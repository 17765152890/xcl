//! Bounded read-only memory loads for synthetic paired-wave mesh validation.
//!
//! This Stage C path is not connected to real-program eligibility or PSO selection.

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_dispatch::{LogicalWaveState, GROUP_THREADS};
use super::mesh_vgpr::{load_vgpr, store_vgpr, VgprScratchLayout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescriptorContract {
    pub register: u8,
    pub words: [u32; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerDescriptorContract {
    pub register: u8,
    pub words: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadOnlyRegion {
    pub guest_base: u64,
    pub memory_offset: u32,
    pub byte_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetSource {
    Zero,
    Scalar(u8),
    Constant(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferFormatSource {
    Raw,
    Descriptor,
    Instruction(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageCOp {
    /// Lane exchange through the GFX10 DS swizzle immediate.
    DsSwizzle {
        dst: u16,
        source: u16,
        control: u16,
    },
    DsRead { dst: u16, address: u16, offsets: [u32; 2], words: u8 },
    DsWrite { address: u16, data: [u16; 2], offsets: [u32; 2], words: u8 },
    ScalarLoad {
        dst: u8,
        base: u8,
        offset: i32,
        soffset: OffsetSource,
        words: u8,
    },
    ScalarBufferLoad {
        dst: u8,
        descriptor: u8,
        offset: u32,
        soffset: OffsetSource,
        words: u8,
    },
    VectorBufferLoad {
        dst: u16,
        address: u16,
        descriptor: u8,
        offset: u32,
        soffset: OffsetSource,
        offen: bool,
        idxen: bool,
        words: u8,
        format: BufferFormatSource,
    },
    /// One dword through the GFX10 global address form.
    GlobalLoad {
        dst: u16,
        address: u16,
        scalar_address: Option<u8>,
        offset: i32,
    },
    /// Integer-coordinate image read through a validated eight-dword GFX10 T#.
    ImageLoad {
        dst: u16,
        coordinates: [u16; 3],
        coordinate_count: u8,
        descriptor: u8,
        dmask: u8,
        d16: bool,
    },
    /// Normalized-coordinate point sample at LOD zero through validated GFX10 T#/S# descriptors.
    ImageSampleLz {
        dst: u16,
        coordinates: [u16; 3],
        coordinate_count: u8,
        /// `true` for the DIM_2D_ARRAY form, whose third coordinate is an
        /// unnormalized array layer rather than normalized texture depth.
        arrayed: bool,
        descriptor: u8,
        sampler: u8,
        dmask: u8,
        d16: bool,
        /// Explicit LOD VGPR for IMAGE_SAMPLE_L. `None` denotes the LZ opcode.
        lod: Option<u16>,
    },
    /// Four neighboring texels from one selected channel of a normalized 2D image.
    ImageGather4Lz {
        dst: u16,
        coordinates: [u16; 2],
        descriptor: u8,
        sampler: u8,
        component: u8,
        d16: bool,
    },
}

fn field(decoded: &Decoded, name: &str) -> Result<u64, String> {
    rdna2::field_value(decoded.enc, &decoded.words, name)
        .ok_or_else(|| format!("{} lacks {name}", decoded.inst.name))
}

fn zero_modifiers(decoded: &Decoded, fields: &[&str]) -> Result<(), String> {
    for &name in fields {
        if rdna2::field_value(decoded.enc, &decoded.words, name).unwrap_or(0) != 0 {
            return Err(format!("{} uses unsupported {name}", decoded.inst.name));
        }
    }
    Ok(())
}

fn offset_source(decoded: &Decoded) -> Result<OffsetSource, String> {
    let selector = field(decoded, "SOFFSET")?;
    Ok(match selector {
        125 | 128 => OffsetSource::Zero,
        0..=124 | 126 | 127 => OffsetSource::Scalar(selector as u8),
        129..=192 => OffsetSource::Constant((selector - 128) as u32),
        193..=208 => OffsetSource::Constant((-(selector as i32 - 192)) as u32),
        240 => OffsetSource::Constant(0.5f32.to_bits()),
        241 => OffsetSource::Constant((-0.5f32).to_bits()),
        242 => OffsetSource::Constant(1.0f32.to_bits()),
        243 => OffsetSource::Constant((-1.0f32).to_bits()),
        244 => OffsetSource::Constant(2.0f32.to_bits()),
        245 => OffsetSource::Constant((-2.0f32).to_bits()),
        246 => OffsetSource::Constant(4.0f32.to_bits()),
        247 => OffsetSource::Constant((-4.0f32).to_bits()),
        248 => OffsetSource::Constant((1.0f32 / (2.0 * std::f32::consts::PI)).to_bits()),
        _ => return Err(format!("unsupported SOFFSET selector {selector}")),
    })
}

fn scalar_span(start: u64, words: u64, what: &str) -> Result<u8, String> {
    let end = start
        .checked_add(words)
        .ok_or_else(|| format!("{what} register overflow"))?;
    if end > 128 {
        return Err(format!("{what} registers are out of bounds"));
    }
    Ok(start as u8)
}

fn scalar_destination(start: u64, words: u64) -> Result<u8, String> {
    let destination = scalar_span(start, words, "scalar destination")?;
    if start + words > 106 {
        return Err("special scalar destinations are outside Stage C".into());
    }
    Ok(destination)
}

pub fn lower_stage_c_one(decoded: &Decoded) -> Result<StageCOp, String> {
    match decoded.inst.name {
        "DS_READ_B32" | "DS_READ2_B32" | "DS_READ2ST64_B32" => {
            zero_modifiers(decoded, &["GDS", "DATA0", "DATA1"])?;
            let pair = decoded.inst.name != "DS_READ_B32";
            let stride = if decoded.inst.name == "DS_READ2ST64_B32" { 256 } else if pair { 4 } else { 1 };
            let offsets = if pair {
                [(field(decoded, "OFFSET0")? * stride) as u32, (field(decoded, "OFFSET1")? * stride) as u32]
            } else {
                [(field(decoded, "OFFSET0")? | (field(decoded, "OFFSET1")? << 8)) as u32, 0]
            };
            Ok(StageCOp::DsRead {
                dst: u16::try_from(field(decoded, "VDST")?).map_err(|_| "DS read destination overflow")?,
                address: u16::try_from(field(decoded, "ADDR")?).map_err(|_| "DS read address overflow")?, offsets,
                words: if pair { 2 } else { 1 },
            })
        }
        "DS_WRITE_B32" | "DS_WRITE2_B32" | "DS_WRITE2ST64_B32" => {
            zero_modifiers(decoded, &["GDS", "VDST"])?;
            let pair = decoded.inst.name != "DS_WRITE_B32";
            let stride = if decoded.inst.name == "DS_WRITE2ST64_B32" { 256 } else if pair { 4 } else { 1 };
            let offsets = if pair {
                [(field(decoded, "OFFSET0")? * stride) as u32, (field(decoded, "OFFSET1")? * stride) as u32]
            } else {
                [(field(decoded, "OFFSET0")? | (field(decoded, "OFFSET1")? << 8)) as u32, 0]
            };
            let data0 = u16::try_from(field(decoded, "DATA0")?).map_err(|_| "DS write data overflow")?;
            Ok(StageCOp::DsWrite {
                address: u16::try_from(field(decoded, "ADDR")?).map_err(|_| "DS write address overflow")?,
                data: [data0, if pair { u16::try_from(field(decoded, "DATA1")?).map_err(|_| "DS write data overflow")? } else { 0 }],
                offsets, words: if pair { 2 } else { 1 },
            })
        }
        "DS_SWIZZLE_B32" => {
            zero_modifiers(decoded, &["GDS", "DATA0", "DATA1"])?;
            let control = u16::try_from(field(decoded, "OFFSET0")? | (field(decoded, "OFFSET1")? << 8))
                .map_err(|_| "DS_SWIZZLE_B32 control overflow")?;
            // In quad mode only the four two-bit selectors are defined. Bitmask mode
            // uses all remaining bits as three five-bit masks.
            if control & 0x8000 == 0 && control & 0x7f00 != 0 {
                return Err("DS_SWIZZLE_B32 quad control has reserved bits".into());
            }
            Ok(StageCOp::DsSwizzle {
                dst: u16::try_from(field(decoded, "VDST")?)
                    .map_err(|_| "DS_SWIZZLE_B32 destination overflow")?,
                source: u16::try_from(field(decoded, "ADDR")?)
                    .map_err(|_| "DS_SWIZZLE_B32 source overflow")?,
                control,
            })
        }
        "IMAGE_GATHER4_LZ" => {
            zero_modifiers(
                decoded,
                &["A16", "DA", "DLC", "GLC", "LWE", "R128", "SLC", "TFE"],
            )?;
            if field(decoded, "UNORM")? != 0 || field(decoded, "DIM")? != 1 {
                return Err("IMAGE_GATHER4_LZ requires normalized 2D coordinates".into());
            }
            let dmask = u8::try_from(field(decoded, "DMASK")?)
                .map_err(|_| "IMAGE_GATHER4_LZ dmask overflow")?;
            if dmask.count_ones() != 1 {
                return Err("IMAGE_GATHER4_LZ requires one selected component".into());
            }
            let nsa = field(decoded, "NSA")?;
            if nsa > 1 {
                return Err("IMAGE_GATHER4_LZ NSA form is outside Stage C".into());
            }
            let base = u16::try_from(field(decoded, "VADDR")?)
                .map_err(|_| "IMAGE_GATHER4_LZ address overflow")?;
            let second = if nsa == 0 {
                base.checked_add(1)
                    .ok_or("IMAGE_GATHER4_LZ address overflow")?
            } else {
                let encoding = rdna2::encoding("MIMG_NSA1")
                    .ok_or("IMAGE_GATHER4_LZ NSA encoding is unavailable")?;
                u16::try_from(
                    rdna2::field_value(encoding, &decoded.words, "VADDRA")
                        .ok_or("IMAGE_GATHER4_LZ lacks VADDRA")?,
                )
                .map_err(|_| "IMAGE_GATHER4_LZ NSA address overflow")?
            };
            Ok(StageCOp::ImageGather4Lz {
                dst: u16::try_from(field(decoded, "VDATA")?)
                    .map_err(|_| "IMAGE_GATHER4_LZ destination overflow")?,
                coordinates: [base, second],
                descriptor: scalar_span(field(decoded, "SRSRC")? * 4, 8, "image descriptor")?,
                sampler: scalar_span(field(decoded, "SSAMP")? * 4, 4, "sampler descriptor")?,
                component: dmask.trailing_zeros() as u8,
                d16: field(decoded, "D16")? != 0,
            })
        }
        "IMAGE_SAMPLE_LZ" | "IMAGE_SAMPLE_L" => {
            let explicit_lod = decoded.inst.name == "IMAGE_SAMPLE_L";
            zero_modifiers(
                decoded,
                &["A16", "DA", "DLC", "GLC", "LWE", "R128", "SLC", "TFE"],
            )?;
            if field(decoded, "UNORM")? != 0 {
                return Err("IMAGE_SAMPLE_LZ requires normalized floating coordinates".into());
            }
            let dim = field(decoded, "DIM")?;
            let (coordinate_count, arrayed) = match dim {
                0 => (1, false),
                1 => (2, false),
                2 => (3, false),
                5 => (3, true),
                _ => {
                    return Err(format!(
                        "IMAGE_SAMPLE_LZ dimension {dim} is outside Stage C"
                    ))
                }
            };
            let dmask = u8::try_from(field(decoded, "DMASK")?)
                .map_err(|_| "IMAGE_SAMPLE_LZ dmask overflow")?;
            if dmask == 0 {
                return Err("IMAGE_SAMPLE_LZ has an empty dmask".into());
            }
            let nsa = field(decoded, "NSA")?;
            let address_count = u64::from(coordinate_count) + u64::from(explicit_lod);
            if nsa > 1 {
                return Err("image sample NSA form is outside Stage C".into());
            }
            let base = u16::try_from(field(decoded, "VADDR")?)
                .map_err(|_| "IMAGE_SAMPLE_LZ address overflow")?;
            let mut coordinates = [base; 3];
            let nsa_encoding = if nsa == 0 {
                None
            } else {
                Some(
                    rdna2::encoding(&format!("MIMG_NSA{nsa}"))
                        .ok_or("IMAGE_SAMPLE_LZ NSA encoding is unavailable")?,
                )
            };
            let mut addresses = [base; 4];
            for index in 1..address_count as usize {
                addresses[index] = if nsa == 0 {
                    base.checked_add(index as u16)
                        .ok_or("IMAGE_SAMPLE_LZ address overflow")?
                } else {
                    let name = format!("VADDR{}", (b'A' + index as u8 - 1) as char);
                    let encoding =
                        nsa_encoding.ok_or("IMAGE_SAMPLE_LZ NSA encoding is unavailable")?;
                    let value = rdna2::field_value(encoding, &decoded.words, &name)
                        .ok_or_else(|| format!("IMAGE_SAMPLE_LZ lacks {name}"))?;
                    u16::try_from(value).map_err(|_| "IMAGE_SAMPLE_LZ NSA address overflow")?
                };
            }
            coordinates[..coordinate_count as usize]
                .copy_from_slice(&addresses[..coordinate_count as usize]);
            Ok(StageCOp::ImageSampleLz {
                dst: u16::try_from(field(decoded, "VDATA")?)
                    .map_err(|_| "IMAGE_SAMPLE_LZ destination overflow")?,
                coordinates,
                coordinate_count,
                arrayed,
                descriptor: scalar_span(field(decoded, "SRSRC")? * 4, 8, "image descriptor")?,
                sampler: scalar_span(field(decoded, "SSAMP")? * 4, 4, "sampler descriptor")?,
                dmask,
                d16: field(decoded, "D16")? != 0,
                lod: explicit_lod.then_some(addresses[address_count as usize - 1]),
            })
        }
        "IMAGE_LOAD" => {
            zero_modifiers(
                decoded,
                &["A16", "DA", "DLC", "GLC", "LWE", "R128", "SLC", "TFE"],
            )?;
            if field(decoded, "UNORM")? != 1 {
                return Err("IMAGE_LOAD requires UNORM integer coordinates".into());
            }
            let dim = field(decoded, "DIM")?;
            if dim > 2 {
                return Err(format!("IMAGE_LOAD dimension {dim} is outside Stage C"));
            }
            let dmask =
                u8::try_from(field(decoded, "DMASK")?).map_err(|_| "IMAGE_LOAD dmask overflow")?;
            if dmask == 0 {
                return Err("IMAGE_LOAD has an empty dmask".into());
            }
            let nsa = field(decoded, "NSA")?;
            if nsa > 1 {
                return Err("IMAGE_LOAD NSA form is outside Stage C".into());
            }
            let base = u16::try_from(field(decoded, "VADDR")?)
                .map_err(|_| "IMAGE_LOAD address overflow")?;
            let mut coordinates = [base; 3];
            let nsa_encoding = if nsa == 0 {
                None
            } else {
                Some(
                    rdna2::encoding(&format!("MIMG_NSA{nsa}"))
                        .ok_or("IMAGE_LOAD NSA encoding is unavailable")?,
                )
            };
            for index in 1..=dim as usize {
                coordinates[index] = if nsa == 0 {
                    base.checked_add(index as u16)
                        .ok_or("IMAGE_LOAD address overflow")?
                } else {
                    let name = format!("VADDR{}", (b'A' + index as u8 - 1) as char);
                    let encoding = nsa_encoding.ok_or("IMAGE_LOAD NSA encoding is unavailable")?;
                    let value = rdna2::field_value(encoding, &decoded.words, &name)
                        .ok_or_else(|| format!("IMAGE_LOAD lacks {name}"))?;
                    u16::try_from(value).map_err(|_| "IMAGE_LOAD NSA address overflow")?
                };
            }
            Ok(StageCOp::ImageLoad {
                dst: u16::try_from(field(decoded, "VDATA")?)
                    .map_err(|_| "IMAGE_LOAD destination overflow")?,
                coordinates,
                coordinate_count: dim as u8 + 1,
                descriptor: scalar_span(field(decoded, "SRSRC")? * 4, 8, "image descriptor")?,
                dmask,
                d16: field(decoded, "D16")? != 0,
            })
        }
        "S_LOAD_DWORD" | "S_LOAD_DWORDX4" | "S_LOAD_DWORDX8" => {
            zero_modifiers(decoded, &["GLC", "DLC"])?;
            let raw = field(decoded, "OFFSET")? as u32;
            let words = match decoded.inst.name {
                "S_LOAD_DWORD" => 1,
                "S_LOAD_DWORDX8" => 8,
                _ => 4,
            };
            Ok(StageCOp::ScalarLoad {
                dst: scalar_destination(field(decoded, "SDATA")?, words)?,
                base: scalar_span(field(decoded, "SBASE")? * 2, 2, "scalar base")?,
                offset: ((raw << 11) as i32) >> 11,
                soffset: offset_source(decoded)?,
                words: words as u8,
            })
        }
        "S_BUFFER_LOAD_DWORD"
        | "S_BUFFER_LOAD_DWORDX2"
        | "S_BUFFER_LOAD_DWORDX4"
        | "S_BUFFER_LOAD_DWORDX8"
        | "S_BUFFER_LOAD_DWORDX16" => {
            zero_modifiers(decoded, &["GLC", "DLC"])?;
            let words = match decoded.inst.name {
                "S_BUFFER_LOAD_DWORD" => 1,
                "S_BUFFER_LOAD_DWORDX2" => 2,
                "S_BUFFER_LOAD_DWORDX4" => 4,
                "S_BUFFER_LOAD_DWORDX8" => 8,
                _ => 16,
            };
            Ok(StageCOp::ScalarBufferLoad {
                dst: scalar_destination(field(decoded, "SDATA")?, words)?,
                descriptor: scalar_span(field(decoded, "SBASE")? * 2, 4, "buffer descriptor")?,
                offset: field(decoded, "OFFSET")? as u32,
                soffset: offset_source(decoded)?,
                words: words as u8,
            })
        }
        "BUFFER_LOAD_DWORD"
        | "BUFFER_LOAD_DWORDX2"
        | "BUFFER_LOAD_DWORDX3"
        | "BUFFER_LOAD_DWORDX4"
        | "BUFFER_LOAD_FORMAT_X"
        | "BUFFER_LOAD_FORMAT_XY"
        | "BUFFER_LOAD_FORMAT_XYZ"
        | "BUFFER_LOAD_FORMAT_XYZW"
        | "TBUFFER_LOAD_FORMAT_X"
        | "TBUFFER_LOAD_FORMAT_XY"
        | "TBUFFER_LOAD_FORMAT_XYZ"
        | "TBUFFER_LOAD_FORMAT_XYZW" => {
            zero_modifiers(decoded, &["GLC", "DLC", "SLC", "TFE", "LDS"])?;
            let (words, format) = match decoded.inst.name {
                "BUFFER_LOAD_DWORD" => (1, BufferFormatSource::Raw),
                "BUFFER_LOAD_DWORDX2" => (2, BufferFormatSource::Raw),
                "BUFFER_LOAD_DWORDX3" => (3, BufferFormatSource::Raw),
                "BUFFER_LOAD_DWORDX4" => (4, BufferFormatSource::Raw),
                "BUFFER_LOAD_FORMAT_X" => (1, BufferFormatSource::Descriptor),
                "BUFFER_LOAD_FORMAT_XY" => (2, BufferFormatSource::Descriptor),
                "BUFFER_LOAD_FORMAT_XYZ" => (3, BufferFormatSource::Descriptor),
                "BUFFER_LOAD_FORMAT_XYZW" => (4, BufferFormatSource::Descriptor),
                "TBUFFER_LOAD_FORMAT_X"
                | "TBUFFER_LOAD_FORMAT_XY"
                | "TBUFFER_LOAD_FORMAT_XYZ"
                | "TBUFFER_LOAD_FORMAT_XYZW" => {
                    let format = u8::try_from(field(decoded, "FORMAT")?)
                        .map_err(|_| "typed buffer format overflow")?;
                    if !STAGE_C_FORMATS.contains(&(format as usize)) {
                        return Err(format!("typed buffer format {format} is outside Stage C"));
                    }
                    (
                        match decoded.inst.name {
                            "TBUFFER_LOAD_FORMAT_X" => 1,
                            "TBUFFER_LOAD_FORMAT_XY" => 2,
                            "TBUFFER_LOAD_FORMAT_XYZ" => 3,
                            _ => 4,
                        },
                        BufferFormatSource::Instruction(format),
                    )
                }
                _ => unreachable!(),
            };
            Ok(StageCOp::VectorBufferLoad {
                dst: u16::try_from(field(decoded, "VDATA")?)
                    .map_err(|_| "vector destination overflow")?,
                address: u16::try_from(field(decoded, "VADDR")?)
                    .map_err(|_| "vector address overflow")?,
                descriptor: scalar_span(field(decoded, "SRSRC")? * 4, 4, "buffer descriptor")?,
                offset: field(decoded, "OFFSET")? as u32,
                soffset: offset_source(decoded)?,
                offen: field(decoded, "OFFEN")? != 0,
                idxen: field(decoded, "IDXEN")? != 0,
                words,
                format,
            })
        }
        "GLOBAL_LOAD_DWORD" => {
            zero_modifiers(decoded, &["DLC", "GLC", "LDS", "SLC"])?;
            let scalar = field(decoded, "SADDR")?;
            let scalar_address = match scalar {
                0x7d | 0x7f => None,
                _ => Some(scalar_span(scalar, 2, "global scalar address")?),
            };
            let raw = field(decoded, "OFFSET")? as u32;
            Ok(StageCOp::GlobalLoad {
                dst: u16::try_from(field(decoded, "VDST")?)
                    .map_err(|_| "global destination overflow")?,
                address: u16::try_from(field(decoded, "ADDR")?)
                    .map_err(|_| "global address overflow")?,
                scalar_address,
                offset: ((raw << 20) as i32) >> 20,
            })
        }
        name => Err(format!("unsupported Stage C opcode/form {name}")),
    }
}

fn scalar(state: &LogicalWaveState, register: u8) -> Option<u32> {
    match register {
        106 => Some(state.vcc[0]),
        107 => Some(state.vcc[1]),
        126 => Some(state.exec[0]),
        127 => Some(state.exec[1]),
        125 => Some(0),
        _ => state.scalar.get(register as usize).copied(),
    }
}

fn source(state: &LogicalWaveState, source: OffsetSource) -> Result<u32, &'static str> {
    match source {
        OffsetSource::Zero => Ok(0),
        OffsetSource::Scalar(register) => scalar(state, register).ok_or("invalid scalar offset"),
        OffsetSource::Constant(value) => Ok(value),
    }
}

fn pair(state: &LogicalWaveState, register: u8) -> Result<u64, &'static str> {
    let low = scalar(state, register).ok_or("invalid scalar pair")?;
    let high = scalar(
        state,
        register.checked_add(1).ok_or("scalar pair overflow")?,
    )
    .ok_or("invalid scalar pair")?;
    Ok(low as u64 | (high as u64) << 32)
}

fn descriptor(state: &LogicalWaveState, register: u8) -> Result<[u32; 4], &'static str> {
    let mut words = [0; 4];
    for (index, word) in words.iter_mut().enumerate() {
        *word = scalar(
            state,
            register
                .checked_add(index as u8)
                .ok_or("descriptor register overflow")?,
        )
        .ok_or("invalid descriptor register")?;
    }
    Ok(words)
}

fn image_descriptor(state: &LogicalWaveState, register: u8) -> Result<[u32; 8], &'static str> {
    let mut words = [0; 8];
    for (index, word) in words.iter_mut().enumerate() {
        *word = scalar(
            state,
            register
                .checked_add(index as u8)
                .ok_or("image descriptor register overflow")?,
        )
        .ok_or("invalid image descriptor register")?;
    }
    Ok(words)
}

pub(crate) fn validated_image_descriptor(
    words: [u32; 8],
    coordinate_count: u8,
) -> Result<crate::gpu::desc::Image, &'static str> {
    validated_image_descriptor_for(words, coordinate_count, false)
}

pub(crate) fn validated_image_descriptor_for(
    words: [u32; 8],
    coordinate_count: u8,
    arrayed: bool,
) -> Result<crate::gpu::desc::Image, &'static str> {
    use crate::gpu::desc::ImageType;
    let image = crate::gpu::desc::image(&words).ok_or("invalid image descriptor type")?;
    let expected = match (coordinate_count, arrayed) {
        (3, true) => ImageType::D2Array,
        (1, false) => ImageType::D1,
        (2, false) => ImageType::D2,
        (3, false) => ImageType::D3,
        _ => return Err("invalid image coordinate count"),
    };
    if image.ty != expected
        || image.swizzle_mode != 0
        || image.base_level != 0
        || image.last_level != 0
        || image.max_mip != 0
        || image.base_array != 0
        || image.min_lod != 0.0
        || image.dst_sel != [4, 5, 6, 7]
        || !matches!(image.format, 75..=77)
        || words[6] != 0
        || words[7] != 0
    {
        return Err("image descriptor is outside the linear no-mip Stage C subset");
    }
    Ok(image)
}

fn point_clamp_array_layer(bits: u32, layers: u32) -> Result<u32, &'static str> {
    let coordinate = f32::from_bits(bits);
    if !coordinate.is_finite() || layers == 0 {
        return Err("sample array layer is not finite");
    }
    let rounded = coordinate.round_ties_even();
    Ok(if rounded <= 0.0 {
        0
    } else if rounded >= layers as f32 {
        layers - 1
    } else {
        rounded as u32
    })
}

pub(crate) fn validated_sampler_descriptor(words: [u32; 4]) -> Result<(), &'static str> {
    // Point filtering, clamp-to-edge on U/V/W, zero LOD bias/range, no comparison,
    // anisotropy, reduction, border pointer, or implementation-specific controls.
    if words != [2 | (2 << 3) | (2 << 6), 0, 0, 0] {
        return Err("sampler descriptor is outside the point-clamp LOD-zero Stage C subset");
    }
    Ok(())
}

fn point_clamp_coordinate(bits: u32, dimension: u32) -> Result<u32, &'static str> {
    let coordinate = f32::from_bits(bits);
    if !coordinate.is_finite() || dimension == 0 {
        return Err("sample coordinate is not finite");
    }
    let scaled = (coordinate * dimension as f32).floor();
    Ok(if scaled <= 0.0 {
        0
    } else if scaled >= dimension as f32 {
        dimension - 1
    } else {
        scaled as u32
    })
}

fn gather_clamp_coordinates(bits: u32, dimension: u32) -> Result<[u32; 2], &'static str> {
    let coordinate = f32::from_bits(bits);
    if !coordinate.is_finite() || dimension == 0 {
        return Err("gather coordinate is not finite");
    }
    let base = (coordinate * dimension as f32 - 0.5).floor();
    let clamp = |value: f32| {
        if value <= 0.0 {
            0
        } else if value >= dimension as f32 {
            dimension - 1
        } else {
            value as u32
        }
    };
    Ok([clamp(base), clamp(base + 1.0)])
}

fn validate_regions(memory: &[u32], regions: &[ReadOnlyRegion]) -> Result<(), &'static str> {
    let memory_bytes = memory.len().checked_mul(4).ok_or("memory size overflow")? as u64;
    for (index, region) in regions.iter().enumerate() {
        if region.guest_base & 3 != 0
            || region.memory_offset & 3 != 0
            || region.byte_len == 0
            || region.byte_len & 3 != 0
        {
            return Err("read-only region is not dword aligned");
        }
        let guest_end = region
            .guest_base
            .checked_add(region.byte_len as u64)
            .ok_or("read-only region address overflow")?;
        let memory_end = region
            .memory_offset
            .checked_add(region.byte_len)
            .ok_or("read-only backing offset overflow")? as u64;
        if memory_end > memory_bytes {
            return Err("read-only region exceeds backing memory");
        }
        for other in &regions[..index] {
            let other_end = other.guest_base + other.byte_len as u64;
            if region.guest_base < other_end && other.guest_base < guest_end {
                return Err("read-only guest regions overlap");
            }
        }
    }
    Ok(())
}

fn read_word(
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    address: u64,
) -> Result<u32, &'static str> {
    if address & 3 != 0 {
        return Err("memory load is misaligned");
    }
    let end = address.checked_add(4).ok_or("memory address overflow")?;
    for region in regions {
        let region_end = region.guest_base + region.byte_len as u64;
        if address >= region.guest_base && end <= region_end {
            let byte = region.memory_offset as u64 + (address - region.guest_base);
            return memory
                .get(byte as usize / 4)
                .copied()
                .ok_or("read-only backing memory is truncated");
        }
    }
    Err("memory load is outside read-only regions")
}

fn read_byte(memory: &[u32], regions: &[ReadOnlyRegion], address: u64) -> Result<u8, &'static str> {
    let end = address.checked_add(1).ok_or("memory address overflow")?;
    for region in regions {
        let region_end = region.guest_base + region.byte_len as u64;
        if address >= region.guest_base && end <= region_end {
            let byte = region.memory_offset as u64 + (address - region.guest_base);
            let word = memory
                .get(byte as usize / 4)
                .copied()
                .ok_or("read-only backing memory is truncated")?;
            return Ok((word >> (8 * (byte as u32 & 3))) as u8);
        }
    }
    Err("memory load is outside read-only regions")
}

fn half_to_f32_bits(value: u32) -> u32 {
    let sign = (value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = value & 0x3ff;
    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((mantissa << (shift + 1) & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    }
}

const STAGE_C_FORMATS: &[usize] = &[8, 20, 21, 22, 24, 29, 50, 56, 60, 69];

fn format_component(
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    address: u64,
    format: usize,
    component: u32,
) -> Result<u32, &'static str> {
    if !STAGE_C_FORMATS.contains(&format) {
        return Err("buffer format is outside the Stage C subset");
    }
    let info = super::emit::buf_format_info(format);
    let components = info & 7;
    let numeric = (info >> 3) & 7;
    let bytes = (0..4)
        .map(|component| (info >> (8 + 6 * component)) & 63)
        .sum::<u32>()
        / 8;
    let mut raw = 0u64;
    for byte in 0..bytes {
        raw |= (read_byte(memory, regions, address + byte as u64)? as u64) << (8 * byte);
    }
    let mut values = [0u32; 4];
    let mut bit_offset = 0;
    for component in 0..4u32 {
        let width = (info >> (8 + 6 * component)) & 63;
        if component >= components {
            values[component as usize] = if component == 3 {
                if matches!(numeric, 4 | 5) {
                    1
                } else {
                    1.0f32.to_bits()
                }
            } else {
                0
            };
            continue;
        }
        let bits = if width == 32 {
            (raw >> bit_offset) as u32
        } else {
            ((raw >> bit_offset) as u32) & ((1u32 << width) - 1)
        };
        let shift = (32 - width) & 31;
        let signed = if width == 32 {
            bits as i32
        } else {
            ((bits << shift) as i32) >> shift
        };
        values[component as usize] = match numeric {
            0 => {
                let max = if width == 32 {
                    u32::MAX as f32
                } else {
                    ((1u32 << width) - 1) as f32
                };
                if bits == 0 {
                    0
                } else if bits as f32 >= max {
                    1.0f32.to_bits()
                } else {
                    ((bits as f32) / max).to_bits()
                }
            }
            1 => {
                let max = if width == 32 {
                    i32::MAX as f32
                } else {
                    ((1u32 << (width - 1)) - 1) as f32
                };
                if signed == 0 {
                    0
                } else if signed as f32 >= max {
                    1.0f32.to_bits()
                } else {
                    ((signed as f32 / max).max(-1.0)).to_bits()
                }
            }
            4 => bits,
            5 => signed as u32,
            6 if width == 32 => bits,
            6 if width == 16 => half_to_f32_bits(bits),
            _ => return Err("buffer numeric format is outside the Stage C subset"),
        };
        bit_offset += width;
    }
    Ok(values[component as usize])
}

fn format_value(
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    address: u64,
    format: usize,
    selector: u32,
) -> Result<u32, &'static str> {
    let numeric = (super::emit::buf_format_info(format) >> 3) & 7;
    match selector {
        0 => Ok(0),
        1 => Ok(if matches!(numeric, 4 | 5) {
            1
        } else {
            1.0f32.to_bits()
        }),
        4..=7 => format_component(memory, regions, address, format, selector - 4),
        _ => Err("buffer destination selector is outside the Stage C subset"),
    }
}

fn add_signed(base: u64, offset: i32) -> Result<u64, &'static str> {
    if offset >= 0 {
        base.checked_add(offset as u64)
    } else {
        base.checked_sub(offset.unsigned_abs() as u64)
    }
    .ok_or("memory address overflow")
}

fn swizzle_lane(lane: u32, control: u16) -> u32 {
    if control & 0x8000 == 0 {
        let selector = (u32::from(control) >> (2 * (lane & 3))) & 3;
        (lane & !3) | selector
    } else {
        let and_mask = (u32::from(control) >> 10) & 0x1f;
        let or_mask = (u32::from(control) >> 5) & 0x1f;
        let xor_mask = u32::from(control) & 0x1f;
        // The masks address one 32-lane row. Wave64's upper row stays upper.
        (lane & 32) | ((((lane & 31) & and_mask) | or_mask) ^ xor_mask)
    }
}

fn execute_checked(
    scratch: &mut [u32],
    layout: VgprScratchLayout,
    states: &mut [LogicalWaveState; 2],
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    ops: &[StageCOp],
) -> Result<(), &'static str> {
    let mut lds = [0u32; 8192];
    for &op in ops {
        for wave in 0..2usize {
            match op {
                StageCOp::DsRead { dst, address, offsets, words } => {
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 { continue; }
                        let thread = wave as u32 * 64 + lane;
                        let base = load_vgpr(scratch, layout, 0, thread, address as u32)?;
                        let mut values = [0u32; 2];
                        for index in 0..words as usize { values[index] = lds[(base.wrapping_add(offsets[index]) >> 2) as usize & 8191]; }
                        for index in 0..words as usize { store_vgpr(scratch, layout, 0, thread, dst as u32 + index as u32, values[index])?; }
                    }
                }
                StageCOp::DsWrite { address, data, offsets, words } => {
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 { continue; }
                        let thread = wave as u32 * 64 + lane;
                        let base = load_vgpr(scratch, layout, 0, thread, address as u32)?;
                        let mut values = [0u32; 2];
                        for index in 0..words as usize { values[index] = load_vgpr(scratch, layout, 0, thread, data[index] as u32)?; }
                        for index in 0..words as usize { lds[(base.wrapping_add(offsets[index]) >> 2) as usize & 8191] = values[index]; }
                    }
                }
                StageCOp::DsSwizzle {
                    dst,
                    source,
                    control,
                } => {
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    let mut snapshot = [0u32; 64];
                    for lane in 0..64u32 {
                        let thread = wave as u32 * 64 + lane;
                        snapshot[lane as usize] =
                            load_vgpr(scratch, layout, 0, thread, source as u32)?;
                    }
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        store_vgpr(
                            scratch,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            snapshot[swizzle_lane(lane, control) as usize],
                        )?;
                    }
                }
                StageCOp::ScalarLoad {
                    dst,
                    base,
                    offset,
                    soffset,
                    words,
                } => {
                    let address = add_signed(pair(&states[wave], base)?, offset)?
                        .checked_add(source(&states[wave], soffset)? as u64)
                        .ok_or("memory address overflow")?;
                    let mut values = [0; 16];
                    for index in 0..words as usize {
                        values[index] = read_word(
                            memory,
                            regions,
                            address
                                .checked_add(index as u64 * 4)
                                .ok_or("memory address overflow")?,
                        )?;
                    }
                    states[wave].scalar[dst as usize..dst as usize + words as usize]
                        .copy_from_slice(&values[..words as usize]);
                }
                StageCOp::ScalarBufferLoad {
                    dst,
                    descriptor: register,
                    offset,
                    soffset,
                    words,
                } => {
                    let descriptor = descriptor(&states[wave], register)?;
                    let stride = (descriptor[1] >> 16) & 0x3fff;
                    let limit = if stride == 0 {
                        descriptor[2] as u64
                    } else {
                        (descriptor[2] as u64)
                            .checked_mul(stride as u64)
                            .ok_or("buffer limit overflow")?
                    };
                    let relative = (offset as u64)
                        .checked_add(source(&states[wave], soffset)? as u64)
                        .ok_or("buffer offset overflow")?;
                    let bytes = words as u64 * 4;
                    if relative.checked_add(bytes).ok_or("buffer range overflow")? > limit {
                        return Err("scalar buffer load is out of range");
                    }
                    let base = descriptor[0] as u64 | ((descriptor[1] & 0xffff) as u64) << 32;
                    let address = base
                        .checked_add(relative)
                        .ok_or("memory address overflow")?;
                    let mut values = [0; 16];
                    for index in 0..words as usize {
                        values[index] = read_word(memory, regions, address + index as u64 * 4)?;
                    }
                    states[wave].scalar[dst as usize..dst as usize + words as usize]
                        .copy_from_slice(&values[..words as usize]);
                }
                StageCOp::VectorBufferLoad {
                    dst,
                    address,
                    descriptor: register,
                    offset,
                    soffset,
                    offen,
                    idxen,
                    words,
                    format,
                } => {
                    let descriptor = descriptor(&states[wave], register)?;
                    let stride = (descriptor[1] >> 16) & 0x3fff;
                    let base = descriptor[0] as u64 | ((descriptor[1] & 0xffff) as u64) << 32;
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        let index = if idxen {
                            load_vgpr(scratch, layout, 0, thread, address as u32)?
                        } else {
                            0
                        };
                        let voffset = if offen {
                            load_vgpr(
                                scratch,
                                layout,
                                0,
                                thread,
                                address as u32 + u32::from(idxen),
                            )?
                        } else {
                            0
                        };
                        let relative = (voffset as u64)
                            .checked_add(offset as u64)
                            .ok_or("buffer offset overflow")?;
                        let format_index = match format {
                            BufferFormatSource::Raw => None,
                            BufferFormatSource::Descriptor => {
                                Some(((descriptor[3] >> 12) & 0x7f) as usize)
                            }
                            BufferFormatSource::Instruction(value) => Some(value as usize),
                        };
                        let bytes = if let Some(format) = format_index {
                            if !STAGE_C_FORMATS.contains(&format) {
                                return Err("buffer format is outside the Stage C subset");
                            }
                            let info = super::emit::buf_format_info(format);
                            (0..4)
                                .map(|component| (info >> (8 + 6 * component)) & 63)
                                .sum::<u32>()
                                / 8
                        } else {
                            words as u32 * 4
                        };
                        let in_range = if stride == 0 {
                            relative
                                .checked_add(bytes as u64)
                                .ok_or("buffer range overflow")?
                                <= descriptor[2] as u64
                        } else {
                            index < descriptor[2]
                        };
                        if !in_range {
                            return Err("vector buffer load is out of range");
                        }
                        let address = base
                            .checked_add(source(&states[wave], soffset)? as u64)
                            .and_then(|value| value.checked_add(relative))
                            .and_then(|value| {
                                value.checked_add((stride as u64).checked_mul(index as u64)?)
                            })
                            .ok_or("memory address overflow")?;
                        if let Some(format) = format_index {
                            for component in 0..words as u32 {
                                let selector = (descriptor[3] >> (component * 3)) & 7;
                                let value =
                                    format_value(memory, regions, address, format, selector)?;
                                store_vgpr(
                                    scratch,
                                    layout,
                                    0,
                                    thread,
                                    dst as u32 + component,
                                    value,
                                )?;
                            }
                        } else {
                            for word in 0..words as u32 {
                                let value = read_word(
                                    memory,
                                    regions,
                                    address
                                        .checked_add(word as u64 * 4)
                                        .ok_or("memory address overflow")?,
                                )?;
                                store_vgpr(scratch, layout, 0, thread, dst as u32 + word, value)?;
                            }
                        }
                    }
                }
                StageCOp::GlobalLoad {
                    dst,
                    address,
                    scalar_address,
                    offset,
                } => {
                    let scalar_base = scalar_address
                        .map(|register| pair(&states[wave], register))
                        .transpose()?
                        .unwrap_or(0);
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        let low = load_vgpr(scratch, layout, 0, thread, address as u32)?;
                        let high = load_vgpr(scratch, layout, 0, thread, address as u32 + 1)?;
                        let vector_base = low as u64 | (high as u64) << 32;
                        let effective = scalar_base
                            .checked_add(vector_base)
                            .ok_or("global address overflow")?;
                        let effective = add_signed(effective, offset)?;
                        let value = read_word(memory, regions, effective)?;
                        store_vgpr(scratch, layout, 0, thread, dst as u32, value)?;
                    }
                }
                StageCOp::ImageLoad {
                    dst,
                    coordinates,
                    coordinate_count,
                    descriptor: register,
                    dmask,
                    d16,
                } => {
                    let descriptor = image_descriptor(&states[wave], register)?;
                    let image = validated_image_descriptor(descriptor, coordinate_count)?;
                    let base = image.address;
                    let integer_format = matches!(image.format, 75 | 76);
                    let dimensions = [image.width, image.height, image.depth];
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        let mut position = [0u32; 3];
                        for index in 0..coordinate_count as usize {
                            position[index] =
                                load_vgpr(scratch, layout, 0, thread, coordinates[index] as u32)?;
                            if position[index] >= dimensions[index] {
                                return Err("image coordinate is out of range");
                            }
                        }
                        let row_pitch = (dimensions[0] as u64)
                            .checked_add(15)
                            .map(|value| value / 16)
                            .and_then(|value| value.checked_mul(256))
                            .ok_or("image row pitch overflow")?;
                        let slice_pitch = row_pitch
                            .checked_mul(dimensions[1] as u64)
                            .ok_or("image slice pitch overflow")?;
                        let texel_offset = (position[2] as u64)
                            .checked_mul(slice_pitch)
                            .and_then(|value| {
                                value.checked_add((position[1] as u64).checked_mul(row_pitch)?)
                            })
                            .and_then(|value| {
                                value.checked_add((position[0] as u64).checked_mul(16)?)
                            })
                            .ok_or("image address overflow")?;
                        let address = base
                            .checked_add(texel_offset)
                            .ok_or("image address overflow")?;
                        let mut selected = [0u32; 4];
                        let mut count = 0usize;
                        for component in 0..4u32 {
                            if dmask & (1 << component) != 0 {
                                selected[count] = read_word(
                                    memory,
                                    regions,
                                    address
                                        .checked_add(component as u64 * 4)
                                        .ok_or("image address overflow")?,
                                )?;
                                count += 1;
                            }
                        }
                        if d16 {
                            for output in 0..count.div_ceil(2) {
                                let half = |value: u32| {
                                    if integer_format {
                                        value & 0xffff
                                    } else {
                                        super::mesh_vector::f32_to_f16(f32::from_bits(value)) as u32
                                    }
                                };
                                let low = half(selected[output * 2]);
                                let high = if output * 2 + 1 < count {
                                    half(selected[output * 2 + 1]) << 16
                                } else {
                                    0
                                };
                                store_vgpr(
                                    scratch,
                                    layout,
                                    0,
                                    thread,
                                    dst as u32 + output as u32,
                                    low | high,
                                )?;
                            }
                        } else {
                            for (output, value) in selected[..count].iter().enumerate() {
                                store_vgpr(
                                    scratch,
                                    layout,
                                    0,
                                    thread,
                                    dst as u32 + output as u32,
                                    *value,
                                )?;
                            }
                        }
                    }
                }
                StageCOp::ImageGather4Lz {
                    dst,
                    coordinates,
                    descriptor: image_register,
                    sampler: sampler_register,
                    component,
                    d16,
                } => {
                    let image_words = image_descriptor(&states[wave], image_register)?;
                    let image = validated_image_descriptor(image_words, 2)?;
                    if image.format != 77 {
                        return Err("gathered image format is outside the float Stage C subset");
                    }
                    validated_sampler_descriptor(descriptor(&states[wave], sampler_register)?)?;
                    let row_pitch = (image.width as u64)
                        .checked_add(15)
                        .map(|value| value / 16)
                        .and_then(|value| value.checked_mul(256))
                        .ok_or("image row pitch overflow")?;
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        let xs = gather_clamp_coordinates(
                            load_vgpr(scratch, layout, 0, thread, coordinates[0] as u32)?,
                            image.width,
                        )?;
                        let ys = gather_clamp_coordinates(
                            load_vgpr(scratch, layout, 0, thread, coordinates[1] as u32)?,
                            image.height,
                        )?;
                        // Architectural order starts at lower-left and proceeds counter-clockwise.
                        let footprint = [(xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[0]), (xs[0], ys[0])];
                        let mut values = [0u32; 4];
                        for (index, (x, y)) in footprint.into_iter().enumerate() {
                            let address = image.address
                                .checked_add(y as u64 * row_pitch)
                                .and_then(|value| value.checked_add(x as u64 * 16))
                                .and_then(|value| value.checked_add(component as u64 * 4))
                                .ok_or("image address overflow")?;
                            values[index] = read_word(memory, regions, address)?;
                        }
                        if d16 {
                            for output in 0..2u32 {
                                let low = super::mesh_vector::f32_to_f16(f32::from_bits(values[(output * 2) as usize])) as u32;
                                let high = (super::mesh_vector::f32_to_f16(f32::from_bits(values[(output * 2 + 1) as usize])) as u32) << 16;
                                store_vgpr(scratch, layout, 0, thread, dst as u32 + output, low | high)?;
                            }
                        } else {
                            for (output, value) in values.into_iter().enumerate() {
                                store_vgpr(scratch, layout, 0, thread, dst as u32 + output as u32, value)?;
                            }
                        }
                    }
                }
                StageCOp::ImageSampleLz {
                    dst,
                    coordinates,
                    coordinate_count,
                    arrayed,
                    descriptor: image_register,
                    sampler: sampler_register,
                    dmask,
                    d16,
                    lod,
                } => {
                    let image_words = image_descriptor(&states[wave], image_register)?;
                    let image = validated_image_descriptor_for(
                        image_words,
                        coordinate_count,
                        arrayed,
                    )?;
                    if image.format != 77 {
                        return Err("sampled image format is outside the float Stage C subset");
                    }
                    validated_sampler_descriptor(descriptor(&states[wave], sampler_register)?)?;
                    let dimensions = [image.width, image.height, image.depth];
                    let row_pitch = (dimensions[0] as u64)
                        .checked_add(15)
                        .map(|value| value / 16)
                        .and_then(|value| value.checked_mul(256))
                        .ok_or("image row pitch overflow")?;
                    let slice_pitch = row_pitch
                        .checked_mul(dimensions[1] as u64)
                        .ok_or("image slice pitch overflow")?;
                    let exec = states[wave].exec[0] as u64 | (states[wave].exec[1] as u64) << 32;
                    for lane in 0..64u32 {
                        if exec & (1u64 << lane) == 0 {
                            continue;
                        }
                        let thread = wave as u32 * 64 + lane;
                        if let Some(lod) = lod {
                            let value =
                                f32::from_bits(load_vgpr(scratch, layout, 0, thread, lod as u32)?);
                            if !value.is_finite() || value != 0.0 {
                                return Err(
                                    "explicit image LOD is outside the non-mipped Stage C subset",
                                );
                            }
                        }
                        let mut position = [0u32; 3];
                        for index in 0..coordinate_count as usize {
                            let coordinate =
                                load_vgpr(scratch, layout, 0, thread, coordinates[index] as u32)?;
                            position[index] = if arrayed && index == 2 {
                                point_clamp_array_layer(coordinate, dimensions[index])?
                            } else {
                                point_clamp_coordinate(coordinate, dimensions[index])?
                            };
                        }
                        let texel_offset = (position[2] as u64)
                            .checked_mul(slice_pitch)
                            .and_then(|value| {
                                value.checked_add((position[1] as u64).checked_mul(row_pitch)?)
                            })
                            .and_then(|value| {
                                value.checked_add((position[0] as u64).checked_mul(16)?)
                            })
                            .ok_or("image address overflow")?;
                        let address = image
                            .address
                            .checked_add(texel_offset)
                            .ok_or("image address overflow")?;
                        let mut selected = [0u32; 4];
                        let mut count = 0usize;
                        for component in 0..4u32 {
                            if dmask & (1 << component) != 0 {
                                selected[count] = read_word(
                                    memory,
                                    regions,
                                    address
                                        .checked_add(component as u64 * 4)
                                        .ok_or("image address overflow")?,
                                )?;
                                count += 1;
                            }
                        }
                        if d16 {
                            for output in 0..count.div_ceil(2) {
                                let low = super::mesh_vector::f32_to_f16(f32::from_bits(
                                    selected[output * 2],
                                )) as u32;
                                let high = if output * 2 + 1 < count {
                                    (super::mesh_vector::f32_to_f16(f32::from_bits(
                                        selected[output * 2 + 1],
                                    )) as u32)
                                        << 16
                                } else {
                                    0
                                };
                                store_vgpr(
                                    scratch,
                                    layout,
                                    0,
                                    thread,
                                    dst as u32 + output as u32,
                                    low | high,
                                )?;
                            }
                        } else {
                            for (output, value) in selected[..count].iter().enumerate() {
                                store_vgpr(
                                    scratch,
                                    layout,
                                    0,
                                    thread,
                                    dst as u32 + output as u32,
                                    *value,
                                )?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_ops(layout: VgprScratchLayout, ops: &[StageCOp]) -> Result<(), &'static str> {
    if ops.iter().any(|op| matches!(op, StageCOp::DsSwizzle { .. }))
        && ops.iter().any(|op| matches!(op, StageCOp::DsRead { .. } | StageCOp::DsWrite { .. })) {
        return Err("Stage C cannot combine lane swizzle scratch with LDS in one bounded shader");
    }
    for &op in ops {
        if let StageCOp::ImageSampleLz {
            lod: Some(register),
            ..
        } = op
        {
            if register as u32 >= layout.vgpr_count {
                return Err("Stage C image LOD is out of bounds");
            }
        }
        match op {
            StageCOp::DsRead { dst, address, words, .. } => {
                if (dst as u32).checked_add(words as u32).is_none_or(|end| end > layout.vgpr_count) { return Err("Stage C DS read destination is out of bounds"); }
                if address as u32 >= layout.vgpr_count { return Err("Stage C DS read address is out of bounds"); }
            }
            StageCOp::DsWrite { address, data, words, .. } => {
                if address as u32 >= layout.vgpr_count { return Err("Stage C DS write address is out of bounds"); }
                for register in &data[..words as usize] { if *register as u32 >= layout.vgpr_count { return Err("Stage C DS write data is out of bounds"); } }
            }
            StageCOp::DsSwizzle { dst, source, .. } => {
                if dst as u32 >= layout.vgpr_count {
                    return Err("Stage C swizzle destination is out of bounds");
                }
                if source as u32 >= layout.vgpr_count {
                    return Err("Stage C swizzle source is out of bounds");
                }
            }
            StageCOp::GlobalLoad { dst, address, .. } => {
                if dst as u32 >= layout.vgpr_count {
                    return Err("Stage C global destination is out of bounds");
                }
                if (address as u32)
                    .checked_add(2)
                    .is_none_or(|end| end > layout.vgpr_count)
                {
                    return Err("Stage C global address is out of bounds");
                }
            }
            StageCOp::VectorBufferLoad {
                dst,
                address,
                offen,
                idxen,
                words,
                ..
            } => {
                if (dst as u32)
                    .checked_add(words as u32)
                    .is_none_or(|end| end > layout.vgpr_count)
                {
                    return Err("Stage C VGPR destination is out of bounds");
                }
                let address_words = u32::from(offen) + u32::from(idxen);
                if address_words != 0
                    && (address as u32)
                        .checked_add(address_words)
                        .is_none_or(|end| end > layout.vgpr_count)
                {
                    return Err("Stage C VGPR address is out of bounds");
                }
            }
            StageCOp::ImageLoad {
                dst,
                coordinates,
                coordinate_count,
                dmask,
                d16,
                ..
            }
            | StageCOp::ImageSampleLz {
                dst,
                coordinates,
                coordinate_count,
                dmask,
                d16,
                ..
            } => {
                let outputs = if d16 {
                    dmask.count_ones().div_ceil(2)
                } else {
                    dmask.count_ones()
                };
                if (dst as u32)
                    .checked_add(outputs)
                    .is_none_or(|end| end > layout.vgpr_count)
                {
                    return Err("Stage C image destination is out of bounds");
                }
                for coordinate in &coordinates[..coordinate_count as usize] {
                    if *coordinate as u32 >= layout.vgpr_count {
                        return Err("Stage C image coordinate is out of bounds");
                    }
                }
            }
            StageCOp::ImageGather4Lz {
                dst,
                coordinates,
                d16,
                ..
            } => {
                let outputs = if d16 { 2 } else { 4 };
                if (dst as u32)
                    .checked_add(outputs)
                    .is_none_or(|end| end > layout.vgpr_count)
                {
                    return Err("Stage C image destination is out of bounds");
                }
                if coordinates
                    .iter()
                    .any(|coordinate| *coordinate as u32 >= layout.vgpr_count)
                {
                    return Err("Stage C image coordinate is out of bounds");
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn execute_stage_c(
    scratch: &mut [u32],
    layout: VgprScratchLayout,
    states: &mut [LogicalWaveState; 2],
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    ops: &[StageCOp],
) -> Result<(), &'static str> {
    layout.validate_for_groups(1)?;
    validate_ops(layout, ops)?;
    validate_regions(memory, regions)?;
    let mut next_scratch = scratch.to_vec();
    let mut next_states = *states;
    execute_checked(
        &mut next_scratch,
        layout,
        &mut next_states,
        memory,
        regions,
        ops,
    )?;
    scratch.copy_from_slice(&next_scratch);
    *states = next_states;
    Ok(())
}

fn hlsl_scalar(source: OffsetSource) -> String {
    match source {
        OffsetSource::Zero => "0u".into(),
        OffsetSource::Constant(value) => format!("{value}u"),
        OffsetSource::Scalar(register) => format!("sg(sb,{register}u)"),
    }
}

fn hlsl_body(ops: &[StageCOp]) -> String {
    let mut body = String::new();
    for &op in ops {
        match op {
            StageCOp::DsRead { dst, address, offsets, words } => {
                body.push_str(&format!("{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;uint a=vg(lb,{address}u);uint v0=0u,v1=0u;if(active){{v0=xcl_lds[((a+{}u)>>2)&8191u];", offsets[0]));
                if words == 2 { body.push_str(&format!("v1=xcl_lds[((a+{}u)>>2)&8191u];", offsets[1])); }
                body.push_str(&format!("stvg(lb,{dst}u,v0);"));
                if words == 2 { body.push_str(&format!("stvg(lb,{}u,v1);", dst + 1)); }
                body.push_str("}GroupMemoryBarrierWithGroupSync();}");
            }
            StageCOp::DsWrite { address, data, offsets, words } => {
                body.push_str(&format!("{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;uint a=vg(lb,{address}u),v0=vg(lb,{}u),v1=0u;", data[0]));
                if words == 2 { body.push_str(&format!("v1=vg(lb,{}u);", data[1])); }
                body.push_str(&format!("if(active){{xcl_lds[((a+{}u)>>2)&8191u]=v0;", offsets[0]));
                if words == 2 { body.push_str(&format!("xcl_lds[((a+{}u)>>2)&8191u]=v1;", offsets[1])); }
                body.push_str("}GroupMemoryBarrierWithGroupSync();}");
            }
            StageCOp::DsSwizzle {
                dst,
                source,
                control,
            } => {
                let lane = if control & 0x8000 == 0 {
                    format!(
                        "((lane&~3u)|(({control}u>>(2u*(lane&3u)))&3u))"
                    )
                } else {
                    let and_mask = (control >> 10) & 0x1f;
                    let or_mask = (control >> 5) & 0x1f;
                    let xor_mask = control & 0x1f;
                    format!(
                        "((lane&32u)|((((lane&31u)&{and_mask}u)|{or_mask}u)^{xor_mask}u))"
                    )
                };
                body.push_str(&format!(
                    "xcl_swz[gi]=vg(lb,{source}u);GroupMemoryBarrierWithGroupSync();{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active)stvg(lb,{dst}u,xcl_swz[wave*64u+{lane}]);}}GroupMemoryBarrierWithGroupSync();"
                ));
            }
            StageCOp::ScalarLoad {
                dst,
                base,
                offset,
                soffset,
                words,
            } => {
                let apply_offset = if offset < 0 {
                    let delta = offset.unsigned_abs();
                    format!("ok=ok&&a>={delta}ull;a-={delta}ull;")
                } else {
                    let delta = offset as u32;
                    format!(
                        "ok=ok&&a<=0xffffffffffffffffull-{delta}ull;a+={delta}ull;"
                    )
                };
                body.push_str(&format!("if(lane==0u){{uint64_t a=((uint64_t)sg(sb,{base}u)|((uint64_t)sg(sb,{base_hi}u)<<32));bool ok=true;{apply_offset}uint64_t so=(uint64_t){soffset};ok=ok&&a<=0xffffffffffffffffull-so;a+=so;uint v[{words}];",base_hi=base+1,soffset=hlsl_scalar(soffset)));
                let final_delta = u32::from(words.saturating_sub(1)) * 4;
                body.push_str(&format!(
                    "ok=ok&&a<=0xffffffffffffffffull-{final_delta}ull;"
                ));
                for i in 0..words {
                    body.push_str(&format!(
                        "ok=ok&&rd(a+{}ull,v[{}]);",
                        i as u32 * 4, i
                    ));
                }
                body.push_str("if(ok){");
                for i in 0..words {
                    body.push_str(&format!("stsg(sb,{}u,v[{}]);", dst + i, i));
                }
                body.push_str("}}DeviceMemoryBarrierWithGroupSync();");
            }
            StageCOp::ScalarBufferLoad {
                dst,
                descriptor,
                offset,
                soffset,
                words,
            } => {
                body.push_str(&format!("if(lane==0u){{uint d0=sg(sb,{d}u),d1=sg(sb,{d1}u),d2=sg(sb,{d2}u),stride=(d1>>16)&0x3fffu;uint64_t off=(uint64_t){offset}u+(uint64_t){soff},limit=stride==0u?(uint64_t)d2:(uint64_t)d2*stride,a=(uint64_t)d0|((uint64_t)(d1&0xffffu)<<32);uint v;",d=descriptor,d1=descriptor+1,d2=descriptor+2,soff=hlsl_scalar(soffset)));
                for i in 0..words {
                    body.push_str(&format!(
                        "if((uint64_t)off+{}ull<=limit&&rd(a+off+{}ull,v))stsg(sb,{}u,v);",
                        i as u32 * 4 + 4,
                        i as u32 * 4,
                        dst + i
                    ));
                }
                body.push_str("}DeviceMemoryBarrierWithGroupSync();");
            }
            StageCOp::VectorBufferLoad {
                dst,
                address,
                descriptor,
                offset,
                soffset,
                offen,
                idxen,
                words,
                format,
            } => {
                let index = if idxen {
                    format!("vg(lb,{address}u)")
                } else {
                    "0u".into()
                };
                let voff = if offen {
                    format!("vg(lb,{}u)", address + u16::from(idxen))
                } else {
                    "0u".into()
                };
                body.push_str(&format!("{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active){{uint d0=sg(sb,{d}u),d1=sg(sb,{d1}u),d2=sg(sb,{d2}u),d3=sg(sb,{d3}u),stride=(d1>>16)&0x3fffu,index={index};uint64_t vo=(uint64_t){voff};bool ok=vo<=0xffffffffffffffffull-(uint64_t){offset}u;uint64_t off=vo+(uint64_t){offset}u,scaled=(uint64_t)stride*(uint64_t)index;bool scaledok=true;if(stride!=0u)scaledok=(uint64_t)index<=0xffffffffffffffffull/(uint64_t)stride;ok=ok&&scaledok;uint64_t a=(uint64_t)d0|((uint64_t)(d1&0xffffu)<<32),so=(uint64_t){soff};ok=ok&&a<=0xffffffffffffffffull-so;a+=so;ok=ok&&a<=0xffffffffffffffffull-off;a+=off;ok=ok&&a<=0xffffffffffffffffull-scaled;a+=scaled;uint v[{words}];",d=descriptor,d1=descriptor+1,d2=descriptor+2,d3=descriptor+3,soff=hlsl_scalar(soffset)));
                if format != BufferFormatSource::Raw {
                    let format = match format {
                        BufferFormatSource::Descriptor => "((d3>>12)&0x7fu)".to_owned(),
                        BufferFormatSource::Instruction(value) => format!("{value}u"),
                        BufferFormatSource::Raw => unreachable!(),
                    };
                    body.push_str(&format!(
                        "uint n=fmtbytes({format});bool loadok=ok&&n!=0u&&(stride==0u?(off<=(uint64_t)d2&&(uint64_t)n<=(uint64_t)d2-off):index<d2);"
                    ));
                    for component in 0..words {
                        body.push_str(&format!(
                            "loadok=loadok&&fmtv(a,{format},(d3>>{}u)&7u,v[{}]);",
                            component as u32 * 3,
                            component
                        ));
                    }
                    body.push_str("if(loadok){");
                    for component in 0..words {
                        body.push_str(&format!(
                            "stvg(lb,{}u,v[{}]);",
                            dst + component as u16,
                            component
                        ));
                    }
                    body.push('}');
                } else {
                    let bytes = u32::from(words) * 4;
                    let final_delta = u32::from(words.saturating_sub(1)) * 4;
                    body.push_str(&format!("bool loadok=ok&&(stride==0u?(off<=(uint64_t)d2&&{bytes}ull<=(uint64_t)d2-off):index<d2)&&a<=0xffffffffffffffffull-{final_delta}ull;"));
                    for word in 0..words {
                        body.push_str(&format!(
                            "loadok=loadok&&rd(a+{}ull,v[{}]);",
                            word as u32 * 4,
                            word
                        ));
                    }
                    body.push_str("if(loadok){");
                    for word in 0..words {
                        body.push_str(&format!(
                            "stvg(lb,{}u,v[{}]);",
                            dst + word as u16,
                            word
                        ));
                    }
                    body.push('}');
                }
                body.push_str("}}DeviceMemoryBarrierWithGroupSync();");
            }
            StageCOp::GlobalLoad {
                dst,
                address,
                scalar_address,
                offset,
            } => {
                let scalar = scalar_address.map_or_else(
                    || "0ull".to_owned(),
                    |register| {
                        format!(
                            "((uint64_t)sg(sb,{register}u)|((uint64_t)sg(sb,{}u)<<32))",
                            register + 1
                        )
                    },
                );
                let apply_offset = if offset < 0 {
                    format!(
                        "uint64_t delta={}ull;ok=ok&&a>=delta;a-=delta;",
                        -(offset as i64)
                    )
                } else {
                    format!(
                        "uint64_t delta={}ull;ok=ok&&a<=0xffffffffffffffffull-delta;a+=delta;",
                        offset as u64
                    )
                };
                body.push_str(&format!(
                    "{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active){{uint64_t s={scalar},q=((uint64_t)vg(lb,{address}u)|((uint64_t)vg(lb,{address_hi}u)<<32));bool ok=q<=0xffffffffffffffffull-s;uint64_t a=s+q;{apply_offset}uint v;if(ok&&rd(a,v))stvg(lb,{dst}u,v);}}}}DeviceMemoryBarrierWithGroupSync();",
                    address_hi = address + 1,
                ));
            }
            StageCOp::ImageLoad {
                dst,
                coordinates,
                coordinate_count,
                descriptor,
                dmask,
                d16,
            } => {
                let mut positions = ["0u".to_owned(), "0u".to_owned(), "0u".to_owned()];
                for index in 0..coordinate_count as usize {
                    positions[index] = format!("vg(lb,{}u)", coordinates[index]);
                }
                body.push_str(&format!(
                    "{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active){{uint d0=sg(sb,{d}u),d1=sg(sb,{d1}u),d2=sg(sb,{d2}u),d3=sg(sb,{d3}u),d4=sg(sb,{d4}u),d5=sg(sb,{d5}u),d6=sg(sb,{d6}u),d7=sg(sb,{d7}u),w=(((d1>>30)&3u)|((d2&0xfffu)<<2))+1u,h=((d2>>14)&0x3fffu)+1u,z=(d4&0x1fffu)+1u,x={x},y={y},q={zq},fmt=(d1>>20)&0x1ffu;bool integer=fmt==75u||fmt==76u;uint64_t a=((uint64_t)d0|((uint64_t)(d1&0xffu)<<32))<<8,rp=((uint64_t)w+15ull)/16ull*256ull,sp=rp*h;bool desc=((d3>>28)=={ty}u)&&(d3&0x0fffffffu)==0xfacu&&((d1>>8)&0xfffu)==0u&&((d4>>16)&0x1fffu)==0u&&((d5>>4)&15u)==0u&&d6==0u&&d7==0u&&(fmt>=75u&&fmt<=77u);a+=(uint64_t)q*sp+(uint64_t)y*rp+(uint64_t)x*16ull;uint v[4];bool ok=desc&&x<w&&y<h&&q<z;",
                    d = descriptor,
                    d1 = descriptor + 1,
                    d2 = descriptor + 2,
                    d3 = descriptor + 3,
                    d4 = descriptor + 4,
                    d5 = descriptor + 5,
                    d6 = descriptor + 6,
                    d7 = descriptor + 7,
                    x = positions[0],
                    y = positions[1],
                    zq = positions[2],
                    ty = 7 + coordinate_count,
                ));
                let mut selected = 0u16;
                for component in 0..4u32 {
                    if dmask & (1 << component) != 0 {
                        body.push_str(&format!(
                            "ok=ok&&rd(a+{}ull,v[{}]);",
                            component * 4,
                            selected
                        ));
                        selected += 1;
                    }
                }
                if d16 {
                    for output in 0..selected.div_ceil(2) {
                        let high = if output * 2 + 1 < selected {
                            format!(
                                "(((integer?v[{}]:f32tof16(asfloat(v[{}])))&0xffffu)<<16u)",
                                output * 2 + 1,
                                output * 2 + 1
                            )
                        } else {
                            "0u".to_owned()
                        };
                        body.push_str(&format!(
                            "if(ok)stvg(lb,{}u,((integer?v[{}]:f32tof16(asfloat(v[{}])))&0xffffu)|{});",
                            dst + output,
                            output * 2,
                            output * 2,
                            high
                        ));
                    }
                } else {
                    for output in 0..selected {
                        body.push_str(&format!("if(ok)stvg(lb,{}u,v[{}]);", dst + output, output));
                    }
                }
                body.push_str("}}DeviceMemoryBarrierWithGroupSync();");
            }
            StageCOp::ImageSampleLz {
                dst,
                coordinates,
                coordinate_count,
                arrayed,
                descriptor,
                sampler,
                dmask,
                d16,
                lod,
            } => {
                let mut coords = ["0.0f".to_owned(), "0.0f".to_owned(), "0.0f".to_owned()];
                for index in 0..coordinate_count as usize {
                    coords[index] = format!("asfloat(vg(lb,{}u))", coordinates[index]);
                }
                let lod_ok = lod
                    .map(|register| format!("(isfinite(asfloat(vg(lb,{register}u)))&&asfloat(vg(lb,{register}u))==0.0f)"))
                    .unwrap_or_else(|| "true".to_owned());
                let layer = if arrayed {
                    "(uint)min(max(round(cz),0.0f),(float)(z-1u))"
                } else {
                    "(uint)min(max(floor(cz*(float)z),0.0f),(float)(z-1u))"
                };
                let descriptor_type = if arrayed { 13 } else { 7 + coordinate_count };
                body.push_str(&format!(
                    "{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active){{uint d0=sg(sb,{d}u),d1=sg(sb,{d1}u),d2=sg(sb,{d2}u),d3=sg(sb,{d3}u),d4=sg(sb,{d4}u),d5=sg(sb,{d5}u),d6=sg(sb,{d6}u),d7=sg(sb,{d7}u),s0=sg(sb,{s}u),s1=sg(sb,{s1}u),s2=sg(sb,{s2}u),s3=sg(sb,{s3}u),w=(((d1>>30)&3u)|((d2&0xfffu)<<2))+1u,h=((d2>>14)&0x3fffu)+1u,z=(d4&0x1fffu)+1u,fmt=(d1>>20)&0x1ffu;float cx={cx},cy={cy},cz={cz};bool finite=isfinite(cx)&&isfinite(cy)&&isfinite(cz);uint x=(uint)min(max(floor(cx*(float)w),0.0f),(float)(w-1u)),y=(uint)min(max(floor(cy*(float)h),0.0f),(float)(h-1u)),q={layer};uint64_t a=((uint64_t)d0|((uint64_t)(d1&0xffu)<<32))<<8,rp=((uint64_t)w+15ull)/16ull*256ull,sp=rp*h;bool desc=((d3>>28)=={ty}u)&&(d3&0x0fffffffu)==0xfacu&&((d1>>8)&0xfffu)==0u&&((d4>>16)&0x1fffu)==0u&&((d5>>4)&15u)==0u&&d6==0u&&d7==0u&&fmt==77u&&s0==146u&&s1==0u&&s2==0u&&s3==0u;a+=(uint64_t)q*sp+(uint64_t)y*rp+(uint64_t)x*16ull;uint v[4];bool ok=desc&&finite;",
                    d=descriptor,d1=descriptor+1,d2=descriptor+2,d3=descriptor+3,d4=descriptor+4,d5=descriptor+5,d6=descriptor+6,d7=descriptor+7,
                    s=sampler,s1=sampler+1,s2=sampler+2,s3=sampler+3,cx=coords[0],cy=coords[1],cz=coords[2],ty=descriptor_type,layer=layer
                ));
                body.push_str(&format!("ok=ok&&{lod_ok};"));
                let mut selected = 0u16;
                for component in 0..4u32 {
                    if dmask & (1 << component) != 0 {
                        body.push_str(&format!(
                            "ok=ok&&rd(a+{}ull,v[{}]);",
                            component * 4,
                            selected
                        ));
                        selected += 1;
                    }
                }
                if d16 {
                    for output in 0..selected.div_ceil(2) {
                        let high = if output * 2 + 1 < selected {
                            format!("((f32tof16(asfloat(v[{}]))&0xffffu)<<16u)", output * 2 + 1)
                        } else {
                            "0u".to_owned()
                        };
                        body.push_str(&format!(
                            "if(ok)stvg(lb,{}u,(f32tof16(asfloat(v[{}]))&0xffffu)|{});",
                            dst + output,
                            output * 2,
                            high
                        ));
                    }
                } else {
                    for output in 0..selected {
                        body.push_str(&format!("if(ok)stvg(lb,{}u,v[{}]);", dst + output, output));
                    }
                }
                body.push_str("}}DeviceMemoryBarrierWithGroupSync();");
            }
            StageCOp::ImageGather4Lz {
                dst,
                coordinates,
                descriptor,
                sampler,
                component,
                d16,
            } => {
                body.push_str(&format!(
                    "{{uint2 ex=scratch.Load2(sb+4u);uint bit=1u<<(lane&31u);bool active=((lane<32u?ex.x:ex.y)&bit)!=0u;if(active){{uint d0=sg(sb,{d}u),d1=sg(sb,{d1}u),d2=sg(sb,{d2}u),d3=sg(sb,{d3}u),d4=sg(sb,{d4}u),d5=sg(sb,{d5}u),d6=sg(sb,{d6}u),d7=sg(sb,{d7}u),s0=sg(sb,{s}u),s1=sg(sb,{s1}u),s2=sg(sb,{s2}u),s3=sg(sb,{s3}u),w=(((d1>>30)&3u)|((d2&0xfffu)<<2))+1u,h=((d2>>14)&0x3fffu)+1u;float cx=asfloat(vg(lb,{cx}u)),cy=asfloat(vg(lb,{cy}u)),fx=floor(cx*(float)w-0.5f),fy=floor(cy*(float)h-0.5f);uint x0=(uint)min(max(fx,0.0f),(float)(w-1u)),x1=(uint)min(max(fx+1.0f,0.0f),(float)(w-1u)),y0=(uint)min(max(fy,0.0f),(float)(h-1u)),y1=(uint)min(max(fy+1.0f,0.0f),(float)(h-1u));uint2 p[4]={{uint2(x0,y1),uint2(x1,y1),uint2(x1,y0),uint2(x0,y0)}};uint64_t base=((uint64_t)d0|((uint64_t)(d1&0xffu)<<32))<<8,rp=((uint64_t)w+15ull)/16ull*256ull;bool ok=isfinite(cx)&&isfinite(cy)&&((d3>>28)==9u)&&(d3&0x0fffffffu)==0xfacu&&((d1>>8)&0xfffu)==0u&&((d4>>16)&0x1fffu)==0u&&((d5>>4)&15u)==0u&&d6==0u&&d7==0u&&((d1>>20)&0x1ffu)==77u&&s0==146u&&s1==0u&&s2==0u&&s3==0u;uint v[4];for(uint i=0u;i<4u;i++)ok=ok&&rd(base+(uint64_t)p[i].y*rp+(uint64_t)p[i].x*16ull+{co}ull,v[i]);",
                    d=descriptor,d1=descriptor+1,d2=descriptor+2,d3=descriptor+3,d4=descriptor+4,d5=descriptor+5,d6=descriptor+6,d7=descriptor+7,
                    s=sampler,s1=sampler+1,s2=sampler+2,s3=sampler+3,cx=coordinates[0],cy=coordinates[1],co=component as u64*4
                ));
                if d16 {
                    body.push_str(&format!("if(ok){{stvg(lb,{dst}u,(f32tof16(asfloat(v[0]))&65535u)|((f32tof16(asfloat(v[1]))&65535u)<<16u));stvg(lb,{dst1}u,(f32tof16(asfloat(v[2]))&65535u)|((f32tof16(asfloat(v[3]))&65535u)<<16u));}}",dst1=dst+1));
                } else {
                    for output in 0..4u16 {
                        body.push_str(&format!("if(ok)stvg(lb,{}u,v[{}]);", dst + output, output));
                    }
                }
                body.push_str("}}DeviceMemoryBarrierWithGroupSync();");
            }
        }
    }
    body
}

pub fn stage_c_validation_hlsl(
    layout: VgprScratchLayout,
    states: [LogicalWaveState; 2],
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    ops: &[StageCOp],
) -> Result<String, &'static str> {
    stage_c_validation_hlsl_with_overrides(layout, states, memory, regions, ops, &[], true)
}

fn stage_c_validation_hlsl_with_overrides(
    layout: VgprScratchLayout,
    states: [LogicalWaveState; 2],
    memory: &[u32],
    regions: &[ReadOnlyRegion],
    ops: &[StageCOp],
    overrides: &[(u32, u32, u32)],
    prevalidate: bool,
) -> Result<String, &'static str> {
    layout.validate_for_groups(1)?;
    validate_ops(layout, ops)?;
    validate_regions(memory, regions)?;
    let mut scratch = vec![0; layout.scratch.total_bytes as usize / 4];
    for thread in 0..GROUP_THREADS {
        for register in 0..layout.vgpr_count {
            store_vgpr(
                &mut scratch,
                layout,
                0,
                thread,
                register,
                super::mesh_vector::initial_vgpr_value(register, thread),
            )?;
        }
    }
    for &(thread, register, value) in overrides {
        if thread >= GROUP_THREADS || register >= layout.vgpr_count {
            return Err("Stage C validation override is out of bounds");
        }
        store_vgpr(&mut scratch, layout, 0, thread, register, value)?;
    }
    if prevalidate {
        let mut checked_states = states;
        execute_stage_c(
            &mut scratch,
            layout,
            &mut checked_states,
            memory,
            regions,
            ops,
        )?;
    }
    let mut region_body = String::new();
    let mut byte_region_body = String::new();
    for region in regions {
        let end = region.guest_base + region.byte_len as u64;
        region_body.push_str(&format!("if(a>={base}ull&&e<={end}ull){{v=mem.Load({offset}u+(uint)(a-{base}ull));return true;}}",base=region.guest_base,end=end,offset=region.memory_offset));
        byte_region_body.push_str(&format!("if(a>={base}ull&&a<{end}ull){{uint p={offset}u+(uint)(a-{base}ull);v=(mem.Load(p&~3u)>>(8u*(p&3u)))&255u;return true;}}",base=region.guest_base,end=end,offset=region.memory_offset));
    }
    let mut state_init = String::new();
    for (wave, state) in states.iter().enumerate() {
        state_init.push_str(&format!("if(gi=={}u){{scratch.Store2(sb+4u,uint2({}u,{}u));scratch.Store2(sb+12u,uint2({}u,{}u));",wave*64,state.exec[0],state.exec[1],state.vcc[0],state.vcc[1]));
        for (register, value) in state.scalar.iter().enumerate() {
            if *value != 0 {
                state_init.push_str(&format!(
                    "scratch.Store(sb+36u+{}u,{value}u);",
                    register * 4
                ));
            }
        }
        state_init.push('}');
    }
    let body = hlsl_body(ops);
    let shared = if ops.iter().any(|op| matches!(op, StageCOp::DsRead { .. } | StageCOp::DsWrite { .. })) { "groupshared uint xcl_lds[8192];" } else { "groupshared uint xcl_swz[128];" };
    let mut override_body = String::new();
    for &(thread, register, value) in overrides {
        override_body.push_str(&format!(
            "if(gi=={thread}u)scratch.Store(lb+{}u,{value}u);",
            register * 4
        ));
    }
    Ok(format!(
        r#"RWStructuredBuffer<uint> io:register(u0);RWByteAddressBuffer records:register(u1);RWByteAddressBuffer mem:register(u2);RWByteAddressBuffer scratch:register(u6);{shared}
uint sg(uint sb,uint r){{if(r==106u)return scratch.Load(sb+12u);if(r==107u)return scratch.Load(sb+16u);if(r==126u)return scratch.Load(sb+4u);if(r==127u)return scratch.Load(sb+8u);return scratch.Load(sb+36u+r*4u);}}void stsg(uint sb,uint r,uint v){{scratch.Store(sb+36u+r*4u,v);}}uint vg(uint lb,uint r){{return scratch.Load(lb+r*4u);}}void stvg(uint lb,uint r,uint v){{scratch.Store(lb+r*4u,v);}}bool rd(uint64_t a,out uint v){{uint64_t e=a+4ull;if(e<a||(a&3ull)!=0ull){{v=0u;return false;}}{region_body}v=0u;return false;}}
bool rdb(uint64_t a,out uint v){{{byte_region_body}v=0u;return false;}}uint fmtbytes(uint f){{if(f==8u)return 2u;if(f==20u||f==21u||f==22u||f==24u||f==29u||f==50u||f==56u||f==60u)return 4u;if(f==69u)return 8u;return 0u;}}bool fmtc(uint64_t a,uint f,uint c,out uint v){{v=0u;uint n=fmtbytes(f);if(n==0u)return false;uint2 r=uint2(0u,0u);for(uint i=0u;i<n;i++){{uint b;if(!rdb(a+i,b))return false;if(i<4u)r.x|=b<<(8u*i);else r.y|=b<<(8u*(i-4u));}}if(f==8u){{if(c>0u){{v=c==3u?0x3f800000u:0u;return true;}}int x=(int)(r.x<<16)>>16;v=asuint(x==0?0.0f:(x>=32767?1.0f:max((float)x/32767.0f,-1.0f)));return true;}}if(f==20u||f==21u||f==22u){{v=c==0u?r.x:(c==3u?(f==20u||f==21u?1u:0x3f800000u):0u);return true;}}if(f==24u){{if(c>=2u){{v=c==3u?0x3f800000u:0u;return true;}}int x=(int)((r.x>>(16u*c))<<16)>>16;v=asuint(x==0?0.0f:(x>=32767?1.0f:max((float)x/32767.0f,-1.0f)));return true;}}if(f==29u){{v=c<2u?asuint(f16tof32((r.x>>(16u*c))&65535u)):(c==3u?0x3f800000u:0u);return true;}}if(f==50u){{uint w=c==3u?2u:10u,o=c==3u?30u:10u*c,m=(1u<<w)-1u,x=(r.x>>o)&m;v=asuint((float)x/(float)m);return true;}}if(f==56u){{v=asuint((float)((r.x>>(8u*c))&255u)/255.0f);return true;}}if(f==60u){{v=(r.x>>(8u*c))&255u;return true;}}if(f==69u){{v=((c<2u?r.x:r.y)>>(16u*(c&1u)))&65535u;return true;}}return false;}}bool fmtv(uint64_t a,uint f,uint s,out uint v){{bool ii=f==20u||f==21u||f==60u||f==69u;if(s==0u){{v=0u;return true;}}if(s==1u){{v=ii?1u:0x3f800000u;return true;}}if(s>=4u)return fmtc(a,f,s-4u,v);v=0u;return false;}}
[numthreads({GROUP_THREADS},1,1)]void main(uint gi:SV_GroupIndex,uint3 gid:SV_GroupID){{uint wave=gi/64u,lane=gi%64u,base=gid.x*io[0],sb=base+wave*548u,lb=base+{vgpr_base}u+gi*{stride}u;for(uint r=0u;r<{vgprs}u;r++){{uint value=r==0u?lane:(r==1u?0xf0f00000u^gi*0x01010101u:0xa5a50000u^r*0x1111u^gi);scratch.Store(lb+r*4u,value);}}{override_body}if(lane==0u){{scratch.Store4(sb,uint4(0u,0u,0u,0u));scratch.Store4(sb+16u,uint4(0u,0u,0u,0u));scratch.Store(sb+32u,0u);}}{state_init}DeviceMemoryBarrierWithGroupSync();{body}}}"#,
        vgpr_base = super::mesh_dispatch::MIN_RECORD_BYTES,
        stride = layout.vgpr_count * 4,
        vgprs = layout.vgpr_count
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    fn decode(name: &str, encoding: &str, fields: &[(&str, u64)]) -> StageCOp {
        let code = encode(name, encoding, fields, None).unwrap();
        lower_stage_c_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
    }

    fn initial(layout: VgprScratchLayout) -> Vec<u32> {
        let mut scratch = vec![0; layout.scratch.total_bytes as usize / 4];
        for thread in 0..GROUP_THREADS {
            for register in 0..layout.vgpr_count {
                store_vgpr(
                    &mut scratch,
                    layout,
                    0,
                    thread,
                    register,
                    super::super::mesh_vector::initial_vgpr_value(register, thread),
                )
                .unwrap();
            }
        }
        scratch
    }

    #[test]
    fn ds_swizzle_decodes_snapshots_rows_and_matches_u6() {
        // Fixed encodings: quad swap [1,0,3,2], followed by a 32-lane row reversal.
        let quad_bytes = [0xb1, 0x00, 0xd4, 0xd8, 0x02, 0x00, 0x00, 0x03];
        let reverse_bytes = [0x1f, 0xfc, 0xd4, 0xd8, 0x03, 0x00, 0x00, 0x02];
        let quad = lower_stage_c_one(&rdna2::decode(&quad_bytes, 0).unwrap()).unwrap();
        let reverse = lower_stage_c_one(&rdna2::decode(&reverse_bytes, 0).unwrap()).unwrap();
        assert_eq!(
            quad,
            StageCOp::DsSwizzle {
                dst: 3,
                source: 2,
                control: 0x00b1,
            }
        );
        assert_eq!(
            reverse,
            StageCOp::DsSwizzle {
                dst: 2,
                source: 3,
                control: 0xfc1f,
            }
        );
        assert_eq!(
            encode(
                "DS_SWIZZLE_B32",
                "ENC_DS",
                &[("VDST", 3), ("ADDR", 2), ("OFFSET0", 0xb1), ("OFFSET1", 0)],
                None,
            )
            .unwrap(),
            quad_bytes
        );

        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let mut states = [LogicalWaveState::default(); 2];
        // Lanes 1 and 63 are inactive sources, while their active partners still read them.
        states[0].exec = [0x8000_0005, 0x8000_0001];
        let mut expected = initial(layout);
        let mut expected_states = states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &[0],
            &[],
            &[quad, reverse],
        )
        .unwrap();
        let initial_value = |register, thread| {
            super::super::mesh_vector::initial_vgpr_value(register, thread)
        };
        // Quad swap snapshots v2, then row reversal snapshots v3 (including inactive lanes).
        assert_eq!(load_vgpr(&expected, layout, 0, 0, 3).unwrap(), initial_value(2, 1));
        assert_eq!(load_vgpr(&expected, layout, 0, 0, 2).unwrap(), initial_value(2, 30));
        assert_eq!(load_vgpr(&expected, layout, 0, 32, 2).unwrap(), initial_value(2, 62));
        assert_eq!(load_vgpr(&expected, layout, 0, 63, 2).unwrap(), initial_value(2, 33));
        assert_eq!(
            load_vgpr(&expected, layout, 0, 1, 3).unwrap(),
            initial_value(3, 1),
            "inactive destinations stay unchanged"
        );

        let hlsl = stage_c_validation_hlsl(layout, states, &[0], &[], &[quad, reverse]).unwrap();
        assert!(hlsl.contains("groupshared uint xcl_swz[128]"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[0],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..GROUP_THREADS {
            for register in 0..layout.vgpr_count {
                assert_eq!(
                    load_vgpr(&got, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected, layout, 0, thread, register).unwrap(),
                    "thread {thread} register {register}"
                );
            }
        }

        let reserved = encode(
            "DS_SWIZZLE_B32",
            "ENC_DS",
            &[("VDST", 3), ("ADDR", 2), ("OFFSET1", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&reserved, 0).unwrap()).is_err());
        assert!(execute_stage_c(
            &mut initial(layout),
            VgprScratchLayout::new(3, 1).unwrap(),
            &mut states,
            &[0],
            &[],
            &[quad],
        )
        .is_err());
    }

    #[test]
    fn ds_dword_reads_and_writes_scale_offsets_and_match_actual_u6() {
        let write2_bytes = [0x00, 0x01, 0x38, 0xd8, 0x01, 0x02, 0x03, 0x00];
        let read2_bytes = [0x00, 0x01, 0xdc, 0xd8, 0x01, 0x00, 0x00, 0x04];
        let write2 = lower_stage_c_one(&rdna2::decode(&write2_bytes, 0).unwrap()).unwrap();
        let read2 = lower_stage_c_one(&rdna2::decode(&read2_bytes, 0).unwrap()).unwrap();
        assert_eq!(write2, StageCOp::DsWrite { address: 1, data: [2, 3], offsets: [0, 4], words: 2 });
        assert_eq!(read2, StageCOp::DsRead { dst: 4, address: 1, offsets: [0, 4], words: 2 });
        assert_eq!(encode("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2), ("DATA1", 3), ("OFFSET1", 1)], None).unwrap(), write2_bytes);
        let write_st64 = decode("DS_WRITE2ST64_B32", "ENC_DS", &[("ADDR", 10), ("DATA0", 5), ("DATA1", 6), ("OFFSET0", 1), ("OFFSET1", 2)]);
        let read_st64 = decode("DS_READ2ST64_B32", "ENC_DS", &[("ADDR", 10), ("VDST", 7), ("OFFSET0", 1), ("OFFSET1", 2)]);
        assert!(matches!(write_st64, StageCOp::DsWrite { offsets: [256, 512], .. }));
        let write_one = decode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 11), ("DATA0", 12), ("OFFSET0", 12)]);
        let read_one = decode("DS_READ_B32", "ENC_DS", &[("ADDR", 11), ("VDST", 13), ("OFFSET0", 12)]);
        let ops = [write2, write_st64, write_one, read2, read_st64, read_one];
        let layout = VgprScratchLayout::new(14, 1).unwrap();
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [0x8000_0001, 0x8000_0001];
        let mut expected = initial(layout);
        let mut overrides = Vec::new();
        for (slot, thread) in [0, 31, 32, 63].into_iter().enumerate() {
            for (register, value) in [(1, slot as u32 * 16), (10, 4096 + slot as u32 * 1024), (11, 8192 + slot as u32 * 16)] {
                store_vgpr(&mut expected, layout, 0, thread, register, value).unwrap();
                overrides.push((thread, register, value));
            }
        }
        let mut expected_states = states;
        execute_stage_c(&mut expected, layout, &mut expected_states, &[0], &[], &ops).unwrap();
        for thread in [0, 31, 32, 63] {
            for (dst, source) in [(4, 2), (5, 3), (7, 5), (8, 6), (13, 12)] {
                assert_eq!(load_vgpr(&expected, layout, 0, thread, dst).unwrap(), super::super::mesh_vector::initial_vgpr_value(source, thread));
            }
        }
        let hlsl = stage_c_validation_hlsl_with_overrides(layout, states, &[0], &[], &ops, &overrides, true).unwrap();
        assert!(hlsl.contains("groupshared uint xcl_lds[8192]"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[0], 1, [0.0; 4], layout.scratch).unwrap();
        for thread in 0..GROUP_THREADS { for register in 0..layout.vgpr_count {
            assert_eq!(load_vgpr(&got, layout, 0, thread, register).unwrap(), load_vgpr(&expected, layout, 0, thread, register).unwrap(), "thread {thread} register {register}");
        }}
        let gds = encode("DS_READ_B32", "ENC_DS", &[("ADDR", 1), ("VDST", 4), ("GDS", 1)], None).unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&gds, 0).unwrap()).is_err());
        assert!(stage_c_validation_hlsl(layout, states, &[0], &[], &[write2, StageCOp::DsSwizzle { dst: 2, source: 3, control: 0 }]).is_err());
    }

    #[test]
    fn global_dword_uses_full_addresses_and_matches_u6() {
        // Fixed externally visible encoding fields: signed OFFSET=-4, v[4:5], s[2:3], v6.
        let bytes = [0xfc, 0x8f, 0x30, 0xdc, 0x04, 0x00, 0x02, 0x06];
        let decoded = rdna2::decode(&bytes, 0).unwrap();
        let op = lower_stage_c_one(&decoded).unwrap();
        assert_eq!(
            op,
            StageCOp::GlobalLoad {
                dst: 6,
                address: 4,
                scalar_address: Some(2),
                offset: -4,
            }
        );
        let encoded = encode(
            "GLOBAL_LOAD_DWORD",
            "ENC_FLAT_GLBL",
            &[("VDST", 6), ("ADDR", 4), ("SADDR", 2), ("OFFSET", 0xffc)],
            None,
        )
        .unwrap();
        assert_eq!(encoded, bytes);
        for modifier in ["DLC", "GLC", "LDS", "SLC"] {
            assert!(lower_stage_c_one(
                &rdna2::decode(
                    &encode(
                        "GLOBAL_LOAD_DWORD",
                        "ENC_FLAT_GLBL",
                        &[("VDST", 6), ("ADDR", 4), ("SADDR", 2), (modifier, 1)],
                        None,
                    )
                    .unwrap(),
                    0,
                )
                .unwrap()
            )
            .is_err());
        }

        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = VgprScratchLayout::new(7, 1).unwrap();
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [3, 0];
        states[0].scalar[2] = 0xffff_fff0;
        states[0].scalar[3] = 1;
        let memory = [0x1111_0000, 0x2222_0001, 0x3333_0002, 0x4444_0003];
        let region = ReadOnlyRegion {
            guest_base: 0x2_0000_0000,
            memory_offset: 0,
            byte_len: 16,
        };
        let overrides = [(0, 4, 0x14), (0, 5, 0), (1, 4, 0x18), (1, 5, 0)];
        let mut expected = initial(layout);
        for &(thread, register, value) in &overrides {
            store_vgpr(&mut expected, layout, 0, thread, register, value).unwrap();
        }
        let mut expected_states = states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &[op],
        )
        .unwrap();
        assert_eq!(load_vgpr(&expected, layout, 0, 0, 6).unwrap(), 0x1111_0000);
        assert_eq!(load_vgpr(&expected, layout, 0, 1, 6).unwrap(), 0x2222_0001);
        assert_eq!(
            load_vgpr(&expected, layout, 0, 2, 6).unwrap(),
            super::super::mesh_vector::initial_vgpr_value(6, 2)
        );

        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            states,
            &memory,
            &[region],
            &[op],
            &overrides,
            true,
        )
        .unwrap();
        assert!(hlsl.contains("uint64_t a="));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..GROUP_THREADS {
            assert_eq!(
                load_vgpr(&got, layout, 0, thread, 6).unwrap(),
                load_vgpr(&expected, layout, 0, thread, 6).unwrap(),
                "thread {thread}"
            );
        }
    }

    #[test]
    fn global_dword_bounds_and_overflow_are_atomic() {
        let layout = VgprScratchLayout::new(3, 1).unwrap();
        let op = StageCOp::GlobalLoad {
            dst: 2,
            address: 0,
            scalar_address: Some(2),
            offset: 0,
        };
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [1, 0];
        states[0].scalar[2] = u32::MAX;
        states[0].scalar[3] = u32::MAX;
        let mut scratch = initial(layout);
        store_vgpr(&mut scratch, layout, 0, 0, 0, 1).unwrap();
        store_vgpr(&mut scratch, layout, 0, 0, 1, 0).unwrap();
        let before = scratch.clone();
        let before_states = states;
        assert!(execute_stage_c(
            &mut scratch,
            layout,
            &mut states,
            &[0],
            &[ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4,
            }],
            &[op],
        )
        .is_err());
        assert_eq!(scratch, before);
        assert_eq!(states, before_states);
        assert!(validate_ops(
            VgprScratchLayout::new(2, 1).unwrap(),
            &[StageCOp::GlobalLoad {
                dst: 2,
                address: 0,
                scalar_address: Some(2),
                offset: 0,
            }]
        )
        .is_err());
    }

    #[test]
    fn global_dword_u6_rejects_wrapped_addresses_before_region_lookup() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = VgprScratchLayout::new(9, 1).unwrap();
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [1, 0];
        states[0].scalar[2] = u32::MAX;
        states[0].scalar[3] = u32::MAX;
        let ops = [
            StageCOp::GlobalLoad {
                dst: 2,
                address: 0,
                scalar_address: Some(2),
                offset: 0,
            },
            StageCOp::GlobalLoad {
                dst: 5,
                address: 3,
                scalar_address: None,
                offset: -4,
            },
            StageCOp::GlobalLoad {
                dst: 8,
                address: 6,
                scalar_address: None,
                offset: 8,
            },
        ];
        let overrides = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 6, 0xffff_fffc),
            (0, 7, 0xffff_ffff),
        ];
        let memory = [0xdead_beef];
        let region = ReadOnlyRegion {
            guest_base: 0,
            memory_offset: 0,
            byte_len: 4,
        };
        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            states,
            &memory,
            &[region],
            &ops,
            &overrides,
            false,
        )
        .unwrap();
        assert!(hlsl.contains("q<=0xffffffffffffffffull-s"));
        assert!(hlsl.contains("a>=delta"));
        assert!(hlsl.contains("a<=0xffffffffffffffffull-delta"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for register in [2, 5, 8] {
            assert_eq!(
                load_vgpr(&got, layout, 0, 0, register).unwrap(),
                super::super::mesh_vector::initial_vgpr_value(register, 0),
                "v{register} changed after a rejected wrapped address"
            );
        }
    }

    #[test]
    fn typed_buffer_u6_rejects_wrapped_addresses_before_format_reads() {
        let layout = VgprScratchLayout::new(2, 1).unwrap();
        let op = StageCOp::VectorBufferLoad {
            dst: 1,
            address: 0,
            descriptor: 0,
            offset: 0,
            soffset: OffsetSource::Zero,
            offen: true,
            idxen: false,
            words: 1,
            format: BufferFormatSource::Instruction(24),
        };
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [1, 0];
        states[0].scalar[0] = 0xffff_fffc;
        states[0].scalar[1] = 0x0000_ffff;
        states[0].scalar[2] = 16;
        states[0].scalar[3] = 4;
        let overrides = [(0, 0, 8)];
        let memory = [0xdead_beef];
        let region = ReadOnlyRegion {
            guest_base: 4,
            memory_offset: 0,
            byte_len: 4,
        };

        let mut cpu_scratch = initial(layout);
        store_vgpr(&mut cpu_scratch, layout, 0, 0, 0, 8).unwrap();
        let cpu_before = cpu_scratch.clone();
        let mut cpu_states = states;
        assert!(execute_stage_c(
            &mut cpu_scratch,
            layout,
            &mut cpu_states,
            &memory,
            &[region],
            &[op],
        )
        .is_err());
        assert_eq!(cpu_scratch, cpu_before);

        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            states,
            &memory,
            &[region],
            &[op],
            &overrides,
            false,
        )
        .unwrap();
        assert!(hlsl.contains("a<=0xffffffffffffffffull-off"));
        assert!(hlsl.contains("loadok=loadok&&fmtv"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        assert_eq!(
            load_vgpr(&got, layout, 0, 0, 1).unwrap(),
            super::super::mesh_vector::initial_vgpr_value(1, 0)
        );
    }

    #[test]
    fn scalar_dword_u6_rejects_wrapped_addresses_atomically() {
        let layout = VgprScratchLayout::new(1, 1).unwrap();
        let op = StageCOp::ScalarLoad {
            dst: 100,
            base: 0,
            offset: 8,
            soffset: OffsetSource::Zero,
            words: 1,
        };
        let mut states = [LogicalWaveState::default(); 2];
        states[0].scalar[0] = u32::MAX - 3;
        states[0].scalar[1] = u32::MAX;
        states[0].scalar[100] = 0x1234_5678;
        let memory = [0xdead_beef];
        let region = ReadOnlyRegion {
            guest_base: 4,
            memory_offset: 0,
            byte_len: 4,
        };
        let mut cpu_scratch = initial(layout);
        let cpu_before = states;
        assert!(execute_stage_c(
            &mut cpu_scratch,
            layout,
            &mut states,
            &memory,
            &[region],
            &[op],
        )
        .is_err());
        assert_eq!(states, cpu_before);

        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            cpu_before,
            &memory,
            &[region],
            &[op],
            &[],
            false,
        )
        .unwrap();
        assert!(hlsl.contains("a<=0xffffffffffffffffull-8ull"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        let got_state = super::super::mesh_dispatch::read_logical_state(
            &got,
            layout.scratch,
            0,
            0,
        )
        .unwrap();
        assert_eq!(got_state.scalar[100], 0x1234_5678);
    }

    #[test]
    fn residual_descriptor_formats_decode_exact_spans() {
        for (name, words) in [
            ("BUFFER_LOAD_FORMAT_XY", 2),
            ("BUFFER_LOAD_FORMAT_XYZ", 3),
            ("BUFFER_LOAD_FORMAT_XYZW", 4),
        ] {
            assert!(matches!(
                decode(
                    name,
                    "ENC_MUBUF",
                    &[("VDATA", 9), ("VADDR", 3), ("SRSRC", 5), ("SOFFSET", 128)]
                ),
                StageCOp::VectorBufferLoad {
                    dst: 9,
                    address: 3,
                    descriptor: 20,
                    words: actual,
                    format: BufferFormatSource::Descriptor,
                    ..
                } if actual == words
            ));
        }
    }

    fn image_tsharp(
        address: u64,
        width: u32,
        height: u32,
        depth: u32,
        format: u32,
        ty: u32,
    ) -> [u32; 8] {
        assert_eq!(address & 0xff, 0);
        let base = address >> 8;
        let width = width - 1;
        [
            base as u32,
            ((base >> 32) as u32 & 0xff) | (format << 20) | ((width & 3) << 30),
            ((width >> 2) & 0xfff) | ((height - 1) << 14),
            4 | (5 << 3) | (6 << 6) | (7 << 9) | (ty << 28),
            depth - 1,
            0,
            0,
            0,
        ]
    }

    fn fixture() -> (
        VgprScratchLayout,
        [LogicalWaveState; 2],
        Vec<u32>,
        ReadOnlyRegion,
        Vec<StageCOp>,
    ) {
        let layout = VgprScratchLayout::new(43, 1).unwrap();
        let region = ReadOnlyRegion {
            guest_base: 0x1_0000_0000,
            memory_offset: 0,
            byte_len: 1024,
        };
        let mut memory: Vec<u32> = (0..256).map(|i| 0x6000_0000 + i).collect();
        memory[100] = 0xff00_00ff;
        memory[200] = 0x4433_2211;
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [u32::MAX; 2];
            state.scalar[0] = region.guest_base as u32;
            state.scalar[1] = (region.guest_base >> 32) as u32;
            for base in [8usize, 16] {
                state.scalar[base] = region.guest_base as u32;
                state.scalar[base + 1] = (region.guest_base >> 32) as u32 | (4 << 16);
                state.scalar[base + 2] = 256;
            }
            state.scalar[64] = region.guest_base as u32 + 400;
            state.scalar[65] = (region.guest_base >> 32) as u32;
            state.scalar[66] = 1024;
            state.scalar[67] = (56 << 12) | 7;
            state.scalar[68] = region.guest_base as u32 + 400;
            state.scalar[69] = (region.guest_base >> 32) as u32;
            state.scalar[70] = 1024;
            state.scalar[71] = (56 << 12) | 4 | (5 << 3) | (6 << 6) | (7 << 9);
            for (base, byte_offset) in [(72usize, 400u32), (76, 600), (80, 700), (84, 800)] {
                state.scalar[base] = region.guest_base as u32 + byte_offset;
                state.scalar[base + 1] = (region.guest_base >> 32) as u32;
                state.scalar[base + 2] = 256;
                state.scalar[base + 3] = 4 | 5 << 3 | 6 << 6 | 7 << 9;
            }
        }
        let ops = vec![
            decode(
                "S_LOAD_DWORDX4",
                "ENC_SMEM",
                &[
                    ("SDATA", 20),
                    ("SBASE", 0),
                    ("OFFSET", 16),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "S_BUFFER_LOAD_DWORD",
                "ENC_SMEM",
                &[("SDATA", 24), ("SBASE", 4), ("OFFSET", 8), ("SOFFSET", 125)],
            ),
            decode(
                "S_BUFFER_LOAD_DWORDX2",
                "ENC_SMEM",
                &[
                    ("SDATA", 25),
                    ("SBASE", 4),
                    ("OFFSET", 12),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "S_BUFFER_LOAD_DWORDX4",
                "ENC_SMEM",
                &[
                    ("SDATA", 28),
                    ("SBASE", 4),
                    ("OFFSET", 20),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "S_BUFFER_LOAD_DWORDX16",
                "ENC_SMEM",
                &[
                    ("SDATA", 32),
                    ("SBASE", 4),
                    ("OFFSET", 64),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "BUFFER_LOAD_DWORDX2",
                "ENC_MUBUF",
                &[
                    ("VDATA", 5),
                    ("VADDR", 0),
                    ("SRSRC", 4),
                    ("SOFFSET", 128),
                    ("IDXEN", 1),
                ],
            ),
            decode(
                "BUFFER_LOAD_FORMAT_X",
                "ENC_MUBUF",
                &[("VDATA", 7), ("VADDR", 0), ("SRSRC", 16), ("SOFFSET", 128)],
            ),
            decode(
                "BUFFER_LOAD_DWORDX4",
                "ENC_MUBUF",
                &[
                    ("VDATA", 9),
                    ("VADDR", 0),
                    ("SRSRC", 4),
                    ("SOFFSET", 128),
                    ("IDXEN", 1),
                ],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_XY",
                "ENC_MTBUF",
                &[
                    ("VDATA", 13),
                    ("VADDR", 0),
                    ("SRSRC", 18),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 24),
                ],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_XYZW",
                "ENC_MTBUF",
                &[
                    ("VDATA", 15),
                    ("VADDR", 0),
                    ("SRSRC", 19),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 69),
                ],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_XYZW",
                "ENC_MTBUF",
                &[
                    ("VDATA", 19),
                    ("VADDR", 0),
                    ("SRSRC", 20),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 50),
                ],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_XYZW",
                "ENC_MTBUF",
                &[
                    ("VDATA", 23),
                    ("VADDR", 0),
                    ("SRSRC", 21),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 60),
                ],
            ),
            decode(
                "S_LOAD_DWORDX8",
                "ENC_SMEM",
                &[
                    ("SDATA", 48),
                    ("SBASE", 0),
                    ("OFFSET", 128),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "S_BUFFER_LOAD_DWORDX8",
                "ENC_SMEM",
                &[
                    ("SDATA", 56),
                    ("SBASE", 4),
                    ("OFFSET", 160),
                    ("SOFFSET", 125),
                ],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_XYZ",
                "ENC_MTBUF",
                &[
                    ("VDATA", 27),
                    ("VADDR", 0),
                    ("SRSRC", 21),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 60),
                ],
            ),
            decode(
                "BUFFER_LOAD_DWORDX3",
                "ENC_MUBUF",
                &[
                    ("VDATA", 30),
                    ("VADDR", 0),
                    ("SRSRC", 4),
                    ("SOFFSET", 128),
                    ("IDXEN", 1),
                ],
            ),
            decode(
                "BUFFER_LOAD_FORMAT_XY",
                "ENC_MUBUF",
                &[("VDATA", 33), ("VADDR", 0), ("SRSRC", 17), ("SOFFSET", 128)],
            ),
            decode(
                "BUFFER_LOAD_FORMAT_XYZ",
                "ENC_MUBUF",
                &[("VDATA", 35), ("VADDR", 0), ("SRSRC", 17), ("SOFFSET", 128)],
            ),
            decode(
                "BUFFER_LOAD_FORMAT_XYZW",
                "ENC_MUBUF",
                &[("VDATA", 38), ("VADDR", 0), ("SRSRC", 17), ("SOFFSET", 128)],
            ),
            decode(
                "TBUFFER_LOAD_FORMAT_X",
                "ENC_MTBUF",
                &[
                    ("VDATA", 42),
                    ("VADDR", 0),
                    ("SRSRC", 18),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 24),
                ],
            ),
            decode(
                "S_LOAD_DWORD",
                "ENC_SMEM",
                &[
                    ("SDATA", 100),
                    ("SBASE", 0),
                    ("OFFSET", 12),
                    ("SOFFSET", 125),
                ],
            ),
        ];
        (layout, states, memory, region, ops)
    }

    #[test]
    fn decoded_forms_use_scaled_bases_and_reject_modifiers() {
        let (_, _, _, _, ops) = fixture();
        let typed_x = [0x00, 0x10, 0xc0, 0xe8, 0x00, 0x2a, 0x12, 0x80];
        assert!(matches!(
            lower_stage_c_one(&rdna2::decode(&typed_x, 0).unwrap()).unwrap(),
            StageCOp::VectorBufferLoad {
                dst: 42,
                address: 0,
                descriptor: 72,
                soffset: OffsetSource::Zero,
                offen: true,
                idxen: false,
                words: 1,
                format: BufferFormatSource::Instruction(24),
                ..
            }
        ));
        assert_eq!(
            encode(
                "TBUFFER_LOAD_FORMAT_X",
                "ENC_MTBUF",
                &[
                    ("VDATA", 42),
                    ("VADDR", 0),
                    ("SRSRC", 18),
                    ("SOFFSET", 128),
                    ("OFFEN", 1),
                    ("FORMAT", 24),
                ],
                None,
            )
            .unwrap(),
            typed_x
        );
        let scalar_dword = [0x03, 0x19, 0x00, 0xf4, 0xfc, 0xff, 0x1f, 0x12];
        assert!(matches!(
            lower_stage_c_one(&rdna2::decode(&scalar_dword, 0).unwrap()).unwrap(),
            StageCOp::ScalarLoad {
                dst: 100,
                base: 6,
                offset: -4,
                soffset: OffsetSource::Scalar(9),
                words: 1,
            }
        ));
        assert_eq!(
            encode(
                "S_LOAD_DWORD",
                "ENC_SMEM",
                &[
                    ("SDATA", 100),
                    ("SBASE", 3),
                    ("OFFSET", 0x1f_fffc),
                    ("SOFFSET", 9),
                ],
                None,
            )
            .unwrap(),
            scalar_dword
        );
        for modifier in ["GLC", "DLC"] {
            let code = encode(
                "S_LOAD_DWORD",
                "ENC_SMEM",
                &[
                    ("SDATA", 100),
                    ("SBASE", 0),
                    ("SOFFSET", 125),
                    (modifier, 1),
                ],
                None,
            )
            .unwrap();
            assert!(lower_stage_c_one(&rdna2::decode(&code, 0).unwrap()).is_err());
        }
        assert!(matches!(
            ops[0],
            StageCOp::ScalarLoad {
                base: 0,
                words: 4,
                ..
            }
        ));
        assert!(matches!(
            ops[1],
            StageCOp::ScalarBufferLoad {
                descriptor: 8,
                words: 1,
                ..
            }
        ));
        assert!(matches!(
            ops[5],
            StageCOp::VectorBufferLoad {
                descriptor: 16,
                idxen: true,
                words: 2,
                ..
            }
        ));
        assert!(matches!(
            ops[6],
            StageCOp::VectorBufferLoad {
                descriptor: 64,
                format: BufferFormatSource::Descriptor,
                ..
            }
        ));
        assert!(matches!(
            ops[7],
            StageCOp::VectorBufferLoad {
                words: 4,
                format: BufferFormatSource::Raw,
                ..
            }
        ));
        assert!(matches!(
            ops[8],
            StageCOp::VectorBufferLoad {
                words: 2,
                format: BufferFormatSource::Instruction(24),
                ..
            }
        ));
        assert!(matches!(
            ops[9],
            StageCOp::VectorBufferLoad {
                words: 4,
                format: BufferFormatSource::Instruction(69),
                ..
            }
        ));
        assert!(matches!(
            ops[10],
            StageCOp::VectorBufferLoad {
                format: BufferFormatSource::Instruction(50),
                ..
            }
        ));
        assert!(matches!(
            ops[11],
            StageCOp::VectorBufferLoad {
                format: BufferFormatSource::Instruction(60),
                ..
            }
        ));
        assert!(matches!(
            ops[12],
            StageCOp::ScalarLoad {
                dst: 48,
                words: 8,
                ..
            }
        ));
        assert!(matches!(
            ops[13],
            StageCOp::ScalarBufferLoad {
                dst: 56,
                words: 8,
                ..
            }
        ));
        assert!(matches!(
            ops[14],
            StageCOp::VectorBufferLoad {
                dst: 27,
                words: 3,
                format: BufferFormatSource::Instruction(60),
                ..
            }
        ));
        assert!(matches!(
            ops[15],
            StageCOp::VectorBufferLoad {
                dst: 30,
                words: 3,
                format: BufferFormatSource::Raw,
                ..
            }
        ));
        assert!(matches!(
            ops[20],
            StageCOp::ScalarLoad {
                dst: 100,
                base: 0,
                offset: 12,
                soffset: OffsetSource::Zero,
                words: 1,
            }
        ));
        let bad = encode(
            "BUFFER_LOAD_DWORD",
            "ENC_MUBUF",
            &[
                ("VDATA", 0),
                ("VADDR", 0),
                ("SRSRC", 0),
                ("SOFFSET", 128),
                ("TFE", 1),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&bad, 0).unwrap()).is_err());
        assert!(super::super::mesh_control::lower(&bad).is_err());

        let unsupported_format = encode(
            "TBUFFER_LOAD_FORMAT_XY",
            "ENC_MTBUF",
            &[
                ("VDATA", 0),
                ("VADDR", 0),
                ("SRSRC", 0),
                ("SOFFSET", 128),
                ("FORMAT", 77),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&unsupported_format, 0).unwrap()).is_err());

        let overflowing_destination = encode(
            "S_LOAD_DWORDX8",
            "ENC_SMEM",
            &[
                ("SDATA", 104),
                ("SBASE", 0),
                ("OFFSET", 0),
                ("SOFFSET", 125),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&overflowing_destination, 0).unwrap()).is_err());
    }

    #[test]
    fn cpu_loads_all_forms_and_preserves_inactive_lanes() {
        let (layout, mut states, memory, region, ops) = fixture();
        assert_eq!(
            read_word(&memory, &[region], region.guest_base + 1020).unwrap(),
            memory[255]
        );
        assert!(read_word(&memory, &[region], region.guest_base + 1024).is_err());
        states[1].exec = [1, 0];
        let mut scratch = initial(layout);
        let before = scratch.clone();
        execute_stage_c(&mut scratch, layout, &mut states, &memory, &[region], &ops).unwrap();
        for state in &states {
            assert_eq!(&state.scalar[20..24], &memory[4..8]);
            assert_eq!(state.scalar[24], memory[2]);
            assert_eq!(&state.scalar[25..27], &memory[3..5]);
            assert_eq!(&state.scalar[28..32], &memory[5..9]);
            assert_eq!(&state.scalar[32..48], &memory[16..32]);
            assert_eq!(&state.scalar[48..56], &memory[32..40]);
            assert_eq!(&state.scalar[56..64], &memory[40..48]);
            assert_eq!(state.scalar[100], memory[3]);
        }
        for thread in 0..128 {
            let active = thread < 64 || thread == 64;
            let expected0 = if active {
                memory[(thread % 64) as usize]
            } else {
                load_vgpr(&before, layout, 0, thread, 5).unwrap()
            };
            let expected1 = if active {
                memory[(thread % 64) as usize + 1]
            } else {
                load_vgpr(&before, layout, 0, thread, 6).unwrap()
            };
            let expected_format = if active {
                1.0f32.to_bits()
            } else {
                load_vgpr(&before, layout, 0, thread, 7).unwrap()
            };
            assert_eq!(
                load_vgpr(&scratch, layout, 0, thread, 5).unwrap(),
                expected0
            );
            assert_eq!(
                load_vgpr(&scratch, layout, 0, thread, 6).unwrap(),
                expected1
            );
            assert_eq!(
                load_vgpr(&scratch, layout, 0, thread, 7).unwrap(),
                expected_format
            );
            for word in 0..4u32 {
                let expected = if active {
                    memory[(thread % 64) as usize + word as usize]
                } else {
                    load_vgpr(&before, layout, 0, thread, 9 + word).unwrap()
                };
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 9 + word).unwrap(),
                    expected
                );
            }
            for component in 0..2u32 {
                let expected = if active {
                    format_value(
                        &memory,
                        &[region],
                        region.guest_base + 400 + (thread % 64) as u64,
                        24,
                        4 + component,
                    )
                    .unwrap()
                } else {
                    load_vgpr(&before, layout, 0, thread, 13 + component).unwrap()
                };
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 13 + component).unwrap(),
                    expected
                );
            }
            for component in 0..4u32 {
                let expected = if active {
                    format_value(
                        &memory,
                        &[region],
                        region.guest_base + 600 + (thread % 64) as u64,
                        69,
                        4 + component,
                    )
                    .unwrap()
                } else {
                    load_vgpr(&before, layout, 0, thread, 15 + component).unwrap()
                };
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 15 + component).unwrap(),
                    expected
                );
            }
            for (dst, byte_offset, format) in [(19u32, 700u64, 50usize), (23, 800, 60)] {
                for component in 0..4u32 {
                    let expected = if active {
                        format_value(
                            &memory,
                            &[region],
                            region.guest_base + byte_offset + (thread % 64) as u64,
                            format,
                            4 + component,
                        )
                        .unwrap()
                    } else {
                        load_vgpr(&before, layout, 0, thread, dst + component).unwrap()
                    };
                    assert_eq!(
                        load_vgpr(&scratch, layout, 0, thread, dst + component).unwrap(),
                        expected
                    );
                }
            }
            for component in 0..3u32 {
                let expected = if active {
                    format_value(
                        &memory,
                        &[region],
                        region.guest_base + 800 + (thread % 64) as u64,
                        60,
                        4 + component,
                    )
                    .unwrap()
                } else {
                    load_vgpr(&before, layout, 0, thread, 27 + component).unwrap()
                };
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 27 + component).unwrap(),
                    expected
                );
            }
            for component in 0..3u32 {
                let expected = if active {
                    memory[(thread % 64) as usize + component as usize]
                } else {
                    load_vgpr(&before, layout, 0, thread, 30 + component).unwrap()
                };
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 30 + component).unwrap(),
                    expected
                );
            }
            for (dst, components) in [(33u32, 2u32), (35, 3), (38, 4)] {
                for component in 0..components {
                    let expected = if active {
                        format_value(
                            &memory,
                            &[region],
                            region.guest_base + 400,
                            56,
                            4 + component,
                        )
                        .unwrap()
                    } else {
                        load_vgpr(&before, layout, 0, thread, dst + component).unwrap()
                    };
                    assert_eq!(
                        load_vgpr(&scratch, layout, 0, thread, dst + component).unwrap(),
                        expected
                    );
                }
            }
            let expected = if active {
                format_value(
                    &memory,
                    &[region],
                    region.guest_base + 400 + (thread % 64) as u64,
                    24,
                    4,
                )
                .unwrap()
            } else {
                load_vgpr(&before, layout, 0, thread, 42).unwrap()
            };
            assert_eq!(
                load_vgpr(&scratch, layout, 0, thread, 42).unwrap(),
                expected
            );
        }
        let lane_zero = [
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        ];
        for (dst, components) in [(33u32, 2usize), (35, 3), (38, 4)] {
            for (component, expected) in lane_zero[..components].iter().enumerate() {
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, 0, dst + component as u32).unwrap(),
                    *expected
                );
            }
        }
    }

    #[test]
    fn rejection_is_atomic_for_bounds_alignment_and_overflow() {
        let (layout, states, memory, region, ops) = fixture();
        for (bad_region, bad_states) in [
            (
                ReadOnlyRegion {
                    byte_len: 16,
                    ..region
                },
                states,
            ),
            (region, {
                let mut s = states;
                s[0].scalar[0] += 2;
                s
            }),
            (region, {
                let mut s = states;
                s[0].scalar[0] = u32::MAX - 3;
                s[0].scalar[1] = u32::MAX;
                s
            }),
        ] {
            let mut scratch = initial(layout);
            let before = scratch.clone();
            let mut actual = bad_states;
            let state_before = actual;
            assert!(execute_stage_c(
                &mut scratch,
                layout,
                &mut actual,
                &memory,
                &[bad_region],
                &ops
            )
            .is_err());
            assert_eq!(scratch, before);
            assert_eq!(actual, state_before);
            assert!(
                stage_c_validation_hlsl(layout, bad_states, &memory, &[bad_region], &ops).is_err()
            );
        }
    }

    #[test]
    fn rejects_unsupported_formats_selectors_and_destination_spans_atomically() {
        let (layout, states, memory, region, ops) = fixture();
        for descriptor_word3 in [(77 << 12) | 4, (56 << 12) | 2] {
            let mut bad_states = states;
            bad_states[0].scalar[67] = descriptor_word3;
            let mut scratch = initial(layout);
            let before = scratch.clone();
            let state_before = bad_states;
            assert!(execute_stage_c(
                &mut scratch,
                layout,
                &mut bad_states,
                &memory,
                &[region],
                &ops
            )
            .is_err());
            assert_eq!(scratch, before);
            assert_eq!(bad_states, state_before);
            assert!(stage_c_validation_hlsl(layout, bad_states, &memory, &[region], &ops).is_err());
        }

        let mut bad_ops = ops.clone();
        if let StageCOp::VectorBufferLoad { dst, .. } = &mut bad_ops[5] {
            *dst = layout.vgpr_count as u16 - 1;
        }
        let mut scratch = initial(layout);
        let before = scratch.clone();
        let mut actual_states = states;
        assert!(execute_stage_c(
            &mut scratch,
            layout,
            &mut actual_states,
            &memory,
            &[region],
            &bad_ops
        )
        .is_err());
        assert_eq!(scratch, before);
        assert_eq!(actual_states, states);
    }

    #[test]
    fn widened_vector_offset_matches_gpu_without_u32_wrap() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = VgprScratchLayout::new(3, 1).unwrap();
        let mut states = [LogicalWaveState::default(); 2];
        states[0].exec = [1, 0];
        states[0].scalar[8] = 0;
        states[0].scalar[9] = 1 | (1 << 16);
        states[0].scalar[10] = 1;
        let memory = [0x1234_5678];
        let region = ReadOnlyRegion {
            guest_base: 0x2_00ff_fffc,
            memory_offset: 0,
            byte_len: 4,
        };
        let ops = [StageCOp::VectorBufferLoad {
            dst: 2,
            address: 1,
            descriptor: 8,
            offset: 0x100f_fffc,
            soffset: OffsetSource::Zero,
            offen: true,
            idxen: false,
            words: 1,
            format: BufferFormatSource::Raw,
        }];
        let mut expected_memory = initial(layout);
        let mut expected_states = states;
        execute_stage_c(
            &mut expected_memory,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &ops,
        )
        .unwrap();
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 2).unwrap(),
            memory[0]
        );
        let hlsl = stage_c_validation_hlsl(layout, states, &memory, &[region], &ops).unwrap();
        assert!(hlsl.contains("uint64_t off="));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got_memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        assert_eq!(load_vgpr(&got_memory, layout, 0, 0, 2).unwrap(), memory[0]);
        assert_eq!(
            load_vgpr(&got_memory, layout, 0, 0, 2).unwrap(),
            load_vgpr(&expected_memory, layout, 0, 0, 2).unwrap()
        );
    }

    #[test]
    fn image_load_decodes_observed_forms_and_matches_u6() {
        let plain = decode(
            "IMAGE_LOAD",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("DMASK", 15),
                ("DIM", 0),
                ("UNORM", 1),
            ],
        );
        let packed = decode(
            "IMAGE_LOAD",
            "ENC_MIMG",
            &[
                ("VDATA", 8),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("DMASK", 15),
                ("DIM", 0),
                ("UNORM", 1),
                ("D16", 1),
            ],
        );
        let packed_integer = decode(
            "IMAGE_LOAD",
            "ENC_MIMG",
            &[
                ("VDATA", 10),
                ("VADDR", 0),
                ("SRSRC", 4),
                ("DMASK", 15),
                ("DIM", 0),
                ("UNORM", 1),
                ("D16", 1),
            ],
        );
        let mut nsa_code = encode(
            "IMAGE_LOAD",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 3),
                ("SRSRC", 2),
                ("DMASK", 7),
                ("DIM", 2),
                ("UNORM", 1),
                ("NSA", 1),
            ],
            None,
        )
        .unwrap();
        nsa_code.extend_from_slice(&(9u32 | 10u32 << 8).to_le_bytes());
        let nsa = lower_stage_c_one(&rdna2::decode(&nsa_code, 0).unwrap()).unwrap();
        assert!(matches!(
            nsa,
            StageCOp::ImageLoad {
                coordinates: [3, 9, 10],
                coordinate_count: 3,
                ..
            }
        ));

        let layout = VgprScratchLayout::new(16, 1).unwrap();
        let base = 0x2_0000_0000u64;
        let memory: Vec<u32> = (0..128u32)
            .flat_map(|x| {
                [
                    (x as f32).to_bits(),
                    (x as f32 + 0.5).to_bits(),
                    (-(x as f32)).to_bits(),
                    1.0f32.to_bits(),
                ]
            })
            .collect();
        let region = ReadOnlyRegion {
            guest_base: base,
            memory_offset: 0,
            byte_len: (memory.len() * 4) as u32,
        };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [u32::MAX; 2];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 128, 1, 1, 77, 8));
            state.scalar[16..24].copy_from_slice(&image_tsharp(base, 128, 1, 1, 75, 8));
        }
        let ops = [plain, packed, packed_integer];
        let mut expected = initial(layout);
        let mut expected_states = states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &ops,
        )
        .unwrap();
        for thread in 0..128u32 {
            let x = thread % 64;
            for component in 0..4u32 {
                assert_eq!(
                    load_vgpr(&expected, layout, 0, thread, 4 + component).unwrap(),
                    memory[(x * 4 + component) as usize]
                );
            }
            assert_eq!(
                load_vgpr(&expected, layout, 0, thread, 8).unwrap(),
                super::super::mesh_vector::f32_to_f16(f32::from_bits(memory[(x * 4) as usize]))
                    as u32
                    | (super::super::mesh_vector::f32_to_f16(f32::from_bits(
                        memory[(x * 4 + 1) as usize]
                    )) as u32)
                        << 16
            );
            assert_eq!(
                load_vgpr(&expected, layout, 0, thread, 10).unwrap(),
                (memory[(x * 4) as usize] & 0xffff)
                    | ((memory[(x * 4 + 1) as usize] & 0xffff) << 16)
            );
        }

        let bad = [StageCOp::ImageLoad {
            dst: 12,
            coordinates: [1, 2, 0],
            coordinate_count: 2,
            descriptor: 8,
            dmask: 1,
            d16: false,
        }];
        let mut rejected = initial(layout);
        let before = rejected.clone();
        let mut rejected_states = states;
        assert!(execute_stage_c(
            &mut rejected,
            layout,
            &mut rejected_states,
            &memory,
            &[region],
            &bad,
        )
        .is_err());
        assert_eq!(rejected, before);
        assert_eq!(rejected_states, states);

        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl(layout, states, &memory, &[region], &ops).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 4..12u32 {
                assert_eq!(
                    load_vgpr(&got, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected, layout, 0, thread, register).unwrap(),
                    "thread {thread}, v{register}"
                );
            }
        }
    }

    #[test]
    fn image_linear_pitch_matches_d2_d3_u6_at_width_seventeen() {
        let image_op = |dim: u64, dst: u64, srsrc: u64| {
            let mut code = encode(
                "IMAGE_LOAD",
                "ENC_MIMG",
                &[
                    ("VDATA", dst),
                    ("VADDR", 0),
                    ("SRSRC", srsrc),
                    ("DMASK", 1),
                    ("DIM", dim),
                    ("UNORM", 1),
                    ("NSA", 1),
                ],
                None,
            )
            .unwrap();
            code.extend_from_slice(&0u32.to_le_bytes());
            lower_stage_c_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let layout = VgprScratchLayout::new(8, 1).unwrap();
        let base = 0x3_0000_0000u64;
        let second = base + 4096;
        let mut memory = vec![0u32; 2048];
        // Width 17 at 16 bytes/texel has a 512-byte row pitch, not 272 bytes.
        let d2_offset = 512 + 16;
        let d3_offset = 4096 + 1024 + 512 + 16;
        memory[d2_offset / 4] = 0x2222_0001;
        memory[d3_offset / 4] = 0x3333_0001;
        // Values at the obsolete tightly-packed offsets make a wrong implementation obvious.
        memory[((17 + 1) * 16) / 4] = 0xdead_0002;
        memory[(4096 + ((2 * 17 + 17 + 1) * 16)) / 4] = 0xdead_0003;
        let region = ReadOnlyRegion {
            guest_base: base,
            memory_offset: 0,
            byte_len: (memory.len() * 4) as u32,
        };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [3, 0];
            state.scalar[24..32].copy_from_slice(&image_tsharp(base, 17, 2, 1, 75, 9));
            state.scalar[32..40].copy_from_slice(&image_tsharp(second, 17, 2, 2, 75, 10));
        }
        let ops = [image_op(1, 4, 6), image_op(2, 5, 8)];
        let mut expected = initial(layout);
        let mut expected_states = states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &ops,
        )
        .unwrap();
        for thread in [1u32, 65] {
            assert_eq!(
                load_vgpr(&expected, layout, 0, thread, 4).unwrap(),
                0x2222_0001
            );
            assert_eq!(
                load_vgpr(&expected, layout, 0, thread, 5).unwrap(),
                0x3333_0001
            );
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl(layout, states, &memory, &[region], &ops).unwrap();
        assert!(hlsl.contains("rp=((uint64_t)w+15ull)/16ull*256ull"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 4..=5u32 {
                assert_eq!(
                    load_vgpr(&got, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected, layout, 0, thread, register).unwrap(),
                    "thread {thread}, v{register}"
                );
            }
        }
    }

    #[test]
    fn image_gather4_lz_decodes_orders_footprint_and_matches_u6() {
        let op = decode(
            "IMAGE_GATHER4_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 4),
                ("DIM", 1),
            ],
        );
        assert!(matches!(
            op,
            StageCOp::ImageGather4Lz {
                coordinates: [0, 1],
                descriptor: 8,
                sampler: 24,
                component: 2,
                ..
            }
        ));
        let bad_mask = encode(
            "IMAGE_GATHER4_LZ",
            "ENC_MIMG",
            &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 6), ("DMASK", 3), ("DIM", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_c_one(&rdna2::decode(&bad_mask, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(8, 1).unwrap();
        let base = 0x5_0000_0000u64;
        let mut memory = vec![0u32; 256];
        for y in 0..4usize {
            for x in 0..4usize {
                memory[(y * 256 + x * 16) / 4 + 2] = ((y * 10 + x) as f32).to_bits();
            }
        }
        let region = ReadOnlyRegion { guest_base: base, memory_offset: 0, byte_len: (memory.len() * 4) as u32 };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [1, 0];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 4, 4, 1, 77, 9));
            state.scalar[24..28].copy_from_slice(&[146, 0, 0, 0]);
        }
        let overrides = [
            (0, 0, 0.5f32.to_bits()),
            (0, 1, 0.5f32.to_bits()),
            (64, 0, 0.5f32.to_bits()),
            (64, 1, 0.5f32.to_bits()),
        ];
        let mut expected = initial(layout);
        for &(thread, register, value) in &overrides {
            store_vgpr(&mut expected, layout, 0, thread, register, value).unwrap();
        }
        let mut expected_states = states;
        execute_stage_c(&mut expected, layout, &mut expected_states, &memory, &[region], &[op]).unwrap();
        let ordered = [21.0f32, 22.0, 12.0, 11.0].map(f32::to_bits);
        for thread in [0u32, 64] {
            for (index, value) in ordered.into_iter().enumerate() {
                assert_eq!(load_vgpr(&expected, layout, 0, thread, 4 + index as u32).unwrap(), value);
            }
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl_with_overrides(layout, states, &memory, &[region], &[op], &overrides, true).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in [0u32, 64] {
            for register in 4..8u32 {
                assert_eq!(load_vgpr(&got, layout, 0, thread, register).unwrap(), load_vgpr(&expected, layout, 0, thread, register).unwrap());
            }
        }
    }

    #[test]
    fn image_sample_lz_uses_real_descriptors_point_clamp_and_padded_rows() {
        let op = decode(
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 1),
            ],
        );
        assert!(matches!(
            op,
            StageCOp::ImageSampleLz {
                descriptor: 8,
                sampler: 24,
                coordinate_count: 2,
                ..
            }
        ));
        let layout = VgprScratchLayout::new(8, 1).unwrap();
        let base = 0x4_0000_0000u64;
        let mut memory = vec![0u32; 512];
        let padded = (512 + 16 * 16) / 4;
        let tight = (17 * 16 + 16 * 16) / 4;
        memory[padded..padded + 4].copy_from_slice(&[
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
        ]);
        memory[tight..tight + 4].copy_from_slice(&[9.0f32.to_bits(); 4]);
        let region = ReadOnlyRegion {
            guest_base: base,
            memory_offset: 0,
            byte_len: (memory.len() * 4) as u32,
        };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [1, 0];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 17, 2, 1, 77, 9));
            state.scalar[24..28].copy_from_slice(&[146, 0, 0, 0]);
        }
        let mut scratch = initial(layout);
        for thread in [0, 64] {
            store_vgpr(&mut scratch, layout, 0, thread, 0, 0.99f32.to_bits()).unwrap();
            store_vgpr(&mut scratch, layout, 0, thread, 1, 0.75f32.to_bits()).unwrap();
        }
        execute_stage_c(&mut scratch, layout, &mut states, &memory, &[region], &[op]).unwrap();
        for thread in [0, 64] {
            for component in 0..4u32 {
                assert_eq!(
                    load_vgpr(&scratch, layout, 0, thread, 4 + component).unwrap(),
                    memory[padded + component as usize]
                );
            }
        }

        // The explicit-LOD opcode shares the same proven non-mipped path, but only
        // when its independent LOD operand is finite zero. Packed and NSA address
        // forms both identify that operand without widening the descriptor contract.
        let explicit = decode(
            "IMAGE_SAMPLE_L",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 1),
            ],
        );
        assert!(matches!(
            explicit,
            StageCOp::ImageSampleLz {
                coordinates: [0, 1, _],
                lod: Some(2),
                ..
            }
        ));
        let mut explicit_nsa_code = encode(
            "IMAGE_SAMPLE_L",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 3),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 1),
                ("NSA", 1),
            ],
            None,
        )
        .unwrap();
        explicit_nsa_code.extend_from_slice(&(9u32 | (11u32 << 8)).to_le_bytes());
        let explicit_nsa =
            lower_stage_c_one(&rdna2::decode(&explicit_nsa_code, 0).unwrap()).unwrap();
        assert!(matches!(
            explicit_nsa,
            StageCOp::ImageSampleLz {
                coordinates: [3, 9, _],
                lod: Some(11),
                ..
            }
        ));
        let mut explicit_scratch = initial(layout);
        for thread in [0, 64] {
            store_vgpr(
                &mut explicit_scratch,
                layout,
                0,
                thread,
                0,
                0.99f32.to_bits(),
            )
            .unwrap();
            store_vgpr(
                &mut explicit_scratch,
                layout,
                0,
                thread,
                1,
                0.75f32.to_bits(),
            )
            .unwrap();
            store_vgpr(
                &mut explicit_scratch,
                layout,
                0,
                thread,
                2,
                0.0f32.to_bits(),
            )
            .unwrap();
        }
        execute_stage_c(
            &mut explicit_scratch,
            layout,
            &mut states,
            &memory,
            &[region],
            &[explicit],
        )
        .unwrap();
        assert_eq!(
            load_vgpr(&explicit_scratch, layout, 0, 0, 4).unwrap(),
            1.0f32.to_bits()
        );
        let before = explicit_scratch.clone();
        store_vgpr(&mut explicit_scratch, layout, 0, 0, 2, 1.0f32.to_bits()).unwrap();
        let rejected_input = explicit_scratch.clone();
        assert!(execute_stage_c(
            &mut explicit_scratch,
            layout,
            &mut states,
            &memory,
            &[region],
            &[explicit],
        )
        .is_err());
        assert_eq!(explicit_scratch, rejected_input);
        assert_ne!(explicit_scratch, before);
        let explicit_hlsl = hlsl_body(&[explicit]);
        assert!(explicit_hlsl.contains("==0.0f"));

        let mut bad_states = states;
        bad_states[0].scalar[24] = 0;
        let before = initial(layout);
        let mut rejected = before.clone();
        assert!(execute_stage_c(
            &mut rejected,
            layout,
            &mut bad_states,
            &memory,
            &[region],
            &[op]
        )
        .is_err());
        assert_eq!(rejected, before);

        // A decoded D1 form also traverses the generated cs_6_6 path. Initial v0 values
        // are finite coordinates at the first texel for the two active lanes.
        let d1 = decode(
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 0),
            ],
        );
        let mut gpu_states = [LogicalWaveState::default(); 2];
        for state in &mut gpu_states {
            state.exec = [1, 0];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 17, 1, 1, 77, 8));
            state.scalar[24..28].copy_from_slice(&[146, 0, 0, 0]);
        }
        let hlsl = stage_c_validation_hlsl(layout, gpu_states, &memory, &[region], &[d1]).unwrap();
        if super::super::dxc::location().is_err() {
            return;
        }
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        let mut expected = initial(layout);
        let mut expected_states = gpu_states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &[d1],
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 4..8u32 {
                assert_eq!(
                    load_vgpr(&got, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected, layout, 0, thread, register).unwrap(),
                    "thread {thread}, v{register}"
                );
            }
        }
    }

    #[test]
    fn image_sample_lz_2d_array_rounds_layers_and_matches_actual_u6() {
        // Fixed bytes from the DIM_2D_ARRAY encoding cohort independently pin the
        // three-address NSA layout used by the decoder.
        let fixed = [0xf09c_0c2au32, 0x0167_3b3f, 0x0000_3940]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let decoded = rdna2::decode(&fixed, 0).unwrap();
        assert_eq!(decoded.inst.name, "IMAGE_SAMPLE_LZ");
        let fixed_op = lower_stage_c_one(&decoded).unwrap();
        assert!(matches!(
            fixed_op,
            StageCOp::ImageSampleLz {
                dst: 59,
                coordinates: [63, 64, 57],
                coordinate_count: 3,
                arrayed: true,
                descriptor: 28,
                sampler: 44,
                dmask: 12,
                ..
            }
        ));

        let op = decode(
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 0),
                ("SRSRC", 2),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 5),
            ],
        );
        assert!(matches!(
            op,
            StageCOp::ImageSampleLz {
                coordinates: [0, 1, 2],
                coordinate_count: 3,
                arrayed: true,
                ..
            }
        ));

        let layout = VgprScratchLayout::new(8, 1).unwrap();
        let base = 0x5_0000_0000u64;
        let mut memory = vec![0u32; 1024];
        // width=4 => 256-byte row pitch; height=3 => 768-byte slice pitch.
        let layer_two = (2 * 768 + 256 + 16) / 4;
        let layer_one = (768 + 256 + 16) / 4;
        let layer_three = (3 * 768 + 256 + 16) / 4;
        let expected_two = [21.0f32, 22.0, 23.0, 24.0].map(f32::to_bits);
        let expected_one = [11.0f32, 12.0, 13.0, 14.0].map(f32::to_bits);
        memory[layer_two..layer_two + 4].copy_from_slice(&expected_two);
        memory[layer_one..layer_one + 4].copy_from_slice(&expected_one);
        memory[layer_three..layer_three + 4].copy_from_slice(&[0xdead_beef; 4]);
        let region = ReadOnlyRegion {
            guest_base: base,
            memory_offset: 0,
            byte_len: (memory.len() * 4) as u32,
        };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [1, 0];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 4, 3, 4, 77, 13));
            state.scalar[24..28].copy_from_slice(&[146, 0, 0, 0]);
        }
        let overrides = [
            (0, 0, 0.4f32.to_bits()),
            (0, 1, 0.6f32.to_bits()),
            (0, 2, 2.5f32.to_bits()),
            (64, 0, 0.4f32.to_bits()),
            (64, 1, 0.6f32.to_bits()),
            (64, 2, 1.4f32.to_bits()),
        ];
        let mut expected = initial(layout);
        for &(thread, register, value) in &overrides {
            store_vgpr(&mut expected, layout, 0, thread, register, value).unwrap();
        }
        let mut expected_states = states;
        execute_stage_c(
            &mut expected,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &[op],
        )
        .unwrap();
        for (thread, values) in [(0u32, expected_two), (64, expected_one)] {
            for (component, value) in values.into_iter().enumerate() {
                assert_eq!(
                    load_vgpr(&expected, layout, 0, thread, 4 + component as u32).unwrap(),
                    value
                );
            }
        }

        let mut wrong_states = states;
        for state in &mut wrong_states {
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 4, 3, 4, 77, 10));
        }
        let mut rejected = expected.clone();
        assert!(execute_stage_c(
            &mut rejected,
            layout,
            &mut wrong_states,
            &memory,
            &[region],
            &[op],
        )
        .is_err());
        assert_eq!(rejected, expected);

        if super::super::dxc::location().is_err() {
            return;
        }
        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            states,
            &memory,
            &[region],
            &[op],
            &overrides,
            true,
        )
        .unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in [0u32, 64] {
            for register in 4..8u32 {
                assert_eq!(
                    load_vgpr(&got, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected, layout, 0, thread, register).unwrap(),
                    "thread {thread}, v{register}"
                );
            }
        }
    }

    #[test]
    fn image_sample_lz_decoded_multidimensional_edges_match_actual_u6() {
        let sample = |dim: u64, dst: u64, vaddr: u64, srsrc: u64, d16: bool| {
            decode(
                "IMAGE_SAMPLE_LZ",
                "ENC_MIMG",
                &[
                    ("VDATA", dst),
                    ("VADDR", vaddr),
                    ("SRSRC", srsrc),
                    ("SSAMP", 6),
                    ("DMASK", 15),
                    ("DIM", dim),
                    ("D16", u64::from(d16)),
                ],
            )
        };
        let d1 = sample(0, 8, 0, 2, false);
        let d2 = sample(1, 12, 0, 8, false);
        let d2_d16 = sample(1, 20, 0, 8, true);
        let mut d3_code = encode(
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 16),
                ("VADDR", 3),
                ("SRSRC", 12),
                ("SSAMP", 6),
                ("DMASK", 15),
                ("DIM", 2),
                ("NSA", 1),
            ],
            None,
        )
        .unwrap();
        d3_code.extend_from_slice(&(4u32 | (5u32 << 8)).to_le_bytes());
        let d3 = lower_stage_c_one(&rdna2::decode(&d3_code, 0).unwrap()).unwrap();
        assert!(matches!(
            d3,
            StageCOp::ImageSampleLz {
                coordinates: [3, 4, 5],
                ..
            }
        ));

        let layout = VgprScratchLayout::new(24, 1).unwrap();
        let base = 0x5_0000_0000u64;
        let d2_base = base + 4096;
        let d3_base = base + 8192;
        let mut memory = vec![0u32; 3072];
        let texels = [
            (0usize, [10.0f32, 11.0, 12.0, 13.0]),
            (256usize, [20.0f32, 21.0, 22.0, 23.0]),
            (4096 + 1024, [30.0f32, 31.0, 32.0, 33.0]),
            (4096 + 768, [40.0f32, 41.0, 42.0, 43.0]),
            (8192 + 2304, [50.0f32, 51.0, 52.0, 53.0]),
            (8192, [60.0f32, 61.0, 62.0, 63.0]),
        ];
        for (byte, values) in texels {
            for (component, value) in values.into_iter().enumerate() {
                memory[byte / 4 + component] = value.to_bits();
            }
        }
        // Poison the obsolete tightly packed D2/D3 positions.
        memory[(4096 + (2 * 17 * 16)) / 4] = 0xdead_0002;
        memory[(8192 + ((3 * 17 + 16) * 16)) / 4] = 0xdead_0003;
        let region = ReadOnlyRegion {
            guest_base: base,
            memory_offset: 0,
            byte_len: (memory.len() * 4) as u32,
        };
        let mut states = [LogicalWaveState::default(); 2];
        for state in &mut states {
            state.exec = [3, 0];
            state.scalar[8..16].copy_from_slice(&image_tsharp(base, 17, 1, 1, 77, 8));
            state.scalar[24..28].copy_from_slice(&[146, 0, 0, 0]);
            state.scalar[32..40].copy_from_slice(&image_tsharp(d2_base, 17, 3, 1, 77, 9));
            state.scalar[48..56].copy_from_slice(&image_tsharp(d3_base, 17, 3, 2, 77, 10));
        }
        let mut overrides = Vec::new();
        for wave_base in [0u32, 64] {
            // lane 0: negative/1.0/>1 exercise clamp-to-edge and padded rows/slices.
            overrides.extend_from_slice(&[
                (wave_base, 0, (-0.25f32).to_bits()),
                (wave_base, 1, 1.0f32.to_bits()),
                (wave_base, 3, 1.25f32.to_bits()),
                (wave_base, 4, 0.5f32.to_bits()),
                (wave_base, 5, 1.0f32.to_bits()),
                // lane 1: interior/zero coordinates.
                (wave_base + 1, 0, 0.99f32.to_bits()),
                (wave_base + 1, 1, 0.34f32.to_bits()),
                (wave_base + 1, 3, 0.0f32.to_bits()),
                (wave_base + 1, 4, 0.0f32.to_bits()),
                (wave_base + 1, 5, 0.0f32.to_bits()),
            ]);
        }
        let ops = [d1, d2, d3, d2_d16];
        let hlsl = stage_c_validation_hlsl_with_overrides(
            layout,
            states,
            &memory,
            &[region],
            &ops,
            &overrides,
            true,
        )
        .unwrap();
        if super::super::dxc::location().is_err() {
            return;
        }
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave_base in [0u32, 64] {
            for (thread, d1_values, d2_values, d3_values) in [
                (
                    wave_base,
                    [10.0f32, 11.0, 12.0, 13.0],
                    [30.0f32, 31.0, 32.0, 33.0],
                    [50.0f32, 51.0, 52.0, 53.0],
                ),
                (
                    wave_base + 1,
                    [20.0f32, 21.0, 22.0, 23.0],
                    [40.0f32, 41.0, 42.0, 43.0],
                    [60.0f32, 61.0, 62.0, 63.0],
                ),
            ] {
                for (base_register, values) in [(8u32, d1_values), (12, d2_values), (16, d3_values)]
                {
                    for (component, value) in values.into_iter().enumerate() {
                        assert_eq!(
                            load_vgpr(&got, layout, 0, thread, base_register + component as u32)
                                .unwrap(),
                            value.to_bits(),
                            "thread {thread} v{}",
                            base_register + component as u32
                        );
                    }
                }
                let low = super::super::mesh_vector::f32_to_f16(d2_values[0]) as u32
                    | (super::super::mesh_vector::f32_to_f16(d2_values[1]) as u32) << 16;
                let high = super::super::mesh_vector::f32_to_f16(d2_values[2]) as u32
                    | (super::super::mesh_vector::f32_to_f16(d2_values[3]) as u32) << 16;
                assert_eq!(load_vgpr(&got, layout, 0, thread, 20).unwrap(), low);
                assert_eq!(load_vgpr(&got, layout, 0, thread, 21).unwrap(), high);
            }
            // Inactive lane 2 must retain the validation harness' initial value.
            assert_eq!(
                load_vgpr(&got, layout, 0, wave_base + 2, 8).unwrap(),
                super::super::mesh_vector::initial_vgpr_value(8, wave_base + 2)
            );
        }

        for bad in [f32::NAN.to_bits(), f32::INFINITY.to_bits()] {
            let mut scratch = initial(layout);
            store_vgpr(&mut scratch, layout, 0, 0, 0, bad).unwrap();
            let before = scratch.clone();
            let mut one_wave = states;
            one_wave[0].exec = [1, 0];
            one_wave[1].exec = [0, 0];
            assert!(execute_stage_c(
                &mut scratch,
                layout,
                &mut one_wave,
                &memory,
                &[region],
                &[d1]
            )
            .is_err());
            assert_eq!(scratch, before);

            let mut bad_states = states;
            bad_states[0].exec = [1, 0];
            bad_states[1].exec = [1, 0];
            let bad_overrides = [(0u32, 0u32, bad), (64u32, 0u32, bad)];
            let bad_hlsl = stage_c_validation_hlsl_with_overrides(
                layout,
                bad_states,
                &memory,
                &[region],
                &[d1],
                &bad_overrides,
                false,
            )
            .unwrap();
            let bad_compute = super::super::dxc::compile(&bad_hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{bad_hlsl}"));
            let (_, _, rejected_gpu) = super::super::runner::mesh_draw_with_dispatch_scratch(
                &bad_compute,
                &mesh,
                &pixel,
                1,
                record.record_dwords(),
                &[
                    layout.scratch.record_stride_bytes,
                    record.record_dwords() as u32 * 4,
                ],
                &memory,
                1,
                [0.0; 4],
                layout.scratch,
            )
            .unwrap();
            for thread in [0u32, 1, 64, 65] {
                for register in 7..=12u32 {
                    assert_eq!(
                        load_vgpr(&rejected_gpu, layout, 0, thread, register).unwrap(),
                        super::super::mesh_vector::initial_vgpr_value(register, thread),
                        "nonfinite {bad:#x}, thread {thread}, v{register}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_matches_cpu_through_bounded_u6_validation() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let (layout, mut initial_states, memory, region, ops) = fixture();
        initial_states[1].exec = [1, 0];
        let mut expected_memory = initial(layout);
        let mut expected_states = initial_states;
        execute_stage_c(
            &mut expected_memory,
            layout,
            &mut expected_states,
            &memory,
            &[region],
            &ops,
        )
        .unwrap();
        let hlsl =
            stage_c_validation_hlsl(layout, initial_states, &memory, &[region], &ops).unwrap();
        assert!(!hlsl.contains("GroupMemoryBarrierWithGroupSync"));
        assert!(hlsl.contains("DeviceMemoryBarrierWithGroupSync"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0.xxxx;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, got_memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &memory,
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got = super::super::mesh_dispatch::read_logical_state(
                &got_memory,
                layout.scratch,
                0,
                wave,
            )
            .unwrap();
            assert_eq!(
                &got.scalar[20..64],
                &expected_states[wave as usize].scalar[20..64]
            );
            assert_eq!(
                got.scalar[100],
                expected_states[wave as usize].scalar[100]
            );
        }
        for thread in 0..128 {
            for register in 5..=32 {
                assert_eq!(
                    load_vgpr(&got_memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    "thread {thread}, v{register}"
                );
            }
            assert_eq!(
                load_vgpr(&got_memory, layout, 0, thread, 42).unwrap(),
                load_vgpr(&expected_memory, layout, 0, thread, 42).unwrap(),
                "thread {thread}, v42"
            );
        }
        for thread in [0u32, 63, 64] {
            let lane = (thread % 64) as usize;
            for component in 0..3u32 {
                assert_eq!(
                    load_vgpr(&got_memory, layout, 0, thread, 30 + component).unwrap(),
                    0x6000_0000 + lane as u32 + component,
                    "triple load thread {thread}, component {component}"
                );
            }
        }
        for thread in [65u32, 127] {
            for component in 0..3u32 {
                let register = 30 + component;
                assert_eq!(
                    load_vgpr(&got_memory, layout, 0, thread, register).unwrap(),
                    super::super::mesh_vector::initial_vgpr_value(register, thread),
                    "inactive triple load thread {thread}, v{register}"
                );
            }
        }
    }
}
