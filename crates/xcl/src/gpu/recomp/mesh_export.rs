//! Synthetic composition of logical-wave reductions into a bounded mesh record.
//!
//! This is an end-to-end validation path only. It does not participate in decoded
//! program eligibility or real mesh PSO selection.

use super::mesh_dispatch::{GROUP_THREADS, LOGICAL_WAVE_LANES};
use super::mesh_reduce::GsAllocRequest;
use super::mesh_vgpr::VgprScratchLayout;
use super::MESH_HEADER;

pub const MAX_SYNTHETIC_VERTICES: u32 = 256;
pub const MAX_SYNTHETIC_PRIMITIVES: u32 = 256;
const VERTEX_DWORDS: u32 = 4;
pub const PARAMETER_TARGETS: u32 = 32;
pub const PARAMETER_DWORDS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyntheticRecordLayout {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub record_dwords: u32,
}

impl SyntheticRecordLayout {
    pub fn new(max_vertices: u32, max_primitives: u32) -> Result<Self, &'static str> {
        if max_vertices == 0
            || max_vertices > MAX_SYNTHETIC_VERTICES
            || max_primitives == 0
            || max_primitives > MAX_SYNTHETIC_PRIMITIVES
        {
            return Err("synthetic mesh record limits are invalid");
        }
        let record_dwords = max_vertices
            .checked_mul(VERTEX_DWORDS)
            .and_then(|vertices| vertices.checked_add(MESH_HEADER as u32))
            .and_then(|words| words.checked_add(max_primitives))
            .and_then(|words| {
                max_vertices
                    .checked_mul(PARAMETER_TARGETS)?
                    .checked_mul(PARAMETER_DWORDS)?
                    .checked_add(words)
            })
            .ok_or("synthetic mesh record size overflow")?;
        Ok(Self {
            max_vertices,
            max_primitives,
            record_dwords,
        })
    }

    pub fn triangle_base(self) -> u32 {
        MESH_HEADER as u32 + self.max_vertices * VERTEX_DWORDS
    }

    /// Raw parameter-export storage used only by the bounded validation adapter.
    /// It intentionally does not model stage linkage or interpolation.
    pub fn parameter_base(self) -> u32 {
        self.triangle_base() + self.max_primitives
    }

    pub fn parameter_word(
        self,
        target: u8,
        vertex: u32,
        component: u32,
    ) -> Result<u32, &'static str> {
        if !(32..=63).contains(&target) {
            return Err("synthetic parameter target is out of bounds");
        }
        if vertex >= self.max_vertices || component >= PARAMETER_DWORDS {
            return Err("synthetic parameter index is out of bounds");
        }
        let attribute = u32::from(target - 32);
        self.parameter_base()
            .checked_add(
                vertex
                    .checked_mul(PARAMETER_TARGETS * PARAMETER_DWORDS)
                    .ok_or("synthetic parameter address overflow")?,
            )
            .and_then(|word| word.checked_add(attribute * PARAMETER_DWORDS))
            .and_then(|word| word.checked_add(component))
            .ok_or("synthetic parameter address overflow")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyntheticGroupConfig {
    pub exec: [u64; 2],
    pub vertices: [u64; 2],
    pub primitives: [u64; 2],
    pub invalid_primitives: [u64; 2],
}

fn bit(mask: u64, lane: usize) -> bool {
    mask & (1u64 << lane) != 0
}

fn position(index: u32) -> [u32; 4] {
    let xy = match index % 3 {
        0 => [-0.75f32, -0.75],
        1 => [0.0, 0.75],
        _ => [0.75, -0.75],
    };
    [xy[0].to_bits(), xy[1].to_bits(), 0, 1.0f32.to_bits()]
}

pub fn build_reference_record(
    layout: SyntheticRecordLayout,
    config: SyntheticGroupConfig,
) -> Vec<u32> {
    let mut record = vec![0u32; layout.record_dwords as usize];
    let mut vertex_offsets = [[0u32; 64]; 2];
    let mut primitive_offsets = [[0u32; 64]; 2];
    let mut vertex_totals = [0u32; 2];
    let mut primitive_totals = [0u32; 2];
    for wave in 0..2usize {
        for lane in 0..64usize {
            vertex_offsets[wave][lane] = vertex_totals[wave];
            primitive_offsets[wave][lane] = primitive_totals[wave];
            if bit(config.exec[wave] & config.vertices[wave], lane) {
                vertex_totals[wave] = vertex_totals[wave].wrapping_add(1);
            }
            if bit(config.exec[wave] & config.primitives[wave], lane) {
                primitive_totals[wave] = primitive_totals[wave].wrapping_add(1);
            }
        }
    }
    let allocation = GsAllocRequest::clamped(
        vertex_totals[0].wrapping_add(vertex_totals[1]),
        primitive_totals[0].wrapping_add(primitive_totals[1]),
    );
    record[0] = allocation.vertices.min(layout.max_vertices);
    record[1] = allocation.primitives.min(layout.max_primitives);
    for wave in 0..2usize {
        for lane in 0..64usize {
            let vertex_offset =
                vertex_offsets[wave][lane] + if wave == 1 { vertex_totals[0] } else { 0 };
            if bit(config.exec[wave] & config.vertices[wave], lane)
                && vertex_offset < layout.max_vertices
            {
                let base = MESH_HEADER + vertex_offset as usize * VERTEX_DWORDS as usize;
                record[base..base + 4].copy_from_slice(&position(vertex_offset));
            }
            let primitive_offset =
                primitive_offsets[wave][lane] + if wave == 1 { primitive_totals[0] } else { 0 };
            if bit(config.exec[wave] & config.primitives[wave], lane)
                && primitive_offset < layout.max_primitives
            {
                let invalid = bit(config.invalid_primitives[wave], lane) || record[0] == 0;
                let packed = if invalid {
                    0x8000_0000
                } else {
                    let base = primitive_offset.wrapping_mul(3) % record[0];
                    let b = (base + 1) % record[0];
                    let c = (base + 2) % record[0];
                    base | (b << 10) | (c << 20)
                };
                record[layout.triangle_base() as usize + primitive_offset as usize] = packed;
            }
        }
    }
    record
}

pub fn build_reference_record_into(
    destination: &mut [u32],
    offset: usize,
    layout: SyntheticRecordLayout,
    config: SyntheticGroupConfig,
) -> Result<(), &'static str> {
    let end = offset
        .checked_add(layout.record_dwords as usize)
        .ok_or("synthetic mesh record destination overflow")?;
    let target = destination
        .get_mut(offset..end)
        .ok_or("synthetic mesh record destination is too small")?;
    target.copy_from_slice(&build_reference_record(layout, config));
    Ok(())
}

fn select_mask(
    configs: &[SyntheticGroupConfig; 2],
    field: fn(&SyntheticGroupConfig) -> [u64; 2],
    high: bool,
) -> String {
    let shift = if high { 32 } else { 0 };
    let value = |group: usize, wave: usize| (field(&configs[group])[wave] >> shift) as u32;
    format!(
        "gid.x == 0u ? (wave == 0u ? {:#x}u : {:#x}u) : (wave == 0u ? {:#x}u : {:#x}u)",
        value(0, 0),
        value(0, 1),
        value(1, 0),
        value(1, 1)
    )
}

pub fn export_validation_hlsl(
    scratch_layout: VgprScratchLayout,
    record_layout: SyntheticRecordLayout,
    configs: [SyntheticGroupConfig; 2],
) -> Result<String, &'static str> {
    scratch_layout.validate_for_groups(2)?;
    if scratch_layout.vgpr_count < 4 {
        return Err("synthetic export requires four VGPRs");
    }
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer res : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads({GROUP_THREADS}, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint wave = gi / {LOGICAL_WAVE_LANES}u;
    uint lane = gi % {LOGICAL_WAVE_LANES}u;
    uint record_word = gid.x * {record_dwords}u;
    uint scratch_base = gid.x * io[0];
    uint vgpr_base = scratch_base + {vgpr_offset}u;
    uint lane_base = vgpr_base + gi * {vgpr_stride}u;
    uint mailbox_base = scratch_base + {mailbox_offset}u;
    uint2 exec = uint2({exec_low}, {exec_high});
    uint2 vertex_mask = uint2({vertex_low}, {vertex_high});
    uint2 primitive_mask = uint2({primitive_low}, {primitive_high});
    uint2 invalid_mask = uint2({invalid_low}, {invalid_high});
    uint lane_bit = lane < 32u ? (1u << lane) : (1u << (lane - 32u));
    uint exec_half = lane < 32u ? exec.x : exec.y;
    uint vertex_half = lane < 32u ? vertex_mask.x : vertex_mask.y;
    uint primitive_half = lane < 32u ? primitive_mask.x : primitive_mask.y;
    uint invalid_half = lane < 32u ? invalid_mask.x : invalid_mask.y;
    uint vertex_contribution = ((exec_half & vertex_half & lane_bit) != 0u) ? 1u : 0u;
    uint primitive_contribution = ((exec_half & primitive_half & lane_bit) != 0u) ? 1u : 0u;
    uint invalid = ((invalid_half & lane_bit) != 0u) ? 1u : 0u;
    for (uint word = gi; word < {record_dwords}u; word += {GROUP_THREADS}u)
        res.Store((record_word + word) * 4u, 0u);
    scratch.Store(lane_base, vertex_contribution);
    scratch.Store(lane_base + 4u, primitive_contribution);
    scratch.Store(lane_base + 8u, invalid);
    scratch.Store(lane_base + 12u, gi);
    DeviceMemoryBarrierWithGroupSync();
    uint vertex_prefix = 0u, primitive_prefix = 0u;
    uint vertex_total = 0u, primitive_total = 0u;
    [unroll]
    for (uint other = 0u; other < 64u; ++other) {{
        uint other_base = vgpr_base + (wave * 64u + other) * {vgpr_stride}u;
        uint vc = scratch.Load(other_base);
        uint pc = scratch.Load(other_base + 4u);
        vertex_total += vc;
        primitive_total += pc;
        if (other < lane) {{ vertex_prefix += vc; primitive_prefix += pc; }}
    }}
    if (lane == 0u) {{
        scratch.Store(mailbox_base + wave * 8u, vertex_total);
        scratch.Store(mailbox_base + wave * 8u + 4u, primitive_total);
    }}
    DeviceMemoryBarrierWithGroupSync();
    uint prior_vertices = wave == 0u ? 0u : scratch.Load(mailbox_base);
    uint prior_primitives = wave == 0u ? 0u : scratch.Load(mailbox_base + 4u);
    uint vertex_offset = prior_vertices + vertex_prefix;
    uint primitive_offset = prior_primitives + primitive_prefix;
    uint all_vertices = scratch.Load(mailbox_base) + scratch.Load(mailbox_base + 8u);
    uint all_primitives = scratch.Load(mailbox_base + 4u) + scratch.Load(mailbox_base + 12u);
    uint packed_allocation = min(all_vertices, 0x3ffu) | (min(all_primitives, 0x7ffu) << 12u);
    uint clamped_vertices = min(packed_allocation & 0x3ffu, {max_vertices}u);
    uint clamped_primitives = min((packed_allocation >> 12u) & 0x7ffu, {max_primitives}u);
    if (gi == 0u) {{
        res.Store(record_word * 4u, clamped_vertices);
        res.Store((record_word + 1u) * 4u, clamped_primitives);
    }}
    if (vertex_contribution != 0u && vertex_offset < {max_vertices}u) {{
        uint vertex_word = record_word + {header}u + vertex_offset * 4u;
        uint corner = vertex_offset % 3u;
        float2 xy = corner == 0u ? float2(-0.75, -0.75) : (corner == 1u ? float2(0.0, 0.75) : float2(0.75, -0.75));
        res.Store4(vertex_word * 4u, asuint(float4(xy, 0.0, 1.0)));
    }}
    if (primitive_contribution != 0u && primitive_offset < {max_primitives}u) {{
        uint packed = 0x80000000u;
        if (invalid == 0u && clamped_vertices != 0u) {{
            uint a = (primitive_offset * 3u) % clamped_vertices;
            uint b = (a + 1u) % clamped_vertices;
            uint c = (a + 2u) % clamped_vertices;
            packed = a | (b << 10u) | (c << 20u);
        }}
        res.Store((record_word + {triangle_base}u + primitive_offset) * 4u, packed);
    }}
    DeviceMemoryBarrierWithGroupSync();
}}
"#,
        record_dwords = record_layout.record_dwords,
        vgpr_offset = super::mesh_dispatch::MIN_RECORD_BYTES,
        vgpr_stride = scratch_layout.vgpr_count * 4,
        mailbox_offset = scratch_layout.scratch.mailbox_offset(0, 0)?,
        exec_low = select_mask(&configs, |c| c.exec, false),
        exec_high = select_mask(&configs, |c| c.exec, true),
        vertex_low = select_mask(&configs, |c| c.vertices, false),
        vertex_high = select_mask(&configs, |c| c.vertices, true),
        primitive_low = select_mask(&configs, |c| c.primitives, false),
        primitive_high = select_mask(&configs, |c| c.primitives, true),
        invalid_low = select_mask(&configs, |c| c.invalid_primitives, false),
        invalid_high = select_mask(&configs, |c| c.invalid_primitives, true),
        max_vertices = record_layout.max_vertices,
        max_primitives = record_layout.max_primitives,
        header = MESH_HEADER,
        triangle_base = record_layout.triangle_base(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configs() -> [SyntheticGroupConfig; 2] {
        [
            SyntheticGroupConfig {
                exec: [u64::MAX, u64::MAX],
                vertices: [(1u64 << 0) | (1u64 << 31), (1u64 << 63)],
                primitives: [0, 1u64 << 32],
                invalid_primitives: [0, 0],
            },
            SyntheticGroupConfig {
                exec: [u64::MAX, u64::MAX],
                vertices: [u64::MAX, u64::MAX],
                primitives: [u64::MAX, u64::MAX],
                invalid_primitives: [(1u64 << 1) | (1u64 << 31), 1u64 << 63],
            },
        ]
    }

    #[test]
    fn reference_builder_clamps_and_preserves_guards() {
        let layout = SyntheticRecordLayout::new(8, 4).unwrap();
        assert!(SyntheticRecordLayout::new(0, 1).is_err());
        assert!(SyntheticRecordLayout::new(u32::MAX, u32::MAX).is_err());
        assert!(SyntheticRecordLayout::new(256, 256).is_ok());
        assert_eq!(
            layout.parameter_word(32, 0, 0).unwrap(),
            layout.parameter_base()
        );
        assert_eq!(
            layout.parameter_word(63, 7, 3).unwrap(),
            layout.record_dwords - 1
        );
        assert!(layout.parameter_word(31, 0, 0).is_err());
        assert!(layout.parameter_word(32, 8, 0).is_err());
        let config = configs()[1];
        let record = build_reference_record(layout, config);
        assert_eq!((record[0], record[1]), (8, 4));
        assert_eq!(record[layout.triangle_base() as usize + 1], 0x8000_0000);
        let guard = 0xdead_beefu32;
        let mut destination = vec![guard; layout.record_dwords as usize + 2];
        build_reference_record_into(&mut destination, 1, layout, config).unwrap();
        assert_eq!(destination[0], guard);
        assert_eq!(*destination.last().unwrap(), guard);
        assert!(build_reference_record_into(&mut destination[..4], 1, layout, config).is_err());
    }

    #[test]
    fn gpu_export_matches_full_records_and_renders() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let record_layout = SyntheticRecordLayout::new(8, 4).unwrap();
        let scratch_layout = VgprScratchLayout::new(4, 2).unwrap();
        let configs = configs();
        let hlsl = export_validation_hlsl(scratch_layout, record_layout, configs).unwrap();
        assert_eq!(
            hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
            3
        );
        assert!(!hlsl.contains("groupshared") && !hlsl.contains("xcl_lds"));
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let interface = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: record_layout.max_vertices,
            max_primitives: record_layout.max_primitives,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&interface),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let color = [0.25f32, 0.5, 0.75, 1.0];
        let pixel = super::super::dxc::compile(
            "float4 main() : SV_Target0 { return float4(0.25, 0.5, 0.75, 1.0); }",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (pixels, records, _) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            2,
            record_layout.record_dwords as usize,
            &[
                scratch_layout.scratch.record_stride_bytes,
                record_layout.record_dwords * 4,
            ],
            &[],
            32,
            [0.0; 4],
            scratch_layout.scratch,
        )
        .unwrap();
        let expected: Vec<u32> = configs
            .iter()
            .flat_map(|config| build_reference_record(record_layout, *config))
            .collect();
        assert_eq!(records, expected);
        assert!(pixels.iter().any(|pixel| *pixel == color));
        assert!(pixels.iter().any(|pixel| *pixel == [0.0; 4]));
    }
}
