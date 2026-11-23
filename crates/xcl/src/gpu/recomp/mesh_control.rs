//! Conservative decoding of a verified scalar/mask subset into bounded mesh control IR.
//!
//! This is an eligibility layer only. It is not selected by the real PSO path.

use std::collections::BTreeMap;

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_dispatch::{ControlOp, MaskBinaryOp, MaskRegister};

enum Pending {
    Ready(ControlOp),
    BranchScc {
        set: bool,
        target_pc: u64,
    },
    BranchMask {
        mask: MaskRegister,
        zero: bool,
        target_pc: u64,
    },
    Jump {
        target_pc: u64,
    },
}

fn field(decoded: &Decoded, name: &str) -> Result<u64, String> {
    rdna2::field_value(decoded.enc, &decoded.words, name)
        .ok_or_else(|| format!("{} lacks {name}", decoded.inst.name))
}

fn scalar_register(value: u64) -> Result<u8, String> {
    u8::try_from(value)
        .ok()
        .filter(|&register| register <= 105)
        .ok_or_else(|| format!("unsupported scalar register {value}"))
}

fn scalar_pair(value: u64) -> Result<u8, String> {
    let register = scalar_register(value)?;
    if register >= 105 {
        return Err("scalar pair crosses the supported register file".into());
    }
    Ok(register)
}

fn mask_register(value: u64) -> Result<MaskRegister, String> {
    match value {
        106 => Ok(MaskRegister::Vcc),
        126 => Ok(MaskRegister::Exec),
        _ => Err(format!("unsupported mask register {value}")),
    }
}

fn inline_u32(decoded: &Decoded, name: &str) -> Result<u32, String> {
    let value = field(decoded, name)?;
    match value {
        128 => Ok(0),
        129..=192 => Ok((value - 128) as u32),
        193..=208 => Ok((-(value as i32 - 192)) as u32),
        255 => decoded
            .literal
            .ok_or_else(|| "missing scalar literal".into()),
        _ => Err(format!("{name} is not a supported inline constant")),
    }
}

fn binary_op(name: &str) -> Option<MaskBinaryOp> {
    match name {
        "S_AND_B64" => Some(MaskBinaryOp::And),
        "S_OR_B64" => Some(MaskBinaryOp::Or),
        "S_XOR_B64" => Some(MaskBinaryOp::Xor),
        "S_ANDN2_B64" => Some(MaskBinaryOp::AndNot),
        _ => None,
    }
}

fn lower_one(decoded: &Decoded) -> Result<Pending, String> {
    let name = decoded.inst.name;
    if let Some(op) = binary_op(name) {
        let lhs = mask_register(field(decoded, "SSRC0")?)?;
        let rhs = mask_register(field(decoded, "SSRC1")?)?;
        let dst = field(decoded, "SDST")?;
        return Ok(Pending::Ready(match mask_register(dst) {
            Ok(dst) => ControlOp::MaskBinary { dst, lhs, rhs, op },
            Err(_) => ControlOp::MaskBinaryScalar {
                dst: scalar_pair(dst)?,
                lhs,
                rhs,
                op,
            },
        }));
    }
    let pending = match name {
        "S_MOV_B64" => {
            let (dst, src) = (field(decoded, "SDST")?, field(decoded, "SSRC0")?);
            Pending::Ready(match (mask_register(dst), mask_register(src)) {
                (Ok(dst), Ok(src)) => ControlOp::CopyMask { dst, src },
                (Ok(dst), Err(_)) => ControlOp::ScalarToMask {
                    dst,
                    src: scalar_pair(src)?,
                },
                (Err(_), Ok(src)) => ControlOp::MaskToScalar {
                    dst: scalar_pair(dst)?,
                    src,
                },
                (Err(_), Err(_)) => {
                    return Err("scalar-to-scalar S_MOV_B64 is not in the verified subset".into())
                }
            })
        }
        "S_AND_SAVEEXEC_B64" => Pending::Ready(ControlOp::SaveExecAndMask {
            mask: mask_register(field(decoded, "SSRC0")?)?,
            dst: scalar_pair(field(decoded, "SDST")?)?,
        }),
        "S_ANDN2_WREXEC_B64" => Pending::Ready(ControlOp::WrExecAndNotMask {
            mask: mask_register(field(decoded, "SSRC0")?)?,
            dst: scalar_pair(field(decoded, "SDST")?)?,
        }),
        "S_MOV_B32" => Pending::Ready(ControlOp::ScalarSet {
            register: scalar_register(field(decoded, "SDST")?)?,
            value: inline_u32(decoded, "SSRC0")?,
        }),
        "S_ADD_U32" => {
            let dst = scalar_register(field(decoded, "SDST")?)?;
            if scalar_register(field(decoded, "SSRC0")?)? != dst {
                return Err("S_ADD_U32 requires an in-place destination".into());
            }
            Pending::Ready(ControlOp::ScalarAddCarry {
                register: dst,
                value: inline_u32(decoded, "SSRC1")?,
            })
        }
        "S_CMP_LT_U32" => Pending::Ready(ControlOp::ScalarCompareLess {
            register: scalar_register(field(decoded, "SSRC0")?)?,
            value: inline_u32(decoded, "SSRC1")?,
        }),
        "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => Pending::BranchScc {
            set: name.ends_with("SCC1"),
            target_pc: rdna2::branch_target(decoded).ok_or("missing direct branch target")?,
        },
        "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => Pending::BranchMask {
            mask: MaskRegister::Exec,
            zero: name.ends_with("EXECZ"),
            target_pc: rdna2::branch_target(decoded).ok_or("missing direct branch target")?,
        },
        "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" => Pending::BranchMask {
            mask: MaskRegister::Vcc,
            zero: name.ends_with("VCCZ"),
            target_pc: rdna2::branch_target(decoded).ok_or("missing direct branch target")?,
        },
        "S_BRANCH" => Pending::Jump {
            target_pc: rdna2::branch_target(decoded).ok_or("missing direct branch target")?,
        },
        "S_ENDPGM" => Pending::Ready(ControlOp::Complete),
        _ => return Err(format!("unsupported control opcode {name}")),
    };
    Ok(pending)
}

fn resolve_target(target_pc: u64, pcs: &BTreeMap<u64, u32>) -> Result<u32, String> {
    if target_pc & 3 != 0 {
        return Err(format!("misaligned branch target {target_pc:#x}"));
    }
    pcs.get(&target_pc)
        .copied()
        .ok_or_else(|| format!("branch target {target_pc:#x} is outside the program"))
}

/// Lowers one already-decoded control instruction using the complete program's
/// instruction-boundary map. This adapter lets mixed-program validation decode
/// bytecode once without weakening the standalone control validator.
pub fn lower_decoded_one(decoded: &Decoded, pcs: &BTreeMap<u64, u32>) -> Result<ControlOp, String> {
    Ok(match lower_one(decoded)? {
        Pending::Ready(op) => op,
        Pending::BranchScc { set, target_pc } => ControlOp::BranchScc {
            set,
            target: resolve_target(target_pc, pcs)?,
        },
        Pending::BranchMask {
            mask,
            zero,
            target_pc,
        } => ControlOp::BranchMaskZero {
            mask,
            zero,
            target: resolve_target(target_pc, pcs)?,
        },
        Pending::Jump { target_pc } => ControlOp::Jump {
            target: resolve_target(target_pc, pcs)?,
        },
    })
}

pub fn lower(code: &[u8]) -> Result<Vec<ControlOp>, String> {
    if code.is_empty() {
        return Err("empty control program".into());
    }
    let report = super::mesh_wave::analyze(code);
    if report.undecoded_at.is_some() || report.decoded_bytes != code.len() {
        return Err("incomplete control program decode".into());
    }
    if !report.lane_reads.is_empty()
        || !report.mbcnt_lo.is_empty()
        || !report.mbcnt_hi.is_empty()
        || !report.allocation_messages.is_empty()
        || !report.barriers.is_empty()
        || !report.exports.is_empty()
    {
        return Err("program requires an unlowered logical-wave operation".into());
    }

    let mut decoded = Vec::with_capacity(report.instructions);
    let mut offset = 0usize;
    while offset < code.len() {
        let instruction = rdna2::decode(&code[offset..], offset as u64)
            .map_err(|error| format!("decode at {offset:#x}: {error:?}"))?;
        offset = offset
            .checked_add(instruction.len)
            .ok_or("control program size overflow")?;
        decoded.push(instruction);
    }
    let pcs: BTreeMap<u64, u32> = decoded
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.pc, index as u32))
        .collect();
    let lowered: Vec<ControlOp> = decoded
        .iter()
        .map(|instruction| lower_decoded_one(instruction, &pcs))
        .collect::<Result<_, _>>()?;
    if !lowered.iter().any(|op| matches!(op, ControlOp::Complete)) {
        return Err("control program has no explicit completion".into());
    }
    Ok(lowered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    fn one(name: &str, encoding: &str, fields: &[(&str, u64)], literal: Option<u32>) -> Vec<u8> {
        encode(name, encoding, fields, literal).unwrap()
    }

    #[test]
    fn rejects_incomplete_unknown_and_unsupported_programs() {
        assert!(lower(&[1, 2]).is_err());
        assert!(lower(&0x0000_003fu32.to_le_bytes()).is_err());
        assert!(lower(&one(
            "V_MOV_B32",
            "ENC_VOP1",
            &[("VDST", 0), ("SRC0", 128)],
            None
        ))
        .is_err());
        assert!(lower(&one("S_BARRIER", "ENC_SOPP", &[], None)).is_err());
        assert!(lower(&one(
            "V_MBCNT_LO_U32_B32",
            "ENC_VOP3",
            &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)],
            None
        ))
        .is_err());
        assert!(lower(&one(
            "V_READFIRSTLANE_B32",
            "ENC_VOP1",
            &[("VDST", 4), ("SRC0", 258)],
            None,
        ))
        .is_err());
        assert!(lower(&one("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None)).is_err());
        assert!(lower(&one(
            "EXP",
            "ENC_EXP",
            &[("TGT", 12), ("EN", 1), ("VSRC0", 0)],
            None,
        ))
        .is_err());
        assert!(lower(&one(
            "BUFFER_LOAD_DWORD",
            "ENC_MUBUF",
            &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
            None,
        ))
        .is_err());
        assert!(lower(&one("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)], None)).is_err());
    }

    #[test]
    fn target_resolution_is_fail_closed() {
        let pcs = BTreeMap::from([(0u64, 0u32), (4, 1)]);
        assert!(resolve_target(2, &pcs).unwrap_err().contains("misaligned"));
        assert!(resolve_target(8, &pcs).unwrap_err().contains("outside"));
    }

    #[test]
    fn rejects_out_of_range_direct_target() {
        let code = one("S_BRANCH", "ENC_SOPP", &[("SIMM16", 4)], None);
        assert!(lower(&code).unwrap_err().contains("outside"));
    }

    #[test]
    #[ignore]
    fn private_mesh_programs_remain_ineligible() {
        let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut checked = 0usize;
        for record in crate::gpu::pack::scan(&data, 0x40) {
            let Some(blob) = crate::gpu::pack::inflate(&data, &record) else {
                continue;
            };
            let Some(shader) = crate::gpu::xbox::parse(&blob) else {
                continue;
            };
            for fssm in shader
                .fssm
                .iter()
                .filter(|fssm| fssm.stages.first().is_some_and(|stage| stage.hw == 7))
            {
                for program in &fssm.programs {
                    checked += 1;
                    let code = &fssm.data[program.offset..program.offset + program.len];
                    assert!(
                        lower(code).is_err(),
                        "private mesh program unexpectedly became eligible"
                    );
                }
            }
        }
        assert!(
            checked >= 7,
            "private mesh survey did not cover the expected cohort"
        );
    }
}
