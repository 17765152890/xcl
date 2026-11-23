//! Bounded scalar/control execution for the first surveyed dispatcher family.
//!
//! This synthetic IR is deliberately not connected to real-program eligibility.

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_dispatch::{
    validate_step_cap, LogicalWaveState, LogicalWaveStatus, ScratchLayout, GROUP_THREADS,
    SCALAR_REGISTERS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand32 {
    Register(u8),
    Constant(u32),
    VccZero,
    ExecZero,
    Scc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand64 {
    Register(u8),
    Constant(u64),
    Scc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarBinaryKind {
    AddI32,
    SubI32,
    MulI32,
    MulHiU32,
    MinU32,
    And,
    Or,
    Xor,
    Lshl,
    Lshr,
    AshrI32,
    BfeU32,
    BfeI32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarBinary64Kind {
    And,
    AndNot,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageAOp {
    AddU32 { dst: u8, lhs: Operand32, rhs: Operand32, carry_in: bool },
    MulImmediateI32 { dst: u8, value: i16 },
    GetPc64 { dst: u8, next_pc: u64 },
    BitSet1 {
        dst: u8,
        bit: Operand32,
    },
    Not32 {
        dst: u8,
        source: Operand32,
    },
    BitReverse32 {
        dst: u8,
        source: Operand32,
    },
    CountBits32 {
        dst: u8,
        source: Operand32,
    },
    CountBits64 {
        dst: u8,
        source: Operand64,
    },
    Binary {
        kind: ScalarBinaryKind,
        dst: u8,
        lhs: Operand32,
        rhs: Operand32,
    },
    Binary64 {
        kind: ScalarBinary64Kind,
        dst: u8,
        lhs: Operand64,
        rhs: Operand64,
    },
    CompareEq {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareGtU32 {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareLtU32 {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareLeU32 {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareGeU32 {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareNeU32 {
        lhs: Operand32,
        rhs: Operand32,
    },
    CompareLtI32 { lhs: Operand32, rhs: Operand32 },
    CompareGtI32 { lhs: Operand32, rhs: Operand32 },
    CompareGeI32 { lhs: Operand32, rhs: Operand32 },
    Select32 {
        dst: u8,
        when_set: Operand32,
        when_clear: Operand32,
    },
    MoveImmediate {
        dst: u8,
        value: u32,
    },
    Move32 {
        dst: u8,
        source: Operand32,
    },
    Move64 {
        dst: u8,
        source: Operand64,
    },
    SaveExecAnd64 {
        dst: u8,
        source: Operand64,
    },
    SaveExecOr64 {
        dst: u8,
        source: Operand64,
    },
    WrExecAndNot64 {
        dst: u8,
        source: Operand64,
    },
    /// A hardware-register write that is observationally inert for the bounded
    /// instruction subset: the mode value matches its arithmetic assumptions,
    /// and flat-memory instructions are not admitted. Other writes fail closed.
    ConfigureHardwareRegister {
        register: u8,
        offset: u8,
        width: u8,
        value: u32,
    },
    Select64 {
        dst: u8,
        when_set: Operand64,
        when_clear: Operand64,
    },
    NoEffect,
    Barrier,
    Jump(u32),
    Complete,
}

fn field(decoded: &Decoded, name: &str) -> Result<u64, String> {
    rdna2::field_value(decoded.enc, &decoded.words, name)
        .ok_or_else(|| format!("{} lacks {name}", decoded.inst.name))
}

fn inline32(decoded: &Decoded, value: u64) -> Option<u32> {
    match value {
        128 => Some(0),
        129..=192 => Some((value - 128) as u32),
        193..=208 => Some((-(value as i32 - 192)) as u32),
        240 => Some(0.5f32.to_bits()),
        241 => Some((-0.5f32).to_bits()),
        242 => Some(1.0f32.to_bits()),
        243 => Some((-1.0f32).to_bits()),
        244 => Some(2.0f32.to_bits()),
        245 => Some((-2.0f32).to_bits()),
        246 => Some(4.0f32.to_bits()),
        247 => Some((-4.0f32).to_bits()),
        248 => Some((1.0f32 / (2.0 * std::f32::consts::PI)).to_bits()),
        255 => decoded.literal,
        _ => None,
    }
}

fn operand32(decoded: &Decoded, name: &str) -> Result<Operand32, String> {
    let value = field(decoded, name)?;
    match value {
        0..=105 | 106 | 107 | 108..=124 | 126 | 127 => Ok(Operand32::Register(value as u8)),
        125 => Ok(Operand32::Constant(0)),
        251 => Ok(Operand32::VccZero),
        252 => Ok(Operand32::ExecZero),
        253 => Ok(Operand32::Scc),
        _ => inline32(decoded, value)
            .map(Operand32::Constant)
            .ok_or_else(|| format!("unsupported {name} selector {value}")),
    }
}

fn operand64(decoded: &Decoded, name: &str) -> Result<Operand64, String> {
    let value = field(decoded, name)?;
    match value {
        0..=123 | 126 => Ok(Operand64::Register(value as u8)),
        125 => Ok(Operand64::Constant(0)),
        128..=192 => Ok(Operand64::Constant(value - 128)),
        193..=208 => Ok(Operand64::Constant((-(value as i64 - 192)) as u64)),
        240..=247 => {
            let values = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
            Ok(Operand64::Constant(
                (values[(value - 240) as usize] as f64).to_bits(),
            ))
        }
        248 => Ok(Operand64::Constant(0x3fc4_5f30_6dc9_c882)),
        253 => Ok(Operand64::Scc),
        255 => decoded
            .literal
            .map(|literal| Operand64::Constant(literal as u64))
            .ok_or_else(|| format!("missing literal for {name}")),
        _ => Err(format!("unsupported {name} selector {value}")),
    }
}

pub fn lower_stage_a_one(decoded: &Decoded) -> Result<StageAOp, String> {
    let destination = |name: &str| {
        u8::try_from(field(decoded, name)?)
            .ok()
            .filter(|&value| value < SCALAR_REGISTERS as u8)
            .ok_or_else(|| "scalar destination is out of bounds".to_string())
    };
    let pair_destination = || {
        u8::try_from(field(decoded, "SDST")?)
            .ok()
            .filter(|&value| value <= 123 || value == 126)
            .ok_or_else(|| "scalar pair destination is out of bounds".to_string())
    };
    let ordinary_pair_destination = || {
        u8::try_from(field(decoded, "SDST")?)
            .ok()
            .filter(|&value| value < 105 || value == 106)
            .ok_or_else(|| "saveexec destination is outside the scalar register file".to_string())
    };
    if matches!(decoded.inst.name, "S_ADD_U32" | "S_ADDC_U32") {
        return Ok(StageAOp::AddU32 {
            dst: destination("SDST")?,
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
            carry_in: decoded.inst.name == "S_ADDC_U32",
        });
    }
    let binary = match decoded.inst.name {
        "S_ADD_I32" => Some(ScalarBinaryKind::AddI32),
        "S_SUB_I32" => Some(ScalarBinaryKind::SubI32),
        "S_MUL_I32" => Some(ScalarBinaryKind::MulI32),
        "S_MUL_HI_U32" => Some(ScalarBinaryKind::MulHiU32),
        "S_MIN_U32" => Some(ScalarBinaryKind::MinU32),
        "S_AND_B32" => Some(ScalarBinaryKind::And),
        "S_OR_B32" => Some(ScalarBinaryKind::Or),
        "S_XOR_B32" => Some(ScalarBinaryKind::Xor),
        "S_LSHL_B32" => Some(ScalarBinaryKind::Lshl),
        "S_LSHR_B32" => Some(ScalarBinaryKind::Lshr),
        "S_ASHR_I32" => Some(ScalarBinaryKind::AshrI32),
        "S_BFE_U32" => Some(ScalarBinaryKind::BfeU32),
        "S_BFE_I32" => Some(ScalarBinaryKind::BfeI32),
        _ => None,
    };
    if let Some(kind) = binary {
        let dst = destination("SDST")?;
        return Ok(StageAOp::Binary {
            kind,
            dst,
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        });
    }
    let binary64 = match decoded.inst.name {
        "S_AND_B64" => Some(ScalarBinary64Kind::And),
        "S_ANDN2_B64" => Some(ScalarBinary64Kind::AndNot),
        "S_OR_B64" => Some(ScalarBinary64Kind::Or),
        "S_XOR_B64" => Some(ScalarBinary64Kind::Xor),
        _ => None,
    };
    if let Some(kind) = binary64 {
        return Ok(StageAOp::Binary64 {
            kind,
            dst: pair_destination()?,
            lhs: operand64(decoded, "SSRC0")?,
            rhs: operand64(decoded, "SSRC1")?,
        });
    }
    if decoded.inst.name == "S_BITSET1_B32" {
        return Ok(StageAOp::BitSet1 {
            dst: destination("SDST")?,
            bit: operand32(decoded, "SSRC0")?,
        });
    }
    if decoded.inst.name == "S_NOT_B32" {
        return Ok(StageAOp::Not32 {
            dst: destination("SDST")?,
            source: operand32(decoded, "SSRC0")?,
        });
    }
    if decoded.inst.name == "S_BREV_B32" {
        return Ok(StageAOp::BitReverse32 {
            dst: destination("SDST")?,
            source: operand32(decoded, "SSRC0")?,
        });
    }
    if decoded.inst.name == "S_BCNT1_I32_B32" {
        return Ok(StageAOp::CountBits32 {
            dst: destination("SDST")?,
            source: operand32(decoded, "SSRC0")?,
        });
    }
    if decoded.inst.name == "S_BCNT1_I32_B64" {
        return Ok(StageAOp::CountBits64 {
            dst: destination("SDST")?,
            source: operand64(decoded, "SSRC0")?,
        });
    }
    if decoded.inst.name == "S_ADDK_I32" {
        let dst = destination("SDST")?;
        return Ok(StageAOp::Binary {
            kind: ScalarBinaryKind::AddI32,
            dst,
            lhs: Operand32::Register(dst),
            rhs: Operand32::Constant(field(decoded, "SIMM16")? as u16 as i16 as i32 as u32),
        });
    }
    if decoded.inst.name == "S_MULK_I32" {
        return Ok(StageAOp::MulImmediateI32 {
            dst: destination("SDST")?,
            value: field(decoded, "SIMM16")? as u16 as i16,
        });
    }
    if decoded.inst.name == "S_GETPC_B64" {
        return Ok(StageAOp::GetPc64 {
            dst: pair_destination()?,
            next_pc: decoded.pc.checked_add(decoded.len as u64)
                .ok_or_else(|| "next program counter overflows".to_string())?,
        });
    }
    if matches!(decoded.inst.name, "S_CMPK_EQ_I32" | "S_CMPK_GT_U32") {
        let lhs = Operand32::Register(destination("SDST")?);
        let immediate = field(decoded, "SIMM16")? as u16;
        return Ok(if decoded.inst.name == "S_CMPK_EQ_I32" {
            StageAOp::CompareEq {
                lhs,
                rhs: Operand32::Constant(immediate as i16 as i32 as u32),
            }
        } else {
            StageAOp::CompareGtU32 {
                lhs,
                rhs: Operand32::Constant(immediate as u32),
            }
        });
    }
    if decoded.inst.name == "S_SETREG_IMM32_B32" {
        let encoded = field(decoded, "SIMM16")? as u16;
        let register = (encoded & 0x3f) as u8;
        let offset = ((encoded >> 6) & 0x1f) as u8;
        let width = (((encoded >> 11) & 0x1f) + 1) as u8;
        let value = decoded
            .literal
            .ok_or_else(|| "setreg immediate is missing its literal".to_string())?;
        let supported = matches!(
            (register, offset, width, value),
            (1, 2, 2, 0) | (20, 0, 32, 0) | (21, 0, 32, 1024)
        );
        if !supported {
            return Err(format!(
                "unsupported hardware register write id={register} offset={offset} width={width} value={value:#x}"
            ));
        }
        return Ok(StageAOp::ConfigureHardwareRegister {
            register,
            offset,
            width,
            value,
        });
    }
    Ok(match decoded.inst.name {
        "S_CMP_EQ_U32" => StageAOp::CompareEq {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_GT_U32" => StageAOp::CompareGtU32 {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_LT_U32" => StageAOp::CompareLtU32 {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_LE_U32" => StageAOp::CompareLeU32 {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_GE_U32" => StageAOp::CompareGeU32 {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_LG_U32" => StageAOp::CompareNeU32 {
            lhs: operand32(decoded, "SSRC0")?,
            rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_LT_I32" => StageAOp::CompareLtI32 {
            lhs: operand32(decoded, "SSRC0")?, rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_GT_I32" => StageAOp::CompareGtI32 {
            lhs: operand32(decoded, "SSRC0")?, rhs: operand32(decoded, "SSRC1")?,
        },
        "S_CMP_GE_I32" => StageAOp::CompareGeI32 {
            lhs: operand32(decoded, "SSRC0")?, rhs: operand32(decoded, "SSRC1")?,
        },
        "S_MOV_B32" => StageAOp::Move32 {
            dst: destination("SDST")?,
            source: operand32(decoded, "SSRC0")?,
        },
        "S_MOV_B64" => StageAOp::Move64 {
            dst: pair_destination()?,
            source: operand64(decoded, "SSRC0")?,
        },
        "S_AND_SAVEEXEC_B64" => StageAOp::SaveExecAnd64 {
            dst: ordinary_pair_destination()?,
            source: operand64(decoded, "SSRC0")?,
        },
        "S_OR_SAVEEXEC_B64" => StageAOp::SaveExecOr64 {
            dst: ordinary_pair_destination()?,
            source: operand64(decoded, "SSRC0")?,
        },
        "S_ANDN2_WREXEC_B64" => StageAOp::WrExecAndNot64 {
            dst: ordinary_pair_destination()?,
            source: operand64(decoded, "SSRC0")?,
        },
        "S_CSELECT_B32" => StageAOp::Select32 {
            dst: destination("SDST")?,
            when_set: operand32(decoded, "SSRC0")?,
            when_clear: operand32(decoded, "SSRC1")?,
        },
        "S_MOVK_I32" => StageAOp::MoveImmediate {
            dst: destination("SDST")?,
            value: field(decoded, "SIMM16")? as u16 as i16 as i32 as u32,
        },
        "S_CSELECT_B64" => StageAOp::Select64 {
            dst: pair_destination()?,
            when_set: operand64(decoded, "SSRC0")?,
            when_clear: operand64(decoded, "SSRC1")?,
        },
        "S_WAITCNT" | "S_INST_PREFETCH" | "S_NOP" => StageAOp::NoEffect,
        "S_BARRIER" => StageAOp::Barrier,
        "S_ENDPGM" => StageAOp::Complete,
        name => return Err(format!("unsupported Stage A opcode {name}")),
    })
}

fn register(state: &LogicalWaveState, index: u8) -> Option<u32> {
    match index {
        106 => Some(state.vcc[0]),
        107 => Some(state.vcc[1]),
        126 => Some(state.exec[0]),
        127 => Some(state.exec[1]),
        125 => Some(0),
        index => state.scalar.get(index as usize).copied(),
    }
}

fn set_register(state: &mut LogicalWaveState, index: u8, value: u32) -> bool {
    match index {
        106 => state.vcc[0] = value,
        107 => state.vcc[1] = value,
        126 => state.exec[0] = value,
        127 => state.exec[1] = value,
        125 => {}
        index => match state.scalar.get_mut(index as usize) {
            Some(register) => *register = value,
            None => return false,
        },
    }
    true
}

fn value32(state: &LogicalWaveState, operand: Operand32) -> Option<u32> {
    match operand {
        Operand32::Register(index) => register(state, index),
        Operand32::Constant(value) => Some(value),
        Operand32::VccZero => Some(u32::from(state.vcc == [0, 0])),
        Operand32::ExecZero => Some(u32::from(state.exec == [0, 0])),
        Operand32::Scc => Some(state.scc),
    }
}

fn value64(state: &LogicalWaveState, operand: Operand64) -> Option<u64> {
    match operand {
        Operand64::Constant(value) => Some(value),
        Operand64::Scc => Some(state.scc as u64),
        Operand64::Register(index) => {
            let low = register(state, index)?;
            let high = register(state, index.checked_add(1)?)?;
            Some(low as u64 | (high as u64) << 32)
        }
    }
}

fn bfe_u(value: u32, control: u32) -> u32 {
    let offset = control & 31;
    let width = (control >> 16) & 0x7f;
    if width == 0 {
        0
    } else if width >= 32 {
        value >> offset
    } else {
        (value >> offset) & (u32::MAX >> (32 - width))
    }
}

fn bfe_i(value: u32, control: u32) -> u32 {
    let offset = control & 31;
    let width = (control >> 16) & 0x7f;
    if width == 0 {
        0
    } else if width >= 32 {
        ((value as i32) >> offset) as u32
    } else {
        let shift = 32 - width;
        (((value >> offset) << shift) as i32 >> shift) as u32
    }
}

fn execute(state: &mut LogicalWaveState, op: StageAOp) -> Result<Option<u32>, &'static str> {
    match op {
        StageAOp::AddU32 { dst, lhs, rhs, carry_in } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar source")?;
            let carry = u64::from(carry_in) * u64::from(state.scc != 0);
            let sum = lhs as u64 + rhs as u64 + carry;
            state.scc = u32::from(sum > u32::MAX as u64);
            if !set_register(state, dst, sum as u32) { return Err("invalid scalar destination"); }
        }
        StageAOp::MulImmediateI32 { dst, value } => {
            let lhs = register(state, dst).ok_or("invalid scalar source")?;
            if !set_register(state, dst, lhs.wrapping_mul(value as i32 as u32)) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::GetPc64 { dst, next_pc } => {
            if !set_register(state, dst, next_pc as u32)
                || !set_register(state, dst + 1, (next_pc >> 32) as u32) {
                return Err("invalid scalar pair destination");
            }
        }
        StageAOp::BitSet1 { dst, bit } => {
            let bit = value32(state, bit).ok_or("invalid scalar bit index")?;
            if !set_register(state, dst, 1u32 << (bit & 31)) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::Not32 { dst, source } => {
            let value = !value32(state, source).ok_or("invalid scalar source")?;
            state.scc = u32::from(value != 0);
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::BitReverse32 { dst, source } => {
            let value = value32(state, source)
                .ok_or("invalid scalar source")?
                .reverse_bits();
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::CountBits32 { dst, source } => {
            let value = value32(state, source)
                .ok_or("invalid scalar source")?
                .count_ones();
            state.scc = u32::from(value != 0);
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::CountBits64 { dst, source } => {
            let value = value64(state, source)
                .ok_or("invalid scalar pair source")?
                .count_ones();
            state.scc = u32::from(value != 0);
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::Binary {
            kind,
            dst,
            lhs,
            rhs,
        } => {
            let (lhs, rhs) = (
                value32(state, lhs).ok_or("invalid scalar source")?,
                value32(state, rhs).ok_or("invalid scalar source")?,
            );
            let value = match kind {
                ScalarBinaryKind::AddI32 => lhs.wrapping_add(rhs),
                ScalarBinaryKind::SubI32 => lhs.wrapping_sub(rhs),
                ScalarBinaryKind::MulI32 => lhs.wrapping_mul(rhs),
                ScalarBinaryKind::MulHiU32 => ((lhs as u64 * rhs as u64) >> 32) as u32,
                ScalarBinaryKind::MinU32 => lhs.min(rhs),
                ScalarBinaryKind::And => lhs & rhs,
                ScalarBinaryKind::Or => lhs | rhs,
                ScalarBinaryKind::Xor => lhs ^ rhs,
                ScalarBinaryKind::Lshl => lhs << (rhs & 31),
                ScalarBinaryKind::Lshr => lhs >> (rhs & 31),
                ScalarBinaryKind::AshrI32 => ((lhs as i32) >> (rhs & 31)) as u32,
                ScalarBinaryKind::BfeU32 => bfe_u(lhs, rhs),
                ScalarBinaryKind::BfeI32 => bfe_i(lhs, rhs),
            };
            match kind {
                ScalarBinaryKind::AddI32 => {
                    state.scc = u32::from((((lhs ^ value) & (rhs ^ value)) >> 31) != 0)
                }
                ScalarBinaryKind::SubI32 => {
                    state.scc = u32::from((((lhs ^ rhs) & (lhs ^ value)) >> 31) != 0)
                }
                ScalarBinaryKind::MulI32 | ScalarBinaryKind::MulHiU32 => {}
                ScalarBinaryKind::MinU32 => state.scc = u32::from(lhs < rhs),
                _ => state.scc = u32::from(value != 0),
            }
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::Binary64 {
            kind,
            dst,
            lhs,
            rhs,
        } => {
            let lhs = value64(state, lhs).ok_or("invalid scalar pair source")?;
            let rhs = value64(state, rhs).ok_or("invalid scalar pair source")?;
            let value = match kind {
                ScalarBinary64Kind::And => lhs & rhs,
                ScalarBinary64Kind::AndNot => lhs & !rhs,
                ScalarBinary64Kind::Or => lhs | rhs,
                ScalarBinary64Kind::Xor => lhs ^ rhs,
            };
            state.scc = u32::from(value != 0);
            if !set_register(state, dst, value as u32)
                || !set_register(state, dst + 1, (value >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
        }
        StageAOp::CompareEq { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs == rhs);
        }
        StageAOp::CompareGtU32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs > rhs);
        }
        StageAOp::CompareLtU32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs < rhs);
        }
        StageAOp::CompareLeU32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs <= rhs);
        }
        StageAOp::CompareGeU32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs >= rhs);
        }
        StageAOp::CompareNeU32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")?;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")?;
            state.scc = u32::from(lhs != rhs);
        }
        StageAOp::CompareLtI32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")? as i32;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")? as i32;
            state.scc = u32::from(lhs < rhs);
        }
        StageAOp::CompareGtI32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")? as i32;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")? as i32;
            state.scc = u32::from(lhs > rhs);
        }
        StageAOp::CompareGeI32 { lhs, rhs } => {
            let lhs = value32(state, lhs).ok_or("invalid scalar compare source")? as i32;
            let rhs = value32(state, rhs).ok_or("invalid scalar compare source")? as i32;
            state.scc = u32::from(lhs >= rhs);
        }
        StageAOp::Select32 {
            dst,
            when_set,
            when_clear,
        } => {
            let selected = if state.scc != 0 { when_set } else { when_clear };
            let value = value32(state, selected).ok_or("invalid scalar select source")?;
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::MoveImmediate { dst, value } => {
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::Move32 { dst, source } => {
            let value = value32(state, source).ok_or("invalid scalar source")?;
            if !set_register(state, dst, value) {
                return Err("invalid scalar destination");
            }
        }
        StageAOp::Move64 { dst, source } => {
            let value = value64(state, source).ok_or("invalid scalar pair source")?;
            if !set_register(state, dst, value as u32)
                || !set_register(state, dst + 1, (value >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
        }
        StageAOp::SaveExecAnd64 { dst, source } => {
            let old_exec = state.exec[0] as u64 | (state.exec[1] as u64) << 32;
            let source = value64(state, source).ok_or("invalid scalar pair source")?;
            let value = old_exec & source;
            if !set_register(state, dst, old_exec as u32)
                || !set_register(state, dst + 1, (old_exec >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
            state.exec = [value as u32, (value >> 32) as u32];
            state.scc = u32::from(value != 0);
        }
        StageAOp::SaveExecOr64 { dst, source } => {
            let old_exec = state.exec[0] as u64 | (state.exec[1] as u64) << 32;
            let source = value64(state, source).ok_or("invalid scalar pair source")?;
            let value = source | old_exec;
            if !set_register(state, dst, old_exec as u32)
                || !set_register(state, dst + 1, (old_exec >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
            state.exec = [value as u32, (value >> 32) as u32];
            state.scc = u32::from(value != 0);
        }
        StageAOp::WrExecAndNot64 { dst, source } => {
            let old_exec = state.exec[0] as u64 | (state.exec[1] as u64) << 32;
            let source = value64(state, source).ok_or("invalid scalar pair source")?;
            let value = source & !old_exec;
            if !set_register(state, dst, value as u32)
                || !set_register(state, dst + 1, (value >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
            state.exec = [value as u32, (value >> 32) as u32];
            state.scc = u32::from(value != 0);
        }
        StageAOp::ConfigureHardwareRegister { .. } => {}
        StageAOp::Select64 {
            dst,
            when_set,
            when_clear,
        } => {
            let selected = if state.scc != 0 { when_set } else { when_clear };
            let value = value64(state, selected).ok_or("invalid scalar pair source")?;
            if !set_register(state, dst, value as u32)
                || !set_register(state, dst + 1, (value >> 32) as u32)
            {
                return Err("invalid scalar pair destination");
            }
        }
        StageAOp::NoEffect => {}
        StageAOp::Barrier => return Ok(None),
        StageAOp::Jump(target) => return Ok(Some(target)),
        StageAOp::Complete => state.status = LogicalWaveStatus::Complete,
    }
    Ok(Some(state.pc.saturating_add(1)))
}

pub fn run_stage_a_pair(
    programs: [&[StageAOp]; 2],
    mut states: [LogicalWaveState; 2],
    step_cap: u32,
) -> Result<[LogicalWaveState; 2], &'static str> {
    validate_step_cap(step_cap)?;
    let mut waiting = [None::<u32>; 2];
    for _ in 0..step_cap {
        for wave in 0..2 {
            if states[wave].status != LogicalWaveStatus::Running || waiting[wave].is_some() {
                continue;
            }
            if states[wave].steps >= step_cap {
                states[wave].status = LogicalWaveStatus::StepLimit;
                continue;
            }
            let Some(&op) = programs[wave].get(states[wave].pc as usize) else {
                states[wave].status = LogicalWaveStatus::Fault;
                continue;
            };
            states[wave].steps += 1;
            if op == StageAOp::Barrier {
                waiting[wave] = Some(states[wave].epoch);
            } else {
                match execute(&mut states[wave], op) {
                    Ok(Some(next)) if states[wave].status == LogicalWaveStatus::Running => {
                        states[wave].pc = next
                    }
                    Ok(_) => {}
                    Err(_) => states[wave].status = LogicalWaveStatus::Fault,
                }
            }
        }
        match waiting {
            [Some(left), Some(right)] if left == right => {
                for wave in 0..2 {
                    states[wave].pc += 1;
                    states[wave].epoch += 1;
                    waiting[wave] = None;
                }
            }
            [Some(_), Some(_)] => {
                states[0].status = LogicalWaveStatus::Fault;
                states[1].status = LogicalWaveStatus::Fault;
            }
            [Some(_), None] if states[1].status != LogicalWaveStatus::Running => {
                states[0].status = LogicalWaveStatus::Fault
            }
            [None, Some(_)] if states[0].status != LogicalWaveStatus::Running => {
                states[1].status = LogicalWaveStatus::Fault
            }
            _ => {}
        }
        if states
            .iter()
            .all(|state| state.status != LogicalWaveStatus::Running)
        {
            return Ok(states);
        }
    }
    for state in &mut states {
        if state.status == LogicalWaveStatus::Running {
            state.status = LogicalWaveStatus::StepLimit;
        }
    }
    Ok(states)
}

fn hlsl_operand(operand: Operand32, scalar_base: &str) -> String {
    match operand {
        Operand32::Constant(value) => format!("{value}u"),
        Operand32::VccZero => "(uint)all(vcc == 0u)".into(),
        Operand32::ExecZero => "(uint)all(exec == 0u)".into(),
        Operand32::Scc => "scc".into(),
        Operand32::Register(index) => match index {
            106 => "vcc.x".into(),
            107 => "vcc.y".into(),
            126 => "exec.x".into(),
            127 => "exec.y".into(),
            125 => "0u".into(),
            _ => format!("scratch.Load({scalar_base} + {}u)", index as u32 * 4),
        },
    }
}

fn hlsl_target(index: u8, value: &str, scalar_base: &str) -> String {
    match index {
        106 => format!("vcc.x = {value};"),
        107 => format!("vcc.y = {value};"),
        126 => format!("exec.x = {value};"),
        127 => format!("exec.y = {value};"),
        125 => String::new(),
        _ => format!(
            "scratch.Store({scalar_base} + {}u, {value});",
            index as u32 * 4
        ),
    }
}

fn hlsl_operand64(operand: Operand64, scalar_base: &str) -> String {
    match operand {
        Operand64::Constant(value) => {
            format!("uint2({}u, {}u)", value as u32, (value >> 32) as u32)
        }
        Operand64::Scc => "uint2(scc, 0u)".into(),
        Operand64::Register(106) => "vcc".into(),
        Operand64::Register(126) => "exec".into(),
        Operand64::Register(index) => {
            format!("scratch.Load2({scalar_base} + {}u)", index as u32 * 4)
        }
    }
}

fn hlsl_target64(index: u8, value: &str, scalar_base: &str) -> String {
    match index {
        106 => format!("vcc = {value};"),
        126 => format!("exec = {value};"),
        _ => format!(
            "scratch.Store2({scalar_base} + {}u, {value});",
            index as u32 * 4
        ),
    }
}

fn hlsl_case(index: usize, op: StageAOp) -> String {
    let sb = "scalar_base";
    let body = match op {
        StageAOp::AddU32 { dst, lhs, rhs, carry_in } => {
            let carry = if carry_in { "(uint64_t)(scc != 0u)" } else { "0ull" };
            format!("uint a = {}; uint b = {}; uint64_t sum = (uint64_t)a + (uint64_t)b + {carry}; uint value = (uint)sum; {} scc = sum > 0xFFFFFFFFull; pc++;", hlsl_operand(lhs, sb), hlsl_operand(rhs, sb), hlsl_target(dst, "value", sb))
        }
        StageAOp::MulImmediateI32 { dst, value } => format!(
            "uint value = (uint)((int){} * (int){}); {} pc++;",
            hlsl_operand(Operand32::Register(dst), sb), value as i32, hlsl_target(dst, "value", sb)
        ),
        StageAOp::GetPc64 { dst, next_pc } => format!(
            "{} pc++;", hlsl_target64(dst, &format!("uint2({}u, {}u)", next_pc as u32, (next_pc >> 32) as u32), sb)
        ),
        StageAOp::BitSet1 { dst, bit } => format!(
            "uint value = 1u << (({}) & 31u); {} pc++;",
            hlsl_operand(bit, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::Not32 { dst, source } => format!(
            "uint value = ~({}); {} scc = value != 0u; pc++;",
            hlsl_operand(source, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::BitReverse32 { dst, source } => format!(
            "uint value = reversebits({}); {} pc++;",
            hlsl_operand(source, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::CountBits32 { dst, source } => format!(
            "uint value = countbits({}); {} scc = value != 0u; pc++;",
            hlsl_operand(source, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::CountBits64 { dst, source } => {
            let source = hlsl_operand64(source, sb);
            format!(
                "uint2 source = {source}; uint value = countbits(source.x) + countbits(source.y); {} scc = value != 0u; pc++;",
                hlsl_target(dst, "value", sb)
            )
        }
        StageAOp::Binary {
            kind,
            dst,
            lhs,
            rhs,
        } => {
            let (a, b) = (hlsl_operand(lhs, sb), hlsl_operand(rhs, sb));
            let value = match kind {
                ScalarBinaryKind::AddI32 => format!("a + b"),
                ScalarBinaryKind::SubI32 => "a - b".into(),
                ScalarBinaryKind::MulI32 => "a * b".into(),
                ScalarBinaryKind::MulHiU32 => {
                    "(uint)(((uint64_t)a * (uint64_t)b) >> 32u)".into()
                }
                ScalarBinaryKind::MinU32 => "min(a, b)".into(),
                ScalarBinaryKind::And => "a & b".into(),
                ScalarBinaryKind::Or => "a | b".into(),
                ScalarBinaryKind::Xor => "a ^ b".into(),
                ScalarBinaryKind::Lshl => "a << (b & 31u)".into(),
                ScalarBinaryKind::Lshr => "a >> (b & 31u)".into(),
                ScalarBinaryKind::AshrI32 => "(uint)((int)a >> (b & 31u))".into(),
                ScalarBinaryKind::BfeU32 => "sbfe_u(a, b)".into(),
                ScalarBinaryKind::BfeI32 => "(uint)sbfe_i(a, b)".into(),
            };
            let scc = match kind {
                ScalarBinaryKind::AddI32 => {
                    "scc = (((a ^ value) & (b ^ value)) >> 31u) != 0u;"
                }
                ScalarBinaryKind::SubI32 => {
                    "scc = (((a ^ b) & (a ^ value)) >> 31u) != 0u;"
                }
                ScalarBinaryKind::MulI32 | ScalarBinaryKind::MulHiU32 => "",
                ScalarBinaryKind::MinU32 => "scc = a < b;",
                _ => "scc = value != 0u;",
            };
            format!(
                "uint a = {a}; uint b = {b}; uint value = {value}; {} {scc} pc++;",
                hlsl_target(dst, "value", sb)
            )
        }
        StageAOp::Binary64 {
            kind,
            dst,
            lhs,
            rhs,
        } => {
            let (lhs, rhs) = (hlsl_operand64(lhs, sb), hlsl_operand64(rhs, sb));
            let value = match kind {
                ScalarBinary64Kind::And => "a & b",
                ScalarBinary64Kind::AndNot => "a & ~b",
                ScalarBinary64Kind::Or => "a | b",
                ScalarBinary64Kind::Xor => "a ^ b",
            };
            format!(
                "uint2 a = {lhs}; uint2 b = {rhs}; uint2 value = {value}; {} scc = any(value != 0u); pc++;",
                hlsl_target64(dst, "value", sb)
            )
        }
        StageAOp::CompareEq { lhs, rhs } => format!(
            "scc = ({}) == ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareGtU32 { lhs, rhs } => format!(
            "scc = ({}) > ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareLtU32 { lhs, rhs } => format!(
            "scc = ({}) < ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareLeU32 { lhs, rhs } => format!(
            "scc = ({}) <= ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareGeU32 { lhs, rhs } => format!(
            "scc = ({}) >= ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareNeU32 { lhs, rhs } => format!(
            "scc = ({}) != ({}); pc++;",
            hlsl_operand(lhs, sb),
            hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareLtI32 { lhs, rhs } => format!(
            "scc = (int)({}) < (int)({}); pc++;", hlsl_operand(lhs, sb), hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareGtI32 { lhs, rhs } => format!(
            "scc = (int)({}) > (int)({}); pc++;", hlsl_operand(lhs, sb), hlsl_operand(rhs, sb)
        ),
        StageAOp::CompareGeI32 { lhs, rhs } => format!(
            "scc = (int)({}) >= (int)({}); pc++;", hlsl_operand(lhs, sb), hlsl_operand(rhs, sb)
        ),
        StageAOp::Select32 {
            dst,
            when_set,
            when_clear,
        } => format!(
            "uint value = scc != 0u ? {} : {}; {} pc++;",
            hlsl_operand(when_set, sb),
            hlsl_operand(when_clear, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::MoveImmediate { dst, value } => {
            format!("{} pc++;", hlsl_target(dst, &format!("{value}u"), sb))
        }
        StageAOp::Move32 { dst, source } => format!(
            "uint value = {}; {} pc++;",
            hlsl_operand(source, sb),
            hlsl_target(dst, "value", sb)
        ),
        StageAOp::Move64 { dst, source } => format!(
            "uint2 value = {}; {} pc++;",
            hlsl_operand64(source, sb),
            hlsl_target64(dst, "value", sb)
        ),
        StageAOp::SaveExecAnd64 { dst, source } => format!(
            "uint2 old_exec = exec; uint2 source = {}; exec &= source; {} scc = any(exec != 0u); pc++;",
            hlsl_operand64(source, sb),
            hlsl_target64(dst, "old_exec", sb)
        ),
        StageAOp::SaveExecOr64 { dst, source } => format!(
            "uint2 old_exec = exec; uint2 source = {}; exec = source | old_exec; {} scc = any(exec != 0u); pc++;",
            hlsl_operand64(source, sb),
            hlsl_target64(dst, "old_exec", sb)
        ),
        StageAOp::WrExecAndNot64 { dst, source } => format!(
            "uint2 old_exec = exec; uint2 source = {}; uint2 value = source & ~old_exec; exec = value; {} scc = any(value != 0u); pc++;",
            hlsl_operand64(source, sb),
            hlsl_target64(dst, "value", sb)
        ),
        StageAOp::ConfigureHardwareRegister { .. } => "pc++;".into(),
        StageAOp::Select64 {
            dst,
            when_set,
            when_clear,
        } => {
            let value = format!(
                "(scc != 0u ? {} : {})",
                hlsl_operand64(when_set, sb),
                hlsl_operand64(when_clear, sb)
            );
            format!("{} pc++;", hlsl_target64(dst, &value, sb))
        }
        StageAOp::NoEffect => "pc++;".into(),
        StageAOp::Barrier => "waiting = true;".into(),
        StageAOp::Jump(target) => format!("pc = {target}u;"),
        StageAOp::Complete => "status = 1u;".into(),
    };
    format!("case {index}u: {{ {body} break; }}")
}

pub fn stage_a_validation_hlsl(
    layout: ScratchLayout,
    programs: [&[StageAOp]; 2],
    initial: [LogicalWaveState; 2],
    step_cap: u32,
) -> Result<String, &'static str> {
    validate_step_cap(step_cap)?;
    layout.validate_for_groups(layout.record_count)?;
    if programs.iter().any(|program| program.is_empty()) {
        return Err("empty Stage A program");
    }
    if initial.iter().enumerate().any(|(wave, state)| {
        state.status != LogicalWaveStatus::Running
            || state.steps != 0
            || state.pc as usize >= programs[wave].len()
    }) {
        return Err("invalid initial Stage A state");
    }
    let cases = |program: &[StageAOp]| {
        program
            .iter()
            .copied()
            .enumerate()
            .map(|(i, op)| hlsl_case(i, op))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut init = String::new();
    for (wave, state) in initial.iter().enumerate() {
        for (register, value) in state
            .scalar
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
        {
            init.push_str(&format!(
                "if (wave == {wave}u) scratch.Store(scalar_base + {}u, {value}u);",
                register * 4
            ));
        }
    }
    let select = |value: fn(&LogicalWaveState) -> u32| {
        format!("wave==0u?{}u:{}u", value(&initial[0]), value(&initial[1]))
    };
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer scratch : register(u6);
uint sbfe_u(uint v, uint c) {{ uint o=c&31u,w=(c>>16)&0x7fu; return w==0u?0u:(w>=32u?v>>o:(v>>o)&(0xffffffffu>>(32u-w))); }}
int sbfe_i(uint v, uint c) {{ uint o=c&31u,w=(c>>16)&0x7fu,s=(32u-w)&31u; return w==0u?0:(w>=32u?(int)v>>o:((int)((v>>o)<<s))>>s); }}
[numthreads({GROUP_THREADS},1,1)] void main(uint gi:SV_GroupIndex,uint3 gid:SV_GroupID) {{
 uint wave=gi/64u,lane=gi%64u,base=gid.x*io[0],state_base=base+wave*{state_bytes}u,scalar_base=state_base+{header_bytes}u,mailbox=base+{mailbox_offset}u;
 scratch.Store(scalar_base+lane*4u,0u); scratch.Store(scalar_base+(lane+64u)*4u,0u);
 if(lane==0u){{ scratch.Store4(state_base,uint4({initial_pc},{initial_exec_lo},{initial_exec_hi},{initial_vcc_lo})); scratch.Store4(state_base+16u,uint4({initial_vcc_hi},{initial_scc},{initial_epoch},0u)); scratch.Store(state_base+32u,0u); scratch.Store(mailbox+wave*4u,0u); scratch.Store(mailbox+8u+wave*4u,0u); }}
 if(gi==0u) records.Store4(gid.x*io[1],uint4(0u,0u,0u,0u)); DeviceMemoryBarrierWithGroupSync();
 if(lane==0u){{{init}}} DeviceMemoryBarrierWithGroupSync();
 [loop] for(uint tick=0u;tick<{step_cap}u;tick++){{
  if(lane==0u){{ uint pc=scratch.Load(state_base); uint2 exec=scratch.Load2(state_base+4u),vcc=scratch.Load2(state_base+12u); uint scc=scratch.Load(state_base+20u),generation=scratch.Load(state_base+24u),status=scratch.Load(state_base+28u),steps=scratch.Load(state_base+32u); bool waiting=scratch.Load(mailbox+wave*4u)!=0u;
   if(status==0u&&!waiting){{ if(steps>={step_cap}u) status=2u; else {{ {cases0} steps++; if(waiting)scratch.Store(mailbox+wave*4u,generation+1u); }} }}
   scratch.Store(state_base,pc);scratch.Store2(state_base+4u,exec);scratch.Store2(state_base+12u,vcc);scratch.Store(state_base+20u,scc);scratch.Store(state_base+24u,generation);scratch.Store(state_base+28u,status);scratch.Store(state_base+32u,steps);
  }} DeviceMemoryBarrierWithGroupSync();
  if(lane==0u){{ uint mine=scratch.Load(mailbox+wave*4u),other=scratch.Load(mailbox+(1u-wave)*4u),status=scratch.Load(state_base+28u),other_status=scratch.Load(base+(1u-wave)*{state_bytes}u+28u),decision=0u; if(mine!=0u&&other!=0u)decision=mine==other?1u:2u;else if(mine!=0u&&other==0u&&other_status!=0u)decision=2u;scratch.Store(mailbox+8u+wave*4u,decision);}}
  DeviceMemoryBarrierWithGroupSync();
  if(lane==0u){{uint decision=scratch.Load(mailbox+8u+wave*4u);if(decision==1u){{uint mine=scratch.Load(mailbox+wave*4u);scratch.Store(state_base,scratch.Load(state_base)+1u);scratch.Store(state_base+24u,mine);scratch.Store(mailbox+wave*4u,0u);}}else if(decision==2u){{scratch.Store(state_base+28u,3u);scratch.Store(mailbox+wave*4u,0u);}}scratch.Store(mailbox+8u+wave*4u,0u);}}
  DeviceMemoryBarrierWithGroupSync();
 }}
 if(lane==0u&&scratch.Load(state_base+28u)==0u)scratch.Store(state_base+28u,2u); DeviceMemoryBarrierWithGroupSync();
}}"#,
        state_bytes = (9 + SCALAR_REGISTERS) * 4,
        header_bytes = 9 * 4,
        mailbox_offset = layout.mailbox_offset(0, 0)?,
        initial_pc = select(|state| state.pc),
        initial_exec_lo = select(|state| state.exec[0]),
        initial_exec_hi = select(|state| state.exec[1]),
        initial_vcc_lo = select(|state| state.vcc[0]),
        initial_vcc_hi = select(|state| state.vcc[1]),
        initial_scc = select(|state| state.scc),
        initial_epoch = select(|state| state.epoch),
        cases0 = format!(
            "if(wave==0u){{switch(pc){{{} default:status=3u;break;}}}}else{{switch(pc){{{} default:status=3u;break;}}}}",
            cases(programs[0]),
            cases(programs[1])
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::mesh_dispatch::{read_logical_state, scratch_for_groups};
    use super::*;

    fn programs() -> ([StageAOp; 9], [StageAOp; 11]) {
        let add = StageAOp::Binary {
            kind: ScalarBinaryKind::AddI32,
            dst: 0,
            lhs: Operand32::Register(0),
            rhs: Operand32::Constant(1),
        };
        (
            [
                add,
                StageAOp::Barrier,
                StageAOp::NoEffect,
                add,
                StageAOp::Barrier,
                StageAOp::Complete,
                StageAOp::NoEffect,
                StageAOp::NoEffect,
                StageAOp::NoEffect,
            ],
            [
                StageAOp::NoEffect,
                StageAOp::NoEffect,
                add,
                StageAOp::Barrier,
                StageAOp::NoEffect,
                StageAOp::NoEffect,
                add,
                StageAOp::Barrier,
                StageAOp::Complete,
                StageAOp::NoEffect,
                StageAOp::NoEffect,
            ],
        )
    }

    fn run_gpu(
        programs: [&[StageAOp]; 2],
        initial: [LogicalWaveState; 2],
        step_cap: u32,
        groups: u32,
    ) -> Option<Vec<[LogicalWaveState; 2]>> {
        if super::super::dxc::location().is_err() {
            return None;
        }
        let layout = scratch_for_groups(groups).unwrap();
        let hlsl = stage_a_validation_hlsl(layout, programs, initial, step_cap).unwrap();
        assert!(!hlsl.contains("groupshared") && !hlsl.contains("xcl_lds"));
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
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            groups,
            record.record_dwords(),
            &[
                layout.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout,
        )
        .unwrap();
        Some(
            (0..groups)
                .map(|group| {
                    std::array::from_fn(|wave| {
                        read_logical_state(&memory, layout, group, wave as u32).unwrap()
                    })
                })
                .collect(),
        )
    }

    #[test]
    fn scalar_edge_cases_match_emitter_formulas() {
        let mut state = LogicalWaveState::default();
        state.scalar[0] = 0x7fff_ffff;
        state.scalar[1] = 1;
        state.scc = 1;
        let ops = [
            StageAOp::Binary {
                kind: ScalarBinaryKind::AddI32,
                dst: 2,
                lhs: Operand32::Register(0),
                rhs: Operand32::Register(1),
            },
            StageAOp::Binary {
                kind: ScalarBinaryKind::Lshl,
                dst: 3,
                lhs: Operand32::Constant(1),
                rhs: Operand32::Constant(32),
            },
            StageAOp::Binary {
                kind: ScalarBinaryKind::BfeU32,
                dst: 4,
                lhs: Operand32::Constant(0xf0),
                rhs: Operand32::Constant((4 << 16) | 4),
            },
            StageAOp::Binary {
                kind: ScalarBinaryKind::BfeI32,
                dst: 5,
                lhs: Operand32::Constant(0x80),
                rhs: Operand32::Constant(8 << 16),
            },
            StageAOp::CompareEq {
                lhs: Operand32::Register(4),
                rhs: Operand32::Constant(0xf),
            },
            StageAOp::Select64 {
                dst: 6,
                when_set: Operand64::Constant(0x1122_3344_5566_7788),
                when_clear: Operand64::Constant(0),
            },
            StageAOp::Complete,
        ];
        let out = run_stage_a_pair([&ops, &ops], [state, state], 32).unwrap();
        for state in out {
            assert_eq!(state.scalar[2], 0x8000_0000);
            assert_eq!(state.scalar[3], 1);
            assert_eq!(state.scalar[4], 0xf);
            assert_eq!(state.scalar[5], 0xffff_ff80);
            assert_eq!(
                [state.scalar[6], state.scalar[7]],
                [0x5566_7788, 0x1122_3344]
            );
        }
    }

    #[test]
    fn compare_select_and_movk_match_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let lower = |name, encoding, fields: &[(&str, u64)]| {
            let code = encode(name, encoding, fields, None).unwrap();
            lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let true_path = [
            lower("S_CMP_GT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)]),
            lower(
                "S_CSELECT_B32",
                "ENC_SOP2",
                &[("SDST", 2), ("SSRC0", 193), ("SSRC1", 192)],
            ),
            lower("S_MOVK_I32", "ENC_SOPK", &[("SDST", 3), ("SIMM16", 0x8000)]),
            StageAOp::Barrier,
            StageAOp::Complete,
        ];
        let false_path = [
            lower("S_CMP_GT_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)]),
            lower(
                "S_CSELECT_B32",
                "ENC_SOP2",
                &[("SDST", 2), ("SSRC0", 193), ("SSRC1", 192)],
            ),
            lower("S_MOVK_I32", "ENC_SOPK", &[("SDST", 3), ("SIMM16", 0x7fff)]),
            StageAOp::Barrier,
            StageAOp::Complete,
        ];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[0] = u32::MAX;
        initial[0].scalar[1] = u32::MAX - 1;
        initial[1].scalar[0] = 0;
        initial[1].scalar[1] = u32::MAX;
        let expected = run_stage_a_pair([&true_path, &false_path], initial, 16).unwrap();
        assert_eq!(
            (
                expected[0].scc,
                expected[0].scalar[2],
                expected[0].scalar[3]
            ),
            (1, u32::MAX, 0xffff_8000)
        );
        assert_eq!(
            (
                expected[1].scc,
                expected[1].scalar[2],
                expected[1].scalar[3]
            ),
            (0, 64, 0x7fff)
        );
        assert_eq!(
            run_gpu([&true_path, &false_path], initial, 16, 2).unwrap(),
            vec![expected; 2]
        );

        let unsupported = encode(
            "S_CSELECT_B32",
            "ENC_SOP2",
            &[("SDST", 2), ("SSRC0", 249), ("SSRC1", 128)],
            None,
        )
        .unwrap();
        assert!(lower_stage_a_one(&rdna2::decode(&unsupported, 0).unwrap()).is_err());
    }

    #[test]
    fn addk_brev_and_unsigned_ge_match_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let lower = |name, encoding, fields: &[(&str, u64)]| {
            let code = encode(name, encoding, fields, None).unwrap();
            lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let program = [
            lower("S_ADDK_I32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 1)]),
            lower(
                "S_CSELECT_B32",
                "ENC_SOP2",
                &[("SDST", 3), ("SSRC0", 129), ("SSRC1", 128)],
            ),
            lower(
                "S_CMP_GE_U32",
                "ENC_SOPC",
                &[("SSRC0", 2), ("SSRC1", 6)],
            ),
            lower("S_BREV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 1)]),
            lower(
                "S_CSELECT_B32",
                "ENC_SOP2",
                &[("SDST", 4), ("SSRC0", 129), ("SSRC1", 128)],
            ),
            StageAOp::Complete,
        ];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[0] = i32::MAX as u32;
        initial[0].scalar[1] = 1;
        initial[0].scalar[2] = 0;
        initial[0].scalar[6] = 1;
        initial[1].scalar[0] = u32::MAX;
        initial[1].scalar[1] = 0;
        initial[1].scalar[2] = 5;
        initial[1].scalar[6] = 1;

        let expected = run_stage_a_pair([&program, &program], initial, 16).unwrap();
        assert_eq!(expected[0].scalar[0], 0x8000_0000);
        assert_eq!(expected[0].scalar[3], 1);
        assert_eq!(expected[0].scalar[7], 0x8000_0000);
        assert_eq!(expected[0].scalar[4], 0);
        assert_eq!(expected[1].scalar[0], 0);
        assert_eq!(expected[1].scalar[3], 0);
        assert_eq!(expected[1].scalar[7], 0);
        assert_eq!(expected[1].scalar[4], 1);
        if let Some(gpu) = run_gpu([&program, &program], initial, 16, 1) {
            for wave in 0..2 {
                assert_eq!(gpu[0][wave].scalar[..8], expected[wave].scalar[..8]);
                assert_eq!(gpu[0][wave].scc, expected[wave].scc);
            }
        }

        let invalid = encode(
            "S_BREV_B32",
            "ENC_SOP1",
            &[("SDST", 0), ("SSRC0", 254)],
            None,
        )
        .unwrap();
        assert!(lower_stage_a_one(&rdna2::decode(&invalid, 0).unwrap()).is_err());
    }

    #[test]
    fn compare_le_decoded_edges_match_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let lower = |fields: &[(&str, u64)], literal| {
            let code = encode("S_CMP_LE_U32", "ENC_SOPC", fields, literal).unwrap();
            lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let select = |dst| StageAOp::Select32 {
            dst,
            when_set: Operand32::Constant(1),
            when_clear: Operand32::Constant(0),
        };
        let ops = [
            lower(&[("SSRC0", 0), ("SSRC1", 1)], None),
            select(2),
            lower(&[("SSRC0", 1), ("SSRC1", 0)], None),
            select(3),
            lower(&[("SSRC0", 0), ("SSRC1", 255)], Some(5)),
            select(4),
            lower(&[("SSRC0", 251), ("SSRC1", 252)], None),
            select(5),
            StageAOp::Complete,
        ];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[0] = 4;
        initial[0].scalar[1] = 5;
        initial[0].exec = [1, 0];
        initial[1].scalar[0] = 6;
        initial[1].scalar[1] = 6;
        initial[1].vcc = [1, 0];

        let expected = run_stage_a_pair([&ops, &ops], initial, 32).unwrap();
        assert_eq!(&expected[0].scalar[2..6], &[1, 0, 1, 0]);
        assert_eq!(&expected[1].scalar[2..6], &[1, 1, 0, 1]);
        let Some(gpu) = run_gpu([&ops, &ops], initial, 32, 1) else {
            return;
        };
        for wave in 0..2 {
            assert_eq!(gpu[0][wave].scc, expected[wave].scc);
            assert_eq!(gpu[0][wave].scalar, expected[wave].scalar);
        }

        let invalid = encode(
            "S_CMP_LE_U32",
            "ENC_SOPC",
            &[("SSRC0", 254), ("SSRC1", 0)],
            None,
        )
        .unwrap();
        assert!(lower_stage_a_one(&rdna2::decode(&invalid, 0).unwrap()).is_err());
    }

    #[test]
    fn material_scalar_masks_match_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let lower = |name, encoding, fields: &[(&str, u64)]| {
            let code = encode(name, encoding, fields, None).unwrap();
            lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let program = [
            lower("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 193)]),
            lower("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 129)]),
            lower(
                "S_AND_B64",
                "ENC_SOP2",
                &[("SDST", 4), ("SSRC0", 0), ("SSRC1", 2)],
            ),
            lower(
                "S_OR_B64",
                "ENC_SOP2",
                &[("SDST", 6), ("SSRC0", 4), ("SSRC1", 2)],
            ),
            lower(
                "S_XOR_B64",
                "ENC_SOP2",
                &[("SDST", 8), ("SSRC0", 0), ("SSRC1", 2)],
            ),
            lower(
                "S_ANDN2_B64",
                "ENC_SOP2",
                &[("SDST", 10), ("SSRC0", 0), ("SSRC1", 2)],
            ),
            lower("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 253)]),
            lower("S_MOV_B32", "ENC_SOP1", &[("SDST", 13), ("SSRC0", 130)]),
            lower("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 13)]),
            lower(
                "S_AND_SAVEEXEC_B64",
                "ENC_SOP1",
                &[("SDST", 14), ("SSRC0", 10)],
            ),
            StageAOp::Complete,
        ];
        let initial = [LogicalWaveState {
            exec: [0xffff_ffff, 0x8000_0001],
            ..Default::default()
        }; 2];
        let expected = run_stage_a_pair([&program, &program], initial, 32).unwrap();
        for state in &expected {
            assert_eq!([state.scalar[4], state.scalar[5]], [1, 0]);
            assert_eq!([state.scalar[8], state.scalar[9]], [0xffff_fffe, u32::MAX]);
            assert_eq!(
                [state.scalar[10], state.scalar[11]],
                [0xffff_fffe, u32::MAX]
            );
            assert_eq!([state.scalar[12], state.scalar[13]], [1, 2]);
            assert_eq!([state.scalar[14], state.scalar[15]], initial[0].exec);
            assert_eq!(state.exec, [0xffff_fffe, 0x8000_0001]);
            assert_eq!(state.scc, 1);
        }
        assert_eq!(
            run_gpu([&program, &program], initial, 32, 2).unwrap(),
            vec![expected; 2]
        );

        let unsupported = encode(
            "S_MOV_B64",
            "ENC_SOP1",
            &[("SDST", 124), ("SSRC0", 0)],
            None,
        )
        .unwrap();
        assert!(lower_stage_a_one(&rdna2::decode(&unsupported, 0).unwrap()).is_err());
    }

    #[test]
    fn decoded_exec_mask_writes_match_cpu_and_gpu() {
        // Fixed SOP1 encodings keep opcode and field placement independent of
        // the repository encoder: s_andn2_wrexec_b64 s[18:19], s[20:21] and
        // s_or_saveexec_b64 s[16:17], s[20:21].
        let wr_decoded = rdna2::decode(&[0x14, 0x3a, 0x92, 0xbe], 0).unwrap();
        let or_decoded = rdna2::decode(&[0x14, 0x25, 0x90, 0xbe], 4).unwrap();
        assert_eq!(wr_decoded.inst.name, "S_ANDN2_WREXEC_B64");
        assert_eq!(or_decoded.inst.name, "S_OR_SAVEEXEC_B64");
        let wr = lower_stage_a_one(&wr_decoded).unwrap();
        let save_or = lower_stage_a_one(&or_decoded).unwrap();
        assert_eq!(
            wr,
            StageAOp::WrExecAndNot64 {
                dst: 18,
                source: Operand64::Register(20),
            }
        );
        assert_eq!(
            save_or,
            StageAOp::SaveExecOr64 {
                dst: 16,
                source: Operand64::Register(20),
            }
        );

        let program = [wr, save_or, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].exec = [0x0f0f_0f0f, 0x8000_0001];
        initial[0].scalar[20] = 0xffff_00ff;
        initial[0].scalar[21] = 0xf000_000f;
        initial[1].exec = [0xaaaa_5555, 0x1234_5678];
        initial[1].scalar[20] = 0xaaaa_5555;
        initial[1].scalar[21] = 0x1234_5678;

        let actual = run_stage_a_pair([&program, &program], initial, 8).unwrap();
        let mut expected = initial;
        expected[0].scalar[18] = 0xf0f0_00f0;
        expected[0].scalar[19] = 0x7000_000e;
        expected[0].scalar[16] = 0xf0f0_00f0;
        expected[0].scalar[17] = 0x7000_000e;
        expected[0].exec = [0xffff_00ff, 0xf000_000f];
        expected[0].scc = 1;
        expected[1].scalar[18] = 0;
        expected[1].scalar[19] = 0;
        expected[1].scalar[16] = 0;
        expected[1].scalar[17] = 0;
        expected[1].exec = [0xaaaa_5555, 0x1234_5678];
        expected[1].scc = 1;
        for state in &mut expected {
            state.pc = 2;
            state.status = LogicalWaveStatus::Complete;
            state.steps = 3;
        }
        assert_eq!(actual, expected);
        assert_eq!(
            run_gpu([&program, &program], initial, 8, 2).unwrap(),
            vec![expected; 2]
        );

        // Destination/source aliasing must use the source pair before the
        // writeback replaces it with the newly computed EXEC value.
        let alias_code = rdna2::encode(
            "S_ANDN2_WREXEC_B64",
            "ENC_SOP1",
            &[("SDST", 20), ("SSRC0", 20)],
            None,
        )
        .unwrap();
        let alias = lower_stage_a_one(&rdna2::decode(&alias_code, 0).unwrap()).unwrap();
        let alias_program = [alias, StageAOp::Complete];
        let alias_actual = run_stage_a_pair([&alias_program, &alias_program], initial, 4).unwrap();
        assert_eq!(&alias_actual[0].scalar[20..22], &[0xf0f0_00f0, 0x7000_000e]);
        assert_eq!(alias_actual[0].exec, [0xf0f0_00f0, 0x7000_000e]);
    }

    #[test]
    fn decoded_saveexec_can_snapshot_into_vcc() {
        // Fixed observed SOP1 encoding: s_and_saveexec_b64 vcc, vcc.
        let decoded = rdna2::decode(&[0x6a, 0x24, 0xea, 0xbe], 0).unwrap();
        assert_eq!(decoded.inst.name, "S_AND_SAVEEXEC_B64");
        let op = lower_stage_a_one(&decoded).unwrap();
        assert_eq!(
            op,
            StageAOp::SaveExecAnd64 {
                dst: 106,
                source: Operand64::Register(106),
            }
        );

        let program = [op, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].exec = [0xffff_00ff, 0x8000_00ff];
        initial[0].vcc = [0x0f0f_0f0f, 0xffff_0000];
        initial[1].exec = [0xaaaa_5555, 0x1234_5678];
        initial[1].vcc = [0xff00_ff00, 0x00ff_00ff];
        let expected = run_stage_a_pair([&program, &program], initial, 8).unwrap();
        assert_eq!(expected[0].exec, [0x0f0f_000f, 0x8000_0000]);
        assert_eq!(expected[0].vcc, initial[0].exec);
        assert_eq!(expected[1].exec, [0xaa00_5500, 0x0034_0078]);
        assert_eq!(expected[1].vcc, initial[1].exec);
        assert_eq!(expected[0].scc, 1);
        assert_eq!(expected[1].scc, 1);
        assert_eq!(
            run_gpu([&program, &program], initial, 8, 1).unwrap(),
            vec![expected]
        );
    }

    #[test]
    fn paired_barriers_wait_repeat_and_fail_bounded() {
        let (a, b) = programs();
        let states = run_stage_a_pair([&a, &b], [LogicalWaveState::default(); 2], 32).unwrap();
        assert!(states
            .iter()
            .all(|s| s.status == LogicalWaveStatus::Complete));
        assert_eq!([states[0].epoch, states[1].epoch], [2, 2]);
        assert_eq!([states[0].scalar[0], states[1].scalar[0]], [2, 2]);
        assert_ne!(states[0].steps, states[1].steps);
        let never = [StageAOp::Jump(0)];
        let wait = [StageAOp::Barrier];
        let failed =
            run_stage_a_pair([&wait, &never], [LogicalWaveState::default(); 2], 8).unwrap();
        assert!(failed.iter().all(|s| matches!(
            s.status,
            LogicalWaveStatus::StepLimit | LogicalWaveStatus::Fault
        )));
        let mut mismatch = [LogicalWaveState::default(); 2];
        mismatch[1].epoch = 1;
        let mismatch = run_stage_a_pair([&wait, &wait], mismatch, 8).unwrap();
        assert!(mismatch
            .iter()
            .all(|state| state.status == LogicalWaveStatus::Fault));
    }

    #[test]
    fn gpu_stage_a_matches_cpu_pair() {
        let (a, b) = programs();
        let initial = [LogicalWaveState {
            exec: [u32::MAX; 2],
            ..Default::default()
        }; 2];
        let Some(gpu) = run_gpu([&a, &b], initial, 32, 1) else {
            return;
        };
        let expected = run_stage_a_pair([&a, &b], initial, 32).unwrap();
        assert_eq!(gpu, vec![expected]);

        let never = [StageAOp::Jump(0)];
        let wait = [StageAOp::Barrier];
        let initial = [LogicalWaveState::default(); 2];
        let expected = run_stage_a_pair([&wait, &never], initial, 8).unwrap();
        assert_eq!(
            run_gpu([&wait, &never], initial, 8, 1).unwrap(),
            vec![expected]
        );

        let mut scalar = LogicalWaveState::default();
        scalar.scalar[0] = 0x7fff_ffff;
        scalar.scalar[1] = 1;
        scalar.scc = 1;
        let edge = [
            StageAOp::Binary {
                kind: ScalarBinaryKind::AddI32,
                dst: 2,
                lhs: Operand32::Register(0),
                rhs: Operand32::Register(1),
            },
            StageAOp::Binary {
                kind: ScalarBinaryKind::BfeI32,
                dst: 3,
                lhs: Operand32::Constant(0x80),
                rhs: Operand32::Constant(8 << 16),
            },
            StageAOp::CompareEq {
                lhs: Operand32::Register(3),
                rhs: Operand32::Constant(0xffff_ff80),
            },
            StageAOp::Select64 {
                dst: 4,
                when_set: Operand64::Constant(0x1122_3344_5566_7788),
                when_clear: Operand64::Constant(0),
            },
            StageAOp::Complete,
        ];
        let initial = [scalar; 2];
        let expected = run_stage_a_pair([&edge, &edge], initial, 16).unwrap();
        assert_eq!(
            run_gpu([&edge, &edge], initial, 16, 1).unwrap(),
            vec![expected]
        );

        let mut fast = Vec::new();
        let mut slow = Vec::new();
        for generation in 0..12 {
            fast.push(StageAOp::NoEffect);
            slow.extend(std::iter::repeat_n(StageAOp::NoEffect, 1 + generation % 4));
            fast.push(StageAOp::Barrier);
            slow.push(StageAOp::Barrier);
        }
        fast.push(StageAOp::Complete);
        slow.push(StageAOp::Complete);
        let initial = [LogicalWaveState::default(); 2];
        let expected = run_stage_a_pair([&fast, &slow], initial, 128).unwrap();
        assert_eq!(expected[0].epoch, 12);
        assert_eq!(expected[1].epoch, 12);
        assert_eq!(
            run_gpu([&fast, &slow], initial, 128, 4).unwrap(),
            vec![expected; 4]
        );
    }

    #[test]
    fn decoded_forms_lower_but_real_gate_stays_closed() {
        use crate::gpu::rdna2::encode;
        let cases = [
            (
                "S_ADD_I32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 129)],
            ),
            (
                "S_MUL_I32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 193), ("SSRC1", 129)],
            ),
            (
                "S_AND_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            (
                "S_OR_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            (
                "S_XOR_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            (
                "S_LSHL_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 129)],
            ),
            (
                "S_LSHR_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 129)],
            ),
            (
                "S_BFE_U32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            (
                "S_BFE_I32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            ("S_CMP_EQ_U32", "ENC_SOPC", vec![("SSRC0", 0), ("SSRC1", 1)]),
            ("S_CMP_GT_U32", "ENC_SOPC", vec![("SSRC0", 0), ("SSRC1", 1)]),
            (
                "S_CSELECT_B32",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 4)],
            ),
            (
                "S_CSELECT_B64",
                "ENC_SOP2",
                vec![("SDST", 2), ("SSRC0", 0), ("SSRC1", 4)],
            ),
            (
                "S_MOVK_I32",
                "ENC_SOPK",
                vec![("SDST", 2), ("SIMM16", 0x8000)],
            ),
            ("S_WAITCNT", "ENC_SOPP", vec![("SIMM16", 0)]),
            ("S_INST_PREFETCH", "ENC_SOPP", vec![("SIMM16", 0)]),
            ("S_NOP", "ENC_SOPP", vec![("SIMM16", 0)]),
            ("S_BARRIER", "ENC_SOPP", vec![]),
        ];
        for (name, enc, fields) in cases {
            let code = encode(name, enc, &fields, None).unwrap();
            let d = rdna2::decode(&code, 0).unwrap();
            assert!(lower_stage_a_one(&d).is_ok(), "{name}");
            assert!(super::super::mesh_control::lower(&code).is_err());
        }
        let code = encode(
            "S_AND_B32",
            "ENC_SOP2",
            &[("SDST", 2), ("SSRC0", 106), ("SSRC1", 107)],
            None,
        )
        .unwrap();
        let decoded = rdna2::decode(&code, 0).unwrap();
        assert_eq!(
            lower_stage_a_one(&decoded).unwrap(),
            StageAOp::Binary {
                kind: ScalarBinaryKind::And,
                dst: 2,
                lhs: Operand32::Register(106),
                rhs: Operand32::Register(107),
            }
        );
        let mut vcc = LogicalWaveState::default();
        vcc.vcc = [0xf0, 0x3c];
        let program = [lower_stage_a_one(&decoded).unwrap(), StageAOp::Complete];
        let states = run_stage_a_pair([&program, &program], [vcc; 2], 8).unwrap();
        assert!(states.iter().all(|state| state.scalar[2] == 0x30));
    }

    #[test]
    fn signed_multiply_wraps_and_preserves_scc_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let code = encode(
            "S_MUL_I32",
            "ENC_SOP2",
            &[("SDST", 6), ("SSRC0", 193), ("SSRC1", 255)],
            Some(1234),
        )
        .unwrap();
        let op = lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap();
        assert_eq!(
            op,
            StageAOp::Binary {
                kind: ScalarBinaryKind::MulI32,
                dst: 6,
                lhs: Operand32::Constant(u32::MAX),
                rhs: Operand32::Constant(1234),
            }
        );
        let program = [op, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scc = 0;
        initial[1].scc = 1;
        let expected = run_stage_a_pair([&program, &program], initial, 8).unwrap();
        assert_eq!(expected[0].scalar[6], 0u32.wrapping_sub(1234));
        assert_eq!(expected[1].scalar[6], 0u32.wrapping_sub(1234));
        assert_eq!([expected[0].scc, expected[1].scc], [0, 1]);
        assert_eq!(
            run_gpu([&program, &program], initial, 8, 2).unwrap(),
            vec![expected; 2]
        );
    }

    #[test]
    fn scalar_integer_edges_match_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let cases = [
            (
                "S_MIN_U32",
                "ENC_SOP2",
                vec![("SDST", 8), ("SSRC0", 0), ("SSRC1", 1)],
                None,
                [5, 2],
                [1, 0],
                [(5, 7, 0), (9, 2, 1)],
            ),
            (
                "S_MUL_HI_U32",
                "ENC_SOP2",
                vec![("SDST", 8), ("SSRC0", 0), ("SSRC1", 1)],
                None,
                [1, 2],
                [0, 1],
                [(u32::MAX, 2, 0), (0x8000_0000, 4, 1)],
            ),
            (
                "S_NOT_B32",
                "ENC_SOP1",
                vec![("SDST", 8), ("SSRC0", 0)],
                None,
                [u32::MAX, 0],
                [1, 0],
                [(0, 0, 0), (u32::MAX, 0, 1)],
            ),
            (
                "S_SUB_I32",
                "ENC_SOP2",
                vec![("SDST", 8), ("SSRC0", 0), ("SSRC1", 1)],
                None,
                [0x7fff_ffff, 5],
                [1, 0],
                [(0x8000_0000, 1, 0), (7, 2, 1)],
            ),
        ];

        for (name, encoding, fields, literal, values, sccs, inputs) in cases {
            let code = encode(name, encoding, &fields, literal).unwrap();
            let decoded = rdna2::decode(&code, 0).unwrap();
            let op = lower_stage_a_one(&decoded).unwrap();
            assert!(super::super::mesh_control::lower(&code).is_err());
            let program = [op, StageAOp::Complete];
            let mut initial = [LogicalWaveState::default(); 2];
            for wave in 0..2 {
                initial[wave].scalar[0] = inputs[wave].0;
                initial[wave].scalar[1] = inputs[wave].1;
                initial[wave].scc = inputs[wave].2;
            }
            let expected = run_stage_a_pair([&program, &program], initial, 8).unwrap();
            for wave in 0..2 {
                assert_eq!(expected[wave].scalar[8], values[wave], "{name} wave {wave}");
                assert_eq!(expected[wave].scc, sccs[wave], "{name} SCC wave {wave}");
            }
            assert_eq!(
                run_gpu([&program, &program], initial, 8, 2).unwrap(),
                vec![expected; 2],
                "{name}"
            );
        }
    }

    #[test]
    fn scalar_register_invariants_match_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let mut program = Vec::new();
        for (name, fields, literal) in [
            ("S_BITSET1_B32", vec![("SDST", 5), ("SSRC0", 133)], None),
            ("S_CMP_LG_U32", vec![("SSRC0", 0), ("SSRC1", 128)], None),
            ("S_SETREG_IMM32_B32", vec![("SIMM16", 0x0881)], Some(0)),
            ("S_SETREG_IMM32_B32", vec![("SIMM16", 0xf814)], Some(0)),
            ("S_SETREG_IMM32_B32", vec![("SIMM16", 0xf815)], Some(1024)),
        ] {
            let code = encode(
                name,
                if name == "S_CMP_LG_U32" {
                    "ENC_SOPC"
                } else if name == "S_BITSET1_B32" {
                    "ENC_SOP1"
                } else {
                    "ENC_SOPK"
                },
                &fields,
                literal,
            )
            .unwrap();
            program.push(lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap());
        }
        program.push(StageAOp::Complete);

        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scc = 1;
        initial[1].scc = 1;
        initial[1].scalar[0] = 7;
        let expected = run_stage_a_pair([&program, &program], initial, 16).unwrap();
        assert_eq!(expected[0].scalar[5], 32);
        assert_eq!(expected[1].scalar[5], 32);
        assert_eq!(expected[0].scc, 0);
        assert_eq!(expected[1].scc, 1);
        assert_eq!(
            run_gpu([&program, &program], initial, 16, 2).unwrap(),
            vec![expected; 2]
        );

        let bit_only = [
            StageAOp::BitSet1 {
                dst: 1,
                bit: Operand32::Constant(31),
            },
            StageAOp::Complete,
        ];
        let preserved = run_stage_a_pair([&bit_only, &bit_only], initial, 8).unwrap();
        assert!(preserved
            .iter()
            .all(|state| state.scc == 1 && state.scalar[1] == 0x8000_0000));

        for (encoded, value) in [(0x0881, 1), (0xf814, 1), (0xf816, 0)] {
            let code = encode(
                "S_SETREG_IMM32_B32",
                "ENC_SOPK",
                &[("SIMM16", encoded)],
                Some(value),
            )
            .unwrap();
            assert!(lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).is_err());
        }
    }

    #[test]
    fn inline_selectors_match_emitter_on_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let selectors32 = [
            (240u64, 0.5f32.to_bits()),
            (241, (-0.5f32).to_bits()),
            (242, 1.0f32.to_bits()),
            (243, (-1.0f32).to_bits()),
            (244, 2.0f32.to_bits()),
            (245, (-2.0f32).to_bits()),
            (246, 4.0f32.to_bits()),
            (247, (-4.0f32).to_bits()),
            (248, (1.0f32 / (2.0 * std::f32::consts::PI)).to_bits()),
            (255, 0xdead_beef),
        ];
        let selectors64 = [
            (128u64, 0u64),
            (129, 1),
            (192, 64),
            (193, u64::MAX),
            (208, (-16i64) as u64),
            (240, (0.5f32 as f64).to_bits()),
            (241, (-0.5f32 as f64).to_bits()),
            (242, (1.0f32 as f64).to_bits()),
            (243, (-1.0f32 as f64).to_bits()),
            (244, (2.0f32 as f64).to_bits()),
            (245, (-2.0f32 as f64).to_bits()),
            (246, (4.0f32 as f64).to_bits()),
            (247, (-4.0f32 as f64).to_bits()),
            (248, 0x3fc4_5f30_6dc9_c882),
            (253, 1),
            (255, 0xdead_beef),
        ];
        let mut program = Vec::new();
        for (index, &(selector, expected)) in selectors32.iter().enumerate() {
            let literal = (selector == 255).then_some(expected);
            let code = encode(
                "S_OR_B32",
                "ENC_SOP2",
                &[
                    ("SDST", 64 + index as u64),
                    ("SSRC0", selector),
                    ("SSRC1", 128),
                ],
                literal,
            )
            .unwrap();
            let op = lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap();
            assert!(matches!(
                op,
                StageAOp::Binary {
                    lhs: Operand32::Constant(value),
                    ..
                } if value == expected
            ));
            program.push(op);
        }
        for (index, &(selector, expected)) in selectors64.iter().enumerate() {
            let literal = (selector == 255).then_some(expected as u32);
            let code = encode(
                "S_CSELECT_B64",
                "ENC_SOP2",
                &[
                    ("SDST", (index * 2) as u64),
                    ("SSRC0", selector),
                    ("SSRC1", 128),
                ],
                literal,
            )
            .unwrap();
            let op = lower_stage_a_one(&rdna2::decode(&code, 0).unwrap()).unwrap();
            assert!(match op {
                StageAOp::Select64 {
                    when_set: Operand64::Constant(value),
                    ..
                } => value == expected,
                StageAOp::Select64 {
                    when_set: Operand64::Scc,
                    ..
                } => selector == 253 && expected == 1,
                _ => false,
            });
            program.push(op);
        }
        program.push(StageAOp::Complete);
        let initial = [LogicalWaveState::default(); 2];
        let expected = run_stage_a_pair([&program, &program], initial, 64).unwrap();
        for state in &expected {
            for (index, &(_, value)) in selectors32.iter().enumerate() {
                assert_eq!(state.scalar[64 + index], value);
            }
            for (index, &(_, value)) in selectors64.iter().enumerate() {
                assert_eq!(state.scalar[index * 2], value as u32);
                assert_eq!(state.scalar[index * 2 + 1], (value >> 32) as u32);
            }
        }
        assert_eq!(
            run_gpu([&program, &program], initial, 64, 2).unwrap(),
            vec![expected; 2]
        );
    }

    #[test]
    fn scalar_carry_pc_immediate_and_signed_compares_match_cpu_and_gpu() {
        use crate::gpu::rdna2::encode;

        let lower = |name, encoding, fields: &[(&str, u64)]| {
            let code = encode(name, encoding, fields, None).unwrap();
            let decoded = rdna2::decode(&code, 0).unwrap();
            (lower_stage_a_one(&decoded).unwrap(), decoded.len as u64)
        };

        let (add, _) = lower(
            "S_ADD_U32",
            "ENC_SOP2",
            &[("SDST", 4), ("SSRC0", 0), ("SSRC1", 1)],
        );
        let (addc, _) = lower(
            "S_ADDC_U32",
            "ENC_SOP2",
            &[("SDST", 5), ("SSRC0", 2), ("SSRC1", 3)],
        );
        let (mulk, _) = lower(
            "S_MULK_I32",
            "ENC_SOPK",
            &[("SDST", 6), ("SIMM16", 0xfffd)],
        );
        let (lt, _) = lower(
            "S_CMP_LT_I32",
            "ENC_SOPC",
            &[("SSRC0", 7), ("SSRC1", 8)],
        );
        let (gt, _) = lower(
            "S_CMP_GT_I32",
            "ENC_SOPC",
            &[("SSRC0", 9), ("SSRC1", 10)],
        );
        let (getpc, next_pc) = lower("S_GETPC_B64", "ENC_SOP1", &[("SDST", 12)]);
        assert_eq!(getpc, StageAOp::GetPc64 { dst: 12, next_pc });

        let program = [add, addc, mulk, lt, gt, getpc, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[0] = u32::MAX;
        initial[0].scalar[1] = 1;
        initial[0].scalar[2] = u32::MAX;
        initial[0].scalar[3] = 0;
        initial[0].scalar[6] = 7;
        initial[0].scalar[7] = u32::MAX;
        initial[0].scalar[8] = 0;
        initial[0].scalar[9] = 4;
        initial[0].scalar[10] = (-2i32) as u32;

        initial[1].scalar[0] = 2;
        initial[1].scalar[1] = 3;
        initial[1].scalar[2] = 5;
        initial[1].scalar[3] = 6;
        initial[1].scalar[6] = (-7i32) as u32;
        initial[1].scalar[7] = 1;
        initial[1].scalar[8] = (-1i32) as u32;
        initial[1].scalar[9] = (-4i32) as u32;
        initial[1].scalar[10] = 2;

        let actual = run_stage_a_pair([&program, &program], initial, 16).unwrap();
        let mut expected = initial;
        expected[0].scalar[4] = 0;
        expected[0].scalar[5] = 0;
        expected[0].scalar[6] = (-21i32) as u32;
        expected[0].scalar[12] = next_pc as u32;
        expected[0].scalar[13] = (next_pc >> 32) as u32;
        expected[0].scc = 1;
        expected[1].scalar[4] = 5;
        expected[1].scalar[5] = 11;
        expected[1].scalar[6] = 21;
        expected[1].scalar[12] = next_pc as u32;
        expected[1].scalar[13] = (next_pc >> 32) as u32;
        expected[1].scc = 0;
        for state in &mut expected {
            state.pc = 6;
            state.status = LogicalWaveStatus::Complete;
            state.steps = 7;
        }
        assert_eq!(actual, expected);
        assert_eq!(
            run_gpu([&program, &program], initial, 16, 2).unwrap(),
            vec![expected; 2]
        );
    }

    #[test]
    fn decoded_scalar_xor_updates_scc_and_matches_gpu() {
        // S_XOR_B32 s5, s1, s2. Keeping this byte vector literal makes the
        // decoder expectation independent of the repository encoder.
        let decoded = rdna2::decode(&[0x01, 0x02, 0x05, 0x89], 0).unwrap();
        assert_eq!(decoded.inst.name, "S_XOR_B32");
        let xor = lower_stage_a_one(&decoded).unwrap();
        assert_eq!(
            xor,
            StageAOp::Binary {
                kind: ScalarBinaryKind::Xor,
                dst: 5,
                lhs: Operand32::Register(1),
                rhs: Operand32::Register(2),
            }
        );

        let program = [xor, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[1] = 0xa5a5_5a5a;
        initial[0].scalar[2] = 0xa5a5_5a5a;
        initial[0].scc = 1;
        initial[1].scalar[1] = 0xffff_0000;
        initial[1].scalar[2] = 0x0f0f_f0f0;

        let actual = run_stage_a_pair([&program, &program], initial, 8).unwrap();
        let mut expected = initial;
        expected[0].scalar[5] = 0;
        expected[0].scc = 0;
        expected[1].scalar[5] = 0xf0f0_f0f0;
        expected[1].scc = 1;
        for state in &mut expected {
            state.pc = 1;
            state.status = LogicalWaveStatus::Complete;
            state.steps = 2;
        }
        assert_eq!(actual, expected);
        assert_eq!(
            run_gpu([&program, &program], initial, 8, 2).unwrap(),
            vec![expected; 2]
        );
    }

    #[test]
    fn scalar_immediate_compares_extend_exactly_and_match_gpu() {
        // S_CMPK_EQ_I32 s5, -1 and S_CMPK_GT_U32 s6, 0xffff.
        // Literal byte vectors keep opcode and field placement independent of
        // the repository encoder used by most construction tests.
        let eq_decoded = rdna2::decode(&[0xff, 0xff, 0x85, 0xb1], 0).unwrap();
        let gt_decoded = rdna2::decode(&[0xff, 0xff, 0x86, 0xb5], 0).unwrap();
        assert_eq!(eq_decoded.inst.name, "S_CMPK_EQ_I32");
        assert_eq!(gt_decoded.inst.name, "S_CMPK_GT_U32");
        let eq = lower_stage_a_one(&eq_decoded).unwrap();
        let gt = lower_stage_a_one(&gt_decoded).unwrap();
        assert_eq!(
            eq,
            StageAOp::CompareEq {
                lhs: Operand32::Register(5),
                rhs: Operand32::Constant(u32::MAX),
            }
        );
        assert_eq!(
            gt,
            StageAOp::CompareGtU32 {
                lhs: Operand32::Register(6),
                rhs: Operand32::Constant(0xffff),
            }
        );

        let ge_code = rdna2::encode(
            "S_CMP_GE_I32",
            "ENC_SOPC",
            &[("SSRC0", 7), ("SSRC1", 8)],
            None,
        )
        .unwrap();
        let ge = lower_stage_a_one(&rdna2::decode(&ge_code, 0).unwrap()).unwrap();

        let check = |op: StageAOp, first: (u8, u32), second: (u8, u32), expected_scc: u32| {
            let program = [op, StageAOp::Complete];
            let mut initial = [LogicalWaveState::default(); 2];
            for state in &mut initial {
                state.scalar[first.0 as usize] = first.1;
                state.scalar[second.0 as usize] = second.1;
            }
            let actual = run_stage_a_pair([&program, &program], initial, 8).unwrap();
            assert!(actual.iter().all(|state| state.scc == expected_scc));
            assert_eq!(
                run_gpu([&program, &program], initial, 8, 1).unwrap(),
                vec![actual]
            );
        };

        check(eq, (5, u32::MAX), (0, 0), 1);
        check(eq, (5, 0xffff), (0, 0), 0);
        check(gt, (6, 0x1_0000), (0, 0), 1);
        check(gt, (6, 0xffff), (0, 0), 0);
        check(ge, (7, (-1i32) as u32), (8, (-1i32) as u32), 1);
    }

    #[test]
    fn decoded_arithmetic_shift_and_bit_counts_match_gpu() {
        // Literal encodings keep opcode and field placement independent of the
        // encoder used by construction-oriented tests.
        let ashr_decoded = rdna2::decode(&[0x01, 0x82, 0x05, 0x91], 0).unwrap();
        let bcnt32_decoded = rdna2::decode(&[0x01, 0x0f, 0x86, 0xbe], 0).unwrap();
        let bcnt64_decoded = rdna2::decode(&[0x02, 0x10, 0x87, 0xbe], 0).unwrap();
        assert_eq!(ashr_decoded.inst.name, "S_ASHR_I32");
        assert_eq!(bcnt32_decoded.inst.name, "S_BCNT1_I32_B32");
        assert_eq!(bcnt64_decoded.inst.name, "S_BCNT1_I32_B64");

        let ashr = lower_stage_a_one(&ashr_decoded).unwrap();
        let bcnt32 = lower_stage_a_one(&bcnt32_decoded).unwrap();
        let bcnt64 = lower_stage_a_one(&bcnt64_decoded).unwrap();
        assert_eq!(
            ashr,
            StageAOp::Binary {
                kind: ScalarBinaryKind::AshrI32,
                dst: 5,
                lhs: Operand32::Register(1),
                rhs: Operand32::Constant(2),
            }
        );
        assert_eq!(
            bcnt32,
            StageAOp::CountBits32 {
                dst: 6,
                source: Operand32::Register(1),
            }
        );
        assert_eq!(
            bcnt64,
            StageAOp::CountBits64 {
                dst: 7,
                source: Operand64::Register(2),
            }
        );

        let program = [ashr, bcnt32, bcnt64, StageAOp::Complete];
        let mut initial = [LogicalWaveState::default(); 2];
        initial[0].scalar[1] = 0x8000_0001;
        initial[0].scalar[2] = u32::MAX;
        initial[0].scalar[3] = 0x8000_0000;
        initial[1].scalar[1] = 0;
        let actual = run_stage_a_pair([&program, &program], initial, 8).unwrap();
        let mut expected = initial;
        expected[0].scalar[5] = 0xe000_0000;
        expected[0].scalar[6] = 2;
        expected[0].scalar[7] = 33;
        expected[0].scc = 1;
        expected[1].scalar[5] = 0;
        expected[1].scalar[6] = 0;
        expected[1].scalar[7] = 0;
        expected[1].scc = 0;
        for state in &mut expected {
            state.pc = 3;
            state.status = LogicalWaveStatus::Complete;
            state.steps = 4;
        }
        assert_eq!(actual, expected);
        assert_eq!(
            run_gpu([&program, &program], initial, 8, 2).unwrap(),
            vec![expected; 2]
        );
    }
}
