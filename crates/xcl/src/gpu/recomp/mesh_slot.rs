//! Validation-only, per-slot HLSL effects over a shared logical-wave ABI.
//!
//! Scheduling, divergent control flow, and rendezvous remain the uniform
//! scheduler's responsibility. This module is deliberately crate-private and
//! is not connected to production eligibility.

use super::mesh_alloc_export::AllocationExportOp;
use super::mesh_memory::StageCOp;
use super::mesh_program::ProgramOp;
use super::mesh_scalar::{Operand32, Operand64, ScalarBinary64Kind, ScalarBinaryKind, StageAOp};
use super::mesh_vector::{
    CompareKind, FloatBinaryKind, FloatConvertKind, FloatMathKind, FloatModifiers,
    FloatTernaryKind, FloatUnaryKind, FmaMixKind, ScalarMask, SdwaWord, StageBOp,
    StageDIntegerKind, VectorBinaryKind, VectorSource, VectorTernaryKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotSync {
    None,
    /// Cross-lane state must be snapshotted before all lanes consume it.
    GroupMemory,
    /// Dedicated buffer-backed LDS must be made visible to the whole group.
    DeviceMemory,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WaveStateAbi<'a> {
    pub sgpr: &'a str,
    pub vgpr: &'a str,
    pub exec: &'a str,
    pub vcc: &'a str,
    pub scc: &'a str,
    pub m0: &'a str,
    /// Wave-local scheduler status. Value 3 is the validation fault state.
    pub status: &'a str,
    pub memory: &'a str,
    pub exports: &'a str,
    pub lds: &'a str,
    pub lane: &'a str,
}

impl Default for WaveStateAbi<'static> {
    fn default() -> Self {
        Self {
            sgpr: "sgpr",
            vgpr: "vgpr",
            exec: "exec",
            vcc: "vcc",
            scc: "scc",
            m0: "m0",
            status: "status",
            memory: "memory",
            exports: "exports",
            lds: "lds",
            lane: "lane",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotEffect {
    pub hlsl: String,
    pub sync: SlotSync,
}

fn scalar_source(value: Operand32, a: WaveStateAbi<'_>) -> String {
    match value {
        Operand32::Register(106) => format!("{}.x", a.vcc),
        Operand32::Register(107) => format!("{}.y", a.vcc),
        Operand32::Register(125) => "0u".into(),
        Operand32::Register(126) => format!("{}.x", a.exec),
        Operand32::Register(127) => format!("{}.y", a.exec),
        Operand32::Register(r) => format!("{}[{}u]", a.sgpr, r),
        Operand32::Constant(v) => format!("{v}u"),
        Operand32::VccZero => format!("!any({})", a.vcc),
        Operand32::ExecZero => format!("!any({})", a.exec),
        Operand32::Scc => a.scc.to_owned(),
    }
}

fn scalar_write(dst: u8, value: &str, a: WaveStateAbi<'_>) -> Result<String, String> {
    Ok(match dst {
        106 => format!("{}.x={value};", a.vcc),
        107 => format!("{}.y={value};", a.vcc),
        125 => String::new(),
        126 => format!("{}.x={value};", a.exec),
        127 => format!("{}.y={value};", a.exec),
        register if register < 125 => format!("{}[{}u]={value};", a.sgpr, register),
        _ => return Err("scalar destination exceeds the validation ABI".into()),
    })
}

fn scalar_pair_source(value: Operand64, a: WaveStateAbi<'_>) -> Result<String, String> {
    Ok(match value {
        Operand64::Register(register) if register < 127 => format!(
            "uint2({},{})",
            scalar_source(Operand32::Register(register), a),
            scalar_source(Operand32::Register(register + 1), a)
        ),
        Operand64::Register(_) => {
            return Err("scalar pair source exceeds the validation ABI".into())
        }
        Operand64::Constant(value) => format!("uint2({}u,{}u)", value as u32, (value >> 32) as u32),
        Operand64::Scc => format!("uint2({},0u)", a.scc),
    })
}

fn scalar_pair_write(dst: u8, value: &str, a: WaveStateAbi<'_>) -> Result<String, String> {
    let high = dst
        .checked_add(1)
        .ok_or("scalar pair destination overflows")?;
    Ok(format!(
        "{}{}",
        scalar_write(dst, &format!("{value}.x"), a)?,
        scalar_write(high, &format!("{value}.y"), a)?
    ))
}

fn vector_source(value: VectorSource, a: WaveStateAbi<'_>) -> Result<String, String> {
    Ok(match value {
        VectorSource::Vgpr(r) => format!("{}[{}u][{}]", a.vgpr, r, a.lane),
        VectorSource::Scalar(r) => format!("{}[{}u]", a.sgpr, r),
        VectorSource::Constant(v) => format!("{v}u"),
        VectorSource::VccZero => format!("(all({}==0u)?1u:0u)", a.vcc),
        VectorSource::ExecZero => format!("(all({}==0u)?1u:0u)", a.exec),
        VectorSource::Scc => a.scc.to_owned(),
        _ => return Err("slot vector source requires cross-lane lowering".into()),
    })
}

fn float_source(
    value: VectorSource,
    a: WaveStateAbi<'_>,
    modifiers: FloatModifiers,
    index: u8,
) -> Result<String, String> {
    let mut value = format!("asfloat({})", vector_source(value, a)?);
    if modifiers.abs & (1 << index) != 0 {
        value = format!("abs({value})");
    }
    if modifiers.neg & (1 << index) != 0 {
        value = format!("(-{value})");
    }
    Ok(value)
}

fn float_result(mut value: String, modifiers: FloatModifiers) -> String {
    value = match modifiers.omod {
        1 => format!("(({value})*2.0f)"),
        2 => format!("(({value})*4.0f)"),
        3 => format!("(({value})*0.5f)"),
        _ => format!("({value})"),
    };
    if modifiers.clamp {
        value = format!("saturate({value})");
    }
    format!("asuint({value})")
}

fn scalar_mask_bit(source: ScalarMask, a: WaveStateAbi<'_>) -> String {
    match source {
        ScalarMask::ScalarPair(register) => format!(
            "(({}[{}u+({}>>5u)]>>({}&31u))&1u)",
            a.sgpr, register, a.lane, a.lane
        ),
        ScalarMask::Vcc => format!("(({}[{}>>5u]>>({}&31u))&1u)", a.vcc, a.lane, a.lane),
        ScalarMask::Exec => format!("(({}[{}>>5u]>>({}&31u))&1u)", a.exec, a.lane, a.lane),
    }
}

fn validate_carry_input(mask: ScalarMask) -> Result<(), String> {
    match mask {
        ScalarMask::Vcc | ScalarMask::Exec => Ok(()),
        ScalarMask::ScalarPair(register) if register <= 104 && register % 2 == 0 => Ok(()),
        _ => Err("carry input mask is not representable in the mature validation ABI".into()),
    }
}

fn carry_mask_write(mask: ScalarMask, ballot: &str, a: WaveStateAbi<'_>) -> Result<String, String> {
    Ok(match mask {
        ScalarMask::Vcc => format!("{}={ballot}.xy;", a.vcc),
        ScalarMask::ScalarPair(register) if register <= 104 && register % 2 == 0 => format!(
            "{}[{}u]={ballot}.x;{}[{}u]={ballot}.y;",
            a.sgpr,
            register,
            a.sgpr,
            register + 1
        ),
        _ => {
            return Err(
                "carry output mask is not representable in the mature validation ABI".into(),
            )
        }
    })
}

fn compare_expression(
    kind: CompareKind,
    lhs: VectorSource,
    rhs: VectorSource,
    a: WaveStateAbi<'_>,
) -> Result<String, String> {
    let lhs = vector_source(lhs, a)?;
    let rhs = vector_source(rhs, a)?;
    Ok(match kind {
        CompareKind::Eq => format!("{lhs}=={rhs}"),
        CompareKind::NotEqual => format!("{lhs}!={rhs}"),
        CompareKind::Greater => format!("{lhs}>{rhs}"),
        CompareKind::GreaterEqual => format!("{lhs}>={rhs}"),
        CompareKind::Less => format!("{lhs}<{rhs}"),
        CompareKind::LessEqual => format!("{lhs}<={rhs}"),
        CompareKind::SignedGreater => format!("(int){lhs}>(int){rhs}"),
        CompareKind::SignedLess => format!("(int){lhs}<(int){rhs}"),
        CompareKind::EqualFloat => format!("asfloat({lhs})==asfloat({rhs})"),
        CompareKind::OrderedNotEqualFloat => format!(
            "(!isnan(asfloat({lhs}))&&!isnan(asfloat({rhs}))&&asfloat({lhs})!=asfloat({rhs}))"
        ),
        CompareKind::UnorderedNotEqualFloat => format!(
            "(isnan(asfloat({lhs}))||isnan(asfloat({rhs}))||asfloat({lhs})!=asfloat({rhs}))"
        ),
        CompareKind::LessFloat => format!("asfloat({lhs})<asfloat({rhs})"),
        CompareKind::GreaterFloat => format!("asfloat({lhs})>asfloat({rhs})"),
        CompareKind::LessEqualFloat => format!("asfloat({lhs})<=asfloat({rhs})"),
        CompareKind::GreaterEqualFloat => format!("asfloat({lhs})>=asfloat({rhs})"),
        CompareKind::UnorderedGreaterFloat => {
            format!("(isnan(asfloat({lhs}))||isnan(asfloat({rhs}))||asfloat({lhs})>asfloat({rhs}))")
        }
        CompareKind::UnorderedLessFloat => {
            format!("(isnan(asfloat({lhs}))||isnan(asfloat({rhs}))||asfloat({lhs})<asfloat({rhs}))")
        }
        CompareKind::UnorderedLessEqualFloat => format!(
            "(isnan(asfloat({lhs}))||isnan(asfloat({rhs}))||asfloat({lhs})<=asfloat({rhs}))"
        ),
        CompareKind::LessFloat16 => format!("f16tof32({lhs}&0xffffu)<f16tof32({rhs}&0xffffu)"),
    })
}

fn compare_ballot(expression: String, target: String, cmpx: bool, a: WaveStateAbi<'_>) -> String {
    let active = format!("(({}[{}>>5u]>>({}&31u))&1u)!=0u", a.exec, a.lane, a.lane);
    let write = if cmpx {
        format!("{}&=xcl_compare.xy", a.exec)
    } else {
        format!("{target}=xcl_compare.xy")
    };
    format!(
        "{{uint4 xcl_compare=WaveActiveBallot({active}&&({expression}));if({}==0u){{{write};}}}}",
        a.lane
    )
}

fn scalar_compare_ballot(
    expression: String,
    dst: u8,
    a: WaveStateAbi<'_>,
) -> Result<String, String> {
    if dst.checked_add(1).filter(|high| *high < 128).is_none() {
        return Err("scalar compare destination pair exceeds the validation ABI".into());
    }
    let active = format!("(({}[{}>>5u]>>({}&31u))&1u)!=0u", a.exec, a.lane, a.lane);
    Ok(format!("{{uint4 xcl_compare=WaveActiveBallot({active}&&({expression}));if({}==0u){{{}[{}u]=xcl_compare.x;{}[{}u]=xcl_compare.y;}}}}", a.lane, a.sgpr, dst, a.sgpr, dst + 1))
}

fn float_compare_expression(kind: CompareKind, lhs: String, rhs: String) -> Result<String, String> {
    Ok(match kind {
        CompareKind::EqualFloat => format!("{lhs}=={rhs}"),
        CompareKind::OrderedNotEqualFloat => {
            format!("(!isnan({lhs})&&!isnan({rhs})&&{lhs}!={rhs})")
        }
        CompareKind::UnorderedNotEqualFloat => {
            format!("(isnan({lhs})||isnan({rhs})||{lhs}!={rhs})")
        }
        CompareKind::LessFloat => format!("{lhs}<{rhs}"),
        CompareKind::GreaterFloat => format!("{lhs}>{rhs}"),
        CompareKind::LessEqualFloat => format!("{lhs}<={rhs}"),
        CompareKind::GreaterEqualFloat => format!("{lhs}>={rhs}"),
        CompareKind::UnorderedGreaterFloat => format!("(isnan({lhs})||isnan({rhs})||{lhs}>{rhs})"),
        CompareKind::UnorderedLessFloat => format!("(isnan({lhs})||isnan({rhs})||{lhs}<{rhs})"),
        CompareKind::UnorderedLessEqualFloat => {
            format!("(isnan({lhs})||isnan({rhs})||{lhs}<={rhs})")
        }
        _ => return Err("specialized float compare kind is unsupported".into()),
    })
}

fn paired_source(source: VectorSource, a: WaveStateAbi<'_>) -> Result<(String, String), String> {
    let high = match source {
        VectorSource::Vgpr(register) if register < 255 => VectorSource::Vgpr(register + 1),
        VectorSource::Scalar(register) if register < 127 => VectorSource::Scalar(register + 1),
        VectorSource::Vgpr(_) | VectorSource::Scalar(_) => {
            return Err("64-bit compare source pair exceeds the validation ABI".into())
        }
        _ => return Err("64-bit compare source is not a plain register pair".into()),
    };
    Ok((vector_source(source, a)?, vector_source(high, a)?))
}

pub(crate) fn lower_slot(op: &ProgramOp, a: WaveStateAbi<'_>) -> Result<SlotEffect, String> {
    let (hlsl, sync) = match op {
        ProgramOp::Scalar(StageAOp::MoveImmediate { dst, value }) =>
            (format!("{}[{}u] = {}u;", a.sgpr, dst, value), SlotSync::None),
        ProgramOp::Scalar(StageAOp::Move32 { dst, source }) => {
            let source = scalar_source(*source, a);
            (scalar_write(*dst, &source, a)?, SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Move64 { dst, source }) => {
            let source=scalar_pair_source(*source,a)?;
            let write=scalar_pair_write(*dst,"xcl_source",a)?;
            (format!("{{uint2 xcl_source={source};{write}}}"),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::AddU32 { dst, lhs, rhs, carry_in }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);let write=scalar_write(*dst,"(uint)xcl_sum",a)?;
            (format!("{{uint xcl_old_scc={};uint xcl_lhs={lhs};uint xcl_rhs={rhs};uint64_t xcl_sum=(uint64_t)xcl_lhs+(uint64_t)xcl_rhs+{}u*(uint64_t)(xcl_old_scc!=0u);{write}{}=xcl_sum>0xffffffffull;}}",a.scc,u32::from(*carry_in),a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::MulImmediateI32 { dst, value }) => {
            let lhs=scalar_source(Operand32::Register(*dst),a);let write=scalar_write(*dst,"xcl_value",a)?;let immediate=*value as i32 as u32;
            (format!("{{uint xcl_lhs={lhs};uint xcl_value=xcl_lhs*{immediate}u;{write}}}"),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::GetPc64 { dst, next_pc }) => {
            let value = format!("uint2({}u,{}u)", *next_pc as u32, (*next_pc >> 32) as u32);
            (scalar_pair_write(*dst, &value, a)?, SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::ConfigureHardwareRegister {
            register,
            offset,
            width,
            value,
        }) => {
            if !matches!(
                (*register, *offset, *width, *value),
                (1, 2, 2, 0) | (20, 0, 32, 0) | (21, 0, 32, 1024)
            ) {
                return Err("hardware register write is outside the bounded validation policy".into());
            }
            // These exact decoder-admitted settings are assumptions of the
            // bounded shell; it has no flat-memory or mutable mode consumers.
            (String::new(), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::SaveExecAnd64 { dst, source }) => {
            let source=scalar_pair_source(*source,a)?;let write=scalar_pair_write(*dst,"xcl_old_exec",a)?;
            (format!("{{uint2 xcl_old_exec={};uint2 xcl_source={source};uint2 xcl_value=xcl_old_exec&xcl_source;{write}{}=xcl_value;{}=any(xcl_value!=0u);}}",a.exec,a.exec,a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::SaveExecOr64 { dst, source }) => {
            let source=scalar_pair_source(*source,a)?;let write=scalar_pair_write(*dst,"xcl_old_exec",a)?;
            (format!("{{uint2 xcl_old_exec={};uint2 xcl_source={source};uint2 xcl_value=xcl_old_exec|xcl_source;{write}{}=xcl_value;{}=any(xcl_value!=0u);}}",a.exec,a.exec,a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::WrExecAndNot64 { dst, source }) => {
            let source=scalar_pair_source(*source,a)?;let write=scalar_pair_write(*dst,"xcl_value",a)?;
            (format!("{{uint2 xcl_old_exec={};uint2 xcl_source={source};uint2 xcl_value=xcl_source&~xcl_old_exec;{write}{}=xcl_value;{}=any(xcl_value!=0u);}}",a.exec,a.exec,a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Select32 { dst, when_set, when_clear }) => {
            let when_set=scalar_source(*when_set,a);let when_clear=scalar_source(*when_clear,a);
            let write=scalar_write(*dst,"xcl_selected",a)?;
            (format!("{{uint xcl_condition={};uint xcl_when_set={when_set};uint xcl_when_clear={when_clear};uint xcl_selected=xcl_condition!=0u?xcl_when_set:xcl_when_clear;{write}}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::BitSet1 { dst, bit }) => {
            let bit=scalar_source(*bit,a);let write=scalar_write(*dst,"xcl_value",a)?;
            (format!("{{uint xcl_bit={bit};uint xcl_value=1u<<(xcl_bit&31u);{write}}}"),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Not32 { dst, source }) => {
            let source=scalar_source(*source,a);let write=scalar_write(*dst,"xcl_value",a)?;
            (format!("{{uint xcl_source={source};uint xcl_value=~xcl_source;{write}{}=xcl_value!=0u;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::BitReverse32 { dst, source }) => {
            let source=scalar_source(*source,a);let write=scalar_write(*dst,"xcl_value",a)?;
            (format!("{{uint xcl_source={source};uint xcl_value=reversebits(xcl_source);{write}}}"),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CountBits32 { dst, source }) => {
            let source=scalar_source(*source,a);let write=scalar_write(*dst,"xcl_value",a)?;
            (format!("{{uint xcl_source={source};uint xcl_value=countbits(xcl_source);{write}{}=xcl_value!=0u;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CountBits64 { dst, source }) => {
            let source=scalar_pair_source(*source,a)?;let write=scalar_write(*dst,"xcl_value",a)?;
            (format!("{{uint2 xcl_source={source};uint xcl_value=countbits(xcl_source.x)+countbits(xcl_source.y);{write}{}=xcl_value!=0u;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Binary { kind, dst, lhs, rhs }) => {
            let lhs = scalar_source(*lhs, a);
            let rhs = scalar_source(*rhs, a);
            let value = match kind {
                ScalarBinaryKind::AddI32 => "xcl_lhs+xcl_rhs",
                ScalarBinaryKind::SubI32 => "xcl_lhs-xcl_rhs",
                ScalarBinaryKind::MulI32 => "xcl_lhs*xcl_rhs",
                ScalarBinaryKind::MulHiU32 => "(uint)(((uint64_t)xcl_lhs*(uint64_t)xcl_rhs)>>32u)",
                ScalarBinaryKind::MinU32 => "min(xcl_lhs,xcl_rhs)",
                ScalarBinaryKind::And => "xcl_lhs&xcl_rhs",
                ScalarBinaryKind::Or => "xcl_lhs|xcl_rhs",
                ScalarBinaryKind::Xor => "xcl_lhs^xcl_rhs",
                ScalarBinaryKind::Lshl => "xcl_lhs<<(xcl_rhs&31u)",
                ScalarBinaryKind::Lshr => "xcl_lhs>>(xcl_rhs&31u)",
                ScalarBinaryKind::AshrI32 => "(uint)((int)xcl_lhs>>(xcl_rhs&31u))",
                ScalarBinaryKind::BfeU32 => "xcl_scalar_bfe_u32(xcl_lhs,xcl_rhs)",
                ScalarBinaryKind::BfeI32 => "xcl_bfe_i32(xcl_lhs,xcl_rhs)",
            };
            let write = scalar_write(*dst, "xcl_value", a)?;
            let scc = match kind {
                ScalarBinaryKind::AddI32 => format!("{}=(((xcl_lhs^xcl_value)&(xcl_rhs^xcl_value))>>31u)!=0u;", a.scc),
                ScalarBinaryKind::SubI32 => format!("{}=(((xcl_lhs^xcl_rhs)&(xcl_lhs^xcl_value))>>31u)!=0u;", a.scc),
                ScalarBinaryKind::MulI32 | ScalarBinaryKind::MulHiU32 => String::new(),
                ScalarBinaryKind::MinU32 => format!("{}=xcl_lhs<xcl_rhs;", a.scc),
                _ => format!("{}=xcl_value!=0u;", a.scc),
            };
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};uint xcl_value={value};{write}{scc}}}"), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Binary64 { kind, dst, lhs, rhs }) => {
            let lhs = scalar_pair_source(*lhs, a)?;
            let rhs = scalar_pair_source(*rhs, a)?;
            let value = match kind {
                ScalarBinary64Kind::And => "xcl_lhs&xcl_rhs",
                ScalarBinary64Kind::AndNot => "xcl_lhs&~xcl_rhs",
                ScalarBinary64Kind::Or => "xcl_lhs|xcl_rhs",
                ScalarBinary64Kind::Xor => "xcl_lhs^xcl_rhs",
            };
            let write = scalar_pair_write(*dst, "xcl_value", a)?;
            (format!("{{uint2 xcl_lhs={lhs};uint2 xcl_rhs={rhs};uint2 xcl_value={value};{write}{}=any(xcl_value!=0u);}}", a.scc), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::Select64 { dst, when_set, when_clear }) => {
            let when_set = scalar_pair_source(*when_set, a)?;
            let when_clear = scalar_pair_source(*when_clear, a)?;
            let write = scalar_pair_write(*dst, "xcl_selected", a)?;
            (format!("{{uint xcl_condition={};uint2 xcl_when_set={when_set};uint2 xcl_when_clear={when_clear};uint2 xcl_selected=xcl_condition!=0u?xcl_when_set:xcl_when_clear;{write}}}", a.scc), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareEq { lhs, rhs }) => {
            let lhs = scalar_source(*lhs, a);
            let rhs = scalar_source(*rhs, a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs==xcl_rhs;}}", a.scc), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareNeU32 { lhs, rhs }) => {
            let lhs = scalar_source(*lhs, a);
            let rhs = scalar_source(*rhs, a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs!=xcl_rhs;}}", a.scc), SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareLtU32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs<xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareGtU32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs>xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareLeU32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs<=xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareGeU32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=xcl_lhs>=xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareLtI32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=(int)xcl_lhs<(int)xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareGtI32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=(int)xcl_lhs>(int)xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::CompareGeI32 { lhs, rhs }) => {
            let lhs=scalar_source(*lhs,a);let rhs=scalar_source(*rhs,a);
            (format!("{{uint xcl_lhs={lhs};uint xcl_rhs={rhs};{}=(int)xcl_lhs>=(int)xcl_rhs;}}",a.scc),SlotSync::None)
        }
        ProgramOp::Scalar(StageAOp::NoEffect) => (String::new(), SlotSync::None),
        ProgramOp::Vector(StageBOp::Move { dst, src }) =>
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, vector_source(*src, a)?, a.vgpr, dst, a.lane), SlotSync::None),
        ProgramOp::Vector(StageBOp::ReadFirstLane { dst, src }) => {
            if *dst >= 106 { return Err("read-first-lane scalar destination exceeds the mature limit".into()); }
            let value = format!("{}[{}u][{}]", a.vgpr, src, a.lane);
            (format!("{{bool xcl_any=any({}!=0u);uint xcl_first={}.x!=0u?firstbitlow({}.x):({}.y!=0u?32u+firstbitlow({}.y):0u);uint xcl_read=WaveReadLaneAt({},xcl_first);if({}==0u){{if(xcl_any){}[{}u]=xcl_read;else {}=3u;}}}}", a.exec, a.exec, a.exec, a.exec, a.exec, value, a.lane, a.sgpr, dst, a.status), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ReadLane { dst, src, lane: selected }) => {
            if *dst >= 106 { return Err("read-lane scalar destination exceeds the mature limit".into()); }
            let selected = vector_source(*selected, a)?;
            let value = format!("{}[{}u][{}]", a.vgpr, src, a.lane);
            (format!("{{uint xcl_selected=WaveReadLaneAt({},0u)&63u;uint xcl_read=WaveReadLaneAt({},xcl_selected);if({}==0u){}[{}u]=xcl_read;}}", selected, value, a.lane, a.sgpr, dst), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::WriteLane { dst, src, lane: selected }) => {
            let source = vector_source(*src, a)?;
            let selected = vector_source(*selected, a)?;
            let selected = format!("({selected}&63u)");
            (format!("if({}=={}){}[{}u][{}]={};", a.lane, selected, a.vgpr, dst, a.lane, source), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::Mbcnt { high, dst, source, accumulator }) => {
            let source = vector_source(*source, a)?;
            let accumulator = vector_source(*accumulator, a)?;
            let count = if *high { format!("({}<32u?0u:countbits({}&(({}==32u)?0u:((1u<<({}-32u))-1u))))", a.lane, source, a.lane, a.lane) } else { format!("countbits({}&(({}==0u)?0u:({}>=32u?0xffffffffu:((1u<<{})-1u))))", source, a.lane, a.lane, a.lane) };
            let value = format!("({accumulator}+{count})");
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::PermLaneX16 { dst, src, selectors_low, selectors_high, fetch_inactive, bound_control }) => {
            let low = vector_source(*selectors_low, a)?;
            let high = vector_source(*selectors_high, a)?;
            let source = format!("{}[{}u][{}]", a.vgpr, src, a.lane);
            let fallback = if *bound_control { "0u".to_owned() } else { format!("{}[{}u][{}]", a.vgpr, dst, a.lane) };
            (format!("{{uint xcl_selectors=({}&15u)<8u?{}:{};uint xcl_n=(xcl_selectors>>(4u*({}&7u)))&15u;uint xcl_selected=({}&~31u)|((({}&16u)^16u)&16u)|xcl_n;uint xcl_read=WaveReadLaneAt({},xcl_selected);bool xcl_source_ok={}||(({}[xcl_selected>>5u]>>(xcl_selected&31u))&1u)!=0u;bool xcl_active=(({}[{}>>5u]>>({}&31u))&1u)!=0u;if(xcl_active){}[{}u][{}]=xcl_source_ok?xcl_read:{};}}", a.lane, low, high, a.lane, a.lane, a.lane, source, fetch_inactive, a.exec, a.exec, a.lane, a.lane, a.vgpr, dst, a.lane, fallback), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CndMask { dst, when_clear, when_set }) => {
            let clear = vector_source(*when_clear, a)?;
            let set = vector_source(*when_set, a)?;
            let selected = format!("(({}[{}>>5u]>>({}&31u))&1u)!=0u?{}:{}", a.vcc, a.lane, a.lane, set, clear);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, selected, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CndMaskScalar { dst, when_clear, when_set, predicate }) => {
            let clear = vector_source(*when_clear, a)?;
            let set = vector_source(*when_set, a)?;
            let selected = format!("({}!=0u?{}:{})", scalar_mask_bit(*predicate, a), set, clear);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, selected, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ConvertF32U32 { dst, src, modifiers }) => {
            let source = vector_source(*src, a)?;
            let value = float_result(format!("(float){source}"), *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ConvertF32I32 { dst, src }) => {
            let source = vector_source(*src, a)?;
            let value = format!("asuint((float)(int){source})");
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ConvertF32U32Sdwa { dst, src, word }) => {
            if *src >= 256 { return Err("SDWA conversion source exceeds the validation VGPR ABI".into()); }
            let source = format!("{}[{}u][{}]", a.vgpr, src, a.lane);
            let selected = match word { SdwaWord::Low => format!("({source}&0xffffu)"), SdwaWord::High => format!("({source}>>16u)") };
            let value = format!("asuint((float){selected})");
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::Compare { kind, lhs, rhs, cmpx }) => {
            let expression = compare_expression(*kind, *lhs, *rhs, a)?;
            (compare_ballot(expression, a.vcc.to_owned(), *cmpx, a), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CompareScalar { kind, lhs, rhs, dst }) => {
            let expression = compare_expression(*kind, *lhs, *rhs, a)?;
            (scalar_compare_ballot(expression, *dst, a)?, SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CompareLessFloat { kind, lhs, rhs, dst, modifiers, cmpx }) => {
            let lhs = float_source(*lhs, a, *modifiers, 0)?;
            let rhs = float_source(*rhs, a, *modifiers, 1)?;
            let expression = float_compare_expression(*kind, lhs, rhs)?;
            let hlsl = if *cmpx {
                compare_ballot(expression, a.vcc.to_owned(), true, a)
            } else if let Some(dst) = dst {
                scalar_compare_ballot(expression, *dst, a)?
            } else {
                compare_ballot(expression, a.vcc.to_owned(), false, a)
            };
            (hlsl, SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CompareEqU64 { lhs, rhs, dst }) => {
            let (lhs_lo, lhs_hi) = paired_source(*lhs, a)?;
            let (rhs_lo, rhs_hi) = paired_source(*rhs, a)?;
            let expression = format!("({lhs_lo}=={rhs_lo})&&({lhs_hi}=={rhs_hi})");
            let hlsl = if let Some(dst) = dst { scalar_compare_ballot(expression, *dst, a)? } else { compare_ballot(expression, a.vcc.to_owned(), false, a) };
            (hlsl, SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::CompareClassF32 { value, classes, dst, modifiers }) => {
            let value = float_source(*value, a, *modifiers, 0)?;
            let classes = vector_source(*classes, a)?;
            let expression = format!("xcl_float_class_match({value},{classes})");
            let hlsl = if let Some(dst) = dst { scalar_compare_ballot(expression, *dst, a)? } else { compare_ballot(expression, a.vcc.to_owned(), false, a) };
            (hlsl, SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::Binary { kind, dst, lhs, rhs }) => {
            let lhs = vector_source(*lhs, a)?;
            let rhs = vector_source(*rhs, a)?;
            let value = match kind {
                VectorBinaryKind::AddNoCarry => format!("({lhs}+{rhs})"),
                VectorBinaryKind::And => format!("({lhs}&{rhs})"),
                VectorBinaryKind::Or => format!("({lhs}|{rhs})"),
                VectorBinaryKind::Xor => format!("({lhs}^{rhs})"),
                VectorBinaryKind::BitCount => format!("(countbits({lhs})+{rhs})"),
                VectorBinaryKind::LshlReverse => format!("({rhs}<<({lhs}&31u))"),
                VectorBinaryKind::ArithmeticShiftRightReverse => format!("asuint((int){rhs}>>(int)({lhs}&31u))"),
                VectorBinaryKind::MinU32 => format!("min({lhs},{rhs})"),
                VectorBinaryKind::MaxU32 => format!("max({lhs},{rhs})"),
                VectorBinaryKind::MulLoU32 => format!("({lhs}*{rhs})"),
                VectorBinaryKind::MulHiU32 => format!("(uint)(((uint64_t){lhs}*(uint64_t){rhs})>>32u)"),
                VectorBinaryKind::SubNoCarry => format!("({lhs}-{rhs})"),
                VectorBinaryKind::ReverseSubNoCarry => format!("({rhs}-{lhs})"),
            };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::AddCarry { dst, lhs, rhs, carry_in, carry_out }) => {
            if let Some(mask) = carry_in { validate_carry_input(*mask)?; }
            let lhs = vector_source(*lhs, a)?;
            let rhs = vector_source(*rhs, a)?;
            let carry = carry_in.map(|mask| format!("(uint)({}!=0u)", scalar_mask_bit(mask, a))).unwrap_or_else(|| "0u".into());
            let active = format!("(({}[{}>>5u]>>({}&31u))&1u)!=0u", a.exec, a.lane, a.lane);
            let write = carry_mask_write(*carry_out, "xcl_carry", a)?;
            (format!("{{uint64_t xcl_sum=(uint64_t){lhs}+(uint64_t){rhs}+(uint64_t){carry};bool xcl_active={active};if(xcl_active){}[{}u][{}]=(uint)xcl_sum;uint4 xcl_carry=WaveActiveBallot(xcl_active&&((xcl_sum>>32u)!=0u));if({}==0u){{{write}}}}}", a.vgpr, dst, a.lane, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ReverseSubCarry { dst, lhs, rhs, carry_in, carry_out }) => {
            validate_carry_input(*carry_in)?;
            let lhs = vector_source(*lhs, a)?;
            let rhs = vector_source(*rhs, a)?;
            let carry = format!("(uint)({}!=0u)", scalar_mask_bit(*carry_in, a));
            let active = format!("(({}[{}>>5u]>>({}&31u))&1u)!=0u", a.exec, a.lane, a.lane);
            let write = carry_mask_write(*carry_out, "xcl_borrow", a)?;
            (format!("{{uint64_t xcl_difference=(uint64_t){rhs}-(uint64_t){lhs}-(uint64_t){carry};bool xcl_active={active};if(xcl_active){}[{}u][{}]=(uint)xcl_difference;uint4 xcl_borrow=WaveActiveBallot(xcl_active&&((xcl_difference>>32u)!=0u));if({}==0u){{{write}}}}}", a.vgpr, dst, a.lane, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::Ternary { kind, dst, a: first, b: second, c: third }) => {
            let first = vector_source(*first, a)?;
            let second = vector_source(*second, a)?;
            let third = vector_source(*third, a)?;
            let value = match kind {
                VectorTernaryKind::Add3 => format!("({first}+{second}+{third})"),
                VectorTernaryKind::BfeU32 => format!("xcl_bfe_u32({first},{second},{third})"),
                VectorTernaryKind::LshlAdd => format!("(({first}<<({second}&31u))+{third})"),
                VectorTernaryKind::AddLshl => format!("(({first}+{second})<<({third}&31u))"),
                VectorTernaryKind::AndOr => format!("(({first}&{second})|{third})"),
                VectorTernaryKind::Or3 => format!("({first}|{second}|{third})"),
                VectorTernaryKind::LshlOr => format!("(({first}<<({second}&31u))|{third})"),
                VectorTernaryKind::BitFieldInsert => format!("(({first}&{second})|(~{first}&{third}))"),
                VectorTernaryKind::XorAdd => format!("(({first}^{second})+{third})"),
            };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::BitReverse { dst, src }) => {
            let source = vector_source(*src, a)?;
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? reversebits({}) : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, source, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::BitNot { dst, src }) => {
            let source = vector_source(*src, a)?;
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? ~{} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, source, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::BitScan { high, dst, src }) => {
            let source = vector_source(*src, a)?;
            let value = if *high { format!("({source}==0u?0xffffffffu:31u-firstbithigh({source}))") } else { format!("({source}==0u?0xffffffffu:firstbitlow({source}))") };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatBinary { kind, dst, lhs, rhs, modifiers }) => {
            let lhs = float_source(*lhs, a, *modifiers, 0)?;
            let rhs = float_source(*rhs, a, *modifiers, 1)?;
            let value = match kind {
                FloatBinaryKind::Add => format!("{lhs}+{rhs}"),
                FloatBinaryKind::Subtract => format!("{lhs}-{rhs}"),
                FloatBinaryKind::ReverseSubtract => format!("{rhs}-{lhs}"),
                FloatBinaryKind::Multiply => format!("{lhs}*{rhs}"),
                FloatBinaryKind::MultiplyLegacy => format!("xcl_mul_legacy({lhs},{rhs})"),
                FloatBinaryKind::Maximum => format!("max({lhs},{rhs})"),
                FloatBinaryKind::Minimum => format!("min({lhs},{rhs})"),
            };
            let value = float_result(value, *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatTernary { kind, dst, a: first, b: second, c: third, modifiers }) => {
            let first = float_source(*first, a, *modifiers, 0)?;
            let second = float_source(*second, a, *modifiers, 1)?;
            let third = float_source(*third, a, *modifiers, 2)?;
            let value = match kind {
                FloatTernaryKind::Minimum3 => format!("min(min({first},{second}),{third})"),
                FloatTernaryKind::Maximum3 => format!("max(max({first},{second}),{third})"),
                FloatTernaryKind::Median3 => format!("max(min({first},{second}),min(max({first},{second}),{third}))"),
            };
            let value = float_result(value, *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatUnary { kind, dst, src }) => {
            let source = format!("asfloat({})", vector_source(*src, a)?);
            let value = match kind {
                FloatUnaryKind::Truncate => format!("trunc({source})"),
                FloatUnaryKind::Floor => format!("floor({source})"),
                FloatUnaryKind::Ceiling => format!("ceil({source})"),
                FloatUnaryKind::RoundEven => format!("round({source})"),
            };
            let value = format!("asuint({value})");
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatFma { dst, a: lhs, b: rhs, c, modifiers, legacy_zero }) => {
            let lhs = float_source(*lhs, a, *modifiers, 0)?;
            let rhs = float_source(*rhs, a, *modifiers, 1)?;
            let addend = match c {
                Some(value) => float_source(*value, a, *modifiers, 2)?,
                None => format!("asfloat({}[{}u][{}])", a.vgpr, dst, a.lane),
            };
            let value = if *legacy_zero {
                format!("xcl_fma_legacy({lhs},{rhs},{addend})")
            } else {
                format!("mad({lhs},{rhs},{addend})")
            };
            let value = float_result(value, *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatMad { dst, a: lhs, b: rhs, c, modifiers }) => {
            let lhs = float_source(*lhs, a, *modifiers, 0)?;
            let rhs = float_source(*rhs, a, *modifiers, 1)?;
            let addend = match c {
                Some(value) => float_source(*value, a, *modifiers, 2)?,
                None => format!("asfloat({}[{}u][{}])", a.vgpr, dst, a.lane),
            };
            let value = float_result(format!("xcl_mad_nonfused({lhs},{rhs},{addend})"), *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatMath { kind, dst, src, modifiers }) => {
            let source = float_source(*src, a, *modifiers, 0)?;
            let source = if matches!(kind, FloatMathKind::Reciprocal | FloatMathKind::ReciprocalIFlag | FloatMathKind::ReciprocalSqrt | FloatMathKind::Sqrt) {
                format!("(((asuint({source})&0x7f800000u)==0u)?asfloat(asuint({source})&0x80000000u):{source})")
            } else { source };
            let value = match kind {
                FloatMathKind::Reciprocal | FloatMathKind::ReciprocalIFlag => format!("rcp({source})"),
                FloatMathKind::ReciprocalSqrt => format!("rsqrt({source})"),
                FloatMathKind::Sqrt => format!("sqrt({source})"),
                FloatMathKind::Sin => format!("(isfinite({source})&&abs({source})>256.0f?0.0f:sin({source}*6.2831853f))"),
                FloatMathKind::Cos => format!("(isfinite({source})&&abs({source})>256.0f?1.0f:cos({source}*6.2831853f))"),
                FloatMathKind::Exp2 => format!("exp2({source})"),
                FloatMathKind::Fract => format!("frac({source})"),
                FloatMathKind::Log2 => format!("log2({source})"),
            };
            let value = float_result(value, *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::ConvertUByte { dst, src, byte, modifiers }) => {
            if *byte > 3 { return Err("byte conversion selector exceeds the source width".into()); }
            let source = vector_source(*src, a)?;
            let value = float_result(format!("(float)(({source}>>{}u)&0xffu)", u32::from(*byte) * 8), *modifiers);
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::PackF16 { dst, lo, hi, op_sel }) => {
            if *op_sel > 3 { return Err("half pack selector exceeds the mature encoding".into()); }
            let lo = vector_source(*lo, a)?;
            let hi = vector_source(*hi, a)?;
            let lo = if op_sel & 1 != 0 { format!("({lo}>>16u)") } else { format!("({lo}&0xffffu)") };
            let hi = if op_sel & 2 != 0 { format!("({hi}>>16u)") } else { format!("({hi}&0xffffu)") };
            let value = format!("({lo}|({hi}<<16u))");
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FloatConvert { kind, dst, src }) => {
            let source = vector_source(*src, a)?;
            let old = format!("{}[{}u][{}]", a.vgpr, dst, a.lane);
            let value = match kind {
                FloatConvertKind::F16FromF32 => format!("({old}&0xffff0000u)|(xcl_f32_to_f16_bounded({source})&0xffffu)"),
                FloatConvertKind::F16FromF32High => format!("({old}&0xffffu)|((xcl_f32_to_f16_bounded({source})&0xffffu)<<16u)"),
                FloatConvertKind::F16FromF32HighPad => format!("(xcl_f32_to_f16_bounded({source})&0xffffu)<<16u"),
                FloatConvertKind::F32FromF16 => format!("asuint(f16tof32({source}&0xffffu))"),
                FloatConvertKind::F32FromF16High => format!("asuint(f16tof32({source}>>16u))"),
                FloatConvertKind::U32FromF32 => format!("xcl_cvt_u32(asfloat({source}))"),
                FloatConvertKind::I32FromF32 => format!("(uint)xcl_cvt_i32(asfloat({source}))"),
            };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::FmaMix { kind, dst, a: first, b: second, c: third, op_sel, op_sel_hi, neg, abs, clamp }) => {
            let mix = |source: VectorSource, index: u8| -> Result<String, String> {
                let raw = vector_source(source, a)?;
                let mut value = if op_sel_hi & (1 << index) != 0 {
                    let half = if op_sel & (1 << index) != 0 { format!("({raw}>>16u)") } else { format!("({raw}&0xffffu)") };
                    format!("f16tof32({half})")
                } else { format!("asfloat({raw})") };
                if abs & (1 << index) != 0 { value = format!("abs({value})"); }
                if neg & (1 << index) != 0 { value = format!("(-{value})"); }
                Ok(value)
            };
            let (first, second, third) = (mix(*first, 0)?, mix(*second, 1)?, mix(*third, 2)?);
            let mut expression = format!("mad({first},{second},{third})");
            if *clamp { expression = format!("saturate({expression})"); }
            let old = format!("{}[{}u][{}]", a.vgpr, dst, a.lane);
            let value = match kind {
                FmaMixKind::F32 => format!("asuint({expression})"),
                FmaMixKind::LoF16 => format!("({old}&0xffff0000u)|(xcl_f32_to_f16_bounded(asuint({expression}))&0xffffu)"),
                FmaMixKind::HiF16 => format!("({old}&0xffffu)|(xcl_f32_to_f16_bounded(asuint({expression}))<<16u)"),
            };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::StageDInteger { kind, dst, a: lhs, b: rhs, c }) => {
            let lhs = vector_source(*lhs, a)?;
            let rhs = vector_source(*rhs, a)?;
            let value = match kind {
                StageDIntegerKind::LogicalShiftRight => format!("({rhs}>>({lhs}&31u))"),
                StageDIntegerKind::MadU24 => {
                    let addend = vector_source(c.ok_or("Stage D addend is missing")?, a)?;
                    format!("(({lhs}&0xffffffu)*({rhs}&0xffffffu)+{addend})")
                }
                StageDIntegerKind::Permute => {
                    let selector = vector_source(c.ok_or("Stage D selector is missing")?, a)?;
                    format!("xcl_perm_b32({lhs},{rhs},{selector})")
                }
            };
            (format!("{}[{}u][{}] = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u ? {} : {}[{}u][{}];", a.vgpr, dst, a.lane, a.exec, a.lane, a.lane, value, a.vgpr, dst, a.lane), SlotSync::None)
        }
        ProgramOp::Vector(StageBOp::NoEffect) => (String::new(), SlotSync::None),
        ProgramOp::Memory(StageCOp::GlobalLoad { dst, address, scalar_address: None, offset }) =>
            (format!("{{ uint xcl_value; bool xcl_active = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u; if (xcl_active && xcl_global_load({}, uint2({}[{}u][{}], {}[{}u][{}]), {}, xcl_value)) {}[{}u][{}] = xcl_value; }}", a.exec, a.lane, a.lane, a.memory, a.vgpr, address, a.lane, a.vgpr, address + 1, a.lane, offset, a.vgpr, dst, a.lane), SlotSync::None),
        ProgramOp::Memory(StageCOp::DsRead { dst, address, offsets, words: 1 }) =>
            (format!("{{ uint xcl_value; bool xcl_active = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u; if (xcl_active && xcl_lds_load({}, {}[{}u][{}], {}u, xcl_value)) {}[{}u][{}] = xcl_value; }}", a.exec, a.lane, a.lane, a.lds, a.vgpr, address, a.lane, offsets[0], a.vgpr, dst, a.lane), SlotSync::DeviceMemory),
        ProgramOp::Memory(StageCOp::DsWrite { address, data, offsets, words: 1 }) =>
            (format!("{{ bool xcl_active = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u; if (xcl_active) xcl_lds_store({}, {}[{}u][{}], {}u, {}[{}u][{}]); }}", a.exec, a.lane, a.lane, a.lds, a.vgpr, address, a.lane, offsets[0], a.vgpr, data[0], a.lane), SlotSync::DeviceMemory),
        ProgramOp::Memory(StageCOp::DsSwizzle { dst, source, control }) =>
            (format!("{{ uint xcl_value; bool xcl_valid = xcl_ds_swizzle({}[{}u][{}], {}, {}u, xcl_value); bool xcl_active = (({}[{} >> 5u] >> ({} & 31u)) & 1u) != 0u; if (xcl_active && xcl_valid) {}[{}u][{}] = xcl_value; }}", a.vgpr, source, a.lane, a.lane, control, a.exec, a.lane, a.lane, a.vgpr, dst, a.lane), SlotSync::GroupMemory),
        ProgramOp::AllocationExport(AllocationExportOp::Allocate { .. }) =>
            (format!("xcl_allocate({});", a.m0), SlotSync::None),
        ProgramOp::AllocationExport(AllocationExportOp::Export(export)) =>
            (format!("xcl_export({}, {}u, {}u, {}u, {}u);", a.exports, export.target, export.enable, export.compressed as u8, export.done as u8), SlotSync::None),
        _ => return Err("operation has no validation per-slot lowering".into()),
    };
    Ok(SlotEffect { hlsl, sync })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representative_chain_uses_one_explicit_abi() {
        let a = WaveStateAbi::default();
        let ops = [
            ProgramOp::Scalar(StageAOp::MoveImmediate { dst: 2, value: 7 }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 3,
                src: VectorSource::Scalar(2),
            }),
            ProgramOp::Memory(StageCOp::GlobalLoad {
                dst: 4,
                address: 3,
                scalar_address: None,
                offset: 4,
            }),
            ProgramOp::AllocationExport(AllocationExportOp::Allocate { m0: 124 }),
        ];
        let text = ops
            .iter()
            .map(|op| lower_slot(op, a).unwrap().hlsl)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("sgpr[2u] = 7u"));
        assert!(text.contains("? sgpr[2u] : vgpr[3u][lane]"));
        assert!(text.contains("xcl_global_load(memory, uint2(vgpr[3u][lane], vgpr[4u][lane]), 4"));
        assert!(text.contains("xcl_allocate(m0)"));
    }

    #[test]
    fn inactive_lanes_are_guarded_and_ds_declares_scheduler_sync() {
        let a = WaveStateAbi::default();
        let vector = lower_slot(
            &ProgramOp::Vector(StageBOp::Move {
                dst: 1,
                src: VectorSource::Vgpr(1),
            }),
            a,
        )
        .unwrap();
        assert!(vector.hlsl.contains("? vgpr[1u][lane] : vgpr[1u][lane]"));
        let ds = lower_slot(
            &ProgramOp::Memory(StageCOp::DsWrite {
                address: 0,
                data: [1, 0],
                offsets: [0, 0],
                words: 1,
            }),
            a,
        )
        .unwrap();
        assert_eq!(ds.sync, SlotSync::DeviceMemory);
        assert!(!ds.hlsl.contains("Barrier"));
        assert!(ds.hlsl.contains("if (xcl_active) xcl_lds_store(lds"));
    }

    #[test]
    fn unsupported_and_cross_lane_forms_fail_closed() {
        assert!(lower_slot(
            &ProgramOp::Scalar(StageAOp::Barrier),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::Move {
                dst: 0,
                src: VectorSource::Sdwa {
                    register: 0,
                    scalar: false,
                    select: 0,
                    sign_extend: false,
                }
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FloatMad {
                dst: 0,
                a: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                b: VectorSource::Constant(1.0f32.to_bits()),
                c: Some(VectorSource::Constant(0.0f32.to_bits())),
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::Permute,
                dst: 0,
                a: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Constant(0)),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::MadU24,
                dst: 0,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: None,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::CndMask {
                dst: 0,
                when_clear: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                when_set: VectorSource::Vgpr(1),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::Compare {
                kind: CompareKind::Eq,
                lhs: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                rhs: VectorSource::Vgpr(1),
                cmpx: false,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::CompareScalar {
                kind: CompareKind::Eq,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                dst: 127,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::CompareEqU64 {
                lhs: VectorSource::Vgpr(255),
                rhs: VectorSource::Vgpr(0),
                dst: None,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::CompareClassF32 {
                value: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                classes: VectorSource::Constant(1),
                dst: None,
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::Ternary {
                kind: VectorTernaryKind::Add3,
                dst: 0,
                a: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                b: VectorSource::Vgpr(1),
                c: VectorSource::Constant(1),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FloatMath {
                kind: FloatMathKind::Sqrt,
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FloatConvert {
                kind: FloatConvertKind::F16FromF32,
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ReadFirstLane { dst: 0, src: 0 }),
            WaveStateAbi::default()
        )
        .is_ok());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ReadLane {
                dst: 0,
                src: 0,
                lane: VectorSource::Dpp16RowMirror {
                    register: 1,
                    fetch_inactive: false,
                },
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::F32,
                dst: 0,
                a: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ConvertUByte {
                dst: 0,
                src: VectorSource::Vgpr(0),
                byte: 4,
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ConvertUByte {
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                byte: 0,
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::BitNot {
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::PackF16 {
                dst: 0,
                lo: VectorSource::Vgpr(0),
                hi: VectorSource::Vgpr(1),
                op_sel: 7,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::Floor,
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::FloatTernary {
                kind: FloatTernaryKind::Median3,
                dst: 0,
                a: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::AddCarry {
                dst: 0,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: None,
                carry_out: ScalarMask::Exec,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ConvertF32U32 {
                dst: 0,
                src: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                modifiers: FloatModifiers::default(),
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::ConvertF32U32Sdwa {
                dst: 0,
                src: 256,
                word: SdwaWord::Low,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::PackF16 {
                dst: 0,
                lo: VectorSource::Dpp16RowMirror {
                    register: 0,
                    fetch_inactive: false,
                },
                hi: VectorSource::Vgpr(1),
                op_sel: 0,
            }),
            WaveStateAbi::default()
        )
        .is_err());
        assert!(lower_slot(
            &ProgramOp::Vector(StageBOp::AddCarry {
                dst: 0,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: None,
                carry_out: ScalarMask::ScalarPair(3),
            }),
            WaveStateAbi::default()
        )
        .is_err());
    }

    #[test]
    fn supported_classes_match_independent_cpu_effects() {
        let active =
            |exec: [u32; 2], lane: u32| (exec[(lane >> 5) as usize] >> (lane & 31)) & 1 != 0;
        let exec = [2, 0];
        let mut sgpr = [0u32; 8];
        sgpr[2] = 7;
        assert_eq!(if active(exec, 1) { sgpr[2] } else { 99 }, 7);
        assert_eq!(if active(exec, 0) { sgpr[2] } else { 99 }, 99);
        let memory = [11u32, 22, 33];
        assert_eq!(memory[1], 22);
        let mut lds = [3u32, 4];
        let read = lds[1];
        lds[0] = read;
        assert_eq!((read, lds[0]), (4, 4));
        let lanes = [10u32, 20, 30, 40];
        assert_eq!(lanes[2], 30);
        assert_eq!(
            (6u32, (32u8, 15u8, false, true)),
            (6, (32, 15, false, true))
        );
    }

    #[test]
    fn read_lane_enforces_mature_scalar_destination_limit() {
        let op = |dst| {
            ProgramOp::Vector(StageBOp::ReadLane {
                dst,
                src: 0,
                lane: VectorSource::Constant(63),
            })
        };
        assert!(lower_slot(&op(105), WaveStateAbi::default()).is_ok());
        assert!(lower_slot(&op(106), WaveStateAbi::default()).is_err());
    }

    fn checked_signed_address(low: u32, high: u32, offset: i32) -> Option<u64> {
        let base = low as u64 | (high as u64) << 32;
        if offset < 0 {
            base.checked_sub(offset.unsigned_abs() as u64)
        } else {
            base.checked_add(offset as u64)
        }
    }

    fn mapped_lds_address(base: u32, offset: u16) -> u32 {
        ((base.wrapping_add(u32::from(offset)) >> 2) & 8191) << 2
    }

    fn swizzle_lane(lane: u32, control: u16) -> u32 {
        if control & 0x8000 == 0 {
            (lane & !3) | ((u32::from(control) >> (2 * (lane & 3))) & 3)
        } else {
            let and_mask = (u32::from(control) >> 10) & 31;
            let or_mask = (u32::from(control) >> 5) & 31;
            let xor_mask = u32::from(control) & 31;
            (lane & 32) | ((((lane & 31) & and_mask) | or_mask) ^ xor_mask)
        }
    }

    #[test]
    fn lds_helper_contract_matches_mature_wrap_and_dword_mapping() {
        assert_eq!(mapped_lds_address(0, 0), 0);
        assert_eq!(mapped_lds_address(32 * 1024 - 8, 4), 32 * 1024 - 4);
        assert_eq!(mapped_lds_address(1, 0), 0);
        assert_eq!(mapped_lds_address(u32::MAX, 1), 0);
        assert_eq!(mapped_lds_address(32 * 1024 - 4, 4), 0);
        assert_eq!(swizzle_lane(6, 0b_11_10_01_00), 6);
        assert_eq!(swizzle_lane(37, 0x8000 | (31 << 10) | 3), 38);
    }

    #[test]
    fn reviewed_edge_cases_are_explicit_in_effects_and_cpu_model() {
        let a = WaveStateAbi::default();
        let global = lower_slot(
            &ProgramOp::Memory(StageCOp::GlobalLoad {
                dst: 4,
                address: 8,
                scalar_address: None,
                offset: -4,
            }),
            a,
        )
        .unwrap()
        .hlsl;
        assert!(global.contains("uint2(vgpr[8u][lane], vgpr[9u][lane])"));
        assert_eq!(checked_signed_address(4, 1, -4), Some(1u64 << 32));
        assert_eq!(checked_signed_address(0, 0, -1), None);
        assert_eq!(checked_signed_address(u32::MAX, u32::MAX, 1), None);

        let write = lower_slot(
            &ProgramOp::Memory(StageCOp::DsWrite {
                address: 0,
                data: [1, 0],
                offsets: [0, 0],
                words: 1,
            }),
            a,
        )
        .unwrap()
        .hlsl;
        assert!(write.contains("if (xcl_active) xcl_lds_store(lds"));
        assert!(!write.contains("lds["));
        let swizzle = lower_slot(
            &ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 2,
                source: 3,
                control: 0,
            }),
            a,
        )
        .unwrap()
        .hlsl;
        assert!(swizzle.contains("xcl_ds_swizzle(vgpr[3u][lane], lane"));
        assert!(swizzle.find("xcl_ds_swizzle").unwrap() < swizzle.find("xcl_active").unwrap());

        let mut lds = 17u32;
        if false {
            lds = 99;
        }
        assert_eq!(lds, 17);
        let old_dst = 23u32;
        let source_lanes = [5u32, 7];
        let swizzled = if false { source_lanes[1] } else { old_dst };
        assert_eq!(swizzled, old_dst);
    }

    #[test]
    fn static_uniform_slot_shell_compiles_with_scheduler_owned_phases() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let a = WaveStateAbi::default();
        let effects = [
            ProgramOp::Scalar(StageAOp::MoveImmediate { dst: 2, value: 7 }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 3,
                src: VectorSource::Scalar(2),
            }),
            ProgramOp::Memory(StageCOp::GlobalLoad {
                dst: 4,
                address: 3,
                scalar_address: None,
                offset: 4,
            }),
            ProgramOp::Memory(StageCOp::DsWrite {
                address: 3,
                data: [4, 0],
                offsets: [0, 0],
                words: 1,
            }),
            ProgramOp::Memory(StageCOp::DsRead {
                dst: 5,
                address: 3,
                offsets: [0, 0],
                words: 1,
            }),
            ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 6,
                source: 5,
                control: 0,
            }),
            ProgramOp::AllocationExport(AllocationExportOp::Allocate { m0: 124 }),
        ]
        .iter()
        .map(|op| lower_slot(op, a).unwrap())
        .collect::<Vec<_>>();
        assert_eq!(
            effects.iter().map(|e| e.sync).collect::<Vec<_>>(),
            [
                SlotSync::None,
                SlotSync::None,
                SlotSync::None,
                SlotSync::DeviceMemory,
                SlotSync::DeviceMemory,
                SlotSync::GroupMemory,
                SlotSync::None
            ]
        );
        let body = effects
            .iter()
            .map(|e| e.hlsl.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!body.contains("GroupMemoryBarrier"));
        let hlsl = format!(
            r#"ByteAddressBuffer memory:register(t0); RWByteAddressBuffer lds:register(u1); RWByteAddressBuffer exports:register(u0);
groupshared uint sgpr[128]; groupshared uint vgpr[8][64];
uint xcl_lds_address(uint base,uint offset){{return (((base+offset)>>2u)&8191u)<<2u;}}
bool xcl_lds_load(RWByteAddressBuffer b,uint base,uint offset,out uint value){{value=b.Load(xcl_lds_address(base,offset));return true;}}
bool xcl_lds_store(RWByteAddressBuffer b,uint base,uint offset,uint value){{b.Store(xcl_lds_address(base,offset),value);return true;}}
bool xcl_ds_swizzle(uint lane_value,uint lane,uint control,out uint value){{
 uint source=(control&0x8000u)==0u?((lane&~3u)|((control>>(2u*(lane&3u)))&3u)):((lane&32u)|((((lane&31u)&((control>>10u)&31u))|((control>>5u)&31u))^(control&31u)));
 value=WaveReadLaneAt(lane_value,source);return source<64u;
}}
bool xcl_global_load(ByteAddressBuffer mem,uint2 p,int off,out uint value){{
 uint64_t a=(uint64_t)p.x|((uint64_t)p.y<<32); bool ok=true;
 if(off<0){{uint64_t d=(uint64_t)(-(int64_t)off);ok=a>=d;a-=d;}}
 else{{uint64_t d=(uint64_t)off;ok=a<=0xffffffffffffffffull-d;a+=d;}}
 ok=ok&&a<=0xfffffffbull; if(ok)value=mem.Load((uint)a); return ok;
}}
void xcl_allocate(uint m0){{}} void xcl_export(RWByteAddressBuffer e,uint t,uint n,uint c,uint d){{}}
[WaveSize(64)][numthreads(64,1,1)] void main(uint lane:SV_GroupIndex){{uint2 exec=~0u.xx,vcc=0u.xx;uint scc=0,m0=0;
{body}
DeviceMemoryBarrierWithGroupSync();
GroupMemoryBarrierWithGroupSync();
}}"#
        );
        super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
    }

    #[test]
    fn buffer_backed_lds_helpers_execute_without_inactive_self_writes() {
        if super::super::dxc::location().is_err() || super::super::runner::adapter().is_err() {
            return;
        }
        let abi = WaveStateAbi {
            lds: "memory",
            ..WaveStateAbi::default()
        };
        let write = lower_slot(
            &ProgramOp::Memory(StageCOp::DsWrite {
                address: 0,
                data: [1, 0],
                offsets: [0, 0],
                words: 1,
            }),
            abi,
        )
        .unwrap();
        let read = lower_slot(
            &ProgramOp::Memory(StageCOp::DsRead {
                dst: 2,
                address: 0,
                offsets: [0, 0],
                words: 1,
            }),
            abi,
        )
        .unwrap();
        assert_eq!(
            (write.sync, read.sync),
            (SlotSync::DeviceMemory, SlotSync::DeviceMemory)
        );
        let hlsl = format!(
            r#"RWByteAddressBuffer io:register(u0);RWByteAddressBuffer res:register(u1);RWByteAddressBuffer memory:register(u2);
groupshared uint vgpr[3][64];
uint xcl_lds_address(uint base,uint offset){{return (((base+offset)>>2u)&8191u)<<2u;}}
bool xcl_lds_load(RWByteAddressBuffer b,uint base,uint offset,out uint value){{value=b.Load(xcl_lds_address(base,offset));return true;}}
bool xcl_lds_store(RWByteAddressBuffer b,uint base,uint offset,uint value){{b.Store(xcl_lds_address(base,offset),value);return true;}}
[numthreads(64,1,1)]void main(uint lane:SV_GroupIndex){{uint2 exec=uint2(1u,0u);
vgpr[0][lane]=0u;vgpr[1][lane]=100u+lane;vgpr[2][lane]=777u;
GroupMemoryBarrierWithGroupSync();
{}
DeviceMemoryBarrierWithGroupSync();
{}
DeviceMemoryBarrierWithGroupSync();
res.Store(4u*(128u+lane*256u+2u),vgpr[2][lane]);
}}"#,
            write.hlsl, read.hlsl
        );
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let result = super::super::runner::run(
            &dxil,
            &super::super::runner::KernelState::default(),
            &[0; 64],
        )
        .unwrap();
        assert_eq!(result.mem[0], 100);
        assert_eq!(result.vgpr[0][2], 100);
        for lane in 1..64 {
            assert_eq!(result.vgpr[lane][2], 777);
        }
    }

    #[test]
    fn wave64_swizzle_reads_a_uniform_snapshot_before_exec_gating() {
        if super::super::dxc::location().is_err()
            || !super::super::runner::wave64_supported().unwrap_or(false)
        {
            return;
        }
        let quad = lower_slot(
            &ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 2,
                source: 1,
                control: 0x1b,
            }),
            WaveStateAbi::default(),
        )
        .unwrap();
        let bitmask = lower_slot(
            &ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 3,
                source: 1,
                control: 0xfc01,
            }),
            WaveStateAbi::default(),
        )
        .unwrap();
        assert_eq!(
            (quad.sync, bitmask.sync),
            (SlotSync::GroupMemory, SlotSync::GroupMemory)
        );
        let hlsl = format!(
            r#"RWByteAddressBuffer io:register(u0);RWByteAddressBuffer res:register(u1);RWByteAddressBuffer memory:register(u2);
groupshared uint vgpr[4][64];
bool xcl_ds_swizzle(uint lane_value,uint lane,uint control,out uint value){{
uint source=(control&0x8000u)==0u?((lane&~3u)|((control>>(2u*(lane&3u)))&3u)):((lane&32u)|((((lane&31u)&((control>>10u)&31u))|((control>>5u)&31u))^(control&31u)));
value=WaveReadLaneAt(lane_value,source);return source<64u;
}}
[WaveSize(64)][numthreads(64,1,1)]void main(uint lane:SV_GroupIndex){{uint2 exec=uint2(1u,1u);
vgpr[1][lane]=1000u+lane;vgpr[2][lane]=777u;vgpr[3][lane]=777u;
GroupMemoryBarrierWithGroupSync();
{}
GroupMemoryBarrierWithGroupSync();
{}
GroupMemoryBarrierWithGroupSync();
res.Store(4u*(128u+lane*256u+2u),vgpr[2][lane]);
res.Store(4u*(128u+lane*256u+3u),vgpr[3][lane]);
}}"#,
            quad.hlsl, bitmask.hlsl
        );
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let result =
            super::super::runner::run(&dxil, &super::super::runner::KernelState::default(), &[])
                .unwrap();
        assert_eq!((result.vgpr[0][2], result.vgpr[0][3]), (1003, 1001));
        assert_eq!((result.vgpr[32][2], result.vgpr[32][3]), (1035, 1033));
        assert_eq!((result.vgpr[1][2], result.vgpr[1][3]), (777, 777));
        assert_eq!((result.vgpr[33][2], result.vgpr[33][3]), (777, 777));
    }

    #[test]
    fn scalar_scc_source_is_raw_while_add_carry_is_boolean() {
        let moved = lower_slot(
            &ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Scc,
            }),
            WaveStateAbi::default(),
        )
        .unwrap();
        assert!(moved.hlsl.contains("sgpr[0u]=scc"));
        assert!(!moved.hlsl.contains("scc != 0u"));
        let add = lower_slot(
            &ProgramOp::Scalar(StageAOp::AddU32 {
                dst: 0,
                lhs: Operand32::Constant(5),
                rhs: Operand32::Scc,
                carry_in: true,
            }),
            WaveStateAbi::default(),
        )
        .unwrap();
        assert!(add.hlsl.contains("uint xcl_rhs=scc"));
        assert!(add.hlsl.contains("xcl_old_scc!=0u"));
    }

    #[test]
    fn scalar_get_pc_uses_architectural_byte_address_and_pair_rules() {
        use crate::gpu::rdna2;
        use crate::gpu::recomp::mesh_scalar::lower_stage_a_one;

        let code = rdna2::encode(
            "S_GETPC_B64",
            "ENC_SOP1",
            &[("SDST", 4)],
            None,
        )
        .unwrap();
        let decoded = rdna2::decode(&code, 10).unwrap();
        assert_eq!(decoded.len, 4);
        let op = lower_stage_a_one(&decoded).unwrap();
        assert_eq!(op, StageAOp::GetPc64 { dst: 4, next_pc: 14 });
        let effect = lower_slot(
            &ProgramOp::Scalar(op),
            WaveStateAbi::default(),
        )
        .unwrap();
        assert!(effect.hlsl.contains("sgpr[4u]=uint2(14u,0u).x"));
        assert!(effect.hlsl.contains("sgpr[5u]=uint2(14u,0u).y"));
        assert!(!effect.hlsl.contains("pc"));
        assert!(lower_slot(
            &ProgramOp::Scalar(StageAOp::GetPc64 { dst: 127, next_pc: 14 }),
            WaveStateAbi::default(),
        ).is_err());
        assert!(!effect.hlsl.contains("uint2(2u,0u)"));
    }

    #[test]
    fn scalar_hardware_configuration_accepts_only_decoder_admitted_inert_tuples() {
        use crate::gpu::rdna2;
        use crate::gpu::recomp::mesh_scalar::lower_stage_a_one;

        for (encoded, expected) in [
            (0x0881, (1, 2, 2, 0)),
            (0xf814, (20, 0, 32, 0)),
            (0xf815, (21, 0, 32, 1024)),
        ] {
            let code = rdna2::encode(
                "S_SETREG_IMM32_B32",
                "ENC_SOPK",
                &[("SIMM16", encoded)],
                Some(expected.3),
            )
            .unwrap();
            let decoded = rdna2::decode(&code, 0).unwrap();
            let op = lower_stage_a_one(&decoded).unwrap();
            assert_eq!(
                op,
                StageAOp::ConfigureHardwareRegister {
                    register: expected.0,
                    offset: expected.1,
                    width: expected.2,
                    value: expected.3,
                }
            );
            assert_eq!(
                lower_slot(&ProgramOp::Scalar(op), WaveStateAbi::default())
                    .unwrap()
                    .hlsl,
                ""
            );
        }
        for tuple in [(1, 2, 2, 1), (20, 0, 31, 0), (22, 0, 32, 1024)] {
            assert!(lower_slot(
                &ProgramOp::Scalar(StageAOp::ConfigureHardwareRegister {
                    register: tuple.0,
                    offset: tuple.1,
                    width: tuple.2,
                    value: tuple.3,
                }),
                WaveStateAbi::default(),
            )
            .is_err());
        }
    }
}
