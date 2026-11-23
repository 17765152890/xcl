//! Decode-once, fail-closed composition of verified mesh instruction families.
//!
//! This validator is intentionally disconnected from real PSO eligibility.

use std::collections::{BTreeMap, BTreeSet};

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_alloc_export::{self, AllocationExportOp};
use super::mesh_control;
use super::mesh_dispatch::{validate_step_cap, ControlOp, SCALAR_REGISTERS};
use super::mesh_memory::{
    self, ImageDescriptorContract, OffsetSource, ReadOnlyRegion, SamplerDescriptorContract,
    StageCOp,
};
use super::mesh_scalar::{self, Operand32, Operand64, StageAOp};
use super::mesh_vector::{self, ScalarMask, StageBOp, VectorSource};
use super::mesh_vgpr::VgprScratchLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProgramFamily {
    Control,
    Scalar,
    Vector,
    Memory,
    AllocationExport,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramOp {
    Control(ControlOp),
    Scalar(StageAOp),
    Vector(StageBOp),
    Memory(StageCOp),
    AllocationExport(AllocationExportOp),
    Unsupported(String),
}

impl ProgramOp {
    pub fn family(&self) -> ProgramFamily {
        match self {
            Self::Control(_) => ProgramFamily::Control,
            Self::Scalar(_) => ProgramFamily::Scalar,
            Self::Vector(_) => ProgramFamily::Vector,
            Self::Memory(_) => ProgramFamily::Memory,
            Self::AllocationExport(_) => ProgramFamily::AllocationExport,
            Self::Unsupported(_) => ProgramFamily::Unsupported,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramInstruction {
    pub pc: u64,
    pub byte_len: usize,
    pub words: Vec<u32>,
    pub literal: Option<u32>,
    pub name: &'static str,
    pub encoding: &'static str,
    pub op: ProgramOp,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoverageSummary {
    pub total: usize,
    pub families: BTreeMap<ProgramFamily, usize>,
    pub opcodes: BTreeMap<&'static str, usize>,
    pub unsupported: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshProgram {
    pub byte_len: usize,
    pub instructions: Vec<ProgramInstruction>,
    pub coverage: CoverageSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramRequirements {
    pub vgprs: u32,
    pub sgprs: u32,
    pub uses_read_only_memory: bool,
    pub uses_read_only_image: bool,
    pub uses_sampler: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramLimits<'a> {
    pub vgprs: u32,
    pub sgprs: u32,
    pub records: u32,
    pub step_cap: u32,
    pub read_only: Option<ReadOnlyRegion>,
    pub image_descriptors: &'a [ImageDescriptorContract],
    pub sampler_descriptors: &'a [SamplerDescriptorContract],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImageDescriptorUse {
    pub coordinate_count: u8,
    pub arrayed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DescriptorManifest {
    pub images: BTreeMap<u8, BTreeSet<ImageDescriptorUse>>,
    pub samplers: BTreeSet<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedProgram {
    pub requirements: ProgramRequirements,
    pub scratch: VgprScratchLayout,
    pub descriptors: DescriptorManifest,
}

fn classify(decoded: &Decoded, pcs: &BTreeMap<u64, u32>) -> ProgramOp {
    let attempts = [
        mesh_control::lower_decoded_one(decoded, pcs).map(ProgramOp::Control),
        mesh_scalar::lower_stage_a_one(decoded).map(ProgramOp::Scalar),
        mesh_vector::lower_stage_b_one(decoded).map(ProgramOp::Vector),
        mesh_memory::lower_stage_c_one(decoded).map(ProgramOp::Memory),
        mesh_alloc_export::lower_one(decoded).map(ProgramOp::AllocationExport),
    ];
    for attempt in &attempts {
        if let Ok(op) = attempt {
            return op.clone();
        }
    }
    ProgramOp::Unsupported(
        attempts
            .into_iter()
            .map(Result::unwrap_err)
            .collect::<Vec<_>>()
            .join("; "),
    )
}

pub fn decode(code: &[u8]) -> Result<MeshProgram, String> {
    if code.is_empty() {
        return Err("empty mesh program".into());
    }
    let mut decoded = Vec::new();
    let mut offset = 0usize;
    while offset < code.len() {
        let instruction = rdna2::decode(&code[offset..], offset as u64)
            .map_err(|error| format!("decode at {offset:#x}: {error:?}"))?;
        if instruction.len == 0 || instruction.len > code.len() - offset {
            return Err(format!("invalid instruction boundary at {offset:#x}"));
        }
        offset = offset
            .checked_add(instruction.len)
            .ok_or("mesh program size overflow")?;
        decoded.push(instruction);
    }
    let pcs: BTreeMap<u64, u32> = decoded
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.pc, index as u32))
        .collect();
    let mut coverage = CoverageSummary::default();
    let instructions = decoded
        .iter()
        .map(|instruction| {
            let op = classify(instruction, &pcs);
            coverage.total += 1;
            *coverage.families.entry(op.family()).or_default() += 1;
            *coverage.opcodes.entry(instruction.inst.name).or_default() += 1;
            if let ProgramOp::Unsupported(reason) = &op {
                *coverage.unsupported.entry(reason.clone()).or_default() += 1;
            }
            ProgramInstruction {
                pc: instruction.pc,
                byte_len: instruction.len,
                words: instruction.words.clone(),
                literal: instruction.literal,
                name: instruction.inst.name,
                encoding: instruction.enc.name,
                op,
            }
        })
        .collect();
    Ok(MeshProgram {
        byte_len: code.len(),
        instructions,
        coverage,
    })
}

fn bump(maximum: &mut u32, start: u32, words: u32) {
    *maximum = (*maximum).max(start.saturating_add(words));
}

fn ordinary_sgpr(register: u32) -> bool {
    register < SCALAR_REGISTERS && !matches!(register, 106 | 107 | 125 | 126 | 127)
}

fn bump_sgprs(maximum: &mut u32, start: u32, words: u32) {
    for register in start..start.saturating_add(words) {
        if ordinary_sgpr(register) {
            *maximum = (*maximum).max(register + 1);
        }
    }
}

fn scalar32(maximum: &mut u32, operand: Operand32) {
    if let Operand32::Register(register) = operand {
        if !matches!(register, 106 | 107 | 125 | 126 | 127) {
            bump_sgprs(maximum, register as u32, 1);
        }
    }
}

fn scalar64(maximum: &mut u32, operand: Operand64) {
    if let Operand64::Register(register) = operand {
        if register != 126 {
            bump_sgprs(maximum, register as u32, 2);
        }
    }
}

fn vector_source(requirements: &mut ProgramRequirements, source: VectorSource) {
    match source {
        VectorSource::Vgpr(register) => bump(&mut requirements.vgprs, register as u32, 1),
        VectorSource::Scalar(register) if !matches!(register, 106 | 107 | 125 | 126 | 127) => {
            bump_sgprs(&mut requirements.sgprs, register as u32, 1)
        }
        VectorSource::Scalar(_) => {}
        VectorSource::Constant(_) => {}
        VectorSource::Sdwa {
            register,
            scalar: true,
            ..
        } => bump_sgprs(&mut requirements.sgprs, register as u32, 1),
        VectorSource::Sdwa {
            register,
            scalar: false,
            ..
        }
        | VectorSource::Dpp16RowMirror { register, .. }
        | VectorSource::Dpp8 { register, .. } => bump(&mut requirements.vgprs, register as u32, 1),
        VectorSource::VccZero | VectorSource::ExecZero | VectorSource::Scc => {}
    }
}

fn offset_source(maximum: &mut u32, source: OffsetSource) {
    if let OffsetSource::Scalar(register) = source {
        if !matches!(register, 106 | 107 | 125 | 126 | 127) {
            bump_sgprs(maximum, register as u32, 1);
        }
    }
}

pub fn requirements(program: &MeshProgram) -> Result<ProgramRequirements, String> {
    let mut result = ProgramRequirements::default();
    for instruction in &program.instructions {
        match &instruction.op {
            ProgramOp::Unsupported(reason) => {
                return Err(format!(
                    "unsupported instruction at {:#x} ({}): {reason}",
                    instruction.pc, instruction.name
                ));
            }
            ProgramOp::Control(op) => match *op {
                ControlOp::MaskToScalar { dst, .. }
                | ControlOp::MaskBinaryScalar { dst, .. }
                | ControlOp::SaveExecAnd { dst, .. }
                | ControlOp::SaveExecAndMask { dst, .. }
                | ControlOp::WrExecAndNot { dst, .. }
                | ControlOp::WrExecAndNotMask { dst, .. } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 2)
                }
                ControlOp::ScalarToMask { src, .. } | ControlOp::RestoreExec { src } => {
                    bump_sgprs(&mut result.sgprs, src as u32, 2)
                }
                ControlOp::ScalarSet { register, .. }
                | ControlOp::ScalarAdd { register, .. }
                | ControlOp::ScalarAddCarry { register, .. }
                | ControlOp::ScalarCompareLess { register, .. }
                | ControlOp::BranchScalarLess { register, .. } => {
                    bump_sgprs(&mut result.sgprs, register as u32, 1)
                }
                _ => {}
            },
            ProgramOp::Scalar(op) => match *op {
                StageAOp::AddU32 { dst, lhs, rhs, .. } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, lhs);
                    scalar32(&mut result.sgprs, rhs);
                }
                StageAOp::MulImmediateI32 { dst, .. } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1)
                }
                StageAOp::GetPc64 { dst, .. } => bump_sgprs(&mut result.sgprs, dst as u32, 2),
                StageAOp::BitSet1 { dst, bit } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, bit);
                }
                StageAOp::Not32 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, source);
                }
                StageAOp::BitReverse32 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, source);
                }
                StageAOp::CountBits32 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, source);
                }
                StageAOp::CountBits64 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar64(&mut result.sgprs, source);
                }
                StageAOp::Binary { dst, lhs, rhs, .. } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, lhs);
                    scalar32(&mut result.sgprs, rhs);
                }
                StageAOp::Binary64 { dst, lhs, rhs, .. } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    scalar64(&mut result.sgprs, lhs);
                    scalar64(&mut result.sgprs, rhs);
                }
                StageAOp::CompareEq { lhs, rhs }
                | StageAOp::CompareGtU32 { lhs, rhs }
                | StageAOp::CompareLtU32 { lhs, rhs }
                | StageAOp::CompareLeU32 { lhs, rhs }
                | StageAOp::CompareGeU32 { lhs, rhs }
                | StageAOp::CompareNeU32 { lhs, rhs }
                | StageAOp::CompareLtI32 { lhs, rhs }
                | StageAOp::CompareGtI32 { lhs, rhs }
                | StageAOp::CompareGeI32 { lhs, rhs } => {
                    scalar32(&mut result.sgprs, lhs);
                    scalar32(&mut result.sgprs, rhs);
                }
                StageAOp::Select32 {
                    dst,
                    when_set,
                    when_clear,
                } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, when_set);
                    scalar32(&mut result.sgprs, when_clear);
                }
                StageAOp::MoveImmediate { dst, .. } => bump_sgprs(&mut result.sgprs, dst as u32, 1),
                StageAOp::Move32 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    scalar32(&mut result.sgprs, source);
                }
                StageAOp::Move64 { dst, source }
                | StageAOp::SaveExecAnd64 { dst, source }
                | StageAOp::SaveExecOr64 { dst, source }
                | StageAOp::WrExecAndNot64 { dst, source } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    scalar64(&mut result.sgprs, source);
                }
                StageAOp::Select64 {
                    dst,
                    when_set,
                    when_clear,
                } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    scalar64(&mut result.sgprs, when_set);
                    scalar64(&mut result.sgprs, when_clear);
                }
                _ => {}
            },
            ProgramOp::Vector(op) => match *op {
                StageBOp::Move { dst, src } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, src);
                }
                StageBOp::RelativeMove { dst, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    // M0 is runtime data, so every hardware VGPR from the encoded base
                    // through v255 can be selected. Require the complete bounded file.
                    bump(&mut result.vgprs, 255, 1);
                }
                StageBOp::ReadFirstLane { dst, src } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    bump(&mut result.vgprs, src as u32, 1);
                }
                StageBOp::ReadLane { dst, src, lane } => {
                    bump_sgprs(&mut result.sgprs, dst as u32, 1);
                    bump(&mut result.vgprs, src as u32, 1);
                    vector_source(&mut result, lane);
                }
                StageBOp::WriteLane { dst, src, lane } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, src);
                    vector_source(&mut result, lane);
                }
                StageBOp::PermLaneX16 {
                    dst,
                    src,
                    selectors_low,
                    selectors_high,
                    ..
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    bump(&mut result.vgprs, src as u32, 1);
                    vector_source(&mut result, selectors_low);
                    vector_source(&mut result, selectors_high);
                }
                StageBOp::Mbcnt {
                    dst,
                    source,
                    accumulator,
                    ..
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, source);
                    vector_source(&mut result, accumulator);
                }
                StageBOp::Binary { dst, lhs, rhs, .. }
                | StageBOp::FloatBinary { dst, lhs, rhs, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                }
                StageBOp::AddCarry {
                    dst,
                    lhs,
                    rhs,
                    carry_in,
                    carry_out,
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                    for mask in carry_in.into_iter().chain([carry_out]) {
                        if let ScalarMask::ScalarPair(register) = mask {
                            bump_sgprs(&mut result.sgprs, register as u32, 2);
                        }
                    }
                }
                StageBOp::ReverseSubCarry {
                    dst,
                    lhs,
                    rhs,
                    carry_in,
                    carry_out,
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                    for mask in [carry_in, carry_out] {
                        if let ScalarMask::ScalarPair(register) = mask {
                            bump_sgprs(&mut result.sgprs, register as u32, 2);
                        }
                    }
                }
                StageBOp::Ternary { dst, a, b, c, .. }
                | StageBOp::FloatTernary { dst, a, b, c, .. }
                | StageBOp::StageDInteger {
                    dst,
                    a,
                    b,
                    c: Some(c),
                    ..
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, a);
                    vector_source(&mut result, b);
                    vector_source(&mut result, c);
                }
                StageBOp::StageDInteger {
                    dst, a, b, c: None, ..
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, a);
                    vector_source(&mut result, b);
                }
                StageBOp::BitReverse { dst, src }
                | StageBOp::BitNot { dst, src }
                | StageBOp::BitScan { dst, src, .. }
                | StageBOp::ConvertF32U32 { dst, src, .. }
                | StageBOp::ConvertF32I32 { dst, src }
                | StageBOp::FloatUnary { dst, src, .. }
                | StageBOp::FloatMath { dst, src, .. }
                | StageBOp::ConvertUByte { dst, src, .. }
                | StageBOp::FloatConvert { dst, src, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, src);
                }
                StageBOp::ConvertF32U32Sdwa { dst, src, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    bump(&mut result.vgprs, src as u32, 1);
                }
                StageBOp::Compare { lhs, rhs, .. } => {
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                }
                StageBOp::CompareScalar { lhs, rhs, dst, .. } => {
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                    bump_sgprs(&mut result.sgprs, dst as u32, 2);
                }
                StageBOp::CompareEqU64 { lhs, rhs, dst } => {
                    for source in [lhs, rhs] {
                        match source {
                            VectorSource::Vgpr(register) => {
                                bump(&mut result.vgprs, register as u32, 2)
                            }
                            VectorSource::Scalar(register) => {
                                bump_sgprs(&mut result.sgprs, register as u32, 2)
                            }
                            _ => unreachable!("validated paired source"),
                        }
                    }
                    if let Some(dst) = dst {
                        bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    }
                }
                StageBOp::CompareClassF32 {
                    value,
                    classes,
                    dst,
                    ..
                } => {
                    vector_source(&mut result, value);
                    vector_source(&mut result, classes);
                    if let Some(dst) = dst {
                        bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    }
                }
                StageBOp::CompareLessFloat { lhs, rhs, dst, .. } => {
                    vector_source(&mut result, lhs);
                    vector_source(&mut result, rhs);
                    if let Some(dst) = dst {
                        bump_sgprs(&mut result.sgprs, dst as u32, 2);
                    }
                }
                StageBOp::CndMask {
                    dst,
                    when_clear,
                    when_set,
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, when_clear);
                    vector_source(&mut result, when_set);
                }
                StageBOp::CndMaskScalar {
                    dst,
                    when_clear,
                    when_set,
                    predicate,
                } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, when_clear);
                    vector_source(&mut result, when_set);
                    if let ScalarMask::ScalarPair(register) = predicate {
                        bump_sgprs(&mut result.sgprs, register as u32, 2);
                    }
                }
                StageBOp::FloatFma { dst, a, b, c, .. }
                | StageBOp::FloatMad { dst, a, b, c, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, a);
                    vector_source(&mut result, b);
                    if let Some(c) = c {
                        vector_source(&mut result, c);
                    }
                }
                StageBOp::PackF16 { dst, lo, hi, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, lo);
                    vector_source(&mut result, hi);
                }
                StageBOp::FmaMix { dst, a, b, c, .. } => {
                    bump(&mut result.vgprs, dst as u32, 1);
                    vector_source(&mut result, a);
                    vector_source(&mut result, b);
                    vector_source(&mut result, c);
                }
                StageBOp::NoEffect => {}
            },
            ProgramOp::Memory(op) => {
                if !matches!(
                    op,
                    StageCOp::DsSwizzle { .. } | StageCOp::DsRead { .. } | StageCOp::DsWrite { .. }
                ) {
                    result.uses_read_only_memory = true;
                }
                match *op {
                    StageCOp::DsRead {
                        dst,
                        address,
                        words,
                        ..
                    } => {
                        bump(&mut result.vgprs, dst as u32, words as u32);
                        bump(&mut result.vgprs, address as u32, 1);
                    }
                    StageCOp::DsWrite {
                        address,
                        data,
                        words,
                        ..
                    } => {
                        bump(&mut result.vgprs, address as u32, 1);
                        for register in &data[..words as usize] {
                            bump(&mut result.vgprs, *register as u32, 1);
                        }
                    }
                    StageCOp::DsSwizzle { dst, source, .. } => {
                        bump(&mut result.vgprs, dst as u32, 1);
                        bump(&mut result.vgprs, source as u32, 1);
                    }
                    StageCOp::ScalarLoad {
                        dst,
                        base,
                        soffset,
                        words,
                        ..
                    } => {
                        bump_sgprs(&mut result.sgprs, dst as u32, words as u32);
                        bump_sgprs(&mut result.sgprs, base as u32, 2);
                        offset_source(&mut result.sgprs, soffset);
                    }
                    StageCOp::ScalarBufferLoad {
                        dst,
                        descriptor,
                        soffset,
                        words,
                        ..
                    } => {
                        bump_sgprs(&mut result.sgprs, dst as u32, words as u32);
                        bump_sgprs(&mut result.sgprs, descriptor as u32, 4);
                        offset_source(&mut result.sgprs, soffset);
                    }
                    StageCOp::VectorBufferLoad {
                        dst,
                        address,
                        descriptor,
                        soffset,
                        words,
                        offen,
                        idxen,
                        ..
                    } => {
                        bump(&mut result.vgprs, dst as u32, words as u32);
                        let address_words = u32::from(offen) + u32::from(idxen);
                        if address_words != 0 {
                            bump(&mut result.vgprs, address as u32, address_words);
                        }
                        bump_sgprs(&mut result.sgprs, descriptor as u32, 4);
                        offset_source(&mut result.sgprs, soffset);
                    }
                    StageCOp::GlobalLoad {
                        dst,
                        address,
                        scalar_address,
                        ..
                    } => {
                        bump(&mut result.vgprs, dst as u32, 1);
                        bump(&mut result.vgprs, address as u32, 2);
                        if let Some(register) = scalar_address {
                            bump_sgprs(&mut result.sgprs, register as u32, 2);
                        }
                    }
                    StageCOp::ImageLoad {
                        dst,
                        coordinates,
                        coordinate_count,
                        descriptor,
                        dmask,
                        d16,
                    } => {
                        result.uses_read_only_image = true;
                        let outputs = if d16 {
                            dmask.count_ones().div_ceil(2)
                        } else {
                            dmask.count_ones()
                        };
                        bump(&mut result.vgprs, dst as u32, outputs);
                        for coordinate in &coordinates[..coordinate_count as usize] {
                            bump(&mut result.vgprs, *coordinate as u32, 1);
                        }
                        bump_sgprs(&mut result.sgprs, descriptor as u32, 8);
                    }
                    StageCOp::ImageSampleLz {
                        dst,
                        coordinates,
                        coordinate_count,
                        descriptor,
                        sampler,
                        dmask,
                        d16,
                        lod,
                        ..
                    } => {
                        result.uses_read_only_image = true;
                        result.uses_sampler = true;
                        let outputs = if d16 {
                            dmask.count_ones().div_ceil(2)
                        } else {
                            dmask.count_ones()
                        };
                        bump(&mut result.vgprs, dst as u32, outputs);
                        for coordinate in &coordinates[..coordinate_count as usize] {
                            bump(&mut result.vgprs, *coordinate as u32, 1);
                        }
                        if let Some(lod) = lod {
                            bump(&mut result.vgprs, lod as u32, 1);
                        }
                        bump_sgprs(&mut result.sgprs, descriptor as u32, 8);
                        bump_sgprs(&mut result.sgprs, sampler as u32, 4);
                    }
                    StageCOp::ImageGather4Lz {
                        dst,
                        coordinates,
                        descriptor,
                        sampler,
                        d16,
                        ..
                    } => {
                        result.uses_read_only_image = true;
                        result.uses_sampler = true;
                        bump(&mut result.vgprs, dst as u32, if d16 { 2 } else { 4 });
                        for coordinate in coordinates {
                            bump(&mut result.vgprs, coordinate as u32, 1);
                        }
                        bump_sgprs(&mut result.sgprs, descriptor as u32, 8);
                        bump_sgprs(&mut result.sgprs, sampler as u32, 4);
                    }
                }
            }
            ProgramOp::AllocationExport(op) => match *op {
                AllocationExportOp::Allocate { m0 } => {
                    bump_sgprs(&mut result.sgprs, m0 as u32, 1);
                }
                AllocationExportOp::Export(export) => {
                    result.vgprs = result.vgprs.max(mesh_alloc_export::required_vgprs(export));
                }
            },
        }
    }
    Ok(result)
}

pub fn descriptor_manifest(program: &MeshProgram) -> DescriptorManifest {
    let mut manifest = DescriptorManifest::default();
    for instruction in &program.instructions {
        let ProgramOp::Memory(op) = instruction.op else {
            continue;
        };
        match op {
            StageCOp::ImageLoad {
                descriptor,
                coordinate_count,
                ..
            } => {
                manifest
                    .images
                    .entry(descriptor)
                    .or_default()
                    .insert(ImageDescriptorUse {
                        coordinate_count,
                        arrayed: false,
                    });
            }
            StageCOp::ImageSampleLz {
                descriptor,
                sampler,
                coordinate_count,
                arrayed,
                ..
            } => {
                manifest
                    .images
                    .entry(descriptor)
                    .or_default()
                    .insert(ImageDescriptorUse {
                        coordinate_count,
                        arrayed,
                    });
                manifest.samplers.insert(sampler);
            }
            StageCOp::ImageGather4Lz {
                descriptor,
                sampler,
                ..
            } => {
                manifest
                    .images
                    .entry(descriptor)
                    .or_default()
                    .insert(ImageDescriptorUse {
                        coordinate_count: 2,
                        arrayed: false,
                    });
                manifest.samplers.insert(sampler);
            }
            _ => {}
        }
    }
    manifest
}

fn validate_descriptor_bindings(
    manifest: &DescriptorManifest,
    images: &[ImageDescriptorContract],
    samplers: &[SamplerDescriptorContract],
) -> Result<(), String> {
    let mut image_bindings = BTreeMap::new();
    for contract in images {
        if image_bindings
            .insert(contract.register, contract.words)
            .is_some()
        {
            return Err("duplicate image descriptor contract register".into());
        }
        if !manifest.images.contains_key(&contract.register) {
            return Err("unused image descriptor contract register".into());
        }
    }
    for (&register, uses) in &manifest.images {
        let words = image_bindings
            .get(&register)
            .ok_or("program requires an explicit image descriptor contract")?;
        for usage in uses {
            mesh_memory::validated_image_descriptor_for(
                *words,
                usage.coordinate_count,
                usage.arrayed,
            )
            .map_err(str::to_owned)?;
        }
    }

    let mut sampler_bindings = BTreeMap::new();
    for contract in samplers {
        if sampler_bindings
            .insert(contract.register, contract.words)
            .is_some()
        {
            return Err("duplicate sampler descriptor contract register".into());
        }
        if !manifest.samplers.contains(&contract.register) {
            return Err("unused sampler descriptor contract register".into());
        }
        mesh_memory::validated_sampler_descriptor(contract.words).map_err(str::to_owned)?;
    }
    for register in &manifest.samplers {
        if !sampler_bindings.contains_key(register) {
            return Err("program requires an explicit sampler descriptor contract".into());
        }
    }
    Ok(())
}

pub fn validate_structure(
    program: &MeshProgram,
    limits: ProgramLimits<'_>,
) -> Result<ValidatedProgram, String> {
    validate_step_cap(limits.step_cap).map_err(str::to_owned)?;
    if limits.sgprs > SCALAR_REGISTERS {
        return Err("declared SGPR limit exceeds dispatcher state".into());
    }
    let requirements = requirements(program)?;
    if !program.instructions.iter().any(|instruction| {
        matches!(
            instruction.op,
            ProgramOp::Control(ControlOp::Complete) | ProgramOp::Scalar(StageAOp::Complete)
        )
    }) {
        return Err("program has no explicit completion".into());
    }
    if requirements.vgprs > limits.vgprs {
        return Err(format!(
            "program requires {} VGPRs but {} were declared",
            requirements.vgprs, limits.vgprs
        ));
    }
    if requirements.sgprs > limits.sgprs {
        return Err(format!(
            "program requires {} SGPRs but {} were declared",
            requirements.sgprs, limits.sgprs
        ));
    }
    if requirements.uses_read_only_memory && limits.read_only.is_none() {
        return Err("program requires a read-only memory region".into());
    }
    if let Some(region) = limits.read_only {
        if region.guest_base & 3 != 0
            || region.memory_offset & 3 != 0
            || region.byte_len == 0
            || region.byte_len & 3 != 0
        {
            return Err("read-only memory region is not dword aligned".into());
        }
        region
            .memory_offset
            .checked_add(region.byte_len)
            .ok_or("read-only memory region overflows host addressing")?;
        region
            .guest_base
            .checked_add(region.byte_len as u64)
            .ok_or("read-only memory region overflows guest addressing")?;
    }
    let scratch = VgprScratchLayout::new(limits.vgprs, limits.records).map_err(str::to_owned)?;
    Ok(ValidatedProgram {
        requirements,
        scratch,
        descriptors: descriptor_manifest(program),
    })
}

pub fn validate(
    program: &MeshProgram,
    limits: ProgramLimits<'_>,
) -> Result<ValidatedProgram, String> {
    let validated = validate_structure(program, limits)?;
    validate_descriptor_bindings(
        &validated.descriptors,
        limits.image_descriptors,
        limits.sampler_descriptors,
    )?;
    Ok(validated)
}

/// A validation-only compute shader for one decode-once mixed-family program.
///
/// This deliberately does not participate in graphics PSO selection.  The first
/// tranche admits only programs which need no workgroup-wide synchronization;
/// operations which would require a uniformly reached barrier fail closed until
/// the epoch scheduler can prove that property.
#[derive(Clone, Debug)]
pub struct ProgramValidationShader {
    pub hlsl: String,
    pub validated: ValidatedProgram,
}

/// Emit the existing shared-register Kernel ABI from a validated `MeshProgram`.
///
/// Instruction bytes are reconstructed from the decode-once representation, so
/// branch PCs and original instruction order remain identical.  Kernel mode's
/// guarded block dispatcher supplies the bounded control-flow budget and shares
/// SGPR, VGPR, EXEC, VCC, SCC, M0, memory, and export state across families.
pub fn validation_shader(
    program: &MeshProgram,
    limits: ProgramLimits<'_>,
) -> Result<ProgramValidationShader, String> {
    let validated = validate(program, limits)?;
    let mut code = Vec::with_capacity(program.byte_len);
    let mut expected_pc = 0u64;
    for instruction in &program.instructions {
        if instruction.pc != expected_pc
            || instruction.byte_len != instruction.words.len() * 4
        {
            return Err("MeshProgram instruction boundaries are not contiguous".to_owned());
        }
        for word in &instruction.words {
            code.extend_from_slice(&word.to_le_bytes());
        }
        expected_pc = expected_pc
            .checked_add(instruction.byte_len as u64)
            .ok_or("MeshProgram byte length overflows")?;
    }
    if code.len() != program.byte_len {
        return Err("MeshProgram byte length does not match its instructions".to_owned());
    }

    // MeshProgram is public and intentionally easy to inspect, but emission must
    // never trust callers to keep its decoded ops and aggregate coverage in sync
    // with the instruction words which are actually translated.
    let canonical = decode(&code)?;
    if canonical != *program {
        return Err("MeshProgram does not match a canonical decode of its words".to_owned());
    }
    for instruction in &canonical.instructions {
        match instruction.op {
            ProgramOp::Scalar(StageAOp::Barrier)
            | ProgramOp::Memory(
                StageCOp::DsSwizzle { .. }
                | StageCOp::DsRead { .. }
                | StageCOp::DsWrite { .. },
            ) => {
                return Err(format!(
                    "{} requires a uniformly reached validation barrier",
                    instruction.name
                ));
            }
            _ => {}
        }
    }

    let translation = super::translate(
        &code,
        &super::Options {
            mode: super::Mode::Kernel,
            // Exercise two logical wave32 halves on one wave64-capable kernel.
            wave64: false,
            hw_wave: 64,
            structured: false,
            ..Default::default()
        },
    )?;
    if !translation.complete() {
        return Err(format!(
            "mixed validation translation is incomplete: {:?}",
            translation.unsupported
        ));
    }
    if translation.instructions != program.instructions.len() {
        return Err("mixed validation translation changed the instruction count".to_owned());
    }
    Ok(ProgramValidationShader {
        hlsl: translation.hlsl,
        validated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    fn append(
        code: &mut Vec<u8>,
        name: &str,
        encoding: &str,
        fields: &[(&str, u64)],
        literal: Option<u32>,
    ) {
        code.extend(encode(name, encoding, fields, literal).unwrap());
    }

    fn mixed() -> Vec<u8> {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_MOV_B32",
            "ENC_SOP1",
            &[("SDST", 2), ("SSRC0", 129)],
            None,
        );
        append(
            &mut code,
            "S_ADD_I32",
            "ENC_SOP2",
            &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 129)],
            None,
        );
        append(&mut code, "S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)], None);
        append(&mut code, "S_BARRIER", "ENC_SOPP", &[], None);
        append(
            &mut code,
            "V_AND_B32",
            "ENC_VOP2",
            &[("VDST", 5), ("SRC0", 256), ("VSRC1", 1)],
            None,
        );
        append(
            &mut code,
            "BUFFER_LOAD_DWORD",
            "ENC_MUBUF",
            &[("VDATA", 7), ("VADDR", 5), ("SRSRC", 2), ("SOFFSET", 128)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        code
    }

    fn limits() -> ProgramLimits<'static> {
        ProgramLimits {
            vgprs: 8,
            sgprs: 16,
            records: 1,
            step_cap: 64,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0x1000,
                memory_offset: 0,
                byte_len: 256,
            }),
            image_descriptors: &[],
            sampler_descriptors: &[],
        }
    }

    #[test]
    fn ds_swizzle_accounts_only_vector_registers() {
        let mut code = vec![0xb1, 0x00, 0xd4, 0xd8, 0x02, 0x00, 0x00, 0x03];
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 3,
                source: 2,
                control: 0x00b1,
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (4, 0));
        assert!(!required.uses_read_only_memory);
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 4,
                sgprs: 0,
                records: 1,
                step_cap: 4,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            }
        )
        .is_ok());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 3,
                sgprs: 0,
                records: 1,
                step_cap: 4,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            }
        )
        .is_err());
    }

    #[test]
    fn ds_dword_io_accounts_register_spans_without_external_memory() {
        let mut code = Vec::new();
        append(
            &mut code,
            "DS_WRITE2ST64_B32",
            "ENC_DS",
            &[
                ("ADDR", 4),
                ("DATA0", 9),
                ("DATA1", 10),
                ("OFFSET0", 1),
                ("OFFSET1", 2),
            ],
            None,
        );
        append(
            &mut code,
            "DS_READ2_B32",
            "ENC_DS",
            &[("ADDR", 4), ("VDST", 11), ("OFFSET1", 1)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (13, 0));
        assert!(!required.uses_read_only_memory);
        let exact = ProgramLimits {
            vgprs: 13,
            sgprs: 0,
            records: 1,
            step_cap: 4,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, exact).is_ok());
        assert!(validate(&program, ProgramLimits { vgprs: 12, ..exact }).is_err());
    }

    fn vector_load(offen: bool, idxen: bool, address: u64) -> MeshProgram {
        let mut code = Vec::new();
        append(
            &mut code,
            "BUFFER_LOAD_DWORD",
            "ENC_MUBUF",
            &[
                ("VDATA", 0),
                ("VADDR", address),
                ("SRSRC", 0),
                ("SOFFSET", 128),
                ("OFFEN", u64::from(offen)),
                ("IDXEN", u64::from(idxen)),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        decode(&code).unwrap()
    }

    #[test]
    fn mixed_program_decodes_once_preserves_boundaries_and_validates_atomically() {
        let code = mixed();
        let program = decode(&code).unwrap();
        assert_eq!(program.byte_len, code.len());
        assert_eq!(program.instructions.len(), 7);
        assert_eq!(program.coverage.total, 7);
        assert_eq!(
            program
                .coverage
                .families
                .get(&ProgramFamily::Unsupported)
                .copied()
                .unwrap_or(0),
            0
        );
        assert!(matches!(
            program.instructions[2].op,
            ProgramOp::Control(ControlOp::Jump { target: 3 })
        ));
        assert!(matches!(
            program.instructions[3].op,
            ProgramOp::Scalar(StageAOp::Barrier)
        ));
        assert!(matches!(program.instructions[4].op, ProgramOp::Vector(_)));
        assert!(matches!(program.instructions[5].op, ProgramOp::Memory(_)));
        assert!(program
            .instructions
            .iter()
            .all(|instruction| instruction.byte_len == instruction.words.len() * 4));
        let validated = validate(&program, limits()).unwrap();
        assert_eq!(validated.requirements.vgprs, 8);
        assert_eq!(validated.requirements.sgprs, 12);
        assert!(validated.requirements.uses_read_only_memory);
    }

    #[test]
    fn material_scalar_forms_classify_and_account_for_register_pairs() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_MOV_B64",
            "ENC_SOP1",
            &[("SDST", 0), ("SSRC0", 193)],
            None,
        );
        append(
            &mut code,
            "S_MOV_B32",
            "ENC_SOP1",
            &[("SDST", 2), ("SSRC0", 0)],
            None,
        );
        append(
            &mut code,
            "S_CMP_LT_U32",
            "ENC_SOPC",
            &[("SSRC0", 2), ("SSRC1", 3)],
            None,
        );
        append(
            &mut code,
            "S_CMP_LE_U32",
            "ENC_SOPC",
            &[("SSRC0", 2), ("SSRC1", 255)],
            Some(0xffff_ffff),
        );
        for (name, dst) in [
            ("S_AND_B64", 4),
            ("S_ANDN2_B64", 6),
            ("S_OR_B64", 8),
            ("S_XOR_B64", 10),
        ] {
            append(
                &mut code,
                name,
                "ENC_SOP2",
                &[("SDST", dst), ("SSRC0", 0), ("SSRC1", 4)],
                None,
            );
        }
        append(
            &mut code,
            "S_AND_SAVEEXEC_B64",
            "ENC_SOP1",
            &[("SDST", 12), ("SSRC0", 10)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(program.instructions[..9]
            .iter()
            .all(|instruction| matches!(instruction.op, ProgramOp::Scalar(_))));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 14));

        let mut unsupported = Vec::new();
        append(
            &mut unsupported,
            "S_MOV_B64",
            "ENC_SOP1",
            &[("SDST", 124), ("SSRC0", 0)],
            None,
        );
        append(&mut unsupported, "S_ENDPGM", "ENC_SOPP", &[], None);
        let unsupported = decode(&unsupported).unwrap();
        assert!(matches!(
            unsupported.instructions[0].op,
            ProgramOp::Unsupported(_)
        ));
        assert!(requirements(&unsupported).is_err());
    }

    #[test]
    fn exec_mask_writes_account_for_source_and_destination_pairs() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_ANDN2_WREXEC_B64",
            "ENC_SOP1",
            &[("SDST", 18), ("SSRC0", 20)],
            None,
        );
        append(
            &mut code,
            "S_OR_SAVEEXEC_B64",
            "ENC_SOP1",
            &[("SDST", 16), ("SSRC0", 20)],
            None,
        );
        append(
            &mut code,
            "S_AND_SAVEEXEC_B64",
            "ENC_SOP1",
            &[("SDST", 106), ("SSRC0", 106)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 22));
        validate(
            &program,
            ProgramLimits {
                vgprs: 0,
                sgprs: 22,
                records: 1,
                step_cap: 8,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .unwrap();
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 0,
                sgprs: 21,
                records: 1,
                step_cap: 8,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .is_err());
    }

    #[test]
    fn float_class_compare_accounts_value_mask_and_destination_pair() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CMP_CLASS_F32",
            "ENC_VOP3",
            &[("VDST", 20), ("SRC0", 267), ("SRC1", 18)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Vector(StageBOp::CompareClassF32 { dst: Some(20), .. })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (12, 22));
        let limits = ProgramLimits {
            vgprs: 12,
            sgprs: 22,
            records: 1,
            step_cap: 4,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        validate(&program, limits).unwrap();
        assert!(validate(
            &program,
            ProgramLimits {
                sgprs: 21,
                ..limits
            }
        )
        .is_err());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 11,
                ..limits
            }
        )
        .is_err());
    }

    #[test]
    fn scalar_register_invariants_classify_and_account_exactly() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_BITSET1_B32",
            "ENC_SOP1",
            &[("SDST", 5), ("SSRC0", 4)],
            None,
        );
        append(
            &mut code,
            "S_CMP_LG_U32",
            "ENC_SOPC",
            &[("SSRC0", 5), ("SSRC1", 128)],
            None,
        );
        for (encoded, value) in [(0x0881, 0), (0xf814, 0), (0xf815, 1024)] {
            append(
                &mut code,
                "S_SETREG_IMM32_B32",
                "ENC_SOPK",
                &[("SIMM16", encoded)],
                Some(value),
            );
        }
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert_eq!(requirements(&program).unwrap().sgprs, 6);

        let invalid = encode(
            "S_SETREG_IMM32_B32",
            "ENC_SOPK",
            &[("SIMM16", 0xf816)],
            Some(0),
        )
        .unwrap();
        let invalid = decode(&invalid).unwrap();
        assert!(matches!(
            invalid.instructions[0].op,
            ProgramOp::Unsupported(_)
        ));
    }

    #[test]
    fn scalar_mask_and_sdwa_vector_forms_compose_and_account_resources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CMP_LE_U32",
            "ENC_VOP3",
            &[("VDST", 0), ("SRC0", 256), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CNDMASK_B32",
            "ENC_VOP3",
            &[
                ("VDST", 5),
                ("SRC0", 256),
                ("SRC1", 257),
                ("SRC2", 0),
                ("NEG", 2),
            ],
            None,
        );
        append(
            &mut code,
            "V_CVT_F32_U32",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 9),
                ("SRC0", 249),
                ("VSRC0", 8),
                ("SRC0_SEL", 5),
                ("DST_SEL", 6),
            ],
            None,
        );
        append(
            &mut code,
            "V_CMPX_LG_F32",
            "ENC_VOPC",
            &[("SRC0", 256), ("VSRC1", 1)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Vector(StageBOp::CompareScalar { dst: 0, .. })
        ));
        assert!(matches!(
            program.instructions[1].op,
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                predicate: ScalarMask::ScalarPair(0),
                ..
            })
        ));
        assert!(matches!(
            program.instructions[2].op,
            ProgramOp::Vector(StageBOp::ConvertF32U32Sdwa { src: 8, .. })
        ));
        assert!(matches!(
            program.instructions[3].op,
            ProgramOp::Vector(StageBOp::Compare {
                kind: mesh_vector::CompareKind::OrderedNotEqualFloat,
                cmpx: true,
                ..
            })
        ));
        let requirements = requirements(&program).unwrap();
        assert_eq!(requirements.vgprs, 10);
        assert_eq!(requirements.sgprs, 2);
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 10,
                sgprs: 2,
                records: 1,
                step_cap: 16,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .is_ok());
    }

    #[test]
    fn float_conversion_subtraction_and_truncation_account_exactly() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CVT_F32_I32",
            "ENC_VOP1",
            &[("VDST", 4), ("SRC0", 256)],
            None,
        );
        append(
            &mut code,
            "V_SUB_F32",
            "ENC_VOP2",
            &[("VDST", 5), ("SRC0", 260), ("VSRC1", 1)],
            None,
        );
        append(
            &mut code,
            "V_SUBREV_F32",
            "ENC_VOP3",
            &[("VDST", 6), ("SRC0", 261), ("SRC1", 260)],
            None,
        );
        append(
            &mut code,
            "V_TRUNC_F32",
            "ENC_VOP1",
            &[("VDST", 7), ("SRC0", 262)],
            None,
        );
        append(
            &mut code,
            "V_CVT_I32_F32",
            "ENC_VOP1",
            &[("VDST", 8), ("SRC0", 263)],
            None,
        );
        append(
            &mut code,
            "V_FLOOR_F32",
            "ENC_VOP1",
            &[("VDST", 9), ("SRC0", 264)],
            None,
        );
        append(
            &mut code,
            "V_CEIL_F32",
            "ENC_VOP1",
            &[("VDST", 10), ("SRC0", 265)],
            None,
        );
        append(
            &mut code,
            "V_RNDNE_F32",
            "ENC_VOP1",
            &[("VDST", 11), ("SRC0", 266)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(program.instructions[..8]
            .iter()
            .all(|instruction| matches!(instruction.op, ProgramOp::Vector(_))));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (12, 0));
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 12,
                sgprs: 1,
                records: 1,
                step_cap: 16,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .is_ok());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 11,
                sgprs: 1,
                records: 1,
                step_cap: 16,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .is_err());
    }

    #[test]
    fn half_conversions_and_float_compare_account_exact_resources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[("VDST", 0), ("SRC0", 256), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CMP_EQ_F32",
            "ENC_VOP3",
            &[("VDST", 2), ("SRC0", 256), ("SRC1", 257), ("ABS", 1)],
            None,
        );
        append(
            &mut code,
            "V_CMP_LG_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 256), ("SRC1", 257), ("NEG", 2)],
            None,
        );
        append(
            &mut code,
            "V_CMP_NLE_F32",
            "ENC_VOP3",
            &[("VDST", 6), ("SRC0", 263), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CMP_NEQ_F32",
            "ENC_VOP3",
            &[("VDST", 8), ("SRC0", 263), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CVT_F16_F32",
            "ENC_VOP1",
            &[("VDST", 5), ("SRC0", 258)],
            None,
        );
        append(
            &mut code,
            "V_CVT_F32_F16",
            "ENC_VOP1",
            &[("VDST", 6), ("SRC0", 261)],
            None,
        );
        append(
            &mut code,
            "V_CVT_U32_F32",
            "ENC_VOP1",
            &[("VDST", 7), ("SRC0", 262)],
            None,
        );
        append(
            &mut code,
            "V_CVT_F16_F32",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 20),
                ("SRC0", 249),
                ("VSRC0", 21),
                ("SRC0_SEL", 6),
                ("DST_SEL", 5),
                ("DST_UNUSED", 0),
            ],
            None,
        );
        append(
            &mut code,
            "V_CMP_LT_F16",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", 278), ("SRC1", 279)],
            None,
        );
        append(
            &mut code,
            "V_CMP_LT_U32",
            "ENC_VOPC",
            &[("SRC0", 280), ("VSRC1", 25)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let requirements = requirements(&program).unwrap();
        assert_eq!((requirements.vgprs, requirements.sgprs), (26, 12));
        let limits = |vgprs, sgprs| ProgramLimits {
            vgprs,
            sgprs,
            records: 1,
            step_cap: 32,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, limits(26, 12)).is_ok());
        assert!(validate(&program, limits(25, 12)).is_err());
        assert!(validate(&program, limits(26, 11)).is_err());
    }

    #[test]
    fn float_math_family_classifies_and_accounts_exact_resources() {
        let mut code = Vec::new();
        for (name, dst) in [
            ("V_CVT_F32_UBYTE0", 2),
            ("V_CVT_F32_UBYTE1", 3),
            ("V_CVT_F32_UBYTE2", 15),
            ("V_CVT_F32_UBYTE3", 16),
            ("V_RCP_F32", 4),
            ("V_RCP_IFLAG_F32", 5),
            ("V_RSQ_F32", 6),
            ("V_SQRT_F32", 7),
            ("V_SIN_F32", 8),
            ("V_COS_F32", 9),
        ] {
            append(
                &mut code,
                name,
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            );
        }
        append(
            &mut code,
            "V_MAX_F32",
            "ENC_VOP2",
            &[("VDST", 10), ("SRC0", 256), ("VSRC1", 1)],
            None,
        );
        append(
            &mut code,
            "V_MUL_F32",
            "ENC_VOP3",
            &[("VDST", 17), ("SRC0", 266), ("SRC1", 267)],
            None,
        );
        append(
            &mut code,
            "V_PACK_B32_F16",
            "ENC_VOP3",
            &[("VDST", 11), ("SRC0", 256), ("SRC1", 257), ("OP_SEL", 2)],
            None,
        );
        append(
            &mut code,
            "V_FMA_MIXLO_F16",
            "ENC_VOP3P",
            &[("VDST", 12), ("SRC0", 242), ("SRC1", 244), ("SRC2", 242)],
            None,
        );
        append(
            &mut code,
            "V_FMA_MIXHI_F16",
            "ENC_VOP3P",
            &[("VDST", 13), ("SRC0", 242), ("SRC1", 244), ("SRC2", 242)],
            None,
        );
        append(
            &mut code,
            "V_FMA_MIX_F32",
            "ENC_VOP3P",
            &[("VDST", 14), ("SRC0", 242), ("SRC1", 244), ("SRC2", 242)],
            None,
        );
        append(
            &mut code,
            "V_CMP_GT_F32",
            "ENC_VOP3",
            &[("VDST", 0), ("SRC0", 256), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CMP_LE_F32",
            "ENC_VOPC",
            &[("SRC0", 256), ("VSRC1", 1)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (18, 2));
    }

    #[test]
    fn literal_fma_accounts_for_both_vector_sources_and_destination() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_FMAAK_F32",
            "ENC_VOP2",
            &[("VDST", 9), ("SRC0", 267), ("VSRC1", 12)],
            Some((-1.0f32).to_bits()),
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (13, 0));
        let mut exact = limits();
        exact.vgprs = 13;
        exact.sgprs = 0;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 12;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn literal_mad_and_minimum3_account_for_all_vector_sources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_MADAK_F32",
            "ENC_VOP2",
            &[("VDST", 9), ("SRC0", 267), ("VSRC1", 12)],
            Some((-1.0f32).to_bits()),
        );
        append(
            &mut code,
            "V_MIN3_F32",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", 267), ("SRC1", 268), ("SRC2", 269)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (14, 0));
        let mut exact = limits();
        exact.vgprs = 14;
        exact.sgprs = 0;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 13;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn legacy_fma_accounts_for_all_three_vector_sources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_FMA_LEGACY_F32",
            "ENC_VOP3",
            &[("VDST", 9), ("SRC0", 267), ("SRC1", 268), ("SRC2", 269)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (14, 0));
        let mut exact = limits();
        exact.vgprs = 14;
        exact.sgprs = 0;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 13;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn residual_bit_ops_account_for_all_source_classes_and_destination() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_BFI_B32",
            "ENC_VOP3",
            &[("VDST", 9), ("SRC0", 267), ("SRC1", 20), ("SRC2", 193)],
            None,
        );
        append(
            &mut code,
            "V_XAD_U32",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", 268), ("SRC1", 269), ("SRC2", 130)],
            None,
        );
        append(
            &mut code,
            "V_NOT_B32",
            "ENC_VOP1",
            &[("VDST", 11), ("SRC0", 270)],
            None,
        );
        append(
            &mut code,
            "V_FFBH_U32",
            "ENC_VOP1",
            &[("VDST", 15), ("SRC0", 270)],
            None,
        );
        append(
            &mut code,
            "V_FFBL_B32",
            "ENC_VOP1",
            &[("VDST", 14), ("SRC0", 22)],
            None,
        );
        append(
            &mut code,
            "V_MAX_U32",
            "ENC_VOP2",
            &[("VDST", 16), ("SRC0", 22), ("VSRC1", 15)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (17, 23));
        let mut exact = limits();
        exact.vgprs = 17;
        exact.sgprs = 23;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 16;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 17;
        exact.sgprs = 22;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn u64_vector_compare_accounts_for_paired_sources_and_mask_destination() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CMP_EQ_U64",
            "ENC_VOPC",
            &[("SRC0", 258), ("VSRC1", 0)],
            None,
        );
        append(
            &mut code,
            "V_CMP_EQ_U64",
            "ENC_VOP3",
            &[("VDST", 104), ("SRC0", 100), ("SRC1", 506)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (252, 106));
        let mut exact = limits();
        exact.vgprs = 252;
        exact.sgprs = 106;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 251;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 252;
        exact.sgprs = 105;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn vector_carry_forms_account_for_masks_and_register_limits() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_ADD_CO_U32",
            "VOP3_SDST_ENC",
            &[("VDST", 8), ("SDST", 20), ("SRC0", 265), ("SRC1", 30)],
            None,
        );
        append(
            &mut code,
            "V_ADD_CO_CI_U32",
            "VOP3_SDST_ENC",
            &[
                ("VDST", 10),
                ("SDST", 104),
                ("SRC0", 267),
                ("SRC1", 193),
                ("SRC2", 102),
            ],
            None,
        );
        append(
            &mut code,
            "V_SUBREV_CO_CI_U32",
            "VOP3_SDST_ENC",
            &[
                ("VDST", 12),
                ("SDST", 104),
                ("SRC0", 267),
                ("SRC1", 193),
                ("SRC2", 102),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert_eq!(
            (
                requirements(&program).unwrap().vgprs,
                requirements(&program).unwrap().sgprs
            ),
            (13, 106)
        );
        let mut exact = limits();
        exact.vgprs = 13;
        exact.sgprs = 106;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 12;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 13;
        exact.sgprs = 105;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn vector_shift_or_accounts_for_scalar_literal_and_vector_sources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_LSHL_OR_B32",
            "ENC_VOP3",
            &[("VDST", 240), ("SRC0", 506), ("SRC1", 100), ("SRC2", 255)],
            Some(0x8000_0001),
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (251, 101));
        let mut exact = limits();
        exact.vgprs = 251;
        exact.sgprs = 101;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 250;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 251;
        exact.sgprs = 100;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn vector_median_accounts_for_all_sources_and_destination() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_MED3_F32",
            "ENC_VOP3",
            &[("VDST", 240), ("SRC0", 506), ("SRC1", 100), ("SRC2", 255)],
            Some((-0.0f32).to_bits()),
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (251, 101));
        let mut exact = limits();
        exact.vgprs = 251;
        exact.sgprs = 101;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 250;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 251;
        exact.sgprs = 100;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn cmpx_family_accounts_sources_without_treating_exec_as_input_storage() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_CMPX_EQ_F32",
            "ENC_VOPC",
            &[("SRC0", 100), ("VSRC1", 250)],
            None,
        );
        append(
            &mut code,
            "V_CMPX_NEQ_F32",
            "ENC_VOPC",
            &[("SRC0", 255), ("VSRC1", 249)],
            Some(f32::NAN.to_bits()),
        );
        append(
            &mut code,
            "V_CMPX_GE_F32",
            "ENC_VOPC",
            &[("SRC0", 242), ("VSRC1", 248)],
            None,
        );
        append(
            &mut code,
            "V_CMPX_NGT_F32",
            "ENC_VOPC",
            &[("SRC0", 243), ("VSRC1", 247)],
            None,
        );
        append(
            &mut code,
            "V_CMPX_GE_U32",
            "ENC_VOPC",
            &[("SRC0", 159), ("VSRC1", 246)],
            None,
        );
        append(
            &mut code,
            "V_CMP_NGE_F32",
            "ENC_VOPC",
            &[("SRC0", 100), ("VSRC1", 245)],
            None,
        );
        append(
            &mut code,
            "V_CMP_NGT_F32",
            "ENC_VOPC",
            &[("SRC0", 100), ("VSRC1", 244)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (251, 101));
        let mut exact = limits();
        exact.vgprs = 251;
        exact.sgprs = 101;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 250;
        assert!(validate(&program, exact).is_err());
        exact.vgprs = 251;
        exact.sgprs = 100;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn vector_integer_family_classifies_and_accounts_exact_resources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_MUL_HI_U32",
            "ENC_VOP3",
            &[("VDST", 26), ("SRC0", 266), ("SRC1", 267)],
            None,
        );
        append(
            &mut code,
            "V_ADD_LSHL_U32",
            "ENC_VOP3",
            &[("VDST", 25), ("SRC0", 256), ("SRC1", 257), ("SRC2", 131)],
            None,
        );
        append(
            &mut code,
            "V_CMP_GE_U32",
            "ENC_VOP3",
            &[("VDST", 20), ("SRC0", 256), ("SRC1", 257)],
            None,
        );
        append(
            &mut code,
            "V_CMPX_GT_I32",
            "ENC_VOPC",
            &[("SRC0", 193), ("VSRC1", 0)],
            None,
        );
        append(
            &mut code,
            "V_CMP_GT_I32",
            "ENC_VOPC",
            &[("SRC0", 273), ("VSRC1", 16)],
            None,
        );
        append(
            &mut code,
            "V_CMPX_NE_U32",
            "ENC_VOPC",
            &[("SRC0", 159), ("VSRC1", 0)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let requirements = requirements(&program).unwrap();
        assert_eq!((requirements.vgprs, requirements.sgprs), (27, 22));
        assert!(matches!(
            program.instructions[4].op,
            ProgramOp::Vector(StageBOp::Compare {
                kind: mesh_vector::CompareKind::SignedGreater,
                cmpx: false,
                ..
            })
        ));
    }

    #[test]
    fn observed_sdwa_shift_accounts_transformed_sources_and_destination() {
        let mut code: Vec<u8> = [0x347842f9u32, 0x02860600]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (61, 1));
        let mut exact = limits();
        exact.vgprs = 61;
        exact.sgprs = 1;
        validate(&program, exact).unwrap();
        exact.vgprs = 60;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn invalid_targets_literals_unknown_and_truncation_fail_closed() {
        let mut literal_target = Vec::new();
        append(
            &mut literal_target,
            "S_BRANCH",
            "ENC_SOPP",
            &[("SIMM16", 1)],
            None,
        );
        append(
            &mut literal_target,
            "V_ADD_F32",
            "ENC_VOP2",
            &[("VDST", 1), ("SRC0", 255), ("VSRC1", 0)],
            Some(0x3f80_0000),
        );
        let program = decode(&literal_target).unwrap();
        assert_eq!(program.instructions[1].literal, Some(0x3f80_0000));
        assert_eq!(program.instructions[1].byte_len, 8);
        let reason = match &program.instructions[0].op {
            ProgramOp::Unsupported(reason) => reason,
            other => panic!("unexpected branch classification {other:?}"),
        };
        assert!(reason.contains("outside"));
        assert!(validate(&program, limits()).is_err());

        assert!(decode(&[0xff]).unwrap_err().contains("decode at"));
        assert!(decode(&[0xff, 0xff, 0xff, 0xff]).is_err());
    }

    #[test]
    fn validation_rejects_missing_regions_limits_steps_and_scratch_overflow() {
        let program = decode(&mixed()).unwrap();
        let mut bad = limits();
        bad.read_only = None;
        assert!(validate(&program, bad).unwrap_err().contains("read-only"));
        bad = limits();
        bad.vgprs = 7;
        assert!(validate(&program, bad).unwrap_err().contains("VGPRs"));
        bad = limits();
        bad.sgprs = 11;
        assert!(validate(&program, bad).unwrap_err().contains("SGPRs"));
        bad = limits();
        bad.step_cap = 0;
        assert!(validate(&program, bad).is_err());
        bad = limits();
        bad.records = u32::MAX;
        assert!(validate(&program, bad).is_err());
        bad = limits();
        bad.read_only = Some(ReadOnlyRegion {
            guest_base: u64::MAX,
            memory_offset: u32::MAX,
            byte_len: 2,
        });
        assert!(validate(&program, bad).is_err());

        let no_completion =
            decode(&encode("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None).unwrap()).unwrap();
        assert!(validate(&no_completion, limits())
            .unwrap_err()
            .contains("completion"));
    }

    #[test]
    fn resource_spans_match_memory_address_arity_and_skip_special_sgprs() {
        let no_address = requirements(&vector_load(false, false, 200)).unwrap();
        assert_eq!(no_address.vgprs, 1);
        let two_addresses = requirements(&vector_load(true, true, 5)).unwrap();
        assert_eq!(two_addresses.vgprs, 7);

        let mut global = Vec::new();
        append(
            &mut global,
            "GLOBAL_LOAD_DWORD",
            "ENC_FLAT_GLBL",
            &[("VDST", 9), ("ADDR", 12), ("SADDR", 20), ("OFFSET", 0xffc)],
            None,
        );
        append(&mut global, "S_ENDPGM", "ENC_SOPP", &[], None);
        let global = requirements(&decode(&global).unwrap()).unwrap();
        assert_eq!(global.vgprs, 14);
        assert_eq!(global.sgprs, 22);

        let mut null_scalar = Vec::new();
        append(
            &mut null_scalar,
            "GLOBAL_LOAD_DWORD",
            "ENC_FLAT_GLBL",
            &[("VDST", 1), ("ADDR", 2), ("SADDR", 0x7d)],
            None,
        );
        append(&mut null_scalar, "S_ENDPGM", "ENC_SOPP", &[], None);
        let null_scalar = requirements(&decode(&null_scalar).unwrap()).unwrap();
        assert_eq!(null_scalar.vgprs, 4);
        assert_eq!(null_scalar.sgprs, 0);

        let instruction = |op| ProgramInstruction {
            pc: 0,
            byte_len: 4,
            words: vec![0],
            literal: None,
            name: "S_ENDPGM",
            encoding: "ENC_SOPP",
            op,
        };
        let special_only = MeshProgram {
            byte_len: 8,
            instructions: vec![
                instruction(ProgramOp::Scalar(StageAOp::Select64 {
                    dst: 126,
                    when_set: Operand64::Register(126),
                    when_clear: Operand64::Constant(0),
                })),
                instruction(ProgramOp::Control(ControlOp::Complete)),
            ],
            coverage: CoverageSummary::default(),
        };
        assert_eq!(requirements(&special_only).unwrap().sgprs, 0);

        let adjacent = MeshProgram {
            instructions: vec![
                instruction(ProgramOp::Control(ControlOp::MaskToScalar {
                    dst: 105,
                    src: super::super::mesh_dispatch::MaskRegister::Exec,
                })),
                instruction(ProgramOp::Control(ControlOp::ScalarToMask {
                    dst: super::super::mesh_dispatch::MaskRegister::Exec,
                    src: 107,
                })),
                instruction(ProgramOp::Control(ControlOp::Complete)),
            ],
            ..special_only
        };
        assert_eq!(requirements(&adjacent).unwrap().sgprs, 109);
    }

    #[test]
    fn extended_memory_forms_classify_and_account_for_full_destinations() {
        let mut code = Vec::new();
        append(
            &mut code,
            "BUFFER_LOAD_DWORDX4",
            "ENC_MUBUF",
            &[
                ("VDATA", 10),
                ("VADDR", 3),
                ("SRSRC", 1),
                ("SOFFSET", 128),
                ("OFFEN", 1),
                ("IDXEN", 1),
            ],
            None,
        );
        append(
            &mut code,
            "TBUFFER_LOAD_FORMAT_XYZW",
            "ENC_MTBUF",
            &[
                ("VDATA", 20),
                ("VADDR", 5),
                ("SRSRC", 1),
                ("SOFFSET", 128),
                ("OFFEN", 1),
                ("FORMAT", 69),
            ],
            None,
        );
        append(
            &mut code,
            "TBUFFER_LOAD_FORMAT_XYZ",
            "ENC_MTBUF",
            &[
                ("VDATA", 24),
                ("VADDR", 5),
                ("SRSRC", 1),
                ("SOFFSET", 128),
                ("OFFEN", 1),
                ("FORMAT", 60),
            ],
            None,
        );
        append(
            &mut code,
            "BUFFER_LOAD_DWORDX3",
            "ENC_MUBUF",
            &[
                ("VDATA", 27),
                ("VADDR", 7),
                ("SRSRC", 1),
                ("SOFFSET", 128),
                ("OFFEN", 1),
            ],
            None,
        );
        append(
            &mut code,
            "BUFFER_LOAD_FORMAT_XYZW",
            "ENC_MUBUF",
            &[("VDATA", 40), ("VADDR", 9), ("SRSRC", 1), ("SOFFSET", 128)],
            None,
        );
        append(
            &mut code,
            "TBUFFER_LOAD_FORMAT_X",
            "ENC_MTBUF",
            &[
                ("VDATA", 50),
                ("VADDR", 11),
                ("SRSRC", 1),
                ("SOFFSET", 128),
                ("OFFEN", 1),
                ("FORMAT", 24),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[1].op,
            ProgramOp::Memory(StageCOp::VectorBufferLoad {
                words: 4,
                format: mesh_memory::BufferFormatSource::Instruction(69),
                ..
            })
        ));
        assert!(matches!(
            program.instructions[2].op,
            ProgramOp::Memory(StageCOp::VectorBufferLoad {
                words: 3,
                format: mesh_memory::BufferFormatSource::Instruction(60),
                ..
            })
        ));
        assert!(matches!(
            program.instructions[3].op,
            ProgramOp::Memory(StageCOp::VectorBufferLoad {
                words: 3,
                format: mesh_memory::BufferFormatSource::Raw,
                ..
            })
        ));
        assert!(matches!(
            program.instructions[4].op,
            ProgramOp::Memory(StageCOp::VectorBufferLoad {
                words: 4,
                format: mesh_memory::BufferFormatSource::Descriptor,
                ..
            })
        ));
        assert!(matches!(
            program.instructions[5].op,
            ProgramOp::Memory(StageCOp::VectorBufferLoad {
                dst: 50,
                address: 11,
                words: 1,
                format: mesh_memory::BufferFormatSource::Instruction(24),
                ..
            })
        ));
        assert_eq!(program.coverage.families[&ProgramFamily::Memory], 6);
        assert_eq!(
            program
                .coverage
                .families
                .get(&ProgramFamily::Unsupported)
                .copied()
                .unwrap_or(0),
            0
        );
        let requirements = requirements(&program).unwrap();
        assert_eq!(requirements.vgprs, 51);
        assert_eq!(requirements.sgprs, 8);
        assert!(requirements.uses_read_only_memory);
        let mut exact = limits();
        exact.vgprs = 51;
        exact.sgprs = 8;
        assert!(validate(&program, exact).is_ok());
        exact.vgprs = 50;
        assert!(validate(&program, exact).is_err());
    }

    #[test]
    fn bounded_image_load_classifies_and_accounts_coordinates() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_LOAD",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 5),
                ("SRSRC", 3),
                ("DMASK", 7),
                ("DIM", 2),
                ("UNORM", 1),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::ImageLoad {
                coordinates: [5, 6, 7],
                coordinate_count: 3,
                dmask: 7,
                ..
            })
        ));
        assert_eq!(program.coverage.families[&ProgramFamily::Memory], 1);
        assert_eq!(
            program
                .coverage
                .families
                .get(&ProgramFamily::Unsupported)
                .copied()
                .unwrap_or(0),
            0
        );
        let required = requirements(&program).unwrap();
        assert_eq!(required.vgprs, 15);
        assert_eq!(required.sgprs, 20);
        assert!(required.uses_read_only_memory);
        assert!(required.uses_read_only_image);
        let mut limits = ProgramLimits {
            vgprs: 15,
            sgprs: 20,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 16,
            }),
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, limits).is_err());
        let valid_contract = ImageDescriptorContract {
            register: 12,
            words: [
                0,
                (77 << 20) | (3 << 30),
                (3 << 14),
                4 | (5 << 3) | (6 << 6) | (7 << 9) | (10 << 28),
                0,
                0,
                0,
                0,
            ],
        };
        limits.image_descriptors = std::slice::from_ref(&valid_contract);
        assert!(validate(&program, limits).is_ok());
        let valid = valid_contract.words;
        for invalid in [
            {
                let mut words = valid;
                words[3] |= 1 << 20;
                words
            },
            {
                let mut words = valid;
                words[3] |= 1 << 12;
                words
            },
            {
                let mut words = valid;
                words[3] = (words[3] & 0x0fff_ffff) | (9 << 28);
                words
            },
            {
                let mut words = valid;
                words[1] = (words[1] & !(0x1ff << 20)) | (56 << 20);
                words
            },
        ] {
            let invalid_contract = ImageDescriptorContract {
                register: 12,
                words: invalid,
            };
            assert!(validate(
                &program,
                ProgramLimits {
                    image_descriptors: std::slice::from_ref(&invalid_contract),
                    ..limits
                }
            )
            .is_err());
        }
    }

    #[test]
    fn bounded_image_sample_requires_explicit_image_and_sampler_contracts() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 5),
                ("SRSRC", 3),
                ("SSAMP", 6),
                ("DMASK", 7),
                ("DIM", 1),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::ImageSampleLz {
                coordinates: [5, 6, _],
                coordinate_count: 2,
                descriptor: 12,
                sampler: 24,
                ..
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (15, 28));
        assert!(required.uses_read_only_image && required.uses_sampler);
        let image_contract = ImageDescriptorContract {
            register: 12,
            words: [
                0,
                (77 << 20) | (3 << 30),
                3 | (3 << 14),
                4 | (5 << 3) | (6 << 6) | (7 << 9) | (9 << 28),
                0,
                0,
                0,
                0,
            ],
        };
        let mut limits = ProgramLimits {
            vgprs: 15,
            sgprs: 28,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4096,
            }),
            image_descriptors: std::slice::from_ref(&image_contract),
            sampler_descriptors: &[],
        };
        assert!(validate(&program, limits).is_err());
        let valid_sampler = SamplerDescriptorContract {
            register: 24,
            words: [146, 0, 0, 0],
        };
        limits.sampler_descriptors = std::slice::from_ref(&valid_sampler);
        assert!(validate(&program, limits).is_ok());
        let invalid_sampler = SamplerDescriptorContract {
            register: 24,
            words: [0, 0, 0, 0],
        };
        assert!(validate(
            &program,
            ProgramLimits {
                sampler_descriptors: std::slice::from_ref(&invalid_sampler),
                ..limits
            }
        )
        .is_err());
    }

    #[test]
    fn bounded_array_sample_requires_array_descriptor_and_three_coordinates() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 5),
                ("SRSRC", 3),
                ("SSAMP", 6),
                ("DMASK", 7),
                ("DIM", 5),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::ImageSampleLz {
                coordinates: [5, 6, 7],
                coordinate_count: 3,
                arrayed: true,
                ..
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (15, 28));
        let descriptor = |ty: u32| ImageDescriptorContract {
            register: 12,
            words: [
                0,
                (77 << 20) | (3 << 30),
                3 | (3 << 14),
                4 | (5 << 3) | (6 << 6) | (7 << 9) | (ty << 28),
                3,
                0,
                0,
                0,
            ],
        };
        let image_contract = descriptor(13);
        let sampler_contract = SamplerDescriptorContract {
            register: 24,
            words: [146, 0, 0, 0],
        };
        let mut limits = ProgramLimits {
            vgprs: 15,
            sgprs: 28,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4096,
            }),
            image_descriptors: std::slice::from_ref(&image_contract),
            sampler_descriptors: std::slice::from_ref(&sampler_contract),
        };
        assert!(validate(&program, limits).is_ok());
        limits.vgprs = 7;
        assert!(validate(&program, limits).is_err());
        limits.vgprs = 15;
        let wrong_descriptor = descriptor(10);
        assert!(validate(
            &program,
            ProgramLimits {
                image_descriptors: std::slice::from_ref(&wrong_descriptor),
                ..limits
            }
        )
        .is_err());
    }

    #[test]
    fn bounded_image_gather_accounts_four_results_and_contracts() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_GATHER4_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 5),
                ("SRSRC", 3),
                ("SSAMP", 6),
                ("DMASK", 2),
                ("DIM", 1),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (16, 28));
        assert!(required.uses_read_only_image && required.uses_sampler);
        let image_contract = ImageDescriptorContract {
            register: 12,
            words: [
                0,
                (77 << 20) | (3 << 30),
                3 | (3 << 14),
                4 | (5 << 3) | (6 << 6) | (7 << 9) | (9 << 28),
                0,
                0,
                0,
                0,
            ],
        };
        let sampler_contract = SamplerDescriptorContract {
            register: 24,
            words: [146, 0, 0, 0],
        };
        let mut limits = ProgramLimits {
            vgprs: 15,
            sgprs: 28,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4096,
            }),
            image_descriptors: std::slice::from_ref(&image_contract),
            sampler_descriptors: std::slice::from_ref(&sampler_contract),
        };
        assert!(validate(&program, limits).is_err());
        limits.vgprs = 16;
        assert!(validate(&program, limits).is_ok());
        let wrong_sampler = SamplerDescriptorContract {
            register: 20,
            words: [146, 0, 0, 0],
        };
        assert!(validate(
            &program,
            ProgramLimits {
                sampler_descriptors: std::slice::from_ref(&wrong_sampler),
                ..limits
            }
        )
        .is_err());
    }

    #[test]
    fn explicit_image_lod_is_included_in_vgpr_limits() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_SAMPLE_L",
            "ENC_MIMG",
            &[
                ("VDATA", 4),
                ("VADDR", 20),
                ("SRSRC", 3),
                ("SSAMP", 6),
                ("DMASK", 1),
                ("DIM", 1),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::ImageSampleLz {
                coordinates: [20, 21, _],
                lod: Some(22),
                ..
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (23, 28));
        let image_contract = ImageDescriptorContract {
            register: 12,
            words: [
                0,
                (77 << 20) | (3 << 30),
                3 | (3 << 14),
                4 | (5 << 3) | (6 << 6) | (7 << 9) | (9 << 28),
                0,
                0,
                0,
                0,
            ],
        };
        let sampler_contract = SamplerDescriptorContract {
            register: 24,
            words: [146, 0, 0, 0],
        };
        let mut limits = ProgramLimits {
            vgprs: 22,
            sgprs: 28,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4096,
            }),
            image_descriptors: std::slice::from_ref(&image_contract),
            sampler_descriptors: std::slice::from_ref(&sampler_contract),
        };
        assert!(validate(&program, limits).is_err());
        limits.vgprs = 23;
        assert!(validate(&program, limits).is_ok());
    }

    #[test]
    fn descriptor_manifest_separates_structure_from_complete_bindings() {
        let mut code = Vec::new();
        append(
            &mut code,
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 12),
                ("VADDR", 5),
                ("SRSRC", 3),
                ("SSAMP", 6),
                ("DMASK", 7),
                ("DIM", 1),
            ],
            None,
        );
        append(
            &mut code,
            "IMAGE_SAMPLE_LZ",
            "ENC_MIMG",
            &[
                ("VDATA", 20),
                ("VADDR", 8),
                ("SRSRC", 5),
                ("SSAMP", 8),
                ("DMASK", 1),
                ("DIM", 5),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        let base = ProgramLimits {
            vgprs: 64,
            sgprs: 64,
            records: 1,
            step_cap: 16,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4096,
            }),
            image_descriptors: &[],
            sampler_descriptors: &[],
        };

        let structural = validate_structure(&program, base).unwrap();
        assert_eq!(structural.descriptors.images.len(), 2);
        assert_eq!(
            structural.descriptors.images[&12],
            BTreeSet::from([ImageDescriptorUse {
                coordinate_count: 2,
                arrayed: false,
            }])
        );
        assert_eq!(
            structural.descriptors.images[&20],
            BTreeSet::from([ImageDescriptorUse {
                coordinate_count: 3,
                arrayed: true,
            }])
        );
        assert_eq!(structural.descriptors.samplers, BTreeSet::from([24, 32]));
        assert!(validate(&program, base).is_err());

        let images = [
            ImageDescriptorContract {
                register: 12,
                words: [
                    0,
                    (77 << 20) | (3 << 30),
                    3 | (3 << 14),
                    4 | (5 << 3) | (6 << 6) | (7 << 9) | (9 << 28),
                    0,
                    0,
                    0,
                    0,
                ],
            },
            ImageDescriptorContract {
                register: 20,
                words: [
                    0,
                    (77 << 20) | (3 << 30),
                    3 | (3 << 14),
                    4 | (5 << 3) | (6 << 6) | (7 << 9) | (13 << 28),
                    3,
                    0,
                    0,
                    0,
                ],
            },
        ];
        let samplers = [
            SamplerDescriptorContract {
                register: 24,
                words: [146, 0, 0, 0],
            },
            SamplerDescriptorContract {
                register: 32,
                words: [146, 0, 0, 0],
            },
        ];
        let bound = ProgramLimits {
            image_descriptors: &images,
            sampler_descriptors: &samplers,
            ..base
        };
        assert!(validate(&program, bound).is_ok());

        for incomplete in [&images[..1], &images[1..]] {
            assert!(validate(
                &program,
                ProgramLimits {
                    image_descriptors: incomplete,
                    ..bound
                }
            )
            .is_err());
        }
        assert!(validate(
            &program,
            ProgramLimits {
                sampler_descriptors: &samplers[..1],
                ..bound
            }
        )
        .is_err());

        let duplicate_images = [images[0], images[0], images[1]];
        assert!(validate(
            &program,
            ProgramLimits {
                image_descriptors: &duplicate_images,
                ..bound
            }
        )
        .is_err());
        let duplicate_samplers = [samplers[0], samplers[0], samplers[1]];
        assert!(validate(
            &program,
            ProgramLimits {
                sampler_descriptors: &duplicate_samplers,
                ..bound
            }
        )
        .is_err());

        let mut mismatched_images = images;
        mismatched_images[1].words[3] = (mismatched_images[1].words[3] & 0x0fff_ffff) | (9 << 28);
        assert!(validate(
            &program,
            ProgramLimits {
                image_descriptors: &mismatched_images,
                ..bound
            }
        )
        .is_err());

        let unused_image = ImageDescriptorContract {
            register: 40,
            words: images[0].words,
        };
        let images_with_unused = [images[0], images[1], unused_image];
        assert!(validate(
            &program,
            ProgramLimits {
                image_descriptors: &images_with_unused,
                ..bound
            }
        )
        .is_err());
        let unused_sampler = SamplerDescriptorContract {
            register: 40,
            words: [146, 0, 0, 0],
        };
        let samplers_with_unused = [samplers[0], samplers[1], unused_sampler];
        assert!(validate(
            &program,
            ProgramLimits {
                sampler_descriptors: &samplers_with_unused,
                ..bound
            }
        )
        .is_err());
    }

    #[test]
    fn scalar_multiply_and_x8_loads_compose_with_full_register_spans() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_MUL_I32",
            "ENC_SOP2",
            &[("SDST", 90), ("SSRC0", 1), ("SSRC1", 2)],
            None,
        );
        append(
            &mut code,
            "S_LOAD_DWORDX8",
            "ENC_SMEM",
            &[
                ("SDATA", 64),
                ("SBASE", 0),
                ("OFFSET", 32),
                ("SOFFSET", 125),
            ],
            None,
        );
        append(
            &mut code,
            "S_BUFFER_LOAD_DWORDX8",
            "ENC_SMEM",
            &[
                ("SDATA", 72),
                ("SBASE", 4),
                ("OFFSET", 64),
                ("SOFFSET", 125),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Scalar(StageAOp::Binary {
                kind: mesh_scalar::ScalarBinaryKind::MulI32,
                ..
            })
        ));
        assert!(matches!(
            program.instructions[1].op,
            ProgramOp::Memory(StageCOp::ScalarLoad { words: 8, .. })
        ));
        assert!(matches!(
            program.instructions[2].op,
            ProgramOp::Memory(StageCOp::ScalarBufferLoad { words: 8, .. })
        ));
        assert_eq!(program.coverage.families[&ProgramFamily::Scalar], 1);
        assert_eq!(program.coverage.families[&ProgramFamily::Memory], 2);
        assert_eq!(
            program
                .coverage
                .families
                .get(&ProgramFamily::Unsupported)
                .copied()
                .unwrap_or(0),
            0
        );
        let requirements = requirements(&program).unwrap();
        assert_eq!(requirements.sgprs, 91);
        assert!(requirements.uses_read_only_memory);
    }

    #[test]
    fn scalar_dword_load_accounts_for_base_offset_and_destination() {
        let mut code = vec![0x03, 0x19, 0x00, 0xf4, 0xfc, 0xff, 0x1f, 0x12];
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Memory(StageCOp::ScalarLoad {
                dst: 100,
                base: 6,
                offset: -4,
                soffset: OffsetSource::Scalar(9),
                words: 1,
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 101));
        assert!(required.uses_read_only_memory);
        let mut limits = ProgramLimits {
            vgprs: 0,
            sgprs: 100,
            records: 1,
            step_cap: 8,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4,
            }),
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, limits).is_err());
        limits.sgprs = 101;
        assert!(validate(&program, limits).is_ok());
    }

    #[test]
    fn scalar_integer_family_classifies_and_accounts_exactly() {
        let mut code = Vec::new();
        for (name, encoding, fields) in [
            (
                "S_MIN_U32",
                "ENC_SOP2",
                vec![("SDST", 20), ("SSRC0", 0), ("SSRC1", 1)],
            ),
            (
                "S_MUL_HI_U32",
                "ENC_SOP2",
                vec![("SDST", 21), ("SSRC0", 2), ("SSRC1", 3)],
            ),
            ("S_NOT_B32", "ENC_SOP1", vec![("SDST", 22), ("SSRC0", 4)]),
            (
                "S_SUB_I32",
                "ENC_SOP2",
                vec![("SDST", 23), ("SSRC0", 6), ("SSRC1", 7)],
            ),
            (
                "S_XOR_B32",
                "ENC_SOP2",
                vec![("SDST", 24), ("SSRC0", 25), ("SSRC1", 26)],
            ),
        ] {
            append(&mut code, name, encoding, &fields, None);
        }
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        assert!(program.instructions[..5]
            .iter()
            .all(|instruction| matches!(instruction.op, ProgramOp::Scalar(_))));
        assert_eq!(program.coverage.families[&ProgramFamily::Scalar], 5);
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 27));
    }

    #[test]
    fn scalar_immediate_reverse_and_ge_classify_and_account_exactly() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_ADDK_I32",
            "ENC_SOPK",
            &[("SDST", 30), ("SIMM16", 0xffff)],
            None,
        );
        append(
            &mut code,
            "S_BREV_B32",
            "ENC_SOP1",
            &[("SDST", 31), ("SSRC0", 30)],
            None,
        );
        append(
            &mut code,
            "S_CMP_GE_U32",
            "ENC_SOPC",
            &[("SSRC0", 31), ("SSRC1", 193)],
            None,
        );
        append(
            &mut code,
            "S_CMPK_EQ_I32",
            "ENC_SOPK",
            &[("SDST", 40), ("SIMM16", 0xffff)],
            None,
        );
        append(
            &mut code,
            "S_CMPK_GT_U32",
            "ENC_SOPK",
            &[("SDST", 41), ("SIMM16", 0xffff)],
            None,
        );
        append(
            &mut code,
            "S_CMP_GE_I32",
            "ENC_SOPC",
            &[("SSRC0", 42), ("SSRC1", 43)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(program.instructions[..6]
            .iter()
            .all(|instruction| matches!(instruction.op, ProgramOp::Scalar(_))));
        assert_eq!(program.coverage.families[&ProgramFamily::Scalar], 6);
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 44));
    }

    #[test]
    fn scalar_shift_and_bit_counts_account_for_pair_sources() {
        let mut code = Vec::new();
        append(
            &mut code,
            "S_ASHR_I32",
            "ENC_SOP2",
            &[("SDST", 95), ("SSRC0", 94), ("SSRC1", 130)],
            None,
        );
        append(
            &mut code,
            "S_BCNT1_I32_B32",
            "ENC_SOP1",
            &[("SDST", 96), ("SSRC0", 95)],
            None,
        );
        append(
            &mut code,
            "S_BCNT1_I32_B64",
            "ENC_SOP1",
            &[("SDST", 97), ("SSRC0", 103)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(program.instructions[..3]
            .iter()
            .all(|instruction| matches!(instruction.op, ProgramOp::Scalar(_))));
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (0, 105));

        let limits = ProgramLimits {
            vgprs: 0,
            sgprs: 104,
            records: 1,
            step_cap: 8,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, limits).is_err());
        assert!(validate(
            &program,
            ProgramLimits {
                sgprs: 105,
                ..limits
            }
        )
        .is_ok());
    }

    #[test]
    fn read_only_regions_require_dword_alignment() {
        let program = decode(&mixed()).unwrap();
        for region in [
            ReadOnlyRegion {
                guest_base: 0x1001,
                memory_offset: 0,
                byte_len: 4,
            },
            ReadOnlyRegion {
                guest_base: 0x1000,
                memory_offset: 2,
                byte_len: 4,
            },
            ReadOnlyRegion {
                guest_base: 0x1000,
                memory_offset: 0,
                byte_len: 2,
            },
            ReadOnlyRegion {
                guest_base: 0x1000,
                memory_offset: 0,
                byte_len: 0,
            },
        ] {
            let mut bad = limits();
            bad.read_only = Some(region);
            assert!(validate(&program, bad)
                .unwrap_err()
                .contains("dword aligned"));
        }
    }

    #[test]
    fn unsupported_forms_have_deterministic_coverage() {
        let code = encode("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)], None).unwrap();
        let first = decode(&code).unwrap();
        let second = decode(&code).unwrap();
        assert_eq!(first.coverage, second.coverage);
        assert_eq!(first.coverage.families[&ProgramFamily::Unsupported], 1);
        assert_eq!(first.coverage.opcodes["S_SETPC_B64"], 1);
        assert!(validate(&first, limits())
            .unwrap_err()
            .contains("S_SETPC_B64"));
    }

    #[test]
    fn allocation_exports_compose_with_deterministic_resources() {
        let mut code = Vec::new();
        append(&mut code, "S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None);
        append(
            &mut code,
            "EXP",
            "ENC_EXP",
            &[
                ("TGT", 12),
                ("EN", 5),
                ("VSRC0", 2),
                ("VSRC1", 200),
                ("VSRC2", 7),
                ("VSRC3", 201),
                ("DONE", 1),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families[&ProgramFamily::AllocationExport],
            2
        );
        let required = requirements(&program).unwrap();
        assert_eq!(required.sgprs, 125);
        assert_eq!(required.vgprs, 8);
        let validated = validate(
            &program,
            ProgramLimits {
                vgprs: 8,
                sgprs: 125,
                records: 1,
                step_cap: 16,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            },
        )
        .unwrap();
        assert_eq!(validated.requirements, required);

        let mut parameter = Vec::new();
        append(
            &mut parameter,
            "EXP",
            "ENC_EXP",
            &[("TGT", 32), ("EN", 5), ("VSRC0", 3), ("VSRC2", 9)],
            None,
        );
        append(&mut parameter, "S_ENDPGM", "ENC_SOPP", &[], None);
        let parameter = decode(&parameter).unwrap();
        assert_eq!(
            parameter.coverage.families[&ProgramFamily::AllocationExport],
            1
        );
        assert_eq!(requirements(&parameter).unwrap().vgprs, 10);
        assert!(validate(
            &parameter,
            ProgramLimits {
                vgprs: 10,
                sgprs: 4,
                records: 1,
                step_cap: 16,
                read_only: None,
                image_descriptors: &[],
                sampler_descriptors: &[],
            }
        )
        .is_ok());

        let invalid = encode("EXP", "ENC_EXP", &[("TGT", 31), ("EN", 15)], None).unwrap();
        let invalid = decode(&invalid).unwrap();
        assert_eq!(invalid.coverage.families[&ProgramFamily::Unsupported], 1);
    }

    #[test]
    fn lane_reads_and_writes_account_scalar_and_vector_bounds() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_READLANE_B32",
            "ENC_VOP3",
            &[("VDST", 105), ("SRC0", 256 + 217), ("SRC1", 104)],
            None,
        );
        append(
            &mut code,
            "V_WRITELANE_B32",
            "ENC_VOP3",
            &[("VDST", 217), ("SRC0", 105), ("SRC1", 191)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Vector(StageBOp::ReadLane {
                dst: 105,
                src: 217,
                ..
            })
        ));
        assert!(matches!(
            program.instructions[1].op,
            ProgramOp::Vector(StageBOp::WriteLane { dst: 217, .. })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (218, 106));
        let exact = ProgramLimits {
            vgprs: 218,
            sgprs: 106,
            records: 1,
            step_cap: 8,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, exact).is_ok());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 217,
                ..exact
            }
        )
        .is_err());
        assert!(validate(
            &program,
            ProgramLimits {
                sgprs: 105,
                ..exact
            }
        )
        .is_err());
    }

    #[test]
    fn relative_vector_reads_require_the_bounded_hardware_register_file() {
        let mut code = encode(
            "V_MOVRELS_B32",
            "ENC_VOP1",
            &[("VDST", 10), ("SRC0", 256 + 20)],
            None,
        )
        .unwrap();
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Vector(StageBOp::RelativeMove { dst: 10, base: 20 })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (256, 0));
        let exact = ProgramLimits {
            vgprs: 256,
            sgprs: 0,
            records: 1,
            step_cap: 8,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, exact).is_ok());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 255,
                ..exact
            }
        )
        .is_err());
    }

    #[test]
    fn cross_row_permute_accounts_data_selectors_and_destination() {
        let mut code = Vec::new();
        append(
            &mut code,
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[
                ("VDST", 216),
                ("SRC0", 256 + 217),
                ("SRC1", 104),
                ("SRC2", 105),
                ("OP_SEL", 3),
            ],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert_eq!(
            program.coverage.families.get(&ProgramFamily::Unsupported),
            None
        );
        assert!(matches!(
            program.instructions[0].op,
            ProgramOp::Vector(StageBOp::PermLaneX16 {
                dst: 216,
                src: 217,
                ..
            })
        ));
        let required = requirements(&program).unwrap();
        assert_eq!((required.vgprs, required.sgprs), (218, 106));
        let exact = ProgramLimits {
            vgprs: 218,
            sgprs: 106,
            records: 1,
            step_cap: 8,
            read_only: None,
            image_descriptors: &[],
            sampler_descriptors: &[],
        };
        assert!(validate(&program, exact).is_ok());
        assert!(validate(
            &program,
            ProgramLimits {
                vgprs: 217,
                ..exact
            }
        )
        .is_err());
        assert!(validate(
            &program,
            ProgramLimits {
                sgprs: 105,
                ..exact
            }
        )
        .is_err());
    }

    fn validation_limits() -> ProgramLimits<'static> {
        ProgramLimits {
            vgprs: 8,
            sgprs: 16,
            records: 2,
            step_cap: 32,
            read_only: Some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 256,
            }),
            image_descriptors: &[],
            sampler_descriptors: &[],
        }
    }

    #[test]
    fn validation_shader_emits_and_executes_mixed_dependency_chain() {
        let mut code = Vec::new();
        append(&mut code, "S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)], None);
        append(&mut code, "V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 4)], None);
        append(
            &mut code,
            "BUFFER_LOAD_DWORD",
            "ENC_MUBUF",
            &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128), ("OFFEN", 1)],
            None,
        );
        append(
            &mut code,
            "EXP",
            "ENC_EXP",
            &[("TGT", 12), ("EN", 1), ("VSRC0", 1), ("DONE", 1)],
            None,
        );
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);

        let program = decode(&code).unwrap();
        let emitted = validation_shader(&program, validation_limits()).unwrap();
        assert!(emitted.hlsl.contains("[WaveSize(64)]"));
        assert!(emitted.hlsl.contains("xcl_budget"));
        assert!(emitted.hlsl.contains("register(u1)"));
        let dxil = super::super::dxc::compile(&emitted.hlsl, "main", "cs_6_6", &[]).unwrap();

        if super::super::runner::adapter().is_ok()
            && super::super::runner::wave64_supported() == Ok(true)
        {
            let mut state = super::super::runner::KernelState::default();
            state.sgpr[2] = 256;
            let result = super::super::runner::run(&dxil, &state, &[0x51a7_c0de]).unwrap();
            for lane in 0..64 {
                assert_eq!(result.vgpr[lane][1], 0x51a7_c0de, "lane {lane}");
                assert_eq!(result.export(12, lane, 0), 0x51a7_c0de, "lane {lane}");
            }
        }
    }

    #[test]
    fn validation_shader_preserves_branch_skip_and_caps_loops() {
        let mut skip = Vec::new();
        append(&mut skip, "S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)], None);
        append(&mut skip, "S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 129)], None);
        append(&mut skip, "S_ENDPGM", "ENC_SOPP", &[], None);
        let hlsl = validation_shader(&decode(&skip).unwrap(), validation_limits()).unwrap().hlsl;
        assert!(hlsl.contains("uint pc = 0"));

        let mut looped = Vec::new();
        append(&mut looped, "S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None);
        append(&mut looped, "S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0xfffe)], None);
        append(&mut looped, "S_ENDPGM", "ENC_SOPP", &[], None);
        let hlsl = validation_shader(&decode(&looped).unwrap(), validation_limits()).unwrap().hlsl;
        assert!(hlsl.contains("xcl_budget"));
        assert!(hlsl.contains("xcl_budget--"));
    }

    #[test]
    fn validation_shader_rejects_barriers_and_does_not_open_pso_eligibility() {
        let mut code = Vec::new();
        append(&mut code, "S_BARRIER", "ENC_SOPP", &[], None);
        append(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
        let program = decode(&code).unwrap();
        assert!(validation_shader(&program, validation_limits())
            .unwrap_err()
            .contains("uniformly reached"));

        let mut mixed = Vec::new();
        append(&mut mixed, "S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)], None);
        append(&mut mixed, "V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 4)], None);
        append(&mut mixed, "S_ENDPGM", "ENC_SOPP", &[], None);
        assert!(validation_shader(&decode(&mixed).unwrap(), validation_limits()).is_ok());
        assert!(super::mesh_control::lower(&mixed).is_err());
    }

    #[test]
    fn validation_shader_rejects_mutated_words_with_stale_decoded_state() {
        let mut original = Vec::new();
        append(&mut original, "S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)], None);
        append(&mut original, "S_ENDPGM", "ENC_SOPP", &[], None);

        let mut barrier_mutation = decode(&original).unwrap();
        let barrier = encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap();
        assert_eq!(barrier.len(), barrier_mutation.instructions[0].byte_len);
        barrier_mutation.instructions[0].words = barrier
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        assert!(validation_shader(&barrier_mutation, validation_limits())
            .unwrap_err()
            .contains("canonical decode"));

        let mut operand_mutation = decode(&original).unwrap();
        let changed = encode(
            "S_MOV_B32",
            "ENC_SOP1",
            &[("SDST", 5), ("SSRC0", 128)],
            None,
        )
        .unwrap();
        assert_eq!(changed.len(), operand_mutation.instructions[0].byte_len);
        operand_mutation.instructions[0].words = changed
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        assert!(validation_shader(&operand_mutation, validation_limits())
            .unwrap_err()
            .contains("canonical decode"));
    }
}
