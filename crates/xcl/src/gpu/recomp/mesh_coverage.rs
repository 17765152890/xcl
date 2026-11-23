//! Aggregate-only coverage survey for decoded mesh programs.
//!
//! This analysis module is intentionally disconnected from PSO eligibility.

use std::collections::{BTreeMap, BTreeSet};

use super::mesh_memory::ReadOnlyRegion;
use super::mesh_memory::StageCOp;
use crate::gpu::rdna2;

use super::mesh_program::{self, MeshProgram, ProgramFamily, ProgramLimits, ProgramOp};
use super::mesh_scalar::StageAOp;
use super::mesh_vector::StageBOp;

fn scalar_variant(op: StageAOp) -> &'static str {
    match op {
        StageAOp::AddU32 { .. } => "add_u32",
        StageAOp::MulImmediateI32 { .. } => "mul_immediate_i32",
        StageAOp::GetPc64 { .. } => "get_pc64",
        StageAOp::BitSet1 { .. } => "bit_set_1",
        StageAOp::Not32 { .. } => "not_32",
        StageAOp::BitReverse32 { .. } => "bit_reverse_32",
        StageAOp::CountBits32 { .. } => "count_bits_32",
        StageAOp::CountBits64 { .. } => "count_bits_64",
        StageAOp::Binary { .. } => "binary",
        StageAOp::Binary64 { .. } => "binary_64",
        StageAOp::CompareEq { .. } => "compare_eq",
        StageAOp::CompareGtU32 { .. } => "compare_gt_u32",
        StageAOp::CompareLtU32 { .. } => "compare_lt_u32",
        StageAOp::CompareLeU32 { .. } => "compare_le_u32",
        StageAOp::CompareGeU32 { .. } => "compare_ge_u32",
        StageAOp::CompareNeU32 { .. } => "compare_ne_u32",
        StageAOp::CompareLtI32 { .. } => "compare_lt_i32",
        StageAOp::CompareGtI32 { .. } => "compare_gt_i32",
        StageAOp::CompareGeI32 { .. } => "compare_ge_i32",
        StageAOp::Select32 { .. } => "select_32",
        StageAOp::MoveImmediate { .. } => "move_immediate",
        StageAOp::Move32 { .. } => "move_32",
        StageAOp::Move64 { .. } => "move_64",
        StageAOp::SaveExecAnd64 { .. } => "save_exec_and_64",
        StageAOp::SaveExecOr64 { .. } => "save_exec_or_64",
        StageAOp::WrExecAndNot64 { .. } => "write_exec_and_not_64",
        StageAOp::ConfigureHardwareRegister { .. } => "configure_hardware_register",
        StageAOp::Select64 { .. } => "select_64",
        StageAOp::NoEffect => "no_effect",
        StageAOp::Barrier => "barrier",
        StageAOp::Jump(_) => "jump",
        StageAOp::Complete => "complete",
    }
}

fn memory_variant(op: StageCOp) -> &'static str {
    match op {
        StageCOp::DsSwizzle { .. } => "ds_swizzle",
        StageCOp::DsRead { .. } => "ds_read",
        StageCOp::DsWrite { .. } => "ds_write",
        StageCOp::ScalarLoad { .. } => "scalar_load",
        StageCOp::ScalarBufferLoad { .. } => "scalar_buffer_load",
        StageCOp::VectorBufferLoad { .. } => "vector_buffer_load",
        StageCOp::GlobalLoad { .. } => "global_load",
        StageCOp::ImageLoad { .. } => "image_load",
        StageCOp::ImageSampleLz { .. } => "image_sample_lz",
        StageCOp::ImageGather4Lz { .. } => "image_gather4_lz",
    }
}

fn vector_variant(op: StageBOp) -> &'static str {
    match op {
        StageBOp::Move { .. } => "move",
        StageBOp::RelativeMove { .. } => "relative_move",
        StageBOp::ReadFirstLane { .. } => "read_first_lane",
        StageBOp::ReadLane { .. } => "read_lane",
        StageBOp::WriteLane { .. } => "write_lane",
        StageBOp::PermLaneX16 { .. } => "perm_lane_x16",
        StageBOp::Mbcnt { .. } => "mbcnt",
        StageBOp::Binary { .. } => "binary",
        StageBOp::AddCarry { .. } => "add_carry",
        StageBOp::ReverseSubCarry { .. } => "reverse_sub_carry",
        StageBOp::Ternary { .. } => "ternary",
        StageBOp::BitReverse { .. } => "bit_reverse",
        StageBOp::BitNot { .. } => "bit_not",
        StageBOp::BitScan { .. } => "bit_scan",
        StageBOp::Compare { .. } => "compare",
        StageBOp::CompareScalar { .. } => "compare_scalar",
        StageBOp::CompareEqU64 { .. } => "compare_eq_u64",
        StageBOp::CompareClassF32 { .. } => "compare_class_f32",
        StageBOp::CompareLessFloat { .. } => "compare_less_float",
        StageBOp::CndMask { .. } => "cnd_mask",
        StageBOp::CndMaskScalar { .. } => "cnd_mask_scalar",
        StageBOp::ConvertF32U32 { .. } => "convert_f32_u32",
        StageBOp::ConvertF32I32 { .. } => "convert_f32_i32",
        StageBOp::ConvertF32U32Sdwa { .. } => "convert_f32_u32_sdwa",
        StageBOp::FloatConvert { .. } => "float_convert",
        StageBOp::FloatBinary { .. } => "float_binary",
        StageBOp::FloatTernary { .. } => "float_ternary",
        StageBOp::FloatUnary { .. } => "float_unary",
        StageBOp::FloatMath { .. } => "float_math",
        StageBOp::ConvertUByte { .. } => "convert_ubyte",
        StageBOp::PackF16 { .. } => "pack_f16",
        StageBOp::FmaMix { .. } => "fma_mix",
        StageBOp::FloatFma { .. } => "float_fma",
        StageBOp::FloatMad { .. } => "float_mad",
        StageBOp::StageDInteger { .. } => "stage_d_integer",
        StageBOp::NoEffect => "no_effect",
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationShellCoverage {
    pub attempted_programs: usize,
    pub accepted_programs: usize,
    pub rejection_reasons: BTreeMap<&'static str, usize>,
    pub first_unlowered_families: BTreeMap<ProgramFamily, usize>,
}

impl ValidationShellCoverage {
    fn observe(&mut self, program: &MeshProgram) {
        self.attempted_programs += 1;
        match super::mesh_epoch_hlsl::validation_epoch_hlsl(
            program,
            1,
            super::mesh_dispatch::MAX_CONTROL_STEPS,
        ) {
            Ok(_) => self.accepted_programs += 1,
            Err(error) => {
                *self
                    .rejection_reasons
                    .entry(classify_shell_rejection(&error))
                    .or_default() += 1;
                if let Some(family) = first_unlowered_family(program) {
                    *self.first_unlowered_families.entry(family).or_default() += 1;
                }
            }
        }
    }

    pub fn rejected_programs(&self) -> usize {
        self.attempted_programs - self.accepted_programs
    }
}

fn classify_shell_rejection(error: &str) -> &'static str {
    if error.contains("no validation per-slot lowering") {
        "unsupported_slot_lowering"
    } else if error.contains("shared state requires") {
        "shared_state_limit"
    } else if error.contains("register requirements exceed") {
        "register_limit"
    } else if error.contains("instruction boundaries") || error.contains("program byte length") {
        "invalid_structure"
    } else if error.contains("resource size") || error.contains("layout overflows") {
        "resource_plan_limit"
    } else {
        "other_validation_rejection"
    }
}

fn first_unlowered_family(program: &MeshProgram) -> Option<ProgramFamily> {
    program.instructions.iter().find_map(|instruction| {
        let lowered = match instruction.op {
            ProgramOp::Control(op) => {
                super::mesh_control_slot::lower_control_slot(op, program.instructions.len() as u32)
                    .map(|_| ())
            }
            _ => super::mesh_slot::lower_slot(
                &instruction.op,
                super::mesh_slot::WaveStateAbi::default(),
            )
            .map(|_| ()),
        };
        lowered.err().map(|_| instruction.op.family())
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshCoverageSurvey {
    pub total_programs: usize,
    pub decoded_programs: usize,
    pub structurally_validated_programs: usize,
    pub decode_failures: usize,
    pub unsupported_opcodes: BTreeMap<String, usize>,
    pub unsupported_forms: BTreeMap<String, usize>,
    pub family_instructions: BTreeMap<ProgramFamily, usize>,
    pub maximum_vgprs: u32,
    pub maximum_sgprs: u32,
    pub programs_using_read_only_memory: usize,
    pub unsupported_sets: BTreeMap<Vec<String>, usize>,
    pub validation_shell: ValidationShellCoverage,
    pub scalar_variants: BTreeMap<&'static str, usize>,
    pub vector_variants: BTreeMap<&'static str, usize>,
    pub memory_variants: BTreeMap<&'static str, usize>,
}

impl MeshCoverageSurvey {
    pub fn observe_decode_failure(&mut self) {
        self.total_programs += 1;
        self.decode_failures += 1;
    }

    pub fn observe(&mut self, program: &MeshProgram) {
        self.total_programs += 1;
        self.decoded_programs += 1;
        for (family, count) in &program.coverage.families {
            *self.family_instructions.entry(*family).or_default() += count;
        }

        let mut unsupported = BTreeSet::new();
        for instruction in &program.instructions {
            if let ProgramOp::Scalar(op) = instruction.op {
                *self.scalar_variants.entry(scalar_variant(op)).or_default() += 1;
            }
            if let ProgramOp::Vector(op) = instruction.op {
                *self.vector_variants.entry(vector_variant(op)).or_default() += 1;
            }
            if let ProgramOp::Memory(op) = instruction.op {
                *self.memory_variants.entry(memory_variant(op)).or_default() += 1;
            }
            if matches!(instruction.op, ProgramOp::Unsupported(_)) {
                *self
                    .unsupported_opcodes
                    .entry(instruction.name.to_owned())
                    .or_default() += 1;
                *self
                    .unsupported_forms
                    .entry(format!("{} [{}]", instruction.name, instruction.encoding))
                    .or_default() += 1;
                unsupported.insert(instruction.name.to_owned());
            }
        }
        *self
            .unsupported_sets
            .entry(unsupported.into_iter().collect())
            .or_default() += 1;

        for instruction in &program.instructions {
            let bytes: Vec<u8> = instruction
                .words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            if let Ok(decoded) = rdna2::decode(&bytes, instruction.pc) {
                let (sgpr_inputs, sgpr_outputs) = rdna2::sgpr_uses(&decoded);
                for register in sgpr_inputs.into_iter().chain(sgpr_outputs) {
                    if register < super::mesh_dispatch::SCALAR_REGISTERS as u64
                        && !matches!(register, 106 | 107 | 125 | 126 | 127)
                    {
                        self.maximum_sgprs = self.maximum_sgprs.max(register as u32 + 1);
                    }
                }
                let (vgpr_inputs, vgpr_outputs) = rdna2::vgpr_uses(&decoded);
                for register in vgpr_inputs.into_iter().chain(vgpr_outputs) {
                    self.maximum_vgprs = self.maximum_vgprs.max(register as u32 + 1);
                }
            }
        }
        self.programs_using_read_only_memory += usize::from(
            program
                .coverage
                .families
                .contains_key(&ProgramFamily::Memory),
        );

        if let Ok(requirements) = mesh_program::requirements(program) {
            self.maximum_vgprs = self.maximum_vgprs.max(requirements.vgprs);
            self.maximum_sgprs = self.maximum_sgprs.max(requirements.sgprs);
            let limits = ProgramLimits {
                vgprs: requirements.vgprs.max(1),
                sgprs: requirements.sgprs.max(1),
                records: 1,
                step_cap: super::mesh_dispatch::MAX_CONTROL_STEPS,
                read_only: requirements
                    .uses_read_only_memory
                    .then_some(ReadOnlyRegion {
                        guest_base: 0,
                        memory_offset: 0,
                        byte_len: 4,
                    }),
                image_descriptors: &[],
                sampler_descriptors: &[],
            };
            if mesh_program::validate_structure(program, limits).is_ok() {
                self.structurally_validated_programs += 1;
                self.validation_shell.observe(program);
            }
        }
    }

    pub fn first_material_family(&self) -> Option<(&[String], usize)> {
        self.unsupported_sets
            .iter()
            .filter(|(set, _)| !set.is_empty())
            .max_by(|(left_set, left_count), (right_set, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| right_set.cmp(left_set))
            })
            .map(|(set, count)| (set.as_slice(), *count))
    }

    pub fn aggregate_report(&self) -> String {
        let mut lines = vec![
            format!(
                "programs total={} decoded={} structurally_validated={} decode_failures={}",
                self.total_programs,
                self.decoded_programs,
                self.structurally_validated_programs,
                self.decode_failures
            ),
            format!(
                "requirements max_vgprs={} max_sgprs={} read_only_programs={}",
                self.maximum_vgprs, self.maximum_sgprs, self.programs_using_read_only_memory
            ),
            format!("families {:?}", self.family_instructions),
            format!("unsupported_opcodes {:?}", self.unsupported_opcodes),
            format!("unsupported_forms {:?}", self.unsupported_forms),
            format!("unsupported_sets {:?}", self.unsupported_sets),
            format!("scalar_variants {:?}", self.scalar_variants),
            format!("vector_variants {:?}", self.vector_variants),
            format!("memory_variants {:?}", self.memory_variants),
            format!(
                "validation_shell attempted={} accepted={} rejected={} reasons={:?} first_unlowered_families={:?}",
                self.validation_shell.attempted_programs,
                self.validation_shell.accepted_programs,
                self.validation_shell.rejected_programs(),
                self.validation_shell.rejection_reasons,
                self.validation_shell.first_unlowered_families
            ),
        ];
        if let Some((set, count)) = self.first_material_family() {
            lines.push(format!(
                "first_material_family count={count} opcodes={set:?}"
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::mesh_memory::{BufferFormatSource, OffsetSource};
    use crate::gpu::recomp::mesh_scalar::StageAOp;
    use crate::gpu::{pack, rdna2::encode, xbox};

    fn code(name: &str, encoding: &str, fields: &[(&str, u64)]) -> Vec<u8> {
        encode(name, encoding, fields, None).unwrap()
    }

    #[test]
    fn aggregate_is_deterministic_and_contains_no_program_identity() {
        let mut supported_code = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)]);
        supported_code.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let supported = mesh_program::decode(&supported_code).unwrap();
        let unsupported_a =
            mesh_program::decode(&code("S_TRAP", "ENC_SOPP", &[("SIMM16", 1)])).unwrap();
        let unsupported_b =
            mesh_program::decode(&code("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 0)])).unwrap();
        let mut survey = MeshCoverageSurvey::default();
        survey.observe(&supported);
        survey.observe(&unsupported_b);
        survey.observe(&unsupported_a);
        survey.observe(&unsupported_a);
        survey.observe_decode_failure();

        assert_eq!(survey.total_programs, 5);
        assert_eq!(survey.structurally_validated_programs, 1);
        assert_eq!(survey.maximum_sgprs, 11);
        assert_eq!(survey.unsupported_opcodes["S_TRAP"], 2);
        assert!(survey.scalar_variants.is_empty());
        assert!(survey.memory_variants.is_empty());
        assert_eq!(
            survey.first_material_family(),
            Some((&["S_TRAP".to_owned()][..], 2))
        );
        let report = survey.aggregate_report();
        assert!(report.contains("scalar_variants {}"));
        assert!(report.contains("memory_variants {}"));
        assert!(report.contains("S_TRAP [ENC_SOPP]"));
        assert!(!report.contains("0x"));
        assert!(!report.contains('\\'));
    }

    #[test]
    fn validation_shell_aggregate_is_deterministic_and_redacted() {
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let accepted = mesh_program::decode(&bytes).unwrap();
        let mut rejected = accepted.clone();
        // Scalar Jump is intentionally scheduler-ineligible: decoded branches
        // must use ProgramOp::Control so rendezvous cannot be bypassed.
        rejected.instructions[0].op = ProgramOp::Scalar(StageAOp::Jump(0));
        let mut barrier = accepted.clone();
        barrier.instructions[0].op = ProgramOp::Scalar(StageAOp::Barrier);

        let mut left = MeshCoverageSurvey::default();
        left.observe(&accepted);
        left.observe(&rejected);
        left.observe(&barrier);
        let mut right = MeshCoverageSurvey::default();
        right.observe(&barrier);
        right.observe(&rejected);
        right.observe(&accepted);

        assert_eq!(left.aggregate_report(), right.aggregate_report());
        assert_eq!(left.validation_shell.attempted_programs, 3);
        assert_eq!(left.validation_shell.accepted_programs, 2);
        assert_eq!(left.scalar_variants["barrier"], 1);
        assert_eq!(left.scalar_variants["jump"], 1);
        assert_eq!(left.scalar_variants.len(), 2);
        assert_eq!(
            left.validation_shell.rejection_reasons["unsupported_slot_lowering"],
            1
        );
        assert_eq!(
            left.validation_shell.first_unlowered_families[&ProgramFamily::Scalar],
            1
        );
        let report = left.aggregate_report();
        for forbidden in ["slot 0", "TEST", "0x", "\\", "/"] {
            assert!(!report.contains(forbidden));
        }
    }

    #[test]
    fn memory_variant_aggregate_is_exhaustive_deterministic_and_redacted() {
        let variants = [
            StageCOp::DsSwizzle { dst: 0, source: 1, control: 0 },
            StageCOp::DsRead { dst: 0, address: 1, offsets: [0, 0], words: 1 },
            StageCOp::DsWrite { address: 0, data: [1, 0], offsets: [0, 0], words: 1 },
            StageCOp::ScalarLoad { dst: 0, base: 2, offset: 0, soffset: OffsetSource::Zero, words: 1 },
            StageCOp::ScalarBufferLoad { dst: 0, descriptor: 4, offset: 0, soffset: OffsetSource::Zero, words: 1 },
            StageCOp::VectorBufferLoad { dst: 0, address: 1, descriptor: 4, offset: 0, soffset: OffsetSource::Zero, offen: false, idxen: false, words: 1, format: BufferFormatSource::Raw },
            StageCOp::GlobalLoad { dst: 0, address: 2, scalar_address: None, offset: 0 },
            StageCOp::ImageLoad { dst: 0, coordinates: [1, 0, 0], coordinate_count: 1, descriptor: 4, dmask: 1, d16: false },
            StageCOp::ImageSampleLz { dst: 0, coordinates: [1, 2, 0], coordinate_count: 2, arrayed: false, descriptor: 4, sampler: 8, dmask: 1, d16: false, lod: None },
            StageCOp::ImageGather4Lz { dst: 0, coordinates: [1, 2], descriptor: 4, sampler: 8, component: 0, d16: false },
        ];
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let template = mesh_program::decode(&bytes).unwrap();
        let mut left = MeshCoverageSurvey::default();
        for op in variants {
            let mut program = template.clone();
            program.instructions[0].op = ProgramOp::Memory(op);
            left.observe(&program);
        }
        let mut right = MeshCoverageSurvey::default();
        for op in variants.into_iter().rev() {
            let mut program = template.clone();
            program.instructions[0].op = ProgramOp::Memory(op);
            right.observe(&program);
        }
        assert_eq!(left.memory_variants, right.memory_variants);
        assert_eq!(left.memory_variants.len(), variants.len());
        assert!(variants.into_iter().all(|op| left.memory_variants[memory_variant(op)] == 1));
        let report = left.aggregate_report();
        assert!(report.contains("memory_variants"));
        for forbidden in ["slot 0", "TEST", "0x", "\\", "/"] {
            assert!(!report.contains(forbidden));
        }
    }

    #[test]
    #[ignore]
    fn private_mesh_program_aggregate_survey() {
        let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut survey = MeshCoverageSurvey::default();
        for record in pack::scan(&data, 0x40) {
            let Some(blob) = pack::inflate(&data, &record) else {
                continue;
            };
            let Some(shader) = xbox::parse(&blob) else {
                continue;
            };
            for fssm in shader
                .fssm
                .iter()
                .filter(|fssm| fssm.stages.first().is_some_and(|stage| stage.hw == 7))
            {
                for program in &fssm.programs {
                    let code = &fssm.data[program.offset..program.offset + program.len];
                    match mesh_program::decode(code) {
                        Ok(program) => survey.observe(&program),
                        Err(_) => survey.observe_decode_failure(),
                    }
                }
            }
        }
        assert!(survey.total_programs > 0);
        println!("{}", survey.aggregate_report());
    }
}
