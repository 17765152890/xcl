//! Aggregate-only coverage survey for decoded mesh programs.
//!
//! This analysis module is intentionally disconnected from PSO eligibility.

use std::collections::{BTreeMap, BTreeSet};

use super::mesh_alloc_export::AllocationExportOp;
use super::mesh_memory::ReadOnlyRegion;
use super::mesh_memory::{stage_c_format_supported, BufferFormatSource, StageCOp};
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

fn count_class(value: u8) -> &'static str {
    match value {
        0 => "none",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        8 => "eight",
        16 => "sixteen",
        _ => "other",
    }
}

fn offset_class(value: super::mesh_memory::OffsetSource) -> &'static str {
    match value {
        super::mesh_memory::OffsetSource::Zero => "zero",
        super::mesh_memory::OffsetSource::Scalar(_) => "scalar",
        super::mesh_memory::OffsetSource::Constant(_) => "constant",
    }
}

/// Static, aggregate-safe form attributes. Values such as offsets, registers,
/// addresses, descriptor contents, and instruction identity are never included.
fn memory_form(op: StageCOp) -> String {
    match op {
        StageCOp::DsSwizzle { control, .. } => format!("mode={}", if control & 0x8000 == 0 { "quad" } else { "bitmask" }),
        StageCOp::DsRead { words, .. } | StageCOp::DsWrite { words, .. } => format!("words={}", count_class(words)),
        StageCOp::ScalarLoad { words, soffset, .. } => format!("words={};offset={}", count_class(words), offset_class(soffset)),
        StageCOp::ScalarBufferLoad { words, soffset, .. } => format!("words={};descriptor=buffer;offset={}", count_class(words), offset_class(soffset)),
        StageCOp::VectorBufferLoad { words, soffset, offen, idxen, format, .. } => format!(
            "words={};format={};address={};offset={}",
            count_class(words),
            match format { BufferFormatSource::Raw => "raw", BufferFormatSource::Descriptor => "descriptor", BufferFormatSource::Instruction(_) => "instruction" },
            match (offen, idxen) { (false, false) => "base", (true, false) => "offset", (false, true) => "index", (true, true) => "index_offset" },
            offset_class(soffset),
        ),
        StageCOp::GlobalLoad { scalar_address, .. } => format!("address={}", if scalar_address.is_some() { "vector_scalar" } else { "vector" }),
        StageCOp::ImageLoad { coordinate_count, dmask, d16, .. } => format!("coordinates={};components={};precision={}", count_class(coordinate_count), count_class(dmask.count_ones() as u8), if d16 { "d16" } else { "d32" }),
        StageCOp::ImageSampleLz { coordinate_count, arrayed, dmask, d16, lod, .. } => format!("coordinates={};array={};components={};precision={};lod={}", count_class(coordinate_count), arrayed, count_class(dmask.count_ones() as u8), if d16 { "d16" } else { "d32" }, if lod.is_some() { "explicit" } else { "zero" }),
        StageCOp::ImageGather4Lz { component, d16, .. } => format!("component={};precision={}", if component < 4 { "color" } else { "other" }, if d16 { "d16" } else { "d32" }),
    }
}

fn memory_rejection_reason(op: StageCOp) -> &'static str {
    match op {
        StageCOp::DsRead { dst, words, .. } => {
            if !matches!(words, 1 | 2) { "word_count" }
            else if words == 2 && dst >= 255 { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::DsWrite { words, .. } => if matches!(words, 1 | 2) { "validation_policy" } else { "word_count" },
        StageCOp::DsSwizzle { .. } => "validation_policy",
        StageCOp::ScalarLoad { dst, base, words, .. } => {
            if !matches!(words, 1 | 4 | 8) { "word_count" }
            else if base >= 127 || dst.checked_add(words).is_none_or(|end| end > 106) { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::ScalarBufferLoad { dst, descriptor, words, .. } => {
            if !matches!(words, 1 | 2 | 4 | 8 | 16) { "word_count" }
            else if descriptor > 124 || dst.checked_add(words).is_none_or(|end| end > 106) { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::VectorBufferLoad { dst, address, descriptor, offset, offen, idxen, words, format, .. } => {
            let form_ok = match format {
                BufferFormatSource::Raw => (1..=4).contains(&words),
                BufferFormatSource::Descriptor => matches!(words, 1 | 2 | 3 | 4) && idxen && !offen && (words == 1 || offset == 0),
                BufferFormatSource::Instruction(value) => stage_c_format_supported(value) && (1..=4).contains(&words) && offen && !idxen,
            };
            if matches!(format, BufferFormatSource::Instruction(value) if !stage_c_format_supported(value)) { "format_class" }
            else if !matches!(words, 1..=4) { "word_count" }
            else if !form_ok { "unsupported_form" }
            else if dst.checked_add(u16::from(words)).is_none_or(|end| end > 256)
                || address.checked_add(u16::from(offen) + u16::from(idxen)).is_none_or(|end| end > 256)
                || descriptor > 124 { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::GlobalLoad { dst, address, scalar_address, .. } => {
            if dst >= 256 || address >= 255 || scalar_address.is_some_and(|register| register >= 127) { "register_span" } else { "validation_policy" }
        }
        StageCOp::ImageLoad { dst, coordinates, coordinate_count, descriptor, dmask, d16 } => {
            let components = dmask.count_ones() as u16;
            let outputs = if d16 { components.div_ceil(2) } else { components };
            if d16 && (coordinate_count != 1 || !matches!(components, 3 | 4)) { "precision_mode" }
            else if !(1..=3).contains(&coordinate_count) || dmask == 0 || dmask & !0xf != 0 { "component_shape" }
            else if dst.checked_add(outputs).is_none_or(|end| end > 256) || descriptor > 120
                || coordinates[..usize::from(coordinate_count)].iter().any(|&register| register >= 256) { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::ImageSampleLz { dst, coordinates, coordinate_count, arrayed, descriptor, sampler, dmask, d16, lod } => {
            let components = dmask.count_ones() as u16;
            let outputs = if d16 { components.div_ceil(2) } else { components };
            if d16 && (coordinate_count != 2 || arrayed || components != 3) { "precision_mode" }
            else if !(1..=3).contains(&coordinate_count) || (arrayed && coordinate_count != 3) || dmask == 0 || dmask & !0xf != 0 { "component_shape" }
            else if dst.checked_add(outputs).is_none_or(|end| end > 256) || descriptor > 120 || sampler > 124
                || coordinates[..usize::from(coordinate_count)].iter().any(|&register| register >= 256)
                || lod.is_some_and(|register| register >= 256) { "register_span" }
            else { "validation_policy" }
        }
        StageCOp::ImageGather4Lz { dst, coordinates, descriptor, sampler, component, d16 } => {
            if d16 { "precision_mode" }
            else if component > 3 { "component_shape" }
            else if dst.checked_add(4).is_none_or(|end| end > 256) || descriptor > 120 || sampler > 124
                || coordinates.iter().any(|&register| register >= 256) { "register_span" }
            else { "validation_policy" }
        }
    }
}

fn first_unlowered_memory(program: &MeshProgram) -> Option<String> {
    for instruction in &program.instructions {
        let lowered = match instruction.op {
            ProgramOp::Control(op) => super::mesh_control_slot::lower_control_slot(op, program.instructions.len() as u32).map(|_| ()),
            _ => super::mesh_slot::lower_slot(&instruction.op, super::mesh_slot::WaveStateAbi::default()).map(|_| ()),
        };
        if lowered.is_err() {
            return match instruction.op {
                ProgramOp::Memory(op) => {
                    let reason = memory_rejection_reason(op);
                    let shape = image_sample_shape_subcauses(op);
                    Some(match shape {
                        Some(shape) => format!("{}|{}|reason={reason};shape={shape}", memory_variant(op), memory_form(op)),
                        None => format!("{}|{}|reason={reason}", memory_variant(op), memory_form(op)),
                    })
                }
                _ => None,
            };
        }
    }
    None
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
    pub first_unlowered_memory: BTreeMap<String, usize>,
    pub other_rejection_details: BTreeMap<String, usize>,
    pub image_sample_shape_subcauses: BTreeMap<String, usize>,
}

impl ValidationShellCoverage {
    fn observe(&mut self, program: &MeshProgram) {
        self.attempted_programs += 1;
        if let Some(cause) = first_image_sample_shape_subcauses(program) {
            *self.image_sample_shape_subcauses.entry(cause).or_default() += 1;
        }
        if let Some(classification) = first_unlowered_memory(program) {
            *self.first_unlowered_memory.entry(classification).or_default() += 1;
        }
        let read_only = program
            .coverage
            .families
            .contains_key(&ProgramFamily::Memory)
            .then_some(ReadOnlyRegion {
                guest_base: 0,
                memory_offset: 0,
                byte_len: 4,
            });
        match super::mesh_epoch_hlsl::validation_epoch_hlsl(
            read_only,
            program,
            1,
            super::mesh_dispatch::MAX_CONTROL_STEPS,
        ) {
            Ok(_) => self.accepted_programs += 1,
            Err(error) => {
                let reason = classify_shell_rejection(&error);
                *self.rejection_reasons.entry(reason).or_default() += 1;
                if reason == "other_validation_rejection" {
                    let detail = classify_other_shell_rejection(program, &error);
                    *self.other_rejection_details.entry(detail).or_default() += 1;
                }
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

fn first_image_sample_shape_subcauses(program: &MeshProgram) -> Option<String> {
    first_validation_slot_rejection(program).and_then(|(op, _)| match *op {
        ProgramOp::Memory(memory) => image_sample_shape_subcauses(memory),
        _ => None,
    })
}

fn image_sample_shape_subcauses(op: StageCOp) -> Option<String> {
    let StageCOp::ImageSampleLz { dst, coordinates, coordinate_count, arrayed, descriptor, sampler, dmask, d16, lod } = op else { return None };
    let components = dmask.count_ones() as u16;
    let outputs = if d16 { components.div_ceil(2) } else { components };
    let mut causes = Vec::new();
    if !(1..=3).contains(&coordinate_count) { causes.push("coordinate_count"); }
    if arrayed && coordinate_count != 3 { causes.push("array_shape"); }
    if dmask == 0 { causes.push("component_mask_empty"); }
    if dmask & !0xf != 0 { causes.push("component_mask_out_of_range"); }
    if d16 && (coordinate_count != 2 || arrayed || components != 3) { causes.push("d16_shape"); }
    if outputs != 0 && !super::mesh_slot::vector_destination_span_fits(dst, outputs) { causes.push("destination_span"); }
    if descriptor > 120 { causes.push("descriptor_span"); }
    if sampler > 124 { causes.push("sampler_span"); }
    if (1..=3).contains(&coordinate_count)
        && coordinates[..usize::from(coordinate_count)].iter().any(|register| *register >= 256) { causes.push("coordinate_span"); }
    if lod.is_some_and(|register| register >= 256) { causes.push("lod_span"); }
    (!causes.is_empty()).then(|| causes.join("+"))
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

fn slot_rejection_reason(error: &str) -> &'static str {
    if error.contains("cross-lane lowering") {
        "cross_lane_source"
    } else if error.contains("carry input mask") || error.contains("carry output mask") {
        "carry_mask_form"
    } else if error.contains("compare kind") || error.contains("compare source") {
        "comparison_form"
    } else if error.contains("hardware register write") {
        "hardware_policy"
    } else if error.contains("selector") || error.contains("encoding") {
        "selector_form"
    } else if error.contains("destination") || error.contains("source") || error.contains("limit") || error.contains("exceeds") {
        "register_span"
    } else if error.contains("no validation per-slot lowering") || error.contains("unsupported") {
        "unsupported_form"
    } else {
        "validation_policy"
    }
}

fn op_form(op: &ProgramOp) -> String {
    match op {
        ProgramOp::Control(_) => "family=control;variant=control".into(),
        ProgramOp::Scalar(op) => format!("family=scalar;variant={}", scalar_variant(*op)),
        ProgramOp::Vector(op) => format!("family=vector;variant={}", vector_variant(*op)),
        ProgramOp::Memory(op) => format!("family=memory;variant={};{}", memory_variant(*op), memory_form(*op)),
        ProgramOp::AllocationExport(AllocationExportOp::Allocate { .. }) => "family=allocation_export;variant=allocate".into(),
        ProgramOp::AllocationExport(AllocationExportOp::Export(_)) => "family=allocation_export;variant=export".into(),
        ProgramOp::Unsupported(_) => "family=unsupported;variant=unsupported".into(),
    }
}

fn first_validation_slot_rejection(program: &MeshProgram) -> Option<(&ProgramOp, String)> {
    let abi = super::mesh_epoch_hlsl::validation_wave_abi();
    program.instructions.iter().find_map(|instruction| {
        let lowered = match instruction.op {
            ProgramOp::Control(op) => super::mesh_control_slot::lower_control_slot(op, program.instructions.len() as u32).map(|_| ()),
            _ => super::mesh_slot::lower_slot(&instruction.op, abi).map(|_| ()),
        };
        lowered.err().map(|error| (&instruction.op, error))
    })
}

fn classify_other_shell_rejection(program: &MeshProgram, error: &str) -> String {
    if error.contains("is not validation-lowerable") {
        if let Some((op, slot_error)) = first_validation_slot_rejection(program) {
            let reason = slot_rejection_reason(&slot_error);
            let span = match *op {
                ProgramOp::Memory(memory) => image_sample_shape_subcauses(memory),
                _ => None,
            };
            return match span {
                Some(span) => format!("stage=slot;{};reason={reason};shape={span}", op_form(op)),
                None => format!("stage=slot;{};reason={reason}", op_form(op)),
            };
        }
    }
    let reason = if error.contains("read-only memory program requires") {
        "missing_memory_region"
    } else if error.contains("read-only region is not dword aligned") {
        "memory_region_alignment"
    } else if error.contains("guest memory region overflows") || error.contains("host memory region overflows") {
        "memory_region_overflow"
    } else if error.contains("requires at least one instruction") {
        "empty_program"
    } else if error.contains("step cap") {
        "step_policy"
    } else {
        "validation_policy"
    };
    format!("stage=plan;reason={reason}")
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
                super::mesh_epoch_hlsl::validation_wave_abi(),
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
                "validation_shell attempted={} accepted={} rejected={} reasons={:?} first_unlowered_families={:?} first_unlowered_memory={:?} other_rejection_details={:?} image_sample_shape_subcauses={:?}",
                self.validation_shell.attempted_programs,
                self.validation_shell.accepted_programs,
                self.validation_shell.rejected_programs(),
                self.validation_shell.rejection_reasons,
                self.validation_shell.first_unlowered_families,
                self.validation_shell.first_unlowered_memory,
                self.validation_shell.other_rejection_details
                ,self.validation_shell.image_sample_shape_subcauses
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
    use crate::gpu::recomp::mesh_vector::VectorSource;
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
    fn other_shell_rejections_are_earliest_deterministic_and_redacted() {
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let template = mesh_program::decode(&bytes).unwrap();
        let rejected = |dst, register| {
            let mut program = template.clone();
            program.instructions[0].op = ProgramOp::Vector(StageBOp::Move {
                dst,
                src: VectorSource::Dpp16RowMirror { register, fetch_inactive: true },
            });
            program
        };
        let a = rejected(19, 37);
        let b = rejected(23, 41);
        let mut left = MeshCoverageSurvey::default();
        left.observe(&a);
        left.observe(&b);
        let mut right = MeshCoverageSurvey::default();
        right.observe(&b);
        right.observe(&a);
        assert_eq!(left.aggregate_report(), right.aggregate_report());
        assert_eq!(left.validation_shell.rejection_reasons["other_validation_rejection"], 2);
        assert_eq!(left.validation_shell.other_rejection_details["stage=slot;family=vector;variant=move;reason=cross_lane_source"], 2);
        assert_eq!(
            classify_other_shell_rejection(&template, "uniform epoch plan requires at least one instruction"),
            "stage=plan;reason=empty_program"
        );
        let report = left.aggregate_report();
        for forbidden in ["slot 0", "TEST", "0x", "19", "23", "37", "41", "\\", "/"] {
            assert!(!report.contains(forbidden), "report leaked {forbidden}: {report}");
        }

        let mut earlier = a;
        earlier.instructions.insert(0, template.instructions[0].clone());
        earlier.instructions[0].op = ProgramOp::Scalar(StageAOp::Jump(0));
        earlier.byte_len += earlier.instructions[0].byte_len;
        assert!(first_validation_slot_rejection(&earlier)
            .is_some_and(|(op, _)| matches!(op, ProgramOp::Scalar(StageAOp::Jump(_)))));
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
    fn first_unlowered_memory_forms_are_sorted_deterministic_and_redacted() {
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let template = mesh_program::decode(&bytes).unwrap();
        let rejected = [
            StageCOp::DsRead { dst: 19, address: 37, offsets: [0x1234, 0], words: 3 },
            StageCOp::VectorBufferLoad { dst: 23, address: 41, descriptor: 52, offset: 0x9876, soffset: OffsetSource::Scalar(77), offen: true, idxen: true, words: 1, format: BufferFormatSource::Descriptor },
            StageCOp::ImageLoad { dst: 29, coordinates: [31, 32, 33], coordinate_count: 3, descriptor: 60, dmask: 5, d16: true },
        ];
        let survey = |ops: &[StageCOp]| {
            let mut survey = MeshCoverageSurvey::default();
            for &op in ops {
                let mut program = template.clone();
                program.instructions[0].op = ProgramOp::Memory(op);
                survey.observe(&program);
            }
            survey
        };
        let left = survey(&rejected);
        let mut reversed = rejected;
        reversed.reverse();
        let right = survey(&reversed);
        assert_eq!(left.validation_shell.first_unlowered_memory, right.validation_shell.first_unlowered_memory);
        assert_eq!(left.validation_shell.first_unlowered_memory.len(), 3);
        assert_eq!(left.validation_shell.first_unlowered_memory["ds_read|words=three|reason=word_count"], 1);
        assert_eq!(left.validation_shell.first_unlowered_memory["vector_buffer_load|words=one;format=descriptor;address=index_offset;offset=scalar|reason=unsupported_form"], 1);
        assert_eq!(left.validation_shell.first_unlowered_memory["image_load|coordinates=three;components=two;precision=d16|reason=precision_mode"], 1);
        let report = left.aggregate_report();
        assert_eq!(report, right.aggregate_report());
        for forbidden in ["slot 0", "TEST", "0x", "1234", "9876", "19", "23", "29", "37", "41", "52", "60", "77", "\\", "/"] {
            assert!(!report.contains(forbidden), "report leaked {forbidden}: {report}");
        }
    }

    #[test]
    fn first_unlowered_memory_uses_exact_predicates_and_stops_at_prior_vector() {
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let template = mesh_program::decode(&bytes).unwrap();
        let classify = |op| {
            let mut program = template.clone();
            program.instructions[0].op = ProgramOp::Memory(op);
            let mut survey = MeshCoverageSurvey::default();
            survey.observe(&program);
            survey.validation_shell.first_unlowered_memory
        };
        let load_two = classify(StageCOp::ScalarLoad { dst: 0, base: 2, offset: 0, soffset: OffsetSource::Zero, words: 2 });
        assert_eq!(load_two["scalar_load|words=two;offset=zero|reason=word_count"], 1);
        let load_sixteen = classify(StageCOp::ScalarLoad { dst: 0, base: 2, offset: 0, soffset: OffsetSource::Zero, words: 16 });
        assert_eq!(load_sixteen["scalar_load|words=sixteen;offset=zero|reason=word_count"], 1);
        let span = classify(StageCOp::ScalarBufferLoad { dst: 100, descriptor: 4, offset: 0, soffset: OffsetSource::Zero, words: 16 });
        assert_eq!(span["scalar_buffer_load|words=sixteen;descriptor=buffer;offset=zero|reason=register_span"], 1);

        let mut prior_vector = template.clone();
        prior_vector.instructions[0].op = ProgramOp::Vector(StageBOp::ReadFirstLane { dst: 106, src: 0 });
        prior_vector.instructions[1].op = ProgramOp::Memory(StageCOp::DsRead { dst: 0, address: 0, offsets: [0, 0], words: 3 });
        let mut survey = MeshCoverageSurvey::default();
        survey.observe(&prior_vector);
        assert!(survey.validation_shell.first_unlowered_memory.is_empty());
    }

    #[test]
    fn image_sample_shape_subcauses_are_exhaustive_deterministic_and_redacted() {
        let mut bytes = code("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)]);
        bytes.extend(code("S_ENDPGM", "ENC_SOPP", &[]));
        let template = mesh_program::decode(&bytes).unwrap();
        let make = |dst, coordinates, coordinate_count, arrayed, descriptor, sampler, dmask, d16, lod| { let mut p=template.clone(); p.instructions[0].op=ProgramOp::Memory(StageCOp::ImageSampleLz {
            dst, coordinates, coordinate_count, arrayed, descriptor,
            sampler, dmask, d16, lod,
        }); p.coverage.families.insert(ProgramFamily::Memory,1); p };
        let cases = [
            (make(0,[0,1,0],0,false,8,24,7,false,None),"coordinate_count"),
            (make(0,[0,1,0],2,true,8,24,7,false,None),"array_shape"),
            (make(0,[0,1,0],2,false,8,24,0,false,None),"component_mask_empty"),
            (make(0,[0,1,0],2,false,8,24,16,false,None),"component_mask_out_of_range"),
            (make(0,[0,1,0],1,false,8,24,7,true,Some(2)),"d16_shape"),
            (make(255,[0,1,0],2,false,8,24,7,true,Some(2)),"destination_span"),
            (make(0,[0,1,0],2,false,121,24,7,true,Some(2)),"descriptor_span"),
            (make(0,[0,1,0],2,false,8,125,7,true,Some(2)),"sampler_span"),
            (make(0,[0,256,0],2,false,8,24,7,true,Some(2)),"coordinate_span"),
            (make(0,[0,1,0],2,false,8,24,7,true,Some(256)),"lod_span"),
            (make(255,[256,1,0],2,true,121,125,23,true,None),"array_shape+component_mask_out_of_range+d16_shape+destination_span+descriptor_span+sampler_span+coordinate_span"),
        ];
        let mut left=ValidationShellCoverage::default();
        for (program, expected) in &cases {
            left.observe(program);
            assert!(left.image_sample_shape_subcauses.contains_key(*expected),"missing {expected}: {:?}",left.image_sample_shape_subcauses);
        }
        let mut right=ValidationShellCoverage::default();
        for (program, _) in cases.iter().rev(){right.observe(program);}
        assert_eq!(left,right);
        assert_eq!(left.image_sample_shape_subcauses.values().sum::<usize>(),cases.len());
        let report=format!("{:?}",left.image_sample_shape_subcauses);
        for forbidden in ["255","256","121","125"] { assert!(!report.contains(forbidden),"leaked {forbidden}: {report}"); }

        let mut prior=cases[0].0.clone();
        prior.instructions.insert(0, mesh_program::ProgramInstruction { pc:0,byte_len:4,words:vec![0],literal:None,name:"TEST",encoding:"TEST",op:ProgramOp::Vector(StageBOp::ReadFirstLane{dst:106,src:0}) });
        for (index,instruction) in prior.instructions.iter_mut().enumerate(){instruction.pc=(index*4)as u64;}prior.byte_len+=4;
        let detail=classify_other_shell_rejection(&prior,"is not validation-lowerable");
        assert!(detail.contains("family=vector"));assert!(!detail.contains("shape="));
        assert!(first_image_sample_shape_subcauses(&prior).is_none());
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
