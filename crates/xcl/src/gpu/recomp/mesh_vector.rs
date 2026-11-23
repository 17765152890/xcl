//! Synthetic decoded vector/mask execution over checked external VGPR scratch.
//!
//! This module is not connected to real-program eligibility or PSO selection.

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_dispatch::{LogicalWaveState, GROUP_THREADS, LOGICAL_WAVE_LANES};
use super::mesh_vgpr::{load_vgpr, store_vgpr, VgprScratchLayout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSource {
    Vgpr(u16),
    Scalar(u8),
    Constant(u32),
    Sdwa {
        register: u8,
        scalar: bool,
        select: u8,
        sign_extend: bool,
    },
    Dpp16RowMirror {
        register: u8,
        fetch_inactive: bool,
    },
    Dpp8 {
        register: u8,
        lanes: [u8; 8],
        fetch_inactive: bool,
    },
    VccZero,
    ExecZero,
    Scc,
}

#[cfg(test)]
mod carry_tests {
    use super::*;

    fn init(layout: VgprScratchLayout, exec: [u64; 2]) -> (Vec<u32>, [LogicalWaveState; 2]) {
        let mut memory = vec![0u32; layout.scratch.total_bytes as usize / 4];
        for thread in 0..128u32 { for register in 0..layout.vgpr_count {
            store_vgpr(&mut memory, layout, 0, thread, register, initial_vgpr_value(register, thread)).unwrap();
        }}
        let states = std::array::from_fn(|wave| LogicalWaveState { exec: [exec[wave] as u32, (exec[wave] >> 32) as u32], ..Default::default() });
        (memory, states)
    }

    #[test]
    fn decoded_relative_move_uses_uniform_m0_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;

        let bytes = encode(
            "V_MOVRELS_B32",
            "ENC_VOP1",
            &[("VDST", 10), ("SRC0", 276)],
            None,
        ).unwrap();
        assert_eq!(bytes, [0x14, 0x87, 0x14, 0x7e]);
        let op = lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap();
        assert_eq!(op, StageBOp::RelativeMove { dst: 10, base: 20 });

        let layout = VgprScratchLayout::new(256, 1).unwrap();
        let exec = [0x8000_0000_0000_0001, 1u64 << 17];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        expected_states[0].scalar[124] = 3;
        expected_states[1].scalar[124] = 500;
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &[op]).unwrap();
        for thread in 0..128u32 {
            let wave = (thread / 64) as usize;
            let lane = thread & 63;
            let active = exec[wave] & (1u64 << lane) != 0;
            let expected = if active {
                if wave == 0 { initial_vgpr_value(23, thread) } else { 0 }
            } else {
                initial_vgpr_value(10, thread)
            };
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 10).unwrap(), expected);
        }

        let hlsl = stage_b_validation_hlsl_with_vcc_and_m0(layout, &[op], exec, [0; 2], [3, 500]).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute, &mesh, &pixel, 1, record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &[], 1, [0.0; 4], layout.scratch,
        ).unwrap();
        for thread in 0..128u32 {
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 10).unwrap(),
                load_vgpr(&expected_memory, layout, 0, thread, 10).unwrap(),
            );
        }
    }

    #[test]
    fn decoded_residual_half_forms_match_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;

        let observed_convert = [0xf9, 0x14, 0x0c, 0x7e, 0x06, 0x05, 0x06, 0x00];
        let observed_half_cmp = [0x80, 0x60, 0x92, 0x7d];
        let observed_u32_cmp = [0x81, 0x24, 0x82, 0x7d];
        assert!(matches!(
            lower_stage_b_one(&rdna2::decode(&observed_convert, 0).unwrap()).unwrap(),
            StageBOp::FloatConvert { kind: FloatConvertKind::F16FromF32HighPad, .. }
        ));
        assert!(matches!(
            lower_stage_b_one(&rdna2::decode(&observed_half_cmp, 0).unwrap()).unwrap(),
            StageBOp::Compare { kind: CompareKind::LessFloat16, .. }
        ));
        assert!(matches!(
            lower_stage_b_one(&rdna2::decode(&observed_u32_cmp, 0).unwrap()).unwrap(),
            StageBOp::Compare { kind: CompareKind::Less, .. }
        ));

        let convert = encode("V_CVT_F16_F32", "VOP1_VOP_SDWA", &[
            ("VDST", 1), ("SRC0", 249), ("VSRC0", 0), ("SRC0_SEL", 6),
            ("DST_SEL", 5), ("DST_UNUSED", 0),
        ], None).unwrap();
        let half_cmp = encode("V_CMP_LT_F16", "ENC_VOP3", &[
            ("VDST", 4), ("SRC0", 256), ("SRC1", 258),
        ], None).unwrap();
        let u32_cmp = encode("V_CMP_LT_U32", "ENC_VOPC", &[
            ("SRC0", 129), ("VSRC1", 2),
        ], None).unwrap();
        let mut ops = vec![
            StageBOp::Move { dst: 0, src: VectorSource::Constant(1.5f32.to_bits()) },
            StageBOp::Move { dst: 1, src: VectorSource::Constant(0x3c00) },
            StageBOp::Move { dst: 2, src: VectorSource::Constant(2) },
        ];
        ops.extend([convert.as_slice(), half_cmp.as_slice(), u32_cmp.as_slice()]
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap()));

        let layout = VgprScratchLayout::new(3, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 31) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            assert_eq!(scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(4)), exec[wave]);
            assert_eq!(mask(expected_states[wave].vcc), exec[wave]);
        }
        for thread in 0..128u32 {
            let active = ((exec[(thread / 64) as usize] >> (thread & 63)) & 1) != 0;
            let expected = if active { 0x3e00_0000 } else { initial_vgpr_value(1, thread) };
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 1).unwrap(), expected);
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for wave in 0..2u32 {
            let got = super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave).unwrap();
            assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(4)), exec[wave as usize]);
            assert_eq!(mask(got.vcc), exec[wave as usize]);
        }
        for thread in 0..128u32 {
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 1).unwrap(),
                load_vgpr(&expected_memory, layout, 0, thread, 1).unwrap()
            );
        }
    }

    #[test]
    fn decoded_float_classes_match_all_ieee_classes_and_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;

        let encoded = encode(
            "V_CMP_CLASS_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 256), ("SRC1", 255)],
            Some(1u32 << 9),
        ).unwrap();
        assert_eq!(encoded, [0x04, 0x00, 0x88, 0xd4, 0x00, 0xff, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00]);
        let decoded = lower_stage_b_one(&rdna2::decode(&encoded, 0).unwrap()).unwrap();
        assert_eq!(decoded, StageBOp::CompareClassF32 {
            value: VectorSource::Vgpr(0),
            classes: VectorSource::Constant(1 << 9),
            dst: Some(4),
            modifiers: FloatModifiers::default(),
        });

        let raw_classes = [
            0x7f80_0001, // signaling NaN
            0x7fc0_0001, // quiet NaN
            0xff80_0000, // -infinity
            0xbf80_0000, // negative normal
            0x8000_0001, // negative subnormal
            0x8000_0000, // -zero
            0x0000_0000, // +zero
            0x0000_0001, // positive subnormal
            0x3f80_0000, // positive normal
            0x7f80_0000, // +infinity
        ];
        let ops = std::array::from_fn::<_, 10, _>(|index| StageBOp::CompareClassF32 {
            value: VectorSource::Constant(raw_classes[index]),
            classes: VectorSource::Constant(1u32 << index),
            dst: Some((index * 2) as u8),
            modifiers: FloatModifiers::default(),
        });
        assert_eq!(raw_classes.map(f32_class_index), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let layout = VgprScratchLayout::new(1, 1).unwrap();
        let exec = [u64::MAX, 0x8000_0000_0000_0001];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            for destination in (0..20).step_by(2) {
                assert_eq!(scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(destination)), exec[wave]);
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for wave in 0..2u32 {
            let got = super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave).unwrap();
            for destination in (0..20).step_by(2) {
                assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(destination)), exec[wave as usize]);
            }
        }
    }

    #[test]
    fn decoded_lane_reads_and_writes_ignore_exec_and_match_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;

        let read_0 = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 256), ("SRC1", 128)], None).unwrap();
        let read_31 = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 256), ("SRC1", 159)], None).unwrap();
        let read_32 = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 256), ("SRC1", 160)], None).unwrap();
        let read_127 = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", 256), ("SRC1", 255)], Some(127)).unwrap();
        let write_127 = encode("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 255), ("SRC1", 191)], Some(0xdead_beef)).unwrap();
        assert_eq!(read_0, [0x04, 0x00, 0x60, 0xd7, 0x00, 0x01, 0x01, 0x00]);
        assert_eq!(read_31, [0x05, 0x00, 0x60, 0xd7, 0x00, 0x3f, 0x01, 0x00]);
        assert_eq!(read_32, [0x06, 0x00, 0x60, 0xd7, 0x00, 0x41, 0x01, 0x00]);
        assert_eq!(read_127, [0x07, 0x00, 0x60, 0xd7, 0x00, 0xff, 0x01, 0x00, 0x7f, 0x00, 0x00, 0x00]);
        assert_eq!(write_127, [0x01, 0x00, 0x61, 0xd7, 0xff, 0x7e, 0x01, 0x00, 0xef, 0xbe, 0xad, 0xde]);

        let ops: Vec<_> = [&read_0[..], &read_31, &read_32, &read_127, &write_127]
            .into_iter()
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap())
            .collect();
        assert_eq!(ops[0], StageBOp::ReadLane { dst: 4, src: 0, lane: VectorSource::Constant(0) });
        assert_eq!(ops[3], StageBOp::ReadLane { dst: 7, src: 0, lane: VectorSource::Constant(127) });
        assert_eq!(ops[4], StageBOp::WriteLane { dst: 1, src: VectorSource::Constant(0xdead_beef), lane: VectorSource::Constant(63) });

        let invalid_read = encode("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 4), ("SRC1", 128)], None).unwrap();
        let invalid_write = encode("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 256), ("SRC1", 128)], None).unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_read, 0).unwrap()).is_err());
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_write, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(2, 1).unwrap();
        let exec = [1, 1u64 << 17];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2usize {
            assert_eq!(&expected_states[wave].scalar[4..8], &[0, 31, 32, 63]);
            for lane in 0..64u32 {
                let thread = wave as u32 * 64 + lane;
                let actual = load_vgpr(&expected_memory, layout, 0, thread, 1).unwrap();
                let expected = if lane == 63 { 0xdead_beef } else { initial_vgpr_value(1, thread) };
                assert_eq!(actual, expected, "wave {wave} lane {lane}");
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for wave in 0..2u32 {
            let got = super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave).unwrap();
            assert_eq!(&got.scalar[4..8], &[0, 31, 32, 63]);
            for lane in 0..64u32 {
                let thread = wave * 64 + lane;
                let actual = load_vgpr(&memory, layout, 0, thread, 1).unwrap();
                let expected = if lane == 63 { 0xdead_beef } else { initial_vgpr_value(1, thread) };
                assert_eq!(actual, expected, "gpu wave {wave} lane {lane}");
            }
        }
    }

    #[test]
    fn decoded_permlanex16_crosses_rows_and_honors_fetch_controls() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;

        let observed = encode(
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[("VDST", 1), ("SRC0", 256), ("SRC1", 255), ("SRC2", 3), ("OP_SEL", 1)],
            Some(0x7654_3210),
        ).unwrap();
        assert_eq!(observed, [0x01, 0x08, 0x78, 0xd7, 0x00, 0xff, 0x0d, 0x00, 0x10, 0x32, 0x54, 0x76]);
        assert!(matches!(
            lower_stage_b_one(&rdna2::decode(&observed, 0).unwrap()).unwrap(),
            StageBOp::PermLaneX16 { selectors_low: VectorSource::Constant(0x7654_3210), selectors_high: VectorSource::Scalar(3), .. }
        ));

        let make = |dst, controls| encode(
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[("VDST", dst), ("SRC0", 256), ("SRC1", 128), ("SRC2", 128), ("OP_SEL", controls)],
            None,
        ).unwrap();
        let fetch = make(1, 1);
        let bound = make(2, 2);
        let preserve = make(3, 0);
        let ops = [fetch.as_slice(), bound.as_slice(), preserve.as_slice()]
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap());
        assert!(matches!(ops[0], StageBOp::PermLaneX16 { fetch_inactive: true, bound_control: false, .. }));
        assert!(matches!(ops[1], StageBOp::PermLaneX16 { fetch_inactive: false, bound_control: true, .. }));

        let reserved = make(1, 4);
        assert!(lower_stage_b_one(&rdna2::decode(&reserved, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let exec = [0x0000_0000_0000_ffff, 0x0000_0000_0000_ffff];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2u32 {
            for lane in 0..64u32 {
                let thread = wave * 64 + lane;
                let active = lane < 16;
                let selected = wave * 64 + (lane & !31) + ((lane & 16) ^ 16);
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 1).unwrap(),
                    if active { initial_vgpr_value(0, selected) } else { initial_vgpr_value(1, thread) },
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                    if active { 0 } else { initial_vgpr_value(2, thread) },
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 3).unwrap(),
                    initial_vgpr_value(3, thread),
                );
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute, &mesh, &pixel, 1, record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &[], 1, [0.0; 4], layout.scratch,
        ).unwrap();
        for thread in 0..128u32 {
            for register in 1..=3u32 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    "thread {thread} register {register}",
                );
            }
        }
    }

    #[test]
    fn decoded_carry_adds_snapshot_masks_and_match_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;
        let add = encode("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 2), ("SDST", 4), ("SRC0", 256), ("SRC1", 193)], None).unwrap();
        let add_ci = encode("V_ADD_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 3), ("SDST", 6), ("SRC0", 256), ("SRC1", 193), ("SRC2", 4)], None).unwrap();
        let vop2 = encode("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 193), ("VSRC1", 0)], None).unwrap();
        assert_eq!(add, [0x02, 0x04, 0x0f, 0xd7, 0x00, 0x83, 0x01, 0x00]);
        assert_eq!(add_ci, [0x03, 0x06, 0x28, 0xd5, 0x00, 0x83, 0x11, 0x00]);
        assert_eq!(vop2, [0xc1, 0x00, 0x02, 0x50]);
        let ops = [add.as_slice(), add_ci.as_slice(), vop2.as_slice()].map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap());
        assert_eq!(ops[0], StageBOp::AddCarry { dst: 2, lhs: VectorSource::Vgpr(0), rhs: VectorSource::Constant(u32::MAX), carry_in: None, carry_out: ScalarMask::ScalarPair(4) });
        assert_eq!(ops[1], StageBOp::AddCarry { dst: 3, lhs: VectorSource::Vgpr(0), rhs: VectorSource::Constant(u32::MAX), carry_in: Some(ScalarMask::ScalarPair(4)), carry_out: ScalarMask::ScalarPair(6) });
        assert_eq!(ops[2], StageBOp::AddCarry { dst: 1, lhs: VectorSource::Constant(u32::MAX), rhs: VectorSource::Vgpr(0), carry_in: Some(ScalarMask::Vcc), carry_out: ScalarMask::Vcc });
        let modified = encode("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 2), ("SDST", 4), ("SRC0", 256), ("SRC1", 193), ("NEG", 1)], None).unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified, 0).unwrap()).is_err());
        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 1) | (1u64 << 31) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 { let carry = exec[wave] & !1;
            assert_eq!(scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(4)), carry);
            assert_eq!(scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(6)), carry);
            assert_eq!(mask(expected_states[wave].vcc), carry);
        }
        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for wave in 0..2 { let got = super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave as u32).unwrap(); let carry = exec[wave] & !1;
            assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(4)), carry);
            assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(6)), carry);
            assert_eq!(mask(got.vcc), carry);
        }
        for thread in 0..128 { for register in [1, 2, 3] {
            assert_eq!(load_vgpr(&memory, layout, 0, thread, register).unwrap(), load_vgpr(&expected_memory, layout, 0, thread, register).unwrap());
        }}
    }

    #[test]
    fn decoded_reverse_subtract_carry_tracks_borrow_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;
        let vop2 = encode("V_SUBREV_CO_CI_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 0)], None).unwrap();
        let vop3 = encode("V_SUBREV_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 3), ("SDST", 4), ("SRC0", 129), ("SRC1", 256), ("SRC2", 106)], None).unwrap();
        assert_eq!(vop2, [0x81, 0x00, 0x04, 0x54]);
        assert_eq!(vop3, [0x03, 0x04, 0x2a, 0xd5, 0x81, 0x00, 0xaa, 0x01]);
        let ops = [vop2.as_slice(), vop3.as_slice()].map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap());
        assert_eq!(ops[0], StageBOp::ReverseSubCarry { dst: 2, lhs: VectorSource::Constant(1), rhs: VectorSource::Vgpr(0), carry_in: ScalarMask::Vcc, carry_out: ScalarMask::Vcc });
        assert_eq!(ops[1], StageBOp::ReverseSubCarry { dst: 3, lhs: VectorSource::Constant(1), rhs: VectorSource::Vgpr(0), carry_in: ScalarMask::Vcc, carry_out: ScalarMask::ScalarPair(4) });
        let modified = encode("V_SUBREV_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 3), ("SDST", 4), ("SRC0", 129), ("SRC1", 256), ("SRC2", 106), ("NEG", 1)], None).unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let exec = [0x8000_0000_8000_0007, 0x8000_0000_0000_0007];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        for state in &mut expected_states { state.vcc = [5, 0]; }
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            assert_eq!(mask(expected_states[wave].vcc), 1);
            assert_eq!(scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(4)), 1);
            for lane in 0..64u32 {
                let thread = wave as u32 * 64 + lane;
                for dst in [2, 3] {
                    let actual = load_vgpr(&expected_memory, layout, 0, thread, dst).unwrap();
                    let expected = if exec[wave] & (1u64 << lane) != 0 {
                        let carry = if dst == 2 { lane == 0 || lane == 2 } else { lane == 0 };
                        lane.wrapping_sub(1).wrapping_sub(carry as u32)
                    } else { initial_vgpr_value(dst, thread) };
                    assert_eq!(actual, expected, "cpu wave {wave} lane {lane} dst {dst}");
                }
            }
        }
        let hlsl = stage_b_validation_hlsl_with_vcc(layout, &ops, exec, [5, 5]).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for wave in 0..2u32 {
            let got = super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave).unwrap();
            assert_eq!(mask(got.vcc), 1);
            assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(4)), 1);
            for lane in 0..64u32 { let thread = wave * 64 + lane; for dst in [2, 3] {
                assert_eq!(load_vgpr(&memory, layout, 0, thread, dst).unwrap(), load_vgpr(&expected_memory, layout, 0, thread, dst).unwrap());
            }}
        }
    }

    #[test]
    fn decoded_shift_or_masks_counts_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;
        let encodings = [(2u16, 128u64), (3, 159), (4, 160), (5, 191)].map(
            |(dst, shift)| {
                encode(
                    "V_LSHL_OR_B32",
                    "ENC_VOP3",
                    &[
                        ("VDST", dst as u64),
                        ("SRC0", 256),
                        ("SRC1", shift),
                        ("SRC2", 129),
                    ],
                    None,
                )
                .unwrap()
            },
        );
        assert_eq!(
            encodings[1],
            [0x03, 0x00, 0x6f, 0xd7, 0x00, 0x3f, 0x05, 0x02]
        );
        let ops = encodings
            .iter()
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            ops[1],
            StageBOp::Ternary {
                kind: VectorTernaryKind::LshlOr,
                dst: 3,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Constant(31),
                c: VectorSource::Constant(1),
            }
        );
        let modified = encode(
            "V_LSHL_OR_B32",
            "ENC_VOP3",
            &[
                ("VDST", 2),
                ("SRC0", 256),
                ("SRC1", 129),
                ("SRC2", 128),
                ("NEG", 1),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(6, 1).unwrap();
        let exec = [u64::MAX, 0x8000_0000_0000_0001];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2u32 {
            for lane in 0..64u32 {
                let thread = wave * 64 + lane;
                let source = initial_vgpr_value(0, thread);
                for (dst, shift) in [(2, 0), (3, 31), (4, 32), (5, 63)] {
                    let before = initial_vgpr_value(dst, thread);
                    let expected = if exec[wave as usize] & (1u64 << lane) != 0 {
                        (source << (shift & 31)) | 1
                    } else {
                        before
                    };
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, dst).unwrap(),
                        expected
                    );
                }
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            for register in 2..=5 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap()
                );
            }
        }
    }

    #[test]
    fn decoded_residual_bit_ops_match_independent_values_and_actual_u6() {
        if super::super::dxc::location().is_err() { return; }
        use crate::gpu::rdna2::encode;
        let bfi = encode("V_BFI_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 193), ("SRC2", 194)], None).unwrap();
        let xad = encode("V_XAD_U32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 256), ("SRC1", 193), ("SRC2", 129)], None).unwrap();
        let not = encode("V_NOT_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 256)], None).unwrap();
        let ffbh = encode("V_FFBH_U32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 256)], None).unwrap();
        let ffbl = encode("V_FFBL_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 256)], None).unwrap();
        let max = encode("V_MAX_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 256), ("VSRC1", 0)], None).unwrap();
        let ffbh_zero = encode("V_FFBH_U32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None).unwrap();
        let ffbl_zero = encode("V_FFBL_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None).unwrap();
        assert_eq!(bfi, [0x02, 0x00, 0x4a, 0xd5, 0x00, 0x83, 0x09, 0x03]);
        assert_eq!(xad, [0x03, 0x00, 0x45, 0xd7, 0x00, 0x83, 0x05, 0x02]);
        assert_eq!(not, [0x00, 0x6f, 0x08, 0x7e]);
        assert_eq!(ffbh, [0x00, 0x73, 0x0a, 0x7e]);
        assert_eq!(ffbl, [0x00, 0x75, 0x0c, 0x7e]);
        assert_eq!(max, [0x00, 0x01, 0x0e, 0x28]);
        let ops = [bfi.as_slice(), xad.as_slice(), not.as_slice(), ffbh.as_slice(), ffbl.as_slice(), max.as_slice(), ffbh_zero.as_slice(), ffbl_zero.as_slice()].map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap());
        assert_eq!(ops[0], StageBOp::Ternary { kind: VectorTernaryKind::BitFieldInsert, dst: 2, a: VectorSource::Vgpr(0), b: VectorSource::Constant(u32::MAX), c: VectorSource::Constant(0xffff_fffe) });
        assert_eq!(ops[1], StageBOp::Ternary { kind: VectorTernaryKind::XorAdd, dst: 3, a: VectorSource::Vgpr(0), b: VectorSource::Constant(u32::MAX), c: VectorSource::Constant(1) });
        assert_eq!(ops[2], StageBOp::BitNot { dst: 4, src: VectorSource::Vgpr(0) });
        assert_eq!(ops[3], StageBOp::BitScan { high: true, dst: 5, src: VectorSource::Vgpr(0) });
        assert_eq!(ops[4], StageBOp::BitScan { high: false, dst: 6, src: VectorSource::Vgpr(0) });
        assert_eq!(ops[5], StageBOp::Binary { kind: VectorBinaryKind::MaxU32, dst: 7, lhs: VectorSource::Vgpr(0), rhs: VectorSource::Vgpr(0) });
        let modified = encode("V_BFI_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 193), ("SRC2", 194), ("NEG", 1)], None).unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(10, 1).unwrap();
        let exec = [u64::MAX, 0x8000_0000_0000_0001];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2u32 { for lane in 0..64u32 {
            let thread = wave * 64 + lane;
            if exec[wave as usize] & (1u64 << lane) != 0 {
                let a = initial_vgpr_value(0, thread);
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(), (a & u32::MAX) | (!a & 0xffff_fffe));
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 3).unwrap(), (a ^ u32::MAX).wrapping_add(1));
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 4).unwrap(), !a);
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 5).unwrap(), if a == 0 { u32::MAX } else { a.leading_zeros() });
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 6).unwrap(), if a == 0 { u32::MAX } else { a.trailing_zeros() });
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 7).unwrap(), a);
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 8).unwrap(), u32::MAX);
                assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 9).unwrap(), u32::MAX);
            }
        }}
        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
        let record = super::super::MsInterface { threads: GROUP_THREADS, max_vertices: 1, max_primitives: 1, ..Default::default() };
        let mesh = super::super::dxc::compile(&super::super::mesh_passthrough(&record), "main", "ms_6_6", &[]).unwrap();
        let pixel = super::super::dxc::compile("float4 main():SV_Target0{return 0.0.xxxx;}", "main", "ps_6_6", &[]).unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(&compute, &mesh, &pixel, 1, record.record_dwords(), &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4], &[], 1, [0.0; 4], layout.scratch).unwrap();
        for thread in 0..128 { for register in 2..=9 {
            assert_eq!(load_vgpr(&memory, layout, 0, thread, register).unwrap(), load_vgpr(&expected_memory, layout, 0, thread, register).unwrap());
        }}
    }

    #[test]
    fn decoded_median3_orders_values_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;
        let encoded = encode(
            "V_MED3_F32",
            "ENC_VOP3",
            &[("VDST", 2), ("SRC0", 256), ("SRC1", 242), ("SRC2", 244)],
            None,
        )
        .unwrap();
        assert_eq!(encoded, [0x02, 0x00, 0x57, 0xd5, 0x00, 0xe5, 0xd1, 0x03]);
        let zero_encoded = encode(
            "V_MED3_F32",
            "ENC_VOP3",
            &[("VDST", 3), ("SRC0", 255), ("SRC1", 128), ("SRC2", 242)],
            Some((-0.0f32).to_bits()),
        )
        .unwrap();
        let nan_encoded = encode(
            "V_MED3_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 255), ("SRC1", 243), ("SRC2", 244)],
            Some(f32::NAN.to_bits()),
        )
        .unwrap();
        let ops = [
            encoded.as_slice(),
            zero_encoded.as_slice(),
            nan_encoded.as_slice(),
        ]
        .map(|bytes| {
            lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap()
        });
        assert_eq!(
            ops[0],
            StageBOp::FloatTernary {
                kind: FloatTernaryKind::Median3,
                dst: 2,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Constant(1.0f32.to_bits()),
                c: VectorSource::Constant(2.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            }
        );
        let modified = encode(
            "V_MED3_F32",
            "ENC_VOP3",
            &[
                ("VDST", 2),
                ("SRC0", 256),
                ("SRC1", 242),
                ("SRC2", 244),
                ("OP_SEL", 1),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(5, 1).unwrap();
        let exec = [u64::MAX, u64::MAX];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        let inputs = [3.0f32, -4.0, 1.5, f32::NAN, -0.0, 0.0];
        for thread in 0..128u32 {
            store_vgpr(
                &mut expected_memory,
                layout,
                0,
                thread,
                0,
                inputs[thread as usize % inputs.len()].to_bits(),
            )
            .unwrap();
        }
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        let expected_medians = [2.0f32, 1.0, 1.5, 1.0, 1.0, 1.0];
        let expected_zero_medians = [(-0.0f32).to_bits(); 6];
        for lane in 0..inputs.len() as u32 {
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, lane, 2).unwrap(),
                expected_medians[lane as usize].to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, lane, 3).unwrap(),
                expected_zero_medians[lane as usize]
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, lane, 4).unwrap(),
                (-1.0f32).to_bits()
            );
        }

        let (mut gpu_expected, mut gpu_states) = init(layout, exec);
        execute_stage_b(&mut gpu_expected, layout, &mut gpu_states, &ops).unwrap();

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            for register in 2..=4 {
                let expected = load_vgpr(&gpu_expected, layout, 0, thread, register).unwrap();
                let actual = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                if f32::from_bits(expected).is_nan() {
                    assert!(f32::from_bits(actual).is_nan());
                } else {
                    assert_eq!(actual, expected, "thread {thread} register {register}");
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorBinaryKind {
    AddNoCarry,
    And,
    Or,
    Xor,
    BitCount,
    LshlReverse,
    ArithmeticShiftRightReverse,
    MinU32,
    MaxU32,
    MulLoU32,
    MulHiU32,
    SubNoCarry,
    ReverseSubNoCarry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorTernaryKind {
    Add3,
    BfeU32,
    LshlAdd,
    AddLshl,
    AndOr,
    Or3,
    LshlOr,
    BitFieldInsert,
    XorAdd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTernaryKind {
    Minimum3,
    Maximum3,
    Median3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareKind {
    Eq,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    SignedGreater,
    SignedLess,
    EqualFloat,
    OrderedNotEqualFloat,
    UnorderedNotEqualFloat,
    LessFloat,
    GreaterFloat,
    LessEqualFloat,
    GreaterEqualFloat,
    UnorderedGreaterFloat,
    UnorderedLessFloat,
    UnorderedLessEqualFloat,
    LessFloat16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarMask {
    ScalarPair(u8),
    Vcc,
    Exec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdwaWord {
    Low,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatConvertKind {
    F16FromF32,
    F16FromF32High,
    F16FromF32HighPad,
    F32FromF16,
    F32FromF16High,
    U32FromF32,
    I32FromF32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageBOp {
    Move {
        dst: u16,
        src: VectorSource,
    },
    RelativeMove {
        dst: u16,
        base: u16,
    },
    ReadFirstLane {
        dst: u8,
        src: u16,
    },
    ReadLane {
        dst: u8,
        src: u16,
        lane: VectorSource,
    },
    WriteLane {
        dst: u16,
        src: VectorSource,
        lane: VectorSource,
    },
    PermLaneX16 {
        dst: u16,
        src: u16,
        selectors_low: VectorSource,
        selectors_high: VectorSource,
        fetch_inactive: bool,
        bound_control: bool,
    },
    Mbcnt {
        high: bool,
        dst: u16,
        source: VectorSource,
        accumulator: VectorSource,
    },
    Binary {
        kind: VectorBinaryKind,
        dst: u16,
        lhs: VectorSource,
        rhs: VectorSource,
    },
    AddCarry {
        dst: u16,
        lhs: VectorSource,
        rhs: VectorSource,
        carry_in: Option<ScalarMask>,
        carry_out: ScalarMask,
    },
    ReverseSubCarry {
        dst: u16,
        lhs: VectorSource,
        rhs: VectorSource,
        carry_in: ScalarMask,
        carry_out: ScalarMask,
    },
    Ternary {
        kind: VectorTernaryKind,
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: VectorSource,
    },
    BitReverse {
        dst: u16,
        src: VectorSource,
    },
    BitNot {
        dst: u16,
        src: VectorSource,
    },
    BitScan {
        high: bool,
        dst: u16,
        src: VectorSource,
    },
    Compare {
        kind: CompareKind,
        lhs: VectorSource,
        rhs: VectorSource,
        cmpx: bool,
    },
    CompareScalar {
        kind: CompareKind,
        lhs: VectorSource,
        rhs: VectorSource,
        dst: u8,
    },
    CompareEqU64 {
        lhs: VectorSource,
        rhs: VectorSource,
        dst: Option<u8>,
    },
    CompareClassF32 {
        value: VectorSource,
        classes: VectorSource,
        dst: Option<u8>,
        modifiers: FloatModifiers,
    },
    CompareLessFloat {
        kind: CompareKind,
        lhs: VectorSource,
        rhs: VectorSource,
        dst: Option<u8>,
        modifiers: FloatModifiers,
        cmpx: bool,
    },
    CndMask {
        dst: u16,
        when_clear: VectorSource,
        when_set: VectorSource,
    },
    CndMaskScalar {
        dst: u16,
        when_clear: VectorSource,
        when_set: VectorSource,
        predicate: ScalarMask,
    },
    ConvertF32U32 {
        dst: u16,
        src: VectorSource,
        modifiers: FloatModifiers,
    },
    ConvertF32I32 {
        dst: u16,
        src: VectorSource,
    },
    ConvertF32U32Sdwa {
        dst: u16,
        src: u16,
        word: SdwaWord,
    },
    FloatConvert {
        kind: FloatConvertKind,
        dst: u16,
        src: VectorSource,
    },
    FloatBinary {
        kind: FloatBinaryKind,
        dst: u16,
        lhs: VectorSource,
        rhs: VectorSource,
        modifiers: FloatModifiers,
    },
    FloatTernary {
        kind: FloatTernaryKind,
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: VectorSource,
        modifiers: FloatModifiers,
    },
    FloatUnary {
        kind: FloatUnaryKind,
        dst: u16,
        src: VectorSource,
    },
    FloatMath {
        kind: FloatMathKind,
        dst: u16,
        src: VectorSource,
        modifiers: FloatModifiers,
    },
    ConvertUByte {
        dst: u16,
        src: VectorSource,
        byte: u8,
        modifiers: FloatModifiers,
    },
    PackF16 {
        dst: u16,
        lo: VectorSource,
        hi: VectorSource,
        op_sel: u8,
    },
    FmaMix {
        kind: FmaMixKind,
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: VectorSource,
        op_sel: u8,
        op_sel_hi: u8,
        neg: u8,
        abs: u8,
        clamp: bool,
    },
    FloatFma {
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: Option<VectorSource>,
        modifiers: FloatModifiers,
        legacy_zero: bool,
    },
    FloatMad {
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: Option<VectorSource>,
        modifiers: FloatModifiers,
    },
    StageDInteger {
        kind: StageDIntegerKind,
        dst: u16,
        a: VectorSource,
        b: VectorSource,
        c: Option<VectorSource>,
    },
    NoEffect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmaMixKind {
    F32,
    LoF16,
    HiF16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FloatModifiers {
    pub abs: u8,
    pub neg: u8,
    pub omod: u8,
    pub clamp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatBinaryKind {
    Add,
    Subtract,
    ReverseSubtract,
    Multiply,
    MultiplyLegacy,
    Maximum,
    Minimum,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatUnaryKind {
    Truncate,
    Floor,
    Ceiling,
    RoundEven,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatMathKind {
    Reciprocal,
    ReciprocalIFlag,
    ReciprocalSqrt,
    Sqrt,
    Sin,
    Cos,
    Exp2,
    Fract,
    Log2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageDIntegerKind {
    LogicalShiftRight,
    MadU24,
    Permute,
}

fn field(decoded: &Decoded, name: &str) -> Result<u64, String> {
    rdna2::field_value(decoded.enc, &decoded.words, name)
        .ok_or_else(|| format!("{} lacks {name}", decoded.inst.name))
}

fn inline(decoded: &Decoded, selector: u64) -> Option<u32> {
    match selector {
        125 | 128 => Some(0),
        129..=192 => Some((selector - 128) as u32),
        193..=208 => Some((-(selector as i32 - 192)) as u32),
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

fn source(decoded: &Decoded, selector: u64) -> Result<VectorSource, String> {
    match selector {
        0..=124 | 126 | 127 => Ok(VectorSource::Scalar(selector as u8)),
        256..=511 => Ok(VectorSource::Vgpr((selector - 256) as u16)),
        251 => Ok(VectorSource::VccZero),
        252 => Ok(VectorSource::ExecZero),
        253 => Ok(VectorSource::Scc),
        _ => inline(decoded, selector)
            .map(VectorSource::Constant)
            .ok_or_else(|| format!("unsupported vector source selector {selector}")),
    }
}

fn u64_source(decoded: &Decoded, selector: u64) -> Result<VectorSource, String> {
    match source(decoded, selector)? {
        VectorSource::Vgpr(register) if register < 255 => Ok(VectorSource::Vgpr(register)),
        VectorSource::Scalar(register) if register <= 104 => Ok(VectorSource::Scalar(register)),
        _ => Err(format!(
            "unsupported paired vector source selector {selector}"
        )),
    }
}

fn dst(decoded: &Decoded) -> Result<u16, String> {
    u16::try_from(field(decoded, "VDST")?).map_err(|_| "VGPR destination overflow".into())
}

fn vop2(decoded: &Decoded) -> Result<(VectorSource, VectorSource), String> {
    Ok((
        source(decoded, field(decoded, "SRC0")?)?,
        VectorSource::Vgpr(field(decoded, "VSRC1")? as u16),
    ))
}

fn vop3_sources(decoded: &Decoded) -> Result<[VectorSource; 3], String> {
    Ok([
        source(decoded, field(decoded, "SRC0")?)?,
        source(decoded, field(decoded, "SRC1")?)?,
        source(decoded, field(decoded, "SRC2")?)?,
    ])
}

fn float_modifiers(decoded: &Decoded) -> Result<FloatModifiers, String> {
    if decoded.enc.name != "ENC_VOP3" {
        return Ok(FloatModifiers::default());
    }
    let get = |name| rdna2::field_value(decoded.enc, &decoded.words, name).unwrap_or(0);
    Ok(FloatModifiers {
        abs: get("ABS") as u8,
        neg: get("NEG") as u8,
        omod: get("OMOD") as u8,
        clamp: get("CLAMP") != 0,
    })
}

fn float_compare_modifiers(decoded: &Decoded) -> Result<FloatModifiers, String> {
    let modifiers = float_modifiers(decoded)?;
    if modifiers.omod != 0
        || modifiers.clamp
        || rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0
    {
        return Err(format!(
            "{} uses unsupported float compare modifiers",
            decoded.inst.name
        ));
    }
    Ok(modifiers)
}

fn reject_integer_modifiers(decoded: &Decoded) -> Result<(), String> {
    if matches!(decoded.enc.name, "ENC_VOP3" | "VOP3_SDST_ENC")
        && ["ABS", "NEG", "OMOD", "CLAMP", "OP_SEL"]
            .into_iter()
            .any(|name| rdna2::field_value(decoded.enc, &decoded.words, name).unwrap_or(0) != 0)
    {
        return Err(format!(
            "{} uses unsupported integer VOP3 modifiers",
            decoded.inst.name
        ));
    }
    Ok(())
}

fn sdwa_encoding(decoded: &Decoded) -> Result<&'static rdna2::Encoding, String> {
    let name = match decoded.enc.name {
        "ENC_VOP2" => "VOP2_VOP_SDWA",
        "ENC_VOPC" => "VOPC_VOP_SDWA_SDST_ENC",
        _ => return Err("unsupported SDWA base encoding".into()),
    };
    rdna2::encoding(name).ok_or_else(|| format!("missing {name} encoding"))
}

fn sdwa_source_pair(decoded: &Decoded) -> Result<(VectorSource, VectorSource), String> {
    let encoding = sdwa_encoding(decoded)?;
    let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
    if ["SRC0_ABS", "SRC0_NEG", "SRC1_ABS", "SRC1_NEG"]
        .into_iter()
        .any(|name| ext(name) != 0)
    {
        return Err(format!(
            "{} uses unsupported SDWA modifiers",
            decoded.inst.name
        ));
    }
    let make = |index: usize| -> Result<VectorSource, String> {
        let register = u8::try_from(ext(if index == 0 { "VSRC0" } else { "VSRC1" }))
            .map_err(|_| "SDWA source register overflow")?;
        let select = u8::try_from(ext(if index == 0 { "SRC0_SEL" } else { "SRC1_SEL" }))
            .map_err(|_| "SDWA source selector overflow")?;
        if select > 6 {
            return Err("unsupported SDWA source selector".into());
        }
        Ok(VectorSource::Sdwa {
            register,
            scalar: ext(if index == 0 { "S0" } else { "S1" }) != 0,
            select,
            sign_extend: ext(if index == 0 { "SRC0_SEXT" } else { "SRC1_SEXT" }) != 0,
        })
    };
    Ok((make(0)?, make(1)?))
}

fn dpp_integer_sources(decoded: &Decoded) -> Result<(VectorSource, VectorSource), String> {
    let rhs = VectorSource::Vgpr(
        u16::try_from(field(decoded, "VSRC1")?).map_err(|_| "DPP RHS register overflow")?,
    );
    match decoded.variant {
        rdna2::Variant::Dpp16 => {
            let encoding = rdna2::encoding("VOP2_VOP_DPP16")
                .ok_or_else(|| "missing VOP2 DPP16 encoding".to_owned())?;
            let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
            if ext("DPP_CTRL") != 0x140
                || ext("ROW_MASK") != 0xf
                || ext("BANK_MASK") != 0xf
                || ext("BOUND_CTRL") != 1
                || ext("SRC0_ABS") != 0
                || ext("SRC0_NEG") != 0
                || ext("SRC1_ABS") != 0
                || ext("SRC1_NEG") != 0
            {
                return Err(format!("unsupported {} DPP16 controls", decoded.inst.name));
            }
            Ok((
                VectorSource::Dpp16RowMirror {
                    register: ext("VSRC0") as u8,
                    fetch_inactive: ext("FI") != 0,
                },
                rhs,
            ))
        }
        rdna2::Variant::Dpp8 => {
            let encoding = rdna2::encoding("VOP2_VOP_DPP8")
                .ok_or_else(|| "missing VOP2 DPP8 encoding".to_owned())?;
            let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
            let lane_fields = [
                "LANE_SEL_0",
                "LANE_SEL_1",
                "LANE_SEL_2",
                "LANE_SEL_3",
                "LANE_SEL_4",
                "LANE_SEL_5",
                "LANE_SEL_6",
                "LANE_SEL_7",
            ];
            let lanes = std::array::from_fn(|index| ext(lane_fields[index]) as u8);
            Ok((
                VectorSource::Dpp8 {
                    register: ext("VSRC0") as u8,
                    lanes,
                    fetch_inactive: field(decoded, "SRC0")? == 0xea,
                },
                rhs,
            ))
        }
        _ => Err(format!("unsupported {} DPP variant", decoded.inst.name)),
    }
}

fn scalar_mask(selector: u64) -> Result<ScalarMask, String> {
    match selector {
        0..=104 if selector % 2 == 0 => Ok(ScalarMask::ScalarPair(selector as u8)),
        106 => Ok(ScalarMask::Vcc),
        126 => Ok(ScalarMask::Exec),
        _ => Err(format!("unsupported scalar mask selector {selector}")),
    }
}

fn reject_cndmask_modifiers(decoded: &Decoded) -> Result<(), String> {
    if ["ABS", "OMOD", "CLAMP", "OP_SEL"]
        .into_iter()
        .any(|name| rdna2::field_value(decoded.enc, &decoded.words, name).unwrap_or(0) != 0)
    {
        return Err("V_CNDMASK_B32 uses unsupported VOP3 modifiers".into());
    }
    // NEG is ignored for this integer bit-select, matching the native VALU path.
    Ok(())
}

fn lower_sdwa_convert(decoded: &Decoded) -> Result<StageBOp, String> {
    let encoding =
        rdna2::encoding("VOP1_VOP_SDWA").ok_or_else(|| "missing VOP1 SDWA encoding".to_owned())?;
    let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
    if ext("S0") != 0
        || ext("SRC0_SEXT") != 0
        || ext("SRC0_NEG") != 0
        || ext("SRC0_ABS") != 0
        || ext("DST_SEL") != 6
        || ext("DST_UNUSED") != 0
        || ext("CLAMP") != 0
        || ext("OMOD") != 0
    {
        return Err("unsupported V_CVT_F32_U32 SDWA modifiers".into());
    }
    let word = match ext("SRC0_SEL") {
        4 => SdwaWord::Low,
        5 => SdwaWord::High,
        _ => return Err("unsupported V_CVT_F32_U32 SDWA selector".into()),
    };
    Ok(StageBOp::ConvertF32U32Sdwa {
        dst: dst(decoded)?,
        src: u16::try_from(ext("VSRC0")).map_err(|_| "SDWA VGPR source overflow")?,
        word,
    })
}

fn lower_sdwa_half_convert(decoded: &Decoded) -> Result<StageBOp, String> {
    let encoding =
        rdna2::encoding("VOP1_VOP_SDWA").ok_or_else(|| "missing VOP1 SDWA encoding".to_owned())?;
    let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
    if ext("S1") != 0
        || ext("SRC0_SEXT") != 0
        || ext("SRC0_NEG") != 0
        || ext("SRC0_ABS") != 0
        || ext("SRC1_SEXT") != 0
        || ext("SRC1_NEG") != 0
        || ext("SRC1_ABS") != 0
        || ext("CLAMP") != 0
        || ext("OMOD") != 0
    {
        return Err(format!(
            "{} uses unsupported SDWA modifiers",
            decoded.inst.name
        ));
    }
    let kind = match (
        decoded.inst.name,
        ext("SRC0_SEL"),
        ext("DST_SEL"),
        ext("DST_UNUSED"),
        ext("S0"),
    ) {
        ("V_CVT_F16_F32", 6, 4, 2, 0 | 1) => FloatConvertKind::F16FromF32,
        ("V_CVT_F16_F32", 6, 5, 2, 0 | 1) => FloatConvertKind::F16FromF32High,
        ("V_CVT_F16_F32", 6, 5, 0, 0 | 1) => FloatConvertKind::F16FromF32HighPad,
        ("V_CVT_F32_F16", 5, 6, 0, 0) => FloatConvertKind::F32FromF16High,
        _ => return Err(format!("unsupported {} SDWA selectors", decoded.inst.name)),
    };
    let selector = ext("VSRC0");
    let src = if ext("S0") == 0 {
        VectorSource::Vgpr(u16::try_from(selector).map_err(|_| "SDWA VGPR source overflow")?)
    } else {
        source(decoded, selector)?
    };
    Ok(StageBOp::FloatConvert {
        kind,
        dst: dst(decoded)?,
        src,
    })
}

fn lower_sdwa_ubyte0(decoded: &Decoded) -> Result<StageBOp, String> {
    let encoding =
        rdna2::encoding("VOP1_VOP_SDWA").ok_or_else(|| "missing VOP1 SDWA encoding".to_owned())?;
    let ext = |name| rdna2::field_value(encoding, &decoded.words, name).unwrap_or(0);
    if ext("S0") != 0
        || ext("SRC0_SEXT") != 0
        || ext("SRC0_NEG") != 0
        || ext("SRC0_ABS") != 0
        || ext("SRC0_SEL") > 3
        || ext("DST_SEL") != 6
        || ext("DST_UNUSED") != 0
        || ext("CLAMP") != 0
        || ext("OMOD") != 0
    {
        return Err("unsupported V_CVT_F32_UBYTE0 SDWA modifiers".into());
    }
    Ok(StageBOp::ConvertUByte {
        dst: dst(decoded)?,
        src: VectorSource::Vgpr(
            u16::try_from(ext("VSRC0")).map_err(|_| "SDWA VGPR source overflow")?,
        ),
        byte: ext("SRC0_SEL") as u8,
        modifiers: FloatModifiers::default(),
    })
}

pub fn lower_stage_b_one(decoded: &Decoded) -> Result<StageBOp, String> {
    let name = decoded.inst.name;
    if name == "V_NOP" {
        return Ok(StageBOp::NoEffect);
    }
    if name == "V_MOV_B32" && matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3") {
        reject_integer_modifiers(decoded)?;
        return Ok(StageBOp::Move {
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
        });
    }
    if name == "V_MOVRELS_B32" {
        if decoded.enc.name != "ENC_VOP1" || decoded.variant != rdna2::Variant::Plain {
            return Err("unsupported V_MOVRELS_B32 form".into());
        }
        reject_integer_modifiers(decoded)?;
        let VectorSource::Vgpr(base) = source(decoded, field(decoded, "SRC0")?)? else {
            return Err("V_MOVRELS_B32 source is not a VGPR".into());
        };
        return Ok(StageBOp::RelativeMove { dst: dst(decoded)?, base });
    }
    if name == "V_READFIRSTLANE_B32" && matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3") {
        reject_integer_modifiers(decoded)?;
        let destination = u8::try_from(field(decoded, "VDST")?)
            .map_err(|_| "readfirstlane scalar destination overflow")?;
        if destination >= 106 {
            return Err("readfirstlane special scalar destination is unsupported".into());
        }
        let VectorSource::Vgpr(src) = source(decoded, field(decoded, "SRC0")?)? else {
            return Err("readfirstlane source is not a VGPR".into());
        };
        return Ok(StageBOp::ReadFirstLane {
            dst: destination,
            src,
        });
    }
    if matches!(name, "V_READLANE_B32" | "V_WRITELANE_B32") {
        if decoded.enc.name != "ENC_VOP3" || decoded.variant != rdna2::Variant::Plain {
            return Err(format!("unsupported Stage B lane operation form {name}"));
        }
        reject_integer_modifiers(decoded)?;
        let lane = source(decoded, field(decoded, "SRC1")?)?;
        if !matches!(lane, VectorSource::Scalar(0..=105) | VectorSource::Constant(_)) {
            return Err("lane selector is not an ordinary scalar or constant".into());
        }
        if name == "V_READLANE_B32" {
            let destination = u8::try_from(field(decoded, "VDST")?)
                .map_err(|_| "readlane scalar destination overflow")?;
            if destination >= 106 {
                return Err("readlane special scalar destination is unsupported".into());
            }
            let VectorSource::Vgpr(src) = source(decoded, field(decoded, "SRC0")?)? else {
                return Err("readlane source is not a VGPR".into());
            };
            return Ok(StageBOp::ReadLane { dst: destination, src, lane });
        }
        let src = source(decoded, field(decoded, "SRC0")?)?;
        if !matches!(src, VectorSource::Scalar(0..=105) | VectorSource::Constant(_)) {
            return Err("writelane source is not an ordinary scalar or constant".into());
        }
        return Ok(StageBOp::WriteLane { dst: dst(decoded)?, src, lane });
    }
    if name == "V_PERMLANEX16_B32" {
        if decoded.enc.name != "ENC_VOP3" || decoded.variant != rdna2::Variant::Plain {
            return Err("unsupported V_PERMLANEX16_B32 form".into());
        }
        if ["ABS", "NEG", "OMOD", "CLAMP"]
            .into_iter()
            .any(|name| rdna2::field_value(decoded.enc, &decoded.words, name).unwrap_or(0) != 0)
        {
            return Err("V_PERMLANEX16_B32 uses unsupported modifiers".into());
        }
        let controls = field(decoded, "OP_SEL")?;
        if controls & !3 != 0 {
            return Err("V_PERMLANEX16_B32 uses reserved controls".into());
        }
        let VectorSource::Vgpr(src) = source(decoded, field(decoded, "SRC0")?)? else {
            return Err("V_PERMLANEX16_B32 data source is not a VGPR".into());
        };
        let selectors_low = source(decoded, field(decoded, "SRC1")?)?;
        let selectors_high = source(decoded, field(decoded, "SRC2")?)?;
        if !matches!(selectors_low, VectorSource::Scalar(0..=127) | VectorSource::Constant(_))
            || !matches!(selectors_high, VectorSource::Scalar(0..=127) | VectorSource::Constant(_))
        {
            return Err("V_PERMLANEX16_B32 selectors are not scalar values".into());
        }
        return Ok(StageBOp::PermLaneX16 {
            dst: dst(decoded)?,
            src,
            selectors_low,
            selectors_high,
            fetch_inactive: controls & 1 != 0,
            bound_control: controls & 2 != 0,
        });
    }
    if matches!(name, "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32") && decoded.enc.name == "ENC_VOP3"
    {
        reject_integer_modifiers(decoded)?;
        let values = vop3_sources(decoded)?;
        return Ok(StageBOp::Mbcnt {
            high: name.contains("_HI_"),
            dst: dst(decoded)?,
            source: values[0],
            accumulator: values[1],
        });
    }
    if matches!(name, "V_BFREV_B32" | "V_NOT_B32" | "V_FFBH_U32" | "V_FFBL_B32") {
        reject_integer_modifiers(decoded)?;
        if !matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3")
            || decoded.variant != rdna2::Variant::Plain
        {
            return Err(format!("unsupported Stage B unary integer form {name}"));
        }
        let dst = dst(decoded)?;
        let src = source(decoded, field(decoded, "SRC0")?)?;
        return Ok(if name == "V_BFREV_B32" { StageBOp::BitReverse {
            dst,
            src,
        } } else if name == "V_NOT_B32" { StageBOp::BitNot {
            dst,
            src,
        } } else { StageBOp::BitScan {
            high: name == "V_FFBH_U32",
            dst,
            src,
        }});
    }
    if matches!(
        name,
        "V_ADD_NC_U32"
            | "V_AND_B32"
            | "V_OR_B32"
            | "V_XOR_B32"
            | "V_BCNT_U32_B32"
            | "V_LSHLREV_B32"
            | "V_ASHRREV_I32"
            | "V_MIN_U32"
            | "V_MAX_U32"
            | "V_MUL_LO_U32"
            | "V_MUL_HI_U32"
            | "V_SUB_NC_U32"
            | "V_SUBREV_NC_U32"
    ) {
        reject_integer_modifiers(decoded)?;
        let observed_form = match name {
            "V_MIN_U32" | "V_MAX_U32" => {
                decoded.enc.name == "ENC_VOP2"
                    && matches!(
                        decoded.variant,
                        rdna2::Variant::Plain | rdna2::Variant::Dpp16 | rdna2::Variant::Dpp8
                    )
            }
            "V_SUB_NC_U32" => {
                decoded.enc.name == "ENC_VOP2"
                    && matches!(
                        decoded.variant,
                        rdna2::Variant::Plain | rdna2::Variant::Sdwa
                    )
            }
            "V_MUL_LO_U32" | "V_MUL_HI_U32" | "V_BCNT_U32_B32" => {
                decoded.enc.name == "ENC_VOP3"
            }
            "V_SUBREV_NC_U32" => matches!(decoded.enc.name, "ENC_VOP2" | "ENC_VOP3"),
            "V_ASHRREV_I32" => {
                decoded.enc.name == "ENC_VOP2" && decoded.variant == rdna2::Variant::Plain
            }
            "V_OR_B32" | "V_XOR_B32" => {
                decoded.enc.name == "ENC_VOP2"
                    && matches!(
                        decoded.variant,
                        rdna2::Variant::Plain
                            | rdna2::Variant::Sdwa
                            | rdna2::Variant::Dpp16
                            | rdna2::Variant::Dpp8
                    )
            }
            _ => matches!(decoded.enc.name, "ENC_VOP2" | "ENC_VOP3"),
        };
        if !observed_form {
            return Err(format!("unsupported Stage B integer form {name}"));
        }
        let (lhs, rhs) = if matches!(
            name,
            "V_MIN_U32" | "V_MAX_U32" | "V_AND_B32" | "V_OR_B32" | "V_XOR_B32" | "V_ADD_NC_U32"
        )
            && matches!(
                decoded.variant,
                rdna2::Variant::Dpp16 | rdna2::Variant::Dpp8
            ) {
            dpp_integer_sources(decoded)?
        } else if matches!(
            name,
            "V_SUB_NC_U32"
                | "V_AND_B32"
                | "V_OR_B32"
                | "V_XOR_B32"
                | "V_ADD_NC_U32"
                | "V_LSHLREV_B32"
        )
            && decoded.variant == rdna2::Variant::Sdwa
        {
            let encoding = sdwa_encoding(decoded)?;
            let ext = |field| rdna2::field_value(encoding, &decoded.words, field).unwrap_or(0);
            if ext("DST_SEL") != 6
                || ext("DST_UNUSED") != 0
                || ext("CLAMP") != 0
                || ext("OMOD") != 0
            {
                return Err(format!("unsupported {name} SDWA destination modifiers"));
            }
            sdwa_source_pair(decoded)?
        } else if decoded.enc.name == "ENC_VOP3" {
            let values = vop3_sources(decoded)?;
            (values[0], values[1])
        } else {
            vop2(decoded)?
        };
        return Ok(StageBOp::Binary {
            kind: match name {
                "V_ADD_NC_U32" => VectorBinaryKind::AddNoCarry,
                "V_AND_B32" => VectorBinaryKind::And,
                "V_OR_B32" => VectorBinaryKind::Or,
                "V_XOR_B32" => VectorBinaryKind::Xor,
                "V_BCNT_U32_B32" => VectorBinaryKind::BitCount,
                "V_LSHLREV_B32" => VectorBinaryKind::LshlReverse,
                "V_ASHRREV_I32" => VectorBinaryKind::ArithmeticShiftRightReverse,
                "V_MIN_U32" => VectorBinaryKind::MinU32,
                "V_MAX_U32" => VectorBinaryKind::MaxU32,
                "V_MUL_LO_U32" => VectorBinaryKind::MulLoU32,
                "V_MUL_HI_U32" => VectorBinaryKind::MulHiU32,
                "V_SUB_NC_U32" => VectorBinaryKind::SubNoCarry,
                _ => VectorBinaryKind::ReverseSubNoCarry,
            },
            dst: dst(decoded)?,
            lhs,
            rhs,
        });
    }
    if name == "V_CMP_EQ_U64" && matches!(decoded.enc.name, "ENC_VOPC" | "ENC_VOP3") {
        reject_integer_modifiers(decoded)?;
        let (lhs, rhs, dst) = if decoded.enc.name == "ENC_VOP3" {
            let destination = field(decoded, "VDST")?;
            let dst = if destination == 106 {
                None
            } else {
                match scalar_mask(destination)? {
                    ScalarMask::ScalarPair(register) => Some(register),
                    _ => return Err("64-bit compare destination is invalid".into()),
                }
            };
            (
                u64_source(decoded, field(decoded, "SRC0")?)?,
                u64_source(decoded, field(decoded, "SRC1")?)?,
                dst,
            )
        } else {
            (
                u64_source(decoded, field(decoded, "SRC0")?)?,
                u64_source(decoded, field(decoded, "VSRC1")? + 256)?,
                None,
            )
        };
        return Ok(StageBOp::CompareEqU64 { lhs, rhs, dst });
    }
    if name == "V_CMP_CLASS_F32" && decoded.enc.name == "ENC_VOP3" {
        let destination = field(decoded, "VDST")?;
        let dst = if destination == 106 {
            None
        } else {
            match scalar_mask(destination)? {
                ScalarMask::ScalarPair(register) => Some(register),
                _ => return Err("float class destination is invalid".into()),
            }
        };
        let [value, classes, _] = vop3_sources(decoded)?;
        let modifiers = float_compare_modifiers(decoded)?;
        if modifiers.abs & !1 != 0 || modifiers.neg & !1 != 0 {
            return Err("float class mask uses modifiers".into());
        }
        return Ok(StageBOp::CompareClassF32 { value, classes, dst, modifiers });
    }
    if matches!(name, "V_ADD_CO_U32" | "V_ADD_CO_CI_U32") {
        reject_integer_modifiers(decoded)?;
        let (lhs, rhs, carry_in, carry_out) = match decoded.enc.name {
            "ENC_VOP2" if name == "V_ADD_CO_CI_U32" && decoded.variant == rdna2::Variant::Plain => {
                let (lhs, rhs) = vop2(decoded)?;
                (lhs, rhs, Some(ScalarMask::Vcc), ScalarMask::Vcc)
            }
            "VOP3_SDST_ENC" => {
                let lhs = source(decoded, field(decoded, "SRC0")?)?;
                let rhs = source(decoded, field(decoded, "SRC1")?)?;
                let carry_in = if name == "V_ADD_CO_CI_U32" { Some(scalar_mask(field(decoded, "SRC2")?)?) } else { None };
                (lhs, rhs, carry_in, scalar_mask(field(decoded, "SDST")?)?)
            }
            _ => return Err(format!("unsupported Stage B carry-add form {name}")),
        };
        if carry_out == ScalarMask::Exec { return Err("carry-add cannot write EXEC".into()); }
        return Ok(StageBOp::AddCarry { dst: dst(decoded)?, lhs, rhs, carry_in, carry_out });
    }
    if name == "V_SUBREV_CO_CI_U32" {
        reject_integer_modifiers(decoded)?;
        let (lhs, rhs, carry_in, carry_out) = match decoded.enc.name {
            "ENC_VOP2" if decoded.variant == rdna2::Variant::Plain => {
                let (lhs, rhs) = vop2(decoded)?;
                (lhs, rhs, ScalarMask::Vcc, ScalarMask::Vcc)
            }
            "VOP3_SDST_ENC" => (
                source(decoded, field(decoded, "SRC0")?)?,
                source(decoded, field(decoded, "SRC1")?)?,
                scalar_mask(field(decoded, "SRC2")?)?,
                scalar_mask(field(decoded, "SDST")?)?,
            ),
            _ => return Err("unsupported Stage B reverse-subtract carry form".into()),
        };
        if carry_out == ScalarMask::Exec { return Err("reverse-subtract carry cannot write EXEC".into()); }
        return Ok(StageBOp::ReverseSubCarry { dst: dst(decoded)?, lhs, rhs, carry_in, carry_out });
    }
    if matches!(
        name,
        "V_ADD3_U32"
            | "V_BFE_U32"
            | "V_LSHL_ADD_U32"
            | "V_ADD_LSHL_U32"
            | "V_AND_OR_B32"
            | "V_OR3_B32"
            | "V_LSHL_OR_B32"
            | "V_BFI_B32"
            | "V_XAD_U32"
    ) {
        reject_integer_modifiers(decoded)?;
        if matches!(
            name,
            "V_ADD3_U32" | "V_ADD_LSHL_U32" | "V_AND_OR_B32" | "V_LSHL_OR_B32"
                | "V_BFI_B32" | "V_XAD_U32"
        )
            && decoded.enc.name != "ENC_VOP3"
        {
            return Err(format!("unsupported Stage B ternary form {name}"));
        }
        let [a, b, c] = vop3_sources(decoded)?;
        return Ok(StageBOp::Ternary {
            kind: match name {
                "V_ADD3_U32" => VectorTernaryKind::Add3,
                "V_BFE_U32" => VectorTernaryKind::BfeU32,
                "V_LSHL_ADD_U32" => VectorTernaryKind::LshlAdd,
                "V_ADD_LSHL_U32" => VectorTernaryKind::AddLshl,
                "V_AND_OR_B32" => VectorTernaryKind::AndOr,
                "V_OR3_B32" => VectorTernaryKind::Or3,
                "V_LSHL_OR_B32" => VectorTernaryKind::LshlOr,
                "V_BFI_B32" => VectorTernaryKind::BitFieldInsert,
                _ => VectorTernaryKind::XorAdd,
            },
            dst: dst(decoded)?,
            a,
            b,
            c,
        });
    }
    if matches!(
        name,
        "V_CMPX_EQ_U32"
            | "V_CMPX_GT_I32"
            | "V_CMPX_LT_I32"
            | "V_CMPX_LT_U32"
            | "V_CMPX_GE_U32"
            | "V_CMPX_NE_U32"
            | "V_CMP_EQ_U32"
            | "V_CMP_GE_U32"
            | "V_CMP_GT_U32"
            | "V_CMP_LE_U32"
            | "V_CMP_LT_U32"
            | "V_CMP_LT_F16"
            | "V_CMP_NE_U32"
            | "V_CMP_GT_I32"
    ) {
        if name == "V_CMPX_NE_U32"
            && (decoded.enc.name != "ENC_VOPC" || decoded.variant != rdna2::Variant::Plain)
        {
            return Err("unsupported Stage B V_CMPX_NE_U32 form".into());
        }
        reject_integer_modifiers(decoded)?;
        let (lhs, rhs, scalar_dst) = if decoded.variant == rdna2::Variant::Sdwa {
            if !matches!(
                name,
                "V_CMPX_EQ_U32" | "V_CMPX_LT_I32" | "V_CMPX_LT_U32" | "V_CMP_EQ_U32"
            ) {
                return Err(format!("unsupported Stage B SDWA compare form {name}"));
            }
            let (lhs, rhs) = sdwa_source_pair(decoded)?;
            let scalar_dst = if name.starts_with("V_CMPX") {
                None
            } else {
                let encoding = sdwa_encoding(decoded)?;
                let ext = |field| rdna2::field_value(encoding, &decoded.words, field).unwrap_or(0);
                if ext("SD") == 0 {
                    None
                } else {
                    match scalar_mask(ext("SDST"))? {
                        ScalarMask::ScalarPair(register) => Some(register),
                        ScalarMask::Vcc => None,
                        ScalarMask::Exec => {
                            return Err("Stage B SDWA compare destination is EXEC".into())
                        }
                    }
                }
            };
            (lhs, rhs, scalar_dst)
        } else if decoded.enc.name == "ENC_VOP3" {
            let destination = field(decoded, "VDST")?;
            let scalar_dst = if name.starts_with("V_CMPX") || destination == 106 {
                if destination != if name.starts_with("V_CMPX") { 126 } else { 106 } {
                    return Err("Stage B compare destination is invalid".into());
                }
                None
            } else {
                match scalar_mask(destination)? {
                    ScalarMask::ScalarPair(register) => Some(register),
                    _ => return Err("Stage B compare destination is invalid".into()),
                }
            };
            let values = vop3_sources(decoded)?;
            (values[0], values[1], scalar_dst)
        } else {
            let (lhs, rhs) = vop2(decoded)?;
            (lhs, rhs, None)
        };
        let kind = match name {
            "V_CMPX_EQ_U32" | "V_CMP_EQ_U32" => CompareKind::Eq,
            "V_CMPX_NE_U32" | "V_CMP_NE_U32" => CompareKind::NotEqual,
            "V_CMP_GT_U32" => CompareKind::Greater,
            "V_CMP_GE_U32" => CompareKind::GreaterEqual,
            "V_CMPX_GE_U32" => CompareKind::GreaterEqual,
            "V_CMP_LE_U32" => CompareKind::LessEqual,
            "V_CMPX_LT_U32" => CompareKind::Less,
            "V_CMP_LT_U32" => CompareKind::Less,
            "V_CMP_LT_F16" => CompareKind::LessFloat16,
            "V_CMPX_GT_I32" | "V_CMP_GT_I32" => CompareKind::SignedGreater,
            _ => CompareKind::SignedLess,
        };
        return Ok(if let Some(dst) = scalar_dst {
            StageBOp::CompareScalar {
                kind,
                lhs,
                rhs,
                dst,
            }
        } else {
            StageBOp::Compare {
                kind,
                lhs,
                rhs,
                cmpx: name.starts_with("V_CMPX"),
            }
        });
    }
    if matches!(
        name,
        "V_CMP_EQ_F32"
            | "V_CMP_LG_F32"
            | "V_CMP_LT_F32"
            | "V_CMP_GT_F32"
            | "V_CMP_LE_F32"
            | "V_CMP_GE_F32"
            | "V_CMP_NLE_F32"
            | "V_CMP_NEQ_F32"
            | "V_CMPX_EQ_F32"
            | "V_CMPX_GT_F32"
            | "V_CMPX_LT_F32"
            | "V_CMPX_LE_F32"
            | "V_CMPX_NEQ_F32"
            | "V_CMPX_GE_F32"
            | "V_CMPX_NGT_F32"
            | "V_CMP_NGE_F32"
            | "V_CMP_NGT_F32"
    ) && matches!(decoded.enc.name, "ENC_VOPC" | "ENC_VOP3")
    {
        let (lhs, rhs, dst, modifiers) = if decoded.enc.name == "ENC_VOP3" {
            let destination = field(decoded, "VDST")?;
            let cmpx = name.starts_with("V_CMPX");
            let expected_special = if cmpx { 126 } else { 106 };
            let dst = if destination == expected_special {
                None
            } else if cmpx {
                return Err("float CMPX destination is not EXEC".into());
            } else {
                match scalar_mask(destination)? {
                    ScalarMask::ScalarPair(register) => Some(register),
                    _ => return Err("float compare destination is invalid".into()),
                }
            };
            let values = vop3_sources(decoded)?;
            (values[0], values[1], dst, float_compare_modifiers(decoded)?)
        } else {
            let (lhs, rhs) = vop2(decoded)?;
            (lhs, rhs, None, FloatModifiers::default())
        };
        return Ok(StageBOp::CompareLessFloat {
            kind: match name {
                "V_CMP_EQ_F32" | "V_CMPX_EQ_F32" => CompareKind::EqualFloat,
                "V_CMP_LG_F32" => CompareKind::OrderedNotEqualFloat,
                "V_CMP_NEQ_F32" | "V_CMPX_NEQ_F32" => {
                    CompareKind::UnorderedNotEqualFloat
                }
                "V_CMP_GT_F32" => CompareKind::GreaterFloat,
                "V_CMP_LE_F32" | "V_CMPX_LE_F32" => CompareKind::LessEqualFloat,
                "V_CMP_GE_F32" | "V_CMPX_GE_F32" => CompareKind::GreaterEqualFloat,
                "V_CMP_NLE_F32" => CompareKind::UnorderedGreaterFloat,
                "V_CMP_NGE_F32" => CompareKind::UnorderedLessFloat,
                "V_CMP_NGT_F32" | "V_CMPX_NGT_F32" => {
                    CompareKind::UnorderedLessEqualFloat
                }
                _ => CompareKind::LessFloat,
            },
            lhs,
            rhs,
            dst,
            modifiers,
            cmpx: name.starts_with("V_CMPX"),
        });
    }
    if name == "V_CNDMASK_B32" && decoded.enc.name == "ENC_VOP3" {
        reject_cndmask_modifiers(decoded)?;
        let [when_clear, when_set, predicate] = vop3_sources(decoded)?;
        let predicate = match predicate {
            VectorSource::Scalar(selector) => scalar_mask(selector as u64)?,
            _ => return Err("Stage B conditional mask predicate is invalid".into()),
        };
        return Ok(match predicate {
            ScalarMask::Vcc => StageBOp::CndMask {
                dst: dst(decoded)?,
                when_clear,
                when_set,
            },
            _ => StageBOp::CndMaskScalar {
                dst: dst(decoded)?,
                when_clear,
                when_set,
                predicate,
            },
        });
    }
    if name == "V_CNDMASK_B32" && decoded.enc.name == "ENC_VOP2" {
        let (when_clear, when_set) = vop2(decoded)?;
        return Ok(StageBOp::CndMask {
            dst: dst(decoded)?,
            when_clear,
            when_set,
        });
    }
    if name == "V_CVT_F32_U32" && decoded.variant == rdna2::Variant::Sdwa {
        return lower_sdwa_convert(decoded);
    }
    if name == "V_CVT_F32_U32" && matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3") {
        return Ok(StageBOp::ConvertF32U32 {
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
            modifiers: float_modifiers(decoded)?,
        });
    }
    if name == "V_CVT_F32_I32"
        && decoded.enc.name == "ENC_VOP1"
        && decoded.variant == rdna2::Variant::Plain
    {
        return Ok(StageBOp::ConvertF32I32 {
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
        });
    }
    if matches!(
        name,
        "V_CVT_F16_F32" | "V_CVT_F32_F16" | "V_CVT_U32_F32" | "V_CVT_I32_F32"
    ) {
        if decoded.variant == rdna2::Variant::Sdwa
            && matches!(name, "V_CVT_F16_F32" | "V_CVT_F32_F16")
        {
            return lower_sdwa_half_convert(decoded);
        }
        if decoded.enc.name != "ENC_VOP1" || decoded.variant != rdna2::Variant::Plain {
            return Err(format!("unsupported Stage B conversion form {name}"));
        }
        return Ok(StageBOp::FloatConvert {
            kind: match name {
                "V_CVT_F16_F32" => FloatConvertKind::F16FromF32,
                "V_CVT_F32_F16" => FloatConvertKind::F32FromF16,
                "V_CVT_U32_F32" => FloatConvertKind::U32FromF32,
                _ => FloatConvertKind::I32FromF32,
            },
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
        });
    }
    if name == "V_CVT_F32_UBYTE0" && decoded.variant == rdna2::Variant::Sdwa {
        return lower_sdwa_ubyte0(decoded);
    }
    if matches!(
        name,
        "V_CVT_F32_UBYTE0" | "V_CVT_F32_UBYTE1" | "V_CVT_F32_UBYTE2" | "V_CVT_F32_UBYTE3"
    ) && matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3")
        && decoded.variant == rdna2::Variant::Plain
    {
        let modifiers = float_modifiers(decoded)?;
        if modifiers.abs != 0
            || modifiers.neg != 0
            || rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0
        {
            return Err(format!("{name} uses unsupported source modifiers"));
        }
        return Ok(StageBOp::ConvertUByte {
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
            byte: match name {
                "V_CVT_F32_UBYTE0" => 0,
                "V_CVT_F32_UBYTE1" => 1,
                "V_CVT_F32_UBYTE2" => 2,
                _ => 3,
            },
            modifiers,
        });
    }
    if matches!(
        name,
        "V_RCP_F32"
            | "V_RCP_IFLAG_F32"
            | "V_RSQ_F32"
            | "V_SQRT_F32"
            | "V_SIN_F32"
            | "V_COS_F32"
            | "V_EXP_F32"
            | "V_FRACT_F32"
            | "V_LOG_F32"
    ) && matches!(decoded.enc.name, "ENC_VOP1" | "ENC_VOP3")
        && decoded.variant == rdna2::Variant::Plain
    {
        if matches!(name, "V_EXP_F32" | "V_FRACT_F32" | "V_LOG_F32")
            && decoded.enc.name == "ENC_VOP3"
            && rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0
        {
            return Err(format!("{name} uses unsupported selector modifiers"));
        }
        return Ok(StageBOp::FloatMath {
            kind: match name {
                "V_RCP_F32" => FloatMathKind::Reciprocal,
                "V_RCP_IFLAG_F32" => FloatMathKind::ReciprocalIFlag,
                "V_RSQ_F32" => FloatMathKind::ReciprocalSqrt,
                "V_SQRT_F32" => FloatMathKind::Sqrt,
                "V_SIN_F32" => FloatMathKind::Sin,
                "V_COS_F32" => FloatMathKind::Cos,
                "V_EXP_F32" => FloatMathKind::Exp2,
                "V_FRACT_F32" => FloatMathKind::Fract,
                _ => FloatMathKind::Log2,
            },
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
            modifiers: float_modifiers(decoded)?,
        });
    }
    if matches!(name, "V_MIN3_F32" | "V_MAX3_F32" | "V_MED3_F32")
        && decoded.enc.name == "ENC_VOP3"
        && decoded.variant == rdna2::Variant::Plain
    {
        if rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0 {
            return Err(format!("{name} uses unsupported selector modifiers"));
        }
        let [a, b, c] = vop3_sources(decoded)?;
        return Ok(StageBOp::FloatTernary {
            kind: match name {
                "V_MIN3_F32" => FloatTernaryKind::Minimum3,
                "V_MAX3_F32" => FloatTernaryKind::Maximum3,
                _ => FloatTernaryKind::Median3,
            },
            dst: dst(decoded)?,
            a,
            b,
            c,
            modifiers: float_modifiers(decoded)?,
        });
    }
    if name == "V_PACK_B32_F16" && decoded.enc.name == "ENC_VOP3" {
        let modifiers = float_modifiers(decoded)?;
        if modifiers.abs != 0 || modifiers.neg != 0 || modifiers.omod != 0 || modifiers.clamp {
            return Err("V_PACK_B32_F16 uses unsupported modifiers".into());
        }
        let values = vop3_sources(decoded)?;
        return Ok(StageBOp::PackF16 {
            dst: dst(decoded)?,
            lo: values[0],
            hi: values[1],
            op_sel: field(decoded, "OP_SEL").unwrap_or(0) as u8,
        });
    }
    if matches!(
        name,
        "V_FMA_MIX_F32" | "V_FMA_MIXLO_F16" | "V_FMA_MIXHI_F16"
    ) && decoded.enc.name == "ENC_VOP3P"
        && decoded.variant == rdna2::Variant::Plain
    {
        let values = vop3_sources(decoded)?;
        return Ok(StageBOp::FmaMix {
            kind: match name {
                "V_FMA_MIX_F32" => FmaMixKind::F32,
                "V_FMA_MIXLO_F16" => FmaMixKind::LoF16,
                _ => FmaMixKind::HiF16,
            },
            dst: dst(decoded)?,
            a: values[0],
            b: values[1],
            c: values[2],
            op_sel: field(decoded, "OP_SEL").unwrap_or(0) as u8,
            op_sel_hi: (field(decoded, "OP_SEL_HI").unwrap_or(0)
                | field(decoded, "OP_SEL_HI_2").unwrap_or(0) << 2) as u8,
            neg: field(decoded, "NEG").unwrap_or(0) as u8,
            abs: field(decoded, "NEG_HI").unwrap_or(0) as u8,
            clamp: field(decoded, "CLAMP").unwrap_or(0) != 0,
        });
    }
    if matches!(
        name,
        "V_TRUNC_F32" | "V_FLOOR_F32" | "V_CEIL_F32" | "V_RNDNE_F32"
    )
        && decoded.enc.name == "ENC_VOP1"
        && decoded.variant == rdna2::Variant::Plain
    {
        return Ok(StageBOp::FloatUnary {
            kind: match name {
                "V_TRUNC_F32" => FloatUnaryKind::Truncate,
                "V_FLOOR_F32" => FloatUnaryKind::Floor,
                "V_CEIL_F32" => FloatUnaryKind::Ceiling,
                _ => FloatUnaryKind::RoundEven,
            },
            dst: dst(decoded)?,
            src: source(decoded, field(decoded, "SRC0")?)?,
        });
    }
    if matches!(
        name,
        "V_ADD_F32"
            | "V_SUB_F32"
            | "V_SUBREV_F32"
            | "V_MUL_F32"
            | "V_MUL_LEGACY_F32"
            | "V_MAX_F32"
            | "V_MIN_F32"
    ) {
        if name == "V_MIN_F32"
            && (decoded.variant != rdna2::Variant::Plain
                || (decoded.enc.name == "ENC_VOP3"
                    && rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0))
        {
            return Err("unsupported V_MIN_F32 encoding or selector modifiers".into());
        }
        if name == "V_MUL_F32"
            && (decoded.variant != rdna2::Variant::Plain
                || (decoded.enc.name == "ENC_VOP3"
                    && rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0))
        {
            return Err("unsupported V_MUL_F32 encoding or selector modifiers".into());
        }
        let (lhs, rhs) = if decoded.enc.name == "ENC_VOP3" {
            let values = vop3_sources(decoded)?;
            (values[0], values[1])
        } else if decoded.enc.name == "ENC_VOP2" {
            vop2(decoded)?
        } else {
            return Err(format!("unsupported Stage D opcode/form {name}"));
        };
        return Ok(StageBOp::FloatBinary {
            kind: match name {
                "V_ADD_F32" => FloatBinaryKind::Add,
                "V_SUB_F32" => FloatBinaryKind::Subtract,
                "V_SUBREV_F32" => FloatBinaryKind::ReverseSubtract,
                "V_MUL_F32" => FloatBinaryKind::Multiply,
                "V_MAX_F32" => FloatBinaryKind::Maximum,
                "V_MIN_F32" => FloatBinaryKind::Minimum,
                _ => FloatBinaryKind::MultiplyLegacy,
            },
            dst: dst(decoded)?,
            lhs,
            rhs,
            modifiers: float_modifiers(decoded)?,
        });
    }
    if name == "V_FMAAK_F32" {
        if decoded.enc.name != "ENC_VOP2" || decoded.variant != rdna2::Variant::Plain {
            return Err("unsupported V_FMAAK_F32 form".into());
        }
        let (a, b) = vop2(decoded)?;
        let literal = decoded
            .literal
            .ok_or_else(|| "V_FMAAK_F32 lacks its literal addend".to_string())?;
        return Ok(StageBOp::FloatFma {
            dst: dst(decoded)?,
            a,
            b,
            c: Some(VectorSource::Constant(literal)),
            modifiers: FloatModifiers::default(),
            legacy_zero: false,
        });
    }
    if name == "V_MADAK_F32" {
        if decoded.enc.name != "ENC_VOP2" || decoded.variant != rdna2::Variant::Plain {
            return Err("unsupported V_MADAK_F32 form".into());
        }
        let (a, b) = vop2(decoded)?;
        let literal = decoded
            .literal
            .ok_or_else(|| "V_MADAK_F32 lacks its literal addend".to_string())?;
        return Ok(StageBOp::FloatMad {
            dst: dst(decoded)?,
            a,
            b,
            c: Some(VectorSource::Constant(literal)),
            modifiers: FloatModifiers::default(),
        });
    }
    if name == "V_FMA_LEGACY_F32" {
        if decoded.enc.name != "ENC_VOP3"
            || decoded.variant != rdna2::Variant::Plain
            || rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0
        {
            return Err("unsupported V_FMA_LEGACY_F32 form or selector modifiers".into());
        }
        let values = vop3_sources(decoded)?;
        return Ok(StageBOp::FloatFma {
            dst: dst(decoded)?,
            a: values[0],
            b: values[1],
            c: Some(values[2]),
            modifiers: float_modifiers(decoded)?,
            legacy_zero: true,
        });
    }
    if matches!(name, "V_FMAC_F32" | "V_FMA_F32") {
        let values = if decoded.enc.name == "ENC_VOP3" {
            vop3_sources(decoded)?
        } else if name == "V_FMAC_F32" && decoded.enc.name == "ENC_VOP2" {
            let (a, b) = vop2(decoded)?;
            [a, b, VectorSource::Constant(0)]
        } else {
            return Err(format!("unsupported Stage D opcode/form {name}"));
        };
        return Ok(StageBOp::FloatFma {
            dst: dst(decoded)?,
            a: values[0],
            b: values[1],
            c: (name == "V_FMA_F32").then_some(values[2]),
            modifiers: float_modifiers(decoded)?,
            legacy_zero: false,
        });
    }
    if matches!(name, "V_MAC_F32" | "V_MAD_F32") {
        let values = if decoded.enc.name == "ENC_VOP3" {
            vop3_sources(decoded)?
        } else if name == "V_MAC_F32"
            && decoded.enc.name == "ENC_VOP2"
            && decoded.variant == rdna2::Variant::Plain
        {
            let (a, b) = vop2(decoded)?;
            [a, b, VectorSource::Constant(0)]
        } else {
            return Err(format!("unsupported console MAD opcode/form {name}"));
        };
        if decoded.variant != rdna2::Variant::Plain {
            return Err(format!("unsupported console MAD variant {name}"));
        }
        if decoded.enc.name == "ENC_VOP3"
            && rdna2::field_value(decoded.enc, &decoded.words, "OP_SEL").unwrap_or(0) != 0
        {
            return Err(format!("unsupported console MAD selector modifiers {name}"));
        }
        return Ok(StageBOp::FloatMad {
            dst: dst(decoded)?,
            a: values[0],
            b: values[1],
            c: (name == "V_MAD_F32").then_some(values[2]),
            modifiers: float_modifiers(decoded)?,
        });
    }
    if name == "V_LSHRREV_B32" {
        let (a, b) = if decoded.enc.name == "ENC_VOP3" {
            let values = vop3_sources(decoded)?;
            (values[0], values[1])
        } else if decoded.enc.name == "ENC_VOP2" {
            vop2(decoded)?
        } else {
            return Err("unsupported Stage D shift form".into());
        };
        return Ok(StageBOp::StageDInteger {
            kind: StageDIntegerKind::LogicalShiftRight,
            dst: dst(decoded)?,
            a,
            b,
            c: None,
        });
    }
    if matches!(name, "V_MAD_U32_U24" | "V_PERM_B32") && decoded.enc.name == "ENC_VOP3" {
        let [a, b, c] = vop3_sources(decoded)?;
        return Ok(StageBOp::StageDInteger {
            kind: if name == "V_MAD_U32_U24" {
                StageDIntegerKind::MadU24
            } else {
                StageDIntegerKind::Permute
            },
            dst: dst(decoded)?,
            a,
            b,
            c: Some(c),
        });
    }
    if name == "V_CMPX_LG_F32" {
        if decoded.enc.name != "ENC_VOPC" || decoded.variant != rdna2::Variant::Plain {
            return Err("unsupported Stage D floating compare form".into());
        }
        let (lhs, rhs) = vop2(decoded)?;
        return Ok(StageBOp::Compare {
            kind: CompareKind::OrderedNotEqualFloat,
            lhs,
            rhs,
            cmpx: true,
        });
    }
    if name == "V_CMPX_GT_U32" {
        let (lhs, rhs) = if decoded.enc.name == "ENC_VOP3" {
            let expected = if name.starts_with("V_CMPX") { 126 } else { 106 };
            if field(decoded, "VDST")? != expected {
                return Err("Stage D compare destination is invalid".into());
            }
            let values = vop3_sources(decoded)?;
            (values[0], values[1])
        } else if decoded.enc.name == "ENC_VOPC" {
            vop2(decoded)?
        } else {
            return Err(format!("unsupported Stage D compare form {name}"));
        };
        return Ok(StageBOp::Compare {
            kind: CompareKind::Greater,
            lhs,
            rhs,
            cmpx: name.starts_with("V_CMPX"),
        });
    }
    Err(format!("unsupported Stage B opcode/form {name}"))
}

fn mask(words: [u32; 2]) -> u64 {
    words[0] as u64 | (words[1] as u64) << 32
}

fn scalar(state: &LogicalWaveState, index: u8) -> Option<u32> {
    match index {
        106 => Some(state.vcc[0]),
        107 => Some(state.vcc[1]),
        126 => Some(state.exec[0]),
        127 => Some(state.exec[1]),
        125 => Some(0),
        _ => state.scalar.get(index as usize).copied(),
    }
}

fn value(
    memory: &[u32],
    layout: VgprScratchLayout,
    state: &LogicalWaveState,
    thread: u32,
    source: VectorSource,
) -> Result<u32, &'static str> {
    match source {
        VectorSource::Vgpr(register) => load_vgpr(memory, layout, 0, thread, register as u32),
        VectorSource::Scalar(register) => scalar(state, register).ok_or("scalar source is invalid"),
        VectorSource::Constant(value) => Ok(value),
        VectorSource::Sdwa {
            register,
            scalar: scalar_source,
            select,
            sign_extend,
        } => {
            let raw = if scalar_source {
                scalar(state, register).ok_or("SDWA scalar source is invalid")?
            } else {
                load_vgpr(memory, layout, 0, thread, register as u32)?
            };
            let (shift, width) = match select {
                0..=3 => (select as u32 * 8, 8),
                4..=5 => ((select as u32 - 4) * 16, 16),
                6 => (0, 32),
                _ => return Err("SDWA source selector is invalid"),
            };
            let selected = raw >> shift;
            Ok(if sign_extend && width < 32 {
                ((selected << (32 - width)) as i32 >> (32 - width)) as u32
            } else if width < 32 {
                selected & (u32::MAX >> (32 - width))
            } else {
                selected
            })
        }
        VectorSource::Dpp16RowMirror {
            register,
            fetch_inactive,
        } => {
            let lane = thread % 64;
            let source_lane = (lane & !15) | (15 - (lane & 15));
            if !fetch_inactive && mask(state.exec) & (1u64 << source_lane) == 0 {
                Ok(0)
            } else {
                let source_thread = thread - lane + source_lane;
                load_vgpr(memory, layout, 0, source_thread, register as u32)
            }
        }
        VectorSource::Dpp8 {
            register,
            lanes,
            fetch_inactive,
        } => {
            let lane = thread % 64;
            let source_lane = (lane & !7) | lanes[(lane & 7) as usize] as u32;
            if !fetch_inactive && mask(state.exec) & (1u64 << source_lane) == 0 {
                Ok(0)
            } else {
                let source_thread = thread - lane + source_lane;
                load_vgpr(memory, layout, 0, source_thread, register as u32)
            }
        }
        VectorSource::VccZero => Ok(u32::from(mask(state.vcc) == 0)),
        VectorSource::ExecZero => Ok(u32::from(mask(state.exec) == 0)),
        VectorSource::Scc => Ok(state.scc),
    }
}

fn scalar_mask_value(state: &LogicalWaveState, source: ScalarMask) -> u64 {
    match source {
        ScalarMask::ScalarPair(register) => {
            state.scalar[register as usize] as u64
                | (state.scalar[register as usize + 1] as u64) << 32
        }
        ScalarMask::Vcc => mask(state.vcc),
        ScalarMask::Exec => mask(state.exec),
    }
}

fn bfe(value: u32, offset: u32, width: u32) -> u32 {
    let (offset, width) = (offset & 31, width & 31);
    if width == 0 {
        0
    } else {
        (value >> offset) & (u32::MAX >> ((32 - width) & 31))
    }
}

fn float_source(value: u32, modifiers: FloatModifiers, index: u8) -> f32 {
    let mut value = f32::from_bits(value);
    if modifiers.abs & (1 << index) != 0 {
        value = value.abs();
    }
    if modifiers.neg & (1 << index) != 0 {
        value = -value;
    }
    value
}

fn float_result(mut value: f32, modifiers: FloatModifiers) -> u32 {
    value = match modifiers.omod {
        1 => value * 2.0,
        2 => value * 4.0,
        3 => value * 0.5,
        _ => value,
    };
    if modifiers.clamp {
        value = value.clamp(0.0, 1.0);
    }
    value.to_bits()
}

fn flush_f32_denormal(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0 {
        f32::from_bits(bits & 0x8000_0000)
    } else {
        value
    }
}

fn mix_source(value: u32, op_sel: u8, op_sel_hi: u8, neg: u8, abs: u8, index: u8) -> f32 {
    let mut value = if op_sel_hi & (1 << index) != 0 {
        let half = if op_sel & (1 << index) != 0 {
            (value >> 16) as u16
        } else {
            value as u16
        };
        f16_to_f32(half)
    } else {
        f32::from_bits(value)
    };
    if abs & (1 << index) != 0 {
        value = value.abs();
    }
    if neg & (1 << index) != 0 {
        value = -value;
    }
    value
}

fn round_shift_even(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7c00 | (((mantissa >> 13) as u16) | 0x0200);
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7bff;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let rounded = round_shift_even(mantissa | 0x80_0000, (14 - half_exponent) as u32);
        return sign | rounded as u16;
    }
    let mut rounded = round_shift_even(mantissa, 13);
    let mut exponent = half_exponent as u16;
    if rounded == 0x400 {
        rounded = 0;
        exponent += 1;
        if exponent >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | (exponent << 10) | rounded as u16
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value as u32 & 0x8000) << 16) as u32;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = value & 0x03ff;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let leading = 15 - mantissa.leading_zeros() as i32;
            let normalized = (mantissa as u32) << (10 - leading);
            let exponent = (127 - 14 - 10 + leading) as u32;
            sign | (exponent << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((mantissa as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((mantissa as u32) << 13),
    };
    f32::from_bits(bits)
}

fn permute_bytes(source0: u32, source1: u32, selector: u32) -> u32 {
    let mut result = 0u32;
    for index in 0..4 {
        let key = (selector >> (index * 8)) & 0xff;
        let byte = match key {
            13.. => 0xff,
            12 => 0,
            8..=11 => {
                let bit = match key {
                    8 => source1 >> 15,
                    9 => source1 >> 31,
                    10 => source0 >> 15,
                    _ => source0 >> 31,
                } & 1;
                if bit != 0 {
                    0xff
                } else {
                    0
                }
            }
            0..=3 => (source1 >> (key * 8)) & 0xff,
            _ => (source0 >> ((key - 4) * 8)) & 0xff,
        };
        result |= byte << (index * 8);
    }
    result
}

fn validate_ops(layout: VgprScratchLayout, ops: &[StageBOp]) -> Result<(), &'static str> {
    let source = |source: VectorSource| match source {
        VectorSource::Vgpr(register) if register as u32 >= layout.vgpr_count => {
            Err("Stage B VGPR source is out of bounds")
        }
        VectorSource::Sdwa {
            register,
            scalar: false,
            ..
        }
        | VectorSource::Dpp16RowMirror { register, .. }
        | VectorSource::Dpp8 { register, .. }
            if register as u32 >= layout.vgpr_count =>
        {
            Err("Stage B transformed VGPR source is out of bounds")
        }
        VectorSource::Sdwa {
            register,
            scalar: true,
            ..
        } if register >= 106 => Err("Stage B SDWA scalar source is out of bounds"),
        _ => Ok(()),
    };
    let destination = |register: u16| {
        if register as u32 >= layout.vgpr_count {
            Err("Stage B VGPR destination is out of bounds")
        } else {
            Ok(())
        }
    };
    for &op in ops {
        match op {
            StageBOp::Move { dst, src } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::RelativeMove { dst, base } => {
                destination(dst)?;
                source(VectorSource::Vgpr(base))?;
            }
            StageBOp::ReadFirstLane { dst, src } => {
                if dst >= 106 {
                    return Err("Stage B scalar destination is out of bounds");
                }
                source(VectorSource::Vgpr(src))?;
            }
            StageBOp::ReadLane { dst, src, lane } => {
                if dst >= 106 {
                    return Err("Stage B scalar destination is out of bounds");
                }
                source(VectorSource::Vgpr(src))?;
                source(lane)?;
            }
            StageBOp::WriteLane { dst, src, lane } => {
                destination(dst)?;
                source(src)?;
                source(lane)?;
            }
            StageBOp::PermLaneX16 { dst, src, selectors_low, selectors_high, .. } => {
                destination(dst)?;
                source(VectorSource::Vgpr(src))?;
                source(selectors_low)?;
                source(selectors_high)?;
            }
            StageBOp::Mbcnt {
                dst,
                source: mask,
                accumulator,
                ..
            } => {
                destination(dst)?;
                source(mask)?;
                source(accumulator)?;
            }
            StageBOp::Binary { dst, lhs, rhs, .. } => {
                destination(dst)?;
                source(lhs)?;
                source(rhs)?;
            }
            StageBOp::AddCarry { dst, lhs, rhs, carry_in, carry_out } => {
                destination(dst)?;
                source(lhs)?;
                source(rhs)?;
                for mask in carry_in.into_iter().chain([carry_out]) {
                    if matches!(mask, ScalarMask::ScalarPair(register) if register > 104 || register % 2 != 0) {
                        return Err("Stage B carry mask is out of bounds");
                    }
                }
            }
            StageBOp::ReverseSubCarry { dst, lhs, rhs, carry_in, carry_out } => {
                destination(dst)?;
                source(lhs)?;
                source(rhs)?;
                for mask in [carry_in, carry_out] {
                    if matches!(mask, ScalarMask::ScalarPair(register) if register > 104 || register % 2 != 0) {
                        return Err("Stage B carry mask is out of bounds");
                    }
                }
            }
            StageBOp::Ternary { dst, a, b, c, .. } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                source(c)?;
            }
            StageBOp::BitReverse { dst, src } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::BitNot { dst, src } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::BitScan { dst, src, .. } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::Compare { lhs, rhs, .. } => {
                source(lhs)?;
                source(rhs)?;
            }
            StageBOp::CompareScalar { lhs, rhs, dst, .. } => {
                source(lhs)?;
                source(rhs)?;
                if dst > 104 || dst % 2 != 0 {
                    return Err("Stage B scalar mask destination is out of bounds");
                }
            }
            StageBOp::CompareEqU64 { lhs, rhs, dst } => {
                for value in [lhs, rhs] {
                    source(value)?;
                    match value {
                        VectorSource::Vgpr(register) => source(VectorSource::Vgpr(register + 1))?,
                        VectorSource::Scalar(register) => source(VectorSource::Scalar(register + 1))?,
                        _ => return Err("64-bit compare source is not paired"),
                    }
                }
                if matches!(dst, Some(register) if register > 104 || register % 2 != 0) {
                    return Err("64-bit compare destination is out of bounds");
                }
            }
            StageBOp::CompareClassF32 { value, classes, dst, .. } => {
                source(value)?;
                source(classes)?;
                if matches!(dst, Some(register) if register > 104 || register % 2 != 0) {
                    return Err("float class destination is out of bounds");
                }
            }
            StageBOp::CompareLessFloat { lhs, rhs, dst, .. } => {
                source(lhs)?;
                source(rhs)?;
                if matches!(dst, Some(register) if register > 104 || register % 2 != 0) {
                    return Err("Stage B scalar mask destination is out of bounds");
                }
            }
            StageBOp::CndMask {
                dst,
                when_clear,
                when_set,
            } => {
                destination(dst)?;
                source(when_clear)?;
                source(when_set)?;
            }
            StageBOp::CndMaskScalar {
                dst,
                when_clear,
                when_set,
                predicate,
            } => {
                destination(dst)?;
                source(when_clear)?;
                source(when_set)?;
                if matches!(predicate, ScalarMask::ScalarPair(register) if register > 104 || register % 2 != 0)
                {
                    return Err("Stage B scalar mask source is out of bounds");
                }
            }
            StageBOp::ConvertF32U32 { dst, src, .. } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::ConvertF32I32 { dst, src }
            | StageBOp::FloatUnary { dst, src, .. }
            | StageBOp::FloatMath { dst, src, .. }
            | StageBOp::ConvertUByte { dst, src, .. } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::ConvertF32U32Sdwa { dst, src, .. } => {
                destination(dst)?;
                source(VectorSource::Vgpr(src))?;
            }
            StageBOp::FloatConvert { dst, src, .. } => {
                destination(dst)?;
                source(src)?;
            }
            StageBOp::FloatBinary { dst, lhs, rhs, .. } => {
                destination(dst)?;
                source(lhs)?;
                source(rhs)?;
            }
            StageBOp::FloatTernary { dst, a, b, c, .. } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                source(c)?;
            }
            StageBOp::FloatFma { dst, a, b, c, .. } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                if let Some(c) = c {
                    source(c)?;
                }
            }
            StageBOp::FloatMad { dst, a, b, c, .. } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                if let Some(c) = c {
                    source(c)?;
                }
            }
            StageBOp::PackF16 {
                dst,
                lo,
                hi,
                op_sel,
            } => {
                destination(dst)?;
                source(lo)?;
                source(hi)?;
                if op_sel & !3 != 0 {
                    return Err("Stage B pack selector is invalid");
                }
            }
            StageBOp::FmaMix {
                dst,
                a,
                b,
                c,
                op_sel,
                op_sel_hi,
                neg,
                abs,
                ..
            } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                source(c)?;
                if op_sel & !7 != 0 || op_sel_hi & !7 != 0 || neg & !7 != 0 || abs & !7 != 0 {
                    return Err("Stage B mixed-FMA selector is invalid");
                }
            }
            StageBOp::StageDInteger { dst, a, b, c, .. } => {
                destination(dst)?;
                source(a)?;
                source(b)?;
                if let Some(c) = c {
                    source(c)?;
                }
            }
            StageBOp::NoEffect => {}
        }
    }
    Ok(())
}

pub fn execute_stage_b(
    memory: &mut [u32],
    layout: VgprScratchLayout,
    states: &mut [LogicalWaveState; 2],
    ops: &[StageBOp],
) -> Result<(), &'static str> {
    layout.validate_for_groups(1)?;
    validate_ops(layout, ops)?;
    if ops
        .iter()
        .any(|op| matches!(op, StageBOp::ReadFirstLane { .. }))
        && states.iter().any(|state| mask(state.exec) == 0)
    {
        return Err("readfirstlane requires a nonempty EXEC mask");
    }
    for &op in ops {
        for wave in 0..2usize {
            let old_exec = mask(states[wave].exec);
            let old_vcc = mask(states[wave].vcc);
            let snapshot = memory.to_vec();
            let mut predicates = 0u64;
            for lane in 0..64u32 {
                let thread = wave as u32 * 64 + lane;
                let active = old_exec & (1u64 << lane) != 0;
                let read = |source| value(&snapshot, layout, &states[wave], thread, source);
                match op {
                    StageBOp::Move { dst, src } if active => {
                        store_vgpr(memory, layout, 0, thread, dst as u32, read(src)?)?;
                    }
                    StageBOp::RelativeMove { dst, base } if active => {
                        let index = (base as u32).checked_add(states[wave].scalar[124]);
                        let selected = index
                            .filter(|&register| register < layout.vgpr_count)
                            .map(|register| load_vgpr(&snapshot, layout, 0, thread, register))
                            .transpose()?
                            .unwrap_or(0);
                        store_vgpr(memory, layout, 0, thread, dst as u32, selected)?;
                    }
                    StageBOp::ReadFirstLane { dst, src } if lane == 0 => {
                        let first = old_exec.trailing_zeros();
                        let first_thread = wave as u32 * 64 + first;
                        let selected = load_vgpr(&snapshot, layout, 0, first_thread, src as u32)?;
                        states[wave].scalar[dst as usize] = selected;
                    }
                    StageBOp::ReadLane { dst, src, lane: selected_lane } if lane == 0 => {
                        let selected_lane = read(selected_lane)? & 63;
                        let selected_thread = wave as u32 * 64 + selected_lane;
                        states[wave].scalar[dst as usize] =
                            load_vgpr(&snapshot, layout, 0, selected_thread, src as u32)?;
                    }
                    StageBOp::WriteLane { dst, src, lane: selected_lane } => {
                        if lane == (read(selected_lane)? & 63) {
                            store_vgpr(memory, layout, 0, thread, dst as u32, read(src)?)?;
                        }
                    }
                    StageBOp::PermLaneX16 {
                        dst,
                        src,
                        selectors_low,
                        selectors_high,
                        fetch_inactive,
                        bound_control,
                    } if active => {
                        let selectors = if lane & 15 < 8 {
                            read(selectors_low)?
                        } else {
                            read(selectors_high)?
                        };
                        let selected_in_row = (selectors >> (4 * (lane & 7))) & 15;
                        let selected_lane = (lane & !31) | ((lane & 16) ^ 16) | selected_in_row;
                        let selected_active = old_exec & (1u64 << selected_lane) != 0;
                        if fetch_inactive || selected_active {
                            let selected_thread = wave as u32 * 64 + selected_lane;
                            let selected = load_vgpr(&snapshot, layout, 0, selected_thread, src as u32)?;
                            store_vgpr(memory, layout, 0, thread, dst as u32, selected)?;
                        } else if bound_control {
                            store_vgpr(memory, layout, 0, thread, dst as u32, 0)?;
                        }
                    }
                    StageBOp::Mbcnt {
                        high,
                        dst,
                        source,
                        accumulator,
                    } if active => {
                        let result = if high {
                            super::mesh_lane::mbcnt_hi(read(source)?, lane, read(accumulator)?)
                        } else {
                            super::mesh_lane::mbcnt_lo(read(source)?, lane, read(accumulator)?)
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::Binary {
                        kind,
                        dst,
                        lhs,
                        rhs,
                    } if active => {
                        let (lhs, rhs) = (read(lhs)?, read(rhs)?);
                        let result = match kind {
                            VectorBinaryKind::AddNoCarry => lhs.wrapping_add(rhs),
                            VectorBinaryKind::And => lhs & rhs,
                            VectorBinaryKind::Or => lhs | rhs,
                            VectorBinaryKind::Xor => lhs ^ rhs,
                            VectorBinaryKind::BitCount => lhs.count_ones().wrapping_add(rhs),
                            VectorBinaryKind::LshlReverse => rhs << (lhs & 31),
                            VectorBinaryKind::ArithmeticShiftRightReverse => {
                                ((rhs as i32) >> (lhs & 31)) as u32
                            }
                            VectorBinaryKind::MinU32 => lhs.min(rhs),
                            VectorBinaryKind::MaxU32 => lhs.max(rhs),
                            VectorBinaryKind::MulLoU32 => lhs.wrapping_mul(rhs),
                            VectorBinaryKind::MulHiU32 => ((lhs as u64 * rhs as u64) >> 32) as u32,
                            VectorBinaryKind::SubNoCarry => lhs.wrapping_sub(rhs),
                            VectorBinaryKind::ReverseSubNoCarry => rhs.wrapping_sub(lhs),
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::AddCarry { dst, lhs, rhs, carry_in, .. } if active => {
                        let (lhs, rhs) = (read(lhs)?, read(rhs)?);
                        let carry = carry_in.map(|source| (scalar_mask_value(&states[wave], source) >> lane) & 1).unwrap_or(0);
                        let sum = lhs as u64 + rhs as u64 + carry;
                        store_vgpr(memory, layout, 0, thread, dst as u32, sum as u32)?;
                        if sum >> 32 != 0 { predicates |= 1u64 << lane; }
                    }
                    StageBOp::ReverseSubCarry { dst, lhs, rhs, carry_in, .. } if active => {
                        let (lhs, rhs) = (read(lhs)?, read(rhs)?);
                        let carry = (scalar_mask_value(&states[wave], carry_in) >> lane) & 1;
                        let difference = (rhs as u64).wrapping_sub(lhs as u64).wrapping_sub(carry);
                        store_vgpr(memory, layout, 0, thread, dst as u32, difference as u32)?;
                        if difference >> 32 != 0 { predicates |= 1u64 << lane; }
                    }
                    StageBOp::Ternary { kind, dst, a, b, c } if active => {
                        let (a, b, c) = (read(a)?, read(b)?, read(c)?);
                        let result = match kind {
                            VectorTernaryKind::Add3 => a.wrapping_add(b).wrapping_add(c),
                            VectorTernaryKind::BfeU32 => bfe(a, b, c),
                            VectorTernaryKind::LshlAdd => (a << (b & 31)).wrapping_add(c),
                            VectorTernaryKind::AddLshl => a.wrapping_add(b) << (c & 31),
                            VectorTernaryKind::AndOr => (a & b) | c,
                            VectorTernaryKind::Or3 => a | b | c,
                            VectorTernaryKind::LshlOr => (a << (b & 31)) | c,
                            VectorTernaryKind::BitFieldInsert => (a & b) | (!a & c),
                            VectorTernaryKind::XorAdd => (a ^ b).wrapping_add(c),
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::BitReverse { dst, src } if active => {
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            read(src)?.reverse_bits(),
                        )?;
                    }
                    StageBOp::BitNot { dst, src } if active => {
                        store_vgpr(memory, layout, 0, thread, dst as u32, !read(src)?)?;
                    }
                    StageBOp::BitScan { high, dst, src } if active => {
                        let value = read(src)?;
                        let result = if value == 0 { u32::MAX } else if high {
                            value.leading_zeros()
                        } else {
                            value.trailing_zeros()
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::Compare { kind, lhs, rhs, .. }
                    | StageBOp::CompareScalar { kind, lhs, rhs, .. }
                        if active =>
                    {
                        let (lhs, rhs) = (read(lhs)?, read(rhs)?);
                        let hit = match kind {
                            CompareKind::Eq => lhs == rhs,
                            CompareKind::NotEqual => lhs != rhs,
                            CompareKind::Greater => lhs > rhs,
                            CompareKind::GreaterEqual => lhs >= rhs,
                            CompareKind::Less => lhs < rhs,
                            CompareKind::LessEqual => lhs <= rhs,
                            CompareKind::SignedGreater => (lhs as i32) > (rhs as i32),
                            CompareKind::SignedLess => (lhs as i32) < (rhs as i32),
                            CompareKind::EqualFloat => f32::from_bits(lhs) == f32::from_bits(rhs),
                            CompareKind::OrderedNotEqualFloat => {
                                let lhs = f32::from_bits(lhs);
                                let rhs = f32::from_bits(rhs);
                                !lhs.is_nan() && !rhs.is_nan() && lhs != rhs
                            }
                            CompareKind::UnorderedNotEqualFloat => {
                                let lhs = f32::from_bits(lhs);
                                let rhs = f32::from_bits(rhs);
                                lhs.is_nan() || rhs.is_nan() || lhs != rhs
                            }
                            CompareKind::LessFloat => f32::from_bits(lhs) < f32::from_bits(rhs),
                            CompareKind::GreaterFloat => f32::from_bits(lhs) > f32::from_bits(rhs),
                            CompareKind::LessEqualFloat => {
                                f32::from_bits(lhs) <= f32::from_bits(rhs)
                            }
                            CompareKind::GreaterEqualFloat => {
                                f32::from_bits(lhs) >= f32::from_bits(rhs)
                            }
                            CompareKind::UnorderedGreaterFloat => {
                                let lhs = f32::from_bits(lhs);
                                let rhs = f32::from_bits(rhs);
                                lhs.is_nan() || rhs.is_nan() || lhs > rhs
                            }
                            CompareKind::UnorderedLessFloat => {
                                let lhs = f32::from_bits(lhs);
                                let rhs = f32::from_bits(rhs);
                                lhs.is_nan() || rhs.is_nan() || lhs < rhs
                            }
                            CompareKind::UnorderedLessEqualFloat => {
                                let lhs = f32::from_bits(lhs);
                                let rhs = f32::from_bits(rhs);
                                lhs.is_nan() || rhs.is_nan() || lhs <= rhs
                            }
                            CompareKind::LessFloat16 => {
                                f16_to_f32(lhs as u16) < f16_to_f32(rhs as u16)
                            }
                        };
                        predicates |= u64::from(hit) << lane;
                    }
                    StageBOp::CompareEqU64 { lhs, rhs, .. } if active => {
                        let pair = |source: VectorSource| -> Result<u64, &'static str> {
                            let high = match source {
                                VectorSource::Vgpr(register) => VectorSource::Vgpr(register + 1),
                                VectorSource::Scalar(register) => VectorSource::Scalar(register + 1),
                                _ => return Err("64-bit compare source is not paired"),
                            };
                            Ok(read(source)? as u64 | ((read(high)? as u64) << 32))
                        };
                        predicates |= u64::from(pair(lhs)? == pair(rhs)?) << lane;
                    }
                    StageBOp::CompareClassF32 { value: input, classes, modifiers, .. } if active => {
                        let bits = float_source(read(input)?, modifiers, 0).to_bits();
                        let class = f32_class_index(bits);
                        predicates |= u64::from(((read(classes)? >> class) & 1) != 0) << lane;
                    }
                    StageBOp::CompareLessFloat {
                        kind,
                        lhs,
                        rhs,
                        modifiers,
                        ..
                    } if active => {
                        let lhs = float_source(read(lhs)?, modifiers, 0);
                        let rhs = float_source(read(rhs)?, modifiers, 1);
                        let hit = match kind {
                            CompareKind::EqualFloat => lhs == rhs,
                            CompareKind::OrderedNotEqualFloat => {
                                !lhs.is_nan() && !rhs.is_nan() && lhs != rhs
                            }
                            CompareKind::UnorderedNotEqualFloat => {
                                lhs.is_nan() || rhs.is_nan() || lhs != rhs
                            }
                            CompareKind::LessFloat => lhs < rhs,
                            CompareKind::GreaterFloat => lhs > rhs,
                            CompareKind::LessEqualFloat => lhs <= rhs,
                            CompareKind::GreaterEqualFloat => lhs >= rhs,
                            CompareKind::UnorderedGreaterFloat => {
                                lhs.is_nan() || rhs.is_nan() || lhs > rhs
                            }
                            CompareKind::UnorderedLessFloat => {
                                lhs.is_nan() || rhs.is_nan() || lhs < rhs
                            }
                            CompareKind::UnorderedLessEqualFloat => {
                                lhs.is_nan() || rhs.is_nan() || lhs <= rhs
                            }
                            _ => false,
                        };
                        predicates |= u64::from(hit) << lane;
                    }
                    StageBOp::CndMask {
                        dst,
                        when_clear,
                        when_set,
                    } if active => {
                        let selected = if old_vcc & (1u64 << lane) != 0 {
                            when_set
                        } else {
                            when_clear
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, read(selected)?)?;
                    }
                    StageBOp::CndMaskScalar {
                        dst,
                        when_clear,
                        when_set,
                        predicate,
                    } if active => {
                        let selected =
                            if scalar_mask_value(&states[wave], predicate) & (1u64 << lane) != 0 {
                                when_set
                            } else {
                                when_clear
                            };
                        store_vgpr(memory, layout, 0, thread, dst as u32, read(selected)?)?;
                    }
                    StageBOp::ConvertF32U32 {
                        dst,
                        src,
                        modifiers,
                    } if active => {
                        let converted = read(src)? as f32;
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(converted, modifiers),
                        )?;
                    }
                    StageBOp::ConvertF32I32 { dst, src } if active => {
                        let converted = read(src)? as i32 as f32;
                        store_vgpr(memory, layout, 0, thread, dst as u32, converted.to_bits())?;
                    }
                    StageBOp::ConvertF32U32Sdwa { dst, src, word } if active => {
                        let raw = load_vgpr(&snapshot, layout, 0, thread, src as u32)?;
                        let selected = match word {
                            SdwaWord::Low => raw & 0xffff,
                            SdwaWord::High => raw >> 16,
                        };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            (selected as f32).to_bits(),
                        )?;
                    }
                    StageBOp::FloatConvert { kind, dst, src } if active => {
                        let source = read(src)?;
                        let result = match kind {
                            FloatConvertKind::F16FromF32 => {
                                let current = load_vgpr(&snapshot, layout, 0, thread, dst as u32)?;
                                (current & 0xffff_0000) | f32_to_f16(f32::from_bits(source)) as u32
                            }
                            FloatConvertKind::F16FromF32High => {
                                let current = load_vgpr(&snapshot, layout, 0, thread, dst as u32)?;
                                (current & 0x0000_ffff)
                                    | ((f32_to_f16(f32::from_bits(source)) as u32) << 16)
                            }
                            FloatConvertKind::F16FromF32HighPad => {
                                (f32_to_f16(f32::from_bits(source)) as u32) << 16
                            }
                            FloatConvertKind::F32FromF16 => f16_to_f32(source as u16).to_bits(),
                            FloatConvertKind::F32FromF16High => {
                                f16_to_f32((source >> 16) as u16).to_bits()
                            }
                            FloatConvertKind::U32FromF32 => f32::from_bits(source) as u32,
                            FloatConvertKind::I32FromF32 => {
                                (f32::from_bits(source) as i32) as u32
                            }
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::FloatBinary {
                        kind,
                        dst,
                        lhs,
                        rhs,
                        modifiers,
                    } if active => {
                        let lhs = float_source(read(lhs)?, modifiers, 0);
                        let rhs = float_source(read(rhs)?, modifiers, 1);
                        let result = match kind {
                            FloatBinaryKind::Add => lhs + rhs,
                            FloatBinaryKind::Subtract => lhs - rhs,
                            FloatBinaryKind::ReverseSubtract => rhs - lhs,
                            FloatBinaryKind::Multiply => lhs * rhs,
                            FloatBinaryKind::MultiplyLegacy => {
                                if lhs == 0.0 || rhs == 0.0 {
                                    0.0
                                } else {
                                    lhs * rhs
                                }
                            }
                            FloatBinaryKind::Maximum => {
                                if lhs.is_nan() {
                                    rhs
                                } else if rhs.is_nan() || lhs >= rhs {
                                    lhs
                                } else {
                                    rhs
                                }
                            }
                            FloatBinaryKind::Minimum => {
                                if lhs.is_nan() {
                                    rhs
                                } else if rhs.is_nan() || lhs <= rhs {
                                    lhs
                                } else {
                                    rhs
                                }
                            }
                        };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(result, modifiers),
                        )?;
                    }
                    StageBOp::FloatTernary {
                        kind,
                        dst,
                        a,
                        b,
                        c,
                        modifiers,
                    } if active => {
                        let choose_max = |lhs: f32, rhs: f32| {
                            if lhs.is_nan() {
                                rhs
                            } else if rhs.is_nan() || lhs >= rhs {
                                lhs
                            } else {
                                rhs
                            }
                        };
                        let a = float_source(read(a)?, modifiers, 0);
                        let b = float_source(read(b)?, modifiers, 1);
                        let c = float_source(read(c)?, modifiers, 2);
                        let choose_min = |lhs: f32, rhs: f32| {
                            if lhs.is_nan() {
                                rhs
                            } else if rhs.is_nan() || lhs <= rhs {
                                lhs
                            } else {
                                rhs
                            }
                        };
                        let result = match kind {
                            FloatTernaryKind::Minimum3 => choose_min(choose_min(a, b), c),
                            FloatTernaryKind::Maximum3 => choose_max(choose_max(a, b), c),
                            FloatTernaryKind::Median3 => choose_max(
                                choose_min(a, b),
                                choose_min(choose_max(a, b), c),
                            ),
                        };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(result, modifiers),
                        )?;
                    }
                    StageBOp::FloatUnary { kind, dst, src } if active => {
                        let value = f32::from_bits(read(src)?);
                        let result = match kind {
                            FloatUnaryKind::Truncate => value.trunc(),
                            FloatUnaryKind::Floor => value.floor(),
                            FloatUnaryKind::Ceiling => value.ceil(),
                            FloatUnaryKind::RoundEven => value.round_ties_even(),
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result.to_bits())?;
                    }
                    StageBOp::FloatMath {
                        kind,
                        dst,
                        src,
                        modifiers,
                    } if active => {
                        let mut value = float_source(read(src)?, modifiers, 0);
                        if matches!(
                            kind,
                            FloatMathKind::Reciprocal
                                | FloatMathKind::ReciprocalIFlag
                                | FloatMathKind::ReciprocalSqrt
                                | FloatMathKind::Sqrt
                        ) {
                            value = flush_f32_denormal(value);
                        }
                        let result = match kind {
                            FloatMathKind::Reciprocal | FloatMathKind::ReciprocalIFlag => {
                                1.0 / value
                            }
                            FloatMathKind::ReciprocalSqrt => 1.0 / value.sqrt(),
                            FloatMathKind::Sqrt => value.sqrt(),
                            FloatMathKind::Sin => {
                                if value.is_finite() && value.abs() > 256.0 {
                                    0.0
                                } else {
                                    (value * std::f32::consts::TAU).sin()
                                }
                            }
                            FloatMathKind::Cos => {
                                if value.is_finite() && value.abs() > 256.0 {
                                    1.0
                                } else {
                                    (value * std::f32::consts::TAU).cos()
                                }
                            }
                            FloatMathKind::Exp2 => value.exp2(),
                            FloatMathKind::Fract => value - value.floor(),
                            FloatMathKind::Log2 => value.log2(),
                        };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(result, modifiers),
                        )?;
                    }
                    StageBOp::ConvertUByte {
                        dst,
                        src,
                        byte,
                        modifiers,
                    } if active => {
                        let result = ((read(src)? >> (byte as u32 * 8)) & 0xff) as f32;
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(result, modifiers),
                        )?;
                    }
                    StageBOp::PackF16 {
                        dst,
                        lo,
                        hi,
                        op_sel,
                    } if active => {
                        let lo = read(lo)? >> if op_sel & 1 != 0 { 16 } else { 0 };
                        let hi = read(hi)? >> if op_sel & 2 != 0 { 16 } else { 0 };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            (lo & 0xffff) | (hi << 16),
                        )?;
                    }
                    StageBOp::FmaMix {
                        kind,
                        dst,
                        a,
                        b,
                        c,
                        op_sel,
                        op_sel_hi,
                        neg,
                        abs,
                        clamp,
                    } if active => {
                        let a = mix_source(read(a)?, op_sel, op_sel_hi, neg, abs, 0);
                        let b = mix_source(read(b)?, op_sel, op_sel_hi, neg, abs, 1);
                        let c = mix_source(read(c)?, op_sel, op_sel_hi, neg, abs, 2);
                        let mut result = a.mul_add(b, c);
                        if clamp {
                            result = if result.is_nan() {
                                0.0
                            } else {
                                result.max(0.0).min(1.0)
                            };
                        }
                        let old = load_vgpr(&snapshot, layout, 0, thread, dst as u32)?;
                        let packed = f32_to_f16(result) as u32;
                        let result = match kind {
                            FmaMixKind::F32 => result.to_bits(),
                            FmaMixKind::LoF16 => (old & 0xffff_0000) | packed,
                            FmaMixKind::HiF16 => (old & 0x0000_ffff) | (packed << 16),
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    StageBOp::FloatFma {
                        dst,
                        a,
                        b,
                        c,
                        modifiers,
                        legacy_zero,
                    } if active => {
                        let a = float_source(read(a)?, modifiers, 0);
                        let b = float_source(read(b)?, modifiers, 1);
                        let addend = if let Some(c) = c {
                            float_source(read(c)?, modifiers, 2)
                        } else {
                            f32::from_bits(load_vgpr(&snapshot, layout, 0, thread, dst as u32)?)
                        };
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(
                                if legacy_zero && (a == 0.0 || b == 0.0) {
                                    addend
                                } else {
                                    a.mul_add(b, addend)
                                },
                                modifiers,
                            ),
                        )?;
                    }
                    StageBOp::FloatMad {
                        dst,
                        a,
                        b,
                        c,
                        modifiers,
                    } if active => {
                        let a = float_source(read(a)?, modifiers, 0);
                        let b = float_source(read(b)?, modifiers, 1);
                        let addend = if let Some(c) = c {
                            float_source(read(c)?, modifiers, 2)
                        } else {
                            f32::from_bits(load_vgpr(&snapshot, layout, 0, thread, dst as u32)?)
                        };
                        let product = f32::from_bits((a * b).to_bits());
                        store_vgpr(
                            memory,
                            layout,
                            0,
                            thread,
                            dst as u32,
                            float_result(product + addend, modifiers),
                        )?;
                    }
                    StageBOp::StageDInteger { kind, dst, a, b, c } if active => {
                        let (a, b) = (read(a)?, read(b)?);
                        let result = match kind {
                            StageDIntegerKind::LogicalShiftRight => b >> (a & 31),
                            StageDIntegerKind::MadU24 => (a & 0xff_ffff)
                                .wrapping_mul(b & 0xff_ffff)
                                .wrapping_add(read(c.ok_or("Stage D addend is missing")?)?),
                            StageDIntegerKind::Permute => {
                                permute_bytes(a, b, read(c.ok_or("Stage D selector is missing")?)?)
                            }
                        };
                        store_vgpr(memory, layout, 0, thread, dst as u32, result)?;
                    }
                    _ => {}
                }
            }
            if let StageBOp::Compare { cmpx, .. } = op {
                if cmpx {
                    let exec = old_exec & predicates;
                    states[wave].exec = [exec as u32, (exec >> 32) as u32];
                } else {
                    states[wave].vcc = [predicates as u32, (predicates >> 32) as u32];
                }
            }
            if let StageBOp::CompareScalar { dst, .. } = op {
                states[wave].scalar[dst as usize] = predicates as u32;
                states[wave].scalar[dst as usize + 1] = (predicates >> 32) as u32;
            }
            if let StageBOp::CompareEqU64 { dst, .. } = op {
                if let Some(dst) = dst {
                    states[wave].scalar[dst as usize] = predicates as u32;
                    states[wave].scalar[dst as usize + 1] = (predicates >> 32) as u32;
                } else {
                    states[wave].vcc = [predicates as u32, (predicates >> 32) as u32];
                }
            }
            if let StageBOp::CompareClassF32 { dst, .. } = op {
                if let Some(dst) = dst {
                    states[wave].scalar[dst as usize] = predicates as u32;
                    states[wave].scalar[dst as usize + 1] = (predicates >> 32) as u32;
                } else {
                    states[wave].vcc = [predicates as u32, (predicates >> 32) as u32];
                }
            }
            if let StageBOp::CompareLessFloat { dst, cmpx, .. } = op {
                if cmpx {
                    let exec = old_exec & predicates;
                    states[wave].exec = [exec as u32, (exec >> 32) as u32];
                } else if let Some(dst) = dst {
                    states[wave].scalar[dst as usize] = predicates as u32;
                    states[wave].scalar[dst as usize + 1] = (predicates >> 32) as u32;
                } else {
                    states[wave].vcc = [predicates as u32, (predicates >> 32) as u32];
                }
            }
            if let StageBOp::AddCarry { carry_out, .. }
            | StageBOp::ReverseSubCarry { carry_out, .. } = op {
                match carry_out {
                    ScalarMask::ScalarPair(dst) => {
                        states[wave].scalar[dst as usize] = predicates as u32;
                        states[wave].scalar[dst as usize + 1] = (predicates >> 32) as u32;
                    }
                    ScalarMask::Vcc => states[wave].vcc = [predicates as u32, (predicates >> 32) as u32],
                    ScalarMask::Exec => unreachable!("EXEC carry destination rejected by lowering"),
                }
            }
        }
    }
    Ok(())
}

pub fn initial_vgpr_value(register: u32, thread: u32) -> u32 {
    match register {
        0 => thread % LOGICAL_WAVE_LANES,
        1 => 0xf0f0_0000u32 ^ thread.wrapping_mul(0x0101_0101),
        _ => 0xa5a5_0000u32 ^ register.wrapping_mul(0x1111) ^ thread,
    }
}

fn hlsl_source(source: VectorSource, stride: u32) -> String {
    match source {
        VectorSource::Vgpr(register) => {
            format!("scratch.Load(lane_base + {}u)", register as u32 * 4)
        }
        VectorSource::Constant(value) => format!("{value}u"),
        VectorSource::Scalar(106) => "vcc.x".into(),
        VectorSource::Scalar(107) => "vcc.y".into(),
        VectorSource::Scalar(126) => "exec.x".into(),
        VectorSource::Scalar(127) => "exec.y".into(),
        VectorSource::Scalar(125) => "0u".into(),
        VectorSource::Scalar(register) => {
            format!("scratch.Load(state_base + 36u + {}u)", register as u32 * 4)
        }
        VectorSource::Sdwa {
            register,
            scalar,
            select,
            sign_extend,
        } => {
            let raw = hlsl_source(
                if scalar {
                    VectorSource::Scalar(register)
                } else {
                    VectorSource::Vgpr(register as u16)
                },
                stride,
            );
            let (shift, width) = match select {
                0..=3 => (select as u32 * 8, 8),
                4..=5 => ((select as u32 - 4) * 16, 16),
                6 => (0, 32),
                _ => unreachable!("validated SDWA selector"),
            };
            if width == 32 {
                raw
            } else if sign_extend {
                format!("(uint)(((int)(({raw}>>{shift}u)<<(32u-{width}u)))>>(32u-{width}u))")
            } else {
                format!("(({raw}>>{shift}u)&{}u)", u32::MAX >> (32 - width))
            }
        }
        VectorSource::Dpp16RowMirror {
            register,
            fetch_inactive,
        } => {
            let source_lane = "((lane&~15u)|(15u-(lane&15u)))";
            let value = format!(
                "scratch.Load(vgpr_base+(wave*64u+{source_lane})*{stride}u+{}u)",
                register as u32 * 4
            );
            if fetch_inactive {
                value
            } else {
                format!("((((({source_lane})<32u?exec.x:exec.y)>>( {source_lane}&31u))&1u)!=0u?{value}:0u)")
            }
        }
        VectorSource::Dpp8 {
            register,
            lanes,
            fetch_inactive,
        } => {
            let packed = lanes
                .into_iter()
                .enumerate()
                .fold(0u32, |value, (index, lane)| {
                    value | (lane as u32) << (index * 3)
                });
            let source_lane = format!("((lane&~7u)|(({packed}u>>(3u*(lane&7u)))&7u))");
            let value = format!(
                "scratch.Load(vgpr_base+(wave*64u+{source_lane})*{stride}u+{}u)",
                register as u32 * 4
            );
            if fetch_inactive {
                value
            } else {
                format!("((((({source_lane})<32u?exec.x:exec.y)>>({source_lane}&31u))&1u)!=0u?{value}:0u)")
            }
        }
        VectorSource::VccZero => "(uint)(all(vcc==0u))".into(),
        VectorSource::ExecZero => "(uint)(all(exec==0u))".into(),
        VectorSource::Scc => "scratch.Load(state_base+32u)".into(),
    }
}

fn hlsl_mask_bit(source: ScalarMask) -> String {
    match source {
        ScalarMask::ScalarPair(register) => format!(
            "((scratch.Load(state_base+36u+{}u+(lane>>5u)*4u)>> (lane&31u))&1u)",
            register as u32 * 4
        ),
        ScalarMask::Vcc => "vcc_bit".into(),
        ScalarMask::Exec => "((lane<32u?exec.x:exec.y)&bit)".into(),
    }
}

fn hlsl_float_source(
    source: VectorSource,
    stride: u32,
    modifiers: FloatModifiers,
    index: u8,
) -> String {
    let mut value = format!("asfloat({})", hlsl_source(source, stride));
    if modifiers.abs & (1 << index) != 0 {
        value = format!("abs({value})");
    }
    if modifiers.neg & (1 << index) != 0 {
        value = format!("(-{value})");
    }
    value
}

fn hlsl_float_result(mut value: String, modifiers: FloatModifiers) -> String {
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

fn f32_class_index(bits: u32) -> u32 {
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    let negative = bits >> 31 != 0;
    if exponent == 0xff {
        if mantissa != 0 { if mantissa & 0x40_0000 != 0 { 1 } else { 0 } }
        else if negative { 2 } else { 9 }
    } else if exponent == 0 {
        if mantissa == 0 { if negative { 5 } else { 6 } }
        else if negative { 4 } else { 7 }
    } else if negative { 3 } else { 8 }
}

fn hlsl_op(op: StageBOp, stride: u32) -> String {
    let store = |dst: u16, value: String| {
        format!(
            "if(active)scratch.Store(lane_base+{}u,{value});",
            dst as u32 * 4
        )
    };
    match op {
        StageBOp::Move { dst, src } => store(dst, hlsl_source(src, stride)),
        StageBOp::RelativeMove { dst, base } => format!(
            "{{uint relative=scratch.Load(state_base+532u);uint selected=relative<{}u&&relative+{}u<{}u?scratch.Load(lane_base+(relative+{}u)*4u):0u;{}}}",
            u32::MAX - base as u32,
            base,
            stride / 4,
            base,
            store(dst, "selected".to_owned())
        ),
        StageBOp::ReadFirstLane { dst, src } => format!(
            "if(lane==0u){{uint first=exec.x!=0u?firstbitlow(exec.x):32u+firstbitlow(exec.y);scratch.Store(state_base+36u+{}u,scratch.Load(vgpr_base+(wave*64u+first)*{}u+{}u));}}",
            dst as u32 * 4,
            stride,
            src as u32 * 4
        ),
        StageBOp::ReadLane { dst, src, lane: selected_lane } => format!(
            "if(lane==0u){{uint selected={}&63u;scratch.Store(state_base+36u+{}u,scratch.Load(vgpr_base+(wave*64u+selected)*{}u+{}u));}}",
            hlsl_source(selected_lane, stride),
            dst as u32 * 4,
            stride,
            src as u32 * 4
        ),
        StageBOp::WriteLane { dst, src, lane: selected_lane } => format!(
            "if(lane==({}&63u))scratch.Store(lane_base+{}u,{});",
            hlsl_source(selected_lane, stride),
            dst as u32 * 4,
            hlsl_source(src, stride)
        ),
        StageBOp::PermLaneX16 {
            dst,
            src,
            selectors_low,
            selectors_high,
            fetch_inactive,
            bound_control,
        } => {
            let low = hlsl_source(selectors_low, stride);
            let high = hlsl_source(selectors_high, stride);
            let fallback = if bound_control {
                format!("scratch.Store(lane_base+{}u,0u);", dst as u32 * 4)
            } else {
                String::new()
            };
            format!(
                "if(active){{uint sels=((lane&15u)<8u)?({low}):({high});uint n=(sels>>(4u*(lane&7u)))&15u;uint selected=(lane&~31u)|(((lane&16u)^16u)&16u)|n;uint sbit=1u<<(selected&31u);bool sok={}||(((selected<32u?exec.x:exec.y)&sbit)!=0u);if(sok)scratch.Store(lane_base+{}u,scratch.Load(vgpr_base+(wave*64u+selected)*{}u+{}u));else{{{fallback}}}}}",
                if fetch_inactive { "true" } else { "false" },
                dst as u32 * 4,
                stride,
                src as u32 * 4,
            )
        }
        StageBOp::Mbcnt {
            high,
            dst,
            source,
            accumulator,
        } => {
            let source = hlsl_source(source, stride);
            let accumulator = hlsl_source(accumulator, stride);
            let count = if high {
                format!("(lane<32u?0u:countbits({source}&((lane==32u)?0u:((1u<<(lane-32u))-1u))))")
            } else {
                format!("countbits({source}&((lane==0u)?0u:(lane>=32u?0xffffffffu:((1u<<lane)-1u))))")
            };
            store(dst, format!("({accumulator}+{count})"))
        }
        StageBOp::Binary {
            kind,
            dst,
            lhs,
            rhs,
        } => {
            let (a, b) = (hlsl_source(lhs, stride), hlsl_source(rhs, stride));
            store(
                dst,
                match kind {
                    VectorBinaryKind::AddNoCarry => format!("({a}+{b})"),
                    VectorBinaryKind::And => format!("({a}&{b})"),
                    VectorBinaryKind::Or => format!("({a}|{b})"),
                    VectorBinaryKind::Xor => format!("({a}^{b})"),
                    VectorBinaryKind::BitCount => format!("(countbits({a})+{b})"),
                    VectorBinaryKind::LshlReverse => format!("({b}<<({a}&31u))"),
                    VectorBinaryKind::ArithmeticShiftRightReverse => {
                        format!("asuint((int){b}>>(int)({a}&31u))")
                    }
                    VectorBinaryKind::MinU32 => format!("min({a},{b})"),
                    VectorBinaryKind::MaxU32 => format!("max({a},{b})"),
                    VectorBinaryKind::MulLoU32 => format!("({a}*{b})"),
                    VectorBinaryKind::MulHiU32 => {
                        format!("(uint)(((uint64_t){a}*(uint64_t){b})>>32u)")
                    }
                    VectorBinaryKind::SubNoCarry => format!("({a}-{b})"),
                    VectorBinaryKind::ReverseSubNoCarry => format!("({b}-{a})"),
                },
            )
        }
        StageBOp::AddCarry { dst, lhs, rhs, carry_in, .. } => {
            let (a, b) = (hlsl_source(lhs, stride), hlsl_source(rhs, stride));
            let carry = carry_in.map(|source| format!("(uint)({}!=0u)", hlsl_mask_bit(source))).unwrap_or_else(|| "0u".into());
            format!("uint64_t carry_sum=(uint64_t){a}+(uint64_t){b}+(uint64_t){carry};if(active)scratch.Store(lane_base+{}u,(uint)carry_sum);scratch.Store(mailbox+gi*4u,(uint)(active&&((carry_sum>>32u)!=0u)));", dst as u32 * 4)
        }
        StageBOp::ReverseSubCarry { dst, lhs, rhs, carry_in, .. } => {
            let (a, b) = (hlsl_source(lhs, stride), hlsl_source(rhs, stride));
            let carry = format!("(uint)({}!=0u)", hlsl_mask_bit(carry_in));
            format!("uint64_t carry_difference=(uint64_t){b}-(uint64_t){a}-(uint64_t){carry};if(active)scratch.Store(lane_base+{}u,(uint)carry_difference);scratch.Store(mailbox+gi*4u,(uint)(active&&((carry_difference>>32u)!=0u)));", dst as u32 * 4)
        }
        StageBOp::Ternary { kind, dst, a, b, c } => {
            let (a, b, c) = (
                hlsl_source(a, stride),
                hlsl_source(b, stride),
                hlsl_source(c, stride),
            );
            store(
                dst,
                match kind {
                    VectorTernaryKind::Add3 => format!("({a}+{b}+{c})"),
                    VectorTernaryKind::BfeU32 => format!("bfe_u({a},{b},{c})"),
                    VectorTernaryKind::LshlAdd => format!("(({a}<<({b}&31u))+{c})"),
                    VectorTernaryKind::AddLshl => format!("(({a}+{b})<<({c}&31u))"),
                    VectorTernaryKind::AndOr => format!("(({a}&{b})|{c})"),
                    VectorTernaryKind::Or3 => format!("({a}|{b}|{c})"),
                    VectorTernaryKind::LshlOr => format!("(({a}<<({b}&31u))|{c})"),
                    VectorTernaryKind::BitFieldInsert => format!("(({a}&{b})|(~{a}&{c}))"),
                    VectorTernaryKind::XorAdd => format!("(({a}^{b})+{c})"),
                },
            )
        }
        StageBOp::BitReverse { dst, src } => {
            store(dst, format!("reversebits({})", hlsl_source(src, stride)))
        }
        StageBOp::BitNot { dst, src } => {
            store(dst, format!("~{}", hlsl_source(src, stride)))
        }
        StageBOp::BitScan { high, dst, src } => {
            let source = hlsl_source(src, stride);
            store(dst, if high {
                format!("(({source})==0u?0xFFFFFFFFu:31u-firstbithigh({source}))")
            } else {
                format!("(({source})==0u?0xFFFFFFFFu:firstbitlow({source}))")
            })
        }
        StageBOp::CndMask {
            dst,
            when_clear,
            when_set,
        } => store(
            dst,
            format!(
                "((vcc_bit!=0u)?{}:{})",
                hlsl_source(when_set, stride),
                hlsl_source(when_clear, stride)
            ),
        ),
        StageBOp::CndMaskScalar {
            dst,
            when_clear,
            when_set,
            predicate,
        } => store(
            dst,
            format!(
                "(({}!=0u)?{}:{})",
                hlsl_mask_bit(predicate),
                hlsl_source(when_set, stride),
                hlsl_source(when_clear, stride)
            ),
        ),
        StageBOp::Compare { kind, lhs, rhs, .. }
        | StageBOp::CompareScalar { kind, lhs, rhs, .. } => {
            let (a, b) = (hlsl_source(lhs, stride), hlsl_source(rhs, stride));
            let expression = match kind {
                CompareKind::Eq => format!("{a}=={b}"),
                CompareKind::NotEqual => format!("{a}!={b}"),
                CompareKind::Greater => format!("{a}>{b}"),
                CompareKind::GreaterEqual => format!("{a}>={b}"),
                CompareKind::Less => format!("{a}<{b}"),
                CompareKind::LessEqual => format!("{a}<={b}"),
                CompareKind::SignedGreater => format!("(int){a}>(int){b}"),
                CompareKind::SignedLess => format!("(int){a}<(int){b}"),
                CompareKind::EqualFloat => format!("asfloat({a})==asfloat({b})"),
                CompareKind::OrderedNotEqualFloat => format!(
                    "(!isnan(asfloat({a}))&&!isnan(asfloat({b}))&&asfloat({a})!=asfloat({b}))"
                ),
                CompareKind::UnorderedNotEqualFloat => format!(
                    "(isnan(asfloat({a}))||isnan(asfloat({b}))||asfloat({a})!=asfloat({b}))"
                ),
                CompareKind::LessFloat => format!("asfloat({a})<asfloat({b})"),
                CompareKind::GreaterFloat => format!("asfloat({a})>asfloat({b})"),
                CompareKind::LessEqualFloat => format!("asfloat({a})<=asfloat({b})"),
                CompareKind::GreaterEqualFloat => format!("asfloat({a})>=asfloat({b})"),
                CompareKind::UnorderedGreaterFloat => format!(
                    "(isnan(asfloat({a}))||isnan(asfloat({b}))||asfloat({a})>asfloat({b}))"
                ),
                CompareKind::UnorderedLessFloat => format!(
                    "(isnan(asfloat({a}))||isnan(asfloat({b}))||asfloat({a})<asfloat({b}))"
                ),
                CompareKind::UnorderedLessEqualFloat => format!(
                    "(isnan(asfloat({a}))||isnan(asfloat({b}))||asfloat({a})<=asfloat({b}))"
                ),
                CompareKind::LessFloat16 => {
                    format!("f16tof32({a}&0xffffu)<f16tof32({b}&0xffffu)")
                }
            };
            format!("scratch.Store(mailbox+gi*4u,(uint)(active&&({expression})));")
        }
        StageBOp::CompareLessFloat {
            kind,
            lhs,
            rhs,
            modifiers,
            ..
        } => {
            let lhs = hlsl_float_source(lhs, stride, modifiers, 0);
            let rhs = hlsl_float_source(rhs, stride, modifiers, 1);
            let expression = match kind {
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
                CompareKind::UnorderedGreaterFloat => {
                    format!("(isnan({lhs})||isnan({rhs})||{lhs}>{rhs})")
                }
                CompareKind::UnorderedLessFloat => {
                    format!("(isnan({lhs})||isnan({rhs})||{lhs}<{rhs})")
                }
                CompareKind::UnorderedLessEqualFloat => {
                    format!("(isnan({lhs})||isnan({rhs})||{lhs}<={rhs})")
                }
                _ => "false".into(),
            };
            format!("scratch.Store(mailbox+gi*4u,(uint)(active&&({expression})));" )
        }
        StageBOp::CompareEqU64 { lhs, rhs, .. } => {
            let high = |source| match source {
                VectorSource::Vgpr(register) => VectorSource::Vgpr(register + 1),
                VectorSource::Scalar(register) => VectorSource::Scalar(register + 1),
                _ => unreachable!("validated paired source"),
            };
            let (lo0, hi0) = (hlsl_source(lhs, stride), hlsl_source(high(lhs), stride));
            let (lo1, hi1) = (hlsl_source(rhs, stride), hlsl_source(high(rhs), stride));
            format!("scratch.Store(mailbox+gi*4u,(uint)(active&&({lo0}=={lo1})&&({hi0}=={hi1})));" )
        }
        StageBOp::CompareClassF32 { value, classes, modifiers, .. } => {
            let value = hlsl_float_source(value, stride, modifiers, 0);
            let classes = hlsl_source(classes, stride);
            format!("uint class_bits=asuint({value}),class_exp=(class_bits>>23u)&255u,class_mant=class_bits&0x7fffffu,class_index;if(class_exp==255u)class_index=class_mant!=0u?((class_mant&0x400000u)!=0u?1u:0u):((class_bits>>31u)!=0u?2u:9u);else if(class_exp==0u)class_index=class_mant==0u?((class_bits>>31u)!=0u?5u:6u):((class_bits>>31u)!=0u?4u:7u);else class_index=(class_bits>>31u)!=0u?3u:8u;scratch.Store(mailbox+gi*4u,(uint)(active&&((({classes}>>class_index)&1u)!=0u)));" )
        }
        StageBOp::ConvertF32U32 {
            dst,
            src,
            modifiers,
        } => store(
            dst,
            hlsl_float_result(format!("(float){}", hlsl_source(src, stride)), modifiers),
        ),
        StageBOp::ConvertF32I32 { dst, src } => store(
            dst,
            format!("asuint((float)(int){})", hlsl_source(src, stride)),
        ),
        StageBOp::ConvertF32U32Sdwa { dst, src, word } => {
            let value = hlsl_source(VectorSource::Vgpr(src), stride);
            let selected = match word {
                SdwaWord::Low => format!("({value}&0xffffu)"),
                SdwaWord::High => format!("({value}>>16u)"),
            };
            store(dst, format!("asuint((float){selected})"))
        }
        StageBOp::FloatConvert { kind, dst, src } => {
            let source = hlsl_source(src, stride);
            let value = match kind {
                FloatConvertKind::F16FromF32 => format!(
                    "(scratch.Load(lane_base+{}u)&0xffff0000u)|(f32_to_f16_bounded({source})&0xffffu)",
                    dst as u32 * 4
                ),
                FloatConvertKind::F16FromF32High => format!(
                    "(scratch.Load(lane_base+{}u)&0xffffu)|((f32_to_f16_bounded({source})&0xffffu)<<16u)",
                    dst as u32 * 4
                ),
                FloatConvertKind::F16FromF32HighPad => {
                    format!("(f32_to_f16_bounded({source})&0xffffu)<<16u")
                }
                FloatConvertKind::F32FromF16 => {
                    format!("asuint(f16tof32({source}&0xffffu))")
                }
                FloatConvertKind::F32FromF16High => {
                    format!("asuint(f16tof32({source}>>16u))")
                }
                FloatConvertKind::U32FromF32 => format!("cvt_u32(asfloat({source}))"),
                FloatConvertKind::I32FromF32 => format!("(uint)cvt_i32(asfloat({source}))"),
            };
            store(dst, value)
        }
        StageBOp::FloatBinary {
            kind,
            dst,
            lhs,
            rhs,
            modifiers,
        } => {
            let a = hlsl_float_source(lhs, stride, modifiers, 0);
            let b = hlsl_float_source(rhs, stride, modifiers, 1);
            let expression = match kind {
                FloatBinaryKind::Add => format!("{a}+{b}"),
                FloatBinaryKind::Subtract => format!("{a}-{b}"),
                FloatBinaryKind::ReverseSubtract => format!("{b}-{a}"),
                FloatBinaryKind::Multiply => format!("{a}*{b}"),
                FloatBinaryKind::MultiplyLegacy => format!("mul_legacy({a},{b})"),
                FloatBinaryKind::Maximum => format!("max({a},{b})"),
                FloatBinaryKind::Minimum => format!("min({a},{b})"),
            };
            store(dst, hlsl_float_result(expression, modifiers))
        }
        StageBOp::FloatTernary { kind, dst, a, b, c, modifiers } => {
            let a = hlsl_float_source(a, stride, modifiers, 0);
            let b = hlsl_float_source(b, stride, modifiers, 1);
            let c = hlsl_float_source(c, stride, modifiers, 2);
            let expression = match kind {
                FloatTernaryKind::Minimum3 => format!("min(min({a},{b}),{c})"),
                FloatTernaryKind::Maximum3 => format!("max(max({a},{b}),{c})"),
                FloatTernaryKind::Median3 => {
                    format!("max(min({a},{b}),min(max({a},{b}),{c}))")
                }
            };
            store(dst, hlsl_float_result(expression, modifiers))
        }
        StageBOp::FloatUnary { kind, dst, src } => {
            let source = format!("asfloat({})", hlsl_source(src, stride));
            let expression = match kind {
                FloatUnaryKind::Truncate => format!("trunc({source})"),
                FloatUnaryKind::Floor => format!("floor({source})"),
                FloatUnaryKind::Ceiling => format!("ceil({source})"),
                FloatUnaryKind::RoundEven => format!("round({source})"),
            };
            store(dst, format!("asuint({expression})"))
        }
        StageBOp::FloatMath { kind, dst, src, modifiers } => {
            let source = hlsl_float_source(src, stride, modifiers, 0);
            let source = if matches!(
                kind,
                FloatMathKind::Reciprocal
                    | FloatMathKind::ReciprocalIFlag
                    | FloatMathKind::ReciprocalSqrt
                    | FloatMathKind::Sqrt
            ) {
                format!(
                    "(((asuint({source})&0x7f800000u)==0u)?asfloat(asuint({source})&0x80000000u):{source})"
                )
            } else {
                source
            };
            let expression = match kind {
                FloatMathKind::Reciprocal | FloatMathKind::ReciprocalIFlag => format!("rcp({source})"),
                FloatMathKind::ReciprocalSqrt => format!("rsqrt({source})"),
                FloatMathKind::Sqrt => format!("sqrt({source})"),
                FloatMathKind::Sin => format!(
                    "(isfinite({source})&&abs({source})>256.0f?0.0f:sin({source}*6.2831853f))"
                ),
                FloatMathKind::Cos => format!(
                    "(isfinite({source})&&abs({source})>256.0f?1.0f:cos({source}*6.2831853f))"
                ),
                FloatMathKind::Exp2 => format!("exp2({source})"),
                FloatMathKind::Fract => format!("frac({source})"),
                FloatMathKind::Log2 => format!("log2({source})"),
            };
            store(dst, hlsl_float_result(expression, modifiers))
        }
        StageBOp::ConvertUByte { dst, src, byte, modifiers } => {
            let source = hlsl_source(src, stride);
            store(dst, hlsl_float_result(format!("(float)(({source}>>{}u)&0xffu)", byte as u32 * 8), modifiers))
        }
        StageBOp::PackF16 { dst, lo, hi, op_sel } => {
            let lo = hlsl_source(lo, stride);
            let hi = hlsl_source(hi, stride);
            let lo = if op_sel & 1 != 0 { format!("({lo}>>16u)") } else { format!("({lo}&0xffffu)") };
            let hi = if op_sel & 2 != 0 { format!("({hi}>>16u)") } else { format!("({hi}&0xffffu)") };
            store(dst, format!("({lo}|({hi}<<16u))"))
        }
        StageBOp::FmaMix { kind, dst, a, b, c, op_sel, op_sel_hi, neg, abs, clamp } => {
            let mix = |src: VectorSource, index: u8| {
                let raw = hlsl_source(src, stride);
                let mut value = if op_sel_hi & (1 << index) != 0 {
                    let half = if op_sel & (1 << index) != 0 { format!("({raw}>>16u)") } else { format!("({raw}&0xffffu)") };
                    format!("f16tof32({half})")
                } else { format!("asfloat({raw})") };
                if abs & (1 << index) != 0 { value = format!("abs({value})"); }
                if neg & (1 << index) != 0 { value = format!("(-{value})"); }
                value
            };
            let (a, b, c) = (mix(a, 0), mix(b, 1), mix(c, 2));
            let mut expression = format!("mad({a},{b},{c})");
            if clamp { expression = format!("saturate({expression})"); }
            let expression = match kind {
                FmaMixKind::F32 => format!("asuint({expression})"),
                FmaMixKind::LoF16 => format!("(scratch.Load(lane_base+{}u)&0xffff0000u)|(f32_to_f16_bounded(asuint({expression}))&0xffffu)", dst as u32 * 4),
                FmaMixKind::HiF16 => format!("(scratch.Load(lane_base+{}u)&0xffffu)|(f32_to_f16_bounded(asuint({expression}))<<16u)", dst as u32 * 4),
            };
            store(dst, expression)
        }
        StageBOp::FloatFma {
            dst,
            a,
            b,
            c,
            modifiers,
            legacy_zero,
        } => {
            let a = hlsl_float_source(a, stride, modifiers, 0);
            let b = hlsl_float_source(b, stride, modifiers, 1);
            let c = c
                .map(|value| hlsl_float_source(value, stride, modifiers, 2))
                .unwrap_or_else(|| format!("asfloat(scratch.Load(lane_base+{}u))", dst as u32 * 4));
            let expression = if legacy_zero {
                format!("fma_legacy({a},{b},{c})")
            } else {
                format!("mad({a},{b},{c})")
            };
            store(dst, hlsl_float_result(expression, modifiers))
        }
        StageBOp::FloatMad {
            dst,
            a,
            b,
            c,
            modifiers,
        } => {
            let a = hlsl_float_source(a, stride, modifiers, 0);
            let b = hlsl_float_source(b, stride, modifiers, 1);
            let c = c
                .map(|value| hlsl_float_source(value, stride, modifiers, 2))
                .unwrap_or_else(|| format!("asfloat(scratch.Load(lane_base+{}u))", dst as u32 * 4));
            store(
                dst,
                hlsl_float_result(format!("mad_nonfused({a},{b},{c})"), modifiers),
            )
        }
        StageBOp::StageDInteger { kind, dst, a, b, c } => {
            let (a, b) = (hlsl_source(a, stride), hlsl_source(b, stride));
            let expression = match kind {
                StageDIntegerKind::LogicalShiftRight => format!("{b}>>({a}&31u)"),
                StageDIntegerKind::MadU24 => format!(
                    "({a}&0xffffffu)*({b}&0xffffffu)+{}",
                    hlsl_source(c.expect("validated Stage D addend"), stride)
                ),
                StageDIntegerKind::Permute => format!(
                    "perm_b32({a},{b},{})",
                    hlsl_source(c.expect("validated Stage D selector"), stride)
                ),
            };
            store(dst, expression)
        }
        StageBOp::NoEffect => String::new(),
    }
}

fn stage_b_validation_hlsl_with_vcc(
    layout: VgprScratchLayout,
    ops: &[StageBOp],
    exec_masks: [u64; 2],
    initial_vcc: [u64; 2],
) -> Result<String, &'static str> {
    stage_b_validation_hlsl_with_vcc_and_m0(layout, ops, exec_masks, initial_vcc, [0; 2])
}

fn stage_b_validation_hlsl_with_vcc_and_m0(
    layout: VgprScratchLayout,
    ops: &[StageBOp],
    exec_masks: [u64; 2],
    initial_vcc: [u64; 2],
    initial_m0: [u32; 2],
) -> Result<String, &'static str> {
    layout.validate_for_groups(1)?;
    validate_ops(layout, ops)?;
    if ops
        .iter()
        .any(|op| matches!(op, StageBOp::ReadFirstLane { .. }))
        && exec_masks.contains(&0)
    {
        return Err("readfirstlane requires a nonempty EXEC mask");
    }
    let stride = layout.vgpr_count * 4;
    let mut body = String::new();
    for &op in ops {
        let compare = matches!(
            op,
            StageBOp::Compare { .. }
                | StageBOp::CompareScalar { .. }
                | StageBOp::CompareEqU64 { .. }
                | StageBOp::CompareClassF32 { .. }
                | StageBOp::CompareLessFloat { .. }
                | StageBOp::AddCarry { .. }
                | StageBOp::ReverseSubCarry { .. }
        );
        body.push_str(&format!("{{uint2 exec=scratch.Load2(state_base+4u),vcc=scratch.Load2(state_base+12u);uint bit=1u<<(lane&31u);bool active=((lane<32u?exec.x:exec.y)&bit)!=0u;uint vcc_bit=(lane<32u?vcc.x:vcc.y)&bit;{}\nDeviceMemoryBarrierWithGroupSync();\n",hlsl_op(op,stride)));
        if let StageBOp::Compare { cmpx, .. } = op {
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}{} }}",if cmpx{"scratch.Store2(state_base+4u,exec&gathered);"}else{"scratch.Store2(state_base+12u,gathered);"}));
        }
        if let StageBOp::CompareScalar { dst, .. } = op {
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}scratch.Store2(state_base+36u+{}u,gathered);}}", dst as u32 * 4));
        }
        if let StageBOp::CompareEqU64 { dst, .. } = op {
            let target = dst
                .map(|register| format!("state_base+36u+{}u", register as u32 * 4))
                .unwrap_or_else(|| "state_base+12u".to_owned());
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}scratch.Store2({target},gathered);}}"));
        }
        if let StageBOp::CompareClassF32 { dst, .. } = op {
            let target = dst
                .map(|register| format!("state_base+36u+{}u", register as u32 * 4))
                .unwrap_or_else(|| "state_base+12u".to_owned());
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}scratch.Store2({target},gathered);}}"));
        }
        if let StageBOp::CompareLessFloat { dst, cmpx, .. } = op {
            let target = if let Some(dst) = dst {
                format!("state_base+36u+{}u", dst as u32 * 4)
            } else {
                "state_base+12u".to_owned()
            };
            let write = if cmpx {
                "scratch.Store2(state_base+4u,exec&gathered);".to_owned()
            } else {
                format!("scratch.Store2({target},gathered);")
            };
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}{write}}}"));
        }
        if let StageBOp::AddCarry { carry_out, .. }
        | StageBOp::ReverseSubCarry { carry_out, .. } = op {
            let target = match carry_out {
                ScalarMask::ScalarPair(register) => format!("state_base+36u+{}u", register as u32 * 4),
                ScalarMask::Vcc => "state_base+12u".to_owned(),
                ScalarMask::Exec => unreachable!("EXEC carry destination rejected by lowering"),
            };
            body.push_str(&format!("if(lane==0u){{uint2 gathered=uint2(0u,0u);[unroll]for(uint i=0u;i<64u;i++){{if(scratch.Load(mailbox+(wave*64u+i)*4u)!=0u){{if(i<32u)gathered.x|=1u<<i;else gathered.y|=1u<<(i-32u);}}}}scratch.Store2({target},gathered);}}"));
        }
        if compare {
            body.push('\n');
        }
        body.push_str("DeviceMemoryBarrierWithGroupSync();}\n");
    }
    Ok(format!(
        r#"RWStructuredBuffer<uint> io:register(u0);RWByteAddressBuffer records:register(u1);RWByteAddressBuffer scratch:register(u6);
uint bfe_u(uint v,uint o,uint w){{o&=31u;w&=31u;return w==0u?0u:(v>>o)&(0xffffffffu>>((32u-w)&31u));}}
float mul_legacy(float a,float b){{return (a==0.0f||b==0.0f)?0.0f:a*b;}}
float fma_legacy(float a,float b,float c){{return (a==0.0f||b==0.0f)?c:mad(a,b,c);}}
float mad_nonfused(float a,float b,float c){{precise float p=a*b;precise float r=p+c;return r;}}
uint cvt_u32(float f){{return isnan(f)||f<=0.0f?0u:(f>=4294967296.0f?0xffffffffu:(uint)f);}}
int cvt_i32(float f){{return isnan(f)?0:(f<=-2147483648.0f?(int)0x80000000u:(f>=2147483648.0f?0x7fffffff:(int)f));}}
uint f32_to_f16_bounded(uint v){{if((v&0x7f800000u)==0x7f800000u&&(v&0x007fffffu)!=0u)return 0x7e00u;return f32tof16(asfloat(v));}}
uint perm_b32(uint a,uint b,uint s){{uint r=0u;[unroll]for(uint i=0u;i<4u;i++){{uint k=(s>>(8u*i))&255u,v;if(k>=13u)v=255u;else if(k==12u)v=0u;else if(k>=8u){{uint q=k==8u?b>>15:(k==9u?b>>31:(k==10u?a>>15:a>>31));v=(q&1u)!=0u?255u:0u;}}else v=((k<4u?b>>(8u*k):a>>(8u*(k-4u)))&255u);r|=v<<(8u*i);}}return r;}}
[numthreads({GROUP_THREADS},1,1)]void main(uint gi:SV_GroupIndex,uint3 gid:SV_GroupID){{uint wave=gi/64u,lane=gi%64u,base=gid.x*io[0],state_base=base+wave*548u,vgpr_base=base+{vgpr_offset}u,lane_base=vgpr_base+gi*{stride}u,mailbox=base+{mailbox_offset}u;for(uint r=0u;r<{vgprs}u;r++){{uint value=r==0u?lane:(r==1u?0xf0f00000u^gi*0x01010101u:0xa5a50000u^r*0x1111u^gi);scratch.Store(lane_base+r*4u,value);}}scratch.Store(state_base+36u+lane*4u,0u);scratch.Store(state_base+36u+(lane+64u)*4u,0u);if(lane==0u){{uint64_t m=wave==0u?{mask0}ull:{mask1}ull,c=wave==0u?{vcc0}ull:{vcc1}ull;scratch.Store4(state_base,uint4(0u,(uint)m,(uint)(m>>32),(uint)c));scratch.Store4(state_base+16u,uint4((uint)(c>>32),0u,0u,0u));scratch.Store(state_base+32u,0u);scratch.Store(state_base+532u,wave==0u?{m00}u:{m01}u);}}if(gi==0u)records.Store4(gid.x*io[1],uint4(0u,0u,0u,0u));DeviceMemoryBarrierWithGroupSync();{body}}}"#,
        vgpr_offset = super::mesh_dispatch::MIN_RECORD_BYTES,
        stride = stride,
        mailbox_offset = layout.scratch.mailbox_offset(0, 0)?,
        vgprs = layout.vgpr_count,
        mask0 = exec_masks[0],
        mask1 = exec_masks[1],
        vcc0 = initial_vcc[0],
        vcc1 = initial_vcc[1],
        m00 = initial_m0[0],
        m01 = initial_m0[1]
    ))
}

pub fn stage_b_validation_hlsl(
    layout: VgprScratchLayout,
    ops: &[StageBOp],
    exec_masks: [u64; 2],
) -> Result<String, &'static str> {
    stage_b_validation_hlsl_with_vcc(layout, ops, exec_masks, [0; 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init(layout: VgprScratchLayout, exec: [u64; 2]) -> (Vec<u32>, [LogicalWaveState; 2]) {
        let mut memory = vec![0u32; layout.scratch.total_bytes as usize / 4];
        for thread in 0..128u32 {
            for register in 0..layout.vgpr_count {
                store_vgpr(
                    &mut memory,
                    layout,
                    0,
                    thread,
                    register,
                    initial_vgpr_value(register, thread),
                )
                .unwrap();
            }
        }
        let states = std::array::from_fn(|wave| LogicalWaveState {
            exec: [exec[wave] as u32, (exec[wave] >> 32) as u32],
            ..Default::default()
        });
        (memory, states)
    }

    fn ops() -> Vec<StageBOp> {
        vec![
            StageBOp::Binary {
                kind: VectorBinaryKind::And,
                dst: 2,
                lhs: VectorSource::Vgpr(1),
                rhs: VectorSource::Constant(0xffff),
            },
            StageBOp::Ternary {
                kind: VectorTernaryKind::BfeU32,
                dst: 3,
                a: VectorSource::Vgpr(1),
                b: VectorSource::Constant(8),
                c: VectorSource::Constant(8),
            },
            StageBOp::BitReverse {
                dst: 4,
                src: VectorSource::Vgpr(0),
            },
            StageBOp::Compare {
                kind: CompareKind::Greater,
                lhs: VectorSource::Constant(40),
                rhs: VectorSource::Vgpr(0),
                cmpx: false,
            },
            StageBOp::CndMask {
                dst: 5,
                when_clear: VectorSource::Vgpr(0),
                when_set: VectorSource::Vgpr(1),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::LshlReverse,
                dst: 6,
                lhs: VectorSource::Constant(2),
                rhs: VectorSource::Vgpr(0),
            },
            StageBOp::Ternary {
                kind: VectorTernaryKind::LshlAdd,
                dst: 7,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Constant(1),
                c: VectorSource::Constant(3),
            },
            StageBOp::Ternary {
                kind: VectorTernaryKind::Or3,
                dst: 8,
                a: VectorSource::Vgpr(2),
                b: VectorSource::Vgpr(3),
                c: VectorSource::Vgpr(4),
            },
            StageBOp::ConvertF32U32 {
                dst: 9,
                src: VectorSource::Vgpr(0),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Add,
                dst: 10,
                lhs: VectorSource::Vgpr(9),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                modifiers: FloatModifiers {
                    neg: 1,
                    omod: 1,
                    ..FloatModifiers::default()
                },
            },
            StageBOp::FloatFma {
                dst: 10,
                a: VectorSource::Constant(2.0f32.to_bits()),
                b: VectorSource::Constant(2.0f32.to_bits()),
                c: None,
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            },
            StageBOp::FloatFma {
                dst: 11,
                a: VectorSource::Vgpr(9),
                b: VectorSource::Constant(2.0f32.to_bits()),
                c: Some(VectorSource::Constant((-1.0f32).to_bits())),
                modifiers: FloatModifiers {
                    abs: 4,
                    clamp: true,
                    ..FloatModifiers::default()
                },
                legacy_zero: false,
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::MultiplyLegacy,
                dst: 12,
                lhs: VectorSource::Constant(0),
                rhs: VectorSource::Constant(f32::INFINITY.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::StageDInteger {
                kind: StageDIntegerKind::LogicalShiftRight,
                dst: 13,
                a: VectorSource::Constant(1),
                b: VectorSource::Vgpr(0),
                c: None,
            },
            StageBOp::StageDInteger {
                kind: StageDIntegerKind::MadU24,
                dst: 14,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Constant(3),
                c: Some(VectorSource::Constant(7)),
            },
            StageBOp::StageDInteger {
                kind: StageDIntegerKind::Permute,
                dst: 15,
                a: VectorSource::Constant(0x1122_3344),
                b: VectorSource::Constant(0x5566_7788),
                c: Some(VectorSource::Constant(0x0706_0100)),
            },
            StageBOp::Compare {
                kind: CompareKind::Eq,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(31),
                cmpx: false,
            },
            StageBOp::Compare {
                kind: CompareKind::Greater,
                lhs: VectorSource::Constant(32),
                rhs: VectorSource::Vgpr(0),
                cmpx: true,
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::AddNoCarry,
                dst: 2,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(u32::MAX),
            },
            StageBOp::Mbcnt {
                high: false,
                dst: 3,
                source: VectorSource::Constant(u32::MAX),
                accumulator: VectorSource::Constant(7),
            },
            StageBOp::Mbcnt {
                high: true,
                dst: 4,
                source: VectorSource::Constant(u32::MAX),
                accumulator: VectorSource::Vgpr(3),
            },
            StageBOp::Move {
                dst: 5,
                src: VectorSource::Constant(0x1234_5678),
            },
            StageBOp::ReadFirstLane { dst: 10, src: 0 },
            StageBOp::CompareScalar {
                kind: CompareKind::LessEqual,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(31),
                dst: 0,
            },
            StageBOp::CndMaskScalar {
                dst: 6,
                when_clear: VectorSource::Constant(0xaaaa_aaaa),
                when_set: VectorSource::Constant(0xbbbb_bbbb),
                predicate: ScalarMask::ScalarPair(0),
            },
            StageBOp::ConvertF32U32Sdwa {
                dst: 7,
                src: 1,
                word: SdwaWord::High,
            },
            StageBOp::Compare {
                kind: CompareKind::OrderedNotEqualFloat,
                lhs: VectorSource::Constant(f32::NAN.to_bits()),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                cmpx: false,
            },
            StageBOp::Compare {
                kind: CompareKind::OrderedNotEqualFloat,
                lhs: VectorSource::Constant(1.0f32.to_bits()),
                rhs: VectorSource::Constant(2.0f32.to_bits()),
                cmpx: true,
            },
            StageBOp::ConvertF32I32 {
                dst: 16,
                src: VectorSource::Constant(i32::MIN as u32),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 17,
                lhs: VectorSource::Constant(8.5f32.to_bits()),
                rhs: VectorSource::Constant(2.25f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::ReverseSubtract,
                dst: 18,
                lhs: VectorSource::Constant(2.25f32.to_bits()),
                rhs: VectorSource::Constant(8.5f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 19,
                src: VectorSource::Constant((-1.75f32).to_bits()),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::F16FromF32,
                dst: 2,
                src: VectorSource::Constant((-0.0f32).to_bits()),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::F32FromF16,
                dst: 3,
                src: VectorSource::Constant(0x7c00),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::U32FromF32,
                dst: 4,
                src: VectorSource::Constant(f32::INFINITY.to_bits()),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::F16FromF32High,
                dst: 5,
                src: VectorSource::Constant(1.0f32.to_bits()),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::F32FromF16High,
                dst: 6,
                src: VectorSource::Constant(0x3c00_0000),
            },
            StageBOp::FloatConvert {
                kind: FloatConvertKind::U32FromF32,
                dst: 7,
                src: VectorSource::Constant(f32::NAN.to_bits()),
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::LessFloat,
                lhs: VectorSource::Constant((-0.0f32).to_bits()),
                rhs: VectorSource::Constant(0.0f32.to_bits()),
                dst: Some(2),
                modifiers: FloatModifiers::default(),
                cmpx: false,
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::LessFloat,
                lhs: VectorSource::Constant(f32::NAN.to_bits()),
                rhs: VectorSource::Constant(f32::INFINITY.to_bits()),
                dst: Some(4),
                modifiers: FloatModifiers::default(),
                cmpx: false,
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::GreaterFloat,
                lhs: VectorSource::Constant(2.0f32.to_bits()),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                dst: Some(12),
                modifiers: FloatModifiers::default(),
                cmpx: false,
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::LessEqualFloat,
                lhs: VectorSource::Constant(1.0f32.to_bits()),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                dst: Some(14),
                modifiers: FloatModifiers::default(),
                cmpx: false,
            },
            StageBOp::ConvertUByte {
                dst: 0,
                src: VectorSource::Constant(0x0000_3412),
                byte: 0,
                modifiers: FloatModifiers::default(),
            },
            StageBOp::ConvertUByte {
                dst: 1,
                src: VectorSource::Constant(0x0000_3412),
                byte: 1,
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Maximum,
                dst: 8,
                lhs: VectorSource::Constant(2.0f32.to_bits()),
                rhs: VectorSource::Constant(3.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Reciprocal,
                dst: 9,
                src: VectorSource::Constant(2.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::ReciprocalIFlag,
                dst: 10,
                src: VectorSource::Constant(4.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::ReciprocalSqrt,
                dst: 11,
                src: VectorSource::Constant(4.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sqrt,
                dst: 12,
                src: VectorSource::Constant(4.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sin,
                dst: 13,
                src: VectorSource::Constant(0.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Cos,
                dst: 14,
                src: VectorSource::Constant(0.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::PackF16 {
                dst: 15,
                lo: VectorSource::Constant(0xaaaa_3c00),
                hi: VectorSource::Constant(0x4000_bbbb),
                op_sel: 0,
            },
            StageBOp::FmaMix {
                kind: FmaMixKind::LoF16,
                dst: 16,
                a: VectorSource::Constant(1.0f32.to_bits()),
                b: VectorSource::Constant(2.0f32.to_bits()),
                c: VectorSource::Constant(1.0f32.to_bits()),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::MinU32,
                dst: 20,
                lhs: VectorSource::Constant(u32::MAX),
                rhs: VectorSource::Vgpr(0),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::MulLoU32,
                dst: 21,
                lhs: VectorSource::Constant(u32::MAX),
                rhs: VectorSource::Constant(2),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::MulHiU32,
                dst: 22,
                lhs: VectorSource::Constant(u32::MAX),
                rhs: VectorSource::Constant(2),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::SubNoCarry,
                dst: 23,
                lhs: VectorSource::Constant(0),
                rhs: VectorSource::Constant(1),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::ReverseSubNoCarry,
                dst: 24,
                lhs: VectorSource::Constant(1),
                rhs: VectorSource::Constant(0),
            },
            StageBOp::Ternary {
                kind: VectorTernaryKind::AddLshl,
                dst: 25,
                a: VectorSource::Constant(u32::MAX),
                b: VectorSource::Constant(2),
                c: VectorSource::Constant(1),
            },
            StageBOp::Ternary {
                kind: VectorTernaryKind::AndOr,
                dst: 26,
                a: VectorSource::Constant(0xf0f0_0f0f),
                b: VectorSource::Constant(0x0ff0_ffff),
                c: VectorSource::Constant(0x8000_0000),
            },
            StageBOp::Binary {
                kind: VectorBinaryKind::ArithmeticShiftRightReverse,
                dst: 0,
                lhs: VectorSource::Constant(4),
                rhs: VectorSource::Constant((-128i32) as u32),
            },
            StageBOp::FloatMad {
                dst: 27,
                a: VectorSource::Constant(0x3f80_0001),
                b: VectorSource::Constant(0x3f80_0001),
                c: Some(VectorSource::Constant(0xbf80_0002)),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMad {
                dst: 28,
                a: VectorSource::Constant(2.0f32.to_bits()),
                b: VectorSource::Constant(3.0f32.to_bits()),
                c: None,
                modifiers: FloatModifiers::default(),
            },
            StageBOp::CompareScalar {
                kind: CompareKind::Eq,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(0),
                dst: 20,
            },
            StageBOp::CompareScalar {
                kind: CompareKind::GreaterEqual,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(5),
                dst: 22,
            },
            StageBOp::CompareScalar {
                kind: CompareKind::NotEqual,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(31),
                dst: 24,
            },
            StageBOp::Compare {
                kind: CompareKind::SignedGreater,
                lhs: VectorSource::Constant(10),
                rhs: VectorSource::Vgpr(0),
                cmpx: true,
            },
            StageBOp::Compare {
                kind: CompareKind::SignedLess,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(10),
                cmpx: true,
            },
            StageBOp::Compare {
                kind: CompareKind::Less,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Constant(10),
                cmpx: true,
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::GreaterFloat,
                lhs: VectorSource::Constant(2.0f32.to_bits()),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                dst: None,
                modifiers: FloatModifiers::default(),
                cmpx: true,
            },
            StageBOp::CompareLessFloat {
                kind: CompareKind::LessFloat,
                lhs: VectorSource::Constant(f32::NAN.to_bits()),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                dst: None,
                modifiers: FloatModifiers::default(),
                cmpx: true,
            },
            StageBOp::NoEffect,
        ]
    }

    #[test]
    fn cpu_preserves_inactive_lanes_and_gathers_masks() {
        let layout = VgprScratchLayout::new(29, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let (mut memory, mut states) = init(layout, exec);
        let before = memory.clone();
        execute_stage_b(&mut memory, layout, &mut states, &ops()).unwrap();
        for wave in 0..2 {
            for lane in 0..64 {
                let thread = (wave * 64 + lane) as u32;
                if exec[wave] & (1u64 << lane) == 0 {
                    for reg in 0..29 {
                        assert_eq!(
                            load_vgpr(&memory, layout, 0, thread, reg).unwrap(),
                            load_vgpr(&before, layout, 0, thread, reg).unwrap()
                        );
                    }
                }
            }
        }
        assert_eq!(mask(states[0].exec), 0);
        assert_eq!(mask(states[1].exec), 0);
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 9).unwrap(),
            0.5f32.to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 10).unwrap(),
            0.25f32.to_bits()
        );
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 27).unwrap(), 0);
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 11).unwrap(),
            0.5f32.to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 12).unwrap(),
            2.0f32.to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 14).unwrap(),
            1.0f32.to_bits()
        );
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 15).unwrap(), 0xbbbb_3c00);
        assert_eq!(states[0].scalar[10], 0);
        assert_eq!(states[1].scalar[10], 5);
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 16).unwrap(),
            ((i32::MIN as f32).to_bits() & 0xffff_0000) | 0x4200
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 17).unwrap(),
            6.25f32.to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 18).unwrap(),
            6.25f32.to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 19).unwrap(),
            (-1.0f32).to_bits()
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 2).unwrap() & 0xffff,
            0x8000
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 3).unwrap(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 4).unwrap(), u32::MAX);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 5).unwrap() >> 16, 0x3c00);
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 6).unwrap(),
            1.0f32.to_bits()
        );
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 7).unwrap(), 0);
        assert_eq!((states[0].scalar[2], states[0].scalar[3]), (0, 0));
        assert_eq!((states[0].scalar[4], states[0].scalar[5]), (0, 0));
        assert_eq!((states[0].scalar[20], states[0].scalar[21]), (0, 0));
        assert_eq!(
            (states[0].scalar[22], states[0].scalar[23]),
            (0x8000_0001, 0)
        );
        assert_eq!(
            (states[0].scalar[24], states[0].scalar[25]),
            (0x8000_0001, 0)
        );
        assert_eq!(
            load_vgpr(&memory, layout, 0, 0, 20).unwrap(),
            18.0f32.to_bits()
        );
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 21).unwrap(), u32::MAX - 1);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 22).unwrap(), 1);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 23).unwrap(), u32::MAX);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 24).unwrap(), u32::MAX);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 25).unwrap(), 2);
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 26).unwrap(), 0x80f0_0f0f);
    }

    #[test]
    fn half_conversion_edges_are_ieee_bounded() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16(f32::MAX), 0x7bff);
        assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(3.0 * 2.0f32.powi(-25)), 0x0002);
        let nan = f32_to_f16(f32::NAN);
        assert_eq!(nan & 0x7c00, 0x7c00);
        assert_ne!(nan & 0x03ff, 0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn checked_layout_rejects_vector_registers_before_execution() {
        let layout = VgprScratchLayout::new(2, 1).unwrap();
        let (mut memory, mut states) = init(layout, [u64::MAX; 2]);
        let original_memory = memory.clone();
        let original_states = states;
        let bad_source = StageBOp::Binary {
            kind: VectorBinaryKind::And,
            dst: 0,
            lhs: VectorSource::Vgpr(2),
            rhs: VectorSource::Constant(1),
        };
        assert!(execute_stage_b(&mut memory, layout, &mut states, &[bad_source]).is_err());
        assert_eq!(memory, original_memory);
        assert_eq!(states, original_states);
        assert!(stage_b_validation_hlsl(layout, &[bad_source], [u64::MAX; 2]).is_err());

        let bad_destination = StageBOp::BitReverse {
            dst: 2,
            src: VectorSource::Vgpr(0),
        };
        assert!(execute_stage_b(&mut memory, layout, &mut states, &[bad_destination]).is_err());
        assert_eq!(memory, original_memory);
        assert_eq!(states, original_states);
        assert!(stage_b_validation_hlsl(layout, &[bad_destination], [u64::MAX; 2]).is_err());

        let bad_unary = StageBOp::FloatUnary {
            kind: FloatUnaryKind::Truncate,
            dst: 0,
            src: VectorSource::Vgpr(2),
        };
        assert!(execute_stage_b(&mut memory, layout, &mut states, &[bad_unary]).is_err());
        assert_eq!(memory, original_memory);
        assert_eq!(states, original_states);
        assert!(stage_b_validation_hlsl(layout, &[bad_unary], [u64::MAX; 2]).is_err());
    }

    #[test]
    fn float_conversion_and_unary_edges_are_exact() {
        let layout = VgprScratchLayout::new(5, 1).unwrap();
        let (mut memory, mut states) = init(layout, [1, 1]);
        let nan = f32::from_bits(0x7fc1_2345);
        let ops = [
            StageBOp::ConvertF32I32 {
                dst: 0,
                src: VectorSource::Constant(i32::MIN as u32),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 1,
                src: VectorSource::Constant((-0.0f32).to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 2,
                src: VectorSource::Constant(f32::INFINITY.to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 3,
                src: VectorSource::Constant(nan.to_bits()),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 4,
                lhs: VectorSource::Constant(0.0f32.to_bits()),
                rhs: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers: FloatModifiers::default(),
            },
        ];
        execute_stage_b(&mut memory, layout, &mut states, &ops).unwrap();
        for thread in [0, 64] {
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 0).unwrap(),
                (i32::MIN as f32).to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 1).unwrap(),
                (-0.0f32).to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 2).unwrap(),
                f32::INFINITY.to_bits()
            );
            assert!(f32::from_bits(load_vgpr(&memory, layout, 0, thread, 3).unwrap()).is_nan());
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 4).unwrap(),
                0.0f32.to_bits()
            );
        }
    }

    fn float_edge_ops() -> Vec<StageBOp> {
        vec![
            StageBOp::ConvertF32I32 {
                dst: 0,
                src: VectorSource::Constant(i32::MIN as u32),
            },
            StageBOp::ConvertF32I32 {
                dst: 1,
                src: VectorSource::Constant(i32::MAX as u32),
            },
            StageBOp::ConvertF32I32 {
                dst: 2,
                src: VectorSource::Constant((1i32 << 24) as u32 - 1),
            },
            StageBOp::ConvertF32I32 {
                dst: 3,
                src: VectorSource::Constant((1i32 << 24) as u32),
            },
            StageBOp::ConvertF32I32 {
                dst: 4,
                src: VectorSource::Constant((1i32 << 24) as u32 + 1),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 5,
                lhs: VectorSource::Constant(0.0f32.to_bits()),
                rhs: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 6,
                lhs: VectorSource::Constant((-0.0f32).to_bits()),
                rhs: VectorSource::Constant(0.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 7,
                lhs: VectorSource::Constant(f32::INFINITY.to_bits()),
                rhs: VectorSource::Constant(f32::INFINITY.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::ReverseSubtract,
                dst: 8,
                lhs: VectorSource::Constant(f32::NEG_INFINITY.to_bits()),
                rhs: VectorSource::Constant(f32::NEG_INFINITY.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Subtract,
                dst: 9,
                lhs: VectorSource::Constant(0x7fc1_2345),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 10,
                src: VectorSource::Constant(0.0f32.to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 11,
                src: VectorSource::Constant((-0.0f32).to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 12,
                src: VectorSource::Constant(f32::INFINITY.to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 13,
                src: VectorSource::Constant(f32::NEG_INFINITY.to_bits()),
            },
            StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 14,
                src: VectorSource::Constant(0x7fc1_2345),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Maximum,
                dst: 15,
                lhs: VectorSource::Constant(0x7fc1_2345),
                rhs: VectorSource::Constant(1.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Maximum,
                dst: 16,
                lhs: VectorSource::Constant((-0.0f32).to_bits()),
                rhs: VectorSource::Constant(0.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Reciprocal,
                dst: 17,
                src: VectorSource::Constant(0.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Reciprocal,
                dst: 18,
                src: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::ReciprocalSqrt,
                dst: 19,
                src: VectorSource::Constant(f32::INFINITY.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sqrt,
                dst: 20,
                src: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sin,
                dst: 21,
                src: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Cos,
                dst: 22,
                src: VectorSource::Constant(f32::INFINITY.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::ConvertUByte {
                dst: 23,
                src: VectorSource::Constant(0xff80_7f00),
                byte: 2,
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Reciprocal,
                dst: 24,
                src: VectorSource::Constant(1),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::ReciprocalSqrt,
                dst: 25,
                src: VectorSource::Constant(1),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sqrt,
                dst: 26,
                src: VectorSource::Constant(1),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sin,
                dst: 27,
                src: VectorSource::Constant(f32::MAX.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Cos,
                dst: 28,
                src: VectorSource::Constant(f32::MAX.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Sin,
                dst: 29,
                src: VectorSource::Constant(257.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Cos,
                dst: 30,
                src: VectorSource::Constant(257.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            },
        ]
    }

    #[test]
    fn float_edges_match_actual_u6_execution() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = VgprScratchLayout::new(31, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let ops = float_edge_ops();
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();

        for wave in 0..2 {
            for lane in 0..64 {
                if exec[wave] & (1u64 << lane) == 0 {
                    continue;
                }
                let thread = (wave * 64 + lane) as u32;
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                    (((1i32 << 24) - 1) as f32).to_bits()
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 3).unwrap(),
                    ((1i32 << 24) as f32).to_bits()
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 4).unwrap(),
                    (((1i32 << 24) + 1) as f32).to_bits()
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 24).unwrap(),
                    f32::INFINITY.to_bits()
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 25).unwrap(),
                    f32::INFINITY.to_bits()
                );
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, 26).unwrap(),
                    0.0f32.to_bits()
                );
                for register in [27, 29] {
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                        0.0f32.to_bits()
                    );
                }
                for register in [28, 30] {
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                        1.0f32.to_bits()
                    );
                }
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();

        for thread in 0..128u32 {
            for register in 0..31u32 {
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                let actual = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                if f32::from_bits(expected).is_nan() {
                    assert!(
                        f32::from_bits(actual).is_nan(),
                        "thread {thread} register {register}: {actual:#010x} is not NaN"
                    );
                } else {
                    assert_eq!(actual, expected, "thread {thread} register {register}");
                }
            }
        }
    }

    #[test]
    fn decoded_mixed_fma_outputs_match_actual_u6() {
        use crate::gpu::rdna2::encode;

        if super::super::dxc::location().is_err() {
            return;
        }
        let decoded_op = |name: &str, fields: &[(&str, u64)], literal: Option<u32>| {
            let bytes = encode(name, "ENC_VOP3P", fields, literal).unwrap();
            lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap()
        };
        let ops = vec![
            decoded_op(
                "V_FMA_MIX_F32",
                &[("VDST", 6), ("SRC0", 242), ("SRC1", 244), ("SRC2", 243)],
                None,
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[
                    ("VDST", 14),
                    ("SRC0", 255),
                    ("SRC1", 242),
                    ("SRC2", 128),
                    ("CLAMP", 1),
                ],
                Some(f32::NAN.to_bits()),
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[
                    ("VDST", 15),
                    ("SRC0", 255),
                    ("SRC1", 242),
                    ("SRC2", 128),
                    ("CLAMP", 1),
                ],
                Some((-0.0f32).to_bits()),
            ),
            decoded_op(
                "V_FMA_MIXLO_F16",
                &[
                    ("VDST", 7),
                    ("SRC0", 242),
                    ("SRC1", 244),
                    ("SRC2", 243),
                    ("OP_SEL", 7),
                    ("OP_SEL_HI", 3),
                    ("OP_SEL_HI_2", 1),
                ],
                None,
            ),
            decoded_op(
                "V_FMA_MIXHI_F16",
                &[
                    ("VDST", 8),
                    ("SRC0", 242),
                    ("SRC1", 244),
                    ("SRC2", 243),
                    ("OP_SEL", 7),
                    ("OP_SEL_HI", 3),
                    ("OP_SEL_HI_2", 1),
                ],
                None,
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[("VDST", 9), ("SRC0", 255), ("SRC1", 242), ("SRC2", 128)],
                Some(f32::NAN.to_bits()),
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[("VDST", 10), ("SRC0", 255), ("SRC1", 242), ("SRC2", 128)],
                Some(f32::INFINITY.to_bits()),
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[("VDST", 11), ("SRC0", 255), ("SRC1", 242), ("SRC2", 128)],
                Some((-0.0f32).to_bits()),
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[("VDST", 12), ("SRC0", 255), ("SRC1", 242), ("SRC2", 128)],
                Some(1),
            ),
            decoded_op(
                "V_FMA_MIX_F32",
                &[
                    ("VDST", 13),
                    ("SRC0", 245),
                    ("SRC1", 242),
                    ("SRC2", 240),
                    ("NEG", 2),
                    ("NEG_HI", 1),
                    ("CLAMP", 1),
                ],
                None,
            ),
        ];
        assert!(matches!(
            ops[0],
            StageBOp::FmaMix {
                kind: FmaMixKind::F32,
                ..
            }
        ));
        assert!(matches!(
            ops[3],
            StageBOp::FmaMix {
                kind: FmaMixKind::LoF16,
                ..
            }
        ));
        assert!(matches!(
            ops[4],
            StageBOp::FmaMix {
                kind: FmaMixKind::HiF16,
                ..
            }
        ));

        let layout = VgprScratchLayout::new(16, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        let before = expected_memory.clone();
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            for lane in 0..64 {
                let thread = (wave * 64 + lane) as u32;
                if exec[wave] & (1u64 << lane) != 0 {
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 6).unwrap(),
                        1.0f32.to_bits()
                    );
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 7).unwrap() & 0xffff_0000,
                        load_vgpr(&before, layout, 0, thread, 7).unwrap() & 0xffff_0000
                    );
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 8).unwrap() & 0x0000_ffff,
                        load_vgpr(&before, layout, 0, thread, 8).unwrap() & 0x0000_ffff
                    );
                    assert!(f32::from_bits(
                        load_vgpr(&expected_memory, layout, 0, thread, 9).unwrap()
                    )
                    .is_nan());
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 10).unwrap(),
                        f32::INFINITY.to_bits()
                    );
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 11).unwrap(),
                        0.0f32.to_bits()
                    );
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, 13).unwrap(),
                        0.0f32.to_bits()
                    );
                    for register in [14, 15] {
                        assert_eq!(
                            load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                            0.0f32.to_bits()
                        );
                    }
                }
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 0..16u32 {
                let actual = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                if f32::from_bits(expected).is_nan() {
                    assert!(
                        f32::from_bits(actual).is_nan(),
                        "thread {thread} register {register}"
                    );
                } else {
                    assert_eq!(actual, expected, "thread {thread} register {register}");
                }
            }
        }
    }

    #[test]
    fn decoded_forms_are_bounded_and_real_gate_stays_closed() {
        use crate::gpu::rdna2::encode;
        let cases = [
            ("V_EXP_F32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 242)]),
            ("V_FRACT_F32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 242)]),
            ("V_LOG_F32", "ENC_VOP1", vec![("VDST", 22), ("SRC0", 242)]),
            (
                "V_MIN_F32",
                "ENC_VOP2",
                vec![("VDST", 23), ("SRC0", 242), ("VSRC1", 0)],
            ),
            (
                "V_MAX3_F32",
                "ENC_VOP3",
                vec![("VDST", 24), ("SRC0", 242), ("SRC1", 243), ("SRC2", 244)],
            ),
            (
                "V_ADD_NC_U32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 193), ("VSRC1", 0)],
            ),
            (
                "V_ADD_NC_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 193)],
            ),
            (
                "V_MIN_U32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 193), ("VSRC1", 0)],
            ),
            (
                "V_MIN_U32",
                "VOP2_VOP_DPP16",
                vec![
                    ("VDST", 2),
                    ("SRC0", 250),
                    ("VSRC0", 0),
                    ("VSRC1", 1),
                    ("DPP_CTRL", 0x140),
                    ("ROW_MASK", 15),
                    ("BANK_MASK", 15),
                    ("BOUND_CTRL", 1),
                ],
            ),
            (
                "V_MIN_U32",
                "VOP2_VOP_DPP8",
                vec![
                    ("VDST", 2),
                    ("SRC0", 233),
                    ("VSRC0", 0),
                    ("VSRC1", 1),
                    ("LANE_SEL_0", 1),
                    ("LANE_SEL_1", 0),
                    ("LANE_SEL_2", 3),
                    ("LANE_SEL_3", 2),
                    ("LANE_SEL_4", 5),
                    ("LANE_SEL_5", 4),
                    ("LANE_SEL_6", 7),
                    ("LANE_SEL_7", 6),
                ],
            ),
            (
                "V_MUL_LO_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_MUL_HI_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_SUB_NC_U32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 193), ("VSRC1", 0)],
            ),
            (
                "V_SUB_NC_U32",
                "VOP2_VOP_SDWA",
                vec![
                    ("VDST", 2),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("VSRC1", 0),
                    ("SRC0_SEL", 1),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
            ),
            (
                "V_SUBREV_NC_U32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 193), ("VSRC1", 0)],
            ),
            (
                "V_SUBREV_NC_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_ASHRREV_I32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 129), ("VSRC1", 0)],
            ),
            (
                "V_MBCNT_LO_U32_B32",
                "ENC_VOP3",
                vec![("VDST", 3), ("SRC0", 193), ("SRC1", 135)],
            ),
            (
                "V_MBCNT_HI_U32_B32",
                "ENC_VOP3",
                vec![("VDST", 4), ("SRC0", 193), ("SRC1", 259)],
            ),
            ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 5), ("SRC0", 242)]),
            (
                "V_READFIRSTLANE_B32",
                "ENC_VOP1",
                vec![("VDST", 10), ("SRC0", 256)],
            ),
            (
                "V_AND_B32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 257), ("VSRC1", 0)],
            ),
            (
                "V_OR_B32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 257), ("VSRC1", 0)],
            ),
            (
                "V_XOR_B32",
                "ENC_VOP2",
                vec![("VDST", 2), ("SRC0", 257), ("VSRC1", 0)],
            ),
            (
                "V_BCNT_U32_B32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_ADD3_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_BFE_U32",
                "ENC_VOP3",
                vec![("VDST", 3), ("SRC0", 257), ("SRC1", 136), ("SRC2", 136)],
            ),
            ("V_BFREV_B32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", 256)]),
            (
                "V_CMPX_EQ_U32",
                "ENC_VOPC",
                vec![("SRC0", 159), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_GT_I32",
                "ENC_VOPC",
                vec![("SRC0", 193), ("VSRC1", 0)],
            ),
            (
                "V_CMP_GT_I32",
                "ENC_VOPC",
                vec![("SRC0", 257), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_GT_F32",
                "ENC_VOPC",
                vec![("SRC0", 242), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_GT_F32",
                "ENC_VOP3",
                vec![("VDST", 126), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CMPX_LT_F32",
                "ENC_VOPC",
                vec![("SRC0", 242), ("VSRC1", 0)],
            ),
            (
                "V_CMP_GE_F32",
                "ENC_VOP3",
                vec![("VDST", 106), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CMPX_LT_I32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_LT_I32",
                "VOPC_VOP_SDWA_SDST_ENC",
                vec![
                    ("SRC0", 249),
                    ("S0", 1),
                    ("VSRC0", 15),
                    ("VSRC1", 0),
                    ("SRC0_SEL", 6),
                    ("SRC1_SEL", 0),
                ],
            ),
            (
                "V_CMPX_LT_U32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_NE_U32",
                "ENC_VOPC",
                vec![("SRC0", 159), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_LT_U32",
                "VOPC_VOP_SDWA_SDST_ENC",
                vec![
                    ("SRC0", 249),
                    ("VSRC0", 0),
                    ("VSRC1", 1),
                    ("SRC0_SEL", 4),
                    ("SRC1_SEL", 6),
                ],
            ),
            (
                "V_CMP_EQ_U32",
                "ENC_VOP3",
                vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CMP_EQ_U32",
                "VOPC_VOP_SDWA_SDST_ENC",
                vec![
                    ("SRC0", 249),
                    ("VSRC0", 0),
                    ("S1", 1),
                    ("VSRC1", 20),
                    ("SRC0_SEL", 2),
                    ("SRC1_SEL", 6),
                    ("SD", 1),
                    ("SDST", 22),
                ],
            ),
            (
                "V_CMP_GE_U32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_GE_U32",
                "ENC_VOP3",
                vec![("VDST", 2), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CMP_NE_U32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_NE_U32",
                "ENC_VOP3",
                vec![("VDST", 4), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CMP_GT_U32",
                "ENC_VOPC",
                vec![("SRC0", 168), ("VSRC1", 0)],
            ),
            (
                "V_CMP_LE_U32",
                "ENC_VOP3",
                vec![("VDST", 106), ("SRC0", 256), ("SRC1", 257), ("SRC2", 128)],
            ),
            (
                "V_CMP_LE_U32",
                "ENC_VOP3",
                vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_CNDMASK_B32",
                "ENC_VOP3",
                vec![("VDST", 5), ("SRC0", 256), ("SRC1", 257), ("SRC2", 106)],
            ),
            (
                "V_CNDMASK_B32",
                "ENC_VOP3",
                vec![
                    ("VDST", 5),
                    ("SRC0", 256),
                    ("SRC1", 257),
                    ("SRC2", 0),
                    ("NEG", 2),
                ],
            ),
            (
                "V_CNDMASK_B32",
                "ENC_VOP3",
                vec![("VDST", 5), ("SRC0", 0), ("SRC1", 257), ("SRC2", 106)],
            ),
            (
                "V_CNDMASK_B32",
                "ENC_VOP3",
                vec![("VDST", 5), ("SRC0", 256), ("SRC1", 128), ("SRC2", 106)],
            ),
            (
                "V_LSHLREV_B32",
                "ENC_VOP2",
                vec![("VDST", 6), ("SRC0", 130), ("VSRC1", 0)],
            ),
            (
                "V_LSHL_ADD_U32",
                "ENC_VOP3",
                vec![("VDST", 7), ("SRC0", 256), ("SRC1", 129), ("SRC2", 131)],
            ),
            (
                "V_ADD_LSHL_U32",
                "ENC_VOP3",
                vec![("VDST", 7), ("SRC0", 256), ("SRC1", 129), ("SRC2", 131)],
            ),
            (
                "V_AND_OR_B32",
                "ENC_VOP3",
                vec![("VDST", 8), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_OR3_B32",
                "ENC_VOP3",
                vec![("VDST", 8), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            ("V_NOP", "ENC_VOP1", vec![("SRC0", 0)]),
            (
                "V_CVT_F32_U32",
                "ENC_VOP1",
                vec![("VDST", 9), ("SRC0", 256)],
            ),
            (
                "V_CVT_F32_U32",
                "ENC_VOP1",
                vec![("VDST", 9), ("SRC0", 251)],
            ),
            (
                "V_CVT_F32_I32",
                "ENC_VOP1",
                vec![("VDST", 16), ("SRC0", 256)],
            ),
            ("V_TRUNC_F32", "ENC_VOP1", vec![("VDST", 19), ("SRC0", 256)]),
            (
                "V_CVT_F32_U32",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 9),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("SRC0_SEL", 5),
                    ("DST_SEL", 6),
                ],
            ),
            (
                "V_ADD_F32",
                "ENC_VOP2",
                vec![("VDST", 10), ("SRC0", 265), ("VSRC1", 9)],
            ),
            (
                "V_ADD_F32",
                "ENC_VOP3",
                vec![
                    ("VDST", 10),
                    ("SRC0", 265),
                    ("SRC1", 242),
                    ("NEG", 1),
                    ("OMOD", 1),
                ],
            ),
            (
                "V_SUB_F32",
                "ENC_VOP2",
                vec![("VDST", 17), ("SRC0", 256), ("VSRC1", 1)],
            ),
            (
                "V_SUB_F32",
                "ENC_VOP3",
                vec![("VDST", 17), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_SUBREV_F32",
                "ENC_VOP2",
                vec![("VDST", 18), ("SRC0", 256), ("VSRC1", 1)],
            ),
            (
                "V_SUBREV_F32",
                "ENC_VOP3",
                vec![("VDST", 18), ("SRC0", 256), ("SRC1", 257)],
            ),
            (
                "V_FMAC_F32",
                "ENC_VOP2",
                vec![("VDST", 10), ("SRC0", 244), ("VSRC1", 9)],
            ),
            (
                "V_FMA_F32",
                "ENC_VOP3",
                vec![("VDST", 11), ("SRC0", 265), ("SRC1", 244), ("SRC2", 243)],
            ),
            (
                "V_MAC_F32",
                "ENC_VOP2",
                vec![("VDST", 27), ("SRC0", 244), ("VSRC1", 9)],
            ),
            (
                "V_MAC_F32",
                "ENC_VOP3",
                vec![("VDST", 27), ("SRC0", 244), ("SRC1", 265)],
            ),
            (
                "V_MAD_F32",
                "ENC_VOP3",
                vec![("VDST", 28), ("SRC0", 244), ("SRC1", 265), ("SRC2", 243)],
            ),
            (
                "V_MUL_LEGACY_F32",
                "ENC_VOP2",
                vec![("VDST", 12), ("SRC0", 128), ("VSRC1", 9)],
            ),
            (
                "V_LSHRREV_B32",
                "ENC_VOP2",
                vec![("VDST", 13), ("SRC0", 129), ("VSRC1", 0)],
            ),
            (
                "V_MAD_U32_U24",
                "ENC_VOP3",
                vec![("VDST", 14), ("SRC0", 256), ("SRC1", 131), ("SRC2", 135)],
            ),
            (
                "V_PERM_B32",
                "ENC_VOP3",
                vec![("VDST", 15), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_CMP_EQ_U32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_GT_U32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMPX_LG_F32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_LT_F32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_EQ_F32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_EQ_F32",
                "ENC_VOP3",
                vec![("VDST", 6), ("SRC0", 256), ("SRC1", 257), ("ABS", 1)],
            ),
            (
                "V_CMP_LG_F32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_LG_F32",
                "ENC_VOP3",
                vec![("VDST", 8), ("SRC0", 256), ("SRC1", 257), ("NEG", 2)],
            ),
            (
                "V_CMP_LT_F32",
                "ENC_VOP3",
                vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257), ("ABS", 1)],
            ),
            (
                "V_CMP_GT_F32",
                "ENC_VOPC",
                vec![("SRC0", 256), ("VSRC1", 0)],
            ),
            (
                "V_CMP_LE_F32",
                "ENC_VOP3",
                vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257), ("NEG", 2)],
            ),
            (
                "V_CVT_F32_UBYTE0",
                "ENC_VOP1",
                vec![("VDST", 0), ("SRC0", 256)],
            ),
            (
                "V_CVT_F32_UBYTE0",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 0),
                    ("SRC0", 249),
                    ("VSRC0", 8),
                    ("SRC0_SEL", 3),
                    ("DST_SEL", 6),
                ],
            ),
            (
                "V_CVT_F32_UBYTE1",
                "ENC_VOP1",
                vec![("VDST", 1), ("SRC0", 256)],
            ),
            ("V_RCP_F32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", 256)]),
            (
                "V_RCP_IFLAG_F32",
                "ENC_VOP1",
                vec![("VDST", 3), ("SRC0", 256)],
            ),
            ("V_RSQ_F32", "ENC_VOP1", vec![("VDST", 4), ("SRC0", 256)]),
            ("V_SQRT_F32", "ENC_VOP1", vec![("VDST", 5), ("SRC0", 256)]),
            ("V_SIN_F32", "ENC_VOP1", vec![("VDST", 6), ("SRC0", 256)]),
            ("V_COS_F32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 256)]),
            (
                "V_MAX_F32",
                "ENC_VOP2",
                vec![("VDST", 8), ("SRC0", 256), ("VSRC1", 1)],
            ),
            (
                "V_PACK_B32_F16",
                "ENC_VOP3",
                vec![("VDST", 9), ("SRC0", 256), ("SRC1", 257), ("OP_SEL", 2)],
            ),
            (
                "V_FMA_MIXLO_F16",
                "ENC_VOP3P",
                vec![("VDST", 10), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_FMA_MIXHI_F16",
                "ENC_VOP3P",
                vec![("VDST", 10), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_FMA_MIX_F32",
                "ENC_VOP3P",
                vec![("VDST", 10), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            ),
            (
                "V_CVT_F16_F32",
                "ENC_VOP1",
                vec![("VDST", 2), ("SRC0", 256)],
            ),
            (
                "V_CVT_F16_F32",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 2),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("SRC0_SEL", 6),
                    ("DST_SEL", 5),
                    ("DST_UNUSED", 2),
                ],
            ),
            (
                "V_CVT_F16_F32",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 2),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("SRC0_SEL", 6),
                    ("DST_SEL", 4),
                    ("DST_UNUSED", 2),
                ],
            ),
            (
                "V_CVT_F16_F32",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 2),
                    ("SRC0", 249),
                    ("S0", 1),
                    ("VSRC0", 4),
                    ("SRC0_SEL", 6),
                    ("DST_SEL", 5),
                    ("DST_UNUSED", 2),
                ],
            ),
            (
                "V_CVT_F32_F16",
                "ENC_VOP1",
                vec![("VDST", 3), ("SRC0", 258)],
            ),
            (
                "V_CVT_F32_F16",
                "VOP1_VOP_SDWA",
                vec![
                    ("VDST", 3),
                    ("SRC0", 249),
                    ("VSRC0", 2),
                    ("SRC0_SEL", 5),
                    ("DST_SEL", 6),
                ],
            ),
            (
                "V_CVT_U32_F32",
                "ENC_VOP1",
                vec![("VDST", 4), ("SRC0", 259)],
            ),
        ];
        for (name, enc, fields) in cases {
            let code = encode(name, enc, &fields, None).unwrap();
            let decoded = rdna2::decode(&code, 0).unwrap();
            assert!(lower_stage_b_one(&decoded).is_ok(), "{name}");
            assert!(super::super::mesh_control::lower(&code).is_err());
        }

        let observed_opsel = encode(
            "V_ADD_F32",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", 256), ("SRC1", 257), ("OP_SEL", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&observed_opsel, 0).unwrap()).is_ok());
        let invalid_predicate = encode(
            "V_CNDMASK_B32",
            "ENC_VOP3",
            &[("VDST", 5), ("SRC0", 256), ("SRC1", 257), ("SRC2", 107)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_predicate, 0).unwrap()).is_err());

        let invalid_scalar_destination = encode(
            "V_CMP_LE_U32",
            "ENC_VOP3",
            &[("VDST", 1), ("SRC0", 256), ("SRC1", 257)],
            None,
        )
        .unwrap();
        assert!(
            lower_stage_b_one(&rdna2::decode(&invalid_scalar_destination, 0).unwrap()).is_err()
        );
        let signed_sdwa = encode(
            "V_CVT_F32_U32",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 9),
                ("SRC0", 249),
                ("VSRC0", 1),
                ("SRC0_SEL", 5),
                ("SRC0_SEXT", 1),
                ("DST_SEL", 6),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&signed_sdwa, 0).unwrap()).is_err());
        let invalid_ubyte_sdwa = encode(
            "V_CVT_F32_UBYTE0",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 0),
                ("SRC0", 249),
                ("VSRC0", 8),
                ("SRC0_SEL", 3),
                ("SRC0_NEG", 1),
                ("DST_SEL", 6),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_ubyte_sdwa, 0).unwrap()).is_err());
        let invalid_ubyte_vop3 = encode(
            "V_CVT_F32_UBYTE0",
            "ENC_VOP3",
            &[("VDST", 0), ("SRC0", 256), ("ABS", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_ubyte_vop3, 0).unwrap()).is_err());

        let literal_move = encode(
            "V_MOV_B32",
            "ENC_VOP1",
            &[("VDST", 5), ("SRC0", 255)],
            Some(0x1234_5678),
        )
        .unwrap();
        assert_eq!(
            lower_stage_b_one(&rdna2::decode(&literal_move, 0).unwrap()).unwrap(),
            StageBOp::Move {
                dst: 5,
                src: VectorSource::Constant(0x1234_5678)
            }
        );
        let modified_integer = encode(
            "V_ADD_NC_U32",
            "ENC_VOP3",
            &[("VDST", 2), ("SRC0", 256), ("SRC1", 257), ("OP_SEL", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&modified_integer, 0).unwrap()).is_err());
        for name in ["V_MAC_F32", "V_MAD_F32"] {
            let modified_mad = encode(
                name,
                "ENC_VOP3",
                &[
                    ("VDST", 27),
                    ("SRC0", 256),
                    ("SRC1", 257),
                    ("SRC2", 258),
                    ("OP_SEL", 1),
                ],
                None,
            )
            .unwrap();
            assert!(lower_stage_b_one(&rdna2::decode(&modified_mad, 0).unwrap()).is_err());
        }
        let unsupported_float_compare = encode(
            "V_CMPX_LG_F32",
            "ENC_VOP3",
            &[("VDST", 126), ("SRC0", 256), ("SRC1", 257)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&unsupported_float_compare, 0).unwrap()).is_err());
        let unsupported_cmpx_ne = encode(
            "V_CMPX_NE_U32",
            "ENC_VOP3",
            &[("VDST", 126), ("SRC0", 159), ("SRC1", 256)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&unsupported_cmpx_ne, 0).unwrap()).is_err());
        for name in ["V_CVT_F32_I32", "V_TRUNC_F32"] {
            let unsupported =
                encode(name, "ENC_VOP3", &[("VDST", 16), ("SRC0", 256)], None).unwrap();
            assert!(lower_stage_b_one(&rdna2::decode(&unsupported, 0).unwrap()).is_err());
        }
        let invalid_float_destination = encode(
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[("VDST", 1), ("SRC0", 256), ("SRC1", 257)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_float_destination, 0).unwrap()).is_err());
        let invalid_float_modifier = encode(
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[("VDST", 106), ("SRC0", 256), ("SRC1", 257), ("CLAMP", 1)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_float_modifier, 0).unwrap()).is_err());
        let invalid_convert_form = encode(
            "V_CVT_U32_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 256)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_convert_form, 0).unwrap()).is_err());
        let invalid_readfirst = encode(
            "V_READFIRSTLANE_B32",
            "ENC_VOP1",
            &[("VDST", 10), ("SRC0", 0)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_readfirst, 0).unwrap()).is_err());
    }

    #[test]
    fn readfirstlane_rejects_empty_exec_before_mutation() {
        let layout = VgprScratchLayout::new(2, 1).unwrap();
        let (mut memory, mut states) = init(layout, [u64::MAX, 0]);
        let before_memory = memory.clone();
        let before_states = states;
        let ops = [StageBOp::ReadFirstLane { dst: 1, src: 0 }];
        assert!(execute_stage_b(&mut memory, layout, &mut states, &ops).is_err());
        assert_eq!(memory, before_memory);
        assert_eq!(states, before_states);
        assert!(stage_b_validation_hlsl(layout, &ops, [u64::MAX, 0]).is_err());
    }

    #[test]
    fn cmpx_preserves_vcc_on_cpu_and_gpu() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let layout = VgprScratchLayout::new(2, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let vcc = [0xaaaa_5555_dead_beef, 0x0123_4567_89ab_cdef];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        for wave in 0..2 {
            expected_states[wave].vcc = [vcc[wave] as u32, (vcc[wave] >> 32) as u32];
        }
        let encoded = encode(
            "V_CMPX_NE_U32",
            "ENC_VOPC",
            &[("SRC0", 159), ("VSRC1", 0)],
            None,
        )
        .unwrap();
        let ops = [
            StageBOp::Compare {
                kind: CompareKind::OrderedNotEqualFloat,
                lhs: VectorSource::Constant(1.0f32.to_bits()),
                rhs: VectorSource::Constant(2.0f32.to_bits()),
                cmpx: true,
            },
            lower_stage_b_one(&rdna2::decode(&encoded, 0).unwrap()).unwrap(),
        ];
        assert_eq!(
            ops[1],
            StageBOp::Compare {
                kind: CompareKind::NotEqual,
                lhs: VectorSource::Constant(31),
                rhs: VectorSource::Vgpr(0),
                cmpx: true,
            }
        );
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        let expected_exec = [exec[0] & !(1u64 << 31), exec[1]];
        for wave in 0..2 {
            assert_eq!(mask(expected_states[wave].vcc), vcc[wave]);
            assert_eq!(mask(expected_states[wave].exec), expected_exec[wave]);
        }

        let hlsl = stage_b_validation_hlsl_with_vcc(layout, &ops, exec, vcc).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got =
                super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave)
                    .unwrap();
            assert_eq!(mask(got.vcc), vcc[wave as usize]);
            assert_eq!(mask(got.exec), expected_exec[wave as usize]);
        }
    }

    #[test]
    fn decoded_cmpx_family_updates_exec_and_preserves_vcc() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let encoded = [
            encode(
                "V_CMPX_EQ_F32",
                "ENC_VOPC",
                &[("SRC0", 128), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_LE_F32",
                "ENC_VOPC",
                &[("SRC0", 128), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_NEQ_F32",
                "ENC_VOPC",
                &[("SRC0", 255), ("VSRC1", 0)],
                Some(f32::NAN.to_bits()),
            )
            .unwrap(),
            encode(
                "V_CMPX_EQ_U32",
                "ENC_VOPC",
                &[("SRC0", 159), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
        ];
        assert_eq!(
            encoded,
            [
                vec![0x80, 0x00, 0x24, 0x7c],
                vec![0x80, 0x00, 0x26, 0x7c],
                vec![0xff, 0x00, 0x3a, 0x7c, 0x00, 0x00, 0xc0, 0x7f],
                vec![0x9f, 0x00, 0xa4, 0x7d],
            ]
        );
        let ops: Vec<_> = encoded
            .iter()
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap())
            .collect();
        let observed_sdwa = [0xf9, 0x50, 0xa4, 0x7d, 0x42, 0x00, 0x86, 0x00];
        assert_eq!(
            lower_stage_b_one(&rdna2::decode(&observed_sdwa, 0).unwrap()).unwrap(),
            StageBOp::Compare {
                kind: CompareKind::Eq,
                lhs: VectorSource::Sdwa {
                    register: 66,
                    scalar: true,
                    select: 6,
                    sign_extend: false,
                },
                rhs: VectorSource::Sdwa {
                    register: 40,
                    scalar: false,
                    select: 0,
                    sign_extend: false,
                },
                cmpx: true,
            }
        );
        assert_eq!(
            ops,
            [
                StageBOp::CompareLessFloat {
                    kind: CompareKind::EqualFloat,
                    lhs: VectorSource::Constant(0),
                    rhs: VectorSource::Vgpr(0),
                    dst: None,
                    modifiers: FloatModifiers::default(),
                    cmpx: true,
                },
                StageBOp::CompareLessFloat {
                    kind: CompareKind::LessEqualFloat,
                    lhs: VectorSource::Constant(0),
                    rhs: VectorSource::Vgpr(0),
                    dst: None,
                    modifiers: FloatModifiers::default(),
                    cmpx: true,
                },
                StageBOp::CompareLessFloat {
                    kind: CompareKind::UnorderedNotEqualFloat,
                    lhs: VectorSource::Constant(f32::NAN.to_bits()),
                    rhs: VectorSource::Vgpr(0),
                    dst: None,
                    modifiers: FloatModifiers::default(),
                    cmpx: true,
                },
                StageBOp::Compare {
                    kind: CompareKind::Eq,
                    lhs: VectorSource::Constant(31),
                    rhs: VectorSource::Vgpr(0),
                    cmpx: true,
                },
            ]
        );

        let layout = VgprScratchLayout::new(1, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 63),
            (1u64 << 0) | (1u64 << 31) | (1u64 << 63),
        ];
        let vcc = [0xaaaa_5555_dead_beef, 0x0123_4567_89ab_cdef];
        let expected_masks = [
            [1u64, 1u64],
            exec,
            exec,
            [1u64 << 31, 1u64 << 31],
        ];
        for (op, expected) in ops.iter().copied().zip(expected_masks) {
            let (mut memory, mut states) = init(layout, exec);
            for wave in 0..2 {
                states[wave].vcc = [vcc[wave] as u32, (vcc[wave] >> 32) as u32];
            }
            execute_stage_b(&mut memory, layout, &mut states, &[op]).unwrap();
            for wave in 0..2 {
                assert_eq!(mask(states[wave].exec), expected[wave]);
                assert_eq!(mask(states[wave].vcc), vcc[wave]);
            }
        }

        let sdwa_bytes = encode(
            "V_CMPX_EQ_U32",
            "VOPC_VOP_SDWA_SDST_ENC",
            &[
                ("SRC0", 249),
                ("S0", 1),
                ("VSRC0", 15),
                ("VSRC1", 0),
                ("SRC0_SEL", 6),
                ("SRC1_SEL", 0),
            ],
            None,
        )
        .unwrap();
        let gpu_op = [lower_stage_b_one(&rdna2::decode(&sdwa_bytes, 0).unwrap()).unwrap()];
        let hlsl = stage_b_validation_hlsl_with_vcc(layout, &gpu_op, exec, vcc).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got = super::super::mesh_dispatch::read_logical_state(
                &memory,
                layout.scratch,
                0,
                wave,
            )
            .unwrap();
            assert_eq!(mask(got.exec), 1);
            assert_eq!(mask(got.vcc), vcc[wave as usize]);
        }

        let invalid_destination = encode(
            "V_CMPX_EQ_F32",
            "ENC_VOP3",
            &[("VDST", 106), ("SRC0", 256), ("SRC1", 257)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_destination, 0).unwrap()).is_err());
    }

    #[test]
    fn decoded_residual_compare_predicates_match_masks_and_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let encoded = [
            encode(
                "V_CMPX_GE_F32",
                "ENC_VOPC",
                &[("SRC0", 242), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_NGT_F32",
                "ENC_VOPC",
                &[("SRC0", 255), ("VSRC1", 0)],
                Some(f32::NAN.to_bits()),
            )
            .unwrap(),
            encode(
                "V_CMPX_GE_U32",
                "ENC_VOPC",
                &[("SRC0", 159), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMP_NGE_F32",
                "ENC_VOPC",
                &[("SRC0", 255), ("VSRC1", 0)],
                Some(f32::NAN.to_bits()),
            )
            .unwrap(),
            encode(
                "V_CMP_NGT_F32",
                "ENC_VOPC",
                &[("SRC0", 128), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
        ];
        assert_eq!(
            encoded,
            [
                vec![0xf2, 0x00, 0x2c, 0x7c],
                vec![0xff, 0x00, 0x36, 0x7c, 0x00, 0x00, 0xc0, 0x7f],
                vec![0x9f, 0x00, 0xac, 0x7d],
                vec![0xff, 0x00, 0x12, 0x7c, 0x00, 0x00, 0xc0, 0x7f],
                vec![0x80, 0x00, 0x16, 0x7c],
            ]
        );
        let ops = encoded
            .iter()
            .map(|bytes| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            ops[0],
            StageBOp::CompareLessFloat {
                kind: CompareKind::GreaterEqualFloat,
                cmpx: true,
                ..
            }
        ));
        assert!(matches!(
            ops[1],
            StageBOp::CompareLessFloat {
                kind: CompareKind::UnorderedLessEqualFloat,
                cmpx: true,
                ..
            }
        ));
        assert!(matches!(
            ops[2],
            StageBOp::Compare {
                kind: CompareKind::GreaterEqual,
                cmpx: true,
                ..
            }
        ));
        assert!(matches!(
            ops[3],
            StageBOp::CompareLessFloat {
                kind: CompareKind::UnorderedLessFloat,
                cmpx: false,
                ..
            }
        ));
        assert!(matches!(
            ops[4],
            StageBOp::CompareLessFloat {
                kind: CompareKind::UnorderedLessEqualFloat,
                cmpx: false,
                ..
            }
        ));

        let layout = VgprScratchLayout::new(1, 1).unwrap();
        let initial_exec = [u64::MAX, u64::MAX];
        let initial_vcc = [0xaaaa_5555_dead_beef, 0x0123_4567_89ab_cdef];
        let (mut expected_memory, mut expected_states) = init(layout, initial_exec);
        for wave in 0..2 {
            expected_states[wave].vcc = [
                initial_vcc[wave] as u32,
                (initial_vcc[wave] >> 32) as u32,
            ];
        }
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            assert_eq!(mask(expected_states[wave].exec), 0xffff_ffff);
            assert_eq!(mask(expected_states[wave].vcc), 0xffff_ffff);
        }

        let hlsl = stage_b_validation_hlsl_with_vcc(layout, &ops, initial_exec, initial_vcc)
            .unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2u32 {
            let got = super::super::mesh_dispatch::read_logical_state(
                &memory,
                layout.scratch,
                0,
                wave,
            )
            .unwrap();
            assert_eq!(mask(got.exec), 0xffff_ffff);
            assert_eq!(mask(got.vcc), 0xffff_ffff);
        }
    }

    #[test]
    fn decoded_u64_equality_checks_both_words_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let vopc_bytes = encode(
            "V_CMP_EQ_U64",
            "ENC_VOPC",
            &[("SRC0", 258), ("VSRC1", 0)],
            None,
        )
        .unwrap();
        let vop3_bytes = encode(
            "V_CMP_EQ_U64",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 256), ("SRC1", 260)],
            None,
        )
        .unwrap();
        assert_eq!(vopc_bytes, [0x02, 0x01, 0xc4, 0x7d]);
        assert_eq!(vop3_bytes, [0x04, 0x00, 0xe2, 0xd4, 0x00, 0x09, 0x02, 0x00]);
        let decoded_vopc = lower_stage_b_one(&rdna2::decode(&vopc_bytes, 0).unwrap()).unwrap();
        let decoded_vop3 = lower_stage_b_one(&rdna2::decode(&vop3_bytes, 0).unwrap()).unwrap();
        assert_eq!(
            [decoded_vopc, decoded_vop3],
            [
                StageBOp::CompareEqU64 {
                    lhs: VectorSource::Vgpr(2),
                    rhs: VectorSource::Vgpr(0),
                    dst: None,
                },
                StageBOp::CompareEqU64 {
                    lhs: VectorSource::Vgpr(0),
                    rhs: VectorSource::Vgpr(4),
                    dst: Some(4),
                },
            ]
        );
        let invalid_vgpr_pair = encode(
            "V_CMP_EQ_U64",
            "ENC_VOPC",
            &[("SRC0", 256), ("VSRC1", 255)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_vgpr_pair, 0).unwrap()).is_err());
        let invalid_scalar_pair = encode(
            "V_CMP_EQ_U64",
            "ENC_VOP3",
            &[("VDST", 106), ("SRC0", 124), ("SRC1", 256)],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&invalid_scalar_pair, 0).unwrap()).is_err());

        let layout = VgprScratchLayout::new(6, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 1) | (1u64 << 31) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let ops = [
            StageBOp::Move { dst: 0, src: VectorSource::Constant(7) },
            StageBOp::Move { dst: 1, src: VectorSource::Constant(9) },
            StageBOp::Move { dst: 2, src: VectorSource::Constant(7) },
            StageBOp::Move { dst: 3, src: VectorSource::Constant(9) },
            decoded_vopc,
            StageBOp::Move { dst: 4, src: VectorSource::Constant(8) },
            StageBOp::Move { dst: 5, src: VectorSource::Constant(9) },
            decoded_vop3,
            StageBOp::Move { dst: 4, src: VectorSource::Constant(7) },
            StageBOp::Move { dst: 5, src: VectorSource::Constant(10) },
            StageBOp::CompareEqU64 {
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(4),
                dst: Some(6),
            },
        ];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(
            &mut expected_memory,
            layout,
            &mut expected_states,
            &ops,
        )
        .unwrap();
        for wave in 0..2 {
            assert_eq!(mask(expected_states[wave].vcc), exec[wave]);
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(4)),
                0
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(6)),
                0
            );
            assert_eq!(mask(expected_states[wave].exec), exec[wave]);
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got =
                super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave)
                    .unwrap();
            assert_eq!(mask(got.vcc), exec[wave as usize]);
            assert_eq!(
                scalar_mask_value(&got, ScalarMask::ScalarPair(4)),
                0
            );
            assert_eq!(scalar_mask_value(&got, ScalarMask::ScalarPair(6)), 0);
            assert_eq!(mask(got.exec), exec[wave as usize]);
        }
    }

    #[test]
    fn decoded_final_shift_and_float_compares_match_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let layout = VgprScratchLayout::new(3, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let sentinel = [0xaaaa_5555_dead_beef, 0x0123_4567_89ab_cdef];
        let shift = encode(
            "V_ASHRREV_I32",
            "ENC_VOP2",
            &[("VDST", 2), ("SRC0", 132), ("VSRC1", 1)],
            None,
        )
        .unwrap();
        let shift = lower_stage_b_one(&rdna2::decode(&shift, 0).unwrap()).unwrap();

        let nle_vopc = encode(
            "V_CMP_NLE_F32",
            "ENC_VOPC",
            &[("SRC0", 255), ("VSRC1", 0)],
            Some(f32::NAN.to_bits()),
        )
        .unwrap();
        let nle_vop3 = encode(
            "V_CMP_NLE_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 242), ("SRC1", 243)],
            None,
        )
        .unwrap();
        let neq_vopc_nan = encode(
            "V_CMP_NEQ_F32",
            "ENC_VOPC",
            &[("SRC0", 255), ("VSRC1", 0)],
            Some(f32::NAN.to_bits()),
        )
        .unwrap();
        let neq_vop3_equal = encode(
            "V_CMP_NEQ_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 242), ("SRC1", 242)],
            None,
        )
        .unwrap();
        let neq_vop3_unequal = encode(
            "V_CMP_NEQ_F32",
            "ENC_VOP3",
            &[("VDST", 6), ("SRC0", 242), ("SRC1", 243)],
            None,
        )
        .unwrap();
        assert_eq!(nle_vopc, [0xff, 0x00, 0x18, 0x7c, 0x00, 0x00, 0xc0, 0x7f]);
        assert_eq!(nle_vop3, [0x04, 0x00, 0x0c, 0xd4, 0xf2, 0xe6, 0x01, 0x00]);
        assert_eq!(neq_vopc_nan, [0xff, 0x00, 0x1a, 0x7c, 0x00, 0x00, 0xc0, 0x7f]);
        assert_eq!(neq_vop3_equal, [0x04, 0x00, 0x0d, 0xd4, 0xf2, 0xe4, 0x01, 0x00]);
        assert_eq!(neq_vop3_unequal, [0x06, 0x00, 0x0d, 0xd4, 0xf2, 0xe6, 0x01, 0x00]);
        let cases = [
            encode(
                "V_CMP_GT_I32",
                "ENC_VOPC",
                &[("SRC0", 257), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_GT_F32",
                "ENC_VOPC",
                &[("SRC0", 242), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_LT_F32",
                "ENC_VOPC",
                &[("SRC0", 242), ("VSRC1", 0)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMPX_GT_F32",
                "ENC_VOP3",
                &[("VDST", 126), ("SRC0", 242), ("SRC1", 243), ("NEG", 2)],
                None,
            )
            .unwrap(),
            encode(
                "V_CMP_GE_F32",
                "ENC_VOP3",
                &[("VDST", 106), ("SRC0", 242), ("SRC1", 242)],
                None,
            )
            .unwrap(),
            nle_vopc,
            nle_vop3,
            neq_vopc_nan,
            neq_vop3_equal,
            neq_vop3_unequal,
        ];

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

        for encoded in cases {
            let compare = lower_stage_b_one(&rdna2::decode(&encoded, 0).unwrap()).unwrap();
            let ops = [shift, compare];
            let (mut expected_memory, mut expected_states) = init(layout, exec);
            for wave in 0..2 {
                expected_states[wave].vcc = [sentinel[wave] as u32, (sentinel[wave] >> 32) as u32];
            }
            execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
            if matches!(
                compare,
                StageBOp::Compare {
                    kind: CompareKind::SignedGreater,
                    cmpx: false,
                    ..
                }
            ) {
                // VGPR 1 begins with a negative i32 in every lane while VGPR 0 is
                // 0..63. This hard-coded result distinguishes signed from unsigned
                // comparison and confirms that non-CMPX writes VCC, not EXEC.
                assert_eq!(mask(expected_states[0].vcc), 0);
                assert_eq!(mask(expected_states[1].vcc), 0);
                assert_eq!(mask(expected_states[0].exec), exec[0]);
                assert_eq!(mask(expected_states[1].exec), exec[1]);
            }
            if let StageBOp::CompareLessFloat {
                kind: CompareKind::UnorderedGreaterFloat,
                dst,
                ..
            } = compare
            {
                let masks = if dst.is_some() {
                    [
                        scalar_mask_value(&expected_states[0], ScalarMask::ScalarPair(4)),
                        scalar_mask_value(&expected_states[1], ScalarMask::ScalarPair(4)),
                    ]
                } else {
                    [mask(expected_states[0].vcc), mask(expected_states[1].vcc)]
                };
                assert_eq!(masks, exec);
                assert_eq!(mask(expected_states[0].exec), exec[0]);
                assert_eq!(mask(expected_states[1].exec), exec[1]);
            }
            if let StageBOp::CompareLessFloat {
                kind: CompareKind::UnorderedNotEqualFloat,
                dst,
                lhs,
                rhs,
                ..
            } = compare
            {
                let masks = if let Some(register) = dst {
                    [
                        scalar_mask_value(&expected_states[0], ScalarMask::ScalarPair(register)),
                        scalar_mask_value(&expected_states[1], ScalarMask::ScalarPair(register)),
                    ]
                } else {
                    [mask(expected_states[0].vcc), mask(expected_states[1].vcc)]
                };
                let expected = if lhs == rhs { [0, 0] } else { exec };
                assert_eq!(masks, expected);
                assert_eq!(mask(expected_states[0].exec), exec[0]);
                assert_eq!(mask(expected_states[1].exec), exec[1]);
            }
            for wave in 0..2 {
                for lane in 0..64 {
                    if exec[wave] & (1u64 << lane) != 0 {
                        let thread = (wave * 64 + lane) as u32;
                        let source = initial_vgpr_value(1, thread) as i32;
                        assert_eq!(
                            load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                            (source >> 4) as u32
                        );
                    }
                }
            }

            let hlsl = stage_b_validation_hlsl_with_vcc(layout, &ops, exec, sentinel).unwrap();
            let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{hlsl}"));
            let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
                &compute,
                &mesh,
                &pixel,
                1,
                record.record_dwords(),
                &[
                    layout.scratch.record_stride_bytes,
                    record.record_dwords() as u32 * 4,
                ],
                &[],
                1,
                [0.0; 4],
                layout.scratch,
            )
            .unwrap();
            for wave in 0..2 {
                let got = super::super::mesh_dispatch::read_logical_state(
                    &memory,
                    layout.scratch,
                    0,
                    wave,
                )
                .unwrap();
                assert_eq!(got, expected_states[wave as usize]);
            }
            for thread in 0..128u32 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, 2).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap()
                );
            }
        }
    }

    #[test]
    fn decoded_float_conversion_edges_match_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let lower = |name: &str, encoding: &str, fields: &[(&str, u64)], literal| {
            let code = encode(name, encoding, fields, literal).unwrap();
            lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let mut ops = Vec::new();
        let move_literal = |value: u32| {
            lower(
                "V_MOV_B32",
                "ENC_VOP1",
                &[("VDST", 0), ("SRC0", 255)],
                Some(value),
            )
        };
        for (dst, value) in [
            (1, 1.00048828125f32.to_bits()),
            (2, 2.0f32.powi(-24).to_bits()),
            (3, f32::MAX.to_bits()),
        ] {
            ops.push(move_literal(value));
            ops.push(lower(
                "V_CVT_F16_F32",
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            ));
        }
        ops.push(lower(
            "V_CVT_F16_F32",
            "ENC_VOP1",
            &[("VDST", 4), ("SRC0", 255)],
            Some(f32::NAN.to_bits()),
        ));
        for (dst, value) in [(5, 0x0001), (6, 0x8000), (7, 0x7c00), (8, 0x7e01)] {
            ops.push(move_literal(value));
            ops.push(lower(
                "V_CVT_F32_F16",
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            ));
        }
        for (dst, value) in [
            (9, 0.0f32.to_bits()),
            (10, 1.75f32.to_bits()),
            (11, 4_294_967_040.0f32.to_bits()),
            (12, 4_294_967_296.0f32.to_bits()),
        ] {
            ops.push(move_literal(value));
            ops.push(lower(
                "V_CVT_U32_F32",
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            ));
        }
        ops.push(move_literal(1.0f32.to_bits()));
        ops.push(lower(
            "V_CVT_F16_F32",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 14),
                ("SRC0", 249),
                ("VSRC0", 0),
                ("SRC0_SEL", 6),
                ("DST_SEL", 4),
                ("DST_UNUSED", 2),
            ],
            None,
        ));
        ops.push(lower(
            "V_CVT_F16_F32",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 15),
                ("SRC0", 249),
                ("S0", 1),
                ("VSRC0", 242),
                ("SRC0_SEL", 6),
                ("DST_SEL", 5),
                ("DST_UNUSED", 2),
            ],
            None,
        ));
        ops.push(move_literal(0x3c00_0000));
        ops.push(lower(
            "V_CVT_F32_F16",
            "VOP1_VOP_SDWA",
            &[
                ("VDST", 13),
                ("SRC0", 249),
                ("VSRC0", 0),
                ("SRC0_SEL", 5),
                ("DST_SEL", 6),
            ],
            None,
        ));
        ops.push(lower(
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[
                ("VDST", 0),
                ("SRC0", 245),
                ("SRC1", 247),
                ("ABS", 1),
                ("NEG", 2),
            ],
            None,
        ));
        ops.push(lower(
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[("VDST", 2), ("SRC0", 245), ("SRC1", 242), ("ABS", 1)],
            None,
        ));
        ops.push(move_literal(f32::NAN.to_bits()));
        ops.push(lower(
            "V_CMP_LT_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 256), ("SRC1", 242), ("NEG", 1)],
            None,
        ));
        ops.push(move_literal((-0.0f32).to_bits()));
        ops.push(lower(
            "V_CMP_EQ_F32",
            "ENC_VOP3",
            &[("VDST", 6), ("SRC0", 256), ("SRC1", 128), ("ABS", 1)],
            None,
        ));
        ops.push(move_literal(1.0f32.to_bits()));
        ops.push(lower(
            "V_CMP_LG_F32",
            "ENC_VOP3",
            &[("VDST", 8), ("SRC0", 256), ("SRC1", 244), ("NEG", 2)],
            None,
        ));
        ops.push(move_literal(f32::NAN.to_bits()));
        ops.push(lower(
            "V_CMP_LG_F32",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", 256), ("SRC1", 244)],
            None,
        ));
        ops.push(move_literal((-0.0f32).to_bits()));
        ops.push(lower(
            "V_CMP_EQ_F32",
            "ENC_VOPC",
            &[("SRC0", 256), ("VSRC1", 0)],
            None,
        ));
        ops.push(lower(
            "V_CVT_U32_F32",
            "ENC_VOP1",
            &[("VDST", 0), ("SRC0", 255)],
            Some((-1.0f32).to_bits()),
        ));

        let layout = VgprScratchLayout::new(16, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 17) | (1u64 << 63),
            (1u64 << 0) | (1u64 << 5) | (1u64 << 40),
        ];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 1).unwrap() & 0xffff,
            0x3c00
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 2).unwrap() & 0xffff,
            1
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 3).unwrap() & 0xffff,
            0x7bff
        );
        assert!(f16_to_f32(load_vgpr(&expected_memory, layout, 0, 0, 4).unwrap() as u16).is_nan());
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 5).unwrap(),
            2.0f32.powi(-24).to_bits()
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 6).unwrap(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 7).unwrap(),
            f32::INFINITY.to_bits()
        );
        assert!(f32::from_bits(load_vgpr(&expected_memory, layout, 0, 0, 8).unwrap()).is_nan());
        assert_eq!(load_vgpr(&expected_memory, layout, 0, 0, 0).unwrap(), 0);
        assert_eq!(load_vgpr(&expected_memory, layout, 0, 0, 9).unwrap(), 0);
        assert_eq!(load_vgpr(&expected_memory, layout, 0, 0, 10).unwrap(), 1);
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 11).unwrap(),
            0xffff_ff00
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 12).unwrap(),
            u32::MAX
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 14).unwrap() & 0xffff,
            0x3c00
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 15).unwrap() >> 16,
            0x3c00
        );
        assert_eq!(
            load_vgpr(&expected_memory, layout, 0, 0, 13).unwrap(),
            1.0f32.to_bits()
        );
        for wave in 0..2 {
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(0)),
                exec[wave]
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(2)),
                0
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(4)),
                0
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(6)),
                exec[wave]
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(8)),
                exec[wave]
            );
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(10)),
                0
            );
            assert_eq!(mask(expected_states[wave].vcc), exec[wave]);
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        assert_eq!(load_vgpr(&memory, layout, 0, 0, 0).unwrap(), 0);
        for wave in 0..2 {
            let got =
                super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave)
                    .unwrap();
            assert_eq!(
                &got.scalar[..12],
                &expected_states[wave as usize].scalar[..12]
            );
            assert_eq!(mask(got.vcc), exec[wave as usize]);
            assert_eq!(mask(got.exec), exec[wave as usize]);
        }
        for thread in 0..128 {
            let active = exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0;
            for register in 0..16 {
                let got = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                if active && register == 4 {
                    assert_eq!(got >> 16, expected >> 16);
                    assert!(f16_to_f32(got as u16).is_nan(), "GPU half NaN {got:#010x}");
                } else if active && register == 8 {
                    assert!(f32::from_bits(got).is_nan());
                } else {
                    assert_eq!(got, expected, "thread {thread} register {register}");
                }
            }
        }
    }

    #[test]
    fn decoded_fmaak_uses_literal_addend_and_fused_rounding() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let bytes = encode(
            "V_FMAAK_F32",
            "ENC_VOP2",
            &[("VDST", 2), ("SRC0", 256), ("VSRC1", 1)],
            Some((-1.0f32).to_bits()),
        )
        .unwrap();
        assert_eq!(bytes, [0x00, 0x03, 0x04, 0x5a, 0x00, 0x00, 0x80, 0xbf]);
        let op = lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap();
        assert_eq!(
            op,
            StageBOp::FloatFma {
                dst: 2,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Constant((-1.0f32).to_bits())),
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            }
        );

        let move_a = encode(
            "V_MOV_B32",
            "ENC_VOP1",
            &[("VDST", 0), ("SRC0", 255)],
            Some(0x3f80_0001),
        )
        .unwrap();
        let move_b = encode(
            "V_MOV_B32",
            "ENC_VOP1",
            &[("VDST", 1), ("SRC0", 255)],
            Some(0x3f7f_fffe),
        )
        .unwrap();
        let ops = [
            lower_stage_b_one(&rdna2::decode(&move_a, 0).unwrap()).unwrap(),
            lower_stage_b_one(&rdna2::decode(&move_b, 0).unwrap()).unwrap(),
            op,
        ];
        let layout = VgprScratchLayout::new(3, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(
            &mut expected_memory,
            layout,
            &mut expected_states,
            &ops,
        )
        .unwrap();
        for thread in 0..128 {
            let active = exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0;
            let expected = if active {
                0xa880_0000
            } else {
                initial_vgpr_value(2, thread)
            };
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                expected,
                "thread {thread}"
            );
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 2).unwrap(),
                load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                "thread {thread}"
            );
        }
    }

    #[test]
    fn decoded_madak_and_min3_keep_nonfused_and_minimum_semantics() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let madak = encode(
            "V_MADAK_F32",
            "ENC_VOP2",
            &[("VDST", 2), ("SRC0", 256), ("VSRC1", 1)],
            Some((-1.0f32).to_bits()),
        )
        .unwrap();
        let min3 = encode(
            "V_MIN3_F32",
            "ENC_VOP3",
            &[("VDST", 3), ("SRC0", 245), ("SRC1", 242), ("SRC2", 244)],
            None,
        )
        .unwrap();
        let max_u32 = encode(
            "V_MAX_U32",
            "ENC_VOP2",
            &[("VDST", 6), ("SRC0", 193), ("VSRC1", 0)],
            None,
        )
        .unwrap();
        assert_eq!(madak, [0x00, 0x03, 0x04, 0x42, 0x00, 0x00, 0x80, 0xbf]);
        assert_eq!(min3, [0x03, 0x00, 0x51, 0xd5, 0xf5, 0xe4, 0xd1, 0x03]);

        let lower = |bytes: &[u8]| lower_stage_b_one(&rdna2::decode(bytes, 0).unwrap()).unwrap();
        let move_value = |dst, value| {
            encode(
                "V_MOV_B32",
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 255)],
                Some(value),
            )
            .unwrap()
        };
        let nan_min = encode(
            "V_MIN3_F32",
            "ENC_VOP3",
            &[("VDST", 4), ("SRC0", 255), ("SRC1", 242), ("SRC2", 244)],
            Some(f32::NAN.to_bits()),
        )
        .unwrap();
        let zero_min = encode(
            "V_MIN3_F32",
            "ENC_VOP3",
            &[("VDST", 5), ("SRC0", 255), ("SRC1", 242), ("SRC2", 244)],
            Some((-0.0f32).to_bits()),
        )
        .unwrap();
        let move_a = move_value(0, 0x3f80_0001);
        let move_b = move_value(1, 0x3f7f_fffe);
        let ops = [
            lower(&move_a),
            lower(&move_b),
            lower(&madak),
            lower(&min3),
            lower(&nan_min),
            lower(&zero_min),
            lower(&max_u32),
        ];
        assert_eq!(
            ops[2],
            StageBOp::FloatMad {
                dst: 2,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Constant((-1.0f32).to_bits())),
                modifiers: FloatModifiers::default(),
            }
        );
        assert_eq!(
            ops[3],
            StageBOp::FloatTernary {
                kind: FloatTernaryKind::Minimum3,
                dst: 3,
                a: VectorSource::Constant((-2.0f32).to_bits()),
                b: VectorSource::Constant(1.0f32.to_bits()),
                c: VectorSource::Constant(2.0f32.to_bits()),
                modifiers: FloatModifiers::default(),
            }
        );

        let layout = VgprScratchLayout::new(7, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for thread in 0..128 {
            let active = exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0;
            for (register, expected) in [
                (2, 0.0f32.to_bits()),
                (3, (-2.0f32).to_bits()),
                (4, 1.0f32.to_bits()),
                (5, (-0.0f32).to_bits()),
                (6, u32::MAX),
            ] {
                assert_eq!(
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    if active { expected } else { initial_vgpr_value(register, thread) },
                    "thread {thread} register {register}",
                );
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            for register in 2..=6 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    "thread {thread} register {register}",
                );
            }
        }
    }

    #[test]
    fn decoded_legacy_fma_returns_addend_for_zero_multiplicand_and_matches_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let bytes = encode(
            "V_FMA_LEGACY_F32",
            "ENC_VOP3",
            &[("VDST", 3), ("SRC0", 256), ("SRC1", 257), ("SRC2", 258)],
            None,
        )
        .unwrap();
        assert_eq!(bytes, [0x03, 0x00, 0x40, 0xd5, 0x00, 0x03, 0x0a, 0x04]);
        let legacy = lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap();
        assert_eq!(
            legacy,
            StageBOp::FloatFma {
                dst: 3,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Vgpr(2)),
                modifiers: FloatModifiers::default(),
                legacy_zero: true,
            }
        );
        let selected = encode(
            "V_FMA_LEGACY_F32",
            "ENC_VOP3",
            &[
                ("VDST", 3),
                ("SRC0", 256),
                ("SRC1", 257),
                ("SRC2", 258),
                ("OP_SEL", 1),
            ],
            None,
        )
        .unwrap();
        assert!(lower_stage_b_one(&rdna2::decode(&selected, 0).unwrap()).is_err());

        let lower_move = |dst, value| {
            let code = encode(
                "V_MOV_B32",
                "ENC_VOP1",
                &[("VDST", dst), ("SRC0", 255)],
                Some(value),
            )
            .unwrap();
            lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let ops = [
            lower_move(0, (-0.0f32).to_bits()),
            lower_move(1, f32::INFINITY.to_bits()),
            lower_move(2, 42.25f32.to_bits()),
            legacy,
            lower_move(5, 0x3f80_0001),
            lower_move(6, 0x3f7f_fffe),
            StageBOp::FloatFma {
                dst: 4,
                a: VectorSource::Vgpr(5),
                b: VectorSource::Vgpr(6),
                c: Some(VectorSource::Constant((-1.0f32).to_bits())),
                modifiers: FloatModifiers::default(),
                legacy_zero: true,
            },
        ];
        let layout = VgprScratchLayout::new(7, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for thread in 0..128 {
            let active = exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0;
            let expected = if active {
                42.25f32.to_bits()
            } else {
                initial_vgpr_value(3, thread)
            };
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 3).unwrap(), expected);
            let fused = if active {
                0xa880_0000
            } else {
                initial_vgpr_value(4, thread)
            };
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 4).unwrap(), fused);
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[layout.scratch.record_stride_bytes, record.record_dwords() as u32 * 4],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            for register in [3, 4] {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    "thread {thread} register {register}"
                );
            }
        }
    }

    #[test]
    fn decoded_i32_conversion_and_floor_edges_match_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let lower = |name: &str, fields: &[(&str, u64)], literal| {
            let code = encode(name, "ENC_VOP1", fields, literal).unwrap();
            lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let cvt_bytes = encode(
            "V_CVT_I32_F32",
            "ENC_VOP1",
            &[("VDST", 1), ("SRC0", 256)],
            None,
        )
        .unwrap();
        let floor_bytes = encode(
            "V_FLOOR_F32",
            "ENC_VOP1",
            &[("VDST", 7), ("SRC0", 256)],
            None,
        )
        .unwrap();
        let ceil_bytes = encode(
            "V_CEIL_F32",
            "ENC_VOP1",
            &[("VDST", 11), ("SRC0", 256)],
            None,
        )
        .unwrap();
        let round_bytes = encode(
            "V_RNDNE_F32",
            "ENC_VOP1",
            &[("VDST", 15), ("SRC0", 256)],
            None,
        )
        .unwrap();
        assert_eq!(cvt_bytes, [0x00, 0x11, 0x02, 0x7e]);
        assert_eq!(floor_bytes, [0x00, 0x49, 0x0e, 0x7e]);
        assert_eq!(ceil_bytes, [0x00, 0x45, 0x16, 0x7e]);
        assert_eq!(round_bytes, [0x00, 0x47, 0x1e, 0x7e]);

        let mut ops = Vec::new();
        for (dst, value) in [
            (1, f32::NAN.to_bits()),
            (2, f32::NEG_INFINITY.to_bits()),
            (3, (-2_147_483_648.0f32).to_bits()),
            (4, 2_147_483_648.0f32.to_bits()),
            (5, 3.75f32.to_bits()),
            (6, (-3.75f32).to_bits()),
        ] {
            ops.push(lower(
                "V_MOV_B32",
                &[("VDST", 0), ("SRC0", 255)],
                Some(value),
            ));
            ops.push(lower(
                "V_CVT_I32_F32",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            ));
        }
        for (dst, value) in [
            (7, (-0.0f32).to_bits()),
            (8, (-3.25f32).to_bits()),
            (9, f32::INFINITY.to_bits()),
            (10, 0x7fc1_2345),
        ] {
            ops.push(lower(
                "V_MOV_B32",
                &[("VDST", 0), ("SRC0", 255)],
                Some(value),
            ));
            ops.push(lower(
                "V_FLOOR_F32",
                &[("VDST", dst), ("SRC0", 256)],
                None,
            ));
        }
        for (name, dst, value) in [
            ("V_CEIL_F32", 11, (-3.25f32).to_bits()),
            ("V_CEIL_F32", 12, (-0.0f32).to_bits()),
            ("V_CEIL_F32", 13, f32::INFINITY.to_bits()),
            ("V_CEIL_F32", 14, 0x7fc1_2345),
            ("V_RNDNE_F32", 15, 2.5f32.to_bits()),
            ("V_RNDNE_F32", 16, 3.5f32.to_bits()),
            ("V_RNDNE_F32", 17, (-2.5f32).to_bits()),
            ("V_RNDNE_F32", 18, (-0.0f32).to_bits()),
        ] {
            ops.push(lower(
                "V_MOV_B32",
                &[("VDST", 0), ("SRC0", 255)],
                Some(value),
            ));
            ops.push(lower(name, &[("VDST", dst), ("SRC0", 256)], None));
        }

        let layout = VgprScratchLayout::new(19, 1).unwrap();
        let exec = [(1u64 << 0) | (1u64 << 63), (1u64 << 5) | (1u64 << 40)];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        let expected = [
            0,
            i32::MIN as u32,
            i32::MIN as u32,
            i32::MAX as u32,
            3,
            (-3i32) as u32,
            (-0.0f32).to_bits(),
            (-4.0f32).to_bits(),
            f32::INFINITY.to_bits(),
        ];
        for (index, value) in expected.into_iter().enumerate() {
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, 0, index as u32 + 1).unwrap(),
                value
            );
        }
        assert!(f32::from_bits(load_vgpr(&expected_memory, layout, 0, 0, 10).unwrap()).is_nan());
        for (register, value) in [
            (11, (-3.0f32).to_bits()),
            (12, (-0.0f32).to_bits()),
            (13, f32::INFINITY.to_bits()),
            (15, 2.0f32.to_bits()),
            (16, 4.0f32.to_bits()),
            (17, (-2.0f32).to_bits()),
            (18, (-0.0f32).to_bits()),
        ] {
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, 0, register).unwrap(),
                value
            );
        }
        assert!(f32::from_bits(load_vgpr(&expected_memory, layout, 0, 0, 14).unwrap()).is_nan());

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128 {
            for register in 0..19 {
                let got = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                if matches!(register, 10 | 14)
                    && exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0
                {
                    assert!(f32::from_bits(got).is_nan());
                } else {
                    assert_eq!(got, expected, "thread {thread} register {register}");
                }
            }
        }
    }

    #[test]
    fn decoded_sdwa_dpp_integer_sources_match_cpu_and_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let lower = |name: &str, encoding: &str, fields: &[(&str, u64)]| {
            let code = encode(name, encoding, fields, None).unwrap();
            lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let observed_shift = {
            let bytes: Vec<u8> = [0x347842f9u32, 0x02860600]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            let op = lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap();
            assert_eq!(op, StageBOp::Binary {
                kind: VectorBinaryKind::LshlReverse,
                dst: 60,
                lhs: VectorSource::Sdwa {
                    register: 0,
                    scalar: true,
                    select: 6,
                    sign_extend: false,
                },
                rhs: VectorSource::Sdwa {
                    register: 33,
                    scalar: false,
                    select: 2,
                    sign_extend: false,
                },
            });
            op
        };
        let dpp16 = |dst, fi| {
            lower(
                "V_MIN_U32",
                "VOP2_VOP_DPP16",
                &[
                    ("VDST", dst),
                    ("SRC0", 250),
                    ("VSRC0", 0),
                    ("VSRC1", 1),
                    ("DPP_CTRL", 0x140),
                    ("ROW_MASK", 15),
                    ("BANK_MASK", 15),
                    ("BOUND_CTRL", 1),
                    ("FI", fi),
                ],
            )
        };
        let ops = vec![
            observed_shift,
            dpp16(2, 0),
            dpp16(3, 1),
            lower(
                "V_MIN_U32",
                "VOP2_VOP_DPP8",
                &[
                    ("VDST", 4),
                    ("SRC0", 234),
                    ("VSRC0", 0),
                    ("VSRC1", 1),
                    ("LANE_SEL_0", 1),
                    ("LANE_SEL_1", 0),
                    ("LANE_SEL_2", 3),
                    ("LANE_SEL_3", 2),
                    ("LANE_SEL_4", 5),
                    ("LANE_SEL_5", 4),
                    ("LANE_SEL_6", 7),
                    ("LANE_SEL_7", 6),
                ],
            ),
            lower(
                "V_SUB_NC_U32",
                "VOP2_VOP_SDWA",
                &[
                    ("VDST", 5),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("S1", 1),
                    ("VSRC1", 20),
                    ("SRC0_SEL", 2),
                    ("SRC0_SEXT", 1),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
            ),
            lower(
                "V_SUB_NC_U32",
                "VOP2_VOP_SDWA",
                &[
                    ("VDST", 6),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("S1", 1),
                    ("VSRC1", 20),
                    ("SRC0_SEL", 2),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
            ),
            lower(
                "V_CMP_EQ_U32",
                "VOPC_VOP_SDWA_SDST_ENC",
                &[
                    ("SRC0", 249),
                    ("VSRC0", 0),
                    ("S1", 1),
                    ("VSRC1", 20),
                    ("SRC0_SEL", 0),
                    ("SRC1_SEL", 6),
                    ("SD", 1),
                    ("SDST", 22),
                ],
            ),
            lower(
                "V_CMPX_LT_I32",
                "VOPC_VOP_SDWA_SDST_ENC",
                &[
                    ("SRC0", 249),
                    ("S0", 1),
                    ("VSRC0", 15),
                    ("VSRC1", 0),
                    ("SRC0_SEL", 6),
                    ("SRC1_SEL", 0),
                ],
            ),
        ];
        let layout = VgprScratchLayout::new(61, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 7) | (1u64 << 31) | (1u64 << 40),
            (1u64 << 0) | (1u64 << 6) | (1u64 << 23) | (1u64 << 47),
        ];
        let vcc = [0xaaaa_5555_dead_beef, 0x0123_4567_89ab_cdef];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        for wave in 0..2 {
            expected_states[wave].vcc = [vcc[wave] as u32, (vcc[wave] >> 32) as u32];
        }
        let before = expected_memory.clone();
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2 {
            let base = (wave * 64) as u32;
            assert_eq!(load_vgpr(&expected_memory, layout, 0, base, 2).unwrap(), 0);
            assert_eq!(load_vgpr(&expected_memory, layout, 0, base, 3).unwrap(), 15);
            assert_eq!(load_vgpr(&expected_memory, layout, 0, base, 4).unwrap(), 1);
            let selected = ((initial_vgpr_value(1, base) >> 16) & 0xff) as u8;
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, base, 5).unwrap(),
                (selected as i8 as i32) as u32
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, base, 6).unwrap(),
                selected as u32
            );
            assert_ne!(
                load_vgpr(&expected_memory, layout, 0, base, 5).unwrap(),
                load_vgpr(&expected_memory, layout, 0, base, 6).unwrap()
            );
            assert_eq!(mask(expected_states[wave].vcc), vcc[wave]);
            assert_eq!(mask(expected_states[wave].exec), exec[wave] & !1);
            assert_eq!(
                scalar_mask_value(&expected_states[wave], ScalarMask::ScalarPair(22)),
                1
            );
        }
        for thread in 0..128u32 {
            let active = exec[(thread / 64) as usize] & (1u64 << (thread % 64)) != 0;
            let expected_shift = if active {
                (initial_vgpr_value(33, thread) >> 16) & 0xff
            } else {
                initial_vgpr_value(60, thread)
            };
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 60).unwrap(),
                expected_shift,
                "thread {thread} observed SDWA shift",
            );
            if exec[(thread / 64) as usize] & (1u64 << (thread % 64)) == 0 {
                for register in 0..7 {
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                        load_vgpr(&before, layout, 0, thread, register).unwrap(),
                        "inactive thread {thread} register {register}"
                    );
                }
            }
        }

        let hlsl = stage_b_validation_hlsl_with_vcc(layout, &ops, exec, vcc).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got = super::super::mesh_dispatch::read_logical_state(
                &memory,
                layout.scratch,
                0,
                wave as u32,
            )
            .unwrap();
            assert_eq!(got.exec, expected_states[wave].exec, "wave {wave} EXEC");
            assert_eq!(got.vcc, expected_states[wave].vcc, "wave {wave} VCC");
            assert_eq!(&got.scalar[22..24], &expected_states[wave].scalar[22..24]);
        }
        for thread in 0..128u32 {
            for register in 0..7u32 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, register).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                    "thread {thread} register {register}"
                );
            }
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 60).unwrap(),
                load_vgpr(&expected_memory, layout, 0, thread, 60).unwrap(),
                "thread {thread} register 60",
            );
        }
    }

    #[test]
    fn gpu_matches_cpu_through_checked_u6_scratch() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = VgprScratchLayout::new(29, 1).unwrap();
        let exec = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        let ops = ops();
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for wave in 0..2 {
            let got =
                super::super::mesh_dispatch::read_logical_state(&memory, layout.scratch, 0, wave)
                    .unwrap();
            assert_eq!(got.exec, expected_states[wave as usize].exec);
            assert_eq!(got.vcc, expected_states[wave as usize].vcc);
            assert_eq!(got.scalar[10], expected_states[wave as usize].scalar[10]);
            for register in 20..26 {
                assert_eq!(
                    got.scalar[register], expected_states[wave as usize].scalar[register],
                    "wave {wave} scalar {register}"
                );
            }
        }
        for thread in 0..128 {
            for reg in 0..29 {
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, reg).unwrap(),
                    load_vgpr(&expected_memory, layout, 0, thread, reg).unwrap(),
                    "thread {thread} reg {reg}"
                );
            }
        }
    }

    #[test]
    fn decoded_byte_multiply_and_residual_and_forms_match_actual_u6() {
        if super::super::dxc::location().is_err() {
            return;
        }
        use crate::gpu::rdna2::encode;

        let lower = |name: &str, encoding: &str, fields: &[(&str, u64)], literal: Option<u32>| {
            let code = encode(name, encoding, fields, literal).unwrap();
            lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
        };
        let reject = |encoding: &str, fields: &[(&str, u64)]| {
            let code = encode("V_MUL_F32", encoding, fields, None).unwrap();
            assert!(lower_stage_b_one(&rdna2::decode(&code, 0).unwrap()).is_err());
        };
        reject(
            "VOP2_VOP_SDWA",
            &[
                ("VDST", 6),
                ("SRC0", 249),
                ("VSRC0", 0),
                ("VSRC1", 1),
                ("SRC0_SEL", 6),
                ("SRC1_SEL", 6),
                ("DST_SEL", 6),
            ],
        );
        reject(
            "VOP2_VOP_DPP16",
            &[
                ("VDST", 6),
                ("SRC0", 250),
                ("VSRC0", 0),
                ("VSRC1", 1),
                ("DPP_CTRL", 0x140),
                ("ROW_MASK", 15),
                ("BANK_MASK", 15),
                ("BOUND_CTRL", 1),
            ],
        );
        reject(
            "ENC_VOP3",
            &[("VDST", 6), ("SRC0", 256), ("SRC1", 257), ("OP_SEL", 1)],
        );
        let ops = vec![
            lower(
                "V_AND_B32",
                "VOP2_VOP_DPP16",
                &[
                    ("VDST", 2),
                    ("SRC0", 250),
                    ("VSRC0", 1),
                    ("VSRC1", 1),
                    ("DPP_CTRL", 0x140),
                    ("ROW_MASK", 15),
                    ("BANK_MASK", 15),
                    ("BOUND_CTRL", 1),
                    ("FI", 1),
                ],
                None,
            ),
            lower(
                "V_AND_B32",
                "VOP2_VOP_SDWA",
                &[
                    ("VDST", 3),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("VSRC1", 1),
                    ("SRC0_SEL", 2),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
                None,
            ),
            lower(
                "V_CVT_F32_UBYTE2",
                "ENC_VOP1",
                &[("VDST", 4), ("SRC0", 255)],
                Some(0xff80_7f00),
            ),
            lower(
                "V_CVT_F32_UBYTE3",
                "ENC_VOP1",
                &[("VDST", 5), ("SRC0", 255)],
                Some(0xff80_7f00),
            ),
            lower(
                "V_MUL_F32",
                "ENC_VOP3",
                &[("VDST", 6), ("SRC0", 128), ("SRC1", 255)],
                Some(f32::INFINITY.to_bits()),
            ),
            lower(
                "V_MUL_F32",
                "ENC_VOP3",
                &[("VDST", 7), ("SRC0", 255), ("SRC1", 244)],
                Some((-0.0f32).to_bits()),
            ),
            lower(
                "V_MUL_F32",
                "ENC_VOP3",
                &[("VDST", 8), ("SRC0", 255), ("SRC1", 242)],
                Some(f32::NAN.to_bits()),
            ),
            lower(
                "V_OR_B32",
                "ENC_VOP2",
                &[("VDST", 9), ("SRC0", 129), ("VSRC1", 0)],
                None,
            ),
            lower(
                "V_XOR_B32",
                "ENC_VOP2",
                &[("VDST", 10), ("SRC0", 129), ("VSRC1", 1)],
                None,
            ),
            lower(
                "V_BCNT_U32_B32",
                "ENC_VOP3",
                &[("VDST", 11), ("SRC0", 129), ("SRC1", 135)],
                None,
            ),
            lower(
                "V_ADD3_U32",
                "ENC_VOP3",
                &[("VDST", 12), ("SRC0", 129), ("SRC1", 130), ("SRC2", 131)],
                None,
            ),
            lower(
                "V_ADD_NC_U32",
                "VOP2_VOP_SDWA",
                &[
                    ("VDST", 13),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("VSRC1", 0),
                    ("SRC0_SEL", 2),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
                None,
            ),
            lower(
                "V_OR_B32",
                "VOP2_VOP_DPP16",
                &[
                    ("VDST", 14),
                    ("SRC0", 250),
                    ("VSRC0", 1),
                    ("VSRC1", 0),
                    ("DPP_CTRL", 0x140),
                    ("ROW_MASK", 15),
                    ("BANK_MASK", 15),
                    ("BOUND_CTRL", 1),
                    ("FI", 1),
                ],
                None,
            ),
            lower(
                "V_XOR_B32",
                "VOP2_VOP_SDWA",
                &[
                    ("VDST", 15),
                    ("SRC0", 249),
                    ("VSRC0", 1),
                    ("VSRC1", 0),
                    ("SRC0_SEL", 2),
                    ("SRC1_SEL", 6),
                    ("DST_SEL", 6),
                ],
                None,
            ),
            lower(
                "V_ADD_NC_U32",
                "VOP2_VOP_DPP16",
                &[
                    ("VDST", 16),
                    ("SRC0", 250),
                    ("VSRC0", 1),
                    ("VSRC1", 0),
                    ("DPP_CTRL", 0x140),
                    ("ROW_MASK", 15),
                    ("BANK_MASK", 15),
                    ("BOUND_CTRL", 1),
                    ("FI", 1),
                ],
                None,
            ),
        ];
        assert!(matches!(
            ops[0],
            StageBOp::Binary {
                kind: VectorBinaryKind::And,
                lhs: VectorSource::Dpp16RowMirror { .. },
                ..
            }
        ));
        assert!(matches!(
            ops[1],
            StageBOp::Binary {
                kind: VectorBinaryKind::And,
                lhs: VectorSource::Sdwa { select: 2, .. },
                ..
            }
        ));
        assert!(matches!(ops[2], StageBOp::ConvertUByte { byte: 2, .. }));
        assert!(matches!(ops[3], StageBOp::ConvertUByte { byte: 3, .. }));
        assert!(matches!(
            ops[4],
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Multiply,
                ..
            }
        ));

        let layout = VgprScratchLayout::new(17, 1).unwrap();
        let exec = [1u64, 1u64];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        let before = expected_memory.clone();
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for wave in 0..2u32 {
            let thread = wave * 64;
            let dpp = load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap();
            assert_ne!(
                dpp, 0,
                "wave {wave} DPP result must observe the mirrored lane"
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 4).unwrap(),
                128.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 5).unwrap(),
                255.0f32.to_bits()
            );
            assert!(
                f32::from_bits(load_vgpr(&expected_memory, layout, 0, thread, 6).unwrap()).is_nan()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 7).unwrap(),
                (-0.0f32).to_bits()
            );
            assert!(
                f32::from_bits(load_vgpr(&expected_memory, layout, 0, thread, 8).unwrap()).is_nan()
            );
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 9).unwrap(), 1);
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 10).unwrap(),
                initial_vgpr_value(1, thread) ^ 1
            );
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 11).unwrap(), 8);
            assert_eq!(load_vgpr(&expected_memory, layout, 0, thread, 12).unwrap(), 6);
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 13).unwrap(),
                (initial_vgpr_value(1, thread) >> 16) & 0xff
            );
            let mirrored = wave * 64 + 15;
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 14).unwrap(),
                initial_vgpr_value(1, mirrored)
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 15).unwrap(),
                (initial_vgpr_value(1, thread) >> 16) & 0xff
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 16).unwrap(),
                initial_vgpr_value(1, mirrored)
            );
        }
        for thread in 0..128u32 {
            if thread % 64 != 0 {
                for register in 0..17u32 {
                    assert_eq!(
                        load_vgpr(&expected_memory, layout, 0, thread, register).unwrap(),
                        load_vgpr(&before, layout, 0, thread, register).unwrap(),
                        "inactive thread {thread} register {register}"
                    );
                }
            }
        }

        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 0..17u32 {
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                let actual = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                if f32::from_bits(expected).is_nan() && matches!(register, 6 | 8) {
                    assert!(f32::from_bits(actual).is_nan());
                } else {
                    assert_eq!(actual, expected, "thread {thread} register {register}");
                }
            }
        }
    }

    #[test]
    fn final_float_math_edges_match_cpu_reference() {
        let layout = VgprScratchLayout::new(5, 1).unwrap();
        let (mut memory, mut states) = init(layout, [1, 1]);
        let modifiers = FloatModifiers::default();
        let ops = [
            StageBOp::FloatMath {
                kind: FloatMathKind::Exp2,
                dst: 0,
                src: VectorSource::Constant(2.0f32.to_bits()),
                modifiers,
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Fract,
                dst: 1,
                src: VectorSource::Constant((-1.25f32).to_bits()),
                modifiers,
            },
            StageBOp::FloatMath {
                kind: FloatMathKind::Log2,
                dst: 2,
                src: VectorSource::Constant(8.0f32.to_bits()),
                modifiers,
            },
            StageBOp::FloatBinary {
                kind: FloatBinaryKind::Minimum,
                dst: 3,
                lhs: VectorSource::Constant(f32::NAN.to_bits()),
                rhs: VectorSource::Constant((-0.0f32).to_bits()),
                modifiers,
            },
            StageBOp::FloatTernary {
                kind: FloatTernaryKind::Maximum3,
                dst: 4,
                a: VectorSource::Constant(f32::NAN.to_bits()),
                b: VectorSource::Constant((-2.0f32).to_bits()),
                c: VectorSource::Constant(3.0f32.to_bits()),
                modifiers,
            },
        ];
        execute_stage_b(&mut memory, layout, &mut states, &ops).unwrap();
        for thread in [0, 64] {
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 0).unwrap(),
                4.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 1).unwrap(),
                0.75f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 2).unwrap(),
                3.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 3).unwrap(),
                (-0.0f32).to_bits()
            );
            assert_eq!(
                load_vgpr(&memory, layout, 0, thread, 4).unwrap(),
                3.0f32.to_bits()
            );
        }
    }

    #[test]
    fn decoded_final_float_math_edges_match_actual_u6() {
        use crate::gpu::rdna2::encode;
        let lower = |name: &str, enc: &str, fields: &[(&str, u64)], literal: Option<u32>| {
            let bytes = encode(name, enc, fields, literal).unwrap();
            lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).unwrap()
        };
        let ops = [
            lower(
                "V_EXP_F32",
                "ENC_VOP1",
                &[("VDST", 0), ("SRC0", 255)],
                Some(f32::NEG_INFINITY.to_bits()),
            ),
            lower("V_EXP_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
            lower(
                "V_FRACT_F32",
                "ENC_VOP1",
                &[("VDST", 2), ("SRC0", 255)],
                Some((-0.0f32).to_bits()),
            ),
            lower(
                "V_FRACT_F32",
                "ENC_VOP1",
                &[("VDST", 3), ("SRC0", 255)],
                Some(f32::INFINITY.to_bits()),
            ),
            lower("V_LOG_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)], None),
            lower("V_LOG_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 244)], None),
            lower(
                "V_MIN_F32",
                "ENC_VOP3",
                &[("VDST", 6), ("SRC0", 255), ("SRC1", 242)],
                Some(f32::NAN.to_bits()),
            ),
            lower(
                "V_MIN_F32",
                "ENC_VOP3",
                &[("VDST", 7), ("SRC0", 242), ("SRC1", 255)],
                Some(f32::NAN.to_bits()),
            ),
            lower(
                "V_MIN_F32",
                "ENC_VOP3",
                &[("VDST", 8), ("SRC0", 255), ("SRC1", 128)],
                Some((-0.0f32).to_bits()),
            ),
            lower(
                "V_MAX3_F32",
                "ENC_VOP3",
                &[("VDST", 9), ("SRC0", 255), ("SRC1", 245), ("SRC2", 244)],
                Some(f32::NAN.to_bits()),
            ),
            lower(
                "V_MAX3_F32",
                "ENC_VOP3",
                &[("VDST", 10), ("SRC0", 255), ("SRC1", 243), ("SRC2", 244)],
                Some(f32::NEG_INFINITY.to_bits()),
            ),
            lower(
                "V_EXP_F32",
                "ENC_VOP1",
                &[("VDST", 11), ("SRC0", 255)],
                Some(f32::MAX.to_bits()),
            ),
            lower(
                "V_LOG_F32",
                "ENC_VOP1",
                &[("VDST", 12), ("SRC0", 255)],
                Some(f32::from_bits(1).to_bits()),
            ),
        ];
        let layout = VgprScratchLayout::new(13, 1).unwrap();
        let exec = [1u64, 1u64];
        let (mut expected_memory, mut expected_states) = init(layout, exec);
        execute_stage_b(&mut expected_memory, layout, &mut expected_states, &ops).unwrap();
        for thread in [0, 64] {
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 0).unwrap(),
                0.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 1).unwrap(),
                1.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 2).unwrap(),
                0.0f32.to_bits()
            );
            assert!(
                f32::from_bits(load_vgpr(&expected_memory, layout, 0, thread, 3).unwrap()).is_nan()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 4).unwrap(),
                f32::NEG_INFINITY.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 5).unwrap(),
                1.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 6).unwrap(),
                1.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 7).unwrap(),
                1.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 8).unwrap(),
                (-0.0f32).to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 9).unwrap(),
                2.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 10).unwrap(),
                2.0f32.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 11).unwrap(),
                f32::INFINITY.to_bits()
            );
            assert_eq!(
                load_vgpr(&expected_memory, layout, 0, thread, 12).unwrap(),
                (-149.0f32).to_bits()
            );
        }
        let hlsl = stage_b_validation_hlsl(layout, &ops, exec).unwrap();
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
            1,
            record.record_dwords(),
            &[
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        for thread in 0..128u32 {
            for register in 0..13u32 {
                let expected = load_vgpr(&expected_memory, layout, 0, thread, register).unwrap();
                let actual = load_vgpr(&memory, layout, 0, thread, register).unwrap();
                if f32::from_bits(expected).is_nan() {
                    assert!(f32::from_bits(actual).is_nan());
                } else {
                    assert_eq!(actual, expected, "thread {thread} register {register}");
                }
            }
        }
        for (name, enc) in [("V_MIN_F32", "VOP2_VOP_DPP16")] {
            let fields = vec![
                ("VDST", 0),
                ("SRC0", 250),
                ("VSRC0", 0),
                ("VSRC1", 1),
                ("DPP_CTRL", 0x140),
                ("ROW_MASK", 15),
                ("BANK_MASK", 15),
            ];
            let bytes = encode(name, enc, &fields, None).unwrap();
            assert!(
                lower_stage_b_one(&rdna2::decode(&bytes, 0).unwrap()).is_err(),
                "{name} {enc}"
            );
        }
        for name in ["V_EXP_F32", "V_FRACT_F32", "V_LOG_F32"] {
            let bytes = encode(
                name,
                "ENC_VOP3",
                &[("VDST", 0), ("SRC0", 256), ("OP_SEL", 1)],
                None,
            )
            .unwrap();
            let decoded = rdna2::decode(&bytes, 0).unwrap();
            let error = lower_stage_b_one(&decoded).unwrap_err();
            assert!(
                error.contains("unsupported selector modifiers"),
                "{name}: {error}"
            );
        }
    }
}
