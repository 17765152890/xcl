//! Paired-half lane primitives for a synthetic logical wave64 dispatcher.
//!
//! This module is not connected to real-program selection.

use super::mesh_dispatch::{
    LogicalWaveStatus, ScratchLayout, GROUP_THREADS, HALF_LANES, LOGICAL_WAVE_LANES,
    SCALAR_REGISTERS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanePrimitiveFault {
    EmptyExec,
}

pub fn read_first_lane(
    values: &[u32; GROUP_THREADS as usize],
    logical_wave: u32,
    exec: u64,
) -> Result<u32, LanePrimitiveFault> {
    if exec == 0 {
        return Err(LanePrimitiveFault::EmptyExec);
    }
    let lane = exec.trailing_zeros() as usize;
    let thread = logical_wave as usize * LOGICAL_WAVE_LANES as usize + lane;
    values
        .get(thread)
        .copied()
        .ok_or(LanePrimitiveFault::EmptyExec)
}

pub fn mbcnt_lo(source_low: u32, lane: u32, accumulator: u32) -> u32 {
    let before = lane.min(HALF_LANES);
    let lane_mask = if before == 32 {
        u32::MAX
    } else {
        (1u32 << before) - 1
    };
    accumulator.wrapping_add((source_low & lane_mask).count_ones())
}

pub fn mbcnt_hi(source_high: u32, lane: u32, accumulator: u32) -> u32 {
    if lane < HALF_LANES {
        return accumulator;
    }
    let before = lane - HALF_LANES;
    let lane_mask = if before == 32 {
        u32::MAX
    } else {
        (1u32 << before) - 1
    };
    accumulator.wrapping_add((source_high & lane_mask).count_ones())
}

pub fn mbcnt_prefix(source: u64, lane: u32, accumulator: u32) -> u32 {
    mbcnt_hi(
        (source >> 32) as u32,
        lane,
        mbcnt_lo(source as u32, lane, accumulator),
    )
}

pub fn lane_validation_hlsl(
    layout: ScratchLayout,
    exec_masks: [u64; 2],
    mbcnt_sources: [u64; 2],
    accumulators: [u32; 2],
) -> Result<String, &'static str> {
    layout.validate_for_groups(1)?;
    let pair = |values: [u64; 2], high: bool| {
        let shift = if high { 32 } else { 0 };
        format!(
            "wave == 0u ? {:#x}u : {:#x}u",
            (values[0] >> shift) as u32,
            (values[1] >> shift) as u32
        )
    };
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads({GROUP_THREADS}, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint wave = gi / {LOGICAL_WAVE_LANES}u;
    uint lane = gi % {LOGICAL_WAVE_LANES}u;
    uint record_base = gid.x * io[0];
    uint state_base = record_base + wave * {state_bytes}u;
    uint scalar_base = state_base + {header_bytes}u;
    uint mailbox_base = record_base + {mailbox_bytes}u;
    uint result_base = record_base + {result_bytes}u;
    uint2 logical_exec = uint2({exec_low}, {exec_high});
    uint2 mbcnt_source = uint2({source_low}, {source_high});
    uint value = gi * 17u + 3u;
    scratch.Store(mailbox_base + gi * 4u, value);
    scratch.Store(scalar_base + lane * 4u, 0u);
    scratch.Store(scalar_base + (lane + 64u) * 4u, 0u);
    if (lane == 0u) {{
        scratch.Store4(state_base, uint4(0u, logical_exec.x, logical_exec.y, 0u));
        scratch.Store4(state_base + 16u, uint4(0u, 0u, 0u, 0u));
        scratch.Store(state_base + 32u, 0u);
    }}
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    uint count = {accumulator};
    if (lane < 32u) {{
        uint before = lane == 0u ? 0u : ((1u << lane) - 1u);
        count += countbits(mbcnt_source.x & before);
    }} else {{
        count += countbits(mbcnt_source.x);
        uint high_lane = lane - 32u;
        uint before = high_lane == 0u ? 0u : ((1u << high_lane) - 1u);
        count += countbits(mbcnt_source.y & before);
    }}
    scratch.Store(result_base + gi * 4u, count);
    if (lane == 0u) {{
        uint status = {complete};
        uint first_value = 0u;
        if (any(logical_exec)) {{
            uint first = logical_exec.x != 0u ? firstbitlow(logical_exec.x) : 32u + firstbitlow(logical_exec.y);
            first_value = scratch.Load(mailbox_base + (wave * 64u + first) * 4u);
        }}
        scratch.Store(state_base + 24u, 1u);
        scratch.Store(state_base + 28u, status);
        scratch.Store(state_base + 32u, 1u);
        scratch.Store(scalar_base, first_value);
    }}
    DeviceMemoryBarrierWithGroupSync();
}}
"#,
        state_bytes = layout.state_offset(0, 1)? - layout.state_offset(0, 0)?,
        header_bytes =
            layout.state_offset(0, 1)? - layout.state_offset(0, 0)? - SCALAR_REGISTERS * 4,
        mailbox_bytes = layout.mailbox_offset(0, 0)?,
        result_bytes = layout.result_offset(0, 0)?,
        exec_low = pair(exec_masks, false),
        exec_high = pair(exec_masks, true),
        source_low = pair(mbcnt_sources, false),
        source_high = pair(mbcnt_sources, true),
        accumulator = format!("wave == 0u ? {}u : {}u", accumulators[0], accumulators[1]),
        complete = format!(
            "wave == 0u ? {}u : {}u",
            if exec_masks[0] == 0 {
                LogicalWaveStatus::Fault as u32
            } else {
                LogicalWaveStatus::Complete as u32
            },
            if exec_masks[1] == 0 {
                LogicalWaveStatus::Fault as u32
            } else {
                LogicalWaveStatus::Complete as u32
            },
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::mesh_dispatch::{read_logical_state, scratch_for_groups};
    use super::*;

    fn run_gpu(
        exec_masks: [u64; 2],
        mbcnt_sources: [u64; 2],
        accumulators: [u32; 2],
    ) -> Option<(ScratchLayout, Vec<u32>)> {
        if super::super::dxc::location().is_err() {
            return None;
        }
        let layout = scratch_for_groups(1).unwrap();
        let hlsl = lane_validation_hlsl(layout, exec_masks, mbcnt_sources, accumulators).unwrap();
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
            "float4 main() : SV_Target0 { return 0.0.xxxx; }",
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
                layout.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout,
        )
        .unwrap();
        Some((layout, memory))
    }

    #[test]
    fn readfirstlane_uses_the_lowest_active_logical_lane() {
        let values = std::array::from_fn(|thread| thread as u32 * 17 + 3);
        assert_eq!(read_first_lane(&values, 0, 1).unwrap(), values[0]);
        assert_eq!(read_first_lane(&values, 0, 1u64 << 31).unwrap(), values[31]);
        assert_eq!(read_first_lane(&values, 0, 1u64 << 32).unwrap(), values[32]);
        assert_eq!(
            read_first_lane(&values, 1, 1u64 << 63).unwrap(),
            values[127]
        );
        assert_eq!(
            read_first_lane(&values, 0, 0),
            Err(LanePrimitiveFault::EmptyExec)
        );
    }

    #[test]
    fn mbcnt_carries_low_activity_into_the_upper_half() {
        assert_eq!(mbcnt_prefix(u64::MAX, 0, 0), 0);
        assert_eq!(mbcnt_prefix(u64::MAX, 31, 0), 31);
        assert_eq!(mbcnt_prefix(u64::MAX, 32, 0), 32);
        assert_eq!(mbcnt_prefix(u64::MAX, 63, 0), 63);
        let sparse = (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 62);
        assert_eq!(mbcnt_prefix(sparse, 32, 7), 9);
        assert_eq!(mbcnt_prefix(sparse, 63, 7), 11);
        assert_eq!(mbcnt_prefix(0, 63, 19), 19);
    }

    #[test]
    fn gpu_lane_primitives_match_native_wave64_reference() {
        let values = std::array::from_fn(|thread| thread as u32 * 17 + 3);
        for (exec_masks, mbcnt_sources, accumulators) in [
            (
                [1u64 << 40, 0],
                [
                    (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 62),
                    u64::MAX,
                ],
                [7, 19],
            ),
            (
                [1, (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63)],
                [0, u64::MAX],
                [0, 5],
            ),
        ] {
            let hlsl = lane_validation_hlsl(
                scratch_for_groups(1).unwrap(),
                exec_masks,
                mbcnt_sources,
                accumulators,
            )
            .unwrap();
            assert_eq!(
                hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
                2
            );
            assert!(!hlsl.contains("groupshared") && !hlsl.contains("xcl_lds"));
            let Some((layout, memory)) = run_gpu(exec_masks, mbcnt_sources, accumulators) else {
                return;
            };
            for wave in 0..2u32 {
                let state = read_logical_state(&memory, layout, 0, wave).unwrap();
                match read_first_lane(&values, wave, exec_masks[wave as usize]) {
                    Ok(value) => {
                        assert_eq!(state.status, LogicalWaveStatus::Complete);
                        assert_eq!(state.scalar[0], value);
                    }
                    Err(LanePrimitiveFault::EmptyExec) => {
                        assert_eq!(state.status, LogicalWaveStatus::Fault);
                        assert_eq!(state.scalar[0], 0);
                    }
                }
                for lane in 0..64u32 {
                    let thread = wave * 64 + lane;
                    let word = layout.result_offset(0, thread).unwrap() as usize / 4;
                    assert_eq!(
                        memory[word],
                        mbcnt_prefix(
                            mbcnt_sources[wave as usize],
                            lane,
                            accumulators[wave as usize]
                        ),
                        "logical wave {wave}, lane {lane}"
                    );
                }
            }
        }
    }
}
