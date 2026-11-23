//! Checked external VGPR storage for synthetic paired-wave dispatcher validation.
//!
//! The extended layout embeds, but does not reinterpret, the original dispatcher
//! layout. No decoded program or real PSO selection path consumes this module.

use super::mesh_dispatch::{
    ScratchLayout, GROUP_THREADS, LOGICAL_WAVE_LANES, MAX_SCRATCH_BYTES, MIN_RECORD_BYTES,
    RECORD_ALIGNMENT,
};

pub const MAX_VGPRS_PER_THREAD: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgprScratchLayout {
    pub scratch: ScratchLayout,
    pub vgpr_count: u32,
}

impl VgprScratchLayout {
    pub fn new(vgpr_count: u32, record_count: u32) -> Result<Self, &'static str> {
        if record_count == 0 {
            return Err("VGPR scratch requires at least one record");
        }
        if vgpr_count > MAX_VGPRS_PER_THREAD {
            return Err("VGPR count exceeds the conservative limit");
        }
        let vgpr_words = GROUP_THREADS
            .checked_mul(vgpr_count)
            .ok_or("VGPR scratch word count overflow")?;
        let record_bytes = vgpr_words
            .checked_mul(4)
            .and_then(|bytes| MIN_RECORD_BYTES.checked_add(bytes))
            .ok_or("VGPR scratch record size overflow")?;
        let stride_units = record_bytes
            .checked_add(RECORD_ALIGNMENT - 1)
            .ok_or("VGPR scratch stride overflow")?
            / RECORD_ALIGNMENT;
        let stride = stride_units
            .checked_mul(RECORD_ALIGNMENT)
            .ok_or("VGPR scratch stride overflow")?;
        let scratch = ScratchLayout::new(0, stride, record_count)?;
        if scratch.total_bytes > MAX_SCRATCH_BYTES {
            return Err("VGPR scratch exceeds the resource limit");
        }
        Ok(Self {
            scratch,
            vgpr_count,
        })
    }

    pub fn validate_for_groups(&self, groups: u32) -> Result<(), &'static str> {
        if groups != self.scratch.record_count {
            return Err("VGPR scratch does not match the dispatch");
        }
        self.validate_exact()?;
        self.scratch.validate_for_groups(groups)
    }

    fn validate_exact(&self) -> Result<(), &'static str> {
        if Self::new(self.vgpr_count, self.scratch.record_count)? != *self {
            return Err("VGPR scratch layout is inconsistent");
        }
        Ok(())
    }

    pub fn vgpr_offset(
        &self,
        record: u32,
        thread: u32,
        register: u32,
    ) -> Result<u32, &'static str> {
        self.validate_exact()?;
        if thread >= GROUP_THREADS {
            return Err("VGPR scratch thread is out of bounds");
        }
        if register >= self.vgpr_count {
            return Err("VGPR scratch register is out of bounds");
        }
        let word = thread
            .checked_mul(self.vgpr_count)
            .and_then(|base| base.checked_add(register))
            .ok_or("VGPR scratch address overflow")?;
        self.scratch
            .record_base(record)?
            .checked_add(MIN_RECORD_BYTES)
            .and_then(|base| {
                word.checked_mul(4)
                    .and_then(|bytes| base.checked_add(bytes))
            })
            .ok_or("VGPR scratch address overflow")
    }
}

pub fn load_vgpr(
    memory: &[u32],
    layout: VgprScratchLayout,
    record: u32,
    thread: u32,
    register: u32,
) -> Result<u32, &'static str> {
    let word = layout.vgpr_offset(record, thread, register)? as usize / 4;
    memory
        .get(word)
        .copied()
        .ok_or("VGPR scratch memory is truncated")
}

pub fn store_vgpr(
    memory: &mut [u32],
    layout: VgprScratchLayout,
    record: u32,
    thread: u32,
    register: u32,
    value: u32,
) -> Result<(), &'static str> {
    let word = layout.vgpr_offset(record, thread, register)? as usize / 4;
    *memory
        .get_mut(word)
        .ok_or("VGPR scratch memory is truncated")? = value;
    Ok(())
}

pub fn masked_vector_move(
    destination: &mut [u32; LOGICAL_WAVE_LANES as usize],
    source: &[u32; LOGICAL_WAVE_LANES as usize],
    exec: u64,
) {
    for lane in 0..LOGICAL_WAVE_LANES as usize {
        if exec & (1u64 << lane) != 0 {
            destination[lane] = source[lane];
        }
    }
}

pub fn masked_vector_add(
    destination: &mut [u32; LOGICAL_WAVE_LANES as usize],
    left: &[u32; LOGICAL_WAVE_LANES as usize],
    right: &[u32; LOGICAL_WAVE_LANES as usize],
    exec: u64,
) {
    for lane in 0..LOGICAL_WAVE_LANES as usize {
        if exec & (1u64 << lane) != 0 {
            destination[lane] = left[lane].wrapping_add(right[lane]);
        }
    }
}

pub fn vgpr_validation_hlsl(
    layout: VgprScratchLayout,
    exec_masks: [u64; 2],
) -> Result<String, &'static str> {
    layout.validate_for_groups(1)?;
    if layout.vgpr_count < 4 {
        return Err("VGPR validation requires four registers");
    }
    let mask_half = |high: bool| {
        let shift = if high { 32 } else { 0 };
        format!(
            "wave == 0u ? {:#x}u : {:#x}u",
            (exec_masks[0] >> shift) as u32,
            (exec_masks[1] >> shift) as u32
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
    uint vgpr_base = record_base + {vgpr_offset}u;
    uint lane_base = vgpr_base + gi * {vgpr_stride}u;
    uint mailbox_base = record_base + {mailbox_offset}u;
    uint result_base = record_base + {result_offset}u;
    uint2 logical_exec = uint2({exec_low}, {exec_high});
    bool active = lane < 32u ? ((logical_exec.x & (1u << lane)) != 0u) : ((logical_exec.y & (1u << (lane - 32u))) != 0u);
    uint initial0 = gi + 1u;
    uint initial1 = 1000u + gi;
    scratch.Store(lane_base, initial0);
    scratch.Store(lane_base + 4u, initial1);
    scratch.Store(lane_base + 8u, 0x5a5a5a5au);
    scratch.Store(lane_base + 12u, 0xa5a5a5a5u);
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    if (active) {{
        uint updated0 = scratch.Load(lane_base) + 5u;
        uint updated2 = updated0 + scratch.Load(lane_base + 4u);
        scratch.Store(lane_base, updated0);
        scratch.Store(lane_base + 8u, updated2);
    }}
    DeviceMemoryBarrierWithGroupSync();
    if (active) scratch.Store(lane_base + 12u, scratch.Load(lane_base + 8u) + lane);
    DeviceMemoryBarrierWithGroupSync();
    uint prefix = 0u;
    uint total = 0u;
    [unroll]
    for (uint other = 0u; other < 64u; ++other) {{
        uint bit = other < 32u ? ((logical_exec.x >> other) & 1u) : ((logical_exec.y >> (other - 32u)) & 1u);
        uint other_thread = wave * 64u + other;
        uint value = scratch.Load(vgpr_base + other_thread * {vgpr_stride}u + 12u);
        if (bit != 0u) {{
            total += value;
            if (other < lane) prefix += value;
        }}
    }}
    DeviceMemoryBarrierWithGroupSync();
    scratch.Store(result_base + gi * 4u, prefix);
    scratch.Store(mailbox_base + gi * 4u, total);
    DeviceMemoryBarrierWithGroupSync();
}}
"#,
        vgpr_offset = MIN_RECORD_BYTES,
        vgpr_stride = layout.vgpr_count * 4,
        mailbox_offset = layout.scratch.mailbox_offset(0, 0)?,
        result_offset = layout.scratch.result_offset(0, 0)?,
        exec_low = mask_half(false),
        exec_high = mask_half(true),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::mesh_reduce::{active_sum, exclusive_prefix_sum};
    use super::*;

    #[test]
    fn layout_checks_counts_limits_tampering_and_offsets() {
        let zero = VgprScratchLayout::new(0, 1).unwrap();
        assert!(zero.vgpr_offset(0, 0, 0).is_err());
        let max = VgprScratchLayout::new(MAX_VGPRS_PER_THREAD, 1).unwrap();
        assert_eq!(max.scratch.record_stride_bytes % RECORD_ALIGNMENT, 0);
        assert!(VgprScratchLayout::new(MAX_VGPRS_PER_THREAD + 1, 1).is_err());
        assert!(VgprScratchLayout::new(4, 0).is_err());
        assert!(VgprScratchLayout::new(MAX_VGPRS_PER_THREAD, u32::MAX).is_err());
        assert!(max.vgpr_offset(1, 0, 0).is_err());
        assert!(max.vgpr_offset(0, GROUP_THREADS, 0).is_err());
        assert!(max.vgpr_offset(0, 0, MAX_VGPRS_PER_THREAD).is_err());
        let mut tampered = max;
        tampered.scratch.record_stride_bytes += RECORD_ALIGNMENT;
        assert!(tampered.validate_for_groups(1).is_err());
        assert!(tampered.vgpr_offset(0, 0, 0).is_err());

        let mut bad_total = max;
        bad_total.scratch.total_bytes -= RECORD_ALIGNMENT;
        let memory = vec![0u32; max.scratch.total_bytes as usize / 4];
        assert!(load_vgpr(&memory, bad_total, 0, 0, 0).is_err());

        let mut bad_count = max;
        bad_count.vgpr_count -= 1;
        let mut memory = memory;
        assert!(store_vgpr(&mut memory, bad_count, 0, 0, 0, 1).is_err());
    }

    #[test]
    fn cpu_masked_operations_and_storage_preserve_inactive_lanes() {
        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let mut memory = vec![0u32; layout.scratch.total_bytes as usize / 4];
        store_vgpr(&mut memory, layout, 0, 127, 3, 99).unwrap();
        assert_eq!(load_vgpr(&memory, layout, 0, 127, 3).unwrap(), 99);
        assert!(load_vgpr(&memory[..4], layout, 0, 127, 3).is_err());
        let source = std::array::from_fn(|lane| lane as u32 + 10);
        let mut destination = [77u32; 64];
        let exec = (1u64 << 31) | (1u64 << 32) | (1u64 << 63);
        masked_vector_move(&mut destination, &source, exec);
        assert_eq!((destination[30], destination[31]), (77, 41));
        assert_eq!((destination[32], destination[63]), (42, 73));
        let left = [u32::MAX; 64];
        let right = [2u32; 64];
        masked_vector_add(&mut destination, &left, &right, exec);
        assert_eq!(destination[31], 1);
        assert_eq!(destination[30], 77);
    }

    fn run_gpu(exec_masks: [u64; 2]) -> Option<(VgprScratchLayout, Vec<u32>)> {
        if super::super::dxc::location().is_err() {
            return None;
        }
        let layout = VgprScratchLayout::new(4, 1).unwrap();
        let hlsl = vgpr_validation_hlsl(layout, exec_masks).unwrap();
        assert_eq!(
            hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
            5
        );
        assert!(!hlsl.contains("groupshared") && !hlsl.contains("xcl_lds"));
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
                layout.scratch.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout.scratch,
        )
        .unwrap();
        Some((layout, memory))
    }

    #[test]
    fn gpu_vgprs_persist_and_reduce_for_independent_logical_waves() {
        let exec_masks = [
            (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
            (1u64 << 5) | (1u64 << 40),
        ];
        let Some((layout, memory)) = run_gpu(exec_masks) else {
            return;
        };
        for wave in 0..2usize {
            let mut reduced = [0u32; 64];
            for lane in 0..64usize {
                let thread = (wave * 64 + lane) as u32;
                let active = exec_masks[wave] & (1u64 << lane) != 0;
                let initial0 = thread + 1;
                let expected0 = if active { initial0 + 5 } else { initial0 };
                let expected2 = if active {
                    expected0.wrapping_add(1000 + thread)
                } else {
                    0x5a5a5a5a
                };
                let expected3 = if active {
                    expected2.wrapping_add(lane as u32)
                } else {
                    0xa5a5a5a5
                };
                assert_eq!(load_vgpr(&memory, layout, 0, thread, 0).unwrap(), expected0);
                assert_eq!(
                    load_vgpr(&memory, layout, 0, thread, 1).unwrap(),
                    1000 + thread
                );
                assert_eq!(load_vgpr(&memory, layout, 0, thread, 2).unwrap(), expected2);
                assert_eq!(load_vgpr(&memory, layout, 0, thread, 3).unwrap(), expected3);
                reduced[lane] = expected3;
            }
            let total = active_sum(&reduced, exec_masks[wave]);
            for lane in 0..64usize {
                let thread = (wave * 64 + lane) as u32;
                let result = layout.scratch.result_offset(0, thread).unwrap() as usize / 4;
                let mailbox = layout.scratch.mailbox_offset(0, thread).unwrap() as usize / 4;
                assert_eq!(
                    memory[result],
                    exclusive_prefix_sum(&reduced, exec_masks[wave], lane as u32)
                );
                assert_eq!(memory[mailbox], total);
            }
        }
    }
}
