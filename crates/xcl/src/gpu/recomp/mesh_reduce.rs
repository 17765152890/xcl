//! External-scratch reductions for paired wave32 halves forming logical wave64.
//!
//! This module only supports synthetic dispatcher validation. It does not enable
//! reductions, allocation messages, or exports for decoded mesh programs.

use super::mesh_dispatch::{ScratchLayout, GROUP_THREADS, LOGICAL_WAVE_LANES};

pub const GS_ALLOC_VERTEX_MAX: u32 = 0x3ff;
pub const GS_ALLOC_PRIMITIVE_MAX: u32 = 0x7ff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GsAllocRequest {
    pub vertices: u32,
    pub primitives: u32,
}

impl GsAllocRequest {
    pub fn clamped(vertices: u32, primitives: u32) -> Self {
        Self {
            vertices: vertices.min(GS_ALLOC_VERTEX_MAX),
            primitives: primitives.min(GS_ALLOC_PRIMITIVE_MAX),
        }
    }

    pub fn pack(self) -> u32 {
        (self.vertices.min(GS_ALLOC_VERTEX_MAX))
            | (self.primitives.min(GS_ALLOC_PRIMITIVE_MAX) << 12)
    }

    pub fn unpack(m0: u32) -> Self {
        Self {
            vertices: m0 & GS_ALLOC_VERTEX_MAX,
            primitives: (m0 >> 12) & GS_ALLOC_PRIMITIVE_MAX,
        }
    }
}

pub fn exclusive_prefix_sum(
    contributions: &[u32; LOGICAL_WAVE_LANES as usize],
    active_mask: u64,
    lane: u32,
) -> u32 {
    contributions
        .iter()
        .take(lane.min(LOGICAL_WAVE_LANES) as usize)
        .enumerate()
        .filter(|(index, _)| active_mask & (1u64 << index) != 0)
        .fold(0u32, |sum, (_, value)| sum.wrapping_add(*value))
}

pub fn active_sum(contributions: &[u32; LOGICAL_WAVE_LANES as usize], active_mask: u64) -> u32 {
    exclusive_prefix_sum(contributions, active_mask, LOGICAL_WAVE_LANES)
}

fn lane_expression(values: &[u32; LOGICAL_WAVE_LANES as usize]) -> String {
    let mut expression = String::new();
    for (lane, value) in values.iter().enumerate().take(values.len() - 1) {
        expression.push_str(&format!("lane == {lane}u ? {value}u : "));
    }
    expression.push_str(&format!("{}u", values[values.len() - 1]));
    expression
}

pub fn reduction_validation_hlsl(
    layout: ScratchLayout,
    active_masks: [u64; 2],
    contributions: [[u32; LOGICAL_WAVE_LANES as usize]; 2],
) -> Result<String, &'static str> {
    layout.validate_for_groups(1)?;
    let mask_half = |high: bool| {
        let shift = if high { 32 } else { 0 };
        format!(
            "wave == 0u ? {:#x}u : {:#x}u",
            (active_masks[0] >> shift) as u32,
            (active_masks[1] >> shift) as u32
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
    uint mailbox_base = record_base + {mailbox_offset}u;
    uint result_base = record_base + {result_offset}u;
    uint2 active_mask = uint2({mask_low}, {mask_high});
    uint contribution = wave == 0u ? ({wave0_values}) : ({wave1_values});
    scratch.Store(mailbox_base + gi * 4u, contribution);
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    uint prefix = 0u;
    uint total = 0u;
    [unroll]
    for (uint other = 0u; other < 64u; ++other) {{
        uint bit = other < 32u ? ((active_mask.x >> other) & 1u) : ((active_mask.y >> (other - 32u)) & 1u);
        uint value = scratch.Load(mailbox_base + (wave * 64u + other) * 4u);
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
        mailbox_offset = layout.mailbox_offset(0, 0)?,
        result_offset = layout.result_offset(0, 0)?,
        mask_low = mask_half(false),
        mask_high = mask_half(true),
        wave0_values = lane_expression(&contributions[0]),
        wave1_values = lane_expression(&contributions[1]),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::mesh_dispatch::scratch_for_groups;
    use super::*;

    fn run_gpu(
        active_masks: [u64; 2],
        contributions: [[u32; LOGICAL_WAVE_LANES as usize]; 2],
    ) -> Option<(ScratchLayout, Vec<u32>)> {
        if super::super::dxc::location().is_err() {
            return None;
        }
        let layout = scratch_for_groups(1).unwrap();
        let hlsl = reduction_validation_hlsl(layout, active_masks, contributions).unwrap();
        assert_eq!(
            hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
            3
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
    fn cpu_reductions_cover_boundaries_masks_and_wrapping() {
        let ones = [1u32; 64];
        assert_eq!(exclusive_prefix_sum(&ones, u64::MAX, 31), 31);
        assert_eq!(exclusive_prefix_sum(&ones, u64::MAX, 32), 32);
        assert_eq!(exclusive_prefix_sum(&ones, u64::MAX, 63), 63);
        assert_eq!(active_sum(&ones, 0), 0);
        let sparse_mask = (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63);
        assert_eq!(exclusive_prefix_sum(&ones, sparse_mask, 32), 2);
        assert_eq!(exclusive_prefix_sum(&ones, sparse_mask, 63), 3);
        let mut overflow = [0u32; 64];
        overflow[0] = u32::MAX;
        overflow[32] = 2;
        assert_eq!(active_sum(&overflow, sparse_mask), 1);
    }

    #[test]
    fn gs_alloc_request_has_exact_clamped_bitfields() {
        for request in [
            GsAllocRequest::clamped(0, 0),
            GsAllocRequest::clamped(17, 29),
            GsAllocRequest::clamped(GS_ALLOC_VERTEX_MAX, GS_ALLOC_PRIMITIVE_MAX),
            GsAllocRequest::clamped(u32::MAX, u32::MAX),
        ] {
            assert_eq!(GsAllocRequest::unpack(request.pack()), request);
            assert_eq!(request.pack() & 0xc00, 0);
        }
        assert_eq!(
            GsAllocRequest::clamped(u32::MAX, u32::MAX),
            GsAllocRequest {
                vertices: GS_ALLOC_VERTEX_MAX,
                primitives: GS_ALLOC_PRIMITIVE_MAX,
            }
        );
        assert_eq!(
            GsAllocRequest::unpack(u32::MAX),
            GsAllocRequest {
                vertices: GS_ALLOC_VERTEX_MAX,
                primitives: GS_ALLOC_PRIMITIVE_MAX,
            }
        );
    }

    #[test]
    fn gpu_reductions_match_logical_wave64_reference() {
        let mut full_and_empty = [[0u32; 64]; 2];
        full_and_empty[0] = std::array::from_fn(|lane| (lane as u32).wrapping_add(1));
        full_and_empty[0][17] = 0;
        full_and_empty[1] = [u32::MAX; 64];

        let mut sparse_and_overflow = [[0u32; 64]; 2];
        sparse_and_overflow[0][0] = u32::MAX;
        sparse_and_overflow[0][31] = 0;
        sparse_and_overflow[0][32] = 2;
        sparse_and_overflow[0][63] = 5;
        sparse_and_overflow[1] = std::array::from_fn(|lane| (lane as u32) * 3);

        for (active_masks, contributions) in [
            ([u64::MAX, 0], full_and_empty),
            (
                [
                    (1u64 << 0) | (1u64 << 31) | (1u64 << 32) | (1u64 << 63),
                    (1u64 << 5) | (1u64 << 40) | (1u64 << 63),
                ],
                sparse_and_overflow,
            ),
        ] {
            let Some((layout, memory)) = run_gpu(active_masks, contributions) else {
                return;
            };
            for wave in 0..2usize {
                let total = active_sum(&contributions[wave], active_masks[wave]);
                for lane in 0..64usize {
                    let thread = (wave * 64 + lane) as u32;
                    let prefix_word = layout.result_offset(0, thread).unwrap() as usize / 4;
                    let total_word = layout.mailbox_offset(0, thread).unwrap() as usize / 4;
                    assert_eq!(
                        memory[prefix_word],
                        exclusive_prefix_sum(&contributions[wave], active_masks[wave], lane as u32),
                        "logical wave {wave}, lane {lane} prefix"
                    );
                    assert_eq!(
                        memory[total_word], total,
                        "logical wave {wave}, lane {lane} total"
                    );
                }
            }
        }
    }
}
