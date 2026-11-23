//! Validation-only HLSL for a static, two-wave uniform slot schedule.
//!
//! This module has no production translator or PSO callers.

use super::mesh_alloc_export::AllocationExportOp;
use super::mesh_memory::ReadOnlyRegion;
use super::mesh_program::{requirements, MeshProgram, ProgramOp};
use super::mesh_slot::{lower_slot, SlotSync, WaveStateAbi};
use super::mesh_vgpr::VgprScratchLayout;

const LDS_SLICE_BYTES: u32 = 32 * 1024;
const MAX_GROUP_SHARED_BYTES: u32 = 32 * 1024;

fn validation_shared_bytes(sgprs: u32) -> Option<u32> {
    2u32
        .checked_mul(sgprs)
        .and_then(|n| n.checked_add(2 * (2 + 2 + 1 + 1 + 1)))
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_add(6))
        // Two wait flags, two generations, and one group-owned action.
        .and_then(|n| n.checked_add(5))
        .and_then(|n| n.checked_mul(4))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValidationEpochShell {
    pub hlsl: String,
    pub input_dwords: u32,
    pub result_dwords: u32,
    pub lds_bytes: u32,
    pub vgpr_scratch: VgprScratchLayout,
}

pub(crate) fn validation_wave_abi() -> WaveStateAbi<'static> {
    WaveStateAbi {
        sgpr: "sgpr[wave]",
        vgpr: "vgpr_scratch",
        vgpr_buffer: true,
        exec: "exec_mask[wave]",
        vcc: "vcc[wave]",
        scc: "scc[wave]",
        m0: "m0[wave]",
        status: "xcl_status[wave]",
        memory: "memory",
        exports: "res",
        lds: "lds",
        lane: "lane",
    }
}

pub(crate) fn validation_epoch_hlsl(
    read_only: Option<ReadOnlyRegion>,
    program: &MeshProgram,
    workgroups: u32,
    step_cap: u32,
) -> Result<ValidationEpochShell, String> {
    if workgroups == 0 {
        return Err("validation epoch shell requires at least one workgroup".into());
    }
    let lds_bytes = workgroups
        .checked_mul(LDS_SLICE_BYTES)
        .ok_or("validation LDS resource size overflows")?;
    super::mesh_dispatch::validate_step_cap(step_cap).map_err(str::to_owned)?;
    validate_boundaries(program)?;
    let requirements = requirements(program)?;
    if requirements.uses_read_only_memory && read_only.is_none() {
        return Err("validation read-only memory program requires an explicit region".into());
    }
    if let Some(region) = read_only {
        if region.guest_base & 3 != 0
            || region.memory_offset & 3 != 0
            || region.byte_len == 0
            || region.byte_len & 3 != 0
        {
            return Err("validation read-only region is not dword aligned".into());
        }
        region
            .guest_base
            .checked_add(u64::from(region.byte_len))
            .ok_or("validation guest memory region overflows")?;
        region
            .memory_offset
            .checked_add(region.byte_len)
            .ok_or("validation host memory region overflows")?;
    }
    let sgprs = requirements.sgprs.max(1);
    let vgprs = requirements.vgprs.max(1);
    if sgprs > 128 || vgprs > 256 {
        return Err("validation epoch register requirements exceed the ABI".into());
    }
    let vgpr_scratch = VgprScratchLayout::new(vgprs, workgroups).map_err(str::to_owned)?;
    vgpr_scratch.validate_for_groups(workgroups).map_err(str::to_owned)?;
    let shared_bytes = validation_shared_bytes(sgprs)
        .ok_or("validation epoch shared-state size overflows")?;
    if shared_bytes > MAX_GROUP_SHARED_BYTES {
        return Err(format!(
            "validation epoch shared state requires {shared_bytes} bytes"
        ));
    }
    let wave_input_dwords = sgprs
        .checked_add(6)
        .ok_or("validation epoch input layout overflows")?;
    let vgpr_input_base = 2u32
        .checked_mul(wave_input_dwords)
        .ok_or("validation epoch input layout overflows")?;
    let input_group_dwords = vgpr_input_base
        .checked_add(
            128u32
                .checked_mul(vgprs)
                .ok_or("validation epoch input layout overflows")?,
        )
        .ok_or("validation epoch input layout overflows")?;
    let result_group_dwords = 128u32
        .checked_add(127 * 256)
        .and_then(|n| n.checked_add(vgprs))
        .ok_or("validation epoch result layout overflows")?;
    let input_dwords = workgroups
        .checked_mul(input_group_dwords)
        .ok_or("validation epoch input resource size overflows")?;
    let result_dwords = workgroups
        .checked_mul(result_group_dwords)
        .ok_or("validation epoch result resource size overflows")?;
    let input_bytes = input_dwords
        .checked_mul(4)
        .ok_or("validation epoch input size overflows")?;
    let result_bytes = result_dwords
        .checked_mul(4)
        .ok_or("validation epoch result size overflows")?;
    let region = read_only.unwrap_or(ReadOnlyRegion {
        guest_base: 0,
        memory_offset: 0,
        byte_len: 0,
    });

    let abi = validation_wave_abi();
    let mut phases = String::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        if matches!(instruction.op, ProgramOp::Scalar(super::mesh_scalar::StageAOp::Barrier)) {
            let next = if index + 1 == program.instructions.len() {
                "xcl_status[wave]=1u;".to_owned()
            } else {
                format!("wave_pc[wave]={}u;", index + 1)
            };
            phases.push_str(&format!(r#"  // uniform scalar rendezvous slot {index}
  if(group_valid&&lane==0u&&xcl_status[wave]==0u&&wave_pc[wave]=={index}u&&!xcl_barrier_waiting[wave]){{
    if(xcl_steps[wave]>={step_cap}u)xcl_status[wave]=2u;
    else{{xcl_steps[wave]++;xcl_barrier_waiting[wave]=1u;}}
  }}
  AllMemoryBarrierWithGroupSync();
  if(group_valid&&gi==0u){{
    xcl_barrier_action=0u;
    if(xcl_barrier_waiting[0]!=0u&&xcl_barrier_waiting[1]!=0u)xcl_barrier_action=xcl_barrier_generation[0]==xcl_barrier_generation[1]?1u:2u;
    else if(xcl_barrier_waiting[0]!=0u&&xcl_status[1]!=0u)xcl_barrier_action=3u;
    else if(xcl_barrier_waiting[1]!=0u&&xcl_status[0]!=0u)xcl_barrier_action=4u;
  }}
  GroupMemoryBarrierWithGroupSync();
  if(group_valid&&lane==0u){{
    if(xcl_barrier_action==1u&&xcl_barrier_waiting[wave]!=0u){{xcl_barrier_waiting[wave]=0u;xcl_barrier_generation[wave]++;{next}}}
    else if(xcl_barrier_action==2u){{xcl_barrier_waiting[wave]=0u;xcl_status[wave]=3u;}}
    else if(xcl_barrier_action==3u&&wave==0u){{xcl_barrier_waiting[wave]=0u;xcl_status[wave]=3u;}}
    else if(xcl_barrier_action==4u&&wave==1u){{xcl_barrier_waiting[wave]=0u;xcl_status[wave]=3u;}}
  }}
  AllMemoryBarrierWithGroupSync();
"#));
            continue;
        }
        let control = match instruction.op {
            ProgramOp::Control(op) => Some(super::mesh_control_slot::lower_control_slot(
                op,
                program.instructions.len() as u32,
            )?),
            _ => None,
        };
        let effect = match control {
            Some(ref hlsl) => super::mesh_slot::SlotEffect {
                hlsl: hlsl.clone(),
                sync: SlotSync::None,
            },
            None => lower_slot(&instruction.op, abi)
                .map_err(|error| format!("slot {index} is not validation-lowerable: {error}"))?,
        };
        let leader_only = matches!(
            instruction.op,
            ProgramOp::Control(_)
                | ProgramOp::Scalar(_)
                | ProgramOp::AllocationExport(AllocationExportOp::Allocate { .. })
                | ProgramOp::AllocationExport(AllocationExportOp::Export(_))
        );
        phases.push_str(&format!("  // uniform static slot {index}\n  if(group_valid&&lane==0u&&xcl_status[wave]==0u&&wave_pc[wave]=={index}u&&xcl_steps[wave]>={step_cap}u)xcl_status[wave]=2u;\n  AllMemoryBarrierWithGroupSync();\n  bool xcl_execute_{index}=group_valid&&xcl_status[wave]==0u&&wave_pc[wave]=={index}u;\n  if(xcl_execute_{index}){{\n"));
        if leader_only {
            phases.push_str("    if(lane==0u){");
            if control.is_some() {
                phases.push_str(&format!("xcl_next_pc[wave]={}u;", index + 1));
            }
            phases.push_str(&effect.hlsl);
            phases.push_str("}\n");
        } else {
            phases.push_str("    ");
            phases.push_str(&effect.hlsl);
            phases.push('\n');
        }
        phases.push_str("  }\n");
        match effect.sync {
            SlotSync::DeviceMemory => phases.push_str("  AllMemoryBarrierWithGroupSync();\n"),
            SlotSync::None | SlotSync::GroupMemory => phases.push_str("  AllMemoryBarrierWithGroupSync();\n"),
        }
        let next = if control.is_some() {
            format!("if(xcl_status[wave]==0u){{if(xcl_next_pc[wave]<{}u)wave_pc[wave]=xcl_next_pc[wave];else if(xcl_next_pc[wave]=={}u)xcl_status[wave]=1u;else xcl_status[wave]=3u;}}", program.instructions.len(), program.instructions.len())
        } else if index + 1 == program.instructions.len() {
            "if(xcl_status[wave]==0u)xcl_status[wave]=1u;".into()
        } else {
            format!("if(xcl_status[wave]==0u)wave_pc[wave]={}u;", index + 1)
        };
        phases.push_str(&format!("  if(xcl_execute_{index}&&lane==0u){{xcl_steps[wave]++;{next}}}\n  AllMemoryBarrierWithGroupSync();\n"));
    }

    let slot_count = program.instructions.len();
    let hlsl = format!(
        r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
RWByteAddressBuffer lds : register(u2);
RWByteAddressBuffer memory : register(u3);
RWByteAddressBuffer vgpr_scratch : register(u6);
groupshared uint sgpr[2][{sgprs}];
groupshared uint2 exec_mask[2];
groupshared uint2 vcc[2];
groupshared uint scc[2];
groupshared uint m0[2];
groupshared uint wave_pc[2];
groupshared uint xcl_next_pc[2];
groupshared uint xcl_status[2];
groupshared uint xcl_steps[2];
groupshared uint xcl_barrier_waiting[2];
groupshared uint xcl_barrier_generation[2];
groupshared uint xcl_barrier_action;
groupshared uint xcl_group_base;
groupshared uint xcl_vgpr_group_base;

uint xcl_vgpr_address(uint wave,uint lane,uint reg) {{ return xcl_vgpr_group_base+{vgpr_data_offset}u+(wave*64u+lane)*{vgpr_stride}u+reg*4u; }}

uint xcl_lds_address(uint base, uint offset) {{ return xcl_group_base + ((((base + offset) >> 2u) & 8191u) << 2u); }}
bool xcl_lds_load(RWByteAddressBuffer b,uint base,uint offset,out uint value) {{ value=b.Load(xcl_lds_address(base,offset));return true; }}
bool xcl_lds_store(RWByteAddressBuffer b,uint base,uint offset,uint value) {{ b.Store(xcl_lds_address(base,offset),value);return true; }}
bool xcl_ds_swizzle(uint lane_value,uint lane,uint control,out uint value) {{
  uint source=(control&0x8000u)==0u?((lane&~3u)|((control>>(2u*(lane&3u)))&3u)):((lane&32u)|((((lane&31u)&((control>>10u)&31u))|((control>>5u)&31u))^(control&31u)));
  value=WaveReadLaneAt(lane_value,source);return source<64u;
}}
float xcl_mul_legacy(float a,float b) {{ return (a==0.0f||b==0.0f)?0.0f:a*b; }}
float xcl_fma_legacy(float a,float b,float c) {{ return (a==0.0f||b==0.0f)?c:mad(a,b,c); }}
float xcl_mad_nonfused(float a,float b,float c) {{ precise float p=a*b; precise float r=p+c; return r; }}
uint xcl_perm_b32(uint a,uint b,uint s) {{ uint r=0u; [unroll] for(uint i=0u;i<4u;i++) {{ uint k=(s>>(8u*i))&255u,v; if(k>=13u)v=255u; else if(k==12u)v=0u; else if(k>=8u) {{ uint q=k==8u?b>>15:(k==9u?b>>31:(k==10u?a>>15:a>>31)); v=(q&1u)!=0u?255u:0u; }} else v=((k<4u?b>>(8u*k):a>>(8u*(k-4u)))&255u); r|=v<<(8u*i); }} return r; }}
uint xcl_bfe_u32(uint value,uint offset,uint width) {{ offset&=31u;width&=31u;return width==0u?0u:(value>>offset)&(0xffffffffu>>((32u-width)&31u)); }}
uint xcl_scalar_bfe_u32(uint value,uint control) {{ uint offset=control&31u,width=(control>>16u)&127u;return width==0u?0u:(width>=32u?value>>offset:(value>>offset)&(0xffffffffu>>(32u-width))); }}
uint xcl_bfe_i32(uint value,uint control) {{ uint offset=control&31u,width=(control>>16u)&127u,shift=(32u-width)&31u;return width==0u?0u:(width>=32u?(uint)((int)value>>offset):(uint)(((int)((value>>offset)<<shift))>>shift)); }}
bool xcl_float_class_match(float value,uint classes) {{ uint bits=asuint(value),e=(bits>>23u)&255u,m=bits&0x7fffffu,index; if(e==255u)index=m!=0u?((m&0x400000u)!=0u?1u:0u):((bits>>31u)!=0u?2u:9u); else if(e==0u)index=m==0u?((bits>>31u)!=0u?5u:6u):((bits>>31u)!=0u?4u:7u); else index=(bits>>31u)!=0u?3u:8u; return ((classes>>index)&1u)!=0u; }}
uint xcl_cvt_u32(float f) {{ return isnan(f)||f<=0.0f?0u:(f>=4294967296.0f?0xffffffffu:(uint)f); }}
int xcl_cvt_i32(float f) {{ return isnan(f)?0:(f<=-2147483648.0f?(int)0x80000000u:(f>=2147483648.0f?0x7fffffff:(int)f)); }}
uint xcl_f32_to_f16_bounded(uint v) {{ if((v&0x7f800000u)==0x7f800000u&&(v&0x007fffffu)!=0u)return 0x7e00u;return f32tof16(asfloat(v)); }}
bool xcl_global_load(RWByteAddressBuffer mem,uint2 p,int off,out uint value) {{
  uint64_t a=(uint64_t)p.x|((uint64_t)p.y<<32);
  if(off<0){{uint64_t d=(uint64_t)(-(int64_t)off);if(a<d)return false;a-=d;}}
  else{{uint64_t d=(uint64_t)off;if(a>0xffffffffffffffffull-d)return false;a+=d;}}
  if((a&3ull)!=0ull||{region_len}u<4u||a<{guest_base}ull)return false;
  uint64_t relative=a-{guest_base}ull;
  if(relative>(uint64_t)({region_len}u-4u))return false;
  if(relative>0xffffffffull-(uint64_t){host_offset}u)return false;
  uint host={host_offset}u+(uint)relative,bytes;mem.GetDimensions(bytes);
  if(bytes<4u||host>bytes-4u)return false;
  value=mem.Load(host);return true;
}}
bool xcl_global_byte(RWByteAddressBuffer mem,uint64_t a,out uint value) {{
  value=0u;if(a<{guest_base}ull)return false;uint64_t relative=a-{guest_base}ull;
  if(relative>=(uint64_t){region_len}u||relative>0xffffffffull-(uint64_t){host_offset}u)return false;
  uint host={host_offset}u+(uint)relative,bytes;mem.GetDimensions(bytes);uint aligned=host&~3u;
  if(bytes<4u||aligned>bytes-4u)return false;value=(mem.Load(aligned)>>((host&3u)*8u))&255u;return true;
}}
uint xcl_fmtbytes(uint f) {{ if(f==8u)return 2u;if(f==20u||f==21u||f==22u||f==24u||f==29u||f==50u||f==56u||f==60u)return 4u;if(f==69u)return 8u;return 0u; }}
bool xcl_fmtc(RWByteAddressBuffer mem,uint64_t a,uint f,uint c,out uint v) {{
  v=0u;uint n=xcl_fmtbytes(f);if(n==0u)return false;uint2 r=0u.xx;
  [unroll]for(uint i=0u;i<8u;i++){{if(i<n){{uint b;if(!xcl_global_byte(mem,a+(uint64_t)i,b))return false;if(i<4u)r.x|=b<<(8u*i);else r.y|=b<<(8u*(i-4u));}}}}
  if(f==8u){{if(c>0u){{v=c==3u?0x3f800000u:0u;return true;}}int x=(int)(r.x<<16)>>16;v=asuint(x==0?0.0f:(x>=32767?1.0f:max((float)x/32767.0f,-1.0f)));return true;}}
  if(f==20u||f==21u||f==22u){{v=c==0u?r.x:(c==3u?(f==20u||f==21u?1u:0x3f800000u):0u);return true;}}
  if(f==24u){{if(c>=2u){{v=c==3u?0x3f800000u:0u;return true;}}int x=(int)((r.x>>(16u*c))<<16)>>16;v=asuint(x==0?0.0f:(x>=32767?1.0f:max((float)x/32767.0f,-1.0f)));return true;}}
  if(f==29u){{v=c<2u?asuint(f16tof32((r.x>>(16u*c))&65535u)):(c==3u?0x3f800000u:0u);return true;}}
  if(f==50u){{uint w=c==3u?2u:10u,o=c==3u?30u:10u*c,m=(1u<<w)-1u,x=(r.x>>o)&m;v=asuint((float)x/(float)m);return true;}}
  if(f==56u){{v=asuint((float)((r.x>>(8u*c))&255u)/255.0f);return true;}}
  if(f==60u){{v=(r.x>>(8u*c))&255u;return true;}}
  if(f==69u){{v=((c<2u?r.x:r.y)>>(16u*(c&1u)))&65535u;return true;}}return false;
}}
bool xcl_fmtv(RWByteAddressBuffer mem,uint64_t a,uint f,uint s,out uint v) {{ bool ii=f==20u||f==21u||f==60u||f==69u;if(s==0u){{v=0u;return true;}}if(s==1u){{v=ii?1u:0x3f800000u;return true;}}if(s>=4u)return xcl_fmtc(mem,a,f,s-4u,v);v=0u;return false; }}
void xcl_allocate(uint value) {{}}
void xcl_export(RWByteAddressBuffer target,uint slot,uint enable,uint compressed,uint done) {{}}

[WaveSize(64)] [numthreads(128,1,1)]
void main(uint gi:SV_GroupIndex,uint3 group_id:SV_GroupID) {{
  uint wave=gi>>6u,lane=gi&63u;
  uint io_size,res_size,lds_size,vgpr_size;io.GetDimensions(io_size);res.GetDimensions(res_size);lds.GetDimensions(lds_size);vgpr_scratch.GetDimensions(vgpr_size);
  bool group_valid=group_id.y==0u&&group_id.z==0u&&group_id.x<{workgroups}u&&io_size>={input_bytes}u&&res_size>={result_bytes}u&&lds_size>={lds_bytes}u&&vgpr_size>={vgpr_scratch_bytes}u;
  if(gi==0u){{xcl_group_base=group_valid?group_id.x*{LDS_SLICE_BYTES}u:0u;xcl_vgpr_group_base=group_valid?group_id.x*{vgpr_record_stride}u:0u;}}
  GroupMemoryBarrierWithGroupSync();
  if(lane==0u){{
    uint wb=group_id.x*{input_group_dwords}u+wave*{wave_input_dwords}u;
    [unroll]for(uint r=0u;r<{sgprs}u;++r)sgpr[wave][r]=0u;
    exec_mask[wave]=0u.xx;vcc[wave]=0u.xx;scc[wave]=0u;m0[wave]=0u;
    if(group_valid){{
      [unroll]for(uint r=0u;r<{sgprs}u;++r)sgpr[wave][r]=io.Load(4u*(wb+r));
      exec_mask[wave]=io.Load2(4u*(wb+{sgprs}u));
      vcc[wave]=io.Load2(4u*(wb+{sgprs_plus_two}u));
      scc[wave]=io.Load(4u*(wb+{sgprs_plus_four}u));
      m0[wave]=io.Load(4u*(wb+{sgprs_plus_five}u));
    }}
    wave_pc[wave]=0u;xcl_next_pc[wave]=0u;xcl_status[wave]=0u;xcl_steps[wave]=0u;xcl_barrier_waiting[wave]=0u;xcl_barrier_generation[wave]=0u;
  }}
  [unroll]for(uint r=0u;r<{vgprs}u;++r){{
    uint xcl_initial_vgpr=0u;
    if(group_valid)xcl_initial_vgpr=io.Load(4u*(group_id.x*{input_group_dwords}u+{vgpr_input_base}u+(wave*64u+lane)*{vgprs}u+r));
    if(group_valid)vgpr_scratch.Store(xcl_vgpr_address(wave,lane,r),xcl_initial_vgpr);
  }}
  AllMemoryBarrierWithGroupSync();
  [loop]for(uint xcl_epoch=0u;xcl_epoch<={step_cap}u;++xcl_epoch){{
    if(group_valid&&lane==0u&&xcl_status[wave]==0u&&wave_pc[wave]>={slot_count}u)xcl_status[wave]=3u;
    GroupMemoryBarrierWithGroupSync();
{phases}  }}
  if(group_valid){{
    if(lane==0u){{res.Store4(4u*(group_id.x*{result_group_dwords}u+wave*4u),uint4(wave_pc[wave],xcl_status[wave],xcl_steps[wave],scc[wave]));res.Store(4u*(group_id.x*{result_group_dwords}u+8u+wave),sgpr[wave][0]);}}
    [unroll]for(uint r=0u;r<{vgprs}u;++r)res.Store(4u*(group_id.x*{result_group_dwords}u+128u+(wave*64u+lane)*256u+r),vgpr_scratch.Load(xcl_vgpr_address(wave,lane,r)));
  }}
}}
"#,
        sgprs_plus_two = sgprs + 2,
        sgprs_plus_four = sgprs + 4,
        sgprs_plus_five = sgprs + 5,
        guest_base = region.guest_base,
        host_offset = region.memory_offset,
        region_len = region.byte_len,
        vgpr_data_offset = super::mesh_dispatch::MIN_RECORD_BYTES,
        vgpr_stride = vgprs * 4,
        vgpr_record_stride = vgpr_scratch.scratch.record_stride_bytes,
        vgpr_scratch_bytes = vgpr_scratch.scratch.total_bytes,
    );
    Ok(ValidationEpochShell {
        hlsl,
        input_dwords,
        result_dwords,
        lds_bytes,
        vgpr_scratch,
    })
}

fn validate_boundaries(program: &MeshProgram) -> Result<(), String> {
    if program.instructions.is_empty() {
        return Err("uniform epoch plan requires at least one instruction".into());
    }
    let mut expected_pc = 0u64;
    for instruction in &program.instructions {
        if instruction.pc != expected_pc || instruction.byte_len != instruction.words.len() * 4 {
            return Err("instruction boundaries are not contiguous".into());
        }
        expected_pc = expected_pc
            .checked_add(instruction.byte_len as u64)
            .ok_or("instruction range overflows")?;
    }
    if expected_pc as usize != program.byte_len {
        return Err("program byte length does not match its slots".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::mesh_dispatch::{
        run_control, ControlOp, LogicalWaveState, LogicalWaveStatus, MaskBinaryOp, MaskRegister,
    };
    use crate::gpu::recomp::mesh_memory::{BufferFormatSource, OffsetSource, StageCOp};
    use crate::gpu::recomp::mesh_program::{CoverageSummary, ProgramInstruction, ProgramRequirements};
    use crate::gpu::recomp::mesh_scalar::{
        Operand32, Operand64, ScalarBinary64Kind, ScalarBinaryKind, StageAOp,
    };
    use crate::gpu::recomp::mesh_vector::{
        CompareKind, FloatBinaryKind, FloatConvertKind, FloatMathKind, FloatModifiers,
        FloatTernaryKind, FloatUnaryKind, FmaMixKind, ScalarMask, SdwaWord, StageBOp,
        StageDIntegerKind, VectorBinaryKind, VectorSource, VectorTernaryKind,
    };

    fn program(ops: Vec<ProgramOp>) -> MeshProgram {
        let instructions = ops
            .into_iter()
            .enumerate()
            .map(|(i, op)| ProgramInstruction {
                pc: (i * 4) as u64,
                byte_len: 4,
                words: vec![0],
                literal: None,
                name: "TEST",
                encoding: "TEST",
                op,
            })
            .collect::<Vec<_>>();
        MeshProgram {
            byte_len: instructions.len() * 4,
            instructions,
            coverage: CoverageSummary::default(),
        }
    }

    fn rendezvous_program() -> MeshProgram {
        program(vec![
            ProgramOp::Scalar(StageAOp::MoveImmediate { dst: 0, value: 5 }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 0,
                src: VectorSource::Scalar(0),
            }),
            ProgramOp::Memory(StageCOp::DsWrite {
                address: 1,
                data: [0, 0],
                offsets: [0, 0],
                words: 1,
            }),
            ProgramOp::Memory(StageCOp::DsRead {
                dst: 2,
                address: 1,
                offsets: [0, 0],
                words: 1,
            }),
            ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 3,
                source: 2,
                control: 0x1b,
            }),
        ])
    }

    fn global_load_program() -> MeshProgram {
        program(vec![ProgramOp::Memory(StageCOp::GlobalLoad {
            dst: 0,
            address: 1,
            scalar_address: None,
            offset: 0,
        })])
    }

    fn branch_program() -> MeshProgram {
        program(vec![
            ProgramOp::Control(ControlOp::BranchScc {
                set: true,
                target: 2,
            }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 0,
                src: VectorSource::Constant(11),
            }),
            ProgramOp::Control(ControlOp::Complete),
        ])
    }

    fn backedge_program() -> MeshProgram {
        program(vec![
            ProgramOp::Control(ControlOp::ScalarSet {
                register: 0,
                value: 0,
            }),
            ProgramOp::Control(ControlOp::ScalarAdd {
                register: 0,
                value: 1,
            }),
            ProgramOp::Control(ControlOp::BranchScalarLess {
                register: 0,
                limit: 3,
                target: 1,
            }),
            ProgramOp::Control(ControlOp::Complete),
        ])
    }

    fn float_binary_program() -> MeshProgram {
        let kinds = [
            FloatBinaryKind::Add,
            FloatBinaryKind::Subtract,
            FloatBinaryKind::ReverseSubtract,
            FloatBinaryKind::Multiply,
            FloatBinaryKind::MultiplyLegacy,
            FloatBinaryKind::Maximum,
            FloatBinaryKind::Minimum,
        ];
        let mut ops = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                ProgramOp::Vector(StageBOp::FloatBinary {
                    kind,
                    dst: (index + 2) as u16,
                    lhs: VectorSource::Vgpr(0),
                    rhs: VectorSource::Vgpr(1),
                    modifiers: FloatModifiers::default(),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::FloatBinary {
            kind: FloatBinaryKind::Add,
            dst: 9,
            lhs: VectorSource::Constant((-0.25f32).to_bits()),
            rhs: VectorSource::Constant((-0.5f32).to_bits()),
            modifiers: FloatModifiers {
                abs: 1,
                neg: 2,
                omod: 1,
                clamp: true,
            },
        }));
        ops.push(ProgramOp::Vector(StageBOp::FloatBinary {
            kind: FloatBinaryKind::MultiplyLegacy,
            dst: 10,
            lhs: VectorSource::Constant(0.0f32.to_bits()),
            rhs: VectorSource::Constant(f32::NAN.to_bits()),
            modifiers: FloatModifiers::default(),
        }));
        ops.push(ProgramOp::Vector(StageBOp::FloatBinary {
            kind: FloatBinaryKind::Maximum,
            dst: 11,
            lhs: VectorSource::Constant(f32::NAN.to_bits()),
            rhs: VectorSource::Constant(2.0f32.to_bits()),
            modifiers: FloatModifiers::default(),
        }));
        ops.push(ProgramOp::Vector(StageBOp::FloatBinary {
            kind: FloatBinaryKind::Minimum,
            dst: 12,
            lhs: VectorSource::Constant((-0.0f32).to_bits()),
            rhs: VectorSource::Constant(0.0f32.to_bits()),
            modifiers: FloatModifiers::default(),
        }));
        ops.push(ProgramOp::Vector(StageBOp::FloatBinary {
            kind: FloatBinaryKind::Maximum,
            dst: 13,
            lhs: VectorSource::Constant(0.0f32.to_bits()),
            rhs: VectorSource::Constant((-0.0f32).to_bits()),
            modifiers: FloatModifiers::default(),
        }));
        program(ops)
    }

    fn float_fma_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 3,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Vgpr(2)),
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            }),
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 4,
                a: VectorSource::Constant(0.0f32.to_bits()),
                b: VectorSource::Constant(f32::INFINITY.to_bits()),
                c: Some(VectorSource::Constant(2.0f32.to_bits())),
                modifiers: FloatModifiers::default(),
                legacy_zero: true,
            }),
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 5,
                a: VectorSource::Constant(2.0f32.to_bits()),
                b: VectorSource::Constant(3.0f32.to_bits()),
                c: None,
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            }),
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 6,
                a: VectorSource::Constant((-0.25f32).to_bits()),
                b: VectorSource::Constant((-0.5f32).to_bits()),
                c: Some(VectorSource::Constant((-0.25f32).to_bits())),
                modifiers: FloatModifiers {
                    abs: 1,
                    neg: 2,
                    omod: 2,
                    clamp: true,
                },
                legacy_zero: false,
            }),
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 7,
                a: VectorSource::Constant(0.0f32.to_bits()),
                b: VectorSource::Constant(f32::INFINITY.to_bits()),
                c: Some(VectorSource::Constant(1.0f32.to_bits())),
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            }),
            ProgramOp::Vector(StageBOp::FloatFma {
                dst: 8,
                a: VectorSource::Constant((-0.0f32).to_bits()),
                b: VectorSource::Constant(2.0f32.to_bits()),
                c: Some(VectorSource::Constant((-0.0f32).to_bits())),
                modifiers: FloatModifiers::default(),
                legacy_zero: false,
            }),
        ])
    }

    fn float_mad_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 3,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Vgpr(2)),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 0,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Vgpr(2)),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 4,
                a: VectorSource::Constant(2.0f32.to_bits()),
                b: VectorSource::Constant(3.0f32.to_bits()),
                c: None,
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 5,
                a: VectorSource::Constant((-0.25f32).to_bits()),
                b: VectorSource::Constant((-0.5f32).to_bits()),
                c: Some(VectorSource::Constant((-0.25f32).to_bits())),
                modifiers: FloatModifiers {
                    abs: 1,
                    neg: 2,
                    omod: 2,
                    clamp: true,
                },
            }),
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 6,
                a: VectorSource::Constant(0.0f32.to_bits()),
                b: VectorSource::Constant(f32::INFINITY.to_bits()),
                c: Some(VectorSource::Constant(1.0f32.to_bits())),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatMad {
                dst: 7,
                a: VectorSource::Constant((-0.0f32).to_bits()),
                b: VectorSource::Constant(2.0f32.to_bits()),
                c: Some(VectorSource::Constant((-0.0f32).to_bits())),
                modifiers: FloatModifiers::default(),
            }),
        ])
    }

    fn stage_d_integer_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::LogicalShiftRight,
                dst: 2,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: None,
            }),
            ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::MadU24,
                dst: 3,
                a: VectorSource::Vgpr(1),
                b: VectorSource::Vgpr(1),
                c: Some(VectorSource::Constant(0xffff_ffff)),
            }),
            ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::Permute,
                dst: 4,
                a: VectorSource::Constant(0xc433_2211),
                b: VectorSource::Constant(0x8877_e655),
                c: Some(VectorSource::Constant(0x0c08_0400)),
            }),
            ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::Permute,
                dst: 5,
                a: VectorSource::Constant(0xc433_2211),
                b: VectorSource::Constant(0x8877_e655),
                c: Some(VectorSource::Constant(0x0d0b_0a09)),
            }),
            ProgramOp::Vector(StageBOp::StageDInteger {
                kind: StageDIntegerKind::LogicalShiftRight,
                dst: 0,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: None,
            }),
        ])
    }

    fn cndmask_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::CndMask {
                dst: 2,
                when_clear: VectorSource::Vgpr(0),
                when_set: VectorSource::Vgpr(1),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 3,
                when_clear: VectorSource::Vgpr(0),
                when_set: VectorSource::Vgpr(1),
                predicate: ScalarMask::ScalarPair(0),
            }),
            ProgramOp::Vector(StageBOp::CndMask {
                dst: 0,
                when_clear: VectorSource::Vgpr(0),
                when_set: VectorSource::Vgpr(1),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 1,
                when_clear: VectorSource::Vgpr(0),
                when_set: VectorSource::Vgpr(1),
                predicate: ScalarMask::ScalarPair(0),
            }),
        ])
    }

    fn compare_program() -> MeshProgram {
        let kinds = [
            CompareKind::Eq,
            CompareKind::NotEqual,
            CompareKind::Greater,
            CompareKind::GreaterEqual,
            CompareKind::Less,
            CompareKind::LessEqual,
            CompareKind::SignedGreater,
            CompareKind::SignedLess,
            CompareKind::EqualFloat,
            CompareKind::OrderedNotEqualFloat,
            CompareKind::UnorderedNotEqualFloat,
            CompareKind::LessFloat,
            CompareKind::GreaterFloat,
            CompareKind::LessEqualFloat,
            CompareKind::GreaterEqualFloat,
            CompareKind::UnorderedGreaterFloat,
            CompareKind::UnorderedLessFloat,
            CompareKind::UnorderedLessEqualFloat,
            CompareKind::LessFloat16,
        ];
        let mut ops = Vec::new();
        for (dst, kind) in kinds.into_iter().enumerate() {
            ops.push(ProgramOp::Vector(StageBOp::Compare {
                kind,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                cmpx: false,
            }));
            ops.push(ProgramOp::Vector(StageBOp::CndMask {
                dst: (dst + 2) as u16,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
            }));
        }
        ops.push(ProgramOp::Vector(StageBOp::CompareScalar {
            kind: CompareKind::Eq,
            lhs: VectorSource::Vgpr(0),
            rhs: VectorSource::Vgpr(1),
            dst: 0,
        }));
        ops.push(ProgramOp::Vector(StageBOp::CndMaskScalar {
            dst: 21,
            when_clear: VectorSource::Constant(0),
            when_set: VectorSource::Constant(1),
            predicate: ScalarMask::ScalarPair(0),
        }));
        ops.push(ProgramOp::Vector(StageBOp::Compare {
            kind: CompareKind::Greater,
            lhs: VectorSource::Vgpr(0),
            rhs: VectorSource::Vgpr(1),
            cmpx: true,
        }));
        ops.push(ProgramOp::Vector(StageBOp::Move {
            dst: 22,
            src: VectorSource::Constant(0xfeed_beef),
        }));
        program(ops)
    }

    fn specialized_compare_program() -> MeshProgram {
        let mut ops = vec![
            ProgramOp::Vector(StageBOp::CompareEqU64 {
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(2),
                dst: None,
            }),
            ProgramOp::Vector(StageBOp::CndMask {
                dst: 7,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
            }),
            ProgramOp::Vector(StageBOp::CompareEqU64 {
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(2),
                dst: Some(0),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 8,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
                predicate: ScalarMask::ScalarPair(0),
            }),
            ProgramOp::Vector(StageBOp::CompareClassF32 {
                value: VectorSource::Vgpr(4),
                classes: VectorSource::Vgpr(6),
                dst: None,
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::CndMask {
                dst: 9,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
            }),
            ProgramOp::Vector(StageBOp::CompareClassF32 {
                value: VectorSource::Vgpr(4),
                classes: VectorSource::Vgpr(6),
                dst: Some(2),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 22,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
                predicate: ScalarMask::ScalarPair(2),
            }),
        ];
        let kinds = [
            CompareKind::EqualFloat,
            CompareKind::OrderedNotEqualFloat,
            CompareKind::UnorderedNotEqualFloat,
            CompareKind::LessFloat,
            CompareKind::GreaterFloat,
            CompareKind::LessEqualFloat,
            CompareKind::GreaterEqualFloat,
            CompareKind::UnorderedGreaterFloat,
            CompareKind::UnorderedLessFloat,
            CompareKind::UnorderedLessEqualFloat,
        ];
        for (index, kind) in kinds.into_iter().enumerate() {
            ops.push(ProgramOp::Vector(StageBOp::CompareLessFloat {
                kind,
                lhs: VectorSource::Vgpr(4),
                rhs: VectorSource::Vgpr(5),
                dst: None,
                modifiers: FloatModifiers::default(),
                cmpx: false,
            }));
            ops.push(ProgramOp::Vector(StageBOp::CndMask {
                dst: (10 + index) as u16,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
            }));
        }
        ops.push(ProgramOp::Vector(StageBOp::CompareLessFloat {
            kind: CompareKind::GreaterFloat,
            lhs: VectorSource::Vgpr(4),
            rhs: VectorSource::Vgpr(5),
            dst: Some(4),
            modifiers: FloatModifiers {
                abs: 1,
                neg: 2,
                omod: 0,
                clamp: false,
            },
            cmpx: false,
        }));
        ops.push(ProgramOp::Vector(StageBOp::CndMaskScalar {
            dst: 20,
            when_clear: VectorSource::Constant(0),
            when_set: VectorSource::Constant(1),
            predicate: ScalarMask::ScalarPair(4),
        }));
        ops.push(ProgramOp::Vector(StageBOp::CompareLessFloat {
            kind: CompareKind::LessFloat,
            lhs: VectorSource::Vgpr(4),
            rhs: VectorSource::Vgpr(5),
            dst: None,
            modifiers: FloatModifiers::default(),
            cmpx: true,
        }));
        ops.push(ProgramOp::Vector(StageBOp::Move {
            dst: 21,
            src: VectorSource::Constant(0xfeed_beef),
        }));
        program(ops)
    }

    fn ternary_program() -> MeshProgram {
        let kinds = [
            VectorTernaryKind::Add3,
            VectorTernaryKind::BfeU32,
            VectorTernaryKind::LshlAdd,
            VectorTernaryKind::AddLshl,
            VectorTernaryKind::AndOr,
            VectorTernaryKind::Or3,
            VectorTernaryKind::LshlOr,
            VectorTernaryKind::BitFieldInsert,
            VectorTernaryKind::XorAdd,
        ];
        let mut ops = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                ProgramOp::Vector(StageBOp::Ternary {
                    kind,
                    dst: (index + 3) as u16,
                    a: VectorSource::Vgpr(0),
                    b: VectorSource::Vgpr(1),
                    c: VectorSource::Vgpr(2),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::Ternary {
            kind: VectorTernaryKind::XorAdd,
            dst: 0,
            a: VectorSource::Vgpr(0),
            b: VectorSource::Vgpr(1),
            c: VectorSource::Vgpr(2),
        }));
        program(ops)
    }

    fn float_math_program() -> MeshProgram {
        let kinds = [
            FloatMathKind::Reciprocal,
            FloatMathKind::ReciprocalIFlag,
            FloatMathKind::ReciprocalSqrt,
            FloatMathKind::Sqrt,
            FloatMathKind::Sin,
            FloatMathKind::Cos,
            FloatMathKind::Exp2,
            FloatMathKind::Fract,
            FloatMathKind::Log2,
        ];
        let mut ops = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                ProgramOp::Vector(StageBOp::FloatMath {
                    kind,
                    dst: (index + 1) as u16,
                    src: VectorSource::Vgpr(0),
                    modifiers: FloatModifiers::default(),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::FloatMath {
            kind: FloatMathKind::Reciprocal,
            dst: 10,
            src: VectorSource::Vgpr(0),
            modifiers: FloatModifiers {
                abs: 1,
                neg: 1,
                omod: 1,
                clamp: true,
            },
        }));
        ops.push(ProgramOp::Vector(StageBOp::FloatMath {
            kind: FloatMathKind::Sqrt,
            dst: 0,
            src: VectorSource::Vgpr(0),
            modifiers: FloatModifiers::default(),
        }));
        program(ops)
    }

    fn float_convert_program() -> MeshProgram {
        let kinds = [
            FloatConvertKind::F16FromF32,
            FloatConvertKind::F16FromF32High,
            FloatConvertKind::F16FromF32HighPad,
            FloatConvertKind::F32FromF16,
            FloatConvertKind::F32FromF16High,
            FloatConvertKind::U32FromF32,
            FloatConvertKind::I32FromF32,
        ];
        let mut ops = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                ProgramOp::Vector(StageBOp::FloatConvert {
                    kind,
                    dst: (index + 1) as u16,
                    src: VectorSource::Vgpr(0),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::FloatConvert {
            kind: FloatConvertKind::F16FromF32,
            dst: 0,
            src: VectorSource::Vgpr(0),
        }));
        program(ops)
    }

    fn lane_remainder_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::ReadLane {
                dst: 0,
                src: 0,
                lane: VectorSource::Constant(63),
            }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 1,
                src: VectorSource::Scalar(0),
            }),
            ProgramOp::Vector(StageBOp::WriteLane {
                dst: 3,
                src: VectorSource::Constant(0xdead_beef),
                lane: VectorSource::Constant(5),
            }),
            ProgramOp::Vector(StageBOp::Mbcnt {
                high: false,
                dst: 4,
                source: VectorSource::Constant(0xffff_ffff),
                accumulator: VectorSource::Constant(10),
            }),
            ProgramOp::Vector(StageBOp::Mbcnt {
                high: true,
                dst: 5,
                source: VectorSource::Constant(0xffff_ffff),
                accumulator: VectorSource::Constant(20),
            }),
            ProgramOp::Vector(StageBOp::PermLaneX16 {
                dst: 6,
                src: 0,
                selectors_low: VectorSource::Constant(0),
                selectors_high: VectorSource::Constant(0xfedc_ba98),
                fetch_inactive: true,
                bound_control: false,
            }),
            ProgramOp::Vector(StageBOp::PermLaneX16 {
                dst: 7,
                src: 0,
                selectors_low: VectorSource::Constant(0),
                selectors_high: VectorSource::Constant(0xfedc_ba98),
                fetch_inactive: false,
                bound_control: true,
            }),
            ProgramOp::Vector(StageBOp::PermLaneX16 {
                dst: 0,
                src: 0,
                selectors_low: VectorSource::Constant(0),
                selectors_high: VectorSource::Constant(0xfedc_ba98),
                fetch_inactive: true,
                bound_control: false,
            }),
        ])
    }

    fn fma_mix_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::F32,
                dst: 3,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            }),
            ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::LoF16,
                dst: 4,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            }),
            ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::HiF16,
                dst: 5,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            }),
            ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::F32,
                dst: 6,
                a: VectorSource::Constant(0xbc00_3800),
                b: VectorSource::Constant(0x4000_3c00),
                c: VectorSource::Constant(0xb800_3400),
                op_sel: 0b010,
                op_sel_hi: 0b111,
                neg: 0b010,
                abs: 0b001,
                clamp: true,
            }),
            ProgramOp::Vector(StageBOp::FmaMix {
                kind: FmaMixKind::F32,
                dst: 0,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                op_sel: 0,
                op_sel_hi: 0,
                neg: 0,
                abs: 0,
                clamp: false,
            }),
        ])
    }

    fn convert_ubyte_program() -> MeshProgram {
        let mut ops = (0u8..4)
            .map(|byte| {
                ProgramOp::Vector(StageBOp::ConvertUByte {
                    dst: u16::from(byte) + 1,
                    src: VectorSource::Vgpr(0),
                    byte,
                    modifiers: FloatModifiers::default(),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::ConvertUByte {
            dst: 5,
            src: VectorSource::Vgpr(0),
            byte: 2,
            modifiers: FloatModifiers {
                abs: 1,
                neg: 1,
                omod: 2,
                clamp: true,
            },
        }));
        ops.push(ProgramOp::Vector(StageBOp::ConvertUByte {
            dst: 0,
            src: VectorSource::Vgpr(0),
            byte: 0,
            modifiers: FloatModifiers::default(),
        }));
        program(ops)
    }

    fn small_integer_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::NoEffect),
            ProgramOp::Vector(StageBOp::AddCarry {
                dst: 2,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: Some(ScalarMask::Vcc),
                carry_out: ScalarMask::ScalarPair(0),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 3,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
                predicate: ScalarMask::ScalarPair(0),
            }),
            ProgramOp::Vector(StageBOp::ReverseSubCarry {
                dst: 4,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: ScalarMask::ScalarPair(0),
                carry_out: ScalarMask::Vcc,
            }),
            ProgramOp::Vector(StageBOp::CndMask {
                dst: 5,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
            }),
            ProgramOp::Vector(StageBOp::BitReverse {
                dst: 6,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::BitNot {
                dst: 7,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::BitScan {
                high: true,
                dst: 8,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::BitScan {
                high: false,
                dst: 9,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::Move {
                dst: 10,
                src: VectorSource::Constant(0xabcdef01),
            }),
            ProgramOp::Vector(StageBOp::AddCarry {
                dst: 11,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: Some(ScalarMask::Exec),
                carry_out: ScalarMask::ScalarPair(2),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 12,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
                predicate: ScalarMask::ScalarPair(2),
            }),
            ProgramOp::Vector(StageBOp::ReverseSubCarry {
                dst: 13,
                lhs: VectorSource::Vgpr(0),
                rhs: VectorSource::Vgpr(1),
                carry_in: ScalarMask::Exec,
                carry_out: ScalarMask::ScalarPair(4),
            }),
            ProgramOp::Vector(StageBOp::CndMaskScalar {
                dst: 14,
                when_clear: VectorSource::Constant(0),
                when_set: VectorSource::Constant(1),
                predicate: ScalarMask::ScalarPair(4),
            }),
            ProgramOp::Vector(StageBOp::BitNot {
                dst: 0,
                src: VectorSource::Vgpr(0),
            }),
        ])
    }

    fn f32_integer_convert_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::ConvertF32U32 {
                dst: 1,
                src: VectorSource::Vgpr(0),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::ConvertF32U32 {
                dst: 2,
                src: VectorSource::Vgpr(0),
                modifiers: FloatModifiers {
                    abs: 1,
                    neg: 1,
                    omod: 1,
                    clamp: true,
                },
            }),
            ProgramOp::Vector(StageBOp::ConvertF32I32 {
                dst: 3,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::ConvertF32U32Sdwa {
                dst: 4,
                src: 0,
                word: SdwaWord::Low,
            }),
            ProgramOp::Vector(StageBOp::ConvertF32U32Sdwa {
                dst: 5,
                src: 0,
                word: SdwaWord::High,
            }),
            ProgramOp::Vector(StageBOp::ConvertF32U32 {
                dst: 0,
                src: VectorSource::Vgpr(0),
                modifiers: FloatModifiers::default(),
            }),
        ])
    }

    fn pack_f16_program() -> MeshProgram {
        let mut ops = (0u8..4)
            .map(|op_sel| {
                ProgramOp::Vector(StageBOp::PackF16 {
                    dst: u16::from(op_sel) + 2,
                    lo: VectorSource::Vgpr(0),
                    hi: VectorSource::Vgpr(1),
                    op_sel,
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::PackF16 {
            dst: 0,
            lo: VectorSource::Vgpr(0),
            hi: VectorSource::Vgpr(1),
            op_sel: 0,
        }));
        program(ops)
    }

    fn float_remainder_program() -> MeshProgram {
        program(vec![
            ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::Truncate,
                dst: 3,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::Floor,
                dst: 4,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::Ceiling,
                dst: 5,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::RoundEven,
                dst: 6,
                src: VectorSource::Vgpr(0),
            }),
            ProgramOp::Vector(StageBOp::FloatTernary {
                kind: FloatTernaryKind::Minimum3,
                dst: 7,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatTernary {
                kind: FloatTernaryKind::Maximum3,
                dst: 8,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatTernary {
                kind: FloatTernaryKind::Median3,
                dst: 9,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                modifiers: FloatModifiers::default(),
            }),
            ProgramOp::Vector(StageBOp::FloatTernary {
                kind: FloatTernaryKind::Median3,
                dst: 10,
                a: VectorSource::Vgpr(0),
                b: VectorSource::Vgpr(1),
                c: VectorSource::Vgpr(2),
                modifiers: FloatModifiers {
                    abs: 1,
                    neg: 2,
                    omod: 1,
                    clamp: true,
                },
            }),
            ProgramOp::Vector(StageBOp::FloatUnary {
                kind: FloatUnaryKind::RoundEven,
                dst: 0,
                src: VectorSource::Vgpr(0),
            }),
        ])
    }

    fn binary_program() -> MeshProgram {
        let kinds = [
            VectorBinaryKind::AddNoCarry,
            VectorBinaryKind::And,
            VectorBinaryKind::Or,
            VectorBinaryKind::Xor,
            VectorBinaryKind::BitCount,
            VectorBinaryKind::LshlReverse,
            VectorBinaryKind::ArithmeticShiftRightReverse,
            VectorBinaryKind::MinU32,
            VectorBinaryKind::MaxU32,
            VectorBinaryKind::MulLoU32,
            VectorBinaryKind::MulHiU32,
            VectorBinaryKind::SubNoCarry,
            VectorBinaryKind::ReverseSubNoCarry,
        ];
        let mut ops = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                ProgramOp::Vector(StageBOp::Binary {
                    kind,
                    dst: (index + 2) as u16,
                    lhs: VectorSource::Vgpr(0),
                    rhs: VectorSource::Vgpr(1),
                })
            })
            .collect::<Vec<_>>();
        ops.push(ProgramOp::Vector(StageBOp::Binary {
            kind: VectorBinaryKind::AddNoCarry,
            dst: 0,
            lhs: VectorSource::Vgpr(0),
            rhs: VectorSource::Vgpr(1),
        }));
        ops.push(ProgramOp::Vector(StageBOp::Binary {
            kind: VectorBinaryKind::AddNoCarry,
            dst: 15,
            lhs: VectorSource::VccZero,
            rhs: VectorSource::ExecZero,
        }));
        ops.push(ProgramOp::Vector(StageBOp::Binary {
            kind: VectorBinaryKind::AddNoCarry,
            dst: 16,
            lhs: VectorSource::Scc,
            rhs: VectorSource::Constant(4),
        }));
        program(ops)
    }

    fn set_compact_input(state: &mut super::super::runner::KernelState, index: usize, value: u32) {
        if index < 128 {
            state.sgpr[index] = value;
        } else {
            let at = index - 128;
            state.vgpr[at / 256][at % 256] = value;
        }
    }

    fn slice_address(group: [u32; 3], groups: u32, resource_bytes: u32, local: u32) -> Option<u32> {
        let required = groups.checked_mul(LDS_SLICE_BYTES)?;
        if group[1] != 0 || group[2] != 0 || group[0] >= groups || resource_bytes < required {
            return None;
        }
        group[0]
            .checked_mul(LDS_SLICE_BYTES)?
            .checked_add(((local >> 2) & 8191) << 2)
    }

    fn checked_region_address(
        address: u64,
        offset: i32,
        access: u32,
        region: ReadOnlyRegion,
        bytes: u32,
    ) -> Option<u32> {
        let address = if offset < 0 {
            address.checked_sub(u64::from(offset.unsigned_abs()))?
        } else {
            address.checked_add(offset as u64)?
        };
        if access == 0 || address & u64::from(access - 1) != 0 {
            return None;
        }
        let relative = address.checked_sub(region.guest_base)?;
        if relative.checked_add(u64::from(access))? > u64::from(region.byte_len) {
            return None;
        }
        let host = u64::from(region.memory_offset).checked_add(relative)?;
        (host.checked_add(u64::from(access))? <= u64::from(bytes)).then_some(host as u32)
    }

    #[test]
    fn plan_checks_slices_state_and_fail_closed_subset() {
        let shell = validation_epoch_hlsl(None, &rendezvous_program(), 2, 16).unwrap();
        assert_eq!(shell.lds_bytes, 65536);
        assert!(shell.input_dwords > 0 && shell.result_dwords > shell.input_dwords);
        assert!(shell.hlsl.contains("group_id.x*32768u"));
        assert!(shell.hlsl.contains("AllMemoryBarrierWithGroupSync"));
        assert!(shell.hlsl.contains("GroupMemoryBarrierWithGroupSync"));
        assert!(validation_epoch_hlsl(None, &rendezvous_program(), 0, 16).is_err());
        assert!(validation_epoch_hlsl(None, &rendezvous_program(), u32::MAX, 16).is_err());
        let barrier = validation_epoch_hlsl(
            None,
            &program(vec![ProgramOp::Scalar(StageAOp::Barrier)]),
            1,
            16,
        )
        .unwrap();
        assert!(barrier.hlsl.contains("xcl_barrier_generation"));
        assert!(barrier.hlsl.contains("xcl_barrier_action"));
        assert_eq!(validation_shared_bytes(115), Some(1024));
        assert_eq!(validation_shared_bytes(116), Some(1032));
        assert!(validation_shared_bytes(128).unwrap() <= MAX_GROUP_SHARED_BYTES);
        let boundary = |scalar_dst| {
            program(vec![
                ProgramOp::Scalar(StageAOp::MoveImmediate { dst: scalar_dst, value: 0 }),
                ProgramOp::Vector(StageBOp::Move {
                    dst: 61,
                    src: VectorSource::Constant(0),
                }),
                ProgramOp::Scalar(StageAOp::Barrier),
            ])
        };
        assert!(validation_epoch_hlsl(None, &boundary(114), 1, 8).is_ok());
        let formerly_limited = validation_epoch_hlsl(None, &boundary(115), 1, 8).unwrap();
        assert_eq!(formerly_limited.vgpr_scratch.vgpr_count, 62);
        assert!(!formerly_limited.hlsl.contains("groupshared uint vgpr"));
        assert!(formerly_limited.hlsl.contains("RWByteAddressBuffer vgpr_scratch : register(u6)"));
        assert!(validation_epoch_hlsl(None, &global_load_program(), 1, 8)
            .unwrap_err()
            .contains("requires an explicit region"));
        let mapped = validation_epoch_hlsl(
            Some(ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 16, byte_len: 240 }),
            &global_load_program(),
            1,
            8,
        )
        .unwrap();
        for required in [
            "a<4294967296ull",
            "relative>(uint64_t)(240u-4u)",
            "0xffffffffull-(uint64_t)16u",
            "host>bytes-4u",
        ] {
            assert!(mapped.hlsl.contains(required), "missing {required}");
        }
        for invalid in [
            ReadOnlyRegion { guest_base: 0, memory_offset: 0, byte_len: 0 },
            ReadOnlyRegion { guest_base: 1, memory_offset: 0, byte_len: 4 },
            ReadOnlyRegion { guest_base: 0, memory_offset: 1, byte_len: 4 },
            ReadOnlyRegion { guest_base: 0, memory_offset: 0, byte_len: 5 },
            ReadOnlyRegion { guest_base: u64::MAX - 3, memory_offset: 0, byte_len: 4 },
            ReadOnlyRegion { guest_base: 0, memory_offset: u32::MAX - 3, byte_len: 4 },
        ] {
            assert!(validation_epoch_hlsl(Some(invalid), &global_load_program(), 1, 8).is_err());
        }
        assert!(validation_epoch_hlsl(
            Some(ReadOnlyRegion { guest_base: 4, memory_offset: 8, byte_len: 4 }),
            &global_load_program(),
            1,
            8,
        )
        .is_ok());
        assert_eq!(slice_address([0, 0, 0], 2, 65536, 0xffff_fffc), Some(32764));
        assert_eq!(slice_address([1, 0, 0], 2, 65536, 0xffff_fffc), Some(65532));
        assert_eq!(slice_address([0, 1, 0], 2, 65536, 0), None);
        assert_eq!(slice_address([0, 0, 1], 2, 65536, 0), None);
        assert_eq!(slice_address([2, 0, 0], 2, 98304, 0), None);
        assert_eq!(slice_address([1, 0, 0], 2, 32768, 0), None);
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 16, byte_len: 240 };
        assert_eq!(checked_region_address(region.guest_base, 0, 4, region, 256), Some(16));
        assert_eq!(checked_region_address(region.guest_base + 1, -1, 4, region, 256), Some(16));
        assert_eq!(checked_region_address(region.guest_base, 1, 4, region, 256), None);
        assert_eq!(checked_region_address(region.guest_base + 240, -4, 4, region, 256), Some(252));
        assert_eq!(checked_region_address(region.guest_base + 236, 0, 4, region, 256), Some(252));
        assert_eq!(checked_region_address(region.guest_base + 237, 0, 4, region, 256), None);
        assert_eq!(checked_region_address(region.guest_base - 4, 0, 4, region, 256), None);
        assert_eq!(checked_region_address(u64::MAX, 1, 4, region, 256), None);
        assert_eq!(checked_region_address(region.guest_base + 1, 0, 4, region, 256), None);
        assert_eq!(checked_region_address(region.guest_base, 0, 4, region, 19), None);
    }

    #[test]
    fn independent_cpu_chain_preserves_inactive_lanes_and_snapshot_sources() {
        let mut vgpr = [[0u32; 128]; 4];
        let mut lds = [0u32; 8192];
        for lane in 0..128usize {
            vgpr[0][lane] = 111;
            vgpr[1][lane] = (lane * 4) as u32;
            vgpr[2][lane] = 9000 + lane as u32;
            vgpr[3][lane] = 777;
        }
        let active = |lane: usize| lane != 1;
        for lane in 0..128usize {
            if active(lane) {
                vgpr[0][lane] = 5;
                lds[((vgpr[1][lane] >> 2) & 8191) as usize] = vgpr[0][lane];
            }
        }
        for lane in 0..128usize {
            if active(lane) {
                vgpr[2][lane] = lds[((vgpr[1][lane] >> 2) & 8191) as usize];
            }
        }
        let snapshot = vgpr[2];
        for lane in 0..128usize {
            if active(lane) {
                let wave_base = lane & !63;
                let source = wave_base | ((lane & 63 & !3) | (3 - (lane & 3)));
                vgpr[3][lane] = snapshot[source];
            }
        }
        assert_eq!((vgpr[0][0], vgpr[2][0], vgpr[3][0]), (5, 5, 5));
        assert_eq!((vgpr[0][1], vgpr[2][1], vgpr[3][1]), (111, 9001, 777));
        assert_eq!(vgpr[3][2], 9001);
        assert_eq!((vgpr[0][64], vgpr[2][64], vgpr[3][64]), (5, 5, 5));
    }

    #[test]
    fn cpu_control_reference_covers_divergence_backedge_and_step_limit() {
        let branch = [
            ControlOp::BranchScc {
                set: true,
                target: 2,
            },
            ControlOp::ScalarSet {
                register: 0,
                value: 11,
            },
            ControlOp::Complete,
        ];
        let taken = run_control(
            &branch,
            LogicalWaveState {
                scc: 1,
                ..Default::default()
            },
            16,
        )
        .unwrap();
        let fallthrough = run_control(&branch, LogicalWaveState::default(), 16).unwrap();
        assert_eq!(
            (taken.status, taken.steps, taken.scalar[0]),
            (LogicalWaveStatus::Complete, 2, 0)
        );
        assert_eq!(
            (fallthrough.status, fallthrough.steps, fallthrough.scalar[0]),
            (LogicalWaveStatus::Complete, 3, 11)
        );

        let looped = [
            ControlOp::ScalarSet {
                register: 0,
                value: 0,
            },
            ControlOp::ScalarAdd {
                register: 0,
                value: 1,
            },
            ControlOp::BranchScalarLess {
                register: 0,
                limit: 3,
                target: 1,
            },
            ControlOp::Complete,
        ];
        let complete = run_control(&looped, LogicalWaveState::default(), 16).unwrap();
        let capped = run_control(&looped, LogicalWaveState::default(), 4).unwrap();
        assert_eq!(
            (complete.status, complete.steps, complete.scalar[0]),
            (LogicalWaveStatus::Complete, 8, 3)
        );
        assert_eq!(
            (capped.status, capped.steps),
            (LogicalWaveStatus::StepLimit, 4)
        );
    }

    #[test]
    fn every_control_variant_lowers_and_compiles_in_uniform_phases() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let ops = vec![
            ControlOp::SetMask {
                dst: MaskRegister::Vcc,
                value: 3,
            },
            ControlOp::CopyMask {
                dst: MaskRegister::Exec,
                src: MaskRegister::Vcc,
            },
            ControlOp::MaskToScalar {
                dst: 2,
                src: MaskRegister::Exec,
            },
            ControlOp::ScalarToMask {
                dst: MaskRegister::Vcc,
                src: 2,
            },
            ControlOp::MaskBinary {
                dst: MaskRegister::Exec,
                lhs: MaskRegister::Exec,
                rhs: MaskRegister::Vcc,
                op: MaskBinaryOp::And,
            },
            ControlOp::MaskScc {
                lhs: MaskRegister::Exec,
                rhs: MaskRegister::Vcc,
            },
            ControlOp::MaskBinaryScalar {
                dst: 4,
                lhs: MaskRegister::Exec,
                rhs: MaskRegister::Vcc,
                op: MaskBinaryOp::Or,
            },
            ControlOp::SaveExecAnd { mask: 3, dst: 6 },
            ControlOp::SaveExecAndMask {
                mask: MaskRegister::Vcc,
                dst: 8,
            },
            ControlOp::WrExecAndNot { mask: 3, dst: 10 },
            ControlOp::WrExecAndNotMask {
                mask: MaskRegister::Vcc,
                dst: 12,
            },
            ControlOp::RestoreExec { src: 6 },
            ControlOp::ScalarSet {
                register: 14,
                value: 1,
            },
            ControlOp::ScalarAdd {
                register: 14,
                value: 1,
            },
            ControlOp::ScalarAddCarry {
                register: 14,
                value: u32::MAX,
            },
            ControlOp::ScalarCompareLess {
                register: 14,
                value: 3,
            },
            ControlOp::BranchScc {
                set: true,
                target: 17,
            },
            ControlOp::BranchMaskZero {
                mask: MaskRegister::Exec,
                zero: false,
                target: 18,
            },
            ControlOp::BranchScalarLess {
                register: 14,
                limit: 3,
                target: 19,
            },
            ControlOp::Jump { target: 20 },
            ControlOp::Complete,
        ];
        let p = program(ops.into_iter().map(ProgramOp::Control).collect());
        let shell = validation_epoch_hlsl(None, &p, 1, 64).unwrap();
        super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
    }

    #[test]
    fn float_binary_family_compiles_and_matches_live_wave64_edges() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_binary_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !2, u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, 2.0f32.to_bits());
                set_compact_input(&mut state, base + 1, 3.0f32.to_bits());
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let expected = [5.0f32, -1.0, 1.0, 6.0, 6.0, 3.0, 2.0].map(f32::to_bits);
        for (index, value) in expected.into_iter().enumerate() {
            assert_eq!(result.raw[128 + index + 2], value);
            assert_eq!(result.raw[128 + 64 * 256 + index + 2], value);
        }
        assert_eq!(result.raw[128 + 9], 1.0f32.to_bits());
        assert_eq!(result.raw[128 + 10], 0.0f32.to_bits());
        assert_eq!(result.raw[128 + 11], 2.0f32.to_bits());
        assert_eq!(result.raw[128 + 12], (-0.0f32).to_bits());
        assert_eq!(result.raw[128 + 13], 0.0f32.to_bits());
        for register in 2..=13 {
            assert_eq!(result.raw[128 + 256 + register], 0x5151_5151);
        }
    }

    #[test]
    fn float_fma_family_compiles_and_matches_live_fused_edges() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_fma_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let a = f32::from_bits(0x3f80_0001);
        let b = f32::from_bits(0x3f80_0001);
        let c = f32::from_bits(0xbf80_0002);
        let fused = a.mul_add(b, c);
        assert_ne!(fused.to_bits(), (a * b + c).to_bits());
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !2, u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, a.to_bits());
                set_compact_input(&mut state, base + 1, b.to_bits());
                set_compact_input(&mut state, base + 2, c.to_bits());
                for register in 3..vgprs {
                    set_compact_input(
                        &mut state,
                        base + register,
                        if register == 5 {
                            4.0f32.to_bits()
                        } else {
                            0x5151_5151
                        },
                    );
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        assert_eq!(at(0, 3), fused.to_bits());
        assert_eq!(at(0, 4), 2.0f32.to_bits());
        assert_eq!(at(0, 5), 10.0f32.to_bits());
        assert_eq!(at(0, 6), 0.0f32.to_bits());
        assert!(f32::from_bits(at(0, 7)).is_nan());
        assert_eq!(at(0, 8), (-0.0f32).to_bits());
        for register in 3..=8 {
            assert_eq!(
                at(1, register),
                if register == 5 {
                    4.0f32.to_bits()
                } else {
                    0x5151_5151
                }
            );
        }
        assert_eq!(at(64, 3), fused.to_bits());
    }

    #[test]
    fn float_mad_family_compiles_and_matches_live_nonfused_edges() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_mad_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let a = f32::from_bits(0x3f80_0001);
        let b = f32::from_bits(0x3f80_0001);
        let c = f32::from_bits(0xbf80_0002);
        let product = f32::from_bits((a * b).to_bits());
        let nonfused = product + c;
        assert_ne!(nonfused.to_bits(), a.mul_add(b, c).to_bits());
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !2, u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, a.to_bits());
                set_compact_input(&mut state, base + 1, b.to_bits());
                set_compact_input(&mut state, base + 2, c.to_bits());
                for register in 3..vgprs {
                    set_compact_input(
                        &mut state,
                        base + register,
                        if register == 4 {
                            4.0f32.to_bits()
                        } else {
                            0x5151_5151
                        },
                    );
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        assert_eq!(at(0, 3), nonfused.to_bits());
        assert_eq!(at(0, 0), nonfused.to_bits());
        assert_eq!(at(0, 4), 10.0f32.to_bits());
        assert_eq!(at(0, 5), 0.0f32.to_bits());
        assert!(f32::from_bits(at(0, 6)).is_nan());
        assert_eq!(at(0, 7), (-0.0f32).to_bits());
        for register in 0..=7 {
            assert_eq!(
                at(1, register),
                match register {
                    0 => a.to_bits(),
                    1 => b.to_bits(),
                    2 => c.to_bits(),
                    4 => 4.0f32.to_bits(),
                    _ => 0x5151_5151,
                }
            );
        }
        assert_eq!(at(64, 3), nonfused.to_bits());
    }

    #[test]
    fn stage_d_integer_family_matches_cpu_edges_aliasing_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = stage_d_integer_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !2, u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, [0u32, 31, 32, 63][lane & 3]);
                set_compact_input(&mut state, base + 1, 0xffff_ffff);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let permute = |a: u32, b: u32, selector: u32| {
            let mut result = 0u32;
            for index in 0..4 {
                let key = (selector >> (index * 8)) & 0xff;
                let byte = match key {
                    13.. => 0xff,
                    12 => 0,
                    8..=11 => {
                        let bit = match key {
                            8 => b >> 15,
                            9 => b >> 31,
                            10 => a >> 15,
                            _ => a >> 31,
                        } & 1;
                        if bit != 0 {
                            0xff
                        } else {
                            0
                        }
                    }
                    0..=3 => (b >> (key * 8)) & 0xff,
                    _ => (a >> ((key - 4) * 8)) & 0xff,
                };
                result |= byte << (index * 8);
            }
            result
        };
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for lane in [0usize, 2, 3, 31, 32, 63, 64, 65] {
            let shift = [0u32, 31, 32, 63][lane & 3];
            let shifted = u32::MAX >> (shift & 31);
            assert_eq!(at(lane, 2), shifted);
            assert_eq!(at(lane, 0), shifted);
            assert_eq!(
                at(lane, 3),
                0xffffffu32.wrapping_mul(0xffffff).wrapping_add(u32::MAX)
            );
            assert_eq!(at(lane, 4), permute(0xc433_2211, 0x8877_e655, 0x0c08_0400));
            assert_eq!(at(lane, 5), permute(0xc433_2211, 0x8877_e655, 0x0d0b_0a09));
        }
        assert_eq!(at(1, 0), 31);
        assert_eq!(at(1, 1), u32::MAX);
        for register in 2..=5 {
            assert_eq!(at(1, register), 0x5151_5151);
        }
    }

    #[test]
    fn cndmask_pair_matches_lane_masks_aliasing_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = cndmask_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let scalar_mask = [0x8000_0001u32, 0x0000_0004];
        let vcc = [[0x5555_5555u32, 0xaaaa_aaaa], [0xaaaa_aaaa, 0x5555_5555]];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            set_compact_input(&mut state, wb, scalar_mask[0]);
            set_compact_input(&mut state, wb + 1, scalar_mask[1]);
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 5), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            set_compact_input(&mut state, wb + sgprs + 2, vcc[wave][0]);
            set_compact_input(&mut state, wb + sgprs + 3, vcc[wave][1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, 0x1111_0000 | lane as u32);
                set_compact_input(&mut state, base + 1, 0x2222_0000 | lane as u32);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 31, 32, 34, 63, 64, 65, 95, 96, 98, 127] {
            let wave = absolute_lane / 64;
            let lane = absolute_lane & 63;
            let clear = 0x1111_0000 | lane as u32;
            let set = 0x2222_0000 | lane as u32;
            let vcc_set = (vcc[wave][lane >> 5] >> (lane & 31)) & 1 != 0;
            let scalar_set = (scalar_mask[lane >> 5] >> (lane & 31)) & 1 != 0;
            assert_eq!(at(absolute_lane, 2), if vcc_set { set } else { clear });
            assert_eq!(at(absolute_lane, 3), if scalar_set { set } else { clear });
            assert_eq!(at(absolute_lane, 0), if vcc_set { set } else { clear });
            let aliased_clear = if vcc_set { set } else { clear };
            assert_eq!(
                at(absolute_lane, 1),
                if scalar_set { set } else { aliased_clear }
            );
        }
        assert_eq!(at(5, 0), 0x1111_0005);
        assert_eq!(at(5, 1), 0x2222_0005);
        assert_eq!(at(5, 2), 0x5151_5151);
        assert_eq!(at(5, 3), 0x5151_5151);
    }

    #[test]
    fn base_compare_kinds_match_cpu_masks_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = compare_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 96).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let pairs = [
            (f32::NAN.to_bits(), 1.0f32.to_bits()),
            ((-0.0f32).to_bits(), 0.0f32.to_bits()),
            (0x8000_0000, 0x7fff_ffff),
            (1.0f32.to_bits(), 2.0f32.to_bits()),
            (0x1234_3c00, 0x1234_3c00),
            (0x4321_bc00, 0x4321_3c00),
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 7), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                let (lhs, rhs) = pairs[lane % pairs.len()];
                set_compact_input(&mut state, base, lhs);
                set_compact_input(&mut state, base + 1, rhs);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let half = |bits: u16| {
            let sign = (u32::from(bits & 0x8000)) << 16;
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = u32::from(bits & 0x03ff);
            let out = match exponent {
                0 if mantissa == 0 => sign,
                0 => {
                    let half_mantissa = bits & 0x03ff;
                    let leading = 15 - half_mantissa.leading_zeros() as i32;
                    let normalized = u32::from(half_mantissa) << (10 - leading);
                    let normalized_exponent = (127 - 14 - 10 + leading) as u32;
                    sign | (normalized_exponent << 23) | ((normalized & 0x03ff) << 13)
                }
                0x1f => sign | 0x7f80_0000 | (mantissa << 13),
                _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
            };
            f32::from_bits(out)
        };
        let kinds = [
            CompareKind::Eq,
            CompareKind::NotEqual,
            CompareKind::Greater,
            CompareKind::GreaterEqual,
            CompareKind::Less,
            CompareKind::LessEqual,
            CompareKind::SignedGreater,
            CompareKind::SignedLess,
            CompareKind::EqualFloat,
            CompareKind::OrderedNotEqualFloat,
            CompareKind::UnorderedNotEqualFloat,
            CompareKind::LessFloat,
            CompareKind::GreaterFloat,
            CompareKind::LessEqualFloat,
            CompareKind::GreaterEqualFloat,
            CompareKind::UnorderedGreaterFloat,
            CompareKind::UnorderedLessFloat,
            CompareKind::UnorderedLessEqualFloat,
            CompareKind::LessFloat16,
        ];
        let expected = |kind: CompareKind, lhs: u32, rhs: u32| {
            let (a, b) = (f32::from_bits(lhs), f32::from_bits(rhs));
            match kind {
                CompareKind::Eq => lhs == rhs,
                CompareKind::NotEqual => lhs != rhs,
                CompareKind::Greater => lhs > rhs,
                CompareKind::GreaterEqual => lhs >= rhs,
                CompareKind::Less => lhs < rhs,
                CompareKind::LessEqual => lhs <= rhs,
                CompareKind::SignedGreater => (lhs as i32) > (rhs as i32),
                CompareKind::SignedLess => (lhs as i32) < (rhs as i32),
                CompareKind::EqualFloat => a == b,
                CompareKind::OrderedNotEqualFloat => !a.is_nan() && !b.is_nan() && a != b,
                CompareKind::UnorderedNotEqualFloat => a.is_nan() || b.is_nan() || a != b,
                CompareKind::LessFloat => a < b,
                CompareKind::GreaterFloat => a > b,
                CompareKind::LessEqualFloat => a <= b,
                CompareKind::GreaterEqualFloat => a >= b,
                CompareKind::UnorderedGreaterFloat => a.is_nan() || b.is_nan() || a > b,
                CompareKind::UnorderedLessFloat => a.is_nan() || b.is_nan() || a < b,
                CompareKind::UnorderedLessEqualFloat => a.is_nan() || b.is_nan() || a <= b,
                CompareKind::LessFloat16 => half(lhs as u16) < half(rhs as u16),
            }
        };
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 4, 5, 31, 32, 63, 64, 95, 96, 127] {
            let lane = absolute_lane & 63;
            let (lhs, rhs) = pairs[lane % pairs.len()];
            for (index, kind) in kinds.into_iter().enumerate() {
                assert_eq!(
                    at(absolute_lane, index + 2),
                    u32::from(expected(kind, lhs, rhs))
                );
            }
            assert_eq!(at(absolute_lane, 21), u32::from(lhs == rhs));
            assert_eq!(
                at(absolute_lane, 22),
                if lhs > rhs { 0xfeed_beef } else { 0x5151_5151 }
            );
        }
        for register in 2..=22 {
            assert_eq!(at(7, register), 0x5151_5151);
        }
    }

    #[test]
    fn specialized_compares_match_cpu_classes_pairs_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = specialized_compare_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 96).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let classes = [
            0x7f80_0001u32,
            0x7fc0_0000,
            0xff80_0000,
            0xbf80_0000,
            0x8000_0001,
            0x8000_0000,
            0,
            1,
            0x3f80_0000,
            0x7f80_0000,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 7), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                let equal = lane & 1 == 0;
                set_compact_input(&mut state, base, lane as u32);
                set_compact_input(&mut state, base + 1, 0x1234_5678);
                set_compact_input(
                    &mut state,
                    base + 2,
                    if equal { lane as u32 } else { lane as u32 + 1 },
                );
                set_compact_input(&mut state, base + 3, 0x1234_5678);
                let class = lane % classes.len();
                set_compact_input(&mut state, base + 4, classes[class]);
                set_compact_input(&mut state, base + 5, 0.0f32.to_bits());
                set_compact_input(&mut state, base + 6, 1u32 << class);
                for register in 7..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let kinds = [
            CompareKind::EqualFloat,
            CompareKind::OrderedNotEqualFloat,
            CompareKind::UnorderedNotEqualFloat,
            CompareKind::LessFloat,
            CompareKind::GreaterFloat,
            CompareKind::LessEqualFloat,
            CompareKind::GreaterEqualFloat,
            CompareKind::UnorderedGreaterFloat,
            CompareKind::UnorderedLessFloat,
            CompareKind::UnorderedLessEqualFloat,
        ];
        let predicate = |kind: CompareKind, a: f32, b: f32| match kind {
            CompareKind::EqualFloat => a == b,
            CompareKind::OrderedNotEqualFloat => !a.is_nan() && !b.is_nan() && a != b,
            CompareKind::UnorderedNotEqualFloat => a.is_nan() || b.is_nan() || a != b,
            CompareKind::LessFloat => a < b,
            CompareKind::GreaterFloat => a > b,
            CompareKind::LessEqualFloat => a <= b,
            CompareKind::GreaterEqualFloat => a >= b,
            CompareKind::UnorderedGreaterFloat => a.is_nan() || b.is_nan() || a > b,
            CompareKind::UnorderedLessFloat => a.is_nan() || b.is_nan() || a < b,
            CompareKind::UnorderedLessEqualFloat => a.is_nan() || b.is_nan() || a <= b,
            _ => unreachable!(),
        };
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 5, 6, 8, 9, 31, 32, 63, 64, 95, 96, 127] {
            let lane = absolute_lane & 63;
            assert_eq!(at(absolute_lane, 7), u32::from(lane & 1 == 0));
            assert_eq!(at(absolute_lane, 8), u32::from(lane & 1 == 0));
            assert_eq!(at(absolute_lane, 9), 1);
            assert_eq!(at(absolute_lane, 22), 1);
            let value = f32::from_bits(classes[lane % classes.len()]);
            for (index, kind) in kinds.into_iter().enumerate() {
                assert_eq!(
                    at(absolute_lane, 10 + index),
                    u32::from(predicate(kind, value, 0.0))
                );
            }
            assert_eq!(at(absolute_lane, 20), u32::from(value.abs() > -0.0f32));
            assert_eq!(
                at(absolute_lane, 21),
                if value < 0.0 {
                    0xfeed_beef
                } else {
                    0x5151_5151
                }
            );
        }
        for register in 7..=22 {
            assert_eq!(at(7, register), 0x5151_5151);
        }
    }

    #[test]
    fn ternary_family_matches_cpu_edges_aliasing_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = ternary_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let values = [0xffff_ffffu32, 0x8000_0001, 0x1234_5678, 0, 1];
        let counts = [0u32, 1, 31, 32, 63];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, values[lane % values.len()]);
                set_compact_input(&mut state, base + 1, counts[lane % counts.len()]);
                set_compact_input(&mut state, base + 2, counts[(lane + 2) % counts.len()]);
                for register in 3..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let bfe = |value: u32, offset: u32, width: u32| {
            let (offset, width) = (offset & 31, width & 31);
            if width == 0 {
                0
            } else {
                (value >> offset) & (u32::MAX >> ((32 - width) & 31))
            }
        };
        let expected = |kind: VectorTernaryKind, a: u32, b: u32, c: u32| match kind {
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
        let kinds = [
            VectorTernaryKind::Add3,
            VectorTernaryKind::BfeU32,
            VectorTernaryKind::LshlAdd,
            VectorTernaryKind::AddLshl,
            VectorTernaryKind::AndOr,
            VectorTernaryKind::Or3,
            VectorTernaryKind::LshlOr,
            VectorTernaryKind::BitFieldInsert,
            VectorTernaryKind::XorAdd,
        ];
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 31, 32, 63, 64, 95, 96, 127] {
            let lane = absolute_lane & 63;
            let (a, b, c) = (
                values[lane % values.len()],
                counts[lane % counts.len()],
                counts[(lane + 2) % counts.len()],
            );
            for (index, kind) in kinds.into_iter().enumerate() {
                assert_eq!(at(absolute_lane, index + 3), expected(kind, a, b, c));
            }
            assert_eq!(
                at(absolute_lane, 0),
                expected(VectorTernaryKind::XorAdd, a, b, c)
            );
        }
        assert_eq!(at(4, 0), values[4]);
        assert_eq!(at(4, 1), counts[4]);
        assert_eq!(at(4, 2), counts[1]);
        for register in 3..=11 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn float_math_family_matches_mature_domains_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_math_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let inputs = [
            0.0f32,
            -0.0,
            f32::from_bits(1),
            1.0,
            4.0,
            -1.0,
            f32::INFINITY,
            f32::NAN,
            257.0,
            -257.0,
            0.25,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 6), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, inputs[lane % inputs.len()].to_bits());
                for register in 1..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at =
            |lane: usize, register: usize| f32::from_bits(result.raw[128 + lane * 256 + register]);
        let close = |actual: f32, expected: f32| {
            if expected.is_nan() {
                actual.is_nan()
            } else if expected.is_infinite() {
                actual == expected
            } else {
                (actual - expected).abs() <= 2.0e-5 * expected.abs().max(1.0)
            }
        };
        for absolute_lane in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 10, 31, 32, 63, 64, 127] {
            let input = inputs[(absolute_lane & 63) % inputs.len()];
            let flushed = if input.is_subnormal() {
                f32::from_bits(input.to_bits() & 0x8000_0000)
            } else {
                input
            };
            assert!(close(at(absolute_lane, 1), 1.0 / flushed));
            assert!(close(at(absolute_lane, 2), 1.0 / flushed));
            assert!(close(at(absolute_lane, 3), 1.0 / flushed.sqrt()));
            assert!(close(at(absolute_lane, 4), flushed.sqrt()));
            let sin_expected = if input.is_finite() && input.abs() > 256.0 {
                0.0
            } else {
                (input * std::f32::consts::TAU).sin()
            };
            let cos_expected = if input.is_finite() && input.abs() > 256.0 {
                1.0
            } else {
                (input * std::f32::consts::TAU).cos()
            };
            assert!(close(at(absolute_lane, 5), sin_expected));
            assert!(close(at(absolute_lane, 6), cos_expected));
            assert!(close(at(absolute_lane, 7), input.exp2()));
            assert!(close(at(absolute_lane, 8), input - input.floor()));
            assert!(close(at(absolute_lane, 9), input.log2()));
            assert_eq!(at(absolute_lane, 10).to_bits(), 0.0f32.to_bits());
            assert!(close(at(absolute_lane, 0), flushed.sqrt()));
        }
        for register in 0..=10 {
            assert_eq!(
                result.raw[128 + 6 * 256 + register],
                if register == 0 {
                    f32::INFINITY.to_bits()
                } else {
                    0x5151_5151
                }
            );
        }
    }

    #[test]
    fn float_convert_family_matches_cpu_edges_aliasing_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_convert_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let inputs = [
            0.0f32.to_bits(),
            (-0.0f32).to_bits(),
            1.0f32.to_bits(),
            (-2.0f32).to_bits(),
            65504.0f32.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NAN.to_bits(),
            1u32,
            4294967296.0f32.to_bits(),
            (-2147483648.0f32).to_bits(),
            2147483648.0f32.to_bits(),
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 6), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, inputs[lane % inputs.len()]);
                for register in 1..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let half_to_f32 = |value: u16| {
            let sign = (u32::from(value & 0x8000)) << 16;
            let exponent = (value >> 10) & 0x1f;
            let mantissa = u32::from(value & 0x03ff);
            let bits = match exponent {
                0 if mantissa == 0 => sign,
                0 => {
                    let half_mantissa = value & 0x03ff;
                    let leading = 15 - half_mantissa.leading_zeros() as i32;
                    let normalized = u32::from(half_mantissa) << (10 - leading);
                    let normalized_exponent = (127 - 14 - 10 + leading) as u32;
                    sign | (normalized_exponent << 23) | ((normalized & 0x03ff) << 13)
                }
                0x1f => sign | 0x7f80_0000 | (mantissa << 13),
                _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
            };
            bits
        };
        assert_eq!(half_to_f32(0x0001), (2.0f32.powi(-24)).to_bits());
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 10, 31, 32, 63, 64, 127] {
            let bits = inputs[(absolute_lane & 63) % inputs.len()];
            let value = f32::from_bits(bits);
            let half = super::super::mesh_vector::f32_to_f16(value) as u32;
            assert_eq!(at(absolute_lane, 1), 0x5151_0000 | half);
            assert_eq!(at(absolute_lane, 2), 0x0000_5151 | (half << 16));
            assert_eq!(at(absolute_lane, 3), half << 16);
            assert_eq!(at(absolute_lane, 4), half_to_f32(bits as u16));
            assert_eq!(at(absolute_lane, 5), half_to_f32((bits >> 16) as u16));
            assert_eq!(at(absolute_lane, 6), value as u32);
            assert_eq!(at(absolute_lane, 7), (value as i32) as u32);
            assert_eq!(at(absolute_lane, 0), (bits & 0xffff_0000) | half);
        }
        assert_eq!(at(7, 4), (2.0f32.powi(-24)).to_bits());
        assert_eq!(at(6, 0), inputs[6]);
        for register in 1..=7 {
            assert_eq!(at(6, register), 0x5151_5151);
        }
    }

    #[test]
    fn lane_remainder_matches_cpu_wave64_visibility_and_masks() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = lane_remainder_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let execs = [
            [u32::MAX & !(1 << 5) & !(1 << 16) & !7, u32::MAX],
            [0, 1 << 3],
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            set_compact_input(&mut state, wb + sgprs, execs[wave][0]);
            set_compact_input(&mut state, wb + sgprs + 1, execs[wave][1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(
                    &mut state,
                    base,
                    0x1000_0000 | ((wave as u32) << 16) | lane as u32,
                );
                for register in 1..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        assert_eq!(at(3, 1), 0x1000_003f);
        assert_eq!(at(35 + 64, 1), 0x1001_003f);
        assert_eq!(at(5, 3), 0xdead_beef);
        assert_eq!(at(4, 3), 0x5151_5151);
        for lane in [3usize, 31, 32, 63] {
            assert_eq!(at(lane, 4), 10 + lane.min(32) as u32);
            assert_eq!(at(lane, 5), 20 + lane.saturating_sub(32) as u32);
        }
        assert_eq!(at(3, 6), 0x1000_0010);
        assert_eq!(at(3, 7), 0);
        assert_eq!(at(8, 6), 0x1000_0018);
        assert_eq!(at(8, 7), 0x1000_0018);
        assert_eq!(at(3, 0), 0x1000_0010);
        assert_eq!(at(8, 0), 0x1000_0018);
        assert_eq!(at(16, 6), 0x5151_5151);
        assert_eq!(at(0, 1), 0x5151_5151);
    }

    #[test]
    fn fma_mix_modes_match_cpu_fused_edges_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = fma_mix_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let fusion = (
            f32::from_bits(0x3f80_0001),
            f32::from_bits(0x3f80_0001),
            f32::from_bits(0xbf80_0002),
        );
        assert_ne!(
            fusion.0.mul_add(fusion.1, fusion.2).to_bits(),
            (fusion.0 * fusion.1 + fusion.2).to_bits()
        );
        let inputs = [
            fusion,
            (0.0, f32::INFINITY, 1.0),
            (-0.0, 2.0, -0.0),
            (f32::from_bits(1), 1.0, 0.0),
            (2.0, 3.0, -1.0),
            (f32::NAN, 1.0, 2.0),
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                let (a, b, c) = inputs[lane % inputs.len()];
                set_compact_input(&mut state, base, a.to_bits());
                set_compact_input(&mut state, base + 1, b.to_bits());
                set_compact_input(&mut state, base + 2, c.to_bits());
                for register in 3..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 5, 31, 32, 63, 64, 127] {
            let (a, b, c) = inputs[(absolute_lane & 63) % inputs.len()];
            let fused = a.mul_add(b, c);
            let packed = super::super::mesh_vector::f32_to_f16(fused) as u32;
            if fused.is_nan() {
                assert!(f32::from_bits(at(absolute_lane, 3)).is_nan());
            } else {
                assert_eq!(at(absolute_lane, 3), fused.to_bits());
            }
            assert_eq!(at(absolute_lane, 4), 0x5151_0000 | packed);
            assert_eq!(at(absolute_lane, 5), 0x0000_5151 | (packed << 16));
            assert_eq!(at(absolute_lane, 6), 0.0f32.to_bits());
            if fused.is_nan() {
                assert!(f32::from_bits(at(absolute_lane, 0)).is_nan());
            } else {
                assert_eq!(at(absolute_lane, 0), fused.to_bits());
            }
        }
        assert_eq!(at(4, 0), 2.0f32.to_bits());
        for register in 3..=6 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn convert_ubyte_selectors_match_cpu_edges_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = convert_ubyte_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let inputs = [
            0x0000_0000u32,
            0xffff_ffff,
            0x0080_7f01,
            0xff00_80fe,
            0x1234_5678,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, inputs[lane % inputs.len()]);
                for register in 1..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 31, 32, 63, 64, 95, 96, 127] {
            let input = inputs[(absolute_lane & 63) % inputs.len()];
            for byte in 0..4usize {
                let selected = ((input >> (byte * 8)) & 0xff) as f32;
                assert_eq!(at(absolute_lane, byte + 1), selected.to_bits());
            }
            let byte2 = ((input >> 16) & 0xff) as f32;
            assert_eq!(
                at(absolute_lane, 5),
                (byte2 * 4.0).clamp(0.0, 1.0).to_bits()
            );
            assert_eq!(at(absolute_lane, 0), ((input & 0xff) as f32).to_bits());
        }
        assert_eq!(at(4, 0), inputs[4]);
        for register in 1..=5 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn small_integer_bundle_matches_cpu_masks_edges_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = small_integer_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let lhs_values = [0u32, 1, 0x8000_0000, u32::MAX, 0x0001_0000];
        let rhs_values = [0u32, u32::MAX, 0x7fff_ffff, 1, 0x0000_ffff];
        let vcc = [[0xaaaa_aaaa, 0x5555_5555], [0x5555_5555, 0xaaaa_aaaa]];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            set_compact_input(&mut state, wb + sgprs + 2, vcc[wave][0]);
            set_compact_input(&mut state, wb + sgprs + 3, vcc[wave][1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, lhs_values[lane % lhs_values.len()]);
                set_compact_input(&mut state, base + 1, rhs_values[lane % rhs_values.len()]);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 31, 32, 63, 64, 95, 96, 127] {
            let wave = absolute_lane / 64;
            let lane = absolute_lane & 63;
            let lhs = lhs_values[lane % lhs_values.len()];
            let rhs = rhs_values[lane % rhs_values.len()];
            let carry_in = (vcc[wave][lane >> 5] >> (lane & 31)) & 1;
            let sum = u64::from(lhs) + u64::from(rhs) + u64::from(carry_in);
            let carry_out = (sum >> 32) != 0;
            let difference = u64::from(rhs)
                .wrapping_sub(u64::from(lhs))
                .wrapping_sub(u64::from(carry_out));
            let borrow = (difference >> 32) != 0;
            assert_eq!(at(absolute_lane, 2), sum as u32);
            assert_eq!(at(absolute_lane, 3), u32::from(carry_out));
            assert_eq!(at(absolute_lane, 4), difference as u32);
            assert_eq!(at(absolute_lane, 5), u32::from(borrow));
            assert_eq!(at(absolute_lane, 6), lhs.reverse_bits());
            assert_eq!(at(absolute_lane, 7), !lhs);
            assert_eq!(
                at(absolute_lane, 8),
                if lhs == 0 {
                    u32::MAX
                } else {
                    lhs.leading_zeros()
                }
            );
            assert_eq!(
                at(absolute_lane, 9),
                if lhs == 0 {
                    u32::MAX
                } else {
                    lhs.trailing_zeros()
                }
            );
            assert_eq!(at(absolute_lane, 10), 0xabcdef01);
            let exec_sum = u64::from(lhs) + u64::from(rhs) + 1;
            assert_eq!(at(absolute_lane, 11), exec_sum as u32);
            assert_eq!(at(absolute_lane, 12), u32::from((exec_sum >> 32) != 0));
            let exec_difference = u64::from(rhs).wrapping_sub(u64::from(lhs)).wrapping_sub(1);
            assert_eq!(at(absolute_lane, 13), exec_difference as u32);
            assert_eq!(
                at(absolute_lane, 14),
                u32::from((exec_difference >> 32) != 0)
            );
            assert_eq!(at(absolute_lane, 0), !lhs);
        }
        assert_eq!(at(4, 0), lhs_values[4]);
        for register in 2..=14 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn f32_integer_converts_match_cpu_rounding_sdwa_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = f32_integer_convert_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let inputs = [
            0u32,
            1,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0x1234_ffff,
            0xffff_1234,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, inputs[lane % inputs.len()]);
                for register in 1..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 5, 6, 7, 8, 9, 31, 32, 63, 64, 127] {
            let input = inputs[(absolute_lane & 63) % inputs.len()];
            let unsigned = (input as f32).to_bits();
            assert_eq!(at(absolute_lane, 1), unsigned);
            assert_eq!(
                at(absolute_lane, 2),
                ((input as f32) * 2.0).clamp(0.0, 1.0).to_bits()
            );
            assert_eq!(at(absolute_lane, 3), ((input as i32) as f32).to_bits());
            assert_eq!(at(absolute_lane, 4), ((input & 0xffff) as f32).to_bits());
            assert_eq!(at(absolute_lane, 5), ((input >> 16) as f32).to_bits());
            assert_eq!(at(absolute_lane, 0), unsigned);
        }
        assert_eq!(at(4, 0), inputs[4]);
        for register in 1..=5 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn pack_f16_selectors_preserve_half_payloads_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = pack_f16_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let lows = [
            0x8000_0001u32,
            0x7c00_7e01,
            0xfc00_7d01,
            0x3c00_0400,
            0x0000_8000,
        ];
        let highs = [
            0x0001_8000u32,
            0x7e01_7c00,
            0x7d01_fc00,
            0x0400_3c00,
            0x8000_0000,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, lows[lane % lows.len()]);
                set_compact_input(&mut state, base + 1, highs[lane % highs.len()]);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        let packed = |lo: u32, hi: u32, selector: u8| {
            let lo = if selector & 1 != 0 {
                lo >> 16
            } else {
                lo & 0xffff
            };
            let hi = if selector & 2 != 0 {
                hi >> 16
            } else {
                hi & 0xffff
            };
            lo | (hi << 16)
        };
        for absolute_lane in [0usize, 1, 2, 3, 31, 32, 63, 64, 95, 96, 127] {
            let lane = absolute_lane & 63;
            let (lo, hi) = (lows[lane % lows.len()], highs[lane % highs.len()]);
            for selector in 0u8..4 {
                assert_eq!(
                    at(absolute_lane, usize::from(selector) + 2),
                    packed(lo, hi, selector)
                );
            }
            assert_eq!(at(absolute_lane, 0), packed(lo, hi, 0));
        }
        assert_eq!(at(4, 0), lows[4]);
        for register in 2..=5 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn float_remainder_matches_cpu_ordering_edges_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = float_remainder_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let inputs = [
            (1.5f32, 2.0, -1.0),
            (-1.5, -2.0, f32::INFINITY),
            (f32::NAN, 1.0, 2.0),
            (1.0, f32::NAN, 2.0),
            (-0.0, 0.0, -0.0),
            (f32::from_bits(1), f32::from_bits(0x8000_0001), 0.0),
            (f32::INFINITY, f32::NEG_INFINITY, 3.0),
            (2.5, 1.5, 3.5),
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                let (a, b, c) = inputs[lane % inputs.len()];
                set_compact_input(&mut state, base, a.to_bits());
                set_compact_input(&mut state, base + 1, b.to_bits());
                set_compact_input(&mut state, base + 2, c.to_bits());
                for register in 3..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let choose_min = |lhs: f32, rhs: f32| {
            if lhs.is_nan() {
                rhs
            } else if rhs.is_nan() || lhs <= rhs {
                lhs
            } else {
                rhs
            }
        };
        let choose_max = |lhs: f32, rhs: f32| {
            if lhs.is_nan() {
                rhs
            } else if rhs.is_nan() || lhs >= rhs {
                lhs
            } else {
                rhs
            }
        };
        let median = |a, b, c| choose_max(choose_min(a, b), choose_min(choose_max(a, b), c));
        let same = |actual: u32, expected: f32| {
            if expected.is_nan() {
                f32::from_bits(actual).is_nan()
            } else {
                actual == expected.to_bits()
            }
        };
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute_lane in [0usize, 1, 2, 3, 5, 6, 7, 31, 32, 63, 64, 127] {
            let (a, b, c) = inputs[(absolute_lane & 63) % inputs.len()];
            assert!(same(at(absolute_lane, 3), a.trunc()));
            assert!(same(at(absolute_lane, 4), a.floor()));
            assert!(same(at(absolute_lane, 5), a.ceil()));
            assert!(same(at(absolute_lane, 6), a.round_ties_even()));
            assert!(same(at(absolute_lane, 7), choose_min(choose_min(a, b), c)));
            assert!(same(at(absolute_lane, 8), choose_max(choose_max(a, b), c)));
            assert!(same(at(absolute_lane, 9), median(a, b, c)));
            let modified = (median(a.abs(), -b, c) * 2.0).clamp(0.0, 1.0);
            assert!(same(at(absolute_lane, 10), modified));
            assert!(same(at(absolute_lane, 0), a.round_ties_even()));
        }
        assert_eq!(at(4, 0), (-0.0f32).to_bits());
        for register in 3..=10 {
            assert_eq!(at(4, register), 0x5151_5151);
        }
    }

    #[test]
    fn integer_binary_family_matches_cpu_edges_aliasing_and_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = binary_program();
        let shell = validation_epoch_hlsl(None, &program, 1, 64).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !(1 << 4), u32::MAX]
            } else {
                [u32::MAX; 2]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            set_compact_input(&mut state, wb + sgprs + 2, if wave == 0 { 0 } else { 1 });
            set_compact_input(&mut state, wb + sgprs + 3, 0);
            set_compact_input(&mut state, wb + sgprs + 4, if wave == 0 { 2 } else { 0 });
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                let shift = [0u32, 31, 32, 63][lane & 3];
                set_compact_input(&mut state, base, shift);
                set_compact_input(&mut state, base + 1, 0x8000_0001);
                for register in 2..vgprs {
                    set_compact_input(&mut state, base + register, 0x5151_5151);
                }
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let expected = |kind: VectorBinaryKind, lhs: u32, rhs: u32| match kind {
            VectorBinaryKind::AddNoCarry => lhs.wrapping_add(rhs),
            VectorBinaryKind::And => lhs & rhs,
            VectorBinaryKind::Or => lhs | rhs,
            VectorBinaryKind::Xor => lhs ^ rhs,
            VectorBinaryKind::BitCount => lhs.count_ones().wrapping_add(rhs),
            VectorBinaryKind::LshlReverse => rhs << (lhs & 31),
            VectorBinaryKind::ArithmeticShiftRightReverse => ((rhs as i32) >> (lhs & 31)) as u32,
            VectorBinaryKind::MinU32 => lhs.min(rhs),
            VectorBinaryKind::MaxU32 => lhs.max(rhs),
            VectorBinaryKind::MulLoU32 => lhs.wrapping_mul(rhs),
            VectorBinaryKind::MulHiU32 => ((u64::from(lhs) * u64::from(rhs)) >> 32) as u32,
            VectorBinaryKind::SubNoCarry => lhs.wrapping_sub(rhs),
            VectorBinaryKind::ReverseSubNoCarry => rhs.wrapping_sub(lhs),
        };
        let kinds = [
            VectorBinaryKind::AddNoCarry,
            VectorBinaryKind::And,
            VectorBinaryKind::Or,
            VectorBinaryKind::Xor,
            VectorBinaryKind::BitCount,
            VectorBinaryKind::LshlReverse,
            VectorBinaryKind::ArithmeticShiftRightReverse,
            VectorBinaryKind::MinU32,
            VectorBinaryKind::MaxU32,
            VectorBinaryKind::MulLoU32,
            VectorBinaryKind::MulHiU32,
            VectorBinaryKind::SubNoCarry,
            VectorBinaryKind::ReverseSubNoCarry,
        ];
        for lane in [0usize, 1, 2, 3, 31, 32, 63, 64, 65] {
            let lhs = [0u32, 31, 32, 63][lane & 3];
            for (index, kind) in kinds.into_iter().enumerate() {
                assert_eq!(
                    result.raw[128 + lane * 256 + index + 2],
                    expected(kind, lhs, 0x8000_0001)
                );
            }
            assert_eq!(result.raw[128 + lane * 256], lhs.wrapping_add(0x8000_0001));
        }
        for register in 0..vgprs {
            assert_eq!(
                result.raw[128 + 4 * 256 + register],
                if register == 0 {
                    0
                } else if register == 1 {
                    0x8000_0001
                } else {
                    0x5151_5151
                }
            );
        }
        assert_eq!(result.raw[128 + 15], 1);
        assert_eq!(result.raw[128 + 16], 2u32.wrapping_add(4));
        assert_eq!(result.raw[128 + 64 * 256 + 15], 0);
        assert_eq!(result.raw[128 + 64 * 256 + 16], 4);
    }

    #[test]
    fn transformed_binary_and_scalar_compare_sources_match_wave64() {
        if super::super::dxc::location().is_err() { return; }
        let reverse = [7, 6, 5, 4, 3, 2, 1, 0];
        let program = program(vec![
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::AddNoCarry, dst: 61,
                lhs: VectorSource::Sdwa { register: 61, scalar: false, select: 0, sign_extend: true }, rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::AddNoCarry, dst: 60,
                lhs: VectorSource::Dpp16RowMirror { register: 61, fetch_inactive: false }, rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::CompareScalar { kind: CompareKind::Eq,
                lhs: VectorSource::Dpp8 { register: 61, lanes: reverse, fetch_inactive: true }, rhs: VectorSource::Constant(7), dst: 0 }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::AddNoCarry, dst: 58, lhs: VectorSource::Scalar(0), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::AddNoCarry, dst: 59, lhs: VectorSource::Scalar(1), rhs: VectorSource::Constant(0) }),
        ]);
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        assert!(shell.hlsl.contains("xcl_vgpr_address(wave,((lane&~15u)"));
        assert!(shell.hlsl.contains("xcl_vgpr_address(wave,((lane&~7u)"));
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) { return; }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let low_exec = if wave == 0 { u32::MAX & !(1 << 8) & !(1 << 15) } else { u32::MAX };
            set_compact_input(&mut state, wb + sgprs, low_exec);
            set_compact_input(&mut state, wb + sgprs + 1, u32::MAX);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base + 58, 0xaaaa_aaaa);
                set_compact_input(&mut state, base + 59, 0xbbbb_bbbb);
                set_compact_input(&mut state, base + 60, 0xdead_beef);
                set_compact_input(&mut state, base + 61, 0x1200_0000 | lane as u32);
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for absolute in [0usize, 7, 8, 15, 16, 31, 32, 63, 64, 71, 72, 79] {
            let lane = absolute & 63;
            let wave = absolute >> 6;
            let active = wave != 0 || !matches!(lane, 8 | 15);
            assert_eq!(at(absolute, 61), if active { lane as u32 } else { 0x1200_0000 | lane as u32 });
            let source_lane = (lane & !15) | (15 - (lane & 15));
            let source_active = wave != 0 || !matches!(source_lane, 8 | 15);
            let expected = if !active { 0xdead_beef } else if source_active { source_lane as u32 } else { 0 };
            assert_eq!(at(absolute, 60), expected);
        }
        assert_eq!([result.raw[1], result.raw[5]], [1, 1]);
        assert_eq!([result.raw[2], result.raw[6]], [5, 5]);
        assert_eq!([result.raw[8], result.raw[9]], [1, 1]);
        for lane in 0..64usize {
            assert_eq!(at(lane, 58), 1, "wave 0 SGPR0 ballot at lane {lane}");
            assert_eq!(at(lane, 59), 0, "wave 0 SGPR1 ballot at lane {lane}");
            assert_eq!(at(64 + lane, 58), 1, "wave 1 SGPR0 ballot at lane {lane}");
            assert_eq!(at(64 + lane, 59), 0, "wave 1 SGPR1 ballot at lane {lane}");
        }
    }

    #[test]
    fn transformed_vector_compare_updates_vcc_and_cmpx_exec() {
        if super::super::dxc::location().is_err() { return; }
        let program = program(vec![
            ProgramOp::Vector(StageBOp::Compare { kind: CompareKind::Eq,
                lhs: VectorSource::Sdwa { register: 61, scalar: false, select: 0, sign_extend: true },
                rhs: VectorSource::Constant(0xffff_ff80), cmpx: false }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 56, lhs: VectorSource::Scalar(106), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 57, lhs: VectorSource::Scalar(107), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Compare { kind: CompareKind::Eq,
                lhs: VectorSource::Dpp8 { register: 61, lanes: [7, 6, 5, 4, 3, 2, 1, 0], fetch_inactive: false },
                rhs: VectorSource::Constant(7), cmpx: false }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 58, lhs: VectorSource::Scalar(106), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 59, lhs: VectorSource::Scalar(107), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Compare { kind: CompareKind::Eq,
                lhs: VectorSource::Dpp16RowMirror { register: 61, fetch_inactive: true },
                rhs: VectorSource::Constant(15), cmpx: true }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 60, lhs: VectorSource::Scalar(126), rhs: VectorSource::Constant(0) }),
            ProgramOp::Vector(StageBOp::Binary { kind: VectorBinaryKind::Or, dst: 61, lhs: VectorSource::Scalar(127), rhs: VectorSource::Constant(0) }),
        ]);
        let shell = validation_epoch_hlsl(None, &program, 1, 32).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) { return; }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let low_exec = if wave == 0 { u32::MAX & !(1 << 7) & !(1 << 15) } else { u32::MAX };
            set_compact_input(&mut state, wb + sgprs, low_exec);
            set_compact_input(&mut state, wb + sgprs + 1, u32::MAX);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                for register in 56..=60 { set_compact_input(&mut state, base + register, 0xa5a5_0000 | register as u32); }
                set_compact_input(&mut state, base + 61, if lane == 0 { 0x80 } else { lane as u32 });
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        assert_eq!([result.raw[1], result.raw[5]], [1, 1]);
        assert_eq!([result.raw[2], result.raw[6]], [9, 9]);
        for lane in 0..64usize {
            let wave0_active = !matches!(lane, 7 | 15);
            assert_eq!([at(lane, 56), at(lane, 57)], if wave0_active { [1, 0] } else { [0xa5a5_0038, 0xa5a5_0039] });
            assert_eq!([at(64 + lane, 56), at(64 + lane, 57)], [1, 0]);
            assert_eq!([at(lane, 58), at(lane, 59)], if wave0_active { [0, 0] } else { [0xa5a5_003a, 0xa5a5_003b] });
            assert_eq!([at(64 + lane, 58), at(64 + lane, 59)], [1, 0]);
            assert_eq!([at(lane, 60), at(lane, 61)], if lane == 0 { [1, 0] } else { [0xa5a5_003c, lane as u32] });
            assert_eq!([at(64 + lane, 60), at(64 + lane, 61)], if lane == 0 { [1, 0] } else { [0xa5a5_003c, lane as u32] });
        }
    }

    #[test]
    fn shell_compiles_and_executes_two_wave_rendezvous() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let shell = validation_epoch_hlsl(None, &rendezvous_program(), 1, 16).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let sgprs = 1usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let vgprs = 4usize;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 {
                [u32::MAX & !2, u32::MAX]
            } else {
                [u32::MAX, u32::MAX]
            };
            set_compact_input(&mut state, wb + sgprs, exec[0]);
            set_compact_input(&mut state, wb + sgprs + 1, exec[1]);
            for lane in 0..64usize {
                let gi = wave * 64 + lane;
                let base = vgpr_base + gi * vgprs;
                set_compact_input(&mut state, base, 111);
                set_compact_input(&mut state, base + 1, (gi * 4) as u32);
                set_compact_input(&mut state, base + 2, 9000 + gi as u32);
                set_compact_input(&mut state, base + 3, 777);
            }
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        let raw = &result.raw;
        let out = |gi: usize, reg: usize| raw[128 + gi * 256 + reg];
        assert_eq!((out(0, 0), out(0, 2), out(0, 3)), (5, 5, 5));
        assert_eq!((out(1, 0), out(1, 2), out(1, 3)), (111, 9001, 777));
        assert_eq!(out(2, 3), 9001);
        assert_eq!((out(64, 0), out(64, 2), out(64, 3)), (5, 5, 5));
        assert_eq!(out(127, 3), 5);
    }

    #[test]
    fn global_load_last_dword_succeeds_and_oob_preserves_destination() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion {
            guest_base: 0x1_0000_0000,
            memory_offset: 16,
            byte_len: 32752,
        };
        let shell = validation_epoch_hlsl(Some(region), &global_load_program(), 1, 16).unwrap();
        assert!(shell
            .hlsl
            .contains("RWByteAddressBuffer memory : register(u3);"));
        assert_eq!(shell.hlsl.matches("xcl_global_load(memory,").count(), 1);
        // The generic runner binds u2. This program has no DS operations, so its
        // already-sized LDS resource can also exercise the read-only helper.
        let hlsl = shell
            .hlsl
            .replacen("xcl_global_load(memory,", "xcl_global_load(lds,", 1);
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let sgprs = 1usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let vgprs = 3usize;
        let run = |address: u64| {
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb + sgprs, 1);
                set_compact_input(&mut state, wb + sgprs + 1, 0);
                for lane in 0..64usize {
                    let base = vgpr_base + (wave * 64 + lane) * vgprs;
                    set_compact_input(&mut state, base, 0x5151_5151);
                    set_compact_input(&mut state, base + 1, address as u32);
                    set_compact_input(&mut state, base + 2, (address >> 32) as u32);
                }
            }
            let mut memory = vec![0u32; 8192];
            memory[8191] = 0xa5a5_5a5a;
            super::super::runner::run(&dxil, &state, &memory).unwrap()
        };
        let valid = run(region.guest_base + u64::from(region.byte_len) - 4);
        assert_eq!(valid.raw[128], 0xa5a5_5a5a);
        assert_eq!(valid.raw[128 + 64 * 256], 0xa5a5_5a5a);
        assert_eq!(valid.raw[128 + 256], 0x5151_5151);
        let oob = run(region.guest_base + u64::from(region.byte_len));
        assert_eq!(oob.raw[128], 0x5151_5151);
        assert_eq!(oob.raw[128 + 64 * 256], 0x5151_5151);
        for invalid in [region.guest_base - 4, region.guest_base + 1] {
            let result = run(invalid);
            assert_eq!(result.raw[128], 0x5151_5151);
            assert_eq!(result.raw[128 + 64 * 256], 0x5151_5151);
        }

        let overflow_program = program(vec![ProgramOp::Memory(StageCOp::GlobalLoad {
            dst: 0,
            address: 1,
            scalar_address: None,
            offset: 1,
        })]);
        let overflow_shell = validation_epoch_hlsl(Some(region), &overflow_program, 1, 16).unwrap();
        assert!(overflow_shell
            .hlsl
            .contains("RWByteAddressBuffer memory : register(u3);"));
        assert_eq!(overflow_shell.hlsl.matches("xcl_global_load(memory,").count(), 1);
        let overflow_hlsl = overflow_shell.hlsl.replacen(
            "xcl_global_load(memory,",
            "xcl_global_load(lds,",
            1,
        );
        let overflow_dxil = super::super::dxc::compile(&overflow_hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{overflow_hlsl}"));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            set_compact_input(&mut state, wb + sgprs, 1);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, 0x5151_5151);
                set_compact_input(&mut state, base + 1, u32::MAX);
                set_compact_input(&mut state, base + 2, u32::MAX);
            }
        }
        let result = super::super::runner::run(&overflow_dxil, &state, &vec![0; 8192]).unwrap();
        assert_eq!(result.raw[128], 0x5151_5151);
        assert_eq!(result.raw[128 + 64 * 256], 0x5151_5151);

        let scalar_program = program(vec![ProgramOp::Memory(StageCOp::GlobalLoad {
            // Deliberately overwrite the low vector-address component.  The
            // helper must snapshot both components before committing dst.
            dst: 1,
            address: 1,
            scalar_address: Some(0),
            offset: 0,
        })]);
        let scalar_shell = validation_epoch_hlsl(Some(region), &scalar_program, 1, 16).unwrap();
        assert!(scalar_shell.hlsl.contains("RWByteAddressBuffer memory : register(u3);"));
        assert_eq!(scalar_shell.hlsl.matches("xcl_global_load(memory,").count(), 1);
        let scalar_hlsl = scalar_shell.hlsl.replacen(
            "xcl_global_load(memory,",
            "xcl_global_load(lds,",
            1,
        );
        let scalar_dxil = super::super::dxc::compile(&scalar_hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{scalar_hlsl}"));
        let run_scalar = |scalar: u64, vector: u64| {
            let mut state = super::super::runner::KernelState::default();
            let wave_stride = 8usize;
            let vgpr_base = 16usize;
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb, scalar as u32);
                set_compact_input(&mut state, wb + 1, (scalar >> 32) as u32);
                set_compact_input(&mut state, wb + 2, 1);
                for lane in 0..64usize {
                    let base = vgpr_base + (wave * 64 + lane) * 3;
                    set_compact_input(&mut state, base, 0x5151_5151);
                    set_compact_input(&mut state, base + 1, vector as u32);
                    set_compact_input(&mut state, base + 2, (vector >> 32) as u32);
                }
            }
            let mut memory = vec![0u32; 8192];
            memory[8191] = 0xa5a5_5a5a;
            super::super::runner::run(&scalar_dxil, &state, &memory).unwrap()
        };
        let mapped = run_scalar(region.guest_base, u64::from(region.byte_len) - 4);
        assert_eq!(
            (mapped.raw[129], mapped.raw[128 + 64 * 256 + 1]),
            (0xa5a5_5a5a, 0xa5a5_5a5a)
        );
        assert_eq!((mapped.raw[130], mapped.raw[128 + 64 * 256 + 2]), (0, 0));
        let overflow = run_scalar(u64::MAX, 1);
        assert_eq!(
            (overflow.raw[129], overflow.raw[128 + 64 * 256 + 1]),
            (1, 1)
        );
        assert_eq!((overflow.raw[130], overflow.raw[128 + 64 * 256 + 2]), (0, 0));
    }

    #[test]
    fn scalar_memory_loads_snapshot_aliases_and_commit_whole_spans() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 64 };
        let cases = [
            (ProgramOp::Memory(StageCOp::ScalarLoad {
                dst: 0,
                base: 0,
                offset: -4,
                soffset: OffsetSource::Zero,
                words: 4,
            }), 3u8, 0x1003u32),
            (ProgramOp::Memory(StageCOp::ScalarBufferLoad {
                dst: 0,
                descriptor: 0,
                offset: 0,
                soffset: OffsetSource::Zero,
                words: 16,
            }), 15u8, 0x100fu32),
        ];
        for (kind, (op, observed, expected)) in cases.into_iter().enumerate() {
            let program = program(vec![
                op,
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Register(observed),
                }),
            ]);
            let shell = validation_epoch_hlsl(Some(region), &program, 1, 8).unwrap();
            let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
            let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                continue;
            }
            let requirements = requirements(&program).unwrap();
            let sgprs = requirements.sgprs.max(1) as usize;
            let wave_stride = sgprs + 6;
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb, region.guest_base as u32 + if kind == 0 { 4 } else { 0 });
                set_compact_input(&mut state, wb + 1, (region.guest_base >> 32) as u32);
                if kind == 1 {
                    set_compact_input(&mut state, wb + 2, region.byte_len);
                    set_compact_input(&mut state, wb + 3, 0xfeed_beef);
                }
            }
            let memory = (0..16u32).map(|value| 0x1000 + value).collect::<Vec<_>>();
            let result = super::super::runner::run(&dxil, &state, &memory).unwrap();
            assert_eq!((result.raw[8], result.raw[9]), (expected, expected));
        }

        // Make only the final dword invalid, then fold every destination into
        // SGPR0. Any eager per-word commit changes the live checksum.
        let mut rejected_ops = vec![ProgramOp::Memory(StageCOp::ScalarBufferLoad {
            dst: 0,
            descriptor: 16,
            offset: 0,
            soffset: OffsetSource::Zero,
            words: 16,
        })];
        for source in 1..16u8 {
            rejected_ops.push(ProgramOp::Scalar(StageAOp::Binary {
                kind: ScalarBinaryKind::Xor,
                dst: 0,
                lhs: Operand32::Register(0),
                rhs: Operand32::Register(source),
            }));
        }
        let rejected = program(rejected_ops);
        let rejected_shell = validation_epoch_hlsl(Some(region), &rejected, 1, 32).unwrap();
        let rejected_hlsl = rejected_shell
            .hlsl
            .replace("xcl_global_load(memory,", "xcl_global_load(lds,");
        let rejected_dxil = super::super::dxc::compile(&rejected_hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{rejected_hlsl}"));
        if super::super::runner::wave64_supported().unwrap_or(false) {
            let requirements = requirements(&rejected).unwrap();
            let sgprs = requirements.sgprs.max(1) as usize;
            let wave_stride = sgprs + 6;
            let mut state = super::super::runner::KernelState::default();
            let expected = (0..16u32).fold(0u32, |checksum, value| checksum ^ (0x8000 + value));
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                for register in 0..16usize {
                    set_compact_input(&mut state, wb + register, 0x8000 + register as u32);
                }
                set_compact_input(&mut state, wb + 16, region.guest_base as u32);
                set_compact_input(&mut state, wb + 17, (region.guest_base >> 32) as u32);
                set_compact_input(&mut state, wb + 18, 60);
                set_compact_input(&mut state, wb + 19, 0);
            }
            let memory = (0..16u32).map(|value| 0x1000 + value).collect::<Vec<_>>();
            let result = super::super::runner::run(&rejected_dxil, &state, &memory).unwrap();
            assert_eq!((result.raw[8], result.raw[9]), (expected, expected));
        }
    }

    #[test]
    fn raw_vector_buffer_loads_snapshot_aliases_mask_exec_and_commit_per_lane() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 60 };
        let program = program(vec![ProgramOp::Memory(StageCOp::VectorBufferLoad {
            dst: 0,
            address: 0,
            descriptor: 0,
            offset: 0,
            soffset: OffsetSource::Zero,
            offen: true,
            idxen: true,
            words: 4,
            format: crate::gpu::recomp::mesh_memory::BufferFormatSource::Raw,
        })]);
        let shell = validation_epoch_hlsl(Some(region), &program, 1, 8).unwrap();
        let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            set_compact_input(&mut state, wb, region.guest_base as u32);
            set_compact_input(&mut state, wb + 1, (region.guest_base >> 32) as u32 | (16 << 16));
            set_compact_input(&mut state, wb + 2, 4);
            set_compact_input(&mut state, wb + 3, 0);
            set_compact_input(&mut state, wb + sgprs, 0b101);
            for lane in 0..64usize {
                let base = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, base, if lane == 2 { 3 } else { 1 });
                set_compact_input(&mut state, base + 1, 0);
                set_compact_input(&mut state, base + 2, 0xaaaa_0002);
                set_compact_input(&mut state, base + 3, 0xaaaa_0003);
            }
        }
        let memory = (0..16u32).map(|value| 0x1000 + value).collect::<Vec<_>>();
        let result = super::super::runner::run(&dxil, &state, &memory).unwrap();
        let at = |lane: usize, register: usize| result.raw[128 + lane * 256 + register];
        for lane in [0usize, 64] {
            assert_eq!((at(lane, 0), at(lane, 1), at(lane, 2), at(lane, 3)), (0x1004, 0x1005, 0x1006, 0x1007));
        }
        for lane in [1usize, 65] {
            assert_eq!((at(lane, 0), at(lane, 1), at(lane, 2), at(lane, 3)), (1, 0, 0xaaaa_0002, 0xaaaa_0003));
        }
        for lane in [2usize, 66] {
            assert_eq!((at(lane, 0), at(lane, 1), at(lane, 2), at(lane, 3)), (3, 0, 0xaaaa_0002, 0xaaaa_0003));
        }
    }

    #[test]
    fn raw_vector_buffer_stride_zero_offsets_and_overflow_are_atomic_live() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 64 };
        let program = program(vec![ProgramOp::Memory(StageCOp::VectorBufferLoad {
            dst: 0,
            address: 0,
            descriptor: 0,
            offset: 4,
            soffset: OffsetSource::Scalar(4),
            offen: true,
            idxen: false,
            words: 4,
            format: crate::gpu::recomp::mesh_memory::BufferFormatSource::Raw,
        })]);
        let shell = validation_epoch_hlsl(Some(region), &program, 1, 8).unwrap();
        let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let run = |base: u64, limit: u32, soffset: u32| {
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb, base as u32);
                set_compact_input(&mut state, wb + 1, (base >> 32) as u32);
                set_compact_input(&mut state, wb + 2, limit);
                set_compact_input(&mut state, wb + 3, 0);
                set_compact_input(&mut state, wb + 4, soffset);
                set_compact_input(&mut state, wb + sgprs, 1);
                for lane in 0..64usize {
                    let at = vgpr_base + (wave * 64 + lane) * vgprs;
                    set_compact_input(&mut state, at, 4);
                    set_compact_input(&mut state, at + 1, 0xaaaa_0001);
                    set_compact_input(&mut state, at + 2, 0xaaaa_0002);
                    set_compact_input(&mut state, at + 3, 0xaaaa_0003);
                }
            }
            let memory = (0..16u32).map(|value| 0x1000 + value).collect::<Vec<_>>();
            super::super::runner::run(&dxil, &state, &memory).unwrap()
        };
        let tuple = |result: &super::super::runner::KernelResult, lane: usize| {
            let at = |register: usize| result.raw[128 + lane * 256 + register];
            (at(0), at(1), at(2), at(3))
        };

        // CPU oracle: relative=voffset+instruction offset=8, then SOFFSET=8
        // contributes to the guest address but not to the descriptor byte limit.
        let relative = 4u64 + 4;
        assert_eq!(relative + 16, 24);
        let success = run(region.guest_base, 64, 8);
        for lane in [0usize, 64] {
            assert_eq!(tuple(&success, lane), (0x1004, 0x1005, 0x1006, 0x1007));
        }

        // Limit 23 excludes only the fourth dword of the 16-byte span.
        assert!(relative + 16 > 23 && relative + 12 <= 23);
        let descriptor_oob = run(region.guest_base, 23, 8);
        let base_overflow = run(u64::MAX - 4, 64, 8);
        for result in [&descriptor_oob, &base_overflow] {
            for lane in [0usize, 64] {
                assert_eq!(tuple(result, lane), (4, 0xaaaa_0001, 0xaaaa_0002, 0xaaaa_0003));
            }
        }
        assert!((u64::MAX - 4).checked_add(8).is_none());
    }

    #[test]
    fn formatted_vector_buffers_decode_convert_alias_and_commit_atomically_live() {
        if super::super::dxc::location().is_err() { return; }
        let descriptor_bytes = crate::gpu::rdna2::encode("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA",0),("VADDR",0),("SRSRC",0),("SOFFSET",128),("IDXEN",1)], None).unwrap();
        assert!(matches!(super::super::mesh_memory::lower_stage_c_one(&crate::gpu::rdna2::decode(&descriptor_bytes,0).unwrap()).unwrap(), StageCOp::VectorBufferLoad { words:1, format:BufferFormatSource::Descriptor, idxen:true, offen:false, .. }));
        for (name, words, dst, first_invalid) in [("BUFFER_LOAD_FORMAT_XY", 2, 254, 255), ("BUFFER_LOAD_FORMAT_XYZ", 3, 253, 254), ("BUFFER_LOAD_FORMAT_XYZW", 4, 252, 253)] {
            let bytes = crate::gpu::rdna2::encode(name, "ENC_MUBUF", &[("VDATA",dst),("VADDR",0),("SRSRC",0),("SOFFSET",128),("IDXEN",1)], None).unwrap();
            let decoded = super::super::mesh_memory::lower_stage_c_one(&crate::gpu::rdna2::decode(&bytes,0).unwrap()).unwrap();
            assert!(matches!(decoded, StageCOp::VectorBufferLoad { dst: actual, words: actual_words, format:BufferFormatSource::Descriptor, idxen:true, offen:false, offset:0, .. } if actual==dst as u16 && actual_words==words));
            assert!(super::super::mesh_slot::lower_slot(&ProgramOp::Memory(decoded), super::super::mesh_slot::WaveStateAbi::default()).is_ok());
            let invalid_bytes = crate::gpu::rdna2::encode(name, "ENC_MUBUF", &[("VDATA",first_invalid),("VADDR",0),("SRSRC",0),("SOFFSET",128),("IDXEN",1)], None).unwrap();
            let invalid = super::super::mesh_memory::lower_stage_c_one(&crate::gpu::rdna2::decode(&invalid_bytes,0).unwrap()).unwrap();
            assert!(matches!(invalid, StageCOp::VectorBufferLoad { dst: actual, words: actual_words, format:BufferFormatSource::Descriptor, idxen:true, offen:false, offset:0, .. } if actual==first_invalid as u16 && actual_words==words));
            assert!(super::super::mesh_slot::lower_slot(&ProgramOp::Memory(invalid), super::super::mesh_slot::WaveStateAbi::default()).is_err());
        }
        let instruction_bytes = crate::gpu::rdna2::encode("TBUFFER_LOAD_FORMAT_XYZW", "ENC_MTBUF", &[("VDATA",4),("VADDR",4),("SRSRC",1),("SOFFSET",128),("OFFEN",1),("OFFSET",4),("FORMAT",56)], None).unwrap();
        assert!(matches!(super::super::mesh_memory::lower_stage_c_one(&crate::gpu::rdna2::decode(&instruction_bytes,0).unwrap()).unwrap(), StageCOp::VectorBufferLoad { words:4, format:BufferFormatSource::Instruction(56), idxen:false, offen:true, .. }));

        let ops = vec![
            ProgramOp::Memory(StageCOp::VectorBufferLoad { dst:0,address:0,descriptor:0,offset:0,soffset:OffsetSource::Zero,offen:false,idxen:true,words:1,format:BufferFormatSource::Descriptor }),
            ProgramOp::Memory(StageCOp::VectorBufferLoad { dst:4,address:4,descriptor:4,offset:4,soffset:OffsetSource::Zero,offen:true,idxen:false,words:4,format:BufferFormatSource::Instruction(56) }),
        ];
        let region=ReadOnlyRegion{guest_base:0x1_0000_0000,memory_offset:0,byte_len:16};
        let p=program(ops);
        let shell=validation_epoch_hlsl(Some(region),&p,1,8).unwrap();
        let hlsl=shell.hlsl.replace("xcl_global_byte(memory,","xcl_global_byte(lds,");
        let _dxil=super::super::dxc::compile(&hlsl,"main","cs_6_6",&[]).unwrap_or_else(|e|panic!("{e}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) { return; }
        let requirements=requirements(&p).unwrap();let sgprs=requirements.sgprs.max(1) as usize;let vgprs=requirements.vgprs.max(1) as usize;
        let wave_stride=sgprs+6;let vgpr_base=2*wave_stride;
        let run=|selectors:u32| {
            let mut state=super::super::runner::KernelState::default();
            for wave in 0..2usize { let wb=wave*wave_stride;
                let descriptor=[region.guest_base as u32,((region.guest_base>>32) as u32)|(4<<16),4,60<<12,region.guest_base as u32,(region.guest_base>>32) as u32,16,selectors];
                for (r,v) in descriptor.into_iter().enumerate(){set_compact_input(&mut state,wb+r,v);}
                set_compact_input(&mut state,wb+sgprs,0b101);
                for lane in 0..64usize {let at=vgpr_base+(wave*64+lane)*vgprs;set_compact_input(&mut state,at,if lane==2{9}else{1});set_compact_input(&mut state,at+4,0);for r in 1..4{set_compact_input(&mut state,at+r,0xAA00_0000|r as u32);}for r in 5..8{set_compact_input(&mut state,at+r,0xBB00_0000|r as u32);}}
            }
            let code=super::super::dxc::compile(&hlsl,"main","cs_6_6",&[]).unwrap();
            super::super::runner::run(&code,&state,&[0x0403_0201,0x0807_0605,0x0c0b_0a09,0x100f_0e0d]).unwrap()
        };
        let success=run((1<<3)|(4<<6)|(5<<9));let at=|r:&super::super::runner::KernelResult,lane:usize,reg:usize|r.raw[128+lane*256+reg];
        for lane in [0usize,64]{assert_eq!(at(&success,lane,0),0);assert_eq!((at(&success,lane,4),at(&success,lane,5),at(&success,lane,6),at(&success,lane,7)),(0,1.0f32.to_bits(),(5.0f32/255.0).to_bits(),(6.0f32/255.0).to_bits()));}
        for lane in [1usize,65]{assert_eq!((at(&success,lane,0),at(&success,lane,1),at(&success,lane,4),at(&success,lane,7)),(1,0xAA00_0001,0,0xBB00_0007));}
        for lane in [2usize,66]{assert_eq!(at(&success,lane,0),9);}
        // Defaults 0 and 1 both succeed, then selector 2 is invalid. An eager
        // per-component implementation would have changed v5 to float one.
        let late_invalid=run((1<<3)|(2<<6)|(7<<9));
        for lane in [0usize,64]{assert_eq!((at(&late_invalid,lane,4),at(&late_invalid,lane,5),at(&late_invalid,lane,6),at(&late_invalid,lane,7)),(0,0xBB00_0005,0xBB00_0006,0xBB00_0007));}
    }

    #[test]
    fn descriptor_xy_xyz_xyzw_indexed_forms_convert_and_commit_atomically() {
        if super::super::dxc::location().is_err() { return; }
        let region=ReadOnlyRegion{guest_base:0x1_0000_0000,memory_offset:0,byte_len:16};
        let program=program(vec![
            ProgramOp::Memory(StageCOp::VectorBufferLoad { dst:0,address:0,descriptor:0,offset:0,soffset:OffsetSource::Zero,offen:false,idxen:true,words:2,format:BufferFormatSource::Descriptor }),
            ProgramOp::Memory(StageCOp::VectorBufferLoad { dst:4,address:8,descriptor:4,offset:0,soffset:OffsetSource::Zero,offen:false,idxen:true,words:4,format:BufferFormatSource::Descriptor }),
            ProgramOp::Memory(StageCOp::VectorBufferLoad { dst:10,address:9,descriptor:8,offset:0,soffset:OffsetSource::Zero,offen:false,idxen:true,words:3,format:BufferFormatSource::Descriptor }),
        ]);
        let compile=|bytes|{let shell=validation_epoch_hlsl(Some(ReadOnlyRegion{byte_len:bytes,..region}),&program,1,8).unwrap();let hlsl=shell.hlsl.replace("xcl_global_byte(memory,","xcl_global_byte(lds,");super::super::dxc::compile(&hlsl,"main","cs_6_6",&[]).unwrap_or_else(|e|panic!("{e}\n{hlsl}"))};
        let dxil=compile(16);
        if !super::super::runner::wave64_supported().unwrap_or(false){return;}
        let requirements=requirements(&program).unwrap();let sgprs=requirements.sgprs.max(1) as usize;let vgprs=requirements.vgprs.max(1) as usize;let wave_stride=sgprs+6;let vgpr_base=2*wave_stride;
        let run=|code:&[u8],late_invalid:bool|{let mut state=super::super::runner::KernelState::default();for wave in 0..2usize{let wb=wave*wave_stride;
            let xy=(56<<12)|4|(5<<3);let xyzw=(56<<12)|4|(5<<3)|((if late_invalid{2}else{0})<<6)|(1<<9);let xyz=(56<<12)|4|(5<<3)|((if late_invalid{2}else{1})<<6);
            let descriptors=[region.guest_base as u32,((region.guest_base>>32)as u32)|(4<<16),4,xy,region.guest_base as u32,((region.guest_base>>32)as u32)|(4<<16),4,xyzw,region.guest_base as u32,((region.guest_base>>32)as u32)|(4<<16),4,xyz];
            for(r,v)in descriptors.into_iter().enumerate(){set_compact_input(&mut state,wb+r,v);}set_compact_input(&mut state,wb+sgprs,0b101);
            for lane in 0..64usize{let at=vgpr_base+(wave*64+lane)*vgprs;set_compact_input(&mut state,at,1);set_compact_input(&mut state,at+1,0xaa00_0001);for r in 2..13{set_compact_input(&mut state,at+r,0xaa00_0000|r as u32);}set_compact_input(&mut state,at+8,1);set_compact_input(&mut state,at+9,1);}}
            super::super::runner::run(code,&state,&[0x0403_0201,0x0807_0605,0,0]).unwrap()};
        let at=|r:&super::super::runner::KernelResult,lane:usize,reg:usize|r.raw[128+lane*256+reg];
        let success=run(&dxil,false);for lane in[0usize,64]{assert_eq!((at(&success,lane,0),at(&success,lane,1)),((5.0f32/255.0).to_bits(),(6.0f32/255.0).to_bits()));assert_eq!((at(&success,lane,4),at(&success,lane,5),at(&success,lane,6),at(&success,lane,7)),((5.0f32/255.0).to_bits(),(6.0f32/255.0).to_bits(),0,1.0f32.to_bits()));assert_eq!((at(&success,lane,10),at(&success,lane,11),at(&success,lane,12)),((5.0f32/255.0).to_bits(),(6.0f32/255.0).to_bits(),1.0f32.to_bits()));}
        for lane in[1usize,65]{assert_eq!((at(&success,lane,0),at(&success,lane,4)),(1,0xaa00_0004));}
        let failed=run(&dxil,true);for lane in[0usize,64]{assert_eq!((at(&failed,lane,4),at(&failed,lane,5),at(&failed,lane,6),at(&failed,lane,7)),(0xaa00_0004,0xaa00_0005,0xaa00_0006,0xaa00_0007));assert_eq!((at(&failed,lane,10),at(&failed,lane,11),at(&failed,lane,12)),(0xaa00_000a,0xaa00_000b,0xaa00_000c));}
    }

    #[test]
    fn image_load_checked_region_masks_components_and_preserves_invalid_lanes() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 512 };
        let program = program(vec![ProgramOp::Memory(StageCOp::ImageLoad {
            dst: 0,
            coordinates: [0, 1, 0],
            coordinate_count: 2,
            descriptor: 0,
            dmask: 0b1010,
            d16: false,
        })]);
        let shell = validation_epoch_hlsl(Some(region), &program, 1, 8).unwrap();
        let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        let shifted = region.guest_base >> 8;
        let descriptor = [
            shifted as u32,
            ((shifted >> 32) as u32 & 0xff) | (75 << 20) | (1 << 30),
            1 << 14,
            0xfac | (9 << 28),
            0,
            0,
            0,
            0,
        ];
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            for (register, value) in descriptor.into_iter().enumerate() {
                set_compact_input(&mut state, wb + register, value);
            }
            set_compact_input(&mut state, wb + sgprs, 0b101);
            for lane in 0..64usize {
                let at = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, at, if lane == 2 { 2 } else { 1 });
                set_compact_input(&mut state, at + 1, 1);
            }
        }
        let mut memory = vec![0u32; 128];
        memory[68..72].copy_from_slice(&[0x1111, 0x2222, 0x3333, 0x4444]);
        let result = super::super::runner::run(&dxil, &state, &memory).unwrap();
        let tuple = |lane: usize| (result.raw[128 + lane * 256], result.raw[129 + lane * 256]);
        for lane in [0usize, 64] {
            assert_eq!(tuple(lane), (0x2222, 0x4444));
        }
        for lane in [1usize, 65] {
            assert_eq!(tuple(lane), (1, 1));
        }
        for lane in [2usize, 66] {
            assert_eq!(tuple(lane), (2, 1));
        }
        // Component 1 is still mapped at byte 276, but component 3 at byte
        // 284 is exactly outside this shorter region. No selected component
        // may commit when the later read fails.
        let short_region = ReadOnlyRegion { byte_len: 284, ..region };
        let short_shell = validation_epoch_hlsl(Some(short_region), &program, 1, 8).unwrap();
        let short_hlsl = short_shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
        let short_dxil = super::super::dxc::compile(&short_hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{short_hlsl}"));
        let short = super::super::runner::run(&short_dxil, &state, &memory).unwrap();
        let short_tuple = |lane: usize| (short.raw[128 + lane * 256], short.raw[129 + lane * 256]);
        for lane in [0usize, 64] {
            assert_eq!(short_tuple(lane), (1, 1));
        }
        assert_eq!((276 < 284, 284 < 284), (true, false));
        assert_eq!(68, (256 + 16) / 4);
    }

    #[test]
    fn d16_image_load_packs_halves_and_commits_atomically() {
        if super::super::dxc::location().is_err() { return; }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 64 };
        let program = program(vec![ProgramOp::Memory(StageCOp::ImageLoad {
            dst: 0, coordinates: [0, 0, 0], coordinate_count: 1,
            descriptor: 0, dmask: 0b1111, d16: true,
        })]);
        let make = |bytes| {
            let shell = validation_epoch_hlsl(Some(ReadOnlyRegion { byte_len: bytes, ..region }), &program, 1, 8).unwrap();
            assert!(shell.hlsl.contains("f32tof16(asfloat(xcl_v[xcl_i]))"));
            let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
            super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"))
        };
        let dxil = make(64);
        let short_dxil = make(28);
        if !super::super::runner::wave64_supported().unwrap_or(false) { return; }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let shifted = region.guest_base >> 8;
        let descriptor = [shifted as u32, ((shifted >> 32) as u32 & 0xff) | (77 << 20) | (1 << 30), 0, 0xfac | (8 << 28), 0, 0, 0, 0];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            for (register, value) in descriptor.into_iter().enumerate() { set_compact_input(&mut state, wb + register, value); }
            set_compact_input(&mut state, wb + sgprs, 0b101);
            for lane in 0..64usize {
                let at = vgpr_base + (wave * 64 + lane) * vgprs;
                set_compact_input(&mut state, at, if lane == 2 { 1 } else { 0 });
                set_compact_input(&mut state, at + 1, 0xfeed_beef);
            }
        }
        let mut memory = vec![0u32; 16];
        memory[4..8].copy_from_slice(&[1.0f32.to_bits(), (-0.0f32).to_bits(), f32::INFINITY.to_bits(), 0x3380_0000]);
        let result = super::super::runner::run(&dxil, &state, &memory).unwrap();
        let tuple = |r: &super::super::runner::KernelResult, lane: usize| (r.raw[128 + lane * 256], r.raw[129 + lane * 256]);
        for lane in [0usize, 64] { assert_eq!(tuple(&result, lane), (0x8000_3c00, 0x0001_7c00)); }
        for lane in [1usize, 65] { assert_eq!(tuple(&result, lane), (0, 0xfeed_beef)); }

        // Exact halfway cases demonstrate round-to-nearest-even.  NaN payload
        // choice is hardware-defined, so only its IEEE half class is asserted.
        memory[4..8].copy_from_slice(&[
            1.00048828125f32.to_bits(),
            1.00146484375f32.to_bits(),
            f32::NAN.to_bits(),
            0x3380_0000,
        ]);
        let edges = super::super::runner::run(&dxil, &state, &memory).unwrap();
        for lane in [0usize, 64] {
            let (packed_ties, packed_nan) = tuple(&edges, lane);
            assert_eq!(packed_ties, 0x3c02_3c00);
            let nan = packed_nan as u16;
            assert_eq!(nan & 0x7c00, 0x7c00);
            assert_ne!(nan & 0x03ff, 0);
            assert_eq!(packed_nan >> 16, 1);
        }

        // Integer image formats preserve raw low 16 bits; interpreting these
        // words as floats and converting them would produce different values.
        let mut integer_state = state.clone();
        let integer_d1 = ((shifted >> 32) as u32 & 0xff) | (75 << 20) | (1 << 30);
        for wave in 0..2usize { set_compact_input(&mut integer_state, wave * wave_stride + 1, integer_d1); }
        memory[4..8].copy_from_slice(&[0xdead_1234, 0xbeef_5678, 0xabcd_9abc, 0x1357_def0]);
        let integer = super::super::runner::run(&dxil, &integer_state, &memory).unwrap();
        for lane in [0usize, 64] { assert_eq!(tuple(&integer, lane), (0x5678_1234, 0xdef0_9abc)); }

        let short = super::super::runner::run(&short_dxil, &state, &memory).unwrap();
        for lane in [0usize, 64] { assert_eq!(tuple(&short, lane), (0, 0xfeed_beef)); }
        assert_eq!((16 + 12 <= 28, 16 + 16 <= 28), (true, false));
    }

    #[test]
    fn image_sample_lz_point_clamps_masks_and_commits_atomically() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 512 };
        let program = program(vec![ProgramOp::Memory(StageCOp::ImageSampleLz {
            dst: 0,
            coordinates: [0, 1, 0],
            coordinate_count: 2,
            arrayed: false,
            descriptor: 0,
            sampler: 8,
            dmask: 0b1010,
            d16: false,
            lod: None,
        })]);
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let shifted = region.guest_base >> 8;
        let descriptor = [
            shifted as u32,
            ((shifted >> 32) as u32 & 0xff) | (77 << 20) | (1 << 30),
            1 << 14,
            0xfac | (9 << 28),
            0,
            0,
            0,
            0,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            for (register, value) in descriptor.into_iter().enumerate() {
                set_compact_input(&mut state, wb + register, value);
            }
            set_compact_input(&mut state, wb + 8, 146);
            set_compact_input(&mut state, wb + sgprs, 0b1101);
            for lane in 0..64usize {
                let at = vgpr_base + (wave * 64 + lane) * vgprs;
                let (x, y) = match lane {
                    2 => (2.0f32, -1.0f32),
                    3 => (f32::NAN, 0.5f32),
                    _ => (0.75f32, 0.75f32),
                };
                set_compact_input(&mut state, at, x.to_bits());
                set_compact_input(&mut state, at + 1, y.to_bits());
            }
        }
        let mut memory = vec![0u32; 128];
        memory[4..8].copy_from_slice(&[0x0101, 0x0202, 0x0303, 0x0404]);
        memory[68..72].copy_from_slice(&[0x1111, 0x2222, 0x3333, 0x4444]);
        let run = |bytes: u32, input: &super::super::runner::KernelState| {
            let shell = validation_epoch_hlsl(Some(ReadOnlyRegion { byte_len: bytes, ..region }), &program, 1, 8).unwrap();
            let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
            let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
            super::super::runner::run(&dxil, input, &memory).unwrap()
        };
        let result = run(512, &state);
        let tuple = |result: &super::super::runner::KernelResult, lane: usize| (result.raw[128 + lane * 256], result.raw[129 + lane * 256]);
        for lane in [0usize, 64] { assert_eq!(tuple(&result, lane), (0x2222, 0x4444)); }
        for lane in [1usize, 65] { assert_eq!(tuple(&result, lane), (0.75f32.to_bits(), 0.75f32.to_bits())); }
        for lane in [2usize, 66] { assert_eq!(tuple(&result, lane), (0x0202, 0x0404)); }
        for lane in [3usize, 67] { assert!(f32::from_bits(tuple(&result, lane).0).is_nan()); assert_eq!(tuple(&result, lane).1, 0.5f32.to_bits()); }

        // Component 1 at 276 is mapped, while component 3 at 284 is not.
        // The earlier successful read must not partially update the result.
        let short = run(284, &state);
        for lane in [0usize, 64] { assert_eq!(tuple(&short, lane), (0.75f32.to_bits(), 0.75f32.to_bits())); }
        let mut filtered = state.clone();
        for wave in 0..2usize {
            set_compact_input(&mut filtered, wave * wave_stride + 8, 147);
        }
        let rejected_sampler = run(512, &filtered);
        for lane in [0usize, 64] {
            assert_eq!(tuple(&rejected_sampler, lane), (0.75f32.to_bits(), 0.75f32.to_bits()));
        }
        assert_eq!((276 < 284, 284 < 284), (true, false));
    }

    #[test]
    fn d16_image_sample_implicit_and_explicit_zero_lod_match() {
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 512 };
        let implicit = program(vec![ProgramOp::Memory(StageCOp::ImageSampleLz {
            dst: 0, coordinates: [0, 1, 0], coordinate_count: 2, arrayed: false,
            descriptor: 0, sampler: 8, dmask: 0b0111, d16: true, lod: None,
        })]);
        let explicit = program(vec![ProgramOp::Memory(StageCOp::ImageSampleLz {
            dst: 0, coordinates: [0, 1, 0], coordinate_count: 2, arrayed: false,
            descriptor: 0, sampler: 8, dmask: 0b0111, d16: true, lod: Some(2),
        })]);
        let implicit_requirements = requirements(&implicit).unwrap();
        let explicit_requirements = requirements(&explicit).unwrap();
        assert_eq!((implicit_requirements.vgprs, explicit_requirements.vgprs), (2, 3));
        let implicit_shell = validation_epoch_hlsl(Some(region), &implicit, 1, 8).unwrap();
        let explicit_shell = validation_epoch_hlsl(Some(region), &explicit, 1, 8).unwrap();
        assert!(implicit_shell.hlsl.contains("xcl_finite&&true&&"));
        assert!(!implicit_shell.hlsl.contains("&&asfloat(") );
        assert!(explicit_shell.hlsl.contains("(isfinite(asfloat("));
        assert!(explicit_shell.hlsl.contains("&&asfloat("));
        assert!(explicit_shell.hlsl.contains("==0.0f)"));
        if super::super::dxc::location().is_err() { return; }
        let compile = |program: &MeshProgram, bytes| {
            let shell = validation_epoch_hlsl(Some(ReadOnlyRegion { byte_len: bytes, ..region }), program, 1, 8).unwrap();
            let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
            super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"))
        };
        let dxil = compile(&implicit, 512);
        let explicit_dxil = compile(&explicit, 512);
        let short_dxil = compile(&implicit, 280);
        if !super::super::runner::wave64_supported().unwrap_or(false) { return; }
        let shifted = region.guest_base >> 8;
        let descriptor = [shifted as u32, ((shifted >> 32) as u32 & 0xff) | (77 << 20) | (1 << 30), 1 << 14, 0xfac | (9 << 28), 0, 0, 0, 0];
        let make_state = |requirements: ProgramRequirements, lod: Option<u32>| {
            let sgprs = requirements.sgprs.max(1) as usize;
            let vgprs = requirements.vgprs.max(1) as usize;
            let wave_stride = sgprs + 6;
            let vgpr_base = 2 * wave_stride;
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                for (register, value) in descriptor.into_iter().enumerate() { set_compact_input(&mut state, wb + register, value); }
                set_compact_input(&mut state, wb + 8, 146);
                set_compact_input(&mut state, wb + sgprs, 0b101);
                for lane in 0..64usize {
                    let at = vgpr_base + (wave * 64 + lane) * vgprs;
                    set_compact_input(&mut state, at, 0.75f32.to_bits());
                    set_compact_input(&mut state, at + 1, 0.75f32.to_bits());
                    if let Some(lod) = lod { set_compact_input(&mut state, at + 2, lod); }
                }
            }
            state
        };
        let implicit_state = make_state(implicit_requirements, None);
        let explicit_zero_state = make_state(explicit_requirements, Some(0.0f32.to_bits()));
        let explicit_nonzero_state = make_state(explicit_requirements, Some(1.0f32.to_bits()));
        let mut memory = vec![0u32; 128];
        memory[68..72].copy_from_slice(&[1.0f32.to_bits(), (-0.0f32).to_bits(), f32::INFINITY.to_bits(), 0]);
        let result = super::super::runner::run(&dxil, &implicit_state, &memory).unwrap();
        let explicit_result = super::super::runner::run(&explicit_dxil, &explicit_zero_state, &memory).unwrap();
        let tuple = |r: &super::super::runner::KernelResult, lane: usize| (r.raw[128 + lane * 256], r.raw[129 + lane * 256]);
        for lane in 0..128usize { assert_eq!(tuple(&result, lane), tuple(&explicit_result, lane)); }
        for lane in [0usize, 64] { assert_eq!(tuple(&result, lane), (0x8000_3c00, 0x0000_7c00)); }
        for lane in [1usize, 65] { assert_eq!(tuple(&result, lane), (0.75f32.to_bits(), 0.75f32.to_bits())); }
        let rejected_lod = super::super::runner::run(&explicit_dxil, &explicit_nonzero_state, &memory).unwrap();
        for lane in [0usize, 64] { assert_eq!(tuple(&rejected_lod, lane), (0.75f32.to_bits(), 0.75f32.to_bits())); }
        let short = super::super::runner::run(&short_dxil, &implicit_state, &memory).unwrap();
        for lane in [0usize, 64] { assert_eq!(tuple(&short, lane), (0.75f32.to_bits(), 0.75f32.to_bits())); }
        assert_eq!((272 + 8 <= 280, 272 + 12 <= 280), (true, false));
    }

    #[test]
    fn d16_sample_destination_span_matches_decoder_and_mature_validator() {
        let decode = |dst| {
            let bytes = crate::gpu::rdna2::encode("IMAGE_SAMPLE_L", "ENC_MIMG", &[
                ("VDATA", dst), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 6),
                ("DMASK", 7), ("DIM", 1), ("D16", 1),
            ], None).unwrap();
            super::super::mesh_memory::lower_stage_c_one(&crate::gpu::rdna2::decode(&bytes, 0).unwrap()).unwrap()
        };
        let valid = decode(254);
        let invalid = decode(255);
        for (op, dst) in [(valid, 254), (invalid, 255)] {
            assert!(matches!(op, StageCOp::ImageSampleLz {
                dst: actual, coordinate_count: 2, arrayed: false, dmask: 7,
                d16: true, lod: Some(2), ..
            } if actual == dst));
        }
        let layout = VgprScratchLayout::new(256, 1).unwrap();
        assert!(super::super::mesh_memory::validate_ops(layout, &[valid]).is_ok());
        assert!(super::super::mesh_memory::validate_ops(layout, &[invalid]).is_err());
        assert!(super::super::mesh_slot::lower_slot(&ProgramOp::Memory(valid), super::super::mesh_slot::WaveStateAbi::default()).is_ok());
        assert!(super::super::mesh_slot::lower_slot(&ProgramOp::Memory(invalid), super::super::mesh_slot::WaveStateAbi::default()).is_err());

        let implicit_bytes = crate::gpu::rdna2::encode("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[
            ("VDATA", 254), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 6),
            ("DMASK", 7), ("DIM", 1), ("D16", 1),
        ], None).unwrap();
        let implicit = super::super::mesh_memory::lower_stage_c_one(
            &crate::gpu::rdna2::decode(&implicit_bytes, 0).unwrap(),
        ).unwrap();
        assert!(matches!(implicit, StageCOp::ImageSampleLz {
            dst: 254, coordinate_count: 2, arrayed: false, dmask: 7,
            d16: true, lod: None, ..
        }));
        assert!(super::super::mesh_memory::validate_ops(layout, &[implicit]).is_ok());
        assert!(super::super::mesh_slot::lower_slot(
            &ProgramOp::Memory(implicit), super::super::mesh_slot::WaveStateAbi::default(),
        ).is_ok());
    }

    #[test]
    fn image_gather4_lz_minimal_live_order() {
        if super::super::dxc::location().is_err() || !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let region = ReadOnlyRegion { guest_base: 0x1_0000_0000, memory_offset: 0, byte_len: 512 };
        let program = program(vec![ProgramOp::Memory(StageCOp::ImageGather4Lz {
            dst: 0,
            coordinates: [0, 1],
            descriptor: 0,
            sampler: 8,
            component: 0,
            d16: false,
        })]);
        let requirements = requirements(&program).unwrap();
        let sgprs = requirements.sgprs.max(1) as usize;
        let vgprs = requirements.vgprs.max(1) as usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let shifted = region.guest_base >> 8;
        let descriptor = [
            shifted as u32,
            ((shifted >> 32) as u32 & 0xff) | (77 << 20) | (1 << 30),
            1 << 14,
            0xfac | (9 << 28),
            0,
            0,
            0,
            0,
        ];
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            for (register, value) in descriptor.into_iter().enumerate() {
                set_compact_input(&mut state, wb + register, value);
            }
            set_compact_input(&mut state, wb + 8, 146);
            set_compact_input(&mut state, wb + sgprs, 0b1_1101);
            for lane in 0..64usize {
                let at = vgpr_base + (wave * 64 + lane) * vgprs;
                let (x, y) = match lane {
                    2 => (-1.0f32, -1.0f32),
                    3 => (2.0f32, 2.0f32),
                    4 => (f32::NAN, 0.5f32),
                    _ => (0.5f32, 0.5f32),
                };
                set_compact_input(&mut state, at, x.to_bits());
                set_compact_input(&mut state, at + 1, y.to_bits());
            }
        }
        let mut memory = vec![0u32; 128];
        memory[0] = 0x100;
        memory[4] = 0x200;
        memory[64] = 0x300;
        memory[68] = 0x400;
        let run = |bytes: u32| {
            let shell = validation_epoch_hlsl(Some(ReadOnlyRegion { byte_len: bytes, ..region }), &program, 1, 8).unwrap();
            let hlsl = shell.hlsl.replace("xcl_global_load(memory,", "xcl_global_load(lds,");
            let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
            super::super::runner::run(&dxil, &state, &memory).unwrap()
        };
        let result = run(512);
        let tuple = |result: &super::super::runner::KernelResult, lane: usize| {
            let at = 128 + lane * 256;
            [result.raw[at], result.raw[at + 1], result.raw[at + 2], result.raw[at + 3]]
        };
        for lane in [0usize, 64] {
            assert_eq!(tuple(&result, lane), [0x300, 0x400, 0x200, 0x100]);
        }
        for lane in [1usize, 65] {
            assert_eq!(tuple(&result, lane), [0.5f32.to_bits(), 0.5f32.to_bits(), 0, 0]);
        }
        for lane in [2usize, 66] {
            assert_eq!(tuple(&result, lane), [0x100; 4]);
        }
        for lane in [3usize, 67] {
            assert_eq!(tuple(&result, lane), [0x400; 4]);
        }
        for lane in [4usize, 68] {
            assert!(f32::from_bits(tuple(&result, lane)[0]).is_nan());
            assert_eq!(&tuple(&result, lane)[1..], &[0.5f32.to_bits(), 0, 0]);
        }

        // The lower-left read at byte 256 succeeds, then the lower-right read
        // at byte 272 fails. All four aliased destinations must remain intact.
        let short = run(260);
        for lane in [0usize, 64] {
            assert_eq!(tuple(&short, lane), [0.5f32.to_bits(), 0.5f32.to_bits(), 0, 0]);
        }
        assert_eq!((256 + 4 <= 260, 272 + 4 <= 260), (true, false));
    }

    #[test]
    fn control_shell_executes_divergent_branches_backedges_and_faults() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let invalid = program(vec![
            ProgramOp::Control(ControlOp::BranchScc {
                set: true,
                target: 99,
            }),
            ProgramOp::Control(ControlOp::Complete),
        ]);
        let cases = [
            (branch_program(), 16u32),
            (backedge_program(), 16),
            (backedge_program(), 4),
            (invalid, 16),
        ];
        let compiled = cases
            .iter()
            .map(|(program, cap)| {
                let shell = validation_epoch_hlsl(None, program, 1, *cap).unwrap();
                let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                    .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
                (shell, dxil)
            })
            .collect::<Vec<_>>();
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let run = |case: usize, scc: [u32; 2], initial_v0: u32| {
            let (shell, dxil) = &compiled[case];
            let program = &cases[case].0;
            let requirements = requirements(program).unwrap();
            let sgprs = requirements.sgprs.max(1) as usize;
            let vgprs = requirements.vgprs.max(1) as usize;
            let wave_stride = sgprs + 6;
            let vgpr_base = 2 * wave_stride;
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb + sgprs, u32::MAX);
                set_compact_input(&mut state, wb + sgprs + 1, u32::MAX);
                set_compact_input(&mut state, wb + sgprs + 4, scc[wave]);
                for lane in 0..64usize {
                    set_compact_input(
                        &mut state,
                        vgpr_base + (wave * 64 + lane) * vgprs,
                        initial_v0,
                    );
                }
            }
            assert!(shell.lds_bytes >= 32768);
            super::super::runner::run(dxil, &state, &vec![0; 8192]).unwrap()
        };

        let branch = run(0, [1, 0], 99);
        assert_eq!((branch.raw[1], branch.raw[2]), (1, 2));
        assert_eq!((branch.raw[5], branch.raw[6]), (1, 3));
        assert_eq!(branch.raw[128], 99);
        assert_eq!(branch.raw[128 + 64 * 256], 11);

        let looped = run(1, [0, 0], 0);
        assert_eq!((looped.raw[1], looped.raw[2], looped.raw[8]), (1, 8, 3));
        assert_eq!((looped.raw[5], looped.raw[6], looped.raw[9]), (1, 8, 3));

        let capped = run(2, [0, 0], 0);
        assert_eq!((capped.raw[1], capped.raw[2], capped.raw[8]), (2, 4, 2));
        assert_eq!((capped.raw[5], capped.raw[6], capped.raw[9]), (2, 4, 2));

        let faulted = run(3, [1, 0], 0);
        assert_eq!((faulted.raw[1], faulted.raw[2]), (3, 1));
        assert_eq!((faulted.raw[5], faulted.raw[6]), (1, 2));
    }

    #[test]
    fn final_lane_family_cpu_edges_and_bounds_are_exact() {
        fn read_first(values: &[u32; 64], exec: [u32; 2]) -> Result<u32, ()> {
            let lane = if exec[0] != 0 {
                exec[0].trailing_zeros()
            } else if exec[1] != 0 {
                32 + exec[1].trailing_zeros()
            } else {
                return Err(());
            };
            Ok(values[lane as usize])
        }
        let values = core::array::from_fn(|lane| 0x1000 + lane as u32);
        assert_eq!(read_first(&values, [1 << 3, 0]), Ok(0x1003));
        assert_eq!(read_first(&values, [0, 1 << 3]), Ok(0x1023));
        assert_eq!(read_first(&values, [0, 0]), Err(()));
        assert!(super::lower_slot(
            &ProgramOp::Vector(StageBOp::ReadFirstLane { dst: 105, src: 0 }),
            WaveStateAbi::default(),
        )
        .is_ok());
        assert!(super::lower_slot(
            &ProgramOp::Vector(StageBOp::ReadFirstLane { dst: 106, src: 0 }),
            WaveStateAbi::default(),
        )
        .is_err());
        assert!(super::lower_slot(
            &ProgramOp::Vector(StageBOp::RelativeMove { dst: 0, base: 0 }),
            WaveStateAbi::default(),
        )
        .is_ok());
    }

    #[test]
    fn relative_move_matches_m0_bounds_aliases_and_exec_live() {
        let cpu_index = |base: u16, m0: u32| {
            (base as u32)
                .checked_add(m0)
                .filter(|&register| register < 256)
        };
        assert_eq!(cpu_index(0, 0), Some(0));
        assert_eq!(cpu_index(61, 0), Some(61));
        assert_eq!(cpu_index(61, 193), Some(254));
        assert_eq!(cpu_index(61, 195), None);
        assert_eq!(cpu_index(61, u32::MAX), None);
        if super::super::dxc::location().is_err() {
            return;
        }
        let bytes = crate::gpu::rdna2::encode(
            "V_MOVRELS_B32",
            "ENC_VOP1",
            &[("VDST", 61), ("SRC0", 256 + 61)],
            None,
        )
        .unwrap();
        let decoded = crate::gpu::rdna2::decode(&bytes, 0).unwrap();
        assert_eq!(
            super::super::mesh_vector::lower_stage_b_one(&decoded).unwrap(),
            StageBOp::RelativeMove { dst: 61, base: 61 }
        );

        let p = program(vec![
            ProgramOp::Vector(StageBOp::Move { dst: 193, src: VectorSource::Constant(0x1930_0001) }),
            ProgramOp::Vector(StageBOp::Move { dst: 254, src: VectorSource::Constant(0x2540_0001) }),
            ProgramOp::Vector(StageBOp::RelativeMove { dst: 0, base: 0 }),
            ProgramOp::Vector(StageBOp::RelativeMove { dst: 61, base: 61 }),
        ]);
        let shell = validation_epoch_hlsl(None, &p, 1, 8).unwrap();
        assert_eq!(shell.vgpr_scratch.vgpr_count, 256);
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let sgprs = 1usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = if wave == 0 { u64::MAX & !(1u64 << 1) } else { u64::MAX };
            set_compact_input(&mut state, wb + sgprs, exec as u32);
            set_compact_input(&mut state, wb + sgprs + 1, (exec >> 32) as u32);
            set_compact_input(&mut state, wb + sgprs + 5, if wave == 0 { 0 } else { 193 });
        }
        for lane in 0..64usize {
            let base = vgpr_base + lane * 256;
            set_compact_input(&mut state, base, 0xA000_0000 | lane as u32);
            set_compact_input(&mut state, base + 61, 0xB000_0000 | lane as u32);
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        for lane in 0..64usize {
            assert_eq!(result.raw[128 + lane * 256], 0xA000_0000 | lane as u32);
            assert_eq!(result.raw[128 + lane * 256 + 61], 0xB000_0000 | lane as u32);
            assert_eq!(result.raw[128 + (64 + lane) * 256], 0x1930_0001);
            assert_eq!(result.raw[128 + (64 + lane) * 256 + 61], 0x2540_0001);
        }

        let invalid = program(vec![ProgramOp::Vector(StageBOp::RelativeMove { dst: 61, base: 61 })]);
        let shell = validation_epoch_hlsl(None, &invalid, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[]).unwrap();
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            let exec = u64::MAX & !(1u64 << 1);
            set_compact_input(&mut state, wb + sgprs, exec as u32);
            set_compact_input(&mut state, wb + sgprs + 1, (exec >> 32) as u32);
            set_compact_input(&mut state, wb + sgprs + 5, if wave == 0 { 195 } else { u32::MAX });
        }
        for lane in 0..64usize {
            set_compact_input(&mut state, vgpr_base + lane * 256 + 61, 0xCC00_0000 | lane as u32);
        }
        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        for wave in 0..2usize {
            for lane in 0..64usize {
                let got = result.raw[128 + (wave * 64 + lane) * 256 + 61];
                let expected = if wave == 0 && lane == 1 { 0xCC00_0001 } else { 0 };
                assert_eq!(got, expected, "invalid wave {wave} lane {lane}");
            }
            assert_eq!(result.raw[wave * 4 + 1], 1, "invalid index is a zero source, not a scheduler fault");
        }
    }

    #[test]
    fn former_shared_limit_vgpr61_round_trips_live_for_both_waves() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let program = program(vec![ProgramOp::Vector(StageBOp::Binary {
            kind: VectorBinaryKind::AddNoCarry,
            dst: 61,
            lhs: VectorSource::Vgpr(61),
            rhs: VectorSource::Constant(1),
        })]);
        let shell = validation_epoch_hlsl(None, &program, 1, 4).unwrap();
        assert_eq!(shell.vgpr_scratch.vgpr_count, 62);
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }

        let sgprs = 1usize;
        let vgprs = 62usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * wave_stride;
            set_compact_input(&mut state, wb + sgprs, u32::MAX);
            set_compact_input(&mut state, wb + sgprs + 1, u32::MAX);
            for lane in 0..64usize {
                let value = 0x6100_0000u32 | ((wave as u32) << 16) | lane as u32;
                let input = vgpr_base + (wave * 64 + lane) * vgprs + 61;
                set_compact_input(&mut state, input, value);
            }
        }

        let result = super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap();
        for wave in 0..2usize {
            for lane in 0..64usize {
                let initial = 0x6100_0000u32 | ((wave as u32) << 16) | lane as u32;
                let output = 128 + (wave * 64 + lane) * 256 + 61;
                assert_eq!(result.raw[output], initial.wrapping_add(1), "wave {wave} lane {lane}");
            }
        }
    }

    #[test]
    fn final_lane_family_compiles_and_executes_wave_local_faults() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let p = program(vec![ProgramOp::Vector(StageBOp::ReadFirstLane {
            dst: 0,
            src: 0,
        })]);
        let shell = validation_epoch_hlsl(None, &p, 1, 8).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let sgprs = 1usize;
        let vgprs = 1usize;
        let wave_stride = sgprs + 6;
        let vgpr_base = 2 * wave_stride;
        let run = |exec: [[u32; 2]; 2]| {
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * wave_stride;
                set_compact_input(&mut state, wb + sgprs, exec[wave][0]);
                set_compact_input(&mut state, wb + sgprs + 1, exec[wave][1]);
                for lane in 0..64usize {
                    let gi = wave * 64 + lane;
                    let base = vgpr_base + gi * vgprs;
                    set_compact_input(&mut state, base, 0x2000 + gi as u32);
                }
            }
            super::super::runner::run(&dxil, &state, &vec![0; 8192]).unwrap()
        };

        let halves = run([[1 << 3, 0], [0, 1 << 3]]);
        assert_eq!((halves.raw[1], halves.raw[5]), (1, 1));
        assert_eq!((halves.raw[8], halves.raw[9]), (0x2003, 0x2063));

        let isolated = run([[0, 0], [0, 1 << 3]]);
        assert_eq!((isolated.raw[1], isolated.raw[2]), (3, 1));
        assert_eq!((isolated.raw[5], isolated.raw[6]), (1, 1));
        assert_eq!(isolated.raw[8], 0); // empty EXEC faults without a scalar write
        assert_eq!(isolated.raw[9], 0x2063);
    }

    #[test]
    fn scalar_binary_family_matches_independent_cpu_edges_and_wave64() {
        fn cpu(kind: ScalarBinaryKind, lhs: u32, rhs: u32, old_scc: u32) -> (u32, u32) {
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
                ScalarBinaryKind::BfeU32 => {
                    let (offset, width) = (rhs & 31, (rhs >> 16) & 127);
                    if width == 0 {
                        0
                    } else if width >= 32 {
                        lhs >> offset
                    } else {
                        (lhs >> offset) & (u32::MAX >> (32 - width))
                    }
                }
                ScalarBinaryKind::BfeI32 => {
                    let (offset, width) = (rhs & 31, (rhs >> 16) & 127);
                    if width == 0 {
                        0
                    } else if width >= 32 {
                        ((lhs as i32) >> offset) as u32
                    } else {
                        let shift = 32 - width;
                        ((((lhs >> offset) << shift) as i32) >> shift) as u32
                    }
                }
            };
            let scc = match kind {
                ScalarBinaryKind::AddI32 => u32::from((((lhs ^ value) & (rhs ^ value)) >> 31) != 0),
                ScalarBinaryKind::SubI32 => u32::from((((lhs ^ rhs) & (lhs ^ value)) >> 31) != 0),
                ScalarBinaryKind::MulI32 | ScalarBinaryKind::MulHiU32 => old_scc,
                ScalarBinaryKind::MinU32 => u32::from(lhs < rhs),
                _ => u32::from(value != 0),
            };
            (value, scc)
        }
        let cases = [
            (ScalarBinaryKind::AddI32, 0x7fff_ffff, 1),
            (ScalarBinaryKind::SubI32, 0x8000_0000, 1),
            (ScalarBinaryKind::MulI32, u32::MAX, 2),
            (ScalarBinaryKind::MulHiU32, u32::MAX, 2),
            (ScalarBinaryKind::MinU32, 9, 3),
            (ScalarBinaryKind::And, 0xf0, 0x0f),
            (ScalarBinaryKind::Or, 0xf0, 0x0f),
            (ScalarBinaryKind::Xor, 0xaa, 0xff),
            (ScalarBinaryKind::Lshl, 1, 32),
            (ScalarBinaryKind::Lshr, 0x8000_0000, 31),
            (ScalarBinaryKind::AshrI32, 0x8000_0000, 31),
            (ScalarBinaryKind::BfeU32, 0xf0, (4 << 16) | 4),
            (ScalarBinaryKind::BfeI32, 0x80, 8 << 16),
        ];
        if super::super::dxc::location().is_err() {
            return;
        }
        let compiled = cases
            .iter()
            .map(|(kind, lhs, rhs)| {
                let p = program(vec![
                    ProgramOp::Scalar(StageAOp::Binary {
                        kind: *kind,
                        dst: 0,
                        lhs: crate::gpu::recomp::mesh_scalar::Operand32::Constant(*lhs),
                        rhs: crate::gpu::recomp::mesh_scalar::Operand32::Constant(*rhs),
                    }),
                    ProgramOp::Scalar(StageAOp::NoEffect),
                ]);
                let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
                let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                    .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
                (shell, dxil)
            })
            .collect::<Vec<_>>();
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        for (index, ((kind, lhs, rhs), (shell, dxil))) in cases.iter().zip(&compiled).enumerate() {
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, 1);
                set_compact_input(&mut state, wb + 5, wave as u32);
            }
            let result =
                super::super::runner::run(dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            for wave in 0..2usize {
                let expected = cpu(*kind, *lhs, *rhs, wave as u32);
                assert_eq!(
                    (result.raw[8 + wave], result.raw[3 + wave * 4]),
                    expected,
                    "case {index} wave {wave}"
                );
            }
        }

        let special_programs = [
            program(vec![ProgramOp::Scalar(StageAOp::Binary {
                kind: ScalarBinaryKind::AddI32,
                dst: 0,
                lhs: crate::gpu::recomp::mesh_scalar::Operand32::Register(0),
                rhs: crate::gpu::recomp::mesh_scalar::Operand32::Scc,
            })]),
            program(vec![ProgramOp::Scalar(StageAOp::Binary {
                kind: ScalarBinaryKind::Xor,
                dst: 0,
                lhs: crate::gpu::recomp::mesh_scalar::Operand32::VccZero,
                rhs: crate::gpu::recomp::mesh_scalar::Operand32::ExecZero,
            })]),
        ];
        for (case, p) in special_programs.iter().enumerate() {
            let shell = validation_epoch_hlsl(None, p, 1, 2).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb, 5);
                set_compact_input(&mut state, wb + 1, 1);
                set_compact_input(&mut state, wb + 3, u32::from(wave == 1));
                set_compact_input(&mut state, wb + 5, wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            let expected = if case == 0 { [5, 6] } else { [1, 0] };
            assert_eq!([result.raw[8], result.raw[9]], expected);
        }
    }

    #[test]
    fn scalar_binary64_family_matches_cpu_pairs_aliases_and_wave64() {
        fn cpu(kind: ScalarBinary64Kind, lhs: u64, rhs: u64) -> (u64, u32) {
            let value = match kind {
                ScalarBinary64Kind::And => lhs & rhs,
                ScalarBinary64Kind::AndNot => lhs & !rhs,
                ScalarBinary64Kind::Or => lhs | rhs,
                ScalarBinary64Kind::Xor => lhs ^ rhs,
            };
            (value, u32::from(value != 0))
        }
        let cases = [
            (
                ScalarBinary64Kind::And,
                0xf0f0_0000_ffff_0000,
                0x0ff0_ffff_00ff_00ff,
            ),
            (ScalarBinary64Kind::AndNot, u64::MAX, 0xffff_0000_ffff_0000),
            (ScalarBinary64Kind::Or, 0x8000_0000_0000_0000, 1),
            (
                ScalarBinary64Kind::Xor,
                0xaaaa_5555_ffff_0000,
                0xaaaa_5555_ffff_0000,
            ),
        ];
        if super::super::dxc::location().is_err() {
            return;
        }
        for (index, (kind, lhs, rhs)) in cases.into_iter().enumerate() {
            let expected = cpu(kind, lhs, rhs);
            for component in 0..2u8 {
                let p = program(vec![
                    ProgramOp::Scalar(StageAOp::Binary64 {
                        kind,
                        dst: 0,
                        lhs: Operand64::Constant(lhs),
                        rhs: Operand64::Constant(rhs),
                    }),
                    ProgramOp::Scalar(StageAOp::Move32 {
                        dst: 0,
                        source: Operand32::Register(component),
                    }),
                    ProgramOp::Scalar(StageAOp::NoEffect),
                ]);
                let shell = validation_epoch_hlsl(None, &p, 1, 6).unwrap();
                let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                    .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
                if !super::super::runner::wave64_supported().unwrap_or(false) {
                    return;
                }
                let mut state = super::super::runner::KernelState::default();
                for wave in 0..2usize {
                    let wb = wave * 8;
                    set_compact_input(&mut state, wb + 2, 1);
                    set_compact_input(&mut state, wb + 6, 1 - wave as u32);
                }
                let result = super::super::runner::run(
                    &dxil,
                    &state,
                    &vec![0; shell.lds_bytes as usize / 4],
                )
                .unwrap();
                let value = if component == 0 {
                    expected.0 as u32
                } else {
                    (expected.0 >> 32) as u32
                };
                assert_eq!(
                    [result.raw[8], result.raw[9]],
                    [value, value],
                    "case {index} component {component}"
                );
                assert_eq!([result.raw[3], result.raw[7]], [expected.1, expected.1]);
            }
        }

        let alias = program(vec![
            ProgramOp::Scalar(StageAOp::Binary64 {
                kind: ScalarBinary64Kind::Xor,
                dst: 0,
                lhs: Operand64::Register(0),
                rhs: Operand64::Constant(0xffff_ffff_0000_0000),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(1),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &alias, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 8;
            set_compact_input(&mut state, wb, 0x1122_3344 + wave as u32);
            set_compact_input(&mut state, wb + 1, 0x5566_7788 + wave as u32);
            set_compact_input(&mut state, wb + 2, 1);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0xaa99_8877, 0xaa99_8876]);

        let special = program(vec![
            ProgramOp::Scalar(StageAOp::Binary64 {
                kind: ScalarBinary64Kind::Or,
                dst: 106,
                lhs: Operand64::Register(106),
                rhs: Operand64::Register(126),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(107),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &special, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 7;
            set_compact_input(&mut state, wb + 1, 1 << wave);
            set_compact_input(&mut state, wb + 2, 0x10 << wave);
            set_compact_input(&mut state, wb + 3, 0x100 << wave);
            set_compact_input(&mut state, wb + 4, 0x1000 << wave);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0x1100, 0x2200]);
        assert_eq!([result.raw[3], result.raw[7]], [1, 1]);
    }

    #[test]
    fn scalar_compare_eq_matches_cpu_constants_and_special_sources_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let cases = [
            (Operand32::Constant(7), Operand32::Constant(7), [1, 1]),
            (Operand32::Constant(7), Operand32::Constant(8), [0, 0]),
            (Operand32::Register(106), Operand32::Register(126), [1, 0]),
            (Operand32::VccZero, Operand32::ExecZero, [0, 1]),
        ];
        for (case, (lhs, rhs, expected)) in cases.into_iter().enumerate() {
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::CompareEq { lhs, rhs }),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Scc,
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, if wave == 0 { 1 } else { 4 });
                set_compact_input(
                    &mut state,
                    wb + 3,
                    if case == 3 && wave == 0 {
                        0
                    } else if wave == 0 {
                        1
                    } else {
                        2
                    },
                );
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], expected);
        }
    }

    #[test]
    fn scalar_compare_ne_matches_cpu_constants_and_special_sources_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let cases = [
            (Operand32::Constant(7), Operand32::Constant(7), [0, 0]),
            (Operand32::Constant(7), Operand32::Constant(8), [1, 1]),
            (Operand32::Register(106), Operand32::Register(126), [0, 1]),
            (Operand32::VccZero, Operand32::ExecZero, [1, 0]),
        ];
        for (case, (lhs, rhs, expected)) in cases.into_iter().enumerate() {
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::CompareNeU32 { lhs, rhs }),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Scc,
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, if wave == 0 { 1 } else { 4 });
                set_compact_input(
                    &mut state,
                    wb + 3,
                    if case == 3 && wave == 0 {
                        0
                    } else if wave == 0 {
                        1
                    } else {
                        2
                    },
                );
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], expected);
        }
    }

    #[test]
    fn scalar_unsigned_compares_match_cpu_boundaries_and_special_sources_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let cases = [
            (
                0u8,
                Operand32::Constant(0),
                Operand32::Constant(u32::MAX),
                [1, 1],
            ),
            (
                1,
                Operand32::Constant(u32::MAX),
                Operand32::Constant(0),
                [1, 1],
            ),
            (2, Operand32::Constant(5), Operand32::Constant(5), [1, 1]),
            (
                3,
                Operand32::Constant(0),
                Operand32::Constant(u32::MAX),
                [0, 0],
            ),
            (
                0,
                Operand32::Register(106),
                Operand32::Register(126),
                [0, 1],
            ),
            (1, Operand32::VccZero, Operand32::ExecZero, [1, 0]),
        ];
        for (case, (kind, lhs, rhs, expected)) in cases.into_iter().enumerate() {
            let op = match kind {
                0 => StageAOp::CompareLtU32 { lhs, rhs },
                1 => StageAOp::CompareGtU32 { lhs, rhs },
                2 => StageAOp::CompareLeU32 { lhs, rhs },
                _ => StageAOp::CompareGeU32 { lhs, rhs },
            };
            let p = program(vec![
                ProgramOp::Scalar(op),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Scc,
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, if wave == 0 { 1 } else { 4 });
                set_compact_input(
                    &mut state,
                    wb + 3,
                    if case == 5 && wave == 0 {
                        0
                    } else if wave == 0 {
                        1
                    } else {
                        2
                    },
                );
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], expected);
        }
    }

    #[test]
    fn scalar_signed_compares_match_cpu_boundaries_and_special_sources_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let cases = [
            (
                0u8,
                Operand32::Constant(i32::MIN as u32),
                Operand32::Constant(i32::MAX as u32),
                [1, 1],
            ),
            (
                1,
                Operand32::Constant(u32::MAX),
                Operand32::Constant(0),
                [0, 0],
            ),
            (
                2,
                Operand32::Constant(u32::MAX),
                Operand32::Constant(u32::MAX),
                [1, 1],
            ),
            (
                0,
                Operand32::Register(106),
                Operand32::Register(126),
                [0, 1],
            ),
            (1, Operand32::VccZero, Operand32::ExecZero, [1, 0]),
        ];
        for (case, (kind, lhs, rhs, expected)) in cases.into_iter().enumerate() {
            let op = match kind {
                0 => StageAOp::CompareLtI32 { lhs, rhs },
                1 => StageAOp::CompareGtI32 { lhs, rhs },
                _ => StageAOp::CompareGeI32 { lhs, rhs },
            };
            let p = program(vec![
                ProgramOp::Scalar(op),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Scc,
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, if wave == 0 { 1 } else { 2 });
                set_compact_input(
                    &mut state,
                    wb + 3,
                    if case == 4 && wave == 0 {
                        0
                    } else if case == 3 && wave == 1 {
                        0x8000_0000
                    } else if wave == 0 {
                        1
                    } else {
                        2
                    },
                );
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], expected);
        }
    }

    #[test]
    fn scalar_select64_matches_cpu_pairs_aliases_and_special_sources_wave64() {
        fn cpu(condition: u32, when_set: u64, when_clear: u64) -> u64 {
            if condition != 0 {
                when_set
            } else {
                when_clear
            }
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        for component in 0..2u8 {
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::Select64 {
                    dst: 0,
                    when_set: Operand64::Register(0),
                    when_clear: Operand64::Constant(0x8000_0000_ffff_ffff),
                }),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Register(component),
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            let set = [0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210];
            for wave in 0..2usize {
                let wb = wave * 8;
                set_compact_input(&mut state, wb, set[wave] as u32);
                set_compact_input(&mut state, wb + 1, (set[wave] >> 32) as u32);
                set_compact_input(&mut state, wb + 2, 1);
                set_compact_input(&mut state, wb + 6, wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            let values = [
                cpu(0, set[0], 0x8000_0000_ffff_ffff),
                cpu(1, set[1], 0x8000_0000_ffff_ffff),
            ];
            let expected = values.map(|v| {
                if component == 0 {
                    v as u32
                } else {
                    (v >> 32) as u32
                }
            });
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], [0, 1]);
        }
        for component in 0..2u8 {
            let source = if component == 0 {
                Operand32::Register(106)
            } else {
                Operand32::Register(107)
            };
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::Select64 {
                    dst: 106,
                    when_set: Operand64::Register(106),
                    when_clear: Operand64::Register(126),
                }),
                ProgramOp::Scalar(StageAOp::Move32 { dst: 0, source }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb + 1, [0x11, 0x33][wave]);
                set_compact_input(&mut state, wb + 2, [0x22, 0x44][wave]);
                set_compact_input(&mut state, wb + 3, [0xaa, 0xcc][wave]);
                set_compact_input(&mut state, wb + 4, [0xbb, 0xdd][wave]);
                set_compact_input(&mut state, wb + 5, wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            let expected = if component == 0 {
                [0x11, 0xcc]
            } else {
                [0x22, 0xdd]
            };
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], [0, 1]);
        }
        let scc_source = program(vec![
            ProgramOp::Scalar(StageAOp::Select64 {
                dst: 0,
                when_set: Operand64::Scc,
                when_clear: Operand64::Constant(0),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(0),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &scc_source, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 8;
            set_compact_input(&mut state, wb + 2, 1);
            set_compact_input(&mut state, wb + 6, wave as u32);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0, 1]);
        assert_eq!([result.raw[3], result.raw[7]], [0, 1]);
    }

    #[test]
    fn scalar_select32_and_move64_match_cpu_aliases_and_special_sources_wave64() {
        fn cpu_select(condition: u32, when_set: u32, when_clear: u32) -> u32 {
            if condition != 0 {
                when_set
            } else {
                when_clear
            }
        }
        fn cpu_move64(value: u64) -> [u32; 2] {
            [value as u32, (value >> 32) as u32]
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        let select = program(vec![ProgramOp::Scalar(StageAOp::Select32 {
            dst: 0,
            when_set: Operand32::Register(0),
            when_clear: Operand32::Constant(0x8000_0001),
        })]);
        let shell = validation_epoch_hlsl(None, &select, 1, 2).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 7;
            set_compact_input(&mut state, wb, 0x1234_0000 + wave as u32);
            set_compact_input(&mut state, wb + 1, 1);
            set_compact_input(&mut state, wb + 5, 2 * wave as u32);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!(
            [result.raw[8], result.raw[9]],
            [
                cpu_select(0, 0x1234_0000, 0x8000_0001),
                cpu_select(2, 0x1234_0001, 0x8000_0001)
            ]
        );
        assert_eq!([result.raw[3], result.raw[7]], [0, 2]);

        for component in 0..2u8 {
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::Move64 {
                    dst: 0,
                    source: Operand64::Constant(0x8000_0000_ffff_ffff),
                }),
                ProgramOp::Scalar(StageAOp::Move32 {
                    dst: 0,
                    source: Operand32::Register(component),
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 8;
                set_compact_input(&mut state, wb + 2, 1);
                set_compact_input(&mut state, wb + 6, wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            let expected = cpu_move64(0x8000_0000_ffff_ffff)[component as usize];
            assert_eq!([result.raw[8], result.raw[9]], [expected, expected]);
            assert_eq!([result.raw[3], result.raw[7]], [0, 1]);
        }

        let alias = program(vec![
            ProgramOp::Scalar(StageAOp::Move64 {
                dst: 0,
                source: Operand64::Register(0),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(1),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &alias, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 8;
            set_compact_input(&mut state, wb, 0x1111_0000 + wave as u32);
            set_compact_input(&mut state, wb + 1, 0x8000_0000 + wave as u32);
            set_compact_input(&mut state, wb + 2, 1);
            set_compact_input(&mut state, wb + 6, wave as u32);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0x8000_0000, 0x8000_0001]);
        assert_eq!([result.raw[3], result.raw[7]], [0, 1]);

        let special = program(vec![
            ProgramOp::Scalar(StageAOp::Move64 {
                dst: 106,
                source: Operand64::Register(126),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(107),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &special, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 7;
            set_compact_input(&mut state, wb + 1, 0x10 + wave as u32);
            set_compact_input(&mut state, wb + 2, 0x20 + wave as u32);
            set_compact_input(&mut state, wb + 5, wave as u32);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0x20, 0x21]);
        assert_eq!([result.raw[3], result.raw[7]], [0, 1]);
    }

    #[test]
    fn scalar_exec_transforms_match_cpu_masks_and_following_exec_wave64() {
        fn cpu(kind: u8, old: u64, source: u64) -> (u64, u64, u32) {
            let value = match kind {
                0 => old & source,
                1 => old | source,
                _ => source & !old,
            };
            let dst = if kind < 2 { old } else { value };
            (dst, value, u32::from(value != 0))
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        let source = 0x8000_0000_0000_0006u64;
        for kind in 0..3u8 {
            let scalar = match kind {
                0 => StageAOp::SaveExecAnd64 {
                    dst: 0,
                    source: Operand64::Constant(source),
                },
                1 => StageAOp::SaveExecOr64 {
                    dst: 0,
                    source: Operand64::Constant(source),
                },
                _ => StageAOp::WrExecAndNot64 {
                    dst: 0,
                    source: Operand64::Constant(source),
                },
            };
            let p = program(vec![
                ProgramOp::Scalar(scalar),
                ProgramOp::Vector(StageBOp::Move {
                    dst: 0,
                    src: VectorSource::Constant(0xdead_beef),
                }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let olds = [0x8000_0000_0000_000bu64, u64::MAX];
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 8;
                set_compact_input(&mut state, wb + 2, olds[wave] as u32);
                set_compact_input(&mut state, wb + 3, (olds[wave] >> 32) as u32);
                for lane in 0..64usize {
                    set_compact_input(&mut state, 16 + (wave * 64 + lane), 0x5151_5151);
                }
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            for wave in 0..2usize {
                let expected = cpu(kind, olds[wave], source);
                assert_eq!(result.raw[8 + wave], expected.0 as u32);
                assert_eq!(result.raw[3 + wave * 4], expected.2);
                for lane in 0..64usize {
                    let active = ((expected.1 >> lane) & 1) != 0;
                    assert_eq!(
                        result.raw[128 + (wave * 64 + lane) * 256],
                        if active { 0xdead_beef } else { 0x5151_5151 }
                    );
                }
            }
        }

        let special = program(vec![
            ProgramOp::Scalar(StageAOp::SaveExecAnd64 {
                dst: 106,
                source: Operand64::Register(126),
            }),
            ProgramOp::Scalar(StageAOp::Move32 {
                dst: 0,
                source: Operand32::Register(107),
            }),
        ]);
        let shell = validation_epoch_hlsl(None, &special, 1, 4).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 7;
            set_compact_input(&mut state, wb + 1, 0x10 + wave as u32);
            set_compact_input(&mut state, wb + 2, 0x20 + wave as u32);
        }
        let result =
            super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                .unwrap();
        assert_eq!([result.raw[8], result.raw[9]], [0x20, 0x21]);
        assert_eq!([result.raw[3], result.raw[7]], [1, 1]);
    }

    #[test]
    fn scalar_unary_bits_match_cpu_edges_aliases_and_special_sources_wave64() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let cases = [
            (
                StageAOp::BitSet1 {
                    dst: 0,
                    bit: Operand32::Constant(31),
                },
                [1u32 << 31, 1u32 << 31],
                [2, 3],
            ),
            (
                StageAOp::BitSet1 {
                    dst: 0,
                    bit: Operand32::Constant(32),
                },
                [1, 1],
                [2, 3],
            ),
            (
                StageAOp::Not32 {
                    dst: 0,
                    source: Operand32::Register(0),
                },
                [!0u32, !u32::MAX],
                [1, 0],
            ),
            (
                StageAOp::BitReverse32 {
                    dst: 0,
                    source: Operand32::Register(0),
                },
                [0x8000_0001u32.reverse_bits(), 0xaaaa_aaaau32.reverse_bits()],
                [2, 3],
            ),
            (
                StageAOp::CountBits32 {
                    dst: 0,
                    source: Operand32::Register(0),
                },
                [0u32.count_ones(), u32::MAX.count_ones()],
                [0, 1],
            ),
            (
                StageAOp::CountBits64 {
                    dst: 0,
                    source: Operand64::Register(126),
                },
                [0u64.count_ones(), u64::MAX.count_ones()],
                [0, 1],
            ),
        ];
        for (case, (op, expected, expected_scc)) in cases.into_iter().enumerate() {
            let p = program(vec![ProgramOp::Scalar(op)]);
            let shell = validation_epoch_hlsl(None, &p, 1, 2).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                let input = if case == 3 {
                    [0x8000_0001, 0xaaaa_aaaa][wave]
                } else {
                    [0, u32::MAX][wave]
                };
                set_compact_input(&mut state, wb, input);
                let exec = if case == 5 { [0, u32::MAX][wave] } else { 1 };
                set_compact_input(&mut state, wb + 1, exec);
                set_compact_input(&mut state, wb + 2, if case == 5 { exec } else { 0 });
                set_compact_input(&mut state, wb + 5, [2, 3][wave]);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], expected_scc);
        }
    }

    #[test]
    fn scalar_add_and_immediate_multiply_match_cpu_boundaries_wave64() {
        fn cpu_add(lhs: u32, rhs: u32, carry_in: bool, old_scc: u32) -> (u32, u32) {
            let sum = lhs as u64 + rhs as u64 + u64::from(carry_in && old_scc != 0);
            (sum as u32, u32::from(sum > u32::MAX as u64))
        }
        if super::super::dxc::location().is_err() {
            return;
        }
        let adds = [
            (
                StageAOp::AddU32 {
                    dst: 0,
                    lhs: Operand32::Constant(u32::MAX),
                    rhs: Operand32::Constant(1),
                    carry_in: false,
                },
                [(0, 1), (0, 1)],
            ),
            (
                StageAOp::AddU32 {
                    dst: 0,
                    lhs: Operand32::Constant(u32::MAX),
                    rhs: Operand32::Constant(0),
                    carry_in: true,
                },
                [cpu_add(u32::MAX, 0, true, 0), cpu_add(u32::MAX, 0, true, 2)],
            ),
            (
                StageAOp::AddU32 {
                    dst: 0,
                    lhs: Operand32::Register(0),
                    rhs: Operand32::Scc,
                    carry_in: true,
                },
                [cpu_add(5, 0, true, 0), cpu_add(5, 2, true, 2)],
            ),
            (
                StageAOp::AddU32 {
                    dst: 0,
                    lhs: Operand32::VccZero,
                    rhs: Operand32::ExecZero,
                    carry_in: false,
                },
                [(1, 0), (0, 0)],
            ),
        ];
        for (case, (op, expected)) in adds.into_iter().enumerate() {
            let p = program(vec![ProgramOp::Scalar(op)]);
            let shell = validation_epoch_hlsl(None, &p, 1, 2).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb, 5);
                set_compact_input(&mut state, wb + 1, 1);
                set_compact_input(
                    &mut state,
                    wb + 3,
                    if case == 3 && wave == 0 { 0 } else { 1 },
                );
                set_compact_input(&mut state, wb + 5, 2 * wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            assert_eq!(
                [
                    (result.raw[8], result.raw[3]),
                    (result.raw[9], result.raw[7])
                ],
                expected
            );
        }

        let multiplies = [
            (2i16, [0x4000_0000, u32::MAX]),
            (-1, [1, 0x8000_0000]),
            (i16::MIN, [2, u32::MAX]),
            (i16::MAX, [u32::MAX, 0x8001_0001]),
        ];
        for (immediate, inputs) in multiplies {
            let p = program(vec![ProgramOp::Scalar(StageAOp::MulImmediateI32 {
                dst: 0,
                value: immediate,
            })]);
            let shell = validation_epoch_hlsl(None, &p, 1, 2).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 7;
                set_compact_input(&mut state, wb, inputs[wave]);
                set_compact_input(&mut state, wb + 1, 1);
                set_compact_input(&mut state, wb + 5, 2 * wave as u32);
            }
            let result =
                super::super::runner::run(&dxil, &state, &vec![0; shell.lds_bytes as usize / 4])
                    .unwrap();
            let expected = inputs.map(|v| v.wrapping_mul(immediate as i32 as u32));
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], [0, 2]);
        }
    }

    #[test]
    fn scalar_get_pc_uses_byte_next_pc_for_two_waves_and_pair_destinations() {
        let next_pc = 0x1234_5678_9abc_def0u64;
        if super::super::dxc::location().is_err() {
            return;
        }
        for (dst, component) in [(4u8, 0u8), (4, 1), (106, 0), (106, 1)] {
            let source = Operand32::Register(dst + component);
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::GetPc64 { dst, next_pc }),
                ProgramOp::Scalar(StageAOp::Move32 { dst: 0, source }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 4).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 11;
                set_compact_input(&mut state, wb + 5, 0x55 + wave as u32);
                set_compact_input(&mut state, wb + 9, wave as u32 + 1);
            }
            let result = super::super::runner::run(
                &dxil,
                &state,
                &vec![0; shell.lds_bytes as usize / 4],
            )
            .unwrap();
            let expected = if component == 0 { next_pc as u32 } else { (next_pc >> 32) as u32 };
            assert_eq!([result.raw[8], result.raw[9]], [expected; 2]);
            assert_eq!([result.raw[3], result.raw[7]], [1, 2]);
        }
    }

    #[test]
    fn scalar_barrier_rendezvous_handles_divergent_arrival_empty_exec_and_repetition() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let p = program(vec![
            ProgramOp::Control(ControlOp::BranchScc { set: true, target: 2 }),
            ProgramOp::Scalar(StageAOp::NoEffect),
            ProgramOp::Scalar(StageAOp::Barrier),
            ProgramOp::Scalar(StageAOp::MoveImmediate { dst: 0, value: 7 }),
            ProgramOp::Scalar(StageAOp::Barrier),
        ]);
        let shell = validation_epoch_hlsl(None, &p, 1, 16).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let mut state = super::super::runner::KernelState::default();
        for wave in 0..2usize {
            let wb = wave * 7;
            set_compact_input(&mut state, wb + 1, if wave == 0 { 0 } else { u32::MAX });
            set_compact_input(&mut state, wb + 2, if wave == 0 { 0 } else { u32::MAX });
            set_compact_input(&mut state, wb + 5, u32::from(wave == 0));
        }
        let result = super::super::runner::run(
            &dxil,
            &state,
            &vec![0; shell.lds_bytes as usize / 4],
        )
        .unwrap();
        assert_eq!([result.raw[1], result.raw[5]], [1, 1]);
        assert_eq!([result.raw[8], result.raw[9]], [7, 7]);
        assert_eq!([result.raw[3], result.raw[7]], [1, 0]);

        let orphan = program(vec![
            ProgramOp::Control(ControlOp::BranchScc { set: true, target: 2 }),
            ProgramOp::Scalar(StageAOp::Barrier),
            ProgramOp::Control(ControlOp::Complete),
        ]);
        let shell = validation_epoch_hlsl(None, &orphan, 1, 8).unwrap();
        let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
        let mut state = super::super::runner::KernelState::default();
        set_compact_input(&mut state, 5, 1);
        let result = super::super::runner::run(
            &dxil,
            &state,
            &vec![0; shell.lds_bytes as usize / 4],
        )
        .unwrap();
        assert_eq!([result.raw[1], result.raw[5]], [1, 3]);
    }

    #[test]
    fn scalar_rendezvous_orders_cross_wave_uav_payloads_live() {
        if super::super::dxc::location().is_err()
            || !super::super::runner::wave64_supported().unwrap_or(false)
        {
            return;
        }
        let hlsl = r#"RWByteAddressBuffer io:register(u0);RWByteAddressBuffer res:register(u1);
groupshared uint waiting[2];groupshared uint release_group;
[WaveSize(64)][numthreads(128,1,1)]void main(uint gi:SV_GroupIndex){
 uint wave=gi>>6u,lane=gi&63u;
 if(lane==0u)waiting[wave]=0u;if(gi==0u)release_group=0u;
 GroupMemoryBarrierWithGroupSync();
 [unroll]for(uint epoch=0u;epoch<2u;epoch++){
  if(lane==0u&&wave==epoch){
   io.Store(4u*wave,wave==0u?0x13579bdfu:0x2468ace0u);
   waiting[wave]=1u;
  }
  AllMemoryBarrierWithGroupSync();
  if(gi==0u)release_group=waiting[0u]&waiting[1u];
  GroupMemoryBarrierWithGroupSync();
  if(lane==0u&&release_group!=0u)waiting[wave]=0u;
  AllMemoryBarrierWithGroupSync();
 }
 if(lane==0u&&release_group!=0u)res.Store(4u*wave,io.Load(4u*(wave^1u)));
}"#;
        let dxil = super::super::dxc::compile(hlsl, "main", "cs_6_6", &[]).unwrap();
        let result = super::super::runner::run(
            &dxil,
            &super::super::runner::KernelState::default(),
            &[],
        )
        .unwrap();
        assert_eq!([result.raw[0], result.raw[1]], [0x2468_ace0, 0x1357_9bdf]);
    }

    #[test]
    fn scalar_hardware_configuration_is_inert_in_bounded_shell_live() {
        if super::super::dxc::location().is_err() {
            return;
        }
        for source in [Operand32::Register(1), Operand32::Register(126)] {
            let p = program(vec![
                ProgramOp::Scalar(StageAOp::ConfigureHardwareRegister {
                    register: 1, offset: 2, width: 2, value: 0,
                }),
                ProgramOp::Scalar(StageAOp::ConfigureHardwareRegister {
                    register: 20, offset: 0, width: 32, value: 0,
                }),
                ProgramOp::Scalar(StageAOp::ConfigureHardwareRegister {
                    register: 21, offset: 0, width: 32, value: 1024,
                }),
                ProgramOp::Scalar(StageAOp::Move32 { dst: 0, source }),
            ]);
            let shell = validation_epoch_hlsl(None, &p, 1, 8).unwrap();
            let dxil = super::super::dxc::compile(&shell.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{}", shell.hlsl));
            if !super::super::runner::wave64_supported().unwrap_or(false) {
                return;
            }
            let mut state = super::super::runner::KernelState::default();
            for wave in 0..2usize {
                let wb = wave * 8;
                set_compact_input(&mut state, wb + 1, 0x1111_0000 + wave as u32);
                set_compact_input(&mut state, wb + 2, 0xaaaa_0000 + wave as u32);
                set_compact_input(&mut state, wb + 3, 0xbbbb_0000 + wave as u32);
                set_compact_input(&mut state, wb + 6, wave as u32 + 1);
            }
            let result = super::super::runner::run(
                &dxil,
                &state,
                &vec![0; shell.lds_bytes as usize / 4],
            )
            .unwrap();
            let expected = if source == Operand32::Register(1) {
                [0x1111_0000, 0x1111_0001]
            } else {
                [0xaaaa_0000, 0xaaaa_0001]
            };
            assert_eq!([result.raw[8], result.raw[9]], expected);
            assert_eq!([result.raw[3], result.raw[7]], [1, 2]);
        }
    }
}
