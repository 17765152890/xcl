//! Console shaders in pipeline states: a shader blob a title passes to Create*PipelineState
//! (DXBC container with XDAT/XBCH/FSSM, no usable DXIL) becomes DXIL through the recompiler,
//! and the pipeline gets XCL's shader ABI root signature instead of the title's one:
//! root UAVs u0 `io` (initial SGPRs = user data), u1 `res`, u2..u5 memory windows, and u6
//! reserved for logical-wave dispatcher scratch, plus the directly indexed descriptor heaps
//! (images / samplers). Mapping the title's root arguments
//! onto that ABI at draw time is the next step (needs the console runtime's user-data layout).

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

use super::{dxc, interface, translate, Mode, Options};
use crate::gpu::xbox;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Pixel,
    Compute,
}

/// A console (xbox-native) shader container?
pub fn is_console(blob: &[u8]) -> bool {
    xbox::is_xbox(blob)
}

/// Memory windows of the title's unified memory (1 without it).
pub fn mem_windows() -> u32 {
    crate::gpu::uma::get().map_or(1, |u| u.shader_windows() as u32)
}

/// A static sampler of a translated program: (S#, comparison) – see `Translation`.
pub type StaticSampler = ([u32; 4], bool);

/// A console shader as PC DXIL and the static samplers it needs in its root signature.
pub struct Compiled {
    pub dxil: Vec<u8>,
    pub samplers: Vec<StaticSampler>,
}

/// Translate a console shader for `stage` to signed DXIL (DXC, disk-cached); its static
/// samplers take registers from `sampler_base` (space 1).
/// Fails when the blob is not a console shader or the program is not fully translated.
pub fn console_to_dxil(blob: &[u8], stage: Stage, sampler_base: u32) -> Result<Compiled, String> {
    let (mut t, profile) = console_translation(blob, stage, sampler_base)?;
    if let Some([r, g, b]) = (stage == Stage::Pixel).then(|| debug_color(blob)).flatten() {
        t.hlsl = t.hlsl.replacen("    return xo;\n}", &format!("    xo.t0 = float4({r:.3}, {g:.3}, {b:.3}, 1.0);\n    return xo;\n}}"), 1);
    }
    // `XCL_DEBUG_LOD0=hash,…`: those pixel shaders sample at LOD 0 instead of with implicit
    // derivatives – tells derivative (helper lane) trouble apart from the rest
    if stage == Stage::Pixel && std::env::var("XCL_DEBUG_LOD0").is_ok_and(|v| v.split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).any(|h| h == crate::gpu::shader::fnv64(blob))) {
        t.hlsl = sample_at_lod0(&t.hlsl);
    }
    // `XCL_DEBUG_PS_SUB=hash:from=>to||from=>to…` (one pixel shader): every `from` in its HLSL
    // becomes `to`
    if let Some(subs) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_SUB").ok()).flatten().and_then(|v| {
        let (h, rest) = v.split_once(':')?;
        (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| rest.to_string())
    }) {
        for sub in subs.split("||") {
            let Some((from, to)) = sub.split_once("=>") else { continue };
            let n = t.hlsl.matches(from).count();
            crate::xlog_info!("[SHADER] XCL_DEBUG_PS_SUB: {n} occurrence(s) of {from:?} replaced");
            t.hlsl = t.hlsl.replace(from, to);
        }
    }
    // `XCL_DEBUG_PS_EXPR=hash:expression` (one pixel shader): its target 0 gets that HLSL
    // expression, e.g. `float4(xi.a1.rgb, 1)` – an input or a register made visible
    if let Some(expr) = (stage == Stage::Pixel).then(|| std::env::var("XCL_DEBUG_PS_EXPR").ok()).flatten().and_then(|v| {
        let (h, e) = v.split_once(':')?;
        (u64::from_str_radix(h.trim(), 16).ok()? == crate::gpu::shader::fnv64(blob)).then(|| e.to_string())
    }) {
        t.hlsl = t.hlsl.replacen("    if (!xcl_live) discard;\n    return xo;\n}", &format!("    xo.t0 = {expr};\n    return xo;\n}}"), 1);
    }
    let dxil = dxc::compile_cached(&t.hlsl, "main", profile)?.0;
    Ok(Compiled { dxil, samplers: t.static_samplers })
}

/// Every `X_T.Sample(args)` of generated HLSL as `X_T.SampleLevel(args, 0.0f)`.
fn sample_at_lod0(hlsl: &str) -> String {
    let mut out = String::with_capacity(hlsl.len());
    let mut rest = hlsl;
    while let Some(at) = rest.find("_T.Sample(") {
        let open = at + "_T.Sample(".len();
        out.push_str(&rest[..at]);
        out.push_str("_T.SampleLevel(");
        let mut depth = 1;
        let mut end = open;
        for (k, ch) in rest[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + k;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&rest[open..end]);
        out.push_str(", 0.0f)");
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `XCL_DEBUG_MAGENTA=hash[:rrggbb],…` (dump hashes, hex; magenta without a colour): those
/// pixel shaders write that opaque colour to target 0 wherever they run – where a draw lands,
/// told apart from what it shades (a colour per shader: which one drew a pixel).
fn debug_color(blob: &[u8]) -> Option<[f32; 3]> {
    static S: std::sync::OnceLock<Vec<(u64, [f32; 3])>> = std::sync::OnceLock::new();
    let wanted = S.get_or_init(|| {
        std::env::var("XCL_DEBUG_MAGENTA")
            .unwrap_or_default()
            .split(',')
            .filter_map(|item| {
                let item = item.trim();
                let (h, c) = item.split_once(':').unwrap_or((item, "ff00ff"));
                let c = u32::from_str_radix(c, 16).ok()?;
                let ch = |shift: u32| ((c >> shift) & 0xFF) as f32 / 255.0;
                Some((u64::from_str_radix(h, 16).ok()?, [ch(16), ch(8), ch(0)]))
            })
            .collect()
    });
    let h = crate::gpu::shader::fnv64(blob);
    wanted.iter().find(|(w, _)| *w == h).map(|(_, c)| *c)
}

/// The HLSL (and its DXC profile) `console_to_dxil` compiles.
pub fn console_to_hlsl(blob: &[u8], stage: Stage) -> Result<(String, &'static str), String> {
    console_translation(blob, stage, 0).map(|(t, p)| (t.hlsl, p))
}

fn console_translation(blob: &[u8], stage: Stage, sampler_base: u32) -> Result<(super::Translation, &'static str), String> {
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    // the hardware variant for this pipeline slot: VS (not the ES variant used with GS), PS
    let want = match stage {
        Stage::Vertex => 4,
        Stage::Pixel => 5,
        Stage::Compute => 6,
    };
    let f = sh
        .fssm
        .iter()
        .find(|f| f.stages.first().is_some_and(|s| s.hw == want))
        .ok_or_else(|| format!("no hardware variant {}", xbox::hw_stage_name(want)))?;
    let mode = interface(blob, &sh, f).ok_or("unknown shader interface")?;
    let profile = match (&mode, stage) {
        (Mode::Vertex(_), Stage::Vertex) => "vs_6_6",
        (Mode::Pixel(_), Stage::Pixel) => "ps_6_6",
        (Mode::Compute(_), Stage::Compute) => "cs_6_6",
        _ => return Err("the shader stage does not match its slot in the PSO".into()),
    };
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    let t = translate(code, &Options { mode, wave64: super::detect_wave64(code), static_sampler_base: sampler_base, mem_windows: mem_windows(), ..Default::default() })?;
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    Ok((t, profile))
}

/// A console geometry stage as XCL runs it, in two passes: the console program as a compute
/// shader writes one record per thread group (vertices + primitives, `record`) into `res`,
/// then a pass-through mesh shader rasterises the records with the pixel shader.
pub struct TwoPass {
    /// compute pass (cs_6_6 DXIL)
    pub cs: Vec<u8>,
    /// pass-through mesh shader (ms_6_6 DXIL)
    pub ms: Vec<u8>,
    pub record: super::MsInterface,
    /// mesh groups of the pass-through per record (> 1: the record is over D3D12's mesh
    /// output limit and goes out in runs of triangles)
    pub mesh_groups_per_record: u32,
    /// static samplers of the compute pass
    pub samplers: Vec<StaticSampler>,
}

/// Checked resource and dispatch sizing for a two-pass geometry submission.
///
/// Callers must supply explicit compute dispatch dimensions. Draw arguments are not
/// interchangeable with dispatch dimensions and must not be used to construct this plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoPassPlan {
    pub dispatch_groups: [u32; 3],
    pub total_groups: u32,
    pub result_bytes: usize,
    pub mesh_groups: u32,
}

impl TwoPassPlan {
    pub fn checked(dispatch_groups: [u32; 3], record_dwords: u32, mesh_groups_per_record: u32) -> Result<Self, String> {
        if record_dwords == 0 {
            return Err("two-pass record size is zero".into());
        }
        if mesh_groups_per_record == 0 {
            return Err("two-pass mesh split is zero".into());
        }
        let total_groups = if dispatch_groups.contains(&0) {
            0
        } else {
            dispatch_groups.into_iter().try_fold(1u32, |product, groups| product.checked_mul(groups))
                .ok_or("two-pass dispatch group count overflow")?
        };
        let result_dwords = total_groups.checked_mul(record_dwords).ok_or("two-pass result size overflow")?;
        let result_bytes_u32 = result_dwords.checked_mul(4).ok_or("two-pass result byte size overflow")?;
        let result_bytes = usize::try_from(result_bytes_u32).map_err(|_| "two-pass result size exceeds the host address space")?;
        let mesh_groups = total_groups.checked_mul(mesh_groups_per_record).ok_or("two-pass mesh group count overflow")?;
        Ok(Self { dispatch_groups, total_groups, result_bytes, mesh_groups })
    }
}

impl TwoPass {
    /// Plan a submission using this pipeline's generated record and pass-through split.
    pub fn plan(&self, dispatch_groups: [u32; 3]) -> Result<TwoPassPlan, String> {
        let record_dwords = u32::try_from(self.record.record_dwords()).map_err(|_| "two-pass record size exceeds u32")?;
        TwoPassPlan::checked(dispatch_groups, record_dwords, self.mesh_groups_per_record)
    }
}

fn finish_two_pass(t: super::Translation, record: super::MsInterface, ps: Option<&[u8]>) -> Result<TwoPass, String> {
    if !t.complete() {
        let missing: Vec<&String> = t.unsupported.keys().collect();
        return Err(format!("incomplete translation, missing: {missing:?}"));
    }
    let cs = dxc::compile_cached(&t.hlsl, "main", "cs_6_6")?.0;
    let ps_inputs = ps.map(|p| crate::gpu::dxbc::signatures(p).0);
    let ms = dxc::compile_cached(&super::emit::mesh_passthrough_for(&record, ps_inputs.as_deref()), "main", "ms_6_6")?.0;
    let mesh_groups_per_record = super::emit::passthrough_split(&record, super::emit::passthrough_elements(&record, ps_inputs.as_deref()));
    Ok(TwoPass { cs, ms, record, mesh_groups_per_record, samplers: t.static_samplers })
}

/// The mesh-shader (NGG) program of a console MS container; the pass-through carries what
/// the pixel shader `ps` (its container: console or PC) reads.
pub fn console_mesh(blob: &[u8], ps: Option<&[u8]>, mesh_wave32: bool) -> Result<TwoPass, String> {
    let sh = xbox::parse(blob).ok_or("not a console shader container")?;
    let f = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 7)).ok_or("no NGG hardware variant")?;
    let mode = interface(blob, &sh, f).ok_or("unknown MS interface (e.g. lines / points)")?;
    let Mode::Mesh(ms) = &mode else { return Err("the container is not an MS".into()) };
    let record = ms.clone();
    let p = f.programs.first().ok_or("FSSM without a program")?;
    let code = &f.data[p.offset..p.offset + p.len];
    finish_two_pass(super::translate_mesh(code, ms.clone(), super::detect_wave64(code), mesh_wave32)?, record, ps)
}

/// Whether the adapter's supported wave range requires split geometry execution.
pub fn geometry_uses_wave32(wave_min: u32, wave_max: u32) -> bool {
    !(wave_min <= 64 && wave_max >= 64)
}

pub fn geometry_uses_wave32_for_device(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            &mut options as *mut _ as _,
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_err()
    {
        return true;
    }
    geometry_uses_wave32(options.WaveLaneCountMin, options.WaveLaneCountMax)
}

/// A legacy GS pipeline: the ES variant of a console VS + the GS as one program.
pub fn console_geometry(vs: &[u8], gs: &[u8], ps: Option<&[u8]>, geometry_wave32: bool) -> Result<TwoPass, String> {
    let (code, g) = super::geometry_program(vs, gs)?;
    let record = g.record();
    let translation = super::translate_geometry(&code, g, geometry_wave32)?;
    finish_two_pass(translation, record, ps)
}

/// XCL shader ABI root signature (see the module docs); `graphics` allows an input layout.
pub fn abi_root_signature(dev: &ID3D12Device, graphics: bool) -> Result<ID3D12RootSignature, String> {
    abi_root_signature_with(dev, graphics, &[])
}

/// A static sampler (register `k`, space 1) from its S#; static samplers have only three
/// border colours (others: transparent black).
pub fn static_sampler_desc(k: u32, (w, comparison): &StaticSampler) -> D3D12_STATIC_SAMPLER_DESC {
    let d = crate::gpu::desc::sampler_d3d12(w, *comparison);
    let border = match d.BorderColor {
        [0.0, 0.0, 0.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        [1.0, 1.0, 1.0, 1.0] => D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        _ => D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
    };
    D3D12_STATIC_SAMPLER_DESC {
        Filter: d.Filter,
        AddressU: d.AddressU,
        AddressV: d.AddressV,
        AddressW: d.AddressW,
        MipLODBias: d.MipLODBias,
        MaxAnisotropy: d.MaxAnisotropy,
        ComparisonFunc: d.ComparisonFunc,
        BorderColor: border,
        MinLOD: d.MinLOD,
        MaxLOD: d.MaxLOD,
        ShaderRegister: k,
        RegisterSpace: 1,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// The ABI root signature with the pipeline's static samplers (registers 0.., space 1).
pub fn abi_root_signature_with(dev: &ID3D12Device, graphics: bool, samplers: &[StaticSampler]) -> Result<ID3D12RootSignature, String> {
    let statics: Vec<D3D12_STATIC_SAMPLER_DESC> = samplers.iter().enumerate().map(|(k, s)| static_sampler_desc(k as u32, s)).collect();
    unsafe {
        // u0 io, u1 res, u2..u5 memory windows, u6 logical-wave dispatcher scratch.
        // Native shaders do not reference u6 and therefore require no extra binding.
        let params: Vec<D3D12_ROOT_PARAMETER1> = (0..=super::mesh_dispatch::SCRATCH_UAV_REGISTER)
            .map(|i| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: i, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            })
            .collect();
        let mut flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        if graphics {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: statics.len() as u32,
                    pStaticSamplers: if statics.is_empty() { std::ptr::null() } else { statics.as_ptr() },
                    Flags: flags,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(|e| format!("{e}"))?;
        let blob = blob.ok_or("root signature")?;
        dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("CreateRootSignature: {e}"))
    }
}

#[cfg(test)]
mod plan_tests {
    use super::TwoPassPlan;

    #[test]
    fn two_pass_plan_sizes_multidimensional_and_empty_dispatches() {
        let plan = TwoPassPlan::checked([2, 3, 4], 5, 2).unwrap();
        assert_eq!(plan.dispatch_groups, [2, 3, 4]);
        assert_eq!(plan.total_groups, 24);
        assert_eq!(plan.result_bytes, 24 * 5 * 4);
        assert_eq!(plan.mesh_groups, 48);

        let empty = TwoPassPlan::checked([0, 7, 9], 5, 3).unwrap();
        assert_eq!(empty.total_groups, 0);
        assert_eq!(empty.result_bytes, 0);
        assert_eq!(empty.mesh_groups, 0);

        for dispatch_groups in [[0, u32::MAX, u32::MAX], [u32::MAX, 0, u32::MAX], [u32::MAX, u32::MAX, 0]] {
            let empty = TwoPassPlan::checked(dispatch_groups, 5, 3).unwrap();
            assert_eq!(empty.total_groups, 0);
            assert_eq!(empty.result_bytes, 0);
            assert_eq!(empty.mesh_groups, 0);
        }
    }

    #[test]
    fn two_pass_plan_rejects_invalid_metadata_and_each_overflow() {
        assert!(TwoPassPlan::checked([1, 1, 1], 0, 1).is_err());
        assert!(TwoPassPlan::checked([1, 1, 1], 1, 0).is_err());
        assert!(TwoPassPlan::checked([u32::MAX, 2, 1], 1, 1).is_err());
        assert!(TwoPassPlan::checked([u32::MAX, 1, 1], 2, 1).is_err());
        assert!(TwoPassPlan::checked([u32::MAX / 4 + 1, 1, 1], 1, 1).is_err());
        assert!(TwoPassPlan::checked([u32::MAX, 1, 1], 1, 2).is_err());
    }
}
