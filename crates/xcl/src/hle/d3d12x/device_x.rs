//! The console device's own methods and the PC ones whose arguments differ on the console.
//!
//! The console's ID3D12Device has the PC ID3D12Device methods at slots 0–43; from 44 on its
//! vtable holds console extensions (as observed on Fortnite): CreatePlacedResourceX #44,
//! CreateCommandQueueX #48, SetDebugFlagsX #52, CreateDerivedPipelineStateX #55, the
//! pipeline-state (de)serialisation
//! #56 / #57 / #66 / #67, CreateGraphicsPipelineStateX #60, CreateSamplerX #61,
//! CreateComputePipelineStateX #62, CreateCommandListX #73, page pools #74 / #75, the frame
//! clock #78–#81 (`frame`), CreatePipelineState #86 (streams, mesh shaders). A title call
//! of any other extension slot is logged and answered S_OK (`real::class_for`) – not run as
//! the PC method at that index.
//!
//! Console queue and list types add DMA (0x10, the copy engine); console queue priorities
//! are mapped onto PC ones.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::core::GUID;

use super::real;
use crate::util::{HRESULT, S_OK};

const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_ACCESSDENIED: HRESULT = 0x8007_0005u32 as i32;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;

/// D3D12_COMMAND_LIST_TYPE of the console's DMA (copy) engine
pub const XBOX_TYPE_DMA: u32 = 0x10;
/// IID of the console's DMA command list (created as a PC graphics command list)
const IID_XBOX_DMA_COMMAND_LIST: GUID = GUID::from_u128(0x30bc77b9_8379_4b25_95de_b8588c42275d);
const IID_GRAPHICS_COMMAND_LIST: GUID = GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
/// D3D12_INDIRECT_ARGUMENT_TYPE of the console's DispatchX
const XBOX_ARG_DISPATCH_X: u32 = 105;

fn guid_bytes(g: &GUID) -> (u32, u16, u16, [u8; 8]) {
    (g.data1, g.data2, g.data3, g.data4)
}

fn once(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < n
}

/// a console list / allocator type → the PC one
pub fn pc_list_type(t: u32) -> u32 {
    if t == XBOX_TYPE_DMA {
        D3D12_COMMAND_LIST_TYPE_COPY.0 as u32
    } else {
        t
    }
}

/// D3D12_COMMAND_QUEUE_DESC { type, priority, flags, node mask } of the console → PC
pub fn pc_queue_desc(d: [u32; 4]) -> [u32; 4] {
    let priority = match d[1] {
        0 | 100 | 10000 => d[1], // NORMAL, HIGH, GLOBAL_REALTIME
        _ => 100,
    };
    [pc_list_type(d[0]), priority, d[2], if d[3] == 0 { 0 } else { 1 }]
}

/// device #8 CreateCommandQueue / #48 CreateCommandQueueX(desc, riid, queue)
pub(super) unsafe extern "system" fn create_command_queue_x(this: *mut c_void, desc: *const [u32; 4], riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let d = if desc.is_null() { [0; 4] } else { pc_queue_desc(desc.read_unaligned()) };
    real::dev_create_command_queue(this, d.as_ptr() as usize, riid, ppv)
}

/// device #9 CreateCommandAllocator(type, riid, allocator)
pub(super) unsafe extern "system" fn create_command_allocator_x(this: *mut c_void, ty: u32, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    real::dev_create_command_allocator(this, pc_list_type(ty) as usize, riid, ppv)
}

unsafe fn create_list(this: *mut c_void, node: u32, ty: u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let dma = !riid.is_null() && guid_bytes(&*riid) == guid_bytes(&IID_XBOX_DMA_COMMAND_LIST);
    let riid = if dma { &IID_GRAPHICS_COMMAND_LIST as *const GUID } else { riid };
    let (node, ty) = (if node == 0 { 0 } else { 1 }, pc_list_type(ty) as usize);
    let hr = if super::console_list::enabled() {
        real::dev_create_command_list_x(this, node, ty, alloc, pso, riid, ppv)
    } else {
        real::dev_create_command_list(this, node, ty, alloc, pso, riid, ppv)
    };
    if dma && hr >= 0 && !ppv.is_null() {
        real::readopt(*ppv, real::Kind::DmaList);
        if std::env::var_os("XCL_WATCH_DMA").is_some() {
            crate::eh::watch_writes(*ppv as usize);
        }
    }
    // the title's inline D3D12.X code writes into the list object: it gets one of the
    // console's shape (`title_list`)
    if hr >= 0 && !ppv.is_null() && !(*ppv).is_null() && !super::console_list::enabled() {
        super::bind::reset(*ppv, alloc);
        *ppv = super::title_list::wrap(*ppv);
    }
    hr
}

/// device #12 CreateCommandList(node mask, type, allocator, initial pipeline, riid, list):
/// console list types and the DMA list's IID
pub(super) unsafe extern "system" fn create_command_list_pc(this: *mut c_void, node: u32, ty: u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    create_list(this, node, ty, alloc, pso, riid, ppv)
}

/// device #73 CreateCommandListX(params { type @0; node mask @8 }, allocator, initial
/// pipeline, riid, list)
pub(super) unsafe extern "system" fn create_command_list_x(this: *mut c_void, params: *const u32, alloc: usize, pso: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let (ty, node) = if params.is_null() { (0, 1) } else { (params.read_unaligned(), params.add(2).read_unaligned()) };
    create_list(this, node, ty, alloc, pso, riid, ppv)
}

/// device #60 CreateGraphicsPipelineStateX(desc, extension count, extensions, riid, pipeline):
/// the desc is a PC D3D12_GRAPHICS_PIPELINE_STATE_DESC with console shaders; the extensions
/// are not modelled
pub(super) unsafe extern "system" fn create_graphics_pso_x(this: *mut c_void, desc: *const c_void, extensions: u32, _ext: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    if extensions != 0 && once(&N, 1) {
        crate::xlog_info!("D3D12.X: CreateGraphicsPipelineStateX with {extensions} extension(s) – not modelled");
    }
    create_graphics_pso_pc(this, desc, riid, ppv)
}

/// device #10 CreateGraphicsPipelineState(desc, riid, pipeline): remembered as a base for
/// derived pipelines
pub(super) unsafe extern "system" fn create_graphics_pso_pc(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let hr = real::dev_create_graphics_pso(this, desc, riid, ppv);
    if hr >= 0 && !ppv.is_null() {
        note_base(*ppv, desc);
    }
    hr
}

/// size of D3D12_GRAPHICS_PIPELINE_STATE_DESC
const GFX_DESC: usize = 0x290;
/// VS, PS, DS, HS, GS { bytecode; length } in the desc
const STAGES: [usize; 5] = [0x08, 0x18, 0x28, 0x38, 0x48];
const SO_ENTRIES: usize = 0x60;
const INPUT_LAYOUT: usize = 0x228;
const CACHED_PSO: usize = 0x278;
/// CreateDerivedPipelineStateX patch: { u32 part; payload } at this stride (the largest
/// payload is the blend state)
const PATCH: usize = 0x14C;

/// A graphics pipeline desc as bytes, aligned like the struct for the runtime's reads.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct GfxDesc([u8; GFX_DESC]);

/// A title graphics pipeline's description, kept for CreateDerivedPipelineStateX. The input
/// layout is copied; the shaders stay the title's and are checked by hash when a pipeline is
/// derived from it.
struct Base {
    desc: GfxDesc,
    shaders: [u64; 5],
    _elems: Vec<[u8; 32]>,
    _names: Vec<CString>,
}

// the desc points into `_elems` / `_names` and the title's shaders; nothing is mutated
unsafe impl Send for Base {}
unsafe impl Sync for Base {}

static BASES: Mutex<Option<HashMap<usize, Arc<Base>>>> = Mutex::new(None);

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

fn rd64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(d[o..o + 8].try_into().unwrap())
}

/// hash of each stage's bytecode (0: none); None when one is not readable
unsafe fn stage_hashes(d: &[u8; GFX_DESC]) -> Option<[u64; 5]> {
    let mut h = [0; 5];
    for (k, &o) in STAGES.iter().enumerate() {
        let (p, n) = (rd64(d, o) as usize, rd64(d, o + 8) as usize);
        if p != 0 && n != 0 {
            if !crate::gpu::shader::readable(p, n) {
                return None;
            }
            h[k] = crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n));
        }
    }
    Some(h)
}

/// Remember a title graphics pipeline's description (not with stream output).
unsafe fn note_base(pso: *mut c_void, desc: *const c_void) {
    if pso.is_null() || desc.is_null() || !crate::gpu::shader::readable(desc as usize, GFX_DESC) {
        return;
    }
    let mut d = GfxDesc((desc as *const [u8; GFX_DESC]).read_unaligned());
    if rd32(&d.0, SO_ENTRIES) != 0 {
        return;
    }
    let Some(shaders) = stage_hashes(&d.0) else { return };
    let (p, n) = (rd64(&d.0, INPUT_LAYOUT) as usize, rd32(&d.0, INPUT_LAYOUT + 8) as usize);
    let (mut elems, mut names) = (Vec::new(), Vec::new());
    if n > 0 {
        if p == 0 || n > 64 || !crate::gpu::shader::readable(p, 32 * n) {
            return;
        }
        for i in 0..n {
            // D3D12_INPUT_ELEMENT_DESC { SemanticName @0; … }
            let mut e = ((p + 32 * i) as *const [u8; 32]).read_unaligned();
            let s = rd64(&e, 0) as *const c_char;
            let name = if s.is_null() { CString::default() } else { CStr::from_ptr(s).to_owned() };
            e[0..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
            names.push(name);
            elems.push(e);
        }
    }
    let at = if elems.is_empty() { 0 } else { elems.as_ptr() as u64 };
    d.0[INPUT_LAYOUT..INPUT_LAYOUT + 8].copy_from_slice(&at.to_le_bytes());
    d.0[CACHED_PSO..CACHED_PSO + 16].fill(0);
    let base = Arc::new(Base { desc: d, shaders, _elems: elems, _names: names });
    BASES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso as usize, base);
}

/// A pipeline released for good is no longer a base.
pub(super) fn forget_base(pso: usize) {
    if let Some(m) = BASES.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
}

/// Apply one CreateDerivedPipelineStateX patch to a desc: part 0 = depth-stencil state, 1 =
/// rasterizer state, 2 = blend state, 3 = sample mask. Ok(false): part 5 (seen on Fortnite,
/// 0x30 bytes, meaning unknown) – not applied. Err: an unknown part.
fn apply_patch(d: &mut [u8; GFX_DESC], patch: &[u8; PATCH]) -> Result<bool, u32> {
    let (at, n) = match rd32(patch, 0) {
        0 => (0x1F0, 0x34),
        1 => (0x1C4, 0x2C),
        2 => (0x78, 0x148),
        3 => (0x1C0, 4),
        5 => return Ok(false),
        t => return Err(t),
    };
    d[at..at + n].copy_from_slice(&patch[4..4 + n]);
    Ok(true)
}

/// device #55 CreateDerivedPipelineStateX(base, patch count, patches, riid, pipeline): the
/// base pipeline's description with the patched parts, created as a new pipeline (itself a
/// base for further ones)
pub(super) unsafe extern "system" fn create_derived_pso_x(this: *mut c_void, base: *mut c_void, count: u32, patches: *const u8, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    static UNMODELLED: AtomicU64 = AtomicU64::new(0);
    if ppv.is_null() {
        return E_INVALIDARG;
    }
    *ppv = std::ptr::null_mut();
    let found = BASES.lock().unwrap().as_ref().and_then(|m| m.get(&(base as usize)).cloned());
    let Some(b) = found else {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – base pipeline {base:?} unknown (its creation failed, or it has stream output)");
        }
        return E_FAIL;
    };
    if stage_hashes(&b.desc.0) != Some(b.shaders) {
        if once(&N, 4) {
            crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – the shaders of base pipeline {base:?} are gone");
        }
        return E_FAIL;
    }
    let n = count as usize;
    if n > 64 || (n > 0 && (patches.is_null() || !crate::gpu::shader::readable(patches as usize, n * PATCH))) {
        return E_INVALIDARG;
    }
    let mut d = b.desc;
    for i in 0..n {
        let patch = &*(patches.add(i * PATCH) as *const [u8; PATCH]);
        match apply_patch(&mut d.0, patch) {
            Ok(true) => {}
            Ok(false) => {
                if once(&UNMODELLED, 1) {
                    crate::xlog_info!("D3D12.X: CreateDerivedPipelineStateX – patch part 5 not modelled (ignored)");
                }
            }
            Err(t) => {
                if once(&N, 4) {
                    crate::xlog_warn!("D3D12.X: CreateDerivedPipelineStateX – unknown patch part {t}");
                }
                return E_FAIL;
            }
        }
    }
    let desc = d.0.as_ptr() as *const c_void;
    let hr = real::dev_create_graphics_pso(this, desc, riid, ppv);
    if hr >= 0 {
        note_base(*ppv, desc);
    }
    hr
}

/// device #86 CreatePipelineState(stream desc, riid, pipeline)
pub(super) unsafe extern "system" fn create_pipeline_state_x(this: *mut c_void, desc: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    real::dev_create_pipeline_state(this, desc, riid, ppv)
}

/// device #56 / #66 SerializePipelineStateX: not available in a release title
pub(super) unsafe extern "system" fn serialize_pso_x(_this: *mut c_void, _pso: *mut c_void) -> HRESULT {
    E_ACCESSDENIED
}

/// device #57 / #67 Deserialize{Graphics,Compute}PipelineStateX(state, riid, pipeline): the
/// serialised form is the console driver's; the title creates the pipeline from its desc
pub(super) unsafe extern "system" fn deserialize_pso_x(this: *mut c_void, state: *const u8, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    // The FSR4 class of blobs: a session object, then sections (pipeline header, hash, raw
    // RDNA2 code, driver metadata) - reconstruct the pipeline through the recompiler. The
    // graphics variant and other formats stay unsupported (below).
    let hr = super::real::deserialize_compute_x(this, state as *const c_void, _riid, ppv);
    if let Some(hr) = hr {
        return hr;
    }
    if once(&N, 1) {
        crate::xlog_warn!("D3D12.X: DeserializePipelineStateX – console pipeline caches are not supported");
    }
    // XCL_FFX_BLOB_DIR=<dir>: dump every deserialise call's blob completely (header qwords,
    // the sub-table at qword[0], and every (ptr, size) segment) so the console driver-cache
    // format can be reverse-engineered offline. The official FSR4 upscaler deserialises its
    // native pipelines here ('FFX_FSR4_PASS_0'...); the sections it passes match the FXCP
    // blocks embedded in the console DLL (root signature, a 0x1c-byte header, a 0x14-byte
    // hash, the raw RDNA2 code, a metadata section).
    if !state.is_null() {
        if let Some(dir) = std::env::var_os("XCL_FFX_BLOB_DIR") {
            dump_blob(state, N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as usize, &dir);
        }
    }
    if !ppv.is_null() {
        *ppv = std::ptr::null_mut();
    }
    E_NOTIMPL
}

/// Full dump of one deserialise blob. The observed header: qword[0] a sub-table pointer,
/// then (ptr, size) pairs of sections (heap and title memory / GPU VAs, which XCL's UMA
/// resolves directly: console address = CPU address).
unsafe fn dump_blob(state: *const u8, call: usize, dir: &std::ffi::OsStr) {
    let d = std::slice::from_raw_parts(state, 0x80);
    let dir = std::path::Path::new(dir).join(format!("call{call}"));
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("header.bin"), d);
    let q = |o: usize| d.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let mut manifest = String::new();
    manifest.push_str(&format!("header @ {state:p} qwords: {}\n", (0..16).map(|i| q(i * 8).map(|v| format!("{v:#x}")).unwrap_or_default()).collect::<Vec<_>>().join(" ")));
    // the sub-table at qword[0]: dump it and what its pointers lead to
    if let Some(t) = q(0) {
        if (0x1_0000..0x8000_0000_0000).contains(&t) {
            let t = std::slice::from_raw_parts(t as *const u8, 0x100);
            let _ = std::fs::write(dir.join("subtable.bin"), t);
            manifest.push_str(&format!("subtable @ {t:p}\n"));
            for i in 0..16u32 {
                let p = u64::from_le_bytes(t[i as usize * 8..i as usize * 8 + 8].try_into().unwrap());
                manifest.push_str(&format!("  subtable qword[{i}] = {p:#x}\n"));
                if (0x1_0000_0000..0x8000_0000_0000).contains(&p) && p & 7 == 0 {
                    let s = std::slice::from_raw_parts(p as *const u8, 0x180);
                    let _ = std::fs::write(dir.join(format!("sub{i}.bin")), s);
                }
            }
        }
    }
    // (ptr, size) pairs from qword[1]: A, hash, code, metadata
    let mut pair = 1usize;
    let mut i = 1usize;
    while let (Some(ptr), Some(sz)) = (q(i * 8), q(i * 8 + 8)) {
        if i > 8 {
            break;
        }
        i += 2;
        if sz == 0 || sz > 0x4_0000_000 || ptr & 7 != 0 {
            continue;
        }
        // console GPU VAs in the UMA equal the host address (XCL's no-copy windows);
        // anything else in a plausible pointer range is a host allocation
        let host = crate::gpu::uma::get()
            .map(|u| u.window_of(ptr as usize).is_some())
            .unwrap_or(false)
            .then_some(ptr as usize)
            .or_else(|| (0x1_0000_0000..0x8000_0000_0000).contains(&ptr).then_some(ptr as usize));
        let name = pair;
        pair += 1;
        match host {
            Some(h) => {
                let s = std::slice::from_raw_parts(h as *const u8, sz as usize);
                let _ = std::fs::write(dir.join(format!("seg{name}.bin")), s);
                manifest.push_str(&format!("seg{name} (ptr {ptr:#x}, size {sz:#x}) -> {} bytes\n", s.len()));
            }
            None => manifest.push_str(&format!("seg{name} (ptr {ptr:#x}, size {sz:#x}) NOT REACHABLE\n")),
        }
    }
    let _ = std::fs::write(dir.join("manifest.txt"), manifest);
    crate::xlog_warn!("D3D12.X: deserialise call {call} dumped to {}", dir.display());
}

/// device #52 SetDebugFlagsX(flags)
pub(super) unsafe extern "system" fn set_debug_flags_x(_this: *mut c_void, _flags: u32) -> HRESULT {
    S_OK
}

static PAGE_POOLS: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

/// device #74 RegisterPagePoolX(address, pages, index out)
pub(super) unsafe extern "system" fn register_page_pool_x(_this: *mut c_void, va: u64, pages: u32, out: *mut u32) -> HRESULT {
    if va == 0 || pages == 0 {
        return E_INVALIDARG;
    }
    let mut p = PAGE_POOLS.lock().unwrap();
    let i = p.iter().position(|&e| e == (va, pages)).unwrap_or_else(|| {
        p.push((va, pages));
        p.len() - 1
    });
    if !out.is_null() {
        *out = i as u32;
    }
    S_OK
}

/// device #75 UnregisterPagePoolX(index)
pub(super) unsafe extern "system" fn unregister_page_pool_x(_this: *mut c_void, index: u64) -> HRESULT {
    match PAGE_POOLS.lock().unwrap().get_mut(index as usize) {
        Some(e) if e.1 != 0 => {
            *e = (0, 0);
            S_OK
        }
        _ => E_FAIL,
    }
}

/// device #61 CreateSamplerX(desc, destination): the title uses it for anisotropic samplers;
/// the console desc (0x6C bytes) carries the PC fields at other offsets (+0x18 address U / V /
/// W, +0x24 / +0x28 min / max LOD, +0x2C bias, +0x34 max anisotropy, +0x40 comparison); its
/// further anisotropy fields are not modelled
pub(super) unsafe extern "system" fn create_sampler_x(this: *mut c_void, x: *const u8, dest: u64) {
    if x.is_null() || !crate::gpu::shader::readable(x as usize, 0x6C) {
        return;
    }
    let u = |o: usize| (x.add(o) as *const u32).read_unaligned();
    let d = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_ANISOTROPIC,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(u(0x18) as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(u(0x1C) as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(u(0x20) as i32),
        MipLODBias: f32::from_bits(u(0x2C)),
        MaxAnisotropy: u(0x34).clamp(1, 16),
        ComparisonFunc: D3D12_COMPARISON_FUNC(u(0x40) as i32),
        BorderColor: [0.0; 4],
        MinLOD: f32::from_bits(u(0x24)),
        MaxLOD: f32::from_bits(u(0x28)),
    };
    real::dev_create_sampler_x(this, &d, dest);
}

/// the console resource desc (0x38 bytes, D3D12_RESOURCE_DESC) as a PC one: its tile layouts
/// and console flag bits mean nothing to a PC driver
pub fn pc_resource_desc(x: &[u8; 0x38]) -> [u8; 0x38] {
    let mut d = *x;
    let dim = u32::from_le_bytes(d[0..4].try_into().unwrap());
    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
    d[8..16].fill(0); // alignment: the driver's
    let layout = if dim == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 { D3D12_TEXTURE_LAYOUT_ROW_MAJOR.0 } else { D3D12_TEXTURE_LAYOUT_UNKNOWN.0 };
    d[0x2C..0x30].copy_from_slice(&(layout as u32).to_le_bytes());
    d[0x30..0x34].copy_from_slice(&(flags & 0xFF).to_le_bytes());
    d
}

/// device #25 GetResourceAllocationInfo(visible mask, count, descs) → {size, alignment},
/// returned through a hidden pointer. Called the C way – (return buffer, device, …), XDefiant –
/// or the C++ way – (device, return buffer, …): whichever argument is the device is `this`.
/// Textures get the console's size (XG: what the title then places them in); buffers and
/// what XG does not model the PC's.
pub(super) unsafe extern "system" fn get_resource_allocation_info_x(a: *mut c_void, b: *mut c_void, mask: u32, n: u32, descs: *const [u8; 0x38]) -> *mut [u64; 2] {
    let (this, ret) = if real::is_adopted(a) { (a, b as *mut [u64; 2]) } else { (b, a as *mut [u64; 2]) };
    type F = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
    let pc: F = std::mem::transmute(real::orig_fn(this, 25));
    let (mut size, mut align) = (0u64, 0u64);
    for k in 0..n as usize {
        let x = if descs.is_null() { [0u8; 0x38] } else { descs.add(k).read_unaligned() };
        let (s, a) = match crate::hle::xg::Texture::new(&x).ok().filter(|_| u32::from_le_bytes(x[0..4].try_into().unwrap()) != D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32) {
            Some(t) => (t.size(), t.alignment()),
            None => {
                let d = pc_resource_desc(&x);
                let mut r = [0u64; 2];
                pc(this, &mut r, mask, 1, &d);
                (r[0], r[1])
            }
        };
        if s == u64::MAX || a == 0 {
            size = u64::MAX;
            break;
        }
        size = size.div_ceil(a) * a + s;
        align = align.max(a);
    }
    if !ret.is_null() {
        *ret = [size, align];
    }
    ret
}

/// Resources the title placed at an address an earlier one was placed at: on the console they
/// share that memory (aliasing – transient render targets reuse it). Logged (first 16).
fn note_alias(va: usize, x: &[u8; 0x38]) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashMap<usize, String>>> = std::sync::Mutex::new(None);
    static N: AtomicU64 = AtomicU64::new(0);
    let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let what = format!("dimension {} {}x{} format {} flags {:#x}", u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30));
    let mut g = SEEN.lock().unwrap();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    if let Some(prev) = m.insert(va, what.clone()) {
        if N.fetch_add(1, Ordering::Relaxed) < 16 {
            crate::xlog_info!("D3D12.X: CreatePlacedResourceX at {va:#x} ({what}) – aliases an earlier resource there ({prev})");
        }
    }
}

/// device #44 CreatePlacedResourceX(address, desc, initial state, clear value, riid, resource):
/// a resource in title memory. A buffer at a 64 KiB-aligned address in unified memory is that
/// memory (placed in its UMA heap); a texture gets a PC texture kept in step with the memory
/// (`title_textures`: detiled as the console lays it out, uploaded again after CPU writes).
pub(super) unsafe extern "system" fn create_placed_resource_x(this: *mut c_void, va: u64, desc: *const u8, state: u32, clear: *const c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    static N: AtomicU64 = AtomicU64::new(0);
    if desc.is_null() || ppv.is_null() || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return E_INVALIDARG;
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let mut d = pc_resource_desc(&x);
    let buffer = u32::from_le_bytes(d[0..4].try_into().unwrap()) == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
    note_alias(va as usize, &x);
    if !buffer {
        // the R9G9B9E5 translation (see dev_create_committed_resource_x): the console
        // UAV texture of this format has no PC typed UAV - R32_TYPELESS with raw views
        let fmt = u32::from_le_bytes(d[32..36].try_into().unwrap());
        let fl = u32::from_le_bytes(d[48..52].try_into().unwrap());
        if fmt == 67 && fl & 0x6 != 0 {
            d[32..36].copy_from_slice(&65u32.to_le_bytes());
        }
    }
    if buffer {
        if let Some(hr) = real::place_at_title_address(this, va as usize, &d, state, clear, riid, ppv) {
            return hr;
        }
    }
    // render targets, depth buffers, UAV textures: in video memory, aliased by address (`vram`)
    let flags = u32::from_le_bytes(d[0x30..0x34].try_into().unwrap());
    if !buffer && super::vram::gpu_owned(flags) {
        let pc = std::ptr::read_unaligned(d.as_ptr() as *const D3D12_RESOURCE_DESC);
        if let Some(hr) = super::vram::place(this, va as usize, &pc, state, clear, riid, ppv) {
            return hr;
        }
    }
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let hr = real::dev_create_committed_resource(this, &heap as *const _ as usize, 0, d.as_ptr() as usize, state as usize, clear as usize, riid, ppv);
    if hr >= 0 && !buffer && !(*ppv).is_null() {
        static M: AtomicU64 = AtomicU64::new(0);
        match super::title_textures::register(this, *ppv, va as usize, &x, D3D12_RESOURCE_STATES(state as i32)) {
            Ok(()) => return hr,
            Err(e) if once(&M, 4) => {
                let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
                crate::xlog_warn!(
                    "D3D12.X: CreatePlacedResourceX texture at {va:#x} (dimension {} {}x{}x{}, {} mips, format {}, {} samples, layout {:#x}, flags {:#x}): not kept in step with the memory ({e})",
                    u(0), u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()), u(0x18), u16::from_le_bytes(x[0x1C..0x1E].try_into().unwrap()),
                    u16::from_le_bytes(x[0x1E..0x20].try_into().unwrap()), u(0x20), u(0x24), u(0x2C), u(0x30)
                )
            }
            Err(_) => {}
        }
    }
    if hr < 0 || once(&N, 4) {
        let w = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
        crate::xlog_info!(
            "D3D12.X: CreatePlacedResourceX at {va:#x}: dimension {} {}x{} format {} flags {:#x} – a resource of its own ({hr:#x})",
            w(0),
            u64::from_le_bytes(x[0x10..0x18].try_into().unwrap()),
            w(0x18),
            w(0x20),
            w(0x30)
        );
    }
    hr
}

/// device #38 GetCopyableFootprints: a console resource desc the PC rejects (total stays
/// UINT64_MAX) is asked again as the PC desc XCL creates for it
#[allow(clippy::too_many_arguments)]
pub(super) unsafe extern "system" fn get_copyable_footprints_x(
    this: *mut c_void,
    desc: *const u8,
    first: u32,
    num: u32,
    base: u64,
    layouts: *mut c_void,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    type F = unsafe extern "system" fn(*mut c_void, *const u8, u32, u32, u64, *mut c_void, *mut u32, *mut u64, *mut u64);
    let f: F = std::mem::transmute(real::orig_fn(this, 38));
    if desc.is_null() || !crate::gpu::shader::readable(desc as usize, 0x38) {
        return f(this, desc, first, num, base, layouts, rows, row_sizes, total);
    }
    let x: [u8; 0x38] = (desc as *const [u8; 0x38]).read_unaligned();
    let mut probe = u64::MAX;
    let t = if total.is_null() { &mut probe as *mut u64 } else { total };
    *t = u64::MAX;
    f(this, x.as_ptr(), first, num, base, layouts, rows, row_sizes, t);
    if *t == u64::MAX {
        let d = pc_resource_desc(&x);
        f(this, d.as_ptr(), first, num, base, layouts, rows, row_sizes, total);
    }
}

/// device #41 CreateCommandSignature(desc, root signature, riid, signature): the console's
/// DispatchX argument (105) is a PC Dispatch; the root signature only goes with arguments
/// that need one (constants, CBV / SRV / UAV)
pub(super) unsafe extern "system" fn create_command_signature_x(this: *mut c_void, desc: *const u8, root: usize, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if desc.is_null() || ppv.is_null() {
        return E_INVALIDARG;
    }
    let mut pc = [0u8; 24];
    std::ptr::copy_nonoverlapping(desc, pc.as_mut_ptr(), 24);
    let n = u32::from_le_bytes(pc[4..8].try_into().unwrap()) as usize;
    let args = u64::from_le_bytes(pc[8..16].try_into().unwrap()) as *const [u32; 4];
    let mut list: Vec<[u32; 4]> = if args.is_null() || n > 64 { Vec::new() } else { (0..n).map(|i| args.add(i).read_unaligned()).collect() };
    for a in list.iter_mut() {
        if a[0] == XBOX_ARG_DISPATCH_X {
            a[0] = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32;
        }
    }
    let needs_root = list.iter().any(|a| (5..=8).contains(&a[0]));
    pc[8..16].copy_from_slice(&(if list.is_empty() { 0 } else { list.as_ptr() as u64 }).to_le_bytes());
    real::dev_create_command_signature(this, pc.as_ptr() as usize, if needs_root { root } else { 0 }, riid, ppv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_types_and_descs_become_pc_ones() {
        assert_eq!(pc_list_type(XBOX_TYPE_DMA), D3D12_COMMAND_LIST_TYPE_COPY.0 as u32);
        assert_eq!(pc_list_type(2), 2);
        assert_eq!(pc_queue_desc([XBOX_TYPE_DMA, 1, 0, 0x10]), [3, 100, 0, 1]);
        assert_eq!(pc_queue_desc([0, 10000, 1, 0]), [0, 10000, 1, 0]);
        let mut x = [0u8; 0x38];
        x[0] = 1; // buffer
        x[8] = 0x40; // alignment
        x[0x2C] = 0x22; // console layout
        x[0x30..0x34].copy_from_slice(&0x1234u32.to_le_bytes());
        let d = pc_resource_desc(&x);
        assert_eq!((d[8], u32::from_le_bytes(d[0x2C..0x30].try_into().unwrap()), u32::from_le_bytes(d[0x30..0x34].try_into().unwrap())), (0, 1, 0x34));
        x[0] = 3; // texture 2D
        assert_eq!(u32::from_le_bytes(pc_resource_desc(&x)[0x2C..0x30].try_into().unwrap()), 0);
    }

    #[test]
    fn derived_pipeline_patches_land_on_the_desc_parts() {
        use std::mem::{offset_of, size_of};
        type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        assert_eq!(size_of::<G>(), GFX_DESC);
        assert_eq!(STAGES, [offset_of!(G, VS), offset_of!(G, PS), offset_of!(G, DS), offset_of!(G, HS), offset_of!(G, GS)]);
        assert_eq!(SO_ENTRIES, offset_of!(G, StreamOutput) + offset_of!(D3D12_STREAM_OUTPUT_DESC, NumEntries));
        assert_eq!((INPUT_LAYOUT, CACHED_PSO), (offset_of!(G, InputLayout), offset_of!(G, CachedPSO)));
        let parts = [
            (0, offset_of!(G, DepthStencilState), size_of::<D3D12_DEPTH_STENCIL_DESC>()),
            (1, offset_of!(G, RasterizerState), size_of::<D3D12_RASTERIZER_DESC>()),
            (2, offset_of!(G, BlendState), size_of::<D3D12_BLEND_DESC>()),
            (3, offset_of!(G, SampleMask), 4),
        ];
        for (tag, at, n) in parts {
            let mut d = [0u8; GFX_DESC];
            let mut patch = [0xAAu8; PATCH];
            patch[0..4].copy_from_slice(&(tag as u32).to_le_bytes());
            assert_eq!(apply_patch(&mut d, &patch), Ok(true));
            assert!(d[at..at + n].iter().all(|&b| b == 0xAA), "part {tag}");
            assert_eq!(d.iter().filter(|&&b| b == 0xAA).count(), n, "part {tag} writes only its {n} bytes");
        }
        let mut d = [0u8; GFX_DESC];
        let mut patch = [0u8; PATCH];
        patch[0] = 5;
        assert_eq!(apply_patch(&mut d, &patch), Ok(false));
        patch[0] = 4;
        assert_eq!(apply_patch(&mut d, &patch), Err(4));
    }

    #[test]
    fn page_pools_register_once_and_unregister() {
        unsafe {
            let mut i = u32::MAX;
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0x1_0000_0000, 16, &mut i), S_OK);
            let mut j = u32::MAX;
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0x1_0000_0000, 16, &mut j), S_OK);
            assert_eq!(i, j);
            assert_eq!(register_page_pool_x(std::ptr::null_mut(), 0, 16, &mut j), E_INVALIDARG);
            assert_eq!(unregister_page_pool_x(std::ptr::null_mut(), i as u64), S_OK);
            assert_eq!(unregister_page_pool_x(std::ptr::null_mut(), i as u64), E_FAIL);
        }
    }
}
