//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    pso: usize,
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    #[cfg(test)]
    last_two_pass_result: Option<(usize, usize)>,
}

/// ABI allocations shared by both halves of a translated two-pass pipeline. The underlying
/// unified-memory blocks remain owned by the command-list recording and are released only
/// when its allocator is reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AbiBindings {
    views: [u64; 6],
    pub result_bytes: usize,
}

impl AbiBindings {
    pub fn result_va(&self) -> u64 {
        self.views[1]
    }
}

static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(STATES.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

fn clear_bound_state(s: &mut State, pso: usize) {
    s.layout = None;
    s.args.clear();
    s.layout_gfx = None;
    s.args_gfx.clear();
    s.pso = pso;
    #[cfg(test)]
    {
        s.last_two_pass_result = None;
    }
}

/// The list was reset onto `allocator` and `pso`: the previous recording's memory is the
/// previous allocator's until that one is reset. Root arguments and arena offsets belong to
/// the previous recording and must not survive Reset.
/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous allocator's until that one is reset, and pipeline
/// and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let (old, blocks) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        clear_bound_state(s, initial_pso);
        s.arena = Default::default();
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
}

/// Clear command-list binding state without ending the current recording. Existing ABI
/// allocations remain owned by this recording, but no root layout, root argument, or pipeline
/// binding may leak across ClearState.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| clear_bound_state(s, pso));
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(s) = STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        console_list::hold_until_reset(s.allocator, s.uma);
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(std::sync::Arc::new(console_list::Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(std::sync::Arc::new(console_list::Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(crate) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(crate) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    if super::real::two_pass_of(pso).is_some() {
        reject_two_pass_draw("ExecuteIndirect");
        super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Rejected, super::structural_trace::Category::IncompatiblePipeline, [max, u32::from(gfx), u32::from(compute)]);
        return;
    }
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    if !gfx && !compute {
        pc(this, sig, max, args, args_off, count_buf, count_off);
        super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [max, 0, 0]);
        return;
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Rejected, super::structural_trace::Category::IncompatiblePipeline, [max, u32::from(gfx), u32::from(compute)]);
        return;
    }
    if gfx {
        if prepare_graphics_command(this).is_ok() {
            pc(this, sig, max, args, args_off, count_buf, count_off);
            super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [max, 1, 0]);
        } else {
            super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Skipped, super::structural_trace::Category::PrerequisiteUnavailable, [max, 1, 0]);
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    match bind_abi(this, list, pso, gfx) {
        Ok(()) => {
            count(if gfx { DRAWS } else { DISPATCHES });
            pc(this, sig, max, args, args_off, count_buf, count_off);
            super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [max, 0, 1]);
        }
        Err(e) => {
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
            super::structural_trace::event(this, super::structural_trace::Kind::ExecuteIndirect, super::structural_trace::Outcome::Skipped, super::structural_trace::Category::PrerequisiteUnavailable, [max, 0, 1]);
        }
    }
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    match prepare_graphics_command(this) {
        Ok(list) => {
            list.DrawInstanced(vertices, instances, first_vertex, first_instance);
            super::structural_trace::event(this, super::structural_trace::Kind::Draw, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [vertices, instances, first_vertex]);
        }
        Err(category) => {
            let outcome = if matches!(category, super::structural_trace::Category::IncompatiblePipeline) { super::structural_trace::Outcome::Rejected } else { super::structural_trace::Outcome::Skipped };
            super::structural_trace::event(this, super::structural_trace::Kind::Draw, outcome, category, [vertices, instances, first_vertex]);
        }
    }
}

#[cfg(test)]
pub(crate) unsafe fn draw_instanced_for_test(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    draw_instanced(this, vertices, instances, first_vertex, first_instance)
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    match prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        Ok(list) => emit_prepared_indexed_draw(this, &list, indices, instances, first_index, base_vertex, first_instance),
        Err(category) => {
            let outcome = if matches!(category, super::structural_trace::Category::IncompatiblePipeline) {
                super::structural_trace::Outcome::Rejected
            } else {
                super::structural_trace::Outcome::Skipped
            };
            super::structural_trace::event(this, super::structural_trace::Kind::DrawIndexed, outcome, category, [indices, instances, first_index]);
        }
    }
}

/// Prepare an indexed draw exactly once. Callers own the trace for a rejected preparation so a
/// console wrapper can report its own operation without emitting a duplicate inner event.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Result<ID3D12GraphicsCommandList, super::structural_trace::Category> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    prepare_graphics_command(this)
}

/// Emit an indexed draw after `prepare_indexed_draw` succeeded, without preparing a second time.
pub(super) unsafe fn emit_prepared_indexed_draw(
    this: *mut c_void,
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
    super::structural_trace::event(this, super::structural_trace::Kind::DrawIndexed, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [indices, instances, first_index]);
}

fn reject_two_pass_draw(what: &str) {
    count(DRAWS_SKIPPED);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        crate::xlog_warn!("D3D12.X: {what} with a two-pass pipeline skipped: dispatch dimensions are unavailable");
    }
}

/// Establish the complete state invariant for a graphics command: reject mesh two-pass
/// metadata, bind the translated graphics ABI (or retain an ordinary PC pipeline), then apply
/// pending draw state so a derived depth-bias PSO remains the final pipeline.
unsafe fn prepare_graphics_command(this: *mut c_void) -> Result<ID3D12GraphicsCommandList, super::structural_trace::Category> {
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    if super::real::two_pass_of(pso).is_some() {
        reject_two_pass_draw("draw");
        return Err(super::structural_trace::Category::IncompatiblePipeline);
    }
    let list = graphics_bound(this).ok_or(super::structural_trace::Category::PrerequisiteUnavailable)?;
    super::list_x::prepare_draw(this);
    Ok(list)
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a pipeline still being built in the background is waited for (its stand-in draws nothing)
    if !super::real::is_translated_graphics(super::pso_async::resolve_wait(pso)) {
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve_wait(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        return Some(list); // a PC pipeline: its own root signature and arguments
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    match bind_abi(this, &list, pso, true) {
        Ok(()) => {
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    with(this, |s| s.pso = pso as usize);
    super::real::bind_pipeline(this, pso);
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    dispatch_kind(this, x, y, z, super::structural_trace::Kind::Dispatch);
}

unsafe fn dispatch_kind(this: *mut c_void, x: u32, y: u32, z: u32, kind: super::structural_trace::Kind) {
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        super::structural_trace::event(this, kind, super::structural_trace::Outcome::Rejected, super::structural_trace::Category::IncompatiblePipeline, [x, y, z]);
        return;
    }
    match translated_dispatch(this, list, pso, x, y, z) {
        Ok(()) => {
            count(DISPATCHES);
            super::structural_trace::event(this, kind, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [x, y, z]);
        }
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
            super::structural_trace::event(this, kind, super::structural_trace::Outcome::Skipped, super::structural_trace::Category::PrerequisiteUnavailable, [x, y, z]);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchXRoute {
    TwoPass(crate::gpu::recomp::pso::TwoPassPlan),
    Compute,
    Skip,
}

fn dispatch_x_route(metadata: Option<&super::real::TwoPassPso>, translated_compute: bool, groups: [u32; 3]) -> Result<DispatchXRoute, String> {
    const MAX_GROUPS_PER_DIMENSION: u32 = 65_535;
    if groups.iter().any(|&groups| groups > MAX_GROUPS_PER_DIMENSION) {
        return Err("DispatchX group count exceeds the D3D12 per-dimension limit".into());
    }
    if let Some(metadata) = metadata {
        let record_dwords = u32::try_from(metadata.record.record_dwords()).map_err(|_| "two-pass record size exceeds u32")?;
        let plan = crate::gpu::recomp::pso::TwoPassPlan::checked(groups, record_dwords, metadata.mesh_groups_per_record)?;
        if plan.mesh_groups > MAX_GROUPS_PER_DIMENSION {
            return Err("two-pass mesh group count exceeds the D3D12 DispatchMesh limit".into());
        }
        return Ok(DispatchXRoute::TwoPass(plan));
    }
    Ok(if translated_compute { DispatchXRoute::Compute } else { DispatchXRoute::Skip })
}

/// Console DispatchX. Mesh/geometry pipelines created as checked two-pass pipelines use the
/// explicit compute -> UAV barrier -> pass-through mesh sequence; ordinary compute pipelines
/// retain the Dispatch path above.
pub(super) unsafe fn dispatch_x(this: *mut c_void, x: u32, y: u32, z: u32) {
    let held_pso = with(this, |s| s.pso);
    let real_pso = super::pso_async::resolve_wait(held_pso);
    let metadata = super::real::two_pass_of(real_pso);
    let route = dispatch_x_route(metadata.as_deref(), super::real::is_translated_compute(real_pso), [x, y, z]);
    match route {
        Ok(DispatchXRoute::Compute) | Ok(DispatchXRoute::Skip) => {
            dispatch_kind(this, x, y, z, super::structural_trace::Kind::DispatchX)
        }
        Ok(DispatchXRoute::TwoPass(plan)) => {
            if plan.total_groups == 0 {
                super::structural_trace::event(this, super::structural_trace::Kind::DispatchX, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [x, y, z]);
                return;
            }
            let result = (|| -> Result<(), String> {
                let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).ok_or("command list")?;
                // Validate the required interface before allocating an ABI record, so a cast
                // failure cannot consume command-list arena space.
                let _: ID3D12GraphicsCommandList6 = list.cast().map_err(|_| "two-pass recording requires ID3D12GraphicsCommandList6")?;
                let bindings = prepare_two_pass_abi(this, &plan)?.ok_or("non-empty two-pass plan produced no ABI bindings")?;
                if !record_two_pass(list, metadata.as_deref(), real_pso, &plan, Some(&bindings))? {
                    return Err("non-empty two-pass submission recorded no commands".into());
                }
                Ok(())
            })();
            match result {
                Ok(()) => {
                    count(DISPATCHES);
                    super::structural_trace::event(this, super::structural_trace::Kind::DispatchX, super::structural_trace::Outcome::Accepted, super::structural_trace::Category::None, [x, y, z]);
                }
                Err(e) => {
                    count(DISPATCHES_SKIPPED);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                        crate::xlog_warn!("D3D12.X: DispatchX with a two-pass pipeline skipped: {e}");
                    }
                    super::structural_trace::event(this, super::structural_trace::Kind::DispatchX, super::structural_trace::Outcome::Skipped, super::structural_trace::Category::PrerequisiteUnavailable, [x, y, z]);
                }
            }
        }
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: DispatchX plan rejected: {e}");
            }
            super::structural_trace::event(this, super::structural_trace::Kind::DispatchX, super::structural_trace::Outcome::Rejected, super::structural_trace::Category::InvalidArguments, [x, y, z]);
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    bind_abi(this, list, pso, false)?;
    list.Dispatch(x, y, z);
    Ok(())
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool) -> Result<(), String> {
    let bindings = prepare_abi(this, gfx, DEFAULT_RESULT_BYTES)?;
    bind_prepared_abi(list, pso, gfx, &bindings)?;
    #[cfg(test)]
    if gfx && GRAPHICS_PREP_TEST_STATE.load(Ordering::SeqCst) == 1 {
        GRAPHICS_PREP_TEST_STATE.store(2, Ordering::SeqCst);
    }
    Ok(())
}

#[cfg(test)]
static GRAPHICS_PREP_TEST_STATE: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) struct GraphicsPrepTestGuard;

#[cfg(test)]
impl Drop for GraphicsPrepTestGuard {
    fn drop(&mut self) { GRAPHICS_PREP_TEST_STATE.store(0, Ordering::SeqCst); }
}

#[cfg(test)]
pub(crate) fn observe_graphics_preparation_for_test() -> GraphicsPrepTestGuard {
    assert_eq!(GRAPHICS_PREP_TEST_STATE.swap(1, Ordering::SeqCst), 0);
    GraphicsPrepTestGuard
}

#[cfg(test)]
pub(crate) fn note_prepare_draw_for_test() {
    let state = GRAPHICS_PREP_TEST_STATE.load(Ordering::SeqCst);
    if state != 0 && state != 3
        && GRAPHICS_PREP_TEST_STATE.compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        GRAPHICS_PREP_TEST_STATE.store(4, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(crate) fn graphics_preparation_order_for_test() -> u64 {
    GRAPHICS_PREP_TEST_STATE.load(Ordering::SeqCst)
}

const DEFAULT_RESULT_BYTES: usize = 256;

/// Allocate one ABI record for a checked two-pass plan. Empty dispatches allocate nothing;
/// callers may bind the returned record first as compute and then as graphics without
/// allocating or changing its result buffer.
pub(super) unsafe fn prepare_two_pass_abi(this: *mut c_void, plan: &crate::gpu::recomp::pso::TwoPassPlan) -> Result<Option<AbiBindings>, String> {
    if plan.total_groups == 0 {
        return Ok(None);
    }
    if plan.result_bytes == 0 {
        return Err("two-pass result buffer is empty".into());
    }
    prepare_abi(this, true, plan.result_bytes).map(Some)
}

unsafe fn prepare_abi(this: *mut c_void, gfx: bool, result_bytes: usize) -> Result<AbiBindings, String> {
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let windows = super::real::uma_window_vas();
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    let (layout, args) = with(this, |s| if gfx { (s.layout_gfx.clone(), s.args_gfx.clone()) } else { (s.layout.clone(), s.args.clone()) });
    // 32-bit console addresses: window 0; small pieces share the command list's arena,
    // while caller-sized two-pass results get a dedicated block when necessary.
    let copy = |bytes: &[u8]| -> Result<usize, String> {
        let p = with(this, |s| s.arena.take(u, bytes.len().max(16), 256, &mut s.uma))
            .ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        Ok(p as usize)
    };
    let zeroed = |bytes: usize| -> Result<usize, String> {
        if bytes == 0 { return Err("ABI allocation is empty".into()); }
        let p = with(this, |s| s.arena.take(u, bytes.max(16), 256, &mut s.uma))
            .ok_or("unified memory is full")?;
        std::ptr::write_bytes(p, 0, bytes);
        Ok(p as usize)
    };
    let built = (|| -> Result<(usize, usize), String> {
        let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { l.inline } as usize).min(args.len());
        let table_at = copy(&args[inline..])?;
        let mut io = match layout.as_ref() {
            Some(l) if gfx => console_list::user_data_gfx(l, &args, table_at as u64),
            Some(l) => console_list::user_data(l, &args, table_at as u64),
            None => vec![table_at as u32, (table_at >> 32) as u32],
        };
        io.resize(crate::gpu::recomp::KERNEL_SGPRS, 0);
        let io_at = copy(&io.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>())?;
        let res_at = zeroed(result_bytes)?;
        Ok((io_at, res_at))
    })();
    let (io_at, res_at) = built?;
    #[cfg(test)]
    with(this, |s| s.last_two_pass_result = Some((res_at, result_bytes)));
    let va = |p: usize| super::real::uma_va_of(p).unwrap_or(first);
    let views = abi_views(va(io_at), va(res_at), &windows)?;
    Ok(AbiBindings { views, result_bytes })
}

#[cfg(test)]
pub(crate) struct DispatchRouteTestGuard(*mut c_void);

#[cfg(test)]
impl Drop for DispatchRouteTestGuard {
    fn drop(&mut self) { forget(self.0); }
}

#[cfg(test)]
pub(crate) fn install_dispatch_route_for_test(list: *mut c_void, pso: usize) -> DispatchRouteTestGuard {
    with(list, |state| {
        state.pso = pso;
        state.last_two_pass_result = None;
    });
    DispatchRouteTestGuard(list)
}

#[cfg(test)]
pub(crate) fn two_pass_result_for_test(list: *mut c_void) -> Option<(usize, usize)> {
    with(list, |state| state.last_two_pass_result)
}

fn abi_views(io: u64, result: u64, windows: &[u64]) -> Result<[u64; 6], String> {
    let first = *windows.first().ok_or("unified memory is not open on the GPU")?;
    Ok([io, result, windows.first().copied().unwrap_or(first), windows.get(1).copied().unwrap_or(first), windows.get(2).copied().unwrap_or(first), windows.get(3).copied().unwrap_or(first)])
}

/// Bind a previously allocated ABI record. Reusing `bindings` is what keeps the result UAV
/// identical across the compute and mesh passes.
pub(super) unsafe fn bind_prepared_abi(list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool, bindings: &AbiBindings) -> Result<(), String> {
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    let rs = super::real::pso_root_signature(real).or_else(|| super::real::abi_rs_cached().map(|p| p as usize)).ok_or("no XCL ABI root signature")?;
    let rs = rs as *mut c_void;
    let rs = ID3D12RootSignature::from_raw_borrowed(&rs).ok_or("root signature")?;
    if gfx {
        list.SetGraphicsRootSignature(rs);
        for (k, &v) in bindings.views.iter().enumerate() {
            list.SetGraphicsRootUnorderedAccessView(k as u32, v);
        }
    } else {
        list.SetComputeRootSignature(rs);
        for (k, &v) in bindings.views.iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, v);
        }
    }
    let real = real as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real).ok_or("pipeline")?;
    list.SetPipelineState(pc);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwoPassCommand {
    BindCompute { pso: usize, root: usize, result_va: u64 },
    Dispatch([u32; 3]),
    UavBarrier { result_va: u64 },
    BindMesh { pso: usize, root: usize, result_va: u64 },
    DispatchMesh(u32),
}

fn two_pass_commands(
    metadata: Option<&super::real::TwoPassPso>,
    mesh_pso: usize,
    plan: &crate::gpu::recomp::pso::TwoPassPlan,
    bindings: Option<&AbiBindings>,
    command_list6: bool,
) -> Result<Vec<TwoPassCommand>, String> {
    if plan.total_groups == 0 {
        if plan.result_bytes != 0 || plan.mesh_groups != 0 || !plan.dispatch_groups.contains(&0) {
            return Err("inconsistent empty two-pass plan".into());
        }
        return Ok(Vec::new());
    }
    if !command_list6 { return Err("two-pass recording requires ID3D12GraphicsCommandList6".into()); }
    let metadata = metadata.ok_or("two-pass pipeline metadata is missing")?;
    let bindings = bindings.ok_or("two-pass ABI bindings are missing")?;
    if metadata.compute == 0 || metadata.compute_rs == 0 || metadata.mesh_rs == 0 || mesh_pso == 0 {
        return Err("two-pass pipeline metadata contains a null object".into());
    }
    if bindings.result_va() == 0 || bindings.result_bytes != plan.result_bytes {
        return Err("two-pass ABI bindings do not match the plan".into());
    }
    let record_dwords = u32::try_from(metadata.record.record_dwords()).map_err(|_| "two-pass record size exceeds u32")?;
    let checked = crate::gpu::recomp::pso::TwoPassPlan::checked(plan.dispatch_groups, record_dwords, metadata.mesh_groups_per_record)?;
    if checked != *plan { return Err("two-pass plan does not match pipeline metadata".into()); }
    let va = bindings.result_va();
    Ok(vec![
        TwoPassCommand::BindCompute { pso: metadata.compute, root: metadata.compute_rs, result_va: va },
        TwoPassCommand::Dispatch(plan.dispatch_groups),
        TwoPassCommand::UavBarrier { result_va: va },
        TwoPassCommand::BindMesh { pso: mesh_pso, root: metadata.mesh_rs, result_va: va },
        TwoPassCommand::DispatchMesh(plan.mesh_groups),
    ])
}

#[allow(dead_code)] // used only when an explicit two-pass caller is wired in
unsafe fn bind_prepared_abi_with_root(
    list: &ID3D12GraphicsCommandList,
    pso: usize,
    root: usize,
    gfx: bool,
    bindings: &AbiBindings,
) -> Result<(), String> {
    let root = root as *mut c_void;
    let root = ID3D12RootSignature::from_raw_borrowed(&root).ok_or("two-pass root signature")?;
    if gfx {
        list.SetGraphicsRootSignature(root);
        for (k, &v) in bindings.views.iter().enumerate() { list.SetGraphicsRootUnorderedAccessView(k as u32, v); }
    } else {
        list.SetComputeRootSignature(root);
        for (k, &v) in bindings.views.iter().enumerate() { list.SetComputeRootUnorderedAccessView(k as u32, v); }
    }
    let pso = super::pso_async::resolve_wait(pso) as *mut c_void;
    let pso = ID3D12PipelineState::from_raw_borrowed(&pso).ok_or("two-pass pipeline")?;
    list.SetPipelineState(pso);
    Ok(())
}

/// Record an explicitly planned two-pass submission. DispatchX supplies the checked plan and
/// prepared ABI; ordinary draw interception intentionally does not infer dispatch dimensions.
pub(super) unsafe fn record_two_pass(
    list: &ID3D12GraphicsCommandList,
    metadata: Option<&super::real::TwoPassPso>,
    mesh_pso: usize,
    plan: &crate::gpu::recomp::pso::TwoPassPlan,
    bindings: Option<&AbiBindings>,
) -> Result<bool, String> {
    if plan.total_groups == 0 {
        two_pass_commands(metadata, mesh_pso, plan, bindings, false)?;
        return Ok(false);
    }
    // Cast before validating the trace so failure cannot leave a partially recorded pass.
    let list6: ID3D12GraphicsCommandList6 = list.cast().map_err(|_| "two-pass recording requires ID3D12GraphicsCommandList6")?;
    let commands = two_pass_commands(metadata, mesh_pso, plan, bindings, true)?;
    let bindings = bindings.unwrap();
    for command in commands {
        match command {
            TwoPassCommand::BindCompute { pso, root, .. } => bind_prepared_abi_with_root(list, pso, root, false, bindings)?,
            TwoPassCommand::Dispatch([x, y, z]) => list.Dispatch(x, y, z),
            TwoPassCommand::UavBarrier { .. } => list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
            }]),
            TwoPassCommand::BindMesh { pso, root, .. } => bind_prepared_abi_with_root(list, pso, root, true, bindings)?,
            TwoPassCommand::DispatchMesh(groups) => list6.DispatchMesh(groups, 1, 1),
        }
    }
    Ok(true)
}

/// Test-only bridge which keeps the live GPU runner on the exact production recording path.
/// The borrowed D3D12 objects remain owned by the runner.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) unsafe fn record_two_pass_for_test(
    list: &ID3D12GraphicsCommandList,
    compute: &ID3D12PipelineState,
    compute_rs: &ID3D12RootSignature,
    mesh: &ID3D12PipelineState,
    mesh_rs: &ID3D12RootSignature,
    record: crate::gpu::recomp::MsInterface,
    mesh_groups_per_record: u32,
    plan: &crate::gpu::recomp::pso::TwoPassPlan,
    views: [u64; 6],
) -> Result<bool, String> {
    let metadata = std::mem::ManuallyDrop::new(super::real::TwoPassPso {
        compute: compute.as_raw() as usize,
        compute_rs: compute_rs.as_raw() as usize,
        mesh_rs: mesh_rs.as_raw() as usize,
        record,
        mesh_groups_per_record,
        kind: "test",
    });
    let bindings = AbiBindings { views, result_bytes: plan.result_bytes };
    record_two_pass(list, Some(&metadata), mesh.as_raw() as usize, plan, Some(&bindings))
}

#[cfg(test)]
mod abi_tests {
    use super::*;

    static DISPATCH_X_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn abi_views_preserve_available_windows_and_repeat_the_first() {
        assert_eq!(abi_views(11, 22, &[100]).unwrap(), [11, 22, 100, 100, 100, 100]);
        assert_eq!(abi_views(11, 22, &[100, 200, 300]).unwrap(), [11, 22, 100, 200, 300, 100]);
        assert!(abi_views(11, 22, &[]).is_err());
    }

    #[test]
    fn prepared_bindings_reuse_the_exact_result_view() {
        let bindings = AbiBindings { views: [11, 22, 100, 200, 300, 400], result_bytes: 4096 };
        let compute = bindings;
        let graphics = bindings;
        assert_eq!(compute.result_va(), 22);
        assert_eq!(graphics.result_va(), compute.result_va());
        assert_eq!(graphics.result_bytes, 4096);
        assert_eq!(DEFAULT_RESULT_BYTES, 256);
    }

    #[test]
    fn empty_two_pass_plan_allocates_nothing_without_requiring_uma() {
        let plan = crate::gpu::recomp::pso::TwoPassPlan::checked([u32::MAX, 0, u32::MAX], 16, 2).unwrap();
        let prepared = unsafe { prepare_two_pass_abi(std::ptr::null_mut(), &plan) }.unwrap();
        assert!(prepared.is_none());
    }

    #[test]
    fn reset_transfers_all_prepared_blocks_to_the_previous_allocator() {
        let list = 0x1234usize as *mut c_void;
        with(list, |state| { state.allocator = 7; state.uma = vec![11, 22, 33]; });
        reset(list, 8, 9);
        assert_eq!(super::super::console_list::take_pending_for_test(7), vec![11, 22, 33]);
        with(list, |state| {
            assert_eq!((state.allocator, state.pso), (8, 9));
            assert!(state.uma.is_empty());
        });
        STATES.lock().unwrap().as_mut().unwrap().remove(&(list as usize));
    }

    fn two_pass_metadata() -> std::mem::ManuallyDrop<super::super::real::TwoPassPso> {
        std::mem::ManuallyDrop::new(super::super::real::TwoPassPso {
            compute: 10,
            compute_rs: 20,
            mesh_rs: 30,
            record: crate::gpu::recomp::MsInterface::default(),
            mesh_groups_per_record: 2,
            kind: "test",
        })
    }

    #[test]
    fn dispatch_x_prefers_two_pass_metadata_over_compute_classification() {
        let metadata = two_pass_metadata();
        let expected = crate::gpu::recomp::pso::TwoPassPlan::checked([2, 3, 1], 4, 2).unwrap();
        assert_eq!(dispatch_x_route(Some(&metadata), true, [2, 3, 1]).unwrap(), DispatchXRoute::TwoPass(expected));
        assert_eq!(dispatch_x_route(None, true, [2, 3, 1]).unwrap(), DispatchXRoute::Compute);
        assert_eq!(dispatch_x_route(None, false, [2, 3, 1]).unwrap(), DispatchXRoute::Skip);
    }

    #[test]
    fn dispatch_x_zero_groups_are_a_checked_noop_and_overflow_fails_before_recording() {
        let metadata = two_pass_metadata();
        let DispatchXRoute::TwoPass(empty) = dispatch_x_route(Some(&metadata), false, [65_535, 0, 65_535]).unwrap() else {
            panic!("two-pass metadata must select the two-pass route")
        };
        assert_eq!((empty.total_groups, empty.result_bytes, empty.mesh_groups), (0, 0, 0));
        assert!(dispatch_x_route(Some(&metadata), false, [65_536, 0, 1]).is_err());
        assert!(dispatch_x_route(Some(&metadata), false, [u32::MAX, 2, 1]).is_err());
    }

    #[test]
    fn dispatch_x_enforces_compute_and_mesh_group_boundaries() {
        let mut metadata = two_pass_metadata();
        metadata.mesh_groups_per_record = 1;
        let DispatchXRoute::TwoPass(at_limit) = dispatch_x_route(Some(&metadata), false, [65_535, 1, 1]).unwrap() else {
            panic!("expected a two-pass route")
        };
        assert_eq!(at_limit.mesh_groups, 65_535);
        assert!(dispatch_x_route(Some(&metadata), false, [65_536, 1, 1]).is_err());

        metadata.mesh_groups_per_record = 2;
        let DispatchXRoute::TwoPass(below_mesh_limit) = dispatch_x_route(Some(&metadata), false, [32_767, 1, 1]).unwrap() else {
            panic!("expected a two-pass route")
        };
        assert_eq!(below_mesh_limit.mesh_groups, 65_534);
        assert!(dispatch_x_route(Some(&metadata), false, [32_768, 1, 1]).is_err());
    }

    #[test]
    fn console_dispatch_x_entry_uses_registered_two_pass_metadata_without_allocating_for_zero_groups() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list_key = 0x7fff_1234usize;
        let list = list_key as *mut c_void;
        let metadata = super::super::real::TwoPassPso {
            compute: 0,
            compute_rs: 20,
            mesh_rs: 30,
            record: crate::gpu::recomp::MsInterface::default(),
            mesh_groups_per_record: 2,
            kind: "test",
        };
        let _registration = super::super::real::install_two_pass_for_test(list_key, metadata);
        with(list, |state| state.pso = list_key);
        assert!(super::super::real::two_pass_of(list_key).is_some());
        let skipped_before = COUNTS[DISPATCHES_SKIPPED].load(Ordering::SeqCst);

        // This is the function installed in console command-list slot 78. The zero Y count
        // proves the real entry performs registry lookup and checked routing, then exits before
        // borrowing a native command list or allocating its shared ABI/result record. Ordinary
        // compute routing would classify the synthetic PSO as untranslated and increment the
        // skipped-dispatch counter, making this assertion mutation-sensitive to the slot wiring.
        unsafe { super::super::list_x::dispatch_x(list, 65_535, 0, 65_535, 0) };
        assert_eq!(COUNTS[DISPATCHES_SKIPPED].load(Ordering::SeqCst), skipped_before);
        with(list, |state| assert!(state.uma.is_empty()));
        STATES.lock().unwrap().as_mut().unwrap().remove(&list_key);
    }

    #[test]
    fn console_draw_indexed_x_entry_rejects_two_pass_before_native_index_binding() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list_key = 0x7fff_5678usize;
        let list = list_key as *mut c_void;
        let metadata = super::super::real::TwoPassPso {
            compute: 0,
            compute_rs: 20,
            mesh_rs: 30,
            record: crate::gpu::recomp::MsInterface::default(),
            mesh_groups_per_record: 2,
            kind: "test",
        };
        let _registration = super::super::real::install_two_pass_for_test(list_key, metadata);
        with(list, |state| state.pso = list_key);
        super::super::structural_trace::start_test_capture();
        let skipped_before = COUNTS[DRAWS_SKIPPED].load(Ordering::SeqCst);

        // This is the function installed in console command-list slot 81. A fake command-list
        // pointer is intentional: two-pass rejection must happen before any native vtable call
        // or temporary index-buffer binding. Removing the ABI preparation from the slot makes
        // this test dereference the fake list instead of incrementing the skipped-draw count.
        unsafe {
            super::super::list_x::draw_indexed_x(
                list,
                0x1000,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT,
                3,
                0,
            )
        };
        assert_eq!(COUNTS[DRAWS_SKIPPED].load(Ordering::SeqCst), skipped_before + 1);
        let trace = super::super::structural_trace::finish_test_capture();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["event"], "draw_indexed_x");
        assert_eq!(trace[0]["outcome"], "rejected");
        assert_eq!(trace[0]["category"], "incompatible_pipeline");
        with(list, |state| assert!(state.uma.is_empty()));
        STATES.lock().unwrap().as_mut().unwrap().remove(&list_key);
    }

    #[test]
    fn standard_graphics_entries_reject_two_pass_before_native_calls() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list_key = 0x7fff_6789usize;
        let list = list_key as *mut c_void;
        let metadata = super::super::real::TwoPassPso {
            compute: 0,
            compute_rs: 20,
            mesh_rs: 30,
            record: crate::gpu::recomp::MsInterface::default(),
            mesh_groups_per_record: 2,
            kind: "test",
        };
        let _registration = super::super::real::install_two_pass_for_test(list_key, metadata);
        with(list, |state| state.pso = list_key);
        super::super::structural_trace::start_test_capture();
        let skipped_before = COUNTS[DRAWS_SKIPPED].load(Ordering::SeqCst);

        // The fake pointer proves rejection precedes command-list borrowing or vtable access.
        unsafe {
            draw_instanced(list, 3, 1, 0, 0);
            draw_indexed_instanced(list, 3, 1, 0, 0, 0);
            execute_indirect(list, 0, 1, 0, 0, 0, 0);
        }
        assert_eq!(COUNTS[DRAWS_SKIPPED].load(Ordering::SeqCst), skipped_before + 3);
        let trace = super::super::structural_trace::finish_test_capture();
        assert_eq!(trace.len(), 3);
        for record in &trace {
            assert_eq!(record["outcome"], "rejected");
            assert_eq!(record["category"], "incompatible_pipeline");
        }
        STATES.lock().unwrap().as_mut().unwrap().remove(&list_key);
    }

    #[test]
    fn console_draw_indexed_x_entry_rejects_invalid_index_format_with_exact_category() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list = 0x7fff_6790usize as *mut c_void;
        super::super::structural_trace::start_test_capture();
        unsafe {
            super::super::list_x::draw_indexed_x(
                list,
                0x1000,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
                3,
                0,
            );
        }
        let trace = super::super::structural_trace::finish_test_capture();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["event"], "draw_indexed_x");
        assert_eq!(trace[0]["outcome"], "rejected");
        assert_eq!(trace[0]["category"], "invalid_arguments");
    }

    #[test]
    fn console_draw_indexed_x_zero_work_entry_preserves_trace_sequence() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list = 0x7fff_6791usize as *mut c_void;
        super::super::structural_trace::start_test_capture();
        unsafe {
            super::super::list_x::draw_indexed_x(
                list,
                0x1000,
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT,
                0,
                0,
            );
        }
        let trace = super::super::structural_trace::finish_test_capture();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["sequence"], 0);
        assert_eq!(trace[0]["event"], "draw_indexed_x");
        assert_eq!(trace[0]["outcome"], "accepted");
        assert_eq!(trace[0]["category"], "none");
        assert_eq!((trace[0]["a"].as_u64(), trace[0]["b"].as_u64(), trace[0]["c"].as_u64()), (Some(0), Some(1), Some(0)));
    }

    fn seed_stale_binding_state(list: *mut c_void, pso: usize) {
        with(list, |state| {
            state.layout = Some(Arc::new(Layout::default()));
            state.args = vec![1, 2, 3, 4];
            state.layout_gfx = Some(Arc::new(Layout::default()));
            state.args_gfx = vec![5, 6, 7, 8];
            state.pso = pso;
            state.last_two_pass_result = Some((0x1234, 64));
        });
    }

    #[test]
    fn reset_sets_initial_pipeline_and_drops_previous_root_state() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list_key = 0x7fff_789ausize;
        let list = list_key as *mut c_void;
        seed_stale_binding_state(list, 0x1111);

        reset(list, 0x2222, 0x3333);
        with(list, |state| {
            assert_eq!((state.allocator, state.pso), (0x2222, 0x3333));
            assert!(state.layout.is_none() && state.args.is_empty());
            assert!(state.layout_gfx.is_none() && state.args_gfx.is_empty());
            assert_eq!(state.last_two_pass_result, None);
            assert!(state.uma.is_empty());
        });
        STATES.lock().unwrap().as_mut().unwrap().remove(&list_key);
    }

    #[test]
    fn clear_state_replaces_pipeline_and_roots_but_keeps_recording_memory() {
        let _serial = DISPATCH_X_TEST_LOCK.lock().unwrap();
        let list_key = 0x7fff_89abusize;
        let list = list_key as *mut c_void;
        seed_stale_binding_state(list, 0x1111);
        // A synthetic value is safe here because the test removes State directly rather than
        // handing it to the allocator-retirement path.
        with(list, |state| {
            state.allocator = 0x2222;
            state.uma = vec![0x4444];
        });

        unsafe { super::super::list_x::on_clear_state(list, 0x3333) };
        with(list, |state| {
            assert_eq!((state.allocator, state.pso), (0x2222, 0x3333));
            assert!(state.layout.is_none() && state.args.is_empty());
            assert!(state.layout_gfx.is_none() && state.args_gfx.is_empty());
            assert_eq!(state.last_two_pass_result, None);
            assert_eq!(state.uma, vec![0x4444]);
        });
        STATES.lock().unwrap().as_mut().unwrap().remove(&list_key);
    }

    #[test]
    fn two_pass_trace_has_exact_order_and_one_result_view() {
        let metadata = two_pass_metadata();
        let plan = crate::gpu::recomp::pso::TwoPassPlan::checked([2, 3, 1], 4, 2).unwrap();
        let bindings = AbiBindings { views: [11, 22, 100, 200, 300, 400], result_bytes: plan.result_bytes };
        assert_eq!(two_pass_commands(Some(&metadata), 40, &plan, Some(&bindings), true).unwrap(), vec![
            TwoPassCommand::BindCompute { pso: 10, root: 20, result_va: 22 },
            TwoPassCommand::Dispatch([2, 3, 1]),
            TwoPassCommand::UavBarrier { result_va: 22 },
            TwoPassCommand::BindMesh { pso: 40, root: 30, result_va: 22 },
            TwoPassCommand::DispatchMesh(12),
        ]);
    }

    #[test]
    fn two_pass_trace_fails_closed_before_recording() {
        let metadata = two_pass_metadata();
        let plan = crate::gpu::recomp::pso::TwoPassPlan::checked([1, 1, 1], 4, 2).unwrap();
        let bindings = AbiBindings { views: [11, 22, 100, 200, 300, 400], result_bytes: plan.result_bytes };
        assert!(two_pass_commands(None, 40, &plan, Some(&bindings), true).is_err());
        assert!(two_pass_commands(Some(&metadata), 40, &plan, None, true).is_err());
        assert!(two_pass_commands(Some(&metadata), 40, &plan, Some(&bindings), false).is_err());
        assert!(two_pass_commands(Some(&metadata), 0, &plan, Some(&bindings), true).is_err());
        let wrong_size = AbiBindings { result_bytes: plan.result_bytes + 4, ..bindings };
        assert!(two_pass_commands(Some(&metadata), 40, &plan, Some(&wrong_size), true).is_err());
        let wrong_plan = crate::gpu::recomp::pso::TwoPassPlan::checked([1, 1, 1], 8, 2).unwrap();
        let wrong_bindings = AbiBindings { result_bytes: wrong_plan.result_bytes, ..bindings };
        assert!(two_pass_commands(Some(&metadata), 40, &wrong_plan, Some(&wrong_bindings), true).is_err());
    }

    #[test]
    fn empty_two_pass_trace_is_a_noop_and_overflow_is_rejected() {
        let empty = crate::gpu::recomp::pso::TwoPassPlan::checked([u32::MAX, 0, u32::MAX], 4, 2).unwrap();
        assert!(two_pass_commands(None, 0, &empty, None, false).unwrap().is_empty());
        assert!(crate::gpu::recomp::pso::TwoPassPlan::checked([u32::MAX, 2, 1], 4, 2).is_err());
    }
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

#[cfg(test)]
mod binding_state_tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }
}
