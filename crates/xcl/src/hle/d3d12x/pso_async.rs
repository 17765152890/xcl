//! Non-blocking pipeline creation. A title creates thousands of pipelines while it loads.
//! Translating a console shader is milliseconds from the cache and a long DXC compile when it
//! is not; the driver's compile of that DXIL is the slow part either way. The title gets a
//! stand-in pipeline (shaders that draw nothing) at once and the real one is built on a
//! worker; binding (`SetPipelineState`, console list replay, the two-pass lookups) uses the
//! real one as soon as it is there. A graphics pipeline draws nothing until then; a compute
//! pipeline is waited for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<(Arc<Pending>, Job)>> {
    static Q: OnceLock<Mutex<mpsc::Sender<(Arc<Pending>, Job)>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<(Arc<Pending>, Job)>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok((p, job)) = next else { return };
                let real = job();
                if real == 0 {
                    FAILED.fetch_add(1, Ordering::Relaxed);
                } else {
                    BUILT.fetch_add(1, Ordering::Relaxed);
                }
                let orphan = {
                    let mut done = p.done.lock().unwrap();
                    let orphan = p.orphan.load(Ordering::Acquire);
                    if !orphan {
                        p.real.store(real, Ordering::Release);
                    }
                    *done = true;
                    p.cv.notify_all();
                    orphan
                };
                // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
                if orphan {
                    release(real);
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    let p = Arc::new(Pending { real: AtomicUsize::new(0), done: Mutex::new(false), cv: Condvar::new(), compute, orphan: Default::default() });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipeline creation returns at once; shader and driver compiles run in the background");
    });
    let _ = queue().lock().unwrap().send((p, job));
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    match entry(pso).map(|p| p.real.load(Ordering::Acquire)) {
        Some(r) if r != 0 => r,
        _ => pso,
    }
}

/// `resolve`, waiting for a pipeline still being built.
pub fn resolve_wait(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// How many uncached driver compiles may run at once. Each one fans out across the GPU
/// driver's own compiler, so a few of them already fill the machine; more of them fight
/// over the same cores and the load takes longer. `XCL_PSO_COMPILES` overrides it.
fn driver_compile_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::env::var("XCL_PSO_COMPILES").ok().and_then(|v| v.parse().ok()).unwrap_or((cores / 4).clamp(2, 4))
    })
}

struct DriverGate {
    /// compiles in the driver, and whether a resource create wants the driver to itself
    running: Mutex<(usize, bool)>,
    cv: Condvar,
}

fn driver_gate() -> &'static DriverGate {
    static G: OnceLock<DriverGate> = OnceLock::new();
    G.get_or_init(|| DriverGate { running: Mutex::new((0, false)), cv: Condvar::new() })
}

/// Hold the driver until the in-flight pipeline compiles finish, and keep new ones from
/// starting. A resource create overlapping a pipeline compile removes the device on this GPU.
pub struct DriverQuiet;

impl Drop for DriverQuiet {
    fn drop(&mut self) {
        let g = driver_gate();
        let mut n = g.running.lock().unwrap();
        n.1 = false;
        g.cv.notify_all();
    }
}

pub fn quiet_driver() -> DriverQuiet {
    let g = driver_gate();
    let mut n = g.running.lock().unwrap();
    n.1 = true;
    if n.0 > 0 {
        crate::xlog_info!("D3D12.X: waiting for {} pipeline compile(s) before creating a resource", n.0);
    }
    while n.0 > 0 {
        n = g.cv.wait(n).unwrap();
    }
    DriverQuiet
}

/// Held across one driver compile of a pipeline the cache did not have.
pub struct DriverGuard;

impl Drop for DriverGuard {
    fn drop(&mut self) {
        let g = driver_gate();
        let mut n = g.running.lock().unwrap();
        n.0 = n.0.saturating_sub(1);
        g.cv.notify_all();
    }
}

/// Block until fewer than `driver_compile_limit` uncached driver compiles are running.
pub fn uncached_compile() -> DriverGuard {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: up to {} new pipelines compile in the driver at once", driver_compile_limit());
    });
    let g = driver_gate();
    let limit = driver_compile_limit();
    let mut n = g.running.lock().unwrap();
    while n.1 || n.0 >= limit {
        n = g.cv.wait(n).unwrap();
    }
    n.0 += 1;
    DriverGuard
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
