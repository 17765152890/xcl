//! Console-built FidelityFX middleware as officially distributed by Microsoft:
//! `Kits/FidelityFX/signedbin` of https://github.com/microsoft/Xbox-GDK-Samples ships
//! AMD-signed Scarlett builds of the loader and the FSR4 upscaler
//! (amd_fidelityfx_loader_xbox.dll / amd_fidelityfx_upscaler_xbox.dll) next to their
//! PC twins (amd_fidelityfx_*_dx12.dll). Anyone can fetch those legally, which makes
//! this suite a publicly reproducible regression harness for console DLL
//! initialization, import routing and exported-function execution - no private game
//! files needed, unlike the retail DLLs of `console_dlls_e2e.rs`.
//!
//! `XCL_SAMPLE_FFX` = folder containing the two `*_xbox.dll` files; skipped when unset.
//!
//! Semantics pinned down from the official binaries' machine code:
//! * the loader's `ffxXboxSetUseFsr4Lite(bool)` / `ffxXboxGetUseFsr4Lite()` first make
//!   an internal ensure-loaded call, then forward to the upscaler's same-named exports
//!   via GetProcAddress - so the pair shares the upscaler's global state, and the very
//!   first call dynamically loads the upscaler (under XCL: through the LoadLibrary
//!   shim, mapped + relocated + TLS carrier + DllMain);
//! * the upscaler's pair is self-contained: Set stores CL into a global byte, Get
//!   zero-extends that byte into EAX - a no-argument bool round trip.

use std::path::PathBuf;
use std::sync::OnceLock;

use xcl::modules::MappedModule;
use xcl::pe::{ExportTarget, ImportName};
use xcl::resolver::{classify, Source};

fn setup() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("XCL_SAMPLE_FFX")?);
        let gb: usize = std::env::var("XCL_TEST_MEM_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        cap_memory(gb << 30);
        xcl::stubs::init_db(&Default::default());
        xcl::hle::init(&xcl::config::Config::default());
        xcl::resolver::set_search_dirs(vec![dir.clone()]);
        let _ = xcl::eh::install_pc_hook();
        xcl::eh::install_crash_handler();
        Some(dir)
    })
    .as_ref()
}

/// Limit this test process's committed memory (see `console_dlls_e2e.rs`).
fn cap_memory(bytes: usize) {
    use windows_sys::Win32::System::JobObjects::*;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = bytes;
        let ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32) != 0
            && AssignProcessToJobObject(job, GetCurrentProcess()) != 0;
        if !ok {
            eprintln!("cap_memory: could not set the memory limit");
        }
    }
}

fn load(dir: &std::path::Path, name: &str) -> std::sync::Arc<MappedModule> {
    xcl::loader::load_game_dll(&dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// These tests share XCL's process-global module table; serialize them so the
/// "not loaded yet" observation and the module identity stay deterministic
/// under `cargo test`'s default parallel execution.
static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    SERIALIZE.lock().unwrap_or_else(|e| e.into_inner())
}

fn proc(m: &MappedModule, name: &str) -> usize {
    match m.exports.by_name.get(name) {
        Some(ExportTarget::Rva(r)) => m.base + *r as usize,
        other => panic!("{}!{name}: {other:?}", m.name),
    }
}

/// Both officially distributed console DLLs map, resolve their imports and run DllMain.
#[test]
fn official_ffx_console_dlls_map_and_initialise() {
    let Some(dir) = setup() else { return };
    let _guard = serialize();
    for name in ["amd_fidelityfx_loader_xbox.dll", "amd_fidelityfx_upscaler_xbox.dll"] {
        let m = load(dir, name);
        assert!(m.initialized.load(std::sync::atomic::Ordering::Acquire), "{name}: DllMain");
    }
}

/// The console-only imports of the upscaler route to XCL's console emulation, never to
/// Windows: d3d12_xs to the D3D12.X HLE, xmem's allocator family to kernelx (cross
/// lookup), PIX instrumentation to the stub module. `classify` dry-runs the same
/// registry the loader uses for static imports.
#[test]
fn upscaler_console_imports_route_to_console_emulation() {
    let Some(_) = setup() else { return };
    let f = |dll: &str, name: &str| classify(dll, &ImportName::Name { hint: 0, name: name.to_string() });
    assert_eq!(
        f("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer"),
        Source::Hle("d3d12_xs.dll".into())
    );
    for n in ["XMemAlloc", "XMemFree", "XMemGetWorkingSetStatistics"] {
        assert_eq!(f("xmem.dll", n), Source::HleOther("kernelx.dll".into()), "xmem!{n}");
    }
    assert_eq!(f("PIXEvt.dll", "PIXEventsReplaceBlock"), Source::Stub);
}

/// Real console code execution, no GPU. The loader's Xbox-only FSR4-Lite accessors
/// lazily load the upscaler on their first call (observed under XCL: the dynamic
/// LoadLibrary goes through the resolver - mapped at a relocated base, TLS carrier
/// installed, DllMain run), then forward to the upscaler's exports via GetProcAddress.
/// The flag round-trips through the upscaler's global state: Set stores CL into a
/// global byte, Get zero-extends it into EAX (machine code of both official binaries).
#[test]
fn ffx_xbox_fsr4_lite_accessors_execute_console_code() {
    let Some(dir) = setup() else { return };
    let _guard = serialize();
    unsafe {
        let loader = load(dir, "amd_fidelityfx_loader_xbox.dll");
        assert!(
            xcl::modules::find_mapped_by_name("amd_fidelityfx_upscaler_xbox.dll").is_none(),
            "the loader has not loaded the upscaler yet"
        );
        let l_set: extern "C" fn(bool) = std::mem::transmute(proc(&loader, "ffxXboxSetUseFsr4Lite"));
        let l_get: extern "C" fn() -> u32 = std::mem::transmute(proc(&loader, "ffxXboxGetUseFsr4Lite"));
        // the first call lazily loads the upscaler through XCL's LoadLibrary shim
        assert_eq!(l_get(), 0, "upscaler default: FSR4 Full (not Lite)");
        let up = xcl::modules::find_mapped_by_name("amd_fidelityfx_upscaler_xbox.dll")
            .expect("the loader's first accessor call loaded the upscaler via XCL");
        assert!(up.initialized.load(std::sync::atomic::Ordering::Acquire), "upscaler DllMain ran");
        // the loader forwards to the upscaler's own exports: one shared global state
        let u_set: extern "C" fn(bool) = std::mem::transmute(proc(&up, "ffxXboxSetUseFsr4Lite"));
        let u_get: extern "C" fn() -> u32 = std::mem::transmute(proc(&up, "ffxXboxGetUseFsr4Lite"));
        l_set(true);
        assert_eq!((l_get(), u_get()), (1, 1), "the forwarded Set stores the upscaler's global");
        u_set(false);
        assert_eq!((l_get(), u_get()), (0, 0), "the direct Set and the forwarded one agree");
        u_set(true);
        assert_eq!(l_get(), 1, "Get reads the same global through both modules");
    }
}
