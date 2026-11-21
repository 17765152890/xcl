//! Import resolution policy (design §5.2).
//!
//! For `dll!function`:
//! 1. routing/aliases → HLE module (+ cross-HLE lookup, + system fallback for the
//!    kernel family), `@system` or `@stub`;
//! 2. a DLL shipped in the game folder → mapped by our loader;
//! 3. a real Windows DLL → LoadLibrary/GetProcAddress;
//! 4. otherwise → automatic logging stub.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use serde::Serialize;

use crate::hle::{registry, HleModule, Route};
use crate::modules;
use crate::pe::{ExportTarget, ImportName};
use crate::stubs;
use crate::util::{dll_key, wide};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "detail")]
pub enum Source {
    /// Implemented by the HLE module the DLL routes to.
    Hle(String),
    /// Implemented by a different HLE module (cross lookup).
    HleOther(String),
    /// Real Windows DLL.
    System(String),
    /// DLL from the game folder, mapped by XCL.
    GameDll(String),
    Stub,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportRecord {
    pub importer: String,
    pub dll: String,
    pub function: String,
    pub delay: bool,
    pub source: Source,
}

static RECORDS: Mutex<Vec<ImportRecord>> = Mutex::new(Vec::new());
static SEARCH_DIRS: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());
static SYS_CACHE: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

pub fn set_search_dirs(dirs: Vec<PathBuf>) {
    *SEARCH_DIRS.write().unwrap() = dirs;
}

pub fn records() -> Vec<ImportRecord> {
    RECORDS.lock().map(|v| v.clone()).unwrap_or_default()
}

fn fname(what: &ImportName) -> String {
    what.to_string()
}

/// Real Windows module handle (cached). Never maps through our loader.
pub fn system_module(dll: &str) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
    let key = dll_key(dll);
    {
        let g = SYS_CACHE.lock().unwrap();
        if let Some(&h) = g.as_ref().and_then(|m| m.get(&key)) {
            return (h != 0).then_some(h);
        }
    }
    let h = unsafe { LoadLibraryW(wide(dll).as_ptr()) } as usize;
    SYS_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, h);
    (h != 0).then_some(h)
}

pub fn system_proc_in(h: usize, what: &ImportName) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    let p = unsafe {
        match what {
            ImportName::Name { name, .. } => {
                let c = std::ffi::CString::new(name.as_str()).ok()?;
                GetProcAddress(h as _, c.as_ptr() as *const u8)
            }
            ImportName::Ordinal(o) => GetProcAddress(h as _, *o as usize as *const u8),
        }
    };
    p.map(|f| f as usize)
}

pub fn system_proc(dll: &str, what: &ImportName) -> Option<usize> {
    system_proc_in(system_module(dll)?, what)
}

fn hle_name_for<'a>(m: &'a HleModule, dll: &str, what: &'a ImportName) -> Option<&'a str> {
    match what {
        ImportName::Name { name, .. } => Some(name.as_str()),
        ImportName::Ordinal(o) => m
            .ordinals
            .get(o)
            .copied()
            .or_else(|| stubs::db().ordinal_name(dll, *o)),
    }
}

/// HLE lookup with cross-module search and (kernel family) system fallback.
fn resolve_hle(m: &HleModule, key: &str, what: &ImportName) -> Option<(usize, Source)> {
    let name = hle_name_for(m, key, what)?;
    if let Some(&a) = m.exports.get(name) {
        return Some((a, Source::Hle(m.name.to_string())));
    }
    if let Some((other, a)) = registry().find_any(name) {
        return Some((a, Source::HleOther(other.name.to_string())));
    }
    let n = ImportName::Name { hint: 0, name: name.to_string() };
    for &c in m.fallback {
        let c = if c == crate::hle::SELF {
            if key == m.name {
                continue;
            }
            key
        } else {
            c
        };
        if let Some(a) = system_proc(c, &n) {
            return Some((a, Source::System(c.to_string())));
        }
    }
    None
}

/// Find a DLL shipped with the game (case-insensitive file name match).
pub fn find_game_dll(name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.is_absolute() && p.is_file() {
        return Some(p.to_path_buf());
    }
    let key = dll_key(name);
    for dir in SEARCH_DIRS.read().ok()?.iter() {
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().to_ascii_lowercase() == key && e.path().is_file() {
                    return Some(e.path());
                }
            }
        }
    }
    None
}

pub fn resolve_export_of_mapped(m: &modules::MappedModule, what: &ImportName, depth: u32) -> Option<usize> {
    let t = match what {
        ImportName::Name { name, .. } => m.exports.by_name.get(name)?,
        ImportName::Ordinal(o) => m.exports.by_ordinal.get(&(*o as u32))?,
    };
    match t {
        ExportTarget::Rva(r) => Some(m.base + *r as usize),
        ExportTarget::Forwarder(f) => {
            if depth > 8 {
                return None;
            }
            let (dll, func) = f.split_once('.')?;
            let what = match func.strip_prefix('#') {
                Some(n) => ImportName::Ordinal(n.parse().ok()?),
                None => ImportName::Name { hint: 0, name: func.to_string() },
            };
            Some(resolve_inner(&dll_key(dll), &what, depth + 1).0)
        }
    }
}

fn stub_for(key: &str, what: &ImportName) -> (usize, Source) {
    let name = fname(what);
    let ret = stubs::db().return_for(key, &name);
    (stubs::make_stub(key, &name, ret), Source::Stub)
}

fn resolve_inner(key: &str, what: &ImportName, depth: u32) -> (usize, Source) {
    let reg = registry();
    match reg.route(key) {
        Some(Route::Stub) => return stub_for(key, what),
        Some(Route::System) => {
            return match system_proc(key, what) {
                Some(a) => (a, Source::System(key.to_string())),
                None => stub_for(key, what),
            }
        }
        Some(Route::Hle(i)) => {
            let m = &reg.modules[*i];
            return resolve_hle(m, key, what).unwrap_or_else(|| stub_for(key, what));
        }
        None => {}
    }
    // DLL shipped with the game → our loader
    if let Some(m) = modules::find_mapped_by_name(key) {
        if let Some(a) = resolve_export_of_mapped(&m, what, depth) {
            return (a, Source::GameDll(key.to_string()));
        }
        return stub_for(key, what);
    }
    if let Some(path) = find_game_dll(key) {
        match crate::loader::load_game_dll(&path) {
            Ok(m) => {
                if let Some(a) = resolve_export_of_mapped(&m, what, depth) {
                    return (a, Source::GameDll(key.to_string()));
                }
                return stub_for(key, what);
            }
            Err(e) => crate::xlog_error!("failed to load {}: {e}", path.display()),
        }
    }
    // Real Windows DLL
    if let Some(a) = system_proc(key, what) {
        return (a, Source::System(key.to_string()));
    }
    // A known console function imported from an unexpected DLL
    if let ImportName::Name { name, .. } = what {
        if let Some((m, a)) = reg.find_any(name) {
            return (a, Source::HleOther(m.name.to_string()));
        }
    }
    stub_for(key, what)
}

/// Resolve one static/delay import and record the outcome for the report.
pub fn resolve_import(dll: &str, what: &ImportName, importer: &str, delay: bool) -> usize {
    let key = dll_key(dll);
    let (addr, source) = resolve_inner(&key, what, 0);
    crate::xlog_debug!("import {importer}: {key}!{what} → {source:?}");
    RECORDS.lock().unwrap().push(ImportRecord {
        importer: importer.to_string(),
        dll: key,
        function: fname(what),
        delay,
        source,
    });
    addr
}

/// Dry-run classification used by `xcl inspect` (does not map game DLLs).
pub fn classify(dll: &str, what: &ImportName) -> Source {
    let key = dll_key(dll);
    let reg = registry();
    match reg.route(&key) {
        Some(Route::Stub) => Source::Stub,
        Some(Route::System) => {
            if system_proc(&key, what).is_some() {
                Source::System(key)
            } else {
                Source::Stub
            }
        }
        Some(Route::Hle(i)) => resolve_hle(&reg.modules[*i], &key, what).map(|x| x.1).unwrap_or(Source::Stub),
        None => {
            if find_game_dll(&key).is_some() {
                return Source::GameDll(key);
            }
            if system_proc(&key, what).is_some() {
                return Source::System(key);
            }
            if let ImportName::Name { name, .. } = what {
                if let Some((m, _)) = reg.find_any(name) {
                    return Source::HleOther(m.name.to_string());
                }
            }
            Source::Stub
        }
    }
}

// ---------------------------------------------------------------------------------------
// Runtime LoadLibrary / GetProcAddress semantics (used by kernelx shims)
// ---------------------------------------------------------------------------------------

/// LoadLibrary as seen by the game. Returns an HMODULE-like value or None.
pub fn load_library(name: &str) -> Option<usize> {
    let key = dll_key(name);
    let reg = registry();
    match reg.route(&key) {
        Some(Route::Hle(i)) => {
            let m = &reg.modules[*i];
            if m.aliases_are_system() && key != m.name {
                // kernel32 / kernelbase / api sets: hand out the real module; GetProcAddress
                // on it still prefers our overrides.
                if let Some(h) = system_module(name) {
                    return Some(h);
                }
            }
            return Some(modules::hle_handle(m.name));
        }
        Some(Route::Stub) => return Some(modules::hle_handle(&key)),
        Some(Route::System) => return system_module(name),
        None => {}
    }
    if let Some(m) = modules::find_mapped_by_name(&key) {
        return Some(m.base);
    }
    let candidate = crate::vfs::get()
        .and_then(|v| v.real_path_for_read(name))
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_game_dll(name));
    if let Some(path) = candidate {
        let under_game = crate::vfs::get().map(|v| v.is_game_path(&path)).unwrap_or(false);
        if under_game || find_game_dll(&key).is_some() {
            match crate::loader::load_game_dll(&path) {
                Ok(m) => return Some(m.base),
                Err(e) => {
                    crate::xlog_error!("LoadLibrary({name}): {e}");
                    return None;
                }
            }
        }
    }
    system_module(name)
}

/// GetProcAddress as seen by the game. Missing functions return None (no stub) so
/// that optional-API probing works.
pub fn get_proc_address(h: usize, what: &ImportName) -> Option<usize> {
    let reg = registry();
    if let Some(name) = modules::hle_name_by_handle(h) {
        let r = match reg.module_by_name(&name).or_else(|| reg.hle_for(&name)) {
            Some(m) => resolve_hle(m, &name, what).map(|x| x.0),
            None => None,
        };
        if r.is_none() {
            // A DLL routed to stubs exists on the console with all of its exports: answer
            // GetProcAddress for any name with a stub (default return value from the db).
            // Unknown DLLs still get NULL so optional-API probing keeps working.
            if matches!(reg.route(&name), Some(Route::Stub)) {
                return Some(stub_for(&name, what).0);
            }
            crate::xlog_warn!("GetProcAddress({name}, {what}) → NULL (not implemented)");
        }
        return r;
    }
    if let Some(m) = modules::find_mapped_by_base(h) {
        return resolve_export_of_mapped(&m, what, 0);
    }
    // A real module: kernel-family modules prefer our overrides.
    let mut buf = [0u16; 260];
    let n = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW(h as _, buf.as_mut_ptr(), 260) };
    if n > 0 {
        let full = String::from_utf16_lossy(&buf[..n as usize]);
        let key = dll_key(&full);
        if let Some(m) = reg.hle_for(&key) {
            if let ImportName::Name { name, .. } = what {
                if let Some(&a) = m.exports.get(name.as_str()) {
                    return Some(a);
                }
            }
        }
    }
    system_proc_in(h, what)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// advapi32 imports go to kernelx: the registry functions the GDK thunks use for the device
    /// type are XCL's, everything else is the real advapi32.
    #[test]
    fn console_user32_and_legacy_dlls_reach_the_pc_functions() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        let (a, s) = resolve_inner("minuserx.dll", &name("PeekMessageW"), 0);
        assert!(a != 0 && matches!(s, Source::System(ref d) if d == "user32.dll"), "{s:?}");
        let (a, s) = resolve_inner("minuserx.dll", &name("CreateWindowExW"), 0);
        assert!(a != 0 && matches!(s, Source::System(_)), "{s:?}");
        let (a, s) = resolve_inner("kernel32legacy.dll", &name("GetProcessAffinityMask"), 0);
        assert!(a != 0 && matches!(s, Source::System(_)), "{s:?}");
        let (a, s) = resolve_inner("minappmodelx.dll", &name("RegisterAppStateChangeNotification"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        let (a, s) = resolve_inner("advapi32legacy.dll", &name("RegOpenKeyW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        // Non-hooked advapi32legacy imports (including LSA policy & privilege APIs) fall through
        // to Windows' advapi32.dll / sechost.dll rather than becoming stubs.
        for sym in ["LsaOpenPolicy", "LsaQueryInformationPolicy", "LsaClose", "LookupPrivilegeValueW"] {
            let (a, s) = resolve_inner("advapi32legacy.dll", &name(sym), 0);
            assert!(a != 0 && matches!(s, Source::System(ref d) if d == "advapi32.dll" || d == "sechost.dll"), "{sym}: {s:?}");
        }
        let h = load_library("advapi32legacy.dll").expect("LoadLibrary(advapi32legacy.dll)");
        assert!(get_proc_address(h, &name("LsaOpenPolicy")).is_some(), "GetProcAddress(advapi32legacy, LsaOpenPolicy)");
    }

    #[test]
    fn advapi32_imports_route_through_kernelx() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        let (a, s) = resolve_inner("advapi32.dll", &name("RegOpenKeyW"), 0);
        assert!(a != 0 && matches!(s, Source::Hle(ref m) if m == crate::hle::KERNELX), "{s:?}");
        let (a, s) = resolve_inner("advapi32.dll", &name("RegSetValueExW"), 0);
        assert!(a != 0 && matches!(s, Source::System(ref d) if d == "advapi32.dll"), "{s:?}");
    }
}
