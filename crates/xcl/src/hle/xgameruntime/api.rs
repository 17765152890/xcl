//! The XGameRuntime thunk protocol – how GDK titles and GDK libraries (XCurl, libHttpClient)
//! reach the runtime. They link a static thunk library that loads `xgameruntime.dll`, calls
//! `InitializeApiImpl(Ex/Ex2)` and gets each API family as a COM-style object from
//! `QueryApiImpl(clsid, iid)`; an API call is a call through a slot of its vtable (0..2 are
//! IUnknown). The by-name exports in `mod.rs` are for titles that import them directly.
//!
//! XCL answers itself for what must stay offline and XCL-managed – XUser (one local user),
//! XGameSave (saves in the XCL data folder), XPackage, XGame (title id from the game's
//! MicrosoftGame.Config), XSystem ids, the XGameUi dialogs that do not take a user –, refuses
//! XStore and XUser device association (the thunk then returns its "API not available"
//! error), and passes everything else to the PC's own XGameRuntime
//! (`System32\xgameruntime.dll`: the same classes under the same GUIDs) – XTaskQueue / XAsync,
//! XThread, XNetworking, XGameInvite… Once the PC runtime is in use, the queues and async blocks
//! are its own, so XCL's async APIs run on its XAsync (`xasync::use_pc`).
//!
//! Classes and slots: from the thunks in the console DLLs (which (clsid, iid) and slot every
//! thunk uses, its argument and result widths) and the PC runtime (API names its slots refer
//! to); slots follow the order of the GDK headers. Slots XCL does not implement are logged and
//! listed in the report (`xgameruntime.unimplemented`).

#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use super::gamesave::*;
use super::*;
use crate::util::{E_INVALIDARG, HRESULT, S_OK};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Guid(pub u32, pub u16, pub u16, pub [u8; 8]);

impl Guid {
    pub const fn from_u128(v: u128) -> Guid {
        Guid((v >> 96) as u32, (v >> 80) as u16, (v >> 64) as u16, (v as u64).to_be_bytes())
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let d = &self.3;
        write!(f, "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{}", self.0, self.1, self.2, d[0], d[1], d[2..].iter().map(|b| format!("{b:02x}")).collect::<String>())
    }
}

const fn g(v: u128) -> Guid {
    Guid::from_u128(v)
}

pub const XUSER: Guid = g(0x01acd177_91f9_4763_a38e_ccbb55ce32e0);
const XUSER_IID: Guid = g(0x26f3c674_a2fe_44fa_b6c4_a323bc94ff53);
/// second interface of the XUser object: XUserGetGamertag with a gamertag component
const XUSER_GAMERTAG_IID: Guid = g(0xcef4fac0_7676_4a94_a119_4c43f9eb5b74);
const XUSER_DEVICE: Guid = g(0x7d824997_10dc_45ab_86b7_2737767c0bf1);
pub const XGAMESAVE: Guid = g(0x704c3f58_e629_4cc2_b197_30511b996fe2);
const XGAMESAVE_IIDS: [Guid; 2] = [g(0x1bfff3af_f14a_40a3_8e35_9ada906593f9), g(0xab4ae4fb_6508_4950_a032_45fd4bf8c43b)];
pub const XPACKAGE: Guid = g(0xaf406016_e850_4aa8_a88d_2f3dcb9dac7e);
const XPACKAGE_IID: Guid = g(0xe2a4734b_2f4a_456d_aa8f_d065e04fb209);
pub const XGAME: Guid = g(0x973a344e_24bf_4d0f_8457_56c534892b29);
const XGAME_IID: Guid = g(0x2549f142_6419_4a06_97b5_931aab7c2f34);
pub const XSYSTEM: Guid = g(0xe349bd1a_fc20_4e40_b99c_4178cc6b409f);
const XSYSTEM_IID: Guid = g(0x1861cf2e_e18b_4834_a9f5_b4a4e6efb4cf);
pub const XGAMEUI: Guid = g(0xdfcd4649_4ff8_4043_ba07_35d607df98b0);
const XGAMEUI_IID: Guid = g(0xeaf669df_5542_4590_99a3_8dc061f837cc);
const XSTORE: Guid = g(0x0dd112ac_7c24_448c_b92b_3960fb5bd30c);
/// XPersistentLocalStorage class candidate from HANDOFF.md
pub const XPLS: Guid = g(0x7f0fe8b8_0000_0000_0000_000000000000);
/// XAsync + XTaskQueue + XThread (iid = clsid)
pub const XASYNC: Guid = g(0x073b7dcb_1fcf_4030_94be_e3c9eb623428);

/// classes that reach Xbox Live / the Store or take XCL's user handles: not available
const REFUSED: [(Guid, &str); 2] = [(XSTORE, "XStore"), (XUSER_DEVICE, "XUser (devices)")];

/// names of the other classes, for the log
fn family(c: &Guid) -> &'static str {
    match c.0 {
        0x073b7dcb => "XAsync/XTaskQueue/XThread",
        0x37e56907 => "XNetworking",
        0x95fd18d2 => "XGameInvite",
        0x8836fe87 => "XGameRuntimeFeature",
        0x0a2192aa => "(console only – stubs on PC)",
        _ => "?",
    }
}

const REGDB_E_CLASSNOTREG: HRESULT = 0x8004_0154_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;

// ---------------------------------------------------------------------------------------
// the PC runtime
// ---------------------------------------------------------------------------------------

type QueryFn = unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> HRESULT;
type InitEx2 = unsafe extern "system" fn(u64, u64, u64, *mut c_void) -> HRESULT;
type InitEx = unsafe extern "system" fn(u64, u64, u64) -> HRESULT;
type Init = unsafe extern "system" fn(u64, u64) -> HRESULT;

struct Pc {
    query: QueryFn,
    init_ex2: Option<InitEx2>,
    init_ex: Option<InitEx>,
    init: Option<Init>,
    uninit: Option<unsafe extern "system" fn()>,
}

static PC: OnceLock<Option<Pc>> = OnceLock::new();

fn pc() -> Option<&'static Pc> {
    PC.get_or_init(|| unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
        let h = LoadLibraryExW(crate::util::wide("xgameruntime.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
        if h.is_null() {
            crate::xlog_warn!("XGameRuntime: no System32\\xgameruntime.dll (Gaming Services) – XTaskQueue/XAsync and the other APIs XCL does not implement are unavailable");
            return None;
        }
        let f = |n: &str| GetProcAddress(h, format!("{n}\0").as_ptr()).map(|p| p as usize);
        let query = f("QueryApiImpl")?;
        crate::xlog_info!("XGameRuntime: APIs XCL does not implement come from the system xgameruntime.dll");
        Some(Pc {
            query: std::mem::transmute::<usize, QueryFn>(query),
            init_ex2: f("InitializeApiImplEx2").map(|p| std::mem::transmute::<usize, InitEx2>(p)),
            init_ex: f("InitializeApiImplEx").map(|p| std::mem::transmute::<usize, InitEx>(p)),
            init: f("InitializeApiImpl").map(|p| std::mem::transmute::<usize, Init>(p)),
            uninit: f("UninitializeApiImpl").map(|p| std::mem::transmute::<usize, unsafe extern "system" fn()>(p)),
        })
    })
    .as_ref()
}

unsafe fn pc_object(c: &Guid, i: &Guid) -> Option<usize> {
    let mut o: *mut c_void = std::ptr::null_mut();
    ((pc()?.query)(c, i, &mut o) >= 0 && !o.is_null()).then_some(o as usize)
}

/// After the PC runtime is initialised: XCL's async APIs on its XAsync.
unsafe fn start_pc_async() {
    if let Some(o) = pc_object(&XASYNC, &XASYNC) {
        super::xasync::use_pc(o);
    }
}

// ---------------------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------------------

pub unsafe extern "system" fn InitializeApiImplEx2(gdk: u64, os: u64, flags: u64, options: *mut c_void) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImplEx2({gdk:#x}, {os:#x}, flags {flags:#x}) – title goes through the GDK thunks");
    pc_init(gdk, os, flags, options);
    S_OK
}

/// Initialise the PC runtime with the title's versions; options this PC runtime does not know
/// (newer GDK) fall back to the older entry points.
unsafe fn pc_init(gdk: u64, os: u64, flags: u64, options: *mut c_void) {
    let Some(p) = pc() else { return };
    let mut hr = p.init_ex2.map(|f| f(gdk, os, flags, options)).unwrap_or(-1);
    if hr < 0 {
        hr = p.init_ex.map(|f| f(gdk, os, flags)).unwrap_or(-1);
    }
    if hr < 0 {
        hr = p.init.map(|f| f(gdk >> 32, gdk & 0xFFFF_FFFF)).unwrap_or(-1);
    }
    if hr < 0 {
        crate::xlog_warn!("XGameRuntime PC: initialization → {:#x}", hr as u32);
    }
    start_pc_async();
}

pub unsafe extern "system" fn InitializeApiImplEx(gdk: u64, os: u64, flags: u64) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImplEx({gdk:#x}, {os:#x}, flags {flags:#x})");
    pc_init(gdk, os, flags, std::ptr::null_mut());
    S_OK
}

pub unsafe extern "system" fn InitializeApiImpl(gdk: u64, os: u64) -> HRESULT {
    crate::xlog_info!("XGameRuntime: InitializeApiImpl({gdk:#x}, {os:#x})");
    pc_init(gdk << 32 | (os & 0xFFFF_FFFF), 0, 0, std::ptr::null_mut());
    S_OK
}

pub unsafe extern "system" fn UninitializeApiImpl() {
    if let Some(f) = pc().and_then(|p| p.uninit) {
        f();
    }
}

pub unsafe extern "system" fn XErrorReport(hr: HRESULT, message: *const u8) {
    crate::xlog_warn!("XErrorReport {:#x}: {}", hr as u32, crate::util::from_cstr_ptr(message).unwrap_or_default());
}

pub unsafe extern "system" fn QueryApiImpl(clsid: *const Guid, iid: *const Guid, out: *mut *mut c_void) -> HRESULT {
    if clsid.is_null() || iid.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    *out = std::ptr::null_mut();
    let (c, i) = (*clsid, *iid);
    if let Some((_, name)) = REFUSED.iter().find(|r| r.0 == c) {
        note(c, i, format!("{name}: unavailable (offline)"));
        return REGDB_E_CLASSNOTREG;
    }
    if let Some(o) = own(&c, &i) {
        note(c, i, format!("{}: XCL", o.iface.name));
        if !o.iface.forward.is_empty() && o.pc.load(Ordering::Acquire) == 0 {
            o.pc.store(pc_object(&c, &i).unwrap_or(0), Ordering::Release);
        }
        *out = o as *const Obj as *mut c_void;
        return S_OK;
    }
    match pc_object(&c, &i) {
        Some(p) => {
            note(c, i, format!("{}: PC", family(&c)));
            *out = p as *mut c_void;
            S_OK
        }
        None => {
            note(c, i, format!("{}: missing", family(&c)));
            REGDB_E_CLASSNOTREG
        }
    }
}

// ---------------------------------------------------------------------------------------
// XCL's API objects
// ---------------------------------------------------------------------------------------

struct Iface {
    name: &'static str,
    clsid: Guid,
    /// interface ids answered (the clsid itself too)
    iids: &'static [Guid],
    /// (slot, function): the API function itself, without `this` (a shim drops it)
    slots: Vec<(usize, usize)>,
    /// slots passed to the PC runtime's object of the same interface
    forward: &'static [usize],
    /// unimplemented slots that return bool: answer false
    zero: &'static [usize],
}

#[repr(C)]
struct Obj {
    vtbl: *const usize,
    iface: Iface,
    /// the PC runtime's object, for `forward`
    pc: AtomicUsize,
    /// slot → API function the shim calls
    targets: [usize; SLOTS],
}

unsafe impl Sync for Obj {}
unsafe impl Send for Obj {}

const SLOTS: usize = 80;

unsafe extern "system" fn query_interface(this: *const Obj, iid: *const Guid, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() || iid.is_null() {
        return E_INVALIDARG;
    }
    let f = &(*this).iface;
    if *iid == f.clsid || f.iids.contains(&*iid) {
        *out = this as *mut c_void;
        return S_OK;
    }
    *out = std::ptr::null_mut();
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(_this: *const Obj) -> u32 {
    1
}

static UNIMPLEMENTED: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

unsafe fn unknown_slot(this: *const Obj, slot: usize) -> usize {
    let f = &(*this).iface;
    let key = format!("{}#{slot}", f.name);
    let first = {
        let mut g = UNIMPLEMENTED.lock().unwrap();
        let n = g.entry(key.clone()).or_insert(0);
        *n += 1;
        *n == 1
    };
    if first {
        crate::xlog_warn!("XGameRuntime: {key} – not supported (E_NOTIMPL)");
    }
    if f.zero.contains(&slot) {
        0
    } else {
        E_NOTIMPL as u32 as usize
    }
}

type Fwd = unsafe extern "system" fn(*const Obj, usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

/// Call slot `slot` of the PC object with the caller's arguments (up to 9; extra ones are
/// ignored by the callee).
unsafe fn forward(this: *const Obj, slot: usize, a: [usize; 9]) -> usize {
    let o = (*this).pc.load(Ordering::Acquire);
    if o == 0 {
        return unknown_slot(this, slot);
    }
    let f: Fwd = std::mem::transmute(*(*(o as *const *const usize)).add(slot));
    f(o as *const Obj, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
}

/// Slot `slot` of an XCL object: its API function with the caller's arguments after `this`.
unsafe fn call_target(this: *const Obj, slot: usize, a: [usize; 9]) -> usize {
    let f: unsafe extern "system" fn(usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize = std::mem::transmute((*this).targets[slot]);
    f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
}

macro_rules! per_slot {
    ($($n:literal)*) => {
        const SHIMS: [Fwd; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj, a: usize, b: usize, c: usize, d: usize, e: usize, g: usize, h: usize, i: usize, j: usize) -> usize {
                call_target(this, $n, [a, b, c, d, e, g, h, i, j])
            }
            f
        }),*];
        const STUBS: [unsafe extern "system" fn(*const Obj) -> usize; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj) -> usize {
                unknown_slot(this, $n)
            }
            f
        }),*];
        const FORWARDS: [Fwd; SLOTS] = [$({
            unsafe extern "system" fn f(this: *const Obj, a: usize, b: usize, c: usize, d: usize, e: usize, g: usize, h: usize, i: usize, j: usize) -> usize {
                forward(this, $n, [a, b, c, d, e, g, h, i, j])
            }
            f
        }),*];
    };
}
per_slot!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
    40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79);

fn object(iface: Iface) -> &'static Obj {
    let mut v: Vec<usize> = (0..SLOTS).map(|k| STUBS[k] as usize).collect();
    v[0] = query_interface as usize;
    v[1] = add_ref as usize;
    v[2] = add_ref as usize;
    for &k in iface.forward {
        v[k] = FORWARDS[k] as usize;
    }
    let mut targets = [0usize; SLOTS];
    for &(k, f) in &iface.slots {
        v[k] = SHIMS[k] as usize;
        targets[k] = f;
    }
    let vtbl = Box::leak(v.into_boxed_slice()).as_ptr();
    Box::leak(Box::new(Obj { vtbl, iface, pc: AtomicUsize::new(0), targets }))
}

fn objects() -> &'static [&'static Obj] {
    static O: OnceLock<Vec<&'static Obj>> = OnceLock::new();
    O.get_or_init(|| {
        vec![
            object(Iface { name: "XUser", clsid: XUSER, iids: &[XUSER_IID], slots: xuser_slots(), forward: &[], zero: &[] }),
            object(Iface { name: "XUser (gamertag)", clsid: XUSER, iids: &[XUSER_GAMERTAG_IID], slots: vec![(3, XUserGetGamertag as usize)], forward: &[], zero: &[] }),
            object(Iface { name: "XGameSave", clsid: XGAMESAVE, iids: &XGAMESAVE_IIDS, slots: xgamesave_slots(), forward: &[], zero: &[] }),
            object(Iface { name: "XPackage", clsid: XPACKAGE, iids: &[XPACKAGE_IID], slots: xpackage_slots(), forward: &[], zero: &[22, 34] }),
            object(Iface {
                name: "XGame",
                clsid: XGAME,
                iids: &[XGAME_IID],
                slots: vec![(3, XGameGetXboxTitleId as usize), (4, XLaunchNewGame as usize), (5, XLaunchRestartOnCrash as usize)],
                forward: &[],
                zero: &[],
            }),
            object(Iface {
                name: "XSystem",
                clsid: XSYSTEM,
                iids: &[XSYSTEM_IID],
                slots: vec![
                    (3, XSystemGetConsoleId as usize),
                    (4, XSystemGetXboxLiveSandboxId as usize),
                    (5, XSystemGetDeviceType as usize),
                ],
                forward: &[6, 7, 8, 9, 10, 11, 12, 13],
                zero: &[],
            }),
            // dialogs without a user handle go to the PC runtime (message, error, text entry)
            object(Iface { name: "XGameUi", clsid: XGAMEUI, iids: &[XGAMEUI_IID], slots: Vec::new(), forward: &[3, 4, 14, 15, 17, 18, 19], zero: &[] }),
            object(Iface {
                name: "XPersistentLocalStorage",
                clsid: XPLS,
                iids: &[XPLS],
                slots: vec![
                    (3, XPersistentLocalStorageGetPathSize as usize),
                    (4, XPersistentLocalStorageGetPath as usize),
                    (5, XPersistentLocalStorageGetSpaceInfo as usize),
                ],
                forward: &[],
                zero: &[],
            }),
        ]
    })
}

/// XCL's object for (clsid, iid); the clsid as iid means the class's first interface.
fn own(c: &Guid, i: &Guid) -> Option<&'static Obj> {
    let all = objects();
    all.iter().copied().find(|o| o.iface.clsid == *c && o.iface.iids.contains(i)).or_else(|| all.iter().copied().find(|o| o.iface.clsid == *c && *i == *c))
}

fn xuser_slots() -> Vec<(usize, usize)> {
    vec![
        (3, XUserDuplicateHandle as usize),
        (4, XUserCloseHandle as usize),
        (5, XUserCompare as usize),
        (6, XUserGetMaxUsers as usize),
        (7, XUserAddAsync as usize),
        (8, XUserAddResult as usize),
        (9, XUserGetLocalId as usize),
        (10, find_user as usize),
        (11, XUserGetId as usize),
        (12, find_user as usize),
        (13, XUserGetIsGuest as usize),
        (14, XUserGetState as usize),
        (19, XUserGetAgeGroup as usize),
        (20, XUserCheckPrivilege as usize),
        (33, XUserRegisterForChangeEvent as usize),
        (34, XUserUnregisterForChangeEvent as usize),
        (36, nothing as usize),
        (42, no as usize),
    ]
}

fn xgamesave_slots() -> Vec<(usize, usize)> {
    vec![
        (3, XGameSaveInitializeProvider as usize),
        (4, XGameSaveInitializeProviderAsync as usize),
        (5, XGameSaveInitializeProviderResult as usize),
        (6, XGameSaveCloseProvider as usize),
        (7, XGameSaveGetRemainingQuota as usize),
        (8, XGameSaveGetRemainingQuotaAsync as usize),
        (9, XGameSaveGetRemainingQuotaResult as usize),
        (10, XGameSaveDeleteContainer as usize),
        (11, XGameSaveDeleteContainerAsync as usize),
        (12, XGameSaveDeleteContainerResult as usize),
        (13, XGameSaveGetContainerInfo as usize),
        (14, XGameSaveEnumerateContainerInfo as usize),
        (15, XGameSaveEnumerateContainerInfoByName as usize),
        (16, XGameSaveCreateContainer as usize),
        (17, XGameSaveCloseContainer as usize),
        (18, XGameSaveEnumerateBlobInfo as usize),
        (19, XGameSaveEnumerateBlobInfoByName as usize),
        (20, XGameSaveReadBlobData as usize),
        (21, XGameSaveReadBlobDataAsync as usize),
        (22, XGameSaveReadBlobDataResult as usize),
        (23, XGameSaveCreateUpdate as usize),
        (24, XGameSaveCloseUpdate as usize),
        (25, XGameSaveSubmitBlobWrite as usize),
        (26, XGameSaveSubmitBlobDelete as usize),
        (27, XGameSaveSubmitUpdate as usize),
        (28, XGameSaveSubmitUpdateAsync as usize),
        (29, XGameSaveSubmitUpdateResult as usize),
        (30, XGameSaveFilesGetFolderWithUiAsync as usize),
        (31, XGameSaveFilesGetFolderWithUiResult as usize),
        (32, XGameSaveFilesGetRemainingQuota as usize),
    ]
}

fn xpackage_slots() -> Vec<(usize, usize)> {
    vec![
        (3, XPackageGetCurrentProcessPackageIdentifier as usize),
        (4, XPackageIsPackagedProcess as usize),
        (5, XPackageCreateInstallationMonitor as usize),
        (6, nothing as usize),
        (7, XPackageGetInstallationProgress as usize),
        (8, no as usize),
        (9, XPackageRegisterInstallationProgressChanged as usize),
        (10, XPackageUnregisterInstallationProgressChanged as usize),
        (23, XPackageMount as usize),
        (24, XPackageGetMountPathSize as usize),
        (25, XPackageGetMountPath as usize),
        (26, XPackageCloseMountHandle as usize),
        (39, XPackageEnumeratePackages as usize),
    ]
}

// ---------------------------------------------------------------------------------------
// functions only the thunk path has
// ---------------------------------------------------------------------------------------

unsafe extern "system" fn nothing(_a: usize) {}

unsafe extern "system" fn no(_a: usize) -> bool {
    false
}

/// XUserFindUserByLocalId / XUserFindUserById (slots 10 / 12): the local user by either id.
unsafe extern "system" fn find_user(id: u64, out: *mut *mut c_void) -> HRESULT {
    let hr = XUserFindUserByLocalId(id, out);
    if hr >= 0 {
        return hr;
    }
    XUserFindUserById(id, out)
}

/// The local user has every privilege (no Xbox Live to deny one).
unsafe extern "system" fn XUserCheckPrivilege(_user: *mut c_void, _options: u32, _privilege: u32, has: *mut bool, reasons: *mut u32) -> HRESULT {
    if has.is_null() {
        return E_INVALIDARG;
    }
    *has = true;
    if !reasons.is_null() {
        *reasons = 0;
    }
    S_OK
}

unsafe extern "system" fn XGameSaveFilesGetRemainingQuota(_user: *mut c_void, _config: *const u8, quota: *mut i64) -> HRESULT {
    if quota.is_null() {
        return E_INVALIDARG;
    }
    *quota = 256 << 20;
    S_OK
}

/// The installation monitor of a fully installed package (XCL runs from an unpacked folder).
static MONITOR: u64 = 0x4D4F_4E49;

unsafe extern "system" fn XPackageCreateInstallationMonitor(_id: *const u8, _count: u32, _sel: *const c_void, _interval: u32, _queue: *mut c_void, out: *mut *const u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    *out = &MONITOR;
    S_OK
}

/// XPackageInstallationProgress { totalBytes, installedBytes, launchBytes, launchable,
/// completed }: everything installed. The byte counts are 0x0101 so the flags still read
/// true if the struct has no launchBytes.
unsafe extern "system" fn XPackageGetInstallationProgress(_monitor: *const u64, p: *mut u8) {
    if p.is_null() {
        return;
    }
    for k in 0..3 {
        std::ptr::write_unaligned(p.add(8 * k) as *mut u64, 0x0101);
    }
    *p.add(24) = 1;
    *p.add(25) = 1;
}

unsafe extern "system" fn XPackageRegisterInstallationProgressChanged(_monitor: *const u64, _ctx: *mut c_void, _cb: usize, token: *mut u64) -> HRESULT {
    if !token.is_null() {
        *token = 1;
    }
    S_OK
}

unsafe extern "system" fn XPackageUnregisterInstallationProgressChanged(_monitor: *const u64, _token: u64, _wait: bool) -> bool {
    true
}

/// `<TitleId>` of the game's MicrosoftGame.Config (xcl.toml `title_id` overrides).
pub unsafe extern "system" fn XGameGetXboxTitleId(out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let t = &crate::state::config().title_id;
    *out = u32::from_str_radix(t.trim_start_matches("0x"), 16).unwrap_or(0);
    if *out == 0 {
        crate::xlog_warn!("XGameGetXboxTitleId: no TitleId in MicrosoftGame.Config / xcl.toml → 0");
    }
    S_OK
}

unsafe extern "system" fn XLaunchNewGame(exe: *const u8, args: *const u8, _user: *mut c_void) {
    crate::xlog_warn!(
        "XLaunchNewGame({}, {}) – ignored (XCL runs a single title)",
        crate::util::from_cstr_ptr(exe).unwrap_or_default(),
        crate::util::from_cstr_ptr(args).unwrap_or_default()
    );
}

unsafe extern "system" fn XLaunchRestartOnCrash(_args: *const u8, _reserved: u32) -> HRESULT {
    S_OK
}

// ---------------------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------------------

static SEEN: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn note(c: Guid, i: Guid, what: String) {
    let key = format!("{c}/{i}");
    let mut g = SEEN.lock().unwrap();
    if !g.contains_key(&key) {
        crate::xlog_info!("XGameRuntime: QueryApiImpl {key} → {what}");
        g.insert(key, what);
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiStats {
    /// "clsid/iid" → who answered (XCL, PC, unavailable, missing)
    pub apis: BTreeMap<String, String>,
    /// XCL objects' slots the title called that XCL does not implement: "Iface#slot" → calls
    pub unimplemented: BTreeMap<String, u64>,
}

pub fn stats() -> ApiStats {
    ApiStats { apis: SEEN.lock().unwrap().clone(), unimplemented: UNIMPLEMENTED.lock().unwrap().clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    unsafe fn query(c: Guid, i: Guid) -> (HRESULT, *mut c_void) {
        let mut o = std::ptr::null_mut();
        (QueryApiImpl(&c, &i, &mut o), o)
    }

    unsafe fn slot<T>(o: *mut c_void, k: usize) -> T {
        std::mem::transmute_copy(&*(*(o as *const *const usize)).add(k))
    }

    static DONE: AtomicU32 = AtomicU32::new(0);
    unsafe extern "system" fn done(_b: *mut XAsyncBlock) {
        DONE.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn guids_format_like_the_registry() {
        assert_eq!(XUSER.to_string(), "01acd177-91f9-4763-a38e-ccbb55ce32e0");
        let b: [u8; 16] = unsafe { std::mem::transmute(XUSER) };
        assert_eq!(&b[..4], &[0x77, 0xd1, 0xac, 0x01]);
    }

    /// The thunk protocol end to end on the PC runtime: XUser and XGameSave are XCL's objects,
    /// their async calls complete on the PC runtime's task queues; XStore is refused; XAsync
    /// comes from the PC runtime.
    #[test]
    fn thunk_protocol_with_xcl_objects_on_pc_queues() {
    let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if pc().is_none() {
            println!("no system xgameruntime.dll – skipping");
            return;
        }
        unsafe {
            assert_eq!(InitializeApiImplEx2(0x6336_0000_2712, 0x65f4_0000_1edd, 2, std::ptr::null_mut()), S_OK);
            assert_eq!(query(XSTORE, XSTORE).0, REGDB_E_CLASSNOTREG);
            let (hr, xa) = query(XASYNC, XASYNC);
            assert!(hr >= 0 && !xa.is_null(), "XAsync from PC: {hr:#x}");
            let get_status: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, bool) -> HRESULT = slot(xa, 3);

            // XUserAddAsync / Result through the vtable, completed on the PC process queue
            let (hr, u) = query(XUSER, XUSER_IID);
            assert_eq!(hr, S_OK);
            let add: unsafe extern "system" fn(*mut c_void, u32, *mut XAsyncBlock) -> HRESULT = slot(u, 7);
            let result: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut *mut c_void) -> HRESULT = slot(u, 8);
            let mut b: XAsyncBlock = std::mem::zeroed();
            b.callback = done as usize;
            assert_eq!(add(u, 1, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut user = std::ptr::null_mut();
            assert_eq!(result(u, &mut b, &mut user), S_OK);
            assert!(!user.is_null());
            let (_, gt) = query(XUSER, XUSER_GAMERTAG_IID);
            let gamertag: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, usize, *mut u8, *mut usize) -> HRESULT = slot(gt, 3);
            let mut buf = [0u8; 64];
            let mut used = 0usize;
            assert_eq!(gamertag(gt, user, 0, 64, buf.as_mut_ptr(), &mut used), S_OK);
            assert!(used > 1);
            // an unimplemented slot answers E_NOTIMPL and is reported
            let token: unsafe extern "system" fn(*mut c_void) -> HRESULT = slot(u, 23);
            assert_eq!(token(u), E_NOTIMPL);
            assert!(stats().unimplemented.contains_key("XUser#23"));

            // Test XSystem slot 5 (XSystemGetDeviceType)
            let (_, sys) = query(XSYSTEM, XSYSTEM_IID);
            assert!(!sys.is_null());
            let get_device_type: unsafe extern "system" fn(*mut c_void) -> u32 = slot(sys, 5);
            assert_eq!(get_device_type(sys), crate::hle::xgameruntime::DEVICE_XBOX_SCARLETT_ANACONDA);

            // Test XPersistentLocalStorage thunk provider (0x7f0fe8b8)
            let (hr_pls, pls) = query(XPLS, XPLS);
            assert_eq!(hr_pls, S_OK);
            assert!(!pls.is_null());
            let get_path_size: unsafe extern "system" fn(*mut c_void, *mut usize) -> HRESULT = slot(pls, 3);
            let mut pls_size = 0usize;
            assert_eq!(get_path_size(pls, &mut pls_size), S_OK);
            assert!(pls_size > 0);

            // XGameSave: provider (async), container, write a blob, read it back (async)
            let (_, s) = query(XGAMESAVE, XGAMESAVE_IIDS[0]);
            let init: unsafe extern "system" fn(*mut c_void, *mut c_void, *const u8, bool, *mut XAsyncBlock) -> HRESULT = slot(s, 4);
            let init_r: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, *mut *mut Provider) -> HRESULT = slot(s, 5);
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(init(s, user, c"xcl-thunk-test".as_ptr() as *const u8, false, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut p = std::ptr::null_mut();
            assert_eq!(init_r(s, &mut b, &mut p), S_OK);
            let create: unsafe extern "system" fn(*mut c_void, *mut Provider, *const u8, *mut *mut Container) -> HRESULT = slot(s, 16);
            let mut c = std::ptr::null_mut();
            assert_eq!(create(s, p, c"slot".as_ptr() as *const u8, &mut c), S_OK);
            let mut upd = std::ptr::null_mut();
            assert_eq!(XGameSaveCreateUpdate(c, c"t".as_ptr() as *const u8, &mut upd), S_OK);
            assert_eq!(XGameSaveSubmitBlobWrite(upd, c"b".as_ptr() as *const u8, b"thunk".as_ptr(), 5), S_OK);
            assert_eq!(XGameSaveSubmitUpdate(upd), S_OK);
            XGameSaveCloseUpdate(upd);
            let read: unsafe extern "system" fn(*mut c_void, *mut Container, *const *const u8, u32, *mut XAsyncBlock) -> HRESULT = slot(s, 21);
            let read_r: unsafe extern "system" fn(*mut c_void, *mut XAsyncBlock, usize, *mut Blob, *mut u32) -> HRESULT = slot(s, 22);
            let names = [c"b".as_ptr() as *const u8];
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(read(s, c, names.as_ptr(), 1, &mut b), S_OK);
            assert_eq!(get_status(xa, &mut b, true), S_OK);
            let mut size = 0usize;
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), S_OK);
            let mut out = vec![0u64; size / 8 + 1];
            let mut n = 0u32;
            assert_eq!(read_r(s, &mut b, size, out.as_mut_ptr() as *mut Blob, &mut n), S_OK);
            assert_eq!(n, 1);
            XGameSaveCloseContainer(c);
            XGameSaveCloseProvider(p);
            // completion callbacks run on the PC queue's threads, after the status
            for _ in 0..200 {
                if DONE.load(Ordering::SeqCst) >= 1 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            super::super::xasync::use_pc(0);
            assert!(DONE.load(Ordering::SeqCst) >= 1, "XAsync completion callback");
        }
    }
}
