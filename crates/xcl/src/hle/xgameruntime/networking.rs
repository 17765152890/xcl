//! XNetworking – GDK networking helpers (`XNetworking.h`, public GDK contract).
//!
//! Covers UDP multiplayer port queries, per-URL HTTP security information (UTF-8 and UTF-16),
//! server certificate verification (real Windows CryptoAPI SSL chain + SHA-1 thumbprint pin
//! checking, with plain-HTTP loopback allowance for local servers), and host connectivity
//! hints via `iphlpapi.dll`.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use super::xasync::{begin_with, XAsyncBlock, XAsyncGetResult, XAsyncGetResultSize};
use crate::util::{hresult_from_win32, wide, E_INVALIDARG, HRESULT, S_OK};

/// Default Xbox Live multiplayer UDP port.
pub const DEFAULT_UDP_PORT: u16 = 3074;

/// `WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3`.
pub const DEFAULT_TLS_FLAGS: u32 = 0x0000_0800 | 0x0000_2000;

const CERT_E_UNTRUSTEDROOT: HRESULT = 0x800B_0109u32 as i32;
const WINHTTP_OPTION_URL: u32 = 34;
const WINHTTP_OPTION_SERVER_CERT_CONTEXT: u32 = 78;
const CERT_SHA1_HASH_PROP_ID: u32 = 3;
const CERT_CHAIN_POLICY_SSL: usize = 4;
const AUTHTYPE_SERVER: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XNetworkingThumbprint {
    pub thumbprint_type: u32,
    pub thumbprint_buffer_byte_count: usize,
    pub thumbprint_buffer: *const u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XNetworkingSecurityInformation {
    pub enabled_http_security_protocol_flags: u32,
    pub thumbprint_count: usize,
    pub thumbprints: *const XNetworkingThumbprint,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XNetworkingConnectivityHint {
    pub connectivity_level: u32,
    pub connectivity_cost: u32,
    pub iana_interface_type: u32,
    pub network_initialized: bool,
    pub approaching_data_limit: bool,
    pub over_data_limit: bool,
    pub roaming: bool,
}

static UDP_PORT_ID: u8 = 0;
static SEC_INFO_ID: u8 = 0;
static SEC_INFO_UTF16_ID: u8 = 0;
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------------------
// UDP multiplayer port
// ---------------------------------------------------------------------------------------

pub unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPort(port: *mut u16) -> HRESULT {
    if port.is_null() {
        return E_INVALIDARG;
    }
    *port = DEFAULT_UDP_PORT;
    S_OK
}

pub unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync(b: *mut XAsyncBlock) -> HRESULT {
    begin_with(b, &UDP_PORT_ID as *const u8 as usize, || Ok(DEFAULT_UDP_PORT.to_le_bytes().to_vec()))
}

pub unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult(
    b: *mut XAsyncBlock,
    port: *mut u16,
) -> HRESULT {
    if port.is_null() {
        return E_INVALIDARG;
    }
    XAsyncGetResult(b, &UDP_PORT_ID as *const u8 as *const c_void, std::mem::size_of::<u16>(), port as *mut c_void, std::ptr::null_mut())
}

pub unsafe extern "system" fn XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged(
    _queue: *mut c_void,
    _ctx: *mut c_void,
    _cb: usize,
    token: *mut u64,
) -> HRESULT {
    if token.is_null() {
        return E_INVALIDARG;
    }
    *token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    S_OK
}

pub unsafe extern "system" fn XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged(_token: u64, _wait: bool) -> bool {
    true
}

// ---------------------------------------------------------------------------------------
// Per-URL HTTP security information (TLS 1.2 / 1.3, no certificate pinning)
// ---------------------------------------------------------------------------------------

fn default_security_info_bytes() -> Vec<u8> {
    let info = XNetworkingSecurityInformation {
        enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS,
        thumbprint_count: 0,
        thumbprints: std::ptr::null(),
    };
    let mut out = vec![0u8; std::mem::size_of::<XNetworkingSecurityInformation>()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            &info as *const XNetworkingSecurityInformation as *const u8,
            out.as_mut_ptr(),
            out.len(),
        );
    }
    out
}

pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlAsync(
    url: *const u8,
    b: *mut XAsyncBlock,
) -> HRESULT {
    if url.is_null() {
        return E_INVALIDARG;
    }
    begin_with(b, &SEC_INFO_ID as *const u8 as usize, || Ok(default_security_info_bytes()))
}

pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlAsyncResultSize(
    b: *mut XAsyncBlock,
    size: *mut usize,
) -> HRESULT {
    XAsyncGetResultSize(b, size)
}

/// Note the GDK parameter order: `buffer_used` precedes `buffer` and `out_ptr`.
pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlAsyncResult(
    b: *mut XAsyncBlock,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> HRESULT {
    security_info_result(&SEC_INFO_ID as *const u8 as *const c_void, b, buffer_size, buffer_used, buffer, out_ptr)
}

pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlUtf16Async(
    url: *const u16,
    b: *mut XAsyncBlock,
) -> HRESULT {
    if url.is_null() {
        return E_INVALIDARG;
    }
    begin_with(b, &SEC_INFO_UTF16_ID as *const u8 as usize, || Ok(default_security_info_bytes()))
}

pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize(
    b: *mut XAsyncBlock,
    size: *mut usize,
) -> HRESULT {
    XAsyncGetResultSize(b, size)
}

pub unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult(
    b: *mut XAsyncBlock,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> HRESULT {
    security_info_result(&SEC_INFO_UTF16_ID as *const u8 as *const c_void, b, buffer_size, buffer_used, buffer, out_ptr)
}

unsafe fn security_info_result(
    identity: *const c_void,
    b: *mut XAsyncBlock,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> HRESULT {
    if !out_ptr.is_null() {
        *out_ptr = std::ptr::null_mut();
    }
    let mut used = 0usize;
    let hr = XAsyncGetResult(b, identity, buffer_size, buffer as *mut c_void, &mut used);
    if !buffer_used.is_null() {
        *buffer_used = used;
    }
    if hr >= 0 && !out_ptr.is_null() {
        *out_ptr = buffer as *mut XNetworkingSecurityInformation;
    }
    hr
}

// ---------------------------------------------------------------------------------------
// Server certificate verification (WinHTTP + CryptoAPI)
// ---------------------------------------------------------------------------------------

#[repr(C)]
struct CertContext {
    encoding_type: u32,
    encoded: *const u8,
    encoded_len: u32,
    cert_info: *const c_void,
    cert_store: *mut c_void,
}

#[repr(C)]
struct CertChainPara {
    cb_size: u32,
    usage_match_type: u32,
    usage_identifier_count: u32,
    usage_identifiers: *const *const u8,
}

#[repr(C)]
struct SslExtraCertChainPolicyPara {
    cb_size: u32,
    auth_type: u32,
    fdw_checks: u32,
    server_name: *const u16,
}

#[repr(C)]
struct CertChainPolicyPara {
    cb_size: u32,
    flags: u32,
    extra_policy_para: *const c_void,
}

#[repr(C)]
struct CertChainPolicyStatus {
    cb_size: u32,
    error: u32,
    chain_index: i32,
    element_index: i32,
    extra_policy_status: *mut c_void,
}

type WinHttpQueryOptionFn = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, *mut u32) -> i32;
type CertGetChainFn = unsafe extern "system" fn(
    *mut c_void,
    *const CertContext,
    *const c_void,
    *mut c_void,
    *const CertChainPara,
    u32,
    *mut c_void,
    *mut *const c_void,
) -> i32;
type CertVerifyPolicyFn =
    unsafe extern "system" fn(usize, *const c_void, *const CertChainPolicyPara, *mut CertChainPolicyStatus) -> i32;
type CertFreeChainFn = unsafe extern "system" fn(*const c_void);
type CertFreeContextFn = unsafe extern "system" fn(*const CertContext) -> i32;
type CertGetPropFn = unsafe extern "system" fn(*const CertContext, u32, *mut c_void, *mut u32) -> i32;

struct CryptoProcs {
    winhttp_query_option: WinHttpQueryOptionFn,
    cert_get_chain: CertGetChainFn,
    cert_verify_policy: CertVerifyPolicyFn,
    cert_free_chain: CertFreeChainFn,
    cert_free_context: CertFreeContextFn,
    cert_get_prop: CertGetPropFn,
}

fn crypto_procs() -> Option<&'static CryptoProcs> {
    static PROCS: OnceLock<Option<CryptoProcs>> = OnceLock::new();
    PROCS
        .get_or_init(|| unsafe {
            use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
            let wh = LoadLibraryExW(wide("winhttp.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            let cr = LoadLibraryExW(wide("crypt32.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            if wh.is_null() || cr.is_null() {
                return None;
            }
            let sym = |h, name: &[u8]| GetProcAddress(h, name.as_ptr()).map(|f| f as usize);
            Some(CryptoProcs {
                winhttp_query_option: std::mem::transmute(sym(wh, b"WinHttpQueryOption\0")?),
                cert_get_chain: std::mem::transmute(sym(cr, b"CertGetCertificateChain\0")?),
                cert_verify_policy: std::mem::transmute(sym(cr, b"CertVerifyCertificateChainPolicy\0")?),
                cert_free_chain: std::mem::transmute(sym(cr, b"CertFreeCertificateChain\0")?),
                cert_free_context: std::mem::transmute(sym(cr, b"CertFreeCertificateContext\0")?),
                cert_get_prop: std::mem::transmute(sym(cr, b"CertGetCertificateContextProperty\0")?),
            })
        })
        .as_ref()
}

fn url_host_and_is_plain_loopback(url: &str) -> (String, bool) {
    let lower = url.to_ascii_lowercase();
    let is_http = lower.starts_with("http://");
    let after_scheme = match url.split_once("://") {
        Some((_, r)) => r,
        None => url,
    };
    let authority = after_scheme.split(['/', '?', '#']).next().unwrap_or("");
    let host = if authority.starts_with('[') {
        authority.split(']').next().unwrap_or("").trim_start_matches('[')
    } else {
        authority.split(':').next().unwrap_or("")
    };
    let loopback = host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1";
    (host.to_string(), is_http && loopback)
}

unsafe fn sha1_thumbprint(procs: &CryptoProcs, cert: *const CertContext) -> Vec<u8> {
    let mut buf = [0u8; 20];
    let mut len = buf.len() as u32;
    if (procs.cert_get_prop)(cert, CERT_SHA1_HASH_PROP_ID, buf.as_mut_ptr() as *mut c_void, &mut len) == 0 {
        return Vec::new();
    }
    buf[..len as usize].to_vec()
}

/// Extract leaf-to-root certificate contexts from the primary simple chain in `CERT_CHAIN_CONTEXT`.
///
/// The chain structures are read through the SDK's declarations (`windows-sys`), not hand-written
/// offsets: on x64 `cChain` / `cElement` sit at +12 and `rgpChain` / `rgpElement` at +16.
///
/// # Safety
/// `chain` is null or a chain context returned by `CertGetCertificateChain` and not freed yet.
/// The returned contexts belong to the chain and are valid only until `CertFreeCertificateChain`.
unsafe fn extract_chain_certs(chain: *const c_void) -> Vec<*const CertContext> {
    use windows_sys::Win32::Security::Cryptography::{CERT_CHAIN_CONTEXT, CERT_CHAIN_ELEMENT, CERT_SIMPLE_CHAIN};

    let chain = chain as *const CERT_CHAIN_CONTEXT;
    if chain.is_null() {
        return Vec::new();
    }
    // SAFETY: the caller passes a live chain context, laid out by crypt32 as the SDK declares it.
    let chain = &*chain;
    if chain.cChain == 0 || chain.rgpChain.is_null() {
        return Vec::new();
    }
    // SAFETY: `rgpChain` holds `cChain` (> 0) pointers; the first one is the primary simple chain.
    let first = *chain.rgpChain as *const CERT_SIMPLE_CHAIN;
    if first.is_null() {
        return Vec::new();
    }
    let first = &*first;
    if first.cElement == 0 || first.rgpElement.is_null() {
        return Vec::new();
    }
    // SAFETY: `rgpElement` holds `cElement` element pointers, leaf first, owned by the chain.
    let elems = std::slice::from_raw_parts(first.rgpElement as *const *const CERT_CHAIN_ELEMENT, first.cElement as usize);
    elems
        .iter()
        .filter(|e| !e.is_null())
        // SAFETY: a non-null element pointer from the chain points at a live CERT_CHAIN_ELEMENT.
        .map(|&e| (*e).pCertContext as *const CertContext)
        .filter(|c| !c.is_null())
        .collect()
}

/// Verify optional SHA-1 thumbprint pins against the leaf-to-root certificate chain.
pub unsafe fn verify_thumbprints(security_info: *const XNetworkingSecurityInformation, chain_sha1s: &[Vec<u8>]) -> bool {
    if security_info.is_null() {
        return true;
    }
    let info = std::ptr::read_unaligned(security_info);
    if info.thumbprint_count == 0 || info.thumbprints.is_null() {
        return true;
    }
    for i in 0..info.thumbprint_count {
        let tp = std::ptr::read_unaligned(info.thumbprints.add(i));
        if tp.thumbprint_buffer.is_null() || tp.thumbprint_buffer_byte_count == 0 {
            continue;
        }
        let expected = std::slice::from_raw_parts(tp.thumbprint_buffer, tp.thumbprint_buffer_byte_count);
        let matched = match tp.thumbprint_type {
            0 => chain_sha1s.first().is_some_and(|c| c.as_slice() == expected),
            1 => chain_sha1s.get(1).is_some_and(|c| c.as_slice() == expected),
            2 => chain_sha1s.last().is_some_and(|c| c.as_slice() == expected),
            _ => chain_sha1s.iter().any(|c| c.as_slice() == expected),
        };
        if matched {
            return true;
        }
    }
    false
}

pub unsafe extern "system" fn XNetworkingVerifyServerCertificate(
    request_handle: *mut c_void,
    security_info: *const XNetworkingSecurityInformation,
) -> HRESULT {
    if request_handle.is_null() {
        return E_INVALIDARG;
    }
    let Some(procs) = crypto_procs() else {
        return crate::util::E_FAIL;
    };

    let mut url_buf = [0u16; 2048];
    let mut url_bytes = (url_buf.len() * 2) as u32;
    let url = if (procs.winhttp_query_option)(request_handle, WINHTTP_OPTION_URL, url_buf.as_mut_ptr() as *mut c_void, &mut url_bytes) != 0 {
        let chars = (url_bytes as usize / 2).min(url_buf.len());
        let trimmed = url_buf[..chars].iter().position(|&c| c == 0).unwrap_or(chars);
        String::from_utf16_lossy(&url_buf[..trimmed])
    } else {
        String::new()
    };
    let (host, is_plain_loopback) = url_host_and_is_plain_loopback(&url);
    if is_plain_loopback {
        return S_OK;
    }

    let mut cert: *const CertContext = std::ptr::null();
    let mut cert_size = std::mem::size_of::<*const CertContext>() as u32;
    if (procs.winhttp_query_option)(
        request_handle,
        WINHTTP_OPTION_SERVER_CERT_CONTEXT,
        &mut cert as *mut *const CertContext as *mut c_void,
        &mut cert_size,
    ) == 0
        || cert.is_null()
    {
        let err = windows_sys::Win32::Foundation::GetLastError();
        return if err != 0 { hresult_from_win32(err) } else { crate::util::E_FAIL };
    }

    let para = CertChainPara {
        cb_size: std::mem::size_of::<CertChainPara>() as u32,
        usage_match_type: 0,
        usage_identifier_count: 0,
        usage_identifiers: std::ptr::null(),
    };
    let mut chain: *const c_void = std::ptr::null();
    let got_chain = (procs.cert_get_chain)(
        std::ptr::null_mut(),
        cert,
        std::ptr::null(),
        (*cert).cert_store,
        &para,
        0,
        std::ptr::null_mut(),
        &mut chain,
    ) != 0;

    let mut hr = if got_chain {
        S_OK
    } else {
        hresult_from_win32(windows_sys::Win32::Foundation::GetLastError())
    };
    if got_chain {
        let host_w = wide(&host);
        let ssl = SslExtraCertChainPolicyPara {
            cb_size: std::mem::size_of::<SslExtraCertChainPolicyPara>() as u32,
            auth_type: AUTHTYPE_SERVER,
            fdw_checks: 0,
            server_name: if host.is_empty() { std::ptr::null() } else { host_w.as_ptr() },
        };
        let policy_para = CertChainPolicyPara {
            cb_size: std::mem::size_of::<CertChainPolicyPara>() as u32,
            flags: 0,
            extra_policy_para: &ssl as *const SslExtraCertChainPolicyPara as *const c_void,
        };
        let mut status = CertChainPolicyStatus {
            cb_size: std::mem::size_of::<CertChainPolicyStatus>() as u32,
            error: 0,
            chain_index: -1,
            element_index: -1,
            extra_policy_status: std::ptr::null_mut(),
        };
        if (procs.cert_verify_policy)(CERT_CHAIN_POLICY_SSL, chain, &policy_para, &mut status) == 0 {
            hr = hresult_from_win32(windows_sys::Win32::Foundation::GetLastError());
        } else if status.error != 0 {
            hr = status.error as HRESULT;
        } else {
            let prints: Vec<Vec<u8>> = extract_chain_certs(chain).into_iter().map(|c| sha1_thumbprint(procs, c)).collect();
            if !verify_thumbprints(security_info, &prints) {
                hr = CERT_E_UNTRUSTEDROOT;
            }
        }
        (procs.cert_free_chain)(chain);
    }
    (procs.cert_free_context)(cert);
    hr
}

// ---------------------------------------------------------------------------------------
// Connectivity hint (iphlpapi.dll)
// ---------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NlNetworkConnectivityHint {
    connectivity_level: u32,
    connectivity_cost: u32,
    approaching_data_limit: u8,
    over_data_limit: u8,
    roaming: u8,
}

type GetConnectivityHintFn = unsafe extern "system" fn(*mut NlNetworkConnectivityHint) -> u32;

fn host_connectivity_hint() -> Option<NlNetworkConnectivityHint> {
    static FN: OnceLock<Option<GetConnectivityHintFn>> = OnceLock::new();
    let f = (*FN.get_or_init(|| unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
        let h = LoadLibraryExW(wide("iphlpapi.dll").as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
        if h.is_null() {
            return None;
        }
        GetProcAddress(h, b"GetNetworkConnectivityHint\0".as_ptr()).map(|p| std::mem::transmute(p))
    }))?;
    let mut hint = NlNetworkConnectivityHint::default();
    if unsafe { f(&mut hint) } == 0 { Some(hint) } else { None }
}

pub unsafe extern "system" fn XNetworkingGetConnectivityHint(out: *mut XNetworkingConnectivityHint) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let host = host_connectivity_hint();
    let level = match host.map(|h| h.connectivity_level) {
        Some(l @ 1..=4) => l,
        Some(5) => 2, // LocalAccess
        _ => 3,       // InternetAccess
    };
    let cost = match host.map(|h| h.connectivity_cost) {
        Some(c @ 1..=3) => c,
        _ => 1, // Unrestricted
    };
    *out = XNetworkingConnectivityHint {
        connectivity_level: level,
        connectivity_cost: cost,
        iana_interface_type: 6, // IF_TYPE_ETHERNET_CSMACD
        network_initialized: true,
        approaching_data_limit: host.is_some_and(|h| h.approaching_data_limit != 0),
        over_data_limit: host.is_some_and(|h| h.over_data_limit != 0),
        roaming: host.is_some_and(|h| h.roaming != 0),
    };
    S_OK
}

pub unsafe extern "system" fn XNetworkingRegisterConnectivityHintChanged(
    _queue: *mut c_void,
    _ctx: *mut c_void,
    _cb: usize,
    token: *mut u64,
) -> HRESULT {
    if token.is_null() {
        return E_INVALIDARG;
    }
    *token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    S_OK
}

pub unsafe extern "system" fn XNetworkingUnregisterConnectivityHintChanged(_token: u64, _wait: bool) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hle::xgameruntime::xasync::XAsyncGetStatus;

    #[test]
    fn struct_layouts_match_gdk_abi() {
        assert_eq!(std::mem::size_of::<XNetworkingThumbprint>(), 24);
        assert_eq!(std::mem::size_of::<XNetworkingSecurityInformation>(), 24);
        assert_eq!(std::mem::size_of::<XNetworkingConnectivityHint>(), 16);
    }

    #[test]
    fn udp_port_sync_and_async() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let mut port = 0u16;
            assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPort(&mut port), S_OK);
            assert_eq!(port, DEFAULT_UDP_PORT);

            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync(&mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut async_port = 0u16;
            assert_eq!(XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult(&mut b, &mut async_port), S_OK);
            assert_eq!(async_port, DEFAULT_UDP_PORT);
        }
    }

    #[test]
    fn security_info_utf8_and_utf16_follow_gdk_arg_order() {
        let _g = crate::hle::xgameruntime::xasync::ASYNC_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            // UTF-8
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XNetworkingQuerySecurityInformationForUrlAsync(b"https://example.invalid/\0".as_ptr(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut sz = 0usize;
            assert_eq!(XNetworkingQuerySecurityInformationForUrlAsyncResultSize(&mut b, &mut sz), S_OK);
            assert_eq!(sz, std::mem::size_of::<XNetworkingSecurityInformation>());
            let mut buf = [0u8; 32];
            let mut used = 0usize;
            let mut ptr: *mut XNetworkingSecurityInformation = std::ptr::null_mut();
            assert_eq!(
                XNetworkingQuerySecurityInformationForUrlAsyncResult(&mut b, buf.len(), &mut used, buf.as_mut_ptr(), &mut ptr),
                S_OK
            );
            assert_eq!(used, sz);
            assert_eq!(ptr as *const u8, buf.as_ptr());
            assert_eq!((*ptr).enabled_http_security_protocol_flags, DEFAULT_TLS_FLAGS);
            assert_eq!((*ptr).thumbprint_count, 0);

            // UTF-16
            let url16 = wide("https://example.invalid/hotfix");
            let mut b2: XAsyncBlock = std::mem::zeroed();
            assert_eq!(XNetworkingQuerySecurityInformationForUrlUtf16Async(url16.as_ptr(), &mut b2), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b2, true), S_OK);
            let mut sz2 = 0usize;
            assert_eq!(XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize(&mut b2, &mut sz2), S_OK);
            assert_eq!(sz2, std::mem::size_of::<XNetworkingSecurityInformation>());
            let mut ptr2: *mut XNetworkingSecurityInformation = std::ptr::null_mut();
            assert_eq!(
                XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult(&mut b2, buf.len(), &mut used, buf.as_mut_ptr(), &mut ptr2),
                S_OK
            );
            assert_eq!((*ptr2).enabled_http_security_protocol_flags, DEFAULT_TLS_FLAGS);
        }
    }

    #[test]
    fn connectivity_hint_and_pin_verification() {
        unsafe {
            let mut hint: XNetworkingConnectivityHint = std::mem::zeroed();
            assert_eq!(XNetworkingGetConnectivityHint(&mut hint), S_OK);
            assert!((1..=4).contains(&hint.connectivity_level));
            assert!((1..=3).contains(&hint.connectivity_cost));
            assert_eq!(hint.iana_interface_type, 6);
            assert!(hint.network_initialized);

            assert_eq!(XNetworkingVerifyServerCertificate(std::ptr::null_mut(), std::ptr::null()), E_INVALIDARG);

            let leaf = vec![0xAAu8; 20];
            let root = vec![0xBBu8; 20];
            let chain = vec![leaf.clone(), root.clone()];
            let pin_ok = XNetworkingThumbprint { thumbprint_type: 2, thumbprint_buffer_byte_count: root.len(), thumbprint_buffer: root.as_ptr() };
            let info_ok = XNetworkingSecurityInformation { enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS, thumbprint_count: 1, thumbprints: &pin_ok };
            assert!(verify_thumbprints(&info_ok, &chain));

            let bad = vec![0xCCu8; 20];
            let pin_bad = XNetworkingThumbprint { thumbprint_type: 0, thumbprint_buffer_byte_count: bad.len(), thumbprint_buffer: bad.as_ptr() };
            let info_bad = XNetworkingSecurityInformation { enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS, thumbprint_count: 1, thumbprints: &pin_bad };
            assert!(!verify_thumbprints(&info_bad, &chain));
        }

        assert_eq!(url_host_and_is_plain_loopback("http://127.0.0.1:3551/account/api/oauth/token"), ("127.0.0.1".into(), true));
        assert_eq!(url_host_and_is_plain_loopback("https://127.0.0.1:3551/secure"), ("127.0.0.1".into(), false));
        assert_eq!(url_host_and_is_plain_loopback("http://[::1]:8080/test"), ("::1".into(), true));
        assert_eq!(url_host_and_is_plain_loopback("https://social.xboxlive.com/users"), ("social.xboxlive.com".into(), false));
    }

    #[test]
    fn cert_context_matches_sdk_layout() {
        use std::mem::{offset_of, size_of};
        use windows_sys::Win32::Security::Cryptography::CERT_CONTEXT;
        assert_eq!(size_of::<CertContext>(), size_of::<CERT_CONTEXT>());
        assert_eq!(offset_of!(CertContext, encoded), offset_of!(CERT_CONTEXT, pbCertEncoded));
        assert_eq!(offset_of!(CertContext, encoded_len), offset_of!(CERT_CONTEXT, cbCertEncoded));
        assert_eq!(offset_of!(CertContext, cert_info), offset_of!(CERT_CONTEXT, pCertInfo));
        assert_eq!(offset_of!(CertContext, cert_store), offset_of!(CERT_CONTEXT, hCertStore));
    }

    /// A chain built from the SDK's own declarations: the walk must find both contexts, leaf first.
    #[test]
    fn chain_certs_follow_sdk_layout() {
        use windows_sys::Win32::Security::Cryptography::{CERT_CHAIN_CONTEXT, CERT_CHAIN_ELEMENT, CERT_CONTEXT, CERT_SIMPLE_CHAIN};
        unsafe {
            let certs: [CERT_CONTEXT; 2] = std::mem::zeroed();
            let mut elems: [CERT_CHAIN_ELEMENT; 2] = std::mem::zeroed();
            for (e, c) in elems.iter_mut().zip(&certs) {
                e.cbSize = std::mem::size_of::<CERT_CHAIN_ELEMENT>() as u32;
                e.pCertContext = c;
            }
            let mut elem_ptrs = [elems.as_mut_ptr(), elems.as_mut_ptr().add(1)];
            let mut simple: CERT_SIMPLE_CHAIN = std::mem::zeroed();
            simple.cbSize = std::mem::size_of::<CERT_SIMPLE_CHAIN>() as u32;
            simple.cElement = elem_ptrs.len() as u32;
            simple.rgpElement = elem_ptrs.as_mut_ptr();
            let mut simple_ptrs = [&mut simple as *mut CERT_SIMPLE_CHAIN];
            let mut ctx: CERT_CHAIN_CONTEXT = std::mem::zeroed();
            ctx.cbSize = std::mem::size_of::<CERT_CHAIN_CONTEXT>() as u32;
            ctx.cChain = 1;
            ctx.rgpChain = simple_ptrs.as_mut_ptr();

            let got = extract_chain_certs(&ctx as *const CERT_CHAIN_CONTEXT as *const c_void);
            let want: Vec<*const CertContext> = certs.iter().map(|c| c as *const CERT_CONTEXT as *const CertContext).collect();
            assert_eq!(got, want);

            ctx.cChain = 0;
            assert!(extract_chain_certs(&ctx as *const CERT_CHAIN_CONTEXT as *const c_void).is_empty());
            assert!(extract_chain_certs(std::ptr::null()).is_empty());
        }
    }

    /// A real chain from crypt32 (a certificate of the machine's root store): the walk finds the
    /// leaf, the certificate the chain was built for.
    #[test]
    fn chain_certs_from_crypt32() {
        use windows_sys::Win32::Security::Cryptography::{CertCloseStore, CertEnumCertificatesInStore, CertOpenSystemStoreW};
        unsafe {
            let procs = crypto_procs().expect("crypt32 / winhttp");
            let store = CertOpenSystemStoreW(0, wide("ROOT").as_ptr());
            assert!(!store.is_null());
            let cert = CertEnumCertificatesInStore(store, std::ptr::null()) as *const CertContext;
            if cert.is_null() {
                eprintln!("root store is empty – skipped");
                CertCloseStore(store, 0);
                return;
            }
            let para = CertChainPara {
                cb_size: std::mem::size_of::<CertChainPara>() as u32,
                usage_match_type: 0,
                usage_identifier_count: 0,
                usage_identifiers: std::ptr::null(),
            };
            let mut chain: *const c_void = std::ptr::null();
            let ok = (procs.cert_get_chain)(std::ptr::null_mut(), cert, std::ptr::null(), store, &para, 0, std::ptr::null_mut(), &mut chain);
            assert_ne!(ok, 0);

            let certs = extract_chain_certs(chain);
            assert!(!certs.is_empty());
            let leaf = sha1_thumbprint(procs, certs[0]);
            assert_eq!(leaf.len(), 20);
            assert_eq!(leaf, sha1_thumbprint(procs, cert));

            (procs.cert_free_chain)(chain);
            (procs.cert_free_context)(cert);
            CertCloseStore(store, 0);
        }
    }
}
