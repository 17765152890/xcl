//! Console fixture: send a dummy report via WinHTTP, without crashing or creating a dump.

use std::ffi::c_void;
use std::ptr::null;

#[link(name = "kernelx", kind = "raw-dylib")]
extern "system" {
    fn GetConsoleType() -> u32;
}

#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> usize;
    fn GetProcAddress(module: usize, name: *const u8) -> usize;
}

type Handle = *mut c_void;

#[link(name = "winhttp", kind = "raw-dylib")]
extern "system" {
    fn WinHttpOpen(agent: *const u16, access: u32, proxy: *const u16, bypass: *const u16, flags: u32) -> Handle;
    fn WinHttpSetTimeouts(session: Handle, resolve: i32, connect: i32, send: i32, receive: i32) -> i32;
    fn WinHttpOpenRequest(connection: Handle, verb: *const u16, path: *const u16, version: *const u16, referer: *const u16, accept: *const *const u16, flags: u32) -> Handle;
    fn WinHttpSendRequest(request: Handle, headers: *const u16, header_len: u32, data: *const c_void, len: u32, total: u32, context: usize) -> i32;
    fn WinHttpCloseHandle(h: Handle) -> i32;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

fn main() {
    let port = std::env::args().nth(1).unwrap().parse::<u16>().unwrap();
    assert_eq!(unsafe { GetConsoleType() }, 6);
    assert_eq!(std::env::var("XCL_PRIVACY_MARKER").unwrap(), "keep=雪");
    assert_eq!(std::env::var_os("LOCALAPPDATA").unwrap(), std::env::var_os("TEMP").unwrap());
    assert!(std::env::var_os("XCL_HOST").is_none());

    let dummy = std::env::vars_os()
        .find(|(key, _)| key.eq_ignore_ascii_case("DUMMY_CRASH_SECRET"))
        .map(|(_, value)| value.into_string().unwrap())
        .unwrap_or_default();
    let body = format!("synthetic-report:{dummy}");

    unsafe {
        let winhttp = LoadLibraryW(wide("winhttp.dll").as_ptr());
        assert_ne!(winhttp, 0);
        let connect = GetProcAddress(winhttp, c"WinHttpConnect".as_ptr().cast());
        assert_ne!(connect, 0);
        let connect: unsafe extern "system" fn(Handle, *const u16, u16, u32) -> Handle = std::mem::transmute(connect);

        const WINHTTP_ACCESS_TYPE_NO_PROXY: u32 = 1;
        let session = WinHttpOpen(wide("XCL privacy probe").as_ptr(), WINHTTP_ACCESS_TYPE_NO_PROXY, null(), null(), 0);
        assert!(!session.is_null(), "WinHttpOpen: {}", std::io::Error::last_os_error());
        assert_ne!(WinHttpSetTimeouts(session, 5000, 5000, 5000, 5000), 0);
        let conn = connect(session, wide("127.0.0.1").as_ptr(), port, 0);
        if conn.is_null() {
            let error = std::io::Error::last_os_error();
            WinHttpCloseHandle(session);
            eprintln!("privacy probe: WinHttpConnect: {error}");
            std::process::exit(error.raw_os_error().unwrap());
        }

        let request = WinHttpOpenRequest(conn, wide("POST").as_ptr(), wide("/synthetic-report").as_ptr(), null(), null(), null(), 0);
        assert!(!request.is_null(), "WinHttpOpenRequest: {}", std::io::Error::last_os_error());
        let len = body.len() as u32;
        assert_ne!(WinHttpSendRequest(request, null(), 0, body.as_ptr().cast(), len, len, 0), 0, "WinHttpSendRequest: {}", std::io::Error::last_os_error());
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(conn);
        WinHttpCloseHandle(session);
    }
}
