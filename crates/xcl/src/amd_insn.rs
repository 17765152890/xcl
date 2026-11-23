//! AMD-only instructions of the console's Zen 2 on a CPU without them (Intel): `CLZERO`,
//! `MONITORX` / `MWAITX` and `RDPRU`. Titles are compiled for the console and use them – a
//! retail game's `memset` clears large blocks with `CLZERO`, its fence waits sleep on `MWAITX` –
//! and on an Intel CPU each one stops the game with an illegal instruction (0xC000001D).
//!
//! A vectored handler, installed only when the CPU lacks one of them, carries the instruction
//! out and resumes after it:
//!
//! * `CLZERO` zeroes the 64-byte line at `rAX`. A fault per line would make large clears crawl,
//!   so when the instruction is the body of one of the two loops compilers emit around it the
//!   handler clears the loop's whole range at once and resumes after the loop:
//!   - `clzero; add rax, 0x40; cmp rax, LIM; jb clzero`
//!   - `mov rax, P; add P, 0x40; clzero; cmp P, LIM; jb mov`
//! * `MONITORX` arms nothing; `MWAITX` yields the thread and returns (it may wake at any time,
//!   so callers check their condition again).
//! * `RDPRU` returns the time stamp counter for MPERF (ecx 0) and APERF (ecx 1), 0 otherwise.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, Once};

use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS};

const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const LINE: u64 = 64;

/// The AMD-only instructions this CPU lacks (empty on AMD Zen CPUs).
pub fn missing() -> Vec<&'static str> {
    use std::arch::x86_64::__cpuid;
    let mut missing = Vec::new();
    // `__cpuid` is safe in newer Rust, `unsafe` in older
    #[allow(unused_unsafe)]
    let cpuid = |leaf| unsafe { __cpuid(leaf) };
    let max_ext = cpuid(0x8000_0000).eax;
    let ext1 = if max_ext >= 0x8000_0001 { cpuid(0x8000_0001).ecx } else { 0 };
    let ext8 = if max_ext >= 0x8000_0008 { cpuid(0x8000_0008).ebx } else { 0 };
    if ext8 & 1 == 0 {
        missing.push("CLZERO");
    }
    if ext1 & (1 << 29) == 0 {
        missing.push("MONITORX/MWAITX");
    }
    if ext8 & (1 << 4) == 0 {
        missing.push("RDPRU");
    }
    missing
}

/// Install the handler if this CPU lacks any of the instructions (once; later calls do nothing).
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let missing = missing();
        if missing.is_empty() {
            return;
        }
        // SAFETY: the handler is a plain function that lives as long as the process
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
        crate::xlog_info!("CPU: {} emulated (AMD-only instructions of the console's CPU)", missing.join(", "));
    });
}

/// What the code at a faulting address was found to be (decoded once per address).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Site {
    /// a lone CLZERO
    Clzero,
    /// `clzero; add rax, 0x40; cmp rax, lim; jb` – resume at `end`
    ClzeroLoopA { lim: u8, end: u64 },
    /// `mov rax, p; add p, 0x40; clzero; cmp p, lim; jb` – resume at `end`
    ClzeroLoopB { p: u8, lim: u8, end: u64 },
    Monitorx,
    Mwaitx,
    Rdpru,
}

static SITES: Mutex<Option<HashMap<u64, Site>>> = Mutex::new(None);

/// `REX.W reg-reg` instruction `opcode` at `b`: (reg, rm) register numbers.
fn rr(b: &[u8], opcode: u8) -> Option<(u8, u8)> {
    if b.len() < 3 || b[0] & 0xF8 != 0x48 || b[1] != opcode || b[2] >> 6 != 3 {
        return None;
    }
    let reg = ((b[0] >> 2) & 1) << 3 | (b[2] >> 3) & 7;
    let rm = (b[0] & 1) << 3 | b[2] & 7;
    Some((reg, rm))
}

/// `add r64, 0x40` (REX.W 83 /0 ib) at `b`: the register.
fn add_line(b: &[u8]) -> Option<u8> {
    if b.len() < 4 || b[0] & 0xFA != 0x48 || b[1] != 0x83 || b[2] & 0xF8 != 0xC0 || b[3] != 0x40 {
        return None;
    }
    Some((b[0] & 1) << 3 | b[2] & 7)
}

/// Decode the instruction at `rip` – `code` holds the 7 bytes before it and 16 from it on
/// (`before` is false when the bytes before are not readable).
fn decode(rip: u64, code: &[u8; 23], before: bool) -> Option<Site> {
    let at = &code[7..];
    if at[0] != 0x0F || at[1] != 0x01 {
        return None;
    }
    Some(match at[2] {
        0xFA => Site::Monitorx,
        0xFB => Site::Mwaitx,
        0xFD => Site::Rdpru,
        0xFC => {
            // A: clzero; add rax, 0x40; cmp rax, lim; jb clzero
            if add_line(&at[3..]) == Some(0) {
                if let Some((lim, 0)) = rr(&at[7..], 0x39) {
                    if at[10] == 0x72 && at[11] as i8 as i64 == -12 && lim != 0 {
                        return Some(Site::ClzeroLoopA { lim, end: rip + 12 });
                    }
                }
            }
            // B: mov rax, p; add p, 0x40; clzero; cmp p, lim; jb mov
            if before {
                if let (Some((p, 0)), Some(p2)) = (rr(&code[0..], 0x89), add_line(&code[3..])) {
                    if let Some((lim, p3)) = rr(&at[3..], 0x39) {
                        if p != 0 && p == p2 && p == p3 && lim != p && lim != 0 && at[6] == 0x72 && at[7] as i8 as i64 == -15 {
                            return Some(Site::ClzeroLoopB { p, lim, end: rip + 8 });
                        }
                    }
                }
            }
            Site::Clzero
        }
        _ => return None,
    })
}

fn reg(ctx: &mut CONTEXT, n: u8) -> &mut u64 {
    match n {
        0 => &mut ctx.Rax,
        1 => &mut ctx.Rcx,
        2 => &mut ctx.Rdx,
        3 => &mut ctx.Rbx,
        4 => &mut ctx.Rsp,
        5 => &mut ctx.Rbp,
        6 => &mut ctx.Rsi,
        7 => &mut ctx.Rdi,
        8 => &mut ctx.R8,
        9 => &mut ctx.R9,
        10 => &mut ctx.R10,
        11 => &mut ctx.R11,
        12 => &mut ctx.R12,
        13 => &mut ctx.R13,
        14 => &mut ctx.R14,
        _ => &mut ctx.R15,
    }
}

/// The arithmetic flags `cmp a, b` leaves.
fn set_cmp_flags(ctx: &mut CONTEXT, a: u64, b: u64) {
    let r = a.wrapping_sub(b);
    let mut f = ctx.EFlags & !(0x0001 | 0x0004 | 0x0010 | 0x0040 | 0x0080 | 0x0800);
    if a < b {
        f |= 0x0001; // CF
    }
    if (r as u8).count_ones() % 2 == 0 {
        f |= 0x0004; // PF
    }
    if (a ^ b ^ r) & 0x10 != 0 {
        f |= 0x0010; // AF
    }
    if r == 0 {
        f |= 0x0040; // ZF
    }
    if r >> 63 != 0 {
        f |= 0x0080; // SF
    }
    if ((a ^ b) & (a ^ r)) >> 63 != 0 {
        f |= 0x0800; // OF
    }
    ctx.EFlags = f;
}

/// `[p, p + len)` committed and writable.
fn writable(p: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Memory::{
        VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD,
        PAGE_READWRITE, PAGE_WRITECOPY,
    };
    let Some(end) = p.checked_add(len) else { return false };
    let mut cur = p;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery only reads the address space's bookkeeping
        let n = unsafe { VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0
            || mbi.State != MEM_COMMIT
            || mbi.Protect & PAGE_GUARD != 0
            || mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) == 0
        {
            return false;
        }
        cur = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
    }
    true
}

/// Zero the lines `[first, end)` (`first` line-aligned); false if not all of it is writable.
fn zero_lines(first: u64, end: u64) -> bool {
    if end <= first || !writable(first, end - first) {
        return false;
    }
    // SAFETY: checked committed and writable above; CLZERO would write the same bytes
    unsafe { std::ptr::write_bytes(first as *mut u8, 0, (end - first) as usize) };
    true
}

fn site_at(rip: u64) -> Option<Site> {
    if let Some(s) = SITES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&rip)) {
        return Some(*s);
    }
    let before = crate::gpu::shader::readable(rip as usize - 7, 7);
    if !crate::gpu::shader::readable(rip as usize, 16) {
        return None;
    }
    let mut code = [0u8; 23];
    // SAFETY: both ranges checked readable
    unsafe {
        if before {
            std::ptr::copy_nonoverlapping((rip - 7) as *const u8, code.as_mut_ptr(), 7);
        }
        std::ptr::copy_nonoverlapping(rip as *const u8, code.as_mut_ptr().add(7), 16);
    }
    let site = decode(rip, &code, before)?;
    crate::xlog_info!("CPU: emulating {site:?} at {}", crate::modules::describe_address(rip as usize));
    SITES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(rip, site);
    Some(site)
}

/// Carry out the instruction at `ctx.Rip`; false: not one of ours (or its memory is not
/// writable – left to fault as it would).
fn emulate(ctx: &mut CONTEXT) -> bool {
    let rip = ctx.Rip;
    let Some(site) = site_at(rip) else { return false };
    match site {
        Site::Clzero => {
            let line = ctx.Rax & !(LINE - 1);
            if !zero_lines(line, line + LINE) {
                return false;
            }
            ctx.Rip = rip + 3;
        }
        Site::ClzeroLoopA { lim, end } => {
            // do { clzero [rax]; rax += 64 } while (rax < lim)
            let start = ctx.Rax;
            let limit = *reg(ctx, lim);
            let iterations = if start < limit { (limit - start).div_ceil(LINE) } else { 1 };
            let last = start.wrapping_add((iterations - 1) * LINE);
            if last < start || !zero_lines(start & !(LINE - 1), (last & !(LINE - 1)) + LINE) {
                return false;
            }
            let after = last + LINE;
            ctx.Rax = after;
            set_cmp_flags(ctx, after, limit);
            ctx.Rip = end;
        }
        Site::ClzeroLoopB { p, lim, end } => {
            // clzero [rax]; while (p < lim) { rax = p; p += 64; clzero [rax] }
            let first = ctx.Rax;
            let mut next = *reg(ctx, p);
            let limit = *reg(ctx, lim);
            let mut last = first;
            if next < limit && next > first {
                let more = (limit - next).div_ceil(LINE);
                last = next + (more - 1) * LINE;
                next += more * LINE;
            }
            // the lines are consecutive (p ran one line ahead of rax); otherwise only rax's
            let contiguous = last == first || first.wrapping_add(LINE) == *reg(ctx, p);
            if !contiguous {
                let line = first & !(LINE - 1);
                if !zero_lines(line, line + LINE) {
                    return false;
                }
                ctx.Rip = rip + 3;
                return true;
            }
            if !zero_lines(first & !(LINE - 1), (last & !(LINE - 1)) + LINE) {
                return false;
            }
            ctx.Rax = last;
            *reg(ctx, p) = next;
            set_cmp_flags(ctx, next, limit);
            ctx.Rip = end;
        }
        Site::Monitorx => ctx.Rip = rip + 3,
        Site::Mwaitx => {
            std::thread::yield_now();
            ctx.Rip = rip + 3;
        }
        Site::Rdpru => {
            // SAFETY: rdtsc is available on every x86-64 CPU
            let v = if ctx.Rcx as u32 <= 1 { unsafe { std::arch::x86_64::_rdtsc() } } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
            // CF set: the register was read (OF, SF, ZF, AF, PF cleared)
            ctx.EFlags = (ctx.EFlags & !(0x0001 | 0x0004 | 0x0010 | 0x0040 | 0x0080 | 0x0800)) | 0x0001;
            ctx.Rip = rip + 3;
        }
    }
    true
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let info = &*info;
    if info.ExceptionRecord.is_null() || info.ContextRecord.is_null() || (*info.ExceptionRecord).ExceptionCode as u32 != STATUS_ILLEGAL_INSTRUCTION {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if emulate(&mut *info.ContextRecord) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(before: &[u8], at: &[u8]) -> ([u8; 23], bool) {
        let mut c = [0u8; 23];
        c[7 - before.len()..7].copy_from_slice(before);
        c[7..7 + at.len()].copy_from_slice(at);
        (c, before.len() == 7)
    }

    #[test]
    fn decodes_the_retail_loops() {
        // clzero; add rax, 0x40; cmp rax, r10; jb -12
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::ClzeroLoopA { lim: 10, end: 0x100C }));
        // ... cmp rax, rdx
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::ClzeroLoopA { lim: 2, end: 0x100C }));
        // mov rax, r10; add r10, 0x40; clzero; cmp r10, r11; jb -15
        let (c, b) = bytes(&[0x4C, 0x89, 0xD0, 0x49, 0x83, 0xC2, 0x40], &[0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xDA, 0x72, 0xF1]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::ClzeroLoopB { p: 10, lim: 11, end: 0x1008 }));
        // mov rax, rdx; add rdx, 0x40; clzero; cmp rdx, r8; jb -15
        let (c, b) = bytes(&[0x48, 0x89, 0xD0, 0x48, 0x83, 0xC2, 0x40], &[0x0F, 0x01, 0xFC, 0x4C, 0x39, 0xC2, 0x72, 0xF1]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::ClzeroLoopB { p: 2, lim: 8, end: 0x1008 }));
        // a lone one, and the other instructions
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xFC, 0xC3]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::Clzero));
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xFA]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::Monitorx));
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xFB]);
        assert_eq!(decode(0x1000, &c, b), Some(Site::Mwaitx));
        let (c, b) = bytes(&[], &[0x0F, 0x01, 0xD0]); // xgetbv: not ours
        assert_eq!(decode(0x1000, &c, b), None);
    }

    /// Run `code` (a function of (rcx, rdx)) from executable memory.
    fn run(code: &[u8], a: usize, b: usize) {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE};
        install();
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
            assert!(!p.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), p as *mut u8, code.len());
            let f: extern "win64" fn(usize, usize) = std::mem::transmute(p);
            f(a, b);
            VirtualFree(p, 0, MEM_RELEASE);
        }
    }

    #[repr(C, align(64))]
    struct Buf([u8; 1024]);

    fn check_cleared(buf: &Buf, from: usize, to: usize) {
        for (i, &v) in buf.0.iter().enumerate() {
            assert_eq!(v, if (from..to).contains(&i) { 0 } else { 0xAA }, "byte {i}");
        }
    }

    #[test]
    fn clzero_loops_clear_their_range() {
        // mov rax, rcx; mov r10, rdx; L: clzero; add rax, 0x40; cmp rax, r10; jb L; ret
        let a = [0x48, 0x89, 0xC8, 0x49, 0x89, 0xD2, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4, 0xC3];
        let mut buf = Buf([0xAA; 1024]);
        let base = buf.0.as_mut_ptr() as usize;
        run(&a, base + 128, base + 640);
        check_cleared(&buf, 128, 640);

        // mov r10, rcx; mov r11, rdx; L: mov rax, r10; add r10, 0x40; clzero; cmp r10, r11; jb L; ret
        let b = [0x49, 0x89, 0xCA, 0x49, 0x89, 0xD3, 0x4C, 0x89, 0xD0, 0x49, 0x83, 0xC2, 0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xDA, 0x72, 0xF1, 0xC3];
        let mut buf = Buf([0xAA; 1024]);
        let base = buf.0.as_mut_ptr() as usize;
        run(&b, base + 64, base + 576);
        check_cleared(&buf, 64, 576);

        // a lone clzero with an address inside the line: mov rax, rcx; clzero; ret
        let c = [0x48, 0x89, 0xC8, 0x0F, 0x01, 0xFC, 0xC3];
        let mut buf = Buf([0xAA; 1024]);
        let base = buf.0.as_mut_ptr() as usize;
        run(&c, base + 200, 0);
        check_cleared(&buf, 192, 256);
    }

    #[test]
    fn monitorx_mwaitx_return() {
        if !missing().contains(&"MONITORX/MWAITX") {
            return; // natively, an untimed mwaitx waits for a write
        }
        // mov rax, rcx; xor ecx, ecx; xor edx, edx; monitorx; xor eax, eax; xor ecx, ecx; mwaitx; ret
        let code = [0x48, 0x89, 0xC8, 0x31, 0xC9, 0x31, 0xD2, 0x0F, 0x01, 0xFA, 0x31, 0xC0, 0x31, 0xC9, 0x0F, 0x01, 0xFB, 0xC3];
        let cell = 0u64;
        run(&code, &cell as *const u64 as usize, 0);
    }
}
