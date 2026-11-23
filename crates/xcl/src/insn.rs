//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ,
//! which are AMD's). The exception handler (`eh`) emulates the ones decoded here and steps over
//! them.

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `clzero` loop: `clzero; add rax, 0x40; cmp rax, <reg>; jb -12` – zero the lines from rAX
    /// on until rAX reaches the register (`clzero_loop`; `end_reg` is never rax)
    ClzeroLoop { end_reg: u8 },
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The arithmetic flags in EFLAGS: CF, PF, AF, ZF, SF and OF.
pub const ARITH_FLAGS: u32 = 0x8D5;

/// The arithmetic flags `cmp a, b` (64-bit) leaves.
pub fn cmp_flags(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    let cf = (a < b) as u32;
    let pf = (r as u8).count_ones() % 2 == 0;
    let af = (a ^ b ^ r) & 0x10 != 0;
    let zf = r == 0;
    let sf = r >> 63 != 0;
    let of = ((a ^ b) & (a ^ r)) >> 63 != 0;
    cf | (pf as u32) << 2 | (af as u32) << 4 | (zf as u32) << 6 | (sf as u32) << 7 | (of as u32) << 11
}

/// What the `ClzeroLoop` emulation does on one fault: zero `lines` 64-byte lines from `line`,
/// leave rAX and the flags as the last `add` / `cmp` did and, when the loop is not `done`, stay
/// at the clzero (the next fault goes on from there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClzeroRun {
    pub line: u64,
    pub lines: u64,
    pub rax: u64,
    pub flags: u32,
    pub done: bool,
}

/// The `clzero; add rax, 0x40; cmp rax, end; jb` loop from `rax`, at most `max_lines` (at least
/// one) of its iterations. As the CPU runs it: the body runs at least once (an `end` at or below
/// rAX zeroes one line), each clzero zeroes the line rAX is in (an unaligned rAX zeroes the line
/// below it; rAX moves by whole lines), and the loop ends at the first rAX not below `end`
/// (unsigned). None when rAX would wrap past the top of the address space, where the real loop
/// faults anyway.
pub fn clzero_loop(rax: u64, end: u64, max_lines: u64) -> Option<ClzeroRun> {
    let after_one = rax.checked_add(64)?;
    let total = if end <= after_one { 1 } else { (end - rax).div_ceil(64) };
    let lines = total.min(max_lines.max(1));
    let rax_after = rax.checked_add(lines.checked_mul(64)?)?;
    Some(ClzeroRun { line: rax & !63, lines, rax: rax_after, flags: cmp_flags(rax_after, end), done: lines == total })
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            // clzero (0F 01 FC); add rax, 0x40 (48 83 C0 40); cmp rax, <reg> ([48|4C] 39 modrm); jb -12 (72 F4)
            // (not after an 0x67: that CLZERO is at EAX, and the jb goes back to the 0F, past the
            // prefix – it is stepped alone, the loop is taken from the next one). The ModRM must be
            // mod 3 with rax in r/m – anything else compares another register or memory – and the
            // register is not rax itself (`cmp rax, rax` never loops: the clzero is stepped alone).
            if code.len() >= i + 12
                && !code[..i].contains(&0x67)
                && code[i + 3..i + 7] == [0x48, 0x83, 0xC0, 0x40]
                && (code[i + 7] == 0x48 || code[i + 7] == 0x4C)
                && code[i + 8] == 0x39
                && code[i + 9] & 0xC7 == 0xC0
                && (code[i + 7], code[i + 9]) != (0x48, 0xC0)
                && code[i + 10..i + 12] == [0x72, 0xF4]
            {
                let rex_r = code[i + 7] == 0x4C;
                let modrm = code[i + 9];
                let reg_idx = ((modrm >> 3) & 7) | if rex_r { 8 } else { 0 };
                Some((ConsoleOnly::ClzeroLoop { end_reg: reg_idx }, i + 12))
            } else {
                Some((ConsoleOnly::Clzero, i + 3))
            }
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

/// `code` as hex, for a log line.
pub fn hex(code: &[u8]) -> String {
    code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    /// The loop's tail with `cmp` bytes `rex 39 modrm`.
    fn loop_with(rex: u8, modrm: u8) -> [u8; 12] {
        [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, rex, 0x39, modrm, 0x72, 0xF4]
    }

    /// Only `cmp rax, <reg>` (ModRM mod 3, r/m 0, no REX.B) ends a CLZERO loop (#278).
    #[test]
    fn clzero_loop_needs_cmp_rax_with_a_register() {
        for (reg, modrm) in [(1u8, 0xC8u8), (2, 0xD0), (3, 0xD8), (4, 0xE0), (5, 0xE8), (6, 0xF0), (7, 0xF8)] {
            assert_eq!(decode(&loop_with(0x48, modrm)), Some((ConsoleOnly::ClzeroLoop { end_reg: reg }, 12)), "{modrm:02x}");
            assert_eq!(decode(&loop_with(0x4C, modrm)), Some((ConsoleOnly::ClzeroLoop { end_reg: reg + 8 }, 12)), "{modrm:02x}");
        }
        assert_eq!(decode(&loop_with(0x4C, 0xC0)), Some((ConsoleOnly::ClzeroLoop { end_reg: 8 }, 12)), "cmp rax, r8");
        // not the loop: the clzero alone
        for (rex, modrm, what) in [
            (0x48, 0xC0, "cmp rax, rax: never loops"),
            (0x48, 0xD1, "cmp rcx, rdx"),
            (0x48, 0xCA, "cmp rdx, rcx"),
            (0x48, 0x10, "cmp [rax], rdx"),
            (0x48, 0x50, "cmp [rax+disp8], rdx"),
            (0x48, 0x90, "cmp [rax+disp32], rdx"),
            (0x49, 0xD0, "cmp r8, rdx (REX.B)"),
            (0x4D, 0xD0, "cmp r8, r10 (REX.B)"),
        ] {
            assert_eq!(decode(&loop_with(rex, modrm)), Some((ConsoleOnly::Clzero, 3)), "{what}");
        }
    }

    #[test]
    fn cmp_flags_match_the_cpu() {
        const CF: u32 = 1;
        const PF: u32 = 4;
        const AF: u32 = 0x10;
        const ZF: u32 = 0x40;
        const SF: u32 = 0x80;
        const OF: u32 = 0x800;
        assert_eq!(cmp_flags(5, 5), ZF | PF);
        assert_eq!(cmp_flags(0x40, 0x80), CF | SF | PF); // r = 0xFFFF_FFFF_FFFF_FFC0
        assert_eq!(cmp_flags(0x80, 0x40), 0); // r = 0x40: one bit, odd parity
        assert_eq!(cmp_flags(0x10, 0x01), AF | PF); // r = 0x0F: four bits, borrow out of bit 3
        assert_eq!(cmp_flags(0x8000_0000_0000_0000, 1), OF | AF | PF); // r = 0x7FFF..FF
        assert_eq!(cmp_flags(0, 1), CF | SF | AF | PF);
        assert_eq!(cmp_flags(1, 0) & !ARITH_FLAGS, 0);
    }

    /// The loop as the CPU runs it (#278): whole lines from rAX's, rAX by whole lines, at least
    /// one iteration, and the flags of the last cmp.
    #[test]
    fn clzero_loop_runs_like_the_cpu() {
        /// The loop stepped one iteration at a time, as the CPU does it.
        fn reference(mut rax: u64, end: u64) -> (u64, u64, u64, u32) {
            let (line, mut lines) = (rax & !63, 0);
            loop {
                lines += 1;
                rax += 64;
                if rax >= end {
                    return (line, lines, rax, cmp_flags(rax, end));
                }
            }
        }
        let full = |rax, end| clzero_loop(rax, end, u64::MAX).unwrap();
        // aligned, and a whole number of lines: [0x1000, 0x1100)
        assert_eq!(full(0x1000, 0x1100), ClzeroRun { line: 0x1000, lines: 4, rax: 0x1100, flags: cmp_flags(0x1100, 0x1100), done: true });
        // an unaligned start: the line below it, and rAX keeps its offset
        assert_eq!(full(0x1010, 0x1100), ClzeroRun { line: 0x1000, lines: 4, rax: 0x1110, flags: cmp_flags(0x1110, 0x1100), done: true });
        // an end that is not a whole number of lines away: the last line is zeroed whole
        assert_eq!(full(0x1000, 0x1081), ClzeroRun { line: 0x1000, lines: 3, rax: 0x10C0, flags: cmp_flags(0x10C0, 0x1081), done: true });
        // end at or below rAX: the body still runs once
        for end in [0x1000, 0x0FFF, 0, 0x1040] {
            let r = full(0x1000, end);
            assert_eq!((r.line, r.lines, r.rax, r.done), (0x1000, 1, 0x1040, true), "end {end:#x}");
        }
        // a run cut to max_lines leaves rAX below end (CF set: the jb is taken) and is not done
        let r = clzero_loop(0x1000, 0x2000, 16).unwrap();
        assert_eq!((r.line, r.lines, r.rax, r.done), (0x1000, 16, 0x1400, false));
        assert_eq!(r.flags & 1, 1);
        let rest = full(r.rax, 0x2000);
        assert_eq!((rest.line, rest.lines, rest.rax), (0x1400, 48, 0x2000));
        assert_eq!(clzero_loop(0x1000, 0x2000, 0).unwrap().lines, 1, "at least one line a fault");
        // rAX wrapping past the top of the address space: left to fault
        assert_eq!(clzero_loop(u64::MAX - 0x20, 0, u64::MAX), None);
        assert_eq!(clzero_loop(u64::MAX - 0x1010, u64::MAX, u64::MAX), None, "the last add wraps");
        assert_eq!(clzero_loop(u64::MAX - 0x1000, u64::MAX, u64::MAX).map(|r| (r.lines, r.rax)), Some((64, u64::MAX)), "it just reaches the top");
        // against the stepped loop
        for rax in [0x1_0000u64, 0x1_0001, 0x1_003F, 0x1_0040, 0x1_0077] {
            for end in [0u64, 0x1_0000, 0x1_0001, 0x1_0040, 0x1_0041, 0x1_00C0, 0x1_00FF, 0x1_1000, 0x1_1001] {
                let r = full(rax, end);
                assert_eq!((r.line, r.lines, r.rax, r.flags), reference(rax, end), "rax {rax:#x} end {end:#x}");
                assert!(r.done);
            }
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // clzero loop against r10 (4C 39 D0 = cmp rax, r10)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 10 }, 12))
        );
        // clzero loop against rdx (48 39 D0 = cmp rax, rdx)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 2 }, 12))
        );
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        // SSE4a, as a retail game has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them (Kalmarv on #209): the last of F2 / F3 decides, 66 next
        // to them does not; 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }
}
