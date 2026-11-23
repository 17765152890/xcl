//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ,
//! which are AMD's). The exception handler (`eh`) emulates the ones decoded here and steps
//! over them.

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `insertq xmm(dst), xmm(src), len, idx` (F2 0F 78 /r ib ib): `insertq` of the low
    /// quadwords (a retail game has 19, in its float → half packing)
    Insertq { dst: u8, src: u8, len: u8, idx: u8 },
    /// SSE4a `extrq xmm(dst), len, idx` (66 0F 78 /0 ib ib): `extrq` of the low quadword
    Extrq { dst: u8, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// The mask of an SSE4a bit field of `len` bits (6 bits; 0 means 64).
fn field_mask(len: u8) -> u64 {
    match len & 63 {
        0 => u64::MAX,
        n => (1u64 << n) - 1,
    }
}

/// SSE4a INSERTQ on the low quadwords: the low `len` bits of `src` replace bits `idx ..` of
/// `dst` (the upper quadword is undefined on the hardware; the emulation leaves it).
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (mask, idx) = (field_mask(len), (idx & 63) as u32);
    (dst & !(mask << idx)) | ((src & mask) << idx)
}

/// SSE4a EXTRQ on the low quadword: bits `idx ..` of `v`, `len` of them, in its low bits.
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    (v >> (idx & 63)) & field_mask(len)
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
    let (mut p66, mut pf3, mut pf2, mut rex) = (false, false, false, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF3 => pf3 = true,
            0xF2 => pf2 = true,
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 => {}
            0x40..=0x4F => {
                rex = b;
                i += 1;
                break; // REX is the last prefix
            }
            _ => break,
        }
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => Some((ConsoleOnly::Clzero, i + 3)),
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        // SSE4a, register operands and two immediates (length, index): F2 0F 78 /r insertq,
        // 66 0F 78 /0 extrq
        (0x78, Some(m)) if m >> 6 == 3 && (pf2 || p66) => {
            let (len, idx) = (*code.get(i + 3)? & 63, *code.get(i + 4)? & 63);
            let (reg, rm) = (((m >> 3) & 7) | (rex & 4) << 1, (m & 7) | (rex & 1) << 3);
            if pf2 {
                Some((ConsoleOnly::Insertq { dst: reg, src: rm, len, idx }, i + 5))
            } else if (m >> 3) & 7 == 0 {
                Some((ConsoleOnly::Extrq { dst: rm, len, idx }, i + 5))
            } else {
                None
            }
        }
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

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
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
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }

    #[test]
    fn decodes_sse4a() {
        // a retail game's: insertq xmm0, xmm1, 16, 16; insertq xmm9, xmm7, 8, 0 (REX.R);
        // insertq xmm11, xmm12, 16, 16 (REX.R + REX.B)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, len: 16, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x44, 0x0F, 0x78, 0xCF, 0x08, 0x00]), Some((ConsoleOnly::Insertq { dst: 9, src: 7, len: 8, idx: 0 }, 7)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x78, 0xDC, 0x10, 0x10]), Some((ConsoleOnly::Insertq { dst: 11, src: 12, len: 16, idx: 16 }, 7)));
        // extrq xmm2, 8, 4; extrq xmm10, 16, 32
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x08, 0x04]), Some((ConsoleOnly::Extrq { dst: 2, len: 8, idx: 4 }, 6)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0x78, 0xC2, 0x10, 0x20]), Some((ConsoleOnly::Extrq { dst: 10, len: 16, idx: 32 }, 7)));
        // not SSE4a: 0F 78 without a prefix (vmread), a memory operand, extrq with /1, cut short
        assert_eq!(decode(&[0x0F, 0x78, 0xC1]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0x01, 0x10, 0x10]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xCA, 0x08, 0x04]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x10]), None);
    }

    /// As AMD's manual defines them (APM vol. 4): the field of `len` bits at `idx`, 0 = 64 bits.
    #[test]
    fn sse4a_semantics() {
        assert_eq!(insertq(u64::MAX, 0x1234, 16, 16), 0xFFFF_FFFF_1234_FFFF);
        assert_eq!(insertq(0, 0xABCD_EF12, 8, 0), 0x12, "only the low len bits of the source");
        assert_eq!(insertq(0x1111_1111_1111_1111, 0x2222_2222_2222_2222, 0, 0), 0x2222_2222_2222_2222, "length 0 is 64");
        assert_eq!(insertq(0, 0x3, 2, 62), 0xC000_0000_0000_0000);
        assert_eq!(extrq(0xFFFF_FFFF_1234_FFFF, 16, 16), 0x1234);
        assert_eq!(extrq(0x8000_0000_0000_0001, 0, 0), 0x8000_0000_0000_0001, "length 0 is 64");
        assert_eq!(extrq(0xF0, 4, 4), 0xF);
    }
}
