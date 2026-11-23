//! BCPack's entropy stage: N byte-wise rANS lanes fed from one shared byte stream.
//!
//! Each lane is a cycle model of the decoder in Microsoft's silicon-rans (`mdu_minirans.v`, MIT):
//! a lane alternates phase 0 (take a byte from the feeder into its one-byte input buffer, hand
//! its decoded symbol to the output vector and apply the state update) and phase 1 (parse a
//! fragment header, or move the buffered byte into the state and decode the next symbol). A lane
//! whose input buffer is empty while its fragment has payload left asks for a byte in phase 0;
//! if the feeder does not serve it, it stalls that phase 0 and the next phase 1. The feeder and
//! the vector assembly follow the SixNG notes ("6. BCPack", entropy decoder in full): requesting
//! lanes are served one byte each in lane order, at most 8 bytes a clock, from the one stream
//! (a chunk has no stream header – its first byte is lane 0's first fragment header); a lane's
//! symbol leaves for its slot of the output vector when the slot is empty, and the vector is
//! released once every slot is full.
//!
//! Fragment header: byte 0 = z (alphabet narrowing, bits 7..6), long (bit 5), flush (bit 4),
//! model q (bits 3..0); byte 1 (+ byte 2 when long) = payload length - 1 (or - 257). State: 32 bits,
//! L = 2^24, 16-bit probabilities. After its last symbol a lane's state must be L, L >> 8 or
//! L >> 16 (the encoder may trim up to two trailing zero bytes).

use super::tables::MODELS;

const L: u32 = 1 << 24;
/// bytes the feeder hands out per clock
const FEED: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum St {
    Idle,
    Hdr0,
    Hdr1,
    Hdr2,
    Processing,
    Draining,
}

struct Lane {
    alphabet_bits: u8,
    st: St,
    done: bool,
    ib: u8,
    ib_full: bool,
    input_remaining: u32,
    output_remaining: u32,
    p0: u32,
    p1: u32,
    sym: u16,
    sym_full: bool,
    eab: u8,
    q: u8,
    hdr3: bool,
    flush: bool,
    stall_p1: bool,
    slot: Option<u16>,
    out: Vec<u16>,
}

impl Lane {
    fn need_ib_load(&self) -> bool {
        !self.ib_full && (matches!(self.st, St::Hdr0 | St::Hdr1 | St::Hdr2) || (self.st == St::Processing && self.input_remaining > 0))
    }

    fn blocked_on_output(&self) -> bool {
        self.sym_full && self.slot.is_some()
    }

    /// The state after `sym` was taken from state `x` (the forward step).
    fn forward(&self, x: u32, sym: u16) -> Result<u32, String> {
        let (freq, cum) = forward(self.eab, self.q, sym)?;
        Ok(freq.wrapping_mul(x >> 16).wrapping_add(x & 0xFFFF).wrapping_sub(cum))
    }

    /// Phase 1 of a lane that did not stall.
    fn phase1(&mut self, lane: usize) -> Result<(), String> {
        match self.st {
            St::Idle => {}
            St::Hdr0 => {
                if self.ib_full {
                    let b = self.ib;
                    self.input_remaining = 0;
                    self.st = St::Hdr1;
                    self.q = b & 0xF;
                    self.hdr3 = b & 0x20 != 0;
                    self.flush = b & 0x10 != 0;
                    let e = self.alphabet_bits.wrapping_sub(b >> 6) & 0xF;
                    if e > 8 {
                        return Err(format!("lane {lane}: alphabet narrowing past 1 bit (header {b:#04x})"));
                    }
                    self.eab = e;
                    self.ib_full = false;
                }
            }
            St::Hdr1 => {
                if self.ib_full {
                    self.ib_full = false;
                    if self.hdr3 {
                        self.input_remaining = self.ib as u32;
                        self.st = St::Hdr2;
                    } else {
                        self.input_remaining = self.ib as u32 + 1;
                        self.st = St::Processing;
                    }
                }
            }
            St::Hdr2 => {
                if self.ib_full {
                    self.input_remaining = ((self.ib as u32) << 8 | (self.input_remaining & 0xFF)) + 257;
                    self.st = St::Processing;
                    self.ib_full = false;
                }
            }
            St::Processing | St::Draining => {
                let feed = self.st == St::Processing && self.p0 < L && self.input_remaining > 0 && self.ib_full;
                let x = if feed { (self.p0 << 8) | self.ib as u32 } else { self.p0 };
                let next_full = x > L && self.output_remaining > 0;
                self.p1 = x;
                if feed {
                    self.input_remaining -= 1;
                    self.ib_full = false;
                }
                if next_full {
                    self.output_remaining -= 1;
                    self.sym = inverse(self.eab, self.q, x & 0xFFFF).ok_or_else(|| format!("lane {lane}: no symbol for cf {:#x} (eab {}, q {})", x & 0xFFFF, self.eab, self.q))?;
                }
                self.sym_full = next_full;
                if self.st == St::Processing {
                    if self.input_remaining == 0 {
                        if self.output_remaining > 0 {
                            if x == L {
                                self.st = St::Hdr0;
                                if self.flush {
                                    self.p1 = 0;
                                }
                            } else {
                                self.st = St::Draining;
                            }
                        } else {
                            self.st = St::Idle;
                            self.done = true;
                        }
                    }
                } else if x <= L {
                    if self.output_remaining > 0 {
                        self.st = St::Hdr0;
                        if self.flush {
                            self.p0 = 0;
                            self.p1 = 0;
                        }
                    } else {
                        self.st = St::Idle;
                        self.done = true;
                    }
                }
            }
        }
        Ok(())
    }
}

fn model(eab: u8, q: u8) -> &'static [(u32, u32); 10] {
    &MODELS[eab as usize * 16 + q as usize]
}

/// (freq, cum) of `sym` in model (`eab`, `q`); cumulative value 0 is reserved, so a symbol
/// that would start there is (freq - 1, 1).
fn forward(eab: u8, q: u8, sym: u16) -> Result<(u32, u32), String> {
    let k = 16 - sym.leading_zeros() as usize; // 0 for symbol 0, else the segment
    if k > eab as usize + 1 {
        return Err(format!("symbol {sym} outside the alphabet of eab {eab}"));
    }
    let (base, freq) = model(eab, q)[k];
    let first = if k == 0 { 0 } else { 1u32 << (k - 1) };
    let cum = (base + freq * (sym as u32 - first)) & 0xFFFF;
    Ok(if cum == 0 { (freq - 1, 1) } else { (freq, cum) })
}

/// The symbol of model (`eab`, `q`) whose cumulative range holds `cf`.
pub(super) fn inverse(eab: u8, q: u8, cf: u32) -> Option<u16> {
    let m = model(eab, q);
    // the highest segment whose base `cf` reaches (bases rise with the segment)
    let mut k = 0;
    for s in (1..=eab as usize + 1).rev() {
        if cf >= m[s].0 {
            k = s;
            break;
        }
    }
    let (base, freq) = m[k];
    if freq == 0 || cf < base {
        return None;
    }
    let first = if k == 0 { 0 } else { 1u32 << (k - 1) };
    let count = if k == 0 { 1 } else { 1u32 << (k - 1) };
    // symbol 0's range starts at 1 (value 0 reserved), the others at their cum
    let j = ((cf - base) / freq).min(count - 1);
    Some((first + j) as u16)
}

/// Decode `symbols` symbols in each lane (symbol widths in bits) from `src`: the symbols per
/// lane and the number of source bytes the lanes took.
pub fn decode_lanes(src: &[u8], widths: &[u8], symbols: usize) -> Result<(Vec<Vec<u16>>, usize), String> {
    let mut lanes: Vec<Lane> = widths
        .iter()
        .map(|&w| Lane {
            alphabet_bits: w - 1,
            st: St::Idle,
            done: symbols == 0,
            ib: 0,
            ib_full: false,
            input_remaining: 0,
            output_remaining: symbols as u32,
            p0: 0,
            p1: 0,
            sym: 0,
            sym_full: false,
            eab: 0,
            q: 0,
            hdr3: false,
            flush: false,
            stall_p1: false,
            slot: None,
            out: Vec::with_capacity(symbols),
        })
        .collect();
    if symbols == 0 {
        return Ok((lanes.into_iter().map(|l| l.out).collect(), 0));
    }
    let n = lanes.len();
    let mut pos = 0usize;
    let mut started = false;
    let mut idle_clocks = 0u32;
    loop {
        // ---- phase 0 ----
        let need: Vec<bool> = lanes.iter().map(Lane::need_ib_load).collect();
        let blocked_out: Vec<bool> = lanes.iter().map(Lane::blocked_on_output).collect();
        let mut fed = [None::<u8>; 32];
        let mut served = 0;
        for i in 0..n {
            if need[i] && !blocked_out[i] && served < FEED && pos < src.len() {
                fed[i] = Some(src[pos]);
                pos += 1;
                served += 1;
            }
        }
        let mut progress = served > 0;
        for (i, lane) in lanes.iter_mut().enumerate() {
            let stall = (need[i] && fed[i].is_none()) || blocked_out[i];
            lane.stall_p1 = stall;
            if stall {
                continue;
            }
            if lane.sym_full {
                let x = lane.forward(lane.p1, lane.sym).map_err(|e| format!("lane {i}: {e}"))?;
                // the last symbol leaves: the state must be a legal end state
                if lane.output_remaining == 0 && !(x == L || x == L >> 8 || x == L >> 16) {
                    return Err(format!("lane {i}: final state {x:#x}"));
                }
                if matches!(lane.st, St::Processing | St::Draining) {
                    lane.p0 = x;
                }
            } else if matches!(lane.st, St::Processing | St::Draining) {
                lane.p0 = lane.p1;
            }
            if let Some(b) = fed[i] {
                lane.ib = b;
                lane.ib_full = true;
            }
            if lane.sym_full && lane.slot.is_none() {
                lane.slot = Some(lane.sym);
                lane.sym_full = false;
                progress = true;
            }
        }
        // ---- phase 1 ----
        for (i, lane) in lanes.iter_mut().enumerate() {
            if !started && lane.st == St::Idle && !lane.done {
                // `go`: the first phase 1 starts every lane on its first header
                lane.st = St::Hdr0;
                lane.p0 = 0;
                lane.p1 = 0;
                progress = true;
                continue;
            }
            if lane.stall_p1 {
                continue;
            }
            let before = (lane.st, lane.sym_full, lane.ib_full);
            lane.phase1(i)?;
            progress |= before != (lane.st, lane.sym_full, lane.ib_full);
        }
        started = true;
        // the output vector leaves once every lane's slot holds a symbol
        if lanes.iter().all(|l| l.slot.is_some()) {
            for l in lanes.iter_mut() {
                let s = l.slot.take().unwrap();
                l.out.push(s);
            }
            progress = true;
        }
        if lanes.iter().all(|l| l.out.len() == symbols) {
            break;
        }
        if progress {
            idle_clocks = 0;
        } else {
            idle_clocks += 1;
            if idle_clocks > 8 {
                let stuck: Vec<String> = lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.out.len() < symbols)
                    .take(4)
                    .map(|(i, l)| format!("{i}: {} of {symbols}, {:?}, x {:#x}, {} left", l.out.len(), l.st, l.p0, l.input_remaining))
                    .collect();
                return Err(format!("stream ended at byte {pos} of {} (lanes {})", src.len(), stuck.join("; ")));
            }
        }
    }
    Ok((lanes.into_iter().map(|l| l.out).collect(), pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every model tiles cumulative values 1..65535 exactly once, and the forward and inverse
    /// steps agree on every value.
    #[test]
    fn models_tile_the_probability_range() {
        for eab in 0..9u8 {
            for q in 0..16u8 {
                let mut seen = vec![false; 65536];
                for s in 0..(1u32 << (eab + 1)) {
                    let (freq, cum) = forward(eab, q, s as u16).unwrap();
                    for cf in cum..cum + freq {
                        assert!(!seen[cf as usize], "eab {eab} q {q}: cf {cf} twice");
                        seen[cf as usize] = true;
                        assert_eq!(inverse(eab, q, cf), Some(s as u16), "eab {eab} q {q}: cf {cf}");
                    }
                }
                assert!(seen[1..].iter().all(|&b| b), "eab {eab} q {q}: a gap");
            }
        }
    }
}
