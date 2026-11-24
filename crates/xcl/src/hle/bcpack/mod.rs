//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format,
//! kept in `tables.rs` beside this file ([`Tables::builtin`]).
//!
//! Ported from VIctory (<https://github.com/Marvisak/VIctory>,
//! `crates/victory-formats/src/bcpack.rs`, GPL-3.0-or-later), used here under GPL-2.0-only with the
//! author's permission (see THIRD_PARTY.md). Changed for XCL on 2026-11-24: moved to `hle::bcpack`,
//! the constant tables are generated Rust source (`tables.rs`) instead of an `include_bytes!` binary.

// Parts of `Tables` are read but not used until the BC6H (and BC7) decoders land.
#![allow(dead_code)]

use std::fmt;

mod tables;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const HISTORY: usize = 256;
const DICTIONARY: usize = 255;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;

/// Bytes of the table file (see [`Tables::parse`]).
pub const TABLES_SIZE: usize = MODELS * BUCKETS * 2
    + 256
    + 64
    + 256
    + 64
    + 64
    + 512 * 2
    + DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);

/// Endpoints and indices of BC1, BC3, BC4 and BC5: small unsigned values,
/// summed per element modulo the element's width.
#[derive(Clone, Copy, Default)]
struct Plain {
    ep: [u8; 8],
    ix: [u8; 32],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ep {
    /// Endpoint 0 R G B, then endpoints 1-3, left-aligned and biased.
    e: [[u16; 3]; 4],
}

#[derive(Clone, Copy, Default)]
struct Bc6Ix {
    mode: u8,
    ix: [i16; 16],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ep {
    ctrl: u16,
    /// Per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned.
    rows: [[u16; 8]; 3],
}

#[derive(Clone, Copy, Default)]
struct Bc7Ix {
    bits: [u8; 2],
    ix: [[i16; 16]; 2],
}

/// The format's constant tables.
pub struct Tables {
    freq: [[u32; BUCKETS]; MODELS],
    /// Each bucket's first slot; 65536 past the model's last, and one more
    /// 65536 at the end so a bucket always has a next start.
    start: [[u32; BUCKETS + 1]; MODELS],
    /// Per model, the bucket of the first slot of each 256-slot window: the
    /// search for a slot's bucket starts there and seldom moves on.
    bucket: Box<[[u8; 256]; MODELS]>,
    /// ceil(2^32 / freq): exact division for the offsets inside a bucket.
    recip: [[u64; BUCKETS]; MODELS],
    rank1: [u8; 256],
    rank3a: [u8; 64],
    rank3c: [u8; 256],
    rank4: [u8; 64],
    rank5: [u8; 64],
    ctrl7: [u16; 512],
    bc1: Vec<Plain>,
    bc3: Vec<Plain>,
    bc4: Vec<Plain>,
    bc5: Vec<Plain>,
    bc6: Vec<(Bc6Ep, Bc6Ix)>,
    bc7: Vec<(Bc7Ep, Bc7Ix)>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        head.try_into().unwrap_or([0; N])
    }
    fn u8(&mut self) -> u8 {
        self.bytes::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes())
    }
}

impl Tables {
    /// The built-in tables.
    pub fn builtin() -> Result<Tables, Error> {
        Tables::parse(&tables::TABLES)
    }

    /// Reads the table file: 128 x 9 u16 model frequencies (per bucket, the
    /// frequency of each of its symbols), the rank tables (BC1 256 B, BC3
    /// alpha 64 B and colour 256 B, BC4 64 B, BC5 64 B), 512 u16 BC7 control
    /// words, then 255 dictionary entries per format in format order: BC1
    /// 6 + 16 B, BC3 8 + 32 B, BC4 2 + 16 B, BC5 4 + 32 B, BC6H (u8 mode,
    /// 12 u16, u8 mode, 16 i16), BC7 (u16 control, 24 u16, 2 u8, 32 i16).
    /// Little-endian, no padding. Every model must tile 65536.
    pub fn parse(bytes: &[u8]) -> Result<Tables, Error> {
        if bytes.len() != TABLES_SIZE {
            return Err(Error("table file has the wrong size"));
        }
        let mut r = Reader(bytes);
        let mut freq = [[0u32; BUCKETS]; MODELS];
        let mut start = [[0u32; BUCKETS + 1]; MODELS];
        let mut bucket = Box::new([[0u8; 256]; MODELS]);
        let mut recip = [[0u64; BUCKETS]; MODELS];
        for m in 0..MODELS {
            for f in &mut freq[m] {
                *f = u32::from(r.u16());
            }
            // Model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols.
            let used = m / 16 + 2;
            let mut at = 0u32;
            for b in 0..used {
                if freq[m][b] == 0 {
                    return Err(Error("a model has an empty bucket"));
                }
                start[m][b] = at;
                recip[m][b] = (1u64 << 32).div_ceil(u64::from(freq[m][b]));
                at += freq[m][b] << b.saturating_sub(1);
            }
            if at != 1 << 16 {
                return Err(Error("a model does not tile 65536"));
            }
            start[m][used..].fill(1 << 16);
            for (window, b) in bucket[m].iter_mut().enumerate() {
                let first = (window as u32) << 8;
                *b = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
            }
        }
        let rank1 = r.bytes();
        let rank3a = r.bytes();
        let rank3c = r.bytes();
        let rank4 = r.bytes();
        let rank5 = r.bytes();
        let mut ctrl7 = [0u16; 512];
        for c in &mut ctrl7 {
            *c = r.u16();
        }
        let mut plain = |ep: usize, ix: usize| {
            (0..DICTIONARY)
                .map(|_| {
                    let mut p = Plain::default();
                    for v in &mut p.ep[..ep] {
                        *v = r.u8();
                    }
                    for v in &mut p.ix[..ix] {
                        *v = r.u8();
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        let bc1 = plain(6, 16);
        let bc3 = plain(8, 32);
        let bc4 = plain(2, 16);
        let bc5 = plain(4, 32);
        let bc6 = (0..DICTIONARY)
            .map(|_| {
                // The endpoint entry's mode byte plays no part in decoding.
                r.u8();
                let mut ep = Bc6Ep::default();
                for v in ep.e.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc6Ix {
                    mode: r.u8(),
                    ..Default::default()
                };
                for v in &mut ix.ix {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        let bc7 = (0..DICTIONARY)
            .map(|_| {
                let mut ep = Bc7Ep {
                    ctrl: r.u16(),
                    ..Default::default()
                };
                for v in ep.rows.as_flattened_mut() {
                    *v = r.u16();
                }
                let mut ix = Bc7Ix {
                    bits: [r.u8(), r.u8()],
                    ..Default::default()
                };
                for v in ix.ix.as_flattened_mut() {
                    *v = r.u16() as i16;
                }
                (ep, ix)
            })
            .collect();
        Ok(Tables {
            freq,
            start,
            bucket,
            recip,
            rank1,
            rank3a,
            rank3c,
            rank4,
            rank5,
            ctrl7,
            bc1,
            bc3,
            bc4,
            bc5,
            bc6,
            bc7,
        })
    }
}

/// Unpacks `src` (BC number `bc`, 1, 3, 4 and 5; BC6H and BC7 are not decoded yet) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let blocks = dst.len() / block;
    let (widths, per): (&[u8], usize) = match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        _ => unreachable!("rejected above"),
    };
    let stride = widths.len() * per;
    let mut records = vec![0u8; blocks * stride];
    let mut next = src.iter().copied();
    lanes(tables, widths, per, blocks, |_| next.next(), &mut records)?;
    let records = records.chunks_exact(stride);
    let out = dst.chunks_exact_mut(block);
    let t = tables;
    match bc {
        1 => plain(records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
        3 => plain(records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
        4 => plain(records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
        5 => plain(records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
        _ => unreachable!("rejected above"),
    }
}

// Lanes.

#[derive(Clone, Copy, Default)]
struct Lane {
    /// Symbol bits - 1 at the full alphabet.
    width: u8,
    /// 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    phase: u8,
    long: bool,
    reset: bool,
    model: u8,
    need: bool,
    latch: Option<u8>,
    out: Option<u8>,
    stalled: bool,
    x: u32,
    left: u32,
}

impl Lane {
    /// One step: take the latched byte (header, length or payload) and
    /// decode at most one symbol.
    fn step(&mut self, t: &Tables) -> Result<(), Error> {
        if self.phase != 3 {
            let Some(b) = self.latch.take() else {
                return Ok(());
            };
            self.need = true;
            match self.phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = self.width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.reset = b & 0x10 != 0;
                    self.long = b & 0x20 != 0;
                    self.left = 0;
                    self.phase = 1;
                }
                1 if self.long => {
                    self.left = b.into();
                    self.phase = 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.phase = 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.phase = 3;
                }
            }
            return Ok(());
        }
        let mut x = self.x;
        if x < LOW
            && let Some(b) = self.latch.take()
        {
            self.left = self.left.wrapping_sub(1);
            x = x << 8 | u32::from(b);
        }
        self.need = self.left != 0 && self.latch.is_none();
        if x > LOW {
            let freq = &t.freq[usize::from(self.model)];
            let start = &t.start[usize::from(self.model)];
            let slot = x & 0xffff;
            // The bucket: starts are ascending, and 65536 past the model's last.
            let b = bucket_of(t, self.model, slot);
            // Bucket 0 holds symbol 0, which owns slots 1 .. f[0] - 1 (slot 0
            // is never used). Bucket b > 0 holds 2^(b-1) symbols of frequency
            // f[b] from start[b] on; offset and frequency are below 2^16, where
            // the reciprocal divides exactly.
            let (sym, f, rem) = if b == 0 {
                (0, freq[0] - 1, slot.wrapping_sub(1))
            } else {
                let offset = slot - start[b];
                let q = ((u64::from(offset) * t.recip[usize::from(self.model)][b]) >> 32) as u32;
                ((1u32 << (b - 1)) + q, freq[b], offset - q * freq[b])
            };
            self.x = (x >> 16).wrapping_mul(f).wrapping_add(rem);
            self.out = Some(sym as u8);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.phase = 0;
            self.need = self.latch.is_none();
            if self.reset {
                self.x = 0;
            }
        }
        Ok(())
    }
}

/// The bucket holding `slot` (below 65536): the last whose start is at or
/// below it. The window table gives the bucket of the window's first slot;
/// later buckets that start inside the window are stepped over.
#[inline]
fn bucket_of(t: &Tables, model: u8, slot: u32) -> usize {
    let start = &t.start[usize::from(model)];
    let mut b = usize::from(t.bucket[usize::from(model)][(slot >> 8) as usize]);
    while start[b + 1] <= slot {
        b += 1;
    }
    b
}

/// Decodes `per` symbols per block on every lane into `records` (one record
/// of `widths.len() * per` symbols per block), taking input from `fetch`
/// (given the lane that takes the byte) in the order of the schedule: each
/// round the lanes step in order and take at most 8 bytes between them.
// shortcut: one lane step at a time (BC1 roughly 30-60 MB/s, BC7 15-20 MB/s
// on one core); if texture streaming needs more, run the schedule only to
// split the input per lane and decode the lanes apart.
fn lanes(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    mut fetch: impl FnMut(usize) -> Option<u8>,
    records: &mut [u8],
) -> Result<(), Error> {
    let mut lanes = [Lane::default(); 16];
    let lanes = &mut lanes[..widths.len()];
    for (lane, &width) in lanes.iter_mut().zip(widths) {
        lane.width = width;
        lane.need = true;
    }
    let stride = widths.len() * per;
    let full = (1u32 << widths.len()) - 1;
    let (mut pos, mut done, mut idle) = (0, 0u32, 0);
    // Where symbol `pos` of lane 0 goes; lane j's is `per * j` further on.
    let mut at = 0;
    while pos < blocks * per {
        let (mut taken, mut moved) = (0, false);
        for (j, lane) in lanes.iter_mut().enumerate() {
            let bit = 1 << j;
            let before = taken;
            if !lane.stalled {
                lane.step(t)?;
            }
            let waiting = lane.out.is_some() && done & bit != 0;
            if lane.need && !waiting && taken < ROUND_BYTES {
                // Past the end nothing arrives; in a valid stream only finished lanes ask there.
                if let Some(b) = fetch(j) {
                    lane.latch = Some(b);
                    taken += 1;
                }
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled
                && let Some(sym) = lane.out.take()
            {
                records[at + j * per] = sym;
                done |= bit;
                moved = true;
            }
        }
        if done == full {
            pos += 1;
            at += if pos.is_multiple_of(per) {
                stride - per + 1
            } else {
                1
            };
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}

// Blocks.

/// The record's endpoint and index references.
fn references(s: &[u8], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (interleave(s[0], s[1]), interleave(s[2], s[3]))
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Two 4-bit symbols, the first on the even bits.
fn interleave(a: u8, b: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
    spread(a) | spread(b) << 1
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
fn pick<'a, T>(i: usize, r: usize, history: &'a [T], dictionary: &'a [T], zero: &'a T) -> &'a T {
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
fn zz(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Element widths of the plain formats' endpoints and indices.
struct Widths {
    ep: [u8; 8],
    ix: [u8; 32],
    paired: bool,
}

const fn widths(ep: [u8; 8], ix3: usize, ix2: usize, paired: bool) -> Widths {
    let mut ix = [0; 32];
    let mut k = 0;
    while k < ix3 + ix2 {
        ix[k] = if k < ix3 { 3 } else { 2 };
        k += 1;
    }
    Widths { ep, ix, paired }
}

const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16, false);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16, true);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0, false);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0, true);

fn plain<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[Plain],
    w: &Widths,
    residual: impl Fn(&[u8]) -> Plain,
    pack: fn(&Plain, &mut [u8]),
) -> Result<(), Error> {
    let mut history = vec![Plain::default(); HISTORY];
    let zero = Plain::default();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, w.paired);
        let ep_ref = &pick(i, ep_ref, &history, dictionary, &zero).ep;
        let ix_ref = &pick(i, ix_ref, &history, dictionary, &zero).ix;
        let mut v = residual(s);
        for ((v, r), bits) in v.ep.iter_mut().zip(ep_ref).zip(w.ep) {
            *v = v.wrapping_add(*r) & ((1u16 << bits) - 1) as u8;
        }
        for ((v, r), bits) in v.ix.iter_mut().zip(ix_ref).zip(w.ix) {
            *v = v.wrapping_add(*r) & ((1u8 << bits) - 1);
        }
        history[i % HISTORY] = v;
        pack(&v, block);
    }
    Ok(())
}

/// 2-bit index residuals: one symbol per 2x2 quad.
fn quads(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            ix[base + p] = v >> (2 * k) & 3;
        }
    }
}

/// 3-bit index residuals: one symbol per two pixels in raster order.
fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = v >> 4 & 7;
    }
}

fn bc1_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
        v.ep[k] = zz(s[2 + k], bits);
    }
    quads(&t.rank1, &s[8..12], &mut v.ix);
    v
}

fn bc3_residual(t: &Tables, s: &[u8]) -> Plain {
    // Colour lanes carry each channel's two endpoints together.
    let mut v = Plain {
        ep: [
            zz(s[14], 5),
            zz(s[16], 6),
            zz(s[18], 5),
            zz(s[15], 5),
            zz(s[17], 6),
            zz(s[19], 5),
            zz(s[4], 8),
            zz(s[5], 8),
        ],
        ..Default::default()
    };
    pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
    quads(&t.rank3c, &s[20..24], &mut v.ix[16..]);
    v
}

fn bc4_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    v.ep[..2].copy_from_slice(&[zz(s[2], 8), zz(s[3], 8)]);
    pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
    v
}

fn bc5_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for k in 0..4 {
        v.ep[k] = zz(s[4 + k], 8);
    }
    pairs(&t.rank5, &s[8..24], &mut v.ix);
    v
}

/// Indices along the endpoint line -> BC1 and BC4 codes.
const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

fn bc1_color(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let bits = (0..16).fold(0, |acc, p| {
        acc | BC1_CODE[usize::from(ix[p] & 3)] << (2 * p)
    });
    out[..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
}

/// BC4 half: endpoints are kept 128 below their stored value.
fn bc4_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let bits = (0..16).fold(0u64, |acc, p| {
        acc | BC4_CODE[usize::from(ix[p] & 7)] << (16 + 3 * p)
    });
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
    out[..8].copy_from_slice(&v.to_le_bytes());
}

fn bc1_pack(v: &Plain, out: &mut [u8]) {
    bc1_color(&v.ep, &v.ix, out);
}

fn bc3_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
    bc1_color(&v.ep, &v.ix[16..], &mut out[8..]);
}

fn bc4_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix, out);
}

fn bc5_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
    bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shipped_tables_parse() {
        assert!(super::Tables::builtin().is_ok());
    }

    use super::*;

    /// Tables we made: uniform models, identity ranks, a BC7 code per mode,
    /// empty dictionaries.
    fn tables() -> Vec<u8> {
        let mut t = Vec::new();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend((0..=255u8).map(|v| v & 0x77)); // rank1 (unused by the tests beyond 2-bit fields)
        t.extend(0..64u8); // rank3a
        t.extend(0..=255u8); // rank3c
        t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4)); // rank4: low 3 bits, next 3 bits
        t.extend(0..64u8); // rank5
        for code in 0..512u16 {
            // Code c: mode 6 (one subset), the only one the tests use.
            let ctrl: u16 = if code == 0 { 0x46 } else { 0 };
            t.extend_from_slice(&ctrl.to_le_bytes());
        }
        t.resize(TABLES_SIZE, 0);
        t
    }

    /// A lane's symbols as one fragment: uniform model, rANS from 2^24, no
    /// reset, so the lane keeps 2^24 when it ends.
    fn fragment(width: u8, symbols: &[u8]) -> Vec<u8> {
        let f = 1u32 << (15 - width);
        let mut x = LOW;
        let mut bytes = Vec::new();
        for &s in symbols.iter().rev() {
            let (freq, start) = if s == 0 {
                (f - 1, 1)
            } else {
                (f, u32::from(s) * f)
            };
            while x >= (freq << 16) {
                bytes.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << 16) + x % freq + start;
        }
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.reverse();
        let mut out = vec![0];
        if bytes.len() > 256 {
            out[0] |= 0x20;
            let n = bytes.len() - 257;
            out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
        } else {
            out.push((bytes.len() - 1) as u8);
        }
        out.extend(bytes);
        out
    }

    /// Interleaves per-lane fragments the way the decoder takes bytes. A lane
    /// that has finished still asks: a filler fragment (two payload bytes)
    /// brings its state to 2^24 with a byte latched, which parks it for good.
    fn stream(t: &Tables, widths: &[u8], per: usize, records: &[Vec<u8>]) -> Vec<u8> {
        let stride = widths.len() * per;
        let mut queues: Vec<std::collections::VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].to_vec())
                    .collect();
                let mut bytes = fragment(widths[j], &syms);
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();
        let mut out = Vec::new();
        let mut check = vec![0; records.len() * stride];
        let fetch = |j: usize| {
            let b = queues[j]
                .pop_front()
                .expect("a parked lane asked for input");
            out.push(b);
            Some(b)
        };
        lanes(t, widths, per, records.len(), fetch, &mut check).unwrap();
        assert_eq!(check, records.concat());
        out
    }

    const BC4_LANES: [u8; 12] = [7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5];

    fn zigzag(v: i32) -> u8 {
        if v >= 0 {
            (2 * v) as u8
        } else {
            (-2 * v - 1) as u8
        }
    }

    #[test]
    fn bc4_blocks_with_and_without_references() {
        let t = Tables::parse(&tables()).unwrap();
        // Block 0: endpoints 10, -3 (+128 stored), indices p -> p & 7.
        let mut b0 = vec![0, 0, zigzag(10), zigzag(-3)];
        b0.extend((0..8).map(|k| ((2 * k) & 7) | ((2 * k + 1) & 7) << 3));
        // Block 1 refers to block 0 for both: residual endpoints +1, +1; indices +1.
        let mut b1 = vec![0, 1, zigzag(1), zigzag(1)];
        b1.extend((0..8).map(|_| 1 | 1 << 3));
        let src = stream(&t, &BC4_LANES, 1, &[b0, b1]);
        let mut dst = [0u8; 16];
        decode(4, &src, &mut dst, &t).unwrap();
        let block = |e0: u8, e1: u8, ix: &dyn Fn(usize) -> usize| {
            let bits = (0..16).fold(0u64, |a, p| a | BC4_CODE[ix(p)] << (16 + 3 * p));
            (u64::from(e0) | u64::from(e1) << 8 | bits).to_le_bytes()
        };
        assert_eq!(dst[..8], block(138, 125, &|p| p & 7));
        assert_eq!(dst[8..], block(139, 126, &|p| (p + 1) & 7));
    }

    #[test]
    fn lanes_take_bytes_in_schedule_order_and_long_fragments() {
        let t = Tables::parse(&tables()).unwrap();
        // Many blocks: long fragments (over 256 payload bytes) on the wide lanes.
        let records: Vec<Vec<u8>> = (0..400u32)
            .map(|i| {
                let mut r = vec![0, 0, (i * 7) as u8, (i * 13) as u8];
                r.extend((0..8).map(|k| ((i + k) * 5 % 64) as u8));
                r
            })
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut got = vec![0u8; records.len() * 12];
        let mut next = src.iter().copied();
        lanes(&t, &BC4_LANES, 1, records.len(), |_| next.next(), &mut got).unwrap();
        assert_eq!(got, records.concat());
    }

    #[test]
    fn malformed_input_is_an_error() {
        let t = Tables::parse(&tables()).unwrap();
        let records: Vec<Vec<u8>> = (0..64u8)
            .map(|i| vec![0, 0, i, i, 1, 2, 3, 4, 5, 6, 7, 8])
            .collect();
        let src = stream(&t, &BC4_LANES, 1, &records);
        let mut dst = vec![0u8; 64 * 8];
        decode(4, &src, &mut dst, &t).unwrap();
        // Truncated.
        assert!(decode(4, &src[..src.len() / 2], &mut dst, &t).is_err());
        // Headers promising more payload than the stream holds. (Other corrupt
        // headers decode to garbage: the format has no redundancy.)
        let headers = [[0x30; 8], [0xff; 8]].concat();
        assert!(decode(4, &headers, &mut dst[..8], &t).is_err());
        assert!(decode(4, &[], &mut dst, &t).is_err());
        // Destination not a whole number of blocks, empty, more blocks than coded.
        assert!(decode(4, &src, &mut dst[..12], &t).is_err());
        assert!(decode(4, &src, &mut [], &t).is_err());
        assert!(decode(4, &src, &mut vec![0u8; 65 * 8], &t).is_err());
        // Unsupported formats, bad tables.
        assert!(decode(2, &src, &mut dst, &t).is_err());
        assert!(Tables::parse(&[0; 16]).is_err());
        let mut no_tiling = tables();
        no_tiling[0] = 1;
        assert!(Tables::parse(&no_tiling).is_err());
    }

    #[test]
    fn noise_never_panics() {
        let t = Tables::parse(&tables()).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for round in 0..300 {
            let len = round * 7 % 900;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed as u8
                })
                .collect();
            for bc in [1, 3, 4, 5, 6, 7] {
                let mut dst = vec![0u8; 16 * (1 + round % 40)];
                let _ = decode(bc, &src, &mut dst, &t);
            }
        }
    }

    /// The window table finds the bucket a scan of the starts finds, for
    /// every slot of every model, in the shipped tables and in ours (whose
    /// buckets are wide) and in tables whose buckets are one slot wide.
    #[test]
    fn bucket_lookup_matches_a_scan_of_the_starts() {
        let mut narrow = tables();
        for m in 0..MODELS {
            // Buckets 1.. of one slot each where they fit; bucket 0 takes the rest.
            let used = m / 16 + 2;
            let rest = (1u32 << 16) - (1..used).map(|b| 1u32 << (b - 1)).sum::<u32>();
            for b in 0..used {
                let f = if b == 0 { rest } else { 1 };
                narrow[(m * BUCKETS + b) * 2..][..2].copy_from_slice(&(f as u16).to_le_bytes());
            }
        }
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
            Tables::parse(&narrow).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                let start = &t.start[usize::from(model)];
                for slot in 0..1u32 << 16 {
                    let scan = start[1..].iter().filter(|&&s| s <= slot).count();
                    assert_eq!(
                        bucket_of(&t, model, slot),
                        scan,
                        "model {model} slot {slot}"
                    );
                }
            }
        }
    }

    /// One rANS decode as first written: scan for the bucket, divide for the
    /// symbol, rebuild its frequency and start from the symbol.
    fn reference_decode(t: &Tables, model: u8, x: u32) -> (u8, u32) {
        let (freq, start) = (&t.freq[usize::from(model)], &t.start[usize::from(model)]);
        let slot = x & 0xffff;
        let b = start[1..BUCKETS].iter().filter(|&&s| s <= slot).count();
        let sym = match b {
            0 => 0,
            1 => 1,
            _ => (1u32 << (b - 1)) + (slot - start[b]) / freq[b],
        } as u8;
        let (f, c) = if sym == 0 {
            (freq[0] - 1, 1)
        } else {
            let k = (8 - sym.leading_zeros()) as usize;
            let f = freq[k];
            (
                f,
                ((u32::from(sym) - (1 << (k - 1))) * f + start[k]) & 0xffff,
            )
        };
        (
            sym,
            (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot),
        )
    }

    /// A payload step decodes the symbol and state the first version did, for
    /// every model and slot of the shipped tables and ours, at assorted states.
    #[test]
    fn payload_step_matches_the_reference_decode() {
        for t in [
            Tables::builtin().unwrap(),
            Tables::parse(&tables()).unwrap(),
        ] {
            for model in 0..MODELS as u8 {
                for high in [0x100u32, 0x1234, 0xffff] {
                    for slot in 0..1u32 << 16 {
                        let x = high << 16 | slot;
                        if x <= LOW {
                            continue;
                        }
                        let mut lane = Lane {
                            phase: 3,
                            model,
                            x,
                            left: 1,
                            ..Default::default()
                        };
                        lane.step(&t).unwrap();
                        let (sym, next) = reference_decode(&t, model, x);
                        assert_eq!(
                            (lane.out, lane.x),
                            (Some(sym), next),
                            "model {model} x {x:#x}"
                        );
                    }
                }
            }
        }
    }

    /// Pseudo-random records that fit each lane's alphabet; BC7 keeps control
    /// code 0 (mode 6), the only one the synthetic tables define.
    fn random_records(widths: &[u8], per: usize, blocks: usize, bc7: bool) -> Vec<Vec<u8>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..blocks)
            .map(|_| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let alphabet = 1u32 << (widths[k / per] + 1);
                        (seed as u32 % alphabet) as u8
                    })
                    .collect();
                if bc7 {
                    r[30] = 0;
                    r[31] = 0;
                }
                r
            })
            .collect()
    }

    /// The fast paths decode exactly what one lane step at a time does, on
    /// every format's lane layout, with short and long fragments.
    #[test]
    fn every_layout_round_trips_through_the_lanes() {
        let t = Tables::parse(&tables()).unwrap();
        let layouts: [(&[u8], usize); 6] = [
            (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
            (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
            (&BC4_LANES, 1),
            (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
            (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
            (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
        ];
        for (widths, per) in layouts {
            for blocks in [1, 2, 17, 300, 1000] {
                let records = random_records(widths, per, blocks, false);
                let src = stream(&t, widths, per, &records);
                let mut got = vec![0u8; records.concat().len()];
                let mut next = src.iter().copied();
                lanes(&t, widths, per, blocks, |_| next.next(), &mut got).unwrap();
                assert_eq!(got, records.concat(), "{widths:?} x{per}, {blocks} blocks");
            }
        }
    }

    /// Decode throughput on synthetic streams (no game data):
    /// `cargo test --release -p victory-formats bcpack_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn bcpack_throughput() {
        let t = Tables::parse(&tables()).unwrap();
        let cases: [(u8, &[u8], usize, usize); 3] = [
            (1, &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1, 8),
            (4, &BC4_LANES, 1, 8),
            (7, &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2, 16),
        ];
        for (bc, widths, per, block) in cases {
            // One fragment per lane holds at most 65792 bytes.
            let blocks = 1 << 14;
            let records = random_records(widths, per, blocks, bc == 7);
            let src = stream(&t, widths, per, &records);
            let mut dst = vec![0u8; blocks * block];
            decode(bc, &src, &mut dst, &t).unwrap();
            let runs = 40;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                decode(bc, std::hint::black_box(&src), &mut dst, &t).unwrap();
            }
            let secs = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "BC{bc}: {blocks} blocks, {:.1} MB out/s, {:.1} ns/block",
                (blocks * block) as f64 / secs / 1e6,
                secs * 1e9 / blocks as f64
            );
        }
    }

    /// Compares with reference outputs made outside the repository:
    /// `VICTORY_BCPACK_TABLES` (the table file) and `VICTORY_BCPACK_CASES`, a
    /// directory of `<name>.bc<n>` packed chunks with `<name>.out` beside them.
    #[test]
    #[ignore = "needs the format tables and reference chunks"]
    fn reference_chunks() {
        let tables = std::env::var_os("VICTORY_BCPACK_TABLES").expect("VICTORY_BCPACK_TABLES");
        let cases = std::env::var_os("VICTORY_BCPACK_CASES").expect("VICTORY_BCPACK_CASES");
        let t = Tables::parse(&std::fs::read(tables).unwrap()).unwrap();
        let mut n = 0;
        for entry in std::fs::read_dir(cases).unwrap() {
            let path = entry.unwrap().path();
            let Some(bc) = path
                .extension()
                .and_then(|e| e.to_str()?.strip_prefix("bc")?.parse().ok())
            else {
                continue;
            };
            let want = std::fs::read(path.with_extension("out")).unwrap();
            let mut got = vec![0u8; want.len()];
            decode(bc, &std::fs::read(&path).unwrap(), &mut got, &t).unwrap();
            assert!(got == want, "{}", path.display());
            n += 1;
        }
        assert!(n > 0);
    }
}
