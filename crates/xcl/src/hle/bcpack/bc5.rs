//! BC5 blocks from the 12 lanes of a BCPack BC5 stream (two symbols per lane and block: units
//! `2i` and `2i + 1`, called `u0` and `u1` here).
//!
//! For block `i`, in stream order:
//! * lane 0 holds a back-reference in two 4-bit halves: bit `k` of `u0` is bit `2k` of the
//!   symbol, bit `k` of `u1` bit `2k + 1`; the symbol then reads as BC4's lane 0
//!   ([`super::bc4`]). Symbol 0 predicts 128 for all four endpoints, and before block 0 the
//!   prediction comes from a built-in history ([`Tables::history`]),
//! * lane 1 was 0 in every stream seen, in both units,
//! * lane 2 is the difference of R0 (`u0`) and R1 (`u1`) from the prediction, lane 3 the same for
//!   G0 and G1, as BC4's lanes 2 and 3,
//! * lanes 4–7 are the red selectors, one row each from the top, the left half of the row in
//!   `u0` and the right half in `u1`, through a pair table as BC4's
//!   ([`Tables::pair_codes`]); lanes 8–11 the green ones.
//!
//! As for BC4, XCL has no source it can take for the two tables, so nothing calls this yet.

use super::bc4::{back_reference, delta, selector_rows};
use super::{decode_lanes, Error};

/// The constants of the BC5 rebuild that are not in a stream.
pub struct Tables {
    /// The endpoints `[R0, R1, G0, G1]` of the blocks before a stream's first:
    /// `history[256 - k]` is the block `k` before block 0 (k = 1..=255; entry 0 is never read).
    /// `None`: not known.
    pub history: [Option<[u8; 4]>; 256],
    /// The selectors of each pair symbol: bits 0–2 the left texel, 3–5 the right one.
    pub pair_codes: [u8; 64],
}

/// Unpack a BC5 stream to its `out_len` bytes of blocks.
pub fn decode(src: &[u8], out_len: usize, tables: &Tables) -> Result<Vec<u8>, Error> {
    if out_len % 16 != 0 {
        return Err(Error::OutputSize);
    }
    assemble(&decode_lanes(5, src, out_len)?, tables)
}

/// The 8-bit symbol whose even bits are `even` and odd bits `odd` (BC3 and BC5 lane 0).
pub(super) fn interleave(even: u8, odd: u8) -> u8 {
    (0..4).map(|k| (even >> k & 1) << (2 * k) | (odd >> k & 1) << (2 * k + 1)).fold(0, |x, y| x | y)
}

/// The blocks of `symbols` (24 per block, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, Error> {
    let blocks = symbols.len() / 24;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 16).map_err(|_| Error::OutputSize)?;
    out.resize(blocks * 16, 0u8);
    for (i, s) in symbols.chunks_exact(24).enumerate() {
        let (u0, u1) = s.split_at(12);
        if u0[1] != 0 || u1[1] != 0 {
            return Err(Error::Lane1 { block: i });
        }
        let predicted = match back_reference(interleave(u0[0], u1[0])) {
            0 => [128; 4],
            d if d <= i => {
                let p = &out[(i - d) * 16..];
                [p[0], p[1], p[8], p[9]]
            }
            d => {
                let slot = 256 + i - d;
                tables.history[slot].ok_or(Error::History { block: i, slot })?
            }
        };
        let o = &mut out[i * 16..i * 16 + 16];
        for (c, lanes) in [4..8, 8..12].into_iter().enumerate() {
            o[8 * c] = predicted[2 * c].wrapping_add(delta(u0[2 + c]));
            o[8 * c + 1] = predicted[2 * c + 1].wrapping_add(delta(u1[2 + c]));
            let halves = lanes.flat_map(|l| [u0[l], u1[l]]);
            o[8 * c + 2..8 * c + 8].copy_from_slice(&selector_rows(halves, &tables.pair_codes).to_le_bytes()[..6]);
        }
    }
    Ok(out)
}
