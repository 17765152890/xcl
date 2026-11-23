//! BC1 blocks from BCPack's 12 lane symbols (one symbol per lane per 8-byte block).
//!
//! From the SixNG notes, "BC1 blocks (codec 0x04)":
//! - lane 0: distance d = unzigzag(v) mod 256; d = 0 predicts zero for the six endpoint fields,
//!   d <= i block i - d of this chunk, d > i preset slot (i - d) mod 256; lane 1 is unused;
//! - lanes 2-7: zigzag residuals (0, -1, +1, -2 …) of R0 G0 B0 R1 G1 B1, added modulo the
//!   field widths 5 6 5 5 6 5; three-colour blocks (c0 <= c1) are written as decoded;
//! - lanes 8-11: the selector quadrants top-left, top-right, bottom-left, bottom-right, each a
//!   codebook entry of four 2-bit codes (TL, TR, BL, BR of the 2x2 group); selectors are not
//!   predicted.

use super::tables::{BC1_CODEBOOK, BC1_PRESETS};

const WIDTHS: [u32; 6] = [5, 6, 5, 5, 6, 5];

/// The 12 lane symbol widths of BC1.
pub const LANE_BITS: [u8; 12] = [8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8];

/// 0, -1, +1, -2 … from 0, 1, 2, 3 …
fn unzigzag(v: u16) -> i32 {
    if v & 1 == 0 {
        (v >> 1) as i32
    } else {
        -(((v as i32) + 1) >> 1)
    }
}

/// The blocks of a chunk; the count of blocks that read a preset slot nobody fitted.
pub fn blocks(lanes: &[Vec<u16>], n: usize) -> (Vec<u8>, usize) {
    let mut fields: Vec<[u8; 6]> = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n * 8);
    let mut unfitted = 0;
    for i in 0..n {
        let v = lanes[0][i];
        let d = (if v & 1 == 0 { v >> 1 } else { 255 - (v >> 1) }) as usize & 0xFF;
        let reference = if d == 0 {
            [0; 6]
        } else if d <= i {
            fields[i - d]
        } else {
            let slot = (i as isize - d as isize).rem_euclid(256) as usize;
            BC1_PRESETS[slot].unwrap_or_else(|| {
                unfitted += 1;
                [0; 6]
            })
        };
        let mut f = [0u8; 6];
        for k in 0..6 {
            let m = 1i32 << WIDTHS[k];
            f[k] = (reference[k] as i32 + unzigzag(lanes[2 + k][i])).rem_euclid(m) as u8;
        }
        fields.push(f);
        let c0 = (f[0] as u16) << 11 | (f[1] as u16) << 5 | f[2] as u16;
        let c1 = (f[3] as u16) << 11 | (f[4] as u16) << 5 | f[5] as u16;
        let mut idx = 0u32;
        for quad in 0..4 {
            let entry = BC1_CODEBOOK[lanes[8 + quad][i] as usize & 0xFF];
            let (qr, qc) = (quad / 2 * 2, quad % 2 * 2);
            for p in 0..4 {
                let (row, col) = (qr + p / 2, qc + p % 2);
                idx |= (entry[p] as u32 & 3) << (2 * (4 * row + col));
            }
        }
        out.extend_from_slice(&c0.to_le_bytes());
        out.extend_from_slice(&c1.to_le_bytes());
        out.extend_from_slice(&idx.to_le_bytes());
    }
    (out, unfitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unzigzag_order() {
        assert_eq!((0..6).map(unzigzag).collect::<Vec<_>>(), vec![0, -1, 1, -2, 2, -3]);
    }

    /// Hand-made lanes: block 0 from zero, block 1 one back (copies block 0's endpoints) with a
    /// residual, uniform selector quadrants.
    #[test]
    fn blocks_follow_prediction_residuals_and_quadrants() {
        let mut lanes = vec![Vec::new(); 12];
        // block 0: d = 0; R0 +3 (v 6), G0 -1 (v 1), others 0; quadrants: symbol 0 = code 2 everywhere
        let b0 = [0u16, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        // block 1: d = 1 (v 2); B1 +2 (v 4); quadrant TL symbol 1 (code 3), others symbol 2 (code 0)
        let b1 = [2u16, 0, 0, 0, 0, 0, 0, 4, 1, 2, 2, 2];
        for (l, lane) in lanes.iter_mut().enumerate() {
            lane.push(b0[l]);
            lane.push(b1[l]);
        }
        let (out, unfitted) = blocks(&lanes, 2);
        assert_eq!(unfitted, 0);
        // block 0: R0 3, G0 63 (0 - 1 mod 64), B0 0, c1 = 0
        assert_eq!(u16::from_le_bytes([out[0], out[1]]), 3 << 11 | 63 << 5);
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), 0);
        assert_eq!(u32::from_le_bytes([out[4], out[5], out[6], out[7]]), 0xAAAA_AAAA);
        // block 1: c0 as block 0, B1 = 2
        assert_eq!(u16::from_le_bytes([out[8], out[9]]), 3 << 11 | 63 << 5);
        assert_eq!(u16::from_le_bytes([out[10], out[11]]), 2);
        // TL quadrant (rows 0-1, cols 0-1) code 3, the rest 0
        let idx = u32::from_le_bytes([out[12], out[13], out[14], out[15]]);
        let code = |r: u32, c: u32| (idx >> (2 * (4 * r + c))) & 3;
        assert_eq!([code(0, 0), code(0, 1), code(1, 0), code(1, 1)], [3, 3, 3, 3]);
        assert_eq!([code(0, 2), code(2, 0), code(3, 3)], [0, 0, 0]);
    }
}
