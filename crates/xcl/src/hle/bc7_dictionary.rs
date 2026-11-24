//! Reads the BC7 dictionary section, supplied separately from the other BCPack tables.
//! Layout source: the format description in github.com/Marvisak/VIctory/pull/658.
//! Bytes 52–53 are retained verbatim; the first entry can differ from its index widths.

const ENTRY_BYTES: usize = 116;

#[derive(Debug, PartialEq)]
pub struct Entry {
    pub control: u16,
    pub parameters: [u16; 24],
    pub index_widths: [u8; 2],
    pub extra_bytes: [u8; 2],
    pub first_indices: [i16; 16],
    pub second_indices: [i16; 15],
}

impl Entry {
    /// Check the two documented index invariants for the BC7 dictionary.
    /// Only set 0 stores pixel 0; set 1 stores the remaining 15 slots.
    pub fn indices_fit_layout(&self) -> bool {
        let [first, second] = self.index_widths;
        if second == 0 && self.second_indices.iter().any(|&index| index != 0) {
            return false;
        }
        if first == 0 {
            return self.first_indices.iter().all(|&index| index == 0);
        }
        let anchor = self.first_indices[0];
        // A nonnegative anchor needs strictly fewer bits than the whole index.
        anchor >= 0 && (u16::BITS - (anchor as u16).leading_zeros()) < u32::from(first)
    }
}

/// Returns None if any entry, including the last one, is incomplete.
pub fn entries(section: &[u8]) -> Option<impl Iterator<Item = Entry> + '_> {
    let records = section.chunks_exact(ENTRY_BYTES);
    if !records.remainder().is_empty() {
        return None;
    }
    Some(records.map(|record| {
        let word = |at| u16::from_le_bytes([record[at], record[at + 1]]);
        Entry {
            control: word(0),
            parameters: std::array::from_fn(|n| word(2 + 2 * n)),
            index_widths: [record[50], record[51]],
            extra_bytes: [record[52], record[53]],
            first_indices: std::array::from_fn(|n| word(54 + 2 * n) as i16),
            second_indices: std::array::from_fn(|n| word(86 + 2 * n) as i16),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_two_records_with_distinct_boundaries() {
        // Build sequentially, independently of the reader's field offsets.
        let mut input = Vec::new();
        for control in [0x1240u16, 0x3441] {
            input.extend(control.to_le_bytes());
            for n in 0..24u16 {
                input.extend((0x1200 + n).to_le_bytes());
            }
            input.extend([4, 0, 2, 0]);
            for n in 0..16i16 {
                input.extend((3 - n).to_le_bytes());
            }
            input.extend([0u8; 30]);
        }
        let mut decoded = entries(&input).unwrap();
        for control in [0x1240, 0x3441] {
            let entry = decoded.next().unwrap();
            assert_eq!(entry.control, control);
            assert_eq!(entry.parameters, std::array::from_fn(|n| 0x1200 + n as u16));
            assert_eq!(entry.index_widths, [4, 0]);
            assert_eq!(entry.extra_bytes, [2, 0]);
            assert_eq!(entry.first_indices, std::array::from_fn(|n| 3 - n as i16));
            assert_eq!(entry.second_indices, [0; 15]);
            assert!(entry.indices_fit_layout());
        }
        assert!(decoded.next().is_none());
    }

    #[test]
    fn zero_width_sets_and_the_anchor_are_checked_independently() {
        let input = [0; ENTRY_BYTES];
        let mut entry = entries(&input).unwrap().next().unwrap();
        assert!(entry.indices_fit_layout());
        entry.second_indices[14] = -1;
        assert!(!entry.indices_fit_layout());
        entry.index_widths[1] = 2;
        assert!(entry.indices_fit_layout());
        entry.first_indices[15] = -1;
        assert!(!entry.indices_fit_layout());
        entry.index_widths[0] = 4;
        for (anchor, expected) in [(-1, false), (0, true), (7, true), (8, false)] {
            entry.first_indices[0] = anchor;
            assert_eq!(entry.indices_fit_layout(), expected);
        }
    }

    #[test]
    fn reads_signed_values_to_the_end_of_the_second_set() {
        let mut input = [0u8; ENTRY_BYTES];
        input[86..88].copy_from_slice(&i16::MIN.to_le_bytes());
        input[114..116].copy_from_slice(&i16::MAX.to_le_bytes());
        let entry = entries(&input).unwrap().next().unwrap();
        assert_eq!(entry.second_indices[0], i16::MIN);
        assert_eq!(entry.second_indices[14], i16::MAX);
    }

    #[test]
    fn refuses_every_partial_final_record() {
        let input = [0u8; 2 * ENTRY_BYTES];
        for size in 0..=input.len() {
            match size % ENTRY_BYTES {
                0 => assert_eq!(entries(&input[..size]).unwrap().count(), size / ENTRY_BYTES),
                _ => assert!(entries(&input[..size]).is_none(), "size {size}"),
            }
        }
    }
}
