//! BCPack: the Xbox Series texture codec DirectStorage unpacks in hardware. It codes BC blocks,
//! not pixels: an entropy stage (`rans`, N lanes of rANS symbols from one byte stream) and a
//! per-format block stage that rebuilds ordinary BC blocks in the texture's stored order.
//!
//! Written from the SixNG format notes (https://github.com/TrinitronClient/GTA6-Asset-Info,
//! "6. BCPack"); the constant tables in `tables.rs` are generated from their JSON
//! (`scripts/gen_bcunpack_tables.py`), whose values come from MIT-licensed work (THIRD_PARTY.md).
//! Only BC1 is rebuilt so far; the other formats need fitted tables the notes do not carry.

pub mod bc1;
pub mod rans;
mod tables;

/// `DSTORAGE_BCPACK_MODE` values this unpacker rebuilds.
pub fn supported(mode: u8) -> bool {
    mode == 1
}

/// The `dst_len` bytes of BC blocks that `src` (one BCPack chunk) codes in `mode`.
pub fn unpack(mode: u8, src: &[u8], dst_len: usize) -> Result<Vec<u8>, String> {
    match mode {
        1 => {
            if dst_len % 8 != 0 {
                return Err(format!("BC1: {dst_len} bytes is not whole blocks"));
            }
            let n = dst_len / 8;
            let (lanes, _read) = rans::decode_lanes(src, &bc1::LANE_BITS, n)?;
            let (out, unfitted) = bc1::blocks(&lanes, n);
            if unfitted > 0 {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::xlog_warn!("BCPack BC1: {unfitted} block(s) read a preset slot no table has (zeros used)");
                }
            }
            Ok(out)
        }
        _ => Err(format!("mode {mode} has no unpacker")),
    }
}

#[cfg(test)]
mod tests {
    /// `XCL_SAMPLE_BCPACK=<folder>` of chunks `XCL_BCPACK_DUMP` wrote (`<mode>_<bytes>_<n>.bin`,
    /// from the user's own game): every chunk of a supported mode decodes, every lane ending in a
    /// legal state. Passes without doing anything when the variable is not set.
    #[test]
    fn sample_chunks_decode() {
        let Some(dir) = std::env::var_os("XCL_SAMPLE_BCPACK") else { return };
        let mut entries: Vec<_> = std::fs::read_dir(&dir).expect("XCL_SAMPLE_BCPACK").filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        let (mut ok, mut failed) = (0, Vec::new());
        for path in entries {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let parts: Vec<&str> = name.split('_').collect();
            let (Some(mode), Some(bytes)) = (parts.first().and_then(|m| m.parse::<u8>().ok()), parts.get(1).and_then(|b| b.parse::<usize>().ok())) else { continue };
            if !super::supported(mode) {
                continue;
            }
            let chunk = std::fs::read(&path).unwrap();
            match super::unpack(mode, &chunk, bytes) {
                Ok(out) => {
                    assert_eq!(out.len(), bytes);
                    ok += 1;
                }
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        eprintln!("BCPack samples: {ok} decoded, {} failed", failed.len());
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
