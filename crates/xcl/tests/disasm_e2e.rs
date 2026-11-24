use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, bytes: &[u8]) -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("xcl-disasm-{name}-{}-{nonce}-{seq}.bin", std::process::id()));
        if !bytes.is_empty() {
            std::fs::write(&path, bytes).unwrap();
        }
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run_disasm(path: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("disasm")
        .arg(path)
        .args(args)
        .output()
        .unwrap()
}

fn dxbc_container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let header = 32 + parts.len() * 4;
    let mut v = vec![0u8; header];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*cc);
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

fn le32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn synthetic_fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut v = vec![0u8; code_at];
    v[0..4].copy_from_slice(b"FSSM");
    v[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    for q in (0x18..0xC0).step_by(8) {
        v[q..q + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    v[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    v[0xC8..0xD0].copy_from_slice(&(code.len() as u64 + 8).to_le_bytes());
    for (i, &(hw, api)) in stages.iter().enumerate() {
        let o = 0x100 + i * 0x88;
        let q = 0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64;
        v[o..o + 8].copy_from_slice(&q.to_le_bytes());
    }
    v.extend_from_slice(code);
    v.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf, 0x00, 0x00, 0x9f, 0xbf]);
    v.extend_from_slice(b"\x03\0\0\0metadata\0\0\0\0");
    let size = (v.len() - 0x10) as u32;
    v[8..12].copy_from_slice(&size.to_le_bytes());
    v
}

fn synthetic_xbch(fssm: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm, 6);
    let mut v = b"XBCH".to_vec();
    le32(&mut v, 3);
    le32(&mut v, z.len() as u32);
    le32(&mut v, 0);
    le32(&mut v, fssm.len() as u32);
    le32(&mut v, 0);
    v.extend_from_slice(&z);
    v
}

fn dxil_program_header(kind: u32, size_dw: u32, bitcode_off: u32) -> Vec<u8> {
    let mut v = Vec::new();
    le32(&mut v, kind << 16 | 0x66);
    le32(&mut v, size_dw);
    v.extend_from_slice(b"DXIL");
    le32(&mut v, 0x106);
    le32(&mut v, bitcode_off);
    le32(&mut v, 4);
    v
}

fn synthetic_xbox_container(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs: Vec<(u32, Vec<u8>)> = vec![(0x12, (0u8..16).collect())];
    for f in fssms {
        blobs.push((0x0d, synthetic_xbch(f)));
    }
    let mut nested = dxil_program_header(kind, 0x8000_0007, 0x10);
    nested.extend_from_slice(b"NOBC");
    blobs.push((0x07, nested));

    let table = 0x18 + 12;
    let mut data_at = table + blobs.len() * 12;
    let mut xdat = b"XDAT".to_vec();
    le32(&mut xdat, blobs.len() as u32);
    le32(&mut xdat, table as u32);
    for (id, b) in &blobs {
        le32(&mut xdat, *id);
        le32(&mut xdat, data_at as u32);
        le32(&mut xdat, b.len() as u32);
        data_at += b.len();
    }
    for (_, b) in &blobs {
        xdat.extend_from_slice(b);
    }
    while (0x18 + xdat.len()) % 4 != 0 {
        xdat.push(0);
    }
    let nobc_at = 0x18 + xdat.len();
    let mut dxil = dxil_program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    dxil.extend_from_slice(&xdat);
    dxil.extend_from_slice(b"NOBC");

    dxbc_container(&[(b"SFI0", &[0; 8]), (b"HASH", &[0; 20]), (b"DXIL", &dxil), (b"PRIV", &[7; 16])])
}

#[test]
fn disasm_honors_byte_range_options() {
    let shader = TempFile::new("range", &[0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf]);
    let length = run_disasm(shader.path(), &["--length", "4"]);
    let offset = run_disasm(shader.path(), &["--offset", "0x4", "--length", "0x4"]);
    let malformed = run_disasm(shader.path(), &["--offset", "bogus"]);
    let missing = run_disasm(shader.path(), &["--length"]);

    assert!(length.status.success());
    let listing = String::from_utf8(length.stdout).unwrap();
    assert!(listing.contains("00000000: s_nop"), "{listing}");
    assert!(!listing.contains("s_endpgm"), "{listing}");

    assert!(offset.status.success());
    let listing = String::from_utf8(offset.stdout).unwrap();
    assert!(listing.contains("00000004: s_endpgm"), "{listing}");
    assert!(!listing.contains("s_nop"), "{listing}");

    for (output, flag) in [(malformed, "--offset"), (missing, "--length")] {
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8(output.stderr).unwrap().contains(flag));
    }
}

#[test]
fn disasm_all_flag_and_full_file_rdna2_scan() {
    // Two programs back-to-back: [s_nop 0; s_endpgm] then [s_mov_b32 s0, s1; s_endpgm]
    let blob = [
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x81, 0xbf,
        0x01, 0x03, 0x80, 0xbe,
        0x00, 0x00, 0x81, 0xbf,
    ];
    let shader = TempFile::new("multi", &blob);

    let stop = run_disasm(shader.path(), &["--offset", "0"]);
    assert!(stop.status.success());
    let stop_out = String::from_utf8(stop.stdout).unwrap();
    assert!(stop_out.contains("00000000: s_nop"), "{stop_out}");
    assert!(stop_out.contains("00000004: s_endpgm"), "{stop_out}");
    assert!(!stop_out.contains("s_mov_b32"), "{stop_out}");

    let all = run_disasm(shader.path(), &["--offset", "0", "--all"]);
    assert!(all.status.success());
    let all_out = String::from_utf8(all.stdout).unwrap();
    assert!(all_out.contains("00000008: s_mov_b32 s0, s1"), "{all_out}");
    assert!(all_out.contains("0000000c: s_endpgm"), "{all_out}");

    let scan = run_disasm(shader.path(), &[]);
    assert!(scan.status.success());
    let scan_out = String::from_utf8(scan.stdout).unwrap();
    assert!(scan_out.contains("format: rdna2-raw"), "{scan_out}");
    assert!(scan_out.contains("; blob: program at +0x0 (8 bytes)"), "{scan_out}");
    assert!(scan_out.contains("; blob: program at +0x8 (8 bytes)"), "{scan_out}");
    assert!(scan_out.contains("s_mov_b32 s0, s1"), "{scan_out}");

    let unknown = TempFile::new("unknown", &[0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x80, 0xbf]);
    let unk = run_disasm(unknown.path(), &[]);
    assert!(unk.status.success());
    let unk_out = String::from_utf8(unk.stdout).unwrap();
    assert!(unk_out.contains("format: unknown"), "{unk_out}");
    assert!(unk_out.contains("no RDNA2 programs found"), "{unk_out}");
}

#[test]
fn disasm_classifies_pc_dxbc_and_dxil_containers() {
    let dxbc_blob = dxbc_container(&[(b"SHEX", &[0x10, 0x00, 0x00, 0x00])]);
    let dxbc_file = TempFile::new("pc-dxbc", &dxbc_blob);
    let dxbc_out = run_disasm(dxbc_file.path(), &[]);
    assert!(dxbc_out.status.success());
    let dxbc_txt = String::from_utf8(dxbc_out.stdout).unwrap();
    assert!(dxbc_txt.contains("format: dxbc"), "{dxbc_txt}");
    assert!(dxbc_txt.contains("SHEX"), "{dxbc_txt}");
    assert!(dxbc_txt.contains("PC-format shader"), "{dxbc_txt}");

    let mut pc_dxil = dxil_program_header(0, 8, 0x10);
    pc_dxil.extend_from_slice(b"BC\xc0\xde");
    let dxil_blob = dxbc_container(&[(b"DXIL", &pc_dxil)]);
    let dxil_file = TempFile::new("pc-dxil", &dxil_blob);
    let dxil_out = run_disasm(dxil_file.path(), &[]);
    assert!(dxil_out.status.success());
    let dxil_txt = String::from_utf8(dxil_out.stdout).unwrap();
    assert!(dxil_txt.contains("format: dxil"), "{dxil_txt}");
    assert!(dxil_txt.contains("DXIL"), "{dxil_txt}");
    assert!(dxil_txt.contains("PC-format shader"), "{dxil_txt}");
}

#[test]
fn disasm_unpacks_xbox_native_fssm_and_writes_fssm_file() {
    // s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
    let code = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];
    let fssm = synthetic_fssm(&code, &[(5, 1)]);
    let container = synthetic_xbox_container(0, std::slice::from_ref(&fssm));

    let blob_file = TempFile::new("xbox-native", &container);
    let fssm_out_file = TempFile::new("fssm-out", &[]);
    let fssm_out_str = fssm_out_file.path().to_str().unwrap();

    let out = run_disasm(blob_file.path(), &["--fssm", fssm_out_str, "--hex"]);
    assert!(out.status.success());
    let txt = String::from_utf8(out.stdout).unwrap();
    assert!(txt.contains("format: xbox-native"), "{txt}");
    assert!(txt.contains("console container: stage PS (DXIL), SM 6.6, XDAT @ +"), "{txt}");
    assert!(txt.contains("XDAT 0x12"), "{txt}");
    assert!(txt.contains("XDAT 0x0d"), "{txt}");
    assert!(txt.contains("XDAT 0x07"), "{txt}");
    assert!(txt.contains("FSSM #0: XBCH v3"), "{txt}");
    assert!(txt.contains("hardware stage: PS"), "{txt}");
    assert!(txt.contains("code: +0x308"), "{txt}");
    assert!(txt.contains("data after the code"), "{txt}");
    assert!(txt.contains("metadata"), "{txt}");
    assert!(txt.contains("saved FSSM to"), "{txt}");
    assert!(txt.contains("s_mov_b32 s0, s1"), "{txt}");
    assert!(txt.contains("v_add_f32 v0, v1, v5"), "{txt}");
    assert!(txt.contains("s_endpgm"), "{txt}");

    let written = std::fs::read(fssm_out_file.path()).unwrap();
    assert_eq!(written, fssm);
}

#[test]
fn disasm_rejects_out_of_range_offset_and_missing_file() {
    let shader = TempFile::new("small", &[0, 0, 0x81, 0xbf]);
    let oob = run_disasm(shader.path(), &["--offset", "0x100"]);
    assert_eq!(oob.status.code(), Some(2));
    let oob_err = String::from_utf8(oob.stderr).unwrap();
    assert!(oob_err.contains("past the end of the file"), "{oob_err}");

    let missing_path = std::env::temp_dir().join(format!("xcl-disasm-nonexistent-{}.bin", std::process::id()));
    let missing = run_disasm(&missing_path, &[]);
    assert_eq!(missing.status.code(), Some(2));
    let missing_err = String::from_utf8(missing.stderr).unwrap();
    assert!(missing_err.contains("xcl disasm:"), "{missing_err}");
}
