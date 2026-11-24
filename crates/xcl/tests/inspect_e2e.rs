use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use xcl::pe::{Layout, Pe, DIR_DELAY_IMPORT, DIR_IMPORT};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn image(bad_directory: Option<usize>) -> Vec<u8> {
    let mut data = vec![0u8; 0x400];
    for (offset, value) in [(0, 0x5A4Du16), (0x84, 0x8664), (0x86, 1), (0x94, 0xF0), (0x96, 0x22), (0x98, 0x20B), (0xDC, 3)] {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (0x3C, 0x80u32), (0x80, 0x4550), (0xB8, 0x1000), (0xBC, 0x200),
        (0xD0, 0x2000), (0xD4, 0x200), (0x104, 16),
        (0x190, 0x200), (0x194, 0x1000), (0x198, 0x200), (0x19C, 0x200), (0x1AC, 0x40000040),
    ] {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    data[0xB0..0xB8].copy_from_slice(&0x140000000u64.to_le_bytes());
    data[0x188..0x18E].copy_from_slice(b".rdata");
    if let Some(directory) = bad_directory {
        let offset = 0x108 + directory * 8;
        data[offset..offset + 4].copy_from_slice(&0x3000u32.to_le_bytes());
        data[offset + 4..offset + 8].copy_from_slice(&32u32.to_le_bytes());
    }
    data
}

#[test]
fn malformed_import_tables_fail_inspection() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = TempDir(std::env::temp_dir().join(format!("xcl-inspect-imports-{}-{nonce}", std::process::id())));
    std::fs::create_dir(&dir.0).unwrap();
    let path = dir.0.join("fixture.exe");

    // Missing import directories are valid and must not be confused with unreadable tables.
    std::fs::write(&path, image(None)).unwrap();
    assert_eq!(xcl::inspect::console_imports(&path), Some(vec![]));
    let output = Command::new(env!("CARGO_BIN_EXE_xcl")).arg("inspect").arg(&path).arg("--json").output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["imports"], serde_json::json!([]));

    for directory in [DIR_IMPORT, DIR_DELAY_IMPORT] {
        let bytes = image(Some(directory));
        let pe = Pe::parse(&bytes, Layout::File).unwrap();
        let imports = if directory == DIR_IMPORT { pe.imports() } else { pe.delay_imports() };
        assert!(imports.is_err());
        std::fs::write(&path, bytes).unwrap();
        for json in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_xcl"));
            command.arg("inspect").arg(&path);
            if json {
                command.arg("--json");
            }
            let output = command.output().unwrap();
            assert_eq!(output.status.code(), Some(3), "directory={directory}, json={json}: {}", String::from_utf8_lossy(&output.stdout));
            assert!(output.stdout.is_empty(), "malformed imports must not produce a successful report");
            assert!(!output.stderr.is_empty(), "inspection failure must have a diagnostic");
        }
        assert_eq!(xcl::inspect::console_imports(&path), None);
    }
}
