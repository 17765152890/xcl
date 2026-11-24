use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn full_scan_and_streaming_probe_agree_on_header_evidence() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-{}-{nonce}.rpf", std::process::id()));
    let mut data = vec![0; 0x860];
    data[..4].copy_from_slice(b"9FPR");
    data[0x800..0x804].copy_from_slice(b"9FPR");
    std::fs::write(&archive, data).unwrap();

    let run = |mode| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg(mode)
            .output()
    };
    let full = run("--full");
    let probe = run("--probe");
    std::fs::remove_file(&archive).unwrap();

    let summary = |output: std::process::Output| {
        assert!(output.status.success(), "{:?}", output);
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .skip(1) // The banner describes the selected mode.
            .collect::<Vec<_>>()
            .join("\n")
    };
    let full = summary(full.unwrap());
    let probe = summary(probe.unwrap());
    assert_eq!(full, probe);
    assert!(full.contains("RPF9 archives (headers every 2 KB): 2"));
    assert!(full.contains("TOC encryption: unknown"));
}

#[test]
fn explicit_probe_rejects_conflicting_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!(
        "xcl-rpf-options-{}-{nonce}.rpf",
        std::process::id()
    ));
    std::fs::write(&archive, b"9FPR").unwrap();

    let options: [&[&str]; 4] = [
        &["--full"],
        &["--find", "record"],
        &["--extract", "record"],
        &["--out", "unused"],
    ];
    let outputs = options.map(|args| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg("--probe")
            .args(args)
            .output()
    });
    std::fs::remove_file(&archive).unwrap();

    for (args, output) in options.into_iter().zip(outputs) {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(String::from_utf8(output.stderr).unwrap().contains(args[0]));
    }
}

#[test]
fn extraction_preserves_long_duplicate_names_and_existing_files() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = std::env::temp_dir().join(format!("xcl-rpf-names-{}-{nonce}", std::process::id()));
    let output_dir = root.join("out");
    std::fs::create_dir_all(&output_dir).unwrap();
    // Verbatim paths isolate filename-component limits from MAX_PATH handling.
    let output_dir = std::fs::canonicalize(output_dir).unwrap();
    let name = format!("PS_{}", "x".repeat(251));
    std::fs::write(output_dir.join(&name), b"preserve me").unwrap();

    let mut data = b"9FPR".to_vec();
    data.resize(0x40, 0);
    for payload in [b"first payload".as_slice(), b"second payload".as_slice()] {
        let compressed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        data.push((name.len() + 1) as u8);
        data.extend_from_slice(name.as_bytes());
        data.push(0);
        data.push(50);
        for value in [compressed.len() as u32, 0, compressed.len() as u32, payload.len() as u32] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        data.extend_from_slice(&compressed);
    }
    let archive = root.join("synthetic.rpf");
    std::fs::write(&archive, data).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&archive)
        .arg("--extract")
        .arg(&name)
        .arg("--out")
        .arg(&output_dir)
        .output();
    let preserved = std::fs::read(output_dir.join(&name));
    let files = std::fs::read_dir(&output_dir).unwrap()
        .map(|entry| entry.and_then(|entry| std::fs::read(entry.path())))
        .collect::<Result<Vec<_>, _>>();
    std::fs::remove_dir_all(root).unwrap();

    let output = output.unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(preserved.unwrap(), b"preserve me");
    let mut files = files.unwrap();
    files.sort();
    assert_eq!(files, vec![b"first payload".to_vec(), b"preserve me".to_vec(), b"second payload".to_vec()]);
}
