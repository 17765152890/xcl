//! Opt-in, bounded and redacted structural command trace.
//!
//! This is evidence plumbing, not a diagnostic dump: it never accepts paths, addresses,
//! bytecode, descriptors, names or arbitrary strings. I/O is best-effort and cannot affect
//! command execution.

use serde::Serialize;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

const SCHEMA_VERSION: u8 = 1;
const DEFAULT_EVENT_CAP: u64 = 4096;
const MAX_EVENT_CAP: u64 = 65_536;
const MAX_LINE_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    Create,
    Reset,
    ClearState,
    Close,
    Submit,
    Release,
    Draw,
    DrawIndexed,
    DrawIndexedX,
    Dispatch,
    DispatchX,
    ExecuteIndirect,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    Accepted,
    Skipped,
    Rejected,
}

/// Coarse, stable categories deliberately avoid carrying implementation errors or content text.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Category {
    None,
    PrerequisiteUnavailable,
    NativeFailure,
    InvalidArguments,
    IncompatiblePipeline,
}

#[derive(Serialize)]
struct Record {
    schema_version: u8,
    run_id: u64,
    sequence: u64,
    list: u32,
    epoch: u32,
    event: Kind,
    outcome: Outcome,
    category: Category,
    a: u32,
    b: u32,
    c: u32,
}

struct Trace<W: Write> {
    writer: W,
    enabled: bool,
    cap: u64,
    run_id: u64,
    sequence: u64,
    next_id: u32,
    ids: HashMap<usize, u32>,
    epochs: HashMap<usize, u32>,
}

impl<W: Write> Trace<W> {
    fn new(writer: W, cap: u64, run_id: u64) -> Self {
        Self {
            writer,
            enabled: true,
            cap: cap.min(MAX_EVENT_CAP),
            run_id,
            sequence: 0,
            next_id: 1,
            ids: HashMap::new(),
            epochs: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn disabled(writer: W) -> Self {
        let mut trace = Self::new(writer, 0, 0);
        trace.enabled = false;
        trace
    }

    fn object_id(&mut self, raw: usize) -> u32 {
        if let Some(id) = self.ids.get(&raw) {
            return *id;
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1).max(1);
        self.ids.insert(raw, id);
        id
    }

    fn emit(&mut self, raw: usize, kind: Kind, outcome: Outcome, category: Category, fields: [u32; 3]) {
        if !self.enabled || self.sequence >= self.cap {
            return;
        }
        // Lifecycle hooks also observe native lists that never entered the scoped trace.
        if matches!(kind, Kind::Close | Kind::Submit | Kind::Release) && !self.epochs.contains_key(&raw) {
            return;
        }
        let list = self.object_id(raw);
        let epoch = self.epochs.entry(raw).or_insert(0);
        if matches!(outcome, Outcome::Accepted) && matches!(kind, Kind::Create | Kind::Reset | Kind::ClearState) {
            *epoch = epoch.saturating_add(1);
        }
        let record = Record {
            schema_version: SCHEMA_VERSION,
            run_id: self.run_id,
            sequence: self.sequence,
            list,
            epoch: *epoch,
            event: kind,
            outcome,
            category,
            a: fields[0],
            b: fields[1],
            c: fields[2],
        };
        let Ok(mut line) = serde_json::to_vec(&record) else {
            self.enabled = false;
            return;
        };
        if line.len() + 1 > MAX_LINE_BYTES {
            self.enabled = false;
            return;
        }
        line.push(b'\n');
        if self.writer.write_all(&line).and_then(|_| self.writer.flush()).is_err() {
            self.enabled = false;
            return;
        }
        self.sequence += 1;
        if matches!(kind, Kind::Release) {
            self.ids.remove(&raw);
            self.epochs.remove(&raw);
        }
    }
}

struct RuntimeTrace(Trace<BufWriter<std::fs::File>>);

#[cfg(test)]
thread_local! {
    static TEST_TRACE: std::cell::RefCell<Option<Trace<Vec<u8>>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn start_test_capture() {
    TEST_TRACE.with(|trace| *trace.borrow_mut() = Some(Trace::new(Vec::new(), 64, 7)));
}

#[cfg(test)]
pub(crate) fn finish_test_capture() -> Vec<serde_json::Value> {
    let bytes = TEST_TRACE.with(|trace| trace.borrow_mut().take().unwrap().writer);
    String::from_utf8(bytes).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}

fn runtime() -> Option<&'static Mutex<RuntimeTrace>> {
    static TRACE: OnceLock<Option<Mutex<RuntimeTrace>>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os("XCL_STRUCTURAL_TRACE");
            let cap = std::env::var("XCL_STRUCTURAL_TRACE_CAP").ok();
            initialize(path.as_deref(), cap.as_deref()).map(Mutex::new)
        })
        .as_ref()
}

fn initialize(path: Option<&std::ffi::OsStr>, cap: Option<&str>) -> Option<RuntimeTrace> {
    let path = path?;
    let cap = cap.and_then(|value| value.parse::<u64>().ok()).unwrap_or(DEFAULT_EVENT_CAP).min(MAX_EVENT_CAP);
    let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_nanos() as u64)
        .unwrap_or_default();
    // RandomState keys the hasher per process. This leaves only an opaque run
    // pseudonym in the trace rather than a recoverable timestamp or process ID.
    let mut run = std::collections::hash_map::RandomState::new().build_hasher();
    run.write_u64(time);
    run.write_u32(std::process::id());
    let run_id = run.finish();
    Some(RuntimeTrace(Trace::new(BufWriter::new(file), cap, run_id)))
}

pub(super) fn event(list: *mut c_void, kind: Kind, outcome: Outcome, category: Category, fields: [u32; 3]) {
    #[cfg(test)]
    if TEST_TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        let Some(trace) = trace.as_mut() else { return false };
        trace.emit(list as usize, kind, outcome, category, fields);
        true
    }) {
        return;
    }
    let Some(trace) = runtime() else { return };
    if let Ok(mut trace) = trace.lock() {
        trace.0.emit(list as usize, kind, outcome, category, fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn deterministic_ids_epochs_and_schema_expose_no_raw_identity() {
        let mut trace = Trace::new(Vec::new(), 8, 7);
        trace.emit(0xdead_beef, Kind::Create, Outcome::Accepted, Category::None, [0, 0, 0]);
        trace.emit(0xdead_beef, Kind::DispatchX, Outcome::Accepted, Category::None, [2, 3, 1]);
        trace.emit(0xcafe_babe, Kind::Create, Outcome::Accepted, Category::None, [0, 0, 0]);
        let text = String::from_utf8(trace.writer).unwrap();
        assert_eq!(text.lines().count(), 3);
        assert!(text.contains(r#""schema_version":1,"run_id":7,"sequence":0,"list":1,"epoch":1"#));
        assert!(text.contains(r#""sequence":1,"list":1,"epoch":1,"event":"dispatch_x""#));
        assert!(text.contains(r#""sequence":2,"list":2,"epoch":1"#));
        assert!(!text.contains("dead") && !text.contains("cafe") && !text.contains(":\\"));
    }

    #[test]
    fn rejected_reset_keeps_epoch_and_real_lifecycle_hooks_are_wired() {
        TEST_TRACE.with(|trace| *trace.borrow_mut() = Some(Trace::new(Vec::new(), 8, 7)));
        let list = 0x7fff_1234usize as *mut c_void;
        unsafe {
            super::super::list_x::on_create(
                list,
                0x100,
                0,
                windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT,
            );
        }
        event(list, Kind::Reset, Outcome::Rejected, Category::NativeFailure, [0, 0, 0]);
        super::super::list_x::on_release(list as usize);
        let bytes = TEST_TRACE.with(|trace| trace.borrow_mut().take().unwrap().writer);
        let text = String::from_utf8(bytes).unwrap();
        let records: Vec<serde_json::Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0]["event"], "create");
        assert_eq!(records[0]["epoch"], 1);
        assert_eq!(records[1]["event"], "reset");
        assert_eq!(records[1]["outcome"], "rejected");
        assert_eq!(records[1]["category"], "native_failure");
        assert_eq!(records[1]["epoch"], 1);
        assert_eq!(records[2]["event"], "release");
        assert_eq!(records[2]["list"], 1);
    }

    #[test]
    fn cap_disabled_mode_and_hostile_numeric_values_are_bounded() {
        let mut trace = Trace::new(Vec::new(), 2, 7);
        for _ in 0..4 {
            trace.emit(usize::MAX, Kind::Dispatch, Outcome::Rejected, Category::InvalidArguments, [u32::MAX; 3]);
        }
        let text = String::from_utf8(trace.writer).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(text.lines().all(|line| line.len() < MAX_LINE_BYTES));
        assert!(!text.contains(&format!("{:#x}", usize::MAX)));

        let mut disabled = Trace::disabled(Vec::new());
        disabled.emit(1, Kind::Draw, Outcome::Accepted, Category::None, [1, 2, 3]);
        assert!(disabled.writer.is_empty());
    }

    struct Fails {
        calls: usize,
    }

    impl Write for Fails {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            Err(io::Error::other("synthetic failure"))
        }

        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }

    #[test]
    fn io_failure_latches_off_without_retrying() {
        let mut trace = Trace::new(Fails { calls: 0 }, 8, 7);
        trace.emit(1, Kind::Draw, Outcome::Accepted, Category::None, [1, 1, 0]);
        trace.emit(1, Kind::Draw, Outcome::Accepted, Category::None, [1, 1, 0]);
        assert!(!trace.enabled);
        assert_eq!(trace.writer.calls, 1);
        assert_eq!(trace.sequence, 0);
    }

    #[test]
    fn release_without_an_observed_create_is_not_emitted() {
        let mut trace = Trace::new(Vec::new(), 8, 7);
        trace.emit(0x1000, Kind::Release, Outcome::Accepted, Category::None, [0, 0, 0]);
        assert!(trace.writer.is_empty());
        assert_eq!(trace.sequence, 0);

        trace.emit(0x1000, Kind::Create, Outcome::Accepted, Category::None, [0, 0, 0]);
        trace.emit(0x1000, Kind::Release, Outcome::Accepted, Category::None, [0, 0, 0]);
        assert_eq!(String::from_utf8(trace.writer).unwrap().lines().count(), 2);
    }

    #[test]
    fn tracked_submit_without_close_is_emitted_and_validator_rejects_it() {
        let mut trace = Trace::new(Vec::new(), 8, 7);
        trace.emit(0x1000, Kind::Create, Outcome::Accepted, Category::None, [0, 0, 0]);
        trace.emit(0x1000, Kind::Submit, Outcome::Accepted, Category::None, [0, 0, 0]);
        let text = String::from_utf8(trace.writer).unwrap();
        assert_eq!(text.lines().count(), 2);
        let error = crate::structural_trace_validate::validate_reader(std::io::Cursor::new(text)).unwrap_err();
        assert_eq!(error.line, 2);
        assert_eq!(error.category, "lifecycle_submit");
    }

    #[test]
    fn absent_and_invalid_output_paths_never_create_or_activate_a_sink() {
        let candidate = std::env::temp_dir().join(format!("xcl_structural_trace_absent_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&candidate);
        assert!(initialize(None, None).is_none());
        assert!(!candidate.exists());

        let directory = std::env::temp_dir();
        assert!(initialize(Some(directory.as_os_str()), Some("1")).is_none());
    }
}
