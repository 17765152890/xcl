//! Offline, fail-closed validation for redacted structural command traces.
//!
//! A successful validation is evidence hygiene only. It never enables a shader, pipeline, or
//! production eligibility path.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{BufRead, Read};
use std::path::Path;

const SCHEMA_VERSION: u8 = 1;
const MAX_EVENTS: u64 = 65_536;
const MAX_LINE_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Event {
    Create,
    Reset,
    ClearState,
    Release,
    Draw,
    DrawIndexed,
    DrawIndexedX,
    Dispatch,
    DispatchX,
    ExecuteIndirect,
}

impl Event {
    fn name(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Reset => "reset",
            Self::ClearState => "clear_state",
            Self::Release => "release",
            Self::Draw => "draw",
            Self::DrawIndexed => "draw_indexed",
            Self::DrawIndexedX => "draw_indexed_x",
            Self::Dispatch => "dispatch",
            Self::DispatchX => "dispatch_x",
            Self::ExecuteIndirect => "execute_indirect",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Accepted,
    Skipped,
    Rejected,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Category {
    None,
    PrerequisiteUnavailable,
    NativeFailure,
    InvalidArguments,
    IncompatiblePipeline,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema_version: u8,
    run_id: u64,
    sequence: u64,
    list: u32,
    epoch: u32,
    event: Event,
    outcome: Outcome,
    category: Category,
    a: u32,
    b: u32,
    c: u32,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ValidationSummary {
    pub schema_version: u8,
    pub events: u64,
    pub lists: u32,
    pub epochs: u64,
    pub accepted: u64,
    pub skipped: u64,
    pub rejected: u64,
    pub event_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ValidationError {
    pub line: u64,
    pub category: &'static str,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "structural trace rejected at line {}: {}",
            self.line, self.category
        )
    }
}

impl std::error::Error for ValidationError {}

fn reject(line: u64, category: &'static str) -> ValidationError {
    ValidationError { line, category }
}

fn command(event: Event) -> bool {
    matches!(
        event,
        Event::Draw
            | Event::DrawIndexed
            | Event::DrawIndexedX
            | Event::Dispatch
            | Event::DispatchX
            | Event::ExecuteIndirect
    )
}

fn validate_outcome(r: &Record, line: u64) -> Result<(), ValidationError> {
    let valid = match (r.outcome, r.category) {
        (Outcome::Accepted, Category::None) => true,
        (Outcome::Skipped, Category::PrerequisiteUnavailable) => command(r.event),
        (Outcome::Rejected, Category::NativeFailure) => r.event == Event::Reset,
        (Outcome::Rejected, Category::InvalidArguments) => {
            matches!(r.event, Event::DrawIndexedX | Event::DispatchX)
        }
        (Outcome::Rejected, Category::IncompatiblePipeline) => command(r.event),
        _ => false,
    };
    valid
        .then_some(())
        .ok_or_else(|| reject(line, "outcome_category"))
}

fn validate_shape(r: &Record, line: u64) -> Result<(), ValidationError> {
    let boolean = |v: u32| v <= 1;
    let valid = match r.event {
        Event::Create => r.a <= 6 && boolean(r.b) && r.c == 0,
        Event::Reset | Event::ClearState => boolean(r.a) && r.b == 0 && r.c == 0,
        Event::Release => [r.a, r.b, r.c] == [0, 0, 0],
        Event::DrawIndexedX => {
            r.c == 0 && boolean(r.b) && (r.category != Category::InvalidArguments || r.b == 0)
        }
        Event::ExecuteIndirect => boolean(r.b) && boolean(r.c) && r.b + r.c <= 1,
        Event::Draw | Event::DrawIndexed | Event::Dispatch | Event::DispatchX => true,
    };
    valid.then_some(()).ok_or_else(|| reject(line, "shape"))
}

pub fn validate(path: &Path) -> Result<ValidationSummary, ValidationError> {
    let file = std::fs::File::open(path).map_err(|_| reject(0, "open"))?;
    validate_reader(std::io::BufReader::new(file))
}

fn validate_reader(mut reader: impl BufRead) -> Result<ValidationSummary, ValidationError> {
    validate_reader_with_cap(&mut reader, MAX_EVENTS)
}

fn validate_reader_with_cap(
    mut reader: impl BufRead,
    max_events: u64,
) -> Result<ValidationSummary, ValidationError> {
    let mut line_number = 0u64;
    let mut sequence = 0u64;
    let mut run_id = None;
    let mut active = BTreeMap::<u32, u32>::new();
    let mut seen = BTreeSet::<u32>::new();
    let mut epochs = 0u64;
    let mut accepted = 0u64;
    let mut skipped = 0u64;
    let mut rejected = 0u64;
    let mut event_counts = BTreeMap::<String, u64>::new();

    loop {
        let mut bytes = Vec::with_capacity(MAX_LINE_BYTES + 1);
        let read = reader
            .by_ref()
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut bytes)
            .map_err(|_| reject(line_number + 1, "read"))?;
        if read == 0 {
            break;
        }
        line_number += 1;
        if line_number > max_events {
            return Err(reject(line_number, "event_cap"));
        }
        if bytes.len() > MAX_LINE_BYTES {
            return Err(reject(line_number, "line_cap"));
        }
        if bytes.last() != Some(&b'\n') {
            return Err(reject(line_number, "truncated_line"));
        }
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        if bytes.is_empty() {
            return Err(reject(line_number, "empty_line"));
        }
        let r: Record =
            serde_json::from_slice(&bytes).map_err(|_| reject(line_number, "json_schema"))?;
        if r.schema_version != SCHEMA_VERSION {
            return Err(reject(line_number, "schema_version"));
        }
        if r.sequence != sequence {
            return Err(reject(line_number, "sequence"));
        }
        sequence += 1;
        match run_id {
            None => run_id = Some(r.run_id),
            Some(id) if id != r.run_id => return Err(reject(line_number, "mixed_run")),
            _ => {}
        }
        if r.list == 0 {
            return Err(reject(line_number, "object_id"));
        }
        validate_outcome(&r, line_number)?;
        validate_shape(&r, line_number)?;

        match r.event {
            Event::Create => {
                if r.outcome != Outcome::Accepted {
                    return Err(reject(line_number, "lifecycle_create"));
                }
                let expected = match active.get(&r.list).copied() {
                    Some(current) => current
                        .checked_add(1)
                        .ok_or_else(|| reject(line_number, "epoch_overflow"))?,
                    None if !seen.contains(&r.list) => 1,
                    None => return Err(reject(line_number, "lifecycle_create")),
                };
                if r.epoch != expected {
                    return Err(reject(line_number, "lifecycle_create"));
                }
                active.insert(r.list, expected);
                seen.insert(r.list);
                epochs += 1;
            }
            Event::Reset | Event::ClearState => {
                let current = *active
                    .get(&r.list)
                    .ok_or_else(|| reject(line_number, "lifecycle_missing_create"))?;
                let expected = if r.outcome == Outcome::Accepted {
                    current
                        .checked_add(1)
                        .ok_or_else(|| reject(line_number, "epoch_overflow"))?
                } else {
                    current
                };
                if r.epoch != expected {
                    return Err(reject(line_number, "epoch"));
                }
                if r.outcome == Outcome::Accepted {
                    active.insert(r.list, expected);
                    epochs += 1;
                }
            }
            Event::Release => {
                let current = active
                    .remove(&r.list)
                    .ok_or_else(|| reject(line_number, "lifecycle_release"))?;
                if r.outcome != Outcome::Accepted || r.epoch != current {
                    return Err(reject(line_number, "lifecycle_release"));
                }
            }
            _ => {
                let current = *active
                    .get(&r.list)
                    .ok_or_else(|| reject(line_number, "lifecycle_missing_create"))?;
                if r.epoch != current {
                    return Err(reject(line_number, "epoch"));
                }
            }
        }

        match r.outcome {
            Outcome::Accepted => accepted += 1,
            Outcome::Skipped => skipped += 1,
            Outcome::Rejected => rejected += 1,
        }
        *event_counts.entry(r.event.name().to_owned()).or_default() += 1;
    }
    if line_number == 0 {
        return Err(reject(0, "empty_trace"));
    }
    if !active.is_empty() {
        return Err(reject(line_number + 1, "incomplete_lifecycle"));
    }
    Ok(ValidationSummary {
        schema_version: SCHEMA_VERSION,
        events: line_number,
        lists: u32::try_from(seen.len()).unwrap_or(u32::MAX),
        epochs,
        accepted,
        skipped,
        rejected,
        event_counts,
    })
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let json = match args {
        [] => false,
        [flag] if flag == "--json" => true,
        _ => {
            eprintln!("usage: xcl trace-validate <trace.jsonl> [--json]");
            return 2;
        }
    };
    match validate(path) {
        Ok(summary) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&summary).unwrap_or_default()
                );
            } else {
                println!("valid structural trace: {} events, {} lists, {} epochs; {} accepted, {} skipped, {} rejected", summary.events, summary.lists, summary.epochs, summary.accepted, summary.skipped, summary.rejected);
            }
            0
        }
        Err(error) => {
            if json {
                eprintln!("{}", serde_json::to_string(&error).unwrap_or_default());
            } else {
                eprintln!("{error}");
            }
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::{self, Read};
    use std::rc::Rc;

    struct LargeLine {
        consumed: Rc<Cell<usize>>,
        remaining: usize,
        chunk: [u8; 4096],
    }

    impl Read for LargeLine {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let available = self.fill_buf()?;
            let count = available.len().min(output.len());
            output[..count].copy_from_slice(&available[..count]);
            self.consume(count);
            Ok(count)
        }
    }

    impl BufRead for LargeLine {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Ok(&self.chunk[..self.remaining.min(self.chunk.len())])
        }

        fn consume(&mut self, amount: usize) {
            let amount = amount.min(self.remaining);
            self.remaining -= amount;
            self.consumed.set(self.consumed.get() + amount);
        }
    }

    fn record(
        seq: u64,
        list: u32,
        epoch: u32,
        event: &str,
        outcome: &str,
        category: &str,
        abc: [u32; 3],
    ) -> String {
        format!(
            r#"{{"schema_version":1,"run_id":7,"sequence":{seq},"list":{list},"epoch":{epoch},"event":"{event}","outcome":"{outcome}","category":"{category}","a":{},"b":{},"c":{}}}"#,
            abc[0], abc[1], abc[2]
        ) + "\n"
    }

    fn valid() -> String {
        [
            record(0, 1, 1, "create", "accepted", "none", [0, 1, 0]),
            record(
                1,
                1,
                1,
                "dispatch_x",
                "rejected",
                "invalid_arguments",
                [u32::MAX, 2, 1],
            ),
            record(2, 1, 1, "reset", "rejected", "native_failure", [1, 0, 0]),
            record(3, 1, 2, "reset", "accepted", "none", [1, 0, 0]),
            record(
                4,
                1,
                2,
                "draw",
                "skipped",
                "prerequisite_unavailable",
                [3, 1, 0],
            ),
            record(5, 1, 2, "release", "accepted", "none", [0, 0, 0]),
        ]
        .concat()
    }

    fn check(text: &str) -> Result<ValidationSummary, ValidationError> {
        validate_reader(std::io::Cursor::new(text.as_bytes()))
    }

    #[test]
    fn valid_trace_has_deterministic_summary() {
        let a = check(&valid()).unwrap();
        let b = check(&valid()).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            (a.events, a.lists, a.epochs, a.accepted, a.skipped, a.rejected),
            (6, 1, 2, 3, 1, 2)
        );
    }

    #[test]
    fn active_create_rebind_advances_epoch_without_creating_a_new_identity() {
        let trace = [
            record(0, 1, 1, "create", "accepted", "none", [3, 0, 0]),
            record(1, 1, 2, "create", "accepted", "none", [0, 0, 0]),
            record(2, 1, 3, "reset", "accepted", "none", [0, 0, 0]),
            record(3, 1, 3, "release", "accepted", "none", [0, 0, 0]),
        ]
        .concat();
        let summary = check(&trace).unwrap();
        assert_eq!((summary.events, summary.lists, summary.epochs), (4, 1, 3));

        let stale_epoch = trace.replacen(
            "\"epoch\":2,\"event\":\"create\"",
            "\"epoch\":1,\"event\":\"create\"",
            1,
        );
        assert_eq!(
            check(&stale_epoch).unwrap_err().category,
            "lifecycle_create"
        );

        let after_release = format!(
            "{}{}",
            [
                record(0, 1, 1, "create", "accepted", "none", [0, 0, 0]),
                record(1, 1, 1, "release", "accepted", "none", [0, 0, 0]),
            ]
            .concat(),
            record(2, 1, 2, "create", "accepted", "none", [0, 0, 0])
        );
        assert_eq!(
            check(&after_release).unwrap_err().category,
            "lifecycle_create"
        );
    }

    #[test]
    fn malformed_hostile_truncated_and_oversized_input_fails_closed() {
        let malformed = check("{}\n").unwrap_err();
        assert_eq!(malformed.category, "json_schema");
        assert_eq!(
            serde_json::to_string(&malformed).unwrap(),
            r#"{"line":1,"category":"json_schema"}"#
        );
        assert_eq!(
            check(&format!("{{{}}}\n", "x".repeat(MAX_LINE_BYTES)))
                .unwrap_err()
                .category,
            "line_cap"
        );
        assert_eq!(
            check(valid().trim_end()).unwrap_err().category,
            "truncated_line"
        );
        let unknown = valid().replacen("\"a\":0", "\"extra\":0,\"a\":0", 1);
        assert_eq!(check(&unknown).unwrap_err().category, "json_schema");
        assert_eq!(
            validate_reader_with_cap(std::io::Cursor::new(valid().as_bytes()), 2)
                .unwrap_err()
                .category,
            "event_cap"
        );

        let consumed = Rc::new(Cell::new(0));
        let huge = LargeLine {
            consumed: Rc::clone(&consumed),
            remaining: 64 * 1024 * 1024,
            chunk: [b'x'; 4096],
        };
        assert_eq!(validate_reader(huge).unwrap_err().category, "line_cap");
        assert_eq!(consumed.get(), MAX_LINE_BYTES + 1);
    }

    #[test]
    fn mixed_runs_sequence_epoch_and_release_finality_are_rejected() {
        assert_eq!(
            check(&valid().replacen("\"run_id\":7", "\"run_id\":8", 1))
                .unwrap_err()
                .category,
            "mixed_run"
        );
        assert_eq!(
            check(&valid().replacen("\"sequence\":1", "\"sequence\":9", 1))
                .unwrap_err()
                .category,
            "sequence"
        );
        assert_eq!(
            check(&valid().replacen(
                "\"epoch\":2,\"event\":\"reset\",\"outcome\":\"accepted\"",
                "\"epoch\":1,\"event\":\"reset\",\"outcome\":\"accepted\"",
                1
            ))
            .unwrap_err()
            .category,
            "epoch"
        );
        let mut after = valid();
        after.push_str(&record(6, 1, 2, "draw", "accepted", "none", [3, 1, 0]));
        assert_eq!(
            check(&after).unwrap_err().category,
            "lifecycle_missing_create"
        );
    }

    #[test]
    fn invalid_outcome_shape_and_incomplete_lifecycle_are_rejected() {
        assert_eq!(
            check(&valid().replacen(
                "\"category\":\"invalid_arguments\"",
                "\"category\":\"none\"",
                1
            ))
            .unwrap_err()
            .category,
            "outcome_category"
        );
        assert_eq!(
            check(&valid().replacen(
                "\"event\":\"release\",\"outcome\":\"accepted\",\"category\":\"none\",\"a\":0",
                "\"event\":\"release\",\"outcome\":\"accepted\",\"category\":\"none\",\"a\":1",
                1
            ))
            .unwrap_err()
            .category,
            "shape"
        );
        let incomplete = valid()
            .lines()
            .take(5)
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        assert_eq!(
            check(&incomplete).unwrap_err().category,
            "incomplete_lifecycle"
        );
    }
}
