# Runtime evidence contract

This document defines the minimum lawful, non-sensitive evidence required before validation-only
mesh or per-slot lowering may participate in real-program eligibility or pipeline selection.
Evidence is observational. Missing, inconsistent, overflowing, or ambiguous evidence keeps the
production path disabled.

## Implementation status

The bounded, versioned structural JSONL sink exists as an explicit opt-in through
`XCL_STRUCTURAL_TRACE`. It covers command-list lifecycle, draw, dispatch, and indirect-command
events with pseudonymous identifiers, fixed categories, and strict caps. The offline admission
validator does not exist yet, and the sink does not by itself enable production eligibility.

Current frame and shader tracing remains interactive diagnostic logging, not compliant evidence
export. Those logs may contain raw addresses or bounded memory-derived diagnostics and must not be
shared as contract-compliant evidence. Only output produced through the structural sink follows
the redaction rules below.

## Required evidence

### Launch identity

- Parsed PE structure: machine, entry RVA, image size, relocations, imports, delay imports, TLS,
  and exception directory.
- Parsed launcher configuration needed for executable selection, memory mode, and storage.
- Resolver outcome for static imports plus actually called unresolved or stubbed APIs.
- Exit or crash status.

Reports must not contain executable bytes.

### Shader and pipeline structure

For each observed pipeline, retain only keyed pseudonymous identifiers and structural metadata:

- pseudonymous shader identifiers and hardware stages;
- stage fields needed to derive mode, user-register count, thread-group dimensions, exports, and
  resource requirements;
- pipeline kind and exact stage pairing;
- keyed pseudonymous root-signature identifier and static-sampler structure;
- translation completeness, compiler result, and stable fail-closed rejection category;
- synchronous, asynchronous, fallback, or two-pass creation mode.

### Root arguments

For every relevant draw or dispatch, capture:

- root-parameter kind, byte offset, declared count, and graphics or compute namespace;
- bounded constants;
- root-buffer address class and bounded structural words;
- descriptor-table offset and bounded descriptor slots.

Persist neither arbitrary memory ranges nor raw pointer values.

### Dispatch and draw geometry

- Exact command entry.
- Dispatch dimensions supplied by that command, never inferred from draw counts.
- Indexed or non-indexed draw arguments and input-layout structure.
- Checked two-pass totals, record size, groups per record, and result size.
- Zero-work behavior and stable limit or overflow rejection categories.

### Resource contracts

For every resource a candidate shader names, capture:

- descriptor type and decoded structural fields;
- dimension, format, mip, array, sample, flag, stride, and bounded-range information;
- read or write role, first and last command use, and title-memory synchronization state;
- render-target, depth, UAV, copy, clear, and barrier relationships;
- exact agreement between required registers and observed bindings.

### Command sequence

Record a bounded structural sequence containing:

- command-list creation, reset, clear-state, and initial or bound pipeline;
- root-signature and root-argument changes;
- barriers;
- draw, dispatch, and indirect commands;
- two-pass compute, UAV barrier, and mesh ordering;
- submission, fence completion, and presentation;
- rejection before native command emission.

The sequence must prove that reset epochs do not reuse stale pipeline or root state and that ABI
allocations remain alive through submission.

## Persisted format

Use bounded, redacted JSON Lines with one structural event per record:

- schema version;
- process-local run identifier;
- monotonic sequence and command-list epoch;
- anonymized object identifiers;
- keyed pseudonymous shader, pipeline, and root-signature identifiers, with the key withheld from
  exported evidence;
- structural arguments and checked-plan fields;
- resource shape, format, and role;
- accepted, skipped, or rejected outcome with a stable category.

The exported evidence bundle must never contain absolute paths, raw virtual addresses, executable
bytes, proprietary shader bytecode, unbounded descriptor payloads, package identifiers, user
identifiers, content names, asset names, or stable raw hashes that fingerprint proprietary content.
Private local dumps may remain available for authorized debugging, but they are not compliant
export evidence and must not be published with the bundle.

## Admission gates

A real program is eligible only when every applicable gate passes:

1. Decoding and structural validation are complete.
2. Stage descriptors and pipeline pairing are internally consistent.
3. The observed root layout supplies each required register or table entry exactly once.
4. Required descriptors match the observed resource contract and no undeclared resource is used.
5. Dispatch dimensions come from an observed dispatch command.
6. Checked two-pass planning succeeds within API limits.
7. Resource extents, formats, alignment, and memory windows validate without clamping.
8. The command sequence proves required barriers and allocation lifetime.
9. Generated shader code compiles.
10. A mutation-sensitive synthetic test covers the same structural contract.
11. A replay or canary run produces no new rejection, device removal, stale-state use, or guard
    corruption.

Any missing or conflicting evidence means ineligible. Never guess dispatch dimensions, root
mappings, resource layouts, or draw-to-dispatch conversion.

## Tests required before enabling eligibility

- Deterministic serialization and redaction.
- Trace bounds and hostile-value truncation.
- Root-layout mismatch and missing-descriptor rejection.
- Dispatch overflow, zero-work, and API-limit handling.
- Reset and clear-state epoch isolation.
- Mutations that omit a required barrier or allocate separate two-pass result buffers.
- Actual command-entry coverage for compute, two-pass, indexed draw, standard draw, and graphics
  indirect execution.
- Replay equivalence against synthetic CPU and GPU references.
- A negative test proving incomplete evidence cannot enable production eligibility.

## Collection procedure

1. Inspect the executable and review structural PE and import output only.
2. Run once with ordinary aggregate reporting.
3. If graphics initialization is reached, enable bounded frame tracing.
4. Narrow later local captures to specific shaders and enforce a small event cap.
5. Export only the sink's redacted structural JSONL plus separately reviewed aggregate results.
6. Implement and run the offline validator against every admission gate.
7. Add a synthetic regression reproducing the observed contract.
8. Enable only that exact verified contract behind a fail-closed predicate.

## Prohibited scope

- No DRM bypass.
- No key discovery, extraction, derivation, or publication.
- No package decryption.
- No protected-package probing intended to recover encrypted content.
- No inclusion or publication of executable bytes, proprietary shader bytecode, content payloads,
  or identifying asset names in the exported evidence bundle.

Authorization to inspect an unpacked executable does not authorize any prohibited action above.
