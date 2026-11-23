---
name: xcl-report
description: >-
  Read XCL report.json and choose the next fix, and run the project's tests
  and title smoke. Use when the user pastes a report or log, asks what is
  blocking a title, or asks to run cargo test, scripts/test.ps1, scripts/smoke.ps1,
  or XCL_SAMPLE_* checks.
---

# Reports and tests

`report.rs` writes `report.json` on every run. It is the to-do list. Fields:

| Field | Meaning |
|---|---|
| `crash`, `exit_code` | Process end. Fix this before tuning frames. |
| `environment` | Preflight: DXC, GPU features (`preflight.rs`). |
| `imports_by_source`, `stubbed_imports` | Static resolution. Labels look like `hle:kernelx.dll`, `hle-cross:…`, `system`, `game-dll:…`, `stub`. A stub that was never called is not the blocker. |
| `called_stubs` | Runtime stubs, by count, with `first_caller` RVA. |
| `vfs` | `unmapped_paths`, `redirected_writes`, `whiteouts`. |
| `dstorage` | BCPack requests refused, by mode. `XCL_DSTORAGE_BCPACK_FILL=1` paints the destination and still fails the request. |
| `notes` | Free-form warnings from the run. |
| `shaders` | Formats, hardware stages, recomp completeness, missing mnemonics. |
| `dxc` | Which `dxcompiler.dll` / `dxil.dll` loaded. |
| `d3d12` | Adapter, unknown IIDs and slots, translated vs fallback pipelines, `presents`, `last_frame_dump`. |
| `audio` | Engines, voices, formats played as silence. |
| `xgameruntime` | GDK classes asked for, XCL slots not implemented. |
| `modules` | Mapped images, base and size. |

`d3d12.presents` is a present count. A non-dark sample (`XCL_FRAME_DUMP`) does not prove rendering.

## Tests

```powershell
cargo test -p xcl --lib
cargo test --workspace -- --test-threads=4
.\scripts\test.ps1
.\scripts\smoke.ps1 -Game <folder>
```

The merge bar (`CONTRIBUTING.md`) is the workspace test with real counts, plus two smoke blocks when `crates/xcl/src` outside tests changed: one built from `main`, one from the PR head with no uncommitted changes. Paste the `xcl-smoke v1` blocks unchanged. A hand-written block or a count you did not see fails review. Smoke sets `XCL_NETWORK=0`, `XCL_RUN_SECONDS` (default 60), and `XCL_ACTIVITY=1`. Compare frames shown and peak commit against `main`. Any `device removals` or `crashes` above 0 fails `PR checks`.

GPU tests call `gpu::recomp::runner::skip_test` and still pass. Report `SKIPPED` lines; they are not coverage. `XCL_REQUIRE_GPU=wave64` or `mesh` (comma-separated) makes those skips fail. `XCL_ADAPTER=1` or a name fragment picks another adapter.

Opt-in retail data, never committed: `XCL_SAMPLE_RPF`, `XCL_SAMPLE_DLLS`, `XCL_SAMPLE_TTPF`, `XCL_SAMPLE_BK2`, `XCL_SAMPLE_BVH4`, `XCL_SAMPLE_AWC`. Unset means the test passes without running. `scripts/test.ps1` lists which are unset.

CI (`.github/workflows/ci.yml`) runs `scripts/test.ps1 --skip test_title_runs_and_game_folder_is_untouched` on a GPU-less runner. Do not delete that e2e test; it checks the game folder stays byte-identical.

Shader compile spot-check: `xcl shaders <pack> --recomp --compile 64`. Do not compile the whole pack.

`XCL_TRACE_IMPORTS` logs arguments on resolved calls. See `xcl-imports`. It is for a short run; `all` wraps system imports too.
