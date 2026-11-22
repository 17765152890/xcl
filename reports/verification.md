# Verification report — test evidence, logs, regressions, limitations

All commands run through `dev\env\build.cmd` (portable MSVC 14.52 + Rust 1.100 + DXC of
the portable SDK, user-scope). GPU: NVIDIA GeForce RTX 4060 Ti. No test result below is
fabricated; failed iterations are reported as failures.

## 1. Baseline (pristine upstream tree @ `18cb400`)

- `cargo build --workspace` — **OK**, 44.12 s, 1 pre-existing upstream warning.
- `cargo test --workspace` — **263 passed, 0 failed** (240 unit + 12 console_dlls +
  1 console_shaders + 1 e2e + 6 gpu_pso + 3 xg_oracle), 26 ignored/opt-in.
- Genuine GPU work on the RTX 4060 Ti in the unit suite: UMA initialized
  ("2048 MB of graphics memory at 0x400000000, GPU-accessible without copies"),
  dozens of DXC-compiled compute dispatches, TLS carrier for 49 threads.
- Suites gated on the author's private data (`XCL_SAMPLE_DLLS/RPF/XG/...`) self-skip
  and pass empty — by upstream design.

## 2. After commit `2395f12` (PIXEvt stub module)

`cargo test -p xcl --lib` → **241 passed, 0 failed, 13 ignored** (240 baseline + the new
`pixevt_stub_module_answers_loadlibrary_and_getprocaddress`). No regressions.

Before/after status of the fixed behavior (code-path analysis + locked by the test):

| Guest call | before (base `18cb400`) | after |
|---|---|---|
| `LoadLibrary("PIXEvt.dll")` | NULL (real loader, file does not exist on Windows) | stub-module handle |
| `GetProcAddress(h, "PIXEventsReplaceBlock")` | NULL (no HLE module by that name) | 0-returning stub |
| any `GetProcAddress` on a stub-routed handle | NULL | db-backed stub |
| `GetProcAddress` on *unknown* DLLs | NULL (optional-API probing) | unchanged — NULL |

## 3. After commit `12fdc25` (FFX harness)

Single-suite run (`XCL_SAMPLE_FFX` = the folder with the two official DLLs,
`--test-threads=1 --nocapture`) — actual log:

```
running 3 tests
test ffx_xbox_fsr4_lite_accessors_execute_console_code ... xcl: loaded amd_fidelityfx_loader_xbox.dll @ 0x180000000 (size 0x8000, preferred 0x180000000)
xcl: DllMain(amd_fidelityfx_loader_xbox.dll, PROCESS_ATTACH)
xcl: loaded amd_fidelityfx_upscaler_xbox.dll @ 0x2a0d2bb0000 (size 0xac4000, preferred 0x180000000, relocated)
xcl: TLS: amd_fidelityfx_upscaler_xbox.dll: template 8 B, block 16 B, index 10, threads initialized: 2
xcl: DllMain(amd_fidelityfx_upscaler_xbox.dll, PROCESS_ATTACH)
ok
test official_ffx_console_dlls_map_and_initialise ... ok
test upscaler_console_imports_route_to_console_emulation ... ok

test result: ok. 3 passed; 0 failed
```

The upscaler mapping in that log is **not the test's own doing**: the console loader's
machine code called `LoadLibraryA` for the upscaler (its internal ensure-loaded helper),
and XCL's runtime shim served it — mapped at a *relocated* base (the loader had taken
the preferred `0x180000000`), TLS carrier installed, DllMain run. The accessor flag then
round-trips through the upscaler's global state, agreeing across both modules' exports.

Full workspace with the harness active:

```
cargo test --workspace -- --test-threads=4   (XCL_SAMPLE_FFX set)
→ 267 passed, 0 failed   (241 unit + 12 + 1 + 1 + 6 + 3 + 3 FFX)
→ ALL_TESTS_GREEN
```

## 4. Failed iterations (kept honest)

1. Accessor test v1 asserted a "not loaded → returns 0" short-circuit model:
   **FAILED** (`left: 1, right: 0`) — XCL's log showed the upscaler being loaded during
   the test, proving the ensure-loaded-forwarding semantics; test corrected (see the
   implementation report).
2. Test v2 under default `--test-threads=4`: **FAILED** — the "not loaded yet"
   pre-assertion raced the parallel suite-mate that maps the same DLL; fixed by
   serializing the suite's tests (shared process-global module table).

Both failures were behavioral findings, not flakiness being papered over: v1 corrected
the documented semantics of the official binaries, v2 exposed that these tests mutate
process-global state and must not run concurrently.

## 5. NVIDIA RTX 4060 Ti vs the author's AMD RDNA4 environment

| Property | RDNA4 (RX 9070, upstream) | RTX 4060 Ti (this run) | Note |
|---|---|---|---|
| UMA `OpenExistingHeapFromAddress` (2 GiB window) | works (measured upstream) | **works** ("2048 MB … GPU-accessible without copies") | XCL's no-copy memory model holds on NVIDIA |
| `StandardSwizzle64KBSupported` | 0 (upstream note) | **0** (probe run) | XCL's own detiling stays mandatory on both vendors — design validated cross-vendor |
| DXC + DXIL for translated shaders | Windows SDK DXC | portable-SDK DXC via `XCL_DXC_DIR` | all `gpu_*` unit tests compile and dispatch |
| GPU unit tests (compute, mesh, GS, BVH, textures) | pass | **pass** (240-test suite, 9.4 s) | second-vendor validation of the recompiler backend |

## 6. Limitations and blockers (documented, not implemented around)

- The FFX harness proves **initialization, routing and exported-function execution** —
  not game compatibility and not rendering. `ffxCreateContext`/FSR4 GPU translation on
  the official DLL is the next bounded target (roadmap Track 1).
- Private-data suites (`console_dlls_e2e`, `gpu_pso_e2e` retail cases, `xg_oracle`
  console-library comparisons) still self-skip in any environment without that data.
  The new harness is the only console-binary suite executable by the public.
- No legal console *game* executable exists to test with; that remains the field-wide
  blocker (research workspace `docs/test-executable.md`).
