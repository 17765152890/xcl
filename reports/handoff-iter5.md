# Handoff — iteration 5 (`experimental/ffx-functional`, 20 commits ahead of upstream `main` @ `e511b94`)

The full pass set of the authentic console FSR4 upscaler now deserializes, translates and
executes on the RTX 4060 Ti; the frontier is inside the frame's GPU execution, bisected
to one dispatch, with the leading cause fixed on the final run in flight (see §4).

## What is verified (machine-confirmed, commits in brackets)

1. **The serialized-blob format is fully solved** (iter-4, `87ed77e`): the deserializer
   creates real PC compute PSOs from the console driver's private pipeline cache — the
   FXCP bank embedded in the console DLL (927 blocks; block #200 = `FFX_FSR4_PASS_0`).
2. **The "NVIDIA size wall" was a wave-lane limit** (iter-5, `ed509a2`): the RTX 4060
   Ti's D3D12 `WaveLaneCount` range is **32..32** — `[WaveSize(64)]` shaders are
   rejected outright (WARP's range is also 32..32; the Radeon iGPU's is **64..64**, which
   inverts which shaders *it* takes). A 949 KB DXIL is accepted once the wave size is
   legal — there is no size limit.
3. **The rendezvous form** (`0f82075`): wave64 console programs on a 32..32 adapter
   translate to `[WaveSize(32)]` + groupshared ballot/cross-lane exchanges, **no barrier
   intrinsics** (DXC lowers them to `dx.op.barrier` behind the per-lane publish branches,
   which deadlocks the group). GPU-verified by a synthetic console CS (exact console
   semantics, cross-half readfirstlane + ballot-bit exec words) through the full retail
   pipeline (`wave64_compute_rendezvous_on_32_lane_waves`).
4. **The LDS budget** (`a182af2`): wave64 + console-LDS passes carry the exchange in the
   LDS block's tail dwords (D3D12's 32 KB groupshared limit is absolute). GPU-verified
   with the LDS variant of the synthetic test.
5. **The full pass set executes** (`2970a6c` lock): 30+ pipelines created, `ffxDispatch`
   code 0 (OK), 26 console dispatches recorded, all records recognized, executed on the
   GPU.
6. **The mid-frame fault bisected** (`0b7e962`): `XCL_EXEC_FIRST_DISPATCHES=n` —
   dispatch 14 (grid 507,1,1 = the 26 KB wave64 pass) is the first that faults; n ≤ 13
   survives. Bounded self-healing rendezvous spins landed (a desynced pair resyncs at
   the next site; no watchdog reset) — and the fault persisted, so it is an *invalid
   access*, not a hang.
7. **The invalid access identified and fixed** (`10faf5e`): the console's t# descriptors
   are read-write, so the FSR4 pass image-stores the neural-input texture without ever
   creating a UAV view; the recompiler's stores index the PC UAV slot at +1, which the
   iteration-3 flag-drop made impossible — the store hit an uninitialized descriptor.
   The R9G9B9E5 texture is now created as **R32_TYPELESS with the flags kept**, the
   console's views translate to **R32_UINT** (the console t# keeps format 132), the
   SRV creation also creates the missing PC UAV at +1, and the recompiler's format-132
   image path packs/unpacks 9-9-9-E5 in the shader (self-inverse with the format's own
   9-bit quantisation — both writer and reader are translated by the same code).

## The run in flight (resume here)

Three marches bracket the frontier, each verdict machine-measured:

1. **The RGBE-fix march**: the translation engaged, all 30 pipelines built, 26
   dispatches recorded and closed cleanly — **dispatch 14 still faults**. The RGBE slot
   fix was *necessary* (the UAV descriptor really was uninitialized) but not the whole
   fault.
2. **The bisect timings** (cached DXILs, ~5 s per run): n ≤ 13 all ~4.5 s — **no desync
   in any earlier dispatch**; n=14 was 17.2 s with the 2^28 bound — the TDR resetting
   the device while the bound was still counting.
3. **The 2^20-bound march** (the bound now fits under the watchdog — a desynced site
   heals in milliseconds): **dispatch 14 STILL faults, now at n=14 = 73.3 s** — not a
   TDR stall, not an immediate access fault: **over a minute of degraded execution**.
   The desync is real and *structural*: the pass's halves disagree on rendezvous
   sites; each disagreement heals at the next site, but the degraded values (exec
   words, readfirstlane results) send the scc-break loops far past the console's
   intended trip counts, burning the block-chain budget until the device gives out.

The next session's primary target is therefore the **desync root cause** — the cheapest
decisive tool: a per-site firing counter (the bound's timeout path writes the site index
to the `res` window; the harness's readback then names the exact rendezvous where the
halves first disagree). Hunt order in the pass's translated flow
(`fsr4_hlsl\24ddadf7.hlsl`, 34 sites):

- **DPP shuffles crossing the 32-lane boundary** — the translated prelude computes them
  as native-wave-local WaveReadLaneAt; a console DPP that broadcasts across the boundary
  produces per-half different values, and any such value reaching a *break condition*
  desynchronises the loop trip counts (the sites' guards stay uniform, but the loop
  entry can diverge);
- a value-level `if` the translator emits **around** a ballot/rla call site (the
  synthetic test covers linear flow only);
- the `scc` computation chain feeding the five `[loop] for(;;)` breaks — every input
  should be exchange-sourced or scalar-uniform; any per-half source is the bug.

## Exact reproduction commands

```
:: full dispatch march (all gates; the frontier-lock test):
cmd /c "set XCL_SAMPLE_FFX=D:\OpenCode\test\xbox-compat\poc\targets&& set XCL_FFX_DISPATCH=1&& set XCL_CONSOLE_LISTS=1&& set XCL_PRINT=1&& set XCL_FFX_DUMP=D:\OpenCode\test\xbox-compat\dev\reports\fsr4_dispatch_output.ppm&& D:\OpenCode\test\xbox-compat\dev\env\build.cmd cargo test --test ffx_official_console_dlls dispatches_and_executes -- --nocapture"

:: per-pass HLSL dumps (the blob-hash-named files in fsr4_hlsl\):
::   add  set XCL_FFX_HLSL=D:\OpenCode\test\xbox-compat\dev\reports\fsr4_hlsl
:: dispatch-bisect: add  set XCL_EXEC_FIRST_DISPATCHES=n   (n = 1..26)
:: direct child (no cargo, live logs): the exe at
::   dev\xcl\target\debug\build\xcl\<hash>\out\ffx_official_console_dlls-<hash>.exe
::   with XCL_SAMPLE_FFX + XCL_FFX_DISPATCH=1 + XCL_CONSOLE_LISTS=1 + XCL_FFX_CHILD=1
::   + XCL_PRINT=1 + XCL_DXC_DIR=D:\OpenCode\test\toolchains\msvc\msvc\Windows Kits\10\bin\10.0.28000.0\x64
:: blob dump of a deserialise call:  set XCL_FFX_BLOB_DIR=<dir>  (XCL_FFX_CHILD not set)

:: the rendezvous regression test (GPU, ~1 s):
cmd /c "set XCL_DXC_DIR=D:\OpenCode\test\toolchains\msvc\msvc\Windows Kits\10\bin\10.0.28000.0\x64&& D:\OpenCode\test\xbox-compat\dev\env\build.cmd cargo test -p xcl --lib wave64_compute_rendezvous -- --nocapture"
::   variants: XCL_TEST_W64_RENDEZVOUS=first (single rendezvous), =lds (LDS-tail exchange)

:: adapter comparisons (the bisect test picks the high-performance adapter):
::   XCL_ADAPTER=<substring> for XCL's device; XCL_FFX_BISECT_ADAPTER=n + XCL_FFX_BISECT_DXIL=<file>
::   for the driver-acceptance test; XCL_FFX_BISECT_NO_RS=1, XCL_FFX_BISECT_EMPTY=1 controls.
```

## Open blockers (exact signatures)

- **Dispatch 14 desync → long degraded execution (the measured frontier)**: the 26 KB
  wave64 pass's rendezvous desynchronises in real execution (n ≤ 13 = ~4.5 s clean —
  specific to this pass; the synthetic test with the same protocol passes). The
  2^28-iteration bound could not fit under the ~2 s watchdog; the 2^20 bound
  (`98d0348`) fits — and the pass now shows its true behaviour: **n=14 = 73.3 s of
  degraded execution** before the device is removed (not a TDR stall, not an immediate
  access fault). Each disagreement heals at the next site, but the degraded exec/read
  values send the scc-break loops past the console's trip counts until the device
  gives out. Root-cause hunt and the decisive instrumentation (a bound-firing site
  counter in the `res` window): see "The run in flight" above.
- **The pass geometry is wrong-but-uniform**: the recorded grids evidence (16,16)-shaped
  console groups — (120,68)×(16,16) = 1920×1088 exactly. The deserializer's
  `deserialised_cs_interface()` hardcodes `(256,1,1)` — same 256 threads, wrong axis
  mapping. Fix: the real geometry is in the FXCP metadata's 'A' records (or derivable
  from the grid × coverage); `XCL_FFX_THREADS=x,y,z` overrides it for validation.
- **The D3D12 debug layer is unavailable on this machine** (no `d3d12SDKLayers.dll`,
  the Graphics Tools optional feature; DISM needs an interactive elevated session) —
  GBV cannot run; use the dispatch bisect + the DBWIN listener instead.

## Notes for the next session

- The upstream march moved again (XDefiant running; retail-FSR4 ACCC notes in
  `HANDOFF.md` — "s_buffer_load" addressing, 32 B descriptor slots — consistent with
  what this branch's retail-informed paths implement). Rebase when convenient; our
  commits touch `emit.rs` (rendezvous prelude, format-132 path), `real.rs`
  (deserializer, view translations), `device_x.rs`, `console_list.rs`, `pso.rs`,
  `vfs.rs` (the drive-mount fix upstream still needs — it broke their e2e on a D:
  workspace), `mod.rs` (hw_wave option) and the two test files.
- The iGPU (64..64) rejects the wave32 passes — the hardware-matrix lesson: an
  adapter's wave range decides which translated shaders run; XCL now adapts (the
  rendezvous form for 32-only; native wave64 for 64-capable; wave32 programs unchanged).
- The R9G9B9E5 pack/decode pair is deliberately self-inverse (not console-bit-exact):
  both the writer and reader passes are translated by the same code. Console-exact
  RGBE (matching the hardware's rounding) only matters if an untranslated path ever
  reads the texture — none does.
