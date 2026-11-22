# Technical roadmap — toward executing authentic Xbox Series applications

## Iteration-5 checkpoint (see `verification-iter5.md` + `handoff-iter5.md`)

The deserialized-FSR4 march is **inside GPU execution**: 30+ authentic pipelines created
(wave32, wave64 and console-LDS passes), `ffxDispatch` returns OK, 26 console dispatches
execute on the RTX 4060 Ti. The wave-lane discovery (NVIDIA 32..32 — iteration 4's "size
wall" was [WaveSize(64)] rejection, not size) is fixed by the GPU-verified rendezvous
form; the mid-frame fault is bisected to dispatch 14 and its identified cause (the
never-initialised PC UAV slot behind the console's read-write t# stores) is fixed in
`10faf5e` — the verifying march is the one in flight.

Next targets, in order:

1. **Hunt the dispatch-14 desync root cause** (the measured frontier — see
   `handoff-iter5.md` "The run in flight"): with the 2^20 bound the pass shows **73.3 s
   of degraded execution**, so the halves genuinely disagree on rendezvous sites and the
   degraded values stretch the loop trip counts. The decisive instrumentation is a
   bound-firing site counter in the `res` window (names the first disagreeing site);
   the prime suspect is a console DPP crossing the 32-lane boundary (translated
   wave-local), the secondary the scc input chain feeding the pass's five `[loop]`
   breaks.
2. **The pass geometry**: the grids evidence (16,16)-shaped console groups
   ((120,68)×(16,16) = 1920×1088); the deserializer's `(256,1,1)` is the same thread
   count with the wrong axis mapping. The real values sit in the FXCP metadata's 'A'
   records — decode them (or derive from grid×coverage) and validate via
   `XCL_FFX_THREADS`.
3. **Output correctness**: once the frame executes cleanly, the harness's readback + PPM
   dump measure the real upscaler output (the geometry fix decides whether the
   assertions can pass meaningfully).
4. **The bounded-spin desync**: if the RGBE fix removes the fault, the desync was
   collateral; if the bounds still fire, compare a dispatch-14-only run's output
   against the console expectation.

## Track 0 — unblock FSR4 context creation on the official DLLs (DONE, iteration 3)

Iteration 2 isolated the exact blocker (see `reports/ffx-gpu.md` §2.5-2.7): the
official FSR4 upscaler's 4th internal texture is **DXGI 67
(R9G9B9E5_SHAREDEXP) + ALLOW_UNORDERED_ACCESS**, a typed-UAV capability the
console has and the PC does not (driver-verified on the RTX 4060 Ti). Bounded plan:

1. Extract the FSR4 pipeline shaders from the official DLL's embedded containers
   (`xcl disasm`, the tooling already exists) and determine whether stores to this
   texture are packed u32s (→ raw-view substitution suffices) or unpacked
   components (→ recompiler needs RGBE software packing for image stores, the same
   pattern as the 78 buffer formats).
2. Implement the substitution at the two proven points: the console-desc
   conversion in `dev_create_committed_resource_x` + the UAV-view override
   (create as R32_TYPELESS, UAV as R32_UINT, keep SRV as the native format).
3. The frontier-lock test `ffx_console_upscaler_creates_and_destroys_a_context_on_
   the_pc_device` then flips automatically to the strict success assertions
   (context → provider version → destroy). The dispatch probe follows.

## Track 1 — authentic console middleware as a public test corpus (extend)

The officially distributed FidelityFX console DLLs (Microsoft's Xbox-GDK-Samples
`Kits/FidelityFX/signedbin`) are the only legally downloadable, console-built,
GPU-heavy binaries available to anyone. Use them harder:

1. **FSR4 context creation on the PC device** (`ffxCreateContext` on XCL's
   D3D12.X device) as an opt-in GPU test — the author's probe proved it works on
   RDNA4 with the *retail* DLL; making it a regression test with the *official* DLL
   on **NVIDIA** (RTX 4060 Ti) checks the whole console-PSO → shader-translation →
   PC-driver path on a second vendor. The FSR4 pipelines are heavy (the -O0 DXC
   threshold and the job-object memory cap in the existing harness exist for this).
2. **`ffxDispatch` descriptor-table decoding** — the retail probe already showed
   768 record words all recognized; the official DLL version pins the formats again.
3. Any other console-built middleware that ships in public sample kits is a
   candidate for the same treatment (watch `Kits/` in Xbox-GDK-Samples).

## Track 2 — console fidelity of the runtime surface (incremental)

- `PIXEvt.dll` is now a stub module (this branch). If instrumented titles need
  more than 0-returns (event block bookkeeping), implement per-function semantics.
- Sweep the `HANDOFF.md` "not done yet / deliberate" list for items that no longer
  need a game exe: XMA2 audio decode (TomBox has a decoder), Opus via
  `MSOpusDecoder.dll`, hull/domain/amplification shader stages (none in the
  current corpus, synthetic tests possible via `rdna2::encode`).
- Watch upstream commits: the repo moves daily; rebase `experimental/compat-improvements`
  frequently and keep the diff minimal.

## Track 3 — the graphics frontier (needs an executable, or more probes)

The last unsolved mile for a *game* is the D3D12.X **graphics draw/present**
submission path: root-argument → SGPR binding for graphics draws and the
PresentX/frame-pacing records. Options that do not require a retail exe:

1. **Middleware probes** (same method that cracked compute): the FSR4 upscaler
   and other console middleware record graphics-adjacent commands; extend
   `console_list` interpretation from what they emit.
2. **GDKX-built sample**: if ID@Xbox/GDKX access can be obtained (legally the
   cleanest route), build Microsoft's MIT-licensed `SimpleTriangle` for
   `Gaming.Xbox.Scarlett.x64` — a *minimal, legal, license-clean* game-shaped
   executable. This single artifact would unblock draw/present reverse-engineering
   for the entire field and can be redistributed as source with build instructions
   (not as a binary).
3. Keep `xcl inspect` + the report harness sharp: when a first legal exe arrives,
   `report.json` (called stubs, unknown slots, D3D12 trace) is the tool that
   turns it into a work list.

## Track 4 — hardware matrix (NVIDIA + iGPU)

XCL is developed on AMD RDNA4. This environment adds an RTX 4060 Ti and an AMD
iGPU. Worth verifying on every change:

- UMA via `OpenExistingHeapFromAddress` (4095 MB windows) on NVIDIA,
- DXC/DXIL acceptance of translated shaders on the NVIDIA driver,
- `OpenExistingHeapFromAddress` heap size limits per vendor,
- high-performance-GPU selection with an iGPU present (the RTX must win).

## Track 5 — quality infrastructure

- Run the full suite with `XCL_SAMPLE_FFX` set on every rebase (it is the only
  console-binary test anyone can run without private data).
- Turn the `XCL_PROBE_*` one-off probes into assertions one at a time - a probe
  that passed once is a regression test waiting to be written.
