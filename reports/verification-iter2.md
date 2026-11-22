# Verification report — iteration 2 (`experimental/ffx-gpu-compat`)

All results are real runs on this machine (NVIDIA GeForce RTX 4060 Ti, portable MSVC +
Rust 1.100 + SDK DXC). Failed iterations are reported as failures; every claim below is
machine-verified or explicitly marked as not reached.

## 1. Reproducibility of iteration 1 (checked first)

`cargo test --workspace` on the branch base (`12fdc25`): **267 passed, 0 failed**
(241 unit incl. the pixevt test, 12 console_dlls, 1 console_shaders, 1 e2e, 3 FFX,
6 gpu_pso, 3 xg_oracle) — identical to iteration 1's final state. The
`xcl-probe_standard_swizzle_support` datapoint re-confirmed: `StandardSwizzle64KBSupported
= 0` on this NVIDIA GPU (same as upstream's RDNA4).

## 2. Investigation runs (all with `XCL_SAMPLE_FFX` = the official DLLs)

| # | Attempt | Result |
|---|---|---|
| 1 | Provider `ffxQuery(NULL, GetVersions)` before create | **crash** 0xC0000005 at provider+0x8055 (read at 0x9) — uninitialized global registry; no device calls before it |
| 2 | Retail-era chain via loader and provider | `FFX_API_RETURN_ERROR (1)`, no message callback use, no device calls — pre-device descriptor validation |
| 3 | Guessed version-desc types (4 / 0x00010009 / 5) with a binary-derived id | ERROR 1 for all; type 5 gave **NO_PROVIDER (4)** — the override machinery runs, the id was not a matching provider version |
| 4 | MS sample chain: **BACKEND_XBOX + UpscaleVersion 4.1.2** | **accepted** — real GPU work begins: AddRef, 4 descriptor heaps, 3 committed textures + SRV/UAV views, then E_INVALIDARG on texture 4 |
| 5 | After the state-mask fix | E_INVALIDARG persists with `state 0x8` — isolating the cause to the format: **DXGI 67 = R9G9B9E5_SHAREDEXP + ALLOW_UNORDERED_ACCESS**; upscaler fail-fasts 0xC0000409 |
| 6 | `CheckFormatSupport` on the RTX 4060 Ti | R8G8B8A8 ✓ typed-UAV, R16G16B16A16F ✓, R11G11B10F ✓, **R9G9B9E5 ✗ typed-UAV** — the console-vs-PC capability difference, driver-verified |

## 3. Final test state

- Gated run (`XCL_SAMPLE_FFX` + `XCL_FFX_GPU=1`, `--test-threads=1 --nocapture`):

```
CheckFormatSupport(R9G9B9E5_SHAREDEXP = 67): shader_load true, typed_uav false
[fatal runtime error in the child: Rust cannot catch foreign exceptions, aborting]
VERIFIED FRONTIER: the sample-exact chain is accepted; the upscaler creates real GPU
state through XCL (heaps + 3 textures in the log above); blocked precisely at DXGI 67
(R9G9B9E5_SHAREDEXP) + ALLOW_UNORDERED_ACCESS - a console-vs-PC typed-UAV capability
difference.
test result: ok. 4 passed; 0 failed
```

- Full workspace regression (harness active, GPU gate off):
  **268 passed, 0 failed** (267 + the new GPU milestone test passing as a frontier
  lock). No previously passing test regressed; the `real.rs` change is behind the
  console slot-76 path only (no PC-path behavior change; unit suite unchanged).

## 4. NVIDIA vs the author's AMD environment

| Item | RDNA4 (upstream) | RTX 4060 Ti (here) |
|---|---|---|
| Real backend init | works | works |
| UMA window via `OpenExistingHeapFromAddress` | works | works |
| Console FFX create → device work | works (retail DLLs) | works (official DLLs) — second-vendor confirmation |
| R9G9B9E5 typed-UAV | n/a (retail DLL doesn't use it) | **unsupported → the blocker** (D3D12 typed-UAV format set is PC-wide, not vendor-specific) |
| `StandardSwizzle64KBSupported` | 0 | 0 |

## 5. Blockers (exact failure points)

1. **R9G9B9E5 typed-UAV** (context creation, resource #4 of the official FSR4
   upscaler): console requests `ALLOW_UNORDERED_ACCESS` on DXGI 67; PC
   `CheckFormatSupport` reports no typed UAV → `CreateCommittedResource3` (slot 76)
   returns 0x80070057 → the console DLL fail-fasts 0xC0000409 (uncatchable through
   Rust). Emulation strategy (raw R32 substitution, two variants depending on whether
   the console shaders store packed or unpacked): `reports/ffx-gpu.md` §2.7.
2. **Global provider enumeration**: `ffxQuery(NULL, ffxQueryDescGetVersions)` faults at
   provider+0x8055 on both the provider and (via delegation) the loader — an
   uninitialized registry walk. Microsoft's sample only ever queries versions through
   a live context; unreachable until creation works.

## 6. Not tested / out of scope this iteration

- FSR4 dispatch and any GPU execution of FSR4 pipelines (blocked upstream of it).
- The loader's FSR3-override path (`CreateContextDescOverrideVersion` with a version
  id from a working enumeration) — same upstream blocker.
