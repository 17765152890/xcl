# Verification report — iteration 3 (`experimental/ffx-functional`)

GPU: NVIDIA GeForce RTX 4060 Ti. All results are real runs. Failed iterations are
reported as failures with their exact signatures.

## 1. Baseline

The 268-test regression from iteration 2 confirmed green at the branch point; one
pre-existing flake in the FFX accessor test (an order-dependent "not loaded yet"
pre-assertion) surfaced in a parallel run and was fixed (assertions now hold in any
order; the order-dependence was mine, from iteration 1).

## 2. The R9G9B9E5 blocker — resolved (Priority One)

Investigation chain, each step machine-verified:

1. The failing texture's heap is **DEFAULT (console heap type 1 = PC heap type 1)** —
   not an upload heap (my interim hypothesis, disproven by the SDK enums).
2. The driver rejects **both** the `ALLOW_UNORDERED_ACCESS` flag **and** the UAV
   initial state for DXGI 67 on the PC (typed-UAV format set), while console RDNA2
   natively supports 9-9-9-E5 image stores. Evidence: dropping the flag alone still
   returned E_INVALIDARG (state 0x8 remained); fixing both made creation succeed.
3. The ScarlettX project was examined per request (below): it contains no
   R9G9B9E5 handling — its real-title path avoids the class of problem entirely.

**Fix (commit `0bce8c9`)**: at the committed-resource console→PC boundary (the same
point as the existing Layout translation) — keep the format (SRVs of R9G9B9E5 are
PC-legal), drop the GPU-write flags, and normalise a matching UAV initial state to
COMMON. Verified by the milestone below. Known open risk, now tested at dispatch
(the FSR4 pipelines never got far enough to bind it — see §4).

## 3. Target milestone — FidelityFX context lifecycle: **ACHIEVED (Verified)**

```
ffxCreateContext via loader, sample-exact: BACKEND_XBOX + UpscaleVersion 4.1.2:
    code 0 (OK), context 0x22745786b90
context words [name-string, provider-ctx, 0, cookie]
inner ctx words [vtable 0x180aad7b0, ...]              (a real C++ provider object)
ffxDestroyContext via provider: code 0 (OK)
test result: ok (create/destroy test, gated run)
```

A genuine Xbox FSR4 4.1.2 context — created, structurally verified, and cleanly
destroyed — through XCL's emulated console D3D12.X device on the RTX 4060 Ti, with
the authentic officially-distributed binaries. All GPU resources created through
the console vtable: 4 descriptor heaps, 4 committed 1920×1080 textures
(R8G8B8A8_UNORM, R16G16B16A16_FLOAT ×2, R9G9B9E5_SHAREDEXP-translated),
SRV/UAV views, and the FSR4 root signature captured through XCL's shader path.

Context-bound findings (precise signatures, documented in the test):
- The **loader's** context dispatch (ffxQuery/ffxDestroyContext walk the wrapper)
  faults at loader+0x16a6; the **provider's** context-bound query faults at
  upscaler+0x1b9a on an uninitialised .data registry slot (the same family as the
  context-less GetVersions crash). The destroy works through the provider with the
  inner context — asserted in the test; the query is gated behind XCL_FFX_QUERY=1.

## 4. Stretch goal — dispatch: **Blocked precisely at the console pipeline cache**

The dispatch flow (real input content: gradient color, two-band depth, zero motion;
three frames; console list; execution; readback — all implemented in the harness)
reaches the FSR4 provider's native pipeline setup and is blocked there:

```
[SHADER] rootsig cee296d8158a6d15: 304 B, format dxbc, parts [RTS0]   ← captured OK
xcl: D3D12.X: DeserializePipelineStateX – console pipeline caches are not supported
xcl: [debug] FFX native PSO 'FFX_FSR4_PASS_0': deserialize failed (0x80004001)
xcl: [debug] Assertion failed: CreateModel(internal_context, correctPreset, model)
        == FFX_API_RETURN_OK
    In ffxProvider_FSR4_Common::Dispatch (..\fsr4\internal\fsr4_common.cpp:1401)
→ crash 0xC0000005 in the console DLL (null PSO during recording)
```

The console DLL names its own source file and line. The serialised blob (now
instrumented, commit `8b12698`) is mapped: a sub-table pointer + (ptr, size)
segments; the 0xec0-byte shader payload sits in **title memory at GPU VA
0x402300000** (XCL's UMA resolves it directly — console address = CPU address) and
is **raw console-compiler output with no XBCH/XDAT/FSSM container structure** —
the console driver's private pipeline-cache serialisation. XCL's recompiler
consumes FSSM containers; this payload needs a dedicated console driver-cache
format RE before translation can flow — consistent with upstream's author note
("ACCO-cooked", deferred) and with his newest upstream commit ("the console shader
compiler as a recompiler test generator") targeting exactly this class.

The test locks the frontier reproducibly (child-process reproduction; the parent
asserts the crash code + the provider assertion + the deserializer line). The
full flow below the frontier engages automatically when a deserializer exists.

## 5. ScarlettX examination (requested)

Cloned and reviewed (Freminet/ScarlettX @ bda18b4, "Concept Destruction playable
at full speed on a hardware GPU"). Architecture is the **complement** of XCL's:

| Aspect | ScarlettX | XCL |
|---|---|---|
| D3D12.X | Clean-room proxy ABI (`d3d12_xs_abi.h` from observation); their own renderer/backend | Wrap the real PC driver; guard-split console/PC vtables |
| Real title | Concept Destruction (Unity IL2CPP) — works because **Unity embeds DXIL containers**; ScarlettX strips Xbox-private chunks (`XBOX`/`XS2M`) | Runs the authentic console middleware natively (official FFX binaries) |
| Console-native RDNA2 microcode | **0%** ("RDNA2 XS2 Shader Lifting: 0 of 16"; "explicit unsupported-format error on precompiled RDNA2 XS2") | XCL's recompiler (24k+ retail shaders) |
| FidelityFX | Reimplements the ABI with a "deterministic bilinear upscale & frame hash" stand-in — never executes the real FSR4 GPU code | The real console DLLs, whose context lifecycle now completes on NVIDIA |

**Applicable to our R9G9B9E5 blocker: nothing directly** — no
R9G9B9E5/format-capability handling exists in ScarlettX because its title never
reaches that code. **Architectural insights**: (a) their honest per-subsystem
status discipline (with explicit unsupported-format errors rather than fake
successes) matches our frontier-lock pattern; (b) their clean-room ABI route
dodges exactly the two problems XCL attacks head-on (console-native shader
lifting; authentic middleware execution) — reinforcing that XCL's remaining
frontier (the console pipeline cache) is the genuinely hard part of this problem
class.

## 6. Final state

- Gated runs: create/destroy test **green** (milestone), dispatch test **green**
  (frontier lock), 4 other FFX tests green.
- Full workspace regression at each commit (268+ tests, 0 failures).
- Commits on `experimental/ffx-functional` (from iteration-2's `490f04f`):
  `0bce8c9` (typed-UAV format translation), `01f3f9e` (context-lifecycle harness),
  `8b12698` (deserializer instrumentation), `416edf1` (dispatch frontier lock).
