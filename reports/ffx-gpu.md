# FidelityFX GPU compatibility — technical report (iteration 2)

Branch: `experimental/ffx-gpu-compat` (from `experimental/compat-improvements` @
`12fdc25`; upstream base `18cb4004008fbf34e43926e94f0e955faed5b010`).
Environment: portable MSVC 14.52 + Rust 1.100 + SDK DXC (via `XCL_DXC_DIR`),
GPU: NVIDIA GeForce RTX 4060 Ti. Artifacts: officially distributed console
FidelityFX DLLs (AMD-signed, microsoft/Xbox-GDK-Samples `Kits/FidelityFX/signedbin`),
supplied via `XCL_SAMPLE_FFX` — never committed.

## 0. Independent code review of iteration 1 (before extending)

Reviewed: `2395f12` (PIXEvt stub module) and `12fdc25` (FFX harness), against the
263→267-test delta and the surrounding XCL architecture.

| Area | Finding | Status |
|---|---|---|
| PIXEvt stub routing safety | The `get_proc_address` fallback triggers **only** for pseudo-handles of DLLs routed `Route::Stub`; real HLE modules win first (`module_by_name` / `hle_for` are checked before the fallback), user aliases override routes and are respected, and *unknown* DLLs still return NULL (optional-API probing preserved) | sound |
| Stub return values | `stub_for` uses the db default; for `pixevt.dll` it is 0 — correct for pointer/BOOL-returning PIX event functions (`E_NOTIMPL` would be a bogus non-null pointer) | sound |
| Ordinal imports through the new path | `stub_for → fname(what)` handles `ImportName::Ordinal` (naming `#N`, `ordinal_name` maps consulted in the HLE path); behavior matches the pre-existing static-import stub path | consistent |
| Lifetimes | Stub thunks + `StubInfo` are intentionally process-lifetime (`Box::leak`), matching the existing `make_stub` design; no new unsafe beyond test-side `transmute`s that mirror upstream test style | sound |
| FFX dynamic module loading | The loader's lazy `LoadLibraryA` of the upscaler is served through XCL's resolver search dirs (and the game VFS in real runs) — verified live in iteration 1 (relocated mapping + TLS carrier + DllMain in the test log) | verified |
| 263 → 267 delta | exactly +1 always-on unit test (pixevt) +3 opt-in FFX tests; the only behavior change (`get_proc_address` on stub-routed handles) had no pre-existing coverage to regress (no stub route existed before) | confirmed by full-suite green |

No defect found that warrants a change; one design note recorded: if a future stub-routed
console DLL has exports whose NULL return is *unsafe* (i.e. the caller dereferences
unconditionally), the db must carry a per-function value — the same rule the db already
documents for HRESULT-returning GDK APIs.

## 1. Prerequisites for `ffxCreateContext` (established before implementation)

From the public ffx-api headers shipped in the same kit (`Kits/AMDTK/ffx-api/include`),
cross-checked against the retail-DLL probe already in `console_dlls_e2e.rs`:

- Signature: `ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc,
  const ffxAllocationCallbacks* memCb)` — non-zero return = error code.
- Descriptor chain: `ffxCreateContextDescUpscale` (type `0x00010000`) → `pNext` →
  backend desc (type `0x01000002`, XCL's D3D12.X device pointer). Field offsets
  (type@0, pNext@8, flags@16, maxRenderSize@20, maxUpscaleSize@28, fpMessage@36)
  are byte-identical between the public header and the struct the retail probe used.
- Return codes (ffx_api.h): 0 OK; 1 ERROR; 2 UNKNOWN_DESCTYPE; 3 RUNTIME_ERROR;
  4 NO_PROVIDER; 5 MEMORY; 6 PARAMETER — used for readable assertions.
- Provider machinery available for diagnostics: `ffxQueryDescGetVersions` (type 4)
  enumerates provider version ids/names for a create-desc type on a device;
  `ffxQueryGetProviderVersion` (type 6) reports the version a context actually runs
  (the MS GDK sample shows exactly this in its UI); `ffxOverrideVersion` (type 5)
  chains via pNext if a specific version must be forced.
- Flags used first: the retail-proven set `HDR | DEPTH_INVERTED | AUTO_EXPOSURE`
  (= `1 | 1<<3 | 1<<5`) at 1280x720 → 1920x1080 — the configuration the retail probe
  proved and the one the game's package used.
- Console-side dependencies (from the binary's imports + upstream's reverse engineering):
  `d3d12_xs.dll` device methods (routed to XCL's D3D12.X HLE), `xmem.dll` allocator
  (kernelx cross lookup), `PIXEvt.dll` (stub since iteration 1), console descriptor
  heaps and command lists (XCL `console_heap`/`console_list`), DXC for any translated
  pipeline (SDK DXC via `XCL_DXC_DIR`).

## 2. `ffxCreateContext` on the official console DLL (NVIDIA RTX 4060 Ti)

### 2.1 Verified before any context attempt

- XCL's **real backend initializes on the RTX 4060 Ti**: "backend 'real' – NVIDIA
  GeForce RTX 4060 Ti (7949 MiB VRAM)", UMA window opened as GPU heaps, console device
  created with `D3D12XboxCreateDevice` → `hr 0x00000000`. **[Verified]**
- The official upscaler maps, installs TLS, runs DllMain in this flow unchanged from
  iteration 1. **[Verified]**

### 2.2 Finding: provider enumeration is a loader-only operation

The first run attempted `ffxQuery(NULL, ffxQueryDescGetVersions{device})` on the
*upscaler* before creating a context (to enumerate available FSR versions). The call
**never returns**:

```
xcl: unhandled exception 0xc0000005 in amd_fidelityfx_upscaler_xbox.dll+0x8055
      (read at address 0x9) rax=0x1 rcx=<stack> rdx=0x10000 …
```

Diagnosis: the provider DLL's `ffxQuery` with a **null context** walks a global
provider registry that is not initialized in that module — `rdx` still holding the
create-desc type `0x10000` shows the crash inside its descriptor dispatch, before any
console device method is touched (the `XCL_D3D12_TRACE` log is empty at that point).
Microsoft's GDK sample performs exactly this enumeration through the **loader's**
function table (`ffxLoadFunctions`), never on the provider DLL directly; the retail
probe never exercised it either. **[Verified — do not call `ffxQuery(NULL, …)` on the
provider DLL.]** The test harness now documents this and goes straight to
`ffxCreateContext`. (XCL's crash handler cleanly named the faulting module+offset and
the process exited with STATUS_ACCESS_VIOLATION — the diagnostic path works.)

*(context-creation results follow)*

### 2.3 Both create paths fail fast with `FFX_API_RETURN_ERROR` (code 1)

With the retail-proven chain (upscale desc → backend desc, no version desc) — via the
**loader** (authentic flow) and directly via the upscaler — both return `code 1 (ERROR)`
in <1 s, with the message callback never invoked and the `XCL_D3D12_TRACE` log **empty**:
the rejection happens in descriptor validation, before any console device method, any
resource, or any provider-internal load. **[Verified]**

Static analysis of the two official binaries (export stubs disassembled locally) explains
the architecture and the likely gap:

- **Loader `ffxCreateContext`** (RVA 0x1250): null-checks → takes
  `desc.type & 0x00ff0000` as the **effect id** → calls an internal
  *ensure-provider-loaded(effect)* helper (the same one the Fsr4Lite accessors use) →
  for effect `0x00010000` (upscale) jumps to the provider-function check →
  `call provider(out, desc, memCb)` → on success wraps `{provider-name, provider-ctx}`
  and returns 0. So the loader is a thin delegating router; the provider decides.
- **Upscaler `ffxCreateContext`** (RVA 0x11F0): walks the descriptor chain for **version
  descriptors**: type `4` (id @ +0x10), type `0x00010009` (id @ +0x18 — the
  effect-specific "UpscaleVersion" matching Microsoft's sample description
  *"with a linked ffx::CreateContextDescUpscaleVersion declaring the API version the
  application was built against"*), type `0x02000002` (id @ +0x18) — and separately for
  type `5` `ffxOverrideVersion` (id @ +0x10). It then runs an internal
  provider/version lookup before touching the device.
- The provider's version-table init (the one triggered by the first Fsr4Lite accessor
  call) registers the upscaler's version id **`0xf5a5ca1e01001002`** alongside the
  effect id `0x00010000` — read straight from its machine code.

**Conclusion so far**: the official FSR 4.1.2 pair requires descriptor information the
retail-era probe never supplied. Comparing against **Microsoft's own sample**
(`Samples/Graphics/AMD_FSR/AMD_FSR.cpp`, `CreateOrUpdateFsrContext`) and the
**current FidelityFX headers shipped beside the DLLs** (`Kits/FidelityFX/api/include`,
a newer copy than `Kits/AMDTK/ffx-api`) gives the exact Xbox chain:

```
ffx::CreateContextDescUpscale        type 0x00010000
  -> ffx::CreateBackendXboxDesc      type FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_XBOX
                                     = MAKE_BACKEND_SUB_ID(FFX_API_BACKEND_ID_XBOX=0x02000000, 0x02)
                                     = 0x02000002, device @ +0x10
  -> ffx::CreateContextDescUpscaleVersion
                                     type FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION
                                     = MAKE_EFFECT_SUB_ID(0x00010000, 0x0b) = 0x0001000b,
                                     u32 version = FFX_UPSCALER_VERSION
                                     = (4<<22)|(1<<12)|2 = 0x01001002 @ +0x10
```

The retail probe's `0x0100_0002` is the **old DX12 backend id** (`BACKEND_ID_DX12` was
0x01000000 in that generation, `0x00000000` now; VK took 0x01000000). Two independent
deviations from the sample's Xbox flow — wrong backend id, missing version declaration —
either of which explains a bare pre-device `ERROR`. Also corrected by the sample: the
context-less `ffxQueryDescGetVersions` is answered **by the loader** (the sample's
`FfxQueryGlobal`, used to locate FSR3's provider id) — the provider DLL's global path is
the one that crashed; and a type-5 `ffxCreateContextDescOverrideVersion` chains a
version id obtained from that enumeration (my guessed id explains its `NO_PROVIDER`).
Reassuringly, the binary's own registration table (decoded from machine code) stores
`0xf5a5ca1e01001002` — whose **low 32 bits are exactly `FFX_UPSCALER_VERSION
(4.1.2) = 0x01001002`**: the descriptor format the sample declares is the one this DLL
was built to accept.

The current run attempts the sample-exact chain first, then isolates each change
(backend only / retail chain control), via the loader and directly. *(outcome below)*

### 2.4 The sample-exact chain is accepted — create reaches real GPU work

With the corrected chain (BACKEND_XBOX + UpscaleVersion 4.1.2) the official upscaler's
`ffxCreateContext` **passes all validation and starts creating real GPU state through
XCL's console D3D12.X device** — the `XCL_D3D12_TRACE` log shows the console middleware
driving the PC driver:

```
ID3D12Device14::AddRef (#1)
CreateDescriptorHeap (#14)  × 4
CreateCommittedResource3 (#76): 1920x1080, format 28 (R8G8B8A8_UNORM), flags 0x5
CreateCommittedResource3 (#76): 1920x1080, format 10 (R16G16B16A16_FLOAT)   × 2
   … each followed by SetName / GetDesc / GetCPUDescriptorHandleForHeapStart /
     GetDescriptorHandleIncrementSize / CreateShaderResourceView /
     CreateUnorderedAccessView × 2 — the FSR4 internal textures and views
CreateCommittedResource3 (#76): 1920x1080, format 67 (R9G9B9E5_SHAREDEXP),
   flags 0x4 (ALLOW_UNORDERED_ACCESS), state console-bit|UAV → E_INVALIDARG
```

**[Verified]** — this alone advances XCL's console-middleware support materially: the
official FSR4 binary accepts XCL's emulated device far enough to allocate heaps and
three internal textures through the console vtable (slot 76) on the RTX 4060 Ti.

### 2.5 Precise blocker: console resource-state bits at the committed-resource boundary

The 4th internal resource fails: the console passes an initial state of
**`0x400000008`** — the PC `UAV` state (0x8) OR a **console-only qualifier bit**
(0x400000000) that does not exist in the 32-bit PC `D3D12_RESOURCE_STATES` enum.
XCL's console-desc converter already fixes console `Layout` fields (0x101 → UNKNOWN)
at exactly this boundary but passed the state through raw, so the PC driver returned
`E_INVALIDARG`; the console DLL then raises a C++ exception, which fails fast through
the Rust test process (`STATUS_STACK_BUFFER_OVERRUN`) — no fake success, no hang.

Targeted fix (this branch): mask the state to the PC range
(`state & 0xFFFF_FFFF`) in `dev_create_committed_resource_x` — the same
console→PC translation point as the Layout fix, same rationale: console-only
qualifiers belong to the console's memory model, which XCL deliberately reinterprets.

### 2.6 Outcome of the state-mask fix, and the true blocker

With the state masked, the 4th texture's creation is rejected **by the format**: the
console desc asks for `ALLOW_UNORDERED_ACCESS` on **DXGI 67 = `R9G9B9E5_SHAREDEXP`**
— a format the console's RDNA2 accepts as a typed UAV, but the PC typed-UAV format
set does not. Driver-verified on the RTX 4060 Ti:

```
CheckFormatSupport(R8G8B8A8_UNORM = 28):       shader_load true, typed_uav true
CheckFormatSupport(R16G16B16A16_FLOAT = 10):   shader_load true, typed_uav true
CheckFormatSupport(R11G11B10_FLOAT = 26):      shader_load true, typed_uav true
CheckFormatSupport(R9G9B9E5_SHAREDEXP = 67):   shader_load true, typed_uav FALSE
```

(All four console values are *valid PC DXGI enum entries* — the console enum matches
the PC one; the difference is capability, not numbering. Resource formats in the
accepted creations: 28 = R8G8B8A8_UNORM, 10 = R16G16B16A16_FLOAT ×2.)

The upscaler then **fail-fasts (0xC0000409)** inside its own resource-creation error
path — an abort that crosses Rust frames and cannot be caught, so the reproduction
runs in a child process and the parent asserts the exact signature. **[Verified,
reproducible: `XCL_SAMPLE_FFX` + `XCL_FFX_GPU=1`, test
`ffx_console_upscaler_creates_and_destroys_a_context_on_the_pc_device`.]**

### 2.7 Emulation strategy for the blocker (next development target)

The R9G9B9E5 texture is 32 bits/pixel like R32. The PC-legal emulation is a
**same-width raw substitution**: create the texture as `R32_TYPELESS` and give the
console's UAV request an `R32_UINT` view, keeping SRV reads in the native format
(R9G9B9E5 SRVs *are* PC-supported). Bit-correctness then depends on what the console
shaders write: RDNA2 `image_store` to a typed descriptor either
(a) already writes packed u32s — then raw views are bit-identical, and the
substitution alone suffices — or
(b) writes unpacked components — then XCL's recompiler must add software RGBE
packing for image stores of this format, the same way it already packs 78 buffer
formats for `buffer_store_format_*`.
Deciding (a) vs (b) requires disassembling the FSR4 pipeline shaders that target this
texture (their console containers are embedded in the DLL and extractable with
`xcl disasm`) — the bounded next step before implementing the substitution in
`dev_create_committed_resource_x` + the UAV-view override.

## 3. Dispatch / GPU execution — **Not Tested**

Blocked upstream of dispatch: context creation must complete first. The evidence so
far (heaps, textures, views) is all pre-dispatch.

## 4. NVIDIA-specific findings

| Observation | Value |
|---|---|
| XCL real backend | works: "backend 'real' — NVIDIA GeForce RTX 4060 Ti (7949 MiB VRAM)" |
| UMA `OpenExistingHeapFromAddress` | works (2048 MB window) |
| Typed-UAV support for the console FSR4 format set | RGBA8 ✓, RGBA16F ✓, R11G11B10F ✓, **R9G9B9E5 ✗** — the blocker |
| Console fail-fast on creation failure | 0xC0000409, not catchable in Rust → child-process reproduction (as XCL's own launcher does for games) |
| `StandardSwizzle64KBSupported` | 0 — same as the author's RDNA4 |

## 5. Milestone classification (iteration 2)

| Milestone | Status |
|---|---|
| Official console DLLs: load/init/export execution | **Verified** (iterations 1-2, unchanged) |
| `ffxCreateContext` descriptor acceptance (official 4.1.2 chain) | **Verified** (iteration 2) |
| Console typed-UAV format capabilities (R9G9B9E5 + UAV) | **Fixed & Verified** (iteration 3, `0bce8c9`) — flags + matching initial state translated at the committed-resource boundary; creation of the 4th texture succeeds |
| **Full FSR4 context lifecycle (create → verify → destroy)** | **VERIFIED (iteration 3)** — the target milestone: `ffxCreateContext` OK via the loader, context structure verified (wrapper + provider C++ object), `ffxDestroyContext` OK via the provider, on the RTX 4060 Ti |
| Context-bound queries (`ffxQuery` via loader or provider) | **Blocked** — both dispatch paths fault on uninitialised registries (loader+0x16a6 wrapper walk; upscaler+0x1b9a .data slot = -1); reproducible behind `XCL_FFX_QUERY=1` |
| FSR4 root signature capture through XCL's shader path | **Verified** — `[SHADER] rootsig cee296d8158a6d15: 304 B, parts [RTS0]` |
| **FSR4 dispatch / GPU execution** | **The march is inside execution (iteration 5)**: the iteration-4 "NVIDIA size wall" was a misdiagnosis — the real cause is the adapter's wave range (RTX 4060 Ti: **32..32**, so [WaveSize(64)] shaders are rejected outright; the iGPU's 64..64 inverts it; a 949 KB DXIL is accepted once the wave size is legal). XCL now translates wave64 console programs to the **rendezvous form** ([WaveSize(32)] + groupshared ballot/cross-lane exchanges — no barrier intrinsics, which deadlock behind the per-lane publish branches; verified on the GPU with a synthetic console CS through the full retail pipeline), with the exchange aliased into the LDS tail when a pass uses console LDS and every spin bounded/self-healing. The **full pass set deserializes and is accepted (30+ pipelines, 576 B..41 KB RDNA2)**, and **`ffxDispatch` returns OK — 26 console dispatches recorded and executed**. The mid-frame fault was bisected to dispatch 14 (grid 507,1 — the 26 KB wave64 pass) and its cause identified: the console's read-write t# stores index the PC UAV slot at +1, which the iteration-3 flag-drop made impossible to create — the store hit an uninitialized descriptor. Fixed (`10faf5e`): the texture is created as **R32_TYPELESS with the flags kept**, views translate to R32_UINT, the SRV creation also creates the missing UAV, and the recompiler's format-132 image path packs/unpacks 9-9-9-E5 (self-inverse). Open: the verifying march's verdict, the real pass geometry ((16,16)-shaped, evidenced by the (120,68) grid over 1920×1088) vs the deserializer's (256,1,1) guess, and the output-correctness verification when the frame executes through. |

## 3. Dispatch / GPU execution

*(filled after the context milestone)*

## 4. NVIDIA-specific findings

*(filled)*
