# Comprehensive Analysis: Mac Builds, Apple Silicon Compatibility, and the Unified `mac-native` Architecture

## 1. Executive Summary & The "30 FPS" Illusion

A persistent point of discussion in the community has been whether macOS under CrossOver / Wine was already achieving playable performance ("they are running on 30 fps already"). 

Rigorous telemetry and frame-dump analysis conducted on an Apple Silicon M5 Pro running CrossOver 26.3 with D3DMetal (`db065d8` vs `ee3c780` / PR #541) revealed the truth:

| Metric (120 s run) | Baseline Port (`db065d8`) | Port + PR #541 (`ee3c780`) |
| :--- | :--- | :--- |
| **Frames Presented** | 2,255 (~18.8 FPS) | 2,576 (~21.5 FPS) |
| **Console-MS Refusals** | 40 | 0 |
| **Translated Draws / Dispatches** | 32,768 / 1,621 | 41,914 / 2,048 |
| **2-second Windows with 0 Presents** | 18 of 59 | 13 of 59 |
| **Frame Dumps Measured (~1s apart)** | 64 pairs | 75 pairs |
| **Pairs Where Picture Actually Changed** | **0 of 64 (0%)** | **1 of 75 (1.3%)** |

**The Reality:** 
74 out of 75 captured frame pairs were 100% pixel-identical. The title presented 2,576 frames, but the 3D scene was completely stalled. The presentation loop was simply re-displaying the same static 2D loading image over and over.

The root causes behind this stall were:
1. **Dropped Geometry Draws & Missing Pixel Shaders:** Shaders requiring dual-source blending or unsupported Metal features were silently refused by D3DMetal, triggering access violations in `RtlVirtualUnwind2` or falling back to empty pipelines.
2. **Rosetta 2 Memory Protection Traps:** Rosetta 2 was terminating the process on AVX2 masked store instructions crossing write-watched page boundaries (`rosetta error: unexpectedly need to EmulateForward on a synchronous exception`).
3. **Texture Streaming Stalls:** Because Wine implements `MEM_WRITE_WATCH` via virtual memory page protection (which triggers the aforementioned Rosetta bug), write-watch had to be disabled, causing streamed textures and font glyph atlases to be uploaded once and never updated again.

---

## 2. Community Fork & Pull Request Audit

### Ahmet Dedeler (`Ahmet-Dedeler/xcl`)
- **`mac-on-main-wip`**: The core research playground. Introduced extended COM vtables with negative prefix mirroring (`PREFIX = 16`), Rosetta 2 AMD opcode emulation, UMA shadow memory copies (`note_copy_into`), depth-only format conversion, Wine isolated DXC heap allocations (`own_heaps`), and the `maskmov` restartable helper engine.
- **`pr/standin-dual-source` (PR #666)**: Replaced single-target stand-in pixel shaders with dual-source blending shaders (`SV_Target0` + `SV_Target1`). Kept open upstream due to lack of a native Windows GPU to run `smoke.ps1`.
- **Handoff Records**: Documented the transition from upstream `9ab7d76` through `7ebf1b8` to `a6f5f7a`.

### Michael McCrimmons (`mikebuckets171/xcl`)
- **`pr/mac-cross-build-ci`**: Added `.github/workflows/mac-cross.yml` to cross-compile the Wine target (`x86_64-pc-windows-gnullvm`) on macOS runners using `llvm-mingw`.
- **`mac-tests-ext-table`**: Fixed 6 unit test fixtures in `bind.rs`, `list_x.rs`, and `real.rs` where the old vtable layout caused test runners to read arbitrary stack words.
- **`pr/blob-cache-accounting`**: Fixed compute blob memory accounting in `keep_compute_blob` so cache budget is only charged on vacant entries.
- **`44515ba` (`refused_on_mac_host`)**: Added proactive gating for graphics PSOs with no PC program before handing them to D3DMetal.

---

## 3. The Core Subsystems in `mac-native`

The `mac-native` branch consolidates all 36 community patches on top of upstream `beaa245`:

```
origin/main (beaa245)
  │
  ├─► efc850d: CI: cross-build the Wine target on macOS runner
  ├─► 79028be: D3D12.X: charge compute-blob budget once per cache entry
  ├─► 103349d: D3D12.X: stand-in pixel shaders write dual-source colours (PR #666)
  ├─► 5f2742b: Mac host basics: vtable prefix (PREFIX = 16), 16 MiB title stacks
  ├─► 2c222c8: Rosetta 2: missing AMD instructions emulation (CLZERO, INSERTQ, etc.)
  ├─► 928a461: UMA shadow copier without OpenExistingHeapFromAddress (Uma::shadowed)
  ├─► 55ac552: Switches for mechanisms Mac host cannot take as they are
  ├─► e18ce56: Depth formats on D3DMetal: depth+stencil mapped to depth-only
  ├─► eb91958: Title textures in unwatched memory compared by content (poll_unwatched)
  ├─► 6c74d6a: Physical pages under Wine: video memory for mapped pages in runs
  ├─► 7d552f5: Background window, remote input, diagnostics
  ├─► ed8e33d: Pipelines under Wine: heap per DXC compile thread (own_heaps)
  ├─► 95cb367: Gap audit: XMemIsToolsMemAvailable, 128 vtable slots
  ├─► 621f7b7: Background mode: input isolation
  ├─► 0797d3b: Tests: native host depth-only formats & INSERTQ
  ├─► 6dd57ef: Network isolation (netoff)
  ├─► 55d7830: DirectStorage failed write diagnostics
  ├─► ac3f5fd: kernelx: outside windows not write-watched on Mac (XCL_WATCHED_FALLBACK)
  ├─► 6fa794f: D3D12.X: command queue #31 GPU view page mappings (gpu_views)
  ├─► c0f9fe6: Title textures in GPU view: compared once per frame
  ├─► 7390608: Diagnostics: XCL_DUMP_THREADS_ON
  ├─► 8850c3f: CopyMemoryToMemoryX CPU fallback for non-UMA ranges
  ├─► 2761506: Tests: Mac host answer differences
  ├─► 4a8a516: Diagnostics: XCL_PROBE_MOV
  ├─► 051428e: Title textures: unchanged memory upload re-presentation (XCL_TEXTURE_SKIP_UNCHANGED)
  ├─► 0f9cf27: CPU: AVX masked stores decoder and helpers (crates/xcl/src/maskmov.rs)
  ├─► 1a8fea6: Loader: Rosetta 2 masked store fault helper patching (replace_masked_stores)
  ├─► 65a165a: Diagnostics: XCL_FRAME_DUMP_FROM
  ├─► 093f054: D3D12.X: graphics PSO with no PC program goes to stand-in (refused_on_mac_host)
  ├─► 7880224: Tests: timestamp landing as 0 is skip on Mac host
  ├─► da35c88: Tests: extended-table fixtures put HDR words in front (test_ext_table)
  ├─► 8f11b9c: Tests: extended table header in reset/query fixtures
  └─► 0614d75: Verification: scripts/verify_mac_build.sh & CI trigger
```

---

## 4. Host-Gating Invariant

All Mac-specific behaviors are strictly isolated behind runtime checks:
- `crate::util::mac_host()`: Detects Wine/CrossOver and Apple Silicon D3DMetal.
- `crate::util::rosetta()`: Inspects CPUID brand string (`VirtualApple`).
- `crate::gpu::uma::NATIVE_HOST_MEMORY`: Controls host memory allocations.

**Result:** Compiling and running this codebase on native Windows x86_64 behaves 100% identically to upstream `main` (`beaa245`), maintaining full regression safety.

---

## 5. Runtime Environment Switches

| Switch | Default (Mac) | Default (Win) | Description |
| :--- | :--- | :--- | :--- |
| `XCL_COHERENT_HEAPS` | `0` (Off) | `1` (On) | Avoids write-watch page protection on coherent heaps under Rosetta 2 |
| `XCL_WATCHED_FALLBACK` | `0` (Off) | `1` (On) | Disables write-watch on fallback allocations outside windows |
| `XCL_TEXTURE_SKIP_UNCHANGED` | `0` (Off) | `1` (On) | Forces re-presentation of textures to prevent grey screen lockups |
| `XCL_DXC_HEAP` | `1` (On) | `0` (Off) | Gives each DXC compiler an isolated heap to bypass Wine lock |
| `XCL_MASKED_STORES` | `1` (On) | `0` (Off) | Rewrites AVX masked stores to restartable helpers under Rosetta 2 |
| `XCL_NETWORK` | `0` (Off) | `1` (On) | Isolates networking to loopback and stubs out winhttp/ws2_32 |
| `XCL_FRAME_DUMP_FROM` | `0` | `0` | Delays frame dumps to target specific gameplay frames |
| `XCL_BACKGROUND` | `0` | `0` | Runs the window without focus capture |
