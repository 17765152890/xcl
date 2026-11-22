# Verification report — iteration 4 (`experimental/ffx-functional`)

GPU: NVIDIA GeForce RTX 4060 Ti. All results are real runs. The iteration started from the
rebased branch (upstream `main` @ `e511b94` + our 9 commits, no conflicts) and works on the
console driver's serialized pipeline format — the exact frontier iteration 3 locked.

## 1. Upstream comparison and rebase

Upstream moved substantially (8 commits since our iteration-3 base `18cb400`): a real retail
title (XDefiant) now runs — 3,443 pipelines from 966 translated console shaders — via the
normal FSSM-container path; `DeserializePipelineStateX` remains **unimplemented upstream**
(our frontier is untouched). Directly reusable upstream work for our dispatch path:
`bind.rs` (console compute root arguments "the console way", measured with the console's
compiler), `title_list.rs` (console-shaped command lists), `console_heap.rs` (32-B
descriptor slots — from the retail FSR4 probe), and the retail-FSR4 notes in `HANDOFF.md`.

Rebase: clean, no conflicts; 269→280 tests on the new base (see §6 for the one failing
upstream check).

## 2. The serialized pipeline format — decoded (verified offline + at the call)

The runtime blob passed to `DeserializePipelineStateX` (dumped completely via the new
`XCL_FFX_BLOB_DIR` instrumentation, commit `4d3ce5e`):

```
qword[0]    a runtime session object (heap pointers - not serialised data)
qword[1..]  (ptr, size) pairs, four sections:
  A    0x1c B   pipeline header  [0x4005cf00, 0x6c090102, .., variant fields]
  H    0x14 B   [hash32, 3, 0]
  C    code     raw RDNA2 - VERIFIED: s_code_end (0xBF9F0000) × 53, s_endpgm × 1,
                byte-identical to a block embedded in the console DLL
  M    0x278 B  driver metadata (below)
```

The console DLL embeds the whole cache statically: **927 `FXCP` blocks** spanning
0x670b0..0xa48b20, each `[version][FXCP][1][2][total]` + (offset,size) section pairs:
a 0x130-B DXBC/RTS0 root signature, A, H, the code, M. (FXCP blocks are per-pipeline;
the entry prefix references device methods 0x42/0x43 — Serialize/Deserialize slots —
with a per-block GUID.) The runtime selected **block #200** for `FFX_FSR4_PASS_0`
(hash 0x3496c3bc, code 0xec0 B at DLL 0x2e5040, metadata at 0x2e5f00).

Metadata section structure (block #200):

```
[0x403, meta_size - 0x14, code_size_in_dwords, 0, 0, variant]
'A' records: [0x41, kind, packed-array byte offset, root index, dword count, id, -1, 0]
  — the console driver's user-data (SGPR) load commands
'B' records: [0x42, 0, id, count, 0, 0, -1, 0]  — binding records
'C' record:  [0x43, 16-byte code-derived digest]  — pipeline identity
```

`metadata[2] × 4 == code size` holds for every block (verified across all 927), and the
metadata size correlates with the binding layout (21 variants).

## 3. Cross-referencing with the PC twin (ground truth corpus)

The PC twin (`amd_fidelityfx_upscaler_dx12.dll`, 57 MB) embeds **1,778 DXIL compute
kernels** with PSV0 numthreads (verified against a locally compiled
`[numthreads(8,4,2)]` reference: **PSV0 r[36..48]**):

```
(256,1,1): 661    (32,1,1): 399    (8,8,1): 363
(64,1,1):  343     (64,2,1): 6      (8,4,1): 6
```

No cross-platform hash match exists (the console's two hashes are code-derived, partially
byte-reversed fragments; the PC HASH parts differ) — kernel-level PC↔console matching is
fuzzy, but the *geometry distribution* constrains the console threads: block #200's code
uses v0 with masks ≥ 0x1F and a v0−64 decomposition path ⇒ group > 64 threads ⇒ the only
matching PC geometries are (256,1,1) or (64,2,1).

## 4. Block #200's compute ABI — read from its code

Disassembled with XCL's own RDNA2 decoder (`xcl disasm`). The prologue matches the retail
FSR4 dispatch-pass ABI the upstream author measured from the retail pack's ACCC shader:

```
v_lshl_or_b32 v6, s14, 4, v5      ; s14/s15 = TGID x/y (group origins)
v_lshl_add_u32 v9, s15, 4, v4
s_mov_b32 s10, s5 ; s_mov_b32 s11, 0      ; s5 = the SRV descriptor table
s_load_dwordx4 s[16:19], s[10:11], 0x2a0  ; SRV #21 (0x2a0/0x20) — retail note: "SRV×21"
v_add_nc_u32 v4, -64, v0 ; v_cmp_gt_u32 64, v0   ; multi-wave group (wave64)
```

⇒ `CsInterface { threads: [256,1,1] (initial hypothesis, env-tunable), user_sgprs: 14,
tgid: [x,y on], tg_size: off, tid_components: 1 }` — every field but threads verified in
the code; threads to be validated against the console's dispatch record once the pipeline
executes (grid × threads must cover the pass's work exactly).

## 5. The deserializer (this iteration's implementation)

`real::deserialize_compute_x`: parses the four sections (validated: exactly 4 sections,
metadata header `0x403`, `metadata[2]×4 == code size`, s_endpgm/s_code_end tail),
translates the raw RDNA2 through the existing recompiler with the synthesized
`Mode::Compute`, compiles DXIL, creates the PC compute PSO under XCL's ABI root
signature (`abi_rs_for`), and returns a console-shaped wrapped PSO
(`console_list::wrap_pso`) exactly like the retail `CreateComputePipelineState` path —
so upstream's `bind` machinery maps the console root arguments at dispatch.
`XCL_FFX_THREADS=x,y,z` overrides the geometry for the validation loop.

## 6. Side findings

- **VFS drive-mount collision (fixed, commit `35c19f8`)**: upstream's new default mount
  `"D:" = "{game}"` translated the host's own real D: paths (our workspace is on D:) to
  doubled nonexistent paths — the e2e title could not load (os error 3), then the TLS
  carrier failed (error 126). Fixed by skipping mount translation for paths under host
  roots (the mount target and the XCL data folder); regression test included; the e2e
  title runs to completion again.
- **Upstream e2e check fails on this environment (not ours)**: the new testtitle check
  "console device methods (placed title memory, DMA queue + list, page pools)" fails with
  "the DMA copy did not read the title's memory" on the RTX 4060 Ti. None of our commits
  touch lists/queues/placed resources; the check is new upstream and unverified on
  NVIDIA. Exact signature recorded; candidate cause: the console DMA list's vtable slot
  #11 (CopyBufferRegion) over a placed buffer on title memory via the video-memory arena.

## 7. Final verified state (iteration 4)

The **target milestone is achieved**: `FFX_FSR4_PASS_0` — and thirteen more passes —
deserialize into correctly constructed, driver-accepted PC compute pipelines:

```
14 × deserialised console compute pipeline created:
  PASS_0:  3544 B RDNA2 → 33 KB DXIL   (RTX 4060 Ti: S_OK)
  ...     576..21688 B passes ...
  PASS_12: 13784 B → 320 KB DXIL       (S_OK)
  PASS_14: 26048 B → 602 KB DXIL       (E_INVALIDARG - the driver wall)
```

Every pipeline is authentic console code: the sections come from the DLL's embedded FXCP
block, the translator reports no unsupported instructions, and the driver creates real
PSOs from the compiled DXIL. The deserializer's once-multi-minute DXC compiles are
disk-cached (`%LOCALAPPDATA%\xcl\dxil`); the gated dispatch test now reaches pass 14 in
~8 s.

**The frontier after iteration 4** (reproducible, machine-isolated):

- The RTX 4060 Ti rejects pass 14's PSO (E_INVALIDARG, silent, debug layer included).
- Direct bisect (`deserialized_fs_dxil_driver_acceptance`): 320 KB DXIL → S_OK,
  602 KB DXIL → E_INVALIDARG on the RTX; **the same 602 KB DXIL → S_OK on the Radeon
  iGPU** (21 s compile).
- DXC cannot shrink it: -O0 → 1.69 MB, -O1/-O2/-O3 → all ≈602 KB (the size is the
  translated instruction count, not optimizer output).
- The iGPU rejects the earlier passes instead — pass 0's DXIL requires
  **"Resource/Sampler descriptor heap indexing"** (the console's descriptor tables,
  emulated bindless), which the iGPU's driver does not support.

So the vendor matrix of the FSR4 march on this machine:

| requirement | RTX 4060 Ti | Radeon iGPU | (upstream's RDNA4) |
|---|---|---|---|
| descriptor-heap-indexed DXIL (passes 0-13) | ✓ | ✗ | presumably ✓ |
| 602 KB DXIL (pass 14) | ✗ | ✓ | presumably ✓ |

Both GPUs together cover the whole set; each alone blocks a different half. Crossing
pass 14 on the RTX requires smaller translated shaders (recompiler emission work —
upstream territory), not more plumbing.

**Not reached**: genuine dispatch execution (the stretch). It needs the pass set to
complete `CreateModel`; pass 14 stops the march on the RTX, and the early table-binding
passes stop it on the iGPU. The pass-geometry validation (grid × threads) is wired into
the harness for the moment the march gets through; `XCL_FFX_THREADS` overrides the
working hypothesis (256,1,1).

**Emission levers tested for pass 14** (all machine-verified):
- DXC flags: -O0 → 1.69 MB, -O1 → 602.6 KB, -O2 → 602.6 KB, -O3 → 602.9 KB — the DXIL
  size is the translated instruction count, not optimizer output.
- `XCL_RECOMP_NO_STRUCTURED=1` (guarded-block-chain emission): the pass-14 HLSL gets
  *bigger* (1.49 MB vs 1.39 MB) — structuring is not the bloat.
- Remaining lever: reduce the guarded flat-memory-load emission itself (~213 B of HLSL
  per instruction on this kernel vs ~120 on the others) — recompiler work, upstream
  territory.

## 8. Regression state

Full workspace on the rebased+extended branch: 260 unit tests, all integration suites
and the six gated FFX tests green (the dispatch frontier-lock asserts the new signature:
*"VERIFIED FRONTIER (iteration 4): 14 authentic FSR4 console pipelines deserialized,
translated and accepted by the NVIDIA driver"*). One known failure, **not ours**: the new
upstream testtitle check "console device methods (placed title memory, DMA queue + list,
page pools)" — "the DMA copy did not read the title's memory" on the RTX 4060 Ti. It fails
identically without this branch's changes (none touch lists, queues or placed
resources); recorded as an upstream-on-NVIDIA issue with the exact signature.
