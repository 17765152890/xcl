# XCL – work handoff (status as of 2026-11-20, session 5)

Xbox Series X|S → Windows compatibility layer (no CPU emulation), Rust (this repository).
Design: `docs/specs/2026-11-20-xcl-design.md` (sections 5.x, including 5.7a – console shader format, 5.7b – recompiler status). Manual: `README.md`.
Conversation with the user in Polish; everything in the repo (code, comments, messages, docs) in English. Git repo: `main`, pushed to origin; commits only at the user's request.

## Environment
Windows 11, AMD Radeon RX 9070 (RDNA4) + iGPU, Rust 1.96 `x86_64-pc-windows-msvc`, VS 2022, Python 3.12.
Build/test: `.\scripts\test.ps1` (= `cargo build --workspace` + `cargo test --workspace`; e2e needs the test title to be built).
NOTE:
- editing files through PowerShell `Get-Content/Set-Content` corrupts UTF-8 (non-ASCII characters) – use the Write/Edit tools
  or `[IO.File]::WriteAllText(..., UTF8Encoding($false))`;
- in Git Bash a heredoc `<<'EOF'` passed to Python eats backslashes (`\\n` in a Rust string → a real newline character,
  paths `C:\U…` → syntax error) – edit text containing backslashes with the Edit tool;
- `scripts\test.ps1` run in the background with `*>` redirection in PowerShell 5.1 ends in an error (cargo's stderr = an exception under
  `ErrorActionPreference=Stop`) – in the background run it through bash: `cargo build --workspace && cargo test --workspace -- --test-threads=4`.

## Rules from the user / constraints
- Work on the sample files and test: `<samples>\shaders.rpf` (shaders),
  `<samples>\audio.rpf` (audio, 28.5 GB – read it in pieces!). Next to them `rpf-viewer.html` –
  a pack viewer (format of the shader records in RPF9, ISG1/OSG1 signature parser).
- `<samples>\recompiler` – a fragment of AnyPS5 (RDNA → SPIR-V recompiler for PS5) from the user, license
  **GPL-2.0-only** (github.com/boykopovar/AnyPS5). **The user agreed to relicense XCL to GPL-2.0-only (2026-11-20)**
  – done (`LICENSE`, `Cargo.toml`, `THIRD_PARTY.md`), so AnyPS5 logic may be ported with a provenance note in the file.
  Consequences: WinDurango code (LGPL-3.0) only as a reference; dependencies must be GPL-2-compatible (no pure Apache-2.0 / GPL-3).
  Technically AnyPS5 is Vulkan/SPIR-V + the PS5 SRT model – useful parts: T#/S#/V# descriptors, images, interpolation, VS inputs.
- **The user has no time for long compiles/runs (2026-11-20)**: avoid steps that take many minutes
  (e.g. `--compile` on thousands of samples, huge DXC measurements); check on small samples, work fast.

## What works (verified by tests)
- **PE loader** (`loader.rs`, `pe.rs`): mapping, relocations, imports + delay-load, DllMain/TLS callbacks, game DLLs.
- **Static TLS** (`tls.rs`): a generated "carrier DLL"; Win11 findings: the loader allocates a slot only for DLLs with imports
  and ignores `SizeOfZeroFill` (the whole size goes into "raw data").
- **C++ exceptions**: hook on `RtlPcToFileHeader` (retour) + `RtlAddFunctionTable` (`eh.rs`).
- **Resolver** (`resolver.rs`, `hle/mod.rs`): HLE → game DLL → Windows → auto-stub with a log (`stubs.rs`, `data/stubdb.toml`).
- **XGameRuntime offline** (`hle/xgameruntime/`: XTaskQueue, XAsync, XUser, XSystem with XSystemGetDeviceType, XPackage, PLS, save folder).
- **VFS** (`vfs.rs`, `vfs_hooks.rs`): `G:\`, `GAME:`, `T:\`, copy-on-write, whiteouts, KERNELBASE hooks (they also catch the CRT).
- **Launcher/host/report** (`launcher.rs`, `host.rs`, `report.rs`, `inspect.rs`): child process with the game's command line;
  does not pick `xcl.exe` itself; warns when the exe does not import console DLLs; `report.json` (called_stubs, shaders, d3d12, vfs…).
- **RDNA2 decoder** (`gpu/rdna2.rs` + generated `gpu/rdna2_spec.rs` from the AMD Machine-Readable ISA XML, MIT; generator
  `scripts/gen_rdna2.py`; the XML can be downloaded from gpuopen.com/machine-readable-isa): 23 812 instructions from the LLVM tests
  (`tests/data/llvm-amdgpu`) – 100% name/length agreement; `xcl disasm`. `rdna2::branch_target`, `rdna2::all_instructions`.
- **Console ISA extensions** (`gpu/rdna2_xbox.rs`, NEW): `v_mac_f32` (VOP2 0x1F / VOP3 0x11F), `v_madmk_f32` (0x20),
  `v_madak_f32` (0x21), `v_mad_f32` (VOP3 0x141) – kept from gfx10.1, removed in gfx10.3; the decoder takes them only on
  opcodes that are free in the spec. The VALU literal is fetched when ANY SRC0/1/2 field = 255, even one unused by the opcode
  (the compiler hides a hash in `v_nop` this way) – displayed as `v_nop ; lit 0x…`. SDWA/DPP: src0 from the extension dword (was `?249`).
- **Shader capture** (`gpu/shader.rs`): classification DXIL/DXBC/**xbox-native**/unknown container/ELF/raw RDNA2,
  dump to `<data>\shaders\` (+ `*.fssmN.bin`, `*.rdna2.txt`); `report.json → shaders`: `xbox_hw_stages`, `xbox_errors`,
  `rdna2_instructions`, `rdna2_undecoded_dwords`, `rdna2_mnemonics` (histogram sorted in descending order).
- **Console shader format** (`gpu/xbox.rs`, NEW) – worked out on `shaders.rpf` (23 869 shaders, all stages),
  full description in the design §5.7a: DXIL (bitcode `NOBC`) → XDAT (+0x18; offsets from the program header) → XBCH v3 (zlib from +0x18)
  → FSSM v0x101 (code: u64 +0xC0/+0xC8, offset counted from +0x10; stage descriptors `0xC0_0002_AAHH` from +0x100 every 0x88,
  HH = PAL hardware stage: 2 ES, 3 GS, 4 VS, 5 PS, 6 CS, 7 MS/NGG). Code split: segments up to `s_endpgm`, appended to the
  program when connected by a jump (shared "discard" exit in PS, trampolines); constant blocks `s_endpgm | N | N dwords |
  s_endpgm` separately (`data_blocks`). Every step has a fallback (synthetic tests). Result on the whole pack: 24 064 FSSM =
  24 064 programs, 473 data blocks, 0 parse errors, 0 undecoded dwords in 28.97 million instructions, 395 mnemonics.
- **`xcl shaders <file|folder|RPF9 pack> [--top N] [--json]`** (`shaders.rs` + `gpu/pack.rs`, NEW – RPF9 record scanner
  as in the viewer): formats, DXIL and hardware stages, mnemonic histogram, "in how many shaders", implementation order
  (`--json → implementation_order`) and coverage, list of unusual containers (with the record offset); multithreaded, whole pack
  approx. 1.7 s (release). `xcl disasm` for xbox-native: XDAT table, XBCH, stages, FSSM header qwords, `--hex`, `--fssm FILE`.
- **RDNA2 → HLSL recompiler** (`gpu/recomp/`, session 3; description in the design §5.7b): `translate()` → a `switch(pc)` dispatcher over
  blocks (AMD jumps are uniform across the wave → exact without structurization), SGPRs uniform, VGPRs per thread with the EXEC mask,
  masks via `WaveActiveBallot`, cross-lane via `WaveReadLaneAt`. `emit.rs` – semantics: SALU (exact SCC), VALU
  f32/u32/i32 + VOP3 modifiers, SDWA, DPP16/DPP8, `ds_swizzle`, `v_permlane(x)16`, `v_fma_mix*`, cube, VOPC/VOPCX,
  saveexec/wrexec, `s_load`/`s_buffer_load` (V#), MUBUF/MTBUF (raw + 78 GFX10 formats from LLVM `UfmtSymbolicGFX10`,
  DST_SEL), `buffer_store_dword*`, exports; the rest → a comment + the `unsupported` counter. `s_swappc_b64 s[n:n+1], s[n:n+1]`
  = fetch shader call (ignored in kernel mode). **Kernel** mode (compute `[WaveSize(64)]`, registers from/to a buffer,
  flat memory `mem`; program code in `mem` from 0 for constants from `s_getpc`). `dxc.rs` – in-process DXC (dxcompiler.dll from
  the Windows SDK / next to the exe / `XCL_DXC_DIR`). `runner.rs` – D3D12 compute on the high-performance GPU. `rdna2::encode` – an assembler
  for tests (also SDWA/DPP layouts); `rdna2::register_scale` – FIXED disassembler bug: SBASE counts SGPR pairs, SRSRC/SSAMP
  quads (was `s[8:9]` instead of `s[16:17]`). Result: **`VS_BlitGenAmbCube` from the pack runs on the RX 9070 with a correct
  SV_Position**; corpus: 24 064 programs, 0 translation errors, **7 442 (30.9%) in full**, DXC compiles 500/500 samples.
  `xcl shaders … --recomp [--compile N]` (missing instructions: in how many programs + "the only missing one").
  Note: DXC divides floats approximately → the ends of the UNORM/SNORM range are computed exactly, separately.
- **XAudio2** (`hle/xaudio2.rs`, NEW): HLE module `xaudio2_9.dll` (+ `xaudio2_9redist.dll`) → system XAudio2 2.9
  (loaded dynamically). Create: flags outside 0x2000|0x8000|0x10000 are dropped (NOTE: the old note "0x18002, processor=0"
  was wrong – according to xaudio2.h 0x2 is a voice flag, processor = XAUDIO2_DEFAULT_PROCESSOR = 1). Vtable patching IN PLACE
  (originals per (vtable, slot)): IXAudio2 #5 CreateSourceVoice (XMA/XMA2 → PCM16; another format rejected by Windows,
  e.g. Opus 0x704F → also PCM16), #7 CreateMasteringVoice (console device id → default; `CO_E_NOTINITIALIZED` →
  CoInitializeEx(MTA) + retry – discovered in e2e; then 0/0 + the GameEffects category), SourceVoice #21 SubmitSourceBuffer
  (silence of length PlayBegin+PlayLength or an estimated one; context/flags/loops preserved), #18 DestroyVoice.
  Refactor: `HleModule.system_fallback: bool` → `fallback: &[&str]` (`hle::SELF` = "@self") + `aliases_are_system()`;
  the remaining XAudio2 exports (X3DAudio*, CreateFX…) → system xaudio2_9.dll. `report.json → audio`.
- **Real D3D12 backend** (`hle/d3d12x/real.rs`): real driver objects with an extended vtable; layout from `d3d12.h`
  (`scripts/gen_d3d12_layout.py` → `hle/d3d12_layout.rs`). Finding: driver vtables have internal virtuals behind the public ones
  (Device 125 vs 104) → **guards** (`hle/d3d12x/guard.rs`, machine code: a call from the game image = console method → log+0,
  otherwise → the original). Overrides: QI (unknown IIDs), all Create* (adoption), PSO/root signature with a fallback
  (replacement shaders when the driver rejects the console ones). Fallback `null.rs` when there is no GPU / `graphics="null"`.
- **Presentation** (`hle/d3d12x/present.rs`): XCL window + swap chain; back buffer detected via RTV/OMSetRenderTargets/
  ExecuteCommandLists; a frame per call of the console queue method (probably PresentX). UNTESTED on a game.
- Test title `tests/testtitle` (+`tests/testgamedll`): 26 checks, including a triangle rendered on the RX 9070 through our
  `d3d12_xs` and pixel readback; session 2 added: a shader in the console format (XDAT/XBCH/FSSM with `v_mac_f32`) at
  PSO creation → `report.json` (xbox-native, histogram) and XAudio2 (console flags/device, an XMA2 voice plays silence – a 0.1 s buffer
  ends after ~95 ms, the PCM voice unchanged, callbacks with contexts; without an audio device the voice part is skipped).
  Test status (end of session 3): 75 unit tests (including 11 recompiler tests executed on the GPU) + e2e (28 checks
  in the title) – all PASSED. Tests on real data (`shaders::tests::retail_pack_sample`,
  `gpu::recomp::tests::gpu_retail_vs_blit_gen_amb_cube`) work only with
  `XCL_SAMPLE_RPF=<samples>\shaders.rpf` (skipped without the variable) – PASSED.

## Recompiler – session 4 (2026-11-20)
- **Corpus coverage: 99.1% of programs translate in full** (23 846 / 24 064; was 75%): MS 9 282 / 9 282 (100%),
  PS 99.9%, VS 92.0% (only the 195 ES merged with GS are missing), CS/kernel 96.4%. DXC: 300/300 samples OK.
- New in `emit.rs`: MIMG A16/D16 (address layout as in AnyPS5 `componentShape`), `image_get_lod`, `_cl` (LOD clamp);
  `buffer_store_format_*` (packing of 78 formats, `st_bytes` with atomics for partial dwords), `buffer_atomic_*`;
  global/flat load/store; LDS (`ds_read/write*`, `read2/write2(st64)`, atomics → `groupshared`, 32 KB = the D3D12 limit);
  `ds_bpermute`; 16-bit VALU (op_sel, preserving the other half as in gfx10 / AnyPS5 `write16Bits`), f16/u16/i16 comparisons,
  VOP3P `v_pk_*`, `v_dot*`; 64-bit (`v_cmp_*_u64/i64/f64`, b64 shifts, `v_mad_u64_u32`, f64 add/mul/fma/cvt, 64-bit inline
  constants as double); `v_movrel*` (switch on M0).
- **Mesh mode** (`Mode::Mesh`, `MsInterface` from PSV0: 128 threads, ≤128 vertices/triangles): DXC does not allow
  `WaveSize` in an MS, and console code assumes wave64 → two passes: the program as a CS `[WaveSize(64)]` writes a group record
  (`s_sendmsg 9` GS_ALLOC_REQ M0 = prim<<12 | vert, `exp pos0/param/prim`, vertex = thread in the group), then
  `mesh_passthrough()` rasterizes the records. Inputs measured on the corpus (`xcl shaders --iface`): s3[27:24] = wave in the group,
  v5 = group·128 + thread (group = v5>>7, like the gfx10.3 MS emulation in RADV), s16–s24 user data (from `io`).
  Test `gpu_mesh_mode_two_passes`: 2 groups × 2 waves → correct records and pixels on the RX 9070 (`runner::mesh_draw`).
- Control flow: instead of a single `switch(pc)` loop, a chain of `if (pc == block)` in address order; `while` only for
  ranges with backward jumps (merged into nesting). Smaller DXIL (470 → 310 KB on the largest program).
- **DXC: the optimizer is superlinear on large programs** (median 0.6 s, but an MS with a DS/VS wrapper, 0.7–1.2 MB of HLSL:
  50–160 s at -O1..-O3, 5 s at -O0) → `dxc::compile` adds `-O0` by itself for HLSL > 256 KB (the driver optimizes anyway).
  Do NOT run `--compile 3000` (an hour); for a quick check use `--compile 64`.
- Tools: `xcl shaders … --recomp --example MNEMONIC|name=FRAGMENT` (listing + DXC result; `XCL_EXAMPLE_HLSL=file` saves
  the HLSL), `--iface` → inputs/exports/`s_sendmsg`/PSV0 per stage, DXC times with `--compile N`.
  `dxbc::psv()` (PSV0), `rdna2::sgpr_uses()`. Disassembler: VGPRs in a 9-bit field (`v_readfirstlane … v261` → `v5`).
- Tests: 86 unit tests (including 17 recompiler tests on the GPU) – PASSED (`cargo test -p xcl --lib`, ~0.5 s).

- Later in session 4: `src_pops_exiting_wave_id` (POPS – the wait is passed through, no ROV ordering: a known limitation),
  flat apertures = 0, `image_*_o` (offset from a VGPR → coordinates shifted by offset/mip size; GPU test) → PS 12 097 /
  12 101. **Runtime:** a captured console shader is translated right away (mode from the interface), `.fssmN.pM.hlsl` next to the dumps,
  `report.json → shaders.recomp_*` (programs, complete ones, modes, missing mnemonics); `XCL_RECOMP_DXC=1` compiles through
  `dxc::compile_cached` (DXIL cache in `%LOCALAPPDATA%\xcl\dxil` or `XCL_DXIL_CACHE`). e2e checks this.
- Gallery: `retail_ps_gallery` (ignored test, `XCL_GALLERY_DIR`) – 24 PS shaders from the pack drawn on the GPU (PNG + page
  `index.html` in that folder, kept locally – not in the repository: renders of game shaders), approx. 10 s.
- FSSM header: `--example name=…` prints it (the dwords up to the code). Stage descriptor +0x110 looks like RSRC2 (USER_SGPR in
  bits 5:1, e.g. 0x18 → 12); from approx. +0x25C in PS an export patch table (`exp` variants). The root signature → user
  SGPR mapping is done by the D3D12.X runtime (the SGPRs carry pointers to tables) – to be determined only from the game exe.
- GS (195 + 195 ES) postponed: on-chip GS – the ES writes to LDS and does `s_setpc` to the GS, the GS writes vertices to LDS
  (64 B/vertex, base = s2 + ([s8+0xB8] << 8)), `s_sendmsg` emit/cut/done; no copy shader in the pack.
- Tests: 90 unit + e2e (29 checks in the title) – PASSED (`XCL_SAMPLE_RPF=… cargo test --workspace -- --test-threads=4`, a few seconds).

- **UMA – console memory accessible to the GPU** (`gpu/uma.rs`, NEW): a single region (2048 MB by default, `XCL_UMA_MB`,
  max. 4095 MB) at an address aligned to 4 GiB, fully committed, opened through
  `ID3D12Device3::OpenExistingHeapFromAddress` (measured: a partial commit is rejected, 4 GiB is rejected, 4095 MB OK in 0.5 s).
  The low 32 bits of a console address = offset in the buffer → the recompiler's flat `mem` model works on game memory without copies.
  First-fit allocator (64 KB, coalescing, zeroing). What goes there: `VirtualAlloc` with `MEM_GRAPHICS` (+ new HLE `VirtualFree(Ex)`),
  `XMemAlloc` with a memory type ≠ 0. The `real` backend opens the region as a heap + a UAV buffer at device creation
  (`real::uma_gpu_va()`, `report.json → d3d12.uma_*`). Tests: `gpu_unified_memory_arena` (s_load/buffer_load/store directly on
  process memory), `uma::tests`, a check in the test title, e2e checks `uma_gpu_va`.
  Next: bind the shaders from the game's PSOs to this buffer as `mem` (u2), and fill the SGPRs from root arguments (requires the game exe).

- **Without the exe (session 4, end):**
  - `gpu/desc.rs` (NEW): GFX10 T#/S#/V# → D3D12 (DXGI format from the GFX10 IMG_FORMAT table – 0–77 like buffers, 128+ sRGB/
    packed, 169–182 BC1–BC7; SRV with channel mapping from DST_SEL, dimensions/mips/arrays/cube/3D/MSAA; sampler: addressing,
    filters, aniso, comparison, min/max, LOD, bias, border). Fields confirmed against AnyPS5. Unit + GPU tests
    (`gpu_tsharp_to_d3d12_resources`: 6 kinds of textures created from decoded T#s). Not used in the runtime yet.
  - GameInput: the `gameinput.dll` import goes to the system `GameInput.dll` (export `GameInputCreate`) – works without
    emulation, checked in the title. If the console version of the API turns out to be different – an alias in the config / HLE.
  - XGameSave (`hle/xgameruntime/gamesave.rs`, NEW): provider/containers/blobs/updates (+ Async/Result), files in
    `<data>\gamesave\<configId>\<container>\b\<blob>`, names %XX-encoded, writes through a temporary file, 256 MB limit.
  - `xcl rpf <pack> [--find|--extract … --out …]` (NEW): list/extract records; `textures.rpf` – 21 nested
    RPF9s with an encrypted TOC (fixed key from the game, data without zstd/LZ4/zlib signatures – probably Oodle) → unreadable
    without the exe; the game unpacks the archives by itself anyway, XCL will get the textures in memory (GFX10 tiling).
  - Deliberately NOT done (waiting for the exe, so as not to guess): physical pages (`MapTitlePhysicalPages` at the game's address),
    GFX10 tiling decoder (no real data to verify against), Opus through the console API, GS.
  - Tests: 96 unit + e2e (29 checks in the title) – PASSED.

- **Translator hooked into the graphics PSO** (`gpu/recomp/pso.rs` NEW, hook in `hle/d3d12x/real.rs`):
  `CreateGraphicsPipelineState` with a native console VS (and PS) → `pso::console_to_dxil` (hardware VS/PS variant from FSSM,
  mode from the interface, only complete translations, DXC with cache) → a PSO with translated shaders and the **XCL ABI root signature**
  (`pso::abi_root_signature`: root UAV u0 io / u1 res / u2 mem + directly indexed heaps; `real::abi_root_signature()`).
  Failure → the old path (original → replacement shaders). `report.json → d3d12.translated_pipelines / translation_errors`.
  GS/HS/DS not yet. **Missing:** binding at draw time – the game's root arguments → SGPRs in `io`, `mem` → the UMA buffer
  (requires the user data layout of the console runtime, i.e. the exe).
- E2E test without the exe: `crates/xcl/tests/gpu_pso_e2e.rs` – `VS_Rage_TransformPosCpv` + `PS_Rage_Untextured` from `shaders.rpf`
  through the backend's device hook → `runner::draw_on` (new: drawing with a ready PSO on any device) → 64×64 RGBA32F;
  explicit inputs (s8 → table of 2 V# → 4×3 world matrix and 4×4 view-projection, POSITION+COLOR vertices); 435 pixels
  inside compared with interpolation computed on the CPU (±0.01), 3501 outside = background color; the `translated_pipelines` counter +1,
  `fallback_pipelines` unchanged. Checked by sabotage: with translation disabled the test fails (both on the counter and on the pixels).
  `real::init()` is now idempotent (previously a direct call + `backend()` created two devices).
- Tests: 96 unit + title e2e + `gpu_pso_e2e` – PASSED.

- Second test in `gpu_pso_e2e.rs` – textures through the decoder: `VS_Rage_TransformPosCpvTex` + `PS_Rage_Textured` (T# from the table
  s[8:9]+0x20, S# built into the PS code = static sampler: aniso 4×, wrap). Two linear 64×64 RGBA8 textures with known
  patterns in memory at the addresses from the console T#s → `desc::image` / `image_d3d12` (resource + SRV, data read from the address
  in the T#), S# → `desc::sampler_d3d12`; tables in the XCL ABI form (`desc::xcl_image_words`: address → SRV index in the heap).
  One PSO, one command list, two draw calls that differ only in the table pointer in user data (left half: A,
  right: B with DST_SEL zyxw) – `runner::draw_on_multi` + `runner::ViewTexture`/`Draw` (NEW). 4096 pixels = texel (±0.005),
  2048/2048 on the right differ from A. Sabotage checked: DST_SEL ignored → fails; the same table in both draws → fails.

- **GS emulation** (`Mode::Geometry`, `GsInterface`, `recomp::geometry_program(es, gs)`): legacy on-chip GS from the pack
  (195 `GS_VoxelizeGeneric` + 195 ES variants `VS_*`, paired by hash suffix). ES and GS of one pipeline = one CS
  `[WaveSize(64)]`, group = up to 21 triangles (63 ES vertices); `s_setpc_b64` ES → GS code; LDS = `groupshared`;
  ESGS element = 16 B per ES OSG1 register (measured: variant 6be01b34 writes every 16 B, 912dfba0 every 32 B – the console driver
  derives this from the ES); GS inputs: v0/v1 = vertex offsets in dwords (16-bit packed), s3 = vertices | primitives<<8,
  s2 = GS output area after the 64 ES elements; `s_sendmsg` emit/cut → triangles from the strip (odd ones swapped), the thread's output
  base = the lowest address it wrote in the GS phase (does not depend on the value at [s8+0xB8]); result → mesh mode record →
  `mesh_passthrough`. ES attributes fetched beforehand: float4 per ISG1 register in `mem` at io[111], primitive count io[110].
  **Assumptions to check on the game:** the value of [s8+0xB8] (GS output base, 0 in the test), the meaning of s19/s21/s16/s17 (user
  data), triangles only → triangle strip, no GS PSO/drawing in the backend (command list capture needed).
- **Fixed translator bug:** `ds_read*` with a destination register overlapping the address (`ds_read2_b32 v[0:1], v0`) computed the second
  address from the overwritten register – now the address goes into a variable first. Only a real GS exposed it; regression test
  `gpu_ds_read_destination_overlaps_address`.
- Corpus: **24 056 / 24 064 (99.97%)** – GS 390/390, MS, PS, VS 100%; 8 CS with `image_bvh*_intersect_ray` remained.
- GS tests: `gpu_geometry_mode_es_gs_strip` (synthetic ES+GS: 25 triangles = 2 groups, a 4-vertex strip, every vertex
  and index exact, drawing) and `retail_es_gs_voxelize_geometry_mode` (a real pair from the pack: projection onto the plane of the dominant
  axis, exact – z → (x, y), x → (z, y), depth 0.5·d+0.5, axis 4/0, TEXCOORD1 = (-1,-1,1,1), TEXCOORD3 = unit
  normal, 496 pixels). All tests: 98 unit + e2e + 3 in `gpu_pso_e2e` – PASSED.

- **Ray tracing: `image_bvh(64)_intersect_ray`** (8 programs: `CS_RTShadowTrace*` wave32, `S_ScreenProbes_TraceRays_BVH`):
  HLSL function `bvh_intersect` in PRELUDE_MEM following the public RDNA2/RADV format – node = id (type in bits 2:0) at
  base + (id & ~7)·8, base from the descriptor in 256 B units (word 0), sorting = word 1 bit 31; box32 (5) / box16 (4,
  f16): slab test of 4 children, hits sorted by distance (a 5-comparison network), misses 0xFFFFFFFF; triangle (0..3):
  (t, denominator, i, j) as numerators (Möller–Trumbore), miss t = inf; instance/AABB → 0xFFFFFFFF (the shader handles it).
  A16 (dir/inv_dir as halves) supported. The game code confirms the layout (node address = (id >> 3) << 6, triangle_id at +48).
  Test `gpu_bvh_intersect_ray` (hand-built BVH, 3 kinds of rays, sorted/unsorted, box16, hit t=5 i=j=0.5, misses,
  64-bit variant). Approximations: NaN/equality edge cases and the hardware's watertight variant, return modes other than IJ.
- **Corpus: 24 064 / 24 064 (100%)** of programs translate in full; DXC compiles both RT programs.
  Tests: 99 unit + e2e + 3 in `gpu_pso_e2e` – PASSED.

## Session 5 (2026-11-20) – everything that can be done without the exe
- **Compute mode** (`Mode::Compute`, `CsInterface::from_descriptor`): COMPUTE_PGM_RSRC2 from stage descriptor +0x38
  (USER_SGPR bits 5:1, TGID_X/Y/Z bits 7/8/9, TG_SIZE bit 10, TIDIG_COMP_CNT bits 12:11), numthreads from +0x3C/+0x40/+0x44
  (= PSV0 in all 36 checked CS); group ids after the user SGPRs, v0..v2 = thread id. `runner::dispatch(dxil, groups, sgpr, mem)`.
- **VS/PS wave width** (chosen by the PC driver): a wave32 program in a 64-wide wave runs as two halves (`xcl_w32`,
  `xcl_half`, half-aware `rla()`, `ballot()`); Kernel with `hw_wave == 64 && !wave64` likewise. A wave64 program in a 32-wide wave
  breaks only on readlane of a lane ≥ 32, a lane from an SGPR, and ds_bpermute – `recomp::wave_hazards` reports this, does not fix it.
  Corpus: 1127 wave32 VS/PS handled, 749 wave64 at risk (447 lane ≥ 32, 353 lane from an SGPR).
- `xcl shaders … --precompile [--limit N]`: DXIL into the cache (`dxc::compile_cached`, `%LOCALAPPDATA%\xcl\dxil` or
  `XCL_DXIL_CACHE`); 60 records in 4.6 s, repeat run 0.2 s (from the cache).
- **VS inputs:** SV_VertexID / SV_InstanceID with a register in ISG1 land like attributes in v[4 + 4r] (`VInstanced*` read
  the instance id in v8) – FIXED, test `vs_system_values_in_isg1_registers`. The `--iface` analysis counts from the MUBUF/MTBUF VADDR
  only the registers actually used (IDXEN + OFFEN) – most "outlier" VS were false alarms (the upper half of a 64-bit VADDR).
  Layout v4 + 4·register: 779 / 788 VS. Remaining: 9 `VS_PassthroughComposite_*_Wrapped`: `image_load v[8:11]` with only v8
  set – the analysis counts 4 addresses without looking at DIM, almost certainly a false alarm too.
- **Physical pages** (`hle/kernelx.rs`): Allocate/Free/MapTitlePhysicalPages in UMA – 64 KiB pages, allocation = a contiguous
  UMA block, page number = offset / 64 KiB; mapping consecutive pages returns their address in UMA (visible to the GPU, no copy);
  mapping at an address chosen by the game, or of scattered pages → refused + log (aliasing would require a memory section).
  Test `physical_pages_in_uma`.
- **GFX10 tiling** (`gpu/tiling.rs`; `gpu/tiling_tables.rs` from `scripts/gen_tiling.py` – AMD addrlib patterns from Mesa, MIT, sources
  downloaded from gitlab.freedesktop.org into the scratchpad): 14 2D modes (256 B / 4 KB / 64 KB: S, D, S_T, D_T, S_X, D_X, Z_X, R_X)
  × 5 element sizes × pipe/packer configuration (RB+ tables, gfx10.3). Linear offset over GF(2): `ox[x] ^ oy[y]`;
  `Layout::{new, offset, detile, tile, size}`, `tiling::element(IMG_FORMAT)`, `tiling::image_layout(&desc::Image, cfg)`
  (`base_xor` from the low bits of the T# address). **Verified on real data:** the patterns that the TTPF viewer
  (`tools/ttpf-viewer/src/tile.rs`) found by hand on game textures are bit for bit SW_64KB_D (64 KB blocks) and SW_4KB_S
  (4 KB blocks) for 1/2/4/8/16 B elements. All modes × configurations yield a permutation of the block (test). Unknowns:
  the console's pipe/packer configuration (GB_ADDR_CONFIG) for the _X modes – the `PipeConfig` parameter, to be determined on a texture
  from the game (e.g. pick the configuration that gives the smoothest image); mip chain and mip tail – no (each level separately).
  `StandardSwizzle64KBSupported` on the RX 9070 = 0, so untiling stays on our side anyway.
- **BVH builder** (`gpu/bvh.rs`): BLAS in the RDNA2 format (box32 128 B + triangle 64 B, primitive id @48, geometry @52),
  split at the centroid median on the longest axis, 4 children per node, 1 triangle per leaf. `bvh::intersect_node` is the CPU
  mirror of the HLSL emulation, `Bvh::trace` is a stack-based traversal to the closest hit. Tests: traversal = brute force
  (300 triangles, 2000 rays) and `gpu_built_bvh_nodes_match_the_cpu_node_test` (64 lanes, nodes from the builder through
  the translated `image_bvh_intersect_ray` = CPU result). Missing: instance nodes (TLAS – the console driver's layout, traversed
  by the shader code, to be read from `CS_RTShadowTrace*`), box16 in the builder, hooking into `BuildRaytracingAccelerationStructure`.
- **Opus postponed until the exe** (user's decision: "we'll see once we have the exe"). See the Audio section.
- Tests: 111 unit (7 ignored helpers) + title e2e + 3 in `gpu_pso_e2e` – PASSED (full `cargo test --workspace`).

### Console DLLs from the game pack (session 5, without the exe)
The user provided 16 DLLs from chunk 1002 (`<samples>\*.dll`, ordinary PEs): the console CRT (vcruntime140(_1),
msvcp140(_1/_2/_atomic_wait/_codecvt_ids), concrt140, vcomp140, vccorlib140), Bink2Scarlett, XCurl, libHttpClient.GDK,
amd_fidelityfx_loader_xbox / amd_fidelityfx_upscaler_xbox, XboxBvhBuilder. The game exe (`game_scarlett_master.exe`) is still missing.
- `xcl inspect`: after fixes **0 missing imports in all 16**. Added:
  - `XMemVirtualQuery(process, address, info) → BOOL` (signature from the call sites in XboxBvhBuilder, objdump from msys64):
    info = MEMORY_BASIC_INFORMATION + a field @48, bits 0–1 = memory readable by the CPU (message on error:
    "pDesc->SourceAccelerationStructureData must be non-null and point to CPU-readable memory");
  - `XMemGetWorkingSetStatistics(allocator, stats) → FALSE` (the FSR upscaler reads +8 only on success → reports 0);
  - `xaudio2_9.dll` ordinals in `data/stubdb.toml` (from the system DLL; Bink imports #1 = XAudio2Create).
- `crates/xcl/tests/console_dlls_e2e.rs` (`XCL_SAMPLE_DLLS` = folder with the DLLs, the test is skipped without it):
  - the console CRT loads with the XCL loader (HLE/Windows/game DLL imports, TLS, DllMain) and **works**: memcpy, `std::mutex`
    (`_Mtx_*`) with 2 threads (40 000 increments), `_Thrd_hardware_concurrency` / `omp_get_num_procs` /
    `Concurrency::GetProcessorCount` = the number of cores;
  - the other DLLs (Bink, XCurl, libHttpClient.GDK, FFX, XboxBvhBuilder, msvcp140_*, vccorlib140): mapping + TLS + DllMain OK;
  - **the real console BVH builder** (`GetRaytracingAccelerationStructurePrebuildInfo` computes on the CPU; a zeroed object
    = default settings) – the result size matches exactly (17/17 cases)
    `align_1K(256 + N·leaf + max(N−1,1)·128 + 4 B per 64 B of nodes)`: BLAS leaf (triangle) 64 B, TLAS leaf
    (**instance node**) 128 B, box32 128 B, header 256 B, parent pointer 4 B per 64 B (layout as in AMD GPURT / RADV).
- XboxBvhBuilder.dll = the C++ class `XboxBvhBuilder(ID3D12Device11*, bool)`: Build, BuildBatched2X, Copy, GetPrebuildInfo,
  `Rehydrate*` (loading prebuilt BVHs – probably the data from `*_bvh4.rpf`), Set64BitMortonCodes, CpuValidateResultX. Builds on the GPU
  (D3D12SerializeVersionedRootSignature → compute shaders in the DLL).
- **AS format read from the game's RT code** (`CS_RTShadowTrace32TemporalHQ_…_0c84ea88`, listing via `xcl shaders … --example`):
  - AS address from the descriptor (`s_load_dwordx4 [s80:81+0x40]`, words 0–1 >> 8 = base for `image_bvh`), **root = dword @0x30
    of the header** (`s_load_dword s55, [base+48]`); node ids counted from the start of the AS (header included); a root of type 6
    (instance) is handled too;
  - **instance node (type 6, 128 B)**, read from +48: @48 id (24 b) | mask << 24 (test `& 0xE5`), @52 contribution
    (24 b) | flags << 24, @56 BLAS root (30 b; 0x3FFFFFFF = none) | bit 31 = translation only (the shader adds t, does not
    rotate), @60 BLAS address >> 8, @64 world → object as [t.xyz, row 0, row 1, row 2] (o' = M·o + t,
    d' = M·d); BLAS triangle: primitive id @48 (consistent). The shader also reads the BLAS header @52 and @56
    (`[52] >> 6 − [56] >> 2`) – meaning unknown, we write 0. Assumed (the shader does not read them): @0 object → world 3×4
    as in D3D12_RAYTRACING_INSTANCE_DESC, @112..127 zero.
  - `gpu::bvh`: 256 B header with the root @0x30, the root always box32, `build_top(&[Instance])` (TLAS, instance boxes =
    transformed BLAS bounds, inverse matrix in f64), `Bvh::trace_scene(BLAS address lookup, …)` = traversal as
    in the shader, `Bvh::bounds`. Tests: `scene_trace_matches_brute_force` (3 BLAS, 24 instances rotated/scaled/
    translated, some translated only, 3000 rays = brute force over the triangles in world space), `instance_node_layout`
    (every field), sizes ≤ the console model without parent pointers. We do not write parent pointers (location unknown,
    no shader seen so far reads them).
- **FSR – decision (session 5): do not emulate the console FSR4.** The upscaler from the pack is an ML model in fp8
  (`fsr4_model_v07_fp8_…`) with pipelines compiled for the console (parts `ACCC`, `DeserializeComputePipelineStateX`,
  "ACO-cooked", with a fallback "NewBE" path). The game does not need it to run – with the exe: HLE of the FFX API
  (ffxCreateContext/ffxDispatch → simple upscaling with our shader, or hand-off to the PC FSR on RDNA4).
  Descriptor tables and textures – to be determined on the game.
- FSR: the loader has the standard FFX API (ffxCreateContext/Configure/Query/Dispatch/DestroyContext) like the PC
  `amd_fidelityfx_loader_dx12.dll`; the upscaler is **FSR4** (`…fsr4_full_720p1080p_input_sanitize_scarlett_opt2.dll`). Plan:
  alias the loader → FFX DX12 from the PC (FSR4 on RDNA4 = the user's RX 9070); check only with the exe.
- **Bink works:** the real `Bink2Scarlett.dll` under XCL decodes a movie from `movies.rpf` (`bink2_console_dll_decodes_a_game_movie`,
  `XCL_SAMPLE_BK2` = an extracted `.bk2`: BinkOpen → BinkDoFrame/BinkCopyToBuffer(BINKSURFACE32 = 3)/BinkNextFrame; 1280×720,
  60 frames, correct image – checked by eye). Movies in the pack: Bink 2 revision `n`, 720p/1080p, 30 fps, uncompressed in the RPF.
- **`core_bvh4.rpf` = compressed BLASes in the console format** (session 5): RAGE resource = 208 B virtual header + physical
  block (separate zlib streams); physical block +0x100 = the source for `XboxBvhBuilder::Rehydrate…X` (a pure CPU function,
  ignores the device): dword 0 = 0x81020006 (bit 24 = compressed, rehydration clears it), @0x04 1503 (meaning?),
  @0x08 CRC32 (standard table, start 0, no final XOR) over NumDescs + per geometry Flags, IndexFormat, IndexCount,
  VertexCount (only Validate… checks it), @0x0C number of triangle nodes, @0x10 AABB, @0x28 size, @0x30 root,
  @0x34 offset of the first triangle node, @0x38 ?, @0x3C per-axis scale exponents (positions in the BVH = position · 2^(e−127)).
  The blob has no vertices – rehydration reads the game's vertex / index buffers from the geometry descriptions (standard
  D3D12_RAYTRACING_GEOMETRY_DESC; without indices 3p+k). Test `xbox_bvh_builder_rehydrates_game_geometry`
  (`XCL_SAMPLE_BVH4` = the pack, zlib resource @0xa4e26070, synthetic vertex buffer): 4 geometries (644/252/744/4
  triangles), 287 box32, 54 box16, 832 pair nodes – every triangle checked (vertices, I/J rotation, unique primitive,
  primitive ranges without gaps).
- **Node format = AMD GPURT gfx10** (MIT sources downloaded into the scratchpad: `TriangleNode1_0.hlsli`, `IntersectCommon.hlsl`,
  `Common.hlsl`), confirmed on the output of the console builder and on the game's RT code:
  - box16 (type 4) in use alongside box32; triangle = **pair**: v0..v3, type 0 = (v0, v1, v2), type 1 = (v1, v3, v2); @52/@56
    primitive 0/1 (index within the geometry), @60 triangle id (a byte per triangle: bits 1:0 I SRC, 3:2 J SRC, 7 opaque) –
    the hardware rearranges the barycentrics (`SwizzleBarycentrics`);
  - **@48 in the Xbox version: geometry << 8 | second triangle << 7 | opaque** (differs from GPURT): boxes point only to type 0,
    the game's shader, after the type 0 test, checks bit 7 (`v76 = [node+48] >> 7`) and tests `node | 1`; bit 0 goes into the
    opaque/any-hit decision.
  - Fixed: the HLSL emulation `bvh_intersect` (type 1, barycentric rearrangement according to @60), the CPU mirror (`bvh::intersect_node`:
    box16, pairs), `Bvh::trace` (bit 7 → type 1), the builder writes @48 = geometry << 8 | 1, @52 primitive, @56 0xFFFFFFFF,
    @60 = 0x89. Tests: `gpu_triangle_pair_and_box16_nodes` (GPU = CPU, the barycentrics of a rotated triangle 0 and of triangle 1 give
    the exact hit point), the old BVH tests with triangle id 9.
- **XboxBvhBuilder on the XCL device (probe, `XCL_PROBE_BVH_DEVICE=1`, test `xbox_bvh_builder_on_xcl_device_probe`)**:
  constructor on the real backend (RX 9070) → `D3D12SerializeVersionedRootSignature` with **console-only flags
  0x3000000** (Windows: "Unsupported bit-flag set") – HLE strips the bits outside 0xFFF and logs → root signature created →
  then the process exits with 0xC0000409 (fail-fast, no SEH exception – suspicion: a device method writes a larger
  structure than the console one / /GS cookie). `QueryInterface(0742a90b-c387-483f-b946-30a7e4e61458)` in the log does not come
  from the builder (more likely from the PC runtime). **Next step:** trace the device vtable calls from the builder (which slot, arguments).
- `OutputDebugStringA/W` (kernelx HLE) → XCL log `[debug] …` (console DLLs and games write a lot there; that is how the XDXR error surfaced).
- **The XboxBvhBuilder constructor goes through completely on the XCL device** (probe `XCL_PROBE_BVH_DEVICE=1`):
  - `XCL_D3D12_TRACE=1` (NEW, `guard::make_tracer`): every public vtable method called from console code → log
    `D3D12 trace: Iface::Method (#slot) ← module+offset`. That is how the sequence CreateRootSignature → SetName → slot 62 came out.
  - **The console device vtable differs from the PC one:** slot 62 on the console = `CreateComputePipelineState(desc, NULL, NULL,
    riid, ppv)` (two extra arguments), on PC = CreateStateObject. `guard::make_split` (NEW): a call from the game image →
    console version, from the runtime → PC; table `real::console_slots` (for now only device #62).
  - Compute PSO with a console CS → `pso::console_to_dxil(…, Stage::Compute)` (cs_6_6) + XCL ABI root signature
    (`real::translated_compute_pso`). Console bytecode must not be passed to the driver: the AMD compiler (amdxc64.dll)
    crashes on DXIL with NOBC bitcode instead of rejecting it – a failed translation → a replacement shader right away.
  - New instructions in the translator (from the builder's shaders, they were not in `shaders.rpf`): `global/flat_atomic_*` (add, sub, and,
    or, xor, swap, s/umin, s/umax, cmpswap, **fmin/fmax via a CAS loop**), `buffer_load_ubyte/sbyte/ushort/sshort`,
    `buffer_store_byte/short`, `ds_write_b8/b16`, `ds_read_u8/i8/u16/i16` (atomic AND/OR on groupshared). Result: 53/53
    builder shaders translated, 0 replacements. **No unit tests for these instructions** – still to be written.
  - **INCIDENT:** the builder's largest shader (1589 instructions, HLSL < 256 KB → full DXC optimization) compiled for
    ~6 min and exhausted 38 GB of committed memory (32 GB RAM + 7 GB page file) – Windows killed the user's
    session (event log: "The system is low on virtual memory", 23:27). The `-O0` threshold in `dxc::compile` lowered from 256 KB
    to 48 KB of HLSL. The DXIL of this shader (314 KB) is already in the DXC cache. Before further probes with large shaders: a
    memory limit for the test process (Job Object), or DXC compilation in a separate process.
  - Tests of the new instructions: `gpu_global_atomics_and_sub_dword_access` (64 lanes at the same addresses: add with GLC,
    sub, fmax/fmin from ±inf, smax; short/byte in a buffer and in LDS with neighbors in the same dword, neighbor read after a barrier).
  - Tests with the console DLLs run under an **8 GB memory limit** (Job Object in `setup`, test `memory_cap_is_active`).

### Root argument binding on the console (determined on 52 shaders of XboxBvhBuilder.dll)
- **s[0:1] = pointer (40 bit, s1 & 0xFF) to the root argument array**: parameters packed in root signature order,
  root constant = 4 B, root descriptor (CBV/SRV/UAV) = 8 B GPU address, no alignment (a UAV after 19 constants lies at 76).
  The shader builds the V# from the address by itself (num_records −1). Checked automatically: 123 reads from the array in 50 shaders – all
  on parameter boundaries (script `check_layout.py` in the scratchpad; shaders extracted from the DLL by their DXBC signature).
  Descriptor tables – not seen yet (the builder does not use them). Group ids after the user SGPRs (s4 = TGID.x with 4 user SGPRs).
- For XCL: on a dispatch with a translated PSO – assemble this array in UMA from the remembered root arguments, s0/s1 = its address
  in `io`, the rest as in Compute mode.

### D3D12.X: the command list is partly "compiled into" the title's code (probe `XCL_PROBE_BVH_RECORD=1`)
- Recording `BuildRaytracingAccelerationStructure` on a list from the XCL device: the builder calls **console list method #148**
  = `AllocateUploadMemory(size, alignment) → CPU pointer` (for us the guard returns 0 → "D3D12 XDXR Error: Failed to
  allocate upload buffer" → C++ exception), and then **writes directly into the internals of the list object**: +0x18 write pointer,
  +0x30 buffer end (appends a 28 B record), +0x140 flag, +0x544 root signature state; slot 33 here is likely
  SetComputeRootSignature (on PC #29). There are no PM4 headers in the code – the records are in a D3D12.X software format,
  turned into GPU commands by the console driver at Close/Execute.
- **Architectural conclusion (also applies to the game):** replacing the vtable on a real PC object is not enough – console code
  would write all over the internals of the PC driver's object. Console code must get **its own XCL list objects** with the console memory
  layout (a vtable forwarding to the PC list + a "shadow" area for the fields and the record buffer), and XCL must **interpret the records**
  at Close/ExecuteCommandLists. The field layout and record format are to be determined from the inline code (first the builder: compute only,
  then the game). This is probably the largest remaining piece of rendering work.

### D3D12.X command list format (compute) – determined on XboxBvhBuilder.dll
Method: a fake list (`xbox_bvh_builder_records_into_a_fake_console_list`, `XCL_PROBE_BVH_FAKELIST=1`: a vtable of 600 thunks
logging the slot and arguments, #148 allocates from UMA, +0x18/+0x30 = record buffer) – the builder recorded the whole BLAS build without
crashing; plus a static review of all 141 write sites (`record_writers.py` in the scratchpad).
- **List object fields:** +0x00 vtable, +0x18 write pointer, +0x28 limit for dispatch records, +0x30 buffer end,
  +0x140 "dirty" flag (set when the header has bits 31:30 = 0), +0x141 a bit from the pipeline (bit 20 of word +0x20 of the record),
  +0x143 flag, **+0x344 + 4·i = header of root parameter i** (written by a method – for us #33, most likely
  SetComputeRootSignature – the inline code only copies it), +0x544 current pipeline (16 B).
- **Records** (no common frame; the length follows from the header / kind):
  - pipeline 28 B (×34): a copy of +0x10..+0x2B of the builder's "info" object = [PSO pointer (our adopted object)][pointer to
    the D3D12Core object][8 B of flags][4 B]; stands in for slot 25;
  - root parameter i: header = [+0x344 + 4i] (+ (n−1) << 16 for multiple constants), then: descriptor = 64-bit address
    (12 B, stands in for slot 43/45), one constant (8 B, slot 37), n constants (4 + 4n B, slot 39: (i, n, data, offset));
  - Dispatch: `0x4002C400` (or `(0x1000BA00 | flag) << 2`) + x, y, z = 16 B, stands in for slot 14 (= PC Dispatch);
  - `0x4003CC00`, 20 B, stands in for slot 80 (probably indirect dispatch);
  - methods: #148 AllocateUploadMemory(size, alignment) → CPU pointer (= GPU address in UMA), #146 barrier
    (1, pointer to a description), #33 (object pointer – root signature / PSO), #14 Dispatch.
- Type 3 PM4 headers do not appear as constants – the parameter headers come from the list, so **XCL can put its own
  tags there** (e.g. 0xC0000000 | i << 8 | tag) and read the stream unambiguously; the pipeline record can be recognized by the known
  PSO pointer.

### DONE: console command list (compute) – the console BVH builder works on the PC GPU
`hle/d3d12x/console_list.rs` (NEW, enabled with `XCL_CONSOLE_LISTS=1`): test `xbox_bvh_builder_builds_on_the_gpu`
(`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_GPU=1`, ~1 min with an empty DXC cache, then seconds) – **the real XboxBvhBuilder.dll
builds a BLAS from 64 random triangles on the RX 9070 through XCL**: the D3D12.X inline code records into a list in the console layout,
`console_list::close` turns the records into PC commands (10 dispatches of translated console shaders, 9 barriers, 0
unknown records), the result read with `bvh::Bvh::trace` = brute force on 500 rays.
- List object: a vtable of 600 slots (known: 0–2, 14 Dispatch, 25 SetPipelineState, 33 SetComputeRootSignature,
  37/39 constants, 41/43/45 descriptors, 146 barrier, 148 AllocateUploadMemory from UMA; the rest – logging stubs), fields as
  in the format above, records in 1 MB chunks (4 KB of slack past the limit, because the inline code checks `write < end` and only then
  writes).
- Root parameter headers: `0xC0000000 | i << 10`; the inline code adds **constant offset · 4** (bits 9:2) and
  **(count − 1) << 16**; our own tags `0x8C0FE0xx` (pipeline, root signature, barrier); dispatch `0x4002C400` /
  `0x4002E800|4`.
- PSO for console code: an object in the console shape (`wrap_pso`, at +0x10 a tag + a pointer to itself, methods 3–8 →
  the PC PSO); created on the console path (device slot 62) when console lists are enabled. The root argument layout
  is remembered at CreateRootSignature (`note_root_signature`, RTS0 parser).
- Dispatch: root argument array (layout from RTS0) in UMA, `io[0..1]` = its address, XCL ABI root signature, u0 io / u1 res /
  u2 `mem` = the whole UMA buffer, translated PSO.
- **Important: `mem` in Compute mode is `globallycoherent`** – without it the bottom-up tree build (atomics + reading what
  other groups wrote) produced wrong child pointers (on the console the code uses GLC / cache invalidation). Graphics gets an ordinary view.
- **Hooked up automatically** (when `XCL_CONSOLE_LISTS=1`): device #12 CreateCommandList – split thunk: console code
  gets a console list (the initial PSO unwrapped to the PC one), XCL code gets a PC list; list: #9 Close (interpretation),
  #10 Reset, #3–8 → PC list; the queue's ExecuteCommandLists replaces console lists with PC ones; the recording's UMA memory is freed
  on Reset of its allocator (`alloc_reset` → `console_list::allocator_reset`). The test does 2 rounds through the vtable (64 and 200
  triangles, allocator and list Reset between them) – both agree with brute force.
- Not yet: record `0x4003CC00` (indirect dispatch, slot 80), the #146 barrier description (currently always a global
  UAV barrier), descriptor tables, graphics (draw records – to be determined on the game), freeing UMA memory after execution
  (the list holds the blocks until Release).

### TLAS on the GPU – WORKS (session 5): instance node format confirmed on the console's output
- `xbox_bvh_builder_builds_a_tlas_on_the_gpu` (`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_TLAS=1`): the console builder builds on the GPU
  2 BLAS + a TLAS of 16 instances (rotated/scaled and translated only), `bvh::Bvh::trace_scene` (instance layout from the game's RT
  code: @48 id|mask, @56 BLAS root | "translation only" bit, @60 BLAS address >> 8, @64 world→object [t, rows])
  = brute force in world space on 800 rays. What unblocked it was **removing branches from the generated HLSL** (below).
- **Branch-free code (the most important performance change in the translator):** in HLSL 2021 `?:`, `&&`, `||` on scalars short-circuit,
  so each of them was a real branch – `lanebit` (lane activity, at every instruction), `scc = (…) ? 1u : 0u`,
  `v_cndmask`, DPP/permlane, buffer range conditions. Now `select()`/`and()`/`or()`/`(uint)(bool)` (memory reads/writes
  still behind an `if`), VGPR write `v = select(active, value, v)`, the VALU path without a named temporary variable.
  Effect on the builder's shaders: 15 720 instructions – DXC `-O3` 5.8 s, 585 KB DXIL (before: > 150 s / driver:
  out of memory); the shader from yesterday's crash (1 589 instr.) – 0.3 s, 48 KB (before: ~6 min and > 30 GB).
  **DXC optimizes everything again** (`-O0` only for > 8 MB of HLSL). A sample of 64 game shaders: 0 errors; median 0.29 s, p90 ~11 s.

### FSR4 for learning about textures (session 5, in progress)
- Probe `fsr4_console_upscaler_creates_a_context_probe` (`XCL_PROBE_FSR=1`): `ffxCreateContext` with
  `ffxCreateContextDescUpscale` (type 0x10000) + a backend description (type **0x1000002**, device at +0x10) – **the context
  is created on the XCL device**. The only configuration in the pack (FSR says so itself via OutputDebugString): render 1280×720
  (or 1920×1080), output 1920×1080, flags HDR | DEPTH_INVERTED | AUTO_EXPOSURE (1 | 8 | 32).
- More console differences (table `real::console_slots`):
  - device **#76 = CreateCommittedResource** (like PC #27); **console resource description**: Layout (+44) = **0x101** (2D texture
    1920×1080 R8G8B8A8, RT|UAV), +52 = 3, description ≥ 64 B – console layouts → UNKNOWN on PC (log in
    `dev_create_committed_resource_x`);
  - descriptor heap **#9 / #10 (GetCPU/GPUDescriptorHandleForHeapStart) return the handle in rax** (PC: a hidden pointer).
- Context creation: 14 resources, 16 UAVs, 10 SRVs, Map/Unmap, GetCopyableFootprints – all through methods (no inline code).
- **`ffxDispatch` on a console list (`XCL_PROBE_FSR_DISPATCH=1`, 1280×720 inputs: RGBA16F color, R32F depth, RG16F
  vectors; 1920×1080 RGBA16F UAV output) – 768 words, all recognized**: 20 dispatches, 24 barriers, heaps, tables.
- **Descriptor tables – format known:**
  - record: parameter header + **1 word = the 32-bit GPU address of the first descriptor** (root CBV: header + 2 words);
    in the packed argument array a table takes **4 B** (`Param::Table`);
  - shader (RDNA2 code from ACCC, see below): `s_mov_b32 s10, s5; s_mov_b32 s11, 0; s_load_dwordx4 s[16:19], s[10:11], 0x240` –
    **upper half of the table address = 0**, descriptor = **32 B per slot** (T# 8 dw, V# 4 dw in a 32 B slot; SRV×21 = 0x2A0);
  - root CBV: the shader builds the V# by itself (`s_mov_b32 s10, -1; s_mov_b32 s11, 0x31016fac` = DST_SEL xyzw + format 32);
  - user SGPRs (14, RSRC2): s4 = table 0 (UAV), s5 = table 1 (SRV), s[6:7], s[8:9]… = root CBVs (the same packed layout
    from s4 on; s[0:3] – unknown), s14/s15 = TGID x/y.
- **Console heaps in XCL (`hle/d3d12x/console_heap.rs`, NEW)** – a CBV/SRV/UAV heap created by console code
  (#14) gets 32 B slots in UMA (CPU handle = GPU handle = address in UMA → the low 32 bits = offset in `mem`) + a PC heap
  with 2 descriptors per slot (slot k → SRV 2k, UAV 2k+1). #15 returns 32, #18/#19 create a PC descriptor and write into the slot
  a T# in the XCL ABI (`desc::xcl_image_for_view`: word 0 = 2k, format/size/type/mips as in the T#; a 2D UAV also gets an SRV at 2k,
  because RDNA does not distinguish them). List: #32 SetDescriptorHeaps, #26 ResourceBarrier (transitions forwarded to PC), #15
  CopyBufferRegion. The probe test checks that the tables from the stream point to the slots with the input textures (formats 71/22/29).
  - **Buffers in UMA (done):** a buffer created by console code (#76, dimension BUFFER) = a placed resource
    (CreatePlacedResource) on the UMA heap (`UMA_HEAP`, flag ALLOW_CROSS_ADAPTER, 64 KB alignment); resource #11
    GetGPUVirtualAddress (console) returns the address in UMA (= the CPU address, low 32 bits = offset in `mem`); **Map on PC returns
    the same address** (checked once, log); Release (all callers) frees the block. Buffer views (SRV/UAV) get a V#
    (`desc::buffer_words`: raw → stride 0, num_records in bytes; structured/typed → stride, num_records in elements,
    GFX10 format). FSR: 3 buffers in UMA, all V#s in the tables point into UMA.
    - Note for the future: all console buffers (DEFAULT ones too) live in system memory (UMA) – the GPU reads them over
      PCIe; for large game buffers this may be slow.
  - Missing: CBV via #17 (not seen), sampler heaps, freeing slots on heap Release, CopyDescriptors.
- **FSR4 PSO = ACCC ("FSPX" container)** – CreateComputePipelineState (NewBE) receives an SFI0/ISG1/OSG1/PSV0/RTS0/HASH/ACCC blob
  without DXIL. FSPX: header 'FSPX', version 1, n entries, size; 16 B entries (type, offset, size, 0): **0** (0x1C B:
  numthreads u16×3 @+0x0C (256,1,1), RSRC1 @+0x14, RSRC2 @+0x18), **1** (0x14, hash), **6 = RDNA2 code** (the first 0x24 B =
  a header with the hash), **7** (resource declarations: 32 B records of type 0x41/0x42/0x43, offset in dwords = slot·8).
  The translator does not read them yet → the dispatches are skipped. Helper: `raw_rdna2_listing` (`XCL_RAW=<code>`).
- List slots recognized from the FSR code: **#53 / #54 = ClearUnorderedAccessViewUint / Float** (GPU handle, CPU handle, resource,
  values[4], 0, NULL; FSR picks #54 for a target in format 0x12) and **#85 = CopyTextureRegion** (destination: COPY_LOCATION
  SUBRESOURCE_INDEX, source: PLACED_FOOTPRINT, x/y/z = 0, box NULL; PC #16 – the console has this method elsewhere).
  Records → PC at Close (console handles → the slot's PC descriptors: UAV 2k+1). FSR: 2 clears, 4 copies, 0 skipped.
- **Test title** calls heap #9 the way console code does (handle in rax) – previously through windows-rs (PC ABI), and it had been failing since 7919cdc.
  XAudio2 test: XMA2 lower bound 80 → 70 ms (10 ms quantum jitter, once it came out at 79 ms).

### (archive) TLAS on the GPU – blocked by the size of a single shader
- Test `xbox_bvh_builder_builds_a_tlas_on_the_gpu` (`XCL_CONSOLE_LISTS=1 XCL_PROBE_BVH_TLAS=1`): 2 BLAS + 16 instances
  (some translated only), `trace_scene` vs brute force in world space. The BLASes build, the TLAS comes out empty: 1 dispatch
  skipped, because **the AMD driver rejects the PSO with the translated CS of 15 720 instructions (0x8007000E, out of memory)** – even
  with a 20 GB limit; the second large one (6 302 instr.) goes through at 20 GB, not at 8 GB. DXC `-O1`/`-O3` on it > 150 s.
- Translator improvements made along the way (all tests green, BLAS build on the GPU still 500/500):
  - **structured control flow**: a fully structured program (forward jumps nested, if/else via
    `s_branch`, loops with a single back jump at the end, `s_endpgm` at the end) → plain `if`/`for(;;)` without `pc`;
    otherwise a **hybrid**: the longest single-entry/single-exit stretches structured under a single `pc` guard
    (#19: 628 → 93 guards). `XCL_RECOMP_NO_STRUCTURED=1` disables it, `XCL_RECOMP_WHY=1` tells why it is not structured;
  - `global_atomic_fmin/fmax` as a single integer atomic (positive → int, negative → uint reversed) instead of a CAS loop;
  - VGPR write `v = select(active, value, v)` instead of `if` (linear code);
  - `pso::console_to_hlsl` + helper tests `blob_hlsl_shape` / `blob_dxc_cost` (`XCL_BLOB=…`).
- **To do:** leaner HLSL (fewer temporaries: at `-O0` each one is a separate memory cell, tens of
  thousands in this shader) or a different way of splitting compilation; the same will speed up compiling large game shaders.

### Plan: console command list in XCL (next sessions)
1. `hle/d3d12x/console_list.rs`: an object with the console layout (fields as above, record buffer, parameter headers), wrapping
   a PC list; console vtable: 14 Dispatch, 25 SetPipelineState, 33 SetComputeRootSignature (writes the tags, remembers the
   RS layout), 37/39/43/45 substitute parameter methods (append the same records), 80, 146 barrier, 148 AllocateUploadMemory
   (UMA), Close/Reset; the rest – logging stubs. CreateCommandList from console code → this object.
2. PSO: at creation remember PSO → (translated?, RS layout from RTS0).
3. ExecuteCommandLists: interpretation of the records → PC list: on every dispatch a root argument array (layout from the section
   "Root argument binding") in UMA, `io` = [address lo, hi, …], ABI root signature + io/res/mem, translated PSO, Dispatch;
   barriers → UAV barrier on `mem`.
4. Test: the real XboxBvhBuilder builds a BLAS on the GPU through this list → the result read with `bvh::Bvh::trace` = brute force.

## In progress (unfinished)
Nothing open.

## Audio – conclusions from `audio.rpf` (the real game)
- AWC containers ("ADAT", version 1): u64 chunk table (type = byte 7, size = bits 28–55, offset = bits 0–27, offset from the
  start of the AWC), types: 0x55 data, 0x76 format (32 B: samples u32, loop i32, frequency u16 @8, …, **codec @28**, channels @29),
  0x48 stream format (512 KB blocks), 0xA3 seek table (start sample of every 2048 B block), 0x36 peak, 0x81, 0x5C, 0xBD.
- **Codec 0x0B = Opus** in 100% of the samples (92 formats + 17 streams from 160 places in the pack): 48 kHz, 2048 B blocks
  `[u16 LE length][Opus packet]…` + zeros, 20 ms frames (960 samples; TOC 0x68/0x78 Hybrid, 0xF8 CELT, mono). Number of
  packets in a block = the difference in the seek table / 960 – verified. **No XMA2.**
- Conclusion: an XMA2 decoder has low priority for this title. Open: whether the game decodes Opus by itself (then PCM → XAudio2 already
  works), or uses the console's hardware Opus decoding (then the API is needed – check with `xcl inspect` on the exe).
  Windows has `C:\Windows\System32\MSOpusDecoder.dll` (a Media Foundation MFT) – a candidate decoder without bundling libopus.
  Registered as "Microsoft Opus Audio Decoder MFT" {63e17c10-2d43-4c42-8fe3-8d8b63e46a6a} (session 5).
- Streaming AWC (session 5, from fragments of the pack): flags 0xFF25, with bit 0 set u16 chunk indices per stream, then 12 B
  of unknown meaning; the u32 stream infos (number of chunks = bits 31:29) lie right before the chunk table, and the table ends
  at `data_offset` from the header. Chunk 0x48 = {number of blocks u32, block size u32 (512 KB), channels u32, per channel 16 B: id,
  samples, u16, frequency u16, codec u8 …}; 0xA3 = start sample of every 512 KB block. 512 KB block = 2048 B header
  (−1, 254 subblocks, 0, samples in the block, …, from +24 the start sample of every subblock) + 2048 B subblocks `[u16 length][packet]…`.
  Unexplained: 6 packets per subblock with a table difference of 12·960 – TOC 0xFC is CELT FB 20 ms **stereo**, so the table
  probably counts samples × channels.

## Title memory above 4 GB: UMA windows (session 5, 2026-11-21)
- **Gap:** shaders addressed one `mem` buffer with the low 32 bits of the address, and D3D12 opens a heap from an
  address only below 4 GiB (4095 MiB) – so at most 4 GB of title graphics memory, while a Series X title has ~13.5 GB.
- **Now** (`gpu/uma.rs`): up to 4 *windows* of ≤ 4095 MiB, window k at `base + k · 4 GiB`, `base` 16 GiB-aligned ⇒ the
  window of an address is `(addr >> 32) & 3`, the low 32 bits the offset in its heap. `Uma::new(size)` splits into as few
  equal windows as fit; `with_windows(n, size)`, `window(k)`, `window_of(p)`, `alloc_in(k, …)`; allocations never straddle
  windows. Console descriptor heaps and descriptor-table copies stay in window 0 (32-bit console addresses).
- Root signature: u2..u5 = windows 0..3 (`pso::abi_root_signature_with`, 6 root UAVs). Runtime: one heap + placed buffer
  per window (`real::UMA_HEAPS`, `uma_window_vas`, `uma_va_of`); console buffers placed in their window's heap; console
  lists bind all windows. Runner: arenas with several windows (`run_in_arena`).
- Translator: `Options::mem_windows` (from `pso::mem_windows()` = UMA windows). 1 = the single-window HLSL unchanged
  (DXIL cache stays valid). >1: every access site sets `xcl_w` from the high address bits – V# word 1 (MUBUF / MTBUF,
  buffer atomics, d16, s_buffer_load), the SGPR pair (s_load), the 64-bit VGPR address or SADDR pair (FLAT / global,
  atomics), BVH descriptor `dw0 >> 24`; `finish()` declares `mem1`.. and turns every `mem.Method(` (program and
  preludes) into an accessor `xcl_mMethod(` switching on `xcl_w` (`windowed`, `mem_accessors`).
- Config: `uma_mb` up to 16380 (default unchanged: 4095 "advanced", 2048 "standard"; `report.json` shows `uma_used_kb`).
- Tests: `windows_are_4_gib_apart_and_allocations_stay_inside`, GPU `gpu_memory_windows_above_4_gib` (s_load, buffer
  load / store, global load / store / atomic in window 1 next to window 0); DXC with 2 windows on a random 2 % of the pack
  (480 programs, seed 1795270997254744600 region): 0 errors, median 0.29 s (vs ~0.2 s with one window). Survey:
  `XCL_SURVEY_WINDOWS=N`.

## XGameRuntime through the GDK thunks (session 5, 2026-11-21)
- **Gap (blocked the start of every GDK game):** titles do not import XTaskQueueCreate / XUserAddAsync… by name – they have
  compiled-in thunks that load `xgameruntime.dll` and take `InitializeApiImpl(Ex/Ex2)`, `QueryApiImpl`,
  `UninitializeApiImpl`, `XErrorReport`; every API is a vtable slot of an object from `QueryApiImpl(clsid, iid)` (0..2 = IUnknown).
  XCL did not have this → `HCInitialize` from libHttpClient.GDK returned 0x89240101 (the game: XGameRuntimeInitialize likewise).
- **Now** (`hle/xgameruntime/api.rs`): 6 entry points. `QueryApiImpl`: XCL answers by itself for XUser (`01acd177`,
  iid `26f3c674` + `cef4fac0` = XUserGetGamertag), XGameSave (`704c3f58`), XPackage (`af406016`), XGame (`973a344e`:
  XGameGetXboxTitleId from `<TitleId>` in MicrosoftGame.Config / `title_id` in xcl.toml), XSystem (`e349bd1a`: console/sandbox
  id, the rest goes to PC), XGameUi (`dfcd4649`: only dialogs without a user handle – message, error, text – go to PC);
  it refuses XStore (`0dd112ac`) and XUser-device (`7d824997`) – the thunk then returns its own "API unavailable" error;
  it forwards the rest to the system `System32\xgameruntime.dll` (the same classes and GUIDs: 27 of 29 pairs from the console DLLs) –
  XAsync/XTaskQueue/XThread (`073b7dcb`), XNetworking, XGameInvite… After XAsync initialization XCL delegates to the PC one
  (`xasync::use_pc`; slots 3 GetStatus, 4 GetResultSize, 5 Cancel, 6 Run, 7 Begin, 9 Schedule, 10 Complete, 11 GetResult),
  because the queues and blocks then belong to PC – all XCL asyncs (`begin_with`) work unchanged. PC init: Ex2 → Ex → Impl
  (an older runtime does not know the new flags: libHttpClient passes 0x3).
- **Where the slot maps come from:** thunks in XCurl.dll / libHttpClient.GDK.dll (GUID pair, slot, argument and result widths), names
  in the slot code of the PC runtime (XAsync identifiers, `GameSaveImpl::…`), the order of the GDK headers; confirmed
  by matching argument counts and write widths (e.g. XUser 13 writes a byte = GetIsGuest, 14 a dword = GetState).
  XCL functions get the arguments without `this` (per-slot shims). Unhandled slots → E_NOTIMPL (bool → false) + log.
- **Safety:** the PC runtime does not validate foreign handles (XGameUiShowPlayerProfileCardAsync with an XCL handle →
  access violation), which is why nothing that takes an XCL XUser/XGameSave handle goes to PC.
- **Report:** `xgameruntime.apis` (pair → who answered) and `xgameruntime.unimplemented` ("XUser#23" → number of calls).
- **Not determined (needs the exe):** `0a2192aa` is a console-only API (on PC: `WindowsGRTSStub` stubs).
- Tests: `thunk_protocol_with_xcl_objects_on_pc_queues` (XUserAddAsync/Result and XGameSave async through the vtable on
  PC queues, XStore rejected, an unhandled slot in the report), `libhttpclient_initialises_through_the_xgameruntime_thunks`
  (real DLL: HCInitialize → S_OK). Without the system xgameruntime.dll: the XCL classes work, the rest is "missing".

## Tiled textures from console memory (session 5, 2026-11-21)
- **GFX10 mip chain** (`tiling::Chain`, `tiling::image_chain(&desc::Image, cfg)`; addrlib
  Gfx10Lib::ComputeSurfaceInfoMacroTiled/MicroTiled): per slice, first the mip tail block (levels ≤ half the block width,
  at most log2(block) − 4 levels; 64 KB / 4 KB blocks), then the levels from the smallest one to mip 0, slices every
  `slice_bytes`. A level in the tail: origin within the block from its offset (odd bits → x, even → y, ×256 B block).
  `detile_level` / `tile_level`. Level sizes = pixels >> i rounded up to whole elements (as in D3D12).
- **`gpu/texture.rs`**: `placement(T#)` (block-aligned address, bytes, chain, slices), `subresources(T#, memory)`
  (D3D12 order: mip + slice × mips), `uma_bytes(T#)`, `create(dev, queue, T#, memory)` → PC texture (SRV state),
  `upload(...)` – re-upload when the title rewrites the memory. `CONSOLE_PIPES` – the console's pipe/packer configuration for
  the _X modes, still unknown (the archives use modes without XOR).
- **Verified on TTPF** (`XCL_SAMPLE_TTPF`, test `retail_ttpf_textures_detile_with_their_mip_chains`): the model gives
  exactly the space occupied by 908 of 912 textures (outliers: 2 cubes in mode 2, 1 volume, the last one truncated). Number of
  mips = `dword0 >> 28` (0 → 1), slices = `(dword1 >> 10) & 0x7FF` + 1. Mips of R8/RGBA8/RGBA16 textures ≈ the previous
  level downscaled (6/6); 15 levels in the mip tail: error 0.66/255, at the wrong place in the block 150/255.
- GPU test `gpu_console_textures_upload_every_mip_and_slice`: tiled memory → PC texture → readback = the data, bit for
  bit, every mip and slice (RGBA8 4 KB array, BC7 64 KB cube).
- **To do with the exe:** binding to the API through which the title creates textures in its memory (CreatePlacedResourceX / heap
  at an address), and the moment of re-upload (CPU write / GPU copy); linear textures, 3D (thick), MSAA.

## Static samplers (session 5, 2026-11-21)
- **Gap:** the translator always took the PC sampler index from S# word 3 [11:0], but an S# built from constants in the code (the console's
  static root signature samplers) has something else there → wrong sampler / read outside the heap. In PS_PedTextured_Eight_Fire 66
  of 89 sampling operations; the e2e test only worked because those bits were 0 in its S#.
- **Now:** the `constant_samplers` pass (mod.rs) – an S# from `s_mov_b32/s_movk_i32/s_mov_b64` constants in the same block
  before the MIMG → static sampler `xcl_ss{k} : register(s{k}, space1)` (`Translation::static_samplers`,
  `Options::static_sampler_base` – PS after VS); `pso::abi_root_signature_with` + `static_sampler_desc` (from
  `desc::sampler_d3d12`; only 3 border colors); root signature cached per set (`abi_rs_for`), **PSO → root signature
  registry** (`real::pso_root_signature`) – drawing must bind this one instead of the title's; ConsolePso / TwoPassPso
  (`compute_rs`, `mesh_rs`) too. S# from descriptor tables (console heaps) – unchanged (index from word 3).
- GPU test `gpu_static_samplers_from_constants_in_code` (point 2.0 vs linear 1.5 at a texel boundary); PS from the game:
  6 static samplers, 61 uses; the PSO tests pass (PS_Rage_Textured – aniso 4× from the code); random 1% DXC: 0 errors.

## MicrosoftGame.Config (session 5, 2026-11-21)
- From the game's manifest (from the user; the rest of the file is irrelevant for XCL): `PersistentLocalStorage/SizeMB` = 2048,
  `VirtualMachine/SMTEnabled` = true, `XboxAnacondaTitleMemory` / `XboxLockhartTitleMemory` = Advanced.
- `manifest.rs` (NEW) reads these fields from the game folder; `Config::resolve` (launcher) fills in the "auto" settings:
  - `pls_mb` → XPersistentLocalStorageGetSpaceInfo (was hard-coded to 4 GiB);
  - SMT is irrelevant: **the game sees the PC's processors** (user's decision – "we're running on PC"; the version with 14 console
    processors was inconsistent: the CRT saw 14, OpenMP 16);
  - `title_memory` → `uma_mb` 4095 for "advanced" (otherwise 2048; `XCL_UMA_MB` still wins). The
    TitleMemoryStatus budget stays standard (13 824 / 8 192 MB) – the "Advanced" mode numbers are not public
    (docs.xbox.com is behind a login), `title_memory_mb` in xcl.toml overrides it.
- Test: the test manifest (`tests/data/testgame/MicrosoftGame.config`) has these values, e2e checks the launcher log and
  boot.json; `manifest::tests`.

## Instructions outside the pack, console heaps, structure (session 5, 2026-11-21)
- **ISA coverage** (`isa_coverage`, every decoder instruction separately; MIMG with DMASK=1, VOP3_SDST with VCC – previously
  false gaps): **975 / 1157 (84.3%)**, was 766. Added (GPU test `gpu_more_atomics_sub_dword_scratch_valu` +
  `image_atomic_inc` in `gpu_images_sample_load_store_atomic`):
  - a shared 32-bit atomic generator (`emit::rmw32`, `At::Mem` / `At::Ref`) for buffer / flat / global / DS / images:
    inc, dec, rsub, csub, mskor, cmpst / fcmpswap (bitwise comparison), add_f32, signed min/max on LDS and images –
    a bounded CAS loop (4096); fmin/fmax in `mem` still without a loop (the sign trick); **64-bit `_x2`**
    (`rmw64`: InterlockedXxx64 SM 6.6) – add/sub/and/or/xor/swap/min/max/cmpswap; `ds_wrxchg2(st64)_rtn_b32`;
  - flat/global/scratch byte/halfword + **d16 / d16_hi** (also buffer_*_d16, ds_read_*_d16, ds_write_*_d16_hi);
  - **scratch_*** – thread-private memory `static uint xcl_scratch[1024]` (4 KB/thread);
  - `v_sad_u8/hi_u8/u16`, `v_msad_u8`, `v_lerp_u8`, `v_div_scale/fmas/fixup_f32` (no scaling: VCC=0, fixup =
    IEEE special cases), `v_add/sub_nc_i32` with clamp, `v_mad_u32_u16/i32_i16`, `v_cvt_pk_u16_u32/i16_i32`,
    `v_cvt_pknorm_*_f32`, `v_mul_hi_i32_i24`, `v_sat_pk_u8_i16`, `v_swap_b32`; `s_memtime/s_memrealtime` (= 0);
    `buffer_gl0/gl1_inv`, `ds_nop` (no effect); `image_sample_cd*` like `_d`.
  - Still missing: f64 (sqrt/rsq/frexp/…), 64-bit LDS, GWS / append / consume, debug/trap, `s_setpc/s_call`, images
    `pck` / `by2` / `msaa_load` / `store_mip`, d16 formats (`buffer/tbuffer_*_format_d16_*`), `v_cvt_*_f16` norm.
- **Console descriptor heaps – finished** (`console_heap.rs`; test `slots_copy_and_release`):
  - **sampler heaps**: 16 B S# slots in UMA, slot k → PC sampler k, S# word 3 [11:0] = k (`#22 CreateSampler`);
  - **`#17 CreateConstantBufferView`** → a raw V# with the console address (UMA; otherwise an empty V#);
  - **`#24 CopyDescriptorsSimple` / `#23 CopyDescriptors`** – copies the PC descriptors (2 per view slot) + the slots, with
    index renumbering (T# word 0, S# word 3; V# unchanged); mixed console/PC ranges → log, PC only;
  - **heap Release** frees its slots (`heap_release`). Console increment: views 32, samplers 16.
  - #17 / #22–#24 assumed to follow the PC layout (like #14–#19 seen from console code) – to be confirmed with the exe.
- **Structure**: loops with an unconditional back jump (exit via flags + break), **`continue`** (a back jump to the
  head of the nearest loop), **entry into the middle of a loop** by a jump from earlier code (the target moves to the body level
  as pending: the code before it runs from the 2nd iteration on), a range equal to a loop (e.g. an if body) = that loop. Whole pack: programs
  with `pc` blocks **2274 → 2046 (8.5%)**. Remaining: a back jump without a cycle (the compiler shares code between paths –
  would require duplication), `continue` to an outer loop, loops with multiple back edges.

## MS (NGG) and GS pipelines in PSO creation (session 5, 2026-11-21)
- 9 282 MS + 195 GS from `shaders.rpf` (~40% of the pack) were getting replacement shaders until now (`real.rs` handled only VS+PS
  and CS). Now:
  - **`CreatePipelineState` with a stream containing a console MS** (`translated_mesh_stream`) and **`CreateGraphicsPipelineState` with a console
    VS + GS** (`translated_gs_pso`) → `pso::console_mesh` / `pso::console_geometry` → **`pso::TwoPass`**: a compute
    pass (the console program, one record per group in `res`) + a pass-through mesh shader with the translated PS. The title gets
    a mesh PSO (state from the stream / description: blend, rasterizer, depth, formats; RS = XCL ABI); the compute PSO and the record
    layout live in `real::two_pass_of(pso)` (`TwoPassPso`: compute, record, mesh_groups_per_record, kind).
    Console code never reaches the driver (failed translation → E_INVALIDARG). AS – not yet (none in the pack).
  - **Pass-through MS for a specific PS** (`emit::mesh_passthrough_for`): outputs = exactly the inputs of the translated PS
    (the same elements, order and types `{type}4` → identical signature packing, D3D12 links them); values from the record slot
    with the same semantic, missing ones = 0 – **in MS+PS pairs from one effect the signatures do NOT match** (the PS
    reads e.g. TEXCOORD20, which the MS does not write, and vice versa; on the console the driver links attributes by index). Clip
    distances are kept. No `SV_CullPrimitive` (bad triangles are degenerate).
  - **D3D12 limit: 32 KB of output per mesh group** – large PSs (17 attributes × 16 B × 128 vertices) exceed it →
    **split mode** (`emit::passthrough_split`): a record goes into N mesh groups, each with a stretch of the triangles and its own
    copies of 3 vertices; at draw time `DispatchMesh(groups · mesh_groups_per_record)`.
  - Tests: `retail_mesh_pipelines_through_the_hook` (MS+PS pairs with the same pass hash in the name, sample
    `XCL_MS_EVERY`, every 1500th by default; every 700th = 14/14), `retail_gs_pipeline_through_the_hook`, `gpu_mesh_mode_two_passes`
    (split mode = the same pixels). The `gpu_pso_e2e` tests are serialized through `setup` (shared PSO counters).
  - **Note:** `gpu_pso_e2e` regression runs without `XCL_SAMPLE_RPF` finish in 0.00 s (skipped) – always run them with the sample.
- **To do with the exe: drawing.** On DispatchMesh (MS) / Draw* (GS) with such a PSO: first `compute` with the ABI RS (io =
  SGPRs from the root arguments, `res` = a record buffer of groups · `record.record_dwords()` dwords, `mem` = UMA), a UAV barrier,
  then the mesh PSO with `DispatchMesh(groups · mesh_groups_per_record)`; for GS additionally fetching the ES vertices into `mem`
  (IO_GS_PRIMS / IO_GS_VERTICES as in `retail_es_gs_voxelize_geometry_mode`). D3D12.X draw records – from the exe.

## Translator survey over the whole pack (session 5, 2026-11-21)
- Test `retail_translation_survey` (ignored; `XCL_SAMPLE_RPF=…\shaders.rpf`, `XCL_SURVEY_EVERY=N`, `XCL_SURVEY_DXC=1`):
  - **translation: 24 064 / 24 064 FSSM variants in full** (PS 12 101, MS/NGG 9 282, VS 2 236, GS 195, ES 195,
    CS 55), 28.97 million instructions, 0 missing instructions, 0 errors, 0 panics; 110 s (debug, all cores);
  - **DXC (every 100th record, 241 variants): 0 errors**; time median 0.26 s, p90 3.0 s, max 9.1 s
    (PS_PedTextured_Eight_Fire), total 224 s → **whole pack ≈ 6 h of CPU** (on the order of 30 min on all cores, if
    there is enough memory). The first game start without a ready DXIL cache would be very long → background compilation +
    a cache are needed (`dxc::compile_cached`, key = HLSL), or filling the cache from `shaders.rpf` beforehand.
  - **Random 1% samples** (`XCL_SURVEY_SEED=<n>|random`, the seed is printed): seed 1795262526659555600 found a DXC
    error – `image_sample_d(_cl)` on a **cube texture**: RDNA supplies 2D gradients in the face's coordinate system, HLSL
    `TextureCube.SampleGrad` wants 3D and has no version with clamp → now LOD = log2(max |grad| · face size) (with clamp)
    and `SampleLevel` (an approximation; the face gradient scale is to be confirmed in the game). After the fix this seed and a new one
    (1795262814426998400): 0 errors. The every-100th-record sample did not hit this – also check randomly.
  - Compile MS (NGG) and GS/ES as **cs_6_6** (they run as compute + a pass-through mesh shader; the ms/gs profile →
    "WaveSize only allowed on compute").
- **DXC time optimization (session 5, second round)** – comparison on the same sample (every 100th record, 241 variants), old
  / new code one right after the other: total **228 → 152 s (−33%)**, median 0.29 → 0.22 s, p90 2.76 → 2.29 s, max 9.3 → 5.9 s;
  slowest PS (PS_PedTextured_Eight_Fire, 8 600 instructions) ~6.5 → 3.4 s, MS (MSNC_VS_PedTransformSkinAlpha) 3.8 → 1.7 s.
  Time measurements taken at different moments jump by ±20% (machine state) – compare only side by side.
  - `dxc.exe -ftime-report` (SDK) showed: at -O0 DXC still takes ~4 s (so it is not the optimizer), at -O3 the most expensive
    was **Jump Threading** (25%) on the `pc` loop, and in MS shaders **loop unrolling** from helper functions.
  - **Structure:** forward jumps leaving a block (break out of a loop, a jump past an exec join, s_endpgm in the middle – discard
    in PS) as **`xcl_skip_<target>` flags**; loops entered in the middle (`s_branch` right before the head, also via `s_nop`) in
    rotated form; if/else only when no pending target falls into the middle (otherwise a plain if / flag); analysis and emission
    go through the same function `item_at`, analysis with memoization (`memo`). If/loop depth ≤ 24 (`MAX_NEST`; discard chains
    in PS produced hundreds of nesting levels). Shaders with `pc` blocks: 102 → 23 / 241 (remaining pattern: a loop exit
    placed in the middle of the loop and a back jump to it).
  - **Loop-free and branch-free HLSL helpers** (HLSL 2021 turns scalar `?:`, `&&`, `||`, `if` into jumps):
    `fmt_comp`/`fmt_bytes`/`bits_at`/`uf_small` (buffer format decoding – 140 times each in an MS), `mul_legacy`
    (1 200 times in a PS), `ballot`, `first_lane`, `cvt_u32/i32`, `bfe_*`, `sbfe_*`, `ffbh_i`, `alignbit`, `img_cls`,
    `dst_sel`, `f16_rtz`. Note: `unorm`/`snorm` are HLSL keywords (do not name variables that way).
  - **DXC on its own threads** (`dxc.rs`: a pool, one thread per core, 64 MB stack, its own compiler): DXC's LLVM passes
    recursively overflowed the 2 MB stack on large PSs (PS_TexturedLit_8); game threads may have a small stack.
  - Unsuccessful (no gain, within noise): computing `lanebit(exec)` once per exec change, removing comments from the HLSL.
  - Helpers: `retail_shader_cost` (`XCL_NAMES=a,b`, `XCL_DXC_ARGS`, `XCL_HLSL_OUT`, 8 GB limit) and, in the survey,
    `XCL_SURVEY_TRACE`, time split structured / with `pc`, the 10 slowest, the smallest unstructured ones.
- The game **does not use FSPX/ACCC**: all 23 869 shaders in `shaders.rpf` are XDAT/XBCH/FSSM – FSPX is only FSR4's own
  pipelines (irrelevant for the game, possibly a test).

## Console shaders – open questions (from the analysis of `shaders.rpf`)
- Unrecognized FSSM metadata: +0x10 (always 6), +0x78/+0x80, +0xD0, +0xF8, +0x110 (resembles SPI_SHADER_PGM_RSRC1,
  but the bits do not match), after the code a table of 5×u32 entries (3, PC every 8 B, 1, value, 8 – looks like a PC → source
  line mapping), then the shader name; in PS from approx. +0x238 an export patch table (`exp` variants for RT formats). For user SGPRs
  and inputs a cross-reference with ISG1/OSG1/PSV0 is needed (those parts are readable – the signature parser is in `rpf-viewer.html`).
- GS: GS + VS descriptors (copy shader), but FSSM contains only one program – is the copy shader created outside FSSM?
- VS: at the start `s_swappc_b64 s[4:5], s[4:5]` = fetch shader call (vertex attributes → v4…, VS_BlitGenAmbCube
  confirms POSITION.xy in v4/v5); in VS mode the recompiler has to replace it with reading the inputs according to ISG1 (the VGPR layout
  for multiple attributes to be determined on the corpus: which v≥4 are read before being written vs the ISG1 element masks).
- VOP2 0x06 / VOP3 0x140: in gfx10.1 MAC/MAD_LEGACY (unfused), in gfx10.3 FMAC/FMA_LEGACY – unknown which one the console has
  (the decoder uses the names from the gfx10.3 spec).
- Some names in the pack are truncated (names > 255 characters do not fit in the record's length byte) – RPF9 scanner cosmetics.

## Next steps (priority)
Little is left to do without the exe – the rest needs an actual run of the game.
1. Once the user has a legal game exe: `xcl inspect`, `xcl run`, analysis of `report.json`
   (`called_stubs` with `(not on PC)` → mapping of console slots, e.g. PresentX; `d3d12.unknown_iids`; `xgameruntime.apis` / `.unimplemented` (thunk classes and slots
   to add: PLS, XSystemGetDeviceType);
   `shaders.rdna2_mnemonics`; `audio` – voice formats, `unsupported_formats`; imports related to Opus/XMA/AcpHal).
2. Draw-time binding: the game's root arguments → SGPRs in `io`, `mem` → UMA buffer; command list capture for GS
   and Compute dispatches; `XboxBvhBuilder` (or `BuildRaytracingAccelerationStructure`) → `gpu::bvh::build/build_top`
   (geometry from GPU buffers in UMA; Rehydrate = prebuilt BVHs from `*_bvh4.rpf` – format to be checked).
3. Textures: binding `gpu::texture` to the title's API (textures in its memory, re-upload after a write), the console's
   pipe/packer configuration for the _X modes (`texture::CONSOLE_PIPES`), linear / 3D textures.
4. Audio: Opus (the MFT above) for XAudio2 0x704F voices, or the console API – depending on what the exe shows; XMA2 low priority.
5. Smaller: 749 wave64 programs at risk in a 32-wide wave (only if they show up), `image_sample_o` with an offset from a VGPR,
   smaller HLSL, further XGameRuntime APIs (XGameUI, XNetworking…), GameInput → system GameInput.dll,
   mapping physical pages at the game's address.
