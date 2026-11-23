---
name: xcl-d3d12
description: >-
  Change XCL's D3D12.X implementation: device, heaps, command lists, pipeline
  state, present, and UMA. Use when a title has no picture, report.json d3d12
  shows unknown slots or IIDs, or when editing hle/d3d12x/.
---

# D3D12.X

`hle/d3d12x/mod.rs`. Exports include `D3D12XboxCreateDevice` and `IID_ID3D12Device` as data.

`real.rs` is the default: PC D3D12 objects with an extended vtable. Driver vtables have internal methods past the public ones, so `guard.rs` checks the caller: a call from the title image is a console slot (log and return 0), anything else goes to the driver. `null.rs` is selected by `graphics = "null"` or when no device is created. `real::init` is idempotent.

Console recording lives in `console_list.rs`, `list_x.rs`, `device_x.rs`, `console_heap.rs`, `bind.rs`, `frame.rs`, `fence_x.rs`, `vram.rs`, `sparse.rs`, `title_textures.rs`. Public COM layout is generated: `scripts/gen_d3d12_layout.py` → `hle/d3d12_layout.rs`.

Graphics PSO creation calls `gpu/recomp/pso.rs` for native VS/PS (and the mesh/GS paths the backend already hooks). Failure falls back to replacement shaders. Counters: `translated_pipelines`, `translation_errors`, `fallback_pipelines`.

`present.rs`: XCL window and swap chain. `present` config is `auto` or `off`. `d3d12.presents` increments per present, including empty frames. `XCL_FRAME_DUMP=N` fills `last_frame_dump` (every 4th pixel unless `XCL_FRAME_DUMP_FULL=1`).

`CheckFeatureSupport` raytracing tier follows `raytracing` / `XCL_RAYTRACING` (`auto`, `tier1_1`, `tier1_0`, `off`). Adapter: `d3d12_adapter` / `XCL_ADAPTER`. Debug layer: `d3d12_debug` or `XCL_D3D12_DEBUG=1` (needs the Agility SDK layers on Windows 10; see README).

A new console method needs a slot test or an e2e draw (`crates/xcl/tests/gpu_pso_e2e.rs` pattern): translate, draw, read back pixels. Do not treat a present count as success.
