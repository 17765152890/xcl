---
name: xcl-kernelx
description: >-
  Implement Xbox kernelx memory, threads, modules, and console identity in XCL.
  Use when editing hle/kernelx.rs, kernelx/lazy.rs, kernelx/placeholders.rs,
  or when a title fails on VirtualAlloc, title memory, or MapTitlePhysicalPages.
---

# kernelx

Module `kernelx.dll` in `hle/kernelx.rs`, with `lazy.rs` (commit on touch) and `placeholders.rs`. Routing in `hle/mod.rs` also sends `kernel32`, `kernelbase`, `advapi32`, `*legacy`, `minappmodelx`, and `api-ms-win-core-*` / `ext-ms-win-core-*` here. Unimplemented names fall through to `kernelbase` / `kernel32` / `advapi32` / `sechost`.

Title memory budget comes from `MicrosoftGame.Config` (`manifest.rs`) unless `xcl.toml` sets `title_memory_mb`. `device_type` is `anaconda` (Series X, 13824 MiB standard) or `lockhart` (Series S, 8192). `title_memory` is `auto`, `standard`, or `advanced`. `console_type_raw` is an override for `GetConsoleType`; the Series value is not public — do not invent a new constant without saying it is a guess.

`VirtualAlloc` (`MEM_GRAPHICS` is `0x1000_0000`):

- Commit with no fixed address: `d3d12x::sparse` first, then `gpu/uma` alloc.
- Reserve with no fixed address: a sparse window that uses no system RAM until commit.
- `MEM_64K_PAGES` (`0x2040_0000`): 64 KiB pages in a sparse window.
- Commit inside an existing sparse reservation: `commit_sparse`.
- Anything else: Windows `VirtualAlloc`.

`XMemAlloc` with `memory_type != 0` sets `MEM_GRAPHICS` and follows that path. UMA is GPU-visible in 4 GiB windows; the low 32 bits of a console address are the offset in the window. Default size is 2048 MiB, or 4095 for advanced title memory; `uma_mb` and `XCL_UMA_MB` override. `OpenExistingHeapFromAddress` rejects a partial commit and a 4 GiB region.

`placeholders.rs` runs before those branches. A placeholder that returns success without backing pages will corrupt later GPU reads. `MapTitlePhysicalPages` must use the title's addresses.

Thread APIs schedule on the host's processors. Keep the host's TLS callback path (`loader::on_tls_event`) working for threads the title creates.
