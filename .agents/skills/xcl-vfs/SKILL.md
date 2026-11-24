---
name: xcl-vfs
description: >-
  Change XCL path translation, the copy-on-write overlay, or console
  DirectStorage (xcrdapi). Use when a title cannot open archives, report.json
  vfs.unmapped_paths is set, or when editing vfs.rs, vfs_hooks.rs, rpf.rs, or
  hle/dstorage.rs.
---

# Files

`vfs.rs` decides; `vfs_hooks.rs` patches KERNELBASE so CRT and Win32 opens hit that decision.

Mounts (`xcl.toml`, `{game}` and `{data}`): `GAME:`, `G:`, `D:` map into the game folder (`Target::Game`). `S:\` is persistent local storage (`pls\`), `T:\` is `temp\`. `Target::Pass` is a host path left alone.

Writes to a game file copy that file into `<data>\overlay\` and modify the copy. Deletes write a whiteout. `Decision::UseWithDisposition` rewrites `TRUNCATE_EXISTING` so the overlay file is not published before the open succeeds. A test that leaves the title directory changed is a failure.

`xcl rpf` lists or extracts readable RPF9 records (`rpf.rs`, `rpf_probe.rs`). Archives over 1 GB use a bounded header probe; `--probe` is summary-only (no `--find`, `--extract`, `--out`, `--full`). Encrypted TOC or Oodle payloads stay unread. The title unpacks its own archives; do not add a decryptor.

## DirectStorage

`hle/dstorage.rs` is `xcrdapi.dll` (`DStorageGetFactory`). Layouts in that file header were measured on a shipping title: factory, queue, request (`0x48`), file, status array. The worker reads files through the VFS, copies memory, and inflates zlib. A destination page array entry that is 0 at `EnqueueRequest` is read again when the request runs, because titles fill that entry after enqueue. An array that already had its pages is kept as snapshotted. An entry that is still 0 at that later read is `E_ACCESS_VIOLATION`. BCPack is refused (`E_DECOMPRESSION`), counted in `report.json` `dstorage` by mode, and logged once per mode. `XCL_DSTORAGE_BCPACK_FILL=1` writes solid BCn blocks into the destination so the title's wait shows up, and the request still fails. Methods not listed return `E_NOTIMPL`.

A cutscene blocked on asset load is usually a refused request or an unmapped path, not a shader bug. Fix the request the report names, and add a unit test that feeds a buffer through the queue without a retail archive.
