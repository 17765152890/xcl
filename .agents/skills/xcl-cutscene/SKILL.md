---
name: xcl-cutscene
description: >-
  Advance a legally unpacked Xbox Series title through XCL until the first intro
  cutscene plays. Use when the user names a retail title, a first-boot goal, an
  intro movie, or GTA, and the work is compatibility (imports, files, GPU,
  video, audio), not obtaining or decrypting the game.
---

# First intro cutscene

XCL is general. Do not add title-specific forks, hardcoded paths, or game
assets. A milestone for one title is reached by fixing the subsystem its
`report.json` shows is blocking it, with a regression test that does not need
that title's files.

The user owns the unpacked files and runs them. Do not decrypt packages, strip
signatures, or discuss how to obtain a build. If the input is not a normal x64
PE, `xcl inspect` already reports that; stop there.

## What "cutscene" requires

In order, each step is a different skill:

1. `xcl inspect` is static (`xcl-imports`). A crash before the entry point is `xcl-loader`. `called_stubs` only exists after `xcl run`.
2. Archives open through the VFS or `xcrdapi` (`xcl-vfs`). Read `vfs.unmapped_paths` and `dstorage` before touching shaders. BCPack refusal still fails the request when the destination is filled.
3. Title memory and threads (`xcl-kernelx`). The README's large-title note is ~15 GB committed in menus on a 32 GB PC. That is one game's menus, not a budget for this milestone.
4. A device and `PresentX` (`xcl-d3d12`). `d3d12.presents` counts presents, including empty ones.
5. The movie is the title's own codec DLL (Bink 2 already runs as a game DLL) plus its display shaders and a voice (`xcl-shaders`, `xcl-audio`). Do not special-case shader names from one title.
6. `XCL_FRAME_DUMP=N` sets `d3d12.last_frame_dump.non_dark_samples`. A non-dark sample is a hint that something was drawn.

Offline services stay offline. A title that blocks the intro on Xbox Live is reported as such; do not fake a store license or a multiplayer session (`xcl-xgameruntime`).

## Loop

1. Read the latest `%LOCALAPPDATA%\XCL\<title>\logs\report.json` (and `xcl.log`). Previous run is `report.prev.json`.
2. Rank blockers in that order: `crash`, `called_stubs` by count, `vfs.unmapped_paths`, `dstorage`, `d3d12` unknown slots and IIDs, `shaders` missing mnemonics, `xgameruntime` unimplemented slots, `audio` formats played as silence.
3. Change one subsystem per iteration, as its own PR once the user asks to open one. Add or extend a test under `crates/xcl` or `tests/testtitle` that does not need the title's files.
4. The title check is `scripts/smoke.ps1`, from `main` and from the commit under test. The user runs it on their files. Quote the printed `xcl-smoke` block. Do not summarize it into new numbers.

Do not claim a title reached the cutscene without that run. This environment does not have retail game data. A smoke run that never reaches the code you changed still counts as a regression check; name what it does not cover under **Not tested**.
