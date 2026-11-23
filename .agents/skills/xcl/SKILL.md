---
name: xcl
description: >-
  Index for working on XCL, the Xbox Series X|S to Windows compatibility layer.
  Use when starting work in this repo, choosing which subsystem to change, or
  when the user mentions XCL, kernelx, D3D12.X, RDNA2, a pull request, or running
  a Series title on Windows. Read the matching granular skill before editing
  that subsystem. Contributor and agent rules: CONTRIBUTING.md and AGENTS.md.
---

# XCL

Native x64 compatibility layer. The title runs in a child process (`XCL_HOST`,
`host.rs`). XCL maps the PE, resolves imports, and recompiles RDNA2 to HLSL
then DXIL. No CPU emulation. Design: `docs/specs/2026-11-20-xcl-design.md`.
User-facing status: `README.md`. `HANDOFF.md` is a session log.

These skills live only in `.agents/skills/`. Codex and Cursor load that directory. Claude Code does not; `CLAUDE.md` tells it to read the same files. Do not copy them elsewhere.

Repo language is English. License is GPL-2.0-only. No SDK headers or binaries,
no game files, no DRM or package decryption. The game folder is read-only;
writes go to `%LOCALAPPDATA%\XCL\<title>\` (`%XCL_DATA_ROOT%` overrides).

## Which skill

Open that directory's `SKILL.md` before editing the subsystem.

| Work | Path |
|---|---|
| First intro cutscene | `.agents/skills/xcl-cutscene/SKILL.md` |
| `report.json`, tests, samples | `.agents/skills/xcl-report/SKILL.md` |
| PE map, TLS, exceptions, launch | `.agents/skills/xcl-loader/SKILL.md` |
| Imports, stubs, ordinals | `.agents/skills/xcl-imports/SKILL.md` |
| Paths, overlay, DirectStorage | `.agents/skills/xcl-vfs/SKILL.md` |
| Memory, threads, console info | `.agents/skills/xcl-kernelx/SKILL.md` |
| GDK thunks, user, saves | `.agents/skills/xcl-xgameruntime/SKILL.md` |
| Device, lists, present | `.agents/skills/xcl-d3d12/SKILL.md` |
| RDNA2 decode and HLSL | `.agents/skills/xcl-shaders/SKILL.md` |
| GFX10 tiling, descriptors | `.agents/skills/xcl-textures/SKILL.md` |
| XAudio2, XMA, Opus | `.agents/skills/xcl-audio/SKILL.md` |
| Gamepad, keyboard | `.agents/skills/xcl-input/SKILL.md` |

## Shared rules

- One concrete startup or playability blocker at a time. Reproduce it on current `main` and name the furthest screen before writing code. What a report must include is in `xcl-cutscene` and `xcl-report`.
- A missing import is a logging stub (`stubs.rs`, `crates/xcl/data/stubdb.toml`), not a load failure. A stub that returns success so a title gets past a screen it has not actually reached is a speculative stub. Do not add one.
- Do not invent console layouts. Use public docs, existing tables, or a user-supplied report.
- Do not hand-edit generated files: `gpu/rdna2_spec.rs` (`scripts/gen_rdna2.py`), `gpu/tiling_tables.rs` (`scripts/gen_tiling.py`), `hle/d3d12_layout.rs` (`scripts/gen_d3d12_layout.py`), `hle/xgameruntime/gdk_table.rs` (`scripts/gen_gdk_table.py`). Each header says so.
- WinDurango is reference only. AnyPS5 ports need a provenance note (`THIRD_PARTY.md`).
- Edit with the file tools. PowerShell `Set-Content` corrupts UTF-8.
- Do not rewrite `HANDOFF.md` or `docs/specs/` in a feature change. If a statement there is now wrong, say so in the PR description. Update `README.md` when user-visible behavior changes.
- Do not run `cargo fmt` on a file or the crate. Format only the lines you change.
- One logical change. Under ~400 changed lines, excluding generated files and test data. A second bug is a second PR. Search open PRs before starting (`gh pr list -R ivall/xcl --search "<keywords>"`).
- Commit subjects are `<Area>: <what it does>`, about 72 characters, matching `git log` (`VFS`, `D3D12.X`, `Recompiler`, `kernelx`, …). No `fix:` / `feat(x):` prefixes. Commit only when the user asks. Do not push or open a PR unless they ask in this session.
- Local check while iterating: `cargo test -p xcl --lib`. Before calling the suite done: `cargo test --workspace -- --test-threads=4` and report passed / failed / ignored / `SKIPPED` from that run. `.\scripts\test.ps1` is the same build plus tests. Do not `--compile` thousands of shaders; `--compile 64` is the spot check.
- A runtime change (`crates/xcl/src`, not tests) needs `.\scripts\smoke.ps1 -Game <folder>` from `main` and from the PR head with a clean tree. Paste both `xcl-smoke` blocks as the script printed them. Do not write or edit a block by hand. `PR checks` rejects a missing block, a dirty tree, a stale commit, and any device removal or crash.
- GPU tests that cannot run call `gpu::recomp::runner::skip_test` and return. Do not `#[ignore]` a test to make it pass.
- CI skips `test_title_runs_and_game_folder_is_untouched` (no Gaming Services, no real GPU). That test still matters on a PC.
- Disclose AI use in the PR **AI use** section: tool, model, and what it did. Do not invent the model name.
