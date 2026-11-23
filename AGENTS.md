# Agent Guide & Autonomous Contributor Protocol

This guide defines operational rules and constraints for autonomous coding agents, swarm workers, and
automated contributors working on the **XCL** codebase (`D:\xcl`).

---

## 1. Cardinal Directives (Hard Constraints)

1. **Zero Game Data**: Never extract, stage, commit, or reference proprietary retail game assets, executables,
   shaders (DXBC, DXIL, FSSM, disassembled code, or raw RDNA2 byte sequences), textures, or audio in commits,
   test fixtures, or comments.
2. **Zero Cryptographic Keys or DRM Bypasses**: No encryption keys, DRM circumvention algorithms, or proprietary
   console SDK dumps.
3. **Pure Synthetic Verification**: Build tests using programmatic assemblers (`rdna2::encode`), synthetic buffers,
   or the in-repo synthetic console test title (`tests/testtitle`) and DLL (`tests/testgamedll`).
4. **Fix, Don't Disable**: Never introduce skip-lists to mask failing tests or unhandled instructions. Implement
   correct opcode decoding, bounds clamping, or fallbacks.
5. **Deterministic File Writes**: Write files with explicit UTF-8 encoding and standard Unix LF (`\n`) line endings.
   Avoid PowerShell cmdlets (`Set-Content`, `Out-File`) that introduce BOMs or mangle non-ASCII characters.

---

## 2. Swarm Message Board Contributor Etiquette

When participating in multi-agent swarms:

- **Targeted Addressing**:
  - Always address peers directly by name: `@verify-<topic>`, `@author-<topic>`, `@tracker`.
  - **Do NOT CC or mention `@coordinator` on intermediate peer-to-peer handoffs or progress updates**.
  - Tag `@coordinator` ONLY when your final deliverable is completely tested and ready for merge, or if you hit
    an unresolvable architectural conflict.
- **Zero Token Waste**:
  - Do not post conversational pleasantries or empty acknowledgments ("hello", "ready", "understood", "starting now").
  - Start posts directly with claims, actionable queries, or completion summaries.
- **Actionable Status Reports**:
  - A post that merely says "done" is useless. Always include:
    1. Git branch name and commit hash.
    2. Modified files.
    3. Exact test command(s) executed and output summary (e.g. `test result: ok. 12 passed; 0 failed`).
    4. Sabotage validation results.
- **Reactive Idling**:
  - Never poll the board or repository in a busy loop.
  - End your turn with reactive waiting (`swarm_wait`) to sleep until an `@mention` or notification arrives.

---

## 3. Git Worktree Fleet Management

To prevent index lock collisions (`.git/index.lock`) across concurrent agents:

1. **Worktree Isolation**:
   - Never run simultaneous git mutations (`checkout`, `commit`, `rebase`, `add`) inside the main repository tree
     if another agent is active there.
   - Provision an isolated worktree for your task:
     ```powershell
     git worktree add D:\xcl-<feature-slug> -b <branch-name> origin/main
     ```
2. **Branch Exclusivity**:
   - A branch can only be checked out in one worktree at a time.
   - Always base new branches on `origin/main`.
3. **Branch Naming Standard**:
   - Follow the `<type>/<short-kebab-slug>` convention:
     - `fix/<slug>`: bug fixes, bounds guards, cleanup.
     - `feat/<slug>`: feature additions, HLE shims.
     - `docs/<slug>`: documentation and specification updates.
     - `test/<slug>`: synthetic test suites and fixtures.
     - `refactor/<slug>`: structural refactoring.
4. **Clean Teardown**:
   - After verification and PR submission, remove temporary worktrees and prune metadata:
     ```powershell
     git worktree remove D:\xcl-<feature-slug>
     git worktree prune
     ```

---

## 4. Test Execution & Repo Layout Reference

### 4.1 Repository Layout
- `Cargo.toml`: Workspace configuration (`crates/xcl`, `tests/testtitle`, `tests/testgamedll`).
- `crates/xcl/`:
  - `src/loader.rs`: PE mapping, imports, TLS callbacks, relocations.
  - `src/pe.rs`: PE headers and bounds checking.
  - `src/tls.rs`: Carrier DLL generation for static TLS.
  - `src/hle/kernelx.rs`: Console memory allocations, threads, physical pages.
  - `src/hle/xgameruntime/`: GDK runtime shims (XTaskQueue, XUser, XGameSave, PLS).
  - `src/hle/d3d12x/`: Direct3D 12.X console graphics (`real.rs`, `guard.rs`, `frame.rs`, `sparse.rs`).
  - `src/hle/xaudio2.rs`: Audio HLE and XMA/Opus voice emulation.
  - `src/gpu/rdna2.rs`: RDNA2 instruction decoding.
  - `src/gpu/recomp/`: RDNA2 to HLSL recompiler (`emit.rs`, `pso.rs`, `dxc.rs`, `runner.rs`).
  - `src/vfs.rs`, `src/vfs_hooks.rs`: Copy-on-write virtual filesystem overlay.
- `tests/testtitle/`: Synthetic console title (`xcl-testtitle`, `tests/testtitle/src/main.rs`).
- `tests/testgamedll/`: Synthetic companion game DLL (`tests/testgamedll/src/lib.rs`).
- `scripts/test.ps1`: Primary build and test script.
- `docs/specs/2026-11-20-xcl-design.md`: Core architectural specification.

### 4.2 Standard Verification Commands

```powershell
# Compile entire workspace
cargo build --workspace

# Run full test suite with test runner
.\scripts\test.ps1

# Run focused tests on a single subsystem
cargo test -p xcl -- <module_or_test_filter>

# Build the synthetic console title
cargo build -p xcl-testtitle

# Run CI-equivalent test execution (no GPU / no retail samples)
.\scripts\test.ps1 --skip test_title_runs_and_game_folder_is_untouched
```

### 4.3 Opt-in Game Sample Variables
If testing against personal retail dumps, these environment variables enable specific test fixtures. When unset, tests skip cleanly:
- `XCL_SAMPLE_RPF`: Shader pack RPF.
- `XCL_SAMPLE_DLLS`: Console DLL directory.
- `XCL_SAMPLE_TTPF`: Texture archive.
- `XCL_SAMPLE_BK2`: Bink 2 movie.
- `XCL_SAMPLE_BVH4`: Prebuilt BVH package.
- `XCL_SAMPLE_AWC`: Audio archive.
- `XCL_SAMPLE_XG`: XG container.
- `XCL_SAMPLE_DXC`: Console DXC toolchain.

---

## 5. Mandatory Pull Request Description Format

All PR descriptions must strictly follow this structure:

```markdown
## Summary
<Concise overview of what was implemented or resolved, including Issue/Item #>

## Changes
- <Bullet 1>
- <Bullet 2>

## Tested
- [x] Workspace build: `cargo build --workspace`
- [x] Subsystem tests: `cargo test -p xcl -- <filter>`
- [x] Full test script: `.\scripts\test.ps1`
- Test Environment:
  - OS: Windows 11
  - GPU: <GPU Model or 'No GPU / CPU synthetic only'>
- Results: <N passed, 0 failed>
- Sabotage Check: <Confirmed test fails when logic is reverted>

## Not Tested
- <Explicitly state unverified paths, skipped GPU tests, or unset opt-in sample variables>

## AI Use
- [x] AI-assisted development
- Details: <Brief statement of agent contributions>
- Git commit trailers:
  Co-Authored-By: <Agent/Model Name> <noreply@anthropic.com>
```
