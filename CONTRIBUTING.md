# Contributing to XCL

Thank you for your interest in contributing to XCL!

XCL is an open-source Xbox Series X|S (GDK) to Windows compatibility layer running natively on x64
without CPU emulation, translating console RDNA2 GPU bytecode to HLSL/DXIL, and shimming console
runtime libraries (`kernelx.dll`, XGameRuntime, `d3d12_xs.dll`, `xaudio2_9.dll`).

All contributions must adhere to our legal boundaries, synthetic testing practices, swarm coordination
etiquette, and pull request standards.

---

## 1. Hard Rules & Legal Boundaries

These rules are absolute and must be followed without exception:

### 1.1 No Game Data, Ever
- **Never commit, stage, or quote copyrighted retail game assets** in any form:
  - No archive files (`.rpf`, `.ttpf`, `.awc`, `.bk2`, `.bvh4`, etc.).
  - No executable binaries (`.exe`, `.dll`) or carved binary blobs.
  - No shaders in any form (DXBC, DXIL, FSSM, disassembled listings, raw RDNA2 byte sequences, or SPIR-V).
  - No game textures, audio streams, movies, or models.
  - No logs, crash dumps, or test fixtures quoting proprietary asset contents.
- Any unit or integration test requiring shader or binary inputs must construct them **synthetically**:
  - Assemble machine code programmatically using our in-tree assembler (`rdna2::encode`).
  - Use crafted mock buffers, headers, or synthetic PE images.
  - Use our in-tree test binaries (`tests/testtitle` and `tests/testgamedll`).
- Local testing against real retail samples is strictly **opt-in** through local developer environment variables (see [Synthetic Testing Guidelines](#3-synthetic-testing-guidelines)). When unset, tests must cleanly skip (`SKIPPED <test>: <reason>`) and never fail.

### 1.2 No Keys, Decryption, or DRM Circumvention
- Do not commit or implement encryption keys, decryption algorithms, license spoofing, or DRM circumvention routines.
- If a console format or container is encrypted, document the specification limitation in `docs/specs/` and halt.
- XCL is designed to run unpacked, owned files under clean-room High-Level Emulation (HLE).

### 1.3 Clean-Room Implementation & License Compliance
- The codebase is licensed under **GPL-2.0-only** (see `LICENSE`).
- All dependencies must be compatible with GPL-2.0.
- No proprietary console SDK headers or binaries are permitted. WinDurango serves only as a behavioral reference; no code is ported from it. AnyPS5 logic may be ported under GPL-2.0-only with provenance notes.
- **Fix, do not disable**: Do not introduce skip-lists or silent bypasses to mask failing edge cases. Implement proper boundary checks, error codes, and graceful fallbacks.

---

## 2. Swarm Contributor Etiquette & Autonomous Fleet Workflow

When multiple contributors or autonomous swarm agents collaborate in the repository, follow these coordination directives to prevent conflicts and token waste:

### 2.1 Git Worktree Fleet & Index Lock Contention
- **Zero index-lock contention**: Never run conflicting git mutations (`checkout`, `commit`, `rebase`, `add`) concurrently inside the same working tree.
- When working on independent tracks in parallel, provision dedicated Git worktrees:
  ```powershell
  git worktree add D:\xcl-<feature-slug> -b <branch-name> origin/main
  ```
- **Branch exclusivity**: A git branch can only be checked out in a single worktree at any given time.
- Clean up completed worktrees and prune stale references after pull requests merge:
  ```powershell
  git worktree remove D:\xcl-<feature-slug>
  git worktree prune
  ```

### 2.2 Task Claiming & Coordination
- Before starting work, check open issues and pull requests to ensure the item is not already claimed or under active review.
- Announce or claim your track with your intended branch name and item reference before writing code.
- If you discover secondary bugs or follow-ups outside your current scope, document them as separate findings rather than expanding scope unchecked.

### 2.3 Peer-to-Peer Communication & Board Etiquette
- Address messages directly to the specific peer or verifier: `@verify-<topic>` or `@author-<topic>`.
- **Do not CC or tag `@coordinator` on intermediate peer-to-peer handoffs**: Tag `@coordinator` only when the final deliverable is ready for integration, or upon an irrecoverable blocker requiring escalation.
- **No conversational filler**: Omit greetings ("hello", "ready", "working on it"). Every message is broadcast to the swarm.
- **Post actionable results**: State the branch name, commit hash, exact commands run, pass/fail counts, and diff summary.
- **Event-driven waiting**: Do not poll or query status in tight loops. Use reactive sleeping (`swarm_wait`) to await notifications.

---

## 3. Synthetic Testing Guidelines

Testing compatibility layers without copyrighted game assets requires rigorous synthetic verification.

### 3.1 Repository Test Architecture
The repository consists of the following test layers:
- **Unit Tests**: Fast, isolated tests within `crates/xcl/src/` testing algorithms, parsers, and bounds logic.
- **Integration Tests**: End-to-end integration tests in `crates/xcl/tests/`:
  - `gpu_pso_e2e.rs`: Pipeline state creation, resource bindings, draw execution.
  - `console_dlls_e2e.rs`: CRT, mutex, and console module mapping tests.
  - `console_shaders.rs`: Kernel translation and root argument placement tests.
- **Synthetic Test Console Title (`tests/testtitle`)**:
  - In-tree test application (`xcl-testtitle`) compiled as a synthetic console title.
  - Verifies filesystem sandboxing (`GAME:`, `S:\`, `T:\`), PLS quotas, memory budgets, D3D12.X device creation, CPUID detection, and XAudio2 silence voices.
- **Synthetic Test Game DLL (`tests/testgamedll`)**:
  - In-tree synthetic companion DLL (`testgame_mw.dll`) exercising relocations, TLS callbacks, and export resolution.

### 3.2 Running the Test Suite
All tests run with standard Cargo commands or the PowerShell runner:

```powershell
# Build entire workspace (including test console title and DLL)
cargo build --workspace

# Run full test suite across workspace
.\scripts\test.ps1

# Focus on a specific crate or component
cargo test -p xcl -- loader
cargo test -p xcl -- recomp
cargo test -p xcl -- real

# Build the synthetic test title
cargo build -p xcl-testtitle

# CI runner simulation (skips host Gaming Services / Basic Render Driver tests)
.\scripts\test.ps1 --skip test_title_runs_and_game_folder_is_untouched
```

### 3.3 Opt-in Developer Environment Variables
Tests on retail data require explicit local opt-in via environment variables pointing to files you legally own:

| Variable | Description | Default Behavior if Unset |
|---|---|---|
| `XCL_SAMPLE_RPF` | Path to extracted shader pack | Skipped (`SKIPPED: XCL_SAMPLE_RPF not set`) |
| `XCL_SAMPLE_DLLS` | Directory containing console DLLs | Skipped (`SKIPPED: XCL_SAMPLE_DLLS not set`) |
| `XCL_SAMPLE_TTPF` | Path to texture archive | Skipped (`SKIPPED: XCL_SAMPLE_TTPF not set`) |
| `XCL_SAMPLE_BK2` | Path to sample Bink video | Skipped (`SKIPPED: XCL_SAMPLE_BK2 not set`) |
| `XCL_SAMPLE_BVH4` | Path to prebuilt BVH package | Skipped (`SKIPPED: XCL_SAMPLE_BVH4 not set`) |
| `XCL_SAMPLE_AWC` | Path to sample audio archive | Skipped (`SKIPPED: XCL_SAMPLE_AWC not set`) |
| `XCL_SAMPLE_XG` | Path to XG container | Skipped (`SKIPPED: XCL_SAMPLE_XG not set`) |
| `XCL_SAMPLE_DXC` | Directory with console DXC toolchain | Skipped (`SKIPPED: XCL_SAMPLE_DXC not set`) |

### 3.4 Sabotage Testing & Bounds Verification
When implementing or verifying bug fixes:
- Verify boundary conditions: out-param pointer sizes, integer overflows, out-of-bounds array reads, null pointer validation.
- Perform sabotage checks: Temporarily revert your fix or inject an intentional discrepancy to confirm that the test suite fails reliably. A test that passes regardless of the code under test is invalid.

---

## 4. Branch Naming Conventions

All branches must follow the standard hierarchical naming format:

```
<type>/<short-kebab-slug>
```

### 4.1 Allowed Branch Types
- **`fix/`**: Bug fixes, safety clamps, boundary checks, error cleanups.
  - Examples: `fix/loader-error-cleanup`, `fix/queue-slot18-safety`, `fix/pls-quota-clamp`, `fix/config-load-resolve`
- **`feat/`**: New feature implementations, new HLE APIs, hardware support.
  - Examples: `feat/present-hdr-colorspace`, `feat/cpuid-startup-check`
- **`docs/`**: Documentation, design specifications, contributor guides.
  - Examples: `docs/contributing-guidelines`, `docs/wave64-guide`
- **`test/`**: Test suite expansions, synthetic test cases, benchmark additions.
  - Examples: `test/testtitle-synthetic-checks`, `test/recomp-edge-cases`
- **`refactor/`**: Code cleanups, modularization, performance improvements without functional changes.
  - Examples: `refactor/d3d12-vtable-dispatch`, `refactor/vfs-mount-table`

Always branch from the latest `origin/main`.

---

## 5. Pull Request Description Template

Every Pull Request submitted to XCL must include the following sections. Copy this template into your PR description:

```markdown
## Summary
Concise description of the change and the issue/item reference it addresses (e.g., Fixes #106 Item 3).

## Changes
- Concise bullet points describing exact code changes made.
- Note any new error codes, clamps, or fallback behaviors.

## Tested
- [x] Full workspace build: `cargo build --workspace`
- [x] Component test execution: `cargo test -p xcl -- <filter>`
- [x] Suite test script: `.\scripts\test.ps1`
- Test platform details:
  - OS: Windows 11 (Build XXXXX)
  - GPU & Driver: AMD Radeon RX 9070 (Adrenalin XX.X.X) / NVIDIA RTX XXXX
  - Outcome: 370+ passed, 0 failed.
- Specific test cases added or verified:
  - `crate::submodule::test_name` passed.
  - Sabotage check: confirmed test fails when fix is reverted.

## Not Tested
- Explicit list of what was NOT tested (e.g., opt-in game samples unset, wave64 on NVIDIA hardware, HDR swapchain presentation on non-HDR displays).

## AI Use
- [x] AI-assisted development (e.g., Claude Opus 5.5, Gemini 3.8 Flash)
- Details of assistance: Implementation of boundary check and unit test generation.
- Attribution in git commit:
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
```
