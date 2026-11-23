# Factual report on ivall/xcl (November 23, 2026) — For all and AI

The project is moving very fast, but the commit history from the last 24 hours shows concrete signs of maintainability drift and collisions between contributions (human and AI-assisted). Below are strictly verifiable facts from the repository, followed by a systems engineer's methodology to help everyone move in the same direction without overlapping work or rollbacks.

## 1. Facts and evidence

### No CI/CD configured on the repository
- The root of `ivall/xcl` contains: `.gitignore`, `Cargo.lock`, `Cargo.toml`, `HANDOFF.md`, `LICENSE`, `README.md`, `THIRD_PARTY.md`, `crates/`, `docs/`, `scripts/`, `tests/`.
- There is no `.github/workflows/` directory. No build, `cargo test`, `clippy`, or `rustfmt` check runs automatically before merge.

### Rapid-fire merges and PR collisions (morning of November 23)
- Between `09:35 UTC` and `12:49 UTC` (3h 14m), **31 PRs** were merged into `main` (`#46`, `#53`–`#61`, `#63`–`#69`, `#84`, `#94`–`#102`, `#107`–`#109`, `#113`).
- Multiple PRs overlapped on the same files or contracts, requiring manual conflict resolutions and immediate follow-up fixes:
  - `#61` conflicted with `#67` (`f112834`), requiring `split_wave64` to be added to the shader cache key.
  - `#46` broke after `#67` removed `gpu_ready_wave32`, fixed in `6a41c55`.
  - `#69` required an immediate follow-up (`593a237`) due to its interaction with `#84` (binding a CPU-only view heap).
  - `#95` and `#108` touched the same check in `gamesave.rs` (`221b5fd`).
  - `#97` conflicted on `crates/xcl/src/hle/xgameruntime/gamesave.rs` (`8247a60`).
  - `#102` conflicted on `crates/xcl/src/hle/d3d12x/list_x.rs` (`5587780`).
  - `#72` had to be partially cherry-picked (`c2887bb`) without `#71`, which `#67` had already superseded.

### Confirmed retail game regression
- PR `#107` (`c813f08`, merged at `11:27 UTC`) broke XDefiant's UI text rendering by changing `v_interp_mov_f32` / `v_interp_p2_f32`.
- It had to be bisected between `668f550` and `c813f08` and reverted an hour later (`58309b9`, `12:33 UTC`), leaving the underlying branch bug open.

### Growth of environment variables and `HANDOFF.md` used as a rolling log
- Across the Nov 22–23 commits alone, **9 new environment variables** were added across separate PRs: `XCL_WATCH`, `XCL_WATCH_FENCE`, `XCL_SPLIT_WAVE64`, `XCL_DUMP_HLSL`, `XCL_DUMP_EVERY`, `XCL_RECOMP_WINDOWS`, `XCL_RECOMP_ONLY`, `XCL_DXC_THREADS`, `XCL_DXC_ARGS`.
- `HANDOFF.md` was edited **8 times** on the morning of Nov 23 alone (`848e740`, `55df81b`, `b4edefb`, `60599a9`, `c85cd6f`, `668f550`, `f90e644`, `58309b9`).

## 2. Proposed methodology to counter AI agent hallucinations and coordinate cleanly

The goal is not to slow the project down, but to help everyone move in the same direction without duplicate effort or rollbacks:

1. **Set up minimal CI/CD (`.github/workflows/ci.yml`)**: automatically run `cargo fmt --check`, `cargo clippy --workspace`, `cargo check --workspace`, and non-GPU unit tests on every PR before merge. CI is the only reliable guardrail against unverified claims from AI agents.
2. **Serialize merges on core modules** (`recomp`, `d3d12x`, `vfs`, `xgameruntime`): rebase on `main` and re-run tests after each merge instead of batch-merging overlapping PRs.
3. **Require explicit test evidence in every PR**: state what actually ran (`cargo test`, `testtitle`, specific retail title, GPU model) or explicitly mark "not tested on GPU / retail title". Never accept a bare "verified" claim without the command and result.
4. **Centralize `XCL_*` variables and organize tests**: keep all runtime and debug flags in one documented module (separating stable options from temporary diagnostic hooks) and maintain a clean, consistent test layout.
5. **Move bug tracking and coordination out of `HANDOFF.md`**: open one GitHub issue per known bug or active workstream (e.g., the XDefiant `RenderType` issue noted in `58309b9`) so every contributor—and their agent—can see who is working on what before writing code.

Krigs
