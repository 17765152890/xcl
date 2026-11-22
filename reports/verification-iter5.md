# Verification report — iteration 5 (`experimental/ffx-functional`)

GPU: NVIDIA GeForce RTX 4060 Ti. All results are real runs. The iteration's mission was the
NVIDIA pipeline wall — the wall turned out to be something else entirely, and the march went
far past it.

## 1. The iteration-4 "NVIDIA size wall" — misdiagnosed, corrected

The driver-bisect instrument set (DBWIN listener with a self-test line proving the capture
lives, the adapter's wave-lane range printed per run, an empty-shader control, a no-root-
signature control) separated the factors:

| adapter | WaveLaneCount range | empty cs_6_6, no [WaveSize] | empty + [WaveSize(64)] |
|---|---|---|---|
| RTX 4060 Ti | **32..32** | S_OK | **E_INVALIDARG, silent** |
| WARP | 32..32 | S_OK | E_INVALIDARG |
| Radeon iGPU | **64..64** | S_OK | S_OK |

The iteration-4 conclusion "the driver rejects ~600 KB DXIL for size" was wrong: **the
driver rejects [WaveSize(64)] compute shaders outright because its wave range cannot run
64-lane waves.** Pass 14 (26 KB RDNA2 → 606 KB DXIL) was simply the FSR4 pass set's first
wave64 program. Once the wave size is legal, a **949 KB DXIL is accepted** (the 41 KB pass)
— there is no size wall. The D3D12 debug layer was never a witness here: the machine has no
`d3d12SDKLayers.dll` (the Graphics Tools optional feature is not installed; DISM needs an
interactive elevated session) — `D3D12GetDebugInterface` fails silently.

The wave-lane matrix also re-explains the iGPU results from iteration 4: the iGPU's range
is 64..64 — it rejects the *wave32* translated shaders, which is why the "descriptor-heap-
indexed" passes failed there. Upstream's RDNA4 (32..64) accepts everything.

## 2. The rendezvous form (commit `0f82075`) — verified

A wave64 console program on a 32..32 adapter now translates to `[WaveSize(32)]` with the
same [numthreads]: the console wave's two halves run as co-resident native waves and
rendezvous per cross-lane operation through groupshared memory (ballot: the sibling's word
via a seq_cst atomic epoch publish + spin; the cross-lane read: a 64-slot value exchange).
No barrier intrinsics — **DXC lowers them to `dx.op.barrier` behind the per-lane publish
branches, which deadlocks the whole group** (verified by the hang, then by the passing run
without them); the seq_cst atomic publish alone orders the word store against the spin's
read.

Verified by `wave64_compute_rendezvous_on_32_lane_waves`: a synthetic console CS (assembled
RDNA2, wrapped in an FSSM container through the same pipeline as retail shaders) whose
readfirstlane crosses the halves (readers in the high half read lane 8's value, which only
the low half's ballot produced) and whose exec words need the other half's ballot bits —
executed on the RTX 4060 Ti with the exact console semantics (`mem[v0] = 8` for the active
lanes 8..=40), 4 console waves over 8 native waves. The LDS variant
(XCL_TEST_W64_RENDEZVOUS=lds) exercises the tail-aliased exchange below.

## 3. The LDS budget (commit `a182af2`) — verified

The FSR4 pass at 4776 B of RDNA2 is wave64 *and* uses the emulated console LDS: 32 KB of
xcl_lds plus the exchange arrays exceeded the D3D12 groupshared limit (DXC: "Total Thread
Group Shared Memory storage is 33888, exceeded 32768"). The exchange now carries inside the
LDS block's tail dwords — the region a program's own accesses only reach when they wrap
the emitter's `& 8191` mask (already unsupported) — keeping the total at exactly 32 KB.
The pass then compiles and is accepted. The LDS-variant synthetic test proves the LDS
round-trip and the aliased exchange coexist correctly on the GPU.

## 4. The full pass set — created; the dispatch — accepted

With both fixes, the march through the official console DLL's serialized pass set
completes: **30+ authentic deserialized pipelines** (the exact count is in the frontier
test's output; sizes from 576 B to 41 KB of raw RDNA2, up to 949 KB of DXIL, wave32 and
wave64, one with console LDS), and then:

```
ffxDispatch frame 0: code 0x0 (OK)     ← the console provider accepted the dispatch
  recorded 1027 words
  dispatch grid (20,12,1) (120,68,1) (94,1,1) (15,540,1) ... 26 dispatches
  close: Translation { dispatches: 26, skipped: 0, barriers: 26, clears: 1, copies: 4, unknown: None }
```

The frontier moved inside execution: the recorded console commands fault the device
mid-frame (0x887A0006). The grid evidence also reveals the real pass geometry —
(120,68) with (16,16) threads exactly covers 1920×1088 — the deserializer's numthreads
guess (256,1,1) is the same 256 threads but the wrong axis mapping; a per-pass geometry
fix remains future work (it affects output correctness, not the fault).

## 5. The execution fault — bisected to dispatch 14 (commit `0b7e962`)

`XCL_EXEC_FIRST_DISPATCHES=n` (the console list replay) executes only the first n
dispatches; each logs its grid and pipeline. Measured:

| n | verdict |
|---|---|
| 1..13 | frame 0 executes, device alive (the later frame-1 close artifact is the harness's 3-frame loop meeting the bisect env) |
| **14** | **device removed (0x887A0006)** — and with the bounded spins in place, an *immediate* fault, not a TDR stall: an invalid access, not a hang |
| 20..26 | device removed |

**Dispatch 14 = grid (507,1,1), the 26 KB wave64 pass** (606 KB DXIL, hash 24ddadf7 —
25 ballot and 9 cross-lane sites interleaved with the heaviest shader of the set).

(The R9G9B9E5-UAV-as-view-creation hypothesis was disproved along the way: instrumented
UAV-view creation logs no console UAV view of any non-UAV resource.)

## 6. The invalid access — identified, fixed (commit `10faf5e`)

The console's t# descriptors are read-write — the FSR4 pass image-stores the
neural-input texture without ever creating a UAV view. The recompiler's image stores
correctly index the PC UAV slot at +1 — but the iteration-3 flag-drop translation made
that slot *impossible to create* (the PC driver rejects both the UAV flag and UAV views
of DXGI 67), so the store went through an **uninitialized descriptor**: the invalid
access, and the iteration-3 documented risk realised through a subtler mechanism than
"the view creation is void".

The fix translates the format instead of the flags:

- `CreateCommittedResource (console)` and `CreatePlacedResourceX`: DXGI 67 with
  GPU-write flags → **R32_TYPELESS with the flags and the console's UAV state kept**;
- the console's R9G9B9E5 views of such a texture → **R32_UINT** on the PC (a format-67
  view of a typeless resource is invalid), while the console t# keeps format 132; the
  SRV creation also **creates the PC UAV at the slot's +1** (console code never asks
  for it);
- the recompiler's format-132 image path: loads read the raw dword and decode 9-9-9-E5;
  stores pack the shader's float RGB. The pair is deliberately self-inverse (both
  writer and reader are translated by the same code; console-exact rounding would only
  matter if an untranslated path read the texture — none does).

The verifying march (every pass recompiles once — the prelude changed) returned its
verdict: the RGBE translation engaged, all 30 pipelines built, 26 dispatches recorded
and closed cleanly — **and dispatch 14 still faults**. The fix was *necessary* (the
slot really was uninitialized) but not the whole fault. The follow-up bisect timings
were decisive: n ≤ 13 all ~4.5 s (clean — no desync anywhere earlier), n=14 = 17.2 s
with the 2^28 bound (the TDR resetting the device while the bound was still counting),
and — after `98d0348` lowered the bound to 2^20 so a desynchronised site heals in
milliseconds — **n=14 = 73.3 s**: not a TDR stall, not an immediate access fault, but
**over a minute of degraded execution** before the device is removed. The desync is
real and structural (the halves disagree on rendezvous sites; each disagreement heals
at the next site, but the degraded exec/readfirstlane values send the pass's scc-break
loops far past the console's trip counts). The root-cause hunt (a bound-firing site
counter, the DPP cross-boundary suspect, the scc input chain) is precisely laid out in
`handoff-iter5.md`.

## 7. Final state at the checkpoint

- Branch: `experimental/ffx-functional`, **20 commits ahead of upstream `main`
  (`e511b94`), clean tree. The full `gpu::` test module green (141 tests); the FFX
  frontier-lock run passes (the three-frontier lock, `2970a6c`).
- Commits of the iteration: `ed509a2` (bisect diagnostics), `0f82075` (rendezvous
  form + its GPU test), `a182af2` (LDS-tail exchange + the LDS test variant),
  `0b7e962` (bounded spins + the dispatch-bisect replay), `10faf5e` (the
  R32_TYPELESS/RGBE translation), `2970a6c` (the execution-frontier lock),
  `98d0348` (the watchdog-fitting spin bound), plus the earlier `35c19f8`
  (VFS drive-mount fix) and `b8881cf` (XCL_ADAPTER).
- Everything reproducible from `handoff-iter5.md`'s command block; the DXIL cache makes
  reruns fast.
