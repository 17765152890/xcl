# RTX 3080 startup comparison

Both runs used upstream XCL `f4ce27c`, Windows build 26300, an RTX 3080 10 GiB with driver 621.14, and the same 69 verified local files. The test set included the complete `audio.rpf`, `levels.rpf` and `navmeshes.rpf` archives. Verification of the remaining files was in progress, so the results apply to this 69-file set.

| Coherent window | First accepted 3840 × 2160 present | Startup stop |
| --- | ---: | --- |
| 512 MiB (default) | 5.309 s | 7.728 s, `0x80000003` at `+0x1dac734` |
| 4,095 MiB (`XCL_COHERENT_MB=4095`) | 6.185 s | 8.829 s, `0x80000003` at `+0x1dac734` |

XCL reported a 4,549 MiB allocation need on the default run and opened the requested 4,095 MiB window on the second. Both runs stopped before the menu. The timings are single-run startup measurements. [Pinned XCL build](https://github.com/ivall/xcl/commit/f4ce27ceb6d1b0576d0c659b713b674778bf1968) · [Related RTX 3080 heap test](https://github.com/ivall/xcl/pull/632)

## Compare host presents between runs

In this revision, `XCL_FRAME_DUMP=0` selects the default readback interval. Clear the entry in the launch environment to keep frame readback off. Set `XCL_ACTIVITY=1` for periodic host presents, frame event timeouts and DirectStorage error counts.

```powershell
Remove-Item Env:XCL_FRAME_DUMP -ErrorAction SilentlyContinue
$env:XCL_ACTIVITY = '1'
```

The host present total counts calls accepted by DXGI. Confirm visible rendering and working input separately when reporting menu or gameplay progress. [Frame-dump parser](https://github.com/ivall/xcl/blob/f4ce27ceb6d1b0576d0c659b713b674778bf1968/crates/xcl/src/hle/d3d12x/present.rs#L775-L785) · [Activity counters](https://github.com/ivall/xcl/blob/f4ce27ceb6d1b0576d0c659b713b674778bf1968/crates/xcl/src/host.rs#L32-L49)

## VIctory comparison

VIctory `d2a4bae` passed mapping and import binding on the same 69-file set. An isolated boot using native D3D12 logged presented frames through index 16 and then stopped at `0x80000003`, guest image offset `+0x1dac734`. Both runtimes stopped before the menu on this file set. [VIctory source revision](https://github.com/Marvisak/VIctory/commit/d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989)

An additional run with XCL `4017f8f` first presented at 5.445 seconds and hit the same stop at 9.062 seconds. Its activity log counted two host presents and 5,372 DirectStorage requests with zero failures through the last completed interval at 8.004 seconds. The final partial interval falls outside those counts. [XCL source revision](https://github.com/ivall/xcl/commit/4017f8ff8faaae6007fd83910d230a027dd85a33)

## Complete package

The next XCL run added the rest of the pinned package for 109 verified files total. It used the `7ebf1b8` diagnostics branch, a 512 MiB coherent window, `XCL_ACTIVITY=1`, and a cleared frame-dump setting. The first accepted 4K present was at 5.466 seconds. XCL continued presenting until the isolated launcher stopped it at its 60-second limit. The earlier `+0x1dac734` exception did not recur.

The last ten two-second activity windows each counted 6–8 host presents, with no reported frame-event timeout or DirectStorage failure in those windows. That is 3–4 accepted presents per second in the late samples. Visible rendering and input remain to be checked. [Pinned package manifest](https://github.com/4ld0-vito/Leonida/blob/05e821e2/game/xbox-1.0.2.1.json)

### 4,095 MiB on the complete package

The matched 109-file rerun kept the same XCL executable, 60-second isolation limit, host counters, and frame-dump setting. The only launch setting changed was `XCL_COHERENT_MB=4095`. XCL opened the requested window, first presented at 6.705 seconds at 3840 × 2160, and ran to the limit without a reported exception.

Each of the final ten two-second activity windows counted 60 accepted host presents, 1,260 draws and 60 dispatches. That is 30 accepted presents per second for those 20 seconds, up from 3–4 in the corresponding default-window samples. The sampled frame-event timeouts and DirectStorage failures remained at zero. Visual output and working input still need direct verification. [Related RTX 3080 heap test](https://github.com/ivall/xcl/pull/632)

### Clean upstream follow-up

The clean `d208e14` build reproduced the 4,095 MiB result and reached the 90-second launcher limit without a reported exception. All final 16 retained two-second intervals counted 60 accepted presents, with zero frame-event timeouts and DirectStorage failures. XCL logged an initial title-requested frame interval of 33,333 µs. This establishes the startup cadence. A visible menu, controller response and any later cadence change remain to be measured. [Upstream revision](https://github.com/ivall/xcl/commit/d208e14b765f469f4b56301b78afa8dac5378339)

### Visible-window follow-up

An isolated 109-file test using a local windowed build based on `d208e14` displayed the Player1, Account Picker, and Screen Narration prompts. The rest of the image stayed dark in two window observations. XCL reported about 30 FPS, and the final ten two-second activity windows each counted 60 accepted presents. The job reached its 180-second cutoff without a reported exception. This verifies visible startup UI. Physical controller response and world rendering are still unverified.

### Player1 input check

The same 109-file Wacom test used XCL’s built-in input script to send one A press at 60 seconds. The host logged the A button bit (`0x4`). Before the press, the window showed the dark Player1 prompt. After it, the game displayed the colored Navigation Experience setup screen, with Minimap (Dynamic) and Continue. The final activity intervals remained near 30 accepted presents per second, and the job reached its 180-second limit without a reported exception. This verifies the title’s response to an app-local scripted press. Physical input and world rendering remain to be checked.
