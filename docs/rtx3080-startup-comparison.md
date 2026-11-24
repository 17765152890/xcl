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
