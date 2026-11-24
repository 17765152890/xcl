---
name: xcl-audio
description: >-
  Implement Xbox audio in XCL: XAudio2 voices, XMA, Opus, ACPHAL, and movie
  audio. Use when a cutscene is silent, a voice format is rejected, or when
  editing hle/xaudio2.rs, hle/xma2/, hle/opus.rs, hle/acphal.rs, or hle/mfplat.rs.
---

# Audio

`hle/xaudio2.rs` is `xaudio2_9.dll` and `xaudio2_9redist.dll`. Create flags outside the host-supported set are dropped. Vtables are patched in place. `CreateSourceVoice` turns XMA and XMA2 into PCM16; Opus `0x704F` goes through `MSOpusDecoder.dll` (`hle/opus.rs`) and is also submitted as PCM. `CreateMasteringVoice` maps a console device id to the default device and retries after `CoInitializeEx` if COM was not started. Unknown exports fall back to system `xaudio2_9.dll`.

Ordinals for Bink (`XAudio2Create` is ordinal 1) are in `stubdb.toml`. Do not renumber them.

`hle/xma2/` is a pure-Rust XMA1/XMA2 decoder. `tables.rs` is data. A wrong frame decode should fail the voice test, not play silence and return success. Silence is only for a format the report already classifies as dropped.

`hle/acphal.rs` covers `acphal.dll`, `xapu.dll`, and `chad.dll` (APU memory, decoder, DSP). Unimplemented names return `E_NOTIMPL`.

`hle/mfplat.rs` falls back to system `mfplat.dll` for the rest of Media Foundation.

`report.json` `audio` lists engines, voices, and formats played as silence. A cutscene with `presents > 0` and a rejected format is an audio bug. Movie pictures are the title's codec DLL (Bink 2 is already loaded as a game DLL) plus its display shaders; do not special-case a title's shader names.
