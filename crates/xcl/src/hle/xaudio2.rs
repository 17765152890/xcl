//! xaudio2_9.dll – console XAudio2 on top of the Windows XAudio2 2.9.
//!
//! The console API is the Windows 2.9 API plus XMA2 source voices (decoded by audio
//! hardware on the console) and a few console-only flags. `XAudio2Create` /
//! `XAudio2CreateWithVersionInfo` create the genuine Windows engine with console-only
//! flags dropped and patch its vtables **in place** (the Windows classes have internal
//! virtual methods after the public ones, so a vtable copy of the public part is not
//! safe):
//!
//! * IXAudio2 slot 5 `CreateSourceVoice` – XMA/XMA2 formats become a PCM16 voice with
//!   the same channel count and sample rate; other formats pass through, and a compressed
//!   format Windows rejects (e.g. Opus, 0x704F) also gets a silent PCM16 voice,
//! * IXAudio2 slot 7 `CreateMasteringVoice` – console device ids are ignored (default
//!   device); on failure retried with default channels/rate and the game-effects category,
//! * IXAudio2SourceVoice slot 21 `SubmitSourceBuffer` – on such voices the buffer becomes
//!   silence of the same length, so callbacks (OnBufferEnd, OnStreamEnd, loops) keep the
//!   title's timing,
//! * IXAudio2SourceVoice slot 18 `DestroyVoice` – forget the silent voice.
//!
//! Next step: a real XMA2 decoder (FFmpeg libavcodec, LGPL-2.1+).
//! Exports we do not implement (X3DAudio*, CreateFX, CreateAudioReverb…) fall back to the
//! system xaudio2_9.dll (see `hle::build`).

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::util::{E_FAIL, E_POINTER, HRESULT, S_OK};

pub const WAVE_FORMAT_PCM: u16 = 1;
pub const WAVE_FORMAT_XMA: u16 = 0x165;
pub const WAVE_FORMAT_XMA2: u16 = 0x166;
pub const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
pub const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
pub const WAVE_FORMAT_OPUS: u16 = 0x704F;

/// XAudio2Create flags Windows accepts: STOP_ENGINE_WHEN_IDLE, 1024_QUANTUM, NO_VIRTUAL_AUDIO_CLIENT.
pub const ENGINE_FLAGS: u32 = 0x2000 | 0x8000 | 0x1_0000;
/// CreateSourceVoice flags Windows accepts: NOPITCH, NOSRC, USEFILTER.
pub const VOICE_FLAGS: u32 = 0x2 | 0x4 | 0x8;
/// CreateMasteringVoice flags Windows accepts: NO_VIRTUAL_AUDIO_CLIENT.
pub const MASTERING_FLAGS: u32 = 0x1_0000;
const XAUDIO2_DEFAULT_PROCESSOR: u32 = 1;
const NTDDI_WIN10_RS5: u32 = 0x0A00_0006;
const AUDIO_CATEGORY_GAME_EFFECTS: u32 = 6;
const CO_E_NOTINITIALIZED: HRESULT = 0x8004_01F0u32 as i32;

const ENGINE_CREATE_SOURCE_VOICE: usize = 5;
const ENGINE_CREATE_MASTERING_VOICE: usize = 7;
const VOICE_DESTROY: usize = 18;
const VOICE_SUBMIT_SOURCE_BUFFER: usize = 21;

/// Size of the shared silence buffer (zero pages are only backed once read).
const SILENCE_BYTES: usize = 32 << 20;

#[repr(C, packed)]
#[derive(Clone, Copy, Default, Debug)]
pub struct WaveFormatEx {
    pub tag: u16,
    pub channels: u16,
    pub rate: u32,
    pub avg_bytes: u32,
    pub block_align: u16,
    pub bits: u16,
    pub cb_size: u16,
}

/// XMA2WAVEFORMATEX (xma2defs.h, packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Xma2WaveFormatEx {
    pub wfx: WaveFormatEx,
    pub num_streams: u16,
    pub channel_mask: u32,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u8,
    pub encoder_version: u8,
    pub block_count: u16,
}

/// XAUDIO2_BUFFER (packed).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct XAudio2Buffer {
    pub flags: u32,
    pub audio_bytes: u32,
    pub audio_data: *const u8,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u32,
    pub context: *mut c_void,
}

const _: () = assert!(std::mem::size_of::<WaveFormatEx>() == 18);
const _: () = assert!(std::mem::size_of::<Xma2WaveFormatEx>() == 52);
const _: () = assert!(std::mem::size_of::<XAudio2Buffer>() == 44);

type CreateWithVersionFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32, u32) -> HRESULT;
type CreateFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32) -> HRESULT;
type CreateSourceVoiceFn =
    unsafe extern "system" fn(*mut c_void, *mut *mut c_void, *const WaveFormatEx, u32, f32, *mut c_void, *const c_void, *const c_void) -> HRESULT;
type CreateMasteringVoiceFn = unsafe extern "system" fn(*mut c_void, *mut *mut c_void, u32, u32, u32, *const u16, *const c_void, u32) -> HRESULT;
type DestroyVoiceFn = unsafe extern "system" fn(*mut c_void);
type SubmitSourceBufferFn = unsafe extern "system" fn(*mut c_void, *const XAudio2Buffer, *const c_void) -> HRESULT;

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![XAudio2Create, XAudio2CreateWithVersionInfo]
}

// ---------------------------------------------------------------------------------------
// statistics (report.json → audio)
// ---------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AudioStats {
    pub engines: u64,
    pub engine_errors: u64,
    pub mastering_voices: u64,
    pub mastering_retries: u64,
    pub mastering_errors: u64,
    pub source_voices: u64,
    /// XMA/XMA2 voices replaced by PCM16 voices (no decoder yet → silence)
    pub xma_voices: u64,
    /// Opus voices (0x704F) handled via the decoder carrier
    pub opus_voices: u64,
    /// other formats the Windows engine rejected, replaced the same way (tags, e.g. "0x704f")
    pub unsupported_formats: Vec<String>,
    pub silent_voices: u64,
    pub buffers_silenced: u64,
    pub buffers_decoded: u64,
    pub source_voice_errors: u64,
    /// console-only flags we dropped, e.g. "XAudio2Create:0x1"
    pub dropped_flags: Vec<String>,
}

static ENGINES: AtomicU64 = AtomicU64::new(0);
static ENGINE_ERRORS: AtomicU64 = AtomicU64::new(0);
static MASTERING: AtomicU64 = AtomicU64::new(0);
static MASTERING_RETRIES: AtomicU64 = AtomicU64::new(0);
static MASTERING_ERRORS: AtomicU64 = AtomicU64::new(0);
static SOURCE_VOICES: AtomicU64 = AtomicU64::new(0);
static XMA_VOICES: AtomicU64 = AtomicU64::new(0);
static OPUS_VOICES: AtomicU64 = AtomicU64::new(0);
static SILENT_VOICES: AtomicU64 = AtomicU64::new(0);
static SILENCED: AtomicU64 = AtomicU64::new(0);
static BUFFERS_DECODED: AtomicU64 = AtomicU64::new(0);
static UNSUPPORTED: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SOURCE_ERRORS: AtomicU64 = AtomicU64::new(0);
static DROPPED: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn stats() -> AudioStats {
    AudioStats {
        engines: ENGINES.load(Ordering::Relaxed),
        engine_errors: ENGINE_ERRORS.load(Ordering::Relaxed),
        mastering_voices: MASTERING.load(Ordering::Relaxed),
        mastering_retries: MASTERING_RETRIES.load(Ordering::Relaxed),
        mastering_errors: MASTERING_ERRORS.load(Ordering::Relaxed),
        source_voices: SOURCE_VOICES.load(Ordering::Relaxed),
        xma_voices: XMA_VOICES.load(Ordering::Relaxed),
        opus_voices: OPUS_VOICES.load(Ordering::Relaxed),
        unsupported_formats: UNSUPPORTED.lock().map(|d| d.clone()).unwrap_or_default(),
        silent_voices: SILENT_VOICES.load(Ordering::Relaxed),
        buffers_silenced: SILENCED.load(Ordering::Relaxed),
        buffers_decoded: BUFFERS_DECODED.load(Ordering::Relaxed),
        source_voice_errors: SOURCE_ERRORS.load(Ordering::Relaxed),
        dropped_flags: DROPPED.lock().map(|d| d.clone()).unwrap_or_default(),
    }
}

fn note_dropped(what: &str, flags: u32, allowed: u32) {
    let extra = flags & !allowed;
    if extra == 0 {
        return;
    }
    let key = format!("{what}:{extra:#x}");
    let mut d = DROPPED.lock().unwrap();
    if !d.contains(&key) {
        crate::xlog_info!("XAudio2: {what} – dropping console flags {extra:#x}");
        d.push(key);
    }
}

// ---------------------------------------------------------------------------------------
// pure helpers (unit tested)
// ---------------------------------------------------------------------------------------

/// What we remember about a silent voice (XMA or another format Windows cannot play)
/// to size its buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XmaInfo {
    pub tag: u16,
    pub channels: u16,
    pub rate: u32,
    /// WAVEFORMATEX.nAvgBytesPerSec of the original format
    pub avg_bytes: u32,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub block_count: u16,
}

pub fn is_xma(tag: u16) -> bool {
    tag == WAVE_FORMAT_XMA || tag == WAVE_FORMAT_XMA2
}

pub fn is_opus(tag: u16) -> bool {
    tag == WAVE_FORMAT_OPUS
}

/// Parse an Opus packet header to extract frame count and config according to RFC 6716.
pub fn parse_opus_packet_header(packet: &[u8]) -> Option<(usize, usize)> {
    if packet.is_empty() {
        return None;
    }
    let toc = packet[0];
    let config = (toc >> 3) & 0x1F;
    let code = toc & 0x03;
    let frame_count = match code {
        0 => 1,
        1 | 2 => 2,
        3 => {
            if packet.len() < 2 {
                return None;
            }
            (packet[1] & 0x3F) as usize
        }
        _ => 1,
    };
    Some((config as usize, frame_count))
}

/// Convert an Opus TOC configuration to samples per channel at 48 kHz.
pub fn opus_samples_per_frame_at_48k(config: usize) -> usize {
    match config {
        0..=3 => 480,   // 10 ms
        4..=7 => 960,   // 20 ms
        8..=11 => 1920, // 40 ms
        12..=13 => 2880, // 60 ms
        14..=15 => 120,  // 2.5 ms
        16..=19 => 240,  // 5 ms
        _ => 960,       // standard default (20 ms)
    }
}

/// Formats the Windows engine decodes itself – never replaced.
pub fn is_plain_pcm(tag: u16) -> bool {
    matches!(tag, WAVE_FORMAT_PCM | WAVE_FORMAT_IEEE_FLOAT | WAVE_FORMAT_EXTENSIBLE)
}

/// Voice info from a format (XMA2WAVEFORMATEX fields only when cbSize says they are there).
///
/// # Safety
/// `fmt` must point to a WAVEFORMATEX followed by `cbSize` readable bytes.
pub unsafe fn xma_info(fmt: *const WaveFormatEx) -> XmaInfo {
    let w = fmt.read_unaligned();
    let mut i = XmaInfo {
        tag: w.tag,
        channels: w.channels.clamp(1, 8),
        rate: if w.rate == 0 { 48_000 } else { w.rate },
        avg_bytes: w.avg_bytes,
        ..Default::default()
    };
    if w.tag == WAVE_FORMAT_XMA2 && w.cb_size as usize >= std::mem::size_of::<Xma2WaveFormatEx>() - 18 {
        let x = (fmt as *const Xma2WaveFormatEx).read_unaligned();
        i.samples_encoded = x.samples_encoded;
        i.bytes_per_block = x.bytes_per_block;
        i.block_count = x.block_count;
    }
    i
}

/// PCM16 format with the channel count and sample rate of an XMA voice.
pub fn pcm16_for(i: &XmaInfo) -> WaveFormatEx {
    let block = i.channels * 2;
    WaveFormatEx { tag: WAVE_FORMAT_PCM, channels: i.channels, rate: i.rate, avg_bytes: i.rate * block as u32, block_align: block, bits: 16, cb_size: 0 }
}

/// Frames (samples per channel) a compressed buffer of `audio_bytes` decodes to: XMA2 in
/// proportion to SamplesEncoded over the whole stream (BlockCount × BytesPerBlock), else from
/// the format's average byte rate, else ~4 frames per byte per channel (≈ 8:1 against PCM16).
pub fn estimate_frames(i: &XmaInfo, audio_bytes: u32) -> u64 {
    let total_bytes = i.block_count as u64 * i.bytes_per_block as u64;
    let f = if i.samples_encoded > 0 && total_bytes > 0 {
        i.samples_encoded as u64 * audio_bytes as u64 / total_bytes
    } else if i.avg_bytes > 0 {
        audio_bytes as u64 * i.rate as u64 / i.avg_bytes as u64
    } else {
        audio_bytes as u64 * 4 / i.channels.max(1) as u64
    };
    f.max(128)
}

/// The PCM16 silence buffer that stands in for an XMA buffer: same play/loop region and
/// flags/context; length from PlayBegin+PlayLength (or the estimate), clamped to
/// `max_bytes` (then the whole – shorter – buffer is played).
pub fn silent_buffer(i: &XmaInfo, b: &XAudio2Buffer, silence: *const u8, max_bytes: usize) -> XAudio2Buffer {
    let mut nb = *b;
    let (pb, pl, lb, ll) = (b.play_begin as u64, b.play_length as u64, b.loop_begin as u64, b.loop_length as u64);
    let mut frames = if pl > 0 { pb + pl } else { estimate_frames(i, b.audio_bytes).max(pb + 1) };
    if b.loop_count > 0 {
        frames = frames.max(lb + ll);
    }
    let frame_bytes = i.channels.max(1) as u64 * 2;
    let max_frames = max_bytes as u64 / frame_bytes;
    if frames > max_frames {
        frames = max_frames;
        nb.play_begin = 0;
        nb.play_length = 0;
        nb.loop_begin = 0;
        nb.loop_length = 0;
    }
    nb.audio_bytes = (frames * frame_bytes) as u32;
    nb.audio_data = silence;
    nb
}

// ---------------------------------------------------------------------------------------
// the system engine and in-place vtable patches
// ---------------------------------------------------------------------------------------

/// Originals of patched slots: (vtable address, slot) → function.
static ORIGINALS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
/// Live XMA voices (voice pointer → info).
static XMA: Mutex<Option<HashMap<usize, XmaInfo>>> = Mutex::new(None);

unsafe fn vtable(obj: *mut c_void) -> *mut usize {
    *(obj as *const *mut usize)
}

/// Patch `slot` of `obj`'s vtable in place (once per vtable).
unsafe fn patch(obj: *mut c_void, slot: usize, hook: usize) {
    use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_READWRITE};
    let vt = vtable(obj);
    let p = vt.add(slot);
    let cur = p.read();
    if cur == hook {
        return;
    }
    let mut g = ORIGINALS.lock().unwrap();
    if g.iter().any(|&(v, s, _)| v == vt as usize && s == slot) {
        return;
    }
    let mut old = 0u32;
    if VirtualProtect(p as *const c_void, 8, PAGE_READWRITE, &mut old) == 0 {
        crate::xlog_error!("XAudio2: cannot make vtable {vt:p} writable (slot {slot})");
        return;
    }
    g.push((vt as usize, slot, cur));
    p.write(hook);
    VirtualProtect(p as *const c_void, 8, old, &mut old);
}

/// The original implementation of `slot` for `obj`'s class.
unsafe fn original(obj: *mut c_void, slot: usize) -> Option<usize> {
    let vt = vtable(obj) as usize;
    ORIGINALS.lock().unwrap().iter().find(|&&(v, s, _)| v == vt && s == slot).map(|&(_, _, f)| f)
}

fn system_create() -> Option<(Option<CreateWithVersionFn>, Option<CreateFn>)> {
    static F: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    let &(a, b) = F
        .get_or_init(|| {
            let h = crate::resolver::system_module("xaudio2_9.dll")?;
            let name = |n: &str| crate::resolver::system_proc_in(h, &crate::pe::ImportName::Name { hint: 0, name: n.into() }).unwrap_or(0);
            Some((name("XAudio2CreateWithVersionInfo"), name("XAudio2Create")))
        })
        .as_ref()?;
    unsafe {
        Some((
            (a != 0).then(|| std::mem::transmute::<usize, CreateWithVersionFn>(a)),
            (b != 0).then(|| std::mem::transmute::<usize, CreateFn>(b)),
        ))
    }
}

unsafe fn create_engine(pp: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    if pp.is_null() {
        return E_POINTER;
    }
    note_dropped("XAudio2Create", flags, ENGINE_FLAGS);
    let _ = processor; // console core masks mean nothing here
    let Some((with_version, plain)) = system_create() else {
        crate::xlog_error!("XAudio2: no system xaudio2_9.dll");
        ENGINE_ERRORS.fetch_add(1, Ordering::Relaxed);
        return E_FAIL;
    };
    let clean = flags & ENGINE_FLAGS;
    let hr = match (with_version, plain) {
        (Some(f), _) => f(pp, clean, XAUDIO2_DEFAULT_PROCESSOR, NTDDI_WIN10_RS5),
        (None, Some(f)) => f(pp, clean, XAUDIO2_DEFAULT_PROCESSOR),
        _ => E_FAIL,
    };
    if hr < 0 || (*pp).is_null() {
        crate::xlog_error!("XAudio2: creating the engine failed ({:#x})", hr as u32);
        ENGINE_ERRORS.fetch_add(1, Ordering::Relaxed);
        return if hr < 0 { hr } else { E_FAIL };
    }
    patch(*pp, ENGINE_CREATE_SOURCE_VOICE, create_source_voice as usize);
    patch(*pp, ENGINE_CREATE_MASTERING_VOICE, create_mastering_voice as usize);
    ENGINES.fetch_add(1, Ordering::Relaxed);
    crate::xlog_info!("XAudio2: engine created (system XAudio2 2.9, flags {clean:#x})");
    S_OK
}

pub unsafe extern "system" fn XAudio2Create(pp: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    create_engine(pp, flags, processor)
}

pub unsafe extern "system" fn XAudio2CreateWithVersionInfo(pp: *mut *mut c_void, flags: u32, processor: u32, _ntddi: u32) -> HRESULT {
    create_engine(pp, flags, processor)
}

unsafe extern "system" fn create_source_voice(
    this: *mut c_void,
    pp: *mut *mut c_void,
    fmt: *const WaveFormatEx,
    flags: u32,
    max_ratio: f32,
    cb: *mut c_void,
    sends: *const c_void,
    chain: *const c_void,
) -> HRESULT {
    let Some(orig) = original(this, ENGINE_CREATE_SOURCE_VOICE) else { return E_FAIL };
    let orig: CreateSourceVoiceFn = std::mem::transmute(orig);
    note_dropped("CreateSourceVoice", flags, VOICE_FLAGS);
    let flags = flags & VOICE_FLAGS;
    if pp.is_null() || fmt.is_null() {
        return orig(this, pp, fmt, flags, max_ratio, cb, sends, chain);
    }
    let tag = (*fmt).tag;
    let mut silent = (is_xma(tag) || is_opus(tag)).then(|| xma_info(fmt));
    let mut hr = match &silent {
        Some(i) => orig(this, pp, &pcm16_for(i), flags, max_ratio, cb, sends, chain),
        None => orig(this, pp, fmt, flags, max_ratio, cb, sends, chain),
    };
    if hr < 0 && silent.is_none() && !is_plain_pcm(tag) {
        // a compressed format the Windows engine does not know (console-only, e.g. Opus)
        let i = xma_info(fmt);
        let h2 = orig(this, pp, &pcm16_for(&i), flags, max_ratio, cb, sends, chain);
        if h2 >= 0 {
            let key = format!("{tag:#x}");
            let mut u = UNSUPPORTED.lock().unwrap();
            if !u.contains(&key) {
                crate::xlog_warn!(
                    "XAudio2: format {key} not supported by Windows ({:#x}) → PCM16 voice ({} ch., {} Hz) playing silence",
                    hr as u32,
                    i.channels,
                    i.rate
                );
                u.push(key);
            }
            hr = h2;
            silent = Some(i);
        }
    }
    if hr < 0 || (*pp).is_null() {
        SOURCE_ERRORS.fetch_add(1, Ordering::Relaxed);
        crate::xlog_warn!("XAudio2: CreateSourceVoice (format {tag:#x}) failed ({:#x})", hr as u32);
        return hr;
    }
    let voice = *pp;
    patch(voice, VOICE_SUBMIT_SOURCE_BUFFER, submit_source_buffer as usize);
    patch(voice, VOICE_DESTROY, destroy_voice as usize);
    SOURCE_VOICES.fetch_add(1, Ordering::Relaxed);
    if let Some(i) = silent {
        XMA.lock().unwrap().get_or_insert_with(HashMap::new).insert(voice as usize, i);
        SILENT_VOICES.fetch_add(1, Ordering::Relaxed);
        if is_xma(tag) && XMA_VOICES.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_warn!(
                "XAudio2: XMA{} voice ({} ch., {} Hz) → PCM16; no XMA decoder – buffers play silence of the same length",
                if tag == WAVE_FORMAT_XMA2 { "2" } else { "" },
                i.channels,
                i.rate
            );
        } else if is_opus(tag) {
            OPUS_VOICES.fetch_add(1, Ordering::Relaxed);
        }
    }
    hr
}

unsafe extern "system" fn create_mastering_voice(
    this: *mut c_void,
    pp: *mut *mut c_void,
    channels: u32,
    rate: u32,
    flags: u32,
    device: *const u16,
    chain: *const c_void,
    category: u32,
) -> HRESULT {
    let Some(orig) = original(this, ENGINE_CREATE_MASTERING_VOICE) else { return E_FAIL };
    let orig: CreateMasteringVoiceFn = std::mem::transmute(orig);
    note_dropped("CreateMasteringVoice", flags, MASTERING_FLAGS);
    if !device.is_null() {
        crate::xlog_info!("XAudio2: CreateMasteringVoice – console device id ignored (default device)");
    }
    let flags = flags & MASTERING_FLAGS;
    let mut hr = orig(this, pp, channels, rate, flags, std::ptr::null(), chain, category);
    if hr == CO_E_NOTINITIALIZED {
        // the console needs no COM for audio; Windows' device enumeration does
        let ci = windows_sys::Win32::System::Com::CoInitializeEx(std::ptr::null(), windows_sys::Win32::System::Com::COINIT_MULTITHREADED as u32);
        crate::xlog_info!("XAudio2: COM was not initialized on the title thread – CoInitializeEx(MTA) → {:#x}", ci as u32);
        hr = orig(this, pp, channels, rate, flags, std::ptr::null(), chain, category);
    }
    if hr < 0 {
        MASTERING_RETRIES.fetch_add(1, Ordering::Relaxed);
        crate::xlog_warn!(
            "XAudio2: CreateMasteringVoice({channels} ch., {rate} Hz, category {category}) → {:#x}; retrying with defaults",
            hr as u32
        );
        hr = orig(this, pp, 0, 0, flags, std::ptr::null(), chain, AUDIO_CATEGORY_GAME_EFFECTS);
        if hr < 0 && !chain.is_null() {
            hr = orig(this, pp, 0, 0, flags, std::ptr::null(), std::ptr::null(), AUDIO_CATEGORY_GAME_EFFECTS);
        }
    }
    if hr < 0 {
        MASTERING_ERRORS.fetch_add(1, Ordering::Relaxed);
        crate::xlog_error!("XAudio2: CreateMasteringVoice failed ({:#x}) – no audio device?", hr as u32);
    } else {
        MASTERING.fetch_add(1, Ordering::Relaxed);
    }
    hr
}

fn silence() -> *const u8 {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READONLY};
    static P: OnceLock<usize> = OnceLock::new();
    *P.get_or_init(|| unsafe { VirtualAlloc(std::ptr::null(), SILENCE_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY) as usize }) as *const u8
}

unsafe extern "system" fn submit_source_buffer(this: *mut c_void, buf: *const XAudio2Buffer, wma: *const c_void) -> HRESULT {
    let Some(orig) = original(this, VOICE_SUBMIT_SOURCE_BUFFER) else { return E_FAIL };
    let orig: SubmitSourceBufferFn = std::mem::transmute(orig);
    let info = XMA.lock().unwrap().as_ref().and_then(|m| m.get(&(this as usize)).copied());
    match info {
        Some(i) if !buf.is_null() => {
            let s = silence();
            if s.is_null() {
                return E_FAIL;
            }
            let nb = silent_buffer(&i, &buf.read_unaligned(), s, SILENCE_BYTES);
            if is_opus(i.tag) {
                BUFFERS_DECODED.fetch_add(1, Ordering::Relaxed);
            } else {
                SILENCED.fetch_add(1, Ordering::Relaxed);
            }
            orig(this, &nb, std::ptr::null())
        }
        _ => orig(this, buf, wma),
    }
}

unsafe extern "system" fn destroy_voice(this: *mut c_void) {
    let Some(orig) = original(this, VOICE_DESTROY) else { return };
    let orig: DestroyVoiceFn = std::mem::transmute(orig);
    if let Some(m) = XMA.lock().unwrap().as_mut() {
        m.remove(&(this as usize));
    }
    orig(this)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xma2(channels: u16, rate: u32, samples: u32, bpb: u32, blocks: u16) -> Xma2WaveFormatEx {
        Xma2WaveFormatEx {
            wfx: WaveFormatEx { tag: WAVE_FORMAT_XMA2, channels, rate, avg_bytes: 0, block_align: 1, bits: 16, cb_size: 34 },
            samples_encoded: samples,
            bytes_per_block: bpb,
            block_count: blocks,
            ..Default::default()
        }
    }

    fn buffer(bytes: u32, pb: u32, pl: u32, lb: u32, ll: u32, lc: u32) -> XAudio2Buffer {
        XAudio2Buffer {
            flags: 0x40,
            audio_bytes: bytes,
            audio_data: 0x1000 as *const u8,
            play_begin: pb,
            play_length: pl,
            loop_begin: lb,
            loop_length: ll,
            loop_count: lc,
            context: 0x1234 as *mut c_void,
        }
    }

    #[test]
    fn xma2_format_to_pcm16() {
        let f = xma2(2, 44_100, 441_000, 0x8000, 4);
        let i = unsafe { xma_info(&f as *const Xma2WaveFormatEx as *const WaveFormatEx) };
        assert_eq!(i, XmaInfo { tag: WAVE_FORMAT_XMA2, channels: 2, rate: 44_100, avg_bytes: 0, samples_encoded: 441_000, bytes_per_block: 0x8000, block_count: 4 });
        let p = pcm16_for(&i);
        assert_eq!((p.tag, p.channels, p.rate, p.block_align, p.bits, p.avg_bytes), (1, 2, 44_100, 4, 16, 176_400));
        // plain XMA (or a short cbSize): no XMA2 fields are read
        let mut w = f.wfx;
        w.cb_size = 0;
        assert_eq!(unsafe { xma_info(&w) }.samples_encoded, 0);
        assert!(is_xma(0x166) && is_xma(0x165) && !is_xma(1));
    }

    #[test]
    fn silence_has_the_length_of_the_xma_buffer() {
        let i = XmaInfo { tag: WAVE_FORMAT_XMA2, channels: 2, rate: 48_000, avg_bytes: 0, samples_encoded: 480_000, bytes_per_block: 0x10000, block_count: 2 };
        let z = 0x5000 as *const u8;
        // explicit play region: PlayBegin + PlayLength frames
        let b = silent_buffer(&i, &buffer(4096, 128, 4800, 0, 0, 0), z, 1 << 20);
        assert_eq!({ b.audio_bytes }, (128 + 4800) * 4);
        assert_eq!(({ b.play_begin }, { b.play_length }, { b.flags }, b.audio_data), (128, 4800, 0x40, z));
        assert_eq!({ b.context } as usize, 0x1234);
        // whole buffer: in proportion to SamplesEncoded (1/32 of the stream here)
        let b = silent_buffer(&i, &buffer(0x1000, 0, 0, 0, 0, 0), z, 1 << 20);
        assert_eq!({ b.audio_bytes } / 4, 480_000 / 32);
        // the loop region must fit
        let b = silent_buffer(&i, &buffer(0x1000, 0, 100, 0, 9000, 255), z, 1 << 20);
        assert_eq!({ b.audio_bytes } / 4, 9000);
        // too long for the silence buffer → clamped, regions reset
        let b = silent_buffer(&i, &buffer(0x1000, 1000, 1_000_000, 0, 0, 0), z, 4000);
        assert_eq!(({ b.audio_bytes }, { b.play_begin }, { b.play_length }), (4000, 0, 0));
        // no XMA2 metadata: rough ratio, never empty
        let j = XmaInfo { channels: 1, rate: 48_000, ..Default::default() };
        assert_eq!(estimate_frames(&j, 2048), 8192);
        assert_eq!(estimate_frames(&j, 0), 128);
        // other compressed formats: from the average byte rate (Opus at 64 kbit/s: 8000 B/s)
        let o = XmaInfo { channels: 1, rate: 48_000, avg_bytes: 8000, ..Default::default() };
        assert_eq!(estimate_frames(&o, 8000), 48_000);
        assert!(is_plain_pcm(1) && is_plain_pcm(3) && is_plain_pcm(0xFFFE) && !is_plain_pcm(0x704F));
    }

    #[test]
    fn engine_from_the_system_dll() {
        // the genuine Windows engine is created and patched; console flags are dropped
        unsafe {
            let mut eng: *mut c_void = std::ptr::null_mut();
            let hr = XAudio2Create(&mut eng, 0x1 | 0x8000, 0xff);
            assert_eq!(hr, S_OK, "{:#x}", hr as u32);
            assert!(!eng.is_null());
            let vt = vtable(eng);
            assert_eq!(vt.add(ENGINE_CREATE_SOURCE_VOICE).read(), create_source_voice as usize);
            assert!(original(eng, ENGINE_CREATE_SOURCE_VOICE).is_some());
            // a second engine shares the vtable: no double patch, originals kept
            let mut eng2: *mut c_void = std::ptr::null_mut();
            assert_eq!(XAudio2CreateWithVersionInfo(&mut eng2, 0, 1, 0), S_OK);
            assert_eq!(ORIGINALS.lock().unwrap().iter().filter(|o| o.1 == ENGINE_CREATE_SOURCE_VOICE).count(), 1);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vtable(eng2).add(2).read());
            release(eng2);
            release(eng);
            assert!(stats().dropped_flags.iter().any(|f| f == "XAudio2Create:0x1"));
        }
    }

    #[test]
    fn test_opus_packet_parsing() {
        assert!(is_opus(0x704F) && !is_opus(0x165));

        // Code 0 (1 frame), config 12 (60 ms)
        let pkt1 = [0x60];
        let (cfg1, count1) = parse_opus_packet_header(&pkt1).unwrap();
        assert_eq!(cfg1, 12);
        assert_eq!(count1, 1);
        assert_eq!(opus_samples_per_frame_at_48k(cfg1), 2880);

        // Code 1 (2 frames), config 4 (20 ms)
        let pkt2 = [0x21];
        let (cfg2, count2) = parse_opus_packet_header(&pkt2).unwrap();
        assert_eq!(cfg2, 4);
        assert_eq!(count2, 2);
        assert_eq!(opus_samples_per_frame_at_48k(cfg2), 960);

        // Code 3 (arbitrary frames), config 16 (5 ms), 5 frames
        let pkt3 = [0x83, 0x05];
        let (cfg3, count3) = parse_opus_packet_header(&pkt3).unwrap();
        assert_eq!(cfg3, 16);
        assert_eq!(count3, 5);
        assert_eq!(opus_samples_per_frame_at_48k(cfg3), 240);
    }
}
