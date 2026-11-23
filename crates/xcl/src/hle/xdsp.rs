//! The console DSP (`chad.dll`).
//!
//! A title registers one buffer, activates streams, and submits a block at a time. Nothing is
//! convolved or transformed here: connect succeeds, each command is finished in the call, and
//! the output block is silence when it lies inside the registered buffer. The title's mixer
//! then continues instead of treating a missing DSP as fatal.
//!
//! Calls, as titles make them: `XDspConnectWithMaximumStreamLimit(buffer, bytes, impulse
//! seconds, max streams, max channels, &handle)`, `XDspActivate(client, &params, &status,
//! &stream)`, `XDspSubmitCommand(stream, &command, &sequence)`, `XDspDeactivate(stream)`,
//! `XDspDisconnect(client)`. `params` holds the block size at +8 and the channel count at +12.
//! `command` holds the output pointer at +16. `XDspStatus` is four 32-bit fields the title
//! polls: submitted sequence, processed sequence, first failed sequence, result.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Mutex;

use crate::util::{HRESULT, E_INVALIDARG, E_OUTOFMEMORY, S_OK};

const CLIENT_MAGIC: u32 = 0x5844_5343;
const STREAM_MAGIC: u32 = 0x5844_5353;
/// frames a command may clear: a power of two in the range titles submit, and nothing larger
const MIN_FRAMES: u32 = 32;
const MAX_FRAMES: u32 = 8192;

#[repr(C)]
struct Status {
    last_submitted: AtomicU32,
    last_processed: AtomicU32,
    first_failed: AtomicU32,
    result: AtomicI32,
}

struct Client {
    magic: u32,
    base: usize,
    len: u32,
    streams: Mutex<Vec<usize>>,
}

struct Stream {
    magic: u32,
    client: usize,
    frames: u32,
    channels: u32,
    status: Status,
}

struct World {
    clients: HashMap<usize, Box<Client>>,
    streams: HashMap<usize, Box<Stream>>,
}

static WORLD: std::sync::LazyLock<Mutex<World>> =
    std::sync::LazyLock::new(|| Mutex::new(World { clients: HashMap::new(), streams: HashMap::new() }));

fn u32_at(p: *const u8, off: usize) -> u32 {
    unsafe { std::ptr::read_unaligned(p.add(off) as *const u32) }
}

fn ptr_at(p: *const u8, off: usize) -> *mut f32 {
    unsafe { std::ptr::read_unaligned(p.add(off) as *const *mut f32) }
}

/// Bytes of one output block, or 0 when the size is not one we will write.
fn block_bytes(frames: u32, channels: u32) -> usize {
    if !(MIN_FRAMES..=MAX_FRAMES).contains(&frames) || !frames.is_power_of_two() || !(1..=2).contains(&channels) {
        return 0;
    }
    frames as usize * channels as usize * 4
}

fn silence(client: &Client, output: *mut f32, frames: u32, channels: u32) {
    static WARNED: AtomicU32 = AtomicU32::new(0);
    let n = block_bytes(frames, channels);
    if output.is_null() || n == 0 {
        return;
    }
    let start = output as usize;
    let end = start.saturating_add(n);
    let base = client.base;
    let limit = base.saturating_add(client.len as usize);
    if start >= base && end <= limit && start % 16 == 0 {
        unsafe { std::ptr::write_bytes(output as *mut u8, 0, n) };
        return;
    }
    if WARNED.fetch_add(1, Ordering::Relaxed) < 4 {
        crate::xlog_warn!("xdsp: output {start:#x} ({n} bytes) is outside the registered buffer – left untouched");
    }
}

/// XDspConnectWithMaximumStreamLimit(buffer, bytes, impulseSeconds, maxStreams, maxChannels, &handle)
pub unsafe extern "system" fn XDspConnectWithMaximumStreamLimit(
    buffer: *mut f32,
    bytes: u32,
    impulse_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    handle: *mut *mut c_void,
) -> HRESULT {
    if buffer.is_null() || handle.is_null() || bytes == 0 || max_streams == 0 || !(1..=2).contains(&max_channels) {
        if !handle.is_null() {
            *handle = std::ptr::null_mut();
        }
        return E_INVALIDARG;
    }
    let c = Box::new(Client { magic: CLIENT_MAGIC, base: buffer as usize, len: bytes, streams: Mutex::new(Vec::new()) });
    let p = Box::into_raw(c);
    let mut w = WORLD.lock().unwrap();
    if w.clients.try_reserve(1).is_err() {
        drop(Box::from_raw(p));
        *handle = std::ptr::null_mut();
        return E_OUTOFMEMORY;
    }
    w.clients.insert(p as usize, Box::from_raw(p));
    *handle = p as *mut c_void;
    crate::xlog_info!(
        "xdsp: client connected ({bytes:#x} byte buffer, {max_streams} streams, {max_channels} ch, impulse {impulse_seconds}s) – commands finish at once, output is silence"
    );
    S_OK
}

/// XDspActivate(client, &params, &status, &stream)
pub unsafe extern "system" fn XDspActivate(
    client: *mut c_void,
    params: *const c_void,
    status: *mut *mut c_void,
    stream: *mut *mut c_void,
) -> HRESULT {
    if client.is_null() || params.is_null() || status.is_null() || stream.is_null() {
        return E_INVALIDARG;
    }
    let raw = params as *const u8;
    let frames = u32_at(raw, 8);
    let channels = u32_at(raw, 12);
    let mut w = WORLD.lock().unwrap();
    if !w.clients.get(&(client as usize)).is_some_and(|c| c.magic == CLIENT_MAGIC) {
        return E_INVALIDARG;
    }
    let s = Box::new(Stream {
        magic: STREAM_MAGIC,
        client: client as usize,
        frames,
        channels,
        status: Status {
            last_submitted: AtomicU32::new(0),
            last_processed: AtomicU32::new(0),
            first_failed: AtomicU32::new(0),
            result: AtomicI32::new(S_OK),
        },
    });
    let p = Box::into_raw(s);
    if w.streams.try_reserve(1).is_err() {
        drop(Box::from_raw(p));
        return E_OUTOFMEMORY;
    }
    let status_addr = &(*p).status as *const Status as *mut Status;
    w.streams.insert(p as usize, Box::from_raw(p));
    if let Some(c) = w.clients.get(&(client as usize)) {
        c.streams.lock().unwrap().push(p as usize);
    }
    *status = status_addr as *mut c_void;
    *stream = p as *mut c_void;
    crate::xlog_info!("xdsp: stream activated ({frames} frames, {channels} ch) – status is updated in the submit call");
    S_OK
}

/// XDspSubmitCommand(stream, &command, &sequence)
pub unsafe extern "system" fn XDspSubmitCommand(stream: *mut c_void, command: *const c_void, sequence: *mut u32) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    let mut w = WORLD.lock().unwrap();
    let (output, frames, channels, client_key, seq) = {
        let Some(s) = w.streams.get_mut(&(stream as usize)) else {
            return E_INVALIDARG;
        };
        if s.magic != STREAM_MAGIC {
            return E_INVALIDARG;
        }
        let seq = s.status.last_submitted.fetch_add(1, Ordering::Release).wrapping_add(1);
        s.status.last_processed.store(seq, Ordering::Release);
        s.status.first_failed.store(0, Ordering::Release);
        s.status.result.store(S_OK, Ordering::Release);
        (ptr_at(command as *const u8, 16), s.frames, s.channels, s.client, seq)
    };
    if let Some(c) = w.clients.get(&client_key) {
        silence(c, output, frames, channels);
    }
    if !sequence.is_null() {
        *sequence = seq;
    }
    S_OK
}

/// XDspDeactivate(stream)
pub unsafe extern "system" fn XDspDeactivate(stream: *mut c_void) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    let mut w = WORLD.lock().unwrap();
    let Some(s) = w.streams.remove(&(stream as usize)) else {
        return E_INVALIDARG;
    };
    if s.magic != STREAM_MAGIC {
        return E_INVALIDARG;
    }
    if let Some(c) = w.clients.get(&s.client) {
        c.streams.lock().unwrap().retain(|p| *p != stream as usize);
    }
    S_OK
}

/// XDspDisconnect(client)
pub unsafe extern "system" fn XDspDisconnect(client: *mut c_void) -> HRESULT {
    if client.is_null() {
        return E_INVALIDARG;
    }
    let mut w = WORLD.lock().unwrap();
    let Some(c) = w.clients.remove(&(client as usize)) else {
        return E_INVALIDARG;
    };
    if c.magic != CLIENT_MAGIC {
        return E_INVALIDARG;
    }
    let streams = std::mem::take(&mut *c.streams.lock().unwrap());
    for p in streams {
        w.streams.remove(&p);
    }
    S_OK
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![
        XDspConnectWithMaximumStreamLimit,
        XDspActivate,
        XDspSubmitCommand,
        XDspDeactivate,
        XDspDisconnect,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_connect_without_a_buffer_or_a_handle() {
        unsafe {
            let mut h = 1 as *mut c_void;
            assert_eq!(XDspConnectWithMaximumStreamLimit(std::ptr::null_mut(), 0x1_0000, 0, 8, 1, &mut h), E_INVALIDARG);
            assert!(h.is_null());
            let mut buf = vec![1f32; 64];
            assert_eq!(
                XDspConnectWithMaximumStreamLimit(buf.as_mut_ptr(), (buf.len() * 4) as u32, 0, 8, 3, &mut h),
                E_INVALIDARG
            );
        }
    }

    #[test]
    fn command_finishes_immediately_and_silences_only_its_block() {
        unsafe {
            // 16-byte aligned, the alignment the output pointer has to have
            let mut storage = vec![1f32; 4096 + 8];
            let addr = storage.as_mut_ptr() as usize;
            let aligned = (addr + 15) & !15;
            let buf = aligned as *mut f32;
            let floats = 4096 - (aligned - addr) / 4;
            let mut client = std::ptr::null_mut();
            assert_eq!(XDspConnectWithMaximumStreamLimit(buf, (floats * 4) as u32, 0, 4, 1, &mut client), S_OK);

            // type, options, blockFrameCount, channelCount, impulse length, padding, pointer
            let params: [u32; 8] = [0, 0, 512, 1, 0, 0, 0, 0];
            let mut status: *mut c_void = std::ptr::null_mut();
            let mut stream = std::ptr::null_mut();
            assert_eq!(XDspActivate(client, params.as_ptr() as *const c_void, &mut status, &mut stream), S_OK);
            assert!(!status.is_null() && !stream.is_null());

            let output = buf.add(16);
            let mut raw = [0u8; 32];
            raw[16..24].copy_from_slice(&(output as usize).to_le_bytes());
            let mut seq = 0u32;
            assert_eq!(XDspSubmitCommand(stream, raw.as_ptr() as *const c_void, &mut seq), S_OK);
            assert_eq!(seq, 1);
            let st = &*(status as *const Status);
            assert_eq!(st.last_submitted.load(Ordering::Acquire), 1);
            assert_eq!(st.last_processed.load(Ordering::Acquire), 1);
            assert_eq!(st.first_failed.load(Ordering::Acquire), 0);
            let samples = std::slice::from_raw_parts(buf, floats);
            assert!(samples[..16].iter().all(|s| *s == 1.0), "samples before the block stay");
            assert!(samples[16..16 + 512].iter().all(|s| *s == 0.0));
            assert_eq!(samples[16 + 512], 1.0, "the sample after the block stays");

            let mut other = [2f32; 8];
            raw[16..24].copy_from_slice(&(other.as_mut_ptr() as usize).to_le_bytes());
            assert_eq!(XDspSubmitCommand(stream, raw.as_ptr() as *const c_void, &mut seq), S_OK);
            assert_eq!(seq, 2);
            assert!(other.iter().all(|s| *s == 2.0));

            assert_eq!(XDspDeactivate(stream), S_OK);
            assert_eq!(XDspDeactivate(stream), E_INVALIDARG);
            assert_eq!(XDspDisconnect(client), S_OK);
            assert_eq!(XDspDisconnect(client), E_INVALIDARG);
        }
    }
}
