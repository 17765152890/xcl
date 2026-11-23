//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (128 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// bytes of title address space an arena stands for at least (and of video memory it takes)
const ARENA: usize = 128 << 20;
/// arenas start and end at multiples of this
const ARENA_ALIGN: usize = 32 << 20;

struct Arena {
    base: usize,
    size: usize,
    heap: ID3D12Heap,
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arena {}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    if let Some(textures) = PLACED_TEXTURES.lock().unwrap().as_mut() {
        textures.remove(&resource);
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// Where a new arena for `[va, va + bytes)` goes between the `arenas` there are (`(base, size)`
/// each): from the 32 MiB boundary below `va`, 128 MiB or what the resource needs, cut back to
/// the gap between the neighbouring arenas so that no two arenas stand for the same title
/// address (two heaps for one address: resources there no longer alias, and every placement
/// missing the first arena costs 128 MiB of video memory – #104). None: `[va, va + bytes)`
/// crosses the edge of an arena there is – no heap can hold the resource whole.
fn new_arena_range(arenas: &[(usize, usize)], va: usize, bytes: usize) -> Option<(usize, usize)> {
    let end = va + bytes;
    let mut base = va & !(ARENA_ALIGN - 1);
    let mut top = base + (end - base).div_ceil(ARENA_ALIGN).max(ARENA / ARENA_ALIGN) * ARENA_ALIGN;
    for &(b, s) in arenas {
        if b + s <= va {
            base = base.max(b + s);
        } else if b >= end {
            top = top.min(b);
        } else {
            return None;
        }
    }
    Some((base, top - base))
}

/// The arena holding `[va, va + bytes)` – an existing one, or a new one next to the existing
/// ones (`new_arena_range`; None: no video memory, or the range crosses an arena's edge).
unsafe fn arena_for(dev: &ID3D12Device, va: usize, bytes: usize) -> Option<(ID3D12Heap, u64)> {
    let mut g = ARENAS.lock().unwrap();
    if let Some(a) = g.iter().find(|a| a.base <= va && va + bytes <= a.base + a.size) {
        return Some((a.heap.clone(), (va - a.base) as u64));
    }
    let ranges: Vec<(usize, usize)> = g.iter().map(|a| (a.base, a.size)).collect();
    let Some((base, size)) = new_arena_range(&ranges, va, bytes) else {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE.swap(true, std::sync::atomic::Ordering::AcqRel) {
            crate::xlog_warn!("D3D12.X: a GPU-owned texture at {va:#x} ({bytes:#x} B) crosses the edge of a video-memory arena – a resource of its own (it aliases nothing; only the first such one is logged)");
        }
        return None;
    };
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // MSAA targets need 4 MiB placement
        Alignment: D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {base:#x}: CreateHeap: {e}");
        return None;
    }
    let heap = heap?;
    crate::xlog_info!("D3D12.X: video-memory arena {} for title addresses {base:#x}..{:#x}", g.len(), base + size);
    g.push(Arena { base, size, heap: heap.clone() });
    Some((heap, (va - base) as u64))
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description): placed in the arena
/// for that address (None: it cannot be – the caller makes a resource of its own).
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let (heap, offset) = arena_for(dev, va, info.SizeInBytes as usize)?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset as usize, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn arenas_never_overlap() {
        // the first placement: 128 MiB from the 32 MiB boundary below it
        let (b0, s0) = new_arena_range(&[], 0x1_0400_0000 + 4 * MIB, 16 * MIB).unwrap();
        assert_eq!((b0, s0), (0x1_0400_0000, 128 * MIB));
        let arenas = vec![(b0, s0)];
        // #104's sequence: a placement starting below that arena and ending inside it crosses
        // its edge – no arena (it would overlap), the resource is made on its own
        assert_eq!(new_arena_range(&arenas, 0x1_0200_0000, 48 * MIB), None);
        // one straddling the arena's end
        assert_eq!(new_arena_range(&arenas, 0x1_0400_0000 + 120 * MIB, 16 * MIB), None);
        // below the arena, fitting in front of it: the new arena stops at the old one's base
        let (b1, s1) = new_arena_range(&arenas, 0x1_0200_0000, 16 * MIB).unwrap();
        assert_eq!((b1, s1), (0x1_0200_0000, 32 * MIB));
        let arenas = vec![(b0, s0), (b1, s1)];
        // above both: 128 MiB, starting at the first arena's end
        let (b2, s2) = new_arena_range(&arenas, b0 + s0 + MIB, MIB).unwrap();
        assert_eq!((b2, s2), (b0 + s0, 128 * MIB));
        // a gap between arenas narrower than 128 MiB: the new arena fills exactly the gap
        let arenas = vec![(0x2_0000_0000, 128 * MIB), (0x2_0C00_0000, 128 * MIB)];
        let (b3, s3) = new_arena_range(&arenas, 0x2_0900_0000, MIB).unwrap();
        assert_eq!((b3, s3), (0x2_0800_0000, 64 * MIB));
        // a resource larger than 128 MiB: the arena grows to hold it
        let (b4, s4) = new_arena_range(&[], 0x3_0000_0000, 200 * MIB).unwrap();
        assert_eq!((b4, s4), (0x3_0000_0000, 224 * MIB));
        for ((ba, sa), (bb, sb)) in [((b0, s0), (b1, s1)), ((b0, s0), (b2, s2)), ((b1, s1), (b2, s2))] {
            assert!(ba + sa <= bb || bb + sb <= ba, "{ba:#x}+{sa:#x} overlaps {bb:#x}+{sb:#x}");
        }
    }
}
