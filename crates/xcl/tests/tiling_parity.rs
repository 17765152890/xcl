//! Reserved-resource tiling parity between XCL's D3D12.X path and the host D3D12 driver (#480).
//!
//! Oracle split:
//! - Host D3D12 driver is the oracle for what the PC also executes: the same reserved resource
//!   created through XCL's `vram::place_pages` (`CreateReservedResource` with layout rewritten to
//!   `D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE`) must return identical `GetResourceTiling`
//!   values whether queried through XCL's device vtable slot or directly through the driver's
//!   original `ID3D12Device::GetResourceTiling` entry.
//! - Console-only D3D12.X semantics belong in `xg_oracle.rs` (`XCL_SAMPLE_XG`), not here.
//! - Measured RDNA4 numbers are pinned as regression constants so a broken layout rewrite or
//!   format mapping fails loudly.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, layout, real, vram, Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tiling {
    total_tiles: u32,
    width_in_tiles: u32,
    height_in_tiles: u16,
    tile_shape: (u32, u32, u32),
}

struct Case {
    name: &'static str,
    format: DXGI_FORMAT,
    width: u64,
    height: u32,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
    clear_depth: bool,
    // Pinned regression constants measured on:
    // AMD Radeon RX 9060 XT (RDNA4, gfx1200), Adrenalin 26.10.1 / Windows driver 32.0.32015.7029,
    // Windows 11 26200 (D3D12_TILED_RESOURCES_TIER_4).
    expected: Tiling,
}

const RDNA4_CASES: &[Case] = &[
    Case {
        name: "R8G8B8A8_UNORM 8x8",
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        width: 8,
        height: 8,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        expected: Tiling {
            total_tiles: 1,
            width_in_tiles: 1,
            height_in_tiles: 1,
            tile_shape: (128, 128, 1),
        },
    },
    Case {
        name: "R8G8B8A8_UNORM 256x256",
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        width: 256,
        height: 256,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        expected: Tiling {
            total_tiles: 4,
            width_in_tiles: 2,
            height_in_tiles: 2,
            tile_shape: (128, 128, 1),
        },
    },
    Case {
        name: "BC1_UNORM 256x256",
        format: DXGI_FORMAT_BC1_UNORM,
        width: 256,
        height: 256,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        expected: Tiling {
            total_tiles: 1,
            width_in_tiles: 1,
            height_in_tiles: 1,
            tile_shape: (512, 256, 1),
        },
    },
    Case {
        name: "D32_FLOAT 8x8",
        format: DXGI_FORMAT_D32_FLOAT,
        width: 8,
        height: 8,
        flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        state: D3D12_RESOURCE_STATE_DEPTH_WRITE,
        clear_depth: true,
        expected: Tiling {
            total_tiles: 1,
            width_in_tiles: 1,
            height_in_tiles: 1,
            tile_shape: (128, 128, 1),
        },
    },
];

type GetResourceTilingFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    *mut u32,
    *mut D3D12_PACKED_MIP_INFO,
    *mut D3D12_TILE_SHAPE,
    *mut u32,
    u32,
    *mut D3D12_SUBRESOURCE_TILING,
);

/// Query `GetResourceTiling` through XCL's adopted device vtable slot.
unsafe fn query_via_xcl_slot(dev: &ID3D12Device, res: &ID3D12Resource) -> Tiling {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut sub_count = 1u32;
    let mut sub = D3D12_SUBRESOURCE_TILING::default();
    dev.GetResourceTiling(
        res,
        Some(&mut total),
        Some(&mut packed),
        Some(&mut shape),
        Some(&mut sub_count),
        0,
        &mut sub,
    );
    assert_eq!(sub_count, 1, "expected 1 subresource tiling entry");
    Tiling {
        total_tiles: total,
        width_in_tiles: sub.WidthInTiles,
        height_in_tiles: sub.HeightInTiles,
        tile_shape: (shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels),
    }
}

/// Query `GetResourceTiling` directly via the host driver's original vtable entry on the same
/// returned `ID3D12Resource`, bypassing XCL's device vtable.
unsafe fn query_via_driver(dev: &ID3D12Device, res: &ID3D12Resource, slot: usize) -> Tiling {
    let f: GetResourceTilingFn = std::mem::transmute(real::orig_fn(dev.as_raw(), slot));
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut sub_count = 1u32;
    let mut sub = D3D12_SUBRESOURCE_TILING::default();
    f(
        dev.as_raw(),
        res.as_raw(),
        &mut total,
        &mut packed,
        &mut shape,
        &mut sub_count,
        0,
        &mut sub,
    );
    assert_eq!(sub_count, 1, "expected 1 subresource tiling entry");
    Tiling {
        total_tiles: total,
        width_in_tiles: sub.WidthInTiles,
        height_in_tiles: sub.HeightInTiles,
        tile_shape: (shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels),
    }
}

#[test]
fn reserved_resource_tiling_matches_host_driver() {
    if d3d12x::backend() != Backend::Real {
        runner::skip_test("no GPU");
        return;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        runner::skip_test("XCL_VRAM_ARENAS disables reserved-resource page placement");
        return;
    }

    unsafe {
        let mut dev_raw: *mut c_void = std::ptr::null_mut();
        let hr = d3d12x::D3D12XboxCreateDevice(
            std::ptr::null_mut(),
            std::ptr::null(),
            &ID3D12Device::IID as *const windows::core::GUID as *const c_void,
            &mut dev_raw,
        );
        assert!(hr >= 0 && !dev_raw.is_null(), "D3D12XboxCreateDevice failed: {hr:#x}");
        let dev = ID3D12Device::from_raw(dev_raw);

        let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        if dev
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &mut opts as *mut _ as *mut c_void,
                std::mem::size_of_val(&opts) as u32,
            )
            .is_err()
            || opts.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED
        {
            runner::skip_test("no tiled resources");
            return;
        }
        if opts.ResourceHeapTier.0 < D3D12_RESOURCE_HEAP_TIER_2.0 {
            runner::skip_test("no tier 2 resource heaps");
            return;
        }

        let tiling_slot = layout::slot(
            layout::find("ID3D12Device").expect("ID3D12Device in layout"),
            "GetResourceTiling",
        )
        .expect("GetResourceTiling slot");
        let res_iid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

        for (i, case) in RDNA4_CASES.iter().enumerate() {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: case.width,
                Height: case.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: case.format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: case.flags,
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: case.format,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
                },
            };
            let clear_ptr = if case.clear_depth {
                &clear as *const D3D12_CLEAR_VALUE as *const c_void
            } else {
                std::ptr::null()
            };

            let va = 0x20_0000_0000usize + i * 0x10_0000;
            let mut res_raw: *mut c_void = std::ptr::null_mut();
            let placed = vram::place_pages(
                dev.as_raw(),
                va,
                &desc,
                case.state.0 as u32,
                clear_ptr,
                None,
                res_iid,
                &mut res_raw,
            );
            let hr = placed.unwrap_or_else(|| panic!("{}: vram::place_pages returned None", case.name));
            assert!(hr >= 0 && !res_raw.is_null(), "{}: CreateReservedResource failed: {hr:#x}", case.name);
            let res = ID3D12Resource::from_raw(res_raw);

            // 1. Same resource, host driver as oracle: XCL's device slot and the driver's
            //    original GetResourceTiling must return identical tiling.
            let xcl_tiling = query_via_xcl_slot(&dev, &res);
            let driver_tiling = query_via_driver(&dev, &res, tiling_slot);
            assert_eq!(
                xcl_tiling, driver_tiling,
                "{}: XCL slot ({xcl_tiling:?}) != host driver ({driver_tiling:?})",
                case.name
            );

            // 2. Measured RDNA4 numbers pinned as regression constants.
            assert_eq!(
                xcl_tiling, case.expected,
                "{}: tiling ({xcl_tiling:?}) != pinned RDNA4 constant ({:?})",
                case.name, case.expected
            );
        }
    }
}
