//! Persistent on-disk pipeline library cache (ID3D12PipelineLibrary1).
//!
//! Eliminates runtime DXIL-to-ISA driver compilation stalls by caching
//! driver-compiled pipeline states across application launches.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

struct Library {
    library: ID3D12PipelineLibrary1,
    /// Backing blob from disk that must outlive the library instance.
    _blob: Vec<u8>,
    added: u64,
}

// SAFETY: D3D12 pipeline library operations are guarded by the internal Mutex.
unsafe impl Send for Library {}

static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
static DIR: OnceLock<PathBuf> = OnceLock::new();

pub static GRAPHICS_LOADED: AtomicU64 = AtomicU64::new(0);
pub static GRAPHICS_STORED: AtomicU64 = AtomicU64::new(0);
pub static COMPUTE_LOADED: AtomicU64 = AtomicU64::new(0);
pub static COMPUTE_STORED: AtomicU64 = AtomicU64::new(0);

const LIBRARY_FILE: &str = "host-pipelines.bin";

/// Folder of the PSO disk cache: XCL_PSO_CACHE, else %LOCALAPPDATA%\xcl\pso.
pub fn cache_dir() -> PathBuf {
    std::env::var("XCL_PSO_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("LOCALAPPDATA").map(|l| PathBuf::from(l).join("xcl").join("pso")))
        .unwrap_or_else(|_| PathBuf::from("xcl-pso-cache"))
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PIPELINE_LIBRARY").map_or(true, |v| v.trim() != "0"))
}

/// Initialize the pipeline library for the active D3D12 device.
pub fn init(device: &ID3D12Device) {
    if !enabled() {
        return;
    }
    let dir = cache_dir();
    let _ = DIR.set(dir.clone());
    let _ = LIBRARY.get_or_init(|| {
        let dev1 = device.cast::<ID3D12Device1>().ok()?;
        let blob = std::fs::read(dir.join(LIBRARY_FILE)).unwrap_or_default();
        let lib = open(&dev1, blob)?;
        Some(Mutex::new(lib))
    });
}

fn library() -> Option<&'static Mutex<Library>> {
    LIBRARY.get().and_then(|opt| opt.as_ref())
}

fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // vkd3d-proton requires a null blob pointer when length is zero to construct an empty library;
        // passing an empty slice through windows-rs forwards a dangling non-null pointer (0x1) that fails validation.
        let ptr = if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr().cast()
        };
        let mut raw = std::ptr::null_mut();
        unsafe {
            (windows::core::Interface::vtable(device).CreatePipelineLibrary)(
                windows::core::Interface::as_raw(device),
                ptr,
                bytes.len(),
                &ID3D12PipelineLibrary1::IID,
                &mut raw,
            )
            .and_then(|| windows::core::Type::from_abi(raw))
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                crate::xlog_info!("D3D12.X: loaded host pipeline library ({} B)", blob.len());
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => {
                crate::xlog_info!("D3D12.X: pipeline library rejected ({e}), starting fresh library");
            }
        }
    }
    let library = make(&[]).ok()?;
    Some(Library { library, _blob: Vec::new(), added: 0 })
}

/// Computes a unique name for a graphics pipeline based on its state and translated shaders.
pub fn graphics_pso_name(
    desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    vs: &[u8],
    ps: Option<&[u8]>,
    gs: Option<&[u8]>,
) -> String {
    let mut h1 = crate::gpu::shader::fnv64(vs);
    let mut h2 = 0xcbf2_9ce4_8422_2325u64;

    if let Some(p) = ps {
        h1 = (h1 ^ crate::gpu::shader::fnv64(p)).wrapping_mul(0x100_0000_01b3);
    }
    if let Some(g) = gs {
        h2 = (h2 ^ crate::gpu::shader::fnv64(g)).wrapping_mul(0x100_0000_01b3);
    }

    let rs_ptr = desc.pRootSignature.as_ref().map_or(0, |rs| windows::core::Interface::as_raw(rs) as usize as u64);
    h1 = (h1 ^ rs_ptr).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.PrimitiveTopologyType.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.NumRenderTargets as u64)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.DSVFormat.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.SampleDesc.Count as u64)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.RasterizerState.CullMode.0 as u64)).wrapping_mul(0x100_0000_01b3);

    format!("gp_{h1:016x}{h2:016x}")
}

/// Computes a unique name for a compute pipeline based on its root signature and compute shader.
pub fn compute_pso_name(desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC, cs: &[u8]) -> String {
    let mut h1 = crate::gpu::shader::fnv64(cs);
    let rs_ptr = desc.pRootSignature.as_ref().map_or(0, |rs| windows::core::Interface::as_raw(rs) as usize as u64);
    h1 = (h1 ^ rs_ptr).wrapping_mul(0x100_0000_01b3);
    let h2 = (desc.Flags.0 as u64).wrapping_mul(0x100_0000_01b3) ^ 0x9e3779b97f4a7c15u64;

    format!("cp_{h1:016x}{h2:016x}")
}

/// Attempts to load a graphics pipeline from the persistent library.
pub fn load_graphics(
    name: &str,
    desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
) -> Option<ID3D12PipelineState> {
    let lib_mutex = library()?;
    let l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary load call with valid pointer and descriptor references.
    let res: windows::core::Result<ID3D12PipelineState> = unsafe {
        l.library.LoadGraphicsPipeline(PCWSTR(wide.as_ptr()), desc)
    };
    match res {
        Ok(pso) => {
            GRAPHICS_LOADED.fetch_add(1, Ordering::Relaxed);
            Some(pso)
        }
        Err(_) => None,
    }
}

/// Stores a newly compiled graphics pipeline into the library.
pub fn store_graphics(name: &str, pso: &ID3D12PipelineState) {
    let Some(lib_mutex) = library() else { return };
    let mut l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary store call with valid wide name and pipeline state pointer.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) }.is_ok() {
        l.added += 1;
        GRAPHICS_STORED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Attempts to load a compute pipeline from the persistent library.
pub fn load_compute(
    name: &str,
    desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
) -> Option<ID3D12PipelineState> {
    let lib_mutex = library()?;
    let l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary load call with valid pointer and descriptor references.
    let res: windows::core::Result<ID3D12PipelineState> = unsafe {
        l.library.LoadComputePipeline(PCWSTR(wide.as_ptr()), desc)
    };
    match res {
        Ok(pso) => {
            COMPUTE_LOADED.fetch_add(1, Ordering::Relaxed);
            Some(pso)
        }
        Err(_) => None,
    }
}

/// Stores a newly compiled compute pipeline into the library.
pub fn store_compute(name: &str, pso: &ID3D12PipelineState) {
    let Some(lib_mutex) = library() else { return };
    let mut l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary store call with valid wide name and pipeline state pointer.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) }.is_ok() {
        l.added += 1;
        COMPUTE_STORED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Serializes and writes the library to disk if new pipelines were stored.
pub fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);

    let (Some(dir), Some(lib_mutex)) = (DIR.get(), library()) else { return };

    {
        let mut last = LAST.lock().unwrap();
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }

    let blob = {
        let mut l = lib_mutex.lock().unwrap();
        if l.added == 0 {
            return;
        }
        // SAFETY: allocating exact byte buffer queried from GetSerializedSize.
        let bytes = unsafe {
            let size = l.library.GetSerializedSize();
            let mut bytes = vec![0u8; size];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                bytes
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: pipeline library serialization failed: {e}");
                return;
            }
        }
    };

    let target = dir.join(LIBRARY_FILE);
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|()| write_atomically(&target, &blob)) {
        crate::xlog_warn!("D3D12.X: saving pipeline library to {} failed: {e}", target.display());
    } else {
        crate::xlog_info!(
            "D3D12.X: saved pipeline library ({} B, {} graphics loaded, {} compute loaded)",
            blob.len(),
            GRAPHICS_LOADED.load(Ordering::Relaxed),
            COMPUTE_LOADED.load(Ordering::Relaxed)
        );
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_names_are_deterministic_and_differentiate_states() {
        let vs = b"dxil_vertex_shader_bytecode_test";
        let ps1 = b"dxil_pixel_shader_bytecode_test_1";
        let ps2 = b"dxil_pixel_shader_bytecode_test_2";

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;

        let n1 = graphics_pso_name(&desc, vs, Some(ps1), None);
        let n1_dup = graphics_pso_name(&desc, vs, Some(ps1), None);
        let n2 = graphics_pso_name(&desc, vs, Some(ps2), None);

        assert_eq!(n1, n1_dup);
        assert_ne!(n1, n2);

        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        let n3 = graphics_pso_name(&desc, vs, Some(ps1), None);
        assert_ne!(n1, n3);
    }

    #[test]
    fn compute_pipeline_names_differentiate_flags_and_shaders() {
        let cs1 = b"dxil_compute_shader_bytecode_test_1";
        let cs2 = b"dxil_compute_shader_bytecode_test_2";

        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        let n1 = compute_pso_name(&desc, cs1);
        let n1_dup = compute_pso_name(&desc, cs1);
        let n2 = compute_pso_name(&desc, cs2);

        assert_eq!(n1, n1_dup);
        assert_ne!(n1, n2);

        desc.Flags = D3D12_PIPELINE_STATE_FLAGS(1);
        let n3 = compute_pso_name(&desc, cs1);
        assert_ne!(n1, n3);
    }

    #[test]
    fn corrupted_blob_falls_back_to_empty_library() {
        let Ok(dev) = crate::gpu::agility::create_device(
            None::<&windows::core::IUnknown>,
            windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0,
        ) else {
            return;
        };
        let Ok(dev1) = dev.cast::<ID3D12Device1>() else {
            return;
        };

        let corrupted = vec![0x7f; 1024];
        let lib = open(&dev1, corrupted).expect("failed to create fallback empty library after rejecting corrupted blob");
        assert!(lib._blob.is_empty());
        let size = unsafe { lib.library.GetSerializedSize() };
        assert!(size > 0);
    }

    #[test]
    fn atomic_file_writing_prevents_partial_writes() {
        let dir = std::env::temp_dir().join(format!("xcl-pso-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-library.bin");

        let data = b"serialized_pipeline_library_mock_payload";
        assert!(write_atomically(&path, data).is_ok());
        assert_eq!(std::fs::read(&path).unwrap(), data);

        let data2 = b"serialized_pipeline_library_second_write";
        assert!(write_atomically(&path, data2).is_ok());
        assert_eq!(std::fs::read(&path).unwrap(), data2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
