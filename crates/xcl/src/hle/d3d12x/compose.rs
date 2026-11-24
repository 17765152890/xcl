//! Display planes: a title that presents more than one plane (`PresentX` with a game plane and a
//! UI plane, as a retail title does on the Series S profile – the game at 1920x1080, the UI at
//! 3840x2160) has the console's display hardware scale them to the output and blend them, the
//! first plane at the bottom. XCL draws them into its swap chain the same way: plane 0 scaled
//! and opaque, every later plane scaled and blended over it (premultiplied alpha).

use std::mem::transmute_copy;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const HLSL: &str = r#"
#define RS "DescriptorTable(SRV(t0)), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> plane : register(t0);
SamplerState linear_clamp : register(s0);
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
[RootSignature(RS)]
float4 ps(V i) : SV_Target { return plane.Sample(linear_clamp, i.uv); }
"#;

/// Most planes a title hands PresentX (the console has three: game, UI and the system's).
pub const MAX_PLANES: usize = 3;

pub struct Compositor {
    rs: ID3D12RootSignature,
    /// [opaque, blended] for the format they were made for
    pso: [ID3D12PipelineState; 2],
    fmt: DXGI_FORMAT,
    srvs: ID3D12DescriptorHeap,
    rtv: ID3D12DescriptorHeap,
    srv_step: usize,
}

unsafe impl Send for Compositor {}

impl Compositor {
    /// The pipelines for a swap chain of `fmt`.
    pub unsafe fn new(dev: &ID3D12Device, fmt: DXGI_FORMAT) -> Result<Compositor, String> {
        let vs = crate::gpu::recomp::dxc::compile(HLSL, "vs", "vs_6_0", &[])?;
        let ps = crate::gpu::recomp::dxc::compile(HLSL, "ps", "ps_6_0", &[])?;
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &vs).map_err(|e| format!("CreateRootSignature: {e}"))?;
        let make = |blend: bool| -> Result<ID3D12PipelineState, String> {
            let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: transmute_copy(&rs),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let rt = &mut pd.BlendState.RenderTarget[0];
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            if blend {
                rt.BlendEnable = true.into();
                (rt.SrcBlend, rt.DestBlend, rt.BlendOp) = (D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD);
                (rt.SrcBlendAlpha, rt.DestBlendAlpha, rt.BlendOpAlpha) = (D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD);
            }
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RTVFormats[0] = fmt;
            dev.CreateGraphicsPipelineState(&pd).map_err(|e| format!("CreateGraphicsPipelineState: {e}"))
        };
        let pso = [make(false)?, make(true)?];
        let srvs: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: MAX_PLANES as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("CreateDescriptorHeap: {e}"))?;
        let rtv: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 })
            .map_err(|e| format!("CreateDescriptorHeap: {e}"))?;
        let srv_step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        Ok(Compositor { rs, pso, fmt, srvs, rtv, srv_step })
    }

    pub fn format(&self) -> DXGI_FORMAT {
        self.fmt
    }

    /// Record on `list`: `planes` (bottom first) scaled to `target` (`w`×`h`, in RENDER_TARGET
    /// state). The planes are read in PIXEL_SHADER_RESOURCE state (promoted from COMMON, as the
    /// copy of a single plane is). The previous use of the descriptors must be done.
    pub unsafe fn record(&self, dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, target: &ID3D12Resource, w: u32, h: u32, planes: &[ID3D12Resource]) {
        let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(target, None, rtv);
        let cpu = self.srvs.GetCPUDescriptorHandleForHeapStart();
        let gpu = self.srvs.GetGPUDescriptorHandleForHeapStart();
        for (i, plane) in planes.iter().take(MAX_PLANES).enumerate() {
            let d = plane.GetDesc();
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: srv_format(d.Format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
            };
            dev.CreateShaderResourceView(plane, Some(&srv), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + i * self.srv_step });
        }
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
        list.SetGraphicsRootSignature(&self.rs);
        list.SetDescriptorHeaps(&[Some(self.srvs.clone())]);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for i in 0..planes.len().min(MAX_PLANES) {
            list.SetPipelineState(&self.pso[(i > 0) as usize]);
            list.SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + (i * self.srv_step) as u64 });
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
}

/// A view format for a display plane's (possibly typeless) format.
fn srv_format(f: DXGI_FORMAT) -> DXGI_FORMAT {
    match f {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        f => f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typeless_planes_get_unorm_views() {
        assert_eq!(srv_format(DXGI_FORMAT_R8G8B8A8_TYPELESS), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(srv_format(DXGI_FORMAT_R8G8B8A8_UNORM), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(srv_format(DXGI_FORMAT_R10G10B10A2_TYPELESS), DXGI_FORMAT_R10G10B10A2_UNORM);
    }
}
