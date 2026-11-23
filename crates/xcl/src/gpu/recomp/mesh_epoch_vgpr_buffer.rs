//! Checked buffer layout for validation-only epoch VGPR state.
//!
//! This is deliberately not wired into the epoch shell until every vector
//! source and destination can use the same abstraction.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BufferVgprLayout {
    groups: u32,
    registers: u32,
    dwords: u32,
}

impl BufferVgprLayout {
    pub(crate) fn checked(groups: u32, registers: u32) -> Result<Self, &'static str> {
        if groups == 0 || registers == 0 || registers > 256 {
            return Err("buffer VGPR layout dimensions are invalid");
        }
        let dwords = groups
            .checked_mul(2)
            .and_then(|n| n.checked_mul(64))
            .and_then(|n| n.checked_mul(registers))
            .ok_or("buffer VGPR layout overflows")?;
        dwords
            .checked_mul(4)
            .ok_or("buffer VGPR byte size overflows")?;
        Ok(Self {
            groups,
            registers,
            dwords,
        })
    }

    pub(crate) fn bytes(self) -> u32 {
        self.dwords * 4
    }

    pub(crate) fn byte_offset(
        self,
        group: u32,
        wave: u32,
        lane: u32,
        register: u32,
    ) -> Option<u32> {
        if group >= self.groups || wave >= 2 || lane >= 64 || register >= self.registers {
            return None;
        }
        group
            .checked_mul(2)?
            .checked_add(wave)?
            .checked_mul(64)?
            .checked_add(lane)?
            .checked_mul(self.registers)?
            .checked_add(register)?
            .checked_mul(4)
    }

    pub(crate) fn hlsl_helpers(self) -> String {
        format!(
            r#"
bool xcl_vgpr_address(RWByteAddressBuffer b,uint group,uint wave,uint lane,uint reg,out uint address){{
  uint bytes;b.GetDimensions(bytes);
  bool indices=group<{groups}u&&wave<2u&&lane<64u&&reg<{registers}u;
  uint64_t item=(((uint64_t)group*2ull+wave)*64ull+lane)*{registers}ull+reg;
  uint64_t offset=item*4ull;
  bool valid=indices&&offset+4ull<=(uint64_t)bytes&&offset<=0xffffffffull;
  address=(uint)offset;return valid;
}}
bool xcl_vgpr_load(RWByteAddressBuffer b,uint group,uint wave,uint lane,uint reg,out uint value){{
  uint address;if(!xcl_vgpr_address(b,group,wave,lane,reg,address))return false;
  value=b.Load(address);return true;
}}
bool xcl_vgpr_store(RWByteAddressBuffer b,uint group,uint wave,uint lane,uint reg,uint value){{
  uint address;if(!xcl_vgpr_address(b,group,wave,lane,reg,address))return false;
  b.Store(address,value);return true;
}}
"#,
            groups = self.groups,
            registers = self.registers
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_layout_isolated_slices_and_edges() {
        let layout = BufferVgprLayout::checked(3, 256).unwrap();
        assert_eq!(layout.bytes(), 3 * 2 * 64 * 256 * 4);
        assert_eq!(layout.byte_offset(0, 0, 0, 0), Some(0));
        assert_eq!(layout.byte_offset(0, 1, 0, 0), Some(64 * 256 * 4));
        assert_eq!(layout.byte_offset(1, 0, 0, 0), Some(2 * 64 * 256 * 4));
        assert_eq!(layout.byte_offset(2, 1, 63, 255), Some(layout.bytes() - 4));
        assert_eq!(layout.byte_offset(3, 0, 0, 0), None);
        assert_eq!(layout.byte_offset(0, 2, 0, 0), None);
        assert_eq!(layout.byte_offset(0, 0, 64, 0), None);
        assert_eq!(layout.byte_offset(0, 0, 0, 256), None);
        assert!(BufferVgprLayout::checked(u32::MAX, 256).is_err());
    }

    #[test]
    fn helpers_compile_and_preserve_two_wave_aliases_live() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = BufferVgprLayout::checked(1, 4).unwrap();
        let hlsl = format!(
            r#"RWByteAddressBuffer res:register(u1);RWByteAddressBuffer memory:register(u2);
{}
[WaveSize(64)][numthreads(128,1,1)]void main(uint gi:SV_GroupIndex){{
 uint wave=gi>>6u,lane=gi&63u,value=0u;
 xcl_vgpr_store(memory,0u,wave,lane,0u,0x1000u+gi);
 DeviceMemoryBarrierWithGroupSync();
 bool ok=xcl_vgpr_load(memory,0u,wave,lane,0u,value);
 if(ok)xcl_vgpr_store(memory,0u,wave,lane,0u,value);
 res.Store(4u*gi,ok?value:0u);
}}"#,
            layout.hlsl_helpers()
        );
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        if !super::super::runner::wave64_supported().unwrap_or(false) {
            return;
        }
        let result = super::super::runner::run(
            &dxil,
            &super::super::runner::KernelState::default(),
            &vec![0; layout.dwords as usize],
        )
        .unwrap();
        assert_eq!(result.raw[0], 0x1000);
        assert_eq!(result.raw[63], 0x103f);
        assert_eq!(result.raw[64], 0x1040);
        assert_eq!(result.raw[127], 0x107f);
        assert_eq!(result.mem[0], 0x1000);
        assert_eq!(result.mem[64 * 4], 0x1040);
    }
}
