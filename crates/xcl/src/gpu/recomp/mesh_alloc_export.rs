//! Exact decoded allocation/export forms for bounded synthetic mesh validation.
//!
//! This adapter does not participate in real PSO eligibility.

use crate::gpu::rdna2::{self, Decoded};

use super::mesh_export::SyntheticRecordLayout;
use super::mesh_reduce::GsAllocRequest;
use super::MESH_HEADER;

pub const M0_REGISTER: u8 = 124;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshExport {
    pub target: u8,
    pub enable: u8,
    pub compressed: bool,
    pub done: bool,
    pub sources: [u16; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationExportOp {
    Allocate { m0: u8 },
    Export(MeshExport),
}

fn field(decoded: &Decoded, name: &str) -> Result<u64, String> {
    rdna2::field_value(decoded.enc, &decoded.words, name)
        .ok_or_else(|| format!("{} lacks {name}", decoded.inst.name))
}

pub fn lower_one(decoded: &Decoded) -> Result<AllocationExportOp, String> {
    match (decoded.inst.name, decoded.enc.name) {
        ("S_SENDMSG", "ENC_SOPP") => {
            let message = field(decoded, "SIMM16")?;
            if message != 9 {
                return Err(format!("unsupported mesh allocation message {message:#x}"));
            }
            Ok(AllocationExportOp::Allocate { m0: M0_REGISTER })
        }
        ("EXP", "ENC_EXP") => {
            if field(decoded, "VM")? != 0 {
                return Err("mesh export VM form is unsupported".into());
            }
            let target = field(decoded, "TGT")? as u8;
            if !matches!(target, 9 | 12..=15 | 20 | 32..=63) {
                return Err(format!("unsupported mesh export target {target}"));
            }
            Ok(AllocationExportOp::Export(MeshExport {
                target,
                enable: field(decoded, "EN")? as u8,
                compressed: field(decoded, "COMPR")? != 0,
                done: field(decoded, "DONE")? != 0,
                sources: [
                    field(decoded, "VSRC0")? as u16,
                    field(decoded, "VSRC1")? as u16,
                    field(decoded, "VSRC2")? as u16,
                    field(decoded, "VSRC3")? as u16,
                ],
            }))
        }
        (name, _) => Err(format!("unsupported allocation/export opcode/form {name}")),
    }
}

pub fn required_vgprs(export: MeshExport) -> u32 {
    if matches!(export.target, 9 | 13..=15) {
        return 0;
    }
    if export.target == 20 {
        return export.sources[0] as u32 + 1;
    }
    let mut required = 0u32;
    if export.compressed {
        for pair in 0..2usize {
            if export.enable & (3 << (pair * 2)) != 0 {
                required = required.max(export.sources[pair] as u32 + 1);
            }
        }
    } else {
        for component in 0..4usize {
            if export.enable & (1 << component) != 0 {
                required = required.max(export.sources[component] as u32 + 1);
            }
        }
    }
    required
}

pub fn apply_allocation(
    record: &mut [u32],
    layout: SyntheticRecordLayout,
    m0: u32,
) -> Result<GsAllocRequest, &'static str> {
    if record.len() < layout.record_dwords as usize {
        return Err("mesh allocation record is truncated");
    }
    let decoded = GsAllocRequest::unpack(m0);
    let allocation = GsAllocRequest::clamped(
        decoded.vertices.min(layout.max_vertices),
        decoded.primitives.min(layout.max_primitives),
    );
    record[0] = allocation.vertices;
    record[1] = allocation.primitives;
    Ok(allocation)
}

pub fn apply_export(
    record: &mut [u32],
    layout: SyntheticRecordLayout,
    export: MeshExport,
    vgprs: &[u32],
    vertex: u32,
    primitive: u32,
) -> Result<(), &'static str> {
    if record.len() < layout.record_dwords as usize {
        return Err("mesh export record is truncated");
    }
    let component = |index: usize| -> Result<u32, &'static str> {
        let source_index = if export.compressed { index / 2 } else { index };
        let raw = *vgprs
            .get(export.sources[source_index] as usize)
            .ok_or("mesh export VGPR is out of bounds")?;
        Ok(if export.compressed {
            if index & 1 == 0 {
                raw & 0xffff
            } else {
                raw >> 16
            }
        } else {
            raw
        })
    };
    let enabled = |index: usize| {
        if export.compressed {
            export.enable & (3 << (index / 2 * 2)) != 0
        } else {
            export.enable & (1 << index) != 0
        }
    };
    match export.target {
        9 | 13..=15 => Ok(()),
        12 => {
            if vertex >= record[0].min(layout.max_vertices) {
                return Err("mesh vertex export is outside allocation");
            }
            let mut values = [0u32; 4];
            for index in 0..4usize {
                if enabled(index) {
                    values[index] = component(index)?;
                }
            }
            let base = MESH_HEADER as u32
                + vertex
                    .checked_mul(4)
                    .ok_or("mesh vertex export address overflow")?;
            for index in 0..4usize {
                if enabled(index) {
                    *record
                        .get_mut(base as usize + index)
                        .ok_or("mesh vertex export exceeds the record")? = values[index];
                }
            }
            Ok(())
        }
        20 => {
            if primitive >= record[1].min(layout.max_primitives) {
                return Err("mesh primitive export is outside allocation");
            }
            let value = *vgprs
                .get(export.sources[0] as usize)
                .ok_or("mesh export VGPR is out of bounds")?;
            *record
                .get_mut(layout.triangle_base() as usize + primitive as usize)
                .ok_or("mesh primitive export exceeds the record")? = value;
            Ok(())
        }
        32..=63 => {
            if vertex >= record[0].min(layout.max_vertices) {
                return Err("mesh parameter export is outside allocation");
            }
            let mut values = [0u32; 4];
            for index in 0..4usize {
                if enabled(index) {
                    values[index] = component(index)?;
                }
            }
            for index in 0..4usize {
                if enabled(index) {
                    let word = layout.parameter_word(export.target, vertex, index as u32)?;
                    *record
                        .get_mut(word as usize)
                        .ok_or("mesh parameter export exceeds the record")? = values[index];
                }
            }
            Ok(())
        }
        _ => Err("unsupported mesh export target"),
    }
}

pub fn parameter_export_validation_hlsl(
    layout: SyntheticRecordLayout,
    export: MeshExport,
    vgprs: &[u32],
    vertex: u32,
) -> Result<String, &'static str> {
    if !(32..=63).contains(&export.target) {
        return Err("validation export is not a parameter target");
    }
    if vertex >= layout.max_vertices {
        return Err("validation parameter vertex is out of bounds");
    }
    let mut values = [None; 4];
    for index in 0..4usize {
        let enabled = if export.compressed {
            export.enable & (3 << (index / 2 * 2)) != 0
        } else {
            export.enable & (1 << index) != 0
        };
        if enabled {
            let source_index = if export.compressed { index / 2 } else { index };
            let raw = *vgprs
                .get(export.sources[source_index] as usize)
                .ok_or("validation parameter VGPR is out of bounds")?;
            values[index] = Some(if export.compressed {
                if index & 1 == 0 {
                    raw & 0xffff
                } else {
                    raw >> 16
                }
            } else {
                raw
            });
        }
    }
    let mut stores = String::new();
    for (component, value) in values.into_iter().enumerate() {
        if let Some(value) = value {
            let word = layout.parameter_word(export.target, vertex, component as u32)?;
            stores.push_str(&format!(
                "        res.Store((record_word + {word}u) * 4u, {value:#x}u);\n"
            ));
        }
    }
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer res : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads(128, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint record_word = gid.x * {record_dwords}u;
    for (uint word = gi; word < {record_dwords}u; word += 128u)
        res.Store((record_word + word) * 4u, 0u);
    DeviceMemoryBarrierWithGroupSync();
    if (gi == 0u) {{
        res.Store(record_word * 4u, {allocation}u);
{stores}    }}
}}
"#,
        record_dwords = layout.record_dwords,
        allocation = vertex + 1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    fn decoded(name: &str, encoding: &str, fields: &[(&str, u64)]) -> AllocationExportOp {
        let code = encode(name, encoding, fields, None).unwrap();
        lower_one(&rdna2::decode(&code, 0).unwrap()).unwrap()
    }

    fn export(fields: &[(&str, u64)]) -> MeshExport {
        match decoded("EXP", "ENC_EXP", fields) {
            AllocationExportOp::Export(export) => export,
            _ => unreachable!(),
        }
    }

    #[test]
    fn decoded_forms_are_exact_and_resource_spans_follow_enables() {
        assert_eq!(
            decoded("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)]),
            AllocationExportOp::Allocate { m0: M0_REGISTER }
        );
        for message in [0, 3, 0x19, 0xffff] {
            let code = encode("S_SENDMSG", "ENC_SOPP", &[("SIMM16", message)], None).unwrap();
            assert!(lower_one(&rdna2::decode(&code, 0).unwrap()).is_err());
        }
        let plain = export(&[
            ("TGT", 12),
            ("EN", 5),
            ("VSRC0", 2),
            ("VSRC1", 200),
            ("VSRC2", 7),
            ("VSRC3", 201),
            ("DONE", 1),
        ]);
        assert!(plain.done);
        assert_eq!(required_vgprs(plain), 8);
        let compressed = export(&[
            ("TGT", 12),
            ("EN", 4),
            ("COMPR", 1),
            ("VSRC0", 200),
            ("VSRC1", 9),
        ]);
        assert_eq!(required_vgprs(compressed), 10);
        let null = export(&[("TGT", 9), ("EN", 15), ("VSRC3", 250)]);
        assert_eq!(required_vgprs(null), 0);
        for target in [32, 63] {
            let parameter = export(&[("TGT", target), ("EN", 9), ("VSRC0", 3), ("VSRC3", 7)]);
            assert_eq!(required_vgprs(parameter), 8);
        }
        for fields in [
            vec![("TGT", 8)],
            vec![("TGT", 31)],
            vec![("TGT", 12), ("VM", 1)],
        ] {
            let code = encode("EXP", "ENC_EXP", &fields, None).unwrap();
            assert!(lower_one(&rdna2::decode(&code, 0).unwrap()).is_err());
        }
    }

    #[test]
    fn allocation_and_exports_are_bounded_and_preserve_guards() {
        let layout = SyntheticRecordLayout::new(4, 2).unwrap();
        let guard = 0xdead_beef;
        let mut storage = vec![guard; layout.record_dwords as usize + 2];
        let record = &mut storage[1..1 + layout.record_dwords as usize];
        record.fill(0);
        let allocation = apply_allocation(record, layout, u32::MAX).unwrap();
        assert_eq!((allocation.vertices, allocation.primitives), (4, 2));
        let position = export(&[
            ("TGT", 12),
            ("EN", 15),
            ("VSRC0", 0),
            ("VSRC1", 1),
            ("VSRC2", 2),
            ("VSRC3", 3),
        ]);
        apply_export(record, layout, position, &[10, 11, 12, 13], 0, 0).unwrap();
        assert_eq!(&record[MESH_HEADER..MESH_HEADER + 4], &[10, 11, 12, 13]);
        let compressed = export(&[("TGT", 12), ("EN", 1), ("COMPR", 1), ("VSRC0", 0)]);
        apply_export(record, layout, compressed, &[0x3344_1122], 1, 0).unwrap();
        let second = MESH_HEADER + 4;
        assert_eq!(&record[second..second + 2], &[0x1122, 0x3344]);
        let primitive = export(&[("TGT", 20), ("EN", 0), ("COMPR", 1), ("VSRC0", 0)]);
        apply_export(record, layout, primitive, &[0x8000_0000], 0, 1).unwrap();
        assert_eq!(record[layout.triangle_base() as usize + 1], 0x8000_0000);
        assert!(apply_export(record, layout, position, &[0; 4], 4, 0).is_err());
        assert!(apply_export(record, layout, primitive, &[0], 0, 2).is_err());
        assert!(apply_export(record, layout, position, &[], 0, 0).is_err());
        let parameter = export(&[
            ("TGT", 63),
            ("EN", 5),
            ("VSRC0", 0),
            ("VSRC1", 200),
            ("VSRC2", 1),
        ]);
        apply_export(record, layout, parameter, &[0x1111, 0x3333], 1, 0).unwrap();
        assert_eq!(
            record[layout.parameter_word(63, 1, 0).unwrap() as usize],
            0x1111
        );
        assert_eq!(
            record[layout.parameter_word(63, 1, 2).unwrap() as usize],
            0x3333
        );
        let before = record.to_vec();
        let partial = export(&[("TGT", 32), ("EN", 3), ("VSRC0", 0), ("VSRC1", 7)]);
        assert!(apply_export(record, layout, partial, &[1], 0, 0).is_err());
        assert_eq!(record, before);
        assert!(apply_export(record, layout, parameter, &[0; 2], 4, 0).is_err());
        assert_eq!(storage[0], guard);
        assert_eq!(*storage.last().unwrap(), guard);
    }

    #[test]
    fn decoded_allocation_exports_reuse_actual_u6_render_path() {
        if super::super::dxc::location().is_err() {
            return;
        }
        assert!(matches!(
            decoded("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)]),
            AllocationExportOp::Allocate { .. }
        ));
        assert!(matches!(
            decoded("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 15), ("DONE", 1)]),
            AllocationExportOp::Export(_)
        ));
        let record_layout = SyntheticRecordLayout::new(4, 2).unwrap();
        let scratch_layout = super::super::mesh_vgpr::VgprScratchLayout::new(4, 2).unwrap();
        let configs = [
            super::super::mesh_export::SyntheticGroupConfig {
                exec: [u64::MAX; 2],
                vertices: [(1u64 << 0) | (1u64 << 31), 1u64 << 63],
                primitives: [1, 0],
                invalid_primitives: [0, 0],
            },
            super::super::mesh_export::SyntheticGroupConfig {
                exec: [u64::MAX; 2],
                vertices: [u64::MAX; 2],
                primitives: [u64::MAX; 2],
                invalid_primitives: [1u64 << 1, 1u64 << 63],
            },
        ];
        let hlsl = super::super::mesh_export::export_validation_hlsl(
            scratch_layout,
            record_layout,
            configs,
        )
        .unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
        let interface = super::super::MsInterface {
            threads: super::super::mesh_dispatch::GROUP_THREADS,
            max_vertices: record_layout.max_vertices,
            max_primitives: record_layout.max_primitives,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&interface),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return float4(0.2,0.4,0.6,1.0);}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (pixels, records, _) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            2,
            record_layout.record_dwords as usize,
            &[
                scratch_layout.scratch.record_stride_bytes,
                record_layout.record_dwords * 4,
            ],
            &[],
            16,
            [0.0; 4],
            scratch_layout.scratch,
        )
        .unwrap();
        let expected: Vec<u32> = configs
            .iter()
            .flat_map(|config| {
                super::super::mesh_export::build_reference_record(record_layout, *config)
            })
            .collect();
        assert_eq!(records, expected);
        assert!(pixels.iter().any(|pixel| *pixel == [0.2, 0.4, 0.6, 1.0]));
    }

    #[test]
    fn decoded_parameter_export_matches_actual_u6_raw_record() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let layout = SyntheticRecordLayout::new(4, 2).unwrap();
        let parameter = export(&[
            ("TGT", 32),
            ("EN", 15),
            ("COMPR", 1),
            ("VSRC0", 0),
            ("VSRC1", 1),
            ("DONE", 1),
        ]);
        let vgprs = [0x2222_1111, 0x4444_3333];
        let vertex = 2;
        let mut expected = vec![0u32; layout.record_dwords as usize];
        expected[0] = vertex + 1;
        apply_export(&mut expected, layout, parameter, &vgprs, vertex, 0).unwrap();

        let hlsl = parameter_export_validation_hlsl(layout, parameter, &vgprs, vertex).unwrap();
        let compute = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
        let interface = super::super::MsInterface {
            threads: super::super::mesh_dispatch::GROUP_THREADS,
            max_vertices: layout.max_vertices,
            max_primitives: layout.max_primitives,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&interface),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main():SV_Target0{return 0.0;}",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let scratch = super::super::mesh_vgpr::VgprScratchLayout::new(1, 1)
            .unwrap()
            .scratch;
        let (_, records, _) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            layout.record_dwords as usize,
            &[scratch.record_stride_bytes, layout.record_dwords * 4],
            &[],
            1,
            [0.0; 4],
            scratch,
        )
        .unwrap();
        assert_eq!(records, expected);
    }
}
