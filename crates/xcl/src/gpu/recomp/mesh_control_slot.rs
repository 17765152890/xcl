//! Validation-only, leader-lane control effects for the uniform epoch shell.

use super::mesh_dispatch::{ControlOp, MaskBinaryOp, MaskRegister};

fn mask(register: MaskRegister) -> &'static str {
    match register {
        MaskRegister::Exec => "exec_mask[wave]",
        MaskRegister::Vcc => "vcc[wave]",
    }
}

fn binary(op: MaskBinaryOp, lhs: &str, rhs: &str) -> String {
    match op {
        MaskBinaryOp::And => format!("({lhs}&{rhs})"),
        MaskBinaryOp::Or => format!("({lhs}|{rhs})"),
        MaskBinaryOp::Xor => format!("({lhs}^{rhs})"),
        MaskBinaryOp::AndNot => format!("({lhs}&~{rhs})"),
    }
}

fn branch(condition: &str, target: u32, slots: u32) -> String {
    if target < slots {
        format!("if({condition})xcl_next_pc[wave]={target}u;")
    } else {
        format!("if({condition})xcl_status[wave]=3u;")
    }
}

pub(crate) fn lower_control_slot(op: ControlOp, slots: u32) -> Result<String, String> {
    let scalar = |register: u8| format!("sgpr[wave][{}u]", register);
    Ok(match op {
        ControlOp::SetMask { dst, value } => format!(
            "{}=uint2({}u,{}u);",
            mask(dst),
            value as u32,
            (value >> 32) as u32
        ),
        ControlOp::CopyMask { dst, src } => format!("{}={};", mask(dst), mask(src)),
        ControlOp::MaskToScalar { dst, src } => format!(
            "{}={}.x;{}={}.y;",
            scalar(dst),
            mask(src),
            scalar(dst.checked_add(1).ok_or("control scalar pair overflows")?),
            mask(src)
        ),
        ControlOp::ScalarToMask { dst, src } => {
            let dst = mask(dst);
            format!(
                "{dst}=uint2({},{});",
                scalar(src),
                scalar(src.checked_add(1).ok_or("control scalar pair overflows")?)
            )
        }
        ControlOp::RestoreExec { src } => format!(
            "exec_mask[wave]=uint2({},{});",
            scalar(src),
            scalar(src.checked_add(1).ok_or("control scalar pair overflows")?)
        ),
        ControlOp::MaskBinary { dst, lhs, rhs, op } => {
            let value = binary(op, mask(lhs), mask(rhs));
            format!("{}={value};scc[wave]=any({});", mask(dst), mask(dst))
        }
        ControlOp::MaskScc { lhs, rhs } => {
            format!("scc[wave]=any({}&{});", mask(lhs), mask(rhs))
        }
        ControlOp::MaskBinaryScalar { dst, lhs, rhs, op } => {
            let hi = dst.checked_add(1).ok_or("control scalar pair overflows")?;
            let value = binary(op, mask(lhs), mask(rhs));
            format!(
                "{{uint2 xcl_value={value};{}=xcl_value.x;{}=xcl_value.y;scc[wave]=any(xcl_value);}}",
                scalar(dst), scalar(hi)
            )
        }
        ControlOp::SaveExecAnd { mask: value, dst } => {
            let hi = dst.checked_add(1).ok_or("control scalar pair overflows")?;
            format!("{}=exec_mask[wave].x;{}=exec_mask[wave].y;exec_mask[wave]&=uint2({}u,{}u);scc[wave]=any(exec_mask[wave]);", scalar(dst), scalar(hi), value as u32, (value >> 32) as u32)
        }
        ControlOp::SaveExecAndMask { mask: source, dst } => {
            let hi = dst.checked_add(1).ok_or("control scalar pair overflows")?;
            format!("{}=exec_mask[wave].x;{}=exec_mask[wave].y;exec_mask[wave]&={};scc[wave]=any(exec_mask[wave]);", scalar(dst), scalar(hi), mask(source))
        }
        ControlOp::WrExecAndNot { mask: value, dst } => {
            let hi = dst.checked_add(1).ok_or("control scalar pair overflows")?;
            format!("exec_mask[wave]=uint2({}u,{}u)&~exec_mask[wave];{}=exec_mask[wave].x;{}=exec_mask[wave].y;scc[wave]=any(exec_mask[wave]);", value as u32, (value >> 32) as u32, scalar(dst), scalar(hi))
        }
        ControlOp::WrExecAndNotMask { mask: source, dst } => {
            let hi = dst.checked_add(1).ok_or("control scalar pair overflows")?;
            format!("exec_mask[wave]={}&~exec_mask[wave];{}=exec_mask[wave].x;{}=exec_mask[wave].y;scc[wave]=any(exec_mask[wave]);", mask(source), scalar(dst), scalar(hi))
        }
        ControlOp::ScalarSet { register, value } => format!("{}={value}u;", scalar(register)),
        ControlOp::ScalarAdd { register, value } => format!("{}+={value}u;", scalar(register)),
        ControlOp::ScalarAddCarry { register, value } => format!(
            "{{uint xcl_old={0};{0}=xcl_old+{value}u;scc[wave]={0}<xcl_old;}}",
            scalar(register)
        ),
        ControlOp::ScalarCompareLess { register, value } => {
            format!("scc[wave]={}<{}u;", scalar(register), value)
        }
        ControlOp::BranchScc { set, target } => branch(
            &format!("(scc[wave]!=0u)=={}", if set { "true" } else { "false" }),
            target,
            slots,
        ),
        ControlOp::BranchMaskZero {
            mask: register,
            zero,
            target,
        } => branch(
            &format!(
                "all({}==0u)=={}",
                mask(register),
                if zero { "true" } else { "false" }
            ),
            target,
            slots,
        ),
        ControlOp::BranchScalarLess {
            register,
            limit,
            target,
        } => branch(&format!("{}<{}u", scalar(register), limit), target, slots),
        ControlOp::Jump { target } => {
            if target < slots {
                format!("xcl_next_pc[wave]={target}u;")
            } else {
                "xcl_status[wave]=3u;".into()
            }
        }
        ControlOp::Complete => "xcl_status[wave]=1u;".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branches_and_bad_targets_are_explicit_state_only_effects() {
        let branch = lower_control_slot(
            ControlOp::BranchScc {
                set: true,
                target: 2,
            },
            3,
        )
        .unwrap();
        assert!(branch.contains("xcl_next_pc[wave]=2u"));
        let bad = lower_control_slot(ControlOp::Jump { target: 9 }, 3).unwrap();
        assert_eq!(bad, "xcl_status[wave]=3u;");
        for text in [branch, bad] {
            assert!(!text.contains("Barrier"));
        }
    }
}
