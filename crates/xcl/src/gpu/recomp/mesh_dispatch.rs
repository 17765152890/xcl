//! Foundation for dispatching logical wave64 mesh programs on fixed-wave32 hardware.
//!
//! This module deliberately does not enable real programs. It defines the external
//! scratch contract and a rendezvous shape that later instruction lowerings can use.

pub const GROUP_THREADS: u32 = 128;
pub const LOGICAL_WAVE_LANES: u32 = 64;
pub const LOGICAL_WAVES: u32 = 2;
pub const HALF_LANES: u32 = 32;
pub const SCALAR_REGISTERS: u32 = 128;
const STATE_HEADER_WORDS: u32 = 9;
const STATE_WORDS: u32 = STATE_HEADER_WORDS + SCALAR_REGISTERS;
const MAILBOX_WORDS: u32 = GROUP_THREADS;
const RESULT_WORDS: u32 = GROUP_THREADS;
const RECORD_WORDS: u32 = STATE_WORDS * LOGICAL_WAVES + MAILBOX_WORDS + RESULT_WORDS;
pub const MIN_RECORD_BYTES: u32 = RECORD_WORDS * 4;
pub const RECORD_ALIGNMENT: u32 = 256;
pub const SCRATCH_UAV_REGISTER: u32 = 6;
pub const MAX_SCRATCH_BYTES: u32 = 128 * 1024 * 1024;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogicalWaveStatus {
    #[default]
    Running = 0,
    Complete = 1,
    StepLimit = 2,
    Fault = 3,
}

impl LogicalWaveStatus {
    pub fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Running,
            1 => Self::Complete,
            2 => Self::StepLimit,
            _ => Self::Fault,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalWaveState {
    pub pc: u32,
    pub exec: [u32; 2],
    pub vcc: [u32; 2],
    pub scc: u32,
    pub epoch: u32,
    pub status: LogicalWaveStatus,
    pub steps: u32,
    pub scalar: [u32; SCALAR_REGISTERS as usize],
}

impl Default for LogicalWaveState {
    fn default() -> Self {
        Self {
            pc: 0,
            exec: [0; 2],
            vcc: [0; 2],
            scc: 0,
            epoch: 0,
            status: LogicalWaveStatus::Running,
            steps: 0,
            scalar: [0; SCALAR_REGISTERS as usize],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    pub base_bytes: u32,
    pub record_stride_bytes: u32,
    pub record_count: u32,
    pub total_bytes: u32,
}

impl ScratchLayout {
    pub fn new(
        base_bytes: u32,
        record_stride_bytes: u32,
        record_count: u32,
    ) -> Result<Self, &'static str> {
        if base_bytes % RECORD_ALIGNMENT != 0 || record_stride_bytes % RECORD_ALIGNMENT != 0 {
            return Err("dispatcher scratch is not record-aligned");
        }
        if record_stride_bytes < MIN_RECORD_BYTES {
            return Err("dispatcher scratch stride is too small");
        }
        let records_bytes = record_stride_bytes
            .checked_mul(record_count)
            .ok_or("dispatcher scratch size overflow")?;
        let total_bytes = base_bytes
            .checked_add(records_bytes)
            .ok_or("dispatcher scratch address overflow")?;
        Ok(Self {
            base_bytes,
            record_stride_bytes,
            record_count,
            total_bytes,
        })
    }

    pub fn record_base(&self, record: u32) -> Result<u32, &'static str> {
        if record >= self.record_count {
            return Err("dispatcher scratch record is out of bounds");
        }
        self.base_bytes
            .checked_add(
                self.record_stride_bytes
                    .checked_mul(record)
                    .ok_or("dispatcher scratch address overflow")?,
            )
            .ok_or("dispatcher scratch address overflow")
    }

    pub fn state_offset(&self, record: u32, logical_wave: u32) -> Result<u32, &'static str> {
        if logical_wave >= LOGICAL_WAVES {
            return Err("logical wave is out of bounds");
        }
        self.record_base(record)?
            .checked_add(logical_wave * STATE_WORDS * 4)
            .ok_or("dispatcher scratch address overflow")
    }

    pub fn mailbox_offset(&self, record: u32, thread: u32) -> Result<u32, &'static str> {
        if thread >= GROUP_THREADS {
            return Err("dispatcher thread is out of bounds");
        }
        self.record_base(record)?
            .checked_add((STATE_WORDS * LOGICAL_WAVES + thread) * 4)
            .ok_or("dispatcher scratch address overflow")
    }

    pub fn result_offset(&self, record: u32, thread: u32) -> Result<u32, &'static str> {
        if thread >= GROUP_THREADS {
            return Err("dispatcher thread is out of bounds");
        }
        self.record_base(record)?
            .checked_add((STATE_WORDS * LOGICAL_WAVES + MAILBOX_WORDS + thread) * 4)
            .ok_or("dispatcher scratch address overflow")
    }

    pub fn validate_for_groups(&self, groups: u32) -> Result<(), &'static str> {
        if self.base_bytes != 0 || self.record_count != groups {
            return Err("dispatcher scratch does not match the dispatch");
        }
        if Self::new(self.base_bytes, self.record_stride_bytes, self.record_count)? != *self {
            return Err("dispatcher scratch layout is inconsistent");
        }
        if self.total_bytes > MAX_SCRATCH_BYTES {
            return Err("dispatcher scratch exceeds the resource limit");
        }
        Ok(())
    }
}

pub fn scratch_for_groups(groups: u32) -> Result<ScratchLayout, &'static str> {
    if groups == 0 {
        return Err("dispatcher scratch requires at least one record");
    }
    let stride = MIN_RECORD_BYTES
        .checked_add(RECORD_ALIGNMENT - 1)
        .ok_or("dispatcher scratch stride overflow")?
        / RECORD_ALIGNMENT
        * RECORD_ALIGNMENT;
    let layout = ScratchLayout::new(0, stride, groups)?;
    if layout.total_bytes > MAX_SCRATCH_BYTES {
        return Err("dispatcher scratch exceeds the resource limit");
    }
    Ok(layout)
}

pub fn read_logical_state(
    memory: &[u32],
    layout: ScratchLayout,
    record: u32,
    wave: u32,
) -> Result<LogicalWaveState, &'static str> {
    let word = layout.state_offset(record, wave)? as usize / 4;
    let words = memory
        .get(word..word + STATE_WORDS as usize)
        .ok_or("dispatcher state is outside the readback")?;
    let mut scalar = [0u32; SCALAR_REGISTERS as usize];
    scalar.copy_from_slice(&words[STATE_HEADER_WORDS as usize..]);
    Ok(LogicalWaveState {
        pc: words[0],
        exec: [words[1], words[2]],
        vcc: [words[3], words[4]],
        scc: words[5],
        epoch: words[6],
        status: LogicalWaveStatus::from_raw(words[7]),
        steps: words[8],
        scalar,
    })
}

/// A small, deterministic model of the rendezvous contract. Both logical waves
/// always execute all three epochs; inactive waves only stop mutating state.
pub fn model_fixed_wave32(
    input: &[u32; GROUP_THREADS as usize],
) -> ([u32; GROUP_THREADS as usize], [LogicalWaveState; 2]) {
    let mut values = *input;
    let mut mailbox = [0u32; GROUP_THREADS as usize];
    let mut states = [LogicalWaveState {
        exec: [u32::MAX; 2],
        ..Default::default()
    }; 2];
    for epoch in 0..3u32 {
        mailbox.copy_from_slice(&values);
        for thread in 0..GROUP_THREADS as usize {
            let wave = thread / LOGICAL_WAVE_LANES as usize;
            let active_epochs = wave as u32 + 2;
            if epoch < active_epochs {
                let partner = thread ^ HALF_LANES as usize;
                values[thread] = values[thread]
                    .wrapping_add(mailbox[partner])
                    .wrapping_add(epoch);
            }
        }
        for (wave, state) in states.iter_mut().enumerate() {
            if epoch < wave as u32 + 2 {
                state.pc = if wave == 0 { 0x10 } else { 0x20 };
                state.vcc = [0xaaaa_aaaa ^ epoch, 0x5555_5555 ^ epoch];
                state.scc = wave as u32;
                state.epoch = epoch + 1;
                state.scalar[0] = values[wave * LOGICAL_WAVE_LANES as usize];
            }
        }
    }
    (values, states)
}

/// Direct logical-wave64 reference for the synthetic program used to validate
/// paired-half exchange and independent control state.
pub fn model_native_wave64(
    input: &[u32; GROUP_THREADS as usize],
) -> ([u32; GROUP_THREADS as usize], [LogicalWaveState; 2]) {
    let mut output = *input;
    let mut states = [LogicalWaveState {
        exec: [u32::MAX; 2],
        ..Default::default()
    }; 2];
    for wave in 0..LOGICAL_WAVES as usize {
        let begin = wave * LOGICAL_WAVE_LANES as usize;
        let end = begin + LOGICAL_WAVE_LANES as usize;
        for epoch in 0..wave as u32 + 2 {
            let previous = output[begin..end].to_vec();
            for lane in 0..LOGICAL_WAVE_LANES as usize {
                output[begin + lane] = output[begin + lane]
                    .wrapping_add(previous[lane ^ HALF_LANES as usize])
                    .wrapping_add(epoch);
            }
            states[wave].pc = if wave == 0 { 0x10 } else { 0x20 };
            states[wave].vcc = [0xaaaa_aaaa ^ epoch, 0x5555_5555 ^ epoch];
            states[wave].scc = wave as u32;
            states[wave].epoch = epoch + 1;
            states[wave].scalar[0] = output[begin];
        }
    }
    (output, states)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskRegister {
    Exec,
    Vcc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskBinaryOp {
    And,
    Or,
    Xor,
    AndNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlOp {
    SetMask {
        dst: MaskRegister,
        value: u64,
    },
    CopyMask {
        dst: MaskRegister,
        src: MaskRegister,
    },
    MaskToScalar {
        dst: u8,
        src: MaskRegister,
    },
    ScalarToMask {
        dst: MaskRegister,
        src: u8,
    },
    MaskBinary {
        dst: MaskRegister,
        lhs: MaskRegister,
        rhs: MaskRegister,
        op: MaskBinaryOp,
    },
    MaskScc {
        lhs: MaskRegister,
        rhs: MaskRegister,
    },
    MaskBinaryScalar {
        dst: u8,
        lhs: MaskRegister,
        rhs: MaskRegister,
        op: MaskBinaryOp,
    },
    SaveExecAnd {
        mask: u64,
        dst: u8,
    },
    SaveExecAndMask {
        mask: MaskRegister,
        dst: u8,
    },
    WrExecAndNot {
        mask: u64,
        dst: u8,
    },
    WrExecAndNotMask {
        mask: MaskRegister,
        dst: u8,
    },
    RestoreExec {
        src: u8,
    },
    ScalarSet {
        register: u8,
        value: u32,
    },
    ScalarAdd {
        register: u8,
        value: u32,
    },
    ScalarAddCarry {
        register: u8,
        value: u32,
    },
    ScalarCompareLess {
        register: u8,
        value: u32,
    },
    BranchScc {
        set: bool,
        target: u32,
    },
    BranchMaskZero {
        mask: MaskRegister,
        zero: bool,
        target: u32,
    },
    BranchScalarLess {
        register: u8,
        limit: u32,
        target: u32,
    },
    Jump {
        target: u32,
    },
    Complete,
}

fn mask(state: &LogicalWaveState, register: MaskRegister) -> u64 {
    let words = match register {
        MaskRegister::Exec => state.exec,
        MaskRegister::Vcc => state.vcc,
    };
    words[0] as u64 | (words[1] as u64) << 32
}

fn set_mask(state: &mut LogicalWaveState, register: MaskRegister, value: u64) {
    let words = [value as u32, (value >> 32) as u32];
    match register {
        MaskRegister::Exec => state.exec = words,
        MaskRegister::Vcc => state.vcc = words,
    }
}

fn scalar_pair(state: &LogicalWaveState, register: u8) -> Option<u64> {
    let register = register as usize;
    let hi = register.checked_add(1)?;
    Some(*state.scalar.get(register)? as u64 | (*state.scalar.get(hi)? as u64) << 32)
}

fn set_scalar_pair(state: &mut LogicalWaveState, register: u8, value: u64) -> bool {
    let register = register as usize;
    let Some(hi) = register.checked_add(1) else {
        return false;
    };
    if hi >= state.scalar.len() {
        return false;
    }
    state.scalar[register] = value as u32;
    state.scalar[hi] = (value >> 32) as u32;
    true
}

pub const MAX_CONTROL_STEPS: u32 = 256;

pub fn validate_step_cap(step_cap: u32) -> Result<(), &'static str> {
    if step_cap == 0 || step_cap > MAX_CONTROL_STEPS {
        return Err("logical-wave step cap is out of bounds");
    }
    Ok(())
}

fn hlsl_mask(register: MaskRegister) -> &'static str {
    match register {
        MaskRegister::Exec => "exec",
        MaskRegister::Vcc => "vcc",
    }
}

fn hlsl_binary(op: MaskBinaryOp, lhs: &str, rhs: &str) -> String {
    match op {
        MaskBinaryOp::And => format!("({lhs} & {rhs})"),
        MaskBinaryOp::Or => format!("({lhs} | {rhs})"),
        MaskBinaryOp::Xor => format!("({lhs} ^ {rhs})"),
        MaskBinaryOp::AndNot => format!("({lhs} & ~{rhs})"),
    }
}

fn hlsl_case(index: usize, op: ControlOp) -> String {
    let scalar = |register: u8| format!("scalar_base + {}u", register as u32 * 4);
    let mask = |register| hlsl_mask(register);
    let body = match op {
        ControlOp::SetMask { dst, value } => format!(
            "{} = uint2({:#x}u, {:#x}u);",
            mask(dst),
            value as u32,
            (value >> 32) as u32
        ),
        ControlOp::CopyMask { dst, src } => format!("{} = {};", mask(dst), mask(src)),
        ControlOp::MaskToScalar { dst, src } => format!(
            "scratch.Store2({}, {});",
            scalar(dst),
            mask(src)
        ),
        ControlOp::ScalarToMask { dst, src } => {
            format!("{} = scratch.Load2({});", mask(dst), scalar(src))
        }
        ControlOp::RestoreExec { src } => {
            format!("exec = scratch.Load2({});", scalar(src))
        }
        ControlOp::MaskBinary { dst, lhs, rhs, op } => {
            let value = hlsl_binary(op, mask(lhs), mask(rhs));
            format!("{} = {value}; scc = any({});", mask(dst), mask(dst))
        }
        ControlOp::MaskBinaryScalar { dst, lhs, rhs, op } => {
            let value = hlsl_binary(op, mask(lhs), mask(rhs));
            format!("uint2 result = {value}; scratch.Store2({}, result); scc = any(result);", scalar(dst))
        }
        ControlOp::MaskScc { lhs, rhs } => format!("scc = any({} & {});", mask(lhs), mask(rhs)),
        ControlOp::SaveExecAnd { mask: value, dst } => format!(
            "scratch.Store2({}, exec); exec &= uint2({:#x}u, {:#x}u); scc = any(exec);",
            scalar(dst),
            value as u32,
            (value >> 32) as u32
        ),
        ControlOp::SaveExecAndMask { mask: source, dst } => {
            format!("scratch.Store2({}, exec); exec &= {}; scc = any(exec);", scalar(dst), mask(source))
        }
        ControlOp::WrExecAndNot { mask: value, dst } => format!(
            "exec = uint2({:#x}u, {:#x}u) & ~exec; scratch.Store2({}, exec); scc = any(exec);",
            value as u32,
            (value >> 32) as u32,
            scalar(dst)
        ),
        ControlOp::WrExecAndNotMask { mask: source, dst } => {
            format!("exec = {} & ~exec; scratch.Store2({}, exec); scc = any(exec);", mask(source), scalar(dst))
        }
        ControlOp::ScalarSet { register, value } => format!("scratch.Store({}, {value}u);", scalar(register)),
        ControlOp::ScalarAdd { register, value } => format!(
            "scratch.Store({0}, scratch.Load({0}) + {value}u);",
            scalar(register)
        ),
        ControlOp::ScalarAddCarry { register, value } => format!(
            "uint old = scratch.Load({0}); uint sum = old + {value}u; scratch.Store({0}, sum); scc = sum < old;",
            scalar(register)
        ),
        ControlOp::ScalarCompareLess { register, value } => {
            format!("scc = scratch.Load({}) < {value}u;", scalar(register))
        }
        ControlOp::BranchScc { set, target } => format!(
            "if ((scc != 0u) == {}) next_pc = {target}u;",
            if set { "true" } else { "false" }
        ),
        ControlOp::BranchMaskZero { mask: register, zero, target } => format!(
            "if ((all({} == 0u)) == {}) next_pc = {target}u;",
            mask(register),
            if zero { "true" } else { "false" }
        ),
        ControlOp::BranchScalarLess { register, limit, target } => format!(
            "if (scratch.Load({}) < {limit}u) next_pc = {target}u;",
            scalar(register)
        ),
        ControlOp::Jump { target } => format!("next_pc = {target}u;"),
        ControlOp::Complete => "status = 1u;".into(),
    };
    format!("case {index}u: {{ {body} break; }}")
}

pub fn control_program_hlsl(
    program: &[ControlOp],
    initial: [LogicalWaveState; 2],
    step_cap: u32,
) -> Result<String, &'static str> {
    validate_step_cap(step_cap)?;
    if program.is_empty() {
        return Err("empty logical-wave control program");
    }
    if initial.iter().any(|state| {
        state.status != LogicalWaveStatus::Running
            || state.steps != 0
            || state.epoch != 0
            || state.pc as usize >= program.len()
    }) {
        return Err("invalid initial logical-wave state");
    }
    let cases = program
        .iter()
        .copied()
        .enumerate()
        .map(|(index, op)| hlsl_case(index, op))
        .collect::<Vec<_>>()
        .join("\n                ");
    let mut scalar_init = String::new();
    for (wave, state) in initial.iter().enumerate() {
        for (register, &value) in state
            .scalar
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
        {
            scalar_init.push_str(&format!(
                "if (wave == {wave}u) scratch.Store(scalar_base + {}u, {value}u);\n        ",
                register * 4
            ));
        }
    }
    let select = |field: fn(&LogicalWaveState) -> u32| {
        format!(
            "wave == 0u ? {}u : {}u",
            field(&initial[0]),
            field(&initial[1])
        )
    };
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads({GROUP_THREADS}, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint wave = gi / {LOGICAL_WAVE_LANES}u;
    uint lane = gi % {LOGICAL_WAVE_LANES}u;
    uint state_base = gid.x * io[0] + wave * {state_bytes}u;
    uint scalar_base = state_base + {header_bytes}u;
    scratch.Store(scalar_base + lane * 4u, 0u);
    scratch.Store(scalar_base + (lane + 64u) * 4u, 0u);
    if (lane == 0u) {{
        scratch.Store4(state_base, uint4({pc}, {exec_lo}, {exec_hi}, {vcc_lo}));
        scratch.Store4(state_base + 16u, uint4({vcc_hi}, {scc}, 0u, 0u));
        scratch.Store(state_base + 32u, 0u);
    }}
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    if (lane == 0u) {{ {scalar_init} }}
    DeviceMemoryBarrierWithGroupSync();
    [loop] for (uint epoch = 0u; epoch < {step_cap}u; ++epoch) {{
        if (lane == 0u) {{
            uint pc = scratch.Load(state_base);
            uint2 exec = scratch.Load2(state_base + 4u);
            uint2 vcc = scratch.Load2(state_base + 12u);
            uint scc = scratch.Load(state_base + 20u);
            uint status = scratch.Load(state_base + 28u);
            uint steps = scratch.Load(state_base + 32u);
            if (status == 0u) {{
                uint next_pc = pc + 1u;
                switch (pc) {{ {cases} default: status = 3u; break; }}
                steps += 1u;
                if (status == 0u) pc = next_pc;
            }}
            scratch.Store(state_base, pc);
            scratch.Store2(state_base + 4u, exec);
            scratch.Store2(state_base + 12u, vcc);
            scratch.Store(state_base + 20u, scc);
            scratch.Store(state_base + 24u, steps);
            scratch.Store(state_base + 28u, status);
            scratch.Store(state_base + 32u, steps);
        }}
        DeviceMemoryBarrierWithGroupSync();
    }}
    if (lane == 0u && scratch.Load(state_base + 28u) == 0u)
        scratch.Store(state_base + 28u, 2u);
    DeviceMemoryBarrierWithGroupSync();
}}
"#,
        state_bytes = STATE_WORDS * 4,
        header_bytes = STATE_HEADER_WORDS * 4,
        pc = select(|state| state.pc),
        exec_lo = select(|state| state.exec[0]),
        exec_hi = select(|state| state.exec[1]),
        vcc_lo = select(|state| state.vcc[0]),
        vcc_hi = select(|state| state.vcc[1]),
        scc = select(|state| state.scc),
    ))
}

pub fn run_control(
    program: &[ControlOp],
    mut state: LogicalWaveState,
    step_cap: u32,
) -> Result<LogicalWaveState, &'static str> {
    validate_step_cap(step_cap)?;
    while state.status == LogicalWaveStatus::Running {
        if state.steps >= step_cap {
            state.status = LogicalWaveStatus::StepLimit;
            break;
        }
        let Some(&instruction) = program.get(state.pc as usize) else {
            state.status = LogicalWaveStatus::Fault;
            break;
        };
        let mut next_pc = state.pc.saturating_add(1);
        match instruction {
            ControlOp::SetMask { dst, value } => set_mask(&mut state, dst, value),
            ControlOp::CopyMask { dst, src } => {
                let value = mask(&state, src);
                set_mask(&mut state, dst, value);
            }
            ControlOp::MaskToScalar { dst, src } => {
                let value = mask(&state, src);
                if !set_scalar_pair(&mut state, dst, value) {
                    state.status = LogicalWaveStatus::Fault;
                }
            }
            ControlOp::ScalarToMask { dst, src } => match scalar_pair(&state, src) {
                Some(value) => set_mask(&mut state, dst, value),
                None => state.status = LogicalWaveStatus::Fault,
            },
            ControlOp::MaskBinary { dst, lhs, rhs, op } => {
                let (lhs, rhs) = (mask(&state, lhs), mask(&state, rhs));
                let value = match op {
                    MaskBinaryOp::And => lhs & rhs,
                    MaskBinaryOp::Or => lhs | rhs,
                    MaskBinaryOp::Xor => lhs ^ rhs,
                    MaskBinaryOp::AndNot => lhs & !rhs,
                };
                set_mask(&mut state, dst, value);
                state.scc = u32::from(value != 0);
            }
            ControlOp::MaskScc { lhs, rhs } => {
                state.scc = u32::from(mask(&state, lhs) & mask(&state, rhs) != 0);
            }
            ControlOp::MaskBinaryScalar { dst, lhs, rhs, op } => {
                let (lhs, rhs) = (mask(&state, lhs), mask(&state, rhs));
                let value = match op {
                    MaskBinaryOp::And => lhs & rhs,
                    MaskBinaryOp::Or => lhs | rhs,
                    MaskBinaryOp::Xor => lhs ^ rhs,
                    MaskBinaryOp::AndNot => lhs & !rhs,
                };
                if !set_scalar_pair(&mut state, dst, value) {
                    state.status = LogicalWaveStatus::Fault;
                }
                state.scc = u32::from(value != 0);
            }
            ControlOp::SaveExecAnd { mask: source, dst } => {
                let old = mask(&state, MaskRegister::Exec);
                let new = old & source;
                if !set_scalar_pair(&mut state, dst, old) {
                    state.status = LogicalWaveStatus::Fault;
                }
                set_mask(&mut state, MaskRegister::Exec, new);
                state.scc = u32::from(new != 0);
            }
            ControlOp::SaveExecAndMask { mask: source, dst } => {
                let old = mask(&state, MaskRegister::Exec);
                let new = old & mask(&state, source);
                if !set_scalar_pair(&mut state, dst, old) {
                    state.status = LogicalWaveStatus::Fault;
                }
                set_mask(&mut state, MaskRegister::Exec, new);
                state.scc = u32::from(new != 0);
            }
            ControlOp::WrExecAndNot { mask: source, dst } => {
                let new = source & !mask(&state, MaskRegister::Exec);
                if !set_scalar_pair(&mut state, dst, new) {
                    state.status = LogicalWaveStatus::Fault;
                }
                set_mask(&mut state, MaskRegister::Exec, new);
                state.scc = u32::from(new != 0);
            }
            ControlOp::WrExecAndNotMask { mask: source, dst } => {
                let new = mask(&state, source) & !mask(&state, MaskRegister::Exec);
                if !set_scalar_pair(&mut state, dst, new) {
                    state.status = LogicalWaveStatus::Fault;
                }
                set_mask(&mut state, MaskRegister::Exec, new);
                state.scc = u32::from(new != 0);
            }
            ControlOp::RestoreExec { src } => match scalar_pair(&state, src) {
                Some(value) => set_mask(&mut state, MaskRegister::Exec, value),
                None => state.status = LogicalWaveStatus::Fault,
            },
            ControlOp::ScalarSet { register, value } => {
                match state.scalar.get_mut(register as usize) {
                    Some(scalar) => *scalar = value,
                    None => state.status = LogicalWaveStatus::Fault,
                }
            }
            ControlOp::ScalarAdd { register, value } => {
                match state.scalar.get_mut(register as usize) {
                    Some(scalar) => *scalar = scalar.wrapping_add(value),
                    None => state.status = LogicalWaveStatus::Fault,
                }
            }
            ControlOp::ScalarAddCarry { register, value } => {
                match state.scalar.get_mut(register as usize) {
                    Some(scalar) => {
                        let (sum, carry) = scalar.overflowing_add(value);
                        *scalar = sum;
                        state.scc = u32::from(carry);
                    }
                    None => state.status = LogicalWaveStatus::Fault,
                }
            }
            ControlOp::ScalarCompareLess { register, value } => {
                match state.scalar.get(register as usize) {
                    Some(&scalar) => state.scc = u32::from(scalar < value),
                    None => state.status = LogicalWaveStatus::Fault,
                }
            }
            ControlOp::BranchScc { set, target } => {
                if (state.scc != 0) == set {
                    next_pc = target;
                }
            }
            ControlOp::BranchMaskZero {
                mask: register,
                zero,
                target,
            } => {
                if (mask(&state, register) == 0) == zero {
                    next_pc = target;
                }
            }
            ControlOp::BranchScalarLess {
                register,
                limit,
                target,
            } => match state.scalar.get(register as usize) {
                Some(&scalar) if scalar < limit => next_pc = target,
                Some(_) => {}
                None => state.status = LogicalWaveStatus::Fault,
            },
            ControlOp::Jump { target } => next_pc = target,
            ControlOp::Complete => state.status = LogicalWaveStatus::Complete,
        }
        state.steps += 1;
        state.epoch = state.steps;
        if state.status == LogicalWaveStatus::Running {
            state.pc = next_pc;
        }
    }
    Ok(state)
}

/// Standalone shader proving the dispatcher contract is expressible in HLSL.
/// Scratch is an external UAV and therefore consumes none of `xcl_lds`.
pub fn validation_hlsl() -> String {
    format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads({GROUP_THREADS}, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint record_base = gid.x * io[0];
    uint wave = gi / {LOGICAL_WAVE_LANES}u;
    uint lane = gi % {LOGICAL_WAVE_LANES}u;
    uint state_base = record_base + wave * {state_bytes}u;
    uint mailbox_base = record_base + {mailbox_base}u;
    uint result_base = record_base + {result_base}u;
    uint value = gi + 1u;
    if (lane == 0u) {{
        scratch.Store4(state_base, uint4(0u, 0xffffffffu, 0xffffffffu, 0u));
        scratch.Store4(state_base + 16u, uint4(0u, 0u, 0u, 0u));
        scratch.Store(state_base + 32u, 0u);
    }}
    scratch.Store(state_base + {state_header_bytes}u + lane * 4u, 0u);
    scratch.Store(state_base + {state_header_bytes}u + (lane + 64u) * 4u, 0u);
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    [unroll] for (uint epoch = 0u; epoch < 3u; ++epoch) {{
        scratch.Store(mailbox_base + gi * 4u, value);
        DeviceMemoryBarrierWithGroupSync();
        uint paired = scratch.Load(mailbox_base + (wave * {LOGICAL_WAVE_LANES}u + (lane ^ {HALF_LANES}u)) * 4u);
        if (epoch < wave + 2u) value += paired + epoch;
        if (lane == 0u && epoch < wave + 2u) {{
            scratch.Store(state_base, wave == 0u ? 0x10u : 0x20u);
            scratch.Store(state_base + 12u, 0xaaaaaaaau ^ epoch);
            scratch.Store(state_base + 16u, 0x55555555u ^ epoch);
            scratch.Store(state_base + 20u, wave);
            scratch.Store(state_base + 24u, epoch + 1u);
            scratch.Store(state_base + {state_header_bytes}u, value);
        }}
        DeviceMemoryBarrierWithGroupSync();
    }}
    scratch.Store(result_base + gi * 4u, value);
}}
"#,
        state_bytes = STATE_WORDS * 4,
        state_header_bytes = STATE_HEADER_WORDS * 4,
        mailbox_base = STATE_WORDS * LOGICAL_WAVES * 4,
        result_base = (STATE_WORDS * LOGICAL_WAVES + MAILBOX_WORDS) * 4
    )
}

/// Standalone bounded control-flow interpreter used to validate the first
/// logical-wave lowering classes through the real runner. It remains synthetic
/// and is not selected by console program translation.
pub fn control_validation_hlsl(step_cap: u32, infinite_loop: bool) -> Result<String, &'static str> {
    validate_step_cap(step_cap)?;
    let execute = if infinite_loop {
        "pc = 0u;"
    } else {
        r#"switch (pc) {
            case 0u: scalar0 = exec.x; scalar1 = exec.y; exec &= uint2(0xffu, 0u); scc = any(exec); pc = 1u; break;
            case 1u: scc = any(exec & vcc); pc = 2u; break;
            case 2u: pc = scc != 0u ? 5u : 3u; break;
            case 3u: scalar2 += 1u; pc = 4u; break;
            case 4u: pc = 6u; break;
            case 5u: scalar2 += 2u; pc = 6u; break;
            case 6u: exec = uint2(scalar0, scalar1); pc = 7u; break;
            case 7u: scalar3 += 1u; pc = 8u; break;
            case 8u: pc = scalar3 < 3u ? 7u : 9u; break;
            case 9u: status = 1u; break;
            default: status = 3u; break;
        }"#
    };
    Ok(format!(
        r#"RWStructuredBuffer<uint> io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer scratch : register(u6);
[numthreads({GROUP_THREADS}, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {{
    uint wave = gi / {LOGICAL_WAVE_LANES}u;
    uint lane = gi % {LOGICAL_WAVE_LANES}u;
    uint state_base = gid.x * io[0] + wave * {state_bytes}u;
    uint scalar_base = state_base + {header_bytes}u;
    scratch.Store(scalar_base + lane * 4u, 0u);
    scratch.Store(scalar_base + (lane + 64u) * 4u, 0u);
    if (lane == 0u) {{
        scratch.Store4(state_base, uint4(0u, 0xffffffffu, 0xffffffffu, wave == 0u ? 0xfu : 0u));
        scratch.Store4(state_base + 16u, uint4(0u, 0u, 0u, 0u));
        scratch.Store(state_base + 32u, 0u);
    }}
    if (gi == 0u) records.Store4(gid.x * io[1], uint4(0u, 0u, 0u, 0u));
    DeviceMemoryBarrierWithGroupSync();
    if (lane == 0u) scratch.Store(scalar_base + 12u, wave == 0u ? 1u : 0u);
    DeviceMemoryBarrierWithGroupSync();
    [loop] for (uint epoch = 0u; epoch < {step_cap}u; ++epoch) {{
        if (lane == 0u) {{
            uint pc = scratch.Load(state_base);
            uint2 exec = scratch.Load2(state_base + 4u);
            uint2 vcc = scratch.Load2(state_base + 12u);
            uint scc = scratch.Load(state_base + 20u);
            uint status = scratch.Load(state_base + 28u);
            uint steps = scratch.Load(state_base + 32u);
            uint scalar0 = scratch.Load(scalar_base);
            uint scalar1 = scratch.Load(scalar_base + 4u);
            uint scalar2 = scratch.Load(scalar_base + 8u);
            uint scalar3 = scratch.Load(scalar_base + 12u);
            if (status == 0u) {{ {execute} steps += 1u; }}
            scratch.Store(state_base, pc);
            scratch.Store2(state_base + 4u, exec);
            scratch.Store2(state_base + 12u, vcc);
            scratch.Store(state_base + 20u, scc);
            scratch.Store(state_base + 24u, steps);
            scratch.Store(state_base + 28u, status);
            scratch.Store(state_base + 32u, steps);
            scratch.Store(scalar_base, scalar0);
            scratch.Store(scalar_base + 4u, scalar1);
            scratch.Store(scalar_base + 8u, scalar2);
            scratch.Store(scalar_base + 12u, scalar3);
        }}
        DeviceMemoryBarrierWithGroupSync();
    }}
    if (lane == 0u && scratch.Load(state_base + 28u) == 0u)
        scratch.Store(state_base + 28u, 2u);
    DeviceMemoryBarrierWithGroupSync();
}}
"#,
        state_bytes = STATE_WORDS * 4,
        header_bytes = STATE_HEADER_WORDS * 4,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divergent_program() -> Vec<ControlOp> {
        vec![
            ControlOp::SaveExecAnd { mask: 0xff, dst: 0 },
            ControlOp::MaskScc {
                lhs: MaskRegister::Exec,
                rhs: MaskRegister::Vcc,
            },
            ControlOp::BranchScc {
                set: true,
                target: 5,
            },
            ControlOp::ScalarAdd {
                register: 2,
                value: 1,
            },
            ControlOp::Jump { target: 6 },
            ControlOp::ScalarAdd {
                register: 2,
                value: 2,
            },
            ControlOp::RestoreExec { src: 0 },
            ControlOp::ScalarAdd {
                register: 3,
                value: 1,
            },
            ControlOp::BranchScalarLess {
                register: 3,
                limit: 3,
                target: 7,
            },
            ControlOp::Complete,
        ]
    }

    fn decoded_control_program() -> Vec<u8> {
        let one = |name, encoding, fields: &[(&str, u64)]| {
            crate::gpu::rdna2::encode(name, encoding, fields, None).unwrap()
        };
        let mut code = one(
            "S_AND_SAVEEXEC_B64",
            "ENC_SOP1",
            &[("SDST", 0), ("SSRC0", 106)],
        );
        code.extend(one(
            "S_AND_B64",
            "ENC_SOP2",
            &[("SDST", 2), ("SSRC0", 126), ("SSRC1", 106)],
        ));
        code.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)]));
        code.extend(one(
            "S_ADD_U32",
            "ENC_SOP2",
            &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)],
        ));
        code.extend(one("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)]));
        code.extend(one(
            "S_ADD_U32",
            "ENC_SOP2",
            &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 130)],
        ));
        code.extend(one("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]));
        code.extend(one(
            "S_ADD_U32",
            "ENC_SOP2",
            &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 129)],
        ));
        code.extend(one(
            "S_CMP_LT_U32",
            "ENC_SOPC",
            &[("SSRC0", 5), ("SSRC1", 131)],
        ));
        code.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffd)]));
        code.extend(one("S_ENDPGM", "ENC_SOPP", &[]));
        code
    }

    fn initial_control_state(wave: usize) -> LogicalWaveState {
        let mut state = LogicalWaveState {
            exec: [u32::MAX; 2],
            vcc: if wave == 0 { [0xf, 0] } else { [0, 0] },
            ..Default::default()
        };
        state.scalar[3] = u32::from(wave == 0);
        state
    }

    fn read_state(memory: &[u32], layout: ScratchLayout, wave: usize) -> LogicalWaveState {
        read_logical_state(memory, layout, 0, wave as u32).unwrap()
    }

    fn run_control_gpu(hlsl: &str) -> Option<(ScratchLayout, Vec<u32>)> {
        if super::super::dxc::location().is_err() {
            return None;
        }
        let compute = super::super::dxc::compile(hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let layout = scratch_for_groups(1).unwrap();
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main() : SV_Target0 { return 0.0.xxxx; }",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout,
        )
        .unwrap();
        Some((layout, memory))
    }

    #[test]
    fn layout_is_external_aligned_and_bounded() {
        let stride = MIN_RECORD_BYTES.next_multiple_of(RECORD_ALIGNMENT);
        let layout = ScratchLayout::new(512, stride, 3).unwrap();
        assert_eq!(layout.total_bytes, 512 + stride * 3);
        assert_eq!(
            layout.state_offset(2, 1).unwrap(),
            512 + stride * 2 + STATE_WORDS * 4
        );
        assert!(layout.mailbox_offset(0, GROUP_THREADS).is_err());
        assert!(layout.result_offset(3, 0).is_err());
        assert!(ScratchLayout::new(1, stride, 1).is_err());
        assert!(ScratchLayout::new(0, MIN_RECORD_BYTES - 4, 1).is_err());
        assert!(ScratchLayout::new(0, u32::MAX - 255, 2).is_err());
        assert!(scratch_for_groups(0).is_err());
        assert!(scratch_for_groups(MAX_SCRATCH_BYTES / RECORD_ALIGNMENT + 1).is_err());
        let mut inconsistent = scratch_for_groups(2).unwrap();
        inconsistent.total_bytes -= RECORD_ALIGNMENT;
        assert!(inconsistent.validate_for_groups(2).is_err());
        assert!(scratch_for_groups(2)
            .unwrap()
            .validate_for_groups(1)
            .is_err());
    }

    #[test]
    fn paired_halves_match_independent_native_wave64_references() {
        let input = std::array::from_fn(|lane| lane as u32 * 17 + 3);
        let fixed = model_fixed_wave32(&input);
        let native = model_native_wave64(&input);
        assert_eq!(fixed, native);
        assert_eq!([fixed.1[0].epoch, fixed.1[1].epoch], [2, 3]);
        assert_eq!([fixed.1[0].pc, fixed.1[1].pc], [0x10, 0x20]);
        assert_ne!(&fixed.0[..64], &fixed.0[64..]);
    }

    #[test]
    fn control_interpreter_handles_masks_branches_restore_and_backedges() {
        let program = divergent_program();
        let waves = [
            run_control(&program, initial_control_state(0), 24).unwrap(),
            run_control(&program, initial_control_state(1), 24).unwrap(),
        ];
        assert!(waves
            .iter()
            .all(|state| state.status == LogicalWaveStatus::Complete));
        assert_eq!([waves[0].scalar[2], waves[1].scalar[2]], [2, 1]);
        assert_eq!([waves[0].scalar[3], waves[1].scalar[3]], [3, 3]);
        assert_eq!([waves[0].scc, waves[1].scc], [1, 0]);
        assert_eq!([waves[0].steps, waves[1].steps], [10, 13]);
        assert!(waves.iter().all(|state| state.exec == [u32::MAX; 2]));
    }

    #[test]
    fn control_interpreter_models_wrexec_and_mask_branches() {
        let program = [
            ControlOp::SetMask {
                dst: MaskRegister::Exec,
                value: 0xf0,
            },
            ControlOp::WrExecAndNot { mask: 0xff, dst: 4 },
            ControlOp::BranchMaskZero {
                mask: MaskRegister::Exec,
                zero: false,
                target: 4,
            },
            ControlOp::ScalarAdd {
                register: 6,
                value: 99,
            },
            ControlOp::Complete,
        ];
        let state = run_control(&program, LogicalWaveState::default(), 8).unwrap();
        assert_eq!(state.status, LogicalWaveStatus::Complete);
        assert_eq!(state.exec, [0x0f, 0]);
        assert_eq!([state.scalar[4], state.scalar[5]], [0x0f, 0]);
        assert_eq!(state.scalar[6], 0);
        assert_eq!(state.scc, 1);

        let masks = [
            ControlOp::SetMask {
                dst: MaskRegister::Exec,
                value: 0xf0,
            },
            ControlOp::SetMask {
                dst: MaskRegister::Vcc,
                value: 0xaa,
            },
            ControlOp::MaskBinary {
                dst: MaskRegister::Vcc,
                lhs: MaskRegister::Vcc,
                rhs: MaskRegister::Exec,
                op: MaskBinaryOp::And,
            },
            ControlOp::Complete,
        ];
        let state = run_control(&masks, LogicalWaveState::default(), 8).unwrap();
        assert_eq!(state.vcc, [0xa0, 0]);
        assert_eq!(state.scc, 1);
    }

    #[test]
    fn control_interpreter_faults_bounded_infinite_loop() {
        let state = run_control(
            &[ControlOp::Jump { target: 0 }],
            LogicalWaveState::default(),
            7,
        )
        .unwrap();
        assert_eq!(state.status, LogicalWaveStatus::StepLimit);
        assert_eq!((state.pc, state.steps, state.epoch), (0, 7, 7));
        assert!(control_validation_hlsl(0, false).is_err());
        assert!(control_validation_hlsl(MAX_CONTROL_STEPS + 1, false).is_err());
        assert!(run_control(&[], LogicalWaveState::default(), 0).is_err());
        assert!(run_control(&[], LogicalWaveState::default(), MAX_CONTROL_STEPS + 1).is_err());
    }

    #[test]
    fn gpu_control_interpreter_matches_native_wave64_reference() {
        let hlsl = control_validation_hlsl(24, false).unwrap();
        let Some((layout, memory)) = run_control_gpu(&hlsl) else {
            return;
        };
        let program = divergent_program();
        for wave in 0..2 {
            let expected = run_control(&program, initial_control_state(wave), 24).unwrap();
            assert_eq!(read_state(&memory, layout, wave), expected);
        }
    }

    #[test]
    fn gpu_control_interpreter_faults_bounded_infinite_loop() {
        let hlsl = control_validation_hlsl(5, true).unwrap();
        let Some((layout, memory)) = run_control_gpu(&hlsl) else {
            return;
        };
        for wave in 0..2 {
            let state = read_state(&memory, layout, wave);
            assert_eq!(state.status, LogicalWaveStatus::StepLimit);
            assert_eq!((state.pc, state.steps, state.epoch), (0, 5, 5));
        }
    }

    #[test]
    fn gpu_lowered_rdna_control_matches_cpu_reference() {
        let program = super::super::mesh_control::lower(&decoded_control_program()).unwrap();
        let mut initial = [initial_control_state(0), initial_control_state(1)];
        initial[0].scalar[5] = 1;
        initial[1].scalar[5] = 0;
        let hlsl = control_program_hlsl(&program, initial, 24).unwrap();
        let Some((layout, memory)) = run_control_gpu(&hlsl) else {
            return;
        };
        for wave in 0..2 {
            let expected = run_control(&program, initial[wave], 24).unwrap();
            assert_eq!(read_state(&memory, layout, wave), expected);
        }
        let waves = [
            read_state(&memory, layout, 0),
            read_state(&memory, layout, 1),
        ];
        assert_eq!([waves[0].scalar[4], waves[1].scalar[4]], [2, 1]);
        assert_eq!([waves[0].steps, waves[1].steps], [12, 16]);
        assert!(waves
            .iter()
            .all(|state| state.status == LogicalWaveStatus::Complete));
    }

    #[test]
    fn validation_shader_has_only_unconditional_rendezvous() {
        let hlsl = validation_hlsl();
        assert_eq!(
            hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
            3
        );
        assert!(!hlsl.contains("groupshared") && !hlsl.contains("xcl_lds"));
        assert!(!hlsl.contains("if (epoch < wave + 2u) DeviceMemoryBarrier"));
        if super::super::dxc::location().is_ok() {
            super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        }
    }

    #[test]
    fn native_wave64_translation_does_not_include_dispatcher_foundation() {
        let code =
            crate::gpu::rdna2::encode("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 128)], None)
                .unwrap();
        let translated = super::super::translate(
            &code,
            &super::super::Options {
                mode: super::super::Mode::Mesh(super::super::MsInterface {
                    threads: GROUP_THREADS,
                    max_vertices: 1,
                    max_primitives: 1,
                    ..Default::default()
                }),
                wave64: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!translated.hlsl.contains("record_base"));
        assert!(!translated.hlsl.contains("DeviceMemoryBarrierWithGroupSync"));
    }

    #[test]
    fn gpu_dispatch_matches_native_wave64_reference() {
        if let Err(reason) = super::super::runner::adapter() {
            println!("GPU unavailable – skipping ({reason})");
            super::super::runner::skip_test("no GPU");
            return;
        }
        if super::super::dxc::location().is_err() {
            super::super::runner::skip_test("no DXC");
            return;
        }
        let hlsl = validation_hlsl();
        let dxil = super::super::dxc::compile(&hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|error| panic!("{error}\n{hlsl}"));
        let layout = scratch_for_groups(1).unwrap();
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh_hlsl = super::super::mesh_passthrough(&record);
        let mesh = super::super::dxc::compile(&mesh_hlsl, "main", "ms_6_6", &[]).unwrap();
        let pixel_hlsl = "float4 main() : SV_Target0 { return 0.0.xxxx; }";
        let pixel = super::super::dxc::compile(pixel_hlsl, "main", "ps_6_6", &[]).unwrap();
        // The real two-pass runner owns, binds, transitions and reads back u6;
        // its ten-second fence timeout turns divergent rendezvous into a failure.
        let (_, _, memory) = super::super::runner::mesh_draw_with_dispatch_scratch(
            &dxil,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[
                layout.record_stride_bytes,
                record.record_dwords() as u32 * 4,
            ],
            &[],
            1,
            [0.0; 4],
            layout,
        )
        .unwrap();
        let input = std::array::from_fn(|lane| lane as u32 + 1);
        let (expected_values, expected_states) = model_native_wave64(&input);
        let actual_values: [u32; GROUP_THREADS as usize] = std::array::from_fn(|thread| {
            memory[layout.result_offset(0, thread as u32).unwrap() as usize / 4]
        });
        let actual_states: [LogicalWaveState; 2] =
            std::array::from_fn(|wave| read_state(&memory, layout, wave));
        assert_eq!(actual_values, expected_values);
        assert_eq!(actual_states, expected_states);
    }

    #[test]
    fn gpu_native_mesh_path_needs_no_dispatcher_binding() {
        if super::super::dxc::location().is_err() {
            return;
        }
        let compute_hlsl = r#"RWByteAddressBuffer records : register(u1);
[numthreads(128, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    if (gi == 0u) records.Store4(0u, uint4(0u, 0u, 0u, 0u));
}"#;
        let compute = super::super::dxc::compile(compute_hlsl, "main", "cs_6_6", &[]).unwrap();
        let record = super::super::MsInterface {
            threads: GROUP_THREADS,
            max_vertices: 1,
            max_primitives: 1,
            ..Default::default()
        };
        let mesh = super::super::dxc::compile(
            &super::super::mesh_passthrough(&record),
            "main",
            "ms_6_6",
            &[],
        )
        .unwrap();
        let pixel = super::super::dxc::compile(
            "float4 main() : SV_Target0 { return 0.0.xxxx; }",
            "main",
            "ps_6_6",
            &[],
        )
        .unwrap();
        super::super::runner::mesh_draw(
            &compute,
            &mesh,
            &pixel,
            1,
            record.record_dwords(),
            &[],
            &[],
            1,
            [0.0; 4],
        )
        .unwrap();
    }
}
