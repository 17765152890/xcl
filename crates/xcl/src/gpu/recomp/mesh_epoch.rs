//! CPU-only reference model for a uniform 128-thread instruction scheduler.
//!
//! This module validates scheduler shape.  It emits no HLSL and is deliberately
//! disconnected from PSO eligibility.

use super::mesh_dispatch::{
    run_control, validate_step_cap, ControlOp, LogicalWaveState, LogicalWaveStatus,
};
use super::mesh_memory::StageCOp;
use super::mesh_program::{MeshProgram, ProgramOp};
use super::mesh_scalar::StageAOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformBarrier {
    ScalarRendezvous,
    GroupMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformSlot {
    pub index: u32,
    pub pc: u64,
    pub barrier: Option<UniformBarrier>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformEpochPlan {
    pub slots: Vec<UniformSlot>,
    ops: Vec<ProgramOp>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformEpochTrace {
    pub epoch: u32,
    /// Slots visited by the whole 128-thread group, in order.
    pub visited: Vec<u32>,
    pub barriers: Vec<(u32, UniformBarrier)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformEpochResult {
    pub waves: [LogicalWaveState; 2],
    pub epochs: Vec<UniformEpochTrace>,
}

impl UniformEpochPlan {
    pub fn new(program: &MeshProgram) -> Result<Self, String> {
        if program.instructions.is_empty() {
            return Err("uniform epoch plan requires at least one instruction".into());
        }
        let mut expected_pc = 0u64;
        let mut slots = Vec::with_capacity(program.instructions.len());
        for (index, instruction) in program.instructions.iter().enumerate() {
            if instruction.pc != expected_pc || instruction.byte_len != instruction.words.len() * 4
            {
                return Err("instruction boundaries are not contiguous".into());
            }
            expected_pc = expected_pc
                .checked_add(instruction.byte_len as u64)
                .ok_or("instruction range overflows")?;
            if let ProgramOp::Control(op) = instruction.op {
                if let Some(target) = branch_target(op) {
                    if target as usize >= program.instructions.len() {
                        return Err(format!(
                            "control target {target} is outside the static slots"
                        ));
                    }
                }
            }
            let barrier = match instruction.op {
                ProgramOp::Scalar(StageAOp::Barrier) => Some(UniformBarrier::ScalarRendezvous),
                ProgramOp::Memory(
                    StageCOp::DsSwizzle { .. } | StageCOp::DsRead { .. } | StageCOp::DsWrite { .. },
                ) => Some(UniformBarrier::GroupMemory),
                _ => None,
            };
            slots.push(UniformSlot {
                index: index as u32,
                pc: instruction.pc,
                barrier,
            });
        }
        if expected_pc as usize != program.byte_len {
            return Err("program byte length does not match its slots".into());
        }
        let ops = program
            .instructions
            .iter()
            .map(|instruction| instruction.op.clone())
            .collect();
        Ok(Self { slots, ops })
    }

    pub fn run(
        &self,
        program: &MeshProgram,
        mut waves: [LogicalWaveState; 2],
        step_cap: u32,
    ) -> Result<UniformEpochResult, String> {
        validate_step_cap(step_cap).map_err(str::to_owned)?;
        if Self::new(program)? != *self {
            return Err("plan does not match program slots".into());
        }
        let mut epochs = Vec::new();
        let mut waiting = [false; 2];
        let mut barrier_generation = [0u32; 2];
        fault_invalid_pcs(&mut waves, self.slots.len());
        while waves
            .iter()
            .any(|wave| wave.status == LogicalWaveStatus::Running)
        {
            let epoch = epochs.len() as u32;
            let mut trace = UniformEpochTrace {
                epoch,
                visited: Vec::with_capacity(self.slots.len()),
                barriers: Vec::new(),
            };
            for slot in &self.slots {
                trace.visited.push(slot.index);
                if let Some(barrier) = slot.barrier {
                    trace.barriers.push((slot.index, barrier));
                }
                for (wave_index, wave) in waves.iter_mut().enumerate() {
                    if wave.status != LogicalWaveStatus::Running || wave.pc != slot.index {
                        continue;
                    }
                    if waiting[wave_index] {
                        continue;
                    }
                    if wave.steps >= step_cap {
                        wave.status = LogicalWaveStatus::StepLimit;
                        continue;
                    }
                    match &program.instructions[slot.index as usize].op {
                        ProgramOp::Scalar(StageAOp::Barrier) => {
                            waiting[wave_index] = true;
                            wave.steps += 1;
                            continue;
                        }
                        ProgramOp::Control(op) => step_control(*op, wave),
                        _ => {
                            wave.pc = wave.pc.saturating_add(1);
                            wave.steps += 1;
                            wave.epoch = wave.steps;
                        }
                    }
                    if wave.status == LogicalWaveStatus::Running
                        && wave.pc as usize >= self.slots.len()
                    {
                        wave.status = LogicalWaveStatus::Complete;
                    }
                }
                if slot.barrier == Some(UniformBarrier::ScalarRendezvous) {
                    match waiting {
                        [true, true] if barrier_generation[0] == barrier_generation[1] => {
                            for wave in 0..2 {
                                waves[wave].pc += 1;
                                waves[wave].epoch += 1;
                                barrier_generation[wave] += 1;
                                waiting[wave] = false;
                                if waves[wave].pc as usize >= self.slots.len() {
                                    waves[wave].status = LogicalWaveStatus::Complete;
                                }
                            }
                        }
                        [true, true] => {
                            waves[0].status = LogicalWaveStatus::Fault;
                            waves[1].status = LogicalWaveStatus::Fault;
                            waiting = [false; 2];
                        }
                        [true, false] if waves[1].status != LogicalWaveStatus::Running => {
                            waves[0].status = LogicalWaveStatus::Fault;
                            waiting[0] = false;
                        }
                        [false, true] if waves[0].status != LogicalWaveStatus::Running => {
                            waves[1].status = LogicalWaveStatus::Fault;
                            waiting[1] = false;
                        }
                        _ => {}
                    }
                }
            }
            epochs.push(trace);
            fault_invalid_pcs(&mut waves, self.slots.len());
            if epochs.len() > step_cap as usize + 1 {
                return Err("uniform scheduler made no bounded progress".into());
            }
        }
        Ok(UniformEpochResult { waves, epochs })
    }
}

fn fault_invalid_pcs(waves: &mut [LogicalWaveState; 2], slots: usize) {
    for wave in waves {
        if wave.status == LogicalWaveStatus::Running && wave.pc as usize >= slots {
            wave.status = LogicalWaveStatus::Fault;
        }
    }
}

fn branch_target(op: ControlOp) -> Option<u32> {
    match op {
        ControlOp::BranchScc { target, .. }
        | ControlOp::BranchMaskZero { target, .. }
        | ControlOp::BranchScalarLess { target, .. }
        | ControlOp::Jump { target } => Some(target),
        _ => None,
    }
}

fn step_control(op: ControlOp, state: &mut LogicalWaveState) {
    let pc = state.pc as usize;
    let mut program = vec![ControlOp::Complete; pc + 1];
    program[pc] = op;
    let total_steps = state.steps;
    let total_epoch = state.epoch;
    let mut one = state.clone();
    one.steps = 0;
    one.epoch = 0;
    match run_control(&program, one, 1) {
        Ok(mut next) => {
            if next.status == LogicalWaveStatus::StepLimit {
                next.status = LogicalWaveStatus::Running;
            }
            next.steps = total_steps + 1;
            next.epoch = total_epoch + 1;
            *state = next;
        }
        Err(_) => state.status = LogicalWaveStatus::Fault,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::mesh_program::{CoverageSummary, ProgramInstruction};

    fn program(ops: Vec<ProgramOp>) -> MeshProgram {
        let instructions = ops
            .into_iter()
            .enumerate()
            .map(|(i, op)| ProgramInstruction {
                pc: (i * 4) as u64,
                byte_len: 4,
                words: vec![0],
                literal: None,
                name: "TEST",
                encoding: "TEST",
                op,
            })
            .collect::<Vec<_>>();
        MeshProgram {
            byte_len: instructions.len() * 4,
            instructions,
            coverage: CoverageSummary::default(),
        }
    }

    #[test]
    fn divergent_waves_still_visit_every_slot_and_barrier_phase() {
        let p = program(vec![
            ProgramOp::Control(ControlOp::BranchScc {
                set: true,
                target: 2,
            }),
            ProgramOp::Scalar(StageAOp::Barrier),
            ProgramOp::Memory(StageCOp::DsSwizzle {
                dst: 0,
                source: 1,
                control: 0,
            }),
        ]);
        let plan = UniformEpochPlan::new(&p).unwrap();
        let mut waves = [LogicalWaveState::default(), LogicalWaveState::default()];
        waves[0].scc = 1;
        let result = plan.run(&p, waves, 8).unwrap();
        assert!(result.epochs.iter().all(|e| e.visited == [0, 1, 2]));
        assert!(result.epochs.iter().all(|e| e.barriers
            == [
                (1, UniformBarrier::ScalarRendezvous),
                (2, UniformBarrier::GroupMemory)
            ]));
        assert_eq!(result.waves[0].status, LogicalWaveStatus::Complete);
        assert_eq!(result.waves[1].status, LogicalWaveStatus::Fault);
    }

    #[test]
    fn scalar_rendezvous_waits_repeats_and_faults_without_deadlock() {
        let p = program(vec![
            ProgramOp::Scalar(StageAOp::NoEffect),
            ProgramOp::Scalar(StageAOp::Barrier),
            ProgramOp::Scalar(StageAOp::NoEffect),
            ProgramOp::Scalar(StageAOp::Barrier),
        ]);
        let plan = UniformEpochPlan::new(&p).unwrap();
        let mut waves = [LogicalWaveState::default(), LogicalWaveState::default()];
        waves[0].pc = 1;
        waves[0].exec = [0, 0];
        let result = plan.run(&p, waves, 16).unwrap();
        assert!(result.waves.iter().all(|wave| wave.status == LogicalWaveStatus::Complete));
        assert_ne!(result.waves[0].steps, result.waves[1].steps);
        assert_eq!([result.waves[0].pc, result.waves[1].pc], [4, 4]);

        let mut orphaned = [LogicalWaveState::default(), LogicalWaveState::default()];
        orphaned[0].pc = 1;
        orphaned[1].status = LogicalWaveStatus::Fault;
        let result = plan.run(&p, orphaned, 8).unwrap();
        assert_eq!(result.waves[0].status, LogicalWaveStatus::Fault);
    }

    #[test]
    fn branch_outcomes_and_wave_state_are_independent() {
        let p = program(vec![
            ProgramOp::Control(ControlOp::BranchScc {
                set: true,
                target: 2,
            }),
            ProgramOp::Control(ControlOp::Complete),
            ProgramOp::Control(ControlOp::ScalarAdd {
                register: 0,
                value: 7,
            }),
            ProgramOp::Control(ControlOp::Complete),
        ]);
        let plan = UniformEpochPlan::new(&p).unwrap();
        let mut waves = [LogicalWaveState::default(), LogicalWaveState::default()];
        waves[0].scc = 1;
        let result = plan.run(&p, waves, 8).unwrap();
        assert_eq!(result.waves[0].scalar[0], 7);
        assert_eq!(result.waves[1].scalar[0], 0);
    }

    #[test]
    fn loop_is_capped_and_bad_targets_fail_closed() {
        let looping = program(vec![ProgramOp::Control(ControlOp::Jump { target: 0 })]);
        let result = UniformEpochPlan::new(&looping)
            .unwrap()
            .run(
                &looping,
                [LogicalWaveState::default(), LogicalWaveState::default()],
                3,
            )
            .unwrap();
        assert!(result
            .waves
            .iter()
            .all(|w| w.status == LogicalWaveStatus::StepLimit && w.steps == 3));
        let bad = program(vec![ProgramOp::Control(ControlOp::Jump { target: 9 })]);
        assert!(UniformEpochPlan::new(&bad).is_err());
    }

    #[test]
    fn invalid_wave_pc_faults_without_stalling_the_other_wave() {
        let p = program(vec![ProgramOp::Control(ControlOp::Complete)]);
        let plan = UniformEpochPlan::new(&p).unwrap();
        let mut waves = [LogicalWaveState::default(), LogicalWaveState::default()];
        waves[0].pc = 9;
        let result = plan.run(&p, waves, 4).unwrap();
        assert_eq!(result.waves[0].status, LogicalWaveStatus::Fault);
        assert_eq!(result.waves[0].steps, 0);
        assert_eq!(result.waves[1].status, LogicalWaveStatus::Complete);
        assert_eq!(result.waves[1].steps, 1);
        assert_eq!(result.epochs.len(), 1);
    }
}
