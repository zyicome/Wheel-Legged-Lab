# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground locomotion plus externally triggered small-jump task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import (
    CurriculumTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from . import mdp
from .wheel_legged_flat_env_cfg import (
    VmcActionsCfg,
    WheelLeggedEventCfg,
    WheelLeggedFlatEnvCfg,
    WheelLeggedRewardsCfg,
    WheelLeggedRobotSceneCfg,
    WheelLeggedTerminationsCfg,
    WheelLeggedVMCEnv,
    WheelleggedObservationsCfg,
)


@configclass
class JumpStateMachineCfg:
    """Contact-driven jump phase timings and acceptance thresholds."""

    # Stage B2: the former 0.22 m reference changed phase at 0.255 m and only
    # produced a marginal hop.  The open-loop sweep showed that holding a
    # 0.18--0.20 m crouch for roughly 0.25 s is required for a visible jump.
    crouch_length: float = 0.19
    # Transition at 0.22 m is deep enough to store useful energy while still
    # being reachable under the ±10% actuator/domain randomization.
    crouch_ready_length: float = 0.22
    thrust_length: float = 0.30
    # Retract fully in flight.  This converts base motion into useful wheel
    # clearance and leaves the entire leg stroke available for impact absorption.
    landing_length: float = 0.19
    crouch_min_time: float = 0.22
    max_crouch_time: float = 0.42
    # 保持完整蹬伸约 0.20 s。过早切换到飞行收腿会让轮端在卸载前
    # 再次压回地面，表现为“机体上升但 contact 始终存在”。
    min_thrust_time: float = 0.18
    max_thrust_time: float = 0.32
    min_release_vz: float = 0.40
    # 使用测得的实际腿长而非动作目标。0.30 m 目标在高速蹬伸时
    # 会因惯性自然越过名义值；延迟到 0.31 m 再收腿可完成轮端卸载。
    min_release_length: float = 0.31
    max_airborne_wait: float = 0.15
    min_takeoff_vz: float = 0.25
    min_flight_time: float = 0.04
    landing_time: float = 0.20
    recovery_stable_time: float = 0.50
    max_recovery_time: float = 1.50
    contact_force_threshold: float = 2.0
    contact_confirm_steps: int = 2
    start_max_tilt: float = 0.30
    start_max_speed: float = 0.40
    recovery_max_tilt: float = 0.20
    recovery_max_vz: float = 0.30
    success_height_ratio: float = 0.80
    # Zero preserves the original base-height-only acceptance rule. Clearance
    # stages set this above zero so a long-legged grounded extension cannot be
    # reported as a successful obstacle-clearing jump.
    success_wheel_clearance_ratio: float = 0.0
    success_min_air_time: float = 0.12
    success_max_recovery_vx_error: float = float("inf")
    use_leg_reference_assist: bool = True
    # The Stage-B1 actor learned residuals around the former shallow reference.
    # Limit its authority during the B2 transfer so it cannot cancel the proven
    # deep-crouch trajectory; increase this again only after B2 converges.
    leg_action_residual_scale: float = 0.05

    # Optional Stage-C high-jump/landing assistance.  Defaults preserve the
    # already-trained Stage-B2 task and its checkpoint behavior.
    use_height_conditioned_crouch: bool = False
    reference_height_min: float = 0.05
    reference_height_max: float = 0.07
    high_jump_crouch_length: float = 0.19
    use_landing_assist: bool = False
    flight_retract_length: float = 0.19
    prelanding_length: float = 0.25
    prelanding_start_vz: float = 0.05
    prelanding_full_vz: float = -0.65
    landing_absorption_length: float = 0.21
    landing_compression_time: float = 0.18
    prelanding_action_residual_scale: float = 0.10
    landing_action_residual_scale: float = 0.15
    use_phase_dependent_gains: bool = False
    thrust_kp_scale: float = 1.0
    thrust_feedforward_scale: float = 1.0
    landing_kd_scale: float = 1.0
    flight_feedforward_scale: float = 1.0
    landing_feedforward_scale: float = 1.0
    # Optional split flight controller: retract through ascent/apex, then
    # independently tune the descending pre-extension. Defaults are inactive
    # and therefore leave Stage B2/C1 checkpoints unchanged.
    use_split_flight_control: bool = False
    flight_retract_kp_scale: float = 1.0
    flight_retract_kd_scale: float = 1.0
    flight_retract_feedforward_scale: float = 0.0
    prelanding_feedforward_scale: float = 1.0
    prelanding_kd_scale: float = 1.0
    # ``inf`` keeps the Stage-B2 success definition unchanged.  The high-jump
    # stage uses a finite intermediate limit while separately reporting the
    # stricter 0.8 m/s soft-landing rate.
    success_max_landing_speed: float = float("inf")
    soft_landing_speed: float = 0.80


class WheelLeggedJumpEnv(WheelLeggedVMCEnv):
    """Wheel-legged environment with a six-state, contact-driven jump machine."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.jump_phase = torch.full(
            (self.num_envs,), mdp.JUMP_PHASE_IDLE, dtype=torch.long, device=self.device
        )
        self.jump_phase_time = torch.zeros(self.num_envs, device=self.device)
        self.jump_wheel_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.jump_start_height = self._robot.data.root_pos_w[:, 2].clone()
        self.jump_apex_height = self.jump_start_height.clone()
        wheel_body_ids, wheel_body_names = self._robot.find_bodies(".*wheel.*")
        if len(wheel_body_ids) != 2:
            raise RuntimeError(
                f"Jump task expected two articulation wheel bodies, got {wheel_body_names}."
            )
        self._jump_robot_wheel_body_ids = wheel_body_ids
        wheel_height = self._robot.data.body_pos_w[:, wheel_body_ids, 2].amin(dim=1)
        self.jump_start_wheel_height = wheel_height.clone()
        self.jump_wheel_apex_height = wheel_height.clone()
        self.jump_takeoff_vz = torch.zeros(self.num_envs, device=self.device)
        self.jump_target_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_takeoff_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_vz = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_leg_length = torch.zeros(self.num_envs, device=self.device)
        self.jump_takeoff_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_landing_event = torch.zeros_like(self.jump_takeoff_event)
        self.jump_success_event = torch.zeros_like(self.jump_takeoff_event)
        self.jump_failure_event = torch.zeros_like(self.jump_takeoff_event)
        self._jump_air_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jump_ground_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_stable_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_airborne_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_airborne_ever = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._jump_attempts = torch.zeros(self.num_envs, device=self.device)
        self._jump_successes = torch.zeros(self.num_envs, device=self.device)
        self._jump_soft_landings = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_crouch = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_thrust = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_unload = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_performance = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_recovery = torch.zeros(self.num_envs, device=self.device)

        self._jump_contact_sensor = self.scene.sensors["contact_forces"]
        wheel_ids, wheel_names = self._jump_contact_sensor.find_bodies(".*wheel.*")
        if len(wheel_ids) != 2:
            raise RuntimeError(
                f"Jump task expected two wheel bodies, got {wheel_names} "
                f"from {self._jump_contact_sensor.body_names}."
            )
        self._jump_wheel_body_ids = wheel_ids

    def _enter_jump_phase(self, mask: torch.Tensor, phase: int):
        if mask.any():
            self.jump_phase[mask] = phase
            self.jump_phase_time[mask] = 0.0
            # Contact counters are phase-local. Reusing counts accumulated in
            # CROUCH was the cause of downward unloading being called takeoff.
            if phase == mdp.JUMP_PHASE_CROUCH:
                self._jump_air_steps[mask] = 0
                self._jump_ground_steps[mask] = 0
                self._jump_stable_steps[mask] = 0
            elif phase == mdp.JUMP_PHASE_THRUST:
                self._jump_air_steps[mask] = 0
            elif phase == mdp.JUMP_PHASE_FLIGHT:
                self._jump_air_steps[mask] = 0
                self._jump_ground_steps[mask] = 0
            elif phase in (mdp.JUMP_PHASE_LANDING, mdp.JUMP_PHASE_RECOVERY):
                self._jump_stable_steps[mask] = 0

    def _pre_reward_update(self):
        cfg = self.cfg.jump_state_machine
        self.jump_takeoff_event.zero_()
        self.jump_landing_event.zero_()
        self.jump_success_event.zero_()
        self.jump_failure_event.zero_()
        self.jump_phase_time += self.step_dt

        wheel_forces = self._jump_contact_sensor.data.net_forces_w[:, self._jump_wheel_body_ids]
        self.jump_wheel_contact[:] = (
            torch.linalg.vector_norm(wheel_forces, dim=-1) > cfg.contact_force_threshold
        )
        airborne = ~self.jump_wheel_contact.any(dim=1)
        any_contact = self.jump_wheel_contact.any(dim=1)
        both_contact = self.jump_wheel_contact.all(dim=1)
        self._jump_air_steps = torch.where(
            airborne, self._jump_air_steps + 1, torch.zeros_like(self._jump_air_steps)
        )
        self._jump_ground_steps = torch.where(
            any_contact, self._jump_ground_steps + 1, torch.zeros_like(self._jump_ground_steps)
        )

        root_z = self._robot.data.root_pos_w[:, 2]
        wheel_height = self._robot.data.body_pos_w[
            :, self._jump_robot_wheel_body_ids, 2
        ].amin(dim=1)
        root_vz = self._robot.data.root_lin_vel_w[:, 2]
        root_vx = self._robot.data.root_lin_vel_b[:, 0]
        speed_x = self._robot.data.root_lin_vel_b[:, 0].abs()
        tilt = torch.acos(torch.clamp(-self._robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        active = self.jump_phase != mdp.JUMP_PHASE_IDLE
        self.jump_apex_height[active] = torch.maximum(self.jump_apex_height[active], root_z[active])
        self.jump_wheel_apex_height[active] = torch.maximum(
            self.jump_wheel_apex_height[active], wheel_height[active]
        )
        confirmed_airborne = active & self._jump_airborne_ever & airborne
        self._jump_airborne_steps[confirmed_airborne] += 1

        jump_command = self.command_manager.get_command("jump_command")
        idle = self.jump_phase == mdp.JUMP_PHASE_IDLE
        start = (
            idle
            & (jump_command[:, 0] > 0.5)
            & both_contact
            & (tilt < cfg.start_max_tilt)
            & (speed_x < cfg.start_max_speed)
        )
        if start.any():
            self.jump_start_height[start] = root_z[start]
            self.jump_apex_height[start] = root_z[start]
            self.jump_start_wheel_height[start] = wheel_height[start]
            self.jump_wheel_apex_height[start] = wheel_height[start]
            self.jump_takeoff_vz[start] = 0.0
            locomotion_command = self.command_manager.get_command(
                "wheel_legged_commands"
            )
            self.jump_target_vx[start] = locomotion_command[start, 0]
            self.jump_takeoff_vx[start] = 0.0
            self.jump_landing_vz[start] = 0.0
            self.jump_landing_vx[start] = 0.0
            self._jump_airborne_steps[start] = 0
            self._jump_airborne_ever[start] = False
            self._jump_attempts[start] += 1.0
            self._enter_jump_phase(start, mdp.JUMP_PHASE_CROUCH)

        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        mean_leg_length = self.L0.mean(dim=1)
        crouch_reached = mean_leg_length <= cfg.crouch_ready_length
        begin_thrust = (
            crouch & (self.jump_phase_time >= cfg.crouch_min_time) & crouch_reached
        )
        self._enter_jump_phase(begin_thrust, mdp.JUMP_PHASE_THRUST)

        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        missed_crouch = crouch & (self.jump_phase_time >= cfg.max_crouch_time)
        if missed_crouch.any():
            self.jump_failure_event[missed_crouch] = True
            self._jump_fail_crouch[missed_crouch] += 1.0
            self._enter_jump_phase(missed_crouch, mdp.JUMP_PHASE_RECOVERY)

        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        release = (
            thrust
            & (self.jump_phase_time >= cfg.min_thrust_time)
            & (root_vz >= cfg.min_release_vz)
            & (mean_leg_length >= cfg.min_release_length)
        )
        self._enter_jump_phase(release, mdp.JUMP_PHASE_FLIGHT)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        takeoff = (
            flight
            & ~self._jump_airborne_ever
            & (self._jump_air_steps >= cfg.contact_confirm_steps)
            & (root_vz >= cfg.min_takeoff_vz)
        )
        if takeoff.any():
            self.jump_takeoff_vz[takeoff] = root_vz[takeoff]
            self.jump_takeoff_vx[takeoff] = root_vx[takeoff]
            self.jump_takeoff_event[takeoff] = True
            self._jump_airborne_ever[takeoff] = True

        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        missed_takeoff = thrust & (self.jump_phase_time >= cfg.max_thrust_time)
        if missed_takeoff.any():
            self.jump_failure_event[missed_takeoff] = True
            self._jump_fail_thrust[missed_takeoff] += 1.0
            self._enter_jump_phase(missed_takeoff, mdp.JUMP_PHASE_RECOVERY)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        failed_to_unload = (
            flight
            & ~self._jump_airborne_ever
            & (self.jump_phase_time >= cfg.max_airborne_wait)
        )
        if failed_to_unload.any():
            self.jump_failure_event[failed_to_unload] = True
            self._jump_fail_unload[failed_to_unload] += 1.0
            self._enter_jump_phase(failed_to_unload, mdp.JUMP_PHASE_RECOVERY)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        first_touchdown = (
            flight
            & cfg.use_landing_assist
            & self._jump_airborne_ever
            & (self._jump_ground_steps == 1)
            & (root_vz <= 0.0)
        )
        if first_touchdown.any():
            # Latch the physical first-contact state.  Waiting for the
            # multi-frame confirmation records post-impact compression speed
            # instead of the actual touchdown speed.
            self.jump_landing_vz[first_touchdown] = root_vz[first_touchdown]
            self.jump_landing_vx[first_touchdown] = root_vx[first_touchdown]
            self.jump_landing_leg_length[first_touchdown] = mean_leg_length[first_touchdown]

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        landing = (
            flight
            & self._jump_airborne_ever
            & (self.jump_phase_time >= cfg.min_flight_time)
            & (self._jump_ground_steps >= cfg.contact_confirm_steps)
            & (root_vz <= 0.0)
        )
        if landing.any():
            if not cfg.use_landing_assist:
                # Preserve Stage-B2 reward/metric semantics exactly.
                self.jump_landing_vz[landing] = root_vz[landing]
                self.jump_landing_vx[landing] = root_vx[landing]
                self.jump_landing_leg_length[landing] = mean_leg_length[landing]
            self.jump_landing_event[landing] = True
            landing_speed_for_metric = (
                self.jump_landing_vz[landing]
                if cfg.use_landing_assist
                else root_vz[landing]
            )
            self._jump_soft_landings[landing] += (
                landing_speed_for_metric.abs() <= cfg.soft_landing_speed
            ).float()
            self._enter_jump_phase(landing, mdp.JUMP_PHASE_LANDING)

        landing_phase = self.jump_phase == mdp.JUMP_PHASE_LANDING
        begin_recovery = landing_phase & (self.jump_phase_time >= cfg.landing_time)
        self._enter_jump_phase(begin_recovery, mdp.JUMP_PHASE_RECOVERY)

        recovery = self.jump_phase == mdp.JUMP_PHASE_RECOVERY
        stable = (
            recovery
            & both_contact
            & (tilt < cfg.recovery_max_tilt)
            & (root_vz.abs() < cfg.recovery_max_vz)
        )
        self._jump_stable_steps = torch.where(
            stable, self._jump_stable_steps + 1, torch.zeros_like(self._jump_stable_steps)
        )
        stable_required = max(1, round(cfg.recovery_stable_time / self.step_dt))
        recovered = recovery & (self._jump_stable_steps >= stable_required)
        # Base rise alone used to accept a grounded leg extension as a jump.
        # It becomes a valid height criterion when combined with confirmed
        # takeoff and a minimum continuous airborne duration.
        apex_rise = self.jump_apex_height - self.jump_start_height
        height_reached = apex_rise >= cfg.success_height_ratio * jump_command[:, 1]
        wheel_clearance = self.jump_wheel_apex_height - self.jump_start_wheel_height
        clearance_reached = (
            wheel_clearance
            >= cfg.success_wheel_clearance_ratio * jump_command[:, 1]
        )
        air_time_reached = self._jump_airborne_steps * self.step_dt >= cfg.success_min_air_time
        successful = (
            recovered
            & self._jump_airborne_ever
            & height_reached
            & clearance_reached
            & air_time_reached
            & (self.jump_landing_vz.abs() <= cfg.success_max_landing_speed)
            & (
                (root_vx - self.jump_target_vx).abs()
                <= cfg.success_max_recovery_vx_error
            )
        )
        # A failure before takeoff was already counted at its actual cause.
        # Do not count it again as a performance failure after recovery.
        recovered_but_failed = recovered & self._jump_airborne_ever & ~successful
        if successful.any():
            self.jump_success_event[successful] = True
            self._jump_successes[successful] += 1.0
        if recovered_but_failed.any():
            self.jump_failure_event[recovered_but_failed] = True
            self._jump_fail_performance[recovered_but_failed] += 1.0
        self._enter_jump_phase(recovered, mdp.JUMP_PHASE_IDLE)

        recovery = self.jump_phase == mdp.JUMP_PHASE_RECOVERY
        recovery_failed = recovery & (self.jump_phase_time >= cfg.max_recovery_time)
        if recovery_failed.any():
            self.jump_failure_event[recovery_failed] = True
            self._jump_fail_recovery[recovery_failed] += 1.0
            self._enter_jump_phase(recovery_failed, mdp.JUMP_PHASE_IDLE)

    def _log_debug_metrics(self):
        super()._log_debug_metrics()
        if not hasattr(self, "jump_phase"):
            return
        log = dict(self.extras.get("log", {}))
        for phase, name in (
            (mdp.JUMP_PHASE_IDLE, "idle"),
            (mdp.JUMP_PHASE_CROUCH, "crouch"),
            (mdp.JUMP_PHASE_THRUST, "thrust"),
            (mdp.JUMP_PHASE_FLIGHT, "flight"),
            (mdp.JUMP_PHASE_LANDING, "landing"),
            (mdp.JUMP_PHASE_RECOVERY, "recovery"),
        ):
            log[f"jump_phase_{name}"] = (self.jump_phase == phase).float().mean().item()
        command = self.command_manager.get_command("jump_command")
        attempts = self._jump_attempts.sum().clamp_min(1.0)
        enabled = command[:, 1] > 0.0
        log["jump_target_height"] = (
            command[enabled, 1].mean().item() if enabled.any() else 0.0
        )
        valid_takeoff = self._jump_airborne_ever
        valid_landing = self.jump_landing_vz < 0.0
        log["jump_takeoff_vz"] = (
            self.jump_takeoff_vz[valid_takeoff].mean().item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_target_vx"] = (
            self.jump_target_vx[valid_takeoff].abs().mean().item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_takeoff_vx_error"] = (
            (self.jump_takeoff_vx[valid_takeoff] - self.jump_target_vx[valid_takeoff])
            .abs()
            .mean()
            .item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_landing_vz"] = (
            self.jump_landing_vz[valid_landing].mean().item()
            if valid_landing.any()
            else 0.0
        )
        log["jump_landing_vx_error"] = (
            (self.jump_landing_vx[valid_landing] - self.jump_target_vx[valid_landing])
            .abs()
            .mean()
            .item()
            if valid_landing.any()
            else 0.0
        )
        log["jump_landing_leg_length"] = (
            self.jump_landing_leg_length[valid_landing].mean().item()
            if valid_landing.any()
            else 0.0
        )
        apex_rise = self.jump_apex_height - self.jump_start_height
        log["jump_apex_rise"] = (
            apex_rise[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        wheel_clearance = self.jump_wheel_apex_height - self.jump_start_wheel_height
        air_time = self._jump_airborne_steps.float() * self.step_dt
        log["jump_wheel_clearance"] = (
            wheel_clearance[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        log["jump_air_time"] = (
            air_time[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        log["jump_airborne_fraction"] = (~self.jump_wheel_contact.any(dim=1)).float().mean().item()
        log["jump_success_rate"] = (self._jump_successes.sum() / attempts).item()
        log["jump_soft_landing_rate"] = (
            self._jump_soft_landings.sum() / attempts
        ).item()
        log["jump_fail_crouch_rate"] = (self._jump_fail_crouch.sum() / attempts).item()
        log["jump_fail_thrust_rate"] = (self._jump_fail_thrust.sum() / attempts).item()
        log["jump_fail_unload_rate"] = (self._jump_fail_unload.sum() / attempts).item()
        log["jump_fail_performance_rate"] = (
            self._jump_fail_performance.sum() / attempts
        ).item()
        log["jump_fail_recovery_rate"] = (self._jump_fail_recovery.sum() / attempts).item()
        if hasattr(self, "_moving_jump_curriculum_level"):
            log["moving_jump_curriculum_level"] = float(
                self._moving_jump_curriculum_level
            )
            log["moving_jump_curriculum_success"] = (
                self._moving_jump_curriculum_last_success.item()
            )
            log["moving_jump_curriculum_soft"] = (
                self._moving_jump_curriculum_last_soft.item()
            )
            log["moving_jump_curriculum_heading"] = (
                self._moving_jump_curriculum_last_heading.item()
            )
            log["moving_jump_curriculum_passes"] = float(
                self._moving_jump_curriculum_passes
            )
        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        log["jump_crouch_l0"] = self.L0[crouch].mean().item() if crouch.any() else 0.0
        log["jump_thrust_l0"] = self.L0[thrust].mean().item() if thrust.any() else 0.0
        log["jump_thrust_l0_dot"] = (
            self.L0_dot[thrust].mean().item() if thrust.any() else 0.0
        )
        self.extras["log"] = log

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "jump_phase"):
            return
        self.jump_phase[env_ids] = mdp.JUMP_PHASE_IDLE
        self.jump_phase_time[env_ids] = 0.0
        self.jump_wheel_contact[env_ids] = False
        root_z = self._robot.data.root_pos_w[env_ids, 2]
        self.jump_start_height[env_ids] = root_z
        self.jump_apex_height[env_ids] = root_z
        wheel_height = self._robot.data.body_pos_w[env_ids][
            :, self._jump_robot_wheel_body_ids, 2
        ].amin(dim=1)
        self.jump_start_wheel_height[env_ids] = wheel_height
        self.jump_wheel_apex_height[env_ids] = wheel_height
        self.jump_takeoff_vz[env_ids] = 0.0
        self.jump_target_vx[env_ids] = 0.0
        self.jump_takeoff_vx[env_ids] = 0.0
        self.jump_landing_vz[env_ids] = 0.0
        self.jump_landing_vx[env_ids] = 0.0
        self.jump_landing_leg_length[env_ids] = 0.0
        self.jump_takeoff_event[env_ids] = False
        self.jump_landing_event[env_ids] = False
        self.jump_success_event[env_ids] = False
        self.jump_failure_event[env_ids] = False
        self._jump_air_steps[env_ids] = 0
        self._jump_ground_steps[env_ids] = 0
        self._jump_stable_steps[env_ids] = 0
        self._jump_airborne_steps[env_ids] = 0
        self._jump_airborne_ever[env_ids] = False
        self._jump_attempts[env_ids] = 0.0
        self._jump_successes[env_ids] = 0.0
        self._jump_soft_landings[env_ids] = 0.0
        self._jump_fail_crouch[env_ids] = 0.0
        self._jump_fail_thrust[env_ids] = 0.0
        self._jump_fail_unload[env_ids] = 0.0
        self._jump_fail_performance[env_ids] = 0.0
        self._jump_fail_recovery[env_ids] = 0.0


@configclass
class WheelLeggedJumpObservationsCfg:
    """Flat-task observations plus explicit jump state."""

    @configclass
    class PolicyCfg(WheelleggedObservationsCfg.PolicyCfg):
        jump_phase = ObsTerm(func=mdp.jump_phase_one_hot, clip=(0.0, 1.0))
        wheel_contacts = ObsTerm(func=mdp.jump_wheel_contacts, clip=(0.0, 1.0))
        jump_command = ObsTerm(
            func=mdp.jump_command,
            params={"command_name": "jump_command", "scale": (1.0, 10.0, 5.0)},
            clip=(-5.0, 5.0),
        )
        jump_phase_time = ObsTerm(func=mdp.jump_phase_time, clip=(0.0, 1.0))

    @configclass
    class CriticCfg(WheelleggedObservationsCfg.CriticCfg):
        jump_phase = ObsTerm(func=mdp.jump_phase_one_hot, clip=(0.0, 1.0))
        wheel_contacts = ObsTerm(func=mdp.jump_wheel_contacts, clip=(0.0, 1.0))
        jump_command = ObsTerm(
            func=mdp.jump_command,
            params={"command_name": "jump_command", "scale": (1.0, 10.0, 5.0)},
            clip=(-5.0, 5.0),
        )
        jump_phase_time = ObsTerm(func=mdp.jump_phase_time, clip=(0.0, 1.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WheelLeggedJumpCommandsCfg:
    """Low-speed locomotion command plus an independent jump trigger."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        heading_command=False,
        rel_heading_envs=0.0,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05),
            ang_vel_yaw=(-0.10, 0.10),
            height=(0.20, 0.22),
            heading=(0.0, 0.0),
        ),
    )
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.0, 4.0),
        jump_probability=0.9,
        trigger_delay_range=(0.6, 1.0),
        trigger_pulse_time=0.10,
        # Height is accepted only together with confirmed takeoff and at least
        # 0.12 s of air time, so grounded leg extension cannot earn success.
        target_height_range=(0.05, 0.07),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedJumpRewardsCfg(WheelLeggedRewardsCfg):
    """Phase-aware locomotion and small-jump objectives."""

    lin_vel_z = RewardTermCfg(func=mdp.jump_gated_lin_vel_z_l2, weight=-0.5)
    base_height = RewardTermCfg(
        func=mdp.jump_gated_base_height_reward,
        weight=2.5,
        params={"command_name": "wheel_legged_commands", "std": 0.06},
    )
    dof_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5.0e-9,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001)
    action_smooth = RewardTermCfg(func=mdp.action_smooth_l2, weight=-0.001)

    jump_crouch = RewardTermCfg(func=mdp.jump_crouch_tracking, weight=8.0, params={"std": 0.030})
    # The reference action is applied by JumpVMCAction already.  Keep only a
    # weak imitation term so it cannot dominate the physical jump objectives.
    jump_phase_action = RewardTermCfg(func=mdp.jump_leg_action_tracking, weight=1.0)
    jump_thrust_pose = RewardTermCfg(
        func=mdp.jump_thrust_length_tracking,
        weight=5.0,
        params={"std": 0.035},
    )
    jump_thrust_speed = RewardTermCfg(
        func=mdp.jump_thrust_extension_speed,
        weight=5.0,
        params={"target_speed": 1.0},
    )
    jump_takeoff = RewardTermCfg(func=mdp.jump_takeoff_velocity, weight=10.0, params={"std": 0.50})
    jump_takeoff_event = RewardTermCfg(func=mdp.jump_takeoff_event_bonus, weight=100.0)
    jump_height = RewardTermCfg(func=mdp.jump_height_tracking, weight=6.0, params={"std": 0.030})
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=4.0)
    jump_symmetry = RewardTermCfg(func=mdp.jump_leg_symmetry_reward, weight=0.5)
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=1.0,
        params={"std": 0.025},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=50.0,
        params={"std_vz": 0.50, "std_tilt": 0.20},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-25.0)
    jump_recovery = RewardTermCfg(
        func=mdp.jump_recovery_stability,
        weight=1.5,
        params={"std_vz": 0.30, "std_tilt": 0.20},
    )
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=200.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-100.0)


@configclass
class WheelLeggedJumpTerminationsCfg(WheelLeggedTerminationsCfg):
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.20},
    )
    minimum_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.05},
    )


@configclass
class WheelLeggedJumpFlatEnvCfg(WheelLeggedFlatEnvCfg):
    """Initial curriculum task: low-speed locomotion and externally triggered jumps."""

    scene: WheelLeggedRobotSceneCfg = WheelLeggedRobotSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: WheelLeggedJumpObservationsCfg = WheelLeggedJumpObservationsCfg()
    actions: VmcActionsCfg = VmcActionsCfg()
    commands: WheelLeggedJumpCommandsCfg = WheelLeggedJumpCommandsCfg()
    rewards: WheelLeggedJumpRewardsCfg = WheelLeggedJumpRewardsCfg()
    terminations: WheelLeggedJumpTerminationsCfg = WheelLeggedJumpTerminationsCfg()
    events: WheelLeggedEventCfg = WheelLeggedEventCfg()
    curriculum = None
    jump_state_machine: JumpStateMachineCfg = JumpStateMachineCfg()
    env_class = WheelLeggedJumpEnv

    def __post_init__(self):
        super().__post_init__()

        # The flat parent currently rebuilds manager configs in __post_init__,
        # so install the jump variants after the common scene/simulation setup.
        self.observations = WheelLeggedJumpObservationsCfg()
        self.actions = VmcActionsCfg()
        self.actions.vmc.class_type = mdp.JumpVMCAction
        self.commands = WheelLeggedJumpCommandsCfg()
        self.rewards = WheelLeggedJumpRewardsCfg()
        self.terminations = WheelLeggedJumpTerminationsCfg()
        self.events = WheelLeggedEventCfg()
        self.curriculum = None
        self.jump_state_machine = JumpStateMachineCfg()
        self.episode_length_s = 20.0
        # These flat-task regularizers oppose the large, intentional leg
        # excursion used for jumping. Keep only a weak branch preference.
        self.rewards.nominal_state.weight = -0.1
        self.rewards.leg_posture.weight = -0.1

        # Retain moderate domain randomization, but remove pushes during the
        # first jump curriculum so contact transitions remain learnable.
        self.events.push_robot = None

        # Preserve the proven flat-task normalization for inherited terms.
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.theta0_dot.scale = 0.05
        self.observations.policy.L0.scale = 5.0
        self.observations.policy.L0_dot.scale = 0.25
        self.observations.policy.joint_wheel_vel.scale = 0.05
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.theta0_dot.scale = 0.05
        self.observations.critic.L0.scale = 5.0
        self.observations.critic.L0_dot.scale = 0.25
        self.observations.critic.joint_wheel_vel.scale = 0.05
        self.observations.critic.joint_acc.scale = 0.0025
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.height_scan.scale = 5.0
        self.observations.critic.torques.scale = 0.05


@configclass
class WheelLeggedHighJumpLandingCommandsCfg(WheelLeggedJumpCommandsCfg):
    """Stationary 9--12 cm jumps used before the moving-jump curriculum."""

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        jump_probability=0.90,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.09, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedHighJumpLandingRewardsCfg(WheelLeggedJumpRewardsCfg):
    """High-jump reward balance with explicit touchdown and absorption priority."""

    jump_crouch = RewardTermCfg(
        func=mdp.jump_crouch_tracking, weight=7.0, params={"std": 0.025}
    )
    jump_phase_action = RewardTermCfg(func=mdp.jump_leg_action_tracking, weight=1.5)
    jump_thrust_pose = RewardTermCfg(
        func=mdp.jump_thrust_length_tracking,
        weight=5.0,
        params={"std": 0.030},
    )
    jump_thrust_speed = RewardTermCfg(
        func=mdp.jump_thrust_extension_speed,
        weight=6.0,
        params={"target_speed": 1.2},
    )
    jump_takeoff = RewardTermCfg(
        func=mdp.jump_takeoff_velocity, weight=12.0, params={"std": 0.55}
    )
    jump_takeoff_event = RewardTermCfg(func=mdp.jump_takeoff_event_bonus, weight=80.0)
    jump_height = RewardTermCfg(
        func=mdp.jump_height_tracking, weight=10.0, params={"std": 0.035}
    )
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=5.0)
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=4.0,
        params={"std": 0.025},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=100.0,
        params={"std_vz": 0.55, "std_tilt": 0.18},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-40.0)
    jump_recovery = RewardTermCfg(
        func=mdp.jump_recovery_stability,
        weight=3.0,
        params={"std_vz": 0.25, "std_tilt": 0.18},
    )
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=250.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-120.0)


@configclass
class WheelLeggedHighJumpLandingFlatEnvCfg(WheelLeggedJumpFlatEnvCfg):
    """Stage C1: 9--12 cm high jump with pre-extension and impact absorption."""

    commands: WheelLeggedHighJumpLandingCommandsCfg = WheelLeggedHighJumpLandingCommandsCfg()
    rewards: WheelLeggedHighJumpLandingRewardsCfg = WheelLeggedHighJumpLandingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedHighJumpLandingCommandsCfg()
        self.rewards = WheelLeggedHighJumpLandingRewardsCfg()

        state = self.jump_state_machine
        state.use_height_conditioned_crouch = True
        state.reference_height_min = 0.09
        state.reference_height_max = 0.12
        # Deeper commands receive a deeper crouch, while the low end starts
        # close to the proven Stage-B2 reference.
        state.crouch_length = 0.19
        state.high_jump_crouch_length = 0.18
        state.crouch_ready_length = 0.21
        state.crouch_min_time = 0.24
        state.max_crouch_time = 0.48
        state.thrust_length = 0.30
        state.min_thrust_time = 0.20
        state.max_thrust_time = 0.36
        state.min_release_vz = 0.50
        state.min_release_length = 0.31

        state.use_landing_assist = True
        state.flight_retract_length = 0.19
        # A useful pre-extension must exceed the normal ~0.27 m standing leg
        # length.  The 0.30 m reference plus a small airborne feed-forward
        # targets ~0.31 m actual length without the former 0.328 m overshoot.
        state.prelanding_length = 0.30
        # Start before the apex so a zero-feed-forward airborne leg has enough
        # time to reach the requested length without a last-moment impulse.
        state.prelanding_start_vz = 0.30
        state.prelanding_full_vz = -0.30
        state.landing_absorption_length = 0.21
        state.landing_compression_time = 0.20
        state.landing_time = 0.30
        state.prelanding_action_residual_scale = 0.10
        state.landing_action_residual_scale = 0.15

        state.use_phase_dependent_gains = True
        state.thrust_kp_scale = 1.25
        state.thrust_feedforward_scale = 1.15
        state.landing_kd_scale = 2.5
        state.flight_feedforward_scale = 0.55
        state.landing_feedforward_scale = 1.20
        state.recovery_stable_time = 0.50
        state.max_recovery_time = 2.0
        state.success_height_ratio = 0.80
        state.success_min_air_time = 0.12
        # Do not call a hard 1.1 m/s touchdown a success.  The separately
        # logged soft-landing rate retains the final <=0.8 m/s acceptance goal.
        state.success_max_landing_speed = 1.05
        state.soft_landing_speed = 0.80

        # Keep the current stage near-stationary.  Forward-distance learning is
        # intentionally deferred to the next task/checkpoint.
        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.10, 0.10)


@configclass
class WheelLeggedJumpClearanceCommandsCfg(WheelLeggedJumpCommandsCfg):
    """Stationary 8--12 cm wheel-clearance commands."""

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        jump_probability=0.90,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedJumpClearanceRewardsCfg(WheelLeggedHighJumpLandingRewardsCfg):
    """Prioritize physical wheel clearance without discarding safe landing."""

    # Base rise remains useful for takeoff, but it is no longer the dominant
    # height objective. The new term measures the lower of both wheel centers.
    jump_height = RewardTermCfg(
        func=mdp.jump_height_tracking, weight=5.0, params={"std": 0.040}
    )
    jump_wheel_clearance = RewardTermCfg(
        func=mdp.jump_wheel_clearance_tracking,
        weight=16.0,
        params={"std": 0.025},
    )
    jump_takeoff = RewardTermCfg(
        func=mdp.jump_takeoff_velocity, weight=14.0, params={"std": 0.60}
    )
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=6.0)
    # Retraction and pre-extension references are deterministic guard rails;
    # this term makes it costly for the residual actor to cancel them.
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=6.0,
        params={"std": 0.022},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=100.0,
        params={"std_vz": 0.60, "std_tilt": 0.18},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-35.0)
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=300.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-120.0)


@configclass
class WheelLeggedJumpClearanceFlatEnvCfg(WheelLeggedHighJumpLandingFlatEnvCfg):
    """Stage C2: retract in flight to raise both wheel centers by up to 12 cm."""

    commands: WheelLeggedJumpClearanceCommandsCfg = WheelLeggedJumpClearanceCommandsCfg()
    rewards: WheelLeggedJumpClearanceRewardsCfg = WheelLeggedJumpClearanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedJumpClearanceCommandsCfg()
        self.rewards = WheelLeggedJumpClearanceRewardsCfg()

        state = self.jump_state_machine
        state.reference_height_min = 0.08
        state.reference_height_max = 0.12
        state.crouch_length = 0.19
        state.high_jump_crouch_length = 0.18
        state.min_release_vz = 0.50
        # Rapidly retracting the wheels redistributes momentum inside the
        # articulation and can reduce base-link vz during the two contact
        # confirmation frames. Release itself already required 0.50 m/s, so a
        # lower post-release threshold avoids misclassifying real takeoffs as
        # unload failures.
        state.min_takeoff_vz = 0.10

        # Hold the wheels close to the body through the apex. Stage C1 began
        # pre-extension at +0.30 m/s and retained 55% weight feed-forward,
        # which effectively cancelled most of the requested retraction.
        state.flight_retract_length = 0.18
        state.prelanding_length = 0.30
        state.prelanding_start_vz = -0.08
        state.prelanding_full_vz = -0.65
        state.prelanding_action_residual_scale = 0.08
        state.landing_action_residual_scale = 0.15

        state.use_split_flight_control = True
        state.flight_retract_kp_scale = 1.8
        state.flight_retract_kd_scale = 0.50
        state.flight_retract_feedforward_scale = 0.0
        state.prelanding_feedforward_scale = 0.70
        state.prelanding_kd_scale = 1.10
        state.landing_kd_scale = 2.5
        state.landing_feedforward_scale = 1.20

        # Success now requires both useful base motion and actual wheel
        # clearance. A 0.90 ratio makes the 0.12 m command require >=0.108 m
        # during early training; the dense reward still peaks at the full
        # commanded 0.12 m.
        state.success_height_ratio = 0.75
        state.success_wheel_clearance_ratio = 0.90
        state.success_min_air_time = 0.12
        state.success_max_landing_speed = 1.10
        state.soft_landing_speed = 0.80

        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.10, 0.10)


@configclass
class WheelLeggedMovingJumpCommandsCfg(WheelLeggedJumpClearanceCommandsCfg):
    """First moving-jump level: gentle forward/backward motion and yaw."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 7.0),
        heading_command=False,
        rel_heading_envs=0.0,
        hold_command_during_jump=True,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            lin_vel_x=(-0.20, 0.20),
            ang_vel_yaw=(-0.20, 0.20),
            height=(0.20, 0.22),
            heading=(0.0, 0.0),
        ),
    )
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        # Keep enough no-jump windows to protect ordinary locomotion.
        jump_probability=0.80,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedMovingJumpRewardsCfg(WheelLeggedJumpClearanceRewardsCfg):
    """Preserve speed through takeoff, flight, touchdown and recovery."""

    track_lin_vel = RewardTermCfg(
        func=mdp.jump_moving_velocity_tracking,
        weight=4.0,
        params={"std": 0.18},
    )
    jump_flight_velocity = RewardTermCfg(
        func=mdp.jump_flight_velocity_preservation,
        weight=6.0,
        params={"std": 0.22},
    )
    jump_landing_velocity = RewardTermCfg(
        func=mdp.jump_landing_velocity_tracking,
        weight=80.0,
        params={"std": 0.20},
    )
    # Slightly reduce the sparse bonus so speed preservation cannot be traded
    # away merely by satisfying the vertical-jump acceptance condition.
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=260.0)


@configclass
class WheelLeggedMovingJumpFlatEnvCfg(WheelLeggedJumpClearanceFlatEnvCfg):
    """Stage C3-L1: retain 8--12 cm wheel clearance while moving at ±0.20 m/s."""

    commands: WheelLeggedMovingJumpCommandsCfg = WheelLeggedMovingJumpCommandsCfg()
    rewards: WheelLeggedMovingJumpRewardsCfg = WheelLeggedMovingJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedMovingJumpCommandsCfg()
        self.rewards = WheelLeggedMovingJumpRewardsCfg()

        state = self.jump_state_machine
        state.start_max_speed = 0.35
        # At the completion of RECOVERY, forward speed must again agree with
        # the velocity command latched when the jump started.
        state.success_max_recovery_vx_error = 0.15

        # Keep the proven vertical trajectory unchanged in the first moving
        # level. Higher speed and explicit target distance belong to later
        # levels, after this task passes its regression gates.
        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.20, 0.20)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.20, 0.20)


@configclass
class WheelLeggedMovingJumpCurriculumCommandsCfg(WheelLeggedMovingJumpCommandsCfg):
    """World-heading-aligned commands with a final ±1.0 m/s envelope."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 7.0),
        heading_command=True,
        heading_control_stiffness=0.8,
        rel_heading_envs=1.0,
        hold_command_during_jump=True,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            # The curriculum immediately installs the first ±0.20 level.
            lin_vel_x=(-1.0, 1.0),
            # This is the heading controller's correction clamp, not a random
            # yaw-rate command range.
            ang_vel_yaw=(-1.2, 1.2),
            height=(0.20, 0.22),
            # Keep every robot aligned with the world x-axis.
            heading=(0.0, 0.0),
        ),
    )


@configclass
class WheelLeggedMovingJumpCurriculumRewardsCfg(WheelLeggedMovingJumpRewardsCfg):
    """Make heading position and its generated yaw-rate command compatible."""

    track_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp_wl,
        weight=1.5,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_heading = RewardTermCfg(
        func=mdp.track_heading_exp,
        weight=1.5,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class WheelLeggedMovingJumpSpeedCurriculumCfg:
    """Discrete speed levels guarded by locomotion, jump, landing and yaw."""

    speed_levels = CurriculumTermCfg(
        func=mdp.moving_jump_speed_curriculum,
        params={
            "reward_term_name": "track_lin_vel",
            "heading_reward_term_name": "track_heading",
            "command_name": "wheel_legged_commands",
            "speed_levels": (0.2, 0.4, 0.6, 0.8, 1.0),
            "initial_level": 0,
            "tracking_threshold": 0.78,
            "jump_success_threshold": 0.75,
            "soft_landing_threshold": 0.75,
            "heading_threshold": 0.80,
            "min_episodes": 512,
            "min_jump_attempts": 1024,
            "consecutive_passes": 2,
        },
    )


@configclass
class WheelLeggedMovingJumpCurriculumFlatEnvCfg(WheelLeggedMovingJumpFlatEnvCfg):
    """Stage C3: one run from ±0.20 to ±1.00 m/s with heading alignment."""

    commands: WheelLeggedMovingJumpCurriculumCommandsCfg = (
        WheelLeggedMovingJumpCurriculumCommandsCfg()
    )
    rewards: WheelLeggedMovingJumpCurriculumRewardsCfg = (
        WheelLeggedMovingJumpCurriculumRewardsCfg()
    )
    curriculum: WheelLeggedMovingJumpSpeedCurriculumCfg = (
        WheelLeggedMovingJumpSpeedCurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedMovingJumpCurriculumCommandsCfg()
        self.rewards = WheelLeggedMovingJumpCurriculumRewardsCfg()
        self.curriculum = WheelLeggedMovingJumpSpeedCurriculumCfg()
        # The L1 task intentionally rejected starts above 0.35 m/s. The
        # curriculum must permit jump triggers at every configured level.
        self.jump_state_machine.start_max_speed = 1.20
