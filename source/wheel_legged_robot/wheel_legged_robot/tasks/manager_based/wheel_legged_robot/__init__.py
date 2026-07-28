# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Wheel-Legged-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_flat_env_cfg:WheelLeggedVMCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_legged_flat_env_cfg:WheelLeggedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLeggedFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:WheelLeggedFlatTrainerCfg",
    },
)


gym.register(
    id="Wheel-Legged-Jump-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLeggedJumpFlatPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Wheel-Legged-Jump-High-Landing-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wheel_legged_jump_env_cfg:"
            "WheelLeggedHighJumpLandingFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "WheelLeggedHighJumpLandingPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Wheel-Legged-Jump-Clearance-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wheel_legged_jump_env_cfg:"
            "WheelLeggedJumpClearanceFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "WheelLeggedJumpClearancePPORunnerCfg"
        ),
    },
)


gym.register(
    id="Wheel-Legged-Jump-Moving-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wheel_legged_jump_env_cfg:"
            "WheelLeggedMovingJumpFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "WheelLeggedMovingJumpPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Wheel-Legged-Jump-Moving-Curriculum-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_jump_env_cfg:WheelLeggedJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wheel_legged_jump_env_cfg:"
            "WheelLeggedMovingJumpCurriculumFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "WheelLeggedMovingJumpCurriculumPPORunnerCfg"
        ),
    },
)
