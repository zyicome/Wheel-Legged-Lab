# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Control vx/wz/height from the keyboard; J triggers jump tasks.",
)
parser.add_argument("--keyboard_height", type=float, default=0.21, help="Initial body height in keyboard mode.")
parser.add_argument(
    "--keyboard_height_step", type=float, default=0.01, help="Body-height increment for each R/F key press."
)
parser.add_argument(
    "--command_range",
    type=float,
    default=0.1,
    help="Wheel-legged vx limit. Use the vx_lim reached by the loaded checkpoint.",
)
parser.add_argument(
    "--yaw_command_range",
    type=float,
    default=0.5,
    help="Wheel-legged wz limit. Use the wz_lim reached by the loaded checkpoint.",
)
parser.add_argument(
    "--jump_height",
    type=float,
    default=None,
    help="Fix jump target height in meters (for example 0.12 for clearance evaluation).",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=100,
    help="Print compact environment diagnostics every N play steps; <=0 disables it.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_legged_robot.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # Use a deterministic, evaluation-friendly environment configuration.
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
        env_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    if hasattr(env_cfg.commands, "wheel_legged_commands"):
        command_cfg = env_cfg.commands.wheel_legged_commands
        command_cfg.ranges.lin_vel_x = (-args_cli.command_range, args_cli.command_range)
        command_cfg.ranges.ang_vel_yaw = (
            -args_cli.yaw_command_range,
            args_cli.yaw_command_range,
        )
        print(
            "[INFO] Wheel-legged play command ranges: "
            f"vx={command_cfg.ranges.lin_vel_x} m/s, "
            f"wz={command_cfg.ranges.ang_vel_yaw} rad/s"
        )
    if args_cli.jump_height is not None and hasattr(env_cfg.commands, "jump_command"):
        jump_height = max(0.0, args_cli.jump_height)
        env_cfg.commands.jump_command.target_height_range = (jump_height, jump_height)
        print(f"[INFO] Fixed jump target height: {jump_height:.3f} m")
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "command_levels"):
        env_cfg.curriculum.command_levels = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "speed_levels"):
        # Evaluation uses the explicit --command_range selected by the user;
        # do not let a freshly initialized curriculum force it back to 0.2.
        env_cfg.curriculum.speed_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        command_cfg = env_cfg.commands.wheel_legged_commands
        command_cfg.heading_command = False
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.debug_vis = False
        if hasattr(env_cfg.commands, "jump_command"):
            # The keyboard controller supplies the trigger; suppress random
            # autonomous jumps before CommandManager is initialized.
            env_cfg.commands.jump_command.jump_probability = 0.0
            env_cfg.commands.jump_command.resampling_time_range = (1.0e9, 1.0e9)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    keyboard_controller = None
    if args_cli.keyboard:
        from rl_utils import WheelLeggedKeyboardController

        keyboard_controller = WheelLeggedKeyboardController(
            env,
            vx_limit=args_cli.command_range,
            wz_limit=args_cli.yaw_command_range,
            default_height=args_cli.keyboard_height,
            height_step=args_cli.keyboard_height_step,
            jump_height=args_cli.jump_height,
        )

    command_term = env.unwrapped.command_manager._terms.get("wheel_legged_commands")
    if command_term is not None and not args_cli.keyboard:
        command_term.set_debug_vis(True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    from rl_utils import camera_follow, format_debug_metrics

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if keyboard_controller is not None:
            keyboard_controller.update()
            # Make the new command visible to the policy in the same frame.
            obs = env.get_observations()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.print_interval > 0 and timestep % args_cli.print_interval == 0:
            debug_line = format_debug_metrics(extras.get("log", {}))
            if debug_line:
                print(debug_line)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
