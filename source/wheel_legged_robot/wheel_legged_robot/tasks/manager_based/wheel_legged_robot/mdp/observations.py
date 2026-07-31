# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def theta0(env: ManagerBasedRLEnv) -> torch.Tensor:
    # 环境初始化阶段（load_managers）也会调用此函数来检查输出形状，
    # 此时 env.theta0 尚未设置，返回零张量即可。
    if not hasattr(env, "theta0"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.theta0  # 返回形状为 (num_envs, 2)

def theta0_dot(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "theta0_dot"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.theta0_dot  # 返回形状为 (num_envs, 2)

def L0(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.L0  # 返回形状为 (num_envs, 2)

def L0_dot(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "L0_dot"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.L0_dot  # 返回形状为 (num_envs, 2)

def joint_wheel_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The wheel joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return joint_wheel_pos

def joint_wheel_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The wheel joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return joint_wheel_vel

def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint accelerations of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return joint_acc

def wheel_legged_commands(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: tuple[float, float, float] = (2.0, 0.25, 5.0),
) -> torch.Tensor:
    """VMC 策略的命令观测：线速度 x、角速度 yaw、高度目标（取前 3 个命令）。"""
    # 1. 获取命令张量（形状：num_envs, 4）
    commands = env.command_manager.get_command(command_name)  # 假设有 4 列
    # 2. 只取前 3 列：lin_vel_x, ang_vel_yaw, height
    commands = commands[:, :3]
    # 3. 应用缩放
    scale_tensor = torch.tensor(scale, device=env.device)
    return commands * scale_tensor


def obstacle_forward_scan(
    env: ManagerBasedRLEnv,
    distances: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
) -> torch.Tensor:
    """Robot-yaw-aligned forward height scan of the active obstacle geometry."""
    count = len(distances)
    if not hasattr(env, "obstacle_position_w"):
        return torch.zeros(env.num_envs, count, device=env.device)
    samples = torch.tensor(distances, device=env.device).unsqueeze(0)
    root_delta = env._robot.data.root_pos_w[:, :2] - env.obstacle_position_w
    obstacle_from_root = -torch.sum(
        root_delta * env.obstacle_forward_w, dim=1, keepdim=True
    )
    half_width = 0.5 * env.obstacle_width.unsqueeze(1)
    hit = (samples - obstacle_from_root).abs() <= half_width
    return torch.where(hit, env.obstacle_height.unsqueeze(1), 0.0)

def base_mass(env) -> torch.Tensor:
    """返回基座质量减去所有环境均值，形状 (num_envs, 1)"""
    if hasattr(env, "_base_mass"):
        mass = env._base_mass  # (num_envs,)
    else:
        # 若尚未随机化，从 URDF 读取默认质量
        asset = env.scene["robot"]
        mass = asset.data.default_mass[:, 0].to(device=env.device)  # 基座质量（确保在正确设备）
    # 计算均值并中心化
    mass_centered = mass - mass.mean()
    return mass_centered.unsqueeze(-1)  # (num_envs, 1)

def base_com(env) -> torch.Tensor:
    """返回基座质心偏移（x,y,z）"""
    if hasattr(env, "_base_com"):
        return env._base_com  # 形状 (num_envs, 3)
    else:
        return torch.zeros(env.num_envs, 3, device=env.device)

def default_joint_offset(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回默认关节位置的随机偏移量（仅匹配 asset_cfg 指定的关节）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "_default_joint_offset"):
        return env._default_joint_offset[:, asset_cfg.joint_ids]
    else:
        return torch.zeros(env.num_envs, len(asset_cfg.joint_ids), device=env.device)

def friction_coef(env) -> torch.Tensor:
    """返回地面摩擦系数（标量）"""
    if hasattr(env, "_friction_coef"):
        return env._friction_coef.unsqueeze(-1)
    else:
        return torch.ones(env.num_envs, 1, device=env.device) * 0.5

def restitution_coef(env) -> torch.Tensor:
    """返回地面恢复系数（标量）"""
    if hasattr(env, "_restitution_coef"):
        return env._restitution_coef.unsqueeze(-1)
    else:
        return torch.zeros(env.num_envs, 1, device=env.device) * 0.5
