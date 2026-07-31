# Wheel-Legged Robot Learning

基于 NVIDIA Isaac Lab 的开链轮腿平衡机器人强化学习项目，包含平移、旋转、原地跳跃、高跳落地、轮端收腿净空，以及边移动边跳跃等任务。

> 这是一个以学习、复现和交流为主要目标的个人项目。作者目前也处于强化学习与机器人控制的入门学习阶段，代码和训练方案难免存在不足。欢迎大家通过 Issue、Discussion 或 Pull Request 一起分析问题、交流经验并完善项目。希望这份从“先站稳、再移动、再跳起来”逐步迭代的记录，能为同样在学习 Isaac Lab 和机器人强化学习的朋友提供一些参考。

## 项目简介

本项目面向一台两腿、两轮的开链轮腿机器人。机器人需要在欠驱动条件下保持机身平衡，并根据速度、高度和跳跃命令完成运动。

项目采用 Isaac Lab 的 Manager-Based 环境组织方式，以 RSL-RL PPO 为主要训练后端，并在腿部控制中加入虚拟模型控制（VMC）。相比直接输出六个关节目标，策略输出左右虚拟腿角、虚拟腿长和轮速参考，再由 VMC 转换为实际关节力矩。

当前已经实现：

- 平地自平衡；
- 前进、后退与航向控制；
- 机身高度调节；
- 速度与航向课程学习；
- 外部命令触发的六阶段跳跃状态机；
- 原地小跳、高跳与落地吸能；
- 空中收腿，提高轮端真实净空；
- 移动过程中起跳、落地并恢复速度；
- `0.2 → 0.4 → 0.6 → 0.8 → 1.0 m/s` 移动跳跃课程；
- 平地模型到跳跃模型的观测维度迁移；
- 按训练指标自动切换任务的一键分阶段训练；
- Play 调试指标输出与键盘自由控制；
- 跳跃开环物理能力测试。

本项目仍是研究与学习性质的仿真工程，尚未完成真实机器人部署和系统性的 Sim-to-Real 验证。

## 演示、预训练模型与训练曲线

以下演示均使用最终移动跳跃策略 `model_844.pt`：

### 单环境测试

单台机器人执行移动、转向和跳跃：

![单环境移动跳跃演示](docs/media/single_env_demo.gif)

### 多环境测试

多个并行环境中的策略表现：

![多环境移动跳跃演示](docs/media/multi_env_demo.gif)

### 键盘控制

使用键盘实时控制移动、转向和跳跃：

![键盘控制演示](docs/media/keyboard_control_demo.gif)

GIF 会在 GitHub README 中直接循环播放。高清版本可单独下载：
[单环境 MP4](docs/media/single_env_demo.mp4) ·
[多环境 MP4](docs/media/multi_env_demo.mp4) ·
[键盘控制 MP4](docs/media/keyboard_control_demo.mp4)

预训练 checkpoint：

```text
checkpoints/wheel_legged_moving_jump_model_844.pt
```

- 对应环境：`Wheel-Legged-Jump-Moving-Curriculum-Flat-v0`
- 课程终点：`1.0 m/s` 平移速度、`1.2 rad/s` yaw 命令范围
- SHA-256：`d1d271f0c323fa13538b80119839b29eb267942af63a6e0d6c3dbf5ef319deb0`

直接运行预训练策略：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint checkpoints/wheel_legged_moving_jump_model_844.pt \
  --num_envs 1 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

最终阶段的 TensorBoard 曲线由原始 event 文件自动导出：

![model_844 训练曲线](docs/media/training_curves_model_844.png)

原始 TensorBoard event 已保存在 `docs/tensorboard/model_844/`，可以重新
交互式查看：

```bash
tensorboard --logdir docs/tensorboard/model_844 --port 6006
```

如需从其他训练日志自动生成相同格式的静态图：

```bash
python scripts/plot_tensorboard_curves.py \
  /path/to/events.out.tfevents.* \
  --output training_curves.png \
  --title "Wheel-Legged Training"
```

曲线和视频来自本项目当前 checkpoint 的单次训练与 Play，主要用于复现
和展示，不代表多随机种子的统计置信区间。

## 训练路线

项目没有把所有能力一次性塞进同一个奖励函数，而是逐步增加难度：

```text
平移与旋转
Wheel-Legged-Flat-v0
        │
        ▼
基础原地跳跃
Wheel-Legged-Jump-Flat-v0
        │
        ▼
高跳与落地强化
Wheel-Legged-Jump-High-Landing-Flat-v0
        │
        ▼
轮端收腿与净空强化
Wheel-Legged-Jump-Clearance-Flat-v0
        │
        ▼
移动跳跃课程
Wheel-Legged-Jump-Moving-Curriculum-Flat-v0
```

最后得到的是一个带普通移动样本和跳跃样本的统一策略，而不是在移动策略与跳跃策略之间硬切换。

## 可用环境

| 环境 ID | 主要用途 |
|---|---|
| `Wheel-Legged-Flat-v0` | 平地平衡、速度跟踪、高度和航向控制 |
| `Wheel-Legged-Jump-Flat-v0` | 基础原地跳跃与六阶段状态机 |
| `Wheel-Legged-Jump-High-Landing-Flat-v0` | `0.09–0.12 m` 高跳、预伸腿和落地吸能 |
| `Wheel-Legged-Jump-Clearance-Flat-v0` | 空中收腿和 `0.08–0.12 m` 轮端净空 |
| `Wheel-Legged-Jump-Moving-Flat-v0` | 第一档低速移动跳跃 |
| `Wheel-Legged-Jump-Moving-Curriculum-Flat-v0` | 从 `0.2 m/s` 自动训练到 `1.0 m/s` 的移动跳跃 |

## 项目结构

```text
Wheel-Legged-Lab/
├── source/wheel_legged_robot/
│   └── wheel_legged_robot/tasks/manager_based/wheel_legged_robot/
│       ├── agents/                  # PPO 配置
│       ├── assets/                  # 机器人资产配置
│       ├── mdp/
│       │   ├── actions.py           # VMC 动作与力矩映射
│       │   ├── commonds.py          # 速度、高度和航向命令
│       │   ├── curriculums.py       # 移动与移动跳跃课程
│       │   ├── events.py            # 随机化与重置事件
│       │   ├── jump.py              # 跳跃命令、观测和奖励
│       │   ├── observations.py      # 观测项
│       │   └── rewards.py           # 平衡与移动奖励
│       ├── wheel_legged_flat_env_cfg.py
│       └── wheel_legged_jump_env_cfg.py
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                 # RSL-RL 训练
│   │   ├── play.py                  # 模型评估与键盘控制
│   │   ├── train_staged.sh          # 一键分阶段训练入口
│   │   └── staged_training_config.json
│   ├── expand_rsl_checkpoint_for_jump.py
│   └── jump_open_loop_test.py
├── JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md
├── JUMP_TASK_DESIGN.md
└── STAGED_TRAINING.md
```

## 依赖

### 已验证的开发环境

| 组件 | 版本 |
|---|---|
| 操作系统 | Linux | Ubuntu22.04
| Python | 3.11 |
| NVIDIA Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| RSL-RL | 5.0.1 |
| Gymnasium | 1.3.0 |

以上是当前开发机器上的版本记录，不代表其他版本一定不可使用。Isaac Sim、Isaac Lab、PyTorch 和 CUDA 的兼容关系变化较快，建议优先按照 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 创建环境。

### 硬件建议

- 支持 CUDA 的 NVIDIA GPU；
- 默认训练使用 `4096` 个并行环境；
- 显存不足时可先将 `--num_envs` 调低至 `512`、`1024` 或 `2048`；
- Play 和功能验证通常只需要 `1–50` 个环境。

### 机器人模型资产

机器人 URDF 和 STL 已放在扩展内部：

```text
source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/
└── wheel_legged_robot/assets/wheellegged_description/
```

代码通过相对于 Python 包的路径加载模型，不再依赖本地 `robot_lab` 数据目录。机器人模型来自 [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/master/resources/robots/wl)，按上游 BSD 3-Clause 许可证再分发。`wl_dealed.urdf` 在原模型基础上把腿杆碰撞 mesh 替换为 box，以提高仿真稳定性和效率。具体来源、修改和许可证见[资产许可说明](source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/ASSET_LICENSE.md)。

## 安装

### 1. 安装 Isaac Lab

先按照官方文档安装 Isaac Sim 和 Isaac Lab，并确认下面的命令能够正常运行：

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
"${ISAACLAB_ROOT}/isaaclab.sh" -p -c "import isaaclab; print('Isaac Lab is ready')"
```

如果使用 Conda，请先激活安装 Isaac Lab 的环境：

```bash
conda activate env_isaaclab
```

### 2. 克隆项目

克隆本仓库：

```bash
git clone https://github.com/zyicome/Wheel-Legged-Lab.git
cd Wheel-Legged-Lab
```

### 3. 安装扩展

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p -m pip install -e source/wheel_legged_robot
```

列出已注册环境：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/list_envs.py
```

如果可以看到以 `Wheel-Legged-` 开头的环境，说明 Python 扩展已经成功安装。

## 快速开始

以下命令均假设当前目录是项目根目录，并且已经设置：

```bash
export ISAACLAB_ROOT="/absolute/path/to/IsaacLab"
```

### 训练平地移动策略

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --run_name flat_baseline
```

训练输出保存在：

```text
logs/rsl_rl/wheel_legged_flat/<timestamp>_<run_name>/
```

### 单独训练一个跳跃阶段

不同跳跃任务的观测维度一致，可以使用上一阶段 checkpoint 只迁移网络权重：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-High-Landing-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/previous/model_700.pt \
  --load_weights_only \
  --run_name high_landing
```

不要把未经转换的平地 checkpoint 直接加载到跳跃任务。平地策略缺少跳跃观测，需要先使用 `scripts/expand_rsl_checkpoint_for_jump.py` 扩展首层网络和归一化器。

### 训练目标落点阶段

`Wheel-Legged-Jump-Target-Landing-Flat-v0` 在现有 8–12 cm 移动跳跃能力上
加入与速度一致的目标落点。触发跳跃时按照
`target_distance = vx_command × 0.16 s × U(0.9, 1.1)` 锁存带符号距离，并
限制在 `[-0.16, 0.16] m`；因此前进命令对应前方落点，后退命令对应后方落点。
目标点以起跳瞬间的位置和机头方向定义，不会退化成只沿世界 X 轴跳跃。训练覆盖
`vx ∈ [-1, 1] m/s` 和最大 `wz ∈ [-1.2, 1.2] rad/s`。该任务仍使用原来的
48 维跳跃观测，可以直接迁移最终移动跳跃 checkpoint：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Target-Landing-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path checkpoints/wheel_legged_moving_jump_model_844.pt \
  --load_weights_only \
  --run_name target_landing
```

训练时重点观察 `J落点误差` 和 `J落点成功`。默认成功条件是首次触地位置与目标
二维距离不超过 0.05 m，同时仍需满足原有的起跳高度、轮端净空、滞空、柔和
落地、速度恢复和姿态恢复条件。固定 0.10 m 落点进行 Play：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Target-Landing-Flat-v0 \
  --checkpoint /absolute/path/to/target_landing/model_1000.pt \
  --num_envs 10 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10 \
  --jump_distance 0.10
```

### 训练 Oracle 障碍物阶段

`Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0` 在每个环境中放置一个横跨行驶
方向的刚性障碍物。第一阶段随机暴露高度为 2–4 cm、宽度为 3.5 cm，环境根据
障碍物距离和当前速度解析计算触发时机：

```text
trigger_distance = |vx_command| × 0.44 s + 0.03 m
```

低层策略不需要新增障碍物观测，仍接收原来的 48 维跳跃输入，因此可以直接迁移
目标落点模型：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/target_landing/model_xxx.pt \
  --load_weights_only \
  --run_name obstacle_oracle_low
```

第一阶段接近速度限制为 0.45–0.75 m/s。训练时重点观察：

```text
O触发误差  趋近 0
O净空      最终大于 0.015 m
O跨越      最终大于 0.80
O碰撞      最终小于 0.03
O成功      最终大于 0.70
```

使用训练后的模型进行自动越障 Play：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --checkpoint /absolute/path/to/obstacle_oracle/model_xxx.pt \
  --num_envs 10 \
  --command_range 0.75 \
  --yaw_command_range 0.4
```

### 一键完成全部阶段

从平地训练开始，达到各阶段指标后自动切换任务：

```bash
./scripts/rsl_rl/train_staged.sh \
  --isaaclab-path "${ISAACLAB_ROOT}" \
  --num-envs 4096 \
  --device cuda:0 \
  --seed 42
```

如果已经有合格的平地 checkpoint，可以跳过平地训练：

```bash
./scripts/rsl_rl/train_staged.sh \
  --isaaclab-path "${ISAACLAB_ROOT}" \
  --start-checkpoint /absolute/path/to/wheel_legged_flat/model_1999.pt \
  --start-stage flat \
  --start-mode next
```

也可以从任意中间阶段恢复。三个参数必须一起提供：

| 参数 | 含义 |
|---|---|
| `--start-checkpoint` | 用作恢复或迁移起点的模型文件 |
| `--start-stage` | 该模型所属阶段，而不是准备进入的阶段 |
| `--start-mode continue` | 恢复当前阶段的网络、优化器和 iteration；达标后再自动切换 |
| `--start-mode next` | 确认当前阶段已经合格，跳过其验收并将网络权重迁移至下一阶段 |

可用的阶段名称依次为：

```text
flat → jump_flat → high_landing → clearance → moving_curriculum
```

例如，继续训练一个尚未完全达标的 High-Landing 模型：

```bash
./scripts/rsl_rl/train_staged.sh \
  --isaaclab-path "${ISAACLAB_ROOT}" \
  --start-checkpoint \
  /absolute/path/to/wheel_legged_jump_high_landing_flat/model_800.pt \
  --start-stage high_landing \
  --start-mode continue
```

如果已经确认这个 High-Landing 模型合格，直接进入 Clearance：

```bash
./scripts/rsl_rl/train_staged.sh \
  --isaaclab-path "${ISAACLAB_ROOT}" \
  --start-checkpoint \
  /absolute/path/to/wheel_legged_jump_high_landing_flat/model_800.pt \
  --start-stage high_landing \
  --start-mode next
```

最终阶段 `moving_curriculum` 后面没有其他阶段，因此只能选择
`--start-mode continue`。旧参数 `--flat-checkpoint PATH` 仍然兼容，
等价于以 `flat + next` 模式启动。

流水线使用最近一段训练指标的滑动平均决定是否晋级。如果某一阶段达到最大训练轮数仍未通过，流程会停止并保留 checkpoint，不会把未达标模型继续传给更困难的任务。

详细说明见 [STAGED_TRAINING.md](STAGED_TRAINING.md)。

### 查看训练结果

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 50 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

Play 默认关闭观测噪声、地形课程、随机推力和命令课程，使用命令行给定的速度与跳跃范围进行确定性评估。

### 键盘自由控制

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --keyboard \
  --real-time \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

启动后先点击一次 Isaac Sim Viewport，使窗口获得键盘焦点。

| 按键 | 功能 |
|---|---|
| `↑ / ↓` 或数字键盘 `8 / 2` | 前进 / 后退 |
| `Z / X` 或数字键盘 `7 / 9` | 左转 / 右转 |
| `R / F` | 升高 / 降低机身 |
| `J` | 触发一次跳跃 |
| `L` | 清零移动命令并恢复默认高度 |

空格键属于 Isaac Sim 的时间轴暂停快捷键，不用于跳跃。暂停并恢复后如果键盘没有响应，请确认时间轴正在播放，重新点击 Viewport，再按一次 `L` 清除残留按键状态。

### 观察 TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

除了总奖励，建议重点观察：

- 平移和航向跟踪误差；
- `jump_takeoff_vz`、`jump_air_time`；
- `jump_apex_rise`、`jump_wheel_clearance`；
- `jump_success_rate`、`jump_soft_landing_rate`；
- `jump_landing_vz`、`jump_fail_recovery_rate`；
- `joint_margin_min`、`torque_saturation`；
- 最终移动跳跃课程档位和各档通过指标。

总奖励上升不一定意味着机器人真的学会了跳跃，应同时检查接触、滞空、轮端净空和落地恢复。

## 跳跃开环测试

在设计跳跃奖励前，可以先验证 VMC、关节行程、电机力矩和接触模型是否具备真实起跳能力：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/jump_open_loop_test.py \
  --task Wheel-Legged-Flat-v0 \
  --headless
```

脚本会并行扫描下蹲腿长、下蹲时间和蹬伸腿长，并把结果保存为 CSV 和 JSON。详细方法见 [JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md](JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md)。

## 设计文档

- [跳跃开环测试与训练路线](JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md)
- [跳跃状态机、奖励和阶段设计](JUMP_TASK_DESIGN.md)
- [一键分阶段训练说明](STAGED_TRAINING.md)

这些文档保留了项目迭代过程中出现的问题、判断依据和训练经验。部分历史数值来自特定 checkpoint，不应被理解为所有机器人和随机种子都能直接达到的保证。

## 当前限制

- 尚未完成真实机器人部署；
- 尚未系统验证传感器噪声、通信延迟和执行器热衰减；
- 目前主要在平坦地面训练，尚未加入完整障碍物感知和自主起跳时机规划；
- 自动分阶段训练门槛来自当前机器人和已有训练日志，改变质量、尺寸、电机或奖励后需要重新校准；
- 机器人模型来自 Wheel-Legged-Gym，相关 BSD 3-Clause 许可证与修改说明已随资产保留；
- RSL-RL 是当前主要验证后端，CusRL 脚本仍属于实验性支持；
- 当前结果不能替代多随机种子统计和真实硬件安全验证。

## Roadmap

- [x] 将机器人 URDF 与 mesh 迁入项目并改为包内路径；
- [x] 补充 Wheel-Legged-Gym 模型来源、修改记录与 BSD 3-Clause 许可证；
- [x] 补充训练与 Play 的 GIF/视频；
- [x] 发布可复现 checkpoint 与 TensorBoard 曲线；
- [x] 增加平地目标落点命令与落点精度训练阶段；
- [x] 增加低障碍物 Oracle 触发与实体碰撞训练阶段；
- [ ] 增加障碍物高度/宽度课程与前向地形感知；
- [ ] 从外部触发跳跃扩展到感知驱动的自主越障；
- [ ] 加入多随机种子评估和自动回归测试；
- [ ] 完成延迟、噪声、摩擦和电机参数随机化；
- [ ] 探索 Sim-to-Real 与真实机器人部署；
- [ ] 补充英文 README。

## 参与讨论与贡献

这个项目不是一份“已经完成的标准答案”，而是一份持续更新的学习记录。作者对强化学习、控制理论和工程实现仍有许多需要学习的地方，也非常欢迎有经验的开发者指出错误。

如果你在复现过程中发现问题，建议在 Issue 中附上：

- Isaac Sim、Isaac Lab、Python、PyTorch 和 RSL-RL 版本；
- 使用的任务名、命令和 checkpoint；
- GPU 型号与 `--num_envs`；
- 完整报错或关键训练指标；
- 能够复现问题的最小配置。

欢迎提交：

- 文档修正和更清晰的解释；
- 奖励、观测与课程学习改进；
- 不同机器人参数上的复现实验；
- 训练曲线、失败案例和消融实验；
- 键盘控制、评估和部署工具；
- 中英文翻译。

无论结果成功或失败，只要过程和条件记录清楚，都可能对其他学习者有帮助。

## 致谢

本项目的学习与实现离不开以下优秀开源项目：

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)：提供基于 Isaac Sim 的机器人学习框架、Manager-Based 环境、传感器、仿真和强化学习接口。
- [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym)（[BSD-3-Clause](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/master/LICENSE)）：本项目使用了该项目发布的轮腿机器人 URDF 与 STL，并参考了轮腿平衡、平移运动和 VMC 相关思路。当前使用的 `wl_dealed.urdf` 在原 URDF 上调整了腿杆碰撞几何。感谢原作者及贡献者公开模型和训练实现。
- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)（[Apache-2.0](https://github.com/fan-ziqi/robot_lab/blob/main/LICENSE)）：本项目在学习 Isaac Lab Manager-Based 任务组织、环境配置和工程结构时参考了该项目。感谢作者及社区提供清晰、丰富的机器人强化学习实践。
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：提供本项目主要使用的 PPO 训练实现。

上述项目的版权与商标归各自作者和组织所有。若本仓库中包含基于第三方代码修改的文件，应继续保留原始版权声明，并遵守相应许可证要求。

## 许可证与免责声明

除文件头或第三方组件另有声明外，本项目新增代码采用根目录 [Apache License 2.0](LICENSE)。来自 Wheel-Legged-Gym 的机器人 URDF、STL 及本项目的派生 URDF 继续采用 BSD 3-Clause，完整许可证和修改记录见[资产许可说明](source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/ASSET_LICENSE.md)。其他第三方代码仍以各文件或上游项目声明为准。

本项目仅供学习与研究。强化学习策略可能输出突变动作或超出预期的控制命令。部署到真实机器人前，请完成限位、力矩、速度、急停、悬挂测试和人员隔离等安全措施。作者不对直接使用本项目导致的设备损坏或人身风险承担责任。
