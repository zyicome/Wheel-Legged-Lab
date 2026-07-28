# 轮腿机器人一键分阶段训练

## 1. 目标

自动训练并依次迁移：

```text
Wheel-Legged-Flat-v0
  → Wheel-Legged-Jump-Flat-v0
  → Wheel-Legged-Jump-High-Landing-Flat-v0
  → Wheel-Legged-Jump-Clearance-Flat-v0
  → Wheel-Legged-Jump-Moving-Curriculum-Flat-v0
```

外层流水线负责在任务之间迁移，最后一个任务内部已有的课程负责把移动
速度从 `0.2` 自动提高到 `1.0 m/s`。不再需要人工查看一次 Play 再手动
拼接下一条训练命令。

## 2. 一条命令从头训练

先进入项目使用的 Isaac Lab Conda 环境，然后执行：

```bash
cd /wheel_legged_robot

./scripts/rsl_rl/train_staged.sh
```

指定环境数量、设备和随机种子：

```bash
./scripts/rsl_rl/train_staged.sh \
  --num-envs 4096 \
  --device cuda:0 \
  --seed 42
```

只检查将要执行的命令，不启动训练：

```bash
./scripts/rsl_rl/train_staged.sh --dry-run
```

## 3. 从已经训练好的平地模型开始

如果 `Wheel-Legged-Flat-v0` 已经达到要求，可以跳过第一阶段：

```bash
./scripts/rsl_rl/train_staged.sh \
  --flat-checkpoint \
  logs/rsl_rl/wheel_legged_flat/<run>/model_<iteration>.pt
```

流水线会自动调用 `expand_rsl_checkpoint_for_jump.py` 扩展 Actor、Critic
和观测归一化器，并限制继承的探索标准差。不要直接把未转换的平地
checkpoint 加载到跳跃任务。

## 4. 自动晋级原理

每个阶段使用最近 `20` 个 iteration 的训练指标平均值，并要求连续
`5` 次判断全部通过。还设置了最低训练轮数，防止迁移初期的偶然高值
触发过早晋级。

主要验收内容：

| 阶段 | 核心条件 |
|---|---|
| Flat | 速度课程达到 `1.0 m/s`、yaw 课程达到 `2.0 rad/s`，跟踪得分和安全指标合格 |
| Jump Flat | 起跳速度、机身升高、滞空、成功率和恢复失败率合格 |
| High Landing | `9 cm` 以上机身上升，并满足落速、柔和落地和恢复要求 |
| Clearance | 轮端净空达到 `10 cm`，同时保持成功率与安全落地 |
| Moving Curriculum | 到达课程第 4 档，即 `1.0 m/s`，并同时通过移动、跳跃、落地和航向指标 |

完整数值位于：

```text
scripts/rsl_rl/staged_training_config.json
```

这些是带探索噪声的训练态门槛，不等于确定性 Play 的验收值。例如历史
C1 训练日志中的 `jump_success_rate` 可能只有约 `0.49`，但对应的
确定性 Play 表现更高，因此 C1 默认训练门槛设置为 `0.45`，同时要求
高度、落速、柔落和恢复指标共同通过。不要仅把该数值单独理解为最终
成功率要求。

## 5. Checkpoint 如何迁移

- Flat → Jump Flat：先扩展输入维度，再只加载 Actor/Critic 权重。
- 后续阶段：观测和动作维度相同，直接只加载 Actor/Critic 权重。
- 每次切换任务都会使用新的优化器和新的 iteration 计数。
- 最终移动任务仍从课程第 0 档开始，环境内部自动晋级到第 4 档。

`--load_weights_only` 是有意设置的。不同阶段奖励分布发生变化，沿用上一
阶段的 Adam 动量容易在新阶段产生过大的早期更新。

## 6. 日志与失败保护

每次运行会建立：

```text
logs/rsl_rl/staged_pipeline/<timestamp>/
```

其中包括：

- `state.json`：已完成阶段、每阶段 checkpoint 和最终 checkpoint；
- `*_gate.json`：该阶段实际使用的门槛；
- `*_status.json`：停止时的滑动平均指标；
- `model_flat_jump_init.pt`：转换后的平地初始化模型。

每个任务的完整训练日志仍保存在原来的 `logs/rsl_rl/<experiment>/`
目录。

如果某阶段用完最大 iteration 后仍未通过，流水线返回非零状态并停止，
同时保留该阶段最终 checkpoint 和状态文件。这样不会把尚未掌握当前
能力的模型自动传到更困难的任务。

## 7. 调整原则

优先调整每阶段的 `max_iterations`，不要为了让流水线“通过”就明显降低
所有门槛。只有在多次训练都稳定卡在同一指标、并且确定性 Play 已经证实
能力合格时，才根据训练态与 Play 的系统偏差调整对应规则。

建议保留以下硬性安全条件：

- `torque_saturation` 不长期出现；
- `jump_fail_recovery_rate` 接近零；
- 高跳阶段必须同时检查高度和落速；
- 净空阶段必须检查真实 `jump_wheel_clearance`，不能用机身高度替代；
- 最终阶段必须同时检查课程档位、移动跟踪、跳跃成功、柔和落地和航向。
