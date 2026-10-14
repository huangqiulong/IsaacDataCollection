# 动捕数据完整工作流：在 Isaac 中模拟采集 → 数据处理 → 训练使用

> **本文定位**：以真机动作捕捉（Vicon/Xsens/视频姿态估计）为标尺，以 Isaac 的
> **AMP（Adversarial Motion Priors）人形任务**为核心，完整说明动捕数据在仿真里的
> "采集 → 处理 → 使用"链路：从数据从哪来、长什么样，到机器人如何"学会"这些动作。
> 命令均来自源码核对（标 ⚠️ 处为未实测/高成本提示）。
>
> **配套文档**：
> - 全局背景：[robot-data-collection-in-isaac-survey.md](./robot-data-collection-in-isaac-survey.md)（§3）
> - 仿真数据管线（Mimic，另一条"数据生成"主线）：[simulation-data-in-isaac-workflow.md](./simulation-data-in-isaac-workflow.md)

---

## 0. 前置

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate
```

**数据位置**（仓库内置，三段已重定向好的动捕动作）：

```
source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/motions/
├── humanoid_dance.npz     # 跳舞
├── humanoid_run.npz       # 跑步
├── humanoid_walk.npz      # 走路
├── motion_viewer.py       # 可视化工具
└── README.md              # 字段字典（数据契约）
```

**任务**（三段动捕一一对应）：

| 任务 ID | 动作数据 | RL 后端 |
| --- | --- | --- |
| `IsaacContrib-Humanoid-AMP-Dance-Direct` | humanoid_dance.npz | **skrl**（默认）|
| `IsaacContrib-Humanoid-AMP-Run-Direct` | humanoid_run.npz | skrl |
| `IsaacContrib-Humanoid-AMP-Walk-Direct` | humanoid_walk.npz | skrl |

---

## 1. 原理：动捕是什么 → Isaac 里如何使用

### 1.1 真机动捕链路（两段外部环节）

```
动捕棚录制/视频       重定向（retargeting）        机器人侧数据
(Vicon/Xsens BVH,  →  (人体骨骼比例 ≠ 机器人     →  .npz 关节/刚体轨迹
 或视频→SMPL)          骨骼，需映射+物理可行性处理)      (仿真可消费)
```

- **重定向是最大瓶颈**：人体动作直接套到机器人上会穿地/失稳，需要专门工具处理
  （本机未配置；外部代表工具：GMR 等）；
- 只有视频时需先做"视频→SMPL 参数"（GVHMR 类），再重定向——**Isaac 不负责这两步，只消费结果**。

### 1.2 Isaac 的消费方式：AMP（对抗式运动先验）

不把动捕当"逐帧目标轨迹"，而是当"**风格先验**"：

- 判别器学习"什么动作像人（=像动捕数据）"；
- 策略同时拿 任务奖励 + 风格奖励（骗过判别器）；
- 结果：机器人学会'用对的风格'完成移动任务——比逐帧跟踪更自然、更鲁棒。

> 想理解闭环：`npz(动捕风格) → 判别器 → 风格奖励 → 策略训练 → 仿真人按这种风格动起来`。

---

## 2. 模拟采集

### 2.1 数据获取（三种来源）

| 来源 | 做法 | 状态 |
| --- | --- | --- |
| ① 仓库内置（推荐入门） | 直接用三段 npz | ✅ 已就位 |
| ② 官方素材再加工 | 换更多动作（如武术、舞蹈片段）→ 重定向到 G1 骨骼 → 生成同格式 npz | 需外部重定向工具 |
| ③ 自制（进阶） | 真人录制（手机视频→SMPL→重定向）或动捕设备直出 BVH | 需完整外部工具链 |

**数据契约**（无论哪种来源，npz 必须满足 README 格式，见 §3.1 字段表）——
这是"采集"环节最该先吃透的部分：**采出来的数据必须能对上机器人 DOF/body 名称**。

### 2.2 数据检查（先"读"再"用"）

```bash
cd source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/motions

# 字段与形状总览（对应 §3.1 表）
python - <<'PY'
import numpy as np
d = np.load("humanoid_dance.npz", allow_pickle=True)
for k in d.files:
    print(f"{k:28s} {str(d[k].shape):16s} {d[k].dtype}")
PY
```

### 2.3 可视化（"看见"动作）

```bash
cd source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/motions

# 首次需要 matplotlib（清华源）
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install matplotlib

# 播放动捕骨架动画（纯 CPU，不启动 Isaac）
python motion_viewer.py --file humanoid_dance.npz
python motion_viewer.py --file humanoid_run.npz
python motion_viewer.py --file humanoid_walk.npz
# 可选：--render-scene 显示 3D 场景背景；--matplotlib-backend TkAgg（默认）
```

> 快捷键/交互：matplotlib 窗口内拖动旋转；关闭窗口结束。
> **看点**：动作帧率（fps 字段）、关节间协调、落脚时机——这就是机器人要模仿的"风格"。

---

## 3. 数据处理

### 3.1 字段字典（数据契约，来自 motions/README.md）

设 N=帧数、D=骨架 DOF 数、B=刚体数：

| 键 | 类型 | 形状 | 含义 |
| --- | --- | --- | --- |
| `fps` | int64 | () | 采样帧率 |
| `dof_names` | 字符串 | (D,) | 关节名 |
| `dof_positions` | float32 | (N, D) | 关节位置序列 |
| `dof_velocities` | float32 | (N, D) | 关节速度序列 |
| `body_names` | 字符串 | (B,) | 刚体名 |
| `body_positions` | float32 | (N, B, 3) | 刚体位置 |
| `body_rotations` | float32 | (N, B, 4) | 刚体旋转（**wxyz** 四元数）|
| `body_linear_velocities` | float32 | (N, B, 3) | 刚体线速度 |
| `body_angular_velocities` | float32 | (N, B, 3) | 刚体角速度 |

**处理要点**：

1. **四元数顺序**：bg 数据是 `wxyz`（Isaac Lab 内部约定；而 HDF5 遥操作数据是 `xyzw`——
   跨格式时注意，见 hdf5-format-guide §3.6）；
2. **单位/坐标系**：位置单位米、Isaac Lab 世界系（Z 上）；
3. **帧率一致性**：fps 决定 AMP 采样时序，自制数据时不要随意改；
4. **DOF/body 名称必须与目标机器人匹配**（G1 29DoF）——重定向工具的最终输出校验点。

### 3.2 自制/替换动作数据的最小步骤（进阶）

1. 准备重定向后的关节轨迹（外部工具；输出即上表字段）；
2. 用 `dof_names`/`body_names` 对齐 G1 的名称（拼写完全一致）；
3. 存成 `.npz`（`np.savez`）放进 `motions/`；
4. 在任务配置里指过去——`humanoid_amp_env_cfg.py` 中对应类的 `motion_file` 字段：

```python
# humanoid_amp_env_cfg.py（示意，97/102/107 行附近）
motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")
```

5. 跑一遍 §2.2 检查脚本 + §4 训练验证（判别器能正常吃新数据）。

---

## 4. 数据使用：AMP 训练 → 回放评估

### 4.1 训练（skrl 后端，默认自动解析）

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate

# 跳舞（8GB 显存建议 512 环境；想更省显存可 256）
./isaaclab.sh train --task IsaacContrib-Humanoid-AMP-Dance-Direct --num_envs 512
# 跑步 / 走路：换 --task IsaacContrib-Humanoid-AMP-Run-Direct / ...-Walk-Direct
```

配置实测要点（`skrl_dance_amp_cfg.yaml`）：

| 项 | 值 |
| --- | --- |
| rollouts / mini_batches | 16 / 2 |
| learning_rate | 5e-5 |
| timesteps | 80000（总训练量）|
| checkpoint_interval | auto |

- 日志/权重：`logs/skrl/humanoid_amp_dance/<时间戳>/`（终端会打印路径）；
- ⚠️ **预期耗时**：本机未实测完整训练；先用少量迭代观察趋势（阻尼/判别器损失），
  完整训练量（80000 timesteps）预计**小时级**，建议后台跑（`setsid` + 日志重定向）；
- 显存纪律：训练前确认 GPU 空闲（`nvidia-smi`）；同机不要叠加其它仿真。

### 4.2 回放评估

```bash
./isaaclab.sh play --task IsaacContrib-Humanoid-AMP-Dance-Direct \
  --checkpoint logs/skrl/humanoid_amp_dance/<时间戳>/checkpoints/<best/last>.pt
```

**评估看点**：

1. 动作"像不像"目标风格（对比 §2.3 的动捕播放）；
2. 是否物理稳定（不抽搐、不穿地、能持续）；
3. 换 checkpoint 对比（训练前期 vs 后期风格差异明显）。

### 4.3 观察点与调参直觉

| 指标 | 健康信号 | 异常与对策 |
| --- | --- | --- |
| episode 回报 | 缓慢上升并稳定 | 长期不涨 → 检查风格奖励权重 |
| 判别器损失 | 稳定振荡（不塌缩到 0/巨大）| 塌缩 → 降判别器学习率/加正则 |
| 动作自然度 | 越来越像动捕 | 风格太弱 → 提高 style reward 权重 |

---

## 5. 与真机的差异 / 局限

- 本仓库的 npz 是**已重定向成品**，省去了"动捕棚→重定向"两大真机环节；
  真机实际使用时这两步的质量决定了数据上限；
- 重定向误差（人体比例 vs 机器人）会表现为"物理不可行"的参考动作——
  AMP 对此有一定容忍（它学风格而非复刻），但逐帧跟踪类任务会更敏感；
- 仿真动捕数据的"多样性"取决于你提供的动作数量：三段动作 vs 真机数百段演示
  在风格覆盖上差距明显——想丰富，需走 §3.2 的自制路径。

---

## 6. 附录

### 6.1 命令速查

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate

# 检查 / 可视化
cd source/isaaclab_tasks/isaaclab_tasks/contrib/humanoid_amp/motions
python motion_viewer.py --file humanoid_dance.npz
cd ~/code/isaac/isaaclab

# 训练（Dance/Run/Walk）
./isaaclab.sh train --task IsaacContrib-Humanoid-AMP-Dance-Direct --num_envs 512
./isaaclab.sh train --task IsaacContrib-Humanoid-AMP-Run-Direct  --num_envs 512
./isaaclab.sh train --task IsaacContrib-Humanoid-AMP-Walk-Direct --num_envs 512

# 回放
./isaaclab.sh play --task IsaacContrib-Humanoid-AMP-Dance-Direct \
  --checkpoint logs/skrl/humanoid_amp_dance/<时间戳>/checkpoints/last.pt
```

### 6.2 FAQ

| 现象 | 原因 / 解决 |
| --- | --- |
| `motion_viewer.py` 报缺 matplotlib | `uv pip install matplotlib`（清华源）|
| 训练起步很慢 | 首次编译/缓存（热身后会快）；确认没开其它 GPU 任务 |
| 机器人动作抽搐 | 训练不足 / 风格权重失衡；加长训练、调 style 权重 |
| 想听声音/看视频回放 | play 时开 `--visualizer kit` 观察实时效果；视频录制用训练 cfg 里 video 选项 |
| 自制 npz 报名称不匹配 | `dof_names`/`body_names` 必须与 G1 完全一致（见 §3.1/3.2）|
| 显存不足 | `--num_envs 256`；关闭浏览器等占显存进程 |
