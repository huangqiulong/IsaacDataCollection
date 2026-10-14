# 仿真数据完整工作流：在 Isaac 中规模化生成 → 数据处理 → 训练使用

> **本文定位**：把"仿真数据"作为一种独立的采集范式（Simulation-first）进行系统讲解：
> 如何以**少量人类演示为种子**，用 Isaac Lab Mimic 自动**规模化生产**高质量演示数据，
> 再经过质量处理，最终用于训练策略并用成功率闭环验证。
> 命令均来自源码核对（标 ⚠️ 处为高成本提示）。
>
> **配套文档**：
> - 种子数据的获取（遥操作录制）：[teleoperation-hands-on-franka-stack-cube.md](./teleoperation-hands-on-franka-stack-cube.md)
> - 视觉数据版（UMI）：[umi-in-isaac-workflow.md](./umi-in-isaac-workflow.md)
> - 移动操作版（EGO）：[ego-in-isaac-workflow.md](./ego-in-isaac-workflow.md)
> - HDF5 格式：[hdf5-format-guide.md](./hdf5-format-guide.md)

---

## 0. 前置

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets
```

**本文主线**（以 Stack-Cube 为范例，其它操作任务同理换任务名）：

```
人类演示 N 条(种子)           Mimic 标注                   Mimic 生成
dataset_keyboard.hdf5  →  annotated_kbd.hdf5  →  generated_kbd_100.hdf5 (+视频/状态)
   (遥操作录制)              (子任务边界)              (自动×100，只留成功)
                                                          ↓
                       成功率闭环 ← 策略回放 ← BC 训练 ← 数据质检/合并
```

---

## 1. 原理：仿真数据从哪里来

### 1.1 三种生成范式

| 范式 | 机制 | 典型任务 | 特点 |
| --- | --- | --- | --- |
| **示范扩充（Mimic）**⭐ | 人类演示 → 子任务分解 → 物体位姿随机重采样 + 运动规划拼接 | `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` | 数据"像人"、成功率高、最适合操作技能 |
| 技能生成（SkillGen） | 用技能片段库 + 规划器直接合成轨迹（无需重规划人类段）| `IsaacContrib-Stack-Cube-Franka-IK-Rel-Skillgen` | 扩展性更强，需 `--use_skillgen` |
| 域随机化（DR） | 材质/光照/位姿/物理参数随机 | 所有任务（含上述管线的生成阶段）| 决定 sim2real 泛化的多样性 |

### 1.2 为什么仿真数据能"规模化"

- 一条人类演示 = 定义"任务怎么做"的**结构**（先抓 A 再放到 B）；
- Mimic 保留结构，把**物体初始位姿随机重采样**后重新执行成功的段——一次产出无数变体；
- 失败轨迹自动丢弃（**只导出成功演示**），因此**生成成功率就是数据质量的第一指标**。

### 1.3 三类任务的 Mimic 任务名对照

| 数据类型 | Mimic 任务 ID |
| --- | --- |
| state（低维观测）| `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` |
| visuomotor（图像观测）| `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0` |
| skillgen 变体 | `IsaacContrib-Stack-Cube-Franka-IK-Rel-Skillgen` |

---

## 2. 模拟采集（Mimic 五步管线）

### 2.1 管线总览与前置条件

| 步骤 | 工具 | 产物 | 本机耗时参考 |
| --- | --- | --- | --- |
| ① 录种子 | `teleop record`（见遥操作手册）| `dataset_keyboard.hdf5`（≥3 条）| 已实测 ✅ |
| ② 标注 | `annotate_demos.py` | `annotated_kbd.hdf5` | ~2-5 分钟 |
| ③ 生成 | `generate_dataset.py` | `generated_kbd_*.hdf5` | 10 条约 5-10 分钟 |
| ④ 训练 | `robomimic/train.py` | checkpoint（`logs/robomimic/...`）| ~10-30 分钟 |
| ⑤ 评估 | `robomimic/play.py` | 成功率数字 | ~5-10 分钟 |

**前置条件**：种子演示必须与标准流程一致（Stack-Cube 为 蓝→红→绿 的顺序）——
`--auto` 标注依赖子任务完成信号（`subtask_terms`），流程乱了标注就会错（详见遥操作手册 §4.3）。

### 2.2 Step ②：标注（给演示"切段"）

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --visualizer kit --auto \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5
```

- 终端出现 **"Annotation of all demos done"** = 成功；
- 原理：每个子任务（grasp_1/stack_1/grasp_2）的**开始/结束帧**被记录，生成阶段据此"拆装重组"；
- GUI 模式可开着看标注过程；多个演示之间有停顿（自动）。

### 2.3 Step ③：生成（小规模验证 → 放大）

```bash
# 先 10 条验证（8GB 卡小并发）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --visualizer kit --num_envs 16 --generation_num_trials 10 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_small.hdf5

# 跑通后放大（8GB 建议 32~64 并发）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 32 --generation_num_trials 100 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5
```

关键参数（源码核对）：

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `--num_envs` | 并行环境数 | 8GB：16 → 32 → 64 逐级试 |
| `--generation_num_trials` | 尝试生成多少条 | 目标是"成功导出条数"，失败会自动丢弃 |
| `--use_skillgen` | 换 SkillGen 生成 | 配 Skillgen 任务使用（进阶）|
| `--disable_dataset_compression` | 关闭压缩 | 一般不需要；调试数据时可用 |
| `--pause_subtask` | 子任务间暂停 | GUI 下肉眼跟流程用 |

- **生成成功率**（终端统计）：state 模式官方参考 ~40%；<20% 说明种子结构/标注有问题；
- 生成阶段自动做物体位姿随机化——天然带域随机化属性。

### 2.4 进阶：实时"边录边生成"

`consolidated_demo.py` 把"遥操作录制 + Mimic 生成"合为一个流程（实时生成数据集）——
想一口气造大批数据时使用（参数与上面类同，未实测，⚠️ 高级用法）。

---

## 3. 数据处理

### 3.1 结构：生成数据 vs 种子数据

- 结构完全同构（HDF5 `data/demo_i/{actions,obs,states,initial_state} + attrs`，见 hdf5-format-guide）；
- 差异在**内容**：生成条的物体初始位姿更多样、动作由"人类段 + 拼接线"组成；
- `success=True` 只出现在成功条上（失败条不落盘）。

### 3.2 质检（训练前必做）

```bash
# ① 数据集统计（条数/步数/夹爪行为）
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py \
  ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5

# ② 曲线抽查（挑几条看动作连续性——拼接线是否平滑）
python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py \
  ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5 --demo demo_0 --out ~/code/isaac/IsaacDataCollection/datasets/gen_plot_0.png

# ③ 回放抽检（直观验证）
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel --visualizer kit --num_envs 1 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5
```

**质检标准**：步数分布集中（无离群超长条）、动作曲线无跳变、回放成功率高。

### 3.3 数据集管理

```bash
# 合并多个生成批次（不同种子/时间跑的多个 hdf5 → 一个大集）
python scripts/tools/merge_hdf5_datasets.py \
  --input_files ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_batch1.hdf5 ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_batch2.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_merged.hdf5
```

- 命名纪律：`dataset_*(种子) / annotated_* / generated_*` 三级分开，批次带序号；
- 体积估算：state 演示 ~500KB/条（700 步左右）→ 100 条约 **50MB**；图像版膨胀数倍；
- 磁盘紧张时生成完即训、训完归档到 `ops/caches/` 或外置盘。

---

## 4. 数据使用

### 4.1 BC 训练（robomimic）

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --algo bc \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5
  # 可选：--epochs 50（覆盖 JSON 里的训练轮数）；--name my_exp01
```

- 配置：任务自带 `bc_rnn_low_dim.json`（state 版）——训练参数不用手写；
- 日志/权重：`logs/robomimic/<实验名>/<时间戳>/`；
- 先 20~50 epoch 验证链路，再考虑加长。

### 4.2 评估：成功率闭环

```bash
# 标准评估（同一分布下 20 次 rollout）
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_rollouts 20 \
  --checkpoint logs/robomimic/<实验名>/<时间戳>/models/model_xxx.pth

# 鲁棒性评估（多种光照/纹理/种子矩阵——sim2real 预演）
./isaaclab.sh -p scripts/imitation_learning/robomimic/robust_eval.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --input_dir logs/robomimic/<实验名>/<时间戳>/models \
  --num_rollouts 20 --num_seeds 3 \
  --log_dir ~/code/isaac/IsaacDataCollection/datasets/eval_results
```

### 4.3 规模化决策表

| 观察 | 判断 | 行动 |
| --- | --- | --- |
| 生成成功率 ≥30% 且 BC 成功率 >50% | 链路健康 | 放大：补种子到 20-30 条 → 生成 1000 条 → 完整训练 |
| 生成成功率 <20% | 种子/标注问题 | 统一种子流程重录；检查标注边界 |
| BC 成功率 <20% | 数据不足/单一 | 增加生成条数、增加种子多样性 |
| 训练集表现好、鲁棒评估差 | 多样性不足 | 生成时加大随机化；用 robust_eval 定位薄弱设置 |

### 4.4 8GB 显存的规模参考

| 阶段 | 推荐规模 | 备注 |
| --- | --- | --- |
| 生成 | `--num_envs 32`，每批 ≤100 条 | 分批跑多次；显存峰值可控 |
| 训练（state）| 100~500 条数据，50~100 epoch | 模型小，显存充足 |
| 训练（visuomotor）| 见 UMI 文档 §4.1（batch 16 → 必要时调小）| 图像吃显存 |
| 评估 | `--num_rollouts 20`（GUI）/ 50+（无头统计）| 评估与训练不要同时跑 |

---

## 5. 与真机的差异 / sim2real 注意

- **动力学缺口**：接触摩擦、柔性、传感器噪声与真机有差距——生成数据训练的策略部署前
  需做鲁棒评估（§4.2）与真机小规模试验；
- **视觉缺口**（visuomotor）：渲染写实度是主要瓶颈，官方路线是"域随机化 + 少量真机数据微调"；
- **任务结构**：Mimic 生成保留的是"人类演示的结构"——演示本身不标准，生成再多也救不回来；
- **数据法律/成本**：仿真数据成本≈电费+显卡时间，这是相对真机（设备/场地/人力）的核心优势。

---

## 6. 附录

### 6.1 命令速查

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets

# ② 标注（state 版任务；visuomotor 换 ...-Visuomotor-Mimic-v0）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --visualizer kit --auto \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5 --output_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5

# ③ 生成（小 → 大）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --visualizer kit --num_envs 16 --generation_num_trials 10 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_small.hdf5
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 32 --generation_num_trials 100 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5

# 处理（统计 / 曲线 / 回放 / 合并）
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py <hdf5>
python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py <hdf5> --demo demo_0 --out plot.png
./isaaclab.sh -p scripts/tools/replay_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_envs 1 --dataset_file <hdf5>
python scripts/tools/merge_hdf5_datasets.py --input_files <a.hdf5> <b.hdf5> --output_file <merged.hdf5>

# 使用（训练 / 评估 / 鲁棒评估）
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel --algo bc --dataset <hdf5>
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel --visualizer kit --num_rollouts 20 --checkpoint <pth>
./isaaclab.sh -p scripts/imitation_learning/robomimic/robust_eval.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel --input_dir <models 目录> \
  --num_rollouts 20 --num_seeds 3 --log_dir ~/code/isaac/IsaacDataCollection/datasets/eval_results
```

### 6.2 FAQ

| 现象 | 原因 / 解决 |
| --- | --- |
| 标注失败/边界错乱 | 种子演示不符合标准子任务流程——统一种子重录 |
| 生成成功率为 0 | 任务/标注不匹配（task 名与数据集 env 不符）；种子过少（<2 条时先补）|
| 生成 OOM | 降 `--num_envs`（32→16→8）|
| 生成中途想停 | Ctrl+C；已成功条目可能未 flush——重跑时换输出文件名 |
| 训练 loss 不降 | 数据量太少/动作不连续；检查质检步骤 + 增加生成条数 |
| 回放成功率波动大 | 多试几个 checkpoint（不同 epoch）；20 次 rollout 统计噪声较大 |
| 数据集被覆盖 | 生成/训练脚本不会覆盖输入；录制种子时注意文件名（见遥操作手册）|
