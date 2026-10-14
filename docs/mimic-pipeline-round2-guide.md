# Mimic 第二轮数据流程操作指南（全新录 30 条 → 标注 → 放大 → 加长训练 → 评估）

> 日期：2026-10-13 ｜ 关联：《`teleoperation-hands-on-franka-stack-cube.md`》（设备操作/录制细节）、
> `ops/docs/isaac-learning-log.md`（第一轮实测记录与教训）
>
> **目标（本轮方案）**：**全新录制 30 条演示**（不沿用第一轮的旧 10 条），从头完整重跑
> 标注 → 放大（300 条）→ 加长训练（6000 epoch）→ 评估。
> 第一轮基线：9 种子 → 100 条生成 → BC 训练 2000 epoch → **成功率 1/20（5%）**（诊断结论：
> 训练未收敛 + 数据量不足，主因非 bug）。
>
> **为什么全录新 30 条（不合并旧 10 条）**：
> 1. 旧 10 条是第一轮“学习期”录的（730~3451 步、中位 1346，含拖沓/犹豫样本）——BC 会把
>    “犹豫风格”一起学走；全新 30 条按统一纪律录制，风格一致、质量更高；
> 2. 场景多样性不损失：积木位置每条 reset 自动随机，旧数据并无刻意的多样性优势；
> 3. 记录干净：旧文件原样保留作对照，新管线统一 `r2` 命名，避免编号/来源混淆；
> 4. `merge_teleop_datasets.py` 依然有用：**分多次录制时**（如 15+15）用它合并。

所有命令均在 `~/code/isaac/isaaclab` 目录下执行；数据集统一存放 `~/code/isaac/IsaacDataCollection/datasets/`（2026-10-14 由 `isaaclab/datasets/` 迁入），下文命令一律用绝对路径。

---

## 0. 全景流程与耗时

| 步骤 | 做什么 | 预计耗时 |
| --- | --- | --- |
| ① 录制 | 手柄**全新录 30 条**（建议分 2~3 次坐） | 1.5~2 h（人工）|
| ② 质检 | `teleop_dataset_stats.py` 快速筛查 | 秒级 |
| ③（可选）合并 | 仅分多批录制时：a+b → `dataset_gamepad.hdf5` | 秒级 |
| ④ 标注 | annotate（30 条逐条回放）| ~75-90 min |
| ⑤ 放大 | generate 目标 **300 条**（隔夜跑）| **~10-11 h** |
| ⑥ 训练 | BC-RNN **6000 epoch** | ~2.5-3 h |
| ⑦ 评估 | play **horizon 1600** × 20 回放 | ~15-20 min |

---

## 1. 录制第二轮的 30 条（全新）

### 录制纪律（第一轮教训 —— 策略会学走你所有的"犹豫"）

1. **果断**：贴近积木后尽快闭爪，不要长时间悬停、来回微调（第一轮策略学到了"贴着积木犹豫 200+ 步不闭爪"）；
2. **短**：单条目标 **≤900 步**（第一轮中位 971 偏长）；
3. **顺序固定**：蓝底 → 红中 → 绿顶（成功判定要求，见手册 §4.2）；
4. **利用场景随机**：每条 reset 后积木位置自动随机；可以主动尝试不同接近方向；
5. 不触发成功判定不算数，按 R 重开当前条（保护窗口内）；
6. 录满 `--num_demos 30` 自动退出并保存；
7. **建议分 2~3 次坐录制**（每次 10~15 条）：精力集中时质量更稳；每次录制换新文件名，最后用合并工具合并（§3）。

### 命令

```bash
cd ~/code/isaac/isaaclab
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --teleop_device gamepad --visualizer kit \
  --num_demos 30 --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5
```

> ⚠️ **每次都用一个新文件名**：`teleop record` 对同名输出文件是**覆盖写**（recorder 以 `"w"` 模式打开）。
> 一次坐录完 30 条就写 `dataset_gamepad.hdf5`；分多次录则每次写独立文件
> （`dataset_gamepad_a.hdf5`、`dataset_gamepad_b.hdf5`…），最后合并（§3）。第一轮旧文件已于 2026-10-14 清理。

---

## 2. 质检（不启动 Isaac）

```bash
.venv/bin/python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5
```

看什么：

- `success` 应全为 True（失败条不会被导出）；
- `steps` 有无 >1200 的拖沓条（可考虑重录）；
- `jerk`（动作抖动）异常偏大的条；
- 期望 30 条齐全。

---

## 3.（可选）合并 —— 仅当分多批录制时

一次坐录完 30 条则**跳过本步**；分多次录制（如 a + b 两个文件）时先合并：

```bash
.venv/bin/python ~/code/isaac/IsaacDataCollection/scripts/merge_teleop_datasets.py \
  ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5 \
  ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad_a.hdf5 ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad_b.hdf5
```

期望输出：`共 30 条演示, ... success=30/30`。

> 合并 = 逐条原样复制（含 obs/states/属性）并重新编号 `demo_0..29`，工具已做 10+10 自测（逐元素
> 一致性验证）。合并放在 annotate 之前：标注只跑一次、种子池一次建齐。

---

## 4. 标注（annotate，逐条回放 + 自动子任务标注）

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --visualizer kit --auto \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/annotated_gamepad.hdf5
```

- 期望：终端打印 `Exported X (out of 30) annotated episodes.`（X ≈ 27-30；个别条回放终判
  不过被跳过属正常，如第一轮的 9/10）；
- 耗时 ≈2.5 min/条 → 30 条约 **75-90 分钟**；
- 输出**重新编号**（被跳过的条不留空号），后续引用注意。

---

## 5. 放大生成（目标 300 条，建议隔夜）

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --visualizer kit --num_envs 64 --generation_num_trials 300 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_gamepad.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_gamepad_300.hdf5
```

- **语义**：`generation_guarantee=True`（默认）→ "**攒够 300 条成功**"才退出；终端统计行
  `X/Y (Z%)` 就是实时生成成功率（参考：上轮 100 条用时 ~3.5h）；
- **耗时 ~10-11 h** → 建议晚上启动。数据文件随每次成功**增量重写**，若意外中断，已写出
  的部分仍可直接用于训练（想续跑需重跑命令，会**重建**文件）；
- `--num_envs 64` 是上一轮 100 条生成的实际配置（已验证稳定；如显存吃紧降到 32）；
- 观察：成功率 ≥30% 为好；<20% 说明种子质量有问题（回 §4.3 纪律重录）。

---

## 6. 训练（加长到 6000 epoch）

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel --algo bc \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_gamepad_300.hdf5 \
  --name bc_rnn_stack_r2 \
  --epochs 6000
```

- 耗时估算：2000 epoch ≈ 50 min → **6000 epoch ≈ 2.5-3 h**；checkpoint 每 100 epoch 一只
  （`model_epoch_100.pth ... model_epoch_6000.pth`）；
- `--name bc_rnn_stack_r2` 用新实验名（避免与旧实验目录的覆盖询问）；
- 看什么：Loss（NLL）持续下降 —— 上一轮训到 2000 epoch 仍未收敛，这正是加长训练的理由；
- 日志目录：`logs/robomimic/IsaacContrib-Stack-Cube-Franka-IK-Rel/bc_rnn_stack_r2/<时间戳>/`。

---

## 7. 评估（固定 horizon 1600）

```bash
# 先找到 checkpoint 时间戳目录（或从训练终端打印的路径复制）
ls -td logs/robomimic/IsaacContrib-Stack-Cube-Franka-IK-Rel/bc_rnn_stack_r2/*/ | head -1

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --checkpoint logs/robomimic/IsaacContrib-Stack-Cube-Franka-IK-Rel/bc_rnn_stack_r2/<时间戳>/models/model_epoch_6000.pth \
  --num_rollouts 20 --horizon 1600 --visualizer none
```

- ⚠️ **不要再用默认 horizon 800**：本数据 44% 演示长于 800 步（中位 971，≤1600 覆盖 96%）；
- 对照基线：第一轮 **1/20（5%）**；目标 ≥ 4-6/20 起步；
- 想围观过程：把 `--visualizer none` 换成 `--visualizer kit`；
- 如需看中间 checkpoint 的差异：对 `model_epoch_3000/4500/6000` 各跑一遍（每个 ~15 min）。

---

## 8. 结果判读与决策环

| 观察 | 判断 | 行动 |
| --- | --- | --- |
| 生成成功率 ≥30% | 种子结构良好 | 继续 |
| 生成成功率 <20% | 种子质量差 | 回 §1 纪律重录个别条后重跑 |
| BC 成功率 ≥30% | 数据链路有效 | 欢迎；可再加数据/更多 epoch 冲量 |
| BC 成功率 10~30% | 方向正确数据仍偏少 | 再录 20-30 条（r3 管线）→ 生成更大规模（500+）|
| BC 成功率 <10% | 需要更大改变 | 检查评估 horizon / 再录更干净种子 / 试 8000-10000 epoch |

---

## 附：本轮涉及的新工具

| 工具 | 用途 |
| --- | --- |
| `IsaacDataCollection/scripts/merge_teleop_datasets.py` | 合并多批数据集（逐条复制 + 重编号，已验证零丢失）|
| `IsaacDataCollection/scripts/teleop_dataset_stats.py` | 数据集统计与质量筛查（不启动 Isaac）|
| `IsaacDataCollection/scripts/bc_eval_diagnose.py` | BC 策略回放诊断（接近距离/叠放高度/夹爪状态逐段打印）|
| `IsaacDataCollection/scripts/isaaclab_teleop_launch.py` | 手柄录制/热身的启动器（注册 `-Teleop` 任务变体）|

## 附：文件命名约定

```
~/code/isaac/IsaacDataCollection/datasets/
  dataset_gamepad.hdf5           # 手柄原始录制（30 条；分批录时先 a/b 再合并）
  annotated_gamepad.hdf5         # 标注后（29 条）
  generated_gamepad_300.hdf5     # 放大生成后（300 条）
  dataset_keyboard.hdf5          # （另一路）键盘演示 3 条；统计在 stats_keyboard.json
```

> 命名归一（2026-10-14）：去除 `_r2` 后缀；旧归档 `20261012/` 已删除（键盘数据移回本目录）。
