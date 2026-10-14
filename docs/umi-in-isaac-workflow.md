# UMI 式数据完整工作流：在 Isaac 中模拟采集 → 数据处理 → 训练使用

> **本文定位**：以真机 UMI（Universal Manipulation Interface）的数据形态为标尺，
> 在 Isaac Sim/Lab 中用 **Visuomotor 腕部相机任务** 完整复刻"采集 → 处理 → 使用"链路。
> 全部命令基于本机（RTX 4070 Laptop 8GB · Isaac Lab 3.0）实测或源码核对，未实测处已明确标注。
>
> **配套文档**：
> - 全局背景与五类采集对照：[robot-data-collection-in-isaac-survey.md](./robot-data-collection-in-isaac-survey.md)
> - 遥操作（录制原理与设备）：[teleoperation-hands-on-franka-stack-cube.md](./teleoperation-hands-on-franka-stack-cube.md)
> - HDF5 数据格式详解（可视化/时间对齐/四元数）：[hdf5-format-guide.md](./hdf5-format-guide.md)

---

## 0. 前置

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p datasets
```

（环境重建、venv 说明见综述 §0；本机资产已齐备。）

**任务**：`IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor` —— Franka 摞方块 + 双相机（腕部/桌面）。
**动作空间**：末端相对位姿增量（6D）+ 夹爪开合（1D）——与 UMI 的动作语义完全相同。

---

## 1. 原理：UMI 是什么 → Isaac 里的对应物

### 1.1 真机 UMI（三要素）

| 要素 | 真机实现 | 解决什么问题 |
| --- | --- | --- |
| 手持夹爪 | 人手拿一个带相机的二指夹爪在场景中演示 | 不需要机器人即可采数（便宜、场景自由） |
| 腕部同轴相机 | GoPro + 反光镜，视角与夹爪同轴 | 视角稳定，训练时直接对齐机器人腕部相机 |
| SLAM 后处理 | 用相机画面恢复夹爪 6DoF 轨迹 | 得到"相对轨迹"，与机器人末端控制天然对齐 |

- 输出数据 = **夹爪相机视频** + **夹爪位姿序列（6DoF + 开合宽度）**；
- 典型训练目标：Diffusion Policy / 视觉 BC 类策略。

### 1.2 Isaac 中的对应物（一一映射）

| UMI 概念 | Isaac Visuomotor 任务 | 备注 |
| --- | --- | --- |
| 腕部同轴相机 | `wrist_cam`（挂在 `Robot/panda_hand/wrist_cam`）| 200×200、24mm 焦距、rgb + depth |
| （附加）第三人称视角 | `table_cam`（桌边固定）| 真机 UMI 没有，可当"额外观测"或对照 |
| 夹爪相对位姿动作 | 动作 = Δpose(6) + gripper(1)，IK-Rel | 逐项同构 |
| SLAM 恢复位姿 | **不需要**——仿真直接给真值位姿 | 数据更干净 |
| 野外场景多样性 | 域随机化（材质/光照/物位）| 仿真侧补偿手段 |

### 1.3 本机相机参数（源码核对）

| 相机 | 挂载点 | 分辨率 | 内参 | 数据 |
| --- | --- | --- | --- | --- |
| `wrist_cam` | `{ENV}/Robot/panda_hand/wrist_cam`，偏移 (0.13, 0, −0.15) | 200×200 | focal 24mm，aperture 20.955 | rgb + distance_to_image_plane |
| `table_cam` | `{ENV}/table_cam`，偏移 (1.0, 0, 0.4) | 200×200 | 同上 | rgb + distance_to_image_plane |

> `image_obs_list = ["table_cam", "wrist_cam"]`——两个相机都进 policy 观测。

---

## 2. 模拟采集

### 2.1 录制前必须知道的 3 件事

1. **相机默认渲染**：`teleop record` 默认启用相机（不要加 `--disable_external_cameras`，否则相机被剥离，录出来没有图像）。
2. **数据集体积**：图像入库后文件明显变大——每步约 2×(200×200×3) ≈ 240KB 原始像素
   （实际落盘有压缩）。**先录 1 条验证，再批量**。
3. **键盘灵敏度与之前不同**：Visuomotor 任务没有内置 `teleop_devices` 配置，
   `record` 会回退到**内置键盘（灵敏度 0.2 / 0.5，比 §抬手写堆叠任务的 0.05 更快）**——
   点按要更轻、更短。

### 2.2 录制命令（先 1 条验证，再 3-5 条）

```bash
# ① 先录 1 条验证链路（约 1-2 分钟操作）
./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 1 \
  --dataset_file ./datasets/dataset_umi_1.hdf5

# ② 验证通过后补到 3~5 条（换文件名避免覆盖；只导出成功演示）
./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 5 \
  --dataset_file ./datasets/dataset_umi_5.hdf5
```

操作纪律与成功判定**与遥操作手册 §4 完全一致**（从下到上：蓝→红→绿；失败按 R；成功等 10 步保护窗口）。

### 2.3 产物检查（图像是否真的入库）

```bash
HDF5_USE_FILE_LOCKING=FALSE .venv/bin/python - <<'PY'
import h5py
f = h5py.File('datasets/dataset_umi_1.hdf5', 'r')
g = f['data/demo_0']
print('attrs:', dict(g.attrs))                    # success / num_samples
print('obs 键:', list(g['obs'].keys()))            # 期望含 table_cam / wrist_cam
for k in ('table_cam', 'wrist_cam'):
    if k in g['obs']:
        print(f'  {k}: {g["obs"][k].shape} {g["obs"][k].dtype}')  # 期望 (N,200,200,3) uint8
f.close()
PY
```

> **判定标准**：`obs` 组下出现 `table_cam`/`wrist_cam` 且形状为 `(N, 200, 200, 3)`，
> `success=True`。两者齐备才值得进入下一步；若图像缺失，检查是否误加了 `--disable_external_cameras`。

---

## 3. 数据处理

### 3.1 数据结构（与 state 版对照）

| 位置 | 内容 | 与 state 版（无相机任务）差异 |
| --- | --- | --- |
| `data/demo_i/actions` | (T, 7) 相对动作 | 相同 |
| `data/demo_i/obs/*` | 9 个低维键 + **2 个图像键** | **新增 table_cam / wrist_cam** |
| `data/demo_i/states` | 全场景状态 | 相同（真值，相当于"免费 SLAM 输出"） |
| `data/demo_i/initial_state` | 初始状态 | 相同 |
| attrs | `success / num_samples / seed` | 相同 |

字段语义"低维部分"详解见 `hdf5-format-guide.md` §3.3；图像键是 UMI 数据的主角。

### 3.2 导出视频（把 UMI 式数据"看"出来）

```bash
# 导出两个相机的视频（工具默认 keys 已含 table_cam/wrist_cam）
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file ./datasets/dataset_umi_5.hdf5 \
  --output_dir ./datasets/umi_videos \
  --input_keys table_cam wrist_cam \
  --video_width 200 --video_height 200

# 只看某一条：
#   --demo_id 0
ls ./datasets/umi_videos/
```

产物：`umi_videos/table_cam_demo_0.mp4`、`wrist_cam_demo_0.mp4` ——
**`wrist_cam` 视频就是"UMI 同轴视角数据"在仿真里的样子**：能从画面里直接看到夹爪与积木的接近过程。

### 3.3 统计与质检

```bash
# 数据集统计（对 image 键会自动跳过或只报形状；关注 success / 步数 / 夹爪）
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py \
  ./datasets/dataset_umi_5.hdf5

# 曲线可视化（动作/末端轨迹等；图像不画）
python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py \
  ./datasets/dataset_umi_5.hdf5 --demo demo_0 --out ./datasets/umi_plot_0.png
```

**质检要点**（视觉数据的额外门槛）：

- 腕部画面中**目标积木始终可见**（抓取接近段不能出画）；
- 曝光/反光：`distance_to_image_plane` 若也录了，可检查深度合理性；
- 图像与动作的**时间对齐**：obs 为 pre-step 语义（见 hdf5-format-guide §3.4），下游训练不要错位。

### 3.4 数据管理

- 录制文件命名建议：`dataset_umi_{条数}.hdf5`；
- 合并多个文件：`scripts/tools/merge_hdf5_datasets.py`（把多次录制的成功演示合并为一个大集，
  用于后续训练）；
- 体积估算：3 条 × 500 步 ≈ 数百 MB 量级；批量录制前先核对磁盘。

---

## 4. 数据使用

### 4.1 路线 A（主线）：robomimic 视觉 BC 训练

任务自带官方视觉 BC 配置 `bc_rnn_image_200.json`（RNN 视觉策略：batch_size=16、600 epochs、
输入 200×200——与本任务相机分辨率精确匹配）：

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --algo bc \
  --dataset ./datasets/dataset_umi_5.hdf5
```

- ⚠️ **8GB 显存注意**：本机视觉 BC 未实测；若 OOM，编辑/复制 `bc_rnn_image_200.json`
  调小 `batch_size`（16 → 8/4），或减少训练数据条数；
- 直接用 3-5 条录制数据训练**必然过拟合**——正式训练前建议先做 §4.2 的 Mimic 增广；
- 训练日志/权重在 `logs/robomimic/...`。

### 4.2 （推荐）Mimic 增广：3 条 → 上百条

```bash
# ① 标注（visuomotor 版 Mimic 任务）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --visualizer kit --auto \
  --input_file ./datasets/dataset_umi_5.hdf5 \
  --output_file ./datasets/annotated_umi.hdf5

# ② 生成（相机渲染使生成变慢；8GB 用小并发）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 8 --generation_num_trials 100 \
  --input_file ./datasets/annotated_umi.hdf5 \
  --output_file ./datasets/generated_umi_100.hdf5
```

> Mimic 全流程原理与参数细节见 [simulation-data-in-isaac-workflow.md](./simulation-data-in-isaac-workflow.md)。
> 注意：带相机的生成比 state 版**慢且更吃显存**，先 10 条试跑。

### 4.3 评估：策略回放

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit --num_rollouts 20 \
  --checkpoint <logs/robomimic/.../model_xxx.pth>
```

### 4.4 与真机 UMI 的闭环对照

| 环节 | 真机 UMI | 本文仿真链路 |
| --- | --- | --- |
| 采集 | 手持夹爪野外演示（美元级成本）| 键盘遥操作（0 成本、分钟级）|
| 位姿恢复 | SLAM 后处理（有噪声）| 真值 states（无噪声）|
| 数据增广 | 换场景重采 | Mimic 自动生成 ×N |
| 训练 | Diffusion Policy 等 | robomimic 视觉 BC（内置配置）|
| 部署 | 机器人执行 | play.py 回放成功率 |

---

## 5. 与真机的差异 / 局限

- **画面真实感**：仿真渲染缺少运动模糊、曝光变化、镜面反射——真机部署需域随机化/真实数据微调补偿；
- **视角多样性**：真机 UMI 手持晃动带来丰富视角；仿真腕部相机刚性固定（更易训练但更"窄"）；
- **场景多样性**：仿真靠随机化补，不能完全替代野外场景；
- **深度数据**：仿真免费提供 `distance_to_image_plane`，真机 UMI 没有——可只用 rgb 对齐真机做实验。

---

## 6. 附录

### 6.1 命令速查

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate

# 采集（1 条验证 → 5 条正式）
./isaaclab.sh teleop record --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit --teleop_device keyboard --num_demos 1 --dataset_file ./datasets/dataset_umi_1.hdf5

# 处理（导出视频 / 统计 / 曲线）
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py --input_file <hdf5> \
  --output_dir ./datasets/umi_videos --input_keys table_cam wrist_cam --video_width 200 --video_height 200
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py <hdf5>
python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py <hdf5> --demo demo_0

# 使用（增广 → 训练 → 评估）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 --visualizer kit --auto \
  --input_file <hdf5> --output_file ./datasets/annotated_umi.hdf5
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 8 --generation_num_trials 100 \
  --input_file ./datasets/annotated_umi.hdf5 --output_file ./datasets/generated_umi_100.hdf5
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor --algo bc --dataset <hdf5>
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor --visualizer kit --num_rollouts 20 --checkpoint <pth>
```

### 6.2 FAQ

| 现象 | 原因 / 解决 |
| --- | --- |
| obs 组没有图像键 | 误加 `--disable_external_cameras`；或录制时相机未渲染 |
| 数据集特别大 | 图像未压缩入库属正常；用 `merge_hdf5_datasets.py` 前先看体积 |
| 训练 OOM | 调小 `bc_rnn_image_200.json` 的 batch_size；减少数据 |
| 腕部画面看不到积木 | 抓取前对准；检查是否遮挡（桌子/机械臂自遮挡） |
| 生成阶段太慢 | 相机渲染开销；降 `--num_envs`，或先做 state 版 Mimic 验证管线 |
