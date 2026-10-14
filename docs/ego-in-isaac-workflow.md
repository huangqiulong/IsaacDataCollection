# EGO 式数据完整工作流：在 Isaac 中模拟采集 → 数据处理 → 训练使用

> **本文定位**：以真机 EGO（第一人称/自我中心视角）数据为标尺，在 Isaac 中用
> **G1 人形机器人头戴相机（robot_pov_cam）+ 移动操作数据生成（locomanipulation SDG）**
> 完整复刻"采集 → 处理 → 使用"链路。
> 命令均来自源码核对（标 ⚠️ 处为高成本/未实测提示）。
>
> **配套文档**：
> - 全局背景：[robot-data-collection-in-isaac-survey.md](./robot-data-collection-in-isaac-survey.md)（§5）
> - HDF5 格式详解：[hdf5-format-guide.md](./hdf5-format-guide.md)
> - Mimic 生成原理：[simulation-data-in-isaac-workflow.md](./simulation-data-in-isaac-workflow.md)

---

## 0. 前置

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets
```

**链条总览**（约 30~60 分钟，含下载与两次小规模生成）：

```
① 下载 G1 预标注数据集(5MB)  →  ② Mimic 生成"操作段"数据(种子扩充)
   →  ③ SDG 生成"移动+操作"数据(含 robot_pov_cam 第一人称帧)
   →  ④ 导出第一人称视频 / 轨迹可视化 / 转 LeRobot-GR00T 格式
   →  ⑤ 使用(视频分析 / GR00T 微调 / 表征学习)
```

---

## 1. 原理：EGO 是什么 → Isaac 里的对应物

### 1.1 真机 EGO（第一人称示范）

- 用**头戴设备**（Project Aria 眼镜、GoPro、头显+手套）从操作者视角录制"人眼看着手操作"的视频
  （EgoMimic、DexCap、Ego4D 等范式）；
- 核心价值：**视角 = 操作者的注意力焦点**，天然适合学习"视觉-动作耦合"；
- 常配合"视频→手部姿态估计→重定向"得到机器人动作。

### 1.2 Isaac 对应物：机器人身上的第一人称相机

| 场景 | 相机 | 位置 | 分辨率 |
| --- | --- | --- | --- |
| G1 遥操作（人在环） | `RobotHeadCam` | 头部（`torso_link/head_link`）| 640×480 |
| **SDG 数据生成（本文主角）** | `robot_pov_cam` | 由 SDG 环境动态添加 | **160×256（默认）/ 320×512（`--high_res_video`）** |

- **`--sensor_camera_view`**：生成过程中把 Sim 视口直接切到机器人第一人称视角——**所见即所录**；
- 与真机 EGO 差异：仿真相机固定在机器人头上（姿态稳定），人类 EGO 随头部晃动（更"野"）。

### 1.3 为什么选 G1 移动操作任务

`Isaac-G1-SteeringWheel-Locomanipulation` 是官方 locomanipulation SDG 旗舰示例：
G1 在仓库场景里（含货架叉车、箱子、两张桌子）先**移动**再**完成操作**，
整条轨迹带第一人称视频——这就是"EGO 式移动操作数据"的仿真版本。

---

## 2. 模拟采集

### 2.1 步骤 ①：下载官方预标注数据集（~5MB）

```bash
# URL 已验证存在（HTTP 200）；S3 直连较慢，几分钟
wget -c -P ./datasets \
  "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_g1_locomanip.hdf5"
ls -la ~/code/isaac/IsaacDataCollection/datasets/dataset_annotated_g1_locomanip.hdf5
```

### 2.2 步骤 ②：Mimic 生成"操作段"数据（种子扩充）

预标注集是**少量人类演示**；先用 Mimic 把它扩成小批量操作数据（后续 SDG 的种子）：

```bash
# 小规模先跑通（20 条）；8GB 卡用小并发
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --num_envs 8 --generation_num_trials 20 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanip.hdf5
```

> ⚠️ 若 CUDA 下出现不稳定（G1 场景较重），可改 `--device cpu`（稳妥但慢很多）。
> 任务名从数据集的 `env_args` 自动解析（`Isaac-Locomanipulation-G1-Abs-Mimic-v0`）。
> 终端会打印生成成功统计——这是数据质量的第一信号。

### 2.3 步骤 ③：SDG 生成"移动+操作"数据（含第一人称帧）⭐

```bash
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
  --device cuda \
  --kit_args="--enable isaacsim.replicator.experimental.mobility_gen" \
  --task="Isaac-G1-SteeringWheel-Locomanipulation" \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanip.hdf5 \
  --num_runs 1 --lift_step 60 --navigate_step 130 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --sensor_camera_view --visualizer kit
# 高分辨率第一人称帧（320×512，默认 160×256）：
#   在上面命令加 --high_res_video
```

关键参数（源码核对）：

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `--task` | SDG 任务（已注册：`Isaac-G1-SteeringWheel-Locomanipulation`）| 固定 |
| `--dataset` | 输入种子数据集（步骤 ② 产物，"静态操作段"）| 固定 |
| `--lift_step` / `--navigate_step` | 搬起 / 导航 的步数切分（示例值）| 60 / 130 起调 |
| `--num_runs` | 生成多少条轨迹 | 1 条先验证 |
| `--sensor_camera_view` | 视口切到机器人第一人称 | 强烈建议开（可视化）|
| `--high_res_video` | 320×512（默认 160×256）| 内容检查用高清 |
| `--randomize_placement` | 物体/起点随机化 | 提多样性 |
| `--background_usd_path` | 换背景场景（仓库→其它）| 想换环境时用 |

> ⚠️ 该脚本渲染相机 + 运行导航，**耗时与显存都高于纯操作生成**；先 `--num_runs 1` 验证。

### 2.4 检查产物

```bash
HDF5_USE_FILE_LOCKING=FALSE .venv/bin/python - <<'PY'
import h5py
f = h5py.File('/home/dillon/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5', 'r')
print('顶层:', list(f.keys()))
demo = f['data/demo_0']
print('demo_0 组:', list(demo.keys())[:12])
obs = demo['obs'] if 'obs' in demo else None
if obs is not None:
    print('obs 键:', [k for k in obs.keys()][:20])
    if 'robot_pov_cam' in obs:
        a = obs['robot_pov_cam']
        print(f"robot_pov_cam: shape={a.shape} dtype={a.dtype}")  # 期望 (N,160,256,3) uint8
f.close()
PY
```

**判定标准**：存在 `robot_pov_cam` 图像数据集（形状 `(N,160,256,3)`，`--high_res_video` 时为 `(N,320,512,3)`），
N = 演示步数。同时会有导航/物体状态等标量键。

---

## 3. 数据处理

### 3.1 数据结构（与 Mimic 数据集的差异）

| 部分 | 内容 |
| --- | --- |
| `data/demo_i/obs/robot_pov_cam` | **(N,H,W,3) uint8 第一人称视频帧**（本文主角）|
| `data/demo_i/obs/*` | 导航速度、关节状态、物体位姿等低维观测 |
| `data/demo_i/actions` / `states` | 高层动作（腕部目标等）与全量状态 |
| 其它 | SDG 专属键（场景/障碍/路径元数据）——用上节脚本全量列出 |

### 3.2 导出第一人称视频（把"ego 数据"看起来）

```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_videos \
  --input_keys robot_pov_cam \
  --video_width 256 --video_height 160
# --high_res_video 录的用：--video_width 512 --video_height 320

ls ~/code/isaac/IsaacDataCollection/datasets/ego_videos/   # robot_pov_cam_demo_0.mp4
```

打开视频，你会看到"**机器人眼中**"的完整过程：先走位、再到桌边、伸右手把物体搬去另一张桌——
这就是 ego 式数据的仿真版。

### 3.3 导航轨迹可视化（数据质检）

```bash
python scripts/imitation_learning/locomanipulation_sdg/plot_navigation_trajectory.py \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_plots
# 可选：--demo_filter demo_0
```

每条演示输出一张 PNG：底盘点轨迹、起始/目标、障碍物——检查"导航是否合理、有没有穿墙抄近路"。

### 3.4 转换到 LeRobot / GR00T 格式（下游训练入口）

官方把 SDG 数据转成 **LeRobot 格式**（视频 + parquet + meta，含 `modality.json`）的脚本：

```bash
python scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
  ./datasets ~/code/isaac/IsaacDataCollection/datasets/lerobot_g1_locomanip
```

- 输入：目录（批量转该目录下所有 `*.hdf5`）；输出：LeRobot 目录结构；
- 产物要点：`videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4`（**ego 视角视频**）、
  `data/` 下 parquet（state/action）、`meta/{info,episodes,tasks,modality}.json`；
- `modality.json` 定义了 state/action 的分段语义（GR00T 微调直接消费）。

> 本机转换纯 CPU，可直接运行（依赖 pandas/pyarrow/tqdm 等，venv 已备）。

---

## 4. 数据使用

### 4.1 路线 A：GR00T 策略微调（正经用途，⚠️ 本机不适用）

- `gr00t/` 目录提供配套工具链：`data_config.py`（数据配置）、`policy.py`（策略定义）、
  `rollout_policy.py`（部署/回放）、`convert_dataset.py`（格式转换）、`no_flash_attn.patch`（显存补丁）；
- GR00T 类视觉-语言-动作模型的训练需要 **大显存 GPU + 外部 Isaac-GR00T 环境**——
  本机 8GB 不适合训练；转换出的 LeRobot 数据集可在有条件的机器上直接用于微调；
- 若只想"看效果"：用 `rollout_policy.py` 在有 GPU 的环境里回放预训练权重。

### 4.2 路线 B：本机可行的用途（推荐学习路径）

| 用途 | 方法 | 产出 |
| --- | --- | --- |
| 数据形态理解 | 上节 h5py 检查 + 视频导出 | 看懂 ego 数据长什么样 |
| 导航质量分析 | `plot_navigation_trajectory.py` | 轨迹质量报告 |
| 视觉表征素材 | ego_view 视频可直接喂给视觉模型/编码器预训练流程（外部）| 表征/自监督实验 |
| 模仿学习直觉 | 用 state 版 `generated_dataset_g1_locomanip.hdf5` 小规模试训（任何 BC 实现）| 理解"移动+操作"联合动作空间 |

### 4.3 与真机 EGO 的对照

| 环节 | 真机 EGO | 本文仿真链路 |
| --- | --- | --- |
| 采集设备 | 头戴眼镜/相机（人动） | G1 头部固定相机（机器人动）|
| 轨迹落地 | 人手姿态估计→重定向 | 仿真物理直接执行 |
| 数据形态 | 视频(+手部姿态) | 视频 + 全量真值状态/动作 |
| 规模扩展 | 请更多人录（贵） | SDG 换随机种子批量生成 |
| 下游 | VLA/DP 类策略 | LeRobot+GR00T 格式（官方工具链）|

---

## 5. 与真机的差异 / 局限

- **视角物理**：机器人头部相机高度/视野与"人的眼睛"不同；真机 EGO 的头部晃动带来更多样但更抖的数据；
- **缺失环节对照**：真机 EGO 的关键难点"人手→机器人手重定向"在仿真中消失（机器人直接执行）——
  学习仿真链路有助于理解数据结构，但真机部署仍需处理重定向；
- **场景规模**：SDG 提供仓库类场景（叉车/箱子/双桌）；野外多样性仍是仿真短板；
- **分辨率**：160×256 默认偏低（训练可用、人看糊）；检查内容时用 `--high_res_video`。

---

## 6. 附录

### 6.1 命令速查

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets

# ① 下载预标注（5MB）
wget -c -P ./datasets "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_g1_locomanip.hdf5"

# ② Mimic 生成操作数据
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --num_envs 8 --generation_num_trials 20 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanip.hdf5

# ③ SDG 生成（含 ego 相机帧）
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
  --device cuda --kit_args="--enable isaacsim.replicator.experimental.mobility_gen" \
  --task="Isaac-G1-SteeringWheel-Locomanipulation" \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanip.hdf5 \
  --num_runs 1 --lift_step 60 --navigate_step 130 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --sensor_camera_view --visualizer kit

# ④ 处理：视频 / 轨迹图 / LeRobot 转换
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py --input_file <sdg.hdf5> \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_videos --input_keys robot_pov_cam --video_width 256 --video_height 160
python scripts/imitation_learning/locomanipulation_sdg/plot_navigation_trajectory.py \
  --input_file <sdg.hdf5> --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_plots
python scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
  ./datasets ~/code/isaac/IsaacDataCollection/datasets/lerobot_g1_locomanip
```

### 6.2 FAQ

| 现象 | 原因 / 解决 |
| --- | --- |
| SDG 生成报扩展缺失 | `--kit_args="--enable isaacsim.replicator.experimental.mobility_gen"` 未传或拼写错 |
| 生成太慢/爆显存 | 先 `--num_runs 1`；降分辨率（不加 `--high_res_video`）；必要时 `--device cpu` 稳妥慢跑 |
| obs 里没有 robot_pov_cam | `enable_cameras` 未开（不要传 `--disable_external_cameras`）；确认用的 SDG 脚本 |
| 视频导出全黑 | 用错 input_keys；先 h5py 打印 obs 键名对齐 |
| 转换脚本报缺 pandas | `uv pip install pandas pyarrow tqdm`（清华源）|
| GR00T 训练跑不动 | 本机 8GB 训练不可行——转换好的数据集拿到大显卡环境训练 |
