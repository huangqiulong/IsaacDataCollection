# EGO 式数据完整工作流：在 Isaac 中模拟采集 → 数据处理 → 训练使用

> **本文定位**：以真机 EGO（第一人称/自我中心视角）数据为标尺，在 Isaac 中用
> **G1 人形机器人头戴相机（robot_pov_cam）+ 移动操作数据生成（locomanipulation SDG）**
> 完整复刻"采集 → 处理 → 使用"链路。
> 命令均来自源码核对（标 ⚠️ 处为高成本/未实测提示）。
>
> **配套文档**：
>
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
- 常配合"视频 → 手部姿态估计 → 重定向"得到机器人动作——这三步的真实做法是整条链路的核心，详见 §1.2。

### 1.2 详解：视频 → 手部姿态估计 → 重定向（真机链路）

> 本节为真机路线**背景梳理**（非本机实测；以两条代表路线为参照：[EgoMimic](https://egomimic.github.io/)（纯视觉）/ [DexCap](https://dex-cap.github.io/)（便携动捕））。放在这里是为了回答：真机 EGO 数据到底怎么从"视频"变成"机器人动作"？仿真链路又省掉了其中哪些环节？

录制时只拿到"视频 + 设备的粗略自身轨迹"；"手在干什么"是**离线后处理**算出来的，主线是四级流水：

```
① SLAM：恢复设备轨迹 T_world_cam(t)
   ② 手部姿态估计：视频 → 手腕 6DoF + 手指关键点（相机坐标系）
      ③ 坐标变换：乘出世界系手轨迹（头部晃动被抵消）
         ④ 重定向：人手 → 机器人（末端+夹爪 / 灵巧手关节）
            ⑤ 动作生成：Δpose(6) + gripper(1) 相对动作
```

#### 环节 A：恢复设备自身的运动（SLAM / VIO）

- **为什么需要**：手部姿态是在"相机坐标系"里给出的，而相机挂在头上随人到处走——不先解出相机轨迹，"手的位置"没有固定参考系；
- **怎么做**：视觉惯性里程计（VIO/SLAM）。Project Aria 这类专用设备由配套的机器感知服务（MPS）离线输出；GoPro 等纯视觉素材常用 ORB-SLAM3 / VINS-Mono 等开源方案补做；
- **输出**：逐帧相机位姿 `T_world_cam(t)`（世界 ← 相机）；
- **坑**：单目 SLAM 存在**尺度不确定性**——轨迹整体可能缩放大/小（表现为"手漂浮、够不到"），需用 IMU 或已知尺寸物体恢复公制尺度；跟踪丢失帧要插值。

#### 环节 B：逐帧估计手的姿态（"视频 → 手"在这里发生）

两条技术路线并存：

**路线 ①：纯视觉（EgoMimic 路线）**

1. **2D 关键点检测**：每帧找出 21 个手部关键点（手腕 1 + 每指 4 个），轻量代表是 MediaPipe Hands；
2. **3D 提升 / 手模型拟合**：把 2D 关键点回归成 **MANO 手参数**（21 个关节点 + 778 顶点网格；每手 45 维姿态 + 10 维形状 + 根节点平移旋转），代表方法：HaMeR、WiLoR、PyMAF 等；
3. **输出**：每帧手腕 6DoF 位姿 `T_cam_hand(t)` + 手指关节位置（均为相机坐标系）。

> 专用设备把这两步封装成现成服务（Aria 的 MPS 直接输出设备轨迹与手部关键点）——这是 EgoMimic "戴上眼镜就能采数据"的前提。

**路线 ②：动捕硬件（DexCap 路线）——不猜、直接量**

- 便携动捕套件：胸前相机架（RGB-D LiDAR 相机拍环境点云 + SLAM 追踪相机），采集时把追踪相机**移到手背**（量手腕/手掌轨迹），手指动作由**电磁场（EMF）动捕手套**测量（指相对掌心的 3D 位置）；
- 优势：**抗遮挡**——双手交叠或被物体挡住时视觉方案会丢手，EMF 手套照常输出（DexCap 专门对比过 VR 头显视觉追踪在遮挡场景下的失效）。

#### 环节 C：把手"挂"回世界坐标系

坐标链做一次乘法：

```
T_world_hand(t) = T_world_cam(t) · T_cam_hand(t)
```

- 得到**世界系下的手轨迹**——头部晃动被抵消，手相对房间的位置稳定下来；
- **时间戳要统一到同一时钟域**（设备时钟 vs 相机帧时钟），否则快速动作段出现"手比画面慢一帧"的系统性错位；
- 到这里，"人类演示"才变成与机器人数据同构的**轨迹 + 观测**。

#### 环节 D：重定向（人手 → 机器人）

人手 20+ 自由度 vs 机器人形态各异，这层"翻译"按目标形态分两条路线：

**路线 ①：平行夹爪末端（EGO→机械臂；输出与遥操作数据同构）**

| 人的量                      | 映射到             | 做法                                                  |
| --------------------------- | ------------------ | ----------------------------------------------------- |
| 手腕 6DoF 位姿              | 机械臂末端目标位姿 | "手眼"标定修正坐标系后作为 IK 目标（逆解到关节）      |
| 拇指尖-食指尖距离（捏合量） | 夹爪开度           | 距离 → [0, 最大开度] 的线性/阈值映射（抓取段二值化） |
| 手掌朝向（可选）            | 末端姿态           | 直接用腕部旋转，注意手坐标系定义差异                  |

- 核心是**动作语义对齐**：人数据与机器人数据共享同一动作空间（末端位姿 + 夹爪），才谈得上"平等混训"；
- 与本文仿真动作空间一致：Δpose(6) + gripper(1)。

**路线 ②：灵巧手全关节（EGO→多指手）**

- 目标：手腕 6DoF + 手指关键点 → 多指机器人手（如 LEAP 手，16 自由度）的 16 维关节角；
- DexCap 的做法：**指尖 IK（fingertip IK）**——以指尖位置为 IK 目标解出 16 维关节位置；
- 数学本质是一个优化问题：

```
min_q  Σ_i ||p_robot,i(q) − p_human,i||^2  +  正则项
```

其中：

1. `q` = 机器人手关节角（LEAP 为 16 维）
2. `p_robot,i(q)` = 第 i 个指尖的正向运动学位置
3. `p_human,i` = 对应人体指尖位置（可加比例缩放 s 与偏置 b 修正）
4. 正则项 = 关节限位、时间平滑、避免自碰撞

- 两大类实现：**位置重定向**（指尖 IK，如上）与**向量重定向**（按"掌心→指尖"等关键向量对齐映射，数值更稳）；开源参考：DexPilot 一脉的 dex-retargeting / AnyTeleop；
- 坑：人手灵活度通常超过机器人手，超限动作要截断；关节语义差异需人工建映射表。

#### 另一半：观测对齐（混训前必需）

动作能翻译还不够——"人视角的画面"与"机器人相机的画面"不是一回事：

| 技巧       | 实例                                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------- |
| 同机位观测 | DexCap：相机架**快拆**（<20 秒换装）——机器人直接戴"人身上那台"观测相机，观测天然同分布 |
| 视觉差修补 | DexCap：正向运动学把机器人手渲染成点云补进观测；EgoMimic：掩码图像弱化人手/机器手外观差        |
| 补充视角   | EgoMimic：策略额外输入机器人腕部相机画面                                                       |

> **回报（EgoMimic 实测）**：2 小时机器人数据 + 1 小时人手数据 > 3 小时纯机器人数据；"1 小时人手数据"的边际价值显著高于"1 小时机器人数据"——前提正是上面的动作重定向 + 观测对齐。

#### 工程细节清单（交给训练之前）

1. **时间同步与重采样**：统一到固定控制频率（如 20~30 Hz）；
2. **滤波**：一欧元滤波/低通去抖，但**别过度平滑**——抓取瞬间最怕被抹平；
3. **工作空间与安全**：限幅到机器人可达空间，加速度上限；
4. **动作转换**：绝对轨迹 → 每步**相对动作**，并与观测错开一步（obs 语义），对齐 HDF5 的 `(obs_t, action_t)`（见 [hdf5-format-guide.md](./hdf5-format-guide.md)）；
5. **质检**：手轨迹重投影回画面核对；"闭爪帧"与"物体被拿起帧"对齐检查。

#### 常见坑速查

| 现象                        | 可能原因                                   |
| --------------------------- | ------------------------------------------ |
| 手"漂浮"/穿模               | 单目 SLAM 尺度漂移、手部估计深度偏差       |
| 抓取时刻错位                | 时间戳不同步、滤波过度                     |
| 手腕朝向整体偏转（如 90°） | 手腕坐标系定义（掌心系 vs 腕关节系）未对齐 |
| 动作超出机器人可达          | 未做工作空间限幅 / 比例缩放                |
| 快速动作段轨迹乱跳          | 遮挡导致估计跳变、缺时序平滑               |

#### 与本文仿真链路的对照

| 真机环节             | 仿真里为什么不需要                                               |
| -------------------- | ---------------------------------------------------------------- |
| A：SLAM 恢复设备轨迹 | 仿真给真值——相机/机器人位姿就在数据集 `states` 里            |
| B：手部姿态估计      | 动作由遥操作设备直接生成（手柄/键盘/鼠标）——不需要"从视频猜手" |
| C：坐标变换          | 动作本来就定义在世界系/机器人基座系                              |
| D：重定向            | 遥操作动作空间本身就是机器人控制空间，天然同构                   |

> 仿真链路的"干净"不是巧合，而是把 A/B/D 中最难、最易出错的环节**整体绕开**。反过来，真机 EGO 数据要接入本文这条训练链路（HDF5 / LeRobot），必须在外部先做完上面四级流水——Isaac 只消费"已重定向好"的成品。（全身人形的重定向见 [mocap-in-isaac-workflow.md](./mocap-in-isaac-workflow.md)。）

### 1.3 Isaac 对应物：机器人身上的第一人称相机

| 场景                               | 相机              | 位置                             | 分辨率                                                       |
| ---------------------------------- | ----------------- | -------------------------------- | ------------------------------------------------------------ |
| G1 遥操作（人在环）                | `RobotHeadCam`  | 头部（`torso_link/head_link`） | 640×480                                                     |
| **SDG 数据生成（本文主角）** | `robot_pov_cam` | 由 SDG 环境动态添加              | **160×256（默认）/ 320×512（`--high_res_video`）** |

- **`--sensor_camera_view`**：生成过程中把 Sim 视口直接切到机器人第一人称视角——**所见即所录**；
- 与真机 EGO 差异：仿真相机固定在机器人头上（姿态稳定），人类 EGO 随头部晃动（更"野"）。

### 1.4 为什么选 G1 移动操作任务

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
# 小规模先跑通（20 条）；8GB 卡用 4 并发（实测 8 并发会 OOM，见下方注）
HDF5_USE_FILE_LOCKING=FALSE ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --num_envs 4 --generation_num_trials 20 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanip.hdf5
```

> 🧯 **显存不足（2026-10-14 实测定位）**：8GB 显卡（RTX 4070 Laptop）上 `--num_envs 8` 会报
> `torch.OutOfMemoryError: CUDA out of memory`（进程占 6.4GiB、仅剩 ~55MiB）——该场景**每个环境都渲染**
> `robot_pov_cam`（480×640），显存开销远大于纯状态 Mimic（Franka 那种）。稳定配置：
> `--num_envs 4`（可选再加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防碎片）。
> 实测结果：36 次尝试达到 20 条成功演示、共 4790 步、输出约 1.5GB（大头是相机帧）。

> ⚠️ 若 CUDA 下出现不稳定（G1 场景较重），可改 `--device cpu`（稳妥但慢很多）。
> 任务名从数据集的 `env_args` 自动解析（`Isaac-Locomanipulation-G1-Abs-Mimic-v0`）。
> 终端会打印生成成功统计——这是数据质量的第一信号。

> 🧯 **常见报错（2026-10-14 实测定位）**：若报 `ModuleNotFoundError: ... 'pinocchio.pinocchio_pywrap_default'`
> 或 `ImportError: ... undefined symbol: ...ADMMSolverResult...`——是**终端继承了 ROS Humble 环境**所致
> （`~/.bashrc` 自动 `source /opt/ros/humble/setup.bash`，其 Python 3.10 库路径污染了 venv 的 pinocchio）。
> **重开一个 VS Code 工作区终端即可**（`ops/scripts/isaaclab_terminal_rc.sh` 已自动剔除 `/opt/ros/*`）；
> 老终端先执行下面两行清理，再重跑生成命令：

```bash
export PYTHONPATH=$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd: -)
export LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd: -)
```

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

| 参数                                  | 含义                                                            | 建议                 |
| ------------------------------------- | --------------------------------------------------------------- | -------------------- |
| `--task`                            | SDG 任务（已注册：`Isaac-G1-SteeringWheel-Locomanipulation`） | 固定                 |
| `--dataset`                         | 输入种子数据集（步骤 ② 产物，"静态操作段"）                    | 固定                 |
| `--lift_step` / `--navigate_step` | 搬起 / 导航 的步数切分（示例值）                                | 60 / 130 起调        |
| `--num_runs`                        | 生成多少条轨迹                                                  | 1 条先验证           |
| `--sensor_camera_view`              | 视口切到机器人第一人称                                          | 强烈建议开（可视化） |
| `--high_res_video`                  | 320×512（默认 160×256）                                       | 内容检查用高清       |
| `--randomize_placement`             | 物体/起点随机化                                                 | 提多样性             |
| `--background_usd_path`             | 换背景场景（仓库→其它）                                        | 想换环境时用         |

> ⚠️ 该脚本渲染相机 + 运行导航，**耗时与显存都高于纯操作生成**；先 `--num_runs 1` 验证。
> 实测（2026-10-14，RTX 4070 Laptop 8GB / `--device cpu`）：**约 2~3 分钟生成 1 条**（含导航段，873 帧），首跑即成功；
> 输出文件 39MB，`robot_pov_cam` 为 `(873,160,256,3) uint8`。

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

| 部分                                 | 内容                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `data/demo_i/obs/robot_pov_cam`    | **(N,H,W,3) uint8 第一人称视频帧**（本文主角）     |
| `data/demo_i/obs/*`                | 导航速度、关节状态、物体位姿等低维观测                   |
| `data/demo_i/actions` / `states` | 高层动作（腕部目标等）与全量状态                         |
| 其它                                 | SDG 专属键（场景/障碍/路径元数据）——用上节脚本全量列出 |

### 3.2 导出第一人称视频（把"ego 数据"看起来）

```bash
.venv/bin/python scripts/tools/hdf5_to_mp4.py \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_videos \
  --input_keys robot_pov_cam \
  --video_width 256 --video_height 160
# --high_res_video 录的用：--video_width 512 --video_height 320

ls ~/code/isaac/IsaacDataCollection/datasets/ego_videos/   # demo_0_robot_pov_cam.mp4（命名规则:demo_{i}_{相机键}.mp4）
```

> ⚠️ 若报 `FileNotFoundError`：说明 `--input_file` 指向的 SDG 数据集还没生成——
> 它是 **§2.3 步骤③** 的产物（不是步骤②的 Mimic 文件）。先跑步骤③再执行本步。

打开视频，你会看到“**机器人眼中**”的完整过程：到桌边抓起方向盘 → 双手持握走位（导航段）→ 到目标点放下——
这就是 ego 式数据的仿真版。（2026-10-14 实测：873 帧 / 30fps / 约 29 秒）

### 3.3 导航轨迹可视化（数据质检）

```bash
python scripts/imitation_learning/locomanipulation_sdg/plot_navigation_trajectory.py \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_plots
# 可选：--demo_filter demo_0
```

每条演示输出一张 PNG：底盘点轨迹、起始/目标、障碍物——检查"导航是否合理、有没有穿墙抄近路"。

### 3.4 转换到 LeRobot / GR00T 格式（下游训练入口）

官方把 SDG 数据转成 **LeRobot 格式**（视频 + parquet + meta，含 `modality.json`）的脚本。
⚠️ 输入目录要**只放 SDG 文件**（脚本对该目录 `*.hdf5` 全量扫描，混入非 SDG 文件会报错）——先建专用目录：

```bash
mkdir -p ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg
ln -sf ../generated_dataset_g1_locomanipulation_sdg.hdf5 ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg/

.venv/bin/python scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
  ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg \
  ~/code/isaac/IsaacDataCollection/datasets/lerobot_g1_locomanip
```

- 输入：目录（扫描 `*.hdf5`；仅转换物体位移 ≥2m 的演示——纯“原地操作”会被跳过）；输出：LeRobot 目录结构；
- 产物要点：`videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4`（**ego 视角视频**）、
  `data/` 下 parquet（state/action）、`meta/{info,episodes,tasks,modality}.json`；
- `modality.json` 定义了 state/action 的分段语义（GR00T 微调直接消费）。

> 本机转换纯 CPU，秒级完成；依赖 pandas/pyarrow/tqdm（2026-10-15 起 venv 已装齐：pandas 2.3.3 / pyarrow 23.0.1 / tqdm 4.70.1）。
> 实测（2026-10-15）：873 帧演示转出 2.3MB 产物（parquet 873 行 × state 49 维/action 32 维 + 4 个 meta 文件 + 20fps ego 视频）。

---

## 4. 数据使用

### 4.1 路线 A：GR00T 策略微调（正经用途，⚠️ 本机不适用）

- `gr00t/` 目录提供配套工具链：`data_config.py`（数据配置）、`policy.py`（策略定义）、
  `rollout_policy.py`（部署/回放）、`convert_dataset.py`（格式转换）、`no_flash_attn.patch`（显存补丁）；
- GR00T 类视觉-语言-动作模型的训练需要 **大显存 GPU + 外部 Isaac-GR00T 环境**——
  本机 8GB 不适合训练；转换出的 LeRobot 数据集可在有条件的机器上直接用于微调；
- 若只想"看效果"：用 `rollout_policy.py` 在有 GPU 的环境里回放预训练权重。

### 4.2 路线 B：本机可行的用途（推荐学习路径）

| 用途         | 方法                                                                           | 产出                        |
| ------------ | ------------------------------------------------------------------------------ | --------------------------- |
| 数据形态理解 | 上节 h5py 检查 + 视频导出                                                      | 看懂 ego 数据长什么样       |
| 导航质量分析 | `plot_navigation_trajectory.py`                                              | 轨迹质量报告                |
| 视觉表征素材 | ego_view 视频可直接喂给视觉模型/编码器预训练流程（外部）                       | 表征/自监督实验             |
| 模仿学习直觉 | 用 state 版 `generated_dataset_g1_locomanip.hdf5` 小规模试训（任何 BC 实现） | 理解"移动+操作"联合动作空间 |

### 4.3 与真机 EGO 的对照

| 环节     | 真机 EGO              | 本文仿真链路                     |
| -------- | --------------------- | -------------------------------- |
| 采集设备 | 头戴眼镜/相机（人动） | G1 头部固定相机（机器人动）      |
| 轨迹落地 | 人手姿态估计→重定向  | 仿真物理直接执行                 |
| 数据形态 | 视频(+手部姿态)       | 视频 + 全量真值状态/动作         |
| 规模扩展 | 请更多人录（贵）      | SDG 换随机种子批量生成           |
| 下游     | VLA/DP 类策略         | LeRobot+GR00T 格式（官方工具链） |

---

## 5. 与真机的差异 / 局限

- **视角物理**：机器人头部相机高度/视野与"人的眼睛"不同；真机 EGO 的头部晃动带来更多样但更抖的数据；
- **缺失环节对照**：真机 EGO 的关键难点"人手→机器人手重定向"在仿真中消失（机器人直接执行）——
  学习仿真链路有助于理解数据结构，但真机部署仍需处理重定向（三步细节见 §1.2）；
- **场景规模**：SDG 提供仓库类场景（叉车/箱子/双桌）；野外多样性仍是仿真短板；
- **分辨率**：160×256 默认偏低（训练可用、人看糊）；检查内容时用 `--high_res_video`。

---

## 6. 附录

### 6.1 命令速查

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets

# ① 下载预标注（5MB）
wget -c -P ./datasets "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_g1_locomanip.hdf5"

# ② Mimic 生成操作数据（8GB 卡用 4 并发；若报 pinocchio 相关错误：开新的工作区终端，或见 §2.2 注清理 ROS 环境变量）
HDF5_USE_FILE_LOCKING=FALSE ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --num_envs 4 --generation_num_trials 20 \
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
.venv/bin/python scripts/tools/hdf5_to_mp4.py --input_file <sdg.hdf5> \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_videos --input_keys robot_pov_cam --video_width 256 --video_height 160
python scripts/imitation_learning/locomanipulation_sdg/plot_navigation_trajectory.py \
  --input_file <sdg.hdf5> --output_dir ~/code/isaac/IsaacDataCollection/datasets/ego_plots
# LeRobot 转换（输入目录只放 SDG 文件；见 §3.4）
mkdir -p ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg && \
ln -sf ../generated_dataset_g1_locomanipulation_sdg.hdf5 ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg/
.venv/bin/python scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
  ~/code/isaac/IsaacDataCollection/datasets/locomanip_sdg \
  ~/code/isaac/IsaacDataCollection/datasets/lerobot_g1_locomanip
```

### 6.2 FAQ

| 现象                                                                             | 原因 / 解决                                                                                                                                                             |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SDG 生成报扩展缺失                                                               | `--kit_args="--enable isaacsim.replicator.experimental.mobility_gen"` 未传或拼写错                                                                                    |
| 生成太慢/爆显存                                                                  | 先 `--num_runs 1`；降分辨率（不加 `--high_res_video`）；必要时 `--device cpu` 稳妥慢跑                                                                            |
| Mimic 生成 `torch.OutOfMemoryError`                                            | G1 场景每环境渲染 480×640 相机，8GB 卡用 `--num_envs 4`（8 会爆）；可加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；仍不够则降 2 或 `--device cpu`       |
| 报 `pinocchio_pywrap_default` 缺失 / `undefined symbol: ADMMSolverResult...` | 终端继承了 ROS Humble 环境（`~/.bashrc` 自动 source）污染 venv 的 pinocchio——开新的工作区终端，或清理 PYTHONPATH/LD_LIBRARY_PATH 里的 `/opt/ros/*`（见 §2.2 注） |
| obs 里没有 robot_pov_cam                                                         | `enable_cameras` 未开（不要传 `--disable_external_cameras`）；确认用的 SDG 脚本                                                                                     |
| 视频导出全黑                                                                     | 用错 input_keys；先 h5py 打印 obs 键名对齐                                                                                                                              |
| 转换脚本报缺 pandas                                                              | `uv pip install pandas pyarrow tqdm`（清华源）；转换报 KeyError=输入目录混入非 SDG 文件（须按 §3.4 用专用目录）                                                                                                                        |
| GR00T 训练跑不动                                                                 | 本机 8GB 训练不可行——转换好的数据集拿到大显卡环境训练                                                                                                                 |
| 真机 EGO 的"视频→手部姿态估计→重定向"怎么做？                                  | 见 §1.2：SLAM 恢复设备轨迹 → 手部姿态估计（纯视觉 / 动捕手套）→ 世界系坐标变换 → 重定向（平行夹爪：手腕+捏合量；灵巧手：指尖 IK）                                   |
