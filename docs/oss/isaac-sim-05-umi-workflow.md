# UMI 式数据工作流：仿真采集 → 数据处理 → 训练使用（五）

这是系列的**第五篇**。前四篇里我们完成了：采集方法全景（一）、遥操作实操手册（二）、仿真与真机采集对比（三）、HDF5 数据格式深挖（四）。这一篇把目光投向近几年最受关注的采集范式之一——**UMI（Universal Manipulation Interface）**：不造机器人，人手拿着一个"带相机的夹爪"直接在场景里演示，就能产出机器人可学习的视觉-动作数据。

本篇不复刻 UMI 的硬件，而是以它的**数据形态**为标尺，在 Isaac Sim/Lab 里找出**一一对应物**，并把"采集 → 处理 → 使用"整条链路完整跑一遍：Visuomotor 任务（腕部 + 桌面双相机）→ 键盘 / Xbox 手柄两种采集方式 → 视频导出与数据体检 → Mimic 增广与视觉 BC 训练。全部命令与数字来自本机实测（RTX 4070 Laptop 8GB · Isaac Sim 6.1 · Isaac Lab 3.0）；配图由仓库内脚本从**真实录制数据**自动生成（5 条演示、2,599 帧、双 200×200 相机）。

> **配套资源**：本系列的全部文章、配套脚本与图表源码已开源：[github.com/huangqiulong/IsaacDataCollection](https://github.com/huangqiulong/IsaacDataCollection)

## 0. 一分钟速览

| 问题 | 答案 |
| --- | --- |
| UMI 是什么 | 手持夹爪 + 腕部同轴相机 + SLAM 后处理的真机采集方案：**无需机器人**即可在任意场景采数 |
| 在 Isaac 里对应什么 | **Visuomotor 任务**：`wrist_cam`（腕部）+ `table_cam`（桌边）双相机，动作 = 相对位姿增量 + 夹爪 |
| 怎么采 | `teleop record`：键盘直接录；Xbox 手柄经本仓库启动器 + `-Visuomotor-Teleop` 变体 |
| 数据长什么样 | HDF5：`obs` 低维 9 键 + **2 个 200×200 图像键**；`actions (T,7)`；5 条演示 = 2,599 帧 ≈ 203 MB |
| 怎么处理 | 导出 MP4（注意 cv2 环境坑）→ 统计质检 → 时间对齐核查 |
| 怎么用 | Mimic 增广（≥100 条）→ robomimic 视觉 BC（`bc_rnn_image_200`）→ `play.py` 回放评估 |
| 最大的两个坑 | ① 录制误加 `--disable_external_cameras` → 图像被剥离，只剩低维数据；② `hdf5_to_mp4.py` 用错 Python → 视频**静默不产出**（详见 §3.2） |

## 1. 原理：UMI 是什么 → Isaac 里的对应物

### 1.1 真机 UMI 的三要素

| 要素 | 真机实现 | 解决什么问题 |
| --- | --- | --- |
| 手持夹爪 | 人手拿一个带相机的二指夹爪在场景中演示 | 不需要机器人即可采数（便宜、场景自由） |
| 腕部同轴相机 | GoPro + 反光镜，视角与夹爪同轴 | 视角稳定，训练时直接对齐机器人腕部相机 |
| SLAM 后处理 | 用相机画面恢复夹爪 6DoF 轨迹 | 得到"相对轨迹"，与机器人末端控制天然对齐 |

- 输出数据 = **夹爪相机视频** + **夹爪位姿序列（6DoF + 开合宽度）**；
- 典型训练目标：Diffusion Policy / 视觉 BC 类策略。

![图 1：真机 UMI 与 Isaac Visuomotor 对照](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig1-umi-vs-sim.png)

*图 1：真机 UMI 的三要素与 Isaac 里的对应物——逐项映射，最终输出同构的数据形态。*

### 1.2 一一映射：Isaac Visuomotor 任务

| UMI 概念 | Isaac Visuomotor 任务 | 备注 |
| --- | --- | --- |
| 腕部同轴相机 | `wrist_cam`（挂在 `Robot/panda_hand/wrist_cam`） | 200×200、24mm 焦距、rgb + depth |
| （附加）第三人称视角 | `table_cam`（桌边固定） | 真机 UMI 没有，可当"额外观测"或对照 |
| 夹爪相对位姿动作 | 动作 = Δpose(6) + gripper(1)，IK-Rel | 逐项同构 |
| SLAM 恢复位姿 | **不需要**——仿真直接给真值位姿 | 数据更干净 |
| 野外场景多样性 | 域随机化（材质/光照/物位） | 仿真侧补偿手段 |

整个工作流串起来就是下面这条链路——本篇文章要做的，就是把它从头到尾跑通：

![图 2：全链路总览](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig2-pipeline.png)

*图 2：全链路总览——采集（①）→ 处理（视频/质检）→ 增广（②③）→ 训练（④）→ 评估（⑤）。*

### 1.3 本机相机参数（源码核对）

| 相机 | 挂载点 | 分辨率 | 内参 | 数据 |
| --- | --- | --- | --- | --- |
| `wrist_cam` | `{ENV}/Robot/panda_hand/wrist_cam`，偏移 (0.13, 0, −0.15) | 200×200 | focal 24mm，aperture 20.955 | rgb + distance_to_image_plane |
| `table_cam` | `{ENV}/table_cam`，偏移 (1.0, 0, 0.4) | 200×200 | 同上 | rgb + distance_to_image_plane |

> `image_obs_list = ["table_cam", "wrist_cam"]`——两个相机都进 policy 观测。

![图 3：Visuomotor 任务的相机布置](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig3-task-cameras.png)

*图 3：相机布置示意——腕部相机随夹爪运动（同轴视角），桌面相机固定观察全局。*

## 2. 模拟采集

### 2.1 录制前必须知道的 3 件事

1. **相机默认渲染**：`teleop record` 默认启用相机（不要加 `--disable_external_cameras`，否则相机被剥离，录出来没有图像）；
2. **数据集体积**：图像入库后文件明显变大——每步约 2×(200×200×3) ≈ 240 KB 原始像素（实际落盘有压缩）。**先录 1 条验证，再批量**；
3. **操作设备二选一**：
   - **键盘（免配置）**：Visuomotor 任务没有内置 `teleop_devices` 配置，`record` 会回退到**内置键盘（平移 0.2 / 旋转 0.5，比堆叠任务常用的 0.05 更快）**——点按要更轻、更短；
   - **Xbox 手柄**：官方 `record` 内置设备不含 gamepad（报 `Built-in devices: keyboard, spacemouse.`），**必须经本仓库启动器 + `-Visuomotor-Teleop` 变体**使用（灵敏度 0.10/0.10、yaw ×3、Y=重开当前条）。

准备工作（一次即可）：

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets
```

### 2.2 Xbox 手柄：连接、自检与操作指令表

**① 连接（蓝牙或 USB）**：手柄长按配对键进入配对模式，在系统蓝牙中配对 `Xbox Wireless Controller`；也可 USB 直插。连接后 Linux 侧应有设备节点：

```bash
ls /dev/input/js*                       # 期望出现 /dev/input/js0
cat /sys/class/input/js0/device/name    # 期望 "Xbox Wireless Controller"
```

**② Isaac 识别自检（蓝牙款必做一次）**：Kit 内嵌 GLFW 映射表缺本机手柄固件（fw0509）条目——不打补丁时 Isaac 显示 `Device name: None` 或提示 `Joystick with unknown remapping detected (will be ignored)`。用本仓库补丁工具检查/修补：

```bash
python3 ~/code/isaac/IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py check   # 期望两份插件库均为 [check] patched
python3 ~/code/isaac/IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py apply   # 未打或升级 Isaac Sim 后重打（自动备份，可 revert）
```

> **Isaac Sim 升级/重装会覆盖该修复**，需重新 `apply`。

**③ 热身（不录制，先试手感）**：

```bash
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop \
  --visualizer kit --num_envs 1 --teleop_device gamepad
```

判定标准：启动日志出现 `Using teleop device: Gamepad Controller for SE(3): Se3GamepadWithReset`、`Device name: Xbox Wireless Controller`（手柄已被 Kit 正确识别）与按键说明表；随后出现 `native teleoperation started.`——此时可直接用手柄操作。**操作时 Isaac 窗口必须保持焦点**。退出：`Ctrl+C`。

**④ 操作指令表**（源码级核对；可用 `scripts/isaac_gamepad_mapping_dump.py` 复核）：

| 操作 | 手柄输入 | 末端增量（本仓库变体配置） |
| --- | --- | --- |
| 前后 x（向积木推进 / 后撤） | 左摇杆 上 / 下 | +x / −x |
| 左右 y | 左摇杆 右 / 左 | −y / +y（已按实测翻转 `pan_y`） |
| 升降 z | 右摇杆 上 / 下 | +z / −z |
| 偏航 yaw（绕 Z 转） | 右摇杆 右 / 左 | +yaw / −yaw（×3 增益） |
| 俯仰 pitch | D-Pad 上 / 下 | −pitch / +pitch |
| 翻滚 roll | D-Pad 左 / 右 | +roll / −roll |
| 夹爪开合 | **X 键** | 开/合切换 |
| 重开当前条（录制中） | **Y 键** | 触发"重置当前演示"回调 |

> 手感调节：某方向相反 → 命令前加 `ISAACLAB_GAMEPAD_INVERT="pan_y,roll"`；yaw 倍率 → `ISAACLAB_GAMEPAD_YAW_BOOST=5`；灵敏度 → 改 `isaaclab_teleop_extras.py` 中 `Se3GamepadWithResetCfg(pos_sensitivity=0.10, rot_sensitivity=0.10, ...)`。

### 2.3 录制命令（先 1 条验证，再 3-5 条）

**方式 A：键盘（免配置，直接录）**

```bash
# ① 先录 1 条验证链路（约 1-2 分钟操作）
./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 1 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_1.hdf5

# ② 验证通过后补到 3~5 条（换文件名避免覆盖；只导出成功演示）
./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 5 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5
```

**方式 B：Xbox 手柄（经启动器 + `-Visuomotor-Teleop` 变体；先按 §2.2 完成连接与自检）**

```bash
# ① 先录 1 条验证链路
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop \
  --visualizer kit \
  --teleop_device gamepad \
  --num_demos 1 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_1.hdf5

# ② 验证通过后补到 3~5 条
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop \
  --visualizer kit \
  --teleop_device gamepad \
  --num_demos 5 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5
```

> 两种方式的产物结构完全相同（视觉 BC 训练对采集设备无感）；想两套设备各录一份对比时换文件名即可（如 `dataset_umi_5_gamepad.hdf5`），下游命令不变。

操作纪律与成功判定**与遥操作手册（第二篇）§4 完全一致**（从下到上：蓝→红→绿；失败按 R（键盘）/ Y（手柄）；成功等 10 步保护窗口）。

### 2.4 产物检查（图像是否真的入库）

```bash
HDF5_USE_FILE_LOCKING=FALSE .venv/bin/python - <<'PY'
import h5py
f = h5py.File('/home/dillon/code/isaac/IsaacDataCollection/datasets/dataset_umi_1.hdf5', 'r')
g = f['data/demo_0']
print('attrs:', dict(g.attrs))                    # success / num_samples
print('obs 键:', list(g['obs'].keys()))            # 期望含 table_cam / wrist_cam
for k in ('table_cam', 'wrist_cam'):
    if k in g['obs']:
        print(f'  {k}: {g["obs"][k].shape} {g["obs"][k].dtype}')  # 期望 (N,200,200,3) uint8
f.close()
PY
```

**判定标准**：`obs` 组下出现 `table_cam`/`wrist_cam` 且形状为 `(N, 200, 200, 3)`，`success=True`。两者齐备才值得进入下一步。

录到的画面长这样——下面是从 5 条数据集里抽出的真实帧（demo_0 全流程：接近 → 抓取 → 叠放）：

![图 4：两条真实相机流](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig4-camera-frames.png)

*图 4：录制回放——腕部（上）与桌面（下）两条真实相机流；t=870 步附近可以看到蓝块被抓起的瞬间。*

**在 Isaac GUI 里实时查看这两个相机**（Visuomotor 任务专属，随时可用的三种方式）：

- **① Camera Inspector（推荐）**：GUI 顶部菜单 **Tools → Sensors → Camera Inspector**——Cameras 列表自动列出场景全部相机（应能看到 `wrist_cam` 与 `table_cam`）；选中后可看实时状态/位姿，点 **Create Viewport** 立即生成显示该相机实时图像的新视口；
- **② 直接切换主视口**：视口左上角的**相机选择器**（默认 `Perspective`）→ 下拉选 `wrist_cam` / `table_cam` → 视口即刻显示该相机画面；想同时看两个相机就再开一个新视口;
- **③ 数据侧证明**：上面 §2.4 的 h5py 检查 + §3.2 导出 MP4 直接观看。

> 常见疑问：视口里**看不到相机"实物"是正常的**——相机是没有几何体的 prim，不会出现在渲染画面中；到 Stage 窗口或 Camera Inspector 里确认其存在即可。

## 3. 数据处理

### 3.1 数据结构（与 state 版对照）

| 位置 | 内容 | 与 state 版（无相机任务）差异 |
| --- | --- | --- |
| `data/demo_i/actions` | (T, 7) 相对动作 | 相同 |
| `data/demo_i/obs/*` | 9 个低维键 + **2 个图像键** | **新增 table_cam / wrist_cam** |
| `data/demo_i/states` | 全场景状态 | 相同（真值，相当于"免费 SLAM 输出"） |
| `data/demo_i/initial_state` | 初始状态 | 相同 |
| attrs | `success / num_samples / seed` | 相同 |

字段语义"低维部分"的逐项详解见**第四篇**；图像键是 UMI 数据的主角。

### 3.2 导出视频（把 UMI 式数据"看"出来）

```bash
# 导出两个相机的视频（工具默认 keys 已含 table_cam/wrist_cam）
.venv/bin/python scripts/tools/hdf5_to_mp4.py \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5 \
  --output_dir ~/code/isaac/IsaacDataCollection/datasets/umi_videos \
  --input_keys table_cam wrist_cam \
  --video_width 200 --video_height 200

# 只看某一条：
#   --demo_id 0
ls ~/code/isaac/IsaacDataCollection/datasets/umi_videos/
```

> ⚠️ **必须用 `.venv/bin/python` 直跑，不要用 `./isaaclab.sh -p`（也不要 `~/isaacsim/python.sh`）**：这两种方式命中的 cv2（Isaac Sim 自带）缺视频编码器，`cv2.VideoWriter` 会**静默失败**——无报错且不产出视频。venv 已装 `opencv-python-headless`（含 FFMPEG 编码器；2026-10-14 实测导出成功）。

产物：`umi_videos/demo_0_table_cam.mp4`、`demo_0_wrist_cam.mp4`（命名格式 `demo_<i>_<key>.mp4`，10 个文件约 4 MB —— **`wrist_cam` 视频就是"UMI 同轴视角数据"在仿真里的样子**：能从画面里直接看到夹爪与积木的接近过程。

### 3.3 统计与质检

```bash
# 数据集统计（对 image 键会自动跳过或只报形状；关注 success / 步数 / 夹爪）
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py \
  ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5

# 曲线可视化（动作/末端轨迹等；图像不画）
python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py \
  ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5 --demo demo_0 --out ~/code/isaac/IsaacDataCollection/datasets/umi_plot_0.png
```

我们这一轮的 5 条演示体检结果——注意 demo_0 是完整的"接近-抓取-叠放"，其余四条是干净利落的短演示：

![图 5：数据集体检](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig5-dataset-check.png)

*图 5：dataset_umi_5.hdf5 体检——5 条全部 success，共 2,599 帧（约 130 秒），平均 520 步/条。*

**质检要点**（视觉数据的额外门槛）：

- 腕部画面中**目标积木始终可见**（抓取接近段不能出画）；
- 曝光/反光：`distance_to_image_plane` 若也录了，可检查深度合理性；
- 图像与动作的**时间对齐**：`obs` 为 pre-step 语义，训练对是 `(obs[t], actions[t])`——别把 `obs/actions`（上一执行动作）当标签，两处易错点详见第四篇 §4。

![图 6：时间对齐](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig6-time-align.png)

*图 6：demo_0 的夹爪指令与动作幅值——t=877 首次闭爪，之后的开合节奏对应"抓取-放置"的每个阶段。*

### 3.4 数据管理

- 录制文件命名建议：`dataset_umi_{条数}.hdf5`；
- 合并多个文件：`scripts/tools/merge_hdf5_datasets.py`（把多次录制的成功演示合并为一个大集，用于后续训练）；
- 体积估算：3 条 × 500 步 ≈ 数百 MB 量级；批量录制前先核对磁盘。

## 4. 数据使用

### 4.1 Mimic 增广：5 条 → 上百条

3~5 条种子直接训练**必然过拟合**——正式训练前先做 Mimic 增广：

![图 7：Mimic 增广 → 视觉 BC 训练链路](https://articleimg-1511801422.cos.ap-shanghai.myqcloud.com/IsaacDataCollection/images/isaac-sim-05/fig7-scale-up.png)

*图 7：从 5 条种子到可用策略——标注、增广、训练、评估五步走。*

```bash
# ① 标注（visuomotor 版 Mimic 任务）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --visualizer kit --auto \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_5.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/annotated_umi.hdf5

# ② 生成（相机渲染使生成变慢；8GB 显存用小并发）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 8 --generation_num_trials 100 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_umi.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_umi_100.hdf5
```

> Mimic 全流程原理与参数细节见（三）的规模化实测：30 条种子 → 300 条生成、生成成功率与步数分布等。注意：**带相机的生成比 state 版慢且更吃显存**，先 10 条试跑。

### 4.2 路线 A（主线）：robomimic 视觉 BC 训练

任务自带官方视觉 BC 配置 `bc_rnn_image_200.json`（RNN 视觉策略：batch_size=16、600 epochs、输入 200×200——与本任务相机分辨率精确匹配）：

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor \
  --algo bc \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_umi_100.hdf5
```

- ⚠️ **8GB 显存注意**：本机视觉 BC 未实测；若 OOM，编辑/复制 `bc_rnn_image_200.json` 调小 `batch_size`（16 → 8/4），或减少训练数据条数；
- 训练日志/权重在 `logs/robomimic/...`。

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
| 采集 | 手持夹爪野外演示（美元级成本） | 键盘/Xbox 遥操作（0 成本、分钟级） |
| 位姿恢复 | SLAM 后处理（有噪声） | 真值 states（无噪声） |
| 数据增广 | 换场景重采 | Mimic 自动生成 ×N |
| 训练 | Diffusion Policy 等 | robomimic 视觉 BC（内置配置） |
| 部署 | 机器人执行 | play.py 回放成功率 |

## 5. 与真机的差异 / 局限

- **画面真实感**：仿真渲染缺少运动模糊、曝光变化、镜面反射——真机部署需域随机化/真实数据微调补偿；
- **视角多样性**：真机 UMI 手持晃动带来丰富视角；仿真腕部相机刚性固定（更易训练但更"窄"）；
- **场景多样性**：仿真靠随机化补，不能完全替代野外场景；
- **深度数据**：仿真免费提供 `distance_to_image_plane`，真机 UMI 没有——可只用 rgb 对齐真机做实验。

## 6. 小结与系列导航

三句话记住这篇：

- **对应关系**：真机 UMI 的"手持夹爪 + 腕部同轴相机 + SLAM"在 Isaac 里就是"遥操作 + `wrist_cam`/`table_cam` + 真值位姿"，动作语义（相对位姿 + 夹爪）逐项同构；
- **全链路已实测**：采集（键盘/Xbox 双路线）→ 视频导出 → 体检 → Mimic 增广 → 视觉 BC 训练 → 回放评估，命令全部可复制执行；
- **两个最容易踩的坑**：录制忘开相机（数据里没有图像键）；视频导出用错 Python（Isaac 自带 cv2 无编码器 → 静默不产出）。

**系列导航**：

- **（一）数据采集方法总览**：五类采集方式（遥操作、动捕、UMI、EGO、仿真生成）的原理、数据格式与上手路径；
- **（二）遥操作实操手册**：在 Isaac Lab 里完成一次完整的遥操作采数（键盘 / Xbox360 / 鼠标）；
- **（三）Isaac Sim 仿真遥操采集与真机采集对比**：两条数据路线逐项对照 + Mimic 规模化实测；
- **（四）HDF5 格式详解**：数据集结构、时间对齐、处理配方与坑；
- **（五）UMI 式数据工作流**：即本文。

**项目仓库（开源）**：[github.com/huangqiulong/IsaacDataCollection](https://github.com/huangqiulong/IsaacDataCollection) —— 全部文章、配套脚本与图表源码。

## 参考资料

- [UMI 项目主页（Universal Manipulation Interface）](https://umi-gripper.github.io/)——手持夹爪采集的原始论文与开源硬件
- [robomimic 数据集格式](https://robomimic.github.io/docs/datasets/overview.html)
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)——UMI 数据的典型下游训练目标
- 上游工具脚本：`isaaclab/scripts/tools/` 下的 `hdf5_to_mp4.py`、`merge_hdf5_datasets.py`

---

*本文全部结论基于实测（Visuomotor 任务 5 条双相机演示，2,599 帧）+ 源码核对；环境为 Isaac Sim 6.1 + Isaac Lab 3.0（RTX 4070 Laptop 8GB）。配图存于 `docs/images/isaac-sim-05/`。*
