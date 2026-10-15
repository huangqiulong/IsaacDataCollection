# 遥操作实操手册：Franka 摞方块（键盘 / Xbox360 / 鼠标）

> **任务**：在 Isaac Sim / Isaac Lab 3.0 中，用三种输入设备遥操作 Franka Panda 完成"三块积木叠放"，
> 录制演示数据集，并完成「数据处理 → 质量评估 → 下游使用验证」的完整闭环。
> **环境**：本机 RTX 4070 Laptop 8GB / Isaac Sim 6.1 / Isaac Lab 3.0.0（2026-10-09 整理）。
> **配套脚本**（工作区 `IsaacDataCollection/scripts/`，不修改上游 isaaclab 仓库）：
>
> | 脚本                          | 用途                                                         |
> | ----------------------------- | ------------------------------------------------------------ |
> | `isaaclab_teleop_extras.py` | 鼠标设备、手柄重置键、gamepad/mouse 任务变体（供启动器导入） |
> | `isaaclab_teleop_launch.py` | 启动器：注册变体后透传运行官方 run / record / replay 脚本    |
> | `teleop_dataset_stats.py`   | 数据集统计与质量检查（h5py + numpy）                         |

---

## 0. 前置

### 0.1 环境（重要）

当前 `isaaclab/.venv` 为空壳，需先重建（约 8.5GB，一次性）：

```bash
cd ~/code/isaac/isaaclab
rm -rf .venv
uv venv --python "$PWD/_isaac_sim/kit/python/bin/python3" .venv
source .venv/bin/activate
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
./isaaclab.sh -i
```

> 本文所有命令默认：**先 `cd ~/code/isaac/isaaclab && source .venv/bin/activate`**。
> （VS Code 工作区已配置**新开终端自动进入该 venv**：`.vscode/settings.json` + `ops/scripts/isaaclab_terminal_rc.sh`；其他终端请手动激活。）
> 数据集统一存放 `~/code/isaac/IsaacDataCollection/datasets/`（2026-10-14 起从 `isaaclab/datasets/` 迁入），下文命令中出现的 `datasets/...` 均指该目录。

### 0.2 开始前检查清单

- [ ] `nvidia-smi` 无其他计算进程（一次只跑一个 Isaac 进程）；
- [ ] 根分区剩余空间 > 5GB（数据集很小，但日志/渲染缓存会增长）；
- [ ] 有图形界面（GUI 必须，无窗口无法遥操作）：`--visualizer kit`；
- [ ] 手柄/鼠标已插好（若用），Isaac 窗口保持**焦点**（键盘/手柄/鼠标事件都发给 Kit 窗口）。

---

## 1. 选型：为什么是 Franka Panda + 摞方块（Stack Cube）

**机器人**：Franka Emika Panda（7 自由度机械臂 + 平行夹爪），桌面固定基座——Isaac Lab 生态里配套最完整的操作臂。

**任务**：`IsaacContrib-Stack-Cube-Franka-IK-Rel` —— 桌上蓝/红/绿三块积木，叠成塔（**从下到上：蓝 → 红 → 绿**，即红叠蓝、绿叠红；蓝块无需移动）；终判据 `cubes_stacked`（三块叠稳 + 夹爪松开）。

**为什么适合"遥操作第一课"**：

| 理由                 | 说明                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 动作语义通用         | 动作 =**末端相对位姿增量 + 夹爪开合**（Differential IK，相对模式）——与 UMI 数据、Mimic 生成、robomimic 训练全链路对齐 |
| 成功判据明确         | 三块叠好即成功，程序自动判定 → 录制"只保存成功演示"自动生效                                                                  |
| 单条演示短           | 熟练后 10~30 秒/条；新手 1~3 分钟/条                                                                                         |
| 无相机（state 模式） | 渲染负担极小，8GB 显存轻松；带相机的 Visuomotor 版本留到 UMI 一节                                                             |
| 下游原生支持         | 该任务是官方 Isaac Lab Mimic 的旗舰 demo：自动子任务标注（--auto）、生成、BC 训练一键贯通                                     |
| 资产已就绪           | 本机资产包已包含 Franka/桌面/积木，无需下载                                                                                   |

> ⚠️ **2026-10-10 实例**：此前资产包曾缺 `Isaac/Props/Blocks` 整个目录（积木 USD+贴图），运行即报 `blue_block.usd not found`；现已补齐并验证通过。换机器/重装环境后如遇同类报错，用 `python3 ops/scripts/isaac_assets_s3_diff.py --prefix "Assets/Isaac/6.1/Isaac/Props/Blocks" --download --workers 16` 自检补齐。

备选任务（以后再说）：`IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor`（带腕部/桌面相机，属 UMI 一节）。

---

## 2. 设备总览与快速热身（teleop run）

`teleop run` = 只操作、不录制（热身）。三种设备先各玩 10 分钟熟悉手感。

### 2.1 键盘（内置，零依赖，推荐先用）

```bash
./isaaclab.sh teleop run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_envs 1 \
  --teleop_device keyboard
```

**键位表**（`Se3Keyboard`，相对位姿增量控制）：

| 键              | 作用              | 键              | 作用                 |
| --------------- | ----------------- | --------------- | -------------------- |
| **W / S** | 末端 x 前后       | **Z / X** | 绕 x 旋转（roll）    |
| **A / D** | 末端 y 左右       | **T / G** | 绕 y 旋转（pitch）   |
| **Q / E** | 末端 z 上下       | **C / V** | 绕 z 旋转（yaw）     |
| **K**     | 开/合夹爪（切换） | **L**     | 复位（清空当前增量） |

> 手法要点：**点按 = 小步微调**（≈2~3cm/步），按住 = 持续快速移动（容易撞桌）。接近积木时用点按。
> 灵敏度由任务配置给出（pos/rot = 0.05/0.05），比脚本内置默认值（0.2/0.5）细腻，适合桌面小范围操作。

**yaw（C/V）转得慢？——用增强版键盘（2026-10-12 新增）**：

官方键盘绕 Z 轴（C/V）增量与其他旋转一致（0.05 rad/次 ≈2.9°/次），实操常感偏慢。
经启动器使用 `-Teleop` 变体时键盘自动换成 **`Se3KeyboardYawBoost`**：**C/V 灵敏度 ×3**（≈8.6°/次），
其余键不变：

```bash
# 热身（增强键盘）
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --visualizer kit --num_envs 1 --teleop_device keyboard
```

- 调倍率：命令前加 `ISAACLAB_KEYBOARD_YAW_BOOST=5`（任意正数）；
- 验证生效：启动日志打印 `(yaw boost active: x3 on C/V)`；
- 注意：官方直跑命令（`./isaaclab.sh teleop ...`）**不带**增强（用任务自带键盘配置）。

### 2.2 Xbox360 手柄

`teleop run` 官方脚本内置 gamepad 支持，可直接用：

```bash
./isaaclab.sh teleop run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_envs 1 \
  --teleop_device gamepad
```

（若此命令报 "no matching entry exists in env_cfg.teleop_devices"——见 §3.2 用启动器 + 变体任务。）

**按键映射**（源码级导出，可用 `IsaacDataCollection/scripts/isaac_gamepad_mapping_dump.py` 复核）：

| 操作                     | 手柄输入       | 末端增量（默认配置）                         |
| ------------------------ | -------------- | -------------------------------------------- |
| 前后（x）                | 左摇杆 上 / 下 | +x / −x                                     |
| 左右（y）                | 左摇杆 右 / 左 | −y / +y（**已按实测反馈翻转**，见下） |
| 升降（z）                | 右摇杆 上 / 下 | +z / −z                                     |
| 偏航（yaw）              | 右摇杆 右 / 左 | +yaw / −yaw                                 |
| 俯仰（pitch）            | D-Pad 上 / 下  | −pitch / +pitch                             |
| 翻滚（roll）             | D-Pad 左 / 右  | +roll / −roll                               |
| 夹爪                     | **X 键** | 开/合切换                                    |
| 重置（扩展：仅录制变体） | **Y 键** | 重开当前演示（见 §3.2）                     |

> **轴方向翻转（2026-10-12 新增）**：任一方向手感相反时，用环境变量一键翻转（逗号分隔）：
> `ISAACLAB_GAMEPAD_INVERT="pan_y,roll" ./isaaclab.sh -p <启动器> run ...`。
> 可用轴名：`pan_x`（前后）`pan_y`（左右）`pan_z`（升降）`yaw` `pitch` `roll`；
> **默认 `pan_y`**（左摇杆左右；因实测反馈左右反向而翻转）。
>
> **yaw 速度增益（2026-10-12 新增）**：绕 Z 轴旋转（右摇杆左右）默认 **×3**（与键盘的
> `ISAACLAB_KEYBOARD_YAW_BOOST` 对称）；调整倍率：
> `ISAACLAB_GAMEPAD_YAW_BOOST=5 ./isaaclab.sh -p <启动器> run ...`。

Linux 检查：插上 USB（或蓝牙配对）后 `ls /dev/input/js*` 应出现设备节点；手柄是否被 Kit 识别可看启动时终端打印的 `Device name:`。

> ✅ **2026-10-12 已修复蓝牙 Xbox 手柄无法识别的问题**（本机实测）：此前蓝牙手柄（"Xbox Wireless Controller"）
> 在 Kit 里显示 `Device name: None`——根因是内嵌 GLFW 映射表缺该固件版的 GUID；现已用
> `IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py` 修补（含备份，可 revert）。
> **注意：Isaac Sim 升级/重装会覆盖该修复，需重新 `python3 IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py apply`。**
> （Xbox 360 手柄不支持蓝牙，需 USB 有线或专用 2.4G 接收器；蓝牙款实为 Xbox One 世代手柄。）

### 2.3 鼠标

鼠标有**两种玩法**：

**方式 A（零代码，立即可用）——鼠标管视角，机器人用键盘/手柄控制**：

- 环绕视角 = 右键拖拽；平移 = 中键拖拽；缩放 = 滚轮；聚焦选中物体 = `F`（Kit 视口默认，如与实机不符以其提示为准）。
- 也就是说"鼠标 + 键盘"组合：左手键盘控机器人，右手鼠标随时调整观察角度——这是日常实操最常用的配置。

**方式 B（进阶）——纯鼠标控制机器人**（本工作区自定义设备 `Se3Mouse`，需启动器）：

```bash
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --visualizer kit --num_envs 1 \
  --teleop_device mouse
```

| 操作        | 鼠标动作                                       |
| ----------- | ---------------------------------------------- |
| XY 平移     | **按住左键拖动**（屏幕上=+x，屏幕右=+y） |
| pitch / yaw | **按住右键拖动**                         |
| roll        | 按住中键拖动                                   |
| Z 升降      | 滚轮                                           |
| 夹爪        | 键盘 K                                         |
| 重置        | 键盘 R（录制时）                               |

> ⚠️ `Se3Mouse` 为工作区自研（官方无鼠标设备），**尚未上机实测**（环境重建后可跑）。
> 若拖动无反应：确认 Isaac 窗口有焦点、按住键再拖；若方向不合直觉：编辑
> `IsaacDataCollection/scripts/isaaclab_teleop_extras.py` 的 `_on_mouse_event`，对相应系数取负即可。

### 2.4 选择建议

| 设备    | 上手速度         | 精度               | 适合                                             |
| ------- | ---------------- | ------------------ | ------------------------------------------------ |
| 键盘    | ★★★（立刻）   | 中（离散步进）     | 首练、快速小批录制                               |
| Xbox360 | ★★（插上即用） | 中高（连续模拟量） | 流畅录制主力（需启动器）                         |
| 鼠标    | ★（需自研脚本） | 中                 | 只有鼠标/触控板可用时；或"鼠标视角+键盘操作"组合 |

---

## 3. 三设备操作详解

### 3.1 键盘（内置）

- **操作**：见 §2.1 键位表。接近 → 微调对准 → K 合爪 → 抬起 → 平移 → 放下 → K 开爪 → 叠下一块。
- **录制时的额外按键**：
  - `R` = 重开当前演示（比如积木碰倒了）：环境重置，从头开始；
  - 成功且稳定 `--num_success_steps`（默认 10）个仿真步后：自动判定成功 → **自动导出该条演示** → 自动重置开始下一条；
  - 成功已触发后的短暂窗口内按 R 会被忽略（终端有提示，属保护逻辑，等自动重置即可）。
- **只导出成功演示**：失败/半途重开的尝试不会写进数据集——记录 10 条成功演示即可，无需担心废数据。

**热身**与§2.1 相同；**录制**命令：

```bash
./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 10 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
```

### 3.2 Xbox360 手柄

**为什么录制需要一个启动器**：官方 `teleop record` 脚本的内置设备只有 keyboard/spacemouse；
名字为 `gamepad` 的设备必须从任务配置 `env_cfg.teleop_devices.devices` 里解析
（录制脚本对此报错原文：`--teleop_device=gamepad ... Built-in devices: keyboard, spacemouse.`）。

工作区的启动器脚本解决了这个问题：它在启动任何仿真之前，先注册**任务变体**
`IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop`（在官方任务配置上加 `gamepad` 和 `mouse` 两个设备条目），
然后把参数透传给官方脚本运行。变体里的手柄灵敏度为 0.10/0.10（2026-10-14 按需求从 0.05 加倍；键盘保持 0.05）。

**录制**（手柄 Y 键 = 重开当前演示；X 键 = 夹爪）：

```bash
# 1) 热身
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --visualizer kit --num_envs 1 --teleop_device gamepad

# 2) 录制 10 条
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --visualizer kit \
  --teleop_device gamepad \
  --num_demos 10 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5
```

> 手感调节：如太快/太慢，编辑 `isaaclab_teleop_extras.py` 里
> `Se3GamepadWithResetCfg(pos_sensitivity=0.10, rot_sensitivity=0.10, ...)` 的数值后重跑
> （0.10 为 2026-10-14 加倍后的当前值）。

### 3.3 鼠标

- **方式 A**（视角辅助）：见 §2.3，配合键盘操作。
- **方式 B**（纯鼠标控制）：见 §2.3 命令；录制时把 `run` 换成 `record` 并加 `--num_demos/--dataset_file`：

```bash
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \
  --visualizer kit \
  --teleop_device mouse \
  --num_demos 10 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_mouse.hdf5
```

> 鼠标命令是"速度式"：鼠标拖着走，机器人跟着动；停下即停。XY 平移与姿态旋转都需**按住对应键拖动**。
> 灵敏度：0.0005 m/像素、0.004 rad/像素、滚轮 0.015 m/格；在 extras 脚本的 `Se3MouseCfg` 里改。

---

## 4. 正式录制流程（以键盘为例，其余设备只换命令）

**建议目标**：先录 **10 条**（够跑通全流程），熟练后补到 **20~30 条**再放大生成。

### 4.1 完整命令

```bash
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets

./isaaclab.sh teleop record \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit \
  --teleop_device keyboard \
  --num_demos 10 \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
```

### 4.2 会话流程（逐条演练）

1. **启动**：等场景加载完（几十秒；出现 Franka + 三块积木 + 终端打印键位说明）。
2. **第一条：慢动作走一遍流程**（从下到上：**蓝 → 红 → 绿**）——先把**红块**叠到**蓝块**上（蓝块保持不动）→ K 开爪 → 再抓**绿块**叠到**红块**上 → K 开爪并移开机械臂，稳定保持 2~3 秒 → 等终端打印 `Success condition met! Recording completed.` 与新计数。
3. **自动进入下一条**（环境重置，积木随机换位）。
4. **失败了就按 R**：积木碰倒/抓歪 → 按 R 重开本条（失败的尝试不会入库）。
5. **录满 10 条脚本自动退出**，打印 `Recording session completed with 10 successful demonstrations`。
6. 中途想退出：`Ctrl+C`（走正常收尾，**已成功的演示都已保存**）。

> ⚠️ **叠放顺序必须正确**：判定要求 **蓝(底) → 红(中) → 绿(顶)**。若叠成其它顺序（如红底-绿中-蓝顶），任务判定**永远不会触发**——终端无成功打印、数据也不会导出（HDF5 会一直保持空骨架）。

### 4.3 录制纪律（决定数据质量）

- 失败立刻按 R，**不要在失败 episode 里"救回来"**——救回来的轨迹动作杂乱，会降低数据质量；
- 动作宁慢勿快、以点按为主；抓取前对准积木中心再合爪；
- 叠放顺序恒定：**红叠蓝 → 绿叠红**（从下到上 蓝→红→绿），不要换顺序（成功判定与 Mimic 标注都依赖该结构）；
- 每录 5 条可中途停下来 `replay` 检查一遍（§5），再继续。

---

## 5. 回放验证（第一层数据检查）

用录制的数据集回放，验证"数据自洽"（同一物理下动作序列可复现、且都达到成功条件）：

```bash
./isaaclab.sh teleop replay \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_envs 1 \
  --reset_sim_buffer_each_episode \
  --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
```

- **预期**：10/10 条回放成功（每条都能叠好）。
- 个别失败 → 该条轨迹在录制时依赖了大量"实时微调"，回放（无人类修正）复现不了：属正常边界情况，
  若只为跑通流程可忽略；追求高质量时**删除重录**该条（重新录一份同名数据集，或用 stats 脚本定位异常条）。
- 三种设备录的数据集回放命令完全一样（replay 不需要遥操作设备）。

---

## 6. 数据处理：理解录制的原始输出（HDF5）

### 6.1 文件结构

`record_demos` 输出为单个 `.hdf5`（robomimic 兼容 schema）：

```
<dataset>.hdf5
├── attrs: format_version = 1
└── data/                                attrs: total(总步数), env_args(JSON: env_name/type)
    ├── demo_0/
    │   ├── attrs: num_samples(步数), seed(随机种子), success(是否成功=True)
    │   ├── actions               (T, 7)  每步命令: [Δx,Δy,Δz, Δrx,Δry,Δrz, 夹爪(±1)]
    │   ├── processed_actions     (T, N)  经控制器处理后的动作(关节目标, 训练一般不用)
    │   ├── obs/                          策略观测(逐步记录), 本例各 term:
    │   │   ├── joint_pos, joint_vel      关节位置/速度
    │   │   ├── object                    物体(积木)相对状态
    │   │   ├── cube_positions, cube_orientations
    │   │   ├── eef_pos, eef_quat         末端位姿
    │   │   └── gripper_pos               夹爪宽度
    │   ├── initial_state/                该条开始时的完整仿真状态(重置用)
    │   └── states/                       逐步完整仿真状态(物理真值)
    ├── demo_1/ ...
```

要点：

- **训练直接消费**：`obs/*`（观测）+ `actions`（动作）；`initial_state/states` 供重置与回放；
- 该任务无相机，`obs` 里没有图像（Visuomotor 版本才有 `wrist_cam` 等）；
- `success` 全为 True（导出模式=只导出成功演示）。

### 6.2 快速查看（可选）

```bash
python - <<'PY'
import h5py
f = h5py.File("/home/dillon/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5", "r")
d = f["data"]
print("demos:", list(d.keys()))
demo = d["demo_0"]
print("attrs:", dict(demo.attrs))
print("actions:", demo["actions"].shape)
print("obs keys:", list(demo["obs"].keys()))
PY
```

### 6.3 统计脚本（推荐）

```bash
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
# 可另存 JSON: --json ~/code/isaac/IsaacDataCollection/datasets/stats_keyboard.json
```

---

## 7. 质量评估

### 7.1 统计指标怎么读（`teleop_dataset_stats.py` 输出）

| 指标                  | 含义               | 健康参考（本任务经验值）        | 异常信号与处理                                                   |
| --------------------- | ------------------ | ------------------------------- | ---------------------------------------------------------------- |
| `steps`             | 单条演示步数       | 60~600 步（新手偏上限）         | <30 过短（可能误触成功判据）；>1200 过长（失败未按 R，混入试错） |
| `pos_move`          | 末端行程累加 (m)   | 0.1~1.5 m                       | <0.05 m：几乎没动，检查动作是否发出去                            |
| `rot_move`          | 姿态变化累加 (rad) | 约 0.5~6 rad                    | 过大说明反复调姿态，可接受但偏乱                                 |
| `jerk`              | 相邻步动作变化均值 | 越小越平滑；个体<3×中位数      | 某条远大于其他：该条抖动明显（紧张/猛点），考虑重录              |
| `grip_toggles`      | 夹爪开合切换次数   | **≥2**（抓一次、放一次） | =0：夹爪从未开合，说明没执行抓放流程                             |
| `grip_closed_ratio` | 闭合时间占比       | 30%~70%                         | 接近 0：一直没合爪                                               |
| `success`           | 成功标记           | 全 True                         | 出现 False = 异常（正常只导出成功演示）                          |

脚本末尾的**质量提示**会把上述异常逐条列出（阈值是经验值，用于快速筛查，不是硬标准）。

### 7.2 三级质量评估流程（从便宜到终极）

```
L1 数值筛查：   teleop_dataset_stats.py  → 无警告、指标在参考区间
        ↓
L2 回放自洽：   teleop replay            → 10/10 成功、轨迹可复现
        ↓
L3 训练可用性（终极判据）:
        ↓
   Mimic 生成成功率（数据"可被机器理解"）  → 经验 ≥30% 及格、≥50% 良好
        ↓
   BC 策略回放成功率（数据"真的能学到东西"）→ 目标明显高于随机，>50% 说明链路有效
```

### 7.3 常见劣质数据模式与修复

| 现象                     | 原因                                                      | 修复                                                                |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------- |
| 某条 steps 极长、jerk 大 | 失败后没按 R，现场"救"了回来                              | 删/重录该条；下次失败立即 R                                         |
| 生成成功率低（<20%）     | 演示子任务结构不一致（抓取顺序/路径变来变去）；标注不准   | 统一流程重录；确认 annotate 的自动标注结果                          |
| BC 成功率低              | 种子演示太少（10 条偏少）；场景覆盖不足（只从一个方向抓） | 补录到 20~30 条，且**刻意变化**：从不同方位接近、不同堆叠顺序 |
| 回放失败个别条           | 录制时高度依赖人类实时修正                                | 重录该条（放慢、规划好路径）                                        |

---

## 8. 数据使用：Mimic 增广 → BC 训练 → 成功率闭环

> 这是"用你的录制数据"的完整下游流程：以少量人类演示为种子，自动生成大规模演示，
> 训练行为克隆策略，并用回放成功率反向验证数据质量。全部小规模参数已适配 8GB 显存。

### Step 1：标注（annotate，~2-5 分钟）

给每条演示的每个子任务（grasp_1 / stack_1 / grasp_2）自动打边界：

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --visualizer kit --auto \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5
```

- **成功标志**：终端打印 `Exported 9 (out of 10) annotated episodes.`（数字随数据变化）；个别条被跳过属正常；
- **机制**：annotate 会把每条演示在 Mimic 环境里**逐帧回放**，用记录器收集标注（`obs/datagen_info/`：eef/物体/target 位姿 + 子任务信号）——因此它同时是一次“回放验证”：
  - 回放结束时不满足终判据（cubes_stacked）的条 → 打印 `The final task was not completed.` 并跳过；
  - 任一子任务信号从未触发的条 → 也会跳过；
  - **被跳过的条不导出，输出文件按剩余条重新编号**（如 10 条输出 9 条，编号会顺移，引用演示时注意）。
- `--auto` 依赖子任务完成信号（`subtask_terms`），要求你的演示与标准流程一致——这正是 §4.3 要求"顺序一致"的原因。

### Step 2：生成（generate，先小后大）

先小规模验证（10 条）：

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --visualizer kit --num_envs 16 --generation_num_trials 10 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_small.hdf5
```

跑通后放大（8GB 建议 `--num_envs 32~64`）：

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --num_envs 32 --generation_num_trials 100 \
  --input_file ~/code/isaac/IsaacDataCollection/datasets/annotated_kbd.hdf5 \
  --output_file ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5
```

- 终端会打印生成进度/成功统计；**生成成功率**是本任务最直接的数据质量信号
  （官方参考：state 模式约 40%，种子/标注质量好可更高）；
- 失败轨迹自动丢弃，只会导出成功演示；
- ⚠️ **`--generation_num_trials` 语义**：默认 `generation_guarantee=True` 下是**"目标成功条数"**（失败自动重试直到攒够），不是"尝试次数"；
- 实测（2026-10-12，9 条手柄种子 / `--num_envs 16` / 目标 10 条）：**10/10 成功、约 17 分钟**；生成文件随每次成功增量重写，完成后自动退出。
- 📖 第二轮全流程（全新录 30 条 → 标注 → 生成 300 → 训练 6000ep → 评估）→ 见 `mimic-pipeline-round2-guide.md`

### Step 3：BC 训练（~10-30 分钟）

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --algo bc \
  --dataset ~/code/isaac/IsaacDataCollection/datasets/generated_kbd_100.hdf5
```

- 训练日志/checkpoint 在 `logs/robomimic/...`（终端会打印路径）；
- 先用小规模数据集跑 20~50 epoch 验证链路即可，无需一次练到位。

### Step 4：策略回放评估（~5-10 分钟）

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
  --visualizer kit --num_rollouts 20 \
  --checkpoint <上一步 logs/robomimic/.../models/model_xxx.pth>
```

- 终端输出成功条数/成功率（20 次回放）；
- 多试几个 checkpoint（不同 epoch 表现差异大）。

### Step 5：闭环决策

| 观察                             | 判断                 | 行动                                                |
| -------------------------------- | -------------------- | --------------------------------------------------- |
| 生成成功率 ≥30%，BC 成功率 >50% | 数据链路有效 ✅      | 扩规模：补录到 20-30 条 → 生成 1000 条 → 完整训练 |
| 生成成功率 <20%                  | 种子演示结构混乱     | 回 §4.3 统一流程重录；检查 annotate 边界           |
| BC 成功率 <20%                   | 数据量不足或覆盖单一 | 补采多样化演示；增加生成条数                        |

> 数据管理建议：原始/标注/生成文件分名存放（`dataset_*.hdf5` / `annotated_*.hdf5` / `generated_*.hdf5`），
> 每次重录换新文件名，避免覆盖；测试新想法前把 `~/code/isaac/IsaacDataCollection/datasets/` 打包归档。

---

## 9. 常见问题排查

| 报错/现象                                                                              | 原因                                                          | 解决                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Downloaded Isaac Sim packages cannot be combined with a Python virtual environment` | venv 未基于 Isaac Sim 自带 Python（当前环境为空壳时也会出现） | 按 §0.1 重建                                                                                                                                                                          |
| `--teleop_device=gamepad ... Built-in devices: keyboard, spacemouse`                 | 用了官方 record（内置不含 gamepad）                           | 用 §3.2 启动器 +`-Teleop` 任务                                                                                                                                                      |
| 手柄无反应                                                                             | 窗口无焦点 / 设备未识别                                       | 点击 Isaac 窗口；`ls /dev/input/js*`；显示 `Device name: None` 时跑 `python3 IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py check`（2026-10-12 蓝牙手柄已修复；Sim 升级后需重 patch） |
| 鼠标方式 B 无反应                                                                      | 需按住键拖动 / 未实测警告                                     | 按住左键再拖；确认焦点；必要时调 extras 脚本                                                                                                                                           |
| 键盘按了没反应                                                                         | 焦点在终端                                                    | 点击仿真窗口再操作                                                                                                                                                                     |
| 录制退出后少于`--num_demos` 条                                                       | 只导出成功演示；或中途 Ctrl+C                                 | 正常行为；继续录或降低难度                                                                                                                                                             |
| 同名数据集被覆盖                                                                       | 打开同名文件会重建                                            | 每次换新文件名，或先归档                                                                                                                                                               |
| `Success condition met` 后按 R 没反应                                                | 成功触发后的保护窗口                                          | 等自动重置（终端有提示）                                                                                                                                                               |
| 生成阶段 OOM                                                                           | `--num_envs` 太大                                           | 降到 16~32（8GB 卡）                                                                                                                                                                   |
| replay 失败个别条                                                                      | 该条依赖人类实时修正                                          | 重录该条（见 §5）                                                                                                                                                                     |

---

## 10. 附录

### 10.1 命令速查

```bash
# 通用前置
cd ~/code/isaac/isaaclab && source .venv/bin/activate && mkdir -p ~/code/isaac/IsaacDataCollection/datasets

# 键盘：run / record
./isaaclab.sh teleop run    --task IsaacContrib-Stack-Cube-Franka-IK-Rel --visualizer kit --num_envs 1 --teleop_device keyboard
./isaaclab.sh teleop record --task IsaacContrib-Stack-Cube-Franka-IK-Rel --visualizer kit --teleop_device keyboard --num_demos 10 --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5

# 手柄：run / record（经启动器）
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run    --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop --visualizer kit --num_envs 1 --teleop_device gamepad
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop --visualizer kit --teleop_device gamepad --num_demos 10 --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5

# 鼠标：run / record（经启动器）
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run    --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop --visualizer kit --num_envs 1 --teleop_device mouse
./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop --visualizer kit --teleop_device mouse --num_demos 10 --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_mouse.hdf5

# 回放（任意数据集）
./isaaclab.sh teleop replay --task IsaacContrib-Stack-Cube-Franka-IK-Rel --visualizer kit --num_envs 1 --reset_sim_buffer_each_episode --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5

# 统计
python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
```

### 10.2 关键参数

| 参数                                                  | 默认                        | 说明                                    |
| ----------------------------------------------------- | --------------------------- | --------------------------------------- |
| `--num_demos`                                       | 0（无限）                   | 目标成功演示条数；达到即自动退出        |
| `--num_success_steps`                               | 10                          | 需连续保持成功的步数（防止"碰巧碰到"）  |
| `--dataset_file`                                    | `~/code/isaac/IsaacDataCollection/datasets/dataset.hdf5` | 输出文件；同名会覆盖                    |
| `--visualizer kit`                                  | —                          | 必须有 GUI 才能遥操作                   |
| `--reset_sim_buffer_each_episode`                   | off                         | 回放时每条之间强制 sim.reset()，更严格  |
| generate:`--num_envs` / `--generation_num_trials` | —                          | 并行环境数 / 目标生成条数；8GB 用 16~32 |

### 10.3 文件索引

- 本文档：`IsaacDataCollection/docs/teleoperation-hands-on-franka-stack-cube.md`
- 扩展与启动器：`IsaacDataCollection/scripts/isaaclab_teleop_extras.py`、`IsaacDataCollection/scripts/isaaclab_teleop_launch.py`
- 统计脚本：`IsaacDataCollection/scripts/teleop_dataset_stats.py`
- 上游参考：`isaaclab/docs/source/features/imitation-learning/teleop_imitation.rst`（官方 Mimic 全流程）、
  `isaaclab/source/isaaclab/isaaclab/devices/`（设备实现源）
- 关联综述：`IsaacDataCollection/docs/robot-data-collection-in-isaac-survey.md`（§2 遥操作）

### 10.4 预计总耗时（首次）

| 阶段               | 时间                            |
| ------------------ | ------------------------------- |
| 热身三设备         | 30-60 分钟                      |
| 录制 10 条         | 30-60 分钟（新手单条 1-3 分钟） |
| 回放 + 统计        | 15 分钟                         |
| 标注 + 小规模生成  | 10-20 分钟                      |
| BC 训练 + 回放评估 | 20-40 分钟                      |
| **合计**     | **约半天**                |

---

*文档：2026-10-09 · isaac 工作区 · 配套脚本位于 `IsaacDataCollection/scripts/`*
