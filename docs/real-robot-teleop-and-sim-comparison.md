# 真机遥操作:设备、数据与仿真对比实战指南

> **定位**:把"真机遥操作"讲透——**用什么设备、怎么用、产生什么数据、怎么处理、与仿真数据逐项对比**。
> 真机设备信息引自各项目官方站点(链接见 §7,2026-10-12 核实);仿真侧结论引自本工作区实测
> (Isaac Lab 3.0 + `dataset_keyboard.hdf5`,详见 [`hdf5-format-guide.md`](./hdf5-format-guide.md))。
>
> 关联文档:
> - [`robot-data-collection-in-isaac-survey.md`](./robot-data-collection-in-isaac-survey.md) — 五类采集方式全景(§2 遥操作、§6 仿真采集)
> - [`teleoperation-hands-on-franka-stack-cube.md`](./teleoperation-hands-on-franka-stack-cube.md) — 本工作区仿真录数实操
> - [`hdf5-format-guide.md`](./hdf5-format-guide.md) — 仿真 HDF5 数据集全解(字段级)

---

## 0. 一分钟速览

| 问题 | 结论 |
| --- | --- |
| 真机遥操作的本质 | 人用**设备**实时操纵真机完成任务,同步记录"观测 → 动作"序列,供模仿学习使用 |
| 有哪几类设备 | ① 同构主从臂(ALOHA/GELLO)② 通用输入(手柄/SpaceMouse/键盘)③ VR 头显 ④ 力反馈主手 ⑤ 手持夹具(UMI)⑥ 手动示教(freedrive) |
| 数据长什么样 | 原始:**ROS 2 bag**(话题流);成品:**HDF5**(robomimic/ACT 系)/ **LeRobot**(Parquet+MP4) |
| 和仿真数据最核心的差异 | 真机**没有全状态真值、成功靠人工标注、频率/时间戳要自己对齐**;仿真**全真值、自动成功判定、微秒级同步、几乎零成本** |
| 怎么选 | 跟着目标任务走:Franka 系 → GELLO;最省钱 → SO-101 套件;野外视觉策略 → UMI;双手灵巧/人形 → VR;接触精细 → 力反馈主手 |

---

## 1. 真机遥操作是什么

操作者通过**控制器(手柄/VR/主从臂/力反馈主手……)实时操纵一台真实机器人**完成任务,
系统同步记录机器人的观测与动作,得到"人类技能 → 机器人可学习数据"的转换。

**为什么不可替代**:

- 数据**真实可部署**:真实接触摩擦、真实相机噪声、真实光照/场景,不存在 Sim2Real 差距;
- 是模仿学习(IL)数据质量的"金标准"(ALOHA、Mobile ALOHA、UMI 等里程碑工作全部建立在真机遥操作数据上);
- 部分任务(柔顺接触、可变形物体、精细插拔)**今天仍难以在仿真中保真复现**。

**局限**:慢(一人一机)、贵(硬件+人力)、场景固定、失败演示要手动处理。

> 与其它四类真机采集方式(动捕/UMI/EGO/仿真)的关系见综述 §1;**UMI 严格来说不遥控机器人**(手持夹具直接演示),本文在 §2.2e/§6 中对比讨论。

---

## 2. 设备全景(6 大类)

### 2.0 总览表

| # | 类别 | 代表系统 | 原理一句话 | 力反馈 | 成本量级 | 采集效率(例) |
| - | --- | --- | --- | --- | --- | --- |
| a | **同构主从臂** | ALOHA / ALOHA 2、GELLO、SO-101 | 一个小型/同规格"主臂",人拖主臂→从臂逐关节复现 | ❌(有重力补偿) | <$300(GELLO 零件)~ 数万美元(整机套件) | 人手直接带节奏,任务成功率高 |
| b | **通用输入设备** | 游戏手柄、键盘、SpaceMouse | 6DoF 摇杆/按键 → 末端增量或速度指令 | ❌ | 数十~数百美元 | UMI 论文中 SpaceMouse ≈ 35 条/小时 |
| c | **VR 头显** | Open-TeleVision(开源)、各类 Quest 方案 | 头显追踪手柄/手部姿态 → 重定向到机器人 | ❌(视觉沉浸) | 数百美元(消费级头显) | 直观、双手灵巧;有眩晕/重定向成本 |
| d | **力反馈主手** | Geomagic Touch、Force Dimension、Haption | 专业并联/串联主手,双向力觉回路 | ✅(强) | 数千~数万美元 | 精细接触任务(医疗/工业遥操作) |
| e | **手持夹具** | UMI、Fast-UMI、DexUMI | 人**手持带相机的夹爪**直接演示;SLAM 恢复 6DoF 轨迹 | ❌(无需机器人) | 数百美元级(3D 打印件 + GoPro) | **111 条/小时**(UMI 杯子任务实测) |
| f | **手动示教(freedrive)** | Franka FR3/UR 手拖示教、示教器 | 断开伺服,人直接拖动机器人本体记轨迹 | ✅(机械反馈) | 0 增量(用机器人本身) | 慢,但零额外硬件 |

### 2.1 (a)同构主从臂 — 当前数据采集的主力

**ALOHA / ALOHA 2**(Stanford + Google DeepMind,已商业化):

- 双主臂 + 双从臂的**桌面级双臂遥操作工作站**;ALOHA 2 相对初代改进:低摩擦直线导轨主/从夹爪、
  被动重力补偿(替代橡皮筋系统)、20×20 铝型材框架、Intel RealSense D405 相机(全局快门+深度);
- 源码/硬件全开源,套件由 Trossen Robotics 销售(Aloha Stationary / WidowX / ViperX 版本),
  并提供 **MuJoCo Menagerie 高保真模型**(可在仿真里复刻同一平台);
- **Mobile ALOHA**:在 ALOHA 基础上加移动底盘 + 全身遥操作;论文里**每个任务仅 50 条演示**,
  与静态 ALOHA 数据**共训**后成功率最高提升 90%(CoRL 2024)。

**GELLO**(UC Berkeley,低成本平替):

- 思路:为你的目标机械臂**造一个小号的"运动学复刻"主臂**(3D 打印件 + 现成 Dynamixel 舵机),
  人拖 GELLO → 目标臂按关节映射同步;**零件成本 < $300**,组装约 30 分钟;
- 已支持 **Franka FR3/Panda、UR、xArm、I2RT YAM** 等;官方用户研究:比 **VR 手柄和 3D SpaceMouse
  更快、更可靠**(5 项含精细插拔的双臂任务,12 名被试);
- 软件开源(MIT),自带数据保存接口(`--use-save-interface`,按键 `s` 开始 / `q` 停止录制)与
  ROS 2(Franka FR3)支持。**本工作区迁移真机的首选参考**(见 §6.4)。

**SO-100 / SO-101**(TheRobotStudio 开源臂,LeRobot 生态):

- 数百美元级零件的开源"leader+follower"双臂套件,3D 打印+舵机;
- HuggingFace LeRobot 的官方录制示例就是 `so101_leader` 遥操 `so101_follower`
  (`lerobot-record` 一条命令搞定采集+存格式+上传 Hub,见 §4.3)。

### 2.2 (b)(c)(d) 通用输入 / VR / 力反馈

- **手柄/键盘/SpaceMouse**:零门槛、便宜;适合快速验证与移动底盘/无人机场景。
  弱点是**表达力有限**(6DoF 主手难同时精确控位姿+夹爪),UMI 论文的对照实验里
  SpaceMouse 采数效率仅为手持夹具的 ~1/3(35 vs 111 条/小时);
- **VR**:消费级头显(Quest 类)+ 双手手柄/手势追踪。
  代表 **Open-TeleVision**(UCSD, 2024,开源):操作者通过头显以**立体视觉主动感知**机器人周围环境,
  手臂与手部动作**镜像映射**到机器人;在 2 种人形机器人、4 项长程精细任务(分拣罐/插罐/叠衣/卸货)
  上验证并真机部署,项目站点 `robot-tv.github.io`;
  注意点:重定向(人手≠机械手)、延迟导致眩晕、安全限速;
- **力反馈主手**:Geomagic Touch / Force Dimension / Haption 等,**双向力觉**回路让操作者"摸到"远端,
  用于精细接触(医疗、核工业、装配研究);成本高、配置复杂,一般数据采集项目用不到。

### 2.3 (e)手持夹具 UMI — "不用机器人的遥操作"

**Universal Manipulation Interface**(Stanford/TRI, RSS 2024,开源):

- 硬件 = **手持平行夹爪 + GoPro**(数百美元级:3D 打印件 + GoPro),腕部同轴相机 + 镜面提供深度线索;
- 人**直接用手拿着夹具**在任意场景演示任务(家里/餐厅,2 分钟内开始采集);
- 后处理:GoPro SLAM 恢复夹爪 **6DoF 轨迹** → **相对轨迹动作表示** + 推理时**延迟匹配**;
- 关键特性(论文实测):数据**100% 免标定**(甚至允许移动基座)、策略**跨机器人**(UR5e 与 Franka 同一策略)、
  自动校验每条演示是否满足机械臂运动学约束;采集速度 **111 条/小时**(约人手速度 48%,比 SpaceMouse 遥操作快 3 倍+);
- 适合:视觉策略(扩散策略等)+ 野外多样性;缺点:没有"真机执行"数据,动作需机器人侧复现(夹爪行程需 >85mm)。

### 2.4 (f)手动示教与示教器

- **freedrive/动觉示教**:FR3、UR 等协作臂可断开伺服,人手拖动"教"轨迹;
  数据=follower 自身关节轨迹(质量高、无主从差异),但**每次只能教一条且很慢**,不适合大批采集;
- **示教器(Teach Pendant)**:UR 示教器、FR3 Desk 等,逐点位编程;数据是"路点+运动指令",
  离"连续状态-动作演示"较远,一般不用于 IL 数据采集。

### 2.5 选型建议(按目标)

| 你的目标 | 推荐组合 | 理由 |
| --- | --- | --- |
| **本工作区任务的真机版**(Franka 叠方块) | **Franka FR3 + GELLO(Franka 版)** | 运动学复刻、便宜、有 ROS2 支持;动作语义(EE 增量/关节)与现有工作流可对齐 |
| 学术标准双臂平台 | ALOHA 2(Trossen 套件) | 生态成熟(ACT 数据/策略全公开),50 条/任务即可出效果 |
| 最低预算入门 | SO-101 套件 + LeRobot | 数百美元 + `lerobot-record` 一站式采集/存储/共享 |
| 野外视觉策略 | UMI | 免标定、跨机器人、效率最高 |
| 双手灵巧/人形 | VR(Open-TeleVision 类) | 手部动作映射,沉浸感知 |
| 精细接触/力觉研究 | 力反馈主手 | 唯一能"传递力"的方案 |

---

## 3. 使用方法(从装好到录完)

### 3.1 通用五步流程

```
① 硬件搭建 → ② 标定/映射 → ③ 遥操作控制 → ④ 录制 → ⑤ 质检归档
```

**① 硬件搭建**:主臂/从臂固定与零位对齐;相机安装(全局快门优先);网络/总线(USB-CAN、EtherCAT)。

**② 标定/映射**(真机独有的关键环节,仿真里不存在):

- 关节空间主从(GELLO/ALOHA):对齐**零位偏移**(offset)与**关节符号**(joint signs),
  例如 GELLO 对 Panda 的参考值为 `joint-signs = 1 -1 1 1 1 -1 1`,并用
  `gello_get_offset.py` 脚本自动求偏移;
- 位姿空间(VR/SpaceMouse):标定**手→机器人坐标系**映射,用 IK 重定向;设置比例缩放与工作空间边界。

**③ 遥操作控制**:主从关节镜像(GELLO/ALOHA)或末端增量(VR/手柄);有重力补偿(GELLO 的 FACTR、
ALOHA 2 被动机构)手感会好很多;控制回路一般 100~1000 Hz。

**④ 录制**:GELLO `--use-save-interface`(`s` 开始 / `q` 停止);ALOHA 走其记录脚本;
LeRobot 用 `lerobot-record`;自研栈写 ROS 2 录制节点。

**⑤ 质检归档**:检查频率/丢帧/时长/失败标记(见 §5.1),按任务/日期/操作员命名归档。

### 3.2 安全清单(真机必备,先说断)

1. **急停按钮**伸手可及;操作员与机器人工位分离;
2. 速度/力矩**限幅**(首周建议 ≤10% 正常速度),工作空间软限位;
3. **看门狗**:主手断连/超时 → 机器人保持或缓慢停止;
4. 大幅运动前**先离物体**,再进入精细操作;
5. 电池/气路/线缆走线固定,防缠绕(移动平台尤其);
6. 记录**异常事件**(碰撞、急停)并与数据标注关联——**出事的演示大概率是脏数据**。

### 3.3 录制系统架构(推荐)

```
主手设备(100~1000 Hz) ──► 机器人控制接口(如 FR3 的 franka_ros2 / UR 的 RTDE)
        │                          │
        │                          ▼
        └──────────► 录制进程 ◄── 机器人状态话题(/joint_states 100~500 Hz)
                        │
                        ├─ 相机(RealSense/USB,30~60 Hz,硬件/软件同步)
                        └─ 写入:ROS 2 bag(rosbag2) 或 直接 HDF5/LeRobot
```

要点:**录制进程独立于控制回路**(避免卡顿影响安全);所有流带**时间戳**(相机优先硬件触发或统一时钟);采集开始时做一次共同同步事件(闪灯/敲击),供事后快速校验对齐。

---

## 4. 真机产生的数据

### 4.1 数据层级

| 层级 | 形态 | 说明 |
| --- | --- | --- |
| 原始层 | **ROS 2 bag**(rosbag2;Humble 默认 sqlite3,新版 ROS 2 与社区广泛用 **MCAP**) | 逐话题、带时间戳的完整数据流 |
| 成品层 | **HDF5**(robomimic / ACT 系) | 按"演示(episode)"组织,便于训练 |
| 成品层 | **LeRobot 数据集**(HuggingFace) | Parquet(低维)+ MP4(视频)+ JSON/Parquet 元数据 |
| 其它 | RLDS/TFDS(Open X-Embodiment 系)、**Zarr**(UMI 的 SLAM 后处理训练集:`dataset.zarr.zip`)、npz 逐条存等 | 视项目生态而定 |

### 4.2 原始层:ROS 2 bag 里都有什么

典型话题(以 Franka FR3 + 相机 + 自研遥操为例):

| 话题 | 类型 | 频率 | 内容 |
| --- | --- | --- | --- |
| `/joint_states` | `sensor_msgs/JointState` | 100~500 Hz | 关节位置/速度/力矩(双臂+夹爪) |
| `/tf`、`/tf_static` | `tf2_msgs` | 按需 | 坐标系变换(末端、相机外参) |
| `/<robot>/cmd`(遥操指令) | 自定义 / `JointTrajectory` 等 | 同控制回路 | **动作数据**(主手映射后的命令) |
| 夹爪状态 | 自定义 | 100 Hz | 开合宽度/力 |
| `/cameras/*/image_raw/compressed` | `sensor_msgs/CompressedImage` | 30~60 Hz | 相机流(h264/MJPEG) |
| `/clock` | `rosgraph_msgs/Clock` | — | 仿真/同步时钟 |

工具链:采集 `ros2 bag record -a`;查看 `ros2 bag info`;**可视化 PlotJuggler / Foxglove**;
导出 `ros2 bag convert`(可转 MCAP、抽取话题)。

### 4.3 成品层(一):HDF5

两大类布局:

1. **robomimic 兼容**(与 Isaac Lab 同源)——`data/demo_N/{obs/*, actions, states?, initial_state?}` + `env_args`。
   真机数据集同样可用该 schema(robomimic 官方支持真机数据),**这正是与本工作区仿真数据"汇流"的接口格式**;
2. **ACT/ALOHA 系**(视觉双臂常见)——每条演示:
   `observations/images/{cam_high, cam_left_wrist, cam_right_wrist}`(uint8 图像)+
   `observations/qpos`(双臂 6+1 关节)+ `action`(14 维);早期版本无成功标记(全为成功演示)。

### 4.4 成品层(二):LeRobot(HuggingFace 标准,当前 v3.0)

- **文件式存储**:`data/` 为 **Parquet** 分片(多个 episode 合并存)、`videos/` 为 **MP4** 分片(每相机);
- 元数据:`meta/info.json`(schema、fps、路径模板)、`meta/stats.json`(归一化统计)、
  `meta/tasks.jsonl`(语言任务)、`meta/episodes/`(分片 Parquet,记录每条的起止/任务/长度);
- 训练侧:`LeRobotDataset` 直接吃 Hub / 本地数据,支持 `delta_timestamps` 取"过去 T 帧"窗口,
  也有 `StreamingLeRobotDataset` 免下载流式训练;
- 录制:`lerobot-record`(SO-101 leader/follower 为官方示例)——**采集→编码→存格式→上传 Hub 一条龙**;
- 注意:录制完必须 `dataset.finalize()`;`v2.1 → v3.0` 官方有转换脚本。
- 图像增广(亮度/对比度等)在**训练时**做,原始帧不加工,便于复现。

### 4.5 真机数据的"脏"点(与仿真对比前先记住)

| 问题 | 表现 | 影响 |
| --- | --- | --- |
| 时间戳漂移/丢帧 | 相机与机器人时钟不同步;USB 掉帧 | 观测-动作错位,策略学出"抖" |
| 通信延迟 | 主手→从臂 10~100 ms | 需要 UMI 式"延迟匹配"或控制补偿 |
| 无真值状态 | 只有传感器读数,没有"物理真值" | 无法仿真式精确回放;分析靠估计 |
| 人工成功标注 | 成功/失败靠人/脚本判定 | 标注质量=数据质量上限 |
| 标定误差 | 相机外参、手眼标定、零位漂移 | 视觉-动作不一致 |
| 编码/压缩伪影 | 视频有损压缩、运动模糊 | 视觉策略鲁棒性下降 |

---

## 5. 处理与使用

### 5.1 质检(录完先跑)

- `ros2 bag info`:时长/频率/消息数,先看有没有明显掉帧;
- 逐话题频率直方图(自写脚本或 Foxglove/PlotJuggler):找断流、抖动;
- 视频抽帧看:模糊、曝光、遮挡;
- 统计量筛查:步长分布、动作抖动(jerk)、行程是否过短——
  **本工作区 `teleop_dataset_stats.py` 的指标逻辑对真机数据同样适用**(转换后跑,见 §5.4)。

### 5.2 同步与重采样

1. 以**相机时钟**为基准(30 Hz),机器人状态按时间戳**插值到相机帧**(或反之按控制帧对齐);
2. 插值用线性/球面(姿态用 Slerp);注意**不要跨大间隔插值**(丢帧区间直接截断或整条剔除);
3. 统一单位与坐标系(米/弧度;世界系/基座系);
4. 固定 fps 存成品(HDF5/LeRobot),把"时间戳同步"这一步固化到转换脚本里。

### 5.3 分段与标注

- **任务边界**:按"开始采集-成功/放弃"键标记,或按脚本检测;失败段剔除;
- **成功标记**:写入演示属性(robomimic `success` / LeRobot 任务元数据);
- **语言指令/子任务**:LeRobot `tasks.jsonl`、Mimic 的 annotate——为 VLA/增广做准备。

### 5.4 转换到训练格式

**目标一:robomimic HDF5(与本工作区仿真数据同格式,推荐)**

```
rosbag → (脚本) → data/demo_N/
    ├── actions        ← 遥操指令(关节目标或 EE 增量,统一到任务动作空间)
    ├── obs/*          ← /joint_states + FK(末端位姿)+ 夹爪 + 视频帧(如需)
    ├── states/*       ← 可选:关节位置/速度(真机无"物体真值",物体状态建议用相机 + 标定估计)
    ├── initial_state  ← 可选:该条开始时状态快照(用于回放重置)
    └── attrs: num_samples / success
data.attrs: env_args(机器人型号)、total
```

要点:动作空间**必须与任务的仿真版对齐**(如栈方块任务:EE 增量 + 夹爪;GELLO 录的是关节,
用 FK 差分换算);对齐之后,**本工作区全套工具(stats 脚本、绘图脚本、Mimic)可以原样复用**。

**目标二:LeRobot**(若走 HuggingFace 生态 / VLA):`lerobot-record` 直采,或官方数据集转换工具;
可上传 Hub 供云端/他人训练。

**工具速查**:

| 任务 | 工具 |
| --- | --- |
| bag 查看/裁剪/转格式 | `ros2 bag info / convert`;Foxglove |
| 时间线可视化 | PlotJuggler、Foxglove |
| 转 HDF5/LeRobot | 自写转换脚本(v3 参考 `lerobot` 转换器)、robomimic 数据规范 |
| 质检统计 | 本工作区 `IsaacDataCollection/scripts/teleop_dataset_stats.py`(转完格式直接跑) |
| 曲线看图 | 本工作区 `IsaacDataCollection/scripts/hdf5_plot_demo.py` |

### 5.5 训练与闭环

真机数据走同一条 IL 流水线:**BC/ACT/扩散策略训练 → 真机回放/部署评估 → 成功率闭环**。
差别只在评估环境(真机 rollout 而不是仿真 play)。共训(real + sim)见 §6.4。

---

## 6. 真机 vs 仿真:详细对比(核心章节)

> 仿真侧基准 = 本工作区现状:`IsaacContrib-Stack-Cube-Franka-IK-Rel`,键盘/手柄遥操,输出 HDF5
> (3 条演示/2780 步/20 Hz/gzip;字段全表见 `hdf5-format-guide.md` §3)。

### 6.1 总览对照表(18 个维度)

| # | 维度 | 真机遥操作 | 仿真(Isaac Lab) |
| - | --- | --- | --- |
| 1 | 操作设备 | 主从臂/VR/主手/手柄(**硬件采购+维护**) | 键盘/SpaceMouse/手柄/XR(**现有外设即可**) |
| 2 | 机器人本体 | 真机(FR3/UR/ALOHA……),有磨损/维护成本 | 数字资产(USD),零损耗 |
| 3 | 成本结构 | 硬件数千~数万美元 + 人力(慢) | 一次性工作站(RTX 4070 8GB 即可)+ 电费,边际成本≈0 |
| 4 | 数据格式 | rosbag2 → HDF5/LeRobot(需转换) | 直出 HDF5(robomimic 兼容,含 gzip) |
| 5 | 动作空间 | 设备相关(关节目标/EE 增量/速度) | 任务定义(IK 相对增量 + 夹爪),与训练对齐 |
| 6 | 观测 | 传感器真实读数(关节编码器+相机) | 传感器仿真(`obs/*`:关节/末端/物体位姿;可选相机) |
| 7 | **全状态真值** | ❌ 无(需状态估计) | ✅ `states/`+`initial_state` 完整物理真值 |
| 8 | **成功标记** | 人工/脚本判定(质量参差) | **自动判定,只导出成功演示**(`success=True`) |
| 9 | 频率与步长 | 控制 100~1000 Hz;录制 30~100 Hz;相机 30~60 Hz,需同步 | 固定步长(本任务 20 Hz,dt 0.01×decimation 5),天然同步 |
| 10 | 时间对齐 | 需插值/重采样/延迟匹配 | 严格对齐(实测:`actions[t]`↔`obs[t]`,`states[t]==obs[t+1]`) |
| 11 | 可复现性 | 近似(动力学/标定/电池电压等扰动) | 精确(replay 逐值一致,可 `initial_state` 重置) |
| 12 | 规模 | 1 台机(或几台),人工逐条 | 并行 16~4096 环境;Mimic 自动生成 |
| 13 | 多样性来源 | 真实场景("野外",如 UMI 家庭/餐厅) | 域随机化(物位/材质/光照/物理参数) |
| 14 | 视觉保真 | 真实相机(全局快门/真实噪声) | 渲染(需 DR 补偿 Sim2Real) |
| 15 | 风险 | 碰撞、夹坏、急停、人机安全 | 无物理风险,失败可无限重来 |
| 16 | 人力 | 需要操作员(可用性、疲劳、一致性) | 一次脚本,机器自跑 |
| 17 | 失败数据处理 | 手动剔除/标注 | 脚本自动跳过(生成器只留成功轨迹) |
| 18 | 适用阶段 | **部署前微调、真实接触、最终评估** | **大规模预训练、快速迭代、增广、早期评估** |

### 6.2 关键维度展开

**① 数据 schema 与"汇流点"**。两边都可以收敛到 robomimic HDF5:
真机=bags→转换;仿真=直出。汇流后,**同一套质检/训练工具链通吃**(§5.4)。
真正的对齐难点是**动作空间**(关节目标 vs EE 增量,用 FK/IK 互转)和**观测键名与含义**
(如真机 `/joint_states` 是绝对值,仿真 `obs/joint_pos` 是相对默认构型的偏移,必须换算)。

**② 时间与频率**。仿真是"离散步"世界:第 t 步什么都是确定的,录制频率=仿真频率(20 Hz);
真机是"连续时间"世界:控制回路 1 kHz、相机 30 Hz、录制器 50 Hz,**必须显式同步**;
且存在**通信延迟**,这正是 UMI 论文强调"推理时延迟匹配"的原因——不做补偿,策略会抖。

**③ 真值状态 vs 传感器读数**。仿真的 `states/`+`initial_state` 让"精确回放、重置、分层分析"零成本;
真机只有读数与估计值,物体位姿要靠视觉+标定推断(对象位姿真值缺失是"真机数据难做 MIT-Mimic 式增广"的根本原因之一)。
反过来,真机的**接触物理就是真实物理**,仿真里最贵的"物理保真"在真机里免费。

**④ 成功标记**。仿真里 `success` 由任务判定自动产生,且**只导出成功演示**(这是数据质量的系统性保障);
真机靠人按标注键或事后脚本,失败样本混入就是噪声。工程建议:录制界面**必须有一键"作废本条"**,失败立刻标记剔除。

**⑤ 效率与成本(数量级对比)**:

| 采集方式 | 实测/官方数据 | 折算 |
| --- | --- | --- |
| 仿真键盘遥操(本工作区) | 3 条演示 ≈ 139 秒仿真时长 → 1.35 MB HDF5 | 人力=操作时间,数据近似免费 |
| 仿真 Mimic 生成 | 一份 10 条种子 → 机器自动生成(**每小时百条级**,取决于 GPU) | 人力≈0(跑机器) |
| UMI 手持夹具 | **111 条/小时**(杯子任务,含 30 秒/条采集速度) | 3× SpaceMouse 遥操 |
| SpaceMouse 遥操作 | 35 条/小时(同任务对照) | 基线 |
| ALOHA 系真机遥操 | Mobile ALOHA:每任务 50 条即可出效果 | 人力逐条,无并行 |
| 人类手工示范(参照) | 231 条/小时 | 数据的"人类上限" |

**⑥ 多样性与泛化**。仿真多样性=程序化随机化(只覆盖你写进去的东西);
真机多样性=真实世界的长尾(UMI 的爆发力正在于此:任何家庭/餐厅都是新场景)。
仿真覆盖"分布内",真机提供"分布外"。

**⑦ 复现与调试**。仿真 `teleop replay` 能逐值复现(实测工具链已具备);
真机回放是"近似复现"(动力学/温漂/标定漂移),但**真机回放仍是上线前必做的自检**(录的动作能不能再执行一遍完成)。

**⑧ 风险与安全**。仿真里撞坏积木=一键重置;真机里=维修账单。所有真机遥操作文档都应把 §3.2 安全清单放在最前面。

### 6.3 字段级对照示例(同一"叠方块"任务)

| 语义 | 仿真(现状,Isaac HDF5) | 真机(建议,FR3+GELLO → 统一 HDF5) |
| --- | --- | --- |
| 动作(标签) | `actions (T,7)`:EE 增量 Δxyz+Δrpy+夹爪(±1) | 由 GELLO 关节目标 FK 差分换算为同语义 `(T,7)` |
| 关节观测 | `obs/joint_pos (T,9)`(相对默认构型) | `/joint_states`(绝对)→ 减默认构型 |
| 末端观测 | `obs/eef_pos/eef_quat` | 标定后 FK 计算(相机/运动学误差会引入噪声) |
| 夹爪观测 | `obs/gripper_pos (T,2)` | 夹爪编码器读数 |
| 相机 | 本任务无(Visuomotor 版有 `wrist_cam/table_cam`) | RealSense 30 fps(可选做 UMI 式腕部视角) |
| 物体状态 | `obs/cube_positions/orientations`(真值) | ⚠️ 无真值;需标定 + 视觉估计(或仅作观测不做监督) |
| 状态真值 | `states/`、`initial_state/` 全量物理真值 | ⚠️ 无;仅关节/速度读数 |
| 成功标记 | 自动(只导出成功) | 人工标注键 + 事后确认 |
| 记录频率 | 20 Hz 固定 | 录制 50 Hz(从 1 kHz 控制回路降采样) |

> 结论:仿真数据多出的是"**真值、自动标注、精确同步**";真机数据多出的是"**真实物理与真实场景**"。
> 统一格式的目的就是让两者在训练时**混着用**(共训),各取所长。

### 6.4 混合策略与本工作区迁移路线

**为什么混合**:Mobile ALOHA 已经证明"少量真机演示 + 已有数据共训"效果显著;
更普遍的范式是"**仿真大规模预训练(便宜、可控) → 真机数据微调/评估(真实、发版)**"。

**本工作区现状(可以复用的东西)**:

```
[已有] 仿真遥操全链路:teleop run/record/replay → HDF5 → 统计/可视化 → Mimic 增广 → BC 训练/评估
[已有] ROS 2 桥:IsaacSim-ros_workspaces(Humble;Python 3.12/3.10 双构建;DDS 双向验证通过)
      → 真机/外部栈通信(自定义消息、h1 控制器等);MoveIt/Nav2 运行期依赖待装(ops/docs/isaacsim_ros_workspaces-setup.md)
[已有] 数据工具:HDF5 质检/绘图/合并/视频导出(hdf5-format-guide.md 全套)
```

**若上真机,缺口清单(按顺序准备)**:

1. **设备**:Franka FR3 + GELLO(Franka 版)【最贴合本工作区任务】,或 SO-101 套件【最省钱】,或 ALOHA 套件【学术标准】;
2. **机器人控制接口**:FR3 → `franka_ros2`/FCI;UR → RTDE;确保有 E-stop 与速度限制配置;
3. **录制栈**:rosbag2(或直接接 LeRobot `lerobot-record`);相机同步(全局快门 + 软触发);
4. **转换脚本**:bag → robomimic HDF5(§5.4 的映射表),动作空间对齐到与仿真相同(EE 增量+夹爪);
5. **质检**:转换后直接跑本工作区 `teleop_dataset_stats.py` / `hdf5_plot_demo.py`;
6. **闭环**:先回放验证 → 少量真机演示微调 BC/ACT → 真机评估。

---

## 7. 参考资料

**项目官方站点/仓库(2026-10-12 核实)**

- GELLO:项目页 <https://wuphilipp.github.io/gello_site/> · 代码 <https://github.com/wuphilipp/gello_software>(MIT;含 FR3/ROS2 支持)
- ALOHA 2:<https://aloha-2.github.io/>(硬件开源 + MuJoCo 模型;Trossen Robotics 套件)
- Mobile ALOHA:<https://mobile-aloha.github.io/>(CoRL 2024;数据集与 act-plus-plus 代码)
- UMI:<https://umi-gripper.github.io/>(RSS 2024;硬件指南与数据采集教程)
- Open-TeleVision:arXiv 2407.01512 · <https://robot-tv.github.io/>
- LeRobot 数据集格式 v3.0:<https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3>
- robomimic 数据集规范:<https://robomimic.github.io/docs/datasets/overview.html>

**本工作区**

- 仿真遥操实操:`IsaacDataCollection/docs/teleoperation-hands-on-franka-stack-cube.md`
- 采集方式全景:`IsaacDataCollection/docs/robot-data-collection-in-isaac-survey.md`
- 仿真 HDF5 详解:`IsaacDataCollection/docs/hdf5-format-guide.md`
- ROS 2 桥:`ops/docs/isaacsim_ros_workspaces-setup.md`(构建/运行/待装依赖)
- 工具脚本:`IsaacDataCollection/scripts/teleop_dataset_stats.py`、`IsaacDataCollection/scripts/hdf5_plot_demo.py`

---

*文档:2026-10-12 · isaac 工作区 · 真机设备数据引自各官方站点;效率/成本数字标注了来源或量级;仿真侧为工作区实测。*
