# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""Teleop 扩展: 键盘(yaw 增强) + 鼠标设备 + 手柄(带重置键) + 任务变体注册。

背景(为什么需要本模块):
  - Isaac Lab 3.0 的 ``teleop record`` 脚本只内置 keyboard/spacemouse 两种设备;
    官方 Se3Gamepad 存在, 但 record 脚本不认识 "gamepad" 这个名字;
  - 官方没有内置鼠标遥操作设备;
  - 录制脚本通过 ``env_cfg.teleop_devices.devices`` 字典解析 ``--teleop_device <名>``,
    因此本模块在**不修改上游 isaaclab 仓库**的前提下, 注册任务变体, 往该字典里
    加入 "keyboard"(yaw 增强) / "gamepad" / "mouse" 条目。

提供的内容:
  1. ``Se3Mouse`` —— 自定义鼠标 SE(3) 遥操作设备(见下表):
       - 按住左键拖动: XY 平移(屏幕上=+x, 屏幕右=+y)
       - 按住右键拖动: 姿态(屏幕上=+pitch, 屏幕右=+yaw)
       - 按住中键拖动: 翻滚(roll)
       - 滚轮: Z 上下
       - 键盘 K: 开合夹爪; 键盘 R: 触发录制脚本的"重置"回调
     方向如不合直觉: 调整 ``_on_mouse_event`` 中各系数的符号即可。
  2. ``Se3GamepadWithReset`` —— 官方 Se3Gamepad 小包装:
     录制脚本传入的回调键是字符串 "R"(重置), 官方手柄设备只认
    ``carb.input.GamepadInput`` 枚举, 这里把 "R"/"RESET" 译码到手柄 **Y 键**;
    并支持轴方向翻转(``invert_axes`` / 环境变量 ``ISAACLAB_GAMEPAD_INVERT``,
    默认翻转 "pan_y" = 左摇杆左右 —— 2026-10-12 用户实测反馈)与
    **yaw(绕 Z 轴, 右摇杆左右)速度增益**(``yaw_boost`` / 环境变量
    ``ISAACLAB_GAMEPAD_YAW_BOOST``, 默认 ×3)。
  3. ``Se3KeyboardYawBoost`` —— 官方 Se3Keyboard 小包装: 把 C/V(绕 Z 轴 yaw)
    的按键增量为默认的 ``yaw_boost`` 倍(默认 3.0; 环境变量
    ``ISAACLAB_KEYBOARD_YAW_BOOST`` 可覆盖)。其余键与手感不变。
  4. ``register_task_variants()`` —— 注册下列任务变体(在原 cfg 基础上增加
     keyboard(增强)/gamepad/mouse 条目):
       - ``IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop``
       - ``IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop``

用法(推荐经配套启动器):
    ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \\
        --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \\
        --teleop_device gamepad --visualizer kit \\
        --num_demos 10 --dataset_file ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5

状态说明(2026-10-14 更新):
  - Se3GamepadWithReset / Se3KeyboardYawBoost / 任务变体: 基于官方类包装, 风险低;
  - 手柄实测(2026-10-12): 蓝牙手柄已可识别(GLFW 映射修补后); 已按反馈默认翻转
    "pan_y"(左摇杆左右); 其它轴若也需翻转: ISAACLAB_GAMEPAD_INVERT="pan_y,roll" 等;
    yaw(绕 Z 轴)已按需求加速 ×3(ISAACLAB_GAMEPAD_YAW_BOOST 可调);
  - 灵敏度(2026-10-14): 手柄移动/旋转灵敏度整体 ×2(pos/rot 0.05 → 0.10, 见
    _add_teleop_devices; yaw 叠加 ×3 后为 0.30); 键盘/鼠标灵敏度不变;
  - Se3Mouse: 尚未上机实测。若鼠标拖动无反应, 优先检查 Isaac 窗口是否获得焦点;
    若方向不对, 调整下方符号。
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable

import carb
import numpy as np
import omni
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, DeviceCfg, DevicesCfg
from isaaclab.devices.gamepad.se3_gamepad import Se3Gamepad
from isaaclab.devices.gamepad.se3_gamepad_cfg import Se3GamepadCfg
from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard
from isaaclab.devices.keyboard.se3_keyboard_cfg import Se3KeyboardCfg
from isaaclab.utils import configclass

# 任务 cfg 基类(供下方变体类继承; 启动器在 App 启动前 import 本模块, 此处 import
# 已验证安全——仅导入 cfg 类, 不触达 Kit 运行时)
from isaaclab_tasks.contrib.stack.config.franka import stack_ik_rel_env_cfg as _stack_base_cfg
from isaaclab_tasks.contrib.stack.config.franka import stack_ik_rel_visuomotor_env_cfg as _stack_visuo_cfg

# =====================================================================================
# 1) 自定义鼠标设备
# =====================================================================================


class Se3Mouse(DeviceBase):
    """鼠标 SE(3) 遥操作设备(相对位姿增量, 与键盘/手柄语义一致)。

    操作方式:
        - 按住左键拖动: XY 平移 (屏幕上=+x, 屏幕右=+y)
        - 按住右键拖动: pitch / yaw (屏幕上=+pitch, 屏幕右=+yaw)
        - 按住中键拖动: roll
        - 滚轮: Z 上下
        - 键盘 K: 开合夹爪
        - 键盘 R: 触发注册到 "R" 的回调(录制脚本中 = 重开当前演示)

    说明: 命令是"逐帧增量"(速度式)——鼠标不动, 机器人即停;
    与 Se3Keyboard/Se3Gamepad 的手感一致。需要 GUI 窗口(appwindow)可用。
    """

    def __init__(self, cfg: Se3MouseCfg):
        # ------- 配置 -------
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.scroll_sensitivity = cfg.scroll_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        # ------- Omniverse 输入接口 -------
        import carb.input  # noqa: F401  (显式加载原生子模块; App 运行时幂等)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._mouse = self._appwindow.get_mouse()
        self._mouse_sub = self._input.subscribe_to_mouse_events(
            self._mouse,
            lambda event, *args, obj=weakref.proxy(self): obj._on_mouse_event(event, *args),
        )
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = None
        if self._keyboard is not None:
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
        # ------- 命令缓冲 -------
        self._delta_pos = np.zeros(3)  # (x, y, z) [m]
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw) [rad]
        self._close_gripper = False
        self._last_pixel: tuple[float, float] | None = None  # 鼠标位置锚点
        self._left_down = False
        self._right_down = False
        self._middle_down = False
        # ------- 附加回调(录制脚本注册 "R" 等) -------
        self._additional_callbacks: dict[str, Callable] = {}

    def __del__(self):
        """取消订阅。"""
        try:
            if getattr(self, "_mouse_sub", None) is not None:
                self._input.unsubscribe_to_mouse_events(self._mouse, self._mouse_sub)
                self._mouse_sub = None
            if getattr(self, "_keyboard_sub", None) is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
                self._keyboard_sub = None
        except Exception:
            pass

    def __str__(self) -> str:
        msg = f"Mouse Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\tMove XY (hold Left Button + move): screen up=+x, screen right=+y\n"
        msg += "\tRotate pitch/yaw (hold Right Button + move)\n"
        msg += "\tRotate roll (hold Middle Button + move)\n"
        msg += "\tMove Z (mouse wheel)\n"
        msg += "\tToggle gripper: K\n"
        msg += "\tReset recording instance: R\n"
        return msg

    """
    Operations
    """

    def reset(self):
        """清空内部状态(env 重置时调用)。"""
        self._close_gripper = False
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
        self._last_pixel = None
        self._left_down = False
        self._right_down = False
        self._middle_down = False

    def add_callback(self, key: str, func: Callable):
        """注册附加回调(键名字符串, 与 Se3Keyboard 语义一致)。"""
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """返回自上次调用以来累积的命令, 并清零(速度式增量)。

        Returns:
            7 维张量: [x, y, z, rx, ry, rz, gripper] (米/弧度, gripper +1 开 / -1 合)
        """
        delta_pos = self._delta_pos.copy()
        delta_rot = self._delta_rot.copy()
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0

        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec])
        if self.gripper_term:
            command = np.append(command, -1.0 if self._close_gripper else 1.0)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """

    def _on_mouse_event(self, event, *args, **kwargs):
        """鼠标事件回调。"""
        et = event.type
        MouseEventType = carb.input.MouseEventType

        if et == MouseEventType.MOVE:
            px = event.pixel_coords  # 绝对像素坐标 Float2
            x, y = float(px[0]), float(px[1])
            if self._last_pixel is None:
                # 首次移动 / 刚按下按钮: 只做锚定, 不产生位移(避免跳变)
                self._last_pixel = (x, y)
                return True
            dx = x - self._last_pixel[0]
            dy = y - self._last_pixel[1]
            self._last_pixel = (x, y)
            if self._left_down:
                # 屏幕上移(dy<0) => +x; 屏幕右移 => +y  (方向不符时改这里的符号)
                self._delta_pos[0] += (-dy) * self.pos_sensitivity
                self._delta_pos[1] += (dx) * self.pos_sensitivity
            if self._right_down:
                self._delta_rot[1] += (-dy) * self.rot_sensitivity  # pitch
                self._delta_rot[2] += (dx) * self.rot_sensitivity  # yaw
            if self._middle_down:
                self._delta_rot[0] += (dx) * self.rot_sensitivity  # roll
        elif et == MouseEventType.LEFT_BUTTON_DOWN:
            self._left_down = True
            self._last_pixel = None
        elif et == MouseEventType.LEFT_BUTTON_UP:
            self._left_down = False
        elif et == MouseEventType.RIGHT_BUTTON_DOWN:
            self._right_down = True
            self._last_pixel = None
        elif et == MouseEventType.RIGHT_BUTTON_UP:
            self._right_down = False
        elif et == MouseEventType.MIDDLE_BUTTON_DOWN:
            self._middle_down = True
            self._last_pixel = None
        elif et == MouseEventType.MIDDLE_BUTTON_UP:
            self._middle_down = False
        elif et == MouseEventType.SCROLL:
            sd = event.scrollDelta  # Float2(水平, 垂直)
            self._delta_pos[2] += float(sd[1]) * self.scroll_sensitivity

        return True

    def _on_keyboard_event(self, event, *args, **kwargs):
        """键盘事件回调(仅夹爪 K 与附加回调键)。"""
        KeyboardEventType = carb.input.KeyboardEventType
        if event.type == KeyboardEventType.KEY_PRESS:
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        return True


@configclass
class Se3MouseCfg(DeviceCfg):
    """鼠标遥操作设备配置。"""

    gripper_term: bool = True
    pos_sensitivity: float = 0.0005  # [m / 像素]
    rot_sensitivity: float = 0.004  # [rad / 像素]
    scroll_sensitivity: float = 0.015  # [m / 滚轮步]
    class_type: type[DeviceBase] | str = Se3Mouse


# =====================================================================================
# 2) 手柄设备包装(把字符串回调键 "R"/"RESET" 译码到 Y 键)
# =====================================================================================


class Se3GamepadWithReset(Se3Gamepad):
    """官方 Se3Gamepad + 回调键别名 + 可配置的轴方向翻转。

    回调别名: 录制脚本用字符串 "R"/"START"/"STOP"/"RESET" 注册回调; 官方手柄
    设备只认 ``carb.input.GamepadInput`` 枚举键。这里把:
      - "R" / "RESET"  -> Y 键(重新开始当前演示)
      - 其余字符串      -> 忽略(录制流程自动开始, 无需 START/STOP)

    轴翻转: ``cfg.invert_axes`` 中的轴的按压方向取反(交换方向槽 direction 0<->1)。
    可用轴名: "pan_x"(左摇杆上下) "pan_y"(左摇杆左右) "pan_z"(右摇杆上下)
              "yaw"(右摇杆左右) "pitch"(D-Pad 上下) "roll"(D-Pad 左右)

    yaw 增益: ``cfg.yaw_boost``(默认 3.0) 把绕 Z 轴旋转(右摇杆左右)的绑定值放大,
    与键盘的 Se3KeyboardYawBoost 对称; 其余轴不受影响。
    """

    _AXIS_IDS = {"pan_x": 0, "pan_y": 1, "pan_z": 2, "roll": 3, "pitch": 4, "yaw": 5}

    def __init__(self, cfg: Se3GamepadWithResetCfg):
        # 必须在 super().__init__ 之前设置 —— 基类构造会调用 _create_key_bindings()
        self._invert_axes = set(getattr(cfg, "invert_axes", ()) or ())
        self._yaw_boost = float(getattr(cfg, "yaw_boost", 1.0) or 1.0)
        super().__init__(cfg)

    def _create_key_bindings(self):
        super()._create_key_bindings()
        maps = (self._INPUT_STICK_VALUE_MAPPING, self._INPUT_DPAD_VALUE_MAPPING)

        # ① yaw(绕 Z 轴, axis=5)增益 —— 直接放大绑定值(正值放大, 无 max 吞值问题)
        if self._yaw_boost != 1.0:
            for mapping in maps:
                for key, (direction, axis, value) in list(mapping.items()):
                    if axis == 5:
                        mapping[key] = (direction, axis, value * self._yaw_boost)

        # ② 方向翻转: 必须交换方向槽(direction 0<->1), **不能**对 value 取反!
        # 官方 _resolve_command_buffer 用 max(raw[0], raw[1]) 取量值、用
        # (raw[1] > raw[0]) 判符号 —— 负值会被 max 吞掉, 导致该轴两个方向都无响应。
        # (2026-10-12 教训: 取反 value 的写法由用户实测发现"左右无响应"后修正)
        invert_ids = {self._AXIS_IDS[a] for a in self._invert_axes if a in self._AXIS_IDS}
        if not invert_ids:
            return
        for mapping in maps:
            for key, (direction, axis, value) in list(mapping.items()):
                if axis in invert_ids:
                    mapping[key] = (1 - direction, axis, value)

    # 注意: ``carb.input`` 是 Kit 原生子模块, 只有 App 运行时才可访问; 本模块会
    # 在 App 启动前被 import(启动器流程), 因此别名表必须**惰性构建**, 不能在
    # 类体中求值。(2026-10-12 修复: 类体求值会报 `carb has no attribute 'input'`)
    @staticmethod
    def _resolve_button_alias(key):
        import carb.input  # noqa: F401  (显式加载原生子模块; App 运行时幂等)

        aliases = {
            "R": carb.input.GamepadInput.Y,
            "RESET": carb.input.GamepadInput.Y,
        }
        return aliases.get(key)

    def add_callback(self, key, func):
        if isinstance(key, str):
            key = self._resolve_button_alias(key)
            if key is None:
                return
        super().add_callback(key, func)


@configclass
class Se3GamepadWithResetCfg(Se3GamepadCfg):
    """与 Se3GamepadCfg 相同, 但实例化为带 Y=重置 且可翻转轴方向的版本。"""

    # 需要方向翻转的轴(2026-10-12 起): 任意子集
    #   "pan_x"(左摇杆上下) "pan_y"(左摇杆左右) "pan_z"(右摇杆上下)
    #   "yaw"(右摇杆左右)  "pitch"(D-Pad 上下) "roll"(D-Pad 左右)
    # 启动器会用环境变量 ISAACLAB_GAMEPAD_INVERT(逗号分隔)覆盖本默认值
    invert_axes: tuple = ("pan_y",)
    # 绕 Z 轴旋转(右摇杆左右)的速度倍率(2026-10-12 起, 与键盘增强对称);
    # 启动器用环境变量 ISAACLAB_GAMEPAD_YAW_BOOST 覆盖(默认 3.0)
    yaw_boost: float = 3.0
    class_type: type[DeviceBase] | str = Se3GamepadWithReset


# =====================================================================================
# 3) 键盘 yaw(绕 Z 轴)增强
# =====================================================================================


class Se3KeyboardYawBoost(Se3Keyboard):
    """官方 Se3Keyboard + 放大的 yaw(绕 Z 轴, C/V 键)灵敏度。

    官方键盘把 C/V 的增量绑定为 ``rot_sensitivity``(如 0.05 rad/次), 与其它
    旋转键相同; 实操中绕 Z 轴的转动往往偏慢。本类把 C/V 的绑定值乘
    ``cfg.yaw_boost``(默认 3.0), 其余按键(平移/roll/pitch/夹爪)不变。
    """

    def __init__(self, cfg: Se3KeyboardYawBoostCfg):
        # 必须在 super().__init__ 之前设置 —— 基类构造会调用 _create_key_bindings()
        self._yaw_boost = float(getattr(cfg, "yaw_boost", 3.0) or 1.0)
        super().__init__(cfg)

    def _create_key_bindings(self):
        super()._create_key_bindings()
        for key in ("C", "V"):
            self._INPUT_KEY_MAPPING[key] = self._INPUT_KEY_MAPPING[key] * self._yaw_boost

    def __str__(self) -> str:
        msg = super().__str__()
        if self._yaw_boost != 1.0:
            msg += f"\t(yaw boost active: x{self._yaw_boost:g} on C/V)\n"
        return msg


@configclass
class Se3KeyboardYawBoostCfg(Se3KeyboardCfg):
    """与 Se3KeyboardCfg 相同, 另加 yaw_boost(yaw 倍率, 默认 3.0)。"""

    yaw_boost: float = 3.0
    class_type: type[Se3Keyboard] | str = Se3KeyboardYawBoost


# =====================================================================================
# 4) 任务变体注册
# =====================================================================================

_VARIANT_IDS = (
    "IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop",
    "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop",
)
_REGISTERED = False


def _add_teleop_devices(env_cfg) -> None:
    """向任务 cfg 的 teleop_devices 中加入 keyboard(增强) / gamepad / mouse 条目。"""
    # 部分任务(如 Visuomotor)的 cfg 没有 teleop_devices 字段 —— 先补齐再填充
    if getattr(env_cfg, "teleop_devices", None) is None:
        env_cfg.teleop_devices = DevicesCfg(devices={})
    devices = env_cfg.teleop_devices.devices
    devices["keyboard"] = Se3KeyboardYawBoostCfg(
        pos_sensitivity=0.05,
        rot_sensitivity=0.05,
        yaw_boost=float(os.environ.get("ISAACLAB_KEYBOARD_YAW_BOOST", "3.0")),
        sim_device=env_cfg.sim.device,
    )
    # 手柄轴方向翻转(2026-10-12 用户实测反馈: 左右反向 → 默认翻转 pan_y)
    _gp_invert = os.environ.get("ISAACLAB_GAMEPAD_INVERT", "pan_y")
    devices["gamepad"] = Se3GamepadWithResetCfg(
        # 移动/旋转灵敏度(2026-10-14 用户需求: 整体 ×2, 0.05 → 0.10; 覆盖平移三轴 +
        # yaw/roll/pitch; yaw 再叠加下方 ×3 增益, 详见模块状态说明)
        pos_sensitivity=0.10,
        rot_sensitivity=0.10,
        invert_axes=tuple(a.strip() for a in _gp_invert.split(",") if a.strip()),
        # 绕 Z 轴旋转(右摇杆左右)加速(2026-10-12 用户需求: ×3)
        yaw_boost=float(os.environ.get("ISAACLAB_GAMEPAD_YAW_BOOST", "3.0")),
        sim_device=env_cfg.sim.device,
    )
    devices["mouse"] = Se3MouseCfg(sim_device=env_cfg.sim.device)


# —— 任务变体 cfg 类(必须放模块顶层: gym 以 "模块名:类名" 字符串解析) ——
# (2026-10-12 修复: 此前定义在 register_task_variants() 内部, 环境实例化时
#  报 `AttributeError: module 'isaaclab_teleop_extras' has no attribute ...`)


class FrankaCubeStackTeleopEnvCfg(_stack_base_cfg.FrankaCubeStackEnvCfg):
    """摞方块(IK-Rel) + keyboard(增强)/gamepad/mouse 遥操作设备。"""

    def __post_init__(self):
        super().__post_init__()
        _add_teleop_devices(self)


class FrankaCubeStackVisuomotorTeleopEnvCfg(_stack_visuo_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """摞方块(Visuomotor) + keyboard(增强)/gamepad/mouse 遥操作设备。"""

    # Visuomotor 基类没有 teleop_devices 字段 —— 显式声明(默认 None),
    # 由 _add_teleop_devices 在 __post_init__ 中填充。
    # (2026-10-12 修复: 此前直接访问会 AttributeError)
    teleop_devices: DevicesCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        _add_teleop_devices(self)


def register_task_variants() -> None:
    """注册带 keyboard(增强)/gamepad/mouse 设备的任务变体(幂等)。"""
    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    gym.register(
        id="IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}:FrankaCubeStackTeleopEnvCfg",
            "robomimic_bc_cfg_entry_point": (
                "isaaclab_tasks.contrib.stack.config.franka.agents:robomimic/bc_rnn_low_dim.json"
            ),
        },
        disable_env_checker=True,
    )
    gym.register(
        id="IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Teleop",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}:FrankaCubeStackVisuomotorTeleopEnvCfg",
            "robomimic_bc_cfg_entry_point": (
                "isaaclab_tasks.contrib.stack.config.franka.agents:robomimic/bc_rnn_image_200.json"
            ),
        },
        disable_env_checker=True,
    )
    _REGISTERED = True
