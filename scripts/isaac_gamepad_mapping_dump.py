#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""手柄键位映射导出 + 轴方向翻转自检(无头 Kit, 约 1 分钟)。

用途:
  1) 导出 ``Se3GamepadWithReset`` 的完整按键映射(权威表, 直接来自设备对象);
  2) 验证轴方向翻转(``invert_axes``)生效:
     - 对比 ``invert_axes=()`` 与 ``invert_axes=("pan_y",)`` 两个实例;
     - 左摇杆左右(pan_y) 的绑定值符号应相反; 其它轴应完全一致。

说明: 设备构造不依赖手柄是否在线(仅事件订阅), 无手柄也能跑通本自检。
用法:
  cd ~/code/isaac/isaaclab && source .venv/bin/activate
  OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaac_gamepad_mapping_dump.py
"""

from __future__ import annotations

import sys

# 让 isaaclab_teleop_extras 可被 import (位于 IsaacDataCollection/scripts/)
sys.path.insert(0, "/home/dillon/code/isaac/IsaacDataCollection/scripts")

from isaaclab.app import AppLauncher  # noqa: E402

# 必须 enable_cameras=True(headless.rendering experience 才含 omni.appwindow)
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import omni  # noqa: E402

_i = 0
for _i in range(600):  # noqa: B007
    if hasattr(omni, "appwindow"):
        break
    simulation_app.update()
else:
    raise RuntimeError("omni.appwindow 在等待窗口内未就绪")

import carb.input  # noqa: E402,F401
import numpy as np  # noqa: E402

from isaaclab_teleop_extras import Se3GamepadWithReset, Se3GamepadWithResetCfg  # noqa: E402

AXIS_NAMES = {0: "x(前后)", 1: "y(左右)", 2: "z(升降)", 3: "roll", 4: "pitch", 5: "yaw"}

# 物理按压 -> 实际增量符号: direction=0 为正方向按压, direction=1 为负方向按压;
# 生效值 = (±value) 由 (direction, value) 组合决定, 这里归一化为“按压方向的实际符号”
def press_sign(direction: int, value: float) -> str:
    return "+" if ((direction == 0 and value >= 0) or (direction == 1 and value < 0)) else "-"


def make(invert=(), yaw_boost=1.0):
    return Se3GamepadWithReset(
        Se3GamepadWithResetCfg(
            pos_sensitivity=0.05, rot_sensitivity=0.05, invert_axes=invert, yaw_boost=yaw_boost
        )
    )


def dump(dev, title: str) -> None:
    print(f"\n== {title} ==")
    for key, (direction, axis, value) in dev._INPUT_STICK_VALUE_MAPPING.items():
        print(f"  [stick] {key.name:20s} axis={AXIS_NAMES[axis]:8s} 按压方向={press_sign(direction, value)}  value={value:+.4f}")
    for key, (direction, axis, value) in dev._INPUT_DPAD_VALUE_MAPPING.items():
        print(f"  [dpad ] {key.name:20s} axis={AXIS_NAMES[axis]:8s} 按压方向={press_sign(direction, value)}  value={value:+.4f}")


def simulate_press(dev, key_name: str, cur_val: float = 1.0):
    """模拟一次按压, 返回(轴, 经官方 _resolve_command_buffer 解析后的命令值)。

    端到端校验翻转是否真的生效——只检查绑定值符号不够(会被解析逻辑吞掉)。
    """
    for mapping, is_stick in ((dev._INPUT_STICK_VALUE_MAPPING, True), (dev._INPUT_DPAD_VALUE_MAPPING, False)):
        for key, (direction, axis, value) in mapping.items():
            if key.name != key_name:
                continue
            raw = np.zeros((2, 6))
            if is_stick:
                raw[direction, axis] = value * cur_val
            else:
                raw[direction, axis] = value
                raw[1 - direction, axis] = 0
            resolved = np.concatenate(
                [dev._resolve_command_buffer(raw[:, :3]), dev._resolve_command_buffer(raw[:, 3:])]
            )
            return axis, float(resolved[axis])
    raise KeyError(key_name)


def main() -> int:
    dev_plain = make()  # 基准: 不翻转/不加速
    dev_default = make(("pan_y",), 3.0)  # 启动器默认: pan_y 翻转 + yaw ×3

    dump(dev_plain, "基准(不翻转/不加速)")
    dump(dev_default, "启动器默认(pan_y 翻转 + yaw ×3)")

    # —— 端到端验证: 模拟按压 -> 经官方 _resolve_command_buffer 解析后的命令值 ——
    # (2026-10-12 教训: 不能只检查绑定值符号——必须走一遍解析流向!)
    # 期望(相对基准): LEFT_STICK_* 等幅反向; RIGHT_STICK_* ×3; 其余不变
    ok = True
    print("\n== 模拟按压 -> 解析后的命令值 (基准 vs 启动器默认) ==")
    for key_name in (
        "LEFT_STICK_UP",
        "LEFT_STICK_DOWN",
        "LEFT_STICK_RIGHT",
        "LEFT_STICK_LEFT",
        "RIGHT_STICK_UP",
        "RIGHT_STICK_DOWN",
        "RIGHT_STICK_RIGHT",
        "RIGHT_STICK_LEFT",
        "DPAD_UP",
        "DPAD_DOWN",
        "DPAD_RIGHT",
        "DPAD_LEFT",
    ):
        a1, c1 = simulate_press(dev_plain, key_name)
        a2, c2 = simulate_press(dev_default, key_name)
        if key_name in ("LEFT_STICK_LEFT", "LEFT_STICK_RIGHT"):
            good = (a1 == a2) and abs(c1 + c2) < 1e-9 and abs(c1) > 1e-6  # 等幅反向
            note = "翻转"
        elif key_name in ("RIGHT_STICK_LEFT", "RIGHT_STICK_RIGHT"):
            good = (a1 == a2) and abs(c2 - 3.0 * c1) < 1e-9  # ×3
            note = "yaw×3"
        else:
            good = (a1 == a2) and abs(c1 - c2) < 1e-12
            note = "不变"
        if not good:
            ok = False
        print(f"  [{'OK ' if good else 'FAIL'}] {key_name:20s} 基准={c1:+.4f}  默认={c2:+.4f}  ({note})")

    print("\n[gamepad] GAMEPAD-MAPPING-PROBE-" + ("OK" if ok else "FAIL"))
    sys.stdout.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
