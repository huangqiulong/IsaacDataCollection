#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""键盘 yaw 增强数值自检(无头 Kit, 一次约 1 分钟)。

验证 Se3KeyboardYawBoost(IsaacDataCollection/scripts/isaaclab_teleop_extras.py):
  - C/V(绕 Z 轴 yaw)的按键增量 = rot_sensitivity * yaw_boost
        (默认 0.05 × 3.0 = 0.15 rad/次)
  - 其它键不受影响 (Z=rot_sensitivity, W=pos_sensitivity)
  - 设备 __str__ 显示 "(yaw boost active: x3.0 on C/V)"

用法:
  cd ~/code/isaac/isaaclab && source .venv/bin/activate
  OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaac_keyboard_yaw_probe.py
  # 自定义倍率(与环境变量机制联动验证):
  ISAACLAB_KEYBOARD_YAW_BOOST=5.0 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p <本脚本>
"""

from __future__ import annotations

import os
import sys

# 让 isaaclab_teleop_extras 可被 import (位于 IsaacDataCollection/scripts/)
sys.path.insert(0, "/home/dillon/code/isaac/IsaacDataCollection/scripts")

from isaaclab.app import AppLauncher  # noqa: E402

# 注意: 必须 enable_cameras=True —— 这会选用 headless.rendering experience;
# 若为 False 则用 headless.kit, 其中不含 omni.appwindow(设备无法构造)。
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# —— 等待 Kit 扩展加载完成(appwindow 等由扩展注入 omni 命名空间, 并非可 import 模块) ——
import omni  # noqa: E402

_i = 0
for _i in range(600):  # noqa: B007 最多 pump 600 帧
    if hasattr(omni, "appwindow"):
        break
    simulation_app.update()
else:
    raise RuntimeError("omni.appwindow 在等待窗口内未就绪")

# carb.input 是真正可 import 的原生子模块
import carb  # noqa: E402
import carb.input  # noqa: E402,F401

print(f"[probe] omni.appwindow ready (pumped {_i} frames); carb.input loaded: {hasattr(carb, 'input')}")

# —— Kit 就绪后再 import(Se3Keyboard 依赖 carb/omni) ——
import numpy as np  # noqa: E402

from isaaclab_teleop_extras import Se3KeyboardYawBoost, Se3KeyboardYawBoostCfg  # noqa: E402


def main() -> int:
    # 与启动器变体一致的参数; boost 允许环境变量覆盖(用于验证联动)
    boost = float(os.environ.get("ISAACLAB_KEYBOARD_YAW_BOOST", "3.0"))
    pos_s, rot_s = 0.05, 0.05

    dev = Se3KeyboardYawBoost(
        Se3KeyboardYawBoostCfg(pos_sensitivity=pos_s, rot_sensitivity=rot_s, yaw_boost=boost)
    )
    m = dev._INPUT_KEY_MAPPING

    print("[probe] boost =", boost)
    print("[probe] C =", m["C"], " V =", m["V"])
    print("[probe] Z =", m["Z"], " W =", m["W"])
    print("[probe] device str:")
    print(str(dev))

    ok = True
    ok &= bool(np.allclose(m["C"], [0.0, 0.0, rot_s * boost]))
    ok &= bool(np.allclose(m["V"], [0.0, 0.0, -rot_s * boost]))
    ok &= bool(np.allclose(m["Z"], [rot_s, 0.0, 0.0]))
    ok &= bool(np.allclose(m["W"], [pos_s, 0.0, 0.0]))
    ok &= bool(np.allclose(m["K"] if isinstance(m["K"], np.ndarray) else [1.0], [1.0]))  # 夹爪开关

    print("[probe] KEYBOARD-YAW-PROBE-" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
