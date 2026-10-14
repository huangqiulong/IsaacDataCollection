#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""遥操作启动器: 注册 gamepad/mouse 任务变体后, 透传到官方脚本运行。

为什么需要它:
  Isaac Lab 3.0 的 ``teleop record`` 只内置 keyboard/spacemouse; 要用 Xbox 手柄
  (gamepad) 或鼠标(mouse)录制, 须先注册带相应 teleop_devices 的任务变体
  (见 ``isaaclab_teleop_extras.py``)。本脚本负责"先注册、再调官方脚本"。

用法(在任意目录; 建议在 isaaclab 目录下跑, 数据集相对路径即为 isaaclab/datasets):

    # 热身(不录制): 新手柄/鼠标先试手感
    ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py run \\
        --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \\
        --teleop_device gamepad --visualizer kit

    # 录制 10 条:
    ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py record \\
        --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \\
        --teleop_device gamepad --visualizer kit \\
        --num_demos 10 --dataset_file ./datasets/dataset_gamepad.hdf5

    # 回放(与设备无关, 但为保持习惯也支持):
    ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/isaaclab_teleop_launch.py replay \\
        --task IsaacContrib-Stack-Cube-Franka-IK-Rel-Teleop \\
        --visualizer kit --num_envs 1 --dataset_file ./datasets/dataset_gamepad.hdf5

可用 ``--teleop_device``: keyboard | spacemouse | gamepad | mouse
  - keyboard: 经本启动器时使用**增强版**(C/V 绕 Z 轴 yaw 灵敏度 ×3, 可用环境
    变量 ``ISAACLAB_KEYBOARD_YAW_BOOST`` 调整倍率); spacemouse: 官方内置;
  - gamepad/mouse: 需本启动器(因为要注册任务变体)。
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_TARGETS = {
    "run": "scripts/environments/teleoperation/teleop_se3_agent.py",
    "record": "scripts/tools/record_demos.py",
    "replay": "scripts/tools/replay_demos.py",
}


def _find_isaaclab_root() -> Path:
    env = os.environ.get("ISAACLAB_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # 默认: 本脚本位于 <workspace>/IsaacDataCollection/scripts/ 下, isaaclab 与其平级
    ws = Path(__file__).resolve().parents[2]
    cand = ws / "isaaclab"
    if (cand / "isaaclab.sh").is_file():
        return cand
    raise SystemExit(
        "找不到 isaaclab 根目录。请设置环境变量 ISAACLAB_ROOT, 或把本脚本保留在 <workspace>/IsaacDataCollection/scripts/ 下。"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _TARGETS:
        modes = "|".join(_TARGETS)
        print(f"用法: ./isaaclab.sh -p <本脚本> {{{modes}}} [官方脚本参数...]")
        raise SystemExit(2)

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # 注册任务变体(在任何 App/Kit 启动之前完成)
    import isaaclab_teleop_extras as extras  # noqa: E402

    extras.register_task_variants()

    target = _find_isaaclab_root() / _TARGETS[sys.argv[1]]
    if not target.is_file():
        raise SystemExit(f"目标脚本不存在: {target}")

    # 透传参数: 让官方脚本以 __main__ 方式运行(等价 `python <脚本> <参数...>`)
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
