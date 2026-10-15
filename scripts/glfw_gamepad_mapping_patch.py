#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""GLFW gamepad 映射表等长修补: 为 Xbox 蓝牙手柄(fw0509)补映射。

背景(2026-10-10 诊断 / 2026-10-12 执行):
  - "Xbox Wireless Controller"(蓝牙, 045e:0b13, fw0509) 在系统层完全正常
    (js0/event26 事件流已验证), 但 Isaac Sim 报 `Device name: None`。
  - 根因: Kit 内嵌 GLFW 3.4.0 的映射表里有同产品 (0b13) 的 fw0501/0505 条目,
    唯独没有 fw0509 → glfwJoystickIsGamepad() 返回 false → carb 忽略手柄
    ("Joystick with unknown remapping detected (will be ignored)")。
  - 修复: 把 fw0505 条目的 GUID 尾部 "05050000" 等长替换为 "09050000"
    (Xbox 官方手柄各固件按钮/轴布局一致, 可直接复用)。
    **零长度变化, 不影响其它数据; 自动备份, 可 revert。**

用法:
  python3 IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py check    # 只检查状态
  python3 IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py apply    # 备份 + 替换
  python3 IsaacDataCollection/scripts/glfw_gamepad_mapping_patch.py revert   # 从备份恢复

目标文件(两份副本都处理, 内容 md5 相同):
  - ~/isaacsim/kit/kernel/plugins/libcarb.windowing-glfw.plugin.so
  - ~/isaacsim/extscache/carb.windowing.plugins-*/bin/libcarb.windowing-glfw.plugin.so
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

GUID_OLD = b"050000005e040000130b000005050000"  # fw0505 条目(将被替换)
GUID_NEW = b"050000005e040000130b000009050000"  # fw0509(我们的手柄)
FW_OLD = b"05050000"  # GUID_OLD 尾部 8 字节
FW_NEW = b"09050000"

BACKUP_SUFFIX = ".bak-20261012"

TARGETS = [
    "/home/dillon/isaacsim/kit/kernel/plugins/libcarb.windowing-glfw.plugin.so",
] + sorted(
    glob.glob("/home/dillon/isaacsim/extscache/carb.windowing.plugins-*/bin/libcarb.windowing-glfw.plugin.so")
)


def status_of(path: str) -> str:
    """返回文件状态: missing / patched / needs-patch / ambiguous"""
    p = Path(path)
    if not p.is_file():
        return "missing"
    data = p.read_bytes()
    n_old = data.count(GUID_OLD)
    n_new = data.count(GUID_NEW)
    if n_new == 1 and n_old == 0:
        return "patched"
    if n_old == 1 and n_new == 0:
        return "needs-patch"
    return f"ambiguous(old={n_old}, new={n_new})"


def do_check() -> int:
    rc = 0
    for t in TARGETS:
        s = status_of(t)
        print(f"[check] {s:16s} {t}")
        if not (s in ("patched", "needs-patch") or s.startswith("ambiguous")):
            rc = 1
    return rc


def do_apply() -> int:
    rc = 0
    for t in TARGETS:
        s = status_of(t)
        if s == "missing":
            print(f"[apply] SKIP(missing) {t}")
            rc = 1
            continue
        if s == "patched":
            print(f"[apply] SKIP(already patched) {t}")
            continue
        if not s == "needs-patch":
            print(f"[apply] SKIP({s}) {t}")
            rc = 1
            continue

        src = Path(t)
        bak = Path(str(src) + BACKUP_SUFFIX)
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"[apply] 备份 -> {bak.name}")
        else:
            print(f"[apply] 备份已存在(保留): {bak.name}")

        data = src.read_bytes()
        off = data.find(GUID_OLD)
        # 仅替换该条目的 fw 尾部 8 字节(offset+24..+32), 不误伤别处同串
        tail = off + 24
        assert data[tail : tail + 8] == FW_OLD, "意外: GUID 尾部不匹配"
        patched = data[:tail] + FW_NEW + data[tail + 8 :]
        assert len(patched) == len(data), "意外: 长度变化"
        src.write_bytes(patched)

        after = status_of(t)
        print(f"[apply] {t}\n        -> 替换完成, 状态: {after}")
        if after != "patched":
            rc = 1
    return rc


def do_revert() -> int:
    rc = 0
    for t in TARGETS:
        bak = Path(t + BACKUP_SUFFIX)
        if not bak.exists():
            print(f"[revert] 无备份, 跳过: {t}")
            continue
        shutil.copy2(bak, t)
        print(f"[revert] 已从备份恢复: {t} (状态: {status_of(t)})")
    return rc


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("check", "apply", "revert"):
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    print(f"== glfw gamepad mapping patch :: {cmd} ==")
    for t in TARGETS:
        print(f"  目标: {t}")
    print()
    if cmd == "check":
        return do_check()
    if cmd == "apply":
        return do_apply()
    return do_revert()


if __name__ == "__main__":
    raise SystemExit(main())
