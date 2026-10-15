#!/usr/bin/env python3
"""GLFW 手柄识别/映射诊断脚本（Xbox 蓝牙手柄 fw0509 缺映射假说的验证）。

背景（2026-10-10）：
  - 手柄 "Xbox Wireless Controller"（蓝牙, 045e:0b13, fw 0509）连接正常、事件正常，
    但 Isaac Sim 的 carb gamepad 层报 `Device name: None`。
  - Kit 内嵌 GLFW 3.4.0 的映射数据库里有同产品 (0b13) 的 fw 0501/0505 条目，
    唯独没有 fw 0509 → glfwJoystickIsGamepad() 大概率为 false → Kit 不创建 gamepad。
  - 本脚本用可 dlopen 的 libglfw 验证:
      1) GLFW 能否看到手柄（present=1, 名称/ GUID）
      2) is_gamepad 是否为 0（缺映射复现）
      3) glfwUpdateGamepadMappings 注入映射后 is_gamepad 是否变 1（修复可行性）

用法:
  .venv/bin/python IsaacDataCollection/scripts/glfw_gamepad_mapping_test.py [libglfw.so 路径]
  默认用 ~/isaacsim/exts/isaacsim.pip.newton/pip_prebundle/glfw/x11/libglfw.so
"""
from __future__ import annotations

import ctypes
import sys

DEFAULT_LIB = (
    "/home/dillon/isaacsim/exts/isaacsim.pip.newton/pip_prebundle/glfw/x11/libglfw.so"
)

# 期望 GUID（由 sysfs 推导: bustype=0005, vendor=045e, product=0b13, version=0509）
EXPECTED_GUID = "050000005e040000130b000009050000"

# 注入映射: 复用同产品 (0b13) 既有条目 "Xbox Series Controller" 的键位布局, GUID 换成 fw0509
MAPPING = (
    b"050000005e040000130b000009050000,Xbox Wireless Controller,"
    b"a:b0,b:b1,back:b10,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,"
    b"guide:b12,leftshoulder:b6,leftstick:b13,lefttrigger:a5,leftx:a0,lefty:a1,"
    b"rightshoulder:b7,rightstick:b14,righttrigger:a4,rightx:a2,righty:a3,"
    b"start:b11,x:b3,y:b4,platform:Linux,"
)

JOY = 0  # GLFW_JOYSTICK_1


def main() -> int:
    lib_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LIB
    lib = ctypes.CDLL(lib_path)
    print(f"lib: {lib_path}")

    lib.glfwInit.restype = ctypes.c_int
    print("glfwInit:", lib.glfwInit())

    lib.glfwJoystickPresent.argtypes = [ctypes.c_int]
    lib.glfwJoystickPresent.restype = ctypes.c_int
    lib.glfwGetJoystickName.argtypes = [ctypes.c_int]
    lib.glfwGetJoystickName.restype = ctypes.c_char_p
    lib.glfwGetJoystickGUID.argtypes = [ctypes.c_int]
    lib.glfwGetJoystickGUID.restype = ctypes.c_char_p
    lib.glfwJoystickIsGamepad.argtypes = [ctypes.c_int]
    lib.glfwJoystickIsGamepad.restype = ctypes.c_int

    present = lib.glfwJoystickPresent(JOY)
    print("present(0):", present)
    if not present:
        print("!! GLFW 看不到 0 号手柄（检查连接/权限）")
        return 1

    name = lib.glfwGetJoystickName(JOY)
    print("name(0):", name.decode() if name else None)
    guid = lib.glfwGetJoystickGUID(JOY)
    guid_s = guid.decode() if guid else None
    print("guid(0):", guid_s)
    print("expected:", EXPECTED_GUID, "(match)" if guid_s == EXPECTED_GUID else "(MISMATCH!)")
    is_pad = lib.glfwJoystickIsGamepad(JOY)
    print("is_gamepad(0):", is_pad, "  <- 0 表示当前库缺此 GUID 的映射")

    lib.glfwUpdateGamepadMappings.argtypes = [ctypes.c_char_p]
    lib.glfwUpdateGamepadMappings.restype = ctypes.c_int
    r = lib.glfwUpdateGamepadMappings(MAPPING)
    print("glfwUpdateGamepadMappings:", r)
    after = lib.glfwJoystickIsGamepad(JOY)
    print("is_gamepad(0) after inject:", after, "  <- 1 表示映射注入可修复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
