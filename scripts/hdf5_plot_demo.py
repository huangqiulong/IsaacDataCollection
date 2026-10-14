#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""遥操作 HDF5 数据集快速可视化:把一条演示画成曲线图(PNG, 无需启动 Isaac)。

读取 Isaac Lab 录制器导出的 HDF5 演示数据集, 对指定 demo 绘制 4 张子图:
  1. actions(6 维增量曲线 + 夹爪通道);
  2. obs/joint_pos(关节角, 相对默认构型);
  3. obs/eef_pos(末端位置);
  4. 末端 XY 平面轨迹(俯视图, 标出起止点)。

用法:
    source ~/code/isaac/isaaclab/.venv/bin/activate
    python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
    python ~/code/isaac/IsaacDataCollection/scripts/hdf5_plot_demo.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5 \
        --demo demo_1 --out /tmp/demo_1.png

依赖: h5py + numpy + matplotlib(isaaclab venv 均已内置)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt


def _setup_fonts() -> None:
    """自动选择系统中可用的中文字体(避免图内文字变方框)。"""
    from matplotlib import font_manager

    prefer = [
        "Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK TC", "Noto Sans CJK JP",
        "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UMing CN",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [n for n in prefer if n in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 HDF5 演示数据中的一条 demo")
    parser.add_argument("dataset", type=str, help="HDF5 数据集路径")
    parser.add_argument("--demo", type=str, default="demo_0", help="演示组名(默认 demo_0)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 PNG 路径(默认: <数据集名>_<demo>.png, 与数据集同目录)",
    )
    return parser.parse_args()


def step_dt(f: h5py.File) -> float:
    """从 env_args 推算每步时长(秒);失败则回退 0.05(20 Hz)。"""
    try:
        env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
        sim = env_args.get("sim_args", {})
        return float(sim.get("dt", 0.01)) * float(sim.get("decimation", 5))
    except Exception:
        return 0.05


def main() -> None:
    args = parse_args()
    _setup_fonts()
    path = Path(args.dataset).expanduser()
    if not path.is_file():
        raise SystemExit(f"[错误] 文件不存在: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f or args.demo not in f["data"]:
            demos = list(f["data"].keys()) if "data" in f else []
            raise SystemExit(f"[错误] 找不到 {args.demo};可选: {demos}")
        g = f["data"][args.demo]
        dt = step_dt(f)
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}
        actions = np.asarray(g["actions"]) if "actions" in g else None
        joint = np.asarray(g["obs/joint_pos"]) if "obs/joint_pos" in g else None
        eef = np.asarray(g["obs/eef_pos"]) if "obs/eef_pos" in g else None

    n = max(len(x) for x in (actions, joint, eef) if x is not None)
    t = np.arange(n) * dt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{path.name} · {args.demo} · {attrs} · {dt * 1000:.0f} ms/step", fontsize=10)

    ax = axes[0, 0]
    if actions is not None:
        for i, lab in enumerate(["dx", "dy", "dz", "drx", "dry", "drz"]):
            ax.plot(t[: len(actions)], actions[:, i], lw=0.9, label=lab)
        ax.plot(t[: len(actions)], actions[:, 6], lw=0.9, ls="--", label="grip")
        ax.legend(ncol=4, fontsize=7)
    else:
        ax.text(0.5, 0.5, "no actions", ha="center")
    ax.set_title("actions(相对增量命令)")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if joint is not None:
        for j in range(joint.shape[1]):
            ax.plot(t[: len(joint)], joint[:, j], lw=0.8, label=f"j{j}")
        ax.legend(ncol=5, fontsize=7)
    else:
        ax.text(0.5, 0.5, "no obs/joint_pos", ha="center")
    ax.set_title("obs/joint_pos(相对默认构型)")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if eef is not None:
        for i, lab in enumerate(["x", "y", "z"]):
            ax.plot(t[: len(eef)], eef[:, i], lw=0.9, label=lab)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no obs/eef_pos", ha="center")
    ax.set_title("obs/eef_pos(末端位置, m)")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if eef is not None:
        ax.plot(eef[:, 0], eef[:, 1], lw=1.0)
        ax.scatter(eef[0, 0], eef[0, 1], c="g", s=28, label="start", zorder=3)
        ax.scatter(eef[-1, 0], eef[-1, 1], c="r", s=28, label="end", zorder=3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no obs/eef_pos", ha="center")
    ax.set_title("末端 XY 轨迹(俯视)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out).expanduser() if args.out else path.with_name(f"{path.stem}_{args.demo}.png")
    fig.savefig(out, dpi=130)
    print(f"已保存: {out}")
    print(f"步数: {n} | 时长: {n * dt:.1f} s | 子图: actions / joint_pos / eef_pos / XY轨迹")


if __name__ == "__main__":
    main()
