#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""遥操作数据集统计与质量检查(仅依赖 h5py + numpy, 不需要启动 Isaac)。

读取 Isaac Lab 录制器导出的 HDF5 演示数据集(teleop record / Mimic 生成, 同 schema),
输出:
  1. 数据集概况(格式版本 / env_args / 演示条数 / 总步数);
  2. 每条演示的统计(steps / success / 位移累积 / 动作抖动 / 夹爪开合);
  3. 汇总统计与质量提示(阈值均为经验值, 用于快速筛查异常演示);
  4. obs(观测)键与形状一览(确认录制包含了哪些模态)。

指标说明(针对 IK-Rel 类任务: action = 末端相对位姿增量 + 夹爪):
  - pos_move : 位置增量模长累加 [m]  —— 末端"总行程"的近似值
  - rot_move : 旋转向量模长累加 [rad]
  - jerk     : 相邻步动作差(位置部分)的绝对值均值 —— 越小越平滑
  - grip     : 夹爪开合(动作末维负=闭合); toggles=开合切换次数(正常抓放 ≥2)

用法:
    source ~/code/isaac/isaaclab/.venv/bin/activate    # 或任意有 h5py 的 python
    python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5
    python ~/code/isaac/IsaacDataCollection/scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_keyboard.hdf5 --json ~/code/isaac/IsaacDataCollection/datasets/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MIN_REASONABLE_STEPS = 30  # 少于此步数疑似"过短演示"
MAX_REASONABLE_STEPS = 1200  # 多于此步数疑似"一次录了太多试错"
MIN_POS_MOVE = 0.05  # 末端总行程小于 5cm 疑似"几乎没动"

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _demo_sort_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 1 << 30


def _list_demos(data_group: h5py.Group) -> list[str]:
    demos = [k for k in data_group.keys() if k.startswith("demo_")]
    demos.sort(key=_demo_sort_key)
    return demos


def _collect_leaf_shapes(group: h5py.Group, prefix: str = "") -> dict[str, tuple]:
    """递归收集组内的叶子数据集 -> 形状。"""
    out: dict[str, tuple] = {}
    for key in group:
        path = f"{prefix}{key}"
        item = group[key]
        if isinstance(item, h5py.Group):
            out.update(_collect_leaf_shapes(item, path + "/"))
        else:
            out[path] = tuple(item.shape)
    return out


def _analyze_actions(actions: np.ndarray) -> dict:
    """对 (T, A) 的动作序列计算统计量。"""
    stats: dict = {"action_dim": int(actions.shape[1])}
    if actions.ndim != 2 or actions.shape[0] == 0:
        return stats

    stats["pos_move"] = float(np.linalg.norm(actions[:, 0:3], axis=1).sum())
    stats["pos_max"] = float(np.abs(actions[:, 0:3]).max())
    if actions.shape[1] >= 6:
        stats["rot_move"] = float(np.linalg.norm(actions[:, 3:6], axis=1).sum())
        stats["rot_max"] = float(np.abs(actions[:, 3:6]).max())
    if actions.shape[0] > 1:
        stats["jerk"] = float(np.abs(np.diff(actions[:, 0:3], axis=0)).mean())
    else:
        stats["jerk"] = 0.0
    if actions.shape[1] >= 7:
        grip = actions[:, 6]
        stats["grip_closed_ratio"] = float((grip < 0).mean())
        stats["grip_toggles"] = int((np.sign(grip[1:]) != np.sign(grip[:-1])).sum())
    return stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def analyze(dataset_path: Path) -> dict:
    report: dict = {"file": str(dataset_path), "demos": [], "warnings": []}

    with h5py.File(dataset_path, "r") as f:
        report["format_version"] = int(f.attrs.get("format_version", -1))
        if "data" not in f:
            raise SystemExit("[错误] 文件里没有 'data' 组, 不是 Isaac Lab 数据集格式?")
        data = f["data"]
        env_args_raw = data.attrs.get("env_args", "{}")
        try:
            env_args = json.loads(env_args_raw)
        except Exception:
            env_args = {"raw": str(env_args_raw)}
        report["env_args"] = env_args
        report["total_steps_attr"] = int(data.attrs.get("total", -1))

        demo_names = _list_demos(data)
        report["num_demos"] = len(demo_names)
        if not demo_names:
            raise SystemExit("[错误] 数据集里没有 demo_* 组(空数据集?)")

        # obs 结构(取第一条演示)
        first = data[demo_names[0]]
        if "obs" in first:
            report["obs_terms"] = _collect_leaf_shapes(first["obs"])

        for name in demo_names:
            g = data[name]
            row: dict = {
                "demo": name,
                "steps": int(g.attrs.get("num_samples", 0)),
            }
            success = g.attrs.get("success", None)
            row["success"] = None if success is None else bool(success)
            if "seed" in g.attrs:
                row["seed"] = int(g.attrs["seed"])
            if "actions" in g:
                actions = np.asarray(g["actions"])
                row.update(_analyze_actions(actions))
            report["demos"].append(row)

    # ---------------- 汇总 ----------------
    steps = np.array([d["steps"] for d in report["demos"]], dtype=float)
    pos_move = np.array([d.get("pos_move", np.nan) for d in report["demos"]])
    summary = {
        "num_demos": len(report["demos"]),
        "steps_mean": float(steps.mean()),
        "steps_std": float(steps.std()),
        "steps_min": int(steps.min()),
        "steps_max": int(steps.max()),
        "pos_move_mean": float(np.nanmean(pos_move)),
        "success_count": int(sum(1 for d in report["demos"] if d.get("success") is True)),
        "success_unknown": int(sum(1 for d in report["demos"] if d.get("success") is None)),
    }
    report["summary"] = summary

    # ---------------- 质量提示(经验阈值) ----------------
    w = report["warnings"]
    if summary["success_unknown"] > 0:
        w.append(f"{summary['success_unknown']} 条演示没有 success 标记(检查导出模式)。")
    if summary["success_count"] < summary["num_demos"] - summary["success_unknown"]:
        w.append("存在 success=False 的演示(正常录制不应导出失败演示)。")
    if summary["steps_std"] > 0.5 * max(summary["steps_mean"], 1.0):
        w.append(
            f"演示时长差异较大(steps {int(steps.min())}~{int(steps.max())}, std={steps.std():.0f})"
            "——检查是否有演示混入了过多试错动作。"
        )
    jerks = np.array([d.get("jerk", np.nan) for d in report["demos"]])
    med_jerk = float(np.nanmedian(jerks)) if np.isfinite(jerks).any() else np.nan
    for d in report["demos"]:
        if d["steps"] < MIN_REASONABLE_STEPS:
            w.append(f"{d['demo']}: 仅 {d['steps']} 步, 疑似过短。")
        if d["steps"] > MAX_REASONABLE_STEPS:
            w.append(f"{d['demo']}: {d['steps']} 步, 疑似过长(可能未在失败时按 R 重开)。")
        if np.isfinite(d.get("pos_move", np.nan)) and d["pos_move"] < MIN_POS_MOVE:
            w.append(f"{d['demo']}: 末端总行程仅 {d['pos_move'] * 100:.1f} cm, 疑似几乎未动。")
        if d.get("grip_toggles", 0) == 0:
            w.append(f"{d['demo']}: 夹爪从未开合(抓放任务应至少开合一次)。")
        if (
            np.isfinite(med_jerk)
            and med_jerk > 0
            and np.isfinite(d.get("jerk", np.nan))
            and d["jerk"] > 3.0 * med_jerk
        ):
            w.append(f"{d['demo']}: 动作抖动明显(jerk {d['jerk']:.4f} vs 中位数 {med_jerk:.4f})。")
    if not w:
        w.append("未发现明显异常。")

    return report


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------


def print_report(report: dict) -> None:
    line = "=" * 100
    print(line)
    print(f"数据集 : {report['file']}")
    print(f"格式   : format_version={report['format_version']}   env_args={report['env_args']}")
    print(f"演示数 : {report['num_demos']}    步数合计: {report['total_steps_attr']}")
    print(line)

    header = (
        f"{'demo':<9}{'steps':>7}{'success':>9}{'pos_move(m)':>13}{'rot_move(rad)':>15}"
        f"{'jerk':>9}{'grip_toggles':>14}{'grip_closed%':>13}"
    )
    print(header)
    print("-" * len(header))
    for d in report["demos"]:
        success = {None: "?", True: "True", False: "False"}[d.get("success")]
        print(
            f"{d['demo']:<9}{d['steps']:>7}{success:>9}"
            f"{d.get('pos_move', float('nan')):>13.3f}"
            f"{d.get('rot_move', float('nan')):>15.3f}"
            f"{d.get('jerk', float('nan')):>9.4f}"
            f"{d.get('grip_toggles', -1):>14}"
            f"{100 * d.get('grip_closed_ratio', float('nan')):>12.1f}%"
        )

    s = report["summary"]
    print("-" * len(header))
    print(
        f"汇总: steps 均值 {s['steps_mean']:.1f} ± {s['steps_std']:.1f} "
        f"(min {s['steps_min']} / max {s['steps_max']}) | pos_move 均值 {s['pos_move_mean']:.3f} m "
        f"| success {s['success_count']}/{s['num_demos']}"
    )

    if "obs_terms" in report:
        print("\nobs 观测键(第一条演示):")
        for k, shape in report["obs_terms"].items():
            print(f"  - {k:<40} shape={shape}")

    print("\n质量提示:")
    for msg in report["warnings"]:
        print(f"  * {msg}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab 遥操作数据集统计与质量检查")
    parser.add_argument("dataset", type=str, help="HDF5 数据集路径(teleop record 的输出)")
    parser.add_argument("--json", type=str, default=None, help="可选: 把统计结果另存为 JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_file():
        raise SystemExit(f"[错误] 文件不存在: {dataset_path}")

    report = analyze(dataset_path)
    print_report(report)

    if args.json:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"统计结果已保存: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
