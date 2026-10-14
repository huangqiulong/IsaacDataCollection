#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""合并多个 Isaac Lab 数据集为单个 HDF5 文件（演示按顺序重新编号 demo_0..N-1）。

背景:
  - `teleop record` 每次**覆盖**同名输出文件（recorder 以 "w" 模式打开），因此
    多批录制必须各写各的文件（如 dataset_gamepad.hdf5 / dataset_gamepad_b2.hdf5），
    再用本脚本合并成单文件（如 dataset_gamepad_all.hdf5）后统一走 annotate。
  - 支持任意同 schema 数据集（录制原始 / 标注后 / Mimic 生成）：合并方式 = 逐条
    复制 `data/demo_*` 组（含全部子组 / 数据集 / 属性），按输入顺序重新编号。

用法:
    python merge_teleop_datasets.py <output.hdf5> <input1.hdf5> <input2.hdf5> [...]

示例:
    python ~/code/isaac/IsaacDataCollection/scripts/merge_teleop_datasets.py \
        ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad_all.hdf5 \
        ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5 \
        ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad_b2.hdf5

输出:
  - 打印每个输入文件的演示数 / 采用数、输出总览（条数 / 帧数 / success 数）；
  - 校验每条演示含 actions/obs 与 num_samples、success 属性，异常条跳过并告警；
  - 各输入 env_args 不一致时给出警告（仍继续合并，靠首个文件的 env_args）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    ks = [k for k in data_group.keys() if k.startswith("demo_")]
    return sorted(ks, key=lambda x: int(x.split("_")[1]))


def _copy_attrs(src, dst) -> None:
    for a, v in src.attrs.items():
        dst.attrs[a] = v


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    """递归复制组（子组 / 数据集 / 属性）。"""
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            g = dst.create_group(name)
            _copy_group(obj, g)
        else:  # h5py.Dataset
            d = dst.create_dataset(
                name,
                data=obj[()],
                compression=obj.compression,
                compression_opts=obj.compression_opts,
            )
            _copy_attrs(obj, d)
    _copy_attrs(src, dst)


def merge(output: str, inputs: list[str]) -> int:
    out_path = Path(output)
    if out_path.exists():
        print(f"[WARN] 输出文件已存在，将被覆盖: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infos: list[tuple[str, int, int]] = []
    with h5py.File(out_path, "w") as f_out:
        f_out.attrs["format_version"] = 1
        data_out = f_out.create_group("data")

        env_args_ref = None
        demo_idx = 0
        total_steps = 0
        n_success = 0
        for in_path in inputs:
            with h5py.File(in_path, "r") as f_in:
                if "data" not in f_in:
                    raise ValueError(f"缺少 /data 组: {in_path}")
                d_in = f_in["data"]

                ea = d_in.attrs.get("env_args")
                if ea is None:
                    print(f"[WARN] {in_path} 无 env_args 属性")
                elif env_args_ref is None:
                    env_args_ref = ea
                else:
                    same = bool(ea == env_args_ref) if not (
                        hasattr(ea, "dtype") or hasattr(env_args_ref, "dtype")
                    ) else str(ea) == str(env_args_ref)
                    if not same:
                        print(
                            f"[WARN] env_args 与首个文件不一致: {in_path}\n"
                            f"       本文件: {ea}\n       首个  : {env_args_ref}"
                        )

                keys = _sorted_demo_keys(d_in)
                n_ok = 0
                for k in keys:
                    src = d_in[k]
                    if "actions" not in src or "obs" not in src:
                        print(f"[WARN] 跳过异常演示（缺 actions/obs）: {in_path}:{k}")
                        continue
                    ns = src.attrs.get("num_samples")
                    if ns is None:
                        print(f"[WARN] {in_path}:{k} 缺 num_samples 属性")
                    dst = data_out.create_group(f"demo_{demo_idx}")
                    _copy_group(src, dst)
                    total_steps += int(ns) if ns is not None else 0
                    if bool(src.attrs.get("success", False)):
                        n_success += 1
                    demo_idx += 1
                    n_ok += 1
                infos.append((str(in_path), len(keys), n_ok))

                if "mask" in f_in:
                    print(f"[WARN] {in_path} 含 mask 组（未合并）")

        if env_args_ref is not None:
            data_out.attrs["env_args"] = env_args_ref
        data_out.attrs["total"] = total_steps

    print("\n=== 合并完成 ===")
    for p, n0, n1 in infos:
        print(f"  输入 {p}: {n0} 条 -> 采用 {n1} 条")
    print(f"  输出 {out_path}: 共 {demo_idx} 条演示, {total_steps} 帧, success={n_success}/{demo_idx}")
    return demo_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="合并多个 Isaac Lab HDF5 数据集（demo 重新编号）")
    ap.add_argument("output", help="输出 hdf5 路径")
    ap.add_argument("inputs", nargs="+", help="输入 hdf5 路径（按给定顺序合并）")
    args = ap.parse_args()
    n = merge(args.output, args.inputs)
    if n == 0:
        print("[ERROR] 合并结果为空", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
