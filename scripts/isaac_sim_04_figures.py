#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为《HDF5 格式详解:结构、可视化与数据处理实战（四）》生成全部配图（PNG）。

生成 6 张图到 IsaacDataCollection/docs/images/isaac-sim-04/：
  fig1  格式对比：HDF5 与它的邻居们                     [matplotlib]
  fig2  HDF5 三种对象模型（文件/组/数据集/属性）        [graphviz]
  fig3  Isaac Lab 数据集全景结构                        [graphviz]
  fig4  时间对齐关系（actions/obs/states 的错位）        [matplotlib]
  fig5  物体状态真值：三块积木的搬运轨迹（实测）          [matplotlib]
  fig6  HDF5 生态：谁生产、谁消费                       [graphviz]

用法（isaaclab venv 内置 numpy/matplotlib）：
    ~/code/isaac/isaaclab/.venv/bin/python IsaacDataCollection/scripts/isaac_sim_04_figures.py
    ... --only fig3 fig4

依赖：graphviz（dot CLI）；numpy + matplotlib；fig4 需要数据集文件（--dataset 指定）。
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from string import Template

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# 常量与样式（与系列其他篇保持一致）
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-04"
DEFAULT_DATASET = (Path.home() / "code" / "isaac" / "IsaacDataCollection" / "datasets" /
                   "dataset_keyboard.hdf5")

FONT = "Noto Sans CJK SC"

CLR = {
    "font": FONT, "dpi": 150, "dark": "#1F2933",
    "blue": "#3B7DD8", "blue_fill": "#E8F1FC",
    "orange": "#E8833A", "orange_fill": "#FDF0E3",
    "green": "#3FA66B", "green_fill": "#E6F5EC",
    "purple": "#8E64C8", "purple_fill": "#F1EBFA",
    "teal": "#2E9CA6", "teal_fill": "#E3F4F5",
    "gray": "#5A6570", "gray_fill": "#EEF1F4",
    "red": "#D64545",
    "edge": "#8794A1",
}

DOT_HEAD = Template('''digraph G {
  rankdir=$rankdir;
  graph [fontname="$font", bgcolor="white", dpi=$dpi, pad=0.4, nodesep=$nodesep,
         ranksep=$ranksep, fontcolor="$dark", fontsize=17, labelloc=t, label="$title"$graph_extra];
  node [fontname="$font", shape=box, style="rounded,filled", fontsize=$fs,
        margin="0.24,0.13", penwidth=1.6, fontcolor="$dark"$node_extra];
  edge [fontname="$font", fontsize=11, color="$edge", penwidth=1.5, arrowsize=0.85,
        fontcolor="#65707B"$edge_extra];
$inner
}''')


def render_dot(inner: str, out: Path, title: str, *, rankdir: str = "TB",
               nodesep: str = "0.45", ranksep: str = "0.5", fontsize: str = "13",
               graph_extra: str = "", node_extra: str = "", edge_extra: str = "") -> None:
    src = DOT_HEAD.substitute(CLR, inner=inner, title=title, rankdir=rankdir,
                              nodesep=nodesep, ranksep=ranksep, fs=fontsize,
                              graph_extra=graph_extra, node_extra=node_extra,
                              edge_extra=edge_extra)
    with out.open("wb") as fh:
        subprocess.run(["dot", "-Tpng"], input=src.encode("utf-8"),
                       stdout=fh, check=True)
    print(f"[ok] {out.name}")


def setup_matplotlib() -> None:
    prefer = ["Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
              "WenQuanYi Micro Hei", "Droid Sans Fallback"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [n for n in prefer if n in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = CLR["dark"]
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.dpi"] = 110


# --------------------------------------------------------------------------
# fig1 三种对象模型
# --------------------------------------------------------------------------
def fig_hdf5_model(out: Path) -> None:
    inner = f'''
  FILE [label="HDF5 文件\\nfile.hdf5", fillcolor="{CLR["gray_fill"]}",
        color="{CLR["gray"]}", penwidth=1.9];
  G1 [label="组 Group\\n/data（像文件夹，可无限嵌套）", fillcolor="{CLR["blue_fill"]}",
      color="{CLR["blue"]}"];
  G2 [label="组 Group\\n/demo_0/obs", fillcolor="{CLR["blue_fill"]}",
      color="{CLR["blue"]}"];
  D1 [label="数据集 Dataset\\n/demo_0/actions\\n(T,7) float32", fillcolor="{CLR["green_fill"]}",
      color="{CLR["green"]}"];
  D2 [label="数据集 Dataset\\n/joint_pos (T,9)", fillcolor="{CLR["green_fill"]}",
      color="{CLR["green"]}"];
  A1 [label="属性 Attribute\\nformat_version = 1", fillcolor="{CLR["purple_fill"]}",
      color="{CLR["purple"]}", fontsize=11];
  A2 [label="属性 Attribute\\ntotal = 2780", fillcolor="{CLR["purple_fill"]}",
      color="{CLR["purple"]}", fontsize=11];

  FILE -> G1 [label="包含"];
  G1 -> G2 [label="包含"];
  G1 -> D1 [label="包含"];
  G2 -> D2 [label="包含"];
  FILE -> A1 [style=dotted, arrowhead=none, label="挂载"];
  G1 -> A2 [style=dotted, arrowhead=none, label="挂载"];
'''
    render_dot(inner, out, "HDF5 三种对象：组（文件夹）· 数据集（N 维数组）· 属性（元数据）",
               rankdir="TB", nodesep="0.5", ranksep="0.45", fontsize="12.5",
               graph_extra=', width="5.2"', edge_extra=', style=solid')


# --------------------------------------------------------------------------
# fig2 数据集全景结构
# --------------------------------------------------------------------------
def fig_dataset_map(out: Path) -> None:
    inner = f'''
  FILE [label="dataset.hdf5\\nattrs: format_version = 1", fillcolor="{CLR["gray_fill"]}",
        color="{CLR["gray"]}", penwidth=1.9];
  DATA [label="data/\\nattrs: env_args · total（总步数）",
        fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  DEMO [label="demo_N/\\nattrs: num_samples · success",
        fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  ACT [label="actions  (T,7)\\n末端相对增量 + 夹爪", fillcolor="{CLR["green_fill"]}",
       color="{CLR["green"]}"];
  PACT [label="processed_actions  (T,8)\\n控制器处理后的关节目标", fillcolor="{CLR["green_fill"]}",
        color="{CLR["green"]}"];
  OBS [label="obs/（策略观测）\\njoint_pos / joint_vel\\neef_pos / eef_quat / gripper_pos\\ncube_positions / cube_orientations\\nobject（39 维拼接）",
       fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  INIT [label="initial_state/\\n本条开始时的完整状态（重置用）",
        fillcolor="{CLR["teal_fill"]}", color="{CLR["teal"]}"];
  ST [label="states/\\n逐步完整物理状态（物理真值）",
      fillcolor="{CLR["teal_fill"]}", color="{CLR["teal"]}"];

  FILE -> DATA -> DEMO;
  DEMO -> ACT;
  DEMO -> PACT;
  DEMO -> OBS;
  DEMO -> INIT;
  DEMO -> ST;
'''
    render_dot(inner, out, "Isaac Lab 遥操作数据集全景（robomimic 兼容 schema）",
               rankdir="LR", nodesep="0.25", ranksep="0.45", fontsize="12",
               graph_extra=', width="6.4"')


# --------------------------------------------------------------------------
# fig3 时间对齐关系
# --------------------------------------------------------------------------
def fig_time_align(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.6, 5.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cols = [10, 30, 50, 70]           # 4 列
    w, h = 15, 9.5
    y_obs, y_act, y_st = 66, 43, 20

    def box(x, y, color, fill, text, sub=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.35,rounding_size=1.0",
                                    facecolor=fill, edgecolor=color, linewidth=1.7))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), text, ha="center",
                va="center", fontsize=11.5, fontweight="bold", color=CLR["dark"])
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=9, color=CLR["gray"])

    for i, x in enumerate(cols):
        box(x, y_obs, CLR["blue"], CLR["blue_fill"], f"obs[{'t' if i == 0 else i}]")
        if i < 3:
            box(x, y_act, CLR["orange"], CLR["orange_fill"],
                f"actions[{'t' if i == 0 else i}]")
            box(x, y_st, CLR["green"], CLR["green_fill"],
                f"states[{'t' if i == 0 else i}]")
    ax.text(cols[-1] + w + 2, y_obs + h / 2, "…", fontsize=16, va="center",
            color=CLR["gray"])
    ax.text(cols[-1] + w + 2, y_act + h / 2, "…", fontsize=16, va="center",
            color=CLR["gray"])
    ax.text(cols[-1] + w + 2, y_st + h / 2, "…", fontsize=16, va="center",
            color=CLR["gray"])

    # 竖直箭头 obs -> actions -> states（第 1 列示例,其余列淡画）
    for i, x in enumerate(cols[:3]):
        alpha = 1.0 if i == 0 else 0.35
        ax.annotate("", xy=(x + w / 2, y_act + h), xytext=(x + w / 2, y_obs),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=CLR["gray"],
                                    alpha=alpha))
        ax.annotate("", xy=(x + w / 2, y_st + h), xytext=(x + w / 2, y_act),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=CLR["gray"],
                                    alpha=alpha))
    ax.text(cols[0] + w + 1.2, (y_obs + y_act) / 2 + 2, "执行", fontsize=11,
            color=CLR["gray"])
    ax.text(cols[0] + w + 1.2, (y_act + y_st) / 2 - 4.5, "产生", fontsize=11,
            color=CLR["gray"])

    # states[t] == obs[t+1] 的斜虚线（第 1 列 -> 第 2 列）
    ax.annotate("", xy=(cols[1] + 2.5, y_obs), xytext=(cols[0] + w - 2.5, y_st + h),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=CLR["green"],
                                linestyle="--"))
    ax.text(cols[0] + w + 4.5, y_act - 3.5, "states[t] ≡ obs[t+1]（逐值相等）",
            fontsize=11.5, color=CLR["green"], fontweight="bold")

    ax.text(50, 8.5, "两条实测恒等式：  states[t] == obs[t+1]      obs/actions[t] == actions[t-1]",
            ha="center", fontsize=13, fontweight="bold", color=CLR["dark"])
    ax.text(50, 3.5, "训练标签用顶层 actions；obs/actions 是“上一执行动作”（滞后 1 步，别拿它当标签）",
            ha="center", fontsize=11.5, color=CLR["red"])

    fig.suptitle("时间对齐：从观测出发执行动作，结果写进下一帧",
                 fontsize=15.5, fontweight="bold", y=0.985)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig4 积木真值轨迹（实测）
# --------------------------------------------------------------------------
def fig_cube_truth(out: Path, dataset: Path, demo: str = "demo_1") -> None:
    setup_matplotlib()
    f = h5py.File(dataset, "r")
    g = f["data"][demo]
    cp = np.asarray(g["obs/cube_positions"])          # (T, 9) = 3 x pos(3)
    n = len(cp)
    f.close()

    t = np.arange(n) * 0.05
    colors = {"蓝块（底座）": "#3B7DD8", "红块": "#D64545", "绿块": "#3FA66B"}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.97, wspace=0.24)
    fig.suptitle(f"物体状态真值：三块积木的搬运轨迹（{demo} 实测，{n} 步 ≈ {n * 0.05:.0f} 秒）",
                 fontsize=15, fontweight="bold", y=0.975)

    # (a) XY 俯视轨迹
    ax = axes[0]
    for i, (label, color) in enumerate(colors.items()):
        seg = cp[:, i * 3:(i + 1) * 3]
        lw = 2.2 if i == 0 else 1.8
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, alpha=0.9, label=label)
        ax.scatter(seg[0, 0], seg[0, 1], s=60, facecolor="white",
                   edgecolor=color, linewidth=2.2, zorder=5)
        ax.scatter(seg[-1, 0], seg[-1, 1], s=60, marker="s", color=color, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("① XY 俯视轨迹（○ 起点 · ■ 终点）", fontsize=13, fontweight="bold", pad=8)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.grid(alpha=0.22, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) 高度 z 随时间
    ax = axes[1]
    for i, (label, color) in enumerate(colors.items()):
        seg = cp[:, i * 3:(i + 1) * 3]
        ax.plot(t, seg[:, 2], color=color, lw=1.8, label=label, alpha=0.92)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("高度 z (m)")
    ax.set_title("② 高度随时间：抬升-放置的台阶清晰可见", fontsize=13, fontweight="bold", pad=8)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.grid(alpha=0.22, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig5 生态图
# --------------------------------------------------------------------------
def fig_ecosystem(out: Path) -> None:
    inner = f'''
  P1 [label="teleop record\\n遥操作录制", fillcolor="{CLR["blue_fill"]}",
      color="{CLR["blue"]}"];
  P2 [label="Mimic generate\\n自动生成（增广）", fillcolor="{CLR["blue_fill"]}",
      color="{CLR["blue"]}"];
  H5 [label="HDF5 数据集\\nrobomimic 兼容\\n全状态真值 + 成功标记",
      fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}", penwidth=2.0,
      fontsize=13];
  C1 [label="robomimic train / play\\n行为克隆训练与评估",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  C2 [label="teleop replay\\n物理回放验证",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  C3 [label="Mimic annotate\\n子任务标注（读写）",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  C4 [label="h5py / H5Web\\n查看、分析与可视化",
      fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];
  C5 [label="hdf5_to_mp4\\n相机数据导出视频",
      fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];

  {{ rank=same; P1; P2; }}
  P1 -> H5 [label="写"];
  P2 -> H5 [label="写"];
  H5 -> C1 [label="读"];
  H5 -> C2 [label="读"];
  H5 -> C3 [label="读写"];
  H5 -> C4 [label="查看"];
  H5 -> C5 [label="导出"];
'''
    render_dot(inner, out, "HDF5 在 Isaac Lab 生态中的位置：一份格式，全链路通吃",
               rankdir="LR", nodesep="0.3", ranksep="0.62", fontsize="12",
               graph_extra=', width="6.8"')


# --------------------------------------------------------------------------
# fig6 格式对比卡片
# --------------------------------------------------------------------------
def fig_format_compare(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(12.2, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cards = [
        ("HDF5", CLR["blue"], CLR["blue_fill"],
         "组 + N 维数组 + 属性（自描述的单文件数据库）",
         "✓ 自描述 · 单文件 · 任意维 · 分块压缩 · 跨语言",
         "× 结构复杂 · 并发写受限",
         "用途：机器人演示数据集（主线）", True),
        ("NPZ（numpy 打包）", CLR["gray"], CLR["gray_fill"],
         "numpy 数组的 zip 打包",
         "✓ Python 里两行搞定",
         "× 无层级元数据 · 不能增量写",
         "用途：临时数组交换", False),
        ("rosbag2 / MCAP", CLR["orange"], CLR["orange_fill"],
         "带时间戳的消息流（逐话题记录）",
         "✓ ROS 生态原生 · 多话题并行",
         "× 读取需要 ROS 工具链",
         "用途：真机采集的原始层", False),
        ("CSV / JSON", CLR["teal"], CLR["teal_fill"],
         "文本表格 / 对象",
         "✓ 人类可读 · 简单通用",
         "× 大数组无结构 · 体积大",
         "用途：统计、配置（如 stats.json）", False),
    ]

    W, H = 45.5, 40.0
    xs, ys = [3, 51.5], [52, 5]
    for idx, (name, color, fill, model, good, bad, use, star) in enumerate(cards):
        x, y = xs[idx % 2], ys[idx // 2]
        box = FancyBboxPatch((x, y), W, H,
                             boxstyle="round,pad=0.6,rounding_size=1.6",
                             facecolor=fill, edgecolor=color,
                             linewidth=2.6 if star else 1.6)
        ax.add_patch(box)
        ax.text(x + 2.6, y + H - 4.6, name, fontsize=15, fontweight="bold", color=color,
                va="center")
        if star:
            ax.text(x + W - 2.6, y + H - 4.6, "★ 本篇主角", fontsize=11.5,
                    color=color, va="center", ha="right", fontweight="bold")
        ax.text(x + 2.6, y + H - 11.5, model, fontsize=11, color=CLR["dark"], va="center")
        ax.text(x + 2.6, y + H - 19, good, fontsize=11, color=CLR["dark"], va="center")
        ax.text(x + 2.6, y + H - 25.5, bad, fontsize=11, color=CLR["dark"], va="center")
        ax.text(x + 2.6, y + 3.6, use, fontsize=11, color=color, fontweight="bold",
                va="center")

    fig.suptitle("格式对比：HDF5 与它的“邻居们”——各自的位置与边界",
                 fontsize=15.5, fontweight="bold", y=0.985)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
FIGS = {
    "fig1": ("fig1-format-compare.png", lambda out, ds: fig_format_compare(out)),
    "fig2": ("fig2-hdf5-model.png", lambda out, ds: fig_hdf5_model(out)),
    "fig3": ("fig3-dataset-map.png", lambda out, ds: fig_dataset_map(out)),
    "fig4": ("fig4-time-alignment.png", lambda out, ds: fig_time_align(out)),
    "fig5": ("fig5-cube-truth.png", lambda out, ds: fig_cube_truth(out, ds)),
    "fig6": ("fig6-ecosystem.png", lambda out, ds: fig_ecosystem(out)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="数据集文件（fig4 数据源）")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定图")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = args.only if args.only else list(FIGS)
    for name in todo:
        fname, fn = FIGS[name]
        fn(args.out / fname, args.dataset)
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
