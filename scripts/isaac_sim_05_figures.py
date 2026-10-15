#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为《UMI 式数据工作流：仿真采集 → 数据处理 → 训练使用（五）》生成全部配图（PNG）。

生成 7 张图到 IsaacDataCollection/docs/images/isaac-sim-05/：
  fig1  真机 UMI ↔ Isaac Visuomotor 逐项对照        [graphviz]
  fig2  全链路总览：采集 → 处理 → 使用 → 评估         [graphviz]
  fig3  Visuomotor 任务与双相机布置示意               [matplotlib]
  fig4  录制回放：两条相机流真实帧网格（实测）          [matplotlib]
  fig5  dataset_umi_5.hdf5 体检（实测：5 条步数）      [matplotlib]
  fig6  时间对齐：夹爪指令 / 动作幅值 + 关键帧（实测）  [matplotlib]
  fig7  Mimic 增广 → 视觉 BC 训练链路                 [graphviz]

用法（isaaclab venv 内置 numpy/matplotlib/h5py）：
    ~/code/isaac/isaaclab/.venv/bin/python IsaacDataCollection/scripts/isaac_sim_05_figures.py
    ... --only fig3 fig5
    ... --dataset ~/code/isaac/IsaacDataCollection/datasets/dataset_umi_9.hdf5

依赖：graphviz（dot CLI）；numpy + matplotlib + h5py；fig4/5/6 需要数据集文件（--dataset 指定）。
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
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-05"
DEFAULT_DATASET = (Path.home() / "code" / "isaac" / "IsaacDataCollection" / "datasets" /
                   "dataset_umi_5.hdf5")

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
# fig1 真机 UMI ↔ Isaac Visuomotor 对照
# --------------------------------------------------------------------------
def fig_umi_vs_sim(out: Path) -> None:
    inner = f'''
  subgraph cluster_real {{
    label="真机 UMI（手持夹爪 + 相机，数百美元 / 场景自由）";
    style="rounded,filled"; fillcolor="{CLR["gray_fill"]}"; color="{CLR["gray"]}";
    fontsize=13; fontcolor="{CLR["gray"]}"; margin=14;
    R1 [label="① 手持二指夹爪\\n人手在真实场景里演示", fillcolor="{CLR["orange_fill"]}",
        color="{CLR["orange"]}"];
    R2 [label="② 腕部同轴相机\\nGoPro + 反光镜（视角 = 夹爪视角）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R3 [label="③ SLAM 后处理\\n从画面恢复夹爪 6DoF 轨迹（有噪声）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R1 -> R2 [style=dotted, arrowhead=none];
    R2 -> R3 [style=dotted, arrowhead=none];
  }}
  subgraph cluster_sim {{
    label="Isaac Visuomotor 任务（0 成本 · 本机 RTX 4070 实测）";
    style="rounded,filled"; fillcolor="{CLR["blue_fill"]}"; color="{CLR["blue"]}";
    fontsize=13; fontcolor="{CLR["blue"]}"; margin=14;
    S1 [label="① 键盘 / Xbox 手柄遥操作\\n（人手演示的等价物）", fillcolor="white",
        color="{CLR["blue"]}"];
    S2 [label="② wrist_cam + table_cam\\n200×200 双相机（都进 policy 观测）",
        fillcolor="white", color="{CLR["blue"]}"];
    S3 [label="③ 仿真真值位姿\\n无需 SLAM——直接给干净 6DoF", fillcolor="white",
        color="{CLR["blue"]}"];
    S1 -> S2 [style=dotted, arrowhead=none];
    S2 -> S3 [style=dotted, arrowhead=none];
  }}
  OUT [label="同一数据形态：相机图像（腕部同轴视角为主） + 夹爪相对动作 Δpose(6) + gripper(1)",
       fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}", penwidth=2.2, fontsize=13.5];
  R1 -> S1 [label="角色等价", constraint=false, style=dashed];
  R2 -> S2 [label="视角对齐", constraint=false, style=dashed];
  R3 -> S3 [label="仿真更干净", constraint=false, style=dashed];
  R3 -> OUT [label="训练数据"];
  S3 -> OUT [label="训练数据"];
'''
    render_dot(inner, out, "真机 UMI 与 Isaac Visuomotor：逐项对应，数据形态同构",
               rankdir="TB", nodesep="0.5", ranksep="0.55", fontsize="12.5",
               graph_extra=', width="6.6"', edge_extra=', style=solid')


# --------------------------------------------------------------------------
# fig2 全链路总览
# --------------------------------------------------------------------------
def fig_pipeline(out: Path) -> None:
    inner = f'''
  A [label="① 采集：teleop record\\nVisuomotor 任务（键盘 / Xbox 手柄）",
     fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}", penwidth=1.9];
  H [label="dataset_umi_5.hdf5\\n5 条 × 2599 帧 × 双 200×200 相机（约 203MB）",
     fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}", penwidth=2.2, fontsize=13];
  V [label="导出视频：hdf5_to_mp4.py\\n→ umi_videos/demo_i_*.mp4", fillcolor="{CLR["gray_fill"]}",
     color="{CLR["gray"]}"];
  Q [label="统计质检：teleop_dataset_stats.py\\n（success / 步数 / 夹爪）",
     fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];
  AN [label="② 标注：annotate_demos.py\\n子任务边界（--auto）", fillcolor="{CLR["blue_fill"]}",
      color="{CLR["blue"]}"];
  GN [label="③ 增广：generate_dataset.py\\nMimic ×N（带相机更慢，先小并发）",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  G [label="generated_umi_*.hdf5\\n（演示量级 ↑）", fillcolor="{CLR["green_fill"]}",
     color="{CLR["green"]}", penwidth=2.0];
  TR [label="④ 训练：robomimic train.py\\n视觉 BC（bc_rnn_image_200）",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}", penwidth=1.9];
  PL [label="⑤ 评估：play.py 回放\\n20 rollouts → 成功率", fillcolor="{CLR["purple_fill"]}",
      color="{CLR["purple"]}", penwidth=1.9];

  A -> H;
  H -> V;
  H -> Q;
  H -> AN;
  AN -> GN;
  GN -> G;
  G -> TR;
  TR -> PL;
  H -> TR [label="不推荐：3~5 条直接练必然过拟合", style=dashed,
           color="{CLR["red"]}", fontcolor="{CLR["red"]}"];
'''
    render_dot(inner, out, "全链路总览：采集 →（处理）→ 增广 → 训练 → 评估",
               rankdir="LR", nodesep="0.28", ranksep="0.6", fontsize="12",
               graph_extra=', width="8.6"')


# --------------------------------------------------------------------------
# fig3 任务与相机布置示意
# --------------------------------------------------------------------------
def fig_task_cameras(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.8, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def rect(x, y, w, h, color, fill, lw=1.8, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.25,rounding_size=0.9",
                                    facecolor=fill, edgecolor=color, linewidth=lw,
                                    linestyle=ls))

    # 地面 + 工作台
    ax.plot([3, 97], [6, 6], color=CLR["gray"], lw=2.0)
    rect(36, 6, 60, 14, CLR["gray"], "#F7F8FA", lw=1.6)
    ax.plot([36, 96], [20, 20], color=CLR["gray"], lw=2.2)
    ax.text(90, 10.5, "工作台", fontsize=11.5, color=CLR["gray"], ha="center")

    # 积木（蓝底座 + 目标位 + 绿块）
    rect(56, 20, 7, 7, CLR["blue"], CLR["blue_fill"], lw=2.0)
    ax.text(59.5, 23.5, "蓝", fontsize=11, ha="center", va="center", color=CLR["blue"],
            fontweight="bold")
    rect(56, 27, 7, 7, CLR["red"], "white", lw=1.8, ls="--")
    ax.text(59.5, 30.5, "目标位", fontsize=9.5, ha="center", va="center", color=CLR["red"])
    rect(74, 20, 7, 7, CLR["green"], CLR["green_fill"], lw=2.0)
    ax.text(77.5, 23.5, "绿", fontsize=11, ha="center", va="center", color=CLR["green"],
            fontweight="bold")

    # 机械臂（基座 + 连杆 + 手爪）
    rect(6, 6, 10, 16, CLR["gray"], CLR["gray_fill"], lw=1.8)
    ax.text(11, 14, "Franka", fontsize=10.5, ha="center", color=CLR["gray"])
    ax.plot([11, 27, 46], [22, 35, 36.5], color=CLR["dark"], lw=5.5,
            solid_capstyle="round", zorder=4)
    ax.plot([11, 27], [22, 35], color=CLR["edge"], lw=2.0, zorder=4)
    rect(44, 32.5, 9, 6.5, CLR["dark"], "white", lw=1.8)
    ax.text(48.5, 35.7, "panda_hand", fontsize=9, ha="center", va="center", color=CLR["dark"])

    # 腕部相机 + 视锥
    rect(47, 28.6, 5.2, 3.4, CLR["orange"], CLR["orange_fill"], lw=2.0)
    ax.add_patch(plt.Polygon([(47.5, 28.6), (41.5, 20.2), (62.0, 20.2)], closed=True,
                             facecolor=CLR["orange"], alpha=0.13, edgecolor="none"))
    ax.plot([47.5, 41.5], [28.6, 20.2], color=CLR["orange"], lw=1.2, ls="--")
    ax.plot([47.5, 62.0], [28.6, 20.2], color=CLR["orange"], lw=1.2, ls="--")
    ax.annotate("wrist_cam（腕部同轴）\n200×200 · 24mm · Δ=(0.13, 0, −0.15)",
                xy=(49.6, 30.2), xytext=(14, 46), fontsize=11, color=CLR["orange"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CLR["orange"], lw=1.6))

    # 桌面相机 + 视锥
    ax.plot([92, 92], [20, 31], color=CLR["gray"], lw=2.4)
    rect(89.3, 31, 5.4, 3.4, CLR["blue"], CLR["blue_fill"], lw=2.0)
    ax.add_patch(plt.Polygon([(89.3, 31), (63.0, 20.2), (84.0, 20.2)], closed=True,
                             facecolor=CLR["blue"], alpha=0.11, edgecolor="none"))
    ax.plot([89.3, 63.0], [31, 20.2], color=CLR["blue"], lw=1.2, ls="--")
    ax.plot([89.3, 84.0], [31, 20.2], color=CLR["blue"], lw=1.2, ls="--")
    ax.annotate("table_cam（桌边固定）\n200×200 · Δ=(1.0, 0, 0.4)",
                xy=(92, 34.4), xytext=(72, 44), fontsize=11, color=CLR["blue"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CLR["blue"], lw=1.6))

    ax.text(50, 12.4, "image_obs_list = [table_cam, wrist_cam] —— 两台相机都进策略观测",
            ha="center", fontsize=12, color=CLR["dark"], fontweight="bold")
    ax.text(50, 2.6, "提示：相机是无几何体的 prim（视口里看不到“镜头实物”）——用 Tools → Sensors → Camera Inspector 检查与预览",
            ha="center", fontsize=10.5, color=CLR["gray"])

    fig.suptitle("Visuomotor 任务的相机布置（源码参数示意）",
                 fontsize=15.5, fontweight="bold", y=0.985)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig4 数据集体检（实测）
# --------------------------------------------------------------------------
def fig_dataset_check(out: Path, dataset: Path) -> None:
    setup_matplotlib()
    f = h5py.File(dataset, "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    lens = [int(f["data"][d].attrs["num_samples"]) for d in demos]
    succ = [bool(f["data"][d].attrs["success"]) for d in demos]
    f.close()
    n = len(lens)
    total = sum(lens)
    size_mb = dataset.stat().st_size / 1024**2

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)
    colors = [CLR["blue"]] + [CLR["teal"]] * (n - 1)
    bars = ax.bar(range(n), lens, width=0.62, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=1.4)
    for i, (b, L, ok) in enumerate(zip(bars, lens, succ)):
        ax.text(b.get_x() + b.get_width() / 2, L + 28, f"{L}",
                ha="center", fontsize=12, fontweight="bold", color=CLR["dark"])
        ax.text(b.get_x() + b.get_width() / 2, 40, "✓" if ok else "✗",
                ha="center", fontsize=13, color=CLR["green"], fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"demo_{i}" for i in range(n)], fontsize=11.5)
    ax.set_ylabel("演示步数（20Hz）", fontsize=12)
    ax.set_ylim(0, max(lens) * 1.18)
    ax.grid(axis="y", alpha=0.22, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.annotate("最长 1161 步 ≈ 58 秒：\n含完整的接近-抓取-叠放",
                xy=(0, lens[0]), xytext=(1.05, lens[0] * 0.86), fontsize=11,
                color=CLR["blue"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CLR["blue"], lw=1.5))
    ax.text(0.985, 0.97,
            f"共 {n} 条 · {total:,} 帧 ≈ {total * 0.05:.0f} 秒 · 全部 success=True\n"
            f"磁盘 {size_mb:.0f} MB（两条 200×200 图像流，压缩入库）\n"
            f"平均 {total / n:.0f} 步/条 —— 短小干脆，适合低频次分坐录制",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=CLR["dark"],
            bbox=dict(boxstyle="round,pad=0.55", facecolor=CLR["gray_fill"],
                      edgecolor=CLR["edge"], linewidth=1.2))

    fig.suptitle(f"dataset_umi_5.hdf5 体检：{n} 条成功演示（实测）",
                 fontsize=15, fontweight="bold", y=0.965)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig5 真实帧网格（实测）
# --------------------------------------------------------------------------
def fig_camera_frames(out: Path, dataset: Path, demo: str = "demo_0") -> None:
    setup_matplotlib()
    f = h5py.File(dataset, "r")
    g = f["data"][demo]
    T = int(g.attrs["num_samples"])
    idxs = np.linspace(0, T - 1, 5).astype(int)
    wrist = np.asarray(g["obs/wrist_cam"][idxs.tolist()])
    table = np.asarray(g["obs/table_cam"][idxs.tolist()])
    f.close()

    fig, axes = plt.subplots(2, 5, figsize=(12.6, 5.6))
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.075, right=0.99,
                        wspace=0.06, hspace=0.16)
    for j, ix in enumerate(idxs):
        axes[0, j].imshow(wrist[j])
        axes[1, j].imshow(table[j])
        axes[0, j].set_title(f"t = {ix} 步（{ix * 0.05:.1f} s）", fontsize=10.5,
                             color=CLR["dark"])
        for i in (0, 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_color(CLR["edge"])
                s.set_linewidth(1.2)
    fig.text(0.028, 0.70, "腕部\nwrist_cam", rotation=90, fontsize=12.5,
             fontweight="bold", color=CLR["orange"], ha="center", va="center")
    fig.text(0.028, 0.30, "桌面\ntable_cam", rotation=90, fontsize=12.5,
             fontweight="bold", color=CLR["blue"], ha="center", va="center")
    fig.suptitle(f"录制回放：两条真实相机流（{demo} · {T} 步 · 5 帧等间隔抽取）",
                 fontsize=15, fontweight="bold", y=0.955)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig6 时间对齐（实测）
# --------------------------------------------------------------------------
def fig_time_align(out: Path, dataset: Path, demo: str = "demo_0") -> None:
    setup_matplotlib()
    f = h5py.File(dataset, "r")
    g = f["data"][demo]
    T = int(g.attrs["num_samples"])
    a = np.asarray(g["actions"])                     # (T,7)
    idxs = [T // 6, T // 2, T - T // 6]
    frames = np.asarray(g["obs/wrist_cam"][sorted(idxs)])
    order = np.argsort(idxs)
    frames = frames[order]
    idxs_sorted = sorted(idxs)
    f.close()

    gcmd = a[:, 6]
    anorm = np.linalg.norm(a[:, :6], axis=1)
    t = np.arange(T)

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.62,
                          top=0.84, bottom=0.10, left=0.085, right=0.97)
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frames[j])
        ax.set_title(f"wrist_cam @ t={idxs_sorted[j]}（{idxs_sorted[j] * 0.05:.1f} s）",
                     fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(CLR["edge"])

    ax1 = fig.add_subplot(gs[1, :])
    ax1.step(t, gcmd, where="post", color=CLR["orange"], lw=2.0)
    ax1.set_ylabel("夹爪指令\n(−1 合 / +1 开)", fontsize=11)
    ax1.set_ylim(-1.55, 1.55)
    ax1.set_yticks([-1, 0, 1])
    ax1.grid(alpha=0.2, ls="--")
    ax1.set_xlim(0, T - 1)
    close_idx = int(np.argmax(gcmd < 0)) if bool((gcmd < 0).any()) else None
    if close_idx is not None and close_idx > 0:
        ax1.axvline(close_idx, color=CLR["red"], lw=1.4, ls=":")
        ax1.text(close_idx + 12, -1.35, f"首次闭爪指令 @ t={close_idx}", fontsize=10.5,
                 color=CLR["red"], fontweight="bold")

    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(t, anorm, color=CLR["blue"], lw=1.6)
    ax2.set_ylabel("动作幅值\n‖Δpose‖", fontsize=11)
    ax2.set_xlabel(f"步（{demo}，20 Hz）", fontsize=11.5)
    ax2.grid(alpha=0.2, ls="--")
    ax2.set_xlim(0, T - 1)

    fig.text(0.5, 0.905,
             "训练对 = (obs[t], actions[t])；obs 为 pre-step 语义 —— 别把 obs/actions（上一执行动作）当标签",
             ha="center", fontsize=11.5, color=CLR["dark"], fontweight="bold")
    fig.text(0.5, 0.875,
             "（两处易错点详见第四篇 §4：states[t] ≡ obs[t+1]；obs/actions 滞后顶层 actions 一步）",
             ha="center", fontsize=10.5, color=CLR["gray"])
    fig.suptitle(f"时间对齐：夹爪指令与动作幅值（{demo} 实测）",
                 fontsize=15, fontweight="bold", y=0.965)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig7 增广与训练链路
# --------------------------------------------------------------------------
def fig_scale_up(out: Path) -> None:
    inner = f'''
  S [label="5 条种子演示\\n（dataset_umi_5.hdf5 · 双相机 2599 帧）",
     fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}", penwidth=2.0, fontsize=12.5];
  AN [label="annotate_demos.py\\n--auto 子任务标注\\n（Visuomotor-Mimic 任务）",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  GN [label="generate_dataset.py\\nMimic 增广 ×N\\n（相机渲染：更慢、更吃显存）",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  D [label="生成数据 100+ 条\\n（generated_umi_*.hdf5）", fillcolor="{CLR["green_fill"]}",
     color="{CLR["green"]}", penwidth=2.0, fontsize=12.5];
  TR [label="robomimic train.py\\nbc_rnn_image_200.json\\nbatch 16 · 输入 200×200",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}", penwidth=2.0, fontsize=12.5];
  EV [label="play.py 回放评估\\n20 rollouts → 成功率", fillcolor="{CLR["purple_fill"]}",
      color="{CLR["purple"]}"];
  N [label="提示：3~5 条直接训练\\n必然过拟合——先增广再训练", fillcolor="{CLR["orange_fill"]}",
     color="{CLR["red"]}", fontsize=11.5, style="rounded,filled,dashed"];

  S -> AN [label="种子"];
  AN -> GN [label="标注后"];
  GN -> D [label="攒够目标条数"];
  D -> TR [label="训练集"];
  TR -> EV [label="checkpoint"];
  N -> TR [style=dashed, color="{CLR["red"]}", fontcolor="{CLR["red"]}"];
'''
    render_dot(inner, out, "从 5 条种子到可用策略：Mimic 增广 → 视觉 BC 训练",
               rankdir="LR", nodesep="0.3", ranksep="0.62", fontsize="12",
               graph_extra=', width="8.4"')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
FIGS = {
    "fig1": ("fig1-umi-vs-sim.png", lambda out, ds: fig_umi_vs_sim(out)),
    "fig2": ("fig2-pipeline.png", lambda out, ds: fig_pipeline(out)),
    "fig3": ("fig3-task-cameras.png", lambda out, ds: fig_task_cameras(out)),
    "fig4": ("fig4-camera-frames.png", lambda out, ds: fig_camera_frames(out, ds)),
    "fig5": ("fig5-dataset-check.png", lambda out, ds: fig_dataset_check(out, ds)),
    "fig6": ("fig6-time-align.png", lambda out, ds: fig_time_align(out, ds)),
    "fig7": ("fig7-scale-up.png", lambda out, ds: fig_scale_up(out)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="数据集文件（fig4/5/6 数据源）")
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
