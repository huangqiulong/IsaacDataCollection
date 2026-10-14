#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为《Isaac Sim 仿真模拟采集数据方法综述（一）》生成全部配图（PNG）。

生成 8 张图到 IsaacDataCollection/docs/images/isaac-sim-01/：
  fig1  五类采集方式速览卡片                                  [matplotlib]
  fig2  遥操作设备选择矩阵（上手成本 × 表达力）               [matplotlib]
  fig3  动捕数据流：动捕源 → 重定向 → 机器人轨迹              [graphviz]
  fig4  真实动捕数据可视化（humanoid_dance.npz 运动残影）      [matplotlib]
  fig5  真机 UMI ↔ 仿真复刻 三步对照                          [graphviz]
  fig6  人类第一人称 ↔ 机器人 POV 对照                        [graphviz]
  fig7  仿真数据生成：自动化光谱                              [matplotlib]
  fig8  推荐学习路线图                                        [graphviz]

用法（isaaclab venv 内置 numpy/matplotlib）：
    ~/code/isaac/isaaclab/.venv/bin/python IsaacDataCollection/scripts/isaac_survey2_figures.py
    ... --only fig1 fig4        # 只生成指定图

依赖：graphviz（dot CLI）；numpy + matplotlib。
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from string import Template

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# 常量与样式（与 survey-1 系列保持一致）
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-01"
DEFAULT_MOTIONS = (Path.home() / "code" / "isaac" / "isaaclab" / "source" /
                   "isaaclab_tasks" / "isaaclab_tasks" / "contrib" /
                   "humanoid_amp" / "motions")

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
    plt.rcParams["axes.edgecolor"] = "#B8C0C8"
    plt.rcParams["axes.labelcolor"] = CLR["dark"]
    plt.rcParams["text.color"] = CLR["dark"]
    plt.rcParams["xtick.color"] = "#5A6570"
    plt.rcParams["ytick.color"] = "#5A6570"
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.dpi"] = 110


# --------------------------------------------------------------------------
# fig1 五类采集方式速览卡片
# --------------------------------------------------------------------------
def fig1(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.6, 7.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cards = [
        ("① 遥操作", CLR["blue"], CLR["blue_fill"],
         "真机对应：主从臂 / VR / 手柄",
         "产出：HDF5 演示集\n（状态-动作-成功标记）",
         "难度 ★☆☆ · 首跑约 30 分钟"),
        ("② 动捕", CLR["purple"], CLR["purple_fill"],
         "真机对应：Vicon / Xsens / 视频",
         "产出：全身运动轨迹 .npz\n（60 fps · 关节+刚体）",
         "难度 ★★☆ · 数据已内置"),
        ("③ UMI", CLR["teal"], CLR["teal_fill"],
         "真机对应：手持夹爪 + GoPro",
         "产出：腕部相机视频 +\n相对位姿轨迹",
         "难度 ★★☆ · 首跑约 1 小时"),
        ("④ EGO", CLR["orange"], CLR["orange_fill"],
         "真机对应：头戴第一人称设备",
         "产出：机器人 POV 视频帧\n（第一人称视角）",
         "难度 ★★★ · 链条较长"),
        ("⑤ 仿真采集", CLR["green"], CLR["green_fill"],
         "真机无对应：仿真独有",
         "产出：海量演示 / 图像数据集\n（Mimic / RL / SDG）",
         "难度 ★★☆ · 规模由 GPU 决定"),
    ]

    W, H = 29.5, 41.0
    xs = [3.5, 35.2, 67.0]
    ys = [53.5, 8.0]
    for idx, (name, color, fill, line1, line2, line3) in enumerate(cards):
        x = xs[idx % 3]
        y = ys[idx // 3]
        box = FancyBboxPatch((x, y), W, H,
                             boxstyle="round,pad=0.6,rounding_size=1.6",
                             facecolor=fill, edgecolor=color, linewidth=2.0)
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((x, y + H - 8.2), W, 8.2,
                                    boxstyle="round,pad=0.6,rounding_size=1.6",
                                    facecolor=color, edgecolor=color, linewidth=2.0))
        ax.text(x + W / 2, y + H - 5.4, name, ha="center", va="center",
                fontsize=17, fontweight="bold", color="white")
        ax.text(x + 2.2, y + H - 14.5, line1, fontsize=11.5, color=CLR["dark"], va="center")
        ax.text(x + 2.2, y + H - 23.5, line2, fontsize=11.5, color=CLR["dark"],
                va="center", linespacing=1.6)
        ax.text(x + 2.2, y + 4.2, line3, fontsize=11.5, color=color,
                fontweight="bold", va="center")

    # 第 6 格：总结提示块
    x, y = xs[2], ys[1]
    box = FancyBboxPatch((x, y), W, H,
                         boxstyle="round,pad=0.6,rounding_size=1.6",
                         facecolor="white", edgecolor=CLR["gray"],
                         linewidth=1.6, linestyle="--")
    ax.add_patch(box)
    ax.text(x + W / 2, y + H - 8, "共同点", ha="center", va="center",
            fontsize=14, fontweight="bold", color=CLR["gray"])
    ax.text(x + W / 2, y + H / 2 - 3,
            "五类方式全部在\nIsaac Sim 内完成\n\n无需真实机器人\n数据自带真值与\n自动成功标记",
            ha="center", va="center", fontsize=12, color=CLR["dark"],
            linespacing=1.8)

    fig.suptitle("Isaac Sim 里的五类数据采集方式：一张图选好方向",
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig2 遥操作设备选择矩阵
# --------------------------------------------------------------------------
def fig2(out: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.97)

    pts = [
        ("键盘", 1.6, 4.2, 900, CLR["blue"],
         "零门槛 · 内置支持\n相对位姿增量控制"),
        ("Xbox 手柄", 3.2, 6.2, 1100, CLR["blue"],
         "双摇杆 6DoF · 长时间采集\n比键盘舒服"),
        ("SpaceMouse", 4.8, 7.4, 950, CLR["purple"],
         "专业 6 自由度鼠标\n平顺、适合桌面精细操作"),
        ("XR 头显", 7.2, 9.0, 1300, CLR["orange"],
         "双手姿态映射\n灵巧操作 / 人形全身"),
    ]
    for name, x, y, size, color, desc in pts:
        ax.scatter(x, y, s=size, color=color, alpha=0.85, zorder=5,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(name, (x, y), xytext=(0, 26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13.5, fontweight="bold",
                    color=color, zorder=6)
        ax.annotate(desc, (x, y), xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=10.5, color=CLR["dark"],
                    linespacing=1.5)

    ax.set_xlim(0.2, 9.0)
    ax.set_ylim(2.2, 10.6)
    ax.set_xlabel("上手成本  →", fontsize=13)
    ax.set_ylabel("控制表达力 / 自由度  →", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.18, ls="--")

    ax.set_title("遥操作设备怎么选：上手成本 × 控制表达力\n"
                 "（Isaac Lab 内置支持：键盘 / 手柄 / SpaceMouse / XR）",
                 fontsize=14.5, fontweight="bold", pad=14)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig3 动捕数据流
# --------------------------------------------------------------------------
def fig3(out: Path) -> None:
    inner = f'''
  SRC [label="动捕源\\n光学（Vicon）· 惯性（Xsens）\\n单目视频姿态估计（SMPL）",
       fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];
  RAW [label="原始运动数据\\nBVH / SMPL 参数",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  RET [label="重定向 Retargeting\\n人体骨骼 → 机器人关节\\n（可用性最大的瓶颈）",
       fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  NPZ [label="机器人运动轨迹 .npz\\ndof 关节轨迹 + body 刚体轨迹（60 fps）",
       fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  AMP [label="AMP 运动先验\\n策略在物理仿真中\\n学会“像人一样动”",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  TRK [label="运动跟踪任务\\n机器人逐帧跟踪\\n参考轨迹",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];

  SRC -> RAW [label="录制"];
  RAW -> RET [label="转换"];
  RET -> NPZ [label="落地"];
  NPZ -> AMP [label="训练（模仿风格）"];
  NPZ -> TRK [label="训练（复现轨迹）"];
'''
    render_dot(inner, out, "动捕数据流：从人体动作到机器人技能",
               rankdir="TB", nodesep="0.55", ranksep="0.42", graph_extra=', width="5.0"')


# --------------------------------------------------------------------------
# fig4 真实动捕数据可视化（运动残影）
# --------------------------------------------------------------------------
EDGES = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14)]


def fig4(out: Path, motions_dir: Path) -> None:
    setup_matplotlib()
    d = np.load(motions_dir / "humanoid_walk.npz", allow_pickle=True)
    bp = np.asarray(d["body_positions"])          # (N, 15, 3)
    fps = int(d["fps"])
    n = len(bp)
    t = np.arange(n) / fps

    fig = plt.figure(figsize=(12.8, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.06, right=0.97)

    # (a) 残影骨架（x-z 侧视,按时间着色）
    ax = fig.add_subplot(gs[0])
    step = 3
    cmap = plt.get_cmap("plasma")
    segs, cols = [], []
    for i in range(0, n, step):
        c = cmap(0.12 + 0.78 * i / n)
        for a, b in EDGES:
            segs.append([(bp[i, a, 0], bp[i, a, 2]), (bp[i, b, 0], bp[i, b, 2])])
            cols.append(c)
    lc = LineCollection(segs, colors=cols, linewidths=1.3, alpha=0.07)
    ax.add_collection(lc)
    # 每隔 0.25 秒描一帧清晰骨架（步态连环画）
    for i in range(0, n, int(0.25 * fps)):
        c = cmap(0.12 + 0.78 * i / n)
        for a, b in EDGES:
            ax.plot([bp[i, a, 0], bp[i, b, 0]], [bp[i, a, 2], bp[i, b, 2]],
                    color=c, lw=2.2, alpha=0.95, solid_capstyle="round")
    ax.set_xlim(bp[..., 0].min() - 0.25, bp[..., 0].max() + 0.25)
    ax.set_ylim(-0.08, bp[..., 2].max() + 0.15)
    ax.set_aspect("equal")
    ax.set_title("① 步态连环画：154 帧骨架叠加（侧视投影）\n颜色由早到晚：紫 → 黄（向右前进）",
                 fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("高度 z (m)")
    ax.grid(alpha=0.2, ls="--")

    # (b) 骨盆高度 + 双脚水平轨迹
    ax = fig.add_subplot(gs[1])
    pelvis = bp[:, 0, :]
    lfoot, rfoot = bp[:, 14, :], bp[:, 11, :]
    ax.plot(t, pelvis[:, 2], lw=2.0, color=CLR["blue"], label="骨盆高度")
    ax.plot(t, lfoot[:, 2], lw=1.3, color=CLR["teal"], label="左脚高度")
    ax.plot(t, rfoot[:, 2], lw=1.3, color=CLR["orange"], label="右脚高度")
    ax.set_title("② 骨盆与双脚高度随时间变化\n（迈步的交替节奏）",
                 fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("高度 (m)")
    ax.grid(alpha=0.25, ls="--")
    ax.legend(fontsize=10.5, frameon=False, ncol=3, loc="upper center")

    fig.suptitle(f"真实动捕数据长什么样：humanoid_walk.npz（{n} 帧 @ {fps} fps，约 {n / fps:.1f} 秒）",
                 fontsize=15.5, fontweight="bold", y=0.965)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig5 真机 UMI ↔ 仿真复刻
# --------------------------------------------------------------------------
def _two_col_rows(rows, out: Path, title: str, left_tag: str, right_tag: str) -> None:
    """左列（橙=真机）/ 右列（蓝=仿真）的多行映射图。"""
    lines = []
    for i, (a, b) in enumerate(rows, 1):
        lines.append(f'A{i} [label="{left_tag}\\n{a}", fillcolor="{CLR["orange_fill"]}", '
                     f'color="{CLR["orange"]}"];')
        lines.append(f'B{i} [label="{right_tag}\\n{b}", fillcolor="{CLR["blue_fill"]}", '
                     f'color="{CLR["blue"]}"];')
        lines.append(f'{{ rank=same; A{i}; B{i}; }}')
        lines.append(f'A{i} -> B{i} [dir=both];')
    for i in range(1, len(rows)):
        lines.append(f"A{i} -> A{i + 1} [style=invis];")
        lines.append(f"B{i} -> B{i + 1} [style=invis];")
    inner = "\n  ".join(lines)
    render_dot(inner, out, title, rankdir="TB", nodesep="0.8", ranksep="0.38")


def fig5(out: Path) -> None:
    rows = [
        ("手持夹爪 + GoPro（腕部同轴视角）", "Franka + 腕部相机（wrist_cam 同轴视角）"),
        ("SLAM 恢复 6DoF 轨迹（后处理）", "真值位姿直接输出（免标定、免 SLAM）"),
        ("相对位姿动作 + 视频 → 训练", "同构动作语义 + 视频 → 训练"),
    ]
    _two_col_rows(rows, out, "UMI：真机流程  ↔  Isaac 仿真复刻", "真机 UMI", "仿真复刻",
                  )


# --------------------------------------------------------------------------
# fig6 人类 EGO ↔ 机器人 POV
# --------------------------------------------------------------------------
def fig6(out: Path) -> None:
    rows = [
        ("头戴设备（Aria / GoPro）第一人称视角", "机器人胸部相机（G1 robot_pov_cam）"),
        ("“人眼看着手做事”的视频记录", "机器人自主执行，无重定向环节"),
        ("手部姿态估计 + 重定向 → 机器人动作", "移动操作 SDG 数据集自带 POV 视频流"),
    ]
    _two_col_rows(rows, out, "EGO：人类第一人称  ↔  机器人 POV 视角", "人类 EGO", "仿真 POV")


# --------------------------------------------------------------------------
# fig7 自动化光谱
# --------------------------------------------------------------------------
def fig7(out: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.04, right=0.97)

    points = [
        (0.06, "遥操作录数", "人全程在环 · 每小时几十条", CLR["blue"], 1500),
        (0.30, "脚本化生成", "脚本策略自动跑 · 需人设计", CLR["gray"], 1200),
        (0.53, "Mimic 增广", "人给种子 · 机器放大 10 倍+", CLR["purple"], 1600),
        (0.78, "RL 自采集", "人给奖励 · 策略自己练", CLR["teal"], 1300),
        (0.94, "SDG 合成", "全自动批量生成 + 标注", CLR["green"], 1400),
    ]
    # 底部长条
    grad = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(grad, extent=[0, 1, 0.50, 0.76], aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "g", ["#DCE9FA", "#D7F0DE"]), alpha=0.9, zorder=1)
    for x, name, desc, color, size in points:
        ax.scatter(x, 0.63, s=size, color=color, zorder=5, alpha=0.92,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(name, (x, 0.63), xytext=(0, 24), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                    color=color, zorder=6)
        ax.annotate(desc, (x, 0.63), xytext=(0, -50), textcoords="offset points",
                    ha="center", va="top", fontsize=10.5, color=CLR["dark"])

    ax.annotate("", xy=(0.99, 0.22), xytext=(0.01, 0.22),
                arrowprops=dict(arrowstyle="->", lw=2.4, color=CLR["gray"]))
    ax.text(0.5, 0.095, "人工介入递减  →  自动化程度递增",
            ha="center", fontsize=13, color=CLR["gray"], fontweight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title("仿真数据生成的“自动化光谱”：从人全程在环，到全自动批量生产",
                 fontsize=15, fontweight="bold", pad=16)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig8 推荐学习路线
# --------------------------------------------------------------------------
def fig8(out: Path) -> None:
    inner = f'''
  S1 [label="① 遥操作\\n产出：HDF5 演示集\\n难度 ★☆☆（30 分钟）",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  S2 [label="② 仿真采集（Mimic）\\n产出：海量增广演示\\n难度 ★★☆（1 小时）",
      fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  S3 [label="③ UMI\\n产出：腕部视觉数据\\n难度 ★★☆（1 小时）",
      fillcolor="{CLR["teal_fill"]}", color="{CLR["teal"]}"];
  S4 [label="④ EGO\\n产出：第一人称视频\\n难度 ★★★（半天）",
      fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
  S5 [label="⑤ 动捕\\n产出：全身技能轨迹\\n难度 ★★★（进阶）",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];

  S1 -> S2 [label="种子数据"];
  S2 -> S3 [label="换视觉"];
  S3 -> S4 [label="换视角"];
  S4 -> S5 [label="换模态"];
'''
    render_dot(inner, out, "推荐上手路线：按“上手成本递增”从种子数据走到全身技能",
               rankdir="TB", nodesep="0.7", ranksep="0.4", fontsize="12",
               graph_extra=', width="5.0"')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
FIGS = {
    "fig1": ("fig1-five-cards.png", lambda out, md: fig1(out)),
    "fig2": ("fig2-teleop-matrix.png", lambda out, md: fig2(out)),
    "fig3": ("fig3-mocap-pipeline.png", lambda out, md: fig3(out)),
    "fig4": ("fig4-mocap-data.png", lambda out, md: fig4(out, md)),
    "fig5": ("fig5-umi-comparison.png", lambda out, md: fig5(out)),
    "fig6": ("fig6-ego-comparison.png", lambda out, md: fig6(out)),
    "fig7": ("fig7-automation-spectrum.png", lambda out, md: fig7(out)),
    "fig8": ("fig8-learning-path.png", lambda out, md: fig8(out)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--motions", type=Path, default=DEFAULT_MOTIONS,
                    help="动捕数据目录（fig4 数据源）")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定图")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = args.only if args.only else list(FIGS)
    for name in todo:
        fname, fn = FIGS[name]
        fn(args.out / fname, args.motions)
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
