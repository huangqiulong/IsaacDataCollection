#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为《遥操作实操手册：Franka 摞方块（键盘 / Xbox360 / 鼠标）（二）》生成全部配图（PNG）。

生成 6 张图到 IsaacDataCollection/docs/images/isaac-sim-02/：
  fig1  叠方块任务与目标状态示意                       [matplotlib]
  fig2  键盘键位与功能分区                             [matplotlib]
  fig3  录制会话状态机（成功/失败/自动保存）           [graphviz]
  fig4  实测数据体检报告（3 条键盘演示）               [matplotlib]
  fig5  三级质量评估金字塔                             [matplotlib]
  fig6  数据质量闭环：评估结果反哺录制                 [graphviz]

用法（isaaclab venv 内置 numpy/matplotlib）：
    ~/code/isaac/isaaclab/.venv/bin/python IsaacDataCollection/scripts/isaac_sim_02_figures.py
    ... --only fig1 fig4        # 只生成指定图

依赖：graphviz（dot CLI）；numpy + matplotlib；fig4 需要 stats JSON（--stats 指定，默认本机数据集目录）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from string import Template

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon

# --------------------------------------------------------------------------
# 常量与样式（与系列其他篇保持一致）
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-02"
DEFAULT_STATS = (Path.home() / "code" / "isaac" / "IsaacDataCollection" / "datasets" /
                 "stats_keyboard.json")

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
# fig1 任务与目标状态
# --------------------------------------------------------------------------
def fig1(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.8, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    colors = {"蓝": "#3B7DD8", "红": "#D64545", "绿": "#3FA66B"}

    # 左：初始状态（三块散放）
    ax.text(23, 91, "初始：三块积木散放", ha="center", fontsize=15,
            fontweight="bold", color=CLR["dark"])
    ax.add_patch(FancyBboxPatch((6, 24), 34, 56, boxstyle="round,pad=0.8,rounding_size=1.6",
                                facecolor="#F7F9FB", edgecolor="#C8D0D8", linewidth=1.5))
    for name, (cx, cy) in {"蓝": (14, 40), "红": (25, 62), "绿": (28, 34)}.items():
        ax.add_patch(FancyBboxPatch((cx, cy), 12, 9, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    facecolor=colors[name], edgecolor="white", linewidth=1.5))
        ax.text(cx + 6, cy + 4.5, f"{name}块", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
    ax.text(23, 15, "蓝块是底座，全程不动", ha="center", fontsize=12, color=CLR["gray"])

    # 中：箭头
    ax.annotate("", xy=(56, 50), xytext=(45, 50),
                arrowprops=dict(arrowstyle="-|>", lw=3, color=CLR["gray"]))
    ax.text(50.5, 57, "叠放", ha="center", fontsize=12.5, color=CLR["gray"])

    # 右：目标状态（叠塔）
    ax.text(77, 91, "目标：叠成塔（从下到上）", ha="center", fontsize=15,
            fontweight="bold", color=CLR["dark"])
    stack_x, stack_w, h = 66, 20, 10
    for i, name in enumerate(["蓝", "红", "绿"]):
        y = 28 + i * h
        ax.add_patch(FancyBboxPatch((stack_x, y), stack_w, h - 0.5,
                                    boxstyle="round,pad=0.25,rounding_size=0.8",
                                    facecolor=colors[name], edgecolor="white", linewidth=1.5))
        ax.text(stack_x + stack_w / 2, y + (h - 0.5) / 2,
                f"{name}（{'底' if i == 0 else '中' if i == 1 else '顶'}）",
                ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    # 顺序标注
    ax.annotate("", xy=(92, 36), xytext=(92, 54),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=CLR["red"]))
    ax.text(93.5, 45, "红叠蓝", fontsize=12, color=CLR["red"], rotation=90, va="center")
    ax.annotate("", xy=(63.5, 46), xytext=(63.5, 64),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=CLR["green"]))
    ax.text(62, 55, "绿叠红", fontsize=12, color=CLR["green"], rotation=90, va="center")

    ax.text(50, 5, "成功判据：三块叠稳（蓝→红→绿）且夹爪松开，程序自动判定 —— 只保存成功演示",
            ha="center", fontsize=12.5, color=CLR["dark"], fontweight="bold")

    fig.suptitle("Franka 摞方块任务：散放 → 叠塔（IsaacContrib-Stack-Cube-Franka-IK-Rel）",
                 fontsize=15.5, fontweight="bold", y=0.99)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig2 键盘键位图
# --------------------------------------------------------------------------
def fig2(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.8, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    move = set("WASDQE")     # 平移
    rot = set("ZXTGCV")      # 旋转
    rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM")]
    key_w, key_h, gap = 7.2, 8.6, 1.35

    y0 = 74
    for r, row in enumerate(rows):
        offset = 2.5 + r * 4.0
        for c, ch in enumerate(row):
            x = 8 + offset + c * (key_w + gap)
            y = y0 - r * (key_h + 1.8)
            if ch in move:
                face, fcolor = CLR["blue"], "white"
            elif ch in rot:
                face, fcolor = CLR["orange"], "white"
            elif ch == "K":
                face, fcolor = CLR["green"], "white"
            elif ch in ("L", "R"):
                face, fcolor = CLR["purple"], "white"
            else:
                face, fcolor = "#E3E8ED", "#8A97A3"
            ax.add_patch(FancyBboxPatch((x, y), key_w, key_h,
                                        boxstyle="round,pad=0.15,rounding_size=0.9",
                                        facecolor=face, edgecolor="white", linewidth=1.4))
            ax.text(x + key_w / 2, y + key_h / 2, ch, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=fcolor)

    # 图例
    legend = [
        (CLR["blue"], "平移：W/S 前后 · A/D 左右 · Q/E 升降"),
        (CLR["orange"], "旋转：Z/X 绕 x（roll）· T/G 绕 y（pitch）· C/V 绕 z（yaw）"),
        (CLR["green"], "夹爪：K 开 / 合切换"),
        (CLR["purple"], "复位：L 清空增量 · R 重开当前演示（录制中）"),
    ]
    y = 30
    for color, text in legend:
        ax.add_patch(FancyBboxPatch((10, y), 2.6, 2.6,
                                    boxstyle="round,pad=0.1,rounding_size=0.5",
                                    facecolor=color, edgecolor="white"))
        ax.text(14.5, y + 1.3, text, fontsize=12.5, va="center", color=CLR["dark"])
        y -= 6.6

    ax.text(50, 4.5, "手法：点按 = 小步微调（约 2~3 cm/步），按住 = 持续移动；接近积木时用点按",
            ha="center", fontsize=12.5, color=CLR["gray"], fontweight="bold")

    fig.suptitle("Isaac Lab 键盘遥操作：默认键位与功能分区",
                 fontsize=15.5, fontweight="bold", y=0.98)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig3 录制会话状态机
# --------------------------------------------------------------------------
def fig3(out: Path) -> None:
    inner = f'''
  START [label="启动录制会话\\nteleop record --num_demos N",
         fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  OP [label="操作任务\\n红块叠到蓝块 → 绿块叠到红块",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  FAIL [label="失败：碰倒 / 抓歪\\n按 R 重开本条\\n（失败的轨迹不会入库）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
  SUCC [label="成功：叠稳并松开夹爪\\n保持 10 步 → 自动判定成功\\n自动保存 demo_N",
        fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  NEXT [label="自动重置场景\\n开始下一条", fillcolor="{CLR["gray_fill"]}",
        color="{CLR["gray"]}"];
  DONE [label="攒满 N 条 → 会话自动退出\\n（中途 Ctrl+C 也可安全退出，已保存的都在）",
        fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];

  START -> OP;
  OP -> FAIL [label="失败"];
  FAIL -> OP [label="R 重开"];
  OP -> SUCC [label="成功稳定"];
  SUCC -> NEXT [penwidth=1.9, color="{CLR["green"]}"];
  NEXT -> OP [label="未满 N 条"];
  NEXT -> DONE [label="已满 N 条"];
'''
    render_dot(inner, out, "一次录制会话的完整流转：失败重开，成功自动保存",
               rankdir="TB", nodesep="0.5", ranksep="0.42", graph_extra=', width="5.2"')


# --------------------------------------------------------------------------
# fig4 实测数据体检报告
# --------------------------------------------------------------------------
def fig4(out: Path, stats_path: Path) -> None:
    setup_matplotlib()
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    demos = data["demos"]
    names = [d["demo"] for d in demos]
    steps = [d["steps"] for d in demos]
    toggles = [d["grip_toggles"] for d in demos]
    jerk = [d["jerk"] for d in demos]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.06, right=0.97, wspace=0.28)
    fig.suptitle("实测数据体检报告：dataset_keyboard.hdf5（3 条键盘演示）",
                 fontsize=15.5, fontweight="bold", y=0.98)

    # (a) 步数
    ax = axes[0]
    colors = [CLR["orange"] if s > 1200 else CLR["green"] for s in steps]
    bars = ax.bar(names, steps, color=colors, width=0.55)
    for rect, v in zip(bars, steps):
        ax.annotate(f"{v}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
    ax.axhline(1200, color=CLR["gray"], ls="--", lw=1.1)
    ax.annotate("预警线 1200", xy=(2.42, 1200), fontsize=10,
                color=CLR["gray"], va="bottom", ha="right")
    ax.annotate("疑似过长：失败后未按 R？", xy=(0, 1301), xytext=(0.28, 1060),
                fontsize=10, color=CLR["orange"],
                arrowprops=dict(arrowstyle="->", color=CLR["orange"], lw=1.2))
    ax.set_ylim(0, max(steps) * 1.22)
    ax.set_ylabel("步数")
    ax.set_title("① 演示长度（步，20 Hz 记录）", fontsize=13,
                 fontweight="bold", pad=8)
    ax.grid(alpha=0.22, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) 夹爪切换次数
    ax = axes[1]
    bars = ax.bar(names, toggles, color=CLR["green"], width=0.55)
    for rect, v in zip(bars, toggles):
        ax.annotate(f"{v} 次", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
    ax.axhline(2, color=CLR["gray"], ls="--", lw=1.1)
    ax.annotate("健康线 ≥2", xy=(-0.42, 2), fontsize=10, color=CLR["gray"], va="bottom")
    ax.set_ylim(0, max(toggles) * 1.35)
    ax.set_title("② 夹爪开合次数（抓放流程执行）", fontsize=13, fontweight="bold", pad=8)
    ax.grid(alpha=0.22, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c) jerk
    ax = axes[2]
    bars = ax.bar(names, jerk, color=CLR["purple"], width=0.55)
    for rect, v in zip(bars, jerk):
        ax.annotate(f"{v * 1000:.2f}e-3", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, max(jerk) * 1.3)
    ax.set_title("③ 动作抖动 jerk（越小越平滑）", fontsize=13, fontweight="bold", pad=8)
    ax.grid(alpha=0.22, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig5 三级质量评估金字塔
# --------------------------------------------------------------------------
def fig5(out: Path) -> None:
    setup_matplotlib()
    fig = plt.figure(figsize=(11.2, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    levels = [
        # (底部y, 顶部y, 底宽, 颜色)
        (56, 78, 58, CLR["purple"], "L3 · 训练可用性（终极判据）",
         "Mimic 生成成功率 ≥30%（及格）· BC 策略成功率 >50%"),
        (34, 56, 74, CLR["blue"], "L2 · 回放自洽",
         "teleop replay：动作序列在物理里可复现（如 10/10 成功）"),
        (12, 34, 90, CLR["green"], "L1 · 数值筛查（最便宜，每次录完必跑）",
         "统计脚本过一遍：无警告、指标在参考区间"),
    ]
    cx = 50
    for y0, y1, w, color, title, desc in levels:
        top_w = w * 0.62
        pts = [(cx - w / 2, y0), (cx + w / 2, y0), (cx + top_w / 2, y1), (cx - top_w / 2, y1)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.85,
                             edgecolor="white", linewidth=2))
        ax.text(cx, (y0 + y1) / 2 + 3.4, title, ha="center", va="center",
                fontsize=13.5, color="white", fontweight="bold")
        ax.text(cx, (y0 + y1) / 2 - 4.6, desc, ha="center", va="center",
                fontsize=11, color="white")

    ax.annotate("", xy=(93, 80), xytext=(93, 14),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=CLR["gray"]))
    ax.text(95.5, 47, "证据强度 / 验证成本递增", fontsize=12, color=CLR["gray"],
            rotation=90, va="center")

    fig.suptitle("数据质量的“三级评估”：从便宜筛查到终极判据",
                 fontsize=15.5, fontweight="bold", y=0.99)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig6 质量闭环
# --------------------------------------------------------------------------
def fig6(out: Path) -> None:
    inner = f'''
  REC [label="录制种子演示\\n（10~30 条）", fillcolor="{CLR["blue_fill"]}",
       color="{CLR["blue"]}"];
  ANN [label="子任务标注\\nannotate --auto", fillcolor="{CLR["blue_fill"]}",
       color="{CLR["blue"]}"];
  GEN [label="Mimic 规模化生成\\n（目标 100~1000 条）",
       fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  TRA [label="BC 训练\\nrobomimic", fillcolor="{CLR["purple_fill"]}",
       color="{CLR["purple"]}"];
  EVAL [label="回放评估\\n生成成功率 / 策略成功率",
        fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  DEC  [label="闭环决策", shape=diamond, style="filled", fillcolor="{CLR["orange_fill"]}",
        color="{CLR["orange"]}", fontsize=13];
  SCALE [label="扩规模\\n补录到 20~30 条 → 生成 1000 条",
         fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];

  REC -> ANN -> GEN -> TRA -> EVAL -> DEC;
  DEC -> SCALE [label="生成成功率 ≥30%\\n且 BC >50%", color="{CLR["green"]}",
                fontcolor="{CLR["green"]}"];
  DEC -> REC [label="生成成功率 <20%：\\n统一流程重录、检查标注", color="{CLR["orange"]}",
              fontcolor="{CLR["orange"]}", penwidth=1.8, constraint=false];
'''
    render_dot(inner, out, "数据质量闭环：评估成功率决定下一步行动",
               rankdir="TB", nodesep="0.6", ranksep="0.42", fontsize="12",
               graph_extra=', width="5.6"')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
FIGS = {
    "fig1": ("fig1-task-scene.png", lambda out, st: fig1(out)),
    "fig2": ("fig2-keyboard.png", lambda out, st: fig2(out)),
    "fig3": ("fig3-session-flow.png", lambda out, st: fig3(out)),
    "fig4": ("fig4-quality-report.png", lambda out, st: fig4(out, st)),
    "fig5": ("fig5-quality-levels.png", lambda out, st: fig5(out)),
    "fig6": ("fig6-closed-loop.png", lambda out, st: fig6(out)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                    help="统计 JSON（fig4 数据源）")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定图")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = args.only if args.only else list(FIGS)
    for name in todo:
        fname, fn = FIGS[name]
        fn(args.out / fname, args.stats)
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
