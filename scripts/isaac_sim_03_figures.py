#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为《Isaac Sim 仿真遥操采集与真机采集对比（三）》生成全部配图（PNG）。

生成 8 张图到 IsaacDataCollection/docs/images/isaac-sim-03/：
  fig1  真机采集方式 ↔ Isaac Sim 仿真对应（映射图）          [graphviz]
  fig2  Isaac Sim 仿真数据生产方法总览（四类方法 → 汇流）      [graphviz]
  fig3  仿真遥操作工作流（run / record / replay）             [graphviz]
  fig4  实测 HDF5 演示数据曲线（真实数据）                    [matplotlib]
  fig5  Mimic 演示增广五步管线                                [graphviz]
  fig6  规模化实测：30 条种子 → 300 条生成演示                [matplotlib]
  fig7  仿真 vs 真机 六维能力对照（定性示意）                  [matplotlib]
  fig8  混合策略：仿真预训练 + 真机微调                        [graphviz]

用法（isaaclab venv 内置 h5py/matplotlib）：
    source ~/code/isaac/isaaclab/.venv/bin/activate
    python IsaacDataCollection/scripts/isaac_survey_figures.py                # 全部
    python IsaacDataCollection/scripts/isaac_survey_figures.py --only fig4 fig6
    python IsaacDataCollection/scripts/isaac_survey_figures.py --fig4-demo demo_2

依赖：graphviz（dot CLI）；h5py + numpy + matplotlib。
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

# --------------------------------------------------------------------------
# 常量与样式
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-03"
DEFAULT_DATASETS = Path.home() / "code" / "isaac" / "isaaclab" / "datasets"

FONT = "Noto Sans CJK SC"

# 全系列统一语义色
CLR = {
    "font": FONT,
    "dpi": 150,
    "dark": "#1F2933",
    "blue": "#3B7DD8",   "blue_fill": "#E8F1FC",      # 仿真 / 遥操作产出
    "orange": "#E8833A", "orange_fill": "#FDF0E3",    # 真机
    "green": "#3FA66B",  "green_fill": "#E6F5EC",     # 成功 / 数据资产
    "purple": "#8E64C8", "purple_fill": "#F1EBFA",    # 增广 / 训练
    "gray": "#5A6570",   "gray_fill": "#EEF1F4",      # 中性
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
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["figure.dpi"] = 110


def style_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=13.5, pad=8, fontweight="bold")
    ax.grid(alpha=0.25, ls="--", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# fig1 真机采集方式 ↔ 仿真对应
# --------------------------------------------------------------------------
def fig1(out: Path) -> None:
    pairs = [
        ("真机 · 遥操作\\n主从臂 / VR / 手柄", "仿真 · 遥操作录数\\n键盘 / SpaceMouse / 手柄 / XR"),
        ("真机 · 动捕\\nVicon / Xsens / 视频姿态估计", "仿真 · 动捕先验\\nAMP 任务 / 运动跟踪"),
        ("真机 · UMI 手持夹爪\\nGoPro + 夹爪", "仿真 · 腕部相机录数\\nVisuomotor 任务"),
        ("真机 · EGO 第一人称\\n头戴设备", "仿真 · 机器人 POV 相机\\n移动操作 SDG"),
    ]
    lines = []
    for i, (a, b) in enumerate(pairs, 1):
        lines.append(f'A{i} [label="{a}", fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];')
        lines.append(f'B{i} [label="{b}", fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];')
        lines.append(f'{{ rank=same; A{i}; B{i}; }}')
        lines.append(f'A{i} -> B{i} [dir=both];')
    lines.append("A1 -> A2 -> A3 -> A4 [style=invis];")
    lines.append("B1 -> B2 -> B3 -> B4 [style=invis];")
    lines.append(
        f'A5 [label="真机侧没有对应方式", fillcolor="white", color="#B8C0C8", '
        f'fontcolor="#8794A1", style="rounded,dashed"];')
    lines.append(
        f'B5 [label="仿真独有 · 生成式采集\\nMimic 演示增广 / RL 自采集 / SDG 合成数据", '
        f'fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];')
    lines.append("{ rank=same; A5; B5; }")
    lines.append("A4 -> A5 [style=invis]; B4 -> B5 [style=invis];")
    lines.append('A5 -> B5 [style=invis];')
    inner = "\n  ".join(lines)
    render_dot(inner, out, "真机采集方式  ↔  Isaac Sim 仿真对应", nodesep="0.8",
               ranksep="0.38")


# --------------------------------------------------------------------------
# fig2 仿真数据生产方法总览
# --------------------------------------------------------------------------
def fig2(out: Path) -> None:
    inner = f'''
  PLAT [label="Isaac Sim / Isaac Lab\\n物理与传感器仿真 · 任务套件 · 并行环境", fontsize=15,
        penwidth=1.9, fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];

  M1 [label="① 遥操作录数\\n人类操控 · 种子演示\\n产出：HDF5 演示集"];
  M2 [label="② Mimic 演示增广\\n少量种子 → 海量演示\\n产出：HDF5 演示集"];
  M3 [label="③ RL 自采集\\n策略试错 · rollout 即数据\\n产出：轨迹 / 策略权重"];
  M4 [label="④ SDG 合成数据\\nReplicator · 完美标注\\n产出：图像 / 分割 / 位姿集"];
  M1 [fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  M2 [fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  M3 [fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  M4 [fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  {{ rank=same; M1; M2; M3; M4; }}
  PLAT -> M1; PLAT -> M2; PLAT -> M3; PLAT -> M4;

  DATA [label="统一数据格式：HDF5（robomimic 兼容） / LeRobot\\n全状态真值 · 自动成功标记 · 严格时间同步",
        fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}", fontsize=13];
  TRAIN [label="策略训练与评估\\nBC / ACT / 扩散策略 / RL · 回放评估", fontsize=14,
         fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];

  M1 -> DATA; M2 -> DATA; M3 -> DATA; M4 -> DATA;
  DATA -> TRAIN [penwidth=2.0];
'''
    render_dot(inner, out, "Isaac Sim 仿真数据生产：四类方法 · 统一汇流", nodesep="0.35",
               fontsize="14")


# --------------------------------------------------------------------------
# fig3 遥操作工作流
# --------------------------------------------------------------------------
def fig3(out: Path) -> None:
    inner = f'''
  DEV [label="遥操作设备\\n键盘 / Xbox 手柄 / SpaceMouse / XR 头显",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  LOOP [label="末端增量控制回路\\n相对位姿增量 + 夹爪开合", fillcolor="{CLR["gray_fill"]}",
        color="{CLR["gray"]}"];
  RUN [label="teleop run\\n热身：只操作、不保存", fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];
  REC [label="teleop record\\n录制演示", fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  H5  [label="dataset.hdf5\\n成功演示逐条保存 demo_N\\n（20 Hz 固定步长）",
       fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  RPL [label="teleop replay\\n回放校验（逐值复现）", fillcolor="{CLR["purple_fill"]}",
       color="{CLR["purple"]}"];

  DEV -> LOOP [label="实时指令"];
  LOOP -> RUN [label="run 模式"];
  LOOP -> REC [label="record 模式"];
  REC -> H5 [label="任务成功判定自动通过\\n（只保存成功演示）", color="{CLR["green"]}",
             fontcolor="{CLR["green"]}", penwidth=1.9];
  H5 -> RPL [label="读取动作回放"];
'''
    render_dot(inner, out, "仿真遥操作工作流：run / record / replay", rankdir="TB",
               nodesep="0.55", ranksep="0.42", graph_extra=', width="5.2"')


# --------------------------------------------------------------------------
# fig4 实测 HDF5 数据曲线（真实数据）
# --------------------------------------------------------------------------
def fig4(out: Path, ds_root: Path, demo: str = "demo_1") -> None:
    setup_matplotlib()
    f = h5py.File(ds_root / "20261012" / "dataset_keyboard.hdf5", "r")
    g = f["data"][demo]
    actions = np.asarray(g["actions"])
    eef = np.asarray(g["obs/eef_pos"])
    ok = bool(g.attrs.get("success", False))
    f.close()

    dt = 0.05
    n = len(actions)
    t = np.arange(n) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.24, top=0.88, bottom=0.09,
                        left=0.07, right=0.97)
    fig.suptitle(f"Isaac Lab 录制的演示数据（HDF5 · {demo} · {n} 步 ≈ {n * dt:.0f} 秒 · 任务成功）",
                 fontsize=16, fontweight="bold", y=0.97)

    # ① 平移动作（颜色语义与子图③一致，颜色含义写在标题里，避免图例遮挡脉冲数据）
    ax = axes[0][0]
    for i, col in enumerate([CLR["blue"], CLR["orange"], CLR["green"]]):
        ax.plot(t, actions[:, i], lw=1.15, color=col)
    style_ax(ax, "① 动作指令：末端平移增量（Δx/Δy/Δz = 蓝/橙/绿）")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("增量 (m)")

    # ② 夹爪动作
    ax = axes[0][1]
    grip = actions[:, 6]
    ax.step(t, grip, where="post", color=CLR["purple"], lw=1.5)
    ax.fill_between(t, grip, 0, step="post", alpha=0.15, color=CLR["purple"])
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["闭合 −1", "张开 +1"])
    ax.set_ylim(-1.6, 1.6)
    style_ax(ax, "② 动作指令：夹爪开合（离散切换）")
    ax.set_xlabel("t (s)")

    # ③ 末端位置
    ax = axes[1][0]
    for i, (lab, col) in enumerate(zip(["x", "y", "z"],
                                       [CLR["blue"], CLR["orange"], CLR["green"]])):
        ax.plot(t, eef[:, i], lw=1.4, color=col, label=lab)
    style_ax(ax, "③ 观测：末端位置（米 · 世界坐标系）")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("位置 (m)")
    ax.legend(ncol=3, fontsize=10, frameon=False)

    # ④ XY 轨迹
    ax = axes[1][1]
    ax.plot(eef[:, 0], eef[:, 1], lw=1.5, color=CLR["blue"], alpha=0.8)
    ax.scatter(eef[0, 0], eef[0, 1], s=70, color=CLR["green"], zorder=5,
               edgecolor="white", linewidth=1.2, label="起点")
    ax.scatter(eef[-1, 0], eef[-1, 1], s=70, color=CLR["red"], zorder=5,
               edgecolor="white", linewidth=1.2, label="终点")
    ax.set_aspect("equal")
    style_ax(ax, "④ 末端 XY 轨迹（俯视）：完整的“取-放”路径")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=10, frameon=False)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig5 Mimic 五步管线
# --------------------------------------------------------------------------
def fig5(out: Path) -> None:
    inner = f'''
  N1 [label="① 种子演示\\n少量人工遥操作演示\\n（或官方预录数据集）",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  N2 [label="② 子任务标注\\n自动 / 手动切分动作段\\nannotate_demos.py",
      fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  N3 [label="③ 规模化生成（×10 量级）\n并行环境自动重拼接 · 随机物位\n失败轨迹自动丢弃\ngenerate_dataset.py",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  N4 [label="④ 行为克隆训练\nBC / BC-RNN / 扩散策略\ntrain.py",
      fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  N5 [label="⑤ 回放评估\n自动判定成功率\nplay.py",
      fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];

  N1 -> N2 [label="切分"];
  N2 -> N3 [label="重拼接生成"];
  N3 -> N4 [label="训练"];
  N4 -> N5 [label="评估"];
'''
    render_dot(inner, out, "Mimic 演示增广五步管线（Isaac Lab）", rankdir="TB",
               nodesep="0.5", ranksep="0.38", graph_extra=', width="5.0"')


# --------------------------------------------------------------------------
# fig6 规模化实测（真实数据）
# --------------------------------------------------------------------------
def fig6(out: Path, ds_root: Path) -> None:
    setup_matplotlib()

    f = h5py.File(ds_root / "dataset_gamepad.hdf5", "r")
    it = f["data"]
    seed_lens = np.array([int(it[k].attrs.get("num_samples", 0)) for k in it.keys()])
    f.close()
    f = h5py.File(ds_root / "generated_gamepad_r2_300.hdf5", "r")
    it = f["data"]
    gen_lens = np.array([int(it[k].attrs.get("num_samples", 0)) for k in it.keys()])
    f.close()

    seed_n, gen_n = len(seed_lens), len(gen_lens)
    seed_f, gen_f = int(seed_lens.sum()), int(gen_lens.sum())

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.45]})
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.98, wspace=0.34)
    fig.suptitle("同一条任务：30 条人工种子演示 → Mimic 自动生成 300 条（实测）",
                 fontsize=16.5, fontweight="bold", y=0.98)

    # (a) 演示条数
    ax = axes[0]
    bars = ax.bar([0, 1], [seed_n, gen_n], width=0.52,
                  color=[CLR["blue"], CLR["purple"]])
    for rect, v in zip(bars, [seed_n, gen_n]):
        ax.annotate(f"{v} 条", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    ax.annotate("×10", xy=(0.5, max(seed_n, gen_n) * 0.62), ha="center",
                fontsize=15, color=CLR["purple"], fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["人工种子\n（遥操作录制）", "Mimic 生成\n（全自动）"], fontsize=10.5)
    ax.set_ylim(0, gen_n * 1.25)
    style_ax(ax, "① 演示条数")

    # (b) 总帧数
    ax = axes[1]
    bars = ax.bar([0, 1], [seed_f, gen_f], width=0.52,
                  color=[CLR["blue"], CLR["purple"]])
    for rect, v in zip(bars, [seed_f, gen_f]):
        ax.annotate(f"{v:,}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=13.5, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    ax.annotate("×9.5", xy=(0.5, max(seed_f, gen_f) * 0.62), ha="center",
                fontsize=15, color=CLR["purple"], fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["人工种子", "Mimic 生成"], fontsize=10.5)
    ax.set_ylim(0, gen_f * 1.25)
    style_ax(ax, "② 总帧数（20 Hz）")

    # (c) 演示长度分布
    ax = axes[2]
    bins = np.arange(400, 1200, 50)
    n1, _, _ = ax.hist(seed_lens, bins=bins, alpha=0.55, color=CLR["blue"], label="人工种子（30 条）")
    n2, _, _ = ax.hist(gen_lens, bins=bins, alpha=0.55, color=CLR["purple"], label="Mimic 生成（300 条）")
    ymax = max(n1.max(), n2.max())
    ax.set_ylim(0, ymax * 1.28)
    med = int(np.median(gen_lens))
    ax.axvline(med, color=CLR["purple"], ls="--", lw=1.2)
    ax.annotate(f"生成中位数 {med} 步", xy=(med + 60, ymax * 1.12), fontsize=10.5,
                color=CLR["purple"], va="center")
    style_ax(ax, "③ 单条演示长度分布")
    ax.set_xlabel("每演示帧数（1 帧 = 50 ms）")
    ax.set_ylabel("演示条数")
    ax.legend(fontsize=10, frameon=False, loc="upper left")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig7 仿真 vs 真机 六维对照（定性示意）
# --------------------------------------------------------------------------
def fig7(out: Path) -> None:
    setup_matplotlib()
    labels = ["数据规模\n与并行度", "成本优势", "数据质量保障\n(真值/标注/同步)",
              "物理·视觉\n真实感", "安全·低风险", "部署直通性"]
    real = [1.0, 1.5, 1.5, 5.0, 2.0, 5.0]
    sim = [5.0, 5.0, 5.0, 3.0, 5.0, 2.5]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    real_c = real + real[:1]
    sim_c = sim + sim[:1]

    fig = plt.figure(figsize=(9.2, 7.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, real_c, color=CLR["orange"], lw=2.4, label="真机遥操作采集")
    ax.fill(angles, real_c, color=CLR["orange"], alpha=0.18)
    ax.plot(angles, sim_c, color=CLR["blue"], lw=2.4, label="Isaac Sim 仿真采集")
    ax.fill(angles, sim_c, color=CLR["blue"], alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8.5, color="#8794A1")
    ax.tick_params(pad=14)
    ax.grid(color="#C8D0D8", alpha=0.9, lw=0.8)
    ax.spines["polar"].set_color("#C8D0D8")

    ax.set_title("仿真 vs 真机：数据采集能力六维对照\n（越靠外 = 该维度优势越强，定性示意）",
                 fontsize=14.5, fontweight="bold", pad=26)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=11.5, frameon=False)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}")


# --------------------------------------------------------------------------
# fig8 混合策略
# --------------------------------------------------------------------------
def fig8(out: Path) -> None:
    inner = f'''
  SIM [label="仿真数据\\n规模大 · 带真值 · 边际成本 ≈ 0",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  PRE [label="大规模预训练\\nBC / ACT / 扩散策略 / RL",
       fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  REAL [label="少量真机演示\n数百条量级 · 逐条人工采集",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
  FT  [label="微调\n（与仿真数据同一格式）",
       fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  DEP [label="真机部署与评估",
       fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];

  SIM -> PRE [label="数千~数十万条演示"];
  PRE -> FT [label="策略初始化", penwidth=1.9];
  REAL -> FT [label="真实数据校准", style=dashed];
  FT -> DEP [label="逐条回放确认"];
'''
    render_dot(inner, out, "混合策略：仿真大规模预训练 + 少量真机数据微调", rankdir="TB",
               nodesep="0.6", ranksep="0.42", graph_extra=', width="5.4"')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
FIGS = {
    "fig1": lambda out, ds: fig1(out),
    "fig2": lambda out, ds: fig2(out),
    "fig3": lambda out, ds: fig3(out),
    "fig5": lambda out, ds: fig5(out),
    "fig8": lambda out, ds: fig8(out),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    ap.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS,
                    help="Isaac Lab datasets 目录（fig4/fig6 数据源）")
    ap.add_argument("--fig4-demo", default="demo_1", help="fig4 使用哪条演示")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定图")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = args.only if args.only else list(FIGS) + ["fig4", "fig6", "fig7"]
    for name in todo:
        out = args.out / {
            "fig1": "fig1-two-routes.png", "fig2": "fig2-methods.png",
            "fig3": "fig3-teleop-workflow.png", "fig4": "fig4-dataset-sample.png",
            "fig5": "fig5-mimic-pipeline.png", "fig6": "fig6-scale-up.png",
            "fig7": "fig7-sim-vs-real.png", "fig8": "fig8-hybrid-training.png",
        }[name]
        if name in FIGS:
            FIGS[name](out, args.datasets)
        elif name == "fig4":
            fig4(out, args.datasets, args.fig4_demo)
        elif name == "fig6":
            fig6(out, args.datasets)
        elif name == "fig7":
            fig7(out)
        else:
            raise SystemExit(f"未知图名: {name}")
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
