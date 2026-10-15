#!/usr/bin/env python3
"""第七篇（LeRobot 数据格式）配图生成。

生成 docs/images/isaac-sim-07/ 下 6 张图：

  fig1-files.png           v3.0 目录结构与文件角色（lerobot/pusht 实测）
  fig2-reading.png         读取链路：episodes 表 → 行区间 / 视频时间段
  fig3-pusht-frames.png    pusht 帧网格（两集，96×96 上采样）
  fig4-episode-curves.png  一条 episode 的 state / action 曲线
  fig5-v2-vs-v3.png        v2.0（EGO 篇转换产物）vs v3.0（pusht）对照
  fig6-hdf5-to-lerobot.png HDF5（Isaac Lab）→ LeRobot 转换链路

用法
----
    python isaac_sim_07_figures.py [--only fig1 fig3] \
        [--pusht <pusht 数据集目录>] [--g1 <G1 LeRobot 目录>] [--out <输出目录>]

依赖：matplotlib + numpy + pandas + pyarrow + av。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from string import Template

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-07"
DEFAULT_PUSHT = ROOT / "datasets" / "lerobot_pusht"
DEFAULT_G1 = ROOT / "datasets" / "lerobot_g1_locomanip"

FONT = "Noto Sans CJK SC"
CLR = {
    "font": FONT, "dpi": 150, "dark": "#1F2933",
    "blue": "#3B7DD8", "blue_fill": "#E8F1FC",
    "orange": "#E8833A", "orange_fill": "#FDF0E3",
    "green": "#3FA66B", "green_fill": "#E6F5EC",
    "purple": "#8E64C8", "purple_fill": "#F1EBFA",
    "teal": "#2E9CA6", "teal_fill": "#E3F4F5",
    "gray": "#5A6570", "gray_fill": "#EEF1F4",
    "red": "#D64545", "edge": "#8794A1",
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
        subprocess.run(["dot", "-Tpng"], input=src.encode("utf-8"), stdout=fh, check=True)
    print(f"[ok] {out.name}")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    prefer = ["Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
              "WenQuanYi Micro Hei", "Droid Sans Fallback"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [n for n in prefer if n in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.facecolor"] = "white"
    return plt


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# fig1：目录结构与文件角色（v3.0 以 pusht 为例）
# ---------------------------------------------------------------------------

def fig1_files(ctx, plt) -> None:
    p: Path = ctx["pusht"]
    rows = [
        ("meta/info.json", "总目录：版本 / 频率 / features schema / 路径模板"),
        ("data/chunk-000/file-000.parquet", "逐帧数据表（state / action / 时间戳 / 任务号 …）"),
        ("meta/episodes/chunk-000/file-000.parquet", "分集元数据：帧区间、视频时间段、分集统计"),
        ("meta/tasks.parquet", "任务表（task_index ↔ 指令文本）"),
        ("meta/stats.json", "全局统计（各特征 min/max/mean/std）"),
        ("videos/observation.image/chunk-000/file-000.mp4", "图像流（全部集共用，靠时间戳切段）"),
    ]
    fig, ax = plt.subplots(figsize=(12.8, 5.6), dpi=140)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    ax.text(0.5, 0.965, "LeRobot v3.0 目录结构（lerobot/pusht 实测，全量 7.5 MB）",
            ha="center", va="top", fontsize=14)
    ax.text(0.5, 0.917, "chunk-000：第 0~999 集存放在同一批文件里（chunks_size=1000）",
            ha="center", va="top", fontsize=10.5, color=CLR["gray"])

    y = 0.84
    for fname, role in rows:
        size = p / fname
        sz = f" ({human(size.stat().st_size)})" if size.exists() else ""
        ax.add_patch(FancyBboxPatch((0.03, y - 0.062), 0.52, 0.085,
                                    boxstyle="round,pad=0.008", fc="#F5F7FA",
                                    ec=CLR["edge"], lw=1.2))
        ax.text(0.05, y - 0.02, fname + sz, fontsize=10,
                family=["DejaVu Sans Mono", "Noto Sans CJK SC"],
                va="center", ha="left")
        ax.add_patch(FancyBboxPatch((0.58, y - 0.062), 0.39, 0.085,
                                    boxstyle="round,pad=0.008",
                                    fc=CLR["blue_fill"], ec=CLR["blue"], lw=1.2))
        ax.text(0.60, y - 0.02, role, fontsize=10, va="center", ha="left")
        y -= 0.115

    fig.savefig(ctx["out"] / "fig1-files.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig1-files.png")


# ---------------------------------------------------------------------------
# fig2：读取链路
# ---------------------------------------------------------------------------

def fig2_reading(ctx, plt) -> None:
    inner = f'''
  I [label="meta/info.json\\n版本 / fps / features schema\\n路径模板 data_path & video_path",
     fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  E [label="meta/episodes/*.parquet\\n每集：帧数 length、\\ndataset_from/to_index、\\n视频 from/to_timestamp",
     fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
  D [label="data/*.parquet\\n逐帧行（25,650 行，pusht）\\nobservation.state / action /\\ntimestamp / task_index …",
     fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  V [label="videos/*.mp4\\n全部集的连续视频\\n（25,650 帧，pusht）",
     fillcolor="{CLR["teal_fill"]}", color="{CLR["teal"]}"];
  T [label="meta/tasks.parquet\\ntask_index ↔ 指令文本",
     fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  I -> D [label=" 解释列含义"];
  I -> V [label=" 视频编码/fps"];
  E -> D [label=" dataset_from/to_index\\n→ 取行区间"];
  E -> V [label=" from/to_timestamp\\n→ 取视频时间段"];
  T -> D [label=" task_index 关联", style=dashed];
'''
    render_dot(inner, ctx["out"] / "fig2-reading.png",
               "读一集数据：先查 episodes 表，再去 data / videos 取片段",
               rankdir="LR", nodesep="0.5", ranksep="0.9", fontsize="12")


# ---------------------------------------------------------------------------
# fig3：pusht 帧网格
# ---------------------------------------------------------------------------

def fig3_frames(ctx, plt) -> None:
    import pandas as pd
    import egosuite_common as ecm

    p: Path = ctx["pusht"]
    info = json.loads((p / "meta/info.json").read_text())
    fps = info["fps"]
    ep = pd.read_parquet(p / "meta/episodes/chunk-000/file-000.parquet")
    video = p / "videos/observation.image/chunk-000/file-000.mp4"

    chosen = []
    for epi in (0, 1):
        row = ep.iloc[epi]
        from_ts, to_ts, length = (float(row["videos/observation.image/from_timestamp"]),
                                  float(row["videos/observation.image/to_timestamp"]),
                                  int(row["length"]))
        base = round(from_ts * fps)
        idx = [base + round(f * (length - 1)) for f in (0.05, 0.35, 0.65, 0.95)]
        chosen.append((epi, idx))
    all_idx = [i for _, idx in chosen for i in idx]
    frames = ecm.decode_selected_frames(video, all_idx)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.4), dpi=140)
    for r, (epi, idx) in enumerate(chosen):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(frames[i][:, :, ::-1], interpolation="nearest")
            ax.set_title(f"ep{epi} · 帧 {i}（t≈{i/fps:.1f}s）", fontsize=9)
            ax.axis("off")
    fig.suptitle("lerobot/pusht 实拍帧（episode 0 / 1，96×96@10fps，上采样显示）", fontsize=13)
    fig.tight_layout()
    fig.savefig(ctx["out"] / "fig3-pusht-frames.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig3-pusht-frames.png")


# ---------------------------------------------------------------------------
# fig4：一集的 state / action 曲线
# ---------------------------------------------------------------------------

def fig4_episode_curves(ctx, plt) -> None:
    import pandas as pd

    p: Path = ctx["pusht"]
    df = pd.read_parquet(p / "data/chunk-000/file-000.parquet")
    ep = df[df["episode_index"] == 0].reset_index(drop=True)
    t = ep["timestamp"].to_numpy()
    state = np.stack([np.asarray(x, dtype=float) for x in ep["observation.state"].to_list()])
    action = np.stack([np.asarray(x, dtype=float) for x in ep["action"].to_list()])

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.2), dpi=140, sharex=True)
    ax = axes[0]
    ax.plot(t, state[:, 0], label="state.x（方块位置）")
    ax.plot(t, state[:, 1], label="state.y")
    ax.set_ylabel("state（像素坐标 0~512）")
    ax.legend(fontsize=9)
    ax.set_title("lerobot/pusht · episode 0（161 帧 @10fps）")
    ax = axes[1]
    ax.plot(t, action[:, 0], color=CLR["orange"], label="action.x（T 型块目标位置）")
    ax.plot(t, action[:, 1], color=CLR["teal"], label="action.y")
    ax.set_ylabel("action（0~512）")
    ax.set_xlabel("timestamp（s）")
    ax.legend(fontsize=9)
    for ax_ in axes:
        ax_.grid(alpha=0.3)
        ax_.axvline(t[-1], color=CLR["gray"], ls=":", lw=1)
    axes[0].annotate("episode 结束（next.done=True）", xy=(t[-1], state[-1, 1]),
                     xytext=(t[-1] - 4.6, state[-1, 1] + 62), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#333333"))
    fig.tight_layout()
    fig.savefig(ctx["out"] / "fig4-episode-curves.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig4-episode-curves.png")


# ---------------------------------------------------------------------------
# fig5：v2.0 vs v3.0 目录对照
# ---------------------------------------------------------------------------

def fig5_v2_vs_v3(ctx, plt) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=140)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    def header(x, text, color, fill):
        ax.add_patch(FancyBboxPatch((x, 0.88), 0.44, 0.10, boxstyle="round,pad=0.01",
                                    fc=fill, ec=color, lw=1.5))
        ax.text(x + 0.22, 0.93, text, ha="center", va="center", fontsize=12.5)

    def row(x, y, text, mono=True):
        ax.text(x + 0.01, y, text, fontsize=9.5, va="center", ha="left",
                family=["DejaVu Sans Mono", "Noto Sans CJK SC"])

    header(0.03, "v2.0：按“集”切文件（EGO 篇 G1 转换产物）", CLR["orange"], CLR["orange_fill"])
    header(0.53, "v3.0：按“文件”分块（lerobot/pusht 实测）", CLR["blue"], CLR["blue_fill"])

    v2_rows = [
        ("data/chunk-000/episode_000000.parquet", "每集一个 Parquet 文件"),
        ("videos/…/episode_000000.mp4", "每集一个视频文件"),
        ("meta/episodes.jsonl", "分集信息（jsonl 文本）"),
        ("meta/tasks.jsonl", "任务表（jsonl 文本）"),
        ("meta/modality.json", "语义分段（GR00T）"),
        ("meta/info.json", "v2 路径模板 episode_{i:06d}"),
    ]
    v3_rows = [
        ("data/chunk-000/file-000.parquet", "整个 chunk 一个文件"),
        ("videos/observation.image/…", "全部集共用一个视频"),
        ("meta/episodes/…/file-000.parquet", "分集表（Parquet）"),
        ("meta/tasks.parquet", "任务表（Parquet）"),
        ("meta/stats.json", "全局统计"),
        ("meta/info.json", "v3 路径模板 file_{i:03d}"),
    ]
    y0 = 0.80
    for r, ((p2, c2), (p3, c3)) in enumerate(zip(v2_rows, v3_rows)):
        y = y0 - r * 0.095
        row(0.03, y, p2)
        ax.text(0.375, y, "# " + c2, fontsize=9.5, va="center", ha="left",
                color=CLR["gray"])
        row(0.53, y, p3)
        ax.text(0.85, y, "# " + c3, fontsize=9.5, va="center", ha="left",
                color=CLR["gray"])

    ax.text(0.5, 0.16, "同样的骨架：info(描述) + 逐帧表(data) + 视频(videos) + 分集信息(episodes) + 任务(tasks)。\n"
                       "v3 把“每集小文件”改成“大文件 + 精确索引”，文件数更少、更适合海量数据。",
            ha="center", fontsize=10.5, color=CLR["dark"])
    ax.set_title("LeRobot 格式两代对照（均为本机实盘数据）", fontsize=13.5, pad=8)
    fig.savefig(ctx["out"] / "fig5-v2-vs-v3.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig5-v2-vs-v3.png")


# ---------------------------------------------------------------------------
# fig6：HDF5 → LeRobot 转换链路
# ---------------------------------------------------------------------------

def fig6_hdf5_to_lerobot(ctx, plt) -> None:
    g1: Path = ctx["g1"]
    info = json.loads((g1 / "meta/info.json").read_text())
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    act_dim = int(info["features"]["action"]["shape"][0])
    frames = int(info["total_frames"])
    fps = info["fps"]
    vid = info["features"]["observation.images.ego_view"]["shape"]

    inner = f'''
  H [label="HDF5（Isaac Lab SDG 产物）\\n1 集 / {frames} 帧 @20Hz\\nobs 14 键含 robot_pov_cam 160×256\\nactions ({frames},32) + SDG 元数据",
     fillcolor="{CLR["gray_fill"]}", color="{CLR["gray"]}"];
  C [label="gr00t/convert_dataset.py\\n（Isac Lab 官方转换脚本）",
     fillcolor="{CLR["purple_fill"]}", color="{CLR["purple"]}"];
  L [label="LeRobot v2.0\\nobservation.state {state_dim} 维\\naction {act_dim} 维\\nobservation.images.ego_view 视频 {frames} 帧 @{fps:.0f}fps\\nmeta: info / episodes / tasks / modality",
     fillcolor="{CLR["blue_fill"]}", color="{CLR["blue"]}"];
  G [label="GR00T / 下游训练\\n（modality.json 描述 state/action 分段语义）",
     fillcolor="{CLR["green_fill"]}", color="{CLR["green"]}"];
  H -> C [label=" --task 自动解析 / 逐条转换"];
  C -> L [label=" state 49 = 双手位姿7+7 + 手关节7+7 + 物体/目标/夹具 7×3\\naction 32 = 上述手部 28 + 底盘速度 3 + 高度 1"];
  L -> G [label=" 直接加载训练"];
'''
    render_dot(inner, ctx["out"] / "fig6-hdf5-to-lerobot.png",
               "HDF5（Isaac Lab）→ LeRobot：以 EGO 篇的 G1 转换为例",
               rankdir="LR", nodesep="0.5", ranksep="1.0", fontsize="11.5")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

FIGS = {
    "fig1": ("fig1-files.png", fig1_files),
    "fig2": ("fig2-reading.png", fig2_reading),
    "fig3": ("fig3-pusht-frames.png", fig3_frames),
    "fig4": ("fig4-episode-curves.png", fig4_episode_curves),
    "fig5": ("fig5-v2-vs-v3.png", fig5_v2_vs_v3),
    "fig6": ("fig6-hdf5-to-lerobot.png", fig6_hdf5_to_lerobot),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--pusht", type=Path, default=DEFAULT_PUSHT)
    ap.add_argument("--g1", type=Path, default=DEFAULT_G1)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS), default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = {"out": args.out, "pusht": args.pusht, "g1": args.g1}

    plt = setup_matplotlib()
    keys = args.only or list(FIGS)
    for k in keys:
        FIGS[k][1](ctx, plt)
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
