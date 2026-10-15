#!/usr/bin/env python3
"""第六篇（EGO 式数据工作流）配图生成。

生成 docs/images/isaac-sim-06/ 下 8 张图：

  fig1-ego-map.png          真机 EGO ↔ Isaac 对应物
  fig2-two-chains.png       仿真链 / 真机链 总览
  fig3-navigation.png       SDG 导航轨迹质检图（复制自 datasets/ego_plots/）
  fig4-frames-compare.png   仿真 G1 第一人称帧 vs EgoSuite 真机帧
  fig5-hand-overlay.png     真机手部骨架投影（复用 egosuite 管线重渲染）
  fig6-wrist-trajectory.png 双手腕轨迹（复用 egosuite 管线）
  fig7-grip-signal.png      捏合开度与语义分段（复用 egosuite 管线）
  fig8-pseudo-actions.png   伪动作拆解（复用 egosuite 管线）

用法
----
    python isaac_sim_06_figures.py [--only fig1 fig5] \
        [--egosuite-data <EgoSuite 抽取目录>] [--sim-video <G1 ego_view mp4>] \
        [--nav-plot <导航质检图>] [--out <输出目录>]

依赖：matplotlib + numpy + av（抽帧）+ graphviz dot（系统命令）；
fig5~fig8 复用 egosuite 处理数据（默认指向本机实测产物路径）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from string import Template

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                      # IsaacDataCollection/
OUT_DIR = ROOT / "docs" / "images" / "isaac-sim-06"
DEFAULT_EGOSUITE = Path("/media/dillon/0D02109D0D02109D/egosuite/processed/0e589be9")
DEFAULT_SIM_VIDEO = (ROOT / "datasets" / "lerobot_g1_locomanip" / "videos" /
                     "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4")
DEFAULT_NAV_PLOT = ROOT / "datasets" / "ego_plots" / "demo_0.png"

FONT = "Noto Sans CJK SC"
CLR = {
    "font": FONT, "dpi": 150, "dark": "#1F2933",
    "blue": "#3B7DD8", "blue_fill": "#E8F1FC",
    "orange": "#E8833A", "orange_fill": "#FDF0E3",
    "green": "#3FA66B", "green_fill": "#E6F5EC",
    "purple": "#8E64C8", "purple_fill": "#F1EBFA",
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


def count_frames(path: Path) -> int:
    import av
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


# ---------------------------------------------------------------------------
# fig1：真机 EGO ↔ Isaac 对应物
# ---------------------------------------------------------------------------

def fig1_ego_map(ctx, plt) -> None:
    inner = f'''
  subgraph cluster_real {{
    label="真机 EGO（EgoSuite-Open100K 实测：头戴式采集）";
    style="rounded,filled"; fillcolor="{CLR["gray_fill"]}"; color="{CLR["gray"]}";
    fontsize=13; fontcolor="{CLR["gray"]}"; margin=14;
    R1 [label="① 头戴双目相机\\n1920×1456 @30fps（+深度设备）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R2 [label="② 设备轨迹（SLAM 输出）\n/pose/headcam 逐帧位姿",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R3 [label="③ 双手 21 关节姿态\\npos + quat（世界系真值）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R4 [label="④ 事件级语义分段 + 质量标注\\n（4 段子任务 / bad frame 标记）",
        fillcolor="{CLR["orange_fill"]}", color="{CLR["orange"]}"];
    R1 -> R2 -> R3 -> R4 [style=invis];
  }}
  subgraph cluster_sim {{
    label="仿真 EGO（Isaac Lab · G1 locomanipulation 实测）";
    style="rounded,filled"; fillcolor="{CLR["blue_fill"]}"; color="{CLR["blue"]}";
    fontsize=13; fontcolor="{CLR["blue"]}"; margin=14;
    S1 [label="① robot_pov_cam 头戴第一人称\n256×160 @20fps（Mimic 帧 640×480）",
        fillcolor="white", color="{CLR["blue"]}"];
    S2 [label="② 仿真真值位姿\\nrobot_root / joint / links（免 SLAM）",
        fillcolor="white", color="{CLR["blue"]}"];
    S3 [label="③ 双腕 EEF + 手部关节 + 物体位姿\\n（43 关节 G1，全量 states）",
        fillcolor="white", color="{CLR["blue"]}"];
    S4 [label="④ SDG 元数据：导航路径 / 目标位姿 / 任务\\nbase_path、goal、object_pose…",
        fillcolor="white", color="{CLR["blue"]}"];
    S1 -> S2 -> S3 -> S4 [style=invis];
  }}
  R1 -> S1 [style=dashed, label=" 对应：头戴第一人称视角", fontcolor="{CLR["red"]}", color="{CLR["red"]}"];
  R2 -> S2 [style=dashed, label=" 对应：轨迹真值（仿真免估计）", fontcolor="{CLR["red"]}", color="{CLR["red"]}"];
  R3 -> S3 [style=dashed, label=" 对应：手/腕真值", fontcolor="{CLR["red"]}", color="{CLR["red"]}"];
  R4 -> S4 [style=dashed, label=" 对应：任务结构化信息", fontcolor="{CLR["red"]}", color="{CLR["red"]}"];
'''
    render_dot(inner, ctx["out"] / "fig1-ego-map.png",
               "EGO 数据：真机采集 ↔ 仿真生成（一一对应）",
               rankdir="TB", nodesep="0.6", ranksep="0.55", fontsize="12.5")


# ---------------------------------------------------------------------------
# fig2：两条链路总览
# ---------------------------------------------------------------------------

def fig2_two_chains(ctx, plt) -> None:
    inner = f'''
  subgraph cluster_sim {{
    label="仿真链（Isaac Lab · 本机 RTX 4070 实测）";
    style="rounded,filled"; fillcolor="{CLR["blue_fill"]}"; color="{CLR["blue"]}";
    fontsize=13; fontcolor="{CLR["blue"]}"; margin=14;
    A1 [label="官方预标注\\n5 MB · 少量人类演示", fillcolor="white", color="{CLR["blue"]}"];
    A2 [label="Mimic 生成\\n20 条 / 4,790 帧 / 1.5 GB\\n（含 robot_pov_cam 640×480）",
        fillcolor="white", color="{CLR["blue"]}"];
    A3 [label="SDG 生成：移动+操作\\n1 集 / 873 帧 / 39 MB\\nrobot_pov_cam 160×256",
        fillcolor="white", color="{CLR["blue"]}"];
    A4 [label="导出与应用\\nMP4 视频 + 导航质检图", fillcolor="white", color="{CLR["blue"]}"];
    A5 [label="LeRobot v2.0\\n873 帧 @20fps · state 49 / action 32",
        fillcolor="white", color="{CLR["blue"]}"];
    A1 -> A2 [label="种子扩充"];
    A2 -> A3 [label="作为 SDG 种子"];
    A3 -> A4 [label="hdf5_to_mp4 / 轨迹图"];
    A3 -> A5 [label="gr00t/convert_dataset.py"];
  }}
  subgraph cluster_real {{
    label="真机链（EgoSuite-Open100K · HF Bucket 实测）";
    style="rounded,filled"; fillcolor="{CLR["orange_fill"]}"; color="{CLR["orange"]}";
    fontsize=13; fontcolor="{CLR["orange"]}"; margin=14;
    B1 [label="EgoSuite-Open100K\\nHF Bucket 49 TB", fillcolor="white", color="{CLR["orange"]}"];
    B2 [label="下载示范 3 集\\n61 MB · MCAP（14 频道）", fillcolor="white", color="{CLR["orange"]}"];
    B3 [label="解析与抽取\\nMP4 + poses.npz + meta.json", fillcolor="white", color="{CLR["orange"]}"];
    B4 [label="验证与重定向示范\\n骨架投影对齐 / 轨迹 / 伪动作 (675×7)",
        fillcolor="white", color="{CLR["orange"]}"];
    B1 -> B2 [label="hf buckets"];
    B2 -> B3 [label="egosuite_mcap_extract.py"];
    B3 -> B4 [label="egosuite_demo_figures.py"];
  }}
  M [label="训练侧消费\\nHDF5（Isaac Lab）/ LeRobot（下游）", fillcolor="{CLR["green_fill"]}",
     color="{CLR["green"]}"];
  A5 -> M [label="  ", color="{CLR["green"]}"];
  B4 -> M [label="  ", color="{CLR["green"]}"];
'''
    render_dot(inner, ctx["out"] / "fig2-two-chains.png",
               "EGO 式数据工作流：仿真链 + 真机链（全部实测）",
               rankdir="LR", nodesep="0.5", ranksep="0.9", fontsize="12")


# ---------------------------------------------------------------------------
# fig3：导航轨迹质检图（复制）
# ---------------------------------------------------------------------------

def fig3_navigation(ctx, plt) -> None:
    src: Path = ctx["nav_plot"]
    if not src.exists():
        print(f"[skip] fig3（未找到 {src}）")
        return
    shutil.copy2(src, ctx["out"] / "fig3-navigation.png")
    print("[ok] fig3-navigation.png")


# ---------------------------------------------------------------------------
# fig4：仿真 vs 真机 第一人称帧对比
# ---------------------------------------------------------------------------

def fig4_frames_compare(ctx, plt) -> None:
    import egosuite_common as ecm

    sim_video: Path = ctx["sim_video"]
    real_video: Path = Path(ctx["egosuite_data"]) / "video_left.mp4"
    if not (sim_video.exists() and real_video.exists()):
        print("[skip] fig4（缺少视频输入）")
        return

    def pick(path: Path, fracs):
        n = count_frames(path)
        idx = [round(f * (n - 1)) for f in fracs]
        return n, ecm.decode_selected_frames(path, idx), idx

    n_sim, sim_frames, sim_idx = pick(sim_video, (0.18, 0.62))
    n_real, real_frames, real_idx = pick(real_video, (0.12, 0.55))

    plt.rcParams["figure.facecolor"] = "white"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=140)
    for col, (i, img) in enumerate(zip(sim_idx, (sim_frames[k] for k in sorted(sim_frames)))):
        h, w = img.shape[:2]
        axes[0, col].imshow(img[:, :, ::-1])
        axes[0, col].set_title(f"仿真 · G1 robot_pov_cam {w}×{h} @20fps\n帧 {i} / {n_sim}（t≈{i/20:.1f} s）",
                               fontsize=10, color=CLR["blue"])
        axes[0, col].axis("off")
    for col, (i, img) in enumerate(zip(real_idx, (real_frames[k] for k in sorted(real_frames)))):
        h, w = img.shape[:2]
        axes[1, col].imshow(img[:, :, ::-1])
        axes[1, col].set_title(f"真机 · EgoSuite 头戴相机 {w}×{h} @30fps\n帧 {i} / {n_real}（t≈{i/30:.1f} s）",
                               fontsize=10, color=CLR["orange"])
        axes[1, col].axis("off")
    fig.suptitle("第一人称画面：仿真（上）vs 真机（下）——同一条 EGO 视角", fontsize=13)
    fig.tight_layout()
    fig.savefig(ctx["out"] / "fig4-frames-compare.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig4-frames-compare.png")


# ---------------------------------------------------------------------------
# fig5~fig8：复用 egosuite 管线重渲染
# ---------------------------------------------------------------------------

def _load_egosuite(ctx):
    import egosuite_common as ecm
    import egosuite_demo_figures as edf

    data = Path(ctx["egosuite_data"])
    poses = np.load(data / "poses.npz")
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    return data, poses, meta, ecm, edf


def fig5_hand_overlay(ctx, plt) -> None:
    data, poses, meta, ecm, edf = _load_egosuite(ctx)
    with tempfile.TemporaryDirectory() as tmp:
        edf.fig_overlay(plt, Path(tmp), data, poses, meta, 6)
        shutil.copy2(Path(tmp) / "fig2-hand-overlay.png", ctx["out"] / "fig5-hand-overlay.png")
    print("[ok] fig5-hand-overlay.png")


def fig6_wrist_trajectory(ctx, plt) -> None:
    data, poses, meta, ecm, edf = _load_egosuite(ctx)
    t = poses["hand_right_t"]
    ap_r = ecm.aperture_cm(poses["hand_right_pos"])
    min_ap_time = float(t[int(np.argmin(ap_r))])
    with tempfile.TemporaryDirectory() as tmp:
        edf.fig_trajectory(plt, Path(tmp), poses, meta, min_ap_time)
        shutil.copy2(Path(tmp) / "fig3-wrist-trajectory.png",
                     ctx["out"] / "fig6-wrist-trajectory.png")
    print("[ok] fig6-wrist-trajectory.png")


def fig7_grip_signal(ctx, plt) -> None:
    data, poses, meta, ecm, edf = _load_egosuite(ctx)
    ap_r = ecm.aperture_cm(poses["hand_right_pos"])
    ap_l = ecm.aperture_cm(poses["hand_left_pos"])
    with tempfile.TemporaryDirectory() as tmp:
        edf.fig_grip(plt, Path(tmp), poses, meta, ap_l, ap_r)
        shutil.copy2(Path(tmp) / "fig4-grip-signal.png", ctx["out"] / "fig7-grip-signal.png")
    print("[ok] fig7-grip-signal.png")


def fig8_pseudo_actions(ctx, plt) -> None:
    data, poses, meta, ecm, edf = _load_egosuite(ctx)
    t = poses["hand_right_t"]
    ap_r = ecm.aperture_cm(poses["hand_right_pos"])
    actions, dp, rv, grip, lo, hi = edf.build_pseudo_actions(
        poses["hand_right_pos"], poses["hand_right_quat"], ap_r)
    with tempfile.TemporaryDirectory() as tmp:
        edf.fig_actions(plt, Path(tmp), t, actions, dp, rv)
        shutil.copy2(Path(tmp) / "fig5-pseudo-actions.png",
                     ctx["out"] / "fig8-pseudo-actions.png")
    print("[ok] fig8-pseudo-actions.png")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

FIGS = {
    "fig1": ("fig1-ego-map.png", fig1_ego_map),
    "fig2": ("fig2-two-chains.png", fig2_two_chains),
    "fig3": ("fig3-navigation.png", fig3_navigation),
    "fig4": ("fig4-frames-compare.png", fig4_frames_compare),
    "fig5": ("fig5-hand-overlay.png", fig5_hand_overlay),
    "fig6": ("fig6-wrist-trajectory.png", fig6_wrist_trajectory),
    "fig7": ("fig7-grip-signal.png", fig7_grip_signal),
    "fig8": ("fig8-pseudo-actions.png", fig8_pseudo_actions),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--egosuite-data", type=Path, default=DEFAULT_EGOSUITE)
    ap.add_argument("--sim-video", type=Path, default=DEFAULT_SIM_VIDEO)
    ap.add_argument("--nav-plot", type=Path, default=DEFAULT_NAV_PLOT)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS), default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = {"out": args.out, "egosuite_data": args.egosuite_data,
           "sim_video": args.sim_video, "nav_plot": args.nav_plot}

    plt = setup_matplotlib()
    keys = args.only or list(FIGS)
    for k in keys:
        FIGS[k][1](ctx, plt)
    print(f"完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
