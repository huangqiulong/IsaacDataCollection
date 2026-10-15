#!/usr/bin/env python3
"""EgoSuite-Open100K 处理演示：可视化 + "伪动作"构建。

输入为 egosuite_mcap_extract.py 的输出目录（poses.npz / meta.json / video_left.mp4），
产出 5 张图（默认写入仓库 docs/images/egosuite-demo/）与左右手伪动作 npz：

  fig1-pipeline.png          数据链路总览（MCAP → 抽取 → 可视化/动作）
  fig2-hand-overlay.png      手部骨架投影叠加（世界系姿态 × 相机标定验证）
  fig3-wrist-trajectory.png  双手腕轨迹（3D + 俯视）
  fig4-grip-signal.png       捏合开度与腕速（含语义分段标注）
  fig5-pseudo-actions.png    机器人风格伪动作序列（Δpos / Δrot / grip）
  pseudo_actions_{left,right}.npz

用法
----
    python egosuite_demo_figures.py --data <抽取输出目录> [--figs <图片目录>] [--overlay-frames 6]

依赖：numpy、matplotlib、opencv-python（画骨架）、av（抽帧）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egosuite_common import (
    HAND_EDGES,
    aperture_cm,
    decode_selected_frames,
    project_world_points,
    quat2R,
    quat_rel_angle_deg,
    setup_matplotlib,
)

LEFT_CLR = "#2ecc71"   # 左手（与骨架叠加中的绿色一致）
RIGHT_CLR = "#e67e22"  # 右手（橙色）
GREY = "#607d8b"
DEFAULT_FIGS_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "egosuite-demo"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def nearest_idx(ts: np.ndarray, t: float) -> int:
    """单调时间序列中最接近 t 的下标。"""
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i - 1 if (t - ts[i - 1]) <= (ts[i] - t) else i


def rotvec_from_rel(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 轴角向量（SE(3) 对数映射）。"""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(cos_a)
    if a < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (a / (2.0 * np.sin(a)))


def build_pseudo_actions(pos: np.ndarray, quat: np.ndarray, aperture: np.ndarray):
    """由腕部轨迹构建"机器人风格"伪动作：Δpos(3) + Δrotvec(3) + grip(1)。

    pos/quat: (T,21,3)/(T,21,4)，取关节 0（手腕）。返回 (T-1,7) 与辅助量。
    """
    dp = np.diff(pos[:, 0, :], axis=0)                       # (T-1,3) m
    rv = np.zeros((len(pos) - 1, 3))
    for i in range(len(pos) - 1):
        R1 = quat2R(*quat[i, 0])
        R2 = quat2R(*quat[i + 1, 0])
        rv[i] = rotvec_from_rel(R1.T @ R2)                   # 相对旋转（当前系）
    lo, hi = np.percentile(aperture, [5, 95])                # 开合归一化区间
    grip = np.clip((aperture - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    actions = np.concatenate([dp, rv, grip[:-1, None]], axis=1)
    return actions, dp, rv, grip, lo, hi


# ---------------------------------------------------------------------------
# 图 1：数据链路
# ---------------------------------------------------------------------------

def fig_pipeline(plt, out_dir: Path, meta: dict, npz_ok: bool, video_ok: bool) -> None:
    task = (meta.get("session", {}) or {}).get("task_info", {})
    task_name = task.get("task_name", "?")
    seg_n = len(meta.get("semantic_segments", []))

    fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=140)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def box(x, y, w, h, text, fc, ec, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, label=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=18, lw=1.6, color=GREY))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, label, ha="center",
                    va="bottom", fontsize=8.5, color=GREY)

    ya = [0.70, 0.47, 0.24, 0.01]
    for y, t in zip(ya, [
        "双目头戴相机视频\nH.264 · 1920×1456 · 30 fps",
        "手部姿态 · 21 关键点/手\npos + quat（世界系）",
        "相机内参 / 外参\nK · world→cam",
        "语义分段 + 会话元数据\n任务/场景/操作员",
    ]):
        box(0.015, y, 0.255, 0.19, t, "#f5f7fa", "#90a4ae")

    yb = [0.62, 0.37, 0.12]
    for y, t in zip(yb, [
        "video_left.mp4\n(video_right.mp4)",
        "poses.npz\n(时间/双手/头戴相机)",
        "meta.json\n(元数据/分段/标定)",
    ]):
        box(0.375, y, 0.21, 0.19, t, "#e8f5e9", "#81c784")

    yc = [0.70, 0.47, 0.24, 0.01]
    for y, t in zip(yc, [
        "骨架投影叠加（图 2）\n验证「视频 → 手部位姿」",
        "腕部轨迹（图 3）\n3D + 俯视",
        "捏合开度 / 腕速（图 4）\n语义分段对齐",
        "伪动作 Δpose+grip（图 5）\nT×7 → 机器人接口风格",
    ]):
        box(0.70, y, 0.285, 0.19, t, "#fff3e0", "#ffb74d")

    arrow(0.275, 0.5, 0.37, 0.5, "egosuite_mcap_extract.py")
    arrow(0.59, 0.5, 0.695, 0.5, "egosuite_demo_figures.py")

    ax.set_title(f"EgoSuite-Open100K 处理链路 · 「{task_name}」（{seg_n} 段语义标注）",
                 fontsize=13, pad=10)
    fig.savefig(out_dir / "fig1-pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig1-pipeline.png")


# ---------------------------------------------------------------------------
# 图 2：骨架投影叠加
# ---------------------------------------------------------------------------

def fig_overlay(plt, out_dir: Path, data_dir: Path, poses, meta, n_frames: int) -> None:
    import cv2

    video = data_dir / "video_left.mp4"
    if not video.exists():
        print("[skip] fig2（未找到 video_left.mp4）")
        return

    K = np.array(meta["intrinsic"]["left"]["K"], dtype=float).reshape(3, 3)
    hc_t, hc_pos, hc_quat = poses["headcam_t"], poses["headcam_pos"], poses["headcam_quat"]
    t_video = poses["t_video_left"]
    n = len(t_video)
    indices = np.linspace(0, n - 1, n_frames).astype(int)
    frames = decode_selected_frames(video, indices)

    colors = {"left": (110, 220, 80), "right": (60, 150, 255)}  # BGR
    ncols = min(3, n_frames)
    nrows = int(np.ceil(n_frames / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), dpi=130)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")
    for k, idx in enumerate(indices):
        ax = axes[k]
        if idx not in frames:
            continue
        img = frames[idx].copy()
        t_i = t_video[idx]
        hi = nearest_idx(hc_t, t_i)
        cam_pos, cam_quat = hc_pos[hi], hc_quat[hi]
        for side in ("left", "right"):
            pos_i = poses[f"hand_{side}_pos"]
            if idx >= len(pos_i):
                continue
            uv, z = project_world_points(pos_i[idx], cam_pos, cam_quat, K)
            for a, b in HAND_EDGES:
                if z[a] > 0 and z[b] > 0:
                    pa, pb = uv[a].astype(int), uv[b].astype(int)
                    if 0 <= pa[0] < img.shape[1] and 0 <= pa[1] < img.shape[0]:
                        cv2.line(img, tuple(pa), tuple(pb), colors[side], 4, cv2.LINE_AA)
            for j in range(len(uv)):
                if z[j] > 0:
                    u, v = uv[j].astype(int)
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (u, v), 6 if j == 0 else 4, colors[side], -1, cv2.LINE_AA)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t = {t_i:5.1f} s（帧 {idx}）", fontsize=10)

    fig.suptitle("手部骨架投影验证：世界系 21 关键点 × 相机标定 → 像素（绿=左手，橙=右手）",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2-hand-overlay.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig2-hand-overlay.png")


# ---------------------------------------------------------------------------
# 图 3：腕部轨迹
# ---------------------------------------------------------------------------

def fig_trajectory(plt, out_dir: Path, poses, meta, min_ap_time: float) -> None:
    t = poses["hand_right_t"]
    wr = poses["hand_right_pos"][:, 0, :]
    wl = poses["hand_left_pos"][:, 0, :]
    cam0 = poses["headcam_pos"][0]

    fig = plt.figure(figsize=(12.5, 5.2), dpi=140)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(*wl.T, color=LEFT_CLR, lw=2, label="左手腕")
    ax3.plot(*wr.T, color=RIGHT_CLR, lw=2, label="右手腕")
    ax3.scatter(*wl[0], color=LEFT_CLR, marker="o", s=60, edgecolor="k", zorder=5)
    ax3.scatter(*wr[0], color=RIGHT_CLR, marker="o", s=60, edgecolor="k", zorder=5)
    ax3.scatter(*wl[-1], color=LEFT_CLR, marker="s", s=60, edgecolor="k", zorder=5)
    ax3.scatter(*wr[-1], color=RIGHT_CLR, marker="s", s=60, edgecolor="k", zorder=5)
    ax3.scatter(*cam0, color=GREY, marker="*", s=140, label="相机(起点)")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title("世界系 3D 轨迹（○=起点，□=终点）")
    ax3.legend(loc="upper left", fontsize=8.5)
    rng = np.ptp(np.vstack([wr, wl, cam0[None]]), axis=0)
    ax3.set_box_aspect(rng / max(rng.max(), 1e-6))
    ax3.view_init(elev=22, azim=-62)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(wl[:, 0], wl[:, 1], color=LEFT_CLR, lw=2, label="左手腕")
    ax2.plot(wr[:, 0], wr[:, 1], color=RIGHT_CLR, lw=2, label="右手腕")
    ax2.scatter(*wl[0, :2], color=LEFT_CLR, marker="o", s=55, edgecolor="k", zorder=5)
    ax2.scatter(*wr[0, :2], color=RIGHT_CLR, marker="o", s=55, edgecolor="k", zorder=5)
    ax2.scatter(*cam0[:2], color=GREY, marker="*", s=160, label="相机(起点)")
    mi = nearest_idx(t, min_ap_time)
    ax2.annotate("捏合最紧时刻", xy=wr[mi, :2], xytext=(wr[mi, 0] + 0.05, wr[mi, 1] + 0.05),
                 arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=9)
    ax2.scatter(*wr[mi, :2], color=RIGHT_CLR, marker="D", s=55, edgecolor="k", zorder=6)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("俯视（X-Y）")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5); ax2.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"双手腕轨迹：右手移动 {np.linalg.norm(wr[-1]-wr[0])*100:.1f} cm · "
                 f"左手移动 {np.linalg.norm(wl[-1]-wl[0])*100:.1f} cm", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3-wrist-trajectory.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig3-wrist-trajectory.png")


# ---------------------------------------------------------------------------
# 图 4：捏合开度 + 腕速
# ---------------------------------------------------------------------------

def fig_grip(plt, out_dir: Path, poses, meta, ap_l, ap_r) -> None:
    t = poses["hand_right_t"]
    wr = poses["hand_right_pos"][:, 0, :]
    dt = np.gradient(t)
    speed = np.linalg.norm(np.gradient(wr, axis=0), axis=-1) / np.maximum(dt, 1e-6)

    events = []
    for seg in meta.get("semantic_segments", []):
        s = seg.get("segment")
        items = s if isinstance(s, list) else ([s] if s else [])
        for it in items:
            if "end_time" in it and "subtask_description" in it:
                events.append((float(it["end_time"]), it["subtask_description"]))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.4), dpi=140, sharex=True)
    ax = axes[0]
    ax.plot(t, ap_r, color=RIGHT_CLR, lw=1.8, label="右手（拇指-食指）")
    ax.plot(t, ap_l, color=LEFT_CLR, lw=1.4, alpha=0.85, label="左手")
    ax.set_ylabel("捏合开度 (cm)")
    ax.set_title("捏合开度：拇指尖-食指尖距离（= 夹爪开度代理量）")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(t, speed * 100, color=GREY, lw=1.6)
    ax.set_ylabel("右腕速度 (cm/s)")
    ax.set_xlabel("时间 (s)")
    ax.set_title("右腕运动速度")
    ax.grid(alpha=0.3)

    for ax_ in axes:
        for end_t, desc in events:
            if 0 <= end_t <= t[-1]:
                ax_.axvline(end_t, color="#8e24aa", ls="--", lw=1.2, alpha=0.8)
        for end_t, desc in events:
            if 0 <= end_t <= t[-1]:
                ax_.text(end_t, ax_.get_ylim()[1] * 0.96, f" {desc[:22]}",
                         rotation=0, fontsize=8, color="#8e24aa", va="top")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4-grip-signal.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig4-grip-signal.png")


# ---------------------------------------------------------------------------
# 图 5：伪动作
# ---------------------------------------------------------------------------

def fig_actions(plt, out_dir: Path, t, actions, dp, rv) -> None:
    tt = t[:-1]
    rot_deg = np.degrees(np.linalg.norm(rv, axis=1))
    grip = actions[:, 6]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 7.2), dpi=140, sharex=True)
    ax = axes[0]
    for i, c in enumerate(["X", "Y", "Z"]):
        ax.plot(tt, dp[:, i] * 100, lw=1.5, label=f"Δ{c}")
    ax.set_ylabel("Δ位置 (cm/步)")
    ax.set_title("伪动作拆解：手腕相对位姿增量 + 夹爪开合（右手）")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=3)

    ax = axes[1]
    ax.plot(tt, rot_deg, color="#8e24aa", lw=1.5)
    ax.set_ylabel("Δ旋转 (°/步)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.fill_between(tt, 0, grip, color=RIGHT_CLR, alpha=0.45)
    ax.plot(tt, grip, color=RIGHT_CLR, lw=1.4)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_ylabel("grip（0=闭合）")
    ax.set_xlabel("时间 (s)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5-pseudo-actions.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig5-pseudo-actions.png")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="egosuite_mcap_extract.py 输出目录")
    ap.add_argument("--figs", type=Path, default=DEFAULT_FIGS_DIR, help="图片输出目录")
    ap.add_argument("--overlay-frames", type=int, default=6, help="骨架叠加帧数")
    args = ap.parse_args()

    args.figs.mkdir(parents=True, exist_ok=True)
    poses = np.load(args.data / "poses.npz")
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))

    pos_l, pos_r = poses["hand_left_pos"], poses["hand_right_pos"]
    quat_l, quat_r = poses["hand_left_quat"], poses["hand_right_quat"]
    t = poses["hand_right_t"]

    ap_r = aperture_cm(pos_r)
    ap_l = aperture_cm(pos_l)
    min_ap_idx = int(np.argmin(ap_r))
    min_ap_time = float(t[min_ap_idx])

    # ---- 伪动作（双手） ----
    for side, pos, quat, ap_ in (("right", pos_r, quat_r, ap_r), ("left", pos_l, quat_l, ap_l)):
        actions, dp, rv, grip, lo, hi = build_pseudo_actions(pos, quat, ap_)
        np.savez_compressed(
            args.data / f"pseudo_actions_{side}.npz",
            t=t[:-1], actions=actions, pos_delta_m=dp, rotvec_rad=rv,
            grip=grip, aperture_cm=ap_, closed_threshold_cm=np.float32(lo),
            open_threshold_cm=np.float32(hi),
        )
        print(f"[npz] pseudo_actions_{side}.npz: actions {actions.shape}（Δpos(3)+Δrotvec(3)+grip(1)）")

    print(f"[统计] 时长 {t[-1]:.1f} s / {len(t)} 帧；右手捏合开度 "
          f"{ap_r.min():.1f} ~ {ap_r.max():.1f} cm（最紧 @ {min_ap_time:.1f} s）")

    plt = setup_matplotlib()
    fig_pipeline(plt, args.figs, meta, npz_ok=True, video_ok=(args.data / "video_left.mp4").exists())
    fig_overlay(plt, args.figs, args.data, poses, meta, args.overlay_frames)
    fig_trajectory(plt, args.figs, poses, meta, min_ap_time)
    fig_grip(plt, args.figs, poses, meta, ap_l, ap_r)

    actions, dp, rv, grip, lo, hi = build_pseudo_actions(pos_r, quat_r, ap_r)
    fig_actions(plt, args.figs, t, actions, dp, rv)

    print(f"完成，图片输出目录: {args.figs}")


if __name__ == "__main__":
    main()
