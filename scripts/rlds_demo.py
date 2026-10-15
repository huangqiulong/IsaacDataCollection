#!/usr/bin/env python3
"""RLDS 数据集读取与可视化示范（配套第九篇）。

功能
----
1. 用 TFDS 打开一个 RLDS 数据集（社区目录自动识别；失败时回退 builder_from_directory）；
2. 打印“数据集 → Episode → Step”的结构摘要（字段、形状、指令、episode/step 数）；
3. 抽取若干 episode：导出双相机采样帧拼图、动作曲线图、统计 JSON。

用法
----
    conda activate rlds          # 含 tensorflow-cpu + tensorflow_datasets
    python rlds_demo.py imperial_wrist_dataset --version 1.0.0 \
        --data-dir /path/to/rlds_tfds --episodes 3 --out ./rlds_demo_out

依赖：tensorflow-cpu, tensorflow_datasets, numpy, matplotlib。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    prefer = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
              "Droid Sans Fallback", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [n for n in prefer if n in available]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.facecolor"] = "white"
    return plt


def load_dataset(name: str, version: str, data_dir: str):
    """返回 (builder, episode_dataset)。优先社区目录，其次本地目录直读。"""
    import tensorflow_datasets as tfds

    try:
        b = tfds.builder(name, data_dir=data_dir)
        print(f"[加载] tfds 识别成功：{name}")
    except Exception as e:
        print(f"[加载] 社区目录失败（{type(e).__name__}），回退 builder_from_directory")
        b = tfds.builder_from_directory(str(Path(data_dir) / name / version))
    ds = b.as_dataset(split="all")
    return b, ds


def describe(builder) -> dict:
    """打印并返回结构摘要。"""
    print("\n===== 数据集结构 =====")
    info = builder.info
    feats = info.features
    print("[顶层字段]", list(feats.keys()))
    steps_feat = feats["steps"]
    inner = getattr(steps_feat, "features", None)
    if inner is not None:
        for k, v in inner.items():
            print(f"  step.{k}: {v.__class__.__name__}")
    else:
        print("  steps:", steps_feat)
    if "episode_metadata" in feats:
        em = feats["episode_metadata"]
        print("  episode_metadata:", list(em.keys()) if hasattr(em, "keys") else em)
    print("[splits]", {k: v.num_examples for k, v in info.splits.items()})
    desc = getattr(info, "description", "")
    print("[description]", desc[:160].replace("\n", " "))
    return {}


def decode_step(step: dict) -> dict:
    """把一个 step 的 tf.Tensor 转成 numpy / str 的轻量 dict。"""
    def conv(x):
        if hasattr(x, "numpy"):
            v = x.numpy()
            return v.decode() if isinstance(v, bytes) else v
        return x

    out = {}
    for k, v in step.items():
        if isinstance(v, dict):                    # observation
            out[k] = {kk: conv(vv) for kk, vv in v.items()}
        else:
            out[k] = conv(v)
    return out


def scan_episode(ep, index: int, out_dir: Path, plt, sample: int = 4) -> dict:
    """统计一个 episode，导出图像拼图与动作曲线。"""
    steps = list(ep["steps"])
    n = len(steps)
    first = decode_step(steps[0])
    lang = first.get("language_instruction", "")

    # 收集动作序列与图像采样
    import numpy as np
    actions = np.stack([decode_step(s)["action"] for s in steps])
    reward = np.array([float(np.asarray(decode_step(s).get("reward", 0.0))) for s in steps])
    is_first = [bool(decode_step(s)["is_first"]) for s in steps]
    is_last = [bool(decode_step(s)["is_last"]) for s in steps]

    img_keys = [k for k in first.get("observation", {}) if hasattr(first["observation"][k], "shape")
                and np.asarray(first["observation"][k]).ndim == 3]
    frames = [int(round(i * (n - 1) / (sample - 1))) for i in range(sample)]

    # ---- 图像拼图：行=相机，列=时刻 ----
    if img_keys:
        fig, axes = plt.subplots(len(img_keys), sample,
                                 figsize=(2.6 * sample, 2.7 * len(img_keys)),
                                 squeeze=False)
        for r, key in enumerate(img_keys):
            for c, f in enumerate(frames):
                img = decode_step(steps[f])["observation"][key]
                axes[r][c].imshow(img)
                axes[r][c].set_axis_off()
                axes[r][c].set_title(f"{key}  t={f}", fontsize=9)
        fig.suptitle(f"episode #{index} · {lang}\n相机帧采样（{n} 步）", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_dir / f"ep{index:02d}_frames.png", bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] ep{index:02d}_frames.png（相机: {img_keys}）")

    # ---- 动作 + 时间线 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for d in range(actions.shape[1]):
        ax1.plot(actions[:, d], linewidth=1.0, label=f"a[{d}]")
    ax1.set_ylabel("action"); ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=4)
    ax1.set_title(f"episode #{index} · {lang}（{n} 步，action {actions.shape[1]} 维）", fontsize=11)
    ax2.eventplot([np.where(is_first)[0]], colors=["#3FA66B"], lineoffsets=0.6, linelengths=0.7)
    ax2.eventplot([np.where(is_last)[0]], colors=["#D64545"], lineoffsets=0.2, linelengths=0.7)
    ax2.set_yticks([0.6, 0.2]); ax2.set_yticklabels(["is_first", "is_last"], fontsize=9)
    ax2.set_xlabel("step"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{index:02d}_actions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] ep{index:02d}_actions.png")

    return {"episode": index, "steps": n, "instruction": lang,
            "action_dim": int(actions.shape[1]),
            "action_min": actions.min(axis=0).round(4).tolist(),
            "action_max": actions.max(axis=0).round(4).tolist(),
            "reward_sum": float(reward.sum()),
            "image_keys": img_keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="数据集名，如 imperial_wrist_dataset")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--data-dir", default=str(Path.home() / "tensorflow_datasets"))
    ap.add_argument("--episodes", type=int, default=3, help="抽取的 episode 数")
    ap.add_argument("--out", default=None, help="输出目录（默认 <data-dir>/../rlds_demo_out/<name>）")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.data_dir).parent / "rlds_demo_out" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    builder, ds = load_dataset(args.name, args.version, args.data_dir)
    describe(builder)

    plt = setup_matplotlib()
    reports = []
    print(f"\n===== 抽取 {args.episodes} 个 episode =====")
    for i, ep in enumerate(ds.take(args.episodes)):
        reports.append(scan_episode(ep, i, out_dir, plt))

    with (out_dir / "summary.json").open("w") as f:
        json.dump({"dataset": args.name, "version": args.version,
                   "episodes_scanned": reports}, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] summary.json -> {out_dir}")
    print("episode/step 概况:")
    for r in reports:
        print(f"  #{r['episode']}: {r['steps']} 步 | action {r['action_dim']}D | “{r['instruction']}”")


if __name__ == "__main__":
    main()
