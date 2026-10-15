#!/usr/bin/env python3
"""EgoSuite-Open100K MCAP 数据抽取。

从单个 MCAP 演示集中抽取三类产物（与 LeRobot / MCAP 数据契约对应）：
  1. 相机视频：H.264 流重组后封装为 MP4（默认导出 head_left，可加 head_right）
  2. 手部姿态：左右手 21 关键点 (T,21,3) 位置 + (T,21,4) 四元数，
     以及头部/相机位姿，保存为 poses.npz
  3. 元数据：会话信息 / 相机内参外参 / 语义分段 / 质量标注，保存为 meta.json

用法
----
    python egosuite_mcap_extract.py --mcap <file.mcap> --out <输出目录> \
        [--cams left,right] [--keep-h264]

输出
----
    <out>/video_left.mp4 / video_right.mp4
    <out>/poses.npz      （时间、手部、头部位姿，时间以首帧为 0 相对秒）
    <out>/meta.json      （会话元数据、内参、外参、语义分段等）

依赖：mcap、protobuf、av（或 cv2 兜底）。
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader

from egosuite_common import build_descriptor_pool, build_message_classes, decode_message

POSE_SINGLE = {  # 单变换位姿频道（1 个 transform / 帧）
    "/pose/head": "head",
    "/pose/headcam": "headcam",
    "/pose/right_eye_cam": "right_eye_cam",
}
POSE_HANDS = {  # 21 关键点手部频道
    "/pose/left_hand": "left",
    "/pose/right_hand": "right",
}


def h264_to_mp4(h264_path: Path, mp4_path: Path, fps: float) -> None:
    """H.264 裸流 -> MP4（解码 + libx264 转码；失败时 cv2 重编码兜底）。

    注意：PyAV 对 raw-h264 的 demux 包 dts 全为 None，无法直接流拷贝
    （会被时间戳校验拦下），因此采用"解码 + 转码"，几百帧的代价可忽略。
    """
    import av

    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    try:
        with av.open(str(h264_path), format="h264") as inp, \
                av.open(str(mp4_path), "w") as out:
            ist = inp.streams.video[0]
            ost = out.add_stream(codec, rate=max(int(round(fps)), 1))
            ost.width, ost.height = ist.width, ist.height
            ost.pix_fmt = "yuv420p"
            for frame in inp.decode(video=0):
                for pkt in ost.encode(frame):
                    out.mux(pkt)
            for pkt in ost.encode():  # flush
                out.mux(pkt)
        return
    except Exception as exc:  # noqa: BLE001 - 兜底路径
        print(f"  [warn] PyAV 转码失败（{exc}），改用 cv2 重编码")

    import cv2

    with av.open(str(h264_path), format="h264") as container:
        writer = None
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="bgr24")
            if writer is None:
                h, w = arr.shape[:2]
                writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter 打开失败（当前环境无视频编码器）")
            writer.write(arr)
        if writer is not None:
            writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mcap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cams", default="left,right", help="要导出的相机: left,right（逗号分隔）")
    ap.add_argument("--keep-h264", action="store_true", help="保留中间 .h264 文件")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mcap, "rb") as f:
        summary = make_reader(f).get_summary()
    pool = build_descriptor_pool(summary)
    classes = build_message_classes(pool, summary)

    video = {cam: [] for cam in cams}          # (log_time_ns, bytes)
    hand = {side: {"t": [], "pos": [], "quat": []} for side in POSE_HANDS.values()}
    single = {name: {"t": [], "pos": [], "quat": []} for name in POSE_SINGLE.values()}
    segments, bad_total, bad_count = [], 0, 0
    meta_msg = None
    intrinsic, extrinsic = {}, {}

    print(f"[读取] {args.mcap.name} …")
    with open(args.mcap, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            topic = channel.topic
            m = decode_message(schema, message.data, classes)

            if topic.startswith("/sensor/camera/head_") and topic.endswith("/video"):
                cam = "left" if "head_left" in topic else "right"
                if cam in video:
                    video[cam].append((message.log_time, m.data))
            elif topic in POSE_HANDS:
                side = POSE_HANDS[topic]
                t0 = m.header.ts / 1e9
                hand[side]["t"].append(t0)
                hand[side]["pos"].append([[tr.pos.x, tr.pos.y, tr.pos.z] for tr in m.transforms])
                hand[side]["quat"].append([[tr.quat.w, tr.quat.x, tr.quat.y, tr.quat.z] for tr in m.transforms])
            elif topic in POSE_SINGLE and len(m.transforms):
                name = POSE_SINGLE[topic]
                tr = m.transforms[0]
                single[name]["t"].append(m.header.ts / 1e9)
                single[name]["pos"].append([tr.pos.x, tr.pos.y, tr.pos.z])
                single[name]["quat"].append([tr.quat.w, tr.quat.x, tr.quat.y, tr.quat.z])
            elif topic == "/annotation/semantic_segments":
                segments.append(MessageToDict(m, preserving_proto_field_name=True))
            elif topic == "/annotation/bad_frame/pose/hand":
                bad_total += 1
                bad_count += int(m.is_bad)
            elif topic == "/session/metadata" and meta_msg is None:
                meta_msg = m
            elif topic.endswith("/intrinsic") and "head_left" in topic and "left" not in intrinsic:
                intrinsic["left"] = MessageToDict(m, preserving_proto_field_name=True)
            elif topic.endswith("/intrinsic") and "head_right" in topic and "right" not in intrinsic:
                intrinsic["right"] = MessageToDict(m, preserving_proto_field_name=True)
            elif topic.endswith("/extrinsic") and "head_left" in topic and "left" not in extrinsic:
                extrinsic["left"] = MessageToDict(m, preserving_proto_field_name=True)
            elif topic.endswith("/extrinsic") and "head_right" in topic and "right" not in extrinsic:
                extrinsic["right"] = MessageToDict(m, preserving_proto_field_name=True)

    # 时间原点 = 左镜头首帧（或最早时间戳）
    t_ref = None
    for cam in cams:
        if video[cam]:
            t_ref = video[cam][0][0] / 1e9
            break
    if t_ref is None:
        t_ref = min(hand["left"]["t"] or [0.0])
    print(f"[时间] 原点(unix) = {t_ref:.3f} s")

    # ---- 1) 视频导出 ----
    stats = {}
    for cam in cams:
        if not video[cam]:
            continue
        stamps = np.array([t for t, _ in video[cam]], dtype=np.float64) / 1e9 - t_ref
        dur = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        fps = (len(stamps) - 1) / dur if dur > 0 else 30.0
        h264_tmp = out_dir / f"_tmp_{cam}.h264"
        h264_tmp.write_bytes(b"".join(d for _, d in video[cam]))
        mp4_path = out_dir / f"video_{cam}.mp4"
        h264_to_mp4(h264_tmp, mp4_path, fps)
        if not args.keep_h264:
            h264_tmp.unlink()
        else:
            h264_tmp.rename(out_dir / f"video_{cam}.h264")
        stats[cam] = {"frames": len(stamps), "fps": round(fps, 3), "duration_s": round(dur, 3),
                      "mp4_mb": round(mp4_path.stat().st_size / 1e6, 2)}
        print(f"[视频] {mp4_path.name}: {len(stamps)} 帧 @ {fps:.2f} fps, "
              f"{dur:.1f} s, {stats[cam]['mp4_mb']} MB")

    # ---- 2) 姿态 npz ----
    save = {"unix0": t_ref}
    for cam in cams:
        save[f"t_video_{cam}"] = np.array([t for t, _ in video[cam]], dtype=np.float64) / 1e9 - t_ref
    for side, d in hand.items():
        save[f"hand_{side}_t"] = np.asarray(d["t"], dtype=np.float64) - t_ref
        save[f"hand_{side}_pos"] = np.asarray(d["pos"], dtype=np.float32)
        save[f"hand_{side}_quat"] = np.asarray(d["quat"], dtype=np.float32)  # (T,21,4) wxyz
    for name, d in single.items():
        if not d["t"]:
            continue
        save[f"{name}_t"] = np.asarray(d["t"], dtype=np.float64) - t_ref
        save[f"{name}_pos"] = np.asarray(d["pos"], dtype=np.float32)
        save[f"{name}_quat"] = np.asarray(d["quat"], dtype=np.float32)      # (T,4) wxyz
    npz_path = out_dir / "poses.npz"
    np.savez_compressed(npz_path, **save)
    print(f"[姿态] {npz_path.name}: 左手 {save['hand_left_pos'].shape[0]} 帧 / "
          f"右手 {save['hand_right_pos'].shape[0]} 帧 / 头戴相机 {save.get('headcam_pos', np.zeros((0,))).shape[0]} 帧")

    # ---- 3) 元数据 json ----
    meta = {
        "source_mcap": args.mcap.name,
        "session": MessageToDict(meta_msg, preserving_proto_field_name=True) if meta_msg is not None else None,
        "semantic_segments": segments,
        "bad_frames": {"frames": bad_total, "bad": bad_count},
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "video": stats,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[元数据] {meta_path.name}: 语义分段 {len(segments)} 条 / bad 帧 {bad_count}/{bad_total}")
    print(f"完成 -> {out_dir}")


if __name__ == "__main__":
    main()
