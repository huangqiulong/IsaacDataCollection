#!/usr/bin/env python3
"""EgoSuite-Open100K 数据结构检查。

打印单个 MCAP 演示集的结构摘要：内嵌 schema、频道与消息数、时长、
会话元数据（采集设备/场景/任务描述/操作员）、事件级语义分段与
手部姿态质量标注（bad frame）统计。

用法
----
    python egosuite_mcap_inspect.py <file.mcap> [<file2.mcap> ...] [--json out.json]

依赖：mcap、protobuf（见仓库 README）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from egosuite_common import build_descriptor_pool, build_message_classes, decode_message

from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader

WANT_TOPICS = [
    "/session/metadata",
    "/annotation/semantic_segments",
    "/annotation/bad_frame/pose/hand",
    "/pose/head",
    "/pose/headcam",
    "/pose/right_eye_cam",
    "/pose/left_hand",
    "/pose/right_hand",
    "/sensor/camera/head_left/intrinsic",
    "/sensor/camera/head_right/intrinsic",
    "/sensor/camera/head_left/extrinsic",
    "/sensor/camera/head_right/extrinsic",
]


def inspect_one(path: Path) -> dict:
    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    pool = build_descriptor_pool(summary)
    classes = build_message_classes(pool, summary)

    info: dict = {"file": path.name, "size_mb": path.stat().st_size / 1e6}
    st = summary.statistics
    if st:
        info["messages"] = st.message_count
        info["duration_s"] = (st.message_end_time - st.message_start_time) / 1e9
    info["channels"] = {
        summary.channels[cid].topic: {
            "schema": summary.schemas[summary.channels[cid].schema_id].name,
            "count": (st.channel_message_counts.get(cid, 0) if st else 0),
        }
        for cid in summary.channels
    }

    # 采样读取：元数据 / 语义分段（全部）/ bad frame 统计
    seen: dict[str, object] = {}
    segments: list[dict] = []
    bad_total, bad_count, problem_types = 0, 0, set()
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            topic = channel.topic
            if topic == "/annotation/semantic_segments":
                m = decode_message(schema, message.data, classes)
                segments.append(MessageToDict(m, preserving_proto_field_name=True))
            elif topic == "/annotation/bad_frame/pose/hand":
                m = decode_message(schema, message.data, classes)
                bad_total += 1
                if m.is_bad:
                    bad_count += 1
                    problem_types.update(m.problem_type)
            elif topic in WANT_TOPICS and topic not in seen:
                seen[topic] = decode_message(schema, message.data, classes)

    info["semantic_segments"] = segments
    info["bad_frames"] = {
        "frames": bad_total, "bad": bad_count, "problem_types": sorted(problem_types),
    }

    # ---------------- 打印 ----------------
    print(f"\n{'='*78}\n文件: {path.name}  ({info['size_mb']:.1f} MB, "
          f"{info.get('messages', '?')} 条消息, 时长 {info.get('duration_s', 0):.1f} s)\n{'='*78}")

    print("\n[频道] 共 %d 个" % len(info["channels"]))
    for topic, ch in info["channels"].items():
        print(f"  {topic:<45} {ch['count']:>7,}  <- {ch['schema']}")

    meta = seen.get("/session/metadata")
    if meta is not None:
        print("\n[会话元数据]")
        print("  " + text_format.MessageToString(meta).replace("\n", "\n  "))

    if segments:
        print("\n[事件级语义分段]（共 %d 条消息）" % len(segments))
        for i, seg in enumerate(segments):
            print(f"  #{i}: task_description={seg.get('task_description', '')[:80]}")
            for s in seg.get("segment", []) if isinstance(seg.get("segment"), list) else [seg.get("segment", {})]:
                print(f"      - {s.get('subtask_description', '')} (end_time={s.get('end_time')} s)")

    bf = info["bad_frames"]
    print(f"\n[手部姿态质量标注] {bf['frames']} 帧中 {bf['bad']} 帧标记为 bad"
          + (f"（类型: {', '.join(bf['problem_types'])}）" if bf["problem_types"] else ""))

    intr = seen.get("/sensor/camera/head_left/intrinsic")
    if intr is not None:
        K = list(intr.K)
        print(f"\n[相机内参] {intr.width}x{intr.height}  fx={K[0]:.1f} fy={K[4]:.1f} "
              f"cx={K[2]:.1f} cy={K[5]:.1f} (frame_id={intr.frame_id})")

    ext = seen.get("/sensor/camera/head_left/extrinsic")
    if ext is not None and len(ext.transforms):
        tr = ext.transforms[0]
        print(f"[相机外参] {tr.parent_frame_id} -> {tr.child_frame_id}  "
              f"t=({tr.translation.x:.3f}, {tr.translation.y:.3f}, {tr.translation.z:.3f})")

    hands = seen.get("/pose/right_hand")
    if hands is not None:
        print(f"[手部姿态] 每帧 {len(hands.transforms)} 个关键点（pos + quat，世界系）")

    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mcaps", nargs="+", type=Path, help="MCAP 文件路径")
    ap.add_argument("--json", type=Path, default=None, help="可选：将结构化结果写入 JSON")
    args = ap.parse_args()

    results = [inspect_one(p) for p in args.mcaps]
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nJSON 已写入: {args.json}")


if __name__ == "__main__":
    main()
