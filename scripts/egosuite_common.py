#!/usr/bin/env python3
"""EgoSuite-Open100K 处理演示：公共工具模块。

被以下脚本共享：
  - egosuite_mcap_inspect.py   数据结构检查
  - egosuite_mcap_extract.py   数据抽取（视频 / 姿态 / 元数据）

提供：MCAP 内嵌 Protobuf 动态解码、四元数与投影几何、
手部骨架定义、视频抽帧、matplotlib 中文字体配置。

运行环境：见仓库 README（需 mcap / av / numpy / matplotlib）。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Protobuf 动态解码（MCAP 内嵌 FileDescriptorSet，无需 .proto 文件）
# ---------------------------------------------------------------------------


def build_descriptor_pool(summary):
    """从 MCAP summary 中提取全部内嵌 schema，构建 protobuf 描述符池。"""
    from google.protobuf import descriptor_pb2, descriptor_pool

    pool = descriptor_pool.DescriptorPool()
    for _sid, sch in summary.schemas.items():
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(sch.data)
        for fd in fds.file:
            try:
                pool.Add(fd)
            except Exception:
                pass
    return pool


def build_message_classes(pool, summary):
    """(schema_id, name) -> 动态生成的 protobuf Message 类。"""
    from google.protobuf import message_factory

    classes = {}
    for sid, sch in summary.schemas.items():
        classes[(sid, sch.name)] = message_factory.GetMessageClass(
            pool.FindMessageTypeByName(sch.name)
        )
    return classes


def decode_message(schema, data, classes):
    """按 schema 解码一条消息。"""
    msg = classes[(schema.id, schema.name)]()
    msg.ParseFromString(data)
    return msg


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------


def quat2R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """(w, x, y, z) 四元数 -> 3x3 旋转矩阵。"""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def quat_rel_angle_deg(qa, qb) -> np.ndarray:
    """两组 (w,x,y,z) 四元数序列之间的相对旋转角（度）。支持 (T,4) 数组。"""
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    w1, x1, y1, z1 = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    w2, x2, y2, z2 = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    # q = qa^-1 * qb（共轭乘法）
    w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    w = np.clip(np.abs(w), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def project_world_points(points_world, cam_pos, cam_quat_wxyz, K):
    """世界系点 -> 像素坐标（OpenCV 针孔模型）。

    参数
    ----
    points_world : (N,3) 世界系点
    cam_pos      : (3,)  相机在世界系的位置（= /pose/headcam 的 pos）
    cam_quat_wxyz: (4,)  相机姿态四元数 (w,x,y,z)
    K            : (3,3) 相机内参

    返回
    ----
    (uv (N,2) 像素坐标, z (N,) 相机系深度 m)
    """
    R = quat2R(*np.asarray(cam_quat_wxyz, dtype=float))
    R_cw = R.T
    t_cw = -R_cw @ np.asarray(cam_pos, dtype=float)
    pc = np.asarray(points_world, dtype=float) @ R_cw.T + t_cw
    z = pc[:, 2]
    uv = (np.asarray(K, dtype=float) @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


# 21 关键点手部骨架（MANO 约定：0=手腕；1-4 拇指、5-8 食指、
# 9-12 中指、13-16 无名指、17-20 小指，每指从根部到指尖）
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
THUMB_TIP, INDEX_TIP = 4, 8


def aperture_cm(pos: np.ndarray) -> np.ndarray:
    """拇指尖-食指尖距离（cm）——夹爪开度的直接代理量。pos: (T,21,3)。"""
    d = np.linalg.norm(pos[:, THUMB_TIP] - pos[:, INDEX_TIP], axis=-1)
    return d * 100.0


# ---------------------------------------------------------------------------
# 视频抽帧
# ---------------------------------------------------------------------------


def decode_selected_frames(video_path, indices):
    """流式解码视频，仅保留 indices 指定的帧（BGR ndarray）。

    返回 {frame_index: ndarray(H,W,3) BGR}。用于骨架叠加等抽样场景，
    避免整段解码占用大量内存。
    """
    import av

    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return {}
    deadline = wanted[-1]
    frames = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="bgr24")
            if i >= deadline:
                break
    return frames


# ---------------------------------------------------------------------------
# matplotlib 中文字体
# ---------------------------------------------------------------------------


def setup_matplotlib():
    """配置 Agg 后端与中文字体，返回 pyplot。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    prefer = [
        "Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [p for p in prefer if p in installed]
    plt.rcParams["font.sans-serif"] = (picked or ["DejaVu Sans"]) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt
