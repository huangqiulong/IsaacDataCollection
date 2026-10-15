#!/usr/bin/env python3
# Copyright (c) 2026, isaac workspace (IsaacDataCollection/scripts/)
# SPDX-License-Identifier: BSD-3-Clause
"""视觉域随机化(DR)在线演示: 逐一开启各视觉随机设置,每次重置后抓拍桌面相机。

对应文章: IsaacDataCollection/docs/isaac-sim-10-skillgen-dr-guide.md（系列第十篇《SkillGen 与域随机化解析》）

机制要点(源码依据 source/isaaclab_tasks/.../stack/mdp/stack_events.py):
  视觉随机化事件 ``randomize_scene_lighting_domelight`` /
  ``randomize_visual_texture_material`` 只有在 ``env.cfg.eval_mode == True``
  时才真正采样;否则它们只把灯/材质复位到默认值。开启后,``eval_type`` 决定
  本次重置激活哪一路随机化(vanilla/light_intensity/light_color/
  light_texture/table_texture/robot_texture/all)。

用法(在 isaaclab 根目录):
    ./isaaclab.sh -p ~/code/isaac/IsaacDataCollection/scripts/simulation_data_dr_demo.py \
        --settings vanilla light_intensity light_texture table_texture all \
        --resets_per_setting 3 \
        --output_dir ~/code/isaac/IsaacDataCollection/docs/images/isaac-sim-10/dr_raw
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="视觉 DR 在线演示(抓图 + 灯光参数清单)")
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor")
parser.add_argument(
    "--settings",
    nargs="+",
    default=["vanilla", "light_intensity", "light_color", "light_texture", "table_texture", "robot_texture", "all"],
    help="要演示的评估设置(对应 env.cfg.eval_type)",
)
parser.add_argument("--resets_per_setting", type=int, default=3, help="每个设置重置并抓拍的次数")
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(Path.home() / "code/isaac/IsaacDataCollection/docs/images/isaac-sim-10/dr_raw"),
)
parser.add_argument("--steps_after_reset", type=int, default=4, help="重置后推进的仿真步数(等相机刷新)")
parser.add_argument("--cam_size", type=int, default=360, help="相机分辨率(正方形)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

# ---- Kit 启动之后才可导入 ----
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401  (注册任务,供 gym.make 使用)

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# 本地演示用资产子集(兼顾离线可用性,含部分官方清单成员;
# 官方完整清单位于 stack_ik_rel_visuomotor_env_cfg.py 的 EventCfg 参数中,
# 需要完整官方资产的可从官方资产库补下载)
SRL_BG = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds"
SKY_SUBSET = [
    f"{ISAAC_NUCLEUS_DIR}/Environments/Skies/Clear/sunflowers_4k.hdr",  # 环境默认背景
    f"{ISAAC_NUCLEUS_DIR}/Environments/Skies/Cloudy/abandoned_parking_4k.hdr",  # 官方清单成员
    f"{SRL_BG}/default/brown_photostudio.hdr",
    f"{SRL_BG}/indoors/machine_shop_01_2k.hdr",
    f"{SRL_BG}/indoors/floral_tent_2k.hdr",
]
TABLE_SUBSET = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",  # 官方清单成员
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak_Planks/Oak_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Polished/Concrete_Polished_BaseColor.png",
    f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png",
]
ARM_SUBSET = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",  # 官方清单成员
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized/Aluminum_Anodized_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Polished/Concrete_Polished_BaseColor.png",
]


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def get_camera_rgb(env, cam_name: str) -> np.ndarray:
    cam = env.scene[cam_name]
    data = cam.data
    out = getattr(data, "output", None)
    if isinstance(out, dict) and "rgb" in out:
        img = out["rgb"]
    elif hasattr(data, "rgb"):
        img = data.rgb
    else:
        raise RuntimeError(f"无法从 {cam_name} 读取 RGB,可用键: {list(out.keys()) if isinstance(out, dict) else out}")
    first = img[0]
    arr = first.detach().cpu().numpy() if isinstance(first, torch.Tensor) else np.asarray(first)
    if arr.ndim != 3 and isinstance(img, torch.Tensor):  # 兜底:直接取整张
        arr = img.detach().cpu().numpy()
    if arr.ndim == 4:  # 兜底:仍带 env 维
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    return to_uint8(arr)


def read_light_attrs(env) -> dict:
    """直接读回 DomeLight 当前属性: 强度 / 颜色 / 背景贴图。"""
    prim = env.scene["light"].prim

    def g(name):
        attr = prim.GetAttribute(name)
        return attr.Get() if attr else None

    color = g("inputs:color")
    texture = g("inputs:texture:file")
    return {
        "intensity": round(float(g("inputs:intensity")), 1) if g("inputs:intensity") is not None else None,
        "color": [round(float(c), 3) for c in color] if color is not None else None,
        "texture": Path(str(texture)).name if texture else None,
    }


def main() -> None:
    from PIL import Image

    t_start = time.time()
    out_root = Path(args_cli.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1, use_fabric=True)
    # 1) 评估模式: 视觉随机化事件从此生效
    env_cfg.eval_mode = True
    # 2) 用本地子集覆盖官方完整资产清单(机制不变,只限缩资产集合)
    env_cfg.events.randomize_light.params["textures"] = SKY_SUBSET
    env_cfg.events.randomize_table_visual_material.params["textures"] = TABLE_SUBSET
    env_cfg.events.randomize_robot_arm_visual_texture.params["textures"] = ARM_SUBSET
    # 3) 放大相机分辨率,便于文章配图
    env_cfg.scene.table_cam.width = args_cli.cam_size
    env_cfg.scene.table_cam.height = args_cli.cam_size
    env_cfg.scene.wrist_cam.width = args_cli.cam_size
    env_cfg.scene.wrist_cam.height = args_cli.cam_size

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[dr-demo] 场景实体: {sorted(env.scene.keys())}")

    zeros = torch.zeros(env.action_space.shape, device=env.device)
    manifest = {
        "task": args_cli.task,
        "resets_per_setting": args_cli.resets_per_setting,
        "env_origin": [round(float(v), 4) for v in env.scene.env_origins[0].cpu().numpy()],
        "captures": [],
    }

    for setting in args_cli.settings:
        env.cfg.eval_type = setting
        t_setting = time.time()
        print(f"\n[dr-demo] === eval_type = {setting} ===")
        for k in range(args_cli.resets_per_setting):
            env.reset()
            for _ in range(args_cli.steps_after_reset):
                env.step(zeros)

            light = read_light_attrs(env)
            cubes = {}
            for name in ("cube_1", "cube_2", "cube_3"):
                if name in env.scene.keys():
                    p = env.scene[name].data.root_pos_w[0].cpu().numpy()
                    cubes[name] = [round(float(v), 4) for v in p]

            entry = {"setting": setting, "reset": k, "light": light, "cubes": cubes}
            for cam_name in ("table_cam", "wrist_cam"):
                arr = get_camera_rgb(env, cam_name)
                fn = f"{setting}_{k:02d}_{cam_name}.png"
                Image.fromarray(arr).save(out_root / fn)
                entry[f"{cam_name}_file"] = fn
            manifest["captures"].append(entry)

            tex = light["texture"] or "-"
            print(
                f"[dr-demo] {setting} #{k}: 强度={light['intensity']}, 颜色={tuple(light['color']) if light['color'] else '-'},"
                f" 贴图={tex}"
            )
        print(f"[dr-demo] {setting} 用时 {time.time() - t_setting:.1f}s")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[dr-demo] 完成: {len(manifest['captures'])} 轮抓拍,总用时 {time.time() - t_start:.1f}s → {out_root}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
