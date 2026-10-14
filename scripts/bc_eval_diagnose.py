"""BC 策略评估诊断脚本：定位 Mimic→BC 策略在闭环回放中的失败模式。

回放时周期性打印:
  - 动作幅度 |a|
  - 夹爪到 cube_2 / cube_3 的距离 (来自 obs['object'] 的相对向量)
  - 三块积木的高度 z (来自 obs['object'] 局部系位置; 原点 z=0)
  - 抬升量 lift2 / lift3 (相对初始高度)

用法(在 isaaclab 目录下):
  ./isaaclab.sh -p /home/dillon/code/isaac/IsaacDataCollection/scripts/bc_eval_diagnose.py \
      --checkpoint <path.pth> --num_rollouts 3 --horizon 1200 --visualizer none
"""

import argparse
import copy

from isaaclab.app import AppLauncher, scan

# ---- CLI ----
parser = argparse.ArgumentParser(description="BC policy rollout diagnostics.")
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_rollouts", type=int, default=3)
parser.add_argument("--horizon", type=int, default=1200)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_overrides = parser.parse_known_args()

# launch omniverse app
from isaaclab_tasks.utils import resolve_task_config

env_cfg_for_scan, _ = resolve_task_config(args_cli.task, "", overrides=hydra_overrides)
app_launcher = AppLauncher(args_cli, enable_cameras=scan(env_cfg_for_scan, args_cli).has_kit_camera)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

from isaaclab_tasks.utils import parse_env_cfg


def main():
    # 与 play.py 完全一致的 env 构建
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        overrides=hydra_overrides,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(101)
    np.random.seed(101)
    env.seed(101)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

    # obs['object'] 索引: [0:3]c1pos [3:7]c1quat [7:10]c2pos [10:14]c2quat [14:17]c3pos
    # [17:21]c3quat [21:24]g->c1 [24:27]g->c2 [27:30]g->c3 [30:33]c1->c2 [33:36]c2->c3 [36:39]c1->c3
    with torch.inference_mode():
        for r in range(args_cli.num_rollouts):
            policy.start_episode()
            obs_dict, _ = env.reset()
            min_d2 = min_d3 = 1e9
            n_close = 0
            z0 = None
            zs = None
            ok = False
            print(f"\n===== rollout {r} =====", flush=True)
            for t in range(args_cli.horizon):
                obs = copy.deepcopy(obs_dict["policy"])
                for ob in obs:
                    obs[ob] = torch.squeeze(obs[ob])
                actions = policy(obs)
                a = np.asarray(actions).reshape(-1)
                if a[6] < 0:
                    n_close += 1
                act = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
                obs_dict, _, terminated, truncated, _ = env.step(act)

                pol = obs_dict["policy"]
                o = pol["object"][0].cpu().numpy()
                d2 = float(np.linalg.norm(o[24:27]))
                d3 = float(np.linalg.norm(o[27:30]))
                zs = np.array([o[2], o[9], o[16]])  # 三块积木 z (局部系, 原点 z=0 -> 即世界 z)
                if z0 is None:
                    z0 = zs.copy()
                    print(
                        f"[init ] z={np.round(zs, 3)} d2={d2:.3f} d3={d3:.3f}"
                        f" grip={np.round(pol['gripper_pos'][0].cpu().numpy(), 3)}",
                        flush=True,
                    )
                min_d2 = min(min_d2, d2)
                min_d3 = min(min_d3, d3)
                if t % args_cli.print_every == 0:
                    g1 = float(pol["gripper_pos"][0, 0].cpu())
                    print(
                        f"[t={t:4d}] |a|={np.abs(a).mean():.4f} a6={a[6]:+.2f} g1={g1:.3f} d2={d2:.3f} d3={d3:.3f}"
                        f" z={np.round(zs, 3)} lift2={zs[1] - z0[1]:+.3f} lift3={zs[2] - z0[2]:+.3f}",
                        flush=True,
                    )
                ok = bool(success_term.func(env, **success_term.params)[0])
                if ok:
                    print(f"[SUCCESS] t={t}", flush=True)
                    break
                if terminated or truncated:
                    print(f"[TERMINATED] t={t} terminated={terminated} truncated={truncated}", flush=True)
                    break
            print(
                f"[summary r={r}] success={ok} min_d2={min_d2:.3f} min_d3={min_d3:.3f}"
                f" final_z={np.round(zs, 3)} lift2={zs[1] - z0[1]:+.3f} lift3={zs[2] - z0[2]:+.3f}"
                f" close_pct={100.0 * n_close / max(t + 1, 1):.0f}%",
                flush=True,
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
