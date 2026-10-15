# IsaacDataCollection · Hands-on Robot Data Collection with Isaac Sim

[简体中文](README.md) ｜ **English**

> A hands-on knowledge base for **robot data collection with NVIDIA Isaac Sim / Isaac Lab** — technical blog posts, in-depth documents, and companion scripts, all validated on real hardware (single RTX 4070 Laptop, 8 GB).

`Isaac Sim 6.1` · `Isaac Lab 3.0` · `Python` · `BSD-3-Clause`

---

## Overview

The race in robot learning is shifting from *models* to *data*. This repository documents a complete, practical route for collecting training data inside the **Isaac Sim simulator**:

- **A tour of five data-collection paradigms**: teleoperation, motion capture (mocap), UMI, EGO, and simulation-first generation (Mimic / SkillGen / RL / SDG) — plus a dimension-by-dimension comparison with real-robot collection;
- **Reproducible hands-on guides**: from a cold start, teleoperate a Franka arm to stack three cubes using keyboard / Xbox controller, and record HDF5 demonstration datasets;
- **A UMI-style vision workflow**: wrist + table camera data through collection → video export → QC → augmentation → visual BC, end to end (Article 5);
- **An EGO (first-person) data pipeline**: simulation side — G1 humanoid Mimic augmentation (20 demos / 4,790 frames) + locomanipulation SDG (873 first-person frames) → LeRobot conversion; real side — a hands-on EgoSuite-Open100K walkthrough (49 TB-scale HF Bucket → MCAP parsing → 3D hand-skeleton overlay on egocentric video → pseudo-action construction) (Article 6);
- **Mocap retargeting hands-on**: LAFAN1 public data (77 BVH clips) → GMR IK retargeting to a Unitree G1 → CSV/npz training-ready format, fully validated; plus the GVHMR "video → SMPL-X" pipeline and its model-registration gates;
- **A full data loop**: seed demos → Mimic augmentation → behavior cloning (BC) training → success-rate evaluation, with measured numbers at every step;
- **Deep dives into the data format**: HDF5 structure, time-alignment relationships, quaternion order conventions, processing recipes and pitfalls (96-byte truncation, merge fix-ups, etc.), plus a LeRobot v2/v3 deep dive with a `lerobot/pusht` public-dataset walkthrough (Article 7);
- **Mocap end-to-end & RLDS**: LAFAN1 → GMR retargeting to a Unitree G1 (45 fps measured) → training-ready npz, plus the GVHMR video pipeline with its registration gates (Article 8); the RLDS/OXE format deep dive with an 86MB public dataset downloaded, decoded and visualized, plus VS Code TFRecord tooling (Article 9);
- **Planning-based generation & domain randomization**: a SkillGen explainer (cuRobo transition planning, manual annotation contract, hardware gates) plus DR's three layers — including a 7-setting visual-randomization capture walkthrough (Article 10).

Everything has been validated hands-on: 3 keyboard demos (2,780 steps), a UMI-style dual-camera pipeline (5 demos / 2,599 frames / dual 200×200 cameras), Mimic scaling 30 seeds into **300 demonstrations (237k frames)**, BC success rate improving from **1/20 to 5/20**, a G1 locomanipulation generation chain (Mimic 20 demos → SDG 873 frames → LeRobot v2.0 conversion), EgoSuite-Open100K real-world episodes (3 episodes / 61 MB), real mocap data visualizations, an end-to-end mocap retargeting chain (LAFAN1 → Unitree G1: IK retargeting → CSV → training-ready npz), an RLDS/OXE public-dataset download & decode walkthrough (86MB sample), a 7-setting visual domain-randomization capture run (21 resets), and more.

## Preview

<p align="center">
  <img src="docs/images/isaac-sim-01/fig1-five-cards.png" width="49%" alt="Five data-collection paradigms">
  <img src="docs/images/isaac-sim-01/fig4-mocap-data.png" width="49%" alt="Real mocap data visualization">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-02/fig2-keyboard.png" width="49%" alt="Keyboard teleoperation keymap">
  <img src="docs/images/isaac-sim-03/fig6-scale-up.png" width="49%" alt="Mimic scaling results">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-05/fig4-camera-frames.png" width="49%" alt="UMI-style dual-camera real frames (wrist / table)">
  <img src="docs/images/isaac-sim-05/fig1-umi-vs-sim.png" width="49%" alt="Real UMI vs Isaac Visuomotor mapping">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-06/fig5-hand-overlay.png" width="49%" alt="EgoSuite hand-skeleton overlay on egocentric video">
  <img src="docs/images/isaac-sim-06/fig7-grip-signal.png" width="49%" alt="EgoSuite grip aperture with semantic segments">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-06/fig2-two-chains.png" width="49%" alt="EGO workflow: simulation chain + real-data chain">
  <img src="docs/images/isaac-sim-07/fig1-files.png" width="49%" alt="LeRobot v3.0 directory structure">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-08/fig2-walk-compare.png" width="49%" alt="Mocap retargeting: human skeleton vs Unitree G1">
  <img src="docs/images/isaac-sim-09/fig4-frames.png" width="49%" alt="RLDS sample frames (imperial_wrist_dataset)">
</p>
<p align="center">
  <img src="docs/images/isaac-sim-10/fig1-three-paradigms.png" width="49%" alt="Three data-generation paradigms: Mimic / SkillGen / domain randomization">
  <img src="docs/images/isaac-sim-10/fig8-dr-montage.png" width="49%" alt="Visual domain randomization: 7 eval settings × 3 resets">
</p>

> All figures live in `docs/images/isaac-sim-01 ~ 10/`, one folder per article.

## Blog Series (Recommended Reading)

| No. | Article | Topic |
| --- | --- | --- |
| (1) | [Simulation Data Collection Methods in Isaac Sim — Overview](docs/isaac-sim-01-data-collection-survey.md) | Five paradigms: principles, data formats, and how to get started |
| (2) | [Teleoperation Hands-On: Franka Stack-Cube (Keyboard / Xbox360 / Mouse)](docs/isaac-sim-02-teleoperation-hands-on.md) | Warm-up, recording, QC, and the full Mimic→BC loop |
| (3) | [Isaac Sim Teleoperation vs. Real-Robot Collection](docs/isaac-sim-03-sim-vs-real-comparison.md) | A dimension-by-dimension comparison (18 aspects) |
| (4) | [HDF5 Format Explained: Structure, Visualization and Processing](docs/isaac-sim-04-hdf5-format-guide.md) | Dataset structure, time alignment, processing recipes and pitfalls |
| (5) | [UMI-Style Data Workflow: Collection → Processing → Training](docs/isaac-sim-05-umi-workflow.md) | Mapping real-world UMI to Isaac + full-pipeline walkthrough (dual cameras, visual BC) |
| (6) | [EGO-Style Data Workflow: Simulation → Public Dataset → Retargeting Demo](docs/isaac-sim-06-ego-workflow.md) | G1 first-person SDG + EgoSuite-Open100K hands-on (skeleton overlay, pseudo actions) |
| (7) | [LeRobot Data Format Explained: Structure, Download & Hands-On](docs/isaac-sim-07-lerobot-format-guide.md) | Format deep dive + pusht public-dataset walkthrough + VS Code tooling |
| (8) | [Mocap End-to-End: From Public Motion Capture / Monocular Video to Robot Training Data](docs/isaac-sim-08-mocap-workflow.md) | GMR retargeting hands-on (LAFAN1 → G1) + GVHMR video pipeline + AMP consumption |
| (9) | [RLDS Data Format Explained: Structure, Download & Hands-On](docs/isaac-sim-09-rlds-format-guide.md) | OXE format deep dive + 86MB public-dataset walkthrough + VS Code TFRecord tooling |
| (10) | [SkillGen & Domain Randomization Explained: Planning-Based Data Generation in Isaac Lab](docs/isaac-sim-10-skillgen-dr-guide.md) | cuRobo transition planning + manual annotation contract + hardware gates + DR mechanism with a 7-setting walkthrough |

> Articles are written in Chinese (published on Chinese tech platforms); this English README provides navigation and an overview.

## Companion Scripts

| Script | Purpose |
| --- | --- |
| `scripts/teleop_dataset_stats.py` | Dataset statistics & quality check (steps / jerk / gripper toggles, no Isaac required) |
| `scripts/hdf5_plot_demo.py` | Per-demo curve visualization (actions / joints / end-effector trajectory) |
| `scripts/isaaclab_teleop_launch.py` | Launcher enabling gamepad/mouse recording (registers `-Teleop` task variants, upstream repo untouched) |
| `scripts/isaaclab_teleop_extras.py` | Teleop extensions (mouse device, gamepad reset key, etc.) |
| `scripts/glfw_gamepad_mapping_patch.py` | Xbox Bluetooth gamepad (fw0509) GLFW mapping patch |
| `scripts/isaac_gamepad_mapping_dump.py` | Gamepad keymap dump (headless Kit) |
| `scripts/rlds_download.py` | RLDS/OXE dataset downloader (direct GCS, chunk-parallel for large shards) |
| `scripts/rlds_demo.py` | RLDS read/decode/visualize demo (structure summary + camera grids + action curves) |
| `scripts/simulation_data_dr_demo.py` | Visual domain-randomization demo: 7 eval settings × reset captures + light-parameter read-back (launches Isaac) |
| `scripts/egosuite_download.sh` | HF Bucket download helper for EgoSuite-Open100K (list / task / get / sync) |
| `scripts/egosuite_mcap_inspect.py` / `egosuite_mcap_extract.py` | EgoSuite MCAP structure inspection & data extraction (shared module `egosuite_common.py`) |

## Repository Layout

```
IsaacDataCollection/
├── README.md                 # 中文说明
├── README_EN.md              # English README (this file)
├── LICENSE                   # BSD-3-Clause
├── docs/
│   ├── isaac-sim-01~10-*.md  # Blog series (10 articles)
│   └── images/               # Figures per article (isaac-sim-01 ~ 10)
├── datasets/                 # Sample dataset (dataset_gamepad.hdf5: 30 gamepad teleop demos)
└── scripts/                  # Companion tooling
```

## Quick Start

Prerequisites: a machine with Isaac Sim / Isaac Lab installed. The utility scripts themselves only need Python with `h5py` / `numpy` / `matplotlib`.

```bash
# 1) Dataset quality check (replace the HDF5 path with yours)
python scripts/teleop_dataset_stats.py ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5

# 2) Visualize a single demonstration
python scripts/hdf5_plot_demo.py ~/code/isaac/IsaacDataCollection/datasets/dataset_gamepad.hdf5 --demo demo_1

# 3) Usage of the RLDS read/visualize script (full walkthrough in Article 9)
python scripts/rlds_demo.py --help
```

> For the full teleop / Mimic / training commands, see blog posts (2), (4) and (5)–(7); RLDS and DR scripts are covered in posts (9) and (10).

## Tested Environment

- Hardware: RTX 4070 Laptop (8 GB) × 1
- Software: Isaac Sim 6.1 · Isaac Lab 3.0 · Python 3.12
- Tasks: `IsaacContrib-Stack-Cube-Franka-IK-Rel` (three-cube stacking, blue → red → green); plus `IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor` (dual 200×200 camera variant); `Isaac-Locomanipulation-G1-Abs-Mimic-v0` (G1 Mimic augmentation, Article 6) and `Isaac-G1-SteeringWheel-Locomanipulation` (G1 first-person locomanipulation SDG, Article 6)

## License & Acknowledgements

This project (docs and code) is licensed under [BSD-3-Clause](LICENSE).

Thanks to the NVIDIA [Isaac Sim](https://developer.nvidia.com/isaac-sim) / [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [robomimic](https://robomimic.github.io/) ecosystems. Projects referenced in the articles (ALOHA, UMI, GELLO, Open-TeleVision, etc.) belong to their respective authors.
