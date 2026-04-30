# Adaptive Teacher Replay Buffer for Safe DRL
### EECS 545 Class Project — University of Michigan

![PyTorch](https://img.shields.io/badge/PyTorch-3.2.6-red?logo=pytorch)
![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-darkgrey?logo=isaacgym)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)

**Authors:** Cody Sheltraw, Julianne Barteck, Ali Imran, Tobias Chang

---

## Overview

This project extends the [Real-DRL](https://github.com/Charlescai123/Real-DRL/tree/main/isaac-go2) framework for safe locomotion on the Unitree Go2 quadruped in NVIDIA Isaac Gym. We propose two modifications to the PHY-Teacher trigger and replay mechanism:

1. **Adaptive η Expansion** — The self-learning region starts small to trigger early PHY-Teacher activations, warm-starting the teaching-to-learn buffer before expanding to its full size. This addresses the cold-start problem where the teacher buffer is empty at the beginning of training.

2. **Safety-Prioritized Replay Sampling** — Instead of uniformly sampling from the teacher buffer, replay samples are drawn according to a skewed normal distribution over the safety-status indicator V(s), prioritizing near-failure recovery experiences.

### Configurations

| Configuration | Algorithm | PHY-Teacher | Trigger |
|---|---|---|---|
| **SAC Baseline** (no teacher) | SAC | ✗ | — |
| **Vanilla Real-DRL** | SAC | ✓ | Fixed η |
| **Ours** | SAC | ✓ | Adaptive η + prioritized replay |

**Key results:** Our method achieves **zero failure events** during training (vs. 3 for vanilla Real-DRL), reaches **2+ of 4 waypoints** during evaluation (vs. 1 for vanilla Real-DRL), and consumes **3.48% less motor energy**.

---

## Setup

### Dependencies

- Python 3.8+
- PyTorch 1.10.0
- Isaac Gym Preview 4

### Installation

1. Clone this repository `https://github.com/ALI11-2000/545-Project.git`. For baselines use, `git@github.com:Charlescai123/isaac-wild-go2.git`

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate isaac-wild
   ```

3. Install `rsl_rl`:
   ```bash
   cd extern/rsl_rl && pip install -e .
   ```

4. Download and install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym):
   ```bash
   cd isaacgym/python && pip install -e .
   ```

5. Build the Unitree Go2 SDK interface:
   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   cd extern/go2_sdk && mkdir build && cd build
   cmake .. && make
   mv go2_interface* ../../..
   ```

### WandB Configuration

Set your WandB API key in the following two files before running any experiments:

- `src/drl_student/runners/off_policy_runner.py`
- `src/utils/plot_trajectory.py`

---

## Running Experiments

### Baseline: Vanilla Real-DRL (SAC + PHY-Teacher, fixed η)

For the vanilla Real-DRL baseline, clone and run the upstream repository:
[https://github.com/Charlescai123/Real-DRL/tree/main/isaac-go2](https://github.com/Charlescai123/Real-DRL/tree/main/isaac-go2)

### Our Method: Adaptive η + Prioritized Replay

**Training:**
```bash
python -m scripts.sac.train --enable_phy_teacher=True --show_gui=True --experiment_name sac_phy_teacher
```

> To enable the Isaac Gym viewer, set `viewer=True` inside `go2_wild_env.py` as well.

**Evaluation** (update `--logdir` to your checkpoint):
```bash
python -m scripts.sac.eval \
  --logdir=logs/sac_phy_teacher/2026_03_22_22_40_57 \
  --enable_phy_teacher=True \
  --show_gui=True
```

**Plot trajectory:**
```bash
python -m src.utils.plot_trajectory
```

---

## Acknowledgements

Built on top of the [Real-DRL](https://github.com/Charlescai123/Real-DRL) framework by Mao et al.