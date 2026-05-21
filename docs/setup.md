# Setup

Roborto uses **two** conda environments because the Gymnasium-Robotics and Habitat
dependency stacks conflict. The `roborto` package installs into both.

## 1. Manipulation environment (`roborto-gym`)

```bash
make setup-gym          # conda env create -f environment-gymrobotics.yml
conda activate roborto-gym
make test
```

This pulls Gymnasium-Robotics + MuJoCo + Stable-Baselines3. Recent MuJoCo wheels are
self-contained; on a headless server you may need EGL/OSMesa for rendering - see the
MuJoCo docs.

> Verify the environment versions in `configs/env/*.yaml`. Gymnasium-Robotics bumps
> environment version suffixes (e.g. `FetchReach-vN`); confirm the live ids with
> `gymnasium.pprint_registry()` and update the configs.

## 2. Habitat environment (`roborto-hab`)

```bash
make setup-hab          # conda env create -f environment-habitat.yml
conda activate roborto-hab
```

Habitat-Sim is installed from the `aihabitat` conda channel; Habitat-Lab via pip.
Confirm the currently supported Python version and the correct `habitat-sim` build
(headless and/or with-Bullet) against the official Habitat install docs.

### MIG + rendering check - do this first

Habitat-Sim renders via EGL, and MIG slices may not support GPU rendering. Before
committing to a workflow, sanity-check Habitat-Sim on a MIG device:

```bash
CUDA_VISIBLE_DEVICES=<MIG-UUID> python -c "import habitat_sim; print('habitat_sim ok')"
# then load a test scene; if EGL fails, run Habitat on a full (non-MIG) card only.
```

### Scene datasets - gated

HM3D / MP3D / Gibson require accepting their licenses. After that, download via the
Habitat datasets downloader (wrapped by `scripts/download_data.py`). Data is large,
lives under `data/`, and is gitignored - never commit it.

## 3. Weights & Biases

Run `wandb login` once. To run without W&B set `logging.backend=offline`; metrics are
then written to a JSONL file in the run directory.
