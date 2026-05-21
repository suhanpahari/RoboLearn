"""Expand a sweep config into the grid of train.py commands.

python scripts/sweep.py sweep=her_relabel_strategy
"""

import itertools
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].startswith("sweep="):
        sys.exit("usage: python scripts/sweep.py sweep=<name>")
    name = sys.argv[1].split("=", 1)[1]
    spec = OmegaConf.load(REPO / "configs" / "sweep" / f"{name}.yaml")
    axes = OmegaConf.to_container(spec.axes, resolve=True)
    keys = list(axes)

    print(f"# sweep '{name}' over base_experiment={spec.base_experiment}")
    for combo in itertools.product(*(axes[k] for k in keys)):
        overrides = " ".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True))
        print(f"python scripts/train.py experiment={spec.base_experiment} {overrides}")

    # TODO(claude-code): instead of printing, dispatch each run on an idle GPU
    # (found via nvidia-smi) inside tmux, and record a sweep id for later analysis.


if __name__ == "__main__":
    main()
