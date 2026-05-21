"""Global seeding. torch is optional so this works in the manipulation env,
the Habitat env, and CI alike."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch. Call once per run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
