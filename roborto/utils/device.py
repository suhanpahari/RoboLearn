"""Device resolution for a shared, unscheduled GPU box.

The card is chosen *outside* the process via CUDA_VISIBLE_DEVICES=<uuid>
(see the Makefile and docs/PLAN.md section 2). Inside the process the visible
card is always cuda:0, so this module never selects an index itself.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def cuda_visible() -> str:
    """Raw value of CUDA_VISIBLE_DEVICES (empty string if unset)."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def resolve_device(prefer_cuda: bool = True):
    """Return the torch device for this process."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def log_gpu_visibility() -> None:
    """Log which card this process can see, and warn if none was pinned."""
    vis = cuda_visible()
    log.info("CUDA_VISIBLE_DEVICES=%s", vis or "<unset>")
    if not vis:
        log.warning(
            "CUDA_VISIBLE_DEVICES is unset - this run may land on a busy card. "
            "Launch via `make train EXP=... GPU=<uuid>` (find an idle one with nvidia-smi)."
        )
