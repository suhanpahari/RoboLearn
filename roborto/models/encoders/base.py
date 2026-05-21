"""Frozen visual-encoder interface for the Spine A navigation track."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FrozenEncoder(ABC):
    """A visual backbone producing a fixed-dim embedding for policy heads.

    Foundation-model encoders must NOT update their weights during RL training -
    the Spine A study compares *frozen* representations. The trained-from-scratch
    baseline is the deliberate exception (see ScratchEncoder).
    """

    name: str = "base"
    embed_dim: int = 0

    @abstractmethod
    def load(self) -> None:
        """Load weights. Called once before training."""

    @abstractmethod
    def encode(self, images):
        """Map images (B, H, W, C) to features (B, embed_dim)."""
