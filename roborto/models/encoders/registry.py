"""Encoder registry - `build_encoder(name)` resolves a config choice to a class."""

from __future__ import annotations

from roborto.models.encoders.base import FrozenEncoder

_REGISTRY: dict[str, type[FrozenEncoder]] = {}


def register(name: str):
    """Class decorator that registers an encoder under a config name."""

    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"encoder {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_encoder(name: str) -> FrozenEncoder:
    """Instantiate a registered encoder by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    """Names of all registered encoders."""
    return sorted(_REGISTRY)


@register("scratch")
class ScratchEncoder(FrozenEncoder):
    """Trained-from-scratch CNN baseline - the control in the encoder study.

    Unlike the foundation-model encoders this one is trained jointly with the
    policy (configs/encoder/scratch.yaml sets frozen=false). STUB: implement the
    CNN in Phase 3.
    """

    embed_dim = 512

    def load(self) -> None:
        raise NotImplementedError("implement ScratchEncoder.load - Phase 3")

    def encode(self, images):
        raise NotImplementedError("implement ScratchEncoder.encode - Phase 3")


# TODO(claude-code): one registered class per remaining configs/encoder/*.yaml
# (resnet_imagenet, clip, dinov2, vc1, r3m). Foundation-model encoders keep
# weights frozen; load() pulls the published checkpoint.
