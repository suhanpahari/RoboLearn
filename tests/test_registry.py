"""Encoder registry behaviour."""

import pytest

from roborto.models.encoders.registry import available, build_encoder


def test_scratch_is_registered():
    assert "scratch" in available()


def test_build_known_encoder():
    encoder = build_encoder("scratch")
    assert encoder.name == "scratch"


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        build_encoder("does_not_exist")
