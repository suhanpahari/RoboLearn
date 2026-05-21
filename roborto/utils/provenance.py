"""Run provenance - capture exactly what produced a result."""

from __future__ import annotations

import platform
import subprocess
import sys


def git_sha(short: bool = True) -> str:
    """Current commit hash, or 'unknown' outside a git repo."""
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    """True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def env_provenance() -> dict:
    """A snapshot of the environment that a run should record."""
    try:
        import importlib.metadata as md

        packages = {d.metadata["Name"]: d.version for d in md.distributions()}
    except Exception:
        packages = {}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "packages": packages,
    }
