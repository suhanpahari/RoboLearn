"""Run all experiment configs sequentially on one GPU, seed by seed.

Usage:
    python scripts/run_all_exps.py gpu=GPU-xxxx [seeds=0,1,2,...] [family=manipulation]

Or via the Makefile:
    make run-all GPU=GPU-xxxx [SEEDS=0,1,2,...,7]

Experiments in configs/experiment/ are sorted alphabetically and run in order.
Navigation experiments (ddppo / habitat) are skipped automatically until the
Habitat env is set up. Within each experiment all specified seeds run sequentially
on the same GPU. Failures are collected and reported at the end; subsequent
experiments still run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "scripts" / "launch_seeds.py"
EXP_DIR = REPO / "configs" / "experiment"

_NAV_MARKERS = ("ddppo", "habitat_pointnav", "habitat_imagenav", "habitat_objectnav")


def _is_navigation(cfg_path: Path) -> bool:
    """Heuristic: skip experiment configs that reference navigation algorithms/envs."""
    text = cfg_path.read_text().lower()
    return any(m in text for m in _NAV_MARKERS)


def _parse_argv() -> tuple[str, str, str]:
    kv: dict[str, str] = {}
    for tok in sys.argv[1:]:
        if "=" not in tok:
            sys.exit(f"ERROR: expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        kv[k] = v
    gpu = kv.get("gpu") or kv.get("gpus")
    if not gpu:
        sys.exit("ERROR: gpu=<uuid> is required")
    seeds = kv.get("seeds", "0")
    family = kv.get("family", "manipulation")
    return gpu, seeds, family


def main() -> None:
    gpu, seeds, family = _parse_argv()

    all_cfgs = sorted(EXP_DIR.glob("*.yaml"))
    if not all_cfgs:
        sys.exit(f"No experiment configs found in {EXP_DIR}")

    to_run: list[str] = []
    skipped: list[str] = []
    for cfg in all_cfgs:
        name = cfg.stem
        if _is_navigation(cfg):
            skipped.append(name)
        else:
            to_run.append(name)

    print(f"run-all on GPU {gpu}  |  seeds={seeds}  |  family={family}")
    print(f"  Will run : {to_run}")
    if skipped:
        print(f"  Skipping : {skipped} (navigation — Habitat not set up yet)")
    print()

    start_all = time.monotonic()
    failed: list[str] = []

    for exp in to_run:
        print(f"{'=' * 60}")
        print(f"  Experiment: {exp}")
        print(f"{'=' * 60}")
        cmd = [
            sys.executable,
            str(LAUNCH),
            f"experiment={exp}",
            f"gpus={gpu}",
            f"seeds={seeds}",
        ]
        t0 = time.monotonic()
        ret = subprocess.run(cmd, cwd=REPO).returncode
        elapsed = (time.monotonic() - t0) / 60
        if ret == 0:
            print(f"  → {exp} DONE  ({elapsed:.1f} min)\n")
        else:
            print(f"  → {exp} FAILED (exit {ret})  ({elapsed:.1f} min)\n")
            failed.append(exp)

    total = (time.monotonic() - start_all) / 60
    print(f"{'=' * 60}")
    print(f"run-all complete in {total:.1f} min")
    if failed:
        print(f"FAILED experiments: {failed}")
        sys.exit(1)
    print(f"All {len(to_run)} experiments done.")


if __name__ == "__main__":
    main()
