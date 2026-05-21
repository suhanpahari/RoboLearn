"""Launch multiple seeds of an experiment in parallel, one GPU per seed.

Usage:
    python scripts/launch_seeds.py \\
        experiment=exp001_fetch_sac_her \\
        gpus=GPU-aaa,GPU-bbb,GPU-ccc

Or via the Makefile:
    make launch EXP=exp001_fetch_sac_her GPUS=GPU-aaa,GPU-bbb,...

Each seed runs as an independent subprocess with CUDA_VISIBLE_DEVICES pinned to
its GPU UUID. Stdout/stderr for each seed goes to logs/launch/<exp>/<seed>.log
so they don't collide in the terminal. A live status line is printed as seeds
complete. Exit code is non-zero if any seed fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train.py"


def _parse_argv() -> tuple[str, list[str], list[int]]:
    """Parse positional key=value args. Returns (experiment, gpus, seeds)."""
    kv: dict[str, str] = {}
    for tok in sys.argv[1:]:
        if "=" not in tok:
            sys.exit(f"ERROR: expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        kv[k] = v

    experiment = kv.get("experiment") or kv.get("exp")
    gpus_raw = kv.get("gpus") or kv.get("gpu")
    if not experiment:
        sys.exit("ERROR: experiment=<name> is required")
    if not gpus_raw:
        sys.exit("ERROR: gpus=uuid1,uuid2,... is required")

    gpus = [g.strip() for g in gpus_raw.split(",") if g.strip()]
    seeds_raw = kv.get("seeds")
    seeds = [int(s.strip()) for s in seeds_raw.split(",")] if seeds_raw else list(range(len(gpus)))

    if len(seeds) != len(gpus):
        sys.exit(
            f"ERROR: {len(seeds)} seeds but {len(gpus)} GPUs — "
            "provide the same count or omit seeds to use 0..n_gpus-1"
        )
    return experiment, gpus, seeds


def main() -> None:
    experiment, gpus, seeds = _parse_argv()

    log_dir = REPO / "logs" / "launch" / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(seeds)} seeds of '{experiment}' in parallel")
    print(f"  Seeds : {seeds}")
    print(f"  GPUs  : {gpus}")
    print(f"  Logs  : {log_dir}/seed<N>.log")
    print()

    env_base = os.environ.copy()
    procs: list[tuple[int, str, subprocess.Popen, Path]] = []

    for seed, gpu in zip(seeds, gpus, strict=True):
        log_path = log_dir / f"seed{seed}.log"
        env = {**env_base, "CUDA_VISIBLE_DEVICES": gpu}
        cmd = [sys.executable, str(TRAIN), f"experiment={experiment}", f"seed={seed}"]
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO)
        procs.append((seed, gpu, p, log_path))
        print(f"  [seed {seed}] PID {p.pid} → GPU {gpu}  log: seed{seed}.log")

    print()
    print("All seeds launched. Waiting for completion...")
    start = time.monotonic()

    failed: list[int] = []
    for seed, _gpu, p, _log_path in procs:
        ret = p.wait()
        elapsed = time.monotonic() - start
        status = "DONE" if ret == 0 else f"FAILED (exit {ret})"
        print(f"  [seed {seed}] {status}  ({elapsed / 60:.1f} min)")
        if ret != 0:
            failed.append(seed)

    print()
    total = time.monotonic() - start
    if failed:
        print(f"FAILED seeds: {failed}  —  check logs in {log_dir}/")
        sys.exit(1)
    print(f"All {len(seeds)} seeds complete in {total / 60:.1f} min.")


if __name__ == "__main__":
    main()
