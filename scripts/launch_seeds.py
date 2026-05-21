"""Launch multiple seeds of an experiment in parallel, pooled across GPUs.

Usage:
    python scripts/launch_seeds.py \\
        experiment=exp001_fetch_sac_her \\
        gpus=GPU-aaa,GPU-bbb,GPU-ccc \\
        seeds=0,1,2,3,4,5,6,7       # optional; defaults to 0..n_gpus-1

Or via the Makefile:
    make launch EXP=exp001_fetch_sac_her GPUS=uuid1,uuid2,...
    make launch EXP=exp001_fetch_sac_her GPUS=uuid1,uuid2,... SEEDS=0,1,2,3,4,5,6,7

Pool behaviour: if there are more seeds than GPUs, each GPU runs its share of seeds
sequentially — as soon as a seed finishes its GPU picks up the next one from the
queue. All seeds eventually complete in ceil(n_seeds / n_gpus) rounds.

Stdout/stderr per seed → logs/launch/<exp>/seed<N>.log (never mixed in terminal).
Exit code is non-zero if any seed fails.
"""

from __future__ import annotations

import collections
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
    return experiment, gpus, seeds


def _spawn(experiment: str, seed: int, gpu: str, log_dir: Path) -> subprocess.Popen:
    log_path = log_dir / f"seed{seed}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    cmd = [sys.executable, str(TRAIN), f"experiment={experiment}", f"seed={seed}"]
    log_f = open(log_path, "w")  # noqa: SIM115 — intentionally kept open for subprocess lifetime
    p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO)
    return p


def main() -> None:
    experiment, gpus, seeds = _parse_argv()

    log_dir = REPO / "logs" / "launch" / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    n_parallel = min(len(seeds), len(gpus))
    print(f"Launching {len(seeds)} seeds of '{experiment}'  ({n_parallel} GPUs in pool)")
    print(f"  Seeds : {seeds}")
    print(f"  GPUs  : {gpus}")
    print(f"  Logs  : {log_dir}/seed<N>.log")
    if len(seeds) > len(gpus):
        extra = len(seeds) - len(gpus)
        print(f"  Note  : {extra} seeds will queue behind the first {len(gpus)}")
    print()

    # Build a queue of (seed, gpu) pairs — seeds are striped across GPUs so each
    # GPU gets a roughly equal share in round-robin order.
    queue: collections.deque[tuple[int, str]] = collections.deque()
    for i, seed in enumerate(seeds):
        queue.append((seed, gpus[i % len(gpus)]))

    # Spawn the first batch (one process per GPU slot).
    # running: maps pid -> (seed, gpu, Popen)
    running: dict[int, tuple[int, str, subprocess.Popen]] = {}
    for _ in range(n_parallel):
        seed, gpu = queue.popleft()
        p = _spawn(experiment, seed, gpu, log_dir)
        running[p.pid] = (seed, gpu, p)
        print(f"  [seed {seed}] PID {p.pid} → {gpu}  log: seed{seed}.log")

    if queue:
        print(f"  ... {len(queue)} seeds queued")
    print()

    start = time.monotonic()
    failed: list[int] = []

    while running:
        time.sleep(2)
        for pid in list(running):
            seed, gpu, p = running[pid]
            ret = p.poll()
            if ret is None:
                continue  # still running
            del running[pid]
            elapsed = time.monotonic() - start
            status = "DONE" if ret == 0 else f"FAILED (exit {ret})"
            print(f"  [seed {seed}] {status}  ({elapsed / 60:.1f} min elapsed)")
            if ret != 0:
                failed.append(seed)

            # Assign the freed GPU to the next queued seed.
            if queue:
                next_seed, _ = queue.popleft()
                p2 = _spawn(experiment, next_seed, gpu, log_dir)
                running[p2.pid] = (next_seed, gpu, p2)
                print(
                    f"  [seed {next_seed}] PID {p2.pid} → {gpu} (queued)  log: seed{next_seed}.log"
                )

    print()
    total = time.monotonic() - start
    if failed:
        print(f"FAILED seeds: {failed}  —  check logs in {log_dir}/")
        sys.exit(1)
    print(f"All {len(seeds)} seeds complete in {total / 60:.1f} min.")


if __name__ == "__main__":
    main()
