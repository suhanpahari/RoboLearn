# Roborto - Claude Code operating instructions

## Project
Embodied-RL research monorepo. Manipulation via Gymnasium-Robotics (MuJoCo),
navigation via AI Habitat. Research spine A: representations & exploration for
sample-efficient embodied control. Goal: a reproducible repo + arXiv technical
reports for a PhD application.

**Read `docs/PLAN.md` before any non-trivial change.** It is the source of truth.

## Hard rules
- NEVER invent, estimate, or hand-edit experimental results, metrics, or plots.
  Numbers come only from real runs; figures only from scripts/make_figures.py.
- Before implementing a new experiment, write its design note in
  docs/experiments/expNNN_*.md (copy docs/experiments/_TEMPLATE.md) and wait
  for review.
- ALL hyperparameters live in Hydra configs under configs/. Never hardcode them.
- Every change: `make lint` and `make test` must pass. New experiment code must
  also pass `make smoke` (a tiny end-to-end run) before you call it done.
- Set and log seeds; every run records git sha + resolved config + package
  versions (roborto/utils/logging.py already does this - keep it that way).
- results/, data/, wandb/ are gitignored. Only small CSVs in results_summary/.
- Do NOT launch long GPU training runs. Set them up, estimate the cost, and
  stop - the human launches them.
- Small, focused PRs. Match the existing style (ruff). Update docs/ when
  behaviour changes, and append to docs/lab_notebook.md when an experiment ends.

## Environments & GPUs
Two conda envs - roborto-gym (manipulation) and roborto-hab (Habitat) - with
conflicting dependencies; never merge them. See docs/setup.md.
Shared, unscheduled DGX box: scripts/train.py and the Makefile take GPU=<uuid>
and export CUDA_VISIBLE_DEVICES. Never hardcode device indices - MIG UUIDs are
the stable identifier. Every training run must be checkpoint-resumable. Do not
write multi-GPU code unless explicitly asked.

## Build vs reuse
Baselines reuse trusted libraries (Stable-Baselines3 + sb3-contrib for SAC/HER;
Habitat-Lab DD-PPO for navigation). In-repo code is for the contribution only:
HER relabeling (roborto/buffers/her.py) and frozen encoders
(roborto/models/encoders/). Do not reimplement what a trusted library provides.

## Reproducing baselines
When reproducing a published result, cite the source and the target number in
the design note; success means "matched within the confidence interval".

## Workflow for any new experiment
design note -> implement + unit tests -> smoke run -> sanity/baseline check
-> (human launches the full multi-seed runs) -> evaluate on held-out data
-> rliable aggregation -> figures via script -> lab-notebook entry + results row.

## Where things live
- configs/         Hydra groups: env, algo, encoder, experiment, sweep
- roborto/utils/   seeding, device, logging, provenance - these WORK already
- roborto/buffers/ replay buffer + HER relabeler - implemented and tested
- roborto/{algos,models,training,evaluation,analysis}/  mostly stubs to fill in
- scripts/         train.py is wired (Hydra + seed + device + logging); the
                   Trainer itself is a stub for Phase 1-2
