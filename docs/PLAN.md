# Roborto — Project Plan

> A reproducible research playground for embodied reinforcement learning across
> **manipulation** (Farama Gymnasium-Robotics) and **navigation** (AI Habitat).
>
> Purpose: produce (a) a clean, well-engineered public repository and (b) a series of
> honest technical reports on arXiv, as evidence of research ability for a PhD application.

This document is the master spec. It is written to be handed to Claude Code as the
source of truth for what to build. Read it top to bottom before scaffolding anything.

---

## 0. What this project is actually for

A PhD admissions committee is not grading "did the experiments work." They are reading
the repo and reports as evidence of four things:

1. **Research taste** — you picked a coherent question, not a pile of unrelated demos.
2. **Technical execution** — the code is correct, tested, and reproducible.
3. **Rigor** — results have error bars, multiple seeds, honest baselines.
4. **Communication** — the reports are clear and the conclusions are warranted.

Two consequences for every decision below:

- **A coherent narrative beats volume.** One well-investigated question with strong
  baselines and honest ablations is worth more than ten shallow experiments.
- **A negative result is valuable *only if the implementation is trustworthy*.** "It
  failed" is interesting research; "it failed because of a bug" is noise. This is why so
  much of this plan is about correctness infrastructure — it is what makes your failures
  *count*.

---

## 1. Research framing — pick one spine

Using *both* platforms only makes sense if a single question runs through both. Otherwise
you have two disconnected mini-projects. Below are four candidate spines. **Pick exactly
one.** The infrastructure in Sections 3–7 is identical regardless of which you choose.

> Before committing, spend 2–3 days on a focused literature review for your chosen spine.
> You need to know the 5–10 closest papers so your reports position the work correctly and
> you do not accidentally reinvent something. None of the ideas below are claimed to be
> novel — that is yours to establish.

### Spine A (recommended) — Representations & exploration for sample-efficient embodied control
*Central question:* what makes an embodied agent learn fast and generalize, and does the
answer transfer between manipulation and navigation?
- **Manipulation track:** sparse-reward goal-conditioned learning on Fetch/Hand tasks —
  goal relabeling (HER) and exploration are the levers.
- **Navigation track:** frozen visual foundation-model encoders vs trained-from-scratch
  encoders for Habitat PointNav/ImageNav — representation is the lever.
- **Cross-cutting synthesis:** is "a good representation / good exploration" a
  domain-specific or domain-general property?
- *Why recommended:* tractable on modest compute, genuinely spans both platforms, and the
  synthesis question is what makes it a *research project* rather than two benchmarks.

### Spine B — Frozen foundation models as universal perception for control
Train small policy heads on top of frozen encoders (DINOv2, CLIP, VC-1, R3M, ImageNet
ResNet) for both vision-based manipulation and Habitat navigation; probe *which* features
matter and whether one encoder wins everywhere. Cheapest to run (encoders are frozen),
very clean to ablate.

### Spine C — Reproducibility & methodology study
Take 3–4 algorithms, run them properly (many seeds, `rliable` statistics), and quantify
how fragile published RL results are across both domains. Less "new idea," very strong
signal of maturity. Best used as *part* of Spine A, not the whole project.

### Spine D — Zero-shot VLM-guided navigation (stretch / high-risk)
Use a vision-language model as a planner picking frontier subgoals for Habitat ObjectNav,
no policy training. Topical and impressive, but compute- and engineering-heavy. Good as
the *final stretch experiment* inside Spine A, not the foundation.

**Default assumption for the rest of this document: Spine A.** Tell Claude Code which you
chose; everything downstream adapts.

---

## 2. Compute reality check — read before scoping

| Platform | Cost | Verdict on this hardware |
|---|---|---|
| Gymnasium-Robotics (MuJoCo) | Light. SAC+HER on `FetchReach` trains in minutes; `FetchPush`/`PickAndPlace`/`Slide` in hours. Tiny VRAM footprint. | Trivially served. Run many seeds in parallel on a single MIG slice. |
| Habitat — PointNav (depth) | Moderate. Decent SPL reachable in tens of millions of steps, ~1–2 days on one A100. | Easily feasible on one full A100. Start here. |
| Habitat — ImageNav | Moderate–heavy. | Feasible on one full A100. |
| Habitat — ObjectNav from scratch | Heavy. Hundreds of millions of frames; literature results used many GPUs. | *Possible* on one A100 but slow (days–weeks). The blocker here is the shared/unscheduled box, not raw FLOPs — see below. |

### 2.1 Your hardware — shared DGX Station A100

The provided `nvidia-smi` shows a **shared DGX Station A100**: four A100-SXM4-**80GB** cards
usable for compute (the "DGX Display" GPU is a small display adapter — ignore it for
training). Two cards are full 80GB A100s; two are **MIG-partitioned** into `3g.40gb`
slices. There is **no job scheduler** — you select a device by UUID with
`CUDA_VISIBLE_DEVICES=GPU-<uuid> python ...` and take whatever is idle.

This is genuinely strong hardware. The limiting factor is **not compute** — it is that the
box is **shared and unscheduled**. Consequences for the project:

- **Manipulation track:** trivial here. MuJoCo RL barely touches VRAM; run a full
  multi-seed sweep on a single `3g.40gb` MIG slice.
- **Navigation (PointNav / ImageNav):** a single full A100 (a non-MIG card, when idle) is
  more than enough. Habitat-Sim runs thousands of env-steps/sec on one GPU; single-GPU
  DD-PPO reaches good PointNav SPL in ~1–2 days. **No multi-node needed.** Multi-GPU
  DD-PPO is possible *opportunistically* if 2–3 full cards happen to be free at once, but
  do not design experiments that depend on it.
- **ObjectNav from scratch:** capable, but a multi-day run on a shared box you do not
  control is a real risk (someone needs the card; a reboot). Default to **evaluation with
  public checkpoints, a shorter/curriculum ObjectNav, or the zero-shot VLM route** unless
  you can reliably hold a full card for a week+. If you do train it, make it bullet-proof
  resumable.

### 2.2 Operational discipline on a shared, unscheduled box

Bake these into the repo from day one — they are correctness and good-citizenship features:

- **Device selection by UUID, never by index.** `train.py` and the `Makefile` take a
  `GPU=<uuid>` argument and set `CUDA_VISIBLE_DEVICES`; inside the process the device is
  just `cuda:0`. MIG devices have no stable integer index across reboots — UUIDs are
  stable. Check `nvidia-smi` for an idle device before every launch.
- **Every long run goes in `tmux`/`screen`** (or `nohup`) so an SSH disconnect does not
  kill it.
- **Every training run must be checkpoint-resumable** from its last checkpoint. If you
  must free a card or the box reboots, you *resume*, not restart. This is also a CI test.
- **Be a good neighbour:** light work (manipulation, debugging, evaluation,
  frozen-encoder inference) on MIG slices; take a full card only for heavy navigation
  training and only when it is idle; do not squat on a card you are not using.

### 2.3 Habitat setup gotchas (these are real and will eat time)

- **MIG + rendering caveat.** Habitat-Sim does GPU rendering (EGL/Magnum). MIG slices are
  built for compute isolation and have historically limited graphics/OpenGL support, so
  Habitat-Sim rendering **may not work on a `3g.40gb` MIG slice**. Verify this on a MIG
  device in Phase 0; if it fails, run Habitat only on the full (non-MIG) cards. Pure-CUDA
  work — MuJoCo, frozen-encoder inference, policy training — is unaffected by MIG.
- Linux only; conda install strongly recommended; headless/EGL rendering on servers.
- Habitat-Sim and Habitat-Lab have a dependency stack that conflicts with the
  Gymnasium-Robotics stack — **use two separate conda environments** (Section 4).
- Scene datasets are gated. **HM3D** is the most accessible for academic use; you must
  accept terms and download via the official downloader. MP3D/Gibson are also gated. Budget
  a day for data access and downloads. Never redistribute these datasets in your repo.

---

## 3. System architecture — the `Roborto` repository

A research monorepo with one installable package, configuration-driven experiments, and a
strict separation between *code*, *configs*, *results*, and *reports*.

```
roborto/
├── README.md                  # what it is, headline results, repro quickstart
├── CLAUDE.md                  # operating instructions for Claude Code (Section 8)
├── LICENSE                    # MIT
├── pyproject.toml             # package metadata + core dependencies
├── environment-gymrobotics.yml# conda env for the manipulation stack
├── environment-habitat.yml    # conda env for the Habitat stack (separate on purpose)
├── requirements.lock          # pinned, exact versions (per env)
├── Makefile                   # single entry point for all common commands
├── .pre-commit-config.yaml    # ruff + black + isort + basic hygiene
├── .github/workflows/ci.yml   # lint + tests + smoke run on every push
├── .gitignore                 # ignores results/, data/, wandb/, *.ckpt
│
├── docs/
│   ├── PLAN.md                # this file
│   ├── setup.md               # exact install steps for both environments
│   ├── reproduce.md           # one command per published figure
│   ├── lab_notebook.md         # dated, append-only log of every experiment + finding
│   ├── experiments/           # one design note per experiment, written BEFORE coding
│   │   └── exp001_fetch_sac_her.md
│   └── adr/                   # short architecture decision records
│
├── configs/                   # Hydra configs — ALL hyperparameters live here
│   ├── config.yaml            # top-level defaults
│   ├── env/                   # one file per environment
│   ├── algo/                  # one file per algorithm
│   ├── encoder/               # frozen-encoder choices (for Spine A/B)
│   ├── experiment/            # named experiments compose the above
│   └── sweep/                 # hyperparameter sweep definitions
│
├── roborto/                    # the installable Python package
│   ├── envs/
│   │   ├── gym_robotics/       # wrappers, registration, goal-env helpers
│   │   └── habitat/            # task configs, sensor setup, wrappers
│   ├── algos/                  # training algorithms (mostly thin wrappers — see §3.2)
│   ├── models/
│   │   ├── encoders/           # frozen FM encoders + trained-from-scratch baselines
│   │   └── policies/           # policy/value network heads
│   ├── buffers/                # replay buffer + HER relabeling (your own, tested code)
│   ├── training/               # trainer loop, checkpointing, resumption
│   ├── evaluation/             # eval protocols + metrics per platform
│   ├── analysis/               # rliable wrappers, plotting, results-table generation
│   └── utils/                  # seeding, logging, config, determinism
│
├── scripts/
│   ├── train.py               # `python scripts/train.py experiment=exp001`
│   ├── evaluate.py
│   ├── sweep.py
│   ├── download_data.py        # Habitat scene/episode datasets
│   └── make_figures.py         # regenerates every figure in every report
│
├── tests/                      # pytest — covers the non-trivial logic
├── notebooks/                  # exploratory ONLY; never load-bearing
│
├── reports/                    # the arXiv technical reports (LaTeX)
│   └── report-01-baselines/
│       ├── main.tex
│       ├── figures/            # generated by scripts/make_figures.py, not by hand
│       └── refs.bib
│
└── results/                    # GITIGNORED. W&B is the source of truth.
                                # Only small summary CSVs get committed (to results_summary/).
```

### 3.1 Design principles (non-negotiable)

- **Config-driven.** No hyperparameter is ever hardcoded. An experiment is fully described
  by a Hydra config file under `configs/experiment/`. Reproducing a run = re-running its
  config.
- **Seeded and logged.** Every run sets and records its seed; every run logs the exact git
  commit, full resolved config, and package versions to W&B.
- **One command per figure.** `scripts/make_figures.py` regenerates every plot in every
  report from logged data. No figure is ever produced by hand.
- **Results are not code.** `results/` is gitignored. The truth lives in W&B. Only small,
  human-readable summary CSVs are committed.
- **Two environments, one package.** The `roborto` package installs into both conda envs;
  platform-specific code is isolated under `envs/gym_robotics/` and `envs/habitat/` and
  imported lazily so neither env needs the other's dependencies.

### 3.2 Build vs. reuse — be honest about what you implement

Reimplementing every algorithm is a time sink and a correctness risk. The defensible
split for a solo project:

- **Reuse trusted libraries for baselines:** Stable-Baselines3 + sb3-contrib for SAC/HER
  on Gymnasium-Robotics; Habitat-Lab's native DD-PPO for navigation. These are the
  workhorses; you wrap them, you do not rewrite them.
- **Implement yourself (with tests) the parts the research is *about*:** the HER relabeling
  strategy, any new exploration bonus, the frozen-encoder integration, custom reward or
  curriculum logic. This is where your contribution lives and where reviewers will look.
- **Keep one minimal reference implementation** (CleanRL-style, single-file) of your core
  algorithm so you — and readers — can verify the library is doing what you think.

State this split explicitly in the README. "I used SB3 for baselines and implemented the
relabeling study myself" is a *strength*; pretending you wrote everything is fragile.

---

## 4. Environment & dependency setup

- **Two conda environments**, both installing the `roborto` package in editable mode:
  - `roborto-gym` — Python 3.11, `gymnasium`, `gymnasium-robotics` (~1.2.x), `mujoco`,
    `stable-baselines3`, `sb3-contrib`, `torch`.
  - `roborto-hab` — Python per Habitat's current requirement, `habitat-sim` +
    `habitat-lab` (~0.3.x) installed via the official conda channel, `torch`.
- **Pin everything.** After each env works, freeze exact versions into `requirements.lock`.
  Reproducibility depends on this.
- **`docs/setup.md`** contains the exact, tested command sequence for each env, including
  the Habitat headless/EGL flags and the scene-dataset download steps.
- **Always check current released versions** when scaffolding — do not trust version
  numbers in this document; they drift.

---

## 5. Experiment lifecycle

Every experiment follows the same pipeline. This is what makes the repo trustworthy.

1. **Design note** — before any code, write `docs/experiments/expNNN_*.md`: hypothesis,
   what you will run, baselines, metrics, success/failure criteria, predicted outcome.
2. **Implement** — code + Hydra config + unit tests for any new non-trivial logic.
3. **Smoke run** — a tiny run (a few thousand steps) that exercises the whole path and
   asserts shapes, logging, and checkpointing work. This is also a CI target.
4. **Correctness checks** — sanity baselines (random policy, a known-good config that
   should reproduce a published number). A method only "works"/"fails" once the baseline
   reproduces.
5. **Full runs** — **≥5 seeds** (8–10 if compute allows). Launch via `scripts/train.py`.
6. **Evaluate** — fixed held-out evaluation episodes / scenes; never evaluate on training
   data; metrics in Section 5.1.
7. **Analyze** — aggregate with `rliable` (IQM + stratified bootstrap confidence
   intervals, performance profiles). Generate figures via `make_figures.py`.
8. **Record** — append a dated entry to `docs/lab_notebook.md` and a row to the results
   summary table. Honest one-paragraph interpretation, including "this failed because…".

**Definition of done for one experiment:** design note merged → code + tests merged → ≥5
seeds logged to W&B → figures regenerable by script → results-table row added → notebook
entry written.

### 5.1 Metrics & evaluation protocol

- **Gymnasium-Robotics** (goal-conditioned): success rate (achieved-goal within
  threshold), episodic return, and **sample efficiency** (env steps to reach X% success).
  Report learning curves, not just final numbers.
- **Habitat navigation:** Success, **SPL**, SoftSPL, distance-to-goal, collision count.
  Use the standard train/val/test scene splits; report on val/test only.
- **Everywhere:** multi-seed aggregates with confidence intervals. A single-seed number is
  not a result.

---

## 6. The arXiv technical reports

### 6.1 How many, and the honest trade-off

Submitting many thin reports looks worse than one or two solid ones — arXiv is a public,
permanent record and admissions readers can tell padding from substance. Recommended:

- **Report 1 — "Roborto: reproducible baselines and infrastructure for embodied RL."**
  The infrastructure, the reproduced baselines on both platforms, and the
  reproducibility/methodology findings. Establishes credibility. ~6–10 pages.
- **Report 2 — your contribution.** The chosen spine's novel experiments, ablations, and —
  explicitly — the negative results. ~8–14 pages.
- *(Optional)* **Report 3** only if the cross-cutting synthesis is substantial enough to
  stand alone; otherwise fold it into Report 2.

It is also completely fine to ship **one comprehensive technical report** plus the repo.
Decide once Report 1's content is concrete.

### 6.2 Report structure (each report)

Abstract · Introduction (the question, why it matters, contributions) · Background &
related work · Methods (environments, algorithms, your contribution) · Experimental setup
(seeds, compute, hyperparameters — enough to reproduce) · Results (with confidence
intervals) · **Failure analysis & limitations** (a real section, not a footnote) ·
Conclusion · Appendix (full hyperparameters, extra plots) · Reproducibility statement
(link to the exact repo commit).

### 6.3 Writing standards

- All claims supported by multi-seed evidence with error bars.
- Use `rliable` for RL aggregation — it is the current expected standard and signals you
  know the field's reproducibility literature.
- Every figure regenerated by `scripts/make_figures.py`; no hand-made plots.
- The reproducibility statement links a specific git commit/tag, not just "the repo."
- Use a clean LaTeX preprint template (a standard arXiv/NeurIPS-style preprint style).

### 6.4 arXiv logistics — check early

First-time submitters to arXiv's CS categories (cs.RO, cs.LG, cs.AI) often need an
**endorsement** before they can post, and submissions pass through moderation. This can
take time, so verify the current arXiv policy and line up an endorser (an advisor,
collaborator, or mentor who already publishes in that category) **before** Report 1 is
finished — do not discover this the week of a deadline.

---

## 7. Milestones & timeline

A ~14-week core plan. Compress or extend to fit your application deadline (note that CS
PhD deadlines cluster around December).

| Phase | Weeks | Goal | Exit criterion |
|---|---|---|---|
| 0 — Foundations | 1–2 | Repo scaffold, both conda envs, CI, logging, smoke runs on each platform | One real (tiny) training run end-to-end on Fetch *and* Habitat, logged to W&B |
| 1 — Manipulation baselines | 3–4 | Reproduce SAC+HER on Fetch tasks, multi-seed | Published-level success rate matched on ≥2 Fetch tasks, with CIs |
| 2 — Navigation baselines | 4–6 | Reproduce DD-PPO PointNav in Habitat | A defensible SPL on a fixed scene set, with CIs |
| 3 — Contribution | 6–10 | The chosen spine's novel experiments + ablations | Core hypothesis tested across ≥5 seeds; figures generated by script |
| 4 — Synthesis / stretch | 10–12 | Cross-domain analysis or the high-risk idea | A clear answer (positive or negative) with evidence |
| 5 — Write & polish | 12–14 | Finalize reports, polish repo, submit to arXiv | Reports on arXiv; repo README + reproduce.md complete |

Treat Phase 2 as the schedule risk (Habitat install + data access + training time). If it
slips, the manipulation track alone still yields a complete, honest report.

---

## 8. Working with Claude Code

### 8.1 Division of labor — important for your actual goal

Claude Code is for **engineering throughput**: scaffolding, wiring configs, writing tests,
plumbing logging, fixing bugs. The things a PhD committee actually evaluates — the research
question, experiment design choices, interpretation of results, and the writing of the
reports — should be **substantially your own thinking**. You will need to defend this work
in interviews. Keep a personal research journal separate from the auto-generated logs, and
be ready to discuss every design decision. Use AI assistance the way you would use any
tool, and follow the AI-use disclosure norms of the programs you apply to.

### 8.2 `CLAUDE.md` — copy this into the repo root

```markdown
# Roborto — Claude Code operating instructions

## Project
Embodied-RL research monorepo. Manipulation via Gymnasium-Robotics (MuJoCo),
navigation via AI Habitat. Goal: a reproducible repo + arXiv technical reports.
The full spec is docs/PLAN.md — read it before any non-trivial change.

## Hard rules
- NEVER invent, estimate, or hand-edit experimental results, metrics, or plots.
  Numbers come only from real runs logged to W&B. Plots come only from
  scripts/make_figures.py.
- Before implementing a new experiment, write its design note in
  docs/experiments/expNNN_*.md and wait for my review.
- ALL hyperparameters live in Hydra configs under configs/. Never hardcode them.
- Every change: run `make lint` and `make test`. New experiment code must pass
  `make smoke` (a tiny end-to-end run) before you call it done.
- Set and log seeds; log git commit + resolved config + package versions on
  every run.
- results/ , data/ , and wandb/ are gitignored. Only small summary CSVs in
  results_summary/ get committed.
- Do NOT launch long GPU training runs. Set them up, estimate their cost, and
  stop — I launch them.
- Keep PRs small and focused. Match existing style. Update docs/ when behavior
  changes.

## Environments & GPUs
Two conda envs: roborto-gym (manipulation) and roborto-hab (Habitat). They have
conflicting dependencies — never merge them. See docs/setup.md.
Shared, unscheduled DGX box: scripts/train.py and the Makefile take GPU=<uuid>
and export CUDA_VISIBLE_DEVICES; never hardcode device indices (MIG UUIDs are
the stable identifier). Every training run must be checkpoint-resumable. Do not
assume more than one GPU is free; do not write multi-GPU code unless asked.

## Reproducing baselines
When reproducing a published result, cite the source and the target number in
the design note, and treat "matched within CI" as the success criterion.

## Workflow for any new experiment
design note -> implement + tests -> smoke run -> correctness/sanity baseline
-> (I launch full runs) -> evaluate -> rliable analysis -> figures -> lab
notebook entry + results-table row.
```

### 8.3 How to feed work to Claude Code

- Work **phase by phase** from Section 7; do not ask for the whole repo at once.
- One task ≈ one small PR (e.g., "scaffold the package + Makefile + CI", then "add the
  Gymnasium-Robotics env wrappers + smoke test", then "add SAC+HER baseline config").
- For each experiment, have Claude Code draft the design note *first*; you edit it; only
  then implement.
- Make Claude Code write the **tests** for buffer logic, HER relabeling, reward functions,
  and config resolution — these are exactly where silent bugs turn real results into
  garbage.

### 8.4 `Makefile` targets to define early

`make setup-gym` · `make setup-hab` · `make lint` · `make test` · `make smoke` ·
`make train EXP=expNNN GPU=<uuid>` · `make eval EXP=expNNN GPU=<uuid>` ·
`make figures` · `make report R=01`.

`train`/`eval`/`smoke` take `GPU=<uuid>` and export `CUDA_VISIBLE_DEVICES` before
launching. If `GPU` is unset, the target should **error loudly** rather than default to a
device — this prevents a run from silently landing on a card someone else is using. Pair
this with a one-line helper (or just `nvidia-smi`) to find an idle device first.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Habitat install / data access eats weeks | Start Phase 0 Habitat setup in week 1, in parallel with manipulation; HM3D first; the manipulation track is a complete fallback report on its own. |
| Shared, unscheduled GPU box; a long run gets interrupted | All runs checkpoint-resumable; launch in `tmux`; MIG slices for light work, full cards only when idle; verify MIG+Habitat rendering in Phase 0; prefer PointNav/ImageNav over ObjectNav-from-scratch. |
| A "failure" is actually a bug | Mandatory sanity baselines + reproduced published numbers before trusting any negative result. |
| Scope creep across four spines | Commit to one spine after the lit review; archive the others as "future work." |
| arXiv endorsement blocks submission | Check policy and line up an endorser in Phase 0–1, not Phase 5. |
| Repo looks AI-generated / shallow | Clean, meaningful commit history; you write the reports and notebook; small reviewed PRs. |

---

## 10. Quality bar — definition of done for the whole project

- Anyone can clone the repo, follow `docs/setup.md` and `docs/reproduce.md`, and
  regenerate every headline figure.
- Every result has ≥5 seeds and confidence intervals.
- Tests cover the non-trivial logic; CI is green.
- `docs/lab_notebook.md` tells the honest story, including dead ends.
- 1–2 technical reports on arXiv, each with a real failure-analysis section and a
  reproducibility statement linking an exact commit.
- The README opens with the research question and the headline result (or honest
  negative result), not a feature list.

---

## Appendix — candidate experiment seeds

A menu to draw Phase 3/4 experiments from once a spine is chosen. **Not claimed to be
novel** — verify against the literature and position accordingly.

- HER relabeling-strategy comparison, plus a difficulty-aware future-goal sampling variant
  (Fetch Push / PickAndPlace / Slide).
- HER combined with an intrinsic-motivation bonus (e.g. RND or ICM) on the hardest sparse
  Fetch tasks.
- Curriculum over goal distance in Point/Ant Maze and in Fetch.
- Frozen foundation-model encoders (DINOv2, CLIP, VC-1, R3M, ImageNet ResNet) vs
  trained-from-scratch encoders for Habitat PointNav/ImageNav — systematic comparison plus
  representation probing.
- Memory-architecture study (LSTM vs GRU vs Transformer) for navigation under partial
  observability.
- Self-supervised auxiliary tasks for navigation sample efficiency.
- Sim-to-sim robustness: domain randomization on Fetch dynamics; visual-perturbation
  (lighting/texture) robustness in Habitat.
- Zero-shot VLM-as-planner picking frontier subgoals for ObjectNav (stretch; no training).
