# Technical Paper Notes — Roborto

Companion document for writing the arXiv technical reports.
All numbers are taken directly from the implemented code and configs — do not edit them by hand.
Fill in the **[FILL AFTER RUNS]** sections once W&B results are in.

---

## 1. Research Framing

### Central question (Spine A)
*What makes an embodied agent learn fast and generalise, and does the answer transfer
between manipulation and navigation?*

Two levers under study:

| Platform | Primary lever | Task family |
|---|---|---|
| Gymnasium-Robotics (MuJoCo) | Goal relabeling strategy + exploration | Sparse-reward goal-conditioned manipulation |
| AI Habitat | Frozen visual encoder quality | Visual navigation (PointNav / ImageNav) |

**Cross-cutting synthesis question:** Is "good representation / good exploration" a domain-specific
or domain-general property?

### Claim this paper can support (Phase 1 only)
"A correctly-configured SAC+HER agent matches published success rates on sparse-reward
Fetch tasks, establishing a trusted baseline from which the relabeling-strategy study departs."
This is a reproducibility claim + infrastructure claim, not a novelty claim — state it honestly.

### Positioning in one sentence
"We reproduce the SAC+HER baseline on FetchPush, FetchPickAndPlace, and FetchSlide with
proper multi-seed statistics, providing an auditable starting point for the relabeling-strategy
ablation that follows."

---

## 2. Environment

### 2.1 FetchPush-v4 (primary baseline task)

| Property | Value | Source |
|---|---|---|
| Library | `gymnasium-robotics` 1.2.3 | `environment-gymrobotics.yml` |
| Gymnasium ID | `FetchPush-v4` | `configs/env/fetch_push.yaml` |
| Physics engine | MuJoCo 3.8.1 | `environment-gymrobotics.yml` |
| Observation | dict: `observation` (25-d), `achieved_goal` (3-d), `desired_goal` (3-d) | verified via `env.observation_space` |
| Observation total | 31-d (25 + 3 + 3) | — |
| Action space | Box(−1, 1, shape=(4,), dtype=float32) — 3-d Cartesian Δ-EE + gripper | verified via `env.action_space` |
| Reward | Sparse: 0 if ‖achieved − desired‖₂ ≤ 0.05, else −1 | `reward_type: sparse` |
| Episode horizon | 50 steps | `max_episode_steps: 50` |
| Task | Push a puck to a target position on a table | — |
| Goal space | 3-d target puck position | Gymnasium-Robotics docs |

**Also planned (same config family, once FetchPush baseline is confirmed):**
- `FetchReach-v4` — sanity check (easiest task, should reach ~100% quickly)
- `FetchPickAndPlace-v4` — harder (lift + place)
- `FetchSlide-v4` — hardest (no direct contact possible)

### 2.2 Goal-conditioned RL formulation
Fetch tasks implement the `GoalEnv` interface (Plappert et al., 2018). Each observation is
a dict with three components:
- **observation** s: robot proprioception (joint angles, velocities, object position/velocity)
- **achieved_goal** g_a: what the agent has accomplished (current object position)
- **desired_goal** g_d: what the agent must achieve (sampled per episode)

Success criterion: ‖g_a − g_d‖₂ ≤ δ with δ = 0.05 m.
Sparse reward: r(g_a, g_d) = −1{‖g_a − g_d‖₂ > δ}, i.e. 0 on success, −1 otherwise.

---

## 3. Algorithm

### 3.1 Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is an off-policy maximum-entropy actor-critic algorithm.
It maximises J(π) = E[∑ γᵗ (r_t + α H(π(·|s_t)))] where α is the entropy temperature
(auto-tuned). Compared to DDPG it is more stable and sample-efficient on continuous
control tasks with sparse rewards.

**Implementation:** Stable-Baselines3 2.8.0, `SAC` class with `MultiInputPolicy`.
`MultiInputPolicy` uses a `CombinedExtractor` feature extractor (flat MLP per key,
outputs concatenated) feeding into a 2-layer MLP policy/critic.

Default network architecture (SB3 2.8.0, `net_arch=None`):
- Feature extractor output → 256-unit hidden layer → 256-unit hidden layer → action / Q-value

### 3.2 Hindsight Experience Replay (HER)

HER (Andrychowicz et al., 2017) addresses sparse rewards in goal-conditioned environments
by relabeling past transitions with goals that were actually achieved. For a transition
(s_t, a_t, s_{t+1}, g_d) that yielded reward −1, HER substitutes g_d with g_a(s_{t+k})
for some k, recomputes the reward (now 0 = success), and adds this synthetic transition
to the replay buffer. This provides a dense learning signal even when the policy almost
never reaches the true desired goal.

**Relabeling strategies (Andrychowicz et al., 2017 notation):**

| Strategy | Goal selection | SB3 name |
|---|---|---|
| `future` | Uniform from {t+1, …, T} — future states in the same episode | `future` |
| `final` | Last state of the episode | `final` |
| `episode` | Uniform from {0, …, T} — any state in the same episode | `episode` |
| `random` | Uniform from the entire replay buffer | `random` |

**Baseline uses `future` with k=4** (i.e., 4 relabeled transitions per real transition),
consistent with the recommended setting from Andrychowicz et al. The relabeling-strategy
comparison (Phase 1 contribution) will sweep all four strategies.

**In-repo implementation:** `roborto/buffers/her.py` — `HERRelabeler` class implements
all four strategies with a vectorised reward recomputation interface compatible with
`GoalEnv.compute_reward`. The baseline uses SB3's `HerReplayBuffer` directly; the
contribution experiment swaps in `HERRelabeler` to expose strategy-level control.

### 3.3 Exact hyperparameters — exp001 baseline

| Parameter | Value | Config key |
|---|---|---|
| Discount factor γ | 0.98 | `algo.gamma` |
| Learning rate | 1 × 10⁻³ | `algo.learning_rate` |
| Batch size | 256 | `algo.batch_size` |
| Replay buffer size | 1 × 10⁶ | `algo.buffer_size` |
| Soft update τ | 0.005 | `algo.tau` |
| Learning starts | 1,000 steps | `algo.learning_starts` |
| HER strategy | `future` | `algo.her.strategy` |
| HER relabeling ratio k | 4 | `algo.her.n_relabel` |
| Policy / critic network | [256, 256] MLP | SB3 default (`net_arch=None`) |
| Entropy temperature α | auto-tuned | SB3 default |
| Actor / critic optimiser | Adam | SB3 default |
| Total training steps | 500,000 | `trainer.total_steps` |
| Number of seeds | 8 | `trainer.num_seeds` |
| Eval interval | 25,000 steps | `trainer.eval_interval` |
| Eval episodes per checkpoint | 100 | `trainer.eval_episodes` |

**Reproduce with one command:**
```bash
# single seed (replace UUID with an idle full A100 from nvidia-smi -L)
make train EXP=exp001_fetch_sac_her GPU=GPU-5ac08ef4-9737-c367-bf80-200f205f6014

# all 8 seeds sequentially
make run-all GPU=GPU-5ac08ef4-9737-c367-bf80-200f205f6014 SEEDS=0,1,2,3,4,5,6,7
```

---

## 4. Evaluation Protocol

### 4.1 Metrics

**Primary metric: success rate**
Fraction of evaluation episodes in which the agent achieves the goal (‖g_a − g_d‖₂ ≤ 0.05)
at any point before episode termination. Implemented in `roborto/evaluation/metrics.py`.

**Secondary metric: mean episodic return**
Mean undiscounted sum of rewards over evaluation episodes (range [−50, 0] for FetchPush
with horizon 50 and sparse reward −1).

**Sample efficiency:** Steps-to-X% success (read off learning curves).

### 4.2 Evaluation procedure
- Deterministic policy (`model.predict(obs, deterministic=True)`)
- 100 fresh episodes per evaluation checkpoint (seeds offset from training seed by +1000)
- Evaluation environment is a separate instance from the training environment
- Evaluation happens every 25,000 training steps → 20 data points over a 500k run

### 4.3 Multi-seed aggregation (use rliable)
Raw per-seed learning curves are logged to W&B. For the paper:

1. Extract all 8 seeds' learning curves at each eval checkpoint.
2. Report **IQM** (Interquartile Mean) ± **stratified bootstrap 95% CI** using `rliable`
   (Agarwal et al., 2021). IQM is preferred over mean for RL because it is robust to
   outlier seeds.
3. Plot as a shaded learning curve (IQM ± CI vs training steps).
4. Final performance = IQM of success rate at the last checkpoint (step 500,000).

```python
# Sketch — fill in after pulling run data from W&B
import rliable.library as rly
import rliable.metrics as metrics

# scores: np.ndarray of shape (n_seeds, n_checkpoints)
iqm_fn = lambda scores: np.array([metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])])
iqm, iqm_ci = rly.get_interval_estimates({'exp001': scores}, iqm_fn, reps=50_000)
```

Figures are generated by `scripts/make_figures.py` — never hand-made.

### 4.4 Target numbers (baselines to match)
Fill in before finalising the paper. Target sources:

| Task | Algorithm | Target success rate | Source |
|---|---|---|---|
| FetchPush | SAC+HER (future, k=4) | **[FILL]** — ~80–90% | Andrychowicz et al. 2017, Table 1 |
| FetchReach | SAC+HER | **[FILL]** — ~100% | — |
| FetchPickAndPlace | SAC+HER | **[FILL]** — ~50–70% | — |
| FetchSlide | SAC+HER | **[FILL]** — ~30–60% | — |

**Success criterion (from PLAN.md):** final success rate matches the target within the
bootstrap confidence interval on at least 2 tasks.

---

## 5. Infrastructure

### 5.1 Repository structure
```
roborto/
├── roborto/              # installable package (pip install -e .[gym])
│   ├── buffers/her.py    # HERRelabeler — 4 strategies, fully tested
│   ├── buffers/replay_buffer.py  # GoalReplayBuffer — circular, numpy
│   ├── envs/gym_robotics/make.py # make_env(cfg, seed) — GoalEnv builder
│   ├── algos/sac.py      # build_sac(cfg, env, device) — SB3 SAC + HER
│   ├── training/trainer.py # SACHERTrainer — smoke, resume, eval, checkpoint
│   ├── evaluation/metrics.py # success_rate(), spl()
│   └── utils/            # seeding, device, logging, provenance
├── configs/              # Hydra config tree — ALL hyperparameters live here
│   ├── config.yaml       # base defaults
│   ├── env/fetch_push.yaml etc.
│   ├── algo/her.yaml, sac.yaml, ddppo.yaml
│   └── experiment/exp001_fetch_sac_her.yaml
├── scripts/
│   ├── train.py          # entry point (Hydra @main)
│   ├── launch_seeds.py   # parallel/pooled multi-seed launcher
│   └── run_all_exps.py   # loop all manipulation experiments
├── tests/                # 24 unit + smoke tests, all green
└── docs/                 # this file, PLAN.md, setup.md, lab_notebook.md
```

### 5.2 Reproducibility guarantees
Every training run automatically records (to W&B and to `<run_dir>/provenance.json`):
- **Git SHA** of the commit that produced it (`git_sha` in W&B summary)
- **Full resolved Hydra config** (all hyperparameters, no hidden defaults)
- **Exact package versions** of every installed package (via `importlib.metadata`)
- **Global seed** set for Python `random`, NumPy, and PyTorch before any stochastic operation

Config is also saved verbatim to `<run_dir>/config.yaml`.

### 5.3 Seeding
`roborto.utils.seeding.set_global_seed(seed)` sets:
- `os.environ["PYTHONHASHSEED"]`
- `random.seed`
- `numpy.random.seed`
- `torch.manual_seed` + `torch.cuda.manual_seed_all`
- `cudnn.deterministic = True`, `cudnn.benchmark = False`

Training environment: seeded via `env.reset(seed=cfg.seed)`.
Evaluation environment: seeded via `env.reset(seed=cfg.seed + 1000)` (separate instance,
never reused across training and evaluation).

### 5.4 Checkpointing and resumption
Every eval cycle writes:
- `<run_dir>/checkpoint/model.zip` — SB3 model (policy + replay buffer)
- `<run_dir>/checkpoint/step.json` — number of environment steps completed

On relaunch with `trainer.resume=true` (the default), the trainer detects the checkpoint
and resumes from the recorded step. This is critical on a shared, unscheduled GPU box.

### 5.5 Compute
| Component | Hardware | Notes |
|---|---|---|
| Manipulation training | Full NVIDIA A100-SXM4-80GB | MIG slices caused CUDA init errors with SB3 — use full cards only |
| GPU selection | `CUDA_VISIBLE_DEVICES=<UUID>` | Set by Makefile; UUIDs are stable across reboots |
| Wall time estimate | ~6 hrs / seed at 500k steps | Based on empirical rate ~12 hrs/1M steps on A100 |
| Total for 8 seeds (sequential) | ~48 hrs | On 1 GPU; use `make run-all` |
| VRAM | < 2 GB | SAC on FetchPush is tiny |

**Verified GPU:** `GPU-5ac08ef4-9737-c367-bf80-200f205f6014` (NVIDIA A100-SXM4-80GB).
Smoke run confirmed end-to-end on 2026-05-21.

---

## 6. Logging and W&B

**Project:** `https://wandb.ai/sohampahari-research-valeo/roborto`

**Metrics logged at every eval checkpoint (every 25k steps):**
- `eval/success_rate` — primary metric
- `eval/mean_return` — secondary

**Run naming convention:** `<experiment_name>/seed<N>` (e.g., `exp001_sac_her_fetch_push/seed3`)

**To view all seeds overlaid:** W&B → project → Runs → Group by `experiment_name`.

**To pull data for `rliable`:**
```python
import wandb
api = wandb.Api()
runs = api.runs("sohampahari-research-valeo/roborto",
                filters={"config.experiment_name": "exp001_sac_her_fetch_push"})
# extract eval/success_rate history per run, stack into (n_seeds, n_checkpoints)
```

---

## 7. Paper Section Drafts

### 7.1 Abstract skeleton
> We present Roborto, a reproducible research framework for embodied reinforcement
> learning across manipulation (Gymnasium-Robotics/MuJoCo) and navigation (AI Habitat).
> We reproduce the SAC+HER baseline on sparse-reward Fetch tasks with proper multi-seed
> statistics (N=8 seeds, IQM ± 95% CI), achieving [FILL: X%] success on FetchPush and
> [FILL: Y%] on FetchPickAndPlace after 500k environment steps — within the confidence
> interval of [FILL: target from paper]. We then study [...].

### 7.2 Methods — Environments paragraph
> We evaluate on the Fetch manipulation suite from Gymnasium-Robotics (Plappert et al.,
> 2018), implemented with MuJoCo 3.8.1. Each task is a sparse-reward goal-conditioned
> environment (`GoalEnv`): the agent receives observation s ∈ ℝ²⁵, achieved goal
> g_a ∈ ℝ³, and desired goal g_d ∈ ℝ³, with binary reward r = −𝟙{‖g_a − g_d‖₂ > 0.05}.
> The action space is continuous in ℝ⁴ (3D end-effector displacement + gripper width).
> Episode horizon is 50 steps. We report results on FetchPush (primary), FetchReach
> (sanity), FetchPickAndPlace, and FetchSlide.

### 7.3 Methods — Algorithm paragraph
> We train with Soft Actor-Critic (SAC; Haarnoja et al., 2018) combined with Hindsight
> Experience Replay (HER; Andrychowicz et al., 2017), using the `future` relabeling
> strategy with k=4 virtual transitions per real transition. The policy and critic are
> 2-layer MLPs with 256 hidden units. We use a learning rate of 10⁻³, discount γ=0.98,
> soft update τ=0.005, batch size 256, and a replay buffer of 10⁶ transitions. The
> entropy temperature α is auto-tuned. We use Stable-Baselines3 2.8.0 as the SAC
> backbone, with our own `HERRelabeler` implementation for the ablation study.

### 7.4 Methods — Training setup paragraph
> Each agent is trained for 500,000 environment steps. We run N=8 independent seeds per
> experiment. We evaluate the deterministic policy on 100 fresh episodes every 25,000
> training steps using a separate, independently-seeded environment. We report
> Interquartile Mean (IQM) and stratified bootstrap 95% confidence intervals following
> Agarwal et al. (2021) using the `rliable` library. Every run records the exact Git
> commit, full resolved configuration, and package versions for reproducibility.
> Training was performed on an NVIDIA A100-SXM4-80GB GPU.

### 7.5 Results table template (fill after runs)

| Task | Steps | Success Rate (IQM ± 95% CI) | Target | Matched? |
|---|---|---|---|---|
| FetchReach | 500k | **[FILL]** | ~100% | **[FILL]** |
| FetchPush | 500k | **[FILL]** | ~80–90% | **[FILL]** |
| FetchPickAndPlace | 500k | **[FILL]** | ~50–70% | **[FILL]** |
| FetchSlide | 500k | **[FILL]** | ~30–60% | **[FILL]** |

### 7.6 Failure analysis section template
> **Failure analysis.** [FILL — be honest about any seeds that failed to learn,
> any task where success rate is below target, and the likely cause.]
> For example: "FetchSlide showed high variance across seeds (CI width: [FILL]),
> consistent with the difficulty of the task under sparse reward and fixed
> learning rate. We did not tune hyperparameters per task."

---

## 8. Key Citations to Include

| Work | Why cite | BibTeX key |
|---|---|---|
| Andrychowicz et al. 2017 | HER algorithm, target numbers | `andrychowicz2017her` |
| Haarnoja et al. 2018 | SAC algorithm | `haarnoja2018sac` |
| Plappert et al. 2018 | GoalEnv interface, Fetch tasks | `plappert2018fetch` |
| Raffin et al. 2021 | Stable-Baselines3 | `raffin2021sb3` |
| Agarwal et al. 2021 | rliable, IQM, statistical reporting | `agarwal2021rliable` |
| Todorov et al. 2012 | MuJoCo physics | `todorov2012mujoco` |
| Gymnasium-Robotics | Library reference | Farama Foundation |

---

## 9. Workflow Checklist — Phase 1 → Paper

- [x] Repo scaffold (Phase 0): configs, CI, tests, seeding, logging, provenance
- [x] HER relabeler implemented and unit-tested (`roborto/buffers/her.py`)
- [x] Replay buffer implemented and unit-tested (`roborto/buffers/replay_buffer.py`)
- [x] SACHERTrainer implemented with smoke + resume support
- [x] Smoke run confirmed end-to-end on GPU (`make smoke` — 2026-05-21)
- [x] exp001 config set to 500k steps, 8 seeds
- [ ] Full 8-seed runs complete and logged to W&B
- [ ] Pull W&B data, compute IQM + 95% CI with rliable
- [ ] Generate learning-curve figures with `scripts/make_figures.py`
- [ ] Fill in results table (Section 7.5 above)
- [ ] Write failure analysis section
- [ ] Append results to `docs/lab_notebook.md`
- [ ] Confirm git SHA of results commit for reproducibility statement
- [ ] Extend to FetchPickAndPlace and FetchSlide
- [ ] Write Phase 1 contribution: HER relabeling-strategy sweep (exp002)
- [ ] Draft Report 1 in `reports/report-01-baselines/main.tex`
- [ ] Get arXiv endorser lined up (do not leave this to the last week)
