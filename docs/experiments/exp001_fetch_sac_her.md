# exp001 - Reproduce SAC+HER on Fetch

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp001_fetch_sac_her.yaml`

## Hypothesis
A correctly configured SAC + Hindsight Experience Replay agent reaches published-level
success rates on the sparse-reward Fetch tasks. This experiment establishes a trusted
baseline and exercises the full train/eval/analysis pipeline before any novel work.

## Setup
- Env: `FetchPush` (sparse reward, goal-conditioned). Extend to `FetchReach` (sanity)
  and `FetchPickAndPlace` (harder) once the pipeline is verified.
- Algorithm: SAC (Stable-Baselines3) with HER. Use SB3's `HerReplayBuffer` for the
  baseline; the in-repo `roborto.buffers.her.HERRelabeler` is validated against it and
  then used for the relabeling-strategy study that follows.
- Seeds: 8. Compute: a single MIG slice is sufficient - manipulation is light.

## Baselines
Target: the SAC+HER success rates reported in the HER literature
(Andrychowicz et al., 2017) and standard SB3 results. Record the exact numbers being
targeted here before running.

## Metrics & success criteria
Success rate on held-out goals, and environment steps to reach a given success level.
**Success criterion:** final success rate matches the target within a bootstrap
confidence interval on at least two Fetch tasks.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
Sweep the HER relabeling strategy (`configs/sweep/her_relabel_strategy.yaml`) - the
first contribution experiment of the manipulation track.
