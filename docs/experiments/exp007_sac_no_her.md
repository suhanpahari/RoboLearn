# exp007 - SAC without HER on FetchPush (HER Necessity Ablation)

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp007_sac_no_her.yaml`

## Hypothesis
Plain SAC without HER fails to learn on sparse-reward FetchPush within 500k steps
(success rate ≈ 0%). This establishes that HER is necessary, not just beneficial,
for this task and reward setting — motivating the relabeling-strategy study.

## Setup
- Env: FetchPush-v4 (sparse, horizon 50)
- Algorithm: SAC, NO HER (uses standard replay buffer, not HerReplayBuffer)
- Config: algo=sac (not her)
- Seeds: 5, Steps: 500k

## Baselines
Compared against exp001 (SAC+HER, future, k=4). The gap between exp007 and exp001
is the HER contribution.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 500k steps.
Expected: near 0%. If success rate > 10%, the sparse reward is less challenging
than assumed (worth investigating).

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
This result appears in the paper as the lower-bound ablation in the HER comparison
table alongside exp002 (strategy sweep). Together they frame the contribution.
