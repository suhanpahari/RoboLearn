# exp002 - HER Relabeling Strategy Comparison on FetchPush

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp002_her_strategy_sweep.yaml`
- **Sweep:** `configs/sweep/her_relabel_strategy.yaml`

## Hypothesis
The `future` strategy outperforms `final`, `episode`, and `random` on FetchPush because
it provides goals at varied difficulty levels (near-future vs far-future), striking a
balance between easy and hard relabeled goals. `final` will underperform on longer-horizon
tasks because the terminal achieved-goal is often far from intermediate states.

## Setup
- Env: FetchPush-v4 (sparse, horizon 50)
- Algorithm: SAC + HER, sweeping strategy ∈ {final, future, episode, random}
- All other hyperparameters identical to exp001 (γ=0.98, lr=1e-3, k=4, buffer=1M)
- Seeds: 5 per strategy (20 runs total)
- Steps: 500k per run
- Run via: `python scripts/sweep.py sweep=her_relabel_strategy`

## Baselines
Compared against exp001 (`future`, k=4) as the reference.
Target from Andrychowicz et al. 2017: `future` should be the best or near-best strategy.

## Metrics & success criteria
Primary: success rate (IQM ± 95% CI) at 500k steps, per strategy.
Success: ranking matches Andrychowicz et al. 2017 (future ≥ episode ≥ final > random
or similar ordering). Any significant reordering would be a finding worth reporting.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
If `future` wins clearly → use as the fixed strategy for exp003/004/005.
If strategies are indistinguishable → report null result and move on.
If unexpected ordering → investigate and consider exp008 (difficulty-aware variant).
