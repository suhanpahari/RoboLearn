# exp008 - Difficulty-Aware Future-Goal Sampling (Novel Contribution)

- **Phase:** 1   **Track:** manipulation
- **Status:** planned — requires implementation first
- **Config:** `configs/experiment/exp008_her_difficulty_aware.yaml`

## Hypothesis
Weighting future-goal candidates by the displacement ‖g_a(t') − g_a(t)‖₂ between
the achieved goal at the relabeling index t' and the transition's own achieved goal
at t produces a curriculum: easy goals (small displacement) are sampled early,
harder goals (large displacement) later. This should improve sample efficiency on
FetchPickAndPlace relative to uniform future sampling (exp001/exp003 baseline).

## Setup
- Env: FetchPickAndPlace-v4 (chosen because it's hard enough to show a signal)
- Algorithm: SAC + HER with `difficulty_aware` strategy (to implement in `roborto/buffers/her.py`)
- Strategy: sample future index t' with probability ∝ ‖g_a(t') − g_a(t)‖₂ + ε
- Seeds: 8, Steps: 1,000,000
- Compare against: exp003 (uniform `future`, same task and steps)

## Implementation required before running
1. Add `difficulty_aware` to `STRATEGIES` in `roborto/buffers/her.py`
2. Implement weighted sampling in `_sample_goal_indices`
3. Add unit tests in `tests/test_her.py`
4. Pass `make smoke EXP=exp008_her_difficulty_aware GPU=<uuid>`

## Baselines
Primary comparison: exp003 (uniform future, FetchPickAndPlace, 1M steps).
Null hypothesis: no significant difference (uniform future is already near-optimal).

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 1M steps vs exp003.
Success: IQM improvement ≥ 5 percentage points with non-overlapping 95% CIs.
Null: CIs overlap → honest null result, still publishable.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs. If null: report as negative result with analysis of why
uniform future sampling may already be near-optimal._

## Next step
If positive → exp008 is the paper's headline contribution.
If null → fold into the strategy comparison as an ablation table row.
