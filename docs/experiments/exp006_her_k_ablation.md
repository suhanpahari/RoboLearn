# exp006 - HER k (n_relabel) Ablation on FetchPush

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp006_her_k_ablation.yaml`

## Hypothesis
There is a sweet spot for k (number of relabeled transitions per real transition).
Too few (k=1) provides insufficient hindsight signal; too many (k=16) crowds out
real transitions and introduces too much off-policy bias. k=4 (the default) is
expected to be near-optimal per Andrychowicz et al., but the shape of this
sensitivity curve is informative for the paper.

## Setup
- Env: FetchPush-v4 (sparse, horizon 50)
- Algorithm: SAC + HER, strategy=future
- Sweep k ∈ {1, 2, 4, 8, 16}
- Seeds: 5 per k value (25 runs total)
- Steps: 500k per run

## Baselines
Compared against exp001 (k=4) as reference. The k=4 result should match exp001.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 500k steps, plotted vs k.
Success criterion: k=4 is within CI of the optimal k. If k=1 or k=2 match k=4,
then the relabeling ratio is not a critical hyperparameter for this task.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
Use the best k in exp008 and exp009. If performance is flat across k → simplify
and report as a non-finding (also useful — reduces hyperparameter sensitivity concerns).
