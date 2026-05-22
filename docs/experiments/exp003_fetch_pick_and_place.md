# exp003 - SAC+HER Baseline on FetchPickAndPlace

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp003_fetch_pick_and_place.yaml`

## Hypothesis
SAC+HER with the `future` strategy reproduces published-level success rates on
FetchPickAndPlace. This is harder than FetchPush (requires lifting the object), so
lower final success and higher variance across seeds is expected.

## Setup
- Env: FetchPickAndPlace-v4 (sparse, horizon 50)
- Algorithm: SAC + HER, strategy=future, k=4 (identical to exp001)
- Seeds: 8, Steps: 1,000,000 (doubled vs FetchPush — task is harder)
- GPU: full A100 (same as exp001)

## Baselines
Published target: ~50–70% success at 1M steps (Andrychowicz et al. 2017, Table 1).
Record exact number here before running: **[FILL from paper]**.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 1M steps.
Success criterion: within CI of published target on FetchPickAndPlace.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
If baseline confirmed → use as the testbed for exp008 (difficulty-aware HER) and
exp009 (HER + RND), as it is the most informative mid-difficulty task.
