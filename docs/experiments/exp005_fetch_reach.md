# exp005 - SAC+HER Sanity Check on FetchReach

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp005_fetch_reach.yaml`

## Hypothesis
SAC+HER reaches ~100% success on FetchReach within 200k steps. FetchReach is the
simplest Fetch task (move end-effector to a target position, no object involved).
If this fails, there is a bug in the training pipeline.

## Setup
- Env: FetchReach-v4 (sparse, horizon 50)
- Algorithm: SAC + HER, strategy=future, k=4
- Seeds: 5, Steps: 200,000 (should converge well before this)
- GPU: full A100 or MIG slice (task is trivial)

## Baselines
Published target: ~100% success well before 500k steps (Andrychowicz et al. 2017).
This is a pipeline sanity check, not a scientific claim.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 200k steps.
Success criterion: IQM ≥ 95%. If <90%, there is likely a bug.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs. If this fails, investigate before running any other experiment._

## Next step
If passes → report as sanity baseline in the paper (one line, not a main result).
If fails → stop all other experiments and debug the trainer.
