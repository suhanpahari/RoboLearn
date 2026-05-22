# exp004 - SAC+HER Baseline on FetchSlide

- **Phase:** 1   **Track:** manipulation
- **Status:** planned
- **Config:** `configs/experiment/exp004_fetch_slide.yaml`

## Hypothesis
SAC+HER can reach non-trivial success rates on FetchSlide despite its difficulty
(the gripper cannot grasp the puck — must apply force to slide it to target, with
no direct contact possible after release). This is the hardest Fetch task.

## Setup
- Env: FetchSlide-v4 (sparse, horizon 50)
- Algorithm: SAC + HER, strategy=future, k=4
- Seeds: 8, Steps: 1,000,000
- GPU: full A100

## Baselines
Published target: ~30–60% success at 1M steps (Andrychowicz et al. 2017).
Record exact number: **[FILL from paper]**.
Note: FetchSlide success rates have high variance in the literature.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 1M steps.
Success criterion: non-trivial (>10%) and within CI of published target.
A lower-than-expected result is acceptable if the baseline is correctly implemented
(verified via exp001/003 matching targets first).

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs._

## Next step
If below target → check if more steps needed or if learning rate tuning helps.
FetchSlide is a known stretch task; a negative result with honest analysis is publishable.
