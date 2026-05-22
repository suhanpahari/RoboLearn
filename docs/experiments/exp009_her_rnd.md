# exp009 - HER + RND Exploration Bonus on FetchPickAndPlace

- **Phase:** 1   **Track:** manipulation
- **Status:** planned — requires RND implementation first
- **Config:** `configs/experiment/exp009_her_rnd.yaml`

## Hypothesis
Adding a Random Network Distillation (Burda et al., 2018) intrinsic reward bonus
to SAC+HER improves sample efficiency on FetchPickAndPlace by encouraging exploration
of novel object configurations, which are rare under the sparse goal-conditioned reward.
RND provides a state-novelty bonus that decays as the agent revisits states.

## Setup
- Env: FetchPickAndPlace-v4 (sparse, horizon 50)
- Algorithm: SAC + HER (future, k=4) + RND bonus (bonus_scale=0.01, to tune)
- Seeds: 5, Steps: 1,000,000
- Compare against: exp003 (SAC+HER, no bonus, same task and steps)

## Implementation required before running
1. Implement RND module in `roborto/algos/rnd.py`:
   - Fixed random network (target) and trained predictor network
   - Intrinsic reward = MSE(predictor(s), target(s)), normalised by running stats
2. Integrate into SACHERTrainer: add bonus to extrinsic reward before storing in buffer
3. Wire `algo.rnd.enabled` and `algo.rnd.bonus_scale` config keys
4. Unit tests + smoke run before full launch

## Baselines
Primary: exp003 (SAC+HER without bonus, FetchPickAndPlace, 1M steps).
Hypothesis null: RND bonus does not help because HER already provides a sufficient
learning signal — the bottleneck is not exploration but credit assignment.

## Metrics & success criteria
Success rate (IQM ± 95% CI) at 1M steps vs exp003.
Also track: intrinsic reward scale over training (sanity: should decrease as states
are visited more often).
Success: IQM improvement ≥ 5 pp with non-overlapping 95% CIs.

## Results
_Filled in after runs._

## Interpretation
_Filled in after runs. If null: report and discuss why HER may subsume the
exploration bonus — relabeled transitions already implicitly encourage visiting
diverse achieved goals._

## Next step
If positive → combine with exp008 (difficulty-aware + RND) as a compound contribution.
If null → report as negative result alongside exp008 in the failure analysis section.
