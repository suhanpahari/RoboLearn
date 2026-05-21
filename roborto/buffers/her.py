"""Hindsight Experience Replay relabeling.

Implemented in-repo (rather than only using SB3's HerReplayBuffer) because the
relabeling-strategy comparison is part of the Spine A manipulation contribution -
see docs/experiments/exp001_fetch_sac_her.md. Reference: Andrychowicz et al., 2017.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RewardFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
STRATEGIES = ("final", "future", "episode", "random")


class HERRelabeler:
    """Relabels episode transitions with hindsight goals.

    Args:
        strategy: one of STRATEGIES.
        n_relabel: extra relabeled copies per real transition (the 'k' in HER).
        compute_reward: vectorized (achieved, desired) -> reward, from the GoalEnv.
        seed: RNG seed.
    """

    def __init__(
        self,
        strategy: str = "future",
        n_relabel: int = 4,
        compute_reward: RewardFn | None = None,
        seed: int = 0,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        if compute_reward is None:
            raise ValueError("compute_reward is required (pass the GoalEnv's reward fn)")
        self.strategy = strategy
        self.n_relabel = int(n_relabel)
        self.compute_reward = compute_reward
        self._rng = np.random.default_rng(seed)

    def _sample_goal_indices(self, t: int, ep_len: int) -> np.ndarray:
        """Episode indices whose achieved goal becomes the new desired goal for t."""
        if self.strategy == "final":
            return np.full(self.n_relabel, ep_len - 1)
        if self.strategy == "future":
            if t >= ep_len - 1:
                return np.array([], dtype=int)
            return self._rng.integers(t + 1, ep_len, size=self.n_relabel)
        # 'episode' and 'random' both sample uniformly within the episode here;
        # a cross-episode 'random' variant can be added at the buffer level.
        return self._rng.integers(0, ep_len, size=self.n_relabel)

    def relabel_episode(self, episode: dict) -> dict:
        """Return relabeled transitions for one episode.

        Args:
            episode: dict of arrays, each shape (T, ...), with keys
                obs, action, next_obs, next_achieved_goal.

        Returns:
            dict of arrays for the relabeled transitions.
        """
        ep_len = len(episode["action"])
        out: dict[str, list] = {
            k: [] for k in ("obs", "action", "next_obs", "desired_goal", "reward")
        }
        for t in range(ep_len):
            for g in self._sample_goal_indices(t, ep_len):
                new_goal = episode["next_achieved_goal"][g]
                reward = self.compute_reward(
                    episode["next_achieved_goal"][t][None], new_goal[None]
                )[0]
                out["obs"].append(episode["obs"][t])
                out["action"].append(episode["action"][t])
                out["next_obs"].append(episode["next_obs"][t])
                out["desired_goal"].append(new_goal)
                out["reward"].append(reward)
        return {k: np.asarray(v) for k, v in out.items()}


# TODO(research): add a 'difficulty_aware' future-sampling strategy (a Spine A idea
# seed) - weight future goals by achieved-goal displacement. Add to STRATEGIES + tests.
