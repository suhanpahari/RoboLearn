"""Goal-conditioned replay buffer (NumPy) for the manipulation track."""

from __future__ import annotations

import numpy as np


class GoalReplayBuffer:
    """Fixed-size circular buffer of goal-conditioned transitions."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.achieved = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.desired = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add(self, obs, action, reward, next_obs, done, achieved, desired) -> None:
        """Insert one transition, overwriting the oldest when full."""
        i = self._idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.achieved[i] = achieved
        self.desired[i] = desired
        self._idx = (i + 1) % self.capacity
        self._full = self._full or self._idx == 0

    def sample(self, batch_size: int) -> dict:
        """Uniformly sample a batch of transitions as a dict of arrays."""
        idx = self._rng.integers(0, len(self), size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "achieved_goal": self.achieved[idx],
            "desired_goal": self.desired[idx],
        }
