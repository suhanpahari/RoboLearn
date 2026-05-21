"""HER relabeling correctness."""

import numpy as np
import pytest

from roborto.buffers.her import STRATEGIES, HERRelabeler


def _sparse_reward(achieved, desired, eps=0.05):
    dist = np.linalg.norm(achieved - desired, axis=-1)
    return (dist <= eps).astype(np.float32) - 1.0  # sparse rewards in {-1, 0}


def _episode(length=8, goal_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    next_achieved = rng.normal(size=(length, goal_dim)).astype(np.float32)
    return {
        "obs": rng.normal(size=(length, 4)).astype(np.float32),
        "action": rng.normal(size=(length, 2)).astype(np.float32),
        "next_obs": rng.normal(size=(length, 4)).astype(np.float32),
        "next_achieved_goal": next_achieved,
    }


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_relabel_runs_for_all_strategies(strategy):
    her = HERRelabeler(strategy, n_relabel=4, compute_reward=_sparse_reward, seed=1)
    out = her.relabel_episode(_episode())
    assert set(out) == {"obs", "action", "next_obs", "desired_goal", "reward"}
    assert len(out["reward"]) == len(out["action"])


def test_future_indices_are_strictly_future():
    her = HERRelabeler("future", n_relabel=20, compute_reward=_sparse_reward, seed=3)
    for t in range(9):
        idx = her._sample_goal_indices(t, ep_len=10)
        assert np.all(idx > t)
    assert her._sample_goal_indices(9, 10).size == 0


def test_rewards_are_sparse_valued():
    her = HERRelabeler("future", n_relabel=8, compute_reward=_sparse_reward, seed=2)
    out = her.relabel_episode(_episode())
    assert np.all(np.isin(out["reward"], [-1.0, 0.0]))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        HERRelabeler("nonsense", compute_reward=_sparse_reward)


def test_missing_reward_fn_raises():
    with pytest.raises(ValueError):
        HERRelabeler("future", compute_reward=None)
