"""GoalReplayBuffer behaviour."""

import numpy as np

from roborto.buffers.replay_buffer import GoalReplayBuffer


def _add(buf, obs_value):
    buf.add(
        obs=np.array([obs_value]),
        action=np.zeros(1),
        reward=0.0,
        next_obs=np.zeros(1),
        done=0.0,
        achieved=np.zeros(1),
        desired=np.zeros(1),
    )


def test_add_and_len():
    buf = GoalReplayBuffer(capacity=10, obs_dim=1, goal_dim=1, act_dim=1)
    assert len(buf) == 0
    for i in range(5):
        _add(buf, i)
    assert len(buf) == 5


def test_circular_overwrite():
    buf = GoalReplayBuffer(capacity=4, obs_dim=1, goal_dim=1, act_dim=1)
    for i in range(6):
        _add(buf, i)
    assert len(buf) == 4
    stored = set(buf.obs[:, 0])
    assert 5.0 in stored  # most recent kept
    assert 0.0 not in stored  # oldest overwritten


def test_sample_shapes():
    buf = GoalReplayBuffer(capacity=20, obs_dim=3, goal_dim=2, act_dim=4)
    for _ in range(20):
        buf.add(np.zeros(3), np.zeros(4), 1.0, np.zeros(3), 0.0, np.zeros(2), np.zeros(2))
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 3)
    assert batch["action"].shape == (8, 4)
    assert batch["desired_goal"].shape == (8, 2)
