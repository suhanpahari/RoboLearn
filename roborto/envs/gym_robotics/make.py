"""Gymnasium-Robotics environment construction (Fetch / Hand goal-envs).

Imports gymnasium_robotics so its envs get registered, then builds the GoalEnv
from cfg.env.id. The environment exposes env.compute_reward, which lets
roborto.buffers.her.HERRelabeler recompute hindsight rewards without reaching
into internals.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch/Hand/Maze envs on import
from omegaconf import DictConfig


def make_env(cfg: DictConfig, seed: int = 0) -> gym.Env:
    """Build a seeded GoalEnv from the env config.

    Args:
        cfg: the resolved env sub-config (cfg.env from Hydra).
        seed: RNG seed; also sets the action-space seed.

    Returns:
        A seeded gymnasium GoalEnv with dict observation space
        {observation, achieved_goal, desired_goal}.
    """
    env = gym.make(
        cfg.id,
        max_episode_steps=cfg.max_episode_steps,
        reward_type=cfg.reward_type,
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
