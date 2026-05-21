"""SAC (+ HER) for the manipulation track, wrapping Stable-Baselines3.

The baseline uses SB3's built-in HerReplayBuffer. The relabeling-strategy
study (exp002) swaps in roborto.buffers.her.HERRelabeler at the buffer level
while keeping the same SAC policy network.
"""

from __future__ import annotations

import gymnasium as gym
from omegaconf import DictConfig
from stable_baselines3 import SAC, HerReplayBuffer


def build_sac(cfg: DictConfig, env: gym.Env, device: str = "auto") -> SAC:
    """Construct a SB3 SAC agent with HER replay for a goal-conditioned env.

    Args:
        cfg: the full resolved Hydra config.
        env: a seeded GoalEnv (dict obs space).
        device: torch device string. Pass "cpu" for tests or "cuda:0" for
            production (after CUDA_VISIBLE_DEVICES is pinned by the Makefile).

    Returns:
        An untrained SAC model ready for model.learn().
    """
    her_kwargs = {
        "n_sampled_goal": cfg.algo.her.n_relabel,
        "goal_selection_strategy": cfg.algo.her.strategy,
    }
    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=her_kwargs,
        learning_rate=cfg.algo.learning_rate,
        batch_size=cfg.algo.batch_size,
        buffer_size=cfg.algo.buffer_size,
        tau=cfg.algo.tau,
        gamma=cfg.algo.gamma,
        learning_starts=cfg.algo.learning_starts,
        verbose=0,
        device=device,
    )
