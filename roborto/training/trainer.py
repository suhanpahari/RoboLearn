"""Trainer implementations.

build_trainer() dispatches on cfg.algo.family.
SACHERTrainer handles the manipulation track (Phase 1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import SAC

from roborto.algos.sac import build_sac
from roborto.envs.gym_robotics.make import make_env

log = logging.getLogger(__name__)

_SMOKE_STEPS = 2_000
_SMOKE_EVAL_EPISODES = 3
_SMOKE_LEARNING_STARTS = 200


class Trainer:
    """Base trainer. Subclasses implement fit()."""

    def __init__(self, cfg: DictConfig, device, run):
        self.cfg = cfg
        self.device = device
        self.run = run

    def fit(self) -> None:
        raise NotImplementedError


class SACHERTrainer(Trainer):
    """SAC + HER trainer for Gymnasium-Robotics goal-conditioned envs.

    Supports:
    - trainer.smoke=true  — tiny run (2k steps) to verify the pipeline end-to-end
    - trainer.resume=true — resumes from the last checkpoint if one exists

    Checkpoints are saved to <run_dir>/checkpoint/ after every eval cycle.
    """

    def fit(self) -> None:
        cfg = self.cfg
        is_smoke = cfg.trainer.smoke

        total_steps = _SMOKE_STEPS if is_smoke else cfg.trainer.total_steps
        eval_episodes = _SMOKE_EVAL_EPISODES if is_smoke else cfg.trainer.eval_episodes
        eval_interval = min(total_steps, cfg.trainer.eval_interval)

        ckpt_dir = Path(self.run.run_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_path = ckpt_dir / "model"
        step_path = ckpt_dir / "step.json"

        env = make_env(cfg.env, seed=cfg.seed)
        eval_env = make_env(cfg.env, seed=cfg.seed + 1000)

        # Resolve torch device string. self.device is None in tests → fall back to cpu.
        device_str = str(self.device) if self.device is not None else "cpu"

        # Resume from checkpoint if available.
        trained_steps = 0
        if cfg.trainer.resume and model_path.with_suffix(".zip").exists():
            log.info("Resuming from checkpoint at %s", model_path)
            model = SAC.load(str(model_path), env=env, device=device_str)
            if step_path.exists():
                trained_steps = json.loads(step_path.read_text())["step"]
            log.info("Resuming at step %d", trained_steps)
        else:
            model = build_sac(cfg, env, device=device_str)
            if is_smoke:
                # Start learning sooner in smoke mode.
                model.learning_starts = _SMOKE_LEARNING_STARTS

        log.info(
            "SACHERTrainer: %s | total_steps=%d smoke=%s",
            cfg.env.id,
            total_steps,
            is_smoke,
        )

        # Train in chunks for periodic eval and checkpointing.
        while trained_steps < total_steps:
            chunk = min(eval_interval, total_steps - trained_steps)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=(trained_steps == 0),
                progress_bar=False,
            )
            trained_steps += chunk

            metrics = _evaluate(model, eval_env, eval_episodes)
            log.info("step=%d  %s", trained_steps, metrics)
            self.run.log(metrics, step=trained_steps)

            model.save(str(model_path))
            step_path.write_text(json.dumps({"step": trained_steps}))

        env.close()
        eval_env.close()
        log.info("Training complete at step %d.", trained_steps)


def _evaluate(model: SAC, env, n_episodes: int) -> dict:
    """Run n_episodes with the greedy policy; return success rate and mean return."""
    successes, returns = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        successes.append(float(info.get("is_success", 0.0)))
        returns.append(ep_return)
    return {
        "eval/success_rate": float(np.mean(successes)),
        "eval/mean_return": float(np.mean(returns)),
    }


def build_trainer(cfg: DictConfig, device, run) -> Trainer:
    """Dispatch to the right Trainer subclass based on cfg.algo.family."""
    family = cfg.algo.family
    if family == "manipulation":
        return SACHERTrainer(cfg, device, run)
    raise NotImplementedError(
        f"No trainer for algo.family={family!r}. Implement the navigation trainer in Phase 2."
    )
