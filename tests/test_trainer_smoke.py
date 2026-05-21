"""Smoke test for the manipulation trainer.

Exercises the full train → checkpoint → resume path on FetchReach in 2k steps
(cpu-only, no W&B). Marked with the 'smoke' pytest marker so the normal test
suite can skip it; `make smoke` runs it explicitly.
"""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from roborto.training.trainer import SACHERTrainer, build_trainer
from roborto.utils.logging import init_run_logger
from roborto.utils.seeding import set_global_seed


@pytest.fixture()
def smoke_cfg(tmp_path):
    """Minimal Hydra-free config for a single FetchReach smoke run."""
    return OmegaConf.create(
        {
            "seed": 0,
            "experiment_name": "smoke_test",
            "env": {
                "name": "FetchReach",
                "id": "FetchReach-v4",
                "max_episode_steps": 50,
                "reward_type": "sparse",
                "goal_conditioned": True,
            },
            "algo": {
                "name": "sac_her",
                "family": "manipulation",
                "backend": "stable_baselines3",
                "gamma": 0.98,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "buffer_size": 10_000,
                "tau": 0.005,
                "learning_starts": 200,
                "her": {"strategy": "future", "n_relabel": 4},
            },
            "trainer": {
                "total_steps": 1_000_000,
                "eval_interval": 25_000,
                "eval_episodes": 50,
                "smoke": True,
                "resume": False,
            },
            "logging": {"backend": "offline"},
            "paths": {"run_dir": str(tmp_path)},
        }
    )


def test_build_trainer_dispatches(smoke_cfg, tmp_path):
    run = init_run_logger(smoke_cfg, tmp_path / "run")
    try:
        trainer = build_trainer(smoke_cfg, device=None, run=run)
        assert isinstance(trainer, SACHERTrainer)
    finally:
        run.finish()


def test_smoke_run_completes(smoke_cfg, tmp_path):
    """Full smoke run: 2k steps, one eval, one checkpoint written."""
    set_global_seed(0)
    run_dir = tmp_path / "run"
    run = init_run_logger(smoke_cfg, run_dir)
    try:
        trainer = SACHERTrainer(smoke_cfg, device=None, run=run)
        trainer.fit()
    finally:
        run.finish()

    ckpt = run_dir / "checkpoint" / "step.json"
    assert ckpt.exists(), "checkpoint step file not written"
    data = json.loads(ckpt.read_text())
    assert data["step"] == 2_000

    assert (run_dir / "checkpoint" / "model.zip").exists(), "model checkpoint missing"
    assert (run_dir / "metrics.jsonl").exists(), "metrics log not written"


def test_resume_from_checkpoint(smoke_cfg, tmp_path):
    """A second run with resume=True loads the checkpoint and finishes cleanly."""
    set_global_seed(1)
    run_dir = tmp_path / "run"

    cfg_first = OmegaConf.merge(smoke_cfg, {"trainer": {"resume": False}})
    run1 = init_run_logger(cfg_first, run_dir)
    try:
        SACHERTrainer(cfg_first, device=None, run=run1).fit()
    finally:
        run1.finish()

    cfg_resume = OmegaConf.merge(smoke_cfg, {"trainer": {"resume": True}})
    run2 = init_run_logger(cfg_resume, run_dir)
    try:
        SACHERTrainer(cfg_resume, device=None, run=run2).fit()
    finally:
        run2.finish()
