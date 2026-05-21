"""Run logging: Weights & Biases when available, JSONL offline fallback.

Every run records its resolved config and full provenance (git sha, package
versions) so any result can be traced back to the exact code that made it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from roborto.utils.provenance import env_provenance

log = logging.getLogger(__name__)


class RunLogger:
    """Minimal experiment logger with a stable interface (log / finish)."""

    def __init__(self, cfg: DictConfig, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        self._jsonl = None

        provenance = env_provenance()
        (self.run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        (self.run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

        if cfg.logging.backend == "wandb":
            try:
                import wandb

                self._wandb = wandb.init(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    name=f"{cfg.experiment_name}/seed{cfg.seed}",
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                self._wandb.summary["git_sha"] = provenance["git_sha"]
            except Exception as exc:
                log.warning("W&B init failed (%s); using offline JSONL.", exc)

        if self._wandb is None:
            self._jsonl = open(self.run_dir / "metrics.jsonl", "a")

    def log(self, metrics: dict, step: int) -> None:
        """Record a dict of scalar metrics at a given training step."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._jsonl is not None:
            self._jsonl.write(json.dumps({"step": step, "t": time.time(), **metrics}) + "\n")
            self._jsonl.flush()

    def finish(self) -> None:
        """Flush and close the logger. Always call this (use try/finally)."""
        if self._wandb is not None:
            self._wandb.finish()
        if self._jsonl is not None:
            self._jsonl.close()


def init_run_logger(cfg: DictConfig, run_dir: Path) -> RunLogger:
    """Construct the RunLogger for a run."""
    return RunLogger(cfg, run_dir)
