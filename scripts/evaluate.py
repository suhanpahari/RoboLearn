"""Evaluation entry point.

python scripts/evaluate.py experiment=exp001_fetch_sac_her
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO(claude-code): load the checkpoint from cfg.paths.run_dir, run
    # cfg.trainer.eval_episodes on held-out goals/scenes, compute metrics
    # (roborto.evaluation.metrics) and log them.
    raise NotImplementedError("evaluate.py is a scaffold stub - see docs/PLAN.md section 5.")


if __name__ == "__main__":
    main()
