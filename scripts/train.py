"""Training entry point.

    python scripts/train.py experiment=exp001_fetch_sac_her

Launch via `make train EXP=... GPU=<uuid>` so CUDA_VISIBLE_DEVICES is pinned.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from roborto.training.trainer import build_trainer
from roborto.utils.device import log_gpu_visibility, resolve_device
from roborto.utils.logging import init_run_logger
from roborto.utils.seeding import set_global_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Experiment: %s | seed=%s", cfg.experiment_name, cfg.seed)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seed(cfg.seed)
    log_gpu_visibility()
    device = resolve_device()

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = init_run_logger(cfg, run_dir)
    try:
        trainer = build_trainer(cfg, device=device, run=run)
        trainer.fit()
    finally:
        run.finish()


if __name__ == "__main__":
    main()
