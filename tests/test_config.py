"""The Hydra config tree must compose for the base config and every experiment."""

from hydra import compose, initialize
from omegaconf import OmegaConf


def test_base_config_composes():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    assert cfg.seed == 0
    assert "env" in cfg
    assert "algo" in cfg
    OmegaConf.to_yaml(cfg, resolve=True)  # all interpolations resolve


def test_experiment_config_composes():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=exp001_fetch_sac_her"],
        )
    assert cfg.experiment_name == "exp001_sac_her_fetch_push"
    assert cfg.algo.name == "sac_her"
    assert cfg.env.name == "FetchPush"
