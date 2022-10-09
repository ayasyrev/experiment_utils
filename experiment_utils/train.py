import warnings

import hydra
from fastprogress import fastprogress
from omegaconf import DictConfig, OmegaConf

from experiment_utils.experiment import ExpCfg, Experiment
from experiment_utils.utils.utils import SeedCfg

fastprogress.MAX_COLS = 80
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf2", config_name="config")
def train(cfg: DictConfig) -> None:

    # print(OmegaConf.to_yaml(cfg))
    cfg = ExpCfg(**cfg)
    # print(cfg.seed)
    # cfg_seed = Cfg_Seed(**cfg.seed)
    # print(cfg_seed)
    # print(cfg)
    exp = Experiment(cfg)
    exp.run()


if __name__ == "__main__":
    train()
