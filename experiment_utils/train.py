import warnings

import hydra
from fastprogress import fastprogress
from omegaconf import DictConfig

from experiment_utils.experiment import Experiment


fastprogress.MAX_COLS = 80
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:

    exp = Experiment(cfg)
    exp.run()


if __name__ == "__main__":
    train()
