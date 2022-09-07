from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig

from experiment_utils.logger import Logger
from experiment_utils.utils.fastai_utils import get_learner
from experiment_utils.utils.hydra_utils import instantiate_model
from experiment_utils.utils.model_utils import load_model_state
from experiment_utils.utils.utils import set_seed


class Experiment:

    learn: Any
    train_func: Callable
    model: torch.nn.Module | None = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = Logger(cfg)
        # self.learn = None
        # self.train_func = None

    def run(self):
        for repeat in range(self.cfg.repeat):
            set_seed(cfg=self.cfg.seed)
            self.set_model()
            # self.modify_model()
            self.set_learner()
            self.set_train_func()
            self.logger.start_job(self.learn, repeat)
            self.train_func(self.learn)
            self.logger.log_job()

        self.logger.log_run()

    def set_model(self):
        self.model = instantiate_model(self.cfg)
        if self.cfg.model_load:
            if self.cfg.model_load.model_load:
                load_model_state(self.model, self.cfg)
            if (process_model := self.cfg.model_load.get("process_model", None)) is not None:
                func = hydra.utils.instantiate(process_model, _convert_="all")
                func(self.model, self.cfg)

    def set_learner(self):
        self.learn = get_learner(self.cfg, self.model)
        if self.model is None:
            self.model = self.learn.model

    def set_train_func(self):
        self.train_func = hydra.utils.instantiate(self.cfg.train)
