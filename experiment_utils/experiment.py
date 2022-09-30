from typing import Any, Callable

import hydra
import torch
from fastai.basics import Learner, accuracy
from fastai.data.core import DataLoaders
from pt_utils.data.data_config import DataCfg
from pt_utils.data.dataloader import get_dataloaders
# from omegaconf import DictConfig
from pydantic import BaseModel

from experiment_utils.logger import Logger
from experiment_utils.utils.import_utils import FuncCfg, load_obj, load_obj_partial, load_obj_run
from experiment_utils.utils.model_utils import load_model_state
from experiment_utils.utils.utils import Cfg_Seed, set_seed

from .logger import LogCfg


class ModelCfg(FuncCfg):
    name: str = "xresnet18"
    path: str | None = "experiment_utils.models.xresnet"


class FitWarmupAnneal(FuncCfg):
    name: str = "fit_warmup_anneal"
    path: str = "experiment_utils.utils.fastai_utils"
    epochs: int = 5
    lr: float = 0.008
    wd: float = 0.01
    warmup_type: str = "lin"
    warmup_pct: float = 0.01
    warmup_div: float = 100.
    anneal_type: str = "cos"
    anneal_pct: float = 0.75
    anneal_div: float = 100000.0


class ExpCfg(BaseModel):
    seed: Cfg_Seed = Cfg_Seed()
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    # dls = ""
    opt_func: FuncCfg = FuncCfg(name="ranger", path="fastai.basics")
    loss_func: FuncCfg = FuncCfg(name="LabelSmoothingCrossEntropy", path="fastai.losses")
    train = ""
    train_func: FuncCfg = FitWarmupAnneal()
    log = LogCfg()
    name: str = "no_name"
    repeat: int = 1  # to train?


class Experiment:

    learn: Any
    train_func: None | Callable
    model: torch.nn.Module | None = None
    opt_func = None
    loss_func = None
    val_loss_func = None
    metrics = [accuracy]

    def __init__(self, cfg: ExpCfg | None) -> None:
        self.cfg = cfg or ExpCfg()
        self.logger = Logger(self.cfg.log)

    def run(self, repeat: int | None = None):
        repeat = repeat or self.cfg.repeat
        self.logger.start_run(repeat, self.cfg)
        for run_number in range(repeat):
            set_seed(cfg=self.cfg.seed)
            self.set_model()
            # self.modify_model()
            self.set_learner()
            self.set_train_func()
            self.logger.start_job(self.learn, run_number)
            self.train_func(self.learn)
            self.logger.log_job()

        self.logger.log_run()

    def set_model(self):
        self.model = load_obj(self.cfg.model)()
        # if self.cfg.model_load:
        #     pass

    def set_learner(self):
        if self.model is None:
            self.set_model()
        if self.opt_func is None:
            self.set_opt()
        if self.loss_func is None:
            self.set_loss_func()
        self._set_learner()

    def set_opt(self):
        self.opt_func = load_obj(self.cfg.opt_func)

    def set_loss_func(self):
        self.loss_func = load_obj_run(self.cfg.loss_func)

    def set_train_func(self):
        self.train_func = load_obj_partial(self.cfg.train_func)

    def _set_learner(self) -> None:
        """create fastai Learner"""
        self.learn = Learner(
            dls=DataLoaders(*get_dataloaders(self.cfg.data)),
            model=self.model,
            opt_func=self.opt_func,
            metrics=self.metrics,
            loss_func=self.loss_func
        )
