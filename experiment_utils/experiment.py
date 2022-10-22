from functools import partial
from typing import Any, Callable

# import hydra
import torch
from fastai.basics import Learner, accuracy
from fastai.data.core import DataLoaders
from pt_utils.data.data_config import DataCfg
from pt_utils.data.dataloader import get_dataloaders
# from omegaconf import DictConfig
from pydantic import BaseModel

from experiment_utils.logger import Logger
from experiment_utils.utils.import_utils import FuncCfg
from experiment_utils.utils.fastai_utils import create_learner
# from experiment_utils.utils.model_utils import load_model_state
from experiment_utils.utils.utils import SeedCfg, set_seed
from .exp_defaults import loss_func_dict, opt_func_dict, train_func_dict
from .models import models_dict

from .logger import LogCfg


class ModelCfg(FuncCfg):
    name: str = "xresnet18"
    path: str | None = "experiment_utils.models.xresnet"
    weight_path: str | None = None


class FitWarmupAnneal(BaseModel):
    # name: str = "fit_warmup_anneal"
    # path: str = "experiment_utils.utils.fastai_utils"
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
    seed: SeedCfg = SeedCfg()
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    # dls = ""
    opt_func: str = "ranger"
    loss_func: str = "LabelSmoothingCrossEntropy"
    train: FitWarmupAnneal = FitWarmupAnneal()
    train_func: str = "fit_warmup_anneal"
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
        self.model = models_dict[self.cfg.model.name]()
        if self.cfg.model.weight_path is not None:
            self.model.load_state_dict(torch.load(f"{self.cfg.model.weight_path}.pt"))

    def set_learner(self):
        if self.model is None:
            self.set_model()
        if self.opt_func is None:
            self.set_opt()
        if self.loss_func is None:
            self.set_loss_func()
        dls = get_dataloaders(self.cfg.data)
        self.learn = create_learner(dls, self.model, self.opt_func, self.metrics, self.loss_func, self.cfg)

    def set_opt(self):
        self.opt_func = opt_func_dict[self.cfg.opt_func]

    def set_loss_func(self):
        self.loss_func = loss_func_dict[self.cfg.loss_func]()

    def set_train_func(self):
        self.train_func = partial(
            train_func_dict[self.cfg.train_func],
            **self.cfg.train.dict())


