import warnings
from typing import Optional

import torch
from fastai.basics import Learner, accuracy, ranger
from fastai.losses import LabelSmoothingCrossEntropy
from fastai.vision.augment import (FlipItem, Normalize, RandomResizedCrop,
                                   imagenet_stats)
from fastprogress import fastprogress
from pydantic import BaseModel, validator

from experiment_utils.logger import LogCfg, Logger
from experiment_utils.models import models_dict
from experiment_utils.utils.fastai_utils import get_dataloaders
from experiment_utils.utils.utils import set_seed

# from omegaconf import DictConfig


fastprogress.MAX_COLS = 80


warnings.filterwarnings("ignore")


class Cfg(BaseModel):
    repeats: int = 1
    data_path: str = "/Data/imagenette2-320/"
    size: int = 128
    min_scale: float = 0.35
    bs: int = 32

    model: str = "xresnet18"
    model_weights: Optional[str]

    epochs: int = 5
    lr: float = 0.008
    wd: float = 0.01
    cbs: list = []

    num_workers: int = 4
    seed_deterministic: bool = True

    log: LogCfg = LogCfg()

    @validator("model")
    def model_supported(cls, v):
        if v not in models_dict:
            raise ValueError(f"{v} not in models dict.")
        return v


def get_learner(cfg: Cfg) -> Learner:
    """Return fastai Learner from cfg"""

    tfms = [RandomResizedCrop(cfg.size, min_scale=cfg.min_scale), FlipItem(0.5)]
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    dls = get_dataloaders(
        ds_path=cfg.data_path,
        bs=cfg.bs,
        num_workers=cfg.num_workers,
        item_tfms=tfms,
        batch_tfms=batch_tfms
    )
    model = models_dict[cfg.model]()
    opt_fn = ranger
    loss_fn = LabelSmoothingCrossEntropy()

    learn = Learner(
        dls=dls,
        model=model,
        opt_func=opt_fn,
        metrics=[accuracy],
        loss_func=loss_fn
    )

    if cfg.model_weights:
        st_dict = torch.load(cfg.model_weights)
        learn.model.load_state_dict(st_dict)
        # learn.load(file=cfg.model_weights, with_opt=False)
    return learn


def train(cfg: Cfg) -> None:
    logger = Logger(cfg.log)
    logger.start_run(cfg.repeats, cfg_to_log=cfg)

    for repeat in range(cfg.repeats):
        set_seed(
            torch_benchmark=not cfg.seed_deterministic,
            torch_deterministic=cfg.seed_deterministic,
        )
        learn = get_learner(cfg)
        logger.start_job(learn, repeat)

        learn.fit_flat_cos(cfg.epochs, cfg.lr, wd=cfg.wd, cbs=cfg.cbs)

        logger.log_job()

    logger.log_run()


if __name__ == "__main__":
    cfg = Cfg(
        epochs=3,
        model_weights="/Data/model_weights/xresnext18_weight_0.pt",
        repeats=2,
    )
    train(cfg)
