import warnings

import hydra
import torch
from fastai.basics import (CategoryBlock, DataBlock, GrandparentSplitter,
                           Learner, accuracy, get_image_files,
                           parent_label)

from fastai.vision.all import ImageBlock
from fastprogress import fastprogress
from omegaconf import DictConfig

from experiment_utils.logger import Logger


fastprogress.MAX_COLS = 80
warnings.filterwarnings("ignore")


def get_dataloaders(ds_path, bs, num_workers,
                    item_tfms, batch_tfms):

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name='val'),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    return dblock.dataloaders(
        ds_path, path=ds_path, bs=bs, num_workers=num_workers)


def get_learner(cfg):
    '''Return fastai Learner from cfg'''

    dls = get_dataloaders(
        ds_path=cfg.data.data_dir,
        bs=cfg.data.dataloader.batch_size,
        num_workers=cfg.data.dataloader.num_workers,
        item_tfms=hydra.utils.instantiate(cfg.data.item_tfms),
        batch_tfms=hydra.utils.instantiate(cfg.data.batch_tfms),
    )

    model = hydra.utils.instantiate(cfg.model, _convert_='all')

    opt_fn = hydra.utils.call(cfg.opt_fn)

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    learn = Learner(dls=dls,
                    model=model,
                    opt_func=opt_fn,
                    metrics=[accuracy],
                    loss_func=loss_fn)

    if cfg.model_load.model_load:
        learn.load(file=cfg.model_load.file_name, with_opt=cfg.model_load.with_opt)
    return learn


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    logger = Logger(cfg)

    for repeat in range(cfg.run.repeat):
        hydra.utils.call(cfg.seed)
        learn = get_learner(cfg)

        logger.start_job(learn, repeat)

        hydra.utils.call(cfg.train)(learn)

        logger.log_job()

    logger.log_run()


if __name__ == "__main__":
    train()
