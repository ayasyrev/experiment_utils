import warnings

import hydra
from fastai.basics import Learner, accuracy
from fastprogress import fastprogress
from omegaconf import DictConfig

from experiment_utils.logger import Logger

fastprogress.MAX_COLS = 80
warnings.filterwarnings("ignore")


def get_learner(cfg):
    '''Return fastai Learner from cfg'''

    dls = hydra.utils.instantiate(cfg.dls)
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
