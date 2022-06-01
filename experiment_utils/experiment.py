import hydra
from omegaconf import DictConfig, OmegaConf

from experiment_utils.logger import Logger
from experiment_utils.utils.fastai_utils import get_learner
from experiment_utils.utils.utils import set_seed


class Experiment:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = Logger(cfg)
        self.learn = None
        self.train_func = None

    def run(self):
        for repeat in range(self.cfg.repeat):
            set_seed(**OmegaConf.to_object(self.cfg.seed))
            self.set_learner()
            self.set_train_func()
            self.logger.start_job(self.learn, repeat)
            self.train_func(self.learn)
            self.logger.log_job()

        self.logger.log_run()

    def set_learner(self):
        self.learn = get_learner(self.cfg)

    def set_train_func(self):
        self.train_func = hydra.utils.instantiate(self.cfg.train)
