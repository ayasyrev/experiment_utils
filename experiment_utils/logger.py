import csv
from experiment_utils.utils import stat
import time
from typing import List

from fastai.basics import Learner
from omegaconf import DictConfig, OmegaConf


def format_time(seconds: float, long: bool = True) -> str:
    "Format secons to mm:ss, optoinal mm:ss.ms"
    seconds_int = int(seconds)
    min, sec = (seconds_int // 60) % 60, seconds_int % 60
    res = f'{min:02d}:{sec:02d}'
    if long:
        res = '.'.join([res, f'{int((seconds - seconds_int) * 10)}'])
    return res


class Logger:
    """Log results"""
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        with open('cfg.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        self.results = []

    def start_job(self, learn: Learner, repeat: int):
        self.learn = learn
        self.repeat = repeat
        self.log_model()
        if self.cfg.run.repeat > 1:
            print(f"repeat #{repeat + 1} of {self.cfg.run.repeat}")
        self.start_time = time.time()

    def log_model(self):
        with open(f'model_{self.learn.model._get_name()}.txt', 'w') as f:
            f.write(str(self.learn.model))

    def log_job(self) -> None:
        acc = self.learn.recorder.final_record[-1]
        self.results.append(acc)
        print(f"acc: {acc:0.2%}")
        train_time = time.time() - self.start_time
        print(f"train time: {format_time(train_time)}")
        name_suffix = str(int(acc * 10000))
        if self.cfg.run.repeat > 1:
            name_suffix = f"{self.repeat}_{name_suffix}"
        log_result(name_suffix=name_suffix,
                   header=self.learn.recorder.metric_names[1:-1],
                   values=self.learn.recorder.values)
        if self.cfg.run.log_loss:
            log_lr(self.learn.recorder.losses, f"losses_{name_suffix}")
        print(50 * '=')
        if self.cfg.model_save.model_save:
            fn = (f"{self.learn.model._get_name()}_{self.cfg.date_time}_{name_suffix}"
                  if self.cfg.model_save.file_name == 'model'
                  else self.cfg.model_save.file_name)
            self.learn.save(fn, with_opt=self.cfg.model_save.with_opt)
        # return acc

    def log_run(self):
        if self.cfg.run.log_lr:
            log_lr(self.learn.recorder.lrs, 'lrs')

        if self.cfg.run.repeat > 1:
            log_resume(self.results)


def log_result(file_name: str = 'log_res', name_suffix: str = '', header: List[str] = [], values: List = []) -> None:
    if name_suffix != '':
        name_suffix = '_' + name_suffix
    with open(f"{file_name}{name_suffix}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(values)


def log_resume(results: List[float]) -> None:
    """Write results to file

    Args:
        results (List[float]): List of results.
    """
    mean, std = stat(results)
    print(f"mean: {mean:0.2%} std: {std:0.4f}")
    file_name = f"mean_{int(mean*10000)}_std_{int(std*10000):04}.csv"
    with open(file_name, 'w') as f:
        for result in results:
            f.write(f"{result}\n")
        f.write(f"#\n{mean}\n{std}")


def log_lr(values: List[float], name: str) -> None:
    """Write values to csv file"""
    with open(f"{name}.csv", "w") as f:
        f.writelines(map(lambda i: f"{i}\n", values))
