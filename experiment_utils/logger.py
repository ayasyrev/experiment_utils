import csv
import time
from pathlib import Path
from typing import List

import yaml
from fastai.basics import Learner
# from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from experiment_utils.utils.utils import stat


def format_time(seconds: float, long: bool = True) -> str:
    "Format seconds to mm:ss, optional mm:ss.ms"
    seconds_int = int(seconds)
    min, sec = (seconds_int // 60) % 60, seconds_int % 60
    res = f"{min:02d}:{sec:02d}"
    if long:
        res = ".".join([res, f"{int((seconds - seconds_int) * 10)}"])
    return res


class LogCfg(BaseModel):
    log_dir: str = "log_run"
    repeat: int = 1
    date_time: str | None = None  # todo
    model_save: bool = False
    model_save_name: str = "model"
    model_save_opt: bool = False
    log_loss: bool = True
    log_lr: bool = True


class Logger:
    """Log results"""

    def __init__(self, cfg: LogCfg) -> None:
        self.cfg = cfg or LogCfg()
        self.results = []

    def _create_log_dir(self):
        self.log_dir = Path(self.cfg.log_dir)
        if self.log_dir.exists() and self.cfg.log_dir != ".":
            self.log_dir = Path(
                f"{self.cfg.log_dir}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            self.cfg.log_dir = str(self.log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def start_run(self, repeat, cfg_to_log):
        self._create_log_dir()
        self.repeat = repeat
        with open(self.log_dir / "cfg.yaml", "w") as f:
            # f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
            # f.write(OmegaConf.to_yaml(cfg_to_log.dict()))
            f.write(yaml.dump(cfg_to_log.dict()))

    def start_job(self, learn: Learner, run_number: int) -> None:
        self.learn = learn
        self.run_number = run_number
        self.log_model()
        if self.repeat > 1:
            print(f"repeat #{run_number + 1} of {self.repeat}")
        self.start_time = time.time()

    def log_model(self):
        model_name = self.learn.model._get_name()
        if model_name == "Sequential":
            model_name = self.learn.model.extra_repr()
        self.model_name = model_name
        with open(self.log_dir / f"model_{model_name}.txt", "w") as f:
            f.write(str(self.learn.model))

    def log_job(self) -> None:
        acc = self.learn.recorder.final_record[-1]
        self.results.append(acc)
        print(f"acc: {acc:0.2%}")
        train_time = time.time() - self.start_time
        print(f"run time: {format_time(train_time)}")
        name_suffix = str(int(round(acc, 4) * 10000))
        if self.repeat > 1:
            name_suffix = f"{self.run_number}_{name_suffix}"
        self.log_result(name_suffix=name_suffix)
        if self.cfg.log_loss:
            self.log_values(self.learn.recorder.losses, f"losses_{name_suffix}")
        print(50 * "-")
        if self.cfg.model_save:
            fn = (
                # f"{self.learn.model._get_name()}_{self.cfg.run.date_time}_{name_suffix}"
                f"{self.model_name}_{self.cfg.date_time}_{name_suffix}"
                if self.cfg.model_save_name == "model"
                else self.cfg.model_save_name
            )
            self.learn.save(fn, with_opt=self.cfg.model_save_opt)
            print(f"model saved. {fn}, {Path.cwd()}")

    def log_run(self) -> None:
        if self.cfg.log_lr:
            if hasattr(self.learn.recorder, "hps"):
                for hp in self.learn.recorder.hps:
                    self.log_values(self.learn.recorder.hps[hp], f"{hp}s")
            else:
                self.log_values(self.learn.recorder.lrs, "lrs")
        if self.repeat > 1:
            self.log_resume()

    def log_result(self, file_name: str = "log_res", name_suffix: str = "") -> None:
        if name_suffix != "":
            name_suffix = "_" + name_suffix
        with open(self.log_dir / f"{file_name}{name_suffix}.csv", "w") as f:
            writer = csv.writer(f)
            # writer.writerow(self.learn.recorder.metric_names[1:-1])
            writer.writerow(["epoch_num"] + self.learn.recorder.metric_names[1:-1])
            # writer.writerows(self.learn.recorder.values)
            for epoch, ep_value in enumerate(self.learn.recorder.values):
                writer.writerow([epoch] + ep_value)

    def log_resume(self) -> None:
        """Write summary for results to file

        Args:
            results (List[float]): List of results.
        """
        mean, std = stat(self.results)
        print(f"mean: {mean:0.2%} std: {std:0.4f}")
        print(50 * "=")

        file_name = f"mean_{int(mean*10000)}_std_{int(round(std, 4)*10000):04}.csv"
        with open(self.log_dir / file_name, "w") as f:
            for result in self.results:
                f.write(f"{result}\n")
            f.write(f"#\n{mean}\n{std}")

    def log_values(self, values: List[float], name: str) -> None:
        """Write values to csv file

        Args:
            values (List[float]): List of floats to write.
            name (str): Name for file.
        """

        with open(self.log_dir / f"{name}.csv", "w") as f:
            # f.writelines(map(lambda i: f"{i}\n", values))
            for epoch, value in enumerate(values):
                f.write(f"{epoch} {value}\n")
