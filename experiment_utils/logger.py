import csv
import time
from pathlib import Path
from typing import List

from fastai.basics import Learner
from omegaconf import DictConfig, OmegaConf

from experiment_utils.utils.utils import stat


def format_time(seconds: float, long: bool = True) -> str:
    "Format seconds to mm:ss, optional mm:ss.ms"
    seconds_int = int(seconds)
    min, sec = (seconds_int // 60) % 60, seconds_int % 60
    res = f"{min:02d}:{sec:02d}"
    if long:
        res = ".".join([res, f"{int((seconds - seconds_int) * 10)}"])
    return res


class Logger:
    """Log results"""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.log_dir = Path(cfg.run.log_dir)
        if self.log_dir.exists() and cfg.run.log_dir != ".":
            self.log_dir = Path(
                f"{cfg.run.log_dir}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        self.log_dir.mkdir(exist_ok=True, parents=True)
        with open(self.log_dir / "cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        self.results = []

    def start_job(self, learn: Learner, repeat: int) -> None:
        self.learn = learn
        self.repeat = repeat
        self.log_model()
        if self.cfg.repeat > 1:
            print(f"repeat #{repeat + 1} of {self.cfg.repeat}")
        self.start_time = time.time()

    def log_model(self):
        with open(self.log_dir / f"model_{self.learn.model._get_name()}.txt", "w") as f:
            f.write(str(self.learn.model))

    def log_job(self) -> None:
        acc = self.learn.recorder.final_record[-1]
        self.results.append(acc)
        print(f"acc: {acc:0.2%}")
        train_time = time.time() - self.start_time
        print(f"run time: {format_time(train_time)}")
        name_suffix = str(int(round(acc, 4) * 10000))
        if self.cfg.repeat > 1:
            name_suffix = f"{self.repeat}_{name_suffix}"
        self.log_result(name_suffix=name_suffix)
        if self.cfg.run.log_loss:
            self.log_values(self.learn.recorder.losses, f"losses_{name_suffix}")
        print(50 * "-")
        if self.cfg.model_save.model_save:
            fn = (
                f"{self.learn.model._get_name()}_{self.cfg.run.date_time}_{name_suffix}"
                if self.cfg.model_save.file_name == "model"
                else self.cfg.model_save.file_name
            )
            self.learn.save(fn, with_opt=self.cfg.model_save.with_opt)

    def log_run(self) -> None:
        if self.cfg.run.log_lr:
            if hasattr(self.learn.recorder, "hps"):
                for hp in self.learn.recorder.hps:
                    self.log_values(self.learn.recorder.hps[hp], f"{hp}s")
            else:
                self.log_values(self.learn.recorder.lrs, "lrs")
        if self.cfg.repeat > 1:
            self.log_resume()

    def log_result(self, file_name: str = "log_res", name_suffix: str = "") -> None:
        if name_suffix != "":
            name_suffix = "_" + name_suffix
        with open(self.log_dir / f"{file_name}{name_suffix}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.learn.recorder.metric_names[1:-1])
            writer.writerows(self.learn.recorder.values)

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
            f.writelines(map(lambda i: f"{i}\n", values))
