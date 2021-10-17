import csv
from pathlib import Path, PosixPath
from typing import List, Tuple, Union

from experiment_utils.utils.utils import print_stat, stat
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pt_utils.utils import flat_dict
from rich import print


def get_result_files(path: PosixPath) -> List[PosixPath]:
    return [fn for fn in path.iterdir() if fn.name.startswith('log_res')]


class Run:
    def __init__(self, path: Union[str, PosixPath]) -> None:
        self.path = Path(path)
        self.result_files = get_result_files(self.path)
        self.repeats = len(self.result_files)
        self._results = None
        self._metrics = None
        if self.repeats == 1:
            self.accuracy = read_accuracy_from_file(self.result_files[0])
        else:
            mean_fn = [fn for fn in self.path.iterdir() if fn.name.startswith('mean')]
            if len(mean_fn) == 0:  # if run stoped incorrectly, no file with results
                self.accuracy, self.std = calc_mean(self.result_files)
            else:
                self.accuracy, self.std = read_mean_from_file(mean_fn[0])

    def __repr__(self) -> str:
        if self.repeats > 1:
            std_str = f"repeats: {self.repeats}, std: {self.std:0.4f}"
        else:
            std_str = ''
        return f"acc: {self.accuracy:.2%}, path: {self.path.name}  {std_str}"

    def plot_lr(self) -> None:
        lrs = read_values_from_file(self.path / 'lrs.csv')
        plt.plot(lrs)

    def plot_metrics(self, metric: Union[str, List[str]] = None, stack=False) -> None:
        if metric is None:
            metric = self.metrics
        else:
            if type(metric) is str:
                metric = [metric]
            for m in metric:
                assert m in self.metrics
        results = []
        for result in self.results:
            values = {m: [] for m in metric}
            for item in result:
                for m in metric:
                    values[m].append(item[m])
            results.append(values)
        self._plot_results(metric, stack, results)

    def _plot_results(self, metric, stack, results):
        if stack:
            for num, result in enumerate(results):
                for m in metric:
                    plt.plot(result[m], label=f"{m}_{num}")
            plt.legend()
        else:
            fig, axs = plt.subplots(len(results), figsize=(5, 3 * len(results)))
            for num, ax in enumerate(axs):
                for metric in results[num]:
                    ax.plot(results[num][metric], label=metric)
                ax.legend()

    @property
    def results(self) -> List[List]:
        if self._results is None:
            self._results = [read_result_from_file(res_file) for res_file in self.result_files]
        return self._results

    @property
    def metrics(self) -> List[str]:
        if self._metrics is None:
            self._metrics = list(self.results[0][0].keys())
        return self._metrics

    @property
    def result(self) -> str:
        std = f"std {self.std:0.4f}" if self.repeats > 1 else ""
        return f"{self.accuracy:.2%} {std}"


def get_log_dirs(path: Union[str, List[str]]) -> List[PosixPath]:
    """Return list of dirs with logs.

    Args:
        path (Union[str, List[str]]): Path or list of pathes to log dirs.

    Returns:
        List[PosixPath]: List of dirs with results logs.
    """
    if type(path) is list:
        path_list = [Path(i) for i in path]
    else:
        path_list = [Path(path)]
    log_dirs = []
    for path in path_list:
        log_dirs.extend(fn.parent for fn in path.rglob('*cfg.yaml') if len(get_result_files(fn.parent)) > 0)
    log_dirs.sort(key=lambda x: x.stat().st_ctime)
    return log_dirs


def read_mean_from_file(filename: PosixPath) -> Tuple[float]:
    """Read mean, std from file with results."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    mean = float(lines[-2].strip())
    std = float(lines[-1].strip())
    return mean, std


def read_result_from_file(filename: Union[str, PosixPath]) -> List[dict]:
    """Read result from file."""
    res = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            res.append({key: float(value) for key, value in line.items()})
    return res


def read_values_from_file(filename: Union[str, PosixPath]) -> List[float]:
    """Read result from file."""
    values = []
    with open(filename, 'r') as f:
        for val in f.readlines():
            values.append(float(val))
    return values


def read_accuracy_from_file(filename: PosixPath) -> float:
    data = read_result_from_file(filename)
    return data[-1]['accuracy']


def calc_mean(filenames: List[PosixPath]) -> Tuple[float]:
    """Calculate mean from results in files."""
    res = []
    for fn in filenames:
        res.append(read_accuracy_from_file(fn))
    return stat(res)


def get_runs(path: Union[str, PosixPath], sort: bool = True, max_is_best: bool = True) -> List[Run]:
    log_dirs = get_log_dirs(path)
    runs = [Run(fn) for fn in log_dirs]
    if sort:
        runs.sort(key=lambda x: x.accuracy, reverse=max_is_best)
    return runs


def stat_runs(runs: List[Run]) -> Tuple[float]:
    print_stat([run.accuracy for run in runs])


def get_cfg(path: Union[str, PosixPath], flat: bool = False) -> Union[DictConfig, dict]:
    """Read cfg from path.

    Args:
        path (Union[str, PosixPath]): Directory name.
        flat (bool, optional): If True, return flattened dict. Defaults to False.

    Returns:
        Union[DictConfig, dict]: return config as OmegaConf DictConfig or as flattened dict.
    """
    cfg = OmegaConf.load(path / 'cfg.yaml')
    cfg.fn = str(path)
    if flat:
        cfg = flat_dict(cfg)
    return cfg


def print_runs(runs: List[Run], thresold: float = 0, limit: int =0) -> None:
    """Print results.

    Args:
        runs (List[Run]): List of runs.
        thresold (float, optional): Print only runs with accuracy more than thresold. Defaults to 0.
        limit (int, optional): Number of runs to print. Defaults to 0, print all.
    """
    len_runs = len(runs)
    if thresold:
        runs = [run for run in runs if run.accuracy > thresold]

    if len(runs) == 0:
        print(f"No run with thresold {thresold}")
    else:
        thresolded = f", {len(runs)} with acc > {thresold:.2%}" if thresold else ''
        if limit:
            lines_to_print = f", print limited to {limit}." if limit < len(runs) else ''
            runs = runs[:limit]
        else:
            lines_to_print = ''
        print(f"{len_runs} log dirs{thresolded}{lines_to_print}")

        max_path_name = max(len(run.path.name) for run in runs)

        for run in runs:
            std = f"rpts: {run.repeats}  std {run.std:0.4f}" if run.repeats > 1 else ''
            print(f"{run.accuracy:0.2%} . {run.path.name:{max_path_name}} {std}")


def rename_runs(runs: List[Run], thresold: float = 0) -> None:
    runs = [run for run in runs if run.accuracy > thresold]
    for run in runs:
        cfg = get_cfg(run.path)
        if cfg.run.repeat != run.repeats:
            print(f"not completed - {run.path} rep: {cfg.run.repeat}, res {len(run.results)}")
        else:
            if "__" not in run.path.name:
                new_name = run.path.parent / f"{run.path.name}__{int(run.accuracy * 10000)}"
                run.path.rename(new_name)
