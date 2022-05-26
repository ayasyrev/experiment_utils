import csv
from pathlib import Path, PosixPath
from typing import List, Tuple, Union

from experiment_utils.utils.utils import print_stat, show_cfg, stat
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pt_utils.utils import flat_dict
from rich import print


def get_result_files(path: PosixPath, name_pattern: str = "log_res") -> List[PosixPath]:
    """Return list of files for given name pattern.

    Args:
        path (PosixPath): Directory name
        name_pattern (str, optional): Pattern for name. Defaults to 'log_res'.

    Returns:
        List[PosixPath]: List of file names as PosixPath.
    """
    return [fn for fn in path.iterdir() if fn.name.startswith(name_pattern)]


class Run:
    def __init__(self, path: Union[str, PosixPath]) -> None:
        self.path = Path(path)
        self.result_files = get_result_files(self.path)
        self.repeats = len(self.result_files)
        self._results = None
        self._metrics = None
        self._cfg = None
        if self.repeats == 1:
            self.accuracy = read_accuracy_from_file(self.result_files[0])
        else:
            mean_fn = [fn for fn in self.path.iterdir() if fn.name.startswith("mean")]
            if len(mean_fn) == 0:  # if run stopped incorrectly, no file with results
                self.accuracy, self.std = calc_mean(self.result_files)
            else:
                self.accuracy, self.std = read_mean_from_file(mean_fn[0])

    def __repr__(self) -> str:
        if self.repeats > 1:
            std_str = f"repeats: {self.repeats}, std: {self.std:0.4f}"
        else:
            std_str = ""
        return f"acc: {self.accuracy:.2%}, path: {self.path.name}  {std_str}"

    def plot_lr(self) -> None:
        lrs = read_values_from_file(self.path / "lrs.csv")
        plt.plot(lrs)

    def lr_find(self, skip_end=5) -> None:
        lrs = read_values_from_file(self.path / "lrs.csv")[:-skip_end]
        loss_fn = get_result_files(self.path, name_pattern="losses")[0]
        losses = read_values_from_file(loss_fn)[:-skip_end]
        res = read_result_from_file(self.path / self.result_files[0])
        suggests = list(res[0].keys())
        points = [(res[0][name], res[1][name]) for name in suggests]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        print("Suggested lrs:")
        for suggest, point in zip(suggests, points):
            print(f"{suggest:10} {point[0]:0.6f}")
        fig, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        for (val, idx), name, color in zip(points, suggests, colors):
            ax.plot(val, idx, "ro", label=name, c=color)
        ax.legend()

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
            self._results = [
                read_result_from_file(res_file) for res_file in self.result_files
            ]
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

    @property
    def cfg(self) -> DictConfig:
        if self._cfg is None:
            self._cfg = get_cfg(self.path)
        return self._cfg

    def show_cfg(self) -> None:
        show_cfg(self.cfg)


def get_log_dirs(path: Union[str, List[str]]) -> List[PosixPath]:
    """Return list of dirs with logs.

    Args:
        path (Union[str, List[str]]): Path or list of path's to log dirs.

    Returns:
        List[PosixPath]: List of dirs with results logs.
    """
    if type(path) is list:
        path_list = [Path(i) for i in path]
    else:
        path_list = [Path(path)]
    log_dirs = []
    for path in path_list:
        log_dirs.extend(
            fn.parent
            for fn in path.rglob("*cfg.yaml")
            if len(get_result_files(fn.parent)) > 0
        )
    log_dirs.sort(key=lambda x: x.stat().st_ctime)
    return log_dirs


def read_mean_from_file(filename: PosixPath) -> Tuple[float]:
    """Read mean, std from file with results."""
    with open(filename, "r") as f:
        lines = f.readlines()
    try:
        mean = float(lines[-2].strip())
        std = float(lines[-1].strip())
    except Exception:
        print(Exception)
        print(filename)
        mean, std = 0, 0
    return mean, std


def read_result_from_file(filename: Union[str, PosixPath]) -> List[dict]:
    """Read result from file."""
    res = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            res.append({key: float(value) for key, value in line.items()})
    return res


def read_values_from_file(filename: Union[str, PosixPath]) -> List[float]:
    """Read result from file."""
    values = []
    with open(filename, "r") as f:
        for val in f.readlines():
            values.append(float(val))
    return values


def read_accuracy_from_file(filename: PosixPath) -> float:
    data = read_result_from_file(filename)
    try:
        accuracy = data[-1]["accuracy"]
    except:  # noqa E722 todo add exception
        accuracy = 0
    return accuracy


def calc_mean(filenames: List[PosixPath]) -> Tuple[float]:
    """Calculate mean from results in files."""
    res = []
    for fn in filenames:
        res.append(read_accuracy_from_file(fn))
    return stat(res)


def get_runs(
    path: Union[str, PosixPath], sort: bool = True, max_is_best: bool = True
) -> List[Run]:
    """Read Runs from logs.

    Args:
        path (Union[str, PosixPath]): Path for log dirs.
        sort (bool, optional): Sort list with result or not.. Defaults to True.
        max_is_best (bool, optional): Criteria for sort. Defaults to True, max is best.

    Returns:
        List[Run]: List of Runs.
    """
    log_dirs = get_log_dirs(path)
    runs = [Run(fn) for fn in log_dirs]
    sorted_runs = sorted(runs, key=lambda x: x.accuracy, reverse=max_is_best)
    if sort:
        return sorted_runs
    for num, run in enumerate(sorted_runs):
        run.num = num
    runs.sort(key=lambda run: run.path.stat().st_ctime, reverse=False)
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
    cfg = OmegaConf.load(path / "cfg.yaml")
    cfg.fn = str(path)
    if flat:
        cfg = flat_dict(cfg)
    return cfg


def filter_runs(runs: List[Run], threshold: float = 0) -> List[Run]:
    """Filter runs for run with accuracy less than threshold

    Args:
        runs (List[Run]): List of Runs.
        threshold (float, optional): Threshold for filter. Defaults to 0.

    Returns:
        List[Run]: List of runs.
    """
    return [run for run in runs if run.accuracy > threshold]


def print_runs(
    runs: List[Run],
    header: str = None,
    limit: int = 0,
    print_num: bool = False,
    print_parent: bool = False,
) -> None:
    """Print results.
    """
    if limit:
        lines_to_print = f", print limited to {limit}." if limit < len(runs) else ""
        runs = runs[:limit]
    else:
        lines_to_print = ""

    print(f"{header}{lines_to_print}")
    max_path_name = max(len(run.path.name) for run in runs)

    for run in runs:
        std = f"rpts: {run.repeats}  std {run.std:0.4f}" if run.repeats > 1 else ""
        if print_num and hasattr(run, "num"):
            num = f". #{run.num}"
        else:
            num = ""
        print_parent = run.path.parts[-2] if print_parent else ""
        print(
            f"{run.accuracy:0.2%} . {run.path.name:{max_path_name}} {num} {std} {print_parent}"
        )


def rename_runs(runs: List[Run], threshold: float = 0) -> None:
    runs = [run for run in runs if run.accuracy > threshold]
    for run in runs:
        cfg = get_cfg(run.path)
        if cfg.repeat != run.repeats:
            print(
                f"not completed - {run.path} rep: {cfg.repeat}, res {len(run.results)}"
            )
        else:
            if "__" not in run.path.name:
                new_name = run.path.parent / f"{run.path.name}__{int(round(run.accuracy, 4) * 10000)}"
                run.path.rename(new_name)
