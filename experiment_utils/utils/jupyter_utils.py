from pathlib import Path, PosixPath
from typing import List, Union

from experiment_utils import __path__ as experiment_utils_path
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


experiment_utils_path = experiment_utils_path[0]


def initialize_config(
    overrides: List[str] = [],
    config_dir_name: str = "conf",
    config_path: Union[str, PosixPath] = None,
    config_name: str = "config",
) -> DictConfig:
    """Initialise Hydra config when work in jupyter notebook.

    Args:
        overrides (List[str], optional): List of overrides. Defaults to [].
        config_dir_name (str, optional): Name of directory with config structure.
            Defaults to "conf".
        config_path (Union[str, PosixPath], optional): Path to look for config dir.
            Defaults to None, take it from experiment_utils lib.
        config_name (str, optional): Name for config file. Defaults to "config".

    Returns:
        DictConfig: Initialized DictConfig.
    """
    if config_path is None:
        config_path = Path(experiment_utils_path)
    config_dir = (config_path / config_dir_name).absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
