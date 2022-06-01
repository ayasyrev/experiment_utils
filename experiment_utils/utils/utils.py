import os
import random
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print


def stat(results: List[float]) -> Tuple[float]:
    """Return tuple(mean, std) from list of floats."""
    stat = np.array(results)
    return stat.mean(), stat.std()


def print_stat(results: List[float]) -> None:
    """Print mean, std from list of floats."""
    mean, std = stat(results)
    print(
        f"mean: {mean:0.2%} std: {std:0.4f} min: {min(results):0.2%}, max: {max(results):0.2%}"
    )


def set_seed(
    SEED_NUM: int = 42,
    seed_pythonhash: bool = True,
    seed_random: bool = True,
    seed_numpy: bool = True,
    seed_torch: bool = True,
    torch_benchmark: bool = True,
    torch_deterministic: bool = False,
    **kwargs,
) -> None:
    """Set seeds.
        TODO: check https://pytorch.org/docs/stable/notes/randomness.html?highlight=deterministic
    """
    # kwargs for compatibility with hydra.utils.call - can remove later.
    if seed_pythonhash:
        os.environ["PYTHONHASHSEED"] = str(SEED_NUM)
    if seed_random:
        random.seed(SEED_NUM)
    if seed_numpy:
        np.random.seed(SEED_NUM)
    if seed_torch:
        torch.manual_seed(SEED_NUM)
        torch.cuda.manual_seed(SEED_NUM)
        torch.cuda.manual_seed_all(SEED_NUM)

    torch.backends.cudnn.benchmark = torch_benchmark
    torch.backends.cudnn.deterministic = torch_deterministic


def no_seed(**kwargs) -> None:
    """Empty func - seed nothing"""
    pass


def show_cfg(cfg: DictConfig, resolve: bool = True) -> None:
    """Print Omegaconf config.

    Args:
        cfg (DictConfig): config to print
        resolve (bool, optional): Resolve or not. Defaults to True.
    """
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
