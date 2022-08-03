import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich import print


def stat(results: List[float]) -> Tuple[float, float]:
    """Return tuple(mean, std) from list of floats."""
    stat = np.array(results)
    return stat.mean(), stat.std()


def print_stat(results: List[float]) -> None:
    """Print mean, std from list of floats."""
    mean, std = stat(results)
    print(
        f"mean: {mean:0.2%} std: {std:0.4f} min: {min(results):0.2%}, max: {max(results):0.2%}"
    )


@dataclass
class Cfg_Seed:
    SEED_NUM: int = 42
    seed_pythonhash: bool = True
    seed_random: bool = True
    seed_numpy: bool = True
    seed_torch: bool = True
    torch_benchmark: bool = True
    torch_deterministic: bool = False


def set_seed(
    cfg: Optional[Union[Cfg_Seed, DictConfig]] = None,
    SEED_NUM: int = 42,
    seed_pythonhash: bool = True,
    seed_random: bool = True,
    seed_numpy: bool = True,
    seed_torch: bool = True,
    torch_benchmark: bool = True,
    torch_deterministic: bool = False,
    **kwargs,
) -> None:
    """Set seeds. Use with cfg or arguments. If cfg is used - arguments ignorings.
        TODO: check https://pytorch.org/docs/stable/notes/randomness.html?highlight=deterministic
    """
    # kwargs for compatibility with hydra.utils.call - can remove later.
    if cfg is None:
        cfg = Cfg_Seed(SEED_NUM, seed_pythonhash, seed_random, seed_numpy, seed_torch, torch_benchmark, torch_deterministic)
    if cfg.seed_pythonhash:
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED_NUM)
    if cfg.seed_random:
        random.seed(cfg.SEED_NUM)
    if cfg.seed_numpy:
        np.random.seed(cfg.SEED_NUM)
    if cfg.seed_torch:
        torch.manual_seed(cfg.SEED_NUM)
        torch.cuda.manual_seed(cfg.SEED_NUM)
        torch.cuda.manual_seed_all(cfg.SEED_NUM)

    torch.backends.cudnn.benchmark = cfg.torch_benchmark
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    # torch.use_deterministic_algorithms()


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
