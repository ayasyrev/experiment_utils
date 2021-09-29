import os
import random
from typing import List, Tuple

import torch
import numpy as np
from rich import print


def stat(results: List[float]) -> Tuple[float]:
    """Returm mean, std from list of floats."""
    stat = np.array(results)
    return stat.mean(), stat.std()


def print_stat(results: List[float]) -> None:
    """Print mean, std from list of floats."""
    mean, std = stat(results)
    print(f"mean: {mean:0.2%} std: {std:0.4f} min: {min(results):0.2%}, max: {max(results):0.2%}")


def set_seed(SEED_NUM: int = 2021,
             seed_pythonhash: bool = True,
             seed_random: bool = True,
             seed_numpy: bool = True,
             seed_torch: bool = True,
             torch_benchmark: bool = True,
             torch_deterministic: bool = False,
             ) -> None:
    """Set seeds.
        TODO: check https://pytorch.org/docs/stable/notes/randomness.html?highlight=deterministic
    """
    if seed_pythonhash:
        os.environ['PYTHONHASHSEED'] = str(SEED_NUM)
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
