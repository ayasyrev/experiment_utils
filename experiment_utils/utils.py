from typing import List, Tuple
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
