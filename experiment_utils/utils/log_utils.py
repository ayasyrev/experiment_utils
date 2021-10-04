import csv
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich import print

# from torch.utils.tensorboard import SummaryWriter
from pt_utils.utils import flat_dict, clear_dict


def get_log_dirs(path: Union[str, List[str]]):
    if type(path) is list:
        path_list = [Path(i) for i in path]
    else:
        path_list =[Path(path)]
    log_dirs = []
    for path in path_list:
        log_dirs.extend(fn.parent for fn in path.rglob('*.hydra*'))
    return log_dirs


def get_cfg(fn):
    cfg = OmegaConf.load(fn / 'cfg.yaml')
    cfg.fn = str(fn)
    return flat_dict(cfg)


def get_hparams(cfg, unics):
    hparams = {k: cfg.get(k, 'None') for k in unics}
    return hparams



def get_result(cfg):
    log_path = Path(cfg['fn'])
    files = [fn for fn in log_path.iterdir() if fn.name.startswith('log_res')]
#         if len(files) > 0:
#     print(res_file)
    data = read_result(files[0])
    return data


def get_acc(cfg):
    log_path = Path(cfg['fn'])
    files = [fn for fn in log_path.iterdir() if fn.name.startswith('log_res')]
    if len(files) > 0:
#         if len(files) > 0:
#     print(res_file)
        data = read_result(files[0])
        acc = float(data[-1]['accuracy'])
    else:
        acc = 0
    return acc

def find_results(log_dir, thresold=0.8, sort=False, max_is_best=True):
    log_dirs = get_log_dirs(log_dir)
#     log_dirs.sort()
    log_dirs.sort(key=lambda x: int(x.name.split('_')[0]))
    res = []
    for path in log_dirs:
        acc = get_acc(get_cfg(path))
        if acc > thresold:
            res.append((acc, path))
#         if not '__' in path.name:
#             new_name = f"{path.name}__{int(acc*10000)}"
#             print(new_name)
    if sort:
        res.sort(key=lambda x: x[0], reverse=max_is_best)
    return res

def rename_from_res(acc_path: tuple):
       for (acc, path) in acc_path:
        if not '__' in path.name:
            new_name = path.parent / f"{path.name}__{int(acc*10000)}"
            path.rename(new_name)


def rename_logdirs(log_dir, thresold=0.8):
    acc_path = find_results(log_dir, thresold)
    rename_from_res(acc_path)


def print_result(res: tuple, sort=True, max_is_best=True):
    
#     res_to_print = sorted(res, key=lambda x: x[0], reverse=max_is_best) if sort else res
    res_to_print = sorted(res, key=lambda x: x[0], reverse=max_is_best) if sort else res
#         res.sort(key=lambda x: x[0], reverse=max_is_best)
    for acc, path in res_to_print:
        print(f"{acc:0.2%} {path.name}")


def read_result(fn):
    res = []
    with open(fn, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            res.append(line)
    return res

