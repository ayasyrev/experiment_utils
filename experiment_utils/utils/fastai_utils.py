from functools import partial
from pathlib import PosixPath
from typing import Callable, List, Union

import kornia
import numpy as np
from fastai.basics import (CategoryBlock, DataBlock, GrandparentSplitter,
                           Learner, get_image_files, parent_label)
from fastai.callback.all import (ParamScheduler, SchedCos, SchedLin, SchedPoly,
                                 combine_scheds)
from fastai.callback.schedule import (  # noqa F401 import lr_find for patch Learner
    SuggestionMethod, lr_find)
from fastai.data.core import DataLoaders
from fastai.vision.all import ImageBlock
from fastcore.all import L
from pt_utils.data.image_folder_dataset import ImageFolderDataset
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import set_image_backend
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader


def convert_MP_to_blurMP(model, layer_type_old):
    # conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            # layer_old = module
            layer_new = kornia.filters.MaxBlurPool2D(kernel_size=3, ceil_mode=True)
            model._modules[name] = layer_new

    return model


sched_dict = {
    'lin': SchedLin,
    'cos': SchedCos
}


def fit_anneal_warmup(
    self: Learner, epochs, lr=None,
    pct_warmup=0., div_warmup=1, warmup_type='cos',
    pct_start=0.75, div_final=1e5, annealing_type='cos',
    cbs=None, reset_opt=False, wd=None, power=1
):
    "Fit 'self.model' for 'n_cycles' with warmup and annealing."
    # will be deprecated in favor of renamed version - fit_warmup_anneal, than modified universal version
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)
    lr = np.array([h['lr'] for h in self.opt.hypers])

    if annealing_type == 'poly':
        anneal = partial(SchedPoly, power=power)
    else:
        anneal = sched_dict[annealing_type]
    if warmup_type == 'poly':
        warm = partial(SchedPoly, power=power)
    else:
        warm = sched_dict[warmup_type]
    pcts = [pct_warmup, pct_start - pct_warmup, 1 - pct_start]
    scheds = [warm(lr / div_warmup, lr), SchedLin(lr, lr), anneal(lr, lr / div_final)]
    scheds = {'lr': combine_scheds(pcts, scheds)}
    self.fit(epochs, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# renamed version of fit_anneal_warmup. Previos version leaved for compatibilites, will be removed later.
# Renamed argumets.
def fit_warmup_anneal(
    self: Learner, epochs, lr=None,
    warmup_pct=0., warmup_div=1, warmup_type='cos',
    anneal_pct=0.75, anneal_div=1e5, anneal_type='cos',
    cbs=None, reset_opt=False, wd=None, power=1
):
    """Fit 'self.model' for 'n_cycles' with warmup and annealing.
    default - no warmup and 'cos' annealing start at 0.75"""
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)
    lr = np.array([h['lr'] for h in self.opt.hypers])

    if anneal_type == 'poly':
        anneal = partial(SchedPoly, power=power)
    else:
        anneal = sched_dict[anneal_type]
    if warmup_type == 'poly':
        warm = partial(SchedPoly, power=power)
    else:
        warm = sched_dict[warmup_type]
    pcts = [warmup_pct, anneal_pct - warmup_pct, 1 - anneal_pct]
    scheds = [warm(lr / warmup_div, lr), SchedLin(lr, lr), anneal(lr, lr / anneal_div)]
    scheds = {'lr': combine_scheds(pcts, scheds)}
    self.fit(epochs, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


def lrfind(learn: Learner, num_it: int = 100, **kwargs):
    suggest_methods = ['valley', 'slide', 'minimum', 'steep']
    suggest_funcs = (
        SuggestionMethod.Valley,
        SuggestionMethod.Slide,
        SuggestionMethod.Minimum,
        SuggestionMethod.Steep)
    # result = learn.lr_find(suggest_funcs=suggest_funcs)
    learn.lr_find(num_it=num_it)
    lrs, losses = tensor(learn.recorder.lrs[num_it // 10:-5]), tensor(learn.recorder.losses[num_it // 10:-5])
    _suggestions = []
    for func in suggest_funcs:
        _suggestions.append(func(lrs, losses, num_it))
    lrs, points = [], []
    for lr, point in _suggestions:
        lrs.append(lr)
        points.append(point)

    print(20 * '-')
    print('Sugeested lrs:')
    # for num, res in enumerate(result):
    idx_list = []
    for (val, idx), name in zip(points, suggest_methods):
        # print(f"{suggest_methods[num]:10}: {res:0.6f}")
        print(f"{name:10}: {val:0.6f}")
        idx_list.append(float(idx))
    print(20 * '-')
    learn.recorder.final_record = [0]  # for compatibility wyth logger.
    learn.recorder.metric_names = [''] + suggest_methods + ['']
    learn.recorder.values = [[*lrs], [*idx_list]]


def fit(self: Learner, epochs, lr, cbs, reset_opt=False, wd=None):
    """Default Fit 'self.model' for 'n_cycles' with 'lr' using 'cbs'. Optionally 'reset_opt'.
    For run from script with hydra config"""
    self.fit(epochs, lr, cbs=L(cbs), reset_opt=reset_opt, wd=wd)


def get_dataloaders(ds_path, bs, num_workers,
                    item_tfms, batch_tfms):
    """Return dataloaders for fastai learner.
    As at fastai imagenette example.

    Args:
        ds_path (str): path to dataset.
        bs (int): batch_size.
        num_workers (int): number of workers for dataloader.
        item_tfms (llist): list of fastai transforms for batch execution (on gpu).

    Returns:
        fastai dataloaders
    """

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name='val'),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    return dblock.dataloaders(
        ds_path, path=ds_path, bs=bs, num_workers=num_workers)


def dls_from_pytorch(
    train_data_path: Union[str, PosixPath],
    val_data_path: Union[str, PosixPath],
    train_tfms: List,
    val_tfms: List,
    batch_size: int,
    num_workers: int,
    dataset_func: Callable = ImageFolderDataset,
    loader: Callable = default_loader,
    image_backend: str = 'pil',  # 'accimage'
    limit_dataset: Union[bool, int] = False,
    pin_memory: bool = True,
    shuffle: bool = True,
    shuffle_val: bool = False,
    drop_last: bool = True,
    drop_last_val: bool = False,
    persistent_workers: bool = False
):
    """Return fastai dataloaders created from pytorch dataloaders.

    Args:
        train_data_path (Union[str, PosixPath]): path for train data.
        val_data_path (Union[str, PosixPath]): path for validation data.
        train_tfms (List): List of transforms for train data.
        val_tfms (List): List of transforms for validation data
        batch_size (int): Batch size
        num_workers (int): Number of workers
        dataset_func (Callable, optional): Funtion or class to create dataset. Defaults to ImageFolderDataset.
        loader (Callable, optional): Function that load image. Defaults to default_loader.
        image_backend (str, optional): Image backand to use. Defaults to 'pil'.
        pin_memory (bool, optional): Use pin memory. Defaults to True.
        shuffle (bool, optional): Use shuffle for train data. Defaults to True.
        shuffle_val (bool, optional): Use shuffle for validation data. Defaults to False.
        drop_last (bool, optional): If last batch not full drop it or not. Defaults to True.
        drop_last_val (bool, optional): If last batch on validation data not full drop it or not. Defaults to False.
        persistent_workers (bool, optional): Use persistante workers. Defaults to False.

    Returns:
        fastai dataloaders
    """
    set_image_backend(image_backend)
    train_tfms = T.Compose(train_tfms)
    val_tfms = T.Compose(val_tfms)
    train_ds = dataset_func(root=train_data_path, transform=train_tfms, loader=loader, limit_dataset=limit_dataset)
    val_ds = dataset_func(root=val_data_path, transform=val_tfms, loader=loader, limit_dataset=limit_dataset)

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle,
                              drop_last=drop_last, persistent_workers=persistent_workers)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle_val,
                            drop_last=drop_last_val, persistent_workers=persistent_workers)
    return DataLoaders(train_loader, val_loader)
