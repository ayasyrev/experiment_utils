from functools import partial
from pathlib import PosixPath
from typing import Callable, List, Union

import hydra
import numpy as np
import torch
from experiment_utils.utils.hydra_utils import instantiate_model
from fastai.basics import (CategoryBlock, DataBlock, GrandparentSplitter,
                           Learner, accuracy, get_image_files, parent_label,
                           tensor)
from fastai.callback.all import (Callback, MixUp, ParamScheduler, SchedCos,
                                 SchedLin, SchedPoly, combine_scheds)
from fastai.callback.schedule import (  # noqa F401 import lr_find for patch Learner
    SuggestionMethod, lr_find)
from fastai.data.core import DataLoaders
from fastai.vision.all import ImageBlock, Normalize, imagenet_stats
from fastcore.all import L
from omegaconf import DictConfig
from pt_utils.data.image_folder_dataset import ImageFolderDataset
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader
from torchvision import set_image_backend
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader


def convert_MP_to_blurMP(model, layer_type_old):
    import kornia

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


sched_dict = {"lin": SchedLin, "cos": SchedCos}


def fit_anneal_warmup(
    self: Learner,
    epochs,
    lr=None,
    pct_warmup=0.0,
    div_warmup=1,
    warmup_type="cos",
    pct_start=0.75,
    div_final=1e5,
    annealing_type="cos",
    cbs=None,
    reset_opt=False,
    wd=None,
    power=1,
):
    "Fit 'self.model' for 'n_cycles' with warmup and annealing."
    # will be deprecated in favor of renamed version - fit_warmup_anneal, than modified universal version
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper("lr", self.lr if lr is None else lr)
    lr = np.array([h["lr"] for h in self.opt.hypers])

    if annealing_type == "poly":
        anneal = partial(SchedPoly, power=power)
    else:
        anneal = sched_dict[annealing_type]
    if warmup_type == "poly":
        warm = partial(SchedPoly, power=power)
    else:
        warm = sched_dict[warmup_type]
    pcts = [pct_warmup, pct_start - pct_warmup, 1 - pct_start]
    scheds = [warm(lr / div_warmup, lr), SchedLin(lr, lr), anneal(lr, lr / div_final)]
    scheds = {"lr": combine_scheds(pcts, scheds)}
    self.fit(epochs, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# renamed version of fit_anneal_warmup. Previous version leaved for compatibilities, will be removed later.
# Renamed arguments.
def fit_warmup_anneal(
    self: Learner,
    epochs,
    lr=None,
    warmup_pct=0.0,
    warmup_div=1,
    warmup_type="cos",
    anneal_pct=0.75,
    anneal_div=1e5,
    anneal_type="cos",
    cbs=None,
    reset_opt=False,
    wd=None,
    power=1,
):
    """Fit 'self.model' for 'n_cycles' with warmup and annealing.
    default - no warmup and 'cos' annealing start at 0.75"""
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper("lr", self.lr if lr is None else lr)
    lr = np.array([h["lr"] for h in self.opt.hypers])

    if anneal_type == "poly":
        anneal = partial(SchedPoly, power=power)
    else:
        anneal = sched_dict[anneal_type]
    if warmup_type == "poly":
        warm = partial(SchedPoly, power=power)
    else:
        warm = sched_dict[warmup_type]
    pcts = [warmup_pct, anneal_pct - warmup_pct, 1 - anneal_pct]
    scheds = [warm(lr / warmup_div, lr), SchedLin(lr, lr), anneal(lr, lr / anneal_div)]
    scheds = {"lr": combine_scheds(pcts, scheds)}
    self.fit(epochs, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


def lrfind(learn: Learner, num_it: int = 100, **kwargs):
    suggest_methods = ["valley", "slide", "minimum", "steep"]
    suggest_funcs = (
        SuggestionMethod.Valley,
        SuggestionMethod.Slide,
        SuggestionMethod.Minimum,
        SuggestionMethod.Steep,
    )
    # result = learn.lr_find(suggest_funcs=suggest_funcs)
    learn.lr_find(num_it=num_it)
    lrs, losses = (
        tensor(learn.recorder.lrs[num_it // 10:-5]),
        tensor(learn.recorder.losses[num_it // 10:-5]),
    )
    _suggestions = [func(lrs, losses, num_it) for func in suggest_funcs]
    lrs, points = [], []
    for lr, point in _suggestions:
        lrs.append(lr)
        points.append(point)

    print(20 * "-")
    print("Suggested lrs:")
    # for num, res in enumerate(result):
    idx_list = []
    for (val, idx), name in zip(points, suggest_methods):
        # print(f"{suggest_methods[num]:10}: {res:0.6f}")
        print(f"{name:10}: {val:0.6f}")
        idx_list.append(float(idx))
    print(20 * "-")
    learn.recorder.final_record = [0]  # for compatibility with logger.
    learn.recorder.metric_names = [""] + suggest_methods + [""]
    learn.recorder.values = [[*lrs], [*idx_list]]


def fit(self: Learner, epochs, lr, cbs, reset_opt=False, wd=None):
    """Default Fit 'self.model' for 'n_cycles' with 'lr' using 'cbs'. Optionally 'reset_opt'.
    For run from script with hydra config"""
    self.fit(epochs, lr, cbs=L(cbs), reset_opt=reset_opt, wd=wd)


def get_dataloaders(ds_path, bs, num_workers, item_tfms, batch_tfms):
    """Return dataloaders for fastai learner.
    As at fastai imagenette example.

    Args:
        ds_path (str): path to dataset.
        bs (int): batch_size.
        num_workers (int): number of workers for dataloader.
        item_tfms (list): list of fastai transforms for batch execution (on gpu).

    Returns:
        fastai dataloaders
    """
    batch_tfms = [Normalize.from_stats(*imagenet_stats)].extend(batch_tfms)
    # batch_tfms = [Normalize.from_stats(*imagenet_stats)] + batch_tfms

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name="val"),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return dblock.dataloaders(ds_path, path=ds_path, bs=bs, num_workers=num_workers)


def dls_from_pytorch(
    train_data_path: Union[str, PosixPath],
    val_data_path: Union[str, PosixPath],
    train_tfms: List,
    val_tfms: List,
    batch_size: int,
    num_workers: int,
    dataset_func: Callable = ImageFolderDataset,
    loader: Callable = default_loader,
    image_backend: str = "pil",  # 'accimage'
    limit_dataset: Union[bool, int] = False,
    pin_memory: bool = True,
    shuffle: bool = True,
    shuffle_val: bool = False,
    drop_last: bool = True,
    drop_last_val: bool = False,
    persistent_workers: bool = False,
):
    """Return fastai dataloaders created from pytorch dataloaders.

    Args:
        train_data_path (Union[str, PosixPath]): path for train data.
        val_data_path (Union[str, PosixPath]): path for validation data.
        train_tfms (List): List of transforms for train data.
        val_tfms (List): List of transforms for validation data
        batch_size (int): Batch size
        num_workers (int): Number of workers
        dataset_func (Callable, optional): Function or class to create dataset. Defaults to ImageFolderDataset.
        loader (Callable, optional): Function that load image. Defaults to default_loader.
        image_backend (str, optional): Image backend to use. Defaults to 'pil'.
        pin_memory (bool, optional): Use pin memory. Defaults to True.
        shuffle (bool, optional): Use shuffle for train data. Defaults to True.
        shuffle_val (bool, optional): Use shuffle for validation data. Defaults to False.
        drop_last (bool, optional): If last batch not full drop it or not. Defaults to True.
        drop_last_val (bool, optional): If last batch on validation data not full drop it or not. Defaults to False.
        persistent_workers (bool, optional): Use persistance workers. Defaults to False.

    Returns:
        fastai dataloaders
    """
    set_image_backend(image_backend)
    train_tfms = T.Compose(train_tfms)
    val_tfms = T.Compose(val_tfms)
    train_ds = dataset_func(
        root=train_data_path,
        transform=train_tfms,
        loader=loader,
        limit_dataset=limit_dataset,
    )
    val_ds = dataset_func(
        root=val_data_path,
        transform=val_tfms,
        loader=loader,
        limit_dataset=limit_dataset,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_val,
        drop_last=drop_last_val,
        persistent_workers=persistent_workers,
    )
    return DataLoaders(train_loader, val_loader)


class MixUpScheduler(Callback):
    "Schedule mixup"
    order, run_valid = 60, False

    def __init__(
        self,
        epochs: int,
        start_epoch: int = 4,
        end_pct: int = 0.75,
        alpha_start: float = 0.05,
        alpha: float = 0.2,
        steps: int = 0,
    ):
        self.start_epoch = start_epoch
        end_epoch = int(epochs * end_pct) - 1
        assert (
            end_epoch >= start_epoch
        ), f"start epoch for mixup {start_epoch} > than end_epoch {end_epoch}"
        if steps == 0:
            steps = end_epoch - start_epoch
            epoch_step = 1
        else:
            epoch_step = int((end_epoch - start_epoch) / steps)
        alpha_step = (alpha - alpha_start) / steps
        print(
            f"mixup start at ep {start_epoch}, {steps} steps, alpha step: {alpha_step}"
        )
        # self.sched = {step: (alpha_start + num * alpha_step)
        #               for num, step in enumerate(range(start_epoch + 1, end_epoch + 1), start=1)}
        self.sched = {
            step: (alpha_start + num * alpha_step)
            for num, step in enumerate(
                range(start_epoch + epoch_step, end_epoch, epoch_step), start=1
            )
        }
        self.events_epochs = list(self.sched.keys())
        # first_epoch = self.events_epochs.pop(0)
        # self.start_at = first_epoch
        self.mixup = MixUp(alpha_start)
        # self.set_alpha(alpha_step)
        print(self.sched)

    def before_fit(self):
        pass

    def after_epoch(self):
        # print(f"end ep {self.learn.epoch}")
        if self.learn.epoch + 1 == self.start_epoch:
            self.set_mixup()
        if self.learn.epoch + 1 in self.events_epochs:
            self.set_alpha(self.sched[self.learn.epoch + 1])
        # if self.learn.epoch + 1 != self.learn.n_epoch:
        #     self.set_alpha(self.sched[self.learn.epoch + 1])

    def set_mixup(self):
        self.learn.add_cb(self.mixup)
        print("set mixup")

    def set_alpha(self, alpha):
        print(f"set alpha {alpha}")
        self.mixup.distrib = Beta(tensor(alpha), tensor(alpha))


def get_learner(cfg: DictConfig) -> Learner:
    """Return fastai Learner from cfg"""

    model = instantiate_model(cfg)
    if cfg.model_load.model_load:
        load_model_state(model, cfg)

    opt_fn = hydra.utils.call(cfg.opt_fn)
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    dls = hydra.utils.instantiate(cfg.dls)

    learn = Learner(
        dls=dls, model=model, opt_func=opt_fn, metrics=[accuracy], loss_func=loss_fn
    )

    # if cfg.model_load.model_load:
    #     learn.load(file=cfg.model_load.file_name, with_opt=cfg.model_load.with_opt)
    return learn


def load_model_state(model: torch.nn.Module, cfg: DictConfig) -> None:
    model_state = model.state_dict()
    loaded_state = torch.load(f"{cfg.model_load.model_path}{cfg.model_load.file_name}.pt")
    loaded_state_keys = loaded_state.keys()
    missed_keys = [key for key in model_state.keys() if key not in loaded_state_keys]
    # if missed_keys:
    print(f"Missed keys: {missed_keys}")
    for key in missed_keys:
        loaded_state[key] = model_state[key]
    wrong_shape = []
    for key in loaded_state.keys():
        if loaded_state[key].shape != model_state[key].shape:
            wrong_shape.append(key)
            loaded_state[key] = loaded_state[key].view_as(model_state[key])
    if wrong_shape:
        print(f"changed {len(wrong_shape)} modules.")
    if cfg.model_load.se_name:
        se_state = torch.load(f"{cfg.model_load.model_path}{cfg.model_load.se_name}.pt")
        for key in se_state.keys():
            loaded_state[key] = se_state[key].view_as(model_state[key])
        print(f"loaded {len(se_state)} se weights {cfg.model_load.se_name}")
    model.load_state_dict(loaded_state)
