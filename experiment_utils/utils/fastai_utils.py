from functools import partial

import kornia
import numpy as np
from fastai.basics import (CategoryBlock, DataBlock, GrandparentSplitter,
                           Learner, get_image_files, parent_label)
from fastai.callback.all import (ParamScheduler, SchedCos, SchedLin, SchedPoly,
                                 combine_scheds)
from fastai.callback.schedule import (  # noqa F401 import lr_find for patch Learner
    SuggestionMethod, lr_find)
from fastai.vision.all import ImageBlock
from fastcore.all import L
from torch import tensor


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
