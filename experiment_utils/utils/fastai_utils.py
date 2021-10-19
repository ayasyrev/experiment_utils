from functools import partial

import kornia
import numpy as np
from fastai.basics import Learner
from fastai.callback.all import (ParamScheduler, SchedCos, SchedLin, SchedPoly,
                                 combine_scheds)
from fastcore.all import L


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


def fit_anneal_warmap(
    self: Learner, epochs, lr=None,
    pct_warmup=0., div_warmup=1, warmup_type='cos',
    pct_start=0.75, div_final=1e5, annealing_type='cos',
    cbs=None, reset_opt=False, wd=None, power=1
):
    "Fit `self.model` for `n_cycles` with warmap and annealing."
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
