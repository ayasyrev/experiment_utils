from fastai.losses import LabelSmoothingCrossEntropy
from fastai.optimizer import ranger

from .utils.fastai_utils import fit_warmup_anneal


loss_func_dict = {
    "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy,
}

opt_func_dict = {
    "ranger": ranger
}

train_func_dict = {
    "fit_warmup_anneal": fit_warmup_anneal
}
