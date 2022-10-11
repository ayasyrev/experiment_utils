# import torch
import torch.nn as nn

from model_constructor import ModelConstructor


xresnet18 = ModelConstructor(
    name="xResNet18",
    num_classes=10,
    block_sizes=[64, 128, 256, 512],
    layers=[2, 2, 2, 2],
    act_fn=nn.ReLU()
)
