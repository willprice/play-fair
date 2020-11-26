import torch
from torch import nn


def replace_last_linear(
    backbone: nn.Module, output_dim: int, init=True
) -> None:
    old_last_linear: nn.Linear = backbone.last_linear
    backbone.last_linear = nn.Linear(old_last_linear.in_features, output_dim)
    if init:
        torch.nn.init.kaiming_normal_(backbone.last_linear.weight)
        torch.nn.init.zeros_(backbone.last_linear.bias)