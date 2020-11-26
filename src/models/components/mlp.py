from itertools import chain
from typing import List, Any

import torch
from torch import nn
import torch.nn.functional as F


def _strip_nones(ls: List) -> List:
    return [x for x in ls if x is not None]


def _flatten(ls: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(ls))


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_layers: int,
    batch_norm: bool = False,
    dropout: float = 0,
    input_relu: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        *_strip_nones(
            [
                nn.ReLU(inplace=True) if input_relu else None,
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim, bias=not batch_norm),
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(inplace=True),
            ]
            + _flatten(
                [
                    [
                        nn.Linear(hidden_dim, hidden_dim, bias=not batch_norm),
                        nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                        nn.ReLU(inplace=True),
                    ]
                    for _ in range(hidden_layers - 1)
                ]
            )
            + [
                nn.Linear(hidden_dim, output_dim),
            ]
        )
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 1,
        dropout: float = 0,
        batch_norm: bool = False,
        input_relu: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = make_mlp(
            input_dim,
            hidden_dim,
            output_dim,
            hidden_layers,
            batch_norm,
            dropout,
            input_relu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim, (
            f"Expected input to be {self.input_dim}D "
            f"but was {x.shape[-1]}D ("
            f"shape {x.shape})"
        )

        x = self.model(x)

        assert x.shape[-1] == self.output_dim
        return x


class MLPConsensus(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (N, T, C) format -> (N, T*C)
            x = x.reshape((x.shape[0], -1))
        assert x.dim() == 2, f"Expected input to be 2/3D but was of shape {x.shape}"
        assert x.shape[-1] == self.input_dim, (
            f"Expected last dim ({x.shape[-1]}) to "
            f"match input dim ({self.input_dim})"
        )

        x = self.model(x)

        assert x.shape[-1] == self.output_dim
        return x
