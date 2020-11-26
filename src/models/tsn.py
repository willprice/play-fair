from typing import Callable

import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients
from torch import nn

from .components.consensus import AverageConsensus


class FeatureTSN(nn.Sequential):
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        input_relu: bool = True,
        dropout: float = 0,
        softmax: bool = False,
    ):
        super().__init__()
        self.input_relu = input_relu
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, output_dim)
        self.consensus = AverageConsensus()
        self.softmax = nn.Softmax(dim=-1) if softmax else None

    def features(self, xs):
        """

        Args:
            xs: tensor containing frame features of shape
             :math:`N \times T \times C` where :math:`C` is ``feature_dim``

        Returns:
            tensor of shape :math:`N \times T \times C'` where :math:`C'` is
            ``output_dim``
        """
        if self.input_relu:
            xs = F.relu(xs)
        xs = self.dropout(xs)
        xs = self.classifier(xs)
        return xs

    def logits(self, frame_scores):
        """

        Args:
            frame_scores: tensor containing class scores of shape
            :math:`N \times T \times C'`

        Returns:
            class score predictions of shape :math:`N \times C'`

        """
        frame_scores = self.consensus(frame_scores)
        if self.softmax is not None:
            frame_scores = self.softmax(frame_scores)
        return frame_scores

    def forward(self, net_features):
        return self.logits(self.features(net_features))

    def integrated_gradients(self, xs, labels, baselines):
        attributor = IntegratedGradients(self)
        # BS, T, C
        if baselines is not None and baselines.ndim == 2:
            baselines = baselines.unsqueeze(0)
        attributions = attributor.attribute(xs, target=labels, baselines=baselines)
        attributions = attributions.mean(dim=-1)
        return attributions

