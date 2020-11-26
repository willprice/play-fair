import torch
from torch import nn


class ModelFeatureExtractorWrapper(nn.Module):
    """
    To enable nn.DataParallel feature extraction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.features(inputs)