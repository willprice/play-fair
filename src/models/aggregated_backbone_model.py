import torch
from torch import nn

from .types import Model


class AggregatedBackboneModel(Model):
    def __init__(self, backbone: nn.Module, temporal_module: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.temporal_module = temporal_module
        self.input_order = "TCHW"

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from batch video tensor.

        Args:
            inputs: Video tensor of shape :math:`(N, T, C, H, W)`

        Returns:
            features of shape :math:`(N, T, ...)` where the remaining dimensions
            are those of the backbone model's features.

        """
        assert inputs.dim() == 5
        batch_size, time_dim = inputs.shape[:2]
        # To support 2D backbones we collapse the time dimension into the batch
        # dimension to get a backbone output for each frame in each batch item.
        backbone_outputs = self.backbone(
                inputs.view((-1,) + inputs.shape[2:])
        )
        # We restore the temporal dimension to be processed by the temporal module.
        backbone_outputs = backbone_outputs.view((batch_size, time_dim) +
                                                 backbone_outputs.shape[1:])
        return backbone_outputs

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Produce logits from batch features

        Args:
            features: A feature tensor of shape :math:`(N, T, ...)` where the remaining
            dimensions are those of the backbone model's features.

        Returns:
            logit tensor of shape :math:`(N, C)`

        """
        return self.temporal_module(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: video tensor of shape :math:`(N, T, C, H, W)`.

        Returns:
            :math:`(N, C)` output
        """
        features = self.features(inputs)
        return self.logits(features)
