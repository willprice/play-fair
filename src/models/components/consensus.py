import torch.nn
from torch import nn


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, input_tensor):
        if self.consensus_type == "avg":
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == "identity":
            output = input_tensor
        else:
            raise NotImplementedError("Only avg and identity consensus implemented")
        return output


class AverageConsensus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape :math:`(N, T, C)`

        Returns:
            Input tensor averaged over the time dimension of shape :math:`(N, C)`
        """
        assert x.dim() == 3
        return x.mean(dim=1)


class ClassifierConsensus(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_relu: bool = True,
                 dropout: float = 0):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU() if input_relu else None
        self.dropout = nn.Dropout(dropout)
        self.consensus = AverageConsensus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.relu is not None:
            x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return self.consensus(x)


