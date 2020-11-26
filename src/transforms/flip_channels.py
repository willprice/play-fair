import torch
from torchvideo import transforms


class FlipChannels(transforms.StatelessTransform[torch.Tensor, torch.Tensor]):
    def __init__(self, channel_dim: int = 0):
        self.channel_dim = channel_dim

    def _transform(
        self, frames: torch.Tensor, params: None
    ) -> torch.Tensor:
        return torch.flip(frames, (self.channel_dim,))

    def __repr__(self):
        return self.__class__.__name__ + f"(channel_dim={self.channel_dim})"
