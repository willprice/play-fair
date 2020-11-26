from dataclasses import dataclass
from typing import Tuple
from typing import Union

from torch import nn



class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def features(self, input):
        raise NotImplementedError()

    def logits(self, features):
        raise NotImplementedError()

    def forward(self, input):
        features = self.features(input)
        return self.logits(features)
