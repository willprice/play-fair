from abc import ABC
from typing import Dict

from torch.utils.data import Dataset as TorchDataset


class Dataset(ABC):
    def __init__(self, data_type: str):
        assert data_type in ["RGB", "features"]
        self.data_type = data_type

    def class_count(self) -> int:
        raise NotImplementedError()

    def train_dataset(self, *args, **kwargs) -> TorchDataset:
        raise NotImplementedError()

    def validation_dataset(self, *args, **kwargs) -> TorchDataset:
        raise NotImplementedError()

    def class2str(self) -> Dict[int, str]:
        raise NotImplementedError()

    def str2class(self) -> Dict[str, int]:
        raise NotImplementedError()
