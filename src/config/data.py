from pathlib import Path
from typing import Union

import h5py
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset as TorchDataset

from datasets.something_something import (
    SomethingSomethingV2Dataset,
    SomethingSomethingV2FeaturesDataset,
)
from .base import ClassConfig
from .sampling import FrameSamplersConfig
from .transform import TransformsConfig


__all__ = [
    "SomethingSomethingV2DatasetConfig",
    "SomethingSomethingV2FeaturesDatasetConfig",
    "FeaturesDataConfig",
    "RGBDataConfig",
]


class SomethingSomethingV2DatasetConfig(ClassConfig):
    kind: str = Field("SomethingSomethingV2Dataset", const=True)
    root: Path
    class_count: int

    def instantiate(self):
        return SomethingSomethingV2Dataset(self.root)


class SomethingSomethingV2FeaturesDatasetConfig(ClassConfig):
    kind: str = Field("SomethingSomethingV2FeaturesDataset", const=True)
    path: Path
    class_count: int
    in_memory: bool = False

    def instantiate(self):
        h5_file = h5py.File(self.path, mode="r", swmr=True)
        return SomethingSomethingV2FeaturesDataset(h5_file, in_mem=self.in_memory)


class RGBDataConfig(BaseModel):
    frame_samplers: FrameSamplersConfig
    transform: TransformsConfig
    dataset: SomethingSomethingV2DatasetConfig


class FeaturesDataConfig(BaseModel):
    frame_samplers: FrameSamplersConfig
    dataset: SomethingSomethingV2FeaturesDatasetConfig


class DataloaderConfig(BaseModel):
    batch_size: int = 1
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False

    def get_dataloader(self, dataset: TorchDataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            **kwargs,
        )
