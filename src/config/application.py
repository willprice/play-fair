from typing import Any, Callable, Tuple, Union

import torch
from pydantic import BaseModel

from config.data import SomethingSomethingV2DatasetConfig, SomethingSomethingV2FeaturesDatasetConfig
from config.model import FeatureMultiscaleNetworkConfig, FeatureTRNConfig, \
    FeatureTSNConfig, TRNConfig, TSNConfig
from config.sampling import FrameSamplersConfig
from config.transform import TransformsConfig
from datasets.base import Dataset
from models.builder import make_feature_trn, make_feature_tsn, make_trn, make_tsn
from transforms.builder import get_transforms


class RGBConfig(BaseModel):
    frame_samplers: FrameSamplersConfig
    transform: TransformsConfig
    dataset: SomethingSomethingV2DatasetConfig
    model: Union[TRNConfig, TSNConfig]

    def get_transforms(self) -> Tuple[Callable[[Any], torch.Tensor], Callable[[Any], torch.Tensor]]:
        return get_transforms(self.transform, self.model.backbone_settings)

    def get_model(self):
        return self.model.instantiate()

    def get_dataset(self) -> Dataset:
        return self.dataset.instantiate()

    def get_train_frame_sampler(self):
        return self.frame_samplers.train.instantiate()

    def get_test_frame_sampler(self):
        return self.frame_samplers.test.instantiate()


class FeatureConfig(BaseModel):
    frame_samplers: FrameSamplersConfig
    dataset: SomethingSomethingV2FeaturesDatasetConfig
    model: Union[FeatureTRNConfig, FeatureTSNConfig, FeatureMultiscaleNetworkConfig]

    def get_model(self):
        return self.model.instantiate()

    def get_dataset(self) -> Dataset:
        return self.dataset.instantiate()

    def get_train_frame_sampler(self):
        return self.frame_samplers.train.instantiate()

    def get_test_frame_sampler(self):
        return self.frame_samplers.test.instantiate()

