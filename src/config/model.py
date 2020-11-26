from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from config.base import ClassConfig
from config.subset_samplers import SubsetSamplerConfig

__all__ = [
    "RGB2DModelSettings",
    "RGB3DModelSettings",
    "FeatureTSNConfig",
    "TSNConfig",
    "FeatureTRNConfig",
    "TRNConfig",
]


class _ModelSettings(BaseModel):
    # Either CTHW, TCHW for a 3D model, or CHW for a 2D one
    input_space: str
    input_order: str
    input_range: Tuple[float, float]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    pretrained: Union[str, bool] = False


class RGB2DModelSettings(_ModelSettings):
    input_size: Tuple[int, int, int]
    pretrained: Union[str, bool] = "imagenet"


class RGB3DModelSettings(_ModelSettings):
    # -1 indicates any input size is supported
    input_size: Tuple[int, int, int, int]


class TSNConfig(ClassConfig):
    kind = Field("TSN", const=True)
    backbone_settings: RGB2DModelSettings
    class_count: int
    backbone: str = "resnet"
    backbone_dim: int = 256
    backbone_checkpoint: Optional[str] = None
    temporal_module_checkpoint: Optional[str] = None
    dropout: float = 0.7

    def instantiate(self):
        from models.builder import make_tsn

        return make_tsn(self)


class FeatureTSNConfig(BaseModel):
    kind = Field("TSN", const=True)
    class_count: int
    input_dim: int = 256
    dropout: float = 0.7
    input_relu: bool = True
    checkpoint: Optional[str] = None

    def instantiate(self):
        from models.builder import make_feature_tsn

        return make_feature_tsn(self)


class TRNConfig(BaseModel):
    kind = Field("TRN", const=True)
    backbone_settings: RGB2DModelSettings
    backbone_checkpoint: Optional[str] = None
    temporal_module_checkpoint: Optional[str] = None
    class_count: int
    backbone: str = "resnet"
    backbone_dim: int = 256
    hidden_dim: int = 256
    n_hidden_layers: int = 1
    dropout: float = 0.7
    frame_count: int = 8
    batch_norm: bool = True

    def instantiate(self):
        from models.builder import make_trn

        return make_trn(self)


class FeatureTRNConfig(BaseModel):
    kind = Field("TRN", const=True)
    frame_count: int
    class_count: int
    input_dim: int = 256
    hidden_dim: int = 256
    n_hidden_layers: int = 1
    dropout: float = 0.7
    batch_norm: bool = True
    input_relu: bool = False
    checkpoint: Optional[str] = None

    def instantiate(self):
        from models.builder import make_feature_trn

        return make_feature_trn(self)


class FeatureMultiscaleNetworkConfig(ClassConfig):
    kind = Field("MultiscaleNetwork", const=True)
    sub_models: List[FeatureTRNConfig]
    softmax: bool = False
    sampler: SubsetSamplerConfig
    recursive: bool = False

    def instantiate(self):
        from models.builder import make_feature_multiscale_model

        return make_feature_multiscale_model(self)
