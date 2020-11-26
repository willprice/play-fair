from typing import Optional, Tuple

from pydantic import BaseModel

__all__ = ["CropAugmentationConfig", "TrainTransformConfig", "TransformsConfig"]


class CropAugmentationConfig(BaseModel):
    scales: Tuple[int, ...] = (1, 0.875, 0.75)
    fixed_crops: bool = False
    more_fixed_crops: bool = False


class TrainTransformConfig(BaseModel):
    hflip: bool = False
    augment_crop: Optional[CropAugmentationConfig] = None


class TransformsConfig(BaseModel):
    preserve_aspect_ratio: bool = True
    image_scale_factor: float = 1 / 0.875
    train: TrainTransformConfig = TrainTransformConfig()
