import logging
from typing import Any, Callable, Tuple, Union

import torch
from torchvideo.transforms import (
    CenterCropVideo,
    Compose,
    IdentityTransform,
    MultiScaleCropVideo,
    NormalizeVideo,
    PILVideoToTensor,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    ResizeVideo,
)

from config.model import RGB2DModelSettings
from config.transform import TransformsConfig
from transforms import FlipChannels

LOG = logging.getLogger(__name__)


def get_transforms(
    cfg: TransformsConfig, model_settings: RGB2DModelSettings
) -> Tuple[Callable[[Any], torch.Tensor], Callable[[Any], torch.Tensor]]:
    train_transforms = []

    # model_settings.input_size is to be interpreted based on model_settings.input_order
    input_order = model_settings.input_order.lower()
    if len(input_order) == 3:  # If only CHW is given, then assume time comes first.
        input_order = "t" + input_order
    if input_order.endswith("hw"):
        input_height, input_width = model_input_size = model_settings.input_size[-2:]
    else:
        raise NotImplementedError("Unsupported input ordering: {}".format(input_order))

    if cfg.train.hflip:
        LOG.info("Using horizontal flipping")
        train_transforms.append(RandomHorizontalFlipVideo())
    if cfg.preserve_aspect_ratio:
        LOG.info(f"Preserving aspect ratio of videos")
        rescaled_size: Union[int, Tuple[int, int]] = int(
            input_height * cfg.image_scale_factor
        )
    else:
        rescaled_size = (
            int(input_height * cfg.image_scale_factor),
            int(input_width * cfg.image_scale_factor),
        )
        LOG.info(f"Squashing videos to {rescaled_size}")
    train_transforms.append(ResizeVideo(rescaled_size))
    LOG.info(f"Resizing videos to {rescaled_size}")
    if cfg.train.augment_crop is not None:
        LOG.info(
            f"Using multiscale cropping "
            f"(scales: {cfg.train.augment_crop.scales}, "
            f"fixed_crops: {cfg.train.augment_crop.fixed_crops}, "
            f"more_fixed_crops: {cfg.train.augment_crop.more_fixed_crops}"
            f")"
        )
        train_transforms.append(
            MultiScaleCropVideo(
                model_input_size,
                scales=cfg.train.augment_crop.scales,
                fixed_crops=cfg.train.augment_crop.fixed_crops,
                more_fixed_crops=cfg.train.augment_crop.more_fixed_crops,
            )
        )
    else:
        LOG.info(f"Cropping videos to {model_input_size}")
        train_transforms.append(RandomCropVideo(model_input_size))

    channel_dim = input_order.find("c")
    if channel_dim == -1:
        raise ValueError(
            f"Could not determine channel position in input_order {input_order!r}"
        )
    if model_settings.input_space == "BGR":
        LOG.info(f"Flipping channels from RGB to BGR")
        channel_transform = FlipChannels(channel_dim)
    else:
        assert model_settings.input_space == "RGB"
        channel_transform = IdentityTransform()
    common_transforms = [
        PILVideoToTensor(
            rescale=model_settings.input_range[-1] != 255,
            ordering=input_order,
        ),
        channel_transform,
        NormalizeVideo(
            mean=model_settings.mean, std=model_settings.std, channel_dim=channel_dim
        ),
    ]
    train_transform = Compose(train_transforms + common_transforms)
    LOG.info(f"Training transform: {train_transform!r}")
    test_transform = Compose(
        [ResizeVideo(rescaled_size), CenterCropVideo(model_input_size)]
        + common_transforms
    )
    LOG.info(f"Validation transform: {test_transform!r}")
    return train_transform, test_transform
