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
from .flip_channels import FlipChannels

LOG = logging.getLogger(__name__)


def get_transforms(
    args, model_settings: RGB2DModelSettings
) -> Tuple[Callable[[Any], torch.Tensor], Callable[[Any], torch.Tensor]]:
    train_transforms = []

    # model_settings.input_size is to be interpreted based on model_settings.input_order
    input_order = model_settings.input_order.lower()
    if input_order.endswith("hw"):
        input_height, input_width = model_input_size = model_settings.input_size[-2:]
    else:
        raise NotImplementedError("Unsupported input ordering: {}".format(input_order))

    if args.augment_hflip:
        LOG.info("Using horizontal flipping")
        train_transforms.append(RandomHorizontalFlipVideo())
    if args.preserve_aspect_ratio:
        LOG.info(f"Preserving aspect ratio of videos")
        rescaled_size: Union[int, Tuple[int, int]] = int(
            input_height * args.image_scale_factor
        )
    else:
        rescaled_size = (
            int(input_height * args.image_scale_factor),
            int(input_width * args.image_scale_factor),
        )
        LOG.info(f"Squashing videos to {rescaled_size}")
    train_transforms.append(ResizeVideo(rescaled_size))
    LOG.info(f"Resizing videos to {rescaled_size}")
    if args.augment_crop:
        LOG.info(
            f"Using multiscale cropping "
            f"(scales: {args.augment_crop_scales}, "
            f"fixed_crops: {args.augment_crop_fixed_crops}, "
            f"more_fixed_crops: {args.augment_crop_more_fixed_crops}"
            f")"
        )
        train_transforms.append(
            MultiScaleCropVideo(
                model_input_size,
                scales=args.augment_crop_scales,
                fixed_crops=args.augment_crop_fixed_crops,
                more_fixed_crops=args.augment_crop_more_fixed_crops,
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
    validation_transform = Compose(
        [ResizeVideo(rescaled_size), CenterCropVideo(model_input_size)]
        + common_transforms
    )
    LOG.info(f"Validation transform: {validation_transform!r}")
    return train_transform, validation_transform
