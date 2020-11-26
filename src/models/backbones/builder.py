from typing import Optional

from models.utils import replace_last_linear
from models.backbones import bninception, resnet


def load_backbone(
    backbone_type: str,
    *,
    backbone_output_dim: Optional[int] = None,
    pretrained: str = "imagenet",
):
    if pretrained == "scratch":
        pretrained = None

    if "resnet" in backbone_type:
        backbone = resnet.__dict__[backbone_type](pretrained=pretrained == "imagenet")
    elif backbone_type.lower() == "bninception":
        backbone = bninception.__dict__[backbone_type.lower()](
            pretrained=pretrained == "imagenet"
        )
    else:
        raise NotImplementedError()

    if backbone_output_dim is not None:
        replace_last_linear(backbone, backbone_output_dim)
    return backbone