from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING, Union

import torch
from torch import nn

from models.multiscale import MultiscaleModel, RecursiveMultiscaleModel

from config.model import (
    FeatureTRNConfig,
    FeatureTSNConfig,
    FeatureMultiscaleNetworkConfig,
    RGB3DModelSettings,
    TRNConfig,
    TSNConfig,
)
from models.aggregated_backbone_model import AggregatedBackboneModel
from models.backbones.builder import load_backbone
from models.components.consensus import ClassifierConsensus
from models.components.mlp import MLPConsensus
from models.tsn import FeatureTSN
from models.types import Model

LOG = logging.getLogger(__name__)


def make_tsn(cfg: TSNConfig) -> Union[nn.Module, Model]:
    backbone = load_backbone(
        cfg.backbone,
        backbone_output_dim=cfg.backbone_dim,
        pretrained=cfg.backbone_settings.pretrained,
    )
    temporal_module = ClassifierConsensus(
        cfg.backbone_dim, cfg.class_count, dropout=cfg.dropout
    )
    model = AggregatedBackboneModel(backbone, temporal_module)
    _maybe_load_checkpoint(model.backbone, cfg.backbone_checkpoint)
    _maybe_load_checkpoint(model.temporal_module, cfg.temporal_module_checkpoint)

    model_settings = RGB3DModelSettings(
        input_space=backbone.input_space,
        input_order=model.input_order,
        input_size=(-1,) + tuple(backbone.input_size),
        input_range=backbone.input_range,
        mean=tuple(backbone.mean),
        std=tuple(backbone.std),
        class_count=cfg.class_count,
    )
    LOG.info(f"Model settings: {model_settings!r}")
    model.settings = model_settings
    return model


def make_feature_tsn(cfg: FeatureTSNConfig):
    tsn = FeatureTSN(
        feature_dim=cfg.input_dim,
        output_dim=cfg.class_count,
        dropout=cfg.dropout,
        input_relu=cfg.input_relu,
    )
    _maybe_load_checkpoint(tsn, cfg.checkpoint)
    return tsn


def make_trn(cfg: TRNConfig) -> nn.Module:
    backbone = load_backbone(
        cfg.backbone,
        backbone_output_dim=cfg.backbone_dim,
        pretrained=cfg.backbone_settings.pretrained,
    )
    temporal_module = MLPConsensus(
        input_dim=cfg.backbone_dim * cfg.frame_count,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.class_count,
        hidden_layers=cfg.n_hidden_layers,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
    )

    model = AggregatedBackboneModel(backbone, temporal_module)
    _maybe_load_checkpoint(model.backbone, cfg.backbone_checkpoint)
    _maybe_load_checkpoint(model.temporal_module, cfg.temporal_module_checkpoint)
    model_settings = RGB3DModelSettings(
        input_space=backbone.input_space,
        input_order=model.input_order,
        input_size=(-1,) + tuple(backbone.input_size),
        input_range=backbone.input_range,
        mean=tuple(backbone.mean),
        std=tuple(backbone.std),
        class_count=cfg.class_count,
    )
    LOG.info(f"Model settings: {model_settings!r}")
    model.settings = model_settings
    return model


def make_feature_trn(cfg: FeatureTRNConfig):
    trn = MLPConsensus(
        input_dim=cfg.input_dim * cfg.frame_count,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.class_count,
        hidden_layers=cfg.n_hidden_layers,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
        input_relu=cfg.input_relu,
    )
    _maybe_load_checkpoint(trn, cfg.checkpoint)
    return trn


def make_feature_multiscale_model(cfg: FeatureMultiscaleNetworkConfig):
    single_scale_models = [
        sub_model_cfg.instantiate() for sub_model_cfg in cfg.sub_models
    ]
    sampler = cfg.sampler.instantiate()
    if cfg.recursive:
        model = RecursiveMultiscaleModel(
            single_scale_models, softmax=cfg.softmax, sampler=sampler
        )
    else:
        model = MultiscaleModel(
            single_scale_models=single_scale_models,
            softmax=cfg.softmax,
            sampler=sampler,
        )
    return model


def _maybe_load_checkpoint(model, checkpoint_path: Optional[str]):
    if checkpoint_path is not None:
        LOG.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            raise ValueError(
                f"Could not find state dict in checkpoint {checkpoint_path}, "
                f"the only available keys were: {list(ckpt.keys())!r}"
            )
        model.load_state_dict(state_dict)
