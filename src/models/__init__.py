"""Model registry — build any supported model by name."""
from __future__ import annotations

import torch.nn as nn

from src.models.cnn_baseline import CNNBaseline
from src.models.bc_resnet import BCResNet, bc_resnet_1_5
from src.models.kwt import KWT, kwt_2, kwt_medium, kwt_small
from src.models.ast_wrapper import ASTWrapper
from src.utils.constants import AST_PRETRAINED_MODEL, AST_FREEZE_N_LAYERS


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from a config dict containing a ``model`` sub-dict."""
    model_cfg: dict = cfg.get("model", cfg)
    name: str = model_cfg["name"].lower()
    num_classes: int = model_cfg.get("num_classes", 12)

    if name == "cnn_baseline":
        return CNNBaseline(
            num_classes=num_classes,
            channels=model_cfg.get("channels", [32, 64, 128, 256]),
            dropout=model_cfg.get("dropout", 0.1),
        )

    if name == "kwt":
        variant = model_cfg.get("variant", "small")
        kw = dict(
            n_mels=model_cfg.get("n_mels", 64),
            time_frames=model_cfg.get("time_frames", 98),
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 4),
            mlp_dim=model_cfg.get("mlp_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
            trunc_norm_std=model_cfg.get("trunc_norm_std", 0.02),
        )
        if variant == "kwt_2":
            return kwt_2(**kw)
        if variant == "medium":
            return kwt_medium(**kw)
        return kwt_small(**kw)

    if name == "bc_resnet":
        variant = str(model_cfg.get("variant", "1.5"))
        bc = dict(
            n_mels=model_cfg.get("n_mels", 40),
            num_classes=num_classes,
            width_mult=model_cfg.get("width_mult", 1.5),
            first_filters=model_cfg.get("first_filters", 24),
            stage_filters=model_cfg.get("stage_filters", [12, 18, 24, 30]),
            stage_blocks=model_cfg.get("stage_blocks", [2, 2, 4, 4]),
            stage_strides_f=model_cfg.get("stage_strides_f", [1, 2, 2, 1]),
            stage_dilations_t=model_cfg.get("stage_dilations_t", [1, 2, 3, 3]),
            last_filters=model_cfg.get("last_filters", 48),
            sub_bands=model_cfg.get("sub_bands", 5),
            dropout=model_cfg.get("dropout", 0.1),
        )
        if variant == "1.5":
            return bc_resnet_1_5(**bc)
        return BCResNet(**bc)

    if name == "ast":
        return ASTWrapper(
            num_classes=num_classes,
            pretrained_model=model_cfg.get(
                "pretrained_model", AST_PRETRAINED_MODEL
            ),
            freeze_n_layers=model_cfg.get("freeze_n_layers", AST_FREEZE_N_LAYERS),
        )

    raise ValueError(f"Unknown model name: {name!r}")
