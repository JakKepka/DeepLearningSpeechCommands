"""Model registry — build any supported model by name."""
from __future__ import annotations

import torch.nn as nn

from src.models.cnn_baseline import CNNBaseline
from src.models.kwt import KWT, kwt_medium, kwt_small
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
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 4),
            mlp_dim=model_cfg.get("mlp_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
        )
        if variant == "medium":
            return kwt_medium(num_classes=num_classes)
        return kwt_small(num_classes=num_classes)

    if name == "ast":
        return ASTWrapper(
            num_classes=num_classes,
            pretrained_model=model_cfg.get(
                "pretrained_model", AST_PRETRAINED_MODEL
            ),
            freeze_n_layers=model_cfg.get("freeze_n_layers", AST_FREEZE_N_LAYERS),
        )

    raise ValueError(f"Unknown model name: {name!r}")
