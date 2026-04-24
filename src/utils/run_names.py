"""Helpers for stable, informative run names."""
from __future__ import annotations

import re
from typing import Any


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return text.strip("_") or "run"


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value)


def _model_tokens(model_cfg: dict[str, Any]) -> list[str]:
    model_name = str(model_cfg.get("name", "model")).lower()
    tokens: list[str] = []

    if model_name == "kwt":
        if model_cfg.get("d_model") is not None:
            tokens.append(f"d{_format_scalar(model_cfg['d_model'])}")
        if model_cfg.get("n_heads") is not None:
            tokens.append(f"h{_format_scalar(model_cfg['n_heads'])}")
        if model_cfg.get("n_layers") is not None:
            tokens.append(f"L{_format_scalar(model_cfg['n_layers'])}")
    elif model_name == "cnn_baseline":
        channels = model_cfg.get("channels") or []
        if channels:
            chan_text = "-".join(_format_scalar(channel) for channel in channels)
            tokens.append(f"ch{chan_text}")
    elif model_name == "ast":
        if model_cfg.get("freeze_n_layers") is not None:
            tokens.append(f"frz{_format_scalar(model_cfg['freeze_n_layers'])}")
    elif model_name == "bc_resnet":
        if model_cfg.get("variant") is not None:
            tokens.append(f"v{_format_scalar(model_cfg['variant'])}")
        if model_cfg.get("width_mult") is not None:
            width = float(model_cfg["width_mult"])
            tokens.append(f"wm{int(round(width * 10))}")
        if model_cfg.get("first_filters") is not None:
            tokens.append(f"f{_format_scalar(model_cfg['first_filters'])}")

    if model_cfg.get("dropout") is not None:
        dropout = float(model_cfg["dropout"])
        tokens.append(f"do{int(round(dropout * 100))}")

    return tokens


def build_run_name(cfg: dict[str, Any], seed: int) -> str:
    experiment_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    prefix = _slug(f"{experiment_cfg.get('id', 'run')}_{experiment_cfg.get('name', 'model')}")
    tokens = _model_tokens(model_cfg)
    if train_cfg.get("batch_size") is not None:
        tokens.append(f"bs{_format_scalar(train_cfg['batch_size'])}")
    tokens.append(f"seed{seed}")
    return "_".join([prefix, *tokens])