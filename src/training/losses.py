"""Loss functions."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.constants import LABEL_SMOOTHING


def build_loss(
    class_counts: Optional[torch.Tensor] = None,
    rebalance: bool = False,
    rebalance_strength: float = 1.0,
    label_smoothing: float = LABEL_SMOOTHING,
    device: torch.device | str = "cpu",
) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss, optionally with inverse-frequency class weights.

    Args:
        class_counts:    Number of samples per class (shape: [num_classes]).
        rebalance:       If True, compute inverse-frequency weights.
        rebalance_strength: Exponent for inverse-frequency weighting.
                           1.0 = standard inverse-frequency;
                           0.5 = milder rebalancing.
        label_smoothing: Label smoothing factor (0 = disabled).
        device:          Target device for weight tensor.
    """
    weight: Optional[torch.Tensor] = None
    if rebalance and class_counts is not None:
        strength = float(rebalance_strength)
        w = 1.0 / class_counts.float().clamp(min=1.0).pow(strength)
        weight = (w / w.sum()).to(device)

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
