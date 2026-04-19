"""Loss functions."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.constants import LABEL_SMOOTHING


def build_loss(
    class_counts: Optional[torch.Tensor] = None,
    rebalance: bool = False,
    label_smoothing: float = LABEL_SMOOTHING,
    device: torch.device | str = "cpu",
) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss, optionally with inverse-frequency class weights.

    Args:
        class_counts:    Number of samples per class (shape: [num_classes]).
        rebalance:       If True, compute inverse-frequency weights.
        label_smoothing: Label smoothing factor (0 = disabled).
        device:          Target device for weight tensor.
    """
    weight: Optional[torch.Tensor] = None
    if rebalance and class_counts is not None:
        w = 1.0 / class_counts.float().clamp(min=1.0)
        weight = (w / w.sum()).to(device)

    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
