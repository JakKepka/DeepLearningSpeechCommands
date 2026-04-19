"""Weighted random sampler for class-rebalancing."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.labels import NUM_CLASSES


def build_weighted_sampler(
    labels: list[int],
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that up-samples rare classes.

    Args:
        labels:      Class index for every sample in the dataset.
        num_samples: How many draws per epoch (default: len(labels)).
        replacement: Sample with replacement (required for over-sampling).
    """
    label_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=NUM_CLASSES).float()
    # Avoid division by zero for classes not present in the split
    class_weights = 1.0 / counts.clamp(min=1.0)
    sample_weights = class_weights[label_tensor]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples if num_samples is not None else len(labels),
        replacement=replacement,
    )
