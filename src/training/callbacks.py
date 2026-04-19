"""Training callbacks: early stopping and checkpointing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.constants import MIN_DELTA_EARLY_STOPPING


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:    How many epochs without improvement before stopping.
        min_delta:   Minimum change to qualify as improvement.
        mode:        'min' or 'max'.
    """

    def __init__(self, patience: int = 10, min_delta: float = MIN_DELTA_EARLY_STOPPING, mode: str = "max"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save the best model weights to disk.

    Args:
        checkpoint_dir:  Directory to save checkpoints.
        run_name:        Prefix for checkpoint files.
        monitor:         Metric name used for comparison.
        mode:            'min' or 'max'.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        run_name: str = "model",
        monitor: str = "val_acc",
        mode: str = "max",
    ):
        assert mode in ("min", "max")
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None
        self.best_path: Optional[Path] = None

    def __call__(self, model: nn.Module, value: float, epoch: int) -> bool:
        """Save if improved. Returns True when a new best is saved."""
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best = value
            path = self.ckpt_dir / f"{self.run_name}_best.pt"
            torch.save(model.state_dict(), path)
            self.best_path = path
        return improved

    def save_last(self, model: nn.Module, epoch: int) -> Path:
        path = self.ckpt_dir / f"{self.run_name}_last.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
        return path
