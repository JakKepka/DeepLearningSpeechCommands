"""Generate and export complete evaluation reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.confusion import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import collect_predictions, compute_metrics
from src.utils.io import save_csv, save_json


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    run_name: str = "run",
    figures_dir: str | Path = "outputs/figures",
    tables_dir: str | Path = "outputs/tables",
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full evaluation: metrics + confusion matrix + export.

    Returns the metrics dict.
    """
    preds, targets = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds, targets, class_names=class_names)

    if extra_info:
        metrics.update(extra_info)

    # Count model parameters
    n_params = sum(p.numel() for p in model.parameters())
    metrics["n_params"] = n_params

    # Export metrics to JSON and CSV
    tables_dir = Path(tables_dir)
    save_json(metrics, tables_dir / f"{run_name}_metrics.json")
    save_csv([metrics], tables_dir / f"{run_name}_metrics.csv")

    # Confusion matrix
    cm = compute_confusion_matrix(preds, targets, normalize="true")
    figures_dir = Path(figures_dir)
    cm_path = figures_dir / f"{run_name}_confusion.png"
    plot_confusion_matrix(cm, class_names=class_names, title=f"{run_name} — Confusion Matrix", save_path=cm_path)
    plt_close()

    return metrics


def plt_close() -> None:
    import matplotlib.pyplot as plt
    plt.close("all")
