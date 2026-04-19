"""Confusion matrix generation and export."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data.labels import ALL_CLASSES


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    normalize: Optional[str] = "true",
) -> np.ndarray:
    """Compute confusion matrix, optionally normalised by true (row-wise)."""
    return confusion_matrix(targets, preds, labels=list(range(len(ALL_CLASSES))), normalize=normalize)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    cnames = class_names or ALL_CLASSES
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if cm.dtype == float else "d",
        xticklabels=cnames,
        yticklabels=cnames,
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
