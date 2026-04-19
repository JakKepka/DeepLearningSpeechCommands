"""Classification metrics: accuracy, macro F1, per-class recall."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.labels import ALL_CLASSES, NUM_CLASSES, SILENCE_IDX, UNKNOWN_IDX


def compute_metrics(
    preds: torch.Tensor | list[int],
    targets: torch.Tensor | list[int],
    class_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute accuracy, macro F1, and per-class recall.

    Returns a flat dict with keys:
        accuracy, macro_f1,
        recall_silence, recall_unknown,
        recall_<class_name> for each class.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    cnames = class_names or ALL_CLASSES

    acc: float = accuracy_score(targets, preds)
    macro_f1: float = f1_score(targets, preds, average="macro", zero_division=0)
    per_class_recall: np.ndarray = recall_score(
        targets, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )

    result: dict[str, float] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }
    for idx, name in enumerate(cnames):
        key = f"recall_{name}"
        result[key] = float(per_class_recall[idx]) if idx < len(per_class_recall) else 0.0

    # Convenient aliases for critical classes
    result["recall_silence"] = result.get(f"recall_{ALL_CLASSES[SILENCE_IDX]}", 0.0)
    result["recall_unknown"] = result.get(f"recall_{ALL_CLASSES[UNKNOWN_IDX]}", 0.0)

    return result


def compute_full_metrics(
    preds: torch.Tensor | list[int] | np.ndarray,
    targets: torch.Tensor | list[int] | np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Extended metrics for the best-checkpoint test evaluation.

    Returns all keys from ``compute_metrics`` plus per-class precision, per-class
    F1, macro precision/recall averages, and the full confusion matrix as a
    list of lists (rows = true class, columns = predicted class).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    cnames = class_names or ALL_CLASSES
    labels = list(range(NUM_CLASSES))

    acc: float = accuracy_score(targets, preds)
    macro_f1: float = f1_score(targets, preds, average="macro", zero_division=0)
    macro_precision: float = precision_score(targets, preds, average="macro", zero_division=0)
    macro_recall: float = recall_score(targets, preds, average="macro", zero_division=0)

    per_recall = recall_score(targets, preds, labels=labels, average=None, zero_division=0)
    per_precision = precision_score(targets, preds, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=labels)

    result: dict = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm.tolist(),
        "class_names": list(cnames),
    }
    for idx, name in enumerate(cnames):
        if idx < len(per_recall):
            result[f"recall_{name}"] = float(per_recall[idx])
            result[f"precision_{name}"] = float(per_precision[idx])
            result[f"f1_{name}"] = float(per_f1[idx])

    return result


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and return (predictions, targets)."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.tolist())

    return np.array(all_preds), np.array(all_targets)
