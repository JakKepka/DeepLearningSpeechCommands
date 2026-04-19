"""Unit tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_metrics
from src.data.labels import NUM_CLASSES, ALL_CLASSES, SILENCE_IDX, UNKNOWN_IDX


class TestComputeMetrics:
    def _make_perfect(self, n: int = 120):
        targets = list(range(NUM_CLASSES)) * (n // NUM_CLASSES)
        preds = list(targets)
        return preds, targets

    def test_perfect_accuracy(self):
        preds, targets = self._make_perfect()
        m = compute_metrics(preds, targets)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_macro_f1(self):
        preds, targets = self._make_perfect()
        m = compute_metrics(preds, targets)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_perfect_recall_all_classes(self):
        preds, targets = self._make_perfect()
        m = compute_metrics(preds, targets)
        for cls in ALL_CLASSES:
            assert m[f"recall_{cls}"] == pytest.approx(1.0)

    def test_silence_recall_key_present(self):
        preds, targets = self._make_perfect()
        m = compute_metrics(preds, targets)
        assert "recall_silence" in m
        assert "recall_unknown" in m

    def test_worst_case_accuracy(self):
        targets = [0] * 12
        preds = [1] * 12
        m = compute_metrics(preds, targets)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_accepts_tensors(self):
        preds = torch.zeros(12, dtype=torch.long)
        targets = torch.zeros(12, dtype=torch.long)
        m = compute_metrics(preds, targets)
        assert isinstance(m["accuracy"], float)

    def test_partial_accuracy(self):
        targets = [0, 0, 1, 1]
        preds = [0, 1, 1, 1]
        m = compute_metrics(preds, targets)
        assert m["accuracy"] == pytest.approx(0.75)
