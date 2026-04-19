"""Integration tests for inference pipeline and hierarchical classifier."""
from __future__ import annotations

import torch
import pytest

from src.models.cnn_baseline import CNNBaseline
from src.models.kwt import kwt_small
from src.models.hierarchical import HierarchicalClassifier, build_hierarchical
from src.data.labels import SILENCE_IDX, UNKNOWN_IDX, TARGET_COMMAND_INDICES


class TestHierarchicalClassifier:
    def _make_input(self, batch: int = 4):
        return torch.randn(batch, 1, 64, 101)

    def test_forward_returns_two_tensors(self):
        stage1 = kwt_small(num_classes=3)
        stage2 = kwt_small(num_classes=10)
        model = HierarchicalClassifier(stage1, stage2)
        s1, s2 = model(self._make_input())
        assert s1.shape == (4, 3)
        assert s2.shape == (4, 10)

    def test_predict_output_shape(self):
        stage1 = kwt_small(num_classes=3)
        stage2 = kwt_small(num_classes=10)
        model = HierarchicalClassifier(stage1, stage2)
        preds = model.predict(self._make_input())
        assert preds.shape == (4,)

    def test_predict_valid_class_indices(self):
        stage1 = kwt_small(num_classes=3)
        stage2 = kwt_small(num_classes=10)
        model = HierarchicalClassifier(stage1, stage2)
        preds = model.predict(self._make_input(batch=20))
        valid = set([SILENCE_IDX, UNKNOWN_IDX] + TARGET_COMMAND_INDICES)
        for p in preds.tolist():
            assert p in valid

    def test_build_hierarchical_helper(self):
        model = build_hierarchical(CNNBaseline)
        s1, s2 = model(self._make_input())
        assert s1.shape == (4, 3)
        assert s2.shape == (4, 10)


class TestPredictor:
    def test_instantiation(self):
        from src.inference.predict import Predictor
        model = kwt_small(num_classes=12)
        predictor = Predictor(model=model, device=torch.device("cpu"))
        assert predictor is not None
