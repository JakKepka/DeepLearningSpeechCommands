"""Unit tests for model forward passes."""
from __future__ import annotations

import pytest
import torch

from src.models.cnn_baseline import CNNBaseline
from src.models.kwt import KWT, kwt_small, kwt_medium


def _dummy_batch(batch_size: int = 4, n_mels: int = 64, time_frames: int = 101):
    return torch.randn(batch_size, 1, n_mels, time_frames)


class TestCNNBaseline:
    def test_output_shape(self):
        model = CNNBaseline(n_mels=64, num_classes=12)
        x = _dummy_batch()
        out = model(x)
        assert out.shape == (4, 12)

    def test_no_nan(self):
        model = CNNBaseline()
        out = model(_dummy_batch())
        assert not torch.isnan(out).any()

    def test_custom_channels(self):
        model = CNNBaseline(num_classes=12, channels=[16, 32])
        out = model(_dummy_batch())
        assert out.shape == (4, 12)

    def test_param_count(self):
        model = CNNBaseline()
        n = sum(p.numel() for p in model.parameters())
        assert n > 0


class TestKWT:
    def test_small_output_shape(self):
        model = kwt_small(num_classes=12)
        out = model(_dummy_batch())
        assert out.shape == (4, 12)

    def test_medium_output_shape(self):
        model = kwt_medium(num_classes=12)
        out = model(_dummy_batch())
        assert out.shape == (4, 12)

    def test_no_nan(self):
        model = kwt_small()
        out = model(_dummy_batch())
        assert not torch.isnan(out).any()

    def test_different_time_frames(self):
        """Model should handle variable T via positional embedding interpolation."""
        model = KWT(n_mels=64, time_frames=101, d_model=64, n_heads=4, n_layers=2, mlp_dim=128, num_classes=12)
        x = torch.randn(2, 1, 64, 80)  # T=80 != 101
        out = model(x)
        assert out.shape == (2, 12)

    def test_3class_stage1(self):
        model = kwt_small(num_classes=3)
        out = model(_dummy_batch())
        assert out.shape == (4, 3)

    def test_10class_stage2(self):
        model = kwt_small(num_classes=10)
        out = model(_dummy_batch())
        assert out.shape == (4, 10)


class TestModelRegistry:
    def test_build_cnn(self):
        from src.models import build_model
        model = build_model({"model": {"name": "cnn_baseline", "num_classes": 12}})
        out = model(_dummy_batch())
        assert out.shape == (4, 12)

    def test_build_kwt_small(self):
        from src.models import build_model
        model = build_model({"model": {"name": "kwt", "variant": "small", "num_classes": 12}})
        out = model(_dummy_batch())
        assert out.shape == (4, 12)

    def test_build_kwt_medium(self):
        from src.models import build_model
        model = build_model({"model": {"name": "kwt", "variant": "medium", "num_classes": 12}})
        out = model(_dummy_batch())
        assert out.shape == (4, 12)
