"""Integration tests: one training step through model + loss."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_baseline import CNNBaseline
from src.models.kwt import kwt_small
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import build_loss
from src.training.seed import set_seed


def _make_loader(n: int = 32, n_mels: int = 64, time_frames: int = 101, num_classes: int = 12):
    x = torch.randn(n, 1, n_mels, time_frames)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


class TestTrainingStep:
    def test_cnn_single_step(self):
        set_seed(0)
        model = CNNBaseline(num_classes=12)
        loader = _make_loader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            break  # one step

        assert loss.item() > 0

    def test_kwt_single_step(self):
        set_seed(0)
        model = kwt_small(num_classes=12)
        loader = _make_loader()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            break

        assert loss.item() > 0


class TestEarlyStopping:
    def test_stops_after_patience(self):
        # First call initialises best; then patience=3 calls without improvement trigger stop
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)   # initialise best
        for _ in range(3):
            es(0.5)  # no improvement — counter increments
        assert es.should_stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es(0.5)
        es(0.6)  # improvement — counter resets
        assert es.counter == 0
        assert not es.should_stop

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)
        es(1.1)  # no improvement
        es(1.2)  # no improvement → stop
        assert es.should_stop


class TestBuildLoss:
    def test_unweighted(self):
        loss_fn = build_loss()
        x = torch.randn(4, 12)
        y = torch.randint(0, 12, (4,))
        loss = loss_fn(x, y)
        assert loss.item() > 0

    def test_weighted(self):
        counts = torch.ones(12) * 100
        counts[0] = 10  # silence is rare
        loss_fn = build_loss(class_counts=counts, rebalance=True)
        x = torch.randn(4, 12)
        y = torch.randint(0, 12, (4,))
        loss = loss_fn(x, y)
        assert loss.item() > 0
