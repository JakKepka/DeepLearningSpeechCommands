"""Unified training pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import collect_predictions, compute_full_metrics, compute_metrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.seed import set_seed
from src.utils.logging import get_logger
from src.utils.constants import (
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_TYPE,
    WARMUP_EPOCHS,
    GRAD_CLIP_NORM,
    EARLY_STOPPING_PATIENCE,
    DEVICE_AUTO,
)

logger = get_logger()


class Trainer:
    """Generic trainer for speech command classification models.

    Args:
        model:          PyTorch model.
        train_loader:   DataLoader for training data.
        val_loader:     DataLoader for validation data.
        criterion:      Loss function.
        cfg:            Config dict with ``train`` sub-dict.
        run_name:       Identifier for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        cfg: dict[str, Any],
        run_name: str = "run",
        test_loader: Optional[DataLoader] = None,
    ):
        train_cfg = cfg.get("train", cfg)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.run_name = run_name
        self.best_epoch: int = 0

        # Device
        device_str = train_cfg.get("device", "auto")
        if device_str == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
            elif torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"
        self.device = torch.device(device_str)
        self.model.to(self.device)
        self.criterion.to(self.device)

        # Optimiser
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg.get("lr", LEARNING_RATE),
            weight_decay=train_cfg.get("weight_decay", WEIGHT_DECAY),
        )

        # LR scheduler
        max_epochs: int = train_cfg.get("max_epochs", MAX_EPOCHS)
        scheduler_name: str = train_cfg.get("scheduler", SCHEDULER_TYPE)
        warmup_epochs: int = train_cfg.get("warmup_epochs", WARMUP_EPOCHS)
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        elif scheduler_name == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=train_cfg.get("lr", 3e-4),
                epochs=max_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=warmup_epochs / max_epochs,
            )
        else:
            self.scheduler = None

        self.max_epochs = max_epochs
        self.grad_clip_norm: float = train_cfg.get("grad_clip_norm", GRAD_CLIP_NORM)

        # Callbacks
        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint = ModelCheckpoint(ckpt_dir, run_name=run_name, monitor="val_acc")
        self.early_stop = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE), mode="max"
        )

        # History
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns a dict with keys:
            ``history``  — list of per-epoch metric dicts
            ``test``     — test-set metrics evaluated on the best checkpoint
                           (empty dict when no test_loader was supplied)
            ``meta``     — run metadata (best epoch, best val_acc, …)
        """
        logger.info(
            f"Training on {self.device} | {self.max_epochs} epochs | "
            f"train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)}"
        )
        for epoch in range(1, self.max_epochs + 1):
            remaining = self.max_epochs - epoch
            logger.info(f"Starting epoch {epoch}/{self.max_epochs} | remaining: {remaining}")
            t0 = time.perf_counter()
            train_metrics = self._train_epoch(epoch=epoch)
            val_metrics = self._val_epoch(epoch=epoch)
            epoch_time = time.perf_counter() - t0

            row = {"epoch": epoch, "epoch_time": epoch_time, **train_metrics, **val_metrics}
            self.history.append(row)

            is_best = self.checkpoint(self.model, val_metrics["val_acc"], epoch)
            if is_best:
                self.best_epoch = epoch
            self.checkpoint.save_last(self.model, epoch)

            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"loss={train_metrics['train_loss']:.4f} "
                f"acc={train_metrics['train_acc']:.3f} | "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_acc={val_metrics['val_acc']:.3f} | "
                f"{epoch_time:.1f}s"
            )

            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()

            if self.early_stop(val_metrics["val_acc"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # --- Test evaluation on the best checkpoint ---
        test_metrics: dict[str, Any] = {}
        if (
            self.test_loader is not None
            and self.checkpoint.best_path is not None
            and self.checkpoint.best_path.exists()
        ):
            logger.info(
                f"Loading best checkpoint (epoch {self.best_epoch}) for test evaluation …"
            )
            state = torch.load(self.checkpoint.best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            preds, targets = collect_predictions(self.model, self.test_loader, self.device)
            test_metrics = compute_full_metrics(preds, targets)
            test_metrics["best_epoch"] = self.best_epoch
            test_metrics["best_val_acc"] = float(self.checkpoint.best)  # type: ignore[arg-type]
            logger.info(
                f"Test acc={test_metrics['accuracy']:.4f}  "
                f"macro_f1={test_metrics['macro_f1']:.4f}"
            )

        return {
            "history": self.history,
            "test": test_metrics,
            "meta": {
                "run_name": self.run_name,
                "total_epochs": len(self.history),
                "best_epoch": self.best_epoch,
                "best_val_acc": float(self.checkpoint.best) if self.checkpoint.best is not None else None,
            },
        }

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int | None = None) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        desc = "train" if epoch is None else f"train e{epoch}/{self.max_epochs}"
        for batch in tqdm(self.train_loader, desc=desc, leave=False):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * y.size(0)
            batch_preds = logits.argmax(dim=-1)
            correct += (batch_preds == y).sum().item()
            n += y.size(0)
            all_preds.extend(batch_preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

        m = compute_metrics(all_preds, all_targets)
        lr = float(self.optimizer.param_groups[0]["lr"])
        return {
            "train_loss": total_loss / n,
            "train_acc": correct / n,
            "train_macro_f1": m["macro_f1"],
            "lr": lr,
        }

    @torch.inference_mode()
    def _val_epoch(self, epoch: int | None = None) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        all_preds: list[int] = []
        all_targets: list[int] = []

        desc = "val" if epoch is None else f"val   e{epoch}/{self.max_epochs}"
        for batch in tqdm(self.val_loader, desc=desc, leave=False):
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            batch_preds = logits.argmax(dim=-1)
            correct += (batch_preds == y).sum().item()
            n += y.size(0)
            all_preds.extend(batch_preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

        m = compute_metrics(all_preds, all_targets)
        result: dict[str, Any] = {
            "val_loss": total_loss / n,
            "val_acc": correct / n,
            "val_macro_f1": m["macro_f1"],
        }
        # Per-class recall on validation (prefixed with "val_")
        for k, v in m.items():
            if k.startswith("recall_"):
                result[f"val_{k}"] = v
        return result
