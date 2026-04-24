"""Unified training pipeline."""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import collect_predictions, compute_full_metrics, compute_metrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.seed import set_seed
from src.utils.constants import STAGE1_SILENCE, STAGE1_TARGET, STAGE1_UNKNOWN
from src.data.labels import SILENCE_IDX, UNKNOWN_IDX
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
        self.hierarchical: bool = bool(train_cfg.get("hierarchical", False))
        self.hierarchical_stage2_weight: float = float(train_cfg.get("hierarchical_stage2_weight", 1.0))
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

        if self.hierarchical:
            self.stage1_criterion = nn.CrossEntropyLoss()
            self.stage2_criterion = nn.CrossEntropyLoss()
            self.stage1_criterion.to(self.device)
            self.stage2_criterion.to(self.device)

        # Optimiser
        lr: float = float(train_cfg.get("lr", LEARNING_RATE))
        weight_decay: float = float(train_cfg.get("weight_decay", WEIGHT_DECAY))
        optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
        if optimizer_name == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = SGD(
                model.parameters(),
                lr=lr,
                momentum=float(train_cfg.get("momentum", 0.9)),
                weight_decay=weight_decay,
                nesterov=bool(train_cfg.get("nesterov", False)),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

        # LR scheduler
        max_epochs: int = train_cfg.get("max_epochs", MAX_EPOCHS)
        scheduler_name: str = train_cfg.get("scheduler", SCHEDULER_TYPE)
        warmup_epochs: int = train_cfg.get("warmup_epochs", WARMUP_EPOCHS)
        if scheduler_name == "cosine":
            min_lr = float(train_cfg.get("min_lr", 0.0))
            if warmup_epochs > 0 and warmup_epochs < max_epochs:
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=float(train_cfg.get("warmup_start_factor", 1e-6)),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max_epochs - warmup_epochs,
                    eta_min=min_lr,
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=min_lr)
        elif scheduler_name == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
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
        training_started_at = time.perf_counter()
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

            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            if self.early_stop(val_metrics["val_acc"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        training_time = time.perf_counter() - training_started_at

        # --- Test evaluation on the best checkpoint ---
        run_finished_at = time.perf_counter()
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
            if self.hierarchical:
                preds, targets = self._collect_predictions_hierarchical(self.test_loader)
            else:
                preds, targets = collect_predictions(self.model, self.test_loader, self.device)
            test_metrics = compute_full_metrics(preds, targets)
            test_metrics["best_epoch"] = self.best_epoch
            test_metrics["best_val_acc"] = float(self.checkpoint.best)  # type: ignore[arg-type]
            logger.info(
                f"Test acc={test_metrics['accuracy']:.4f}  "
                f"macro_f1={test_metrics['macro_f1']:.4f}"
            )

        total_run_time = time.perf_counter() - run_finished_at + training_time
        epoch_times = [float(row["epoch_time"]) for row in self.history]
        cumulative_epoch_time = 0.0
        for row in self.history:
            cumulative_epoch_time += float(row["epoch_time"])
            row["cumulative_epoch_time"] = cumulative_epoch_time

        best_epoch_time = None
        if self.best_epoch > 0 and self.best_epoch <= len(self.history):
            best_epoch_time = float(self.history[self.best_epoch - 1]["epoch_time"])

        timing = {
            "total_training_time": training_time,
            "total_run_time": total_run_time,
            "avg_epoch_time": statistics.fmean(epoch_times) if epoch_times else 0.0,
            "median_epoch_time": statistics.median(epoch_times) if epoch_times else 0.0,
            "min_epoch_time": min(epoch_times) if epoch_times else 0.0,
            "max_epoch_time": max(epoch_times) if epoch_times else 0.0,
            "first_epoch_time": epoch_times[0] if epoch_times else 0.0,
            "last_epoch_time": epoch_times[-1] if epoch_times else 0.0,
            "best_epoch_time": best_epoch_time,
            "cumulative_epoch_time": cumulative_epoch_time,
        }

        return {
            "history": self.history,
            "test": test_metrics,
            "meta": {
                "run_name": self.run_name,
                "total_epochs": len(self.history),
                "best_epoch": self.best_epoch,
                "best_val_acc": float(self.checkpoint.best) if self.checkpoint.best is not None else None,
                **timing,
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
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            if self.hierarchical:
                stage1_logits, stage2_logits = self.model(x)
                y_stage1 = self._to_stage1_labels(y)
                loss1 = self.stage1_criterion(stage1_logits, y_stage1)

                target_mask = y_stage1 == STAGE1_TARGET
                if target_mask.any():
                    y_stage2 = self._to_stage2_labels(y[target_mask])
                    loss2 = self.stage2_criterion(stage2_logits[target_mask], y_stage2)
                    loss = loss1 + self.hierarchical_stage2_weight * loss2
                else:
                    loss = loss1

                batch_preds = self._hierarchical_preds_from_logits(stage1_logits, stage2_logits)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)
                batch_preds = logits.argmax(dim=-1)
            loss.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * y.size(0)
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
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
            if self.hierarchical:
                stage1_logits, stage2_logits = self.model(x)
                y_stage1 = self._to_stage1_labels(y)
                loss1 = self.stage1_criterion(stage1_logits, y_stage1)
                target_mask = y_stage1 == STAGE1_TARGET
                if target_mask.any():
                    y_stage2 = self._to_stage2_labels(y[target_mask])
                    loss2 = self.stage2_criterion(stage2_logits[target_mask], y_stage2)
                    loss = loss1 + self.hierarchical_stage2_weight * loss2
                else:
                    loss = loss1
                batch_preds = self._hierarchical_preds_from_logits(stage1_logits, stage2_logits)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)
                batch_preds = logits.argmax(dim=-1)
            total_loss += loss.item() * y.size(0)
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

    def _to_stage1_labels(self, y: torch.Tensor) -> torch.Tensor:
        # 0 -> silence, 1 -> unknown, 2..11 -> target-command
        y_stage1 = torch.full_like(y, STAGE1_TARGET)
        y_stage1[y == SILENCE_IDX] = STAGE1_SILENCE
        y_stage1[y == UNKNOWN_IDX] = STAGE1_UNKNOWN
        return y_stage1

    def _to_stage2_labels(self, y_targets_only: torch.Tensor) -> torch.Tensor:
        # Flat labels 2..11 -> stage2 labels 0..9
        return (y_targets_only - 2).clamp(min=0, max=9)

    def _hierarchical_preds_from_logits(
        self,
        stage1_logits: torch.Tensor,
        stage2_logits: torch.Tensor,
    ) -> torch.Tensor:
        stage1_pred = stage1_logits.argmax(dim=-1)
        stage2_pred_flat = stage2_logits.argmax(dim=-1) + 2
        pred = stage2_pred_flat.clone()
        pred[stage1_pred == STAGE1_SILENCE] = SILENCE_IDX
        pred[stage1_pred == STAGE1_UNKNOWN] = UNKNOWN_IDX
        return pred

    @torch.inference_mode()
    def _collect_predictions_hierarchical(self, loader: DataLoader) -> tuple[list[int], list[int]]:
        self.model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1]
            stage1_logits, stage2_logits = self.model(x)
            preds = self._hierarchical_preds_from_logits(stage1_logits, stage2_logits)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
        return all_preds, all_targets
