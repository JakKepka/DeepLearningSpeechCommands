#!/usr/bin/env python3
"""Training script — trains a single model for one seed.

Usage:
    python scripts/train.py --config configs/experiments/A1_cnn_baseline.yaml
    python scripts/train.py --config configs/experiments/A2_kwt.yaml --seed 123
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpeechCommandsDataset
from src.data.feature_cache import precompute_to_memmap
from src.data.sampler import build_weighted_sampler
from src.data.transforms import AudioPipeline
from src.models import build_model
from src.models.hierarchical import HierarchicalClassifier
from src.training.losses import build_loss
from src.training.seed import set_seed
from src.training.trainer import Trainer
from src.utils.config import load_experiment_config
from src.utils.logging import setup_logging
from src.utils.paths import resolve_output_dirs
from src.utils.run_names import build_run_name
from src.utils.constants import BATCH_SIZE, NUM_WORKERS, N_MELS, N_FFT, HOP_LENGTH


def resolve_noise_dir(root: str | Path) -> Path | None:
    root_path = Path(root)
    candidates = [
        root_path / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_",
        root_path / "SpeechCommands" / "speech_commands_v0.01" / "_background_noise_",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in (root_path / "SpeechCommands").rglob("_background_noise_"):
        if candidate.is_dir():
            return candidate
    return None


def log_training_start_report(
    logger,
    *,
    config_path: str,
    run_name: str,
    experiment_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    aug_cfg: dict[str, Any],
    root: str,
    noise_dir: Path | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    rebalance: bool,
    rebalance_mode: str,
    hierarchical: bool,
    n_params: int,
    train_size: int,
    val_size: int,
    test_size: int,
) -> None:
    scheduler = train_cfg.get("scheduler", "none")
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    optimizer = train_cfg.get("optimizer", "adamw")
    augment_enabled = bool(aug_cfg.get("enabled", True))
    spec_aug_cfg = aug_cfg.get("spec_aug", {})

    report_lines = [
        "Training start report:",
        f"  run={run_name}",
        f"  config={config_path}",
        f"  experiment={experiment_cfg.get('id')} ({experiment_cfg.get('name')})",
        f"  model={model_cfg.get('name')} | params={n_params:,} | hierarchical={hierarchical}",
        f"  device={device} | optimizer={optimizer} | lr={train_cfg.get('lr')} | weight_decay={train_cfg.get('weight_decay')}",
        f"  epochs={train_cfg.get('max_epochs')} | early_stop={train_cfg.get('early_stopping_patience')} | scheduler={scheduler} | warmup={warmup_epochs}",
        f"  batch_size={batch_size} | num_workers={num_workers} | prefetch_factor={prefetch_factor} | rebalance={rebalance} ({rebalance_mode})",
        f"  data_root={root} | train/val/test={train_size}/{val_size}/{test_size}",
        f"  augmentation={augment_enabled} | noise_dir={noise_dir if noise_dir is not None else 'not found'}",
    ]

    if augment_enabled:
        report_lines.append(
            "  aug_details="
            f"time_shift={aug_cfg.get('time_shift_samples', 1600)}, "
            f"noise_prob={aug_cfg.get('noise_prob', 0.5)}, "
            f"snr=({aug_cfg.get('noise_snr_min', 5.0)}, {aug_cfg.get('noise_snr_max', 20.0)}), "
            f"specaug_t={spec_aug_cfg.get('time_mask_param', 25)}x{spec_aug_cfg.get('n_time_masks', 2)}, "
            f"specaug_f={spec_aug_cfg.get('freq_mask_param', 14)}x{spec_aug_cfg.get('n_freq_masks', 2)}"
        )

    logger.info("\n".join(report_lines))


def main(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})
    model_cfg = cfg.get("model", {})
    experiment_cfg = cfg.get("experiment", {})
    experiment_id = experiment_cfg.get("id")

    train_cfg.update(resolve_output_dirs(train_cfg, experiment_id))

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed)

    # AST fine-tuning is not stable in the current local macOS setup.
    # Fail fast with a clear message instead of allowing a native segfault
    # during Hugging Face model initialisation.
    if sys.platform == "darwin" and str(model_cfg.get("name", "")).lower() == "ast":
        raise RuntimeError(
            "A4/AST training is not supported in the current macOS local setup. "
            "Use CUDA on Colab/Linux for this experiment. "
            "Recommended: run configs/experiments/A5_ast.yaml on Colab with GPU."
        )

    run_name = build_run_name(cfg, seed)
    log_dir = train_cfg.get("log_dir", "outputs/logs")
    logger = setup_logging(log_dir=log_dir, name=run_name)
    logger.info(f"Config: {args.config} | seed: {seed} | run: {run_name}")

    root = data_cfg.get("root", "./data/raw")
    n_mels = data_cfg.get("n_mels", N_MELS)
    n_fft = data_cfg.get("n_fft", N_FFT)
    hop_length = data_cfg.get("hop_length", HOP_LENGTH)
    noise_dir = resolve_noise_dir(root)

    train_pipeline = AudioPipeline(
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment=aug_cfg.get("enabled", True),
        time_shift_samples=aug_cfg.get("time_shift_samples", 1600),
        noise_dir=noise_dir,
        noise_prob=aug_cfg.get("noise_prob", 0.5),
        noise_snr_range=(
            aug_cfg.get("noise_snr_min", 5.0),
            aug_cfg.get("noise_snr_max", 20.0),
        ),
        spec_aug_time_mask=aug_cfg.get("spec_aug", {}).get("time_mask_param", 25),
        spec_aug_freq_mask=aug_cfg.get("spec_aug", {}).get("freq_mask_param", 14),
        n_time_masks=aug_cfg.get("spec_aug", {}).get("n_time_masks", 2),
        n_freq_masks=aug_cfg.get("spec_aug", {}).get("n_freq_masks", 2),
    )
    eval_pipeline = AudioPipeline(n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, augment=False)

    train_ds = SpeechCommandsDataset(root=root, split="training", transform=train_pipeline)
    val_ds = SpeechCommandsDataset(root=root, split="validation", transform=eval_pipeline)
    test_ds = SpeechCommandsDataset(root=root, split="testing", transform=eval_pipeline)

    batch_size = train_cfg.get("batch_size", BATCH_SIZE)
    num_workers = train_cfg.get("num_workers", NUM_WORKERS)
    rebalance = train_cfg.get("rebalance", False)
    rebalance_mode = str(train_cfg.get("rebalance_mode", "sampler" if rebalance else "none")).lower()
    rebalance_strength = float(train_cfg.get("rebalance_strength", 1.0))
    hierarchical = bool(train_cfg.get("hierarchical", False))
    prefetch_factor = train_cfg.get("prefetch_factor", 4)

    # On macOS, use num_workers=0 to avoid audio backend issues with multiprocessing
    if sys.platform == "darwin":
        if num_workers > 0:
            logger.warning("macOS detected: setting num_workers=0 for audio backend compatibility")
            num_workers = 0

    # Determine device early to set pin_memory correctly
    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)

    # pin_memory is not supported on MPS
    use_pin_memory = device.type == "cuda"

    # Optional feature caching for faster epochs (best with augment=False).
    cache_mode = str(train_cfg.get("cache_features", "none")).lower()
    cache_dtype = str(train_cfg.get("cache_dtype", "float16"))
    cache_batch_size = int(train_cfg.get("cache_batch_size", batch_size))
    cache_overwrite = bool(train_cfg.get("cache_overwrite", False))
    cache_base = Path(train_cfg.get("feature_cache_dir", "./data/feature_cache"))

    train_aug_enabled = bool(aug_cfg.get("enabled", True))
    if cache_mode in {"val_test", "all"}:
        if cache_mode == "all" and train_aug_enabled:
            logger.warning(
                "cache_features=all with augmentation enabled would freeze augmentations. "
                "Switching to cache_features=val_test."
            )
            cache_mode = "val_test"

        cache_base.mkdir(parents=True, exist_ok=True)

        if cache_mode == "all":
            logger.info("Building/using feature cache for train split …")
            train_ds = precompute_to_memmap(
                train_ds,
                cache_base / "train",
                batch_size=cache_batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                dtype=cache_dtype,
                overwrite=cache_overwrite,
            )

        logger.info("Building/using feature cache for validation split …")
        val_ds = precompute_to_memmap(
            val_ds,
            cache_base / "validation",
            batch_size=cache_batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dtype=cache_dtype,
            overwrite=cache_overwrite,
        )

        logger.info("Building/using feature cache for test split …")
        test_ds = precompute_to_memmap(
            test_ds,
            cache_base / "testing",
            batch_size=cache_batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dtype=cache_dtype,
            overwrite=cache_overwrite,
        )

    use_sampler_rebalance = rebalance and rebalance_mode == "sampler"
    use_weight_rebalance = rebalance and rebalance_mode == "class_weights"

    if use_sampler_rebalance:
        sampler = build_weighted_sampler(train_ds.get_labels())
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if rebalance and rebalance_mode not in {"sampler", "class_weights"}:
        logger.warning(
            f"Unknown rebalance_mode={rebalance_mode!r}; falling back to sampler"
        )
        sampler = build_weighted_sampler(train_ds.get_labels())
        shuffle = False
        use_sampler_rebalance = True
        use_weight_rebalance = False

    dl_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle,
        sampler=sampler,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **dl_kwargs,
    )

    if hierarchical:
        # Build two independent models from the same base config.
        cfg_stage1 = copy.deepcopy(cfg)
        cfg_stage1.setdefault("model", {})["num_classes"] = 3
        cfg_stage2 = copy.deepcopy(cfg)
        cfg_stage2.setdefault("model", {})["num_classes"] = 10
        model = HierarchicalClassifier(
            stage1=build_model(cfg_stage1),
            stage2=build_model(cfg_stage2),
        )
        logger.info("Hierarchical mode enabled: training 2-stage model (3-class + 10-class)")
    else:
        model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg.get('model', {}).get('name')} | {n_params:,} parameters")

    log_training_start_report(
        logger,
        config_path=args.config,
        run_name=run_name,
        experiment_cfg=experiment_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        aug_cfg=aug_cfg,
        root=root,
        noise_dir=noise_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        rebalance=rebalance,
        rebalance_mode=rebalance_mode,
        hierarchical=hierarchical,
        n_params=n_params,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )

    class_counts = train_ds.class_counts() if use_weight_rebalance else None
    criterion = build_loss(
        class_counts=class_counts,
        rebalance=use_weight_rebalance,
        rebalance_strength=rebalance_strength,
        device=device,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        cfg=cfg,
        run_name=run_name,
        test_loader=test_loader,
    )
    result = trainer.train()

    # Save history and timing summaries into outputs/tables.
    from src.utils.io import save_csv, save_json
    tables_dir = Path(train_cfg.get("tables_dir", "outputs/tables"))
    save_json(result, tables_dir / f"{run_name}_history.json")

    timing_rows = []
    for row in result.get("history", []):
        timing_rows.append(
            {
                "epoch": row.get("epoch"),
                "epoch_time": row.get("epoch_time"),
                "cumulative_epoch_time": row.get("cumulative_epoch_time"),
                "train_loss": row.get("train_loss"),
                "val_loss": row.get("val_loss"),
                "train_acc": row.get("train_acc"),
                "val_acc": row.get("val_acc"),
                "lr": row.get("lr"),
            }
        )
    save_csv(timing_rows, tables_dir / f"{run_name}_timing.csv")
    save_json(result.get("meta", {}), tables_dir / f"{run_name}_timing_summary.json")

    best_val = result["meta"].get("best_val_acc")
    test_acc = result["test"].get("accuracy")
    logger.info(
        f"Training complete. Best val_acc: {best_val:.4f}"
        + (f" | Test acc: {test_acc:.4f}" if test_acc is not None else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech command classifier")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    main(parser.parse_args())
