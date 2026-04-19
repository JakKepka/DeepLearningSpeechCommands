#!/usr/bin/env python3
"""Training script — trains a single model for one seed.

Usage:
    python scripts/train.py --config configs/experiments/A1_cnn_baseline.yaml
    python scripts/train.py --config configs/experiments/A2_kwt_small.yaml --seed 123
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpeechCommandsDataset
from src.data.sampler import build_weighted_sampler
from src.data.transforms import AudioPipeline
from src.models import build_model
from src.training.losses import build_loss
from src.training.seed import set_seed
from src.training.trainer import Trainer
from src.utils.config import load_experiment_config
from src.utils.logging import setup_logging
from src.utils.constants import BATCH_SIZE, NUM_WORKERS, N_MELS, N_FFT, HOP_LENGTH


def main(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed)

    run_name = f"{cfg.get('experiment', {}).get('id', 'run')}_{cfg.get('experiment', {}).get('name', 'model').replace(' ', '_')}_seed{seed}"
    log_dir = train_cfg.get("log_dir", "outputs/logs")
    logger = setup_logging(log_dir=log_dir, name=run_name)
    logger.info(f"Config: {args.config} | seed: {seed} | run: {run_name}")

    root = data_cfg.get("root", "./data/raw")
    n_mels = data_cfg.get("n_mels", N_MELS)
    n_fft = data_cfg.get("n_fft", N_FFT)
    hop_length = data_cfg.get("hop_length", HOP_LENGTH)
    noise_dir = Path(root) / "SpeechCommands" / "speech_commands_v0.01" / "_background_noise_"

    train_pipeline = AudioPipeline(
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment=aug_cfg.get("enabled", True),
        time_shift_samples=aug_cfg.get("time_shift_samples", 1600),
        noise_dir=noise_dir if noise_dir.exists() else None,
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

    # On macOS, use num_workers=0 to avoid audio backend issues with multiprocessing
    import sys
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
    use_pin_memory = device.type != "mps"

    if rebalance:
        sampler = build_weighted_sampler(train_ds.get_labels())
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg.get('model', {}).get('name')} | {n_params:,} parameters")

    class_counts = train_ds.class_counts() if rebalance else None
    criterion = build_loss(class_counts=class_counts, rebalance=rebalance, device=device)

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

    # Save history (dict with keys: history, test, meta)
    from src.utils.io import save_json
    tables_dir = Path(train_cfg.get("tables_dir", "outputs/tables"))
    save_json(result, tables_dir / f"{run_name}_history.json")
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
