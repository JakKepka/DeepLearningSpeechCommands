#!/usr/bin/env python3
"""Evaluate a trained model on the test split.

Usage:
    python scripts/evaluate.py \\
        --config configs/experiments/A1_cnn_baseline.yaml \\
        --checkpoint outputs/checkpoints/A1/A1_CNN_Baseline_ch32-64-128-256_do10_bs128_seed42_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpeechCommandsDataset
from src.data.transforms import AudioPipeline
from src.data.labels import ALL_CLASSES
from src.evaluation.reports import evaluate_model
from src.models import build_model
from src.utils.config import load_experiment_config
from src.utils.logging import setup_logging
from src.utils.paths import resolve_output_dirs
from src.utils.constants import BATCH_SIZE, NUM_WORKERS, N_MELS, N_FFT, HOP_LENGTH


def main(args: argparse.Namespace) -> None:
    cfg = load_experiment_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    experiment_cfg = cfg.get("experiment", {})
    experiment_id = experiment_cfg.get("id")
    train_cfg.update(resolve_output_dirs(train_cfg, experiment_id))

    logger = setup_logging(name="evaluate")

    root = data_cfg.get("root", "./data/raw")
    n_mels = data_cfg.get("n_mels", N_MELS)
    n_fft = data_cfg.get("n_fft", N_FFT)
    hop_length = data_cfg.get("hop_length", HOP_LENGTH)

    pipeline = AudioPipeline(n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, augment=False)
    test_ds = SpeechCommandsDataset(root=root, split="testing", transform=pipeline)
    
    # On macOS, use num_workers=0 to avoid audio backend issues
    import sys
    num_workers = train_cfg.get("num_workers", NUM_WORKERS)
    if sys.platform == "darwin" and num_workers > 0:
        logger.warning("macOS detected: setting num_workers=0 for audio backend compatibility")
        num_workers = 0
    
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.get("batch_size", BATCH_SIZE),
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(cfg)
    ckpt_path = args.checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    # Support both raw state_dict and wrapped checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)
    model = model.to(device)

    run_name = Path(ckpt_path).stem
    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        class_names=ALL_CLASSES,
        run_name=run_name,
        figures_dir=train_cfg.get("figures_dir", "outputs/figures"),
        tables_dir=train_cfg.get("tables_dir", "outputs/tables"),
    )

    logger.info("=== Test Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<30s}: {v:.4f}")
        else:
            logger.info(f"  {k:<30s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pt checkpoint")
    main(parser.parse_args())
