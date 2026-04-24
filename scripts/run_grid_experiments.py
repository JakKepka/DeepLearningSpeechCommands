#!/usr/bin/env python3
"""Run grid of experiments across multiple models, seeds, and batch sizes.

Default parameters from configs/experiments/grid_search_defaults.yaml are merged with
experiment-specific configs, then overridden by command-line arguments.

Grid search defaults (can be changed in configs/experiments/grid_search_defaults.yaml):
  - models: [A1, A2, A3]
  - seeds: [42]
  - batch_sizes: [128, 256]
  - device: auto
  - num_workers: 4
  - augment: true

Usage:
    # Use all defaults from config
    python scripts/run_grid_experiments.py

    # Override models
    python scripts/run_grid_experiments.py --models A1 A2

    # Override everything
    python scripts/run_grid_experiments.py \
        --models A1 A2 A3 \
        --seeds 42 123 \
        --batch-sizes 256 512 \
        --max-epochs 15 \
        --lr 3.0e-4 \
        --early-stopping-patience 6 \
        --device cuda \
        --num-workers 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.utils.config import load_experiment_config


CONFIG_MAP = {
    "A1": "configs/experiments/A1_cnn_baseline.yaml",
    "A2": "configs/experiments/A2_kwt.yaml",
    "A3": "configs/experiments/A3_kwt_medium.yaml",
    "A4": "configs/experiments/A4_bc_resnet.yaml",
    "A5": "configs/experiments/A5_ast.yaml",
    "C1": "configs/experiments/C1_flat.yaml",
    "C2": "configs/experiments/C2_flat_rebalanced.yaml",
    "C3": "configs/experiments/C3_hierarchical.yaml",
    "C4": "configs/experiments/C4_hierarchical_bc_resnet.yaml",
}

GRID_DEFAULTS_CONFIG = "configs/experiments/grid_search_defaults.yaml"


def load_grid_defaults() -> dict:
    """Load default grid search parameters from config."""
    try:
        cfg = load_experiment_config(GRID_DEFAULTS_CONFIG)
        return cfg.get("grid_search", {})
    except Exception as e:
        print(f"[WARNING] Could not load grid search defaults: {e}")
        return {}


def load_grid_train_overrides() -> dict:
    """Load explicit train overrides from grid_search_defaults.yaml.

    Only values written directly in the file's ``train`` section are returned,
    so precedence is: train/default -> grid train overrides -> CLI overrides.
    """
    try:
        with open(GRID_DEFAULTS_CONFIG) as f:
            raw_cfg = yaml.safe_load(f) or {}
        return raw_cfg.get("train", {})
    except Exception as e:
        print(f"[WARNING] Could not load grid search training overrides: {e}")
        return {}


def run_training(
    model_id: str,
    config_path: str,
    seed: int,
    batch_size: int,
    device: str = "auto",
    augment: bool = True,
    num_workers: int = 4,
    max_epochs: int | None = None,
    lr: float | None = None,
    early_stopping_patience: int | None = None,
) -> int:
    """Load config, override batch_size, seed, and training params, then run training."""
    # Load experiment config (includes model, data, train from defaults)
    cfg = load_experiment_config(config_path)

    # Apply explicit grid-search train overrides (e.g. max_epochs=5)
    grid_train_overrides = load_grid_train_overrides()
    train = cfg.setdefault("train", {})
    train.update(grid_train_overrides)

    # Override train settings
    train["batch_size"] = batch_size
    train["seed"] = seed
    train["device"] = device
    train["num_workers"] = num_workers

    # Optional training parameter overrides
    if max_epochs is not None:
        train["max_epochs"] = max_epochs
    if lr is not None:
        train["lr"] = lr
    if early_stopping_patience is not None:
        train["early_stopping_patience"] = early_stopping_patience

    # Override augmentation if provided
    if not augment:
        aug = cfg.setdefault("augmentation", {})
        aug["enabled"] = False

    # Write temporary config
    tmp_cfg = f"/tmp/{model_id}_seed{seed}_bs{batch_size}.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f)

    # Run training
    cmd = [sys.executable, "scripts/train.py", "--config", tmp_cfg, "--seed", str(seed)]
    print(f"\n{'='*70}")
    print(f"  {model_id} | seed={seed} | bs={batch_size} | device={device}")
    if max_epochs is not None:
        print(f"  max_epochs={max_epochs} | lr={lr} | early_stop={early_stopping_patience}")
    print(f"{'='*70}")
    result = subprocess.run(cmd)
    return result.returncode


def main(args: argparse.Namespace) -> None:
    models = args.models or ["A1", "A2", "A3"]
    seeds = args.seeds or [42]
    batch_sizes = args.batch_sizes or [128, 256]
    device = args.device or "auto"
    augment = args.augment
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    lr = args.lr
    early_stopping_patience = args.early_stopping_patience

    total_runs = len(models) * len(seeds) * len(batch_sizes)
    completed = 0
    failed = 0

    print(f"Grid Experiment Plan:")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Total runs: {total_runs}")
    print(f"  Device: {device}")
    print(f"  Augment: {augment}")
    if max_epochs or lr or early_stopping_patience:
        print(f"  Training overrides:")
        if max_epochs:
            print(f"    max_epochs: {max_epochs}")
        if lr:
            print(f"    lr: {lr}")
        if early_stopping_patience:
            print(f"    early_stopping_patience: {early_stopping_patience}")
    print()

    for model_id in models:
        if model_id not in CONFIG_MAP:
            print(f"[ERROR] Unknown model: {model_id}")
            continue

        config_path = CONFIG_MAP[model_id]
        if not Path(config_path).exists():
            print(f"[ERROR] Config not found: {config_path}")
            continue

        for seed in seeds:
            for batch_size in batch_sizes:
                returncode = run_training(
                    model_id=model_id,
                    config_path=config_path,
                    seed=seed,
                    batch_size=batch_size,
                    device=device,
                    augment=augment,
                    num_workers=num_workers,
                    max_epochs=max_epochs,
                    lr=lr,
                    early_stopping_patience=early_stopping_patience,
                )

                completed += 1
                if returncode != 0:
                    failed += 1
                    print(f"[FAILED] {model_id} seed={seed} bs={batch_size}")
                else:
                    print(f"[OK] {model_id} seed={seed} bs={batch_size}")

    print(f"\n{'='*70}")
    print(f"Grid Experiment Summary:")
    print(f"  Total runs: {total_runs}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Load defaults from config
    grid_defaults = load_grid_defaults()

    parser = argparse.ArgumentParser(
        description="Run grid of experiments across models, seeds, and batch sizes"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=grid_defaults.get("models", ["A1", "A2", "A3"]),
        help=f"Model IDs to train (default from config: {grid_defaults.get('models', ['A1', 'A2', 'A3'])})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=grid_defaults.get("seeds", [42]),
        help=f"Random seeds to use (default from config: {grid_defaults.get('seeds', [42])})",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=grid_defaults.get("batch_sizes", [128, 256]),
        help=f"Batch sizes to try (default from config: {grid_defaults.get('batch_sizes', [128, 256])})",
    )
    parser.add_argument(
        "--device",
        default=grid_defaults.get("device", "auto"),
        help=f"Device: auto | cuda | mps | cpu (default from config: {grid_defaults.get('device', 'auto')})",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=grid_defaults.get("augment", True),
        help=f"Enable augmentation (default from config: {grid_defaults.get('augment', True)})",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="Disable augmentation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=grid_defaults.get("num_workers", 4),
        help=f"Number of DataLoader workers (default from config: {grid_defaults.get('num_workers', 4)})",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs (default: from configs/experiments/grid_search_defaults.yaml)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: from configs/experiments/grid_search_defaults.yaml)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Override early stopping patience (default: from configs/experiments/grid_search_defaults.yaml)",
    )
    main(parser.parse_args())
