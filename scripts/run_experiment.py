#!/usr/bin/env python3
"""Run an experiment with multiple seeds using a single config file.

Usage:
    python scripts/run_experiment.py --config configs/experiments/C1_flat.yaml
    python scripts/run_experiment.py --config configs/experiments/A2_kwt.yaml --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    seeds: list[int] = args.seeds if args.seeds else _get_seeds_from_config(cfg_path)

    print(f"Experiment: {cfg_path.name} | seeds: {seeds}")
    for seed in seeds:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "train.py"),
            "--config", str(cfg_path),
            "--seed", str(seed),
        ]
        print(f"\n{'='*60}\n  seed={seed}\n{'='*60}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[WARNING] seed={seed} exited with code {result.returncode}")


def _get_seeds_from_config(cfg_path: Path) -> list[int]:
    import yaml
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    return data.get("experiment", {}).get("seeds", [42])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment across multiple seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds from config")
    main(parser.parse_args())
