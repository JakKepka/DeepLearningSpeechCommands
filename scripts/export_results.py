#!/usr/bin/env python3
"""Aggregate all per-run metric JSON files into summary tables.

Usage:
    python scripts/export_results.py --tables-dir outputs/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.io import save_csv, save_json

METRIC_KEYS = [
    "accuracy", "macro_f1", "recall_silence", "recall_unknown", "n_params"
]


def main(args: argparse.Namespace) -> None:
    tables_dir = Path(args.tables_dir)
    metric_files = sorted(tables_dir.glob("*_metrics.json"))

    if not metric_files:
        print("No metric JSON files found in", tables_dir)
        return

    # Group by experiment ID (first component before '_')
    groups: dict[str, list[dict]] = {}
    for f in metric_files:
        exp_id = f.stem.split("_")[0]
        groups.setdefault(exp_id, []).append(json.loads(f.read_text()))

    summary_rows: list[dict] = []
    for exp_id, runs in sorted(groups.items()):
        row: dict = {"experiment": exp_id, "n_runs": len(runs)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs if key in r]
            if not vals:
                continue
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    out_csv = tables_dir / "summary.csv"
    out_json = tables_dir / "summary.json"
    save_csv(summary_rows, out_csv)
    save_json(summary_rows, out_json)
    print(f"Summary saved to {out_csv} and {out_json}")
    print(f"\n{'EXP':<8} {'Accuracy':>10} {'±':>6} {'MacroF1':>10} {'±':>6} {'RecSilence':>12} {'RecUnknown':>12}")
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{row['experiment']:<8} "
            f"{row.get('accuracy_mean', float('nan')):>10.4f} "
            f"{row.get('accuracy_std', 0):>6.4f} "
            f"{row.get('macro_f1_mean', float('nan')):>10.4f} "
            f"{row.get('macro_f1_std', 0):>6.4f} "
            f"{row.get('recall_silence_mean', float('nan')):>12.4f} "
            f"{row.get('recall_unknown_mean', float('nan')):>12.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export aggregated results tables")
    parser.add_argument("--tables-dir", default="outputs/tables")
    main(parser.parse_args())
