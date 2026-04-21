#!/usr/bin/env python3
"""Aggregate all per-run metric JSON files into summary tables.

Usage:
    python scripts/export_results.py --tables-dir outputs/tables
"""
from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import save_csv, save_json

METRIC_KEYS = [
    "accuracy", "macro_f1", "recall_silence", "recall_unknown", "n_params"
]


def main(args: argparse.Namespace) -> None:
    tables_dir = Path(args.tables_dir)
    metric_files = sorted(f for f in tables_dir.rglob("*_metrics.json") if f.name != "summary.json")
    history_files = sorted(tables_dir.rglob("*_history.json"))

    # Group by experiment ID and deduplicate by run_name.
    # Prefer *_metrics.json rows; fill missing runs from history[test].
    groups: dict[str, dict[str, dict]] = {}

    for f in metric_files:
        rel = f.relative_to(tables_dir)
        exp_id = rel.parts[0] if len(rel.parts) > 1 else f.stem.split("_")[0]
        run_name = f.stem.replace("_metrics", "")
        groups.setdefault(exp_id, {})[run_name] = json.loads(f.read_text())

    for f in history_files:
        rel = f.relative_to(tables_dir)
        exp_id = rel.parts[0] if len(rel.parts) > 1 else f.stem.split("_")[0]
        run_name = f.stem.replace("_history", "")

        run_map = groups.setdefault(exp_id, {})
        if run_name in run_map:
            continue

        payload = json.loads(f.read_text())
        test = payload.get("test", {}) if isinstance(payload, dict) else {}
        if not isinstance(test, dict) or not test:
            continue

        # Only keep flat numeric metrics used by this exporter.
        row: dict[str, float] = {}
        for key in METRIC_KEYS:
            value = test.get(key)
            if isinstance(value, (int, float)):
                row[key] = float(value)
        if row:
            run_map[run_name] = row

    if not groups:
        print("No metric JSON files or history test metrics found in", tables_dir)
        return

    summary_rows: list[dict] = []
    for exp_id, run_map in sorted(groups.items()):
        runs = list(run_map.values())
        row: dict = {"experiment": exp_id, "n_runs": len(runs)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs if key in r]
            if not vals:
                continue
            row[f"{key}_mean"] = float(statistics.fmean(vals))
            row[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
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
