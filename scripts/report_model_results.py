#!/usr/bin/env python3
"""Create per-model training/test reports from outputs/tables.

For each model directory in outputs/tables/{model}, the script reads all
*_history.json files and produces artifacts in outputs/figures/{model}/summary:
    - training_curves_bs{batch}.png
    - confusion_matrix_bs{batch}.png
  - test_metrics_by_batch.png
  - test_results.csv
  - test_summary_by_batch.csv
  - analysis.txt

Additionally, it creates a global comparison folder:
    outputs/figures/all_models/summary

Usage:
    python scripts/report_model_results.py --tables-dir outputs/tables --figures-dir outputs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.labels import ALL_CLASSES


plt.style.use("seaborn-v0_8-whitegrid")


RUN_SEED_RE = re.compile(r"seed(\d+)")
RUN_BS_RE = re.compile(r"bs(\d+)")
RUN_EMB_RE = re.compile(r"(?:emb|embedding)(\d+)")


def _parse_seed(run_name: str) -> int | None:
    m = RUN_SEED_RE.search(run_name)
    return int(m.group(1)) if m else None


def _parse_batch_size(run_name: str) -> int | None:
    m = RUN_BS_RE.search(run_name)
    return int(m.group(1)) if m else None


def _parse_embedding_id(run_name: str) -> int | None:
    m = RUN_EMB_RE.search(run_name.lower())
    return int(m.group(1)) if m else None


def _load_history(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "history" not in data:
            return None
        return data
    except Exception:
        return None


def _group_history_files(tables_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(tables_dir.rglob("*_history.json")):
        rel = path.relative_to(tables_dir)
        model_id = rel.parts[0] if len(rel.parts) > 1 else path.stem.split("_")[0]
        groups[model_id].append(path)
    return groups


def _stack_with_nan(curves: list[list[float]]) -> np.ndarray:
    if not curves:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, : len(curve)] = np.asarray(curve, dtype=float)
    return arr


def _group_runs_by_batch(runs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        batch_size = run.get("batch_size")
        if batch_size is None:
            continue
        grouped[int(batch_size)].append(run)
    return dict(sorted(grouped.items()))


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )


def _style_axes_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_edgecolor("#334155")


def _save_training_curves(model_id: str, batch_size: int, runs: list[dict[str, Any]], out_path: Path) -> None:
    train_loss_curves: list[list[float]] = []
    val_loss_curves: list[list[float]] = []
    train_acc_curves: list[list[float]] = []
    val_acc_curves: list[list[float]] = []

    for run in runs:
        history = run.get("history", [])
        if not history:
            continue
        train_loss_curves.append([float(r.get("train_loss", math.nan)) for r in history])
        val_loss_curves.append([float(r.get("val_loss", math.nan)) for r in history])
        train_acc_curves.append([float(r.get("train_acc", math.nan)) for r in history])
        val_acc_curves.append([float(r.get("val_acc", math.nan)) for r in history])

    if not train_loss_curves:
        return

    tl = _stack_with_nan(train_loss_curves)
    vl = _stack_with_nan(val_loss_curves)
    ta = _stack_with_nan(train_acc_curves)
    va = _stack_with_nan(val_acc_curves)

    epochs = np.arange(1, max(tl.shape[1], ta.shape[1]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"{model_id} - training curves (batch_size={batch_size})", fontsize=14, fontweight="bold")

    for curve in train_loss_curves:
        axes[0].plot(np.arange(1, len(curve) + 1), curve, color="#2563eb", alpha=0.18, linewidth=1.5)
    for curve in val_loss_curves:
        axes[0].plot(np.arange(1, len(curve) + 1), curve, color="#f97316", alpha=0.18, linewidth=1.5)

    tl_mean = np.nanmean(tl, axis=0)
    tl_std = np.nanstd(tl, axis=0)
    vl_mean = np.nanmean(vl, axis=0)
    vl_std = np.nanstd(vl, axis=0)

    x_l = np.arange(1, len(tl_mean) + 1)
    axes[0].plot(x_l, tl_mean, color="#1d4ed8", label="train mean", linewidth=2.6)
    axes[0].fill_between(x_l, tl_mean - tl_std, tl_mean + tl_std, color="#60a5fa", alpha=0.22, label="train ± std over seeds")
    axes[0].plot(x_l, vl_mean, color="#ea580c", label="val mean", linewidth=2.6)
    axes[0].fill_between(x_l, vl_mean - vl_std, vl_mean + vl_std, color="#fdba74", alpha=0.22, label="val ± std over seeds")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    _style_axes_frame(axes[0])

    for curve in train_acc_curves:
        axes[1].plot(np.arange(1, len(curve) + 1), curve, color="#16a34a", alpha=0.18, linewidth=1.5)
    for curve in val_acc_curves:
        axes[1].plot(np.arange(1, len(curve) + 1), curve, color="#dc2626", alpha=0.18, linewidth=1.5)

    ta_mean = np.nanmean(ta, axis=0)
    ta_std = np.nanstd(ta, axis=0)
    va_mean = np.nanmean(va, axis=0)
    va_std = np.nanstd(va, axis=0)

    x_a = np.arange(1, len(ta_mean) + 1)
    axes[1].plot(x_a, ta_mean, color="#15803d", label="train mean", linewidth=2.6)
    axes[1].fill_between(x_a, ta_mean - ta_std, ta_mean + ta_std, color="#86efac", alpha=0.22, label="train ± std over seeds")
    axes[1].plot(x_a, va_mean, color="#b91c1c", label="val mean", linewidth=2.6)
    axes[1].fill_between(x_a, va_mean - va_std, va_mean + va_std, color="#fca5a5", alpha=0.22, label="val ± std over seeds")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    _style_axes_frame(axes[1])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(model_id: str, batch_size: int, runs: list[dict[str, Any]], out_path: Path) -> int:
    cms: list[np.ndarray] = []
    for run in runs:
        test = run.get("test", {})
        cm = test.get("confusion_matrix") if isinstance(test, dict) else None
        if cm is None:
            continue
        cm_arr = np.asarray(cm, dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm_arr / row_sums
        cms.append(cm_norm)

    if not cms:
        return 0

    cm_mean = np.mean(cms, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_mean,
        annot=True,
        fmt=".2f",
        xticklabels=ALL_CLASSES,
        yticklabels=ALL_CLASSES,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"{model_id} - mean confusion matrix (batch_size={batch_size})", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _style_axes_frame(ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(cms)


def _save_test_reports(model_id: str, runs: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for run in runs:
        test = run.get("test", {})
        if not isinstance(test, dict) or not test:
            continue
        row = {
            "model": model_id,
            "run_name": run.get("run_name"),
            "seed": run.get("seed"),
            "batch_size": run.get("batch_size"),
            "embedding": run.get("embedding"),
            "accuracy": test.get("accuracy"),
            "macro_f1": test.get("macro_f1"),
            "macro_precision": test.get("macro_precision"),
            "macro_recall": test.get("macro_recall"),
            "best_epoch": test.get("best_epoch"),
            "best_val_acc": test.get("best_val_acc"),
        }
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "test_results.csv"

    if rows:
        with open(raw_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bs = row.get("batch_size")
        if bs is not None:
            by_batch[int(bs)].append(row)

    batch_summary_rows: list[dict[str, Any]] = []
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    for bs, batch_rows in sorted(by_batch.items()):
        out: dict[str, Any] = {"batch_size": bs, "n_runs": len(batch_rows)}
        for metric in metrics:
            vals = [float(r[metric]) for r in batch_rows if r.get(metric) is not None]
            if vals:
                out[f"{metric}_mean"] = float(np.mean(vals))
                out[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        batch_summary_rows.append(out)

    batch_csv = out_dir / "test_summary_by_batch.csv"
    if batch_summary_rows:
        with open(batch_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(batch_summary_rows)

    # Plot test accuracy + macro_f1 by batch size
    if batch_summary_rows:
        x = np.arange(len(batch_summary_rows))
        labels = [str(r["batch_size"]) for r in batch_summary_rows]
        acc = [r.get("accuracy_mean", np.nan) for r in batch_summary_rows]
        f1 = [r.get("macro_f1_mean", np.nan) for r in batch_summary_rows]

        fig, ax = plt.subplots(figsize=(10, 5.8))
        width = 0.35
        acc_bars = ax.bar(x - width / 2, acc, width, label="accuracy", color="#2563eb")
        f1_bars = ax.bar(x + width / 2, f1, width, label="macro_f1", color="#ea580c")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{model_id} - test metrics by batch size (mean over seeds)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        _style_axes_frame(ax)
        _annotate_bars(ax, acc_bars)
        _annotate_bars(ax, f1_bars)
        plt.tight_layout()
        fig.savefig(str(out_dir / "test_metrics_by_batch.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Human-readable analysis
    analysis_lines: list[str] = []
    analysis_lines.append(f"Model: {model_id}")
    analysis_lines.append(f"Runs with test metrics: {len(rows)}")
    analysis_lines.append(
        "Training curves are grouped by batch size. Shaded bands show standard deviation over different seeds only, not over batch sizes."
    )
    if batch_summary_rows:
        best = max(
            batch_summary_rows,
            key=lambda r: float(r.get("macro_f1_mean", -1.0)),
        )
        analysis_lines.append(
            "Best batch size by macro_f1_mean: "
            f"{best['batch_size']} (macro_f1_mean={best.get('macro_f1_mean', float('nan')):.4f}, "
            f"accuracy_mean={best.get('accuracy_mean', float('nan')):.4f})"
        )
    else:
        analysis_lines.append("No test metrics available to compute batch-size analysis.")

    with open(out_dir / "analysis.txt", "w") as f:
        f.write("\n".join(analysis_lines) + "\n")

    return {
        "n_test_runs": len(rows),
        "n_batches": len(batch_summary_rows),
        "rows": rows,
        "batch_summary_rows": batch_summary_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_all_models_comparison(
    figures_dir: Path,
    all_test_rows: list[dict[str, Any]],
    all_batch_rows: list[dict[str, Any]],
) -> None:
    out_dir = figures_dir / "all_models" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "all_models_test_results.csv", all_test_rows)
    _write_csv(out_dir / "all_models_summary_by_model_batch.csv", all_batch_rows)

    if not all_test_rows:
        return

    # Overall summary per model across all runs
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_test_rows:
        by_model[str(row["model"])].append(row)

    overall_rows: list[dict[str, Any]] = []
    for model_id, rows in sorted(by_model.items()):
        out: dict[str, Any] = {"model": model_id, "n_runs": len(rows)}
        for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            if vals:
                out[f"{metric}_mean"] = float(np.mean(vals))
                out[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        overall_rows.append(out)

    _write_csv(out_dir / "all_models_summary_by_model.csv", overall_rows)

    # Summary per embedding group (if embedding tags exist in run names)
    emb_rows = [r for r in all_test_rows if r.get("embedding") is not None]
    if emb_rows:
        by_emb: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in emb_rows:
            by_emb[int(row["embedding"])] .append(row)

        emb_summary_rows: list[dict[str, Any]] = []
        for emb_id, rows in sorted(by_emb.items()):
            out: dict[str, Any] = {"embedding": emb_id, "n_runs": len(rows)}
            for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
                vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
                if vals:
                    out[f"{metric}_mean"] = float(np.mean(vals))
                    out[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            emb_summary_rows.append(out)

        _write_csv(out_dir / "all_models_summary_by_embedding.csv", emb_summary_rows)

        # Separate charts per embedding
        for emb_id in sorted(by_emb.keys()):
            emb_batch_rows = [r for r in all_batch_rows if r.get("embedding") == emb_id]
            if not emb_batch_rows:
                continue

            best_batch_rows: list[dict[str, Any]] = []
            by_model_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in emb_batch_rows:
                by_model_batch[str(row["model"])].append(row)

            for model_id, rows in sorted(by_model_batch.items()):
                best = max(rows, key=lambda r: float(r.get("macro_f1_mean", -1.0)))
                best_batch_rows.append(best)

            if best_batch_rows:
                x = np.arange(len(best_batch_rows))
                labels = [str(r["model"]) for r in best_batch_rows]
                acc = [float(r.get("accuracy_mean", np.nan)) for r in best_batch_rows]
                f1 = [float(r.get("macro_f1_mean", np.nan)) for r in best_batch_rows]

                fig, ax = plt.subplots(figsize=(11, 6))
                width = 0.36
                acc_bars = ax.bar(x - width / 2, acc, width, color="#2563eb", label="accuracy")
                f1_bars = ax.bar(x + width / 2, f1, width, color="#ea580c", label="macro_f1")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylim(0.0, 1.02)
                ax.set_xlabel("Model")
                ax.set_ylabel("Score")
                ax.set_title(
                    f"All models - best batch per model for embedding {emb_id} (mean over seeds)",
                    fontweight="bold",
                )
                ax.grid(axis="y", alpha=0.3)
                ax.legend()
                _style_axes_frame(ax)
                _annotate_bars(ax, acc_bars)
                _annotate_bars(ax, f1_bars)
                plt.tight_layout()
                fig.savefig(
                    str(out_dir / f"all_models_best_batch_comparison_embedding{emb_id}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

    # Pick best batch per model (by macro_f1_mean) from batch summaries
    best_batch_rows: list[dict[str, Any]] = []
    by_model_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_batch_rows:
        by_model_batch[str(row["model"])].append(row)

    for model_id, rows in sorted(by_model_batch.items()):
        best = max(rows, key=lambda r: float(r.get("macro_f1_mean", -1.0)))
        best_batch_rows.append(best)

    if best_batch_rows:
        x = np.arange(len(best_batch_rows))
        labels = [str(r["model"]) for r in best_batch_rows]
        acc = [float(r.get("accuracy_mean", np.nan)) for r in best_batch_rows]
        f1 = [float(r.get("macro_f1_mean", np.nan)) for r in best_batch_rows]

        fig, ax = plt.subplots(figsize=(11, 6))
        width = 0.36
        acc_bars = ax.bar(x - width / 2, acc, width, color="#2563eb", label="accuracy")
        f1_bars = ax.bar(x + width / 2, f1, width, color="#ea580c", label="macro_f1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("All models - best batch per model (mean over seeds)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        _style_axes_frame(ax)
        _annotate_bars(ax, acc_bars)
        _annotate_bars(ax, f1_bars)
        plt.tight_layout()
        fig.savefig(str(out_dir / "all_models_best_batch_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Heatmap: macro F1 by model x batch size
    if all_batch_rows:
        models = sorted({str(r["model"]) for r in all_batch_rows})
        batches = sorted({int(r["batch_size"]) for r in all_batch_rows})
        model_idx = {m: i for i, m in enumerate(models)}
        batch_idx = {b: i for i, b in enumerate(batches)}
        matrix = np.full((len(models), len(batches)), np.nan, dtype=float)

        for row in all_batch_rows:
            m = str(row["model"])
            b = int(row["batch_size"])
            matrix[model_idx[m], batch_idx[b]] = float(row.get("macro_f1_mean", np.nan))

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrBr",
            xticklabels=[str(b) for b in batches],
            yticklabels=models,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title("Macro F1 mean by model and batch size", fontweight="bold")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Model")
        _style_axes_frame(ax)
        plt.tight_layout()
        fig.savefig(str(out_dir / "all_models_macro_f1_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Short global analysis
    lines: list[str] = []
    lines.append("Global comparison across all models")
    lines.append(f"Runs included: {len(all_test_rows)}")
    lines.append(f"Model-batch groups: {len(all_batch_rows)}")
    if best_batch_rows:
        top = max(best_batch_rows, key=lambda r: float(r.get("macro_f1_mean", -1.0)))
        lines.append(
            "Best model+batch by macro_f1_mean: "
            f"{top['model']} @ bs={top['batch_size']} "
            f"(macro_f1_mean={float(top.get('macro_f1_mean', float('nan'))):.4f}, "
            f"accuracy_mean={float(top.get('accuracy_mean', float('nan'))):.4f})"
        )
    with open(out_dir / "analysis.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def _collect_runs(history_files: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in history_files:
        data = _load_history(path)
        if data is None:
            continue
        run_name = path.stem.replace("_history", "")
        runs.append(
            {
                "run_name": run_name,
                "seed": _parse_seed(run_name),
                "batch_size": _parse_batch_size(run_name),
                "embedding": _parse_embedding_id(run_name),
                "history": data.get("history", []),
                "test": data.get("test", {}),
                "meta": data.get("meta", {}),
            }
        )
    return runs


def _cleanup_legacy_outputs(out_dir: Path) -> None:
    for old_name in ("training_curves.png", "confusion_matrix_mean.png"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()


def main(args: argparse.Namespace) -> None:
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    grouped = _group_history_files(tables_dir)
    if not grouped:
        print(f"No *_history.json files found in {tables_dir}")
        return

    all_test_rows: list[dict[str, Any]] = []
    all_batch_rows: list[dict[str, Any]] = []

    for model_id, history_files in sorted(grouped.items()):
        runs = _collect_runs(history_files)
        if not runs:
            print(f"[{model_id}] skipped (no valid history files)")
            continue

        out_dir = figures_dir / model_id / "summary"
        _cleanup_legacy_outputs(out_dir)
        runs_by_batch = _group_runs_by_batch(runs)
        n_cm = 0
        for batch_size, batch_runs in runs_by_batch.items():
            _save_training_curves(
                model_id,
                batch_size,
                batch_runs,
                out_dir / f"training_curves_bs{batch_size}.png",
            )
            n_cm += _save_confusion_matrix(
                model_id,
                batch_size,
                batch_runs,
                out_dir / f"confusion_matrix_bs{batch_size}.png",
            )
        info = _save_test_reports(model_id, runs, out_dir)
        for row in info.get("rows", []):
            all_test_rows.append(row)
        for row in info.get("batch_summary_rows", []):
            emb_id = None
            for run in runs:
                if run.get("batch_size") == row.get("batch_size") and run.get("embedding") is not None:
                    emb_id = run.get("embedding")
                    break
            all_batch_rows.append({"model": model_id, "embedding": emb_id, **row})

        print(
            f"[{model_id}] runs={len(runs)} | batches={len(runs_by_batch)} | "
            f"test_runs={info['n_test_runs']} | conf_mats={n_cm} | saved={out_dir}"
        )

    _save_all_models_comparison(figures_dir, all_test_rows, all_batch_rows)
    print(f"[ALL_MODELS] saved={figures_dir / 'all_models' / 'summary'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create per-model report figures and summaries")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    main(parser.parse_args())
