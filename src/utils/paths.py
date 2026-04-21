"""Helpers for consistent output directory layout."""
from __future__ import annotations

from pathlib import Path


def with_experiment_subdir(path: str | Path, experiment_id: str | None) -> Path:
    """Return ``path/experiment_id`` unless it is already the final component."""
    p = Path(path)
    if not experiment_id:
        return p
    return p if p.name == experiment_id else p / experiment_id


def resolve_output_dirs(train_cfg: dict, experiment_id: str | None) -> dict[str, str]:
    """Resolve output directories for a run.

    Tables, figures and checkpoints are grouped by experiment ID to keep
    ``outputs/`` tidy, e.g. ``outputs/tables/A1/...``.
    """
    return {
        "tables_dir": str(with_experiment_subdir(train_cfg.get("tables_dir", "outputs/tables"), experiment_id)),
        "figures_dir": str(with_experiment_subdir(train_cfg.get("figures_dir", "outputs/figures"), experiment_id)),
        "checkpoint_dir": str(with_experiment_subdir(train_cfg.get("checkpoint_dir", "outputs/checkpoints"), experiment_id)),
    }