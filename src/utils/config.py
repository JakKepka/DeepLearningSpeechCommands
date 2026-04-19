"""YAML configuration loading and merging utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple config dicts; later ones override earlier ones."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_update(result, cfg)
    return result


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_experiment_config(experiment_yaml: str | Path) -> dict[str, Any]:
    """Load experiment YAML that may reference default configs."""
    exp = load_yaml(experiment_yaml)
    exp_dir = Path(experiment_yaml).parent
    configs_root = exp_dir.parent  # configs/

    base: dict[str, Any] = {}
    for section_name, default_file in exp.get("defaults", {}).items():
        yaml_path = configs_root / section_name / f"{default_file}.yaml"
        if yaml_path.exists():
            section_cfg = load_yaml(yaml_path)
            _deep_update(base, section_cfg)

    # Experiment-level overrides (everything except "defaults")
    overrides = {k: v for k, v in exp.items() if k not in ("defaults", "experiment")}
    _deep_update(base, overrides)
    base["experiment"] = exp.get("experiment", {})
    return base
