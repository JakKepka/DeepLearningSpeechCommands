"""Feature caching utilities for faster training.

This module allows precomputing transformed features (e.g. log-Mel tensors)
to disk once, then reusing them across epochs/runs via memory-mapped arrays.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MMapFeatureDataset(Dataset):
    """Dataset backed by feature/label memmap files."""

    def __init__(self, cache_dir: str | Path):
        cache_path = Path(cache_dir)
        meta = json.loads((cache_path / "meta.json").read_text(encoding="utf-8"))

        self._shape = tuple(meta["shape"])
        self._dtype = np.dtype(meta.get("dtype", "float32"))
        self._features = np.memmap(
            cache_path / "features.dat",
            mode="r",
            dtype=self._dtype,
            shape=self._shape,
        )
        self._labels = np.memmap(
            cache_path / "labels.dat",
            mode="r",
            dtype=np.int64,
            shape=(self._shape[0],),
        )

    def __len__(self) -> int:
        return int(self._shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(np.asarray(self._features[idx], dtype=np.float32))
        y = int(self._labels[idx])
        return x, y


def _cache_exists(cache_dir: Path) -> bool:
    return (
        (cache_dir / "meta.json").exists()
        and (cache_dir / "features.dat").exists()
        and (cache_dir / "labels.dat").exists()
    )


def precompute_to_memmap(
    dataset: Dataset,
    cache_dir: str | Path,
    *,
    batch_size: int = 256,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    dtype: str = "float32",
    overwrite: bool = False,
) -> MMapFeatureDataset:
    """Build on-disk feature cache and return a memmap-backed dataset."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if _cache_exists(cache_path) and not overwrite:
        return MMapFeatureDataset(cache_path)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    total = len(dataset)
    write_dtype = np.float16 if dtype == "float16" else np.float32

    feat_mm: np.memmap | None = None
    lbl_mm = np.memmap(
        cache_path / "labels.dat",
        mode="w+",
        dtype=np.int64,
        shape=(total,),
    )

    offset = 0
    shape: tuple[int, ...] | None = None
    for xb, yb in loader:
        xb_np = xb.cpu().numpy().astype(write_dtype, copy=False)
        yb_np = yb.cpu().numpy().astype(np.int64, copy=False)

        if feat_mm is None:
            shape = (total, *xb_np.shape[1:])
            feat_mm = np.memmap(
                cache_path / "features.dat",
                mode="w+",
                dtype=write_dtype,
                shape=shape,
            )

        bs = xb_np.shape[0]
        feat_mm[offset : offset + bs] = xb_np
        lbl_mm[offset : offset + bs] = yb_np
        offset += bs

    if feat_mm is None or shape is None:
        raise RuntimeError("Feature cache build failed: dataset appears empty.")

    feat_mm.flush()
    lbl_mm.flush()

    meta = {
        "shape": list(shape),
        "dtype": str(write_dtype),
    }
    (cache_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return MMapFeatureDataset(cache_path)