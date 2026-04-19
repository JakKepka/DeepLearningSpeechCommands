#!/usr/bin/env python3
"""Download and verify the Speech Commands dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torchaudio

from src.data.dataset import SpeechCommandsDataset
from src.data.labels import ALL_CLASSES


def main(args: argparse.Namespace) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Speech Commands to {root} …")
    for split in ("training", "validation", "testing"):
        print(f"  Loading {split} split …")
        ds = SpeechCommandsDataset(root=str(root), split=split, download=True)
        counts = ds.class_counts()
        total = len(ds)
        print(f"    {total} samples")
        for idx, name in enumerate(ALL_CLASSES):
            print(f"      {name:12s}: {counts[idx].item():6d}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Speech Commands dataset")
    parser.add_argument("--root", default="./data/raw", help="Root directory for dataset")
    main(parser.parse_args())
