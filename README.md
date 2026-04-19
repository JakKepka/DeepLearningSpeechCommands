# Speech Commands Classification with Transformers

> **Authors:** Jakub Kępka, Damian Kąkol
> **Course:** Deep Learning 

## Overview

This project investigates keyword spotting on the [Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) in a 12-class setting:

| Class                                                                                     | Type            |
| ----------------------------------------------------------------------------------------- | --------------- |
| `silence`                                                                               | special         |
| `unknown`                                                                               | special         |
| `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go` | target commands |

Three research questions are addressed:

1. Does a lightweight Transformer (KWT-style) outperform a simple CNN?
2. Does a pretrained audio Transformer (AST) further improve accuracy?
3. Does explicit handling of `silence`/`unknown` (class rebalancing or hierarchical pipeline) improve robustness?

---

## Project Structure

```
speech-commands/
├── configs/            # YAML experiment configs (data, model, train, experiments)
├── data/raw/           # Dataset downloaded here (auto-populated)
├── outputs/            # Checkpoints, logs, figures, result tables
├── scripts/            # Entry-point scripts
│   ├── prepare_data.py  – download & inspect dataset
│   ├── train.py         – train a single model
│   ├── evaluate.py      – evaluate on test split
│   ├── run_experiment.py – run an experiment across multiple seeds
│   └── export_results.py – aggregate metrics into summary tables
├── src/
│   ├── data/            – dataset, transforms, sampler, labels
│   ├── models/          – CNN baseline, KWT, AST wrapper, hierarchical
│   ├── training/        – trainer, losses, callbacks, seed
│   ├── evaluation/      – metrics, confusion matrix, reports
│   ├── inference/       – single-file predictor
│   └── utils/           – config loading, I/O, logging
└── tests/               – pytest unit & integration tests
```

---

## Quick Start

### 0. (macOS only) Install Audio Backend

On macOS, you need to install `libsndfile` for audio file loading:

```bash
# Option A: Use the setup script
bash scripts/setup_audio_backend.sh

# Option B: Manual installation
brew install libsndfile
```

If this step is skipped and the backend is missing, silence samples will be generated as zeros (the training will still work, but less realistic).

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

### 2. Download Dataset

```bash
python scripts/prepare_data.py --root ./data/raw
```

### 3. Train a Model

```bash
# CNN baseline
python scripts/train.py --config configs/experiments/A1_cnn_baseline.yaml

# KWT-small
python scripts/train.py --config configs/experiments/A2_kwt_small.yaml

# KWT-medium
python scripts/train.py --config configs/experiments/A3_kwt_medium.yaml

# AST fine-tuning
python scripts/train.py --config configs/experiments/A4_ast.yaml
```

### 4. Evaluate on Test Split

```bash
python scripts/evaluate.py \
    --config configs/experiments/A1_cnn_baseline.yaml \
    --checkpoint outputs/checkpoints/A1_CNN_Baseline_seed42_best.pt
```

### 5. Run Full Experiment (multiple seeds)

```bash
python scripts/run_experiment.py --config configs/experiments/C1_flat.yaml
# Seeds are taken from the config file (default: [42, 123, 456])
```

### 6. Export Summary Table

```bash
python scripts/export_results.py --tables-dir outputs/tables
```

---

## Models

| Model        | File                           | Configuration                        |
| ------------ | ------------------------------ | ------------------------------------ |
| CNN Baseline | `src/models/cnn_baseline.py` | 4 ConvBlocks → GAP → Linear        |
| KWT-small    | `src/models/kwt.py`          | d=128, 4 layers, 4 heads             |
| KWT-medium   | `src/models/kwt.py`          | d=192, 6 layers, 4 heads             |
| AST          | `src/models/ast_wrapper.py`  | Pretrained AudioSet, fine-tuned head |
| Hierarchical | `src/models/hierarchical.py` | Stage1 (3-class) + Stage2 (10-class) |

---

## Experiments

| ID | Model        | Strategy                | Seeds |
| -- | ------------ | ----------------------- | ----- |
| A1 | CNN Baseline | flat 12-class           | 1     |
| A2 | KWT-small    | flat 12-class           | 1     |
| A3 | KWT-medium   | flat 12-class           | 1     |
| A4 | AST          | fine-tuning             | 1     |
| C1 | KWT-small    | flat, no rebalancing    | 3     |
| C2 | KWT-small    | flat + weighted sampler | 3     |
| C3 | KWT-small    | hierarchical pipeline   | 3     |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Reported Metrics

Each run produces in `outputs/tables/`:

- `<run>_metrics.json` — accuracy, macro F1, per-class recall, n_params
- `<run>_metrics.csv` — same as CSV
- `<run>_history.json` — per-epoch train/val loss and accuracy

Each run produces in `outputs/figures/`:

- `<run>_confusion.png` — normalised confusion matrix
