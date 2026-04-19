# Utils Module

This module contains utility functions and centralized project configuration.

## `constants.py` — Centralized Configuration

All magic numbers and default hyperparameters are defined in `constants.py` to avoid hardcoding values throughout the codebase.

### Audio Processing Constants

- **`SAMPLE_RATE`** (16000 Hz): Resampling frequency for all audio
- **`CLIP_DURATION`** (1.0 s): Expected duration per audio clip
- **`CLIP_SAMPLES`** (16000): Samples per clip (= SAMPLE_RATE × CLIP_DURATION)
- **`N_MELS`** (64): Number of Mel-frequency bands in spectrograms
- **`N_FFT`** (512): FFT window size
- **`HOP_LENGTH`** (160): Number of samples between successive frames
- **`F_MIN`** (20.0 Hz): Minimum frequency for Mel scale
- **`F_MAX`** (8000.0 Hz): Maximum frequency for Mel scale

### Model Architecture Constants

#### CNN Baseline
- **`CNN_CHANNELS`** ([32, 64, 128, 256]): Channel progression through ConvBlocks
- **`CNN_DROPOUT`** (0.1): Dropout rate

#### Keyword Transformer (KWT)
- **`KWT_SMALL_CONFIG`**: d=128, 4 layers, 4 heads, mlp_dim=256, dropout=0.1
- **`KWT_MEDIUM_CONFIG`**: d=192, 6 layers, 4 heads, mlp_dim=384, dropout=0.1

#### General
- **`TRUNC_NORM_STD`** (0.02): Standard deviation for `trunc_normal_` initialization
- **`AST_PRETRAINED_MODEL`**: HuggingFace checkpoint ID for Audio Spectrogram Transformer
- **`AST_FREEZE_N_LAYERS`** (6): Number of AST encoder layers to freeze during fine-tuning

### Training Constants

- **`BATCH_SIZE`** (128): Training batch size
- **`NUM_WORKERS`** (4): DataLoader worker processes
- **`MAX_EPOCHS`** (50): Maximum training epochs
- **`SEED_DEFAULT`** (42): Default random seed
- **`LEARNING_RATE`** (3.0e-4): AdamW initial learning rate
- **`WEIGHT_DECAY`** (1.0e-4): L2 regularization
- **`SCHEDULER_TYPE`** ("cosine"): LR scheduler ("cosine" or "onecycle")
- **`WARMUP_EPOCHS`** (5): Number of warmup epochs for OneCycleLR
- **`GRAD_CLIP_NORM`** (1.0): Gradient clipping threshold
- **`EARLY_STOPPING_PATIENCE`** (10): Epochs without improvement before stopping
- **`LABEL_SMOOTHING`** (0.0): Label smoothing for cross-entropy loss

### Data Augmentation Constants

- **`AUGMENTATION_ENABLED`** (True): Whether to apply augmentations during training
- **`TIME_SHIFT_SAMPLES`** (1600): Maximum circular time shift
- **`NOISE_PROBABILITY`** (0.5): Probability of adding background noise
- **`NOISE_SNR_MIN`** (5.0 dB): Minimum signal-to-noise ratio
- **`NOISE_SNR_MAX`** (20.0 dB): Maximum signal-to-noise ratio
- **`NOISE_SNR_RANGE`**: Tuple of (MIN, MAX) for random SNR
- **`SPEC_AUG_TIME_MASK_PARAM`** (25): Max time mask length (SpecAugment)
- **`SPEC_AUG_FREQ_MASK_PARAM`** (14): Max frequency mask length (SpecAugment)
- **`N_TIME_MASKS`** (2): Number of time masks to apply
- **`N_FREQ_MASKS`** (2): Number of frequency masks to apply

### Data Processing Constants

- **`SILENCE_FRACTION`** (0.10): Fraction of training set to add as silence
- **`N_SILENCE_FIXED`** (260): Fixed number of silence samples for val/test splits
- **`REBALANCE_ENABLED`** (False): Whether to use weighted sampling for class rebalancing

### Hierarchical Classification Constants

- **`STAGE1_SILENCE`** (0): Class index for silence in stage-1 (3-class)
- **`STAGE1_TARGET`** (1): Class index for target commands in stage-1
- **`STAGE1_UNKNOWN`** (2): Class index for unknown in stage-1
- **`NUM_STAGE1_CLASSES`** (3): Number of stage-1 superclasses
- **`NUM_STAGE2_CLASSES`** (10): Number of stage-2 command classes

### Output Paths

- **`CHECKPOINT_DIR`**: "outputs/checkpoints"
- **`LOG_DIR`**: "outputs/logs"
- **`FIGURES_DIR`**: "outputs/figures"
- **`TABLES_DIR`**: "outputs/tables"
- **`DATA_ROOT`**: "data/raw"

### Miscellaneous

- **`NUM_CLASSES`** (12): Total classes (10 commands + silence + unknown)
- **`DEVICE_AUTO`** ("auto"): Resolved at runtime to cuda/mps/cpu
- **`MIN_DELTA_EARLY_STOPPING`** (1e-4): Minimum improvement for EarlyStopping

---

## Usage

### Import Constants

```python
from src.utils.constants import (
    SAMPLE_RATE,
    BATCH_SIZE,
    MAX_EPOCHS,
    KWT_SMALL_CONFIG,
    # ... any other constants
)
```

### Override via Config Files

Constants are used as **defaults**. YAML configuration files (in `configs/`) can override them:

```yaml
# configs/data/default.yaml
data:
  sample_rate: 16000        # Uses SAMPLE_RATE from constants
  n_mels: 64                # Uses N_MELS from constants
  custom_param: 999         # New parameters can still be added
```

During training, config values override defaults:

```python
cfg = load_experiment_config("configs/experiments/A1.yaml")
n_mels = cfg.get("data", {}).get("n_mels", N_MELS)  # Prefers config, falls back to constant
```

---

## Other Modules

- **`config.py`**: YAML loading and deep merging (`load_experiment_config()`)
- **`io.py`**: JSON/CSV/pickle save/load
- **`logging.py`**: Logging setup with console + file handlers
