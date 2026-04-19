"""Global constants: audio processing, model architecture, training hyperparameters."""
from __future__ import annotations

# =========================================================================
# Audio Processing
# =========================================================================

# Sample rate and duration
SAMPLE_RATE: int = 16000  # Hz
CLIP_DURATION: float = 1.0  # seconds
CLIP_SAMPLES: int = SAMPLE_RATE * int(CLIP_DURATION)  # 16000 samples

# Log-Mel Spectrogram extraction
N_MELS: int = 64
N_FFT: int = 512
HOP_LENGTH: int = 160
F_MIN: float = 20.0
F_MAX: float = 8000.0

# =========================================================================
# Model Architecture
# =========================================================================

# CNN Baseline
CNN_CHANNELS: list[int] = [32, 64, 128, 256]
CNN_DROPOUT: float = 0.1

# Keyword Transformer (KWT)
KWT_SMALL_CONFIG: dict = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "mlp_dim": 256,
    "dropout": 0.1,
}

KWT_MEDIUM_CONFIG: dict = {
    "d_model": 192,
    "n_heads": 4,
    "n_layers": 6,
    "mlp_dim": 384,
    "dropout": 0.1,
}

# Transformer initialization
TRUNC_NORM_STD: float = 0.02

# Audio Spectrogram Transformer (AST)
AST_PRETRAINED_MODEL: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FREEZE_N_LAYERS: int = 6

# =========================================================================
# Training
# =========================================================================

# Batch and data
BATCH_SIZE: int = 128
NUM_WORKERS: int = 4
MAX_EPOCHS: int = 50
SEED_DEFAULT: int = 42

# Optimizer
LEARNING_RATE: float = 3.0e-4
WEIGHT_DECAY: float = 1.0e-4

# Scheduler
SCHEDULER_TYPE: str = "cosine"  # "cosine" or "onecycle"
WARMUP_EPOCHS: int = 5

# Gradient clipping
GRAD_CLIP_NORM: float = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE: int = 10

# =========================================================================
# Data Augmentation
# =========================================================================

AUGMENTATION_ENABLED: bool = True

# Time-domain augmentation
TIME_SHIFT_SAMPLES: int = 1600

# Background noise injection
NOISE_PROBABILITY: float = 0.5
NOISE_SNR_MIN: float = 5.0
NOISE_SNR_MAX: float = 20.0
NOISE_SNR_RANGE: tuple[float, float] = (NOISE_SNR_MIN, NOISE_SNR_MAX)

# SpecAugment (frequency-time masking)
SPEC_AUG_TIME_MASK_PARAM: int = 25
SPEC_AUG_FREQ_MASK_PARAM: int = 14
N_TIME_MASKS: int = 2
N_FREQ_MASKS: int = 2

# =========================================================================
# Data Processing
# =========================================================================

# Silence sample generation
SILENCE_FRACTION: float = 0.10  # fraction of training set
N_SILENCE_FIXED: int = 260  # fixed count for val/test splits

# Class rebalancing
REBALANCE_ENABLED: bool = False

# =========================================================================
# Hierarchical Classification
# =========================================================================

# Stage-1 superclass labels (in hierarchical pipeline)
STAGE1_SILENCE: int = 0
STAGE1_TARGET: int = 1
STAGE1_UNKNOWN: int = 2
NUM_STAGE1_CLASSES: int = 3
NUM_STAGE2_CLASSES: int = 10  # only target commands

# =========================================================================
# Output Paths (defaults)
# =========================================================================

CHECKPOINT_DIR: str = "./outputs/checkpoints"
LOG_DIR: str = "./outputs/logs"
FIGURES_DIR: str = "./outputs/figures"
TABLES_DIR: str = "./outputs/tables"
DATA_ROOT: str = "./data/raw"

# =========================================================================
# Miscellaneous
# =========================================================================

NUM_CLASSES: int = 12  # 10 commands + silence + unknown
DEVICE_AUTO: str = "auto"  # will be resolved to cuda/mps/cpu at runtime
LABEL_SMOOTHING: float = 0.0
MIN_DELTA_EARLY_STOPPING: float = 1e-4
