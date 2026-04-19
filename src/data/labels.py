"""Label constants and mapping for 12-class Speech Commands."""
from __future__ import annotations

from src.utils.constants import NUM_CLASSES

TARGET_COMMANDS: list[str] = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

# Canonical class ordering: silence=0, unknown=1, then 10 commands
ALL_CLASSES: list[str] = [SILENCE_LABEL, UNKNOWN_LABEL] + TARGET_COMMANDS
# NUM_CLASSES imported from constants (12)

LABEL_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(ALL_CLASSES)}
IDX_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}

SILENCE_IDX: int = LABEL_TO_IDX[SILENCE_LABEL]
UNKNOWN_IDX: int = LABEL_TO_IDX[UNKNOWN_LABEL]
TARGET_COMMAND_INDICES: list[int] = [LABEL_TO_IDX[c] for c in TARGET_COMMANDS]

# -----------------------------------------------------------
# Hierarchical label spaces
# -----------------------------------------------------------
# Stage-1: 3 super-classes (imported from constants)
from src.utils.constants import STAGE1_SILENCE, STAGE1_TARGET, STAGE1_UNKNOWN, NUM_STAGE1_CLASSES

STAGE1_LABEL_MAP = {
    SILENCE_LABEL: STAGE1_SILENCE,
    UNKNOWN_LABEL: STAGE1_UNKNOWN,
}
for _cmd in TARGET_COMMANDS:
    STAGE1_LABEL_MAP[_cmd] = STAGE1_TARGET

# Stage-2: 10 target command classes (re-indexed 0..9)
STAGE2_LABEL_TO_IDX: dict[str, int] = {cmd: i for i, cmd in enumerate(TARGET_COMMANDS)}
STAGE2_IDX_TO_LABEL: dict[int, str] = {i: cmd for cmd, i in STAGE2_LABEL_TO_IDX.items()}
NUM_STAGE2_CLASSES = len(TARGET_COMMANDS)

BACKGROUND_NOISE_FOLDER = "_background_noise_"


def flat_label(raw: str) -> int:
    """Map raw Speech Commands word to flat 12-class index."""
    if raw in TARGET_COMMANDS:
        return LABEL_TO_IDX[raw]
    if raw in (SILENCE_LABEL, "_silence_"):
        return SILENCE_IDX
    return UNKNOWN_IDX
