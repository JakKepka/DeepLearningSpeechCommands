"""Hierarchical two-stage inference pipeline for silence / unknown / commands."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.constants import N_MELS
from src.data.labels import (
    SILENCE_IDX,
    STAGE1_SILENCE,
    STAGE1_TARGET,
    STAGE1_UNKNOWN,
    STAGE2_IDX_TO_LABEL,
    TARGET_COMMANDS,
    UNKNOWN_IDX,
    flat_label,
)


class HierarchicalClassifier(nn.Module):
    """Two-stage hierarchical classifier.

    Stage 1:  3-class — silence | target-command | unknown-or-other
    Stage 2: 10-class — identifies which target command

    At inference:
        - stage1 → silence   ⇒ final = silence (idx 0)
        - stage1 → unknown   ⇒ final = unknown (idx 1)
        - stage1 → target    ⇒ run stage2, map back to flat 12-class index
    """

    def __init__(self, stage1: nn.Module, stage2: nn.Module):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward: returns (stage1_logits, stage2_logits)."""
        return self.stage1(x), self.stage2(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Full hierarchical inference. Returns flat 12-class predictions (B,)."""
        stage1_logits = self.stage1(x)
        stage1_pred = stage1_logits.argmax(dim=-1)  # (B,)

        stage2_logits = self.stage2(x)
        stage2_pred = stage2_logits.argmax(dim=-1)  # (B,) — indices 0..9

        # Map stage-2 index → flat class index
        stage2_flat = torch.tensor(
            [flat_label(TARGET_COMMANDS[i.item()]) for i in stage2_pred],
            device=x.device,
        )

        result = torch.empty_like(stage1_pred)
        for i in range(x.shape[0]):
            if stage1_pred[i] == STAGE1_SILENCE:
                result[i] = SILENCE_IDX
            elif stage1_pred[i] == STAGE1_UNKNOWN:
                result[i] = UNKNOWN_IDX
            else:
                result[i] = stage2_flat[i]
        return result


def build_hierarchical(base_model_cls, n_mels: int = N_MELS, time_frames: int = 101, **model_kwargs):
    """Convenience builder: creates two independent model instances sharing the same arch.

    Only passes ``n_mels`` / ``time_frames`` if the model ``__init__`` accepts them.
    """
    import inspect
    sig = inspect.signature(base_model_cls.__init__).parameters
    kw = dict(**model_kwargs)
    if "n_mels" in sig:
        kw["n_mels"] = n_mels
    if "time_frames" in sig:
        kw["time_frames"] = time_frames

    stage1 = base_model_cls(num_classes=3, **kw)
    stage2 = base_model_cls(num_classes=10, **kw)
    return HierarchicalClassifier(stage1, stage2)
