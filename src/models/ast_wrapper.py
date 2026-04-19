"""Wrapper for Audio Spectrogram Transformer (AST) fine-tuning.

Uses the pretrained AST from HuggingFace Transformers.
Model ID defined in constants.AST_PRETRAINED_MODEL.

The wrapper adapts our (1, n_mels, T) log-Mel spectrograms to the
(B, T, n_mels) format expected by the HuggingFace ASTModel.
Positional embeddings are automatically interpolated for different sizes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.constants import AST_PRETRAINED_MODEL, AST_FREEZE_N_LAYERS, TRUNC_NORM_STD


class ASTWrapper(nn.Module):
    """Fine-tune Audio Spectrogram Transformer for 12-class speech commands.

    Input:  (B, 1, n_mels, time_frames) — log-Mel spectrogram
    Output: (B, num_classes) — logits
    """

    def __init__(
        self,
        num_classes: int = 12,
        pretrained_model: str = AST_PRETRAINED_MODEL,
        freeze_n_layers: int = AST_FREEZE_N_LAYERS,
    ):
        super().__init__()
        try:
            from transformers import ASTModel, ASTConfig
        except ImportError as exc:
            raise ImportError(
                "Install 'transformers' to use ASTWrapper: pip install transformers"
            ) from exc

        self.ast = ASTModel.from_pretrained(pretrained_model, ignore_mismatched_sizes=True)
        hidden_size: int = self.ast.config.hidden_size

        if freeze_n_layers > 0:
            self._freeze_layers(freeze_n_layers)

        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=TRUNC_NORM_STD)

    def _freeze_layers(self, n: int) -> None:
        # Freeze patch embeddings + first n encoder layers
        for param in self.ast.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.ast.encoder.layer):
            if i >= n:
                break
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        B, _C, _F, T = x.shape
        # ASTModel expects (B, T, n_mels)
        x = x.squeeze(1).permute(0, 2, 1).contiguous()
        outputs = self.ast(input_values=x)
        pooled = outputs.pooler_output  # (B, hidden_size)
        return self.head(pooled)


def ast_finetuned(num_classes: int = 12, freeze_n_layers: int = 6, **kwargs) -> ASTWrapper:
    return ASTWrapper(num_classes=num_classes, freeze_n_layers=freeze_n_layers, **kwargs)
