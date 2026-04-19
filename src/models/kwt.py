"""Keyword Transformer (KWT-style) for speech command classification.

Each column of the log-Mel spectrogram (one time step) is treated as a token.
Architecture follows the ViT / KWT design: CLS token + positional embedding +
Transformer encoder + MLP classification head.

Reference: "Keyword Transformer: A Self-Attention Model for Keyword Spotting"
           Berg et al., Interspeech 2021.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.constants import (
    KWT_SMALL_CONFIG,
    KWT_MEDIUM_CONFIG,
    N_MELS,
    TRUNC_NORM_STD,
)


class KWT(nn.Module):
    """Keyword Transformer.

    Input:  (B, 1, n_mels, time_frames)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        time_frames: int = 101,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 12,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames

        # Project each time-frame vector (length n_mels) to d_model
        self.patch_embed = nn.Linear(n_mels, d_model)

        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, time_frames + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=TRUNC_NORM_STD)
        nn.init.trunc_normal_(self.cls_token, std=TRUNC_NORM_STD)
        nn.init.trunc_normal_(self.head.weight, std=TRUNC_NORM_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        B, _C, _F, T = x.shape
        # Treat each time step as a token: (B, T, n_mels)
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.patch_embed(x)  # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Interpolate positional embedding if T differs from expected
        if T + 1 != self.pos_embed.shape[1]:
            pos = self._interpolate_pos_embed(T)
        else:
            pos = self.pos_embed

        x = self.pos_drop(x + pos)
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)

    def _interpolate_pos_embed(self, T: int) -> torch.Tensor:
        cls_pos = self.pos_embed[:, :1, :]
        seq_pos = self.pos_embed[:, 1:, :]  # (1, expected_T, d)
        # Interpolate seq_pos to length T
        seq_pos = seq_pos.permute(0, 2, 1)  # (1, d, expected_T)
        seq_pos = torch.nn.functional.interpolate(seq_pos, size=T, mode="linear", align_corners=False)
        seq_pos = seq_pos.permute(0, 2, 1)  # (1, T, d)
        return torch.cat([cls_pos, seq_pos], dim=1)


# ------------------------------------------------------------------
# Named configurations
# ------------------------------------------------------------------

KWT_CONFIGS: dict[str, dict] = {
    "small": KWT_SMALL_CONFIG,
    "medium": KWT_MEDIUM_CONFIG,
}


def kwt_small(num_classes: int = 12, **kwargs) -> KWT:
    return KWT(**{**KWT_CONFIGS["small"], "num_classes": num_classes, **kwargs})


def kwt_medium(num_classes: int = 12, **kwargs) -> KWT:
    return KWT(**{**KWT_CONFIGS["medium"], "num_classes": num_classes, **kwargs})
