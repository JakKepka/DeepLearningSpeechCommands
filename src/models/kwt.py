"""Keyword Transformer (KWT-style) for speech command classification.

Each column of the log-Mel spectrogram (one time step) is treated as a token.
Architecture follows the ViT / KWT design: CLS token + positional embedding +
Transformer encoder + MLP classification head.

Reference: "Keyword Transformer: A Self-Attention Model for Keyword Spotting"
           Berg et al., Interspeech 2021.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import (
    KWT_2_CONFIG,
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
        time_frames: int = 98,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 12,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 12,
        trunc_norm_std: float = TRUNC_NORM_STD,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.d_model = d_model

        # Each time step (vector of length n_mels) becomes one token.
        self.patch_embed = nn.Linear(n_mels, d_model)

        # CLS token + learned positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, time_frames + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        # PostNorm encoder block.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights(trunc_norm_std)

    def _init_weights(self, trunc_norm_std: float) -> None:
        nn.init.trunc_normal_(self.cls_token, std=trunc_norm_std)
        nn.init.trunc_normal_(self.pos_embed, std=trunc_norm_std)
        nn.init.trunc_normal_(self.head.weight, std=trunc_norm_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        b, c, f, t = x.shape
        if c != 1:
            raise ValueError(f"Expected 1 input channel, got {c}")
        if f != self.n_mels:
            raise ValueError(f"Expected n_mels={self.n_mels}, got {f}")

        # (B, 1, F, T) -> (B, T, F).
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Interpolate positional embedding if time length differs.
        if x.size(1) != self.pos_embed.size(1):
            pos = self._interpolate_pos_embed(x.size(1) - 1)
        else:
            pos = self.pos_embed

        x = self.pos_drop(x + pos)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)

    def _interpolate_pos_embed(self, t: int) -> torch.Tensor:
        """Resize positional embeddings when input time length differs."""
        cls_pos = self.pos_embed[:, :1, :]
        seq_pos = self.pos_embed[:, 1:, :]

        seq_pos = seq_pos.permute(0, 2, 1)
        seq_pos = F.interpolate(
            seq_pos,
            size=t,
            mode="linear",
            align_corners=False,
        )
        seq_pos = seq_pos.permute(0, 2, 1)
        return torch.cat([cls_pos, seq_pos], dim=1)


# ------------------------------------------------------------------
# Named configurations
# ------------------------------------------------------------------

KWT_CONFIGS: dict[str, dict] = {
    "kwt_2": KWT_2_CONFIG,
    "small": KWT_SMALL_CONFIG,
    "medium": KWT_MEDIUM_CONFIG,
}


def kwt_small(num_classes: int = 12, **kwargs) -> KWT:
    return KWT(**{**KWT_CONFIGS["small"], "num_classes": num_classes, **kwargs})


def kwt_2(num_classes: int = 12, **kwargs) -> KWT:
    return KWT(**{**KWT_CONFIGS["kwt_2"], "num_classes": num_classes, **kwargs})


def kwt_medium(num_classes: int = 12, **kwargs) -> KWT:
    return KWT(**{**KWT_CONFIGS["medium"], "num_classes": num_classes, **kwargs})
