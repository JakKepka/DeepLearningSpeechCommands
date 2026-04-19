"""CNN baseline for log-Mel spectrogram input."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.constants import CNN_CHANNELS, CNN_DROPOUT, N_MELS


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """Lightweight CNN for keyword spotting from log-Mel spectrograms.

    Input:  (B, 1, n_mels, time_frames)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        num_classes: int = 12,
        channels: list[int] | None = None,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()
        ch = channels or CNN_CHANNELS

        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in ch:
            blocks.append(ConvBlock(in_ch, out_ch))
            blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(ch[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def cnn_baseline(num_classes: int = 12, **kwargs) -> CNNBaseline:
    return CNNBaseline(num_classes=num_classes, **kwargs)
