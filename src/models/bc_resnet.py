"""BC-ResNet for keyword spotting on log-Mel spectrograms.

Implementation follows the core BC-ResNet idea:
frequency depthwise conv + SubSpectralNorm, temporal depthwise path,
pointwise mixing, dropout, and residual-style fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    """SubSpectral Normalization for tensors shaped (B, C, F, T)."""

    def __init__(self, channels: int, sub_bands: int = 5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        if f % self.sub_bands != 0:
            raise ValueError(
                f"Frequency dim {f} must be divisible by sub_bands={self.sub_bands}"
            )
        x = x.view(b, c * self.sub_bands, f // self.sub_bands, t)
        x = self.bn(x)
        x = x.view(b, c, self.sub_bands, f // self.sub_bands, t)
        return x.reshape(b, c, f, t)


class BCResNormalBlock(nn.Module):
    """Normal BC-Res block with identity shortcut."""

    def __init__(
        self,
        channels: int,
        dilation_t: int,
        dropout: float = 0.1,
        sub_bands: int = 5,
    ):
        super().__init__()
        self.freq_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=channels,
            bias=False,
        )
        self.ssn = SubSpectralNorm(channels, sub_bands=sub_bands)

        self.temp_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 3),
            padding=(0, dilation_t),
            dilation=(1, dilation_t),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        residual_2d = self.freq_dw(x)
        residual_2d = self.ssn(residual_2d)

        y = residual_2d.mean(dim=2, keepdim=True)
        y = self.temp_dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw(y)
        y = self.drop(y)

        out = y + identity + residual_2d
        return F.relu(out)


class BCResTransitionBlock(nn.Module):
    """Transition block with channel change and optional frequency downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int,
        dilation_t: int,
        dropout: float = 0.1,
        sub_bands: int = 5,
    ):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride_f, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.freq_dw = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=out_channels,
            bias=False,
        )
        self.ssn = SubSpectralNorm(out_channels, sub_bands=sub_bands)

        self.temp_dw = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, dilation_t),
            dilation=(1, dilation_t),
            groups=out_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        residual_2d = self.freq_dw(x)
        residual_2d = self.ssn(residual_2d)

        y = residual_2d.mean(dim=2, keepdim=True)
        y = self.temp_dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw(y)
        y = self.drop(y)

        out = y + residual_2d
        return F.relu(out)


class BCResNet(nn.Module):
    """BC-ResNet for KWS.

    Input:  (B, 1, n_mels, time_frames)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        n_mels: int = 40,
        num_classes: int = 12,
        width_mult: float = 1.5,
        first_filters: int = 24,
        stage_filters: list[int] | tuple[int, ...] = (12, 18, 24, 30),
        stage_blocks: list[int] | tuple[int, ...] = (2, 2, 4, 4),
        stage_strides_f: list[int] | tuple[int, ...] = (1, 2, 2, 1),
        stage_dilations_t: list[int] | tuple[int, ...] = (1, 2, 3, 3),
        last_filters: int = 48,
        sub_bands: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.width_mult = width_mult

        self.stem = nn.Conv2d(
            1,
            first_filters,
            kernel_size=5,
            stride=(2, 1),
            padding=2,
            bias=False,
        )

        layers: list[nn.Module] = []
        in_ch = first_filters

        for out_ch, n_blocks, stride_f, dilation_t in zip(
            stage_filters, stage_blocks, stage_strides_f, stage_dilations_t
        ):
            layers.append(
                BCResTransitionBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride_f=stride_f,
                    dilation_t=dilation_t,
                    dropout=dropout,
                    sub_bands=sub_bands,
                )
            )
            for _ in range(n_blocks):
                layers.append(
                    BCResNormalBlock(
                        channels=out_ch,
                        dilation_t=dilation_t,
                        dropout=dropout,
                        sub_bands=sub_bands,
                    )
                )
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        self.dw_head = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=5,
            padding=2,
            groups=in_ch,
            bias=False,
        )
        self.head_pw = nn.Conv2d(in_ch, last_filters, kernel_size=1, bias=False)
        self.classifier = nn.Conv2d(last_filters, num_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]}")
        if x.shape[2] != self.n_mels:
            raise ValueError(f"Expected n_mels={self.n_mels}, got {x.shape[2]}")

        x = self.stem(x)
        x = self.backbone(x)

        x = self.dw_head(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.head_pw(x)
        x = x.mean(dim=3, keepdim=True)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)


def bc_resnet_1_5(num_classes: int = 12, **kwargs) -> BCResNet:
    cfg = {
        "num_classes": num_classes,
        "width_mult": 1.5,
        "first_filters": 24,
        "stage_filters": (12, 18, 24, 30),
        "stage_blocks": (2, 2, 4, 4),
        "stage_strides_f": (1, 2, 2, 1),
        "stage_dilations_t": (1, 2, 3, 3),
        "last_filters": 48,
        "sub_bands": 5,
        "dropout": 0.1,
    }
    cfg.update(kwargs)
    return BCResNet(**cfg)
