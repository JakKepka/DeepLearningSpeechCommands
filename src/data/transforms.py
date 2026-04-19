"""Audio feature extraction and data augmentation transforms."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from src.utils.audio import load_audio

from src.utils.constants import (
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    F_MIN,
    F_MAX,
    TIME_SHIFT_SAMPLES,
    NOISE_PROBABILITY,
    NOISE_SNR_RANGE,
    SPEC_AUG_TIME_MASK_PARAM,
    SPEC_AUG_FREQ_MASK_PARAM,
    N_TIME_MASKS,
    N_FREQ_MASKS,
)


class LogMelSpectrogram(nn.Module):
    """Convert raw waveform → log-Mel spectrogram.

    Output shape: (1, n_mels, time_frames)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.to_db(self.mel(waveform))  # (1, n_mels, T)


class TimeShift(nn.Module):
    """Random circular shift along the time axis."""

    def __init__(self, max_shift_samples: int = TIME_SHIFT_SAMPLES):
        super().__init__()
        self.max_shift = max_shift_samples

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return wav
        return torch.roll(wav, shift, dims=-1)


class AddBackgroundNoise(nn.Module):
    """Mix signal with random noise at a random SNR."""

    def __init__(
        self,
        noise_dir: str | Path,
        snr_range: tuple[float, float] = NOISE_SNR_RANGE,
        p: float = NOISE_PROBABILITY,
    ):
        super().__init__()
        self.p = p
        self.snr_range = snr_range
        self._pool = self._load_noise(Path(noise_dir))

    @staticmethod
    def _load_noise(noise_dir: Path) -> Optional[torch.Tensor]:
        files = list(noise_dir.glob("*.wav"))
        if not files:
            return None
        chunks: list[torch.Tensor] = []
        for f in files:
            wav, sr = load_audio(f)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            chunks.append(wav)
        return torch.cat(chunks, dim=-1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self._pool is None or random.random() >= self.p:
            return wav
        n = wav.shape[-1]
        total = self._pool.shape[-1]
        if total >= n:
            start = random.randint(0, total - n)
            noise = self._pool[..., start : start + n]
        else:
            noise = self._pool.repeat(1, n // total + 1)[..., :n]

        snr = 10 ** (random.uniform(*self.snr_range) / 20.0)
        sig_rms = wav.norm(p=2).clamp(min=1e-9)
        noise_rms = noise.norm(p=2).clamp(min=1e-9)
        scale = sig_rms / (noise_rms * snr)
        return wav + scale * noise


class SpecAugment(nn.Module):
    """SpecAugment: time & frequency masking on log-Mel spectrograms."""

    def __init__(
        self,
        time_mask_param: int = SPEC_AUG_TIME_MASK_PARAM,
        freq_mask_param: int = SPEC_AUG_FREQ_MASK_PARAM,
        n_time_masks: int = N_TIME_MASKS,
        n_freq_masks: int = N_FREQ_MASKS,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        for m in self.freq_masks:
            spec = m(spec)
        for m in self.time_masks:
            spec = m(spec)
        return spec


class AudioPipeline(nn.Module):
    """Waveform → log-Mel spectrogram with optional augmentation.

    When ``augment=True``, applies:
        1. TimeShift
        2. AddBackgroundNoise (if *noise_dir* is provided)
        3. LogMelSpectrogram
        4. SpecAugment
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        augment: bool = False,
        time_shift_samples: int = TIME_SHIFT_SAMPLES,
        noise_dir: Optional[str | Path] = None,
        noise_prob: float = NOISE_PROBABILITY,
        noise_snr_range: tuple[float, float] = NOISE_SNR_RANGE,
        spec_aug_time_mask: int = SPEC_AUG_TIME_MASK_PARAM,
        spec_aug_freq_mask: int = SPEC_AUG_FREQ_MASK_PARAM,
        n_time_masks: int = N_TIME_MASKS,
        n_freq_masks: int = N_FREQ_MASKS,
    ):
        super().__init__()
        self.augment = augment

        if augment:
            self.time_shift: Optional[nn.Module] = TimeShift(time_shift_samples)
            if noise_dir is not None and Path(noise_dir).exists():
                self.bg_noise: Optional[nn.Module] = AddBackgroundNoise(
                    noise_dir, snr_range=noise_snr_range, p=noise_prob
                )
            else:
                self.bg_noise = None
        else:
            self.time_shift = None
            self.bg_noise = None

        self.log_mel = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        )

        self.spec_aug: Optional[nn.Module] = (
            SpecAugment(
                time_mask_param=spec_aug_time_mask,
                freq_mask_param=spec_aug_freq_mask,
                n_time_masks=n_time_masks,
                n_freq_masks=n_freq_masks,
            )
            if augment
            else None
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.augment:
            if self.time_shift is not None:
                wav = self.time_shift(wav)
            if self.bg_noise is not None:
                wav = self.bg_noise(wav)
        spec = self.log_mel(wav)
        if self.spec_aug is not None:
            spec = self.spec_aug(spec)
        return spec
