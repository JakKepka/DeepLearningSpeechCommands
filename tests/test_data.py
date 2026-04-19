"""Unit tests for data pipeline."""
from __future__ import annotations

import torch
import pytest

from src.data.labels import (
    ALL_CLASSES,
    NUM_CLASSES,
    SILENCE_IDX,
    UNKNOWN_IDX,
    TARGET_COMMAND_INDICES,
    flat_label,
    STAGE1_LABEL_MAP,
    STAGE1_SILENCE,
    STAGE1_TARGET,
    STAGE1_UNKNOWN,
    TARGET_COMMANDS,
)
from src.data.transforms import AudioPipeline, LogMelSpectrogram


# ------------------------------------------------------------------ #
# Label mapping                                                        #
# ------------------------------------------------------------------ #

class TestLabelMapping:
    def test_num_classes(self):
        assert NUM_CLASSES == 12

    def test_silence_index(self):
        assert SILENCE_IDX == 0

    def test_unknown_index(self):
        assert UNKNOWN_IDX == 1

    def test_target_commands_count(self):
        assert len(TARGET_COMMAND_INDICES) == 10

    def test_flat_label_target(self):
        assert flat_label("yes") == ALL_CLASSES.index("yes")

    def test_flat_label_silence(self):
        assert flat_label("silence") == SILENCE_IDX
        assert flat_label("_silence_") == SILENCE_IDX

    def test_flat_label_unknown(self):
        assert flat_label("tree") == UNKNOWN_IDX
        assert flat_label("bed") == UNKNOWN_IDX

    def test_stage1_map_silence(self):
        assert STAGE1_LABEL_MAP["silence"] == STAGE1_SILENCE

    def test_stage1_map_target(self):
        for cmd in TARGET_COMMANDS:
            assert STAGE1_LABEL_MAP[cmd] == STAGE1_TARGET

    def test_stage1_map_unknown(self):
        assert STAGE1_LABEL_MAP["unknown"] == STAGE1_UNKNOWN


# ------------------------------------------------------------------ #
# Feature extraction                                                   #
# ------------------------------------------------------------------ #

class TestLogMelSpectrogram:
    def test_output_shape(self):
        mel = LogMelSpectrogram(n_mels=64, n_fft=512, hop_length=160)
        wav = torch.zeros(1, 16000)
        spec = mel(wav)
        assert spec.shape[0] == 1          # channels
        assert spec.shape[1] == 64         # n_mels
        assert spec.shape[2] > 0           # time frames

    def test_output_dtype(self):
        mel = LogMelSpectrogram()
        spec = mel(torch.randn(1, 16000))
        assert spec.dtype == torch.float32


class TestAudioPipeline:
    def test_no_augment_shape(self):
        pipeline = AudioPipeline(n_mels=64, augment=False)
        wav = torch.randn(1, 16000)
        spec = pipeline(wav)
        assert spec.ndim == 3
        assert spec.shape[1] == 64

    def test_augment_shape_consistent(self):
        pipeline = AudioPipeline(n_mels=64, augment=True)
        wav = torch.randn(1, 16000)
        spec = pipeline(wav)
        assert spec.ndim == 3
        assert spec.shape[1] == 64

    def test_with_spec_augment(self):
        pipeline = AudioPipeline(
            n_mels=64, augment=True,
            spec_aug_time_mask=10, spec_aug_freq_mask=8,
            n_time_masks=1, n_freq_masks=1,
        )
        wav = torch.randn(1, 16000)
        spec = pipeline(wav)
        assert spec.shape[1] == 64
