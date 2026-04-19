"""Single-file and batch audio inference."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.data.dataset import CLIP_SAMPLES, SAMPLE_RATE, _pad_or_trim
from src.data.labels import IDX_TO_LABEL
from src.data.transforms import AudioPipeline
from src.utils.audio import load_audio


class Predictor:
    """Load a trained model and run inference on audio files.

    Args:
        model:       Trained nn.Module (in eval mode).
        pipeline:    AudioPipeline for feature extraction (no augmentation).
        device:      Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        pipeline: Optional[AudioPipeline] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = model.eval().to(device)

        if pipeline is None:
            pipeline = AudioPipeline(augment=False)
        self.pipeline = pipeline.to(device)

    def predict_file(self, path: str | Path) -> dict[str, object]:
        """Predict the class label for a single WAV file.

        Returns:
            dict with keys: ``label`` (str), ``class_idx`` (int),
            ``probabilities`` (list[float]).
        """
        import torchaudio

        wav, sr = load_audio(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = _pad_or_trim(wav, CLIP_SAMPLES)

        wav = wav.to(self.device)
        spec = self.pipeline(wav)          # (1, n_mels, T)
        spec = spec.unsqueeze(0)           # (1, 1, n_mels, T)

        with torch.inference_mode():
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            idx = int(probs.argmax().item())

        return {
            "label": IDX_TO_LABEL[idx],
            "class_idx": idx,
            "probabilities": probs.cpu().tolist(),
        }

    def predict_batch(self, paths: list[str | Path]) -> list[dict[str, object]]:
        return [self.predict_file(p) for p in paths]
