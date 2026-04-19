"""Audio loading helpers resilient to torchaudio backend issues on macOS."""
from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import torch

# Set to False to force the stdlib wave fallback (useful for debugging or
# when torchaudio backends are unavailable).
USE_TORCHAUDIO: bool = True


def load_audio(path: str | Path, *, use_torchaudio: bool | None = None) -> tuple[torch.Tensor, int]:
    """Load WAV as float32 tensor shaped (channels, samples).

    Args:
        path: Path to the audio file.
        use_torchaudio: Override the global ``USE_TORCHAUDIO`` flag for this
            call.  ``True`` uses ``torchaudio.load`` (with stdlib fallback on
            error), ``False`` forces the stdlib ``wave`` reader directly.
            Defaults to the module-level ``USE_TORCHAUDIO`` flag (``True``).
    """
    path = Path(path)
    backend = USE_TORCHAUDIO if use_torchaudio is None else use_torchaudio
    if backend:
        try:
            import torchaudio
            return torchaudio.load(str(path))
        except Exception:
            return _load_wav_via_wave(path)
    return _load_wav_via_wave(path)


def _load_wav_via_wave(path: Path) -> tuple[torch.Tensor, int]:
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise RuntimeError(f"Failed to read WAV file {path}: {exc}") from exc

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(
            f"Unsupported WAV sample width ({sampwidth} bytes) for file {path}."
        )

    if data.size == 0:
        return torch.zeros((1, 0), dtype=torch.float32), sr

    data = data.reshape(-1, n_channels).T  # (channels, samples)
    tensor = torch.from_numpy(data).to(dtype=torch.float32)
    return tensor, sr
