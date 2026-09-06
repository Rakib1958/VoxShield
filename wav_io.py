"""Small, dependency-free WAV reader/writer for the project demos."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def read_mono_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM mono WAV as float samples in approximately [-1, 1]."""
    path = Path(path)
    with wave.open(str(path), "rb") as file:
        if file.getcomptype() != "NONE":
            raise ValueError("compressed WAV is not supported; export 16-bit PCM WAV")
        if file.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        if file.getnchannels() != 1:
            raise ValueError("expected mono WAV; convert to mono first")
        sample_rate = file.getframerate()
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    return samples.astype(np.float64) / 32768.0, sample_rate


def write_mono_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> int:
    """Write a 16-bit PCM mono WAV.  Returns the number of clipped samples."""
    path = Path(path)
    signal = np.asarray(signal, dtype=np.float64)
    clipped = int(np.count_nonzero(np.abs(signal) > 1.0))
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())
    return clipped
