"""Audio adapters for the desktop app; DSP modules only receive NumPy arrays."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from wav_io import read_mono_wav, write_mono_wav


def _to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return samples.mean(axis=1)
    raise ValueError("audio data must have one or two dimensions")


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode audio to a mono floating-point NumPy array.

    With optional `soundfile`, this supports WAV, FLAC, OGG, and AIFF (and MP3
    when the installed libsndfile supports it).  Without it, 16-bit mono WAV
    remains available through the project's standard-library WAV adapter.
    """
    path = Path(path)
    try:
        import soundfile as sf  # type: ignore[import-not-found]
        samples, sample_rate = sf.read(path, dtype="float64", always_2d=False)
        return _to_mono(samples), int(sample_rate)
    except ImportError:
        if path.suffix.lower() == ".wav":
            return read_mono_wav(path)
        raise ValueError("Install soundfile to open FLAC, OGG, AIFF, or MP3 audio.")
    except RuntimeError as soundfile_error:
        # pydub is a useful fallback for formats that libsndfile cannot decode,
        # notably some MP3/AAC builds. It requires an ffmpeg installation.
        try:
            from pydub import AudioSegment  # type: ignore[import-not-found]
            segment = AudioSegment.from_file(path)
        except ImportError:
            raise ValueError(f"Could not decode {path.suffix}. Install pydub and ffmpeg for this format.") from soundfile_error
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float64)
        samples = samples.reshape((-1, segment.channels))
        scale = float(1 << (8 * segment.sample_width - 1))
        return _to_mono(samples / scale), int(segment.frame_rate)


def record_mono(seconds: float, sample_rate: int = 44_100) -> tuple[np.ndarray, int]:
    """Record from the default microphone directly into a NumPy array."""
    if seconds <= 0:
        raise ValueError("recording duration must be positive")
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Recording needs sounddevice. Run: py -3 -m pip install sounddevice") from error
    frames = round(seconds * sample_rate)
    recorded = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float64")
    sd.wait()
    return recorded[:, 0].copy(), sample_rate


def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    """Play a NumPy signal through the default output device, blocking until done."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("In-app playback needs sounddevice. Run: py -3 -m pip install sounddevice") from error
    sd.play(np.asarray(samples, dtype=np.float32), sample_rate)
    sd.wait()


class Recorder:
    """Non-blocking microphone recorder with pause/resume support."""

    def __init__(self, sample_rate: int = 44_100) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._accepting = False
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError("Recording needs sounddevice. Run: py -3 -m pip install sounddevice") from error

        def callback(indata, frames, time, status) -> None:
            if status:
                return
            with self._lock:
                if self._accepting:
                    self._chunks.append(indata[:, 0].copy())

        self._accepting = True
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float64", callback=callback)
        self._stream.start()

    def pause(self) -> None:
        self._accepting = False

    def resume(self) -> None:
        self._accepting = True

    def stop(self) -> tuple[np.ndarray, int]:
        self._accepting = False
        if self._stream is None:
            raise ValueError("No recording is active")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            result = np.concatenate(self._chunks) if self._chunks else np.empty(0, dtype=np.float64)
        if not len(result):
            raise ValueError("The recording is empty")
        return result, self.sample_rate


class AudioPlayer:
    """Non-blocking NumPy-array playback with pause/resume support."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)
        self.sample_rate = sample_rate
        self.position = 0
        self.paused = False
        self.volume = 1.0
        self._stream = None
        self._lock = threading.Lock()

    def play(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError("Playback needs sounddevice. Run: py -3 -m pip install sounddevice") from error

        def callback(outdata, frames, time, status) -> None:
            with self._lock:
                if self.paused:
                    outdata.fill(0)
                    return
                remaining = len(self.samples) - self.position
                count = min(frames, max(remaining, 0))
                outdata.fill(0)
                if count:
                    outdata[:count, 0] = self.samples[self.position : self.position + count] * self.volume
                    self.position += count
                if count < frames:
                    raise sd.CallbackStop()

        self.stop()
        self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback)
        self._stream.start()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self.volume = max(0.0, min(float(volume), 2.0))

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None


def write_output_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> int:
    """Write transformed arrays as portable 16-bit PCM WAV files."""
    return write_mono_wav(path, samples, sample_rate)
