"""Optional neural voice-conversion backend, as an alternative to the LPC matcher.

`match_voice_character` in `phase_vocoder.py` is DSP-only: it matches a
reference's pitch and averaged LPC spectral-envelope shape, which is
explainable and always available but has a real ceiling — it can nudge a
voice's broad register and resonance, not reconstruct someone's actual
timbre. This module wraps kNN-VC (Baas, van Niekerk & Kamper — a published,
pretrained, any-to-any voice-conversion pipeline: https://github.com/bshall/knn-vc),
which featurizes speech with a self-supervised model (WavLM), converts by
k-nearest-neighbor matching each source frame's features against a
reference's, and resynthesizes with a pretrained neural vocoder (HiFi-GAN).
It sounds considerably closer to the reference than the LPC path, at the
cost of needing PyTorch, a one-time few-hundred-MB model download over the
internet, and real compute per conversion.

Everything here is optional and lazily imported: importing this module
never requires torch to be installed, and every failure path raises a
plain `ValueError` with a human-readable message, matching the pattern
`audio_io.py` and `secure_audio.py` already use for their optional
dependencies (sounddevice, cryptography) — so a missing/failed ML backend
never take down anything else in the app; the always-available LPC path
is the fallback.

Honesty note for whoever maintains this: the capability-detection and
runtime-estimate logic below is exercised and tested without needing
PyTorch at all. The actual `MLVoiceConverter.convert()` call against
kNN-VC's published `torch.hub` interface has *not* been run end-to-end in
the environment this was written in (no GPU, and the first call downloads
several hundred MB of weights) — it's written to that project's documented
API, wrapped in defensive error handling, but worth a manual test run
before relying on it for a live demo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

# kNN-VC's published models run on 16 kHz mono audio.
_MODEL_SAMPLE_RATE = 16_000
_HUB_REPO = "bshall/knn-vc"


@dataclass
class CapabilityReport:
    torch_available: bool
    gpu_available: bool
    cpu_count: int
    ram_gb: float | None
    estimated_seconds: float
    recommendation: str  # "fast" | "workable" | "slow" | "unavailable"
    reason: str


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    try:
        # Linux-only fallback; harmless if it fails (e.g. on Windows without psutil).
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def capability_report(duration_seconds: float) -> CapabilityReport:
    """Estimate whether this machine can reasonably run the ML backend.

    The estimate is a rough heuristic, not a benchmark of this specific
    machine — the honest framing surfaced in the UI is "roughly", not an
    exact number. It exists so someone doesn't press "Use ML matching",
    wait, and wonder if the app hung: a lower-end laptop is told upfront to
    expect a couple of minutes for a 10-second clip, not 10 seconds.
    """
    try:
        import torch  # type: ignore[import-not-found]
        torch_available = True
        gpu_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_available = False
        gpu_available = False

    cpu_count = os.cpu_count() or 1
    ram = _ram_gb()

    if not torch_available:
        return CapabilityReport(
            torch_available=False, gpu_available=False, cpu_count=cpu_count, ram_gb=ram,
            estimated_seconds=0.0, recommendation="unavailable",
            reason="PyTorch isn't installed. Run: pip install torch torchaudio",
        )

    model_load_overhead = 6.0  # first call per session; cached after that
    if gpu_available:
        factor = 0.6
        recommendation = "fast"
        reason = "A CUDA GPU was detected — conversions should run faster than real time."
    elif cpu_count >= 8 and (ram is None or ram >= 8):
        factor = 6.0
        recommendation = "workable"
        reason = f"No GPU, but {cpu_count} CPU cores is enough to run this — just noticeably slower than real time."
    elif cpu_count >= 4:
        factor = 12.0
        recommendation = "slow"
        reason = f"Only {cpu_count} CPU cores and no GPU — expect roughly a minute per 5 seconds of audio."
    else:
        factor = 20.0
        recommendation = "slow"
        reason = f"Just {cpu_count} CPU core(s) and no GPU — this will likely take a long time; the LPC matcher is the practical choice here."

    estimated_seconds = model_load_overhead + duration_seconds * factor
    return CapabilityReport(
        torch_available=True, gpu_available=gpu_available, cpu_count=cpu_count, ram_gb=ram,
        estimated_seconds=estimated_seconds, recommendation=recommendation, reason=reason,
    )


class MLVoiceConverter:
    """Thin wrapper around kNN-VC, loaded once and reused for later conversions."""

    _cached_pipeline = None  # class-level: one load serves the whole app session

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _pipeline(self):
        if MLVoiceConverter._cached_pipeline is not None:
            return MLVoiceConverter._cached_pipeline
        try:
            import torch
        except ImportError as error:
            raise ValueError(
                "ML voice matching needs PyTorch. Run: pip install torch torchaudio"
            ) from error
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = torch.hub.load(
                _HUB_REPO, "knn_vc", prematched=True, trust_repo=True,
                pretrained=True, device=device,
            )
        except Exception as error:  # noqa: BLE001 — surfacing whatever torch.hub raised, as a plain message
            raise ValueError(
                "Couldn't load the ML voice-conversion model (kNN-VC). The first run needs an internet "
                f"connection to fetch its pretrained weights (a few hundred MB). Underlying error: {error}"
            ) from error
        MLVoiceConverter._cached_pipeline = pipeline
        return pipeline

    def convert(
        self,
        source: np.ndarray,
        source_rate: int,
        reference: np.ndarray,
        reference_rate: int,
    ) -> tuple[np.ndarray, int]:
        """Convert `source` toward `reference`'s voice. Returns (samples, sample_rate)."""
        try:
            import torch
            import torchaudio
        except ImportError as error:
            raise ValueError(
                "ML voice matching needs PyTorch and torchaudio. Run: pip install torch torchaudio"
            ) from error

        pipeline = self._pipeline()

        def _prepare(signal: np.ndarray, rate: int):
            tensor = torch.as_tensor(np.asarray(signal, dtype=np.float32)).unsqueeze(0)
            if rate != _MODEL_SAMPLE_RATE:
                tensor = torchaudio.functional.resample(tensor, rate, _MODEL_SAMPLE_RATE)
            return tensor

        query_wav = _prepare(source, source_rate)
        matching_wav = _prepare(reference, reference_rate)

        with torch.inference_mode():
            query_features = pipeline.get_features(query_wav)
            matching_features = pipeline.get_features(matching_wav)
            matched_features = pipeline.match(query_features, matching_features, topk=4)
            converted = pipeline.vocode(matched_features[None]).squeeze(0)

        return converted.detach().cpu().numpy().astype(np.float64), _MODEL_SAMPLE_RATE


def timed_convert(
    source: np.ndarray, source_rate: int, reference: np.ndarray, reference_rate: int,
) -> tuple[np.ndarray, int, float]:
    """Run `MLVoiceConverter.convert` and report how long it actually took.

    Kept separate from the class so the Match page can log/display real
    elapsed time next to the pre-run estimate from `capability_report`.
    """
    converter = MLVoiceConverter()
    started = time.monotonic()
    samples, sample_rate = converter.convert(source, source_rate, reference, reference_rate)
    return samples, sample_rate, time.monotonic() - started
