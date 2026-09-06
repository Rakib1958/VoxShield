"""First milestone for a phase-vocoder voice changer.

This module deliberately stops before changing time or pitch.  It proves that
our short-time Fourier transform (STFT) analysis and overlap-add synthesis are
correct: `round_trip()` should reproduce an input signal (apart from tiny
floating-point error).  A phase vocoder is built by editing the complex STFT
frames between those two stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int = 44_100
    frame_size: int = 2_048
    hop_size: int = 512

    def __post_init__(self) -> None:
        if self.frame_size <= 0 or self.hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive")
        if self.hop_size > self.frame_size:
            raise ValueError("hop_size must not exceed frame_size")


def sqrt_hann(size: int) -> np.ndarray:
    """Return a periodic square-root Hann window.

    Using this window at both analysis and synthesis means their product is a
    Hann window.  With a 75%-overlap hop (e.g. 512 / 2048), the overlap weights
    can be normalized sample by sample during synthesis.
    """
    return np.sqrt(np.hanning(size + 1)[:-1])


def frame_count(signal_length: int, config: StftConfig) -> int:
    """How many frames are required after zero padding."""
    if signal_length <= config.frame_size:
        return 1
    return int(np.ceil((signal_length - config.frame_size) / config.hop_size)) + 1


def stft(signal: np.ndarray, config: StftConfig) -> tuple[np.ndarray, int]:
    """Analyze a mono signal into one real-FFT spectrum per frame.

    The spectrum is complex. `abs(spectrum)` is the magnitude; `angle(spectrum)`
    is its phase, which carries timing information and must be preserved.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("stft expects a one-dimensional (mono) signal")

    # The Hann window is zero at its ends.  Padding both sides ensures every
    # real input sample, including the first and last, is covered by interior
    # (non-zero) portions of one or more windows.
    edge_padding = config.frame_size
    analysis_signal = np.pad(signal, (edge_padding, edge_padding))
    frames = frame_count(len(analysis_signal), config)
    padded_length = (frames - 1) * config.hop_size + config.frame_size
    padded = np.pad(analysis_signal, (0, padded_length - len(analysis_signal)))
    window = sqrt_hann(config.frame_size)
    spectra = np.empty((frames, config.frame_size // 2 + 1), dtype=np.complex128)

    for index in range(frames):
        start = index * config.hop_size
        # Replace np.fft.rfft here with your own real-FFT adapter if desired.
        spectra[index] = np.fft.rfft(padded[start : start + config.frame_size] * window)
    return spectra, len(signal)


def istft(
    spectra: np.ndarray,
    original_length: int,
    config: StftConfig,
    *,
    hop_size: int | None = None,
    trim_start: int | None = None,
) -> np.ndarray:
    """Reconstruct a mono signal and remove analysis padding."""
    spectra = np.asarray(spectra)
    expected_bins = config.frame_size // 2 + 1
    if spectra.ndim != 2 or spectra.shape[1] != expected_bins:
        raise ValueError(f"expected spectra with shape (frames, {expected_bins})")

    hop_size = config.hop_size if hop_size is None else hop_size
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    output_length = (len(spectra) - 1) * hop_size + config.frame_size
    output = np.zeros(output_length)
    weight = np.zeros(output_length)
    window = sqrt_hann(config.frame_size)

    for index, spectrum in enumerate(spectra):
        start = index * hop_size
        output[start : start + config.frame_size] += np.fft.irfft(spectrum, n=config.frame_size) * window
        weight[start : start + config.frame_size] += window**2

    # Avoid dividing unvisited endpoint samples; they are padding anyway.
    nonzero = weight > np.finfo(float).eps
    output[nonzero] /= weight[nonzero]
    edge_padding = config.frame_size if trim_start is None else trim_start
    return output[edge_padding : edge_padding + original_length]


def round_trip(signal: np.ndarray, config: StftConfig = StftConfig()) -> np.ndarray:
    """STFT followed by inverse STFT—the project's first correctness check."""
    spectra, original_length = stft(signal, config)
    return istft(spectra, original_length, config)


def principal_argument(phase: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi), removing whole turns of phase."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def spectral_peaks(magnitudes: np.ndarray) -> np.ndarray:
    """Return local-maximum bin indices, including a dominant edge when needed."""
    if len(magnitudes) == 1:
        return np.array([0])
    interior = np.flatnonzero(
        (magnitudes[1:-1] >= magnitudes[:-2]) & (magnitudes[1:-1] > magnitudes[2:])
    ) + 1
    edges: list[int] = []
    if magnitudes[0] > magnitudes[1]:
        edges.append(0)
    if magnitudes[-1] >= magnitudes[-2]:
        edges.append(len(magnitudes) - 1)
    peaks = np.concatenate((np.array(edges, dtype=int), interior))
    # Silence or a perfectly flat spectrum contains no strict local maximum.
    return peaks if len(peaks) else np.array([int(np.argmax(magnitudes))])


def phase_vocoder_spectra(
    spectra: np.ndarray,
    analysis_hop: int,
    synthesis_hop: int,
    *,
    phase_locking: bool = False,
) -> np.ndarray:
    """Retain magnitudes while advancing each bin's phase at a new hop.

    Consecutive FFT phases differ for two reasons: the bin's expected rotation
    and a smaller residual caused by the signal being off the bin center.
    After wrapping away whole rotations, that residual gives the bin's measured
    instantaneous frequency.  We advance that measured frequency for the new
    synthesis hop instead of merely copying the old phase.
    """
    spectra = np.asarray(spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a two-dimensional array")
    if analysis_hop <= 0 or synthesis_hop <= 0:
        raise ValueError("hop sizes must be positive")

    frames, bins = spectra.shape
    if frames == 0:
        return spectra.copy()

    frame_size = 2 * (bins - 1)
    angular_bin_frequency = 2 * np.pi * np.arange(bins) / frame_size
    expected_advance = angular_bin_frequency * analysis_hop
    analysis_phase = np.angle(spectra)
    synthesis_phase = np.empty_like(analysis_phase)
    synthesis_phase[0] = analysis_phase[0]

    for frame in range(1, frames):
        observed_advance = analysis_phase[frame] - analysis_phase[frame - 1]
        residual = principal_argument(observed_advance - expected_advance)
        instantaneous_advance = expected_advance + residual
        synthesis_phase[frame] = synthesis_phase[frame - 1] + instantaneous_advance * synthesis_hop / analysis_hop

        if phase_locking:
            # Identity phase locking: each bin keeps its *analysis-frame*
            # phase offset from the nearest spectral peak, while that peak
            # itself follows the instantaneous-frequency phase estimate above.
            peaks = spectral_peaks(np.abs(spectra[frame]))
            bin_numbers = np.arange(bins)
            nearest_peak = peaks[np.argmin(np.abs(bin_numbers[:, None] - peaks), axis=1)]
            relative_phase = principal_argument(
                analysis_phase[frame] - analysis_phase[frame, nearest_peak]
            )
            synthesis_phase[frame] = synthesis_phase[frame, nearest_peak] + relative_phase

    return np.abs(spectra) * np.exp(1j * synthesis_phase)


def time_stretch(
    signal: np.ndarray,
    stretch: float,
    config: StftConfig = StftConfig(),
    *,
    phase_locking: bool = False,
) -> np.ndarray:
    """Change duration while approximately preserving pitch.

    `stretch=1.5` makes the signal 50% longer; `stretch=0.75` makes it shorter.
    The actual factor is quantized slightly because overlap-add needs an integer
    output hop.  For typical frame sizes this difference is negligible.
    """
    if stretch <= 0:
        raise ValueError("stretch must be positive")

    spectra, original_length = stft(signal, config)
    synthesis_hop = max(1, round(config.hop_size * stretch))
    actual_stretch = synthesis_hop / config.hop_size
    corrected = phase_vocoder_spectra(
        spectra, config.hop_size, synthesis_hop, phase_locking=phase_locking
    )
    output_length = round(original_length * actual_stretch)
    trim_start = round(config.frame_size * actual_stretch)
    return istft(
        corrected,
        output_length,
        config,
        hop_size=synthesis_hop,
        trim_start=trim_start,
    )


def linear_resample(signal: np.ndarray, output_length: int) -> np.ndarray:
    """Resample with linear interpolation (intentionally the naive baseline).

    This changes duration and pitch together.  It is included so the contrast
    with `time_stretch` can be heard without a third-party resampling library.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("linear_resample expects a non-empty mono signal")
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if output_length == len(signal):
        return signal.copy()
    positions = np.linspace(0, len(signal) - 1, output_length)
    return np.interp(positions, np.arange(len(signal)), signal)


def naive_time_stretch(signal: np.ndarray, stretch: float) -> np.ndarray:
    """Make audio longer/shorter by resampling; pitch changes with duration."""
    if stretch <= 0:
        raise ValueError("stretch must be positive")
    return linear_resample(signal, round(len(signal) * stretch))


def pitch_shift(
    signal: np.ndarray,
    semitones: float,
    config: StftConfig = StftConfig(),
    *,
    phase_locking: bool = False,
) -> np.ndarray:
    """Shift pitch while returning to the input duration.

    First resampling raises/lowers pitch *and* changes duration.  The phase
    vocoder then restores the duration while retaining that new pitch.
    """
    factor = 2 ** (semitones / 12)
    pitch_changed = linear_resample(signal, max(1, round(len(signal) / factor)))
    duration_restored = time_stretch(
        pitch_changed, factor, config, phase_locking=phase_locking
    )
    # The integer output hop can make the duration differ by a few samples.
    return linear_resample(duration_restored, len(signal))


def anonymize_voice(
    signal: np.ndarray, semitones: float = 4.0, config: StftConfig = StftConfig()
) -> np.ndarray:
    """A transparent, duration-preserving voice-transformation preset.

    It intentionally exposes the pitch offset rather than presenting a fixed
    transformation as "anonymous."  A pitch shift alters vocal pitch and, in
    this simple approach, resonant vocal-tract characteristics too.  It cannot
    remove identity cues such as wording, accent, speech rhythm, or context.
    """
    if semitones == 0:
        raise ValueError("anonymization needs a non-zero semitone offset")
    return pitch_shift(signal, semitones, config, phase_locking=True)


def reduce_background_estimate(
    signal: np.ndarray, config: StftConfig = StftConfig(), strength: float = 1.25
) -> np.ndarray:
    """Reduce steady spectral background energy before a voice transformation.

    This lightweight spectral gate estimates a per-frequency background floor
    from the quieter frames. It can reduce hum or steady accompaniment, but is
    not source separation: music sharing frequencies with speech will remain
    and aggressive settings can affect sustained vowels.
    """
    if strength < 0:
        raise ValueError("strength must be non-negative")
    spectra, original_length = stft(signal, config)
    magnitude = np.abs(spectra)
    background_floor = np.percentile(magnitude, 20, axis=0, keepdims=True)
    gain = np.clip((magnitude - strength * background_floor) / (magnitude + 1e-12), 0.08, 1.0)
    return istft(spectra * gain, original_length, config)


def _estimate_pitch(signal: np.ndarray, sample_rate: int) -> float | None:
    """Estimate a voiced fundamental-frequency median with frame autocorrelation."""
    frame_size = min(2_048, len(signal))
    if frame_size < 256:
        return None
    candidates: list[float] = []
    minimum_lag = max(1, int(sample_rate / 350))
    maximum_lag = min(frame_size - 1, int(sample_rate / 70))
    for start in range(0, len(signal) - frame_size + 1, frame_size // 2):
        frame = signal[start : start + frame_size].astype(np.float64)
        frame -= frame.mean()
        if np.sqrt(np.mean(frame**2)) < 0.01:
            continue
        correlation = np.correlate(frame * np.hanning(frame_size), frame * np.hanning(frame_size), mode="full")[frame_size - 1 :]
        lag = minimum_lag + int(np.argmax(correlation[minimum_lag : maximum_lag + 1]))
        if correlation[lag] > 0.25 * correlation[0]:
            candidates.append(sample_rate / lag)
    return float(np.median(candidates)) if candidates else None


def voice_character_profile(signal: np.ndarray, sample_rate: int, config: StftConfig = StftConfig()) -> dict[str, float | None]:
    """Measure broad, non-identifying pitch and brightness traits of a reference."""
    spectra, _ = stft(signal, config)
    magnitude = np.abs(spectra)
    frequencies = np.linspace(0, 1, magnitude.shape[1])
    centroid = np.sum(magnitude * frequencies, axis=1) / (np.sum(magnitude, axis=1) + 1e-12)
    return {"pitch_hz": _estimate_pitch(np.asarray(signal), sample_rate), "brightness": float(np.median(centroid))}


def match_voice_character(
    source: np.ndarray,
    source_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    config: StftConfig = StftConfig(),
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Move a source toward a reference's broad pitch and brightness profile.

    This is deliberately not voice cloning: it matches aggregate DSP traits,
    not a person's identity, phonetics, or learned vocal representation.
    """
    source_profile = voice_character_profile(source, source_rate, config)
    reference_profile = voice_character_profile(reference, reference_rate, config)
    source_pitch, reference_pitch = source_profile["pitch_hz"], reference_profile["pitch_hz"]
    semitones = 0.0
    if source_pitch and reference_pitch:
        semitones = float(np.clip(12 * np.log2(reference_pitch / source_pitch), -8, 8))
    matched = pitch_shift(source, semitones, config, phase_locking=True) if semitones else np.asarray(source, dtype=np.float64).copy()
    spectra, original_length = stft(matched, config)
    brightness_delta = float(np.clip(reference_profile["brightness"] - source_profile["brightness"], -0.25, 0.25))
    frequency_axis = np.linspace(-1, 1, spectra.shape[1])
    # A gentle spectral tilt moves perceived brightness without trying to
    # reconstruct a target speaker's detailed formant signature.
    tilt = np.exp(2.5 * brightness_delta * frequency_axis)
    matched = istft(spectra * tilt, original_length, config)
    return matched, {"pitch_shift_semitones": semitones, **reference_profile}
