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


def _analysis_frames(signal: np.ndarray, config: StftConfig) -> tuple[np.ndarray, int]:
    """Slice a mono signal into padded, windowed analysis frames.

    Shared by `stft` (which FFTs these frames) and the LPC formant tools
    further down (which need the time-domain frames themselves, not their
    spectra, to estimate an all-pole envelope).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("expects a one-dimensional (mono) signal")

    # The Hann window is zero at its ends.  Padding both sides ensures every
    # real input sample, including the first and last, is covered by interior
    # (non-zero) portions of one or more windows.
    edge_padding = config.frame_size
    analysis_signal = np.pad(signal, (edge_padding, edge_padding))
    frames = frame_count(len(analysis_signal), config)
    padded_length = (frames - 1) * config.hop_size + config.frame_size
    padded = np.pad(analysis_signal, (0, padded_length - len(analysis_signal)))
    window = sqrt_hann(config.frame_size)
    framed = np.empty((frames, config.frame_size))

    for index in range(frames):
        start = index * config.hop_size
        framed[index] = padded[start : start + config.frame_size] * window
    return framed, len(signal)


def stft(signal: np.ndarray, config: StftConfig) -> tuple[np.ndarray, int]:
    """Analyze a mono signal into one real-FFT spectrum per frame.

    The spectrum is complex. `abs(spectrum)` is the magnitude; `angle(spectrum)`
    is its phase, which carries timing information and must be preserved.
    """
    framed, original_length = _analysis_frames(signal, config)
    # Replace np.fft.rfft here with your own real-FFT adapter if desired.
    spectra = np.fft.rfft(framed, axis=1)
    return spectra, original_length


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
    signal: np.ndarray,
    semitones: float = 4.0,
    config: StftConfig = StftConfig(),
    *,
    formant_ratio: float = 1.0,
) -> np.ndarray:
    """A transparent, duration-preserving voice-transformation preset.

    It intentionally exposes the pitch offset (and, now, the formant ratio)
    rather than presenting a fixed transformation as "anonymous." `semitones`
    changes vocal pitch; `formant_ratio` independently changes how large the
    vocal tract sounds (see `shift_voice_character`). Neither can remove
    identity cues such as wording, accent, speech rhythm, or context.
    """
    if semitones == 0 and abs(formant_ratio - 1.0) < 1e-6:
        raise ValueError("anonymization needs a non-zero semitone offset or formant ratio")
    return shift_voice_character(signal, semitones, formant_ratio, config, phase_locking=True)


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


def _levinson_durbin(autocorrelation: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Solve the Yule-Walker equations for an all-pole (LPC) model.

    Returns `(coefficients, error_power)` such that the all-pole filter is
    `A(z) = 1 + coefficients[0]*z^-1 + ... + coefficients[order-1]*z^-order`,
    and `error_power` is the leftover (unpredicted) energy — the gain that
    `1 / A(z)` needs to reproduce the frame's overall loudness.
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    error = float(autocorrelation[0])
    if error <= 1e-12:
        return np.zeros(order), 1e-12
    for i in range(1, order + 1):
        acc = autocorrelation[i] + np.dot(a[1:i], autocorrelation[i - 1 : 0 : -1])
        reflection = -acc / error
        updated = a.copy()
        updated[1:i] = a[1:i] + reflection * a[i - 1 : 0 : -1]
        updated[i] = reflection
        a = updated
        error *= 1 - reflection**2
        if error <= 1e-12:
            error = 1e-12
            break
    return a[1:], error


def default_lpc_order(sample_rate: int) -> int:
    """A common rule of thumb for speech: roughly two poles per kHz, plus a few."""
    return 2 + sample_rate // 1000


def lpc_envelope(frame: np.ndarray, order: int, fft_size: int) -> np.ndarray:
    """Estimate a smooth spectral envelope for one time-domain frame.

    Fits an all-pole filter to the frame's autocorrelation (Levinson-Durbin),
    then evaluates `sqrt(error) / |A(e^jw)|` at `fft_size // 2 + 1`
    frequencies. This is the classic, lightweight way to describe "what the
    vocal tract is doing" (the formants) separately from the finer harmonic
    detail that carries pitch — the frame is assumed already windowed, so no
    extra tapering is applied here.
    """
    frame = np.asarray(frame, dtype=np.float64)
    autocorrelation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :][: order + 1]
    coefficients, error = _levinson_durbin(autocorrelation, order)
    polynomial = np.concatenate(([1.0], coefficients))
    response = np.fft.rfft(polynomial, n=fft_size)
    return np.sqrt(error) / np.maximum(np.abs(response), 1e-9)


def formant_shift(
    signal: np.ndarray,
    ratio: float,
    config: StftConfig = StftConfig(),
    *,
    lpc_order: int | None = None,
) -> np.ndarray:
    """Move formant (vocal-tract resonance) frequencies without touching pitch.

    `ratio > 1` raises the formants (reads as a smaller/younger vocal tract);
    `ratio < 1` lowers them (bigger/older). Each frame's spectrum is divided
    by its own LPC envelope to get a flattened "residual" that carries the
    pitch harmonics and fine detail, the envelope is warped along the
    frequency axis, and the residual is multiplied back in — so the
    excitation (and therefore the pitch) is left alone.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if abs(ratio - 1.0) < 1e-6:
        return np.asarray(signal, dtype=np.float64).copy()

    order = lpc_order or default_lpc_order(config.sample_rate)
    frames, original_length = _analysis_frames(signal, config)
    spectra = np.fft.rfft(frames, axis=1)
    bins = spectra.shape[1]
    bin_index = np.arange(bins, dtype=np.float64)
    warped = np.empty_like(spectra)

    for i, frame in enumerate(frames):
        envelope = lpc_envelope(frame, order, config.frame_size)
        residual = spectra[i] / np.maximum(envelope, 1e-9)
        # Bin k should receive whatever used to sit at k / ratio: raising
        # `ratio` pulls low-frequency envelope detail up into higher bins,
        # i.e. formants move up.
        source_positions = np.clip(bin_index / ratio, 0, bins - 1)
        new_envelope = np.interp(source_positions, bin_index, envelope)
        warped[i] = residual * new_envelope

    return istft(warped, original_length, config)


def shift_voice_character(
    signal: np.ndarray,
    semitones: float,
    formant_ratio: float = 1.0,
    config: StftConfig = StftConfig(),
    *,
    phase_locking: bool = True,
) -> np.ndarray:
    """Shift pitch and vocal-tract character independently.

    `pitch_shift` resamples the signal, which is simple but scales formants
    by the same factor as pitch — the classic "chipmunk/monster" artifact
    once the offset gets past a couple of semitones, because a real voice
    can change register without its vocal tract changing size. Here that
    automatic formant shift is undone with `formant_shift` (dividing out the
    resample factor) and whichever `formant_ratio` was actually asked for is
    applied on top. The result: `semitones` controls how high the voice is,
    `formant_ratio` controls how "big" it sounds, and the two stop fighting.
    """
    pitch_factor = 2 ** (semitones / 12) if semitones else 1.0
    shifted = (
        pitch_shift(signal, semitones, config, phase_locking=phase_locking)
        if semitones
        else np.asarray(signal, dtype=np.float64).copy()
    )
    net_formant_ratio = formant_ratio / pitch_factor
    if abs(net_formant_ratio - 1.0) > 1e-3:
        shifted = formant_shift(shifted, net_formant_ratio, config)
    return shifted


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


def _average_lpc_envelope(signal: np.ndarray, config: StftConfig, order: int) -> np.ndarray:
    """Average the LPC spectral envelope across a signal's louder ("voiced") frames.

    Quiet frames (silence, breath) have unreliable autocorrelation and would
    just average in noise, so frames below 10% of the loudest frame's RMS are
    skipped when there is enough signal to do so.
    """
    frames, _ = _analysis_frames(signal, config)
    if len(frames) == 0:
        return np.ones(config.frame_size // 2 + 1)
    energies = np.sqrt(np.mean(frames**2, axis=1))
    threshold = 0.1 * energies.max()
    voiced = frames[energies > threshold] if threshold > 0 else frames
    if len(voiced) == 0:
        voiced = frames
    envelopes = np.stack([lpc_envelope(frame, order, config.frame_size) for frame in voiced])
    return np.mean(envelopes, axis=0)


def voice_character_profile(
    signal: np.ndarray,
    sample_rate: int,
    config: StftConfig = StftConfig(),
    *,
    lpc_order: int | None = None,
) -> dict[str, object]:
    """Measure broad, non-identifying pitch and vocal-tract traits of a reference.

    `envelope` is the signal's averaged LPC spectral envelope shape — the
    broad resonant "size" of the voice, used by `match_voice_character` to
    nudge a source toward a reference without reconstructing its identity.
    """
    order = lpc_order or default_lpc_order(sample_rate)
    spectra, _ = stft(signal, config)
    magnitude = np.abs(spectra)
    frequencies = np.linspace(0, 1, magnitude.shape[1])
    centroid = np.sum(magnitude * frequencies, axis=1) / (np.sum(magnitude, axis=1) + 1e-12)
    return {
        "pitch_hz": _estimate_pitch(np.asarray(signal), sample_rate),
        "brightness": float(np.median(centroid)),
        "envelope": _average_lpc_envelope(signal, config, order),
    }


def match_voice_character(
    source: np.ndarray,
    source_rate: int,
    reference: np.ndarray,
    reference_rate: int,
    config: StftConfig = StftConfig(),
    *,
    envelope_strength: float = 0.7,
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Move a source toward a reference's broad pitch and vocal-tract character.

    Pitch is matched to the reference's median fundamental with a
    formant-preserving shift, so the pitch move itself doesn't distort
    timbre. Separately, each frame's LPC spectral-envelope *shape* is blended
    `envelope_strength` of the way toward the reference's averaged envelope
    shape, while that frame's own energy is kept — so loudness and voiced/
    unvoiced dynamics still follow the source, only the resonance shape
    leans toward the reference.

    This is deliberately not voice cloning: a single averaged target envelope
    cannot track the reference's frame-by-frame articulation, only nudge the
    source's broad resonant character — no learned vocal identity involved.
    """
    order = default_lpc_order(source_rate)
    source_profile = voice_character_profile(source, source_rate, config, lpc_order=order)
    reference_profile = voice_character_profile(reference, reference_rate, config, lpc_order=order)
    source_pitch, reference_pitch = source_profile["pitch_hz"], reference_profile["pitch_hz"]
    semitones = 0.0
    if source_pitch and reference_pitch:
        semitones = float(np.clip(12 * np.log2(reference_pitch / source_pitch), -8, 8))
    matched = (
        shift_voice_character(source, semitones, 1.0, config, phase_locking=True)
        if semitones
        else np.asarray(source, dtype=np.float64).copy()
    )

    frames, original_length = _analysis_frames(matched, config)
    spectra = np.fft.rfft(frames, axis=1)
    reference_shape = reference_profile["envelope"]
    reference_shape = reference_shape / max(float(np.mean(reference_shape)), 1e-9)
    strength = float(np.clip(envelope_strength, 0.0, 1.0))
    out = np.empty_like(spectra)

    for i, frame in enumerate(frames):
        envelope = lpc_envelope(frame, order, config.frame_size)
        residual = spectra[i] / np.maximum(envelope, 1e-9)
        gain = float(np.mean(envelope))
        source_shape = envelope / max(gain, 1e-9)
        blended_shape = source_shape * (1 - strength) + reference_shape * strength
        out[i] = residual * (blended_shape * gain)

    matched = istft(out, original_length, config)
    return matched, {
        "pitch_shift_semitones": semitones,
        "pitch_hz": reference_profile["pitch_hz"],
        "brightness": reference_profile["brightness"],
    }
