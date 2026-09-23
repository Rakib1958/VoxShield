"""fingerprint.py

Constellation-map peak extraction and combinatorial hashing, built on top of
an existing STFT array (e.g. the output of scipy.signal.stft).

This module is deliberately decoupled from how you computed the STFT: hand it
a 2D magnitude array plus the frequency/time axes scipy.signal.stft gives you
back, and it does the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Peak:
    time_idx: int       # index into the STFT time axis
    freq_idx: int        # index into the STFT frequency axis
    time: float           # seconds
    freq: float           # Hz


@dataclass(frozen=True)
class FingerprintHash:
    hash_value: int
    anchor_time: float   # seconds — this is what gets stored as absolute_time


# ---------------------------------------------------------------------------
# 1. Peak extraction ("constellation map")
# ---------------------------------------------------------------------------

def extract_peaks(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    neighborhood_size: tuple[int, int] = (20, 20),
    amp_min_percentile: float = 75.0,
) -> list[Peak]:
    """Find local-maxima peaks in a 2D STFT magnitude array.

    Parameters
    ----------
    Zxx : complex or real ndarray, shape (n_freqs, n_times)
        The STFT output (e.g. from scipy.signal.stft). Complex input is
        converted to magnitude automatically.
    freqs, times : 1D ndarrays
        The frequency and time axes returned alongside Zxx.
    neighborhood_size : (freq_bins, time_bins)
        Size of the local-maximum window. Bigger windows => fewer, stronger
        peaks; matches scipy.ndimage.maximum_filter's `size` argument.
    amp_min_percentile : float
        Peaks below this percentile of the magnitude distribution are
        discarded as noise. Raise it to get fewer, more robust peaks.
    """
    try:
        from scipy.ndimage import maximum_filter  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Song recognition needs scipy. Run: py -3 -m pip install scipy") from error

    magnitude = np.abs(Zxx)

    # Local maximum filter: a point survives only if it equals the max of its
    # neighborhood (i.e. it is the tallest point in that window).
    local_max = maximum_filter(magnitude, size=neighborhood_size, mode="constant")
    is_peak = magnitude == local_max

    # Drop peaks that are just quiet-background noise.
    threshold = np.percentile(magnitude, amp_min_percentile)
    is_peak &= magnitude > threshold

    freq_idxs, time_idxs = np.nonzero(is_peak)

    peaks = [
        Peak(
            time_idx=int(t_idx),
            freq_idx=int(f_idx),
            time=float(times[t_idx]),
            freq=float(freqs[f_idx]),
        )
        for f_idx, t_idx in zip(freq_idxs, time_idxs)
    ]
    # Chronological order matters for the fan-out step below.
    peaks.sort(key=lambda p: p.time_idx)
    return peaks


# ---------------------------------------------------------------------------
# 2. Combinatorial hashing ("anchor -> target" pairs)
# ---------------------------------------------------------------------------

def _pack_hash(anchor_freq: int, target_freq: int, delta_t_ms: int) -> int:
    """Pack (anchor_freq, target_freq, delta_t) into a single 32-bit int.

    9 bits for each frequency bin (0-511) and 14 bits for the time delta in
    milliseconds (0-16383 ms, ~16s). Adjust the bit widths if your STFT has
    more than 512 frequency bins.
    """
    anchor_freq &= 0x1FF     # 9 bits
    target_freq &= 0x1FF     # 9 bits
    delta_t_ms &= 0x3FFF     # 14 bits
    return (anchor_freq << 23) | (target_freq << 14) | delta_t_ms


def generate_hashes(
    peaks: list[Peak],
    *,
    fan_out: int = 5,
    min_delta_t: float = 0.0,
    max_delta_t: float = 4.0,
) -> list[FingerprintHash]:
    """Turn a list of peaks into anchor/target hashes.

    For each peak (the "anchor"), pair it with up to `fan_out` subsequent
    peaks (the "targets") that fall within [min_delta_t, max_delta_t] seconds
    of it. This is the classic Shazam-style combinatorial hashing scheme —
    it's what makes the fingerprint robust to noise, since a match only needs
    enough surviving hash collisions, not a clean full-spectrum match.
    """
    hashes: list[FingerprintHash] = []

    for i, anchor in enumerate(peaks):
        targets_found = 0
        for target in peaks[i + 1 :]:
            delta_t = target.time - anchor.time
            if delta_t < min_delta_t:
                continue
            if delta_t > max_delta_t:
                break  # peaks are time-sorted, nothing further will qualify
            if targets_found >= fan_out:
                break

            delta_t_ms = int(round(delta_t * 1000))
            hash_value = _pack_hash(anchor.freq_idx, target.freq_idx, delta_t_ms)
            hashes.append(FingerprintHash(hash_value=hash_value, anchor_time=anchor.time))
            targets_found += 1

    return hashes


def fingerprint_stft(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    **kwargs,
) -> list[FingerprintHash]:
    """Convenience wrapper: STFT array in, hash list out."""
    peaks = extract_peaks(Zxx, freqs, times)
    return generate_hashes(peaks, **kwargs)
