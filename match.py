"""match.py

Live-microphone matching: record a few seconds of audio, fingerprint it the
same way ingest.py fingerprinted the library, look up matching hashes in
PostgreSQL, and find the song whose hashes align in time consistently (not
just the song with the most raw hash collisions — that alone is too noisy).

CLI usage:
    python match.py                  # record 5s from the default mic and match
    python match.py --seconds 8

Library usage (e.g. from desktop_app.py, which already records audio itself):
    from match import identify_song
    result = identify_song(samples, sample_rate)
    if result:
        print(result.title, result.youtube_url)
"""

from __future__ import annotations

import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from db import cursor
from fingerprint import FingerprintHash, fingerprint_stft

DEFAULT_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class MatchResult:
    song_id: int
    title: str
    artist: str | None
    youtube_url: str
    confidence: int  # number of hashes agreeing on the winning time offset


# ---------------------------------------------------------------------------
# 1. Record audio (CLI only — desktop_app.py records via audio_io.Recorder)
# ---------------------------------------------------------------------------

def record_audio(seconds: float = 5.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Recording needs sounddevice. Run: py -3 -m pip install sounddevice") from error

    print(f"Listening for {seconds}s...")
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float64")
    sd.wait()
    print("Done recording.")
    return recording[:, 0]


# ---------------------------------------------------------------------------
# 2. Query the database for hash matches
# ---------------------------------------------------------------------------

def query_matches(hashes: list[FingerprintHash]) -> list[tuple[int, int, float, float]]:
    """Returns rows of (hash_value, song_id, db_absolute_time, sample_anchor_time)."""
    if not hashes:
        return []

    hash_to_sample_time = {h.hash_value: h.anchor_time for h in hashes}
    hash_values = list(hash_to_sample_time.keys())

    with cursor() as cur:
        cur.execute(
            """
            SELECT hash_value, song_id, absolute_time
            FROM hashes
            WHERE hash_value = ANY(%s)
            """,
            (hash_values,),
        )
        rows = cur.fetchall()

    return [
        (hash_value, song_id, db_time, hash_to_sample_time[hash_value])
        for hash_value, song_id, db_time in rows
    ]


# ---------------------------------------------------------------------------
# 3. Time-alignment scoring
# ---------------------------------------------------------------------------

def find_best_match(rows: list[tuple[int, int, float, float]]) -> tuple[int, int] | None:
    """Groups matches by song_id, then finds the offset (db_time - sample_time)
    that the most hashes agree on. A real match has many hashes clustering on
    the *same* offset (the song is playing at a fixed alignment); random
    collisions scatter across many different offsets.

    Returns (song_id, agreement_count) for the winner, or None if no matches.
    """
    # song_id -> Counter of rounded offset -> count
    offset_counts: dict[int, Counter] = defaultdict(Counter)

    for _hash_value, song_id, db_time, sample_time in rows:
        offset = round(db_time - sample_time, 1)  # 100ms buckets absorb jitter
        offset_counts[song_id][offset] += 1

    best_song_id = None
    best_score = 0
    for song_id, counts in offset_counts.items():
        _offset, score = counts.most_common(1)[0]
        if score > best_score:
            best_song_id = song_id
            best_score = score

    if best_song_id is None:
        return None
    return best_song_id, best_score


# ---------------------------------------------------------------------------
# 4. Fetch metadata
# ---------------------------------------------------------------------------

def fetch_song(song_id: int) -> tuple[str, str | None, str]:
    with cursor() as cur:
        cur.execute("SELECT title, artist, youtube_url FROM songs WHERE song_id = %s", (song_id,))
        return cur.fetchone()


# ---------------------------------------------------------------------------
# 5. End-to-end: samples in, MatchResult out (no recording, no browser)
# ---------------------------------------------------------------------------

def identify_song(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    min_agreement: int = 5,
) -> MatchResult | None:
    """Fingerprint an already-recorded clip and look up its best DB match.

    Returns None if no match clears `min_agreement` aligned hashes. Raises
    ValueError (with a friendly message) if scipy/psycopg2 aren't installed
    or the database isn't reachable — callers should catch that and show it.
    """
    try:
        from scipy.signal import stft  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Song recognition needs scipy. Run: py -3 -m pip install scipy") from error

    freqs, times, Zxx = stft(samples, fs=sample_rate, nperseg=4096)
    hashes = fingerprint_stft(Zxx, freqs, times)

    rows = query_matches(hashes)
    result = find_best_match(rows)
    if result is None or result[1] < min_agreement:
        return None

    song_id, score = result
    title, artist, youtube_url = fetch_song(song_id)
    return MatchResult(song_id=song_id, title=title, artist=artist,
                        youtube_url=youtube_url, confidence=score)


def identify_and_play(seconds: float = 5.0, min_agreement: int = 5) -> None:
    """CLI convenience: record, identify, print, and open the result in a browser."""
    audio = record_audio(seconds)
    result = identify_song(audio, DEFAULT_SAMPLE_RATE, min_agreement=min_agreement)

    if result is None:
        print("No confident match found.")
        return

    print(f"Match: '{result.title}'" + (f" by {result.artist}" if result.artist else "")
          + f" (confidence={result.confidence})")
    webbrowser.open(result.youtube_url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record from the mic and identify the playing song.")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--min-agreement", type=int, default=5, help="Minimum aligned-hash count to call it a match")
    args = parser.parse_args()

    identify_and_play(seconds=args.seconds, min_agreement=args.min_agreement)
