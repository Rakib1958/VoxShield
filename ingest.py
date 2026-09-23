"""ingest.py

Bulk-insert a song's fingerprint hashes (and its metadata) into PostgreSQL.

Usage:
    from fingerprint import fingerprint_stft
    hashes = fingerprint_stft(Zxx, freqs, times)
    ingest_song(title="Song Name", artist="Some Artist",
                youtube_url="https://youtube.com/watch?v=...", hashes=hashes)
"""

from __future__ import annotations

from db import cursor
from fingerprint import FingerprintHash


def ingest_song(
    *,
    title: str,
    artist: str | None,
    youtube_url: str,
    hashes: list[FingerprintHash],
) -> int:
    """Insert song metadata + its hashes. Returns the new song_id."""
    try:
        from psycopg2.extras import execute_values  # type: ignore[import-not-found]
    except ImportError as error:
        raise ValueError("Song recognition needs psycopg2. Run: py -3 -m pip install psycopg2-binary") from error

    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO songs (title, artist, youtube_url)
            VALUES (%s, %s, %s)
            RETURNING song_id
            """,
            (title, artist, youtube_url),
        )
        song_id = cur.fetchone()[0]

        rows = [(h.hash_value, song_id, h.anchor_time) for h in hashes]
        execute_values(
            cur,
            "INSERT INTO hashes (hash_value, song_id, absolute_time) VALUES %s",
            rows,
            page_size=1000,
        )

    print(f"Ingested '{title}' as song_id={song_id} ({len(hashes)} hashes)")
    return song_id


def ingest_from_wav(path: str, title: str, artist: str | None, youtube_url: str) -> int:
    """Full pipeline: WAV file -> STFT -> hashes -> DB. Requires scipy."""
    import soundfile as sf
    from scipy.signal import stft

    from fingerprint import fingerprint_stft

    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:  # collapse stereo to mono
        audio = audio.mean(axis=1)

    freqs, times, Zxx = stft(audio, fs=sample_rate, nperseg=4096)
    hashes = fingerprint_stft(Zxx, freqs, times)

    return ingest_song(title=title, artist=artist, youtube_url=youtube_url, hashes=hashes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a song's fingerprint into the database.")
    parser.add_argument("wav_path", help="Path to the song's WAV file")
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", default=None)
    parser.add_argument("--youtube-url", required=True)
    args = parser.parse_args()

    ingest_from_wav(args.wav_path, args.title, args.artist, args.youtube_url)
