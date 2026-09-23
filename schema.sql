-- schema.sql
-- PostgreSQL schema for the fingerprint-matching proof of concept.
--
-- Run with:  psql -U <user> -d <database> -f schema.sql

CREATE TABLE IF NOT EXISTS songs (
    song_id      SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    artist       TEXT,
    youtube_url  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- hash_value is a 32-bit integer packed from (anchor_freq, target_freq, delta_t).
-- One song produces many thousands of rows here, so the table is intentionally
-- narrow and the (hash_value) index is what makes matching fast.
CREATE TABLE IF NOT EXISTS hashes (
    hash_value     INTEGER NOT NULL,
    song_id        INTEGER NOT NULL REFERENCES songs(song_id) ON DELETE CASCADE,
    absolute_time  REAL    NOT NULL   -- seconds from the start of the song, at the anchor point
);

CREATE INDEX IF NOT EXISTS idx_hashes_hash_value ON hashes (hash_value);
CREATE INDEX IF NOT EXISTS idx_hashes_song_id ON hashes (song_id);

