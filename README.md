# VoxShield
Phase Vocoder Voice Changer application developed with Python 

## Song recognition (fingerprinting) feature

A Shazam-style module for identifying songs from a short mic recording,
built on top of the same STFT foundation used elsewhere in the project.

- `schema.sql` — PostgreSQL tables: `songs` (with `youtube_url`) and `hashes`
  (`hash_value`, `song_id`, `absolute_time`).
- `fingerprint.py` — turns an STFT array into peaks ("constellation map")
  then into combinatorial `(anchor_freq, target_freq, delta_t)` hashes.
- `db.py` — psycopg2 connection helper (reads `PGHOST`/`PGDATABASE`/etc.
  from the environment).
- `ingest.py` — bulk-inserts a song's fingerprint + metadata into the DB.
- `match.py` — records from the mic, fingerprints it, queries the DB,
  scores candidates by time-alignment agreement (not just raw hash count),
  and opens the winning `youtube_url` in a browser.

### Setup

```bash
pip install psycopg2-binary scipy numpy sounddevice soundfile
createdb shazam_poc
psql -d shazam_poc -f schema.sql
```

### Usage

Ingest a library of songs (each needs a WAV file and a YouTube URL):

```bash
python ingest.py path/to/song.wav --title "Song Name" --artist "Artist" \
    --youtube-url "https://youtube.com/watch?v=XXXXXXXXXXX"
```

Identify whatever's playing near the mic:

```bash
python match.py --seconds 5
```

### How matching works

1. Record N seconds of audio, run it through the same STFT + peak-extraction
   + hashing pipeline used at ingest time.
2. Look up every generated hash in the `hashes` table — a real song produces
   many exact `hash_value` collisions even under mic noise, because the
   hashing is local (anchor/target frequency pairs + a time delta), so noise
   only kills individual hashes rather than the whole fingerprint.
3. **Time alignment**: the song with the most raw hash matches doesn't
   necessarily win — random collisions are common. Instead, for each
   candidate song we compute `db_time - sample_time` for every matching hash
   and bucket those offsets. A true match has a large spike of hashes all
   agreeing on *one* offset (the recording is a snippet of the original
   file, so every hash pair is shifted by the same constant); noise
   collisions scatter across many different offsets. The song with the
   largest single-offset spike (above `--min-agreement`) wins.
4. Fetch that song's `youtube_url` and open it with `webbrowser.open()`.
