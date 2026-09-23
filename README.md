# VoxShield

A desktop voice-transformation studio built with Python and tkinter. VoxShield combines a phase vocoder engine, neural voice conversion, real-time mic processing, steganographic audio vaults, and Shazam-style song recognition into a single offline application.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

---

## Features

### Transform
Pitch-shift and time-stretch audio without the chipmunk effect. The phase vocoder preserves pitch independently of duration by re-integrating instantaneous frequencies across STFT frames. A phase-locked variant anchors neighboring bins to spectral peaks for cleaner sustained vowels.

- Independent pitch (semitones) and duration (stretch factor) controls
- Formant ratio adjustment for vocal-tract size shifting
- Presets: Low pitch, High pitch, Feminine-style, Masculine-style, and more
- Four output modes: naive resample, phase vocoder, phase-locked vocoder, voice transformation
- Built-in recording from the default microphone
- Background noise reduction via spectral subtraction

### Match
Shift a voice toward the pitch and spectral character of a reference recording using two backends:

- **LPC (always available):** Matches the reference's pitch contour and averaged LPC spectral-envelope shape. Lightweight, explainable, no downloads required.
- **kNN-VC (optional):** Neural voice conversion using WavLM feature extraction and HiFi-GAN resynthesis. Produces significantly closer timbre matches at the cost of requiring PyTorch and a one-time model download. Falls back to LPC automatically if unavailable.

### Live
Near-real-time pitch and formant shifting on a live microphone feed. Works with virtual audio cables (e.g., VB-CABLE) to route transformed audio into calls on Zoom, Discord, or any other app that accepts a mic input.

- Adjustable pitch (semitones) and formant ratio
- Output device selection for virtual cable routing
- Automatic virtual cable detection
- Call-mode one-click start

### Recognize
Identify a song playing near the microphone using STFT-based audio fingerprinting, backed by a PostgreSQL database of pre-ingested song hashes.

- Record a short clip (configurable duration) from the mic
- Constellation-map peak extraction from the STFT magnitude
- Combinatorial anchor/target hashing (frequency pairs + time delta packed into 32-bit integers)
- Time-alignment scoring: candidates are ranked by the largest single-offset spike in a `db_time - sample_time` histogram, not by raw hash count
- Automatic YouTube link opening on a confident match

### Vault
Hide audio inside a decoy WAV file using AES-GCM encryption. Ordinary players hear the decoy; only VoxShield with the correct passphrase can unlock the hidden clip.

- Create: embed a real audio clip inside a decoy WAV with a passphrase
- Unlock: recover the hidden audio with the correct passphrase
- The passphrase is never stored inside the vault file

### Compare
A guided listening comparison of the four output modes (naive, phase vocoder, phase-locked, voice transformation) with explanations of what to listen for in each.

### Learn
An interactive walkthrough of the phase vocoder pipeline: framing, windowing, FFT, instantaneous frequency estimation, phase accumulation, and overlap-add synthesis, with annotated code snippets at each step.

---

## Installation

### Prerequisites

- Python 3.10 or later
- tkinter (included with most Python distributions on Windows)
- PostgreSQL (required only for the Recognize feature)

### Core dependencies

```bash
pip install numpy sounddevice soundfile
```

### Optional dependencies

| Package | Required for |
|---------|-------------|
| `scipy` | Song recognition (fingerprinting) |
| `psycopg2-binary` | Song recognition (database) |
| `cryptography` | Vault (AES-GCM encryption) |
| `torch`, `torchaudio` | ML voice matching (kNN-VC) |
| `Pillow` | Blurred backdrop on the feature menu |

Install everything at once:

```bash
pip install numpy sounddevice soundfile scipy psycopg2-binary cryptography Pillow
```

---

## Quick start

```bash
cd VoxShield
python desktop_app.py
```

The app opens to a landing page. Click **Explore features** to open the radial feature wheel and pick any module.

---

## Song recognition setup

### 1. Create the database

Using PostgreSQL (local or hosted on [Neon](https://neon.tech)):

```bash
createdb voxshield
psql -d voxshield -f schema.sql
```

Or run `schema.sql` manually in pgAdmin.

### 2. Configure the connection

Set these environment variables (or use a `.env` file):

```
PGHOST=localhost
PGPORT=5432
PGDATABASE=voxshield
PGUSER=your_user
PGPASSWORD=your_password
PGSSLMODE=prefer
```

For Neon or other hosted Postgres, set `PGSSLMODE=require`.

### 3. Ingest songs

Each song needs a WAV file and a YouTube URL:

```bash
python ingest.py path/to/song.wav --title "Song Name" --artist "Artist" \
    --youtube-url "https://youtube.com/watch?v=XXXXXXXXXXX"
```

### 4. Identify from the CLI (optional)

```bash
python match.py --seconds 5
```

Or use the **Recognize** tab in the app to record and match from the GUI.

### How matching works

1. Record N seconds of audio and run it through the same STFT + peak-extraction + hashing pipeline used at ingest time.
2. Look up every generated hash in the `hashes` table. A real song produces many exact `hash_value` collisions even under mic noise, because the hashing is local (anchor/target frequency pairs + a time delta), so noise only kills individual hashes rather than the whole fingerprint.
3. **Time alignment:** for each candidate song, compute `db_time - sample_time` for every matching hash and bucket those offsets. A true match has a large spike of hashes all agreeing on one offset; noise collisions scatter across many different offsets. The song with the largest single-offset spike (above `min_agreement`) wins.
4. Fetch that song's `youtube_url` and open it in the browser.

---

## Project structure

```
VoxShield/
  desktop_app.py      Main tkinter application (UI, navigation, feature pages)
  phase_vocoder.py     STFT, time-stretch, pitch-shift, formant warp, LPC matching
  audio_io.py          Audio file I/O, recording, playback
  realtime.py          Live microphone pitch/formant shifting
  ml_voice.py          Optional kNN-VC neural voice conversion
  secure_audio.py      AES-GCM steganographic vault (encode/decode)
  fingerprint.py       STFT peak extraction and combinatorial hashing
  db.py                PostgreSQL connection helper (psycopg2)
  ingest.py            Bulk-insert song fingerprints into the database
  match.py             Mic recording, hash lookup, time-alignment scoring
  schema.sql           PostgreSQL schema (songs + hashes tables)
  process_wav.py       CLI batch processing utility
  wav_io.py            Low-level WAV read/write
```

---

## Architecture

```
                    ┌──────────────┐
                    │  desktop_app │  tkinter UI
                    └──────┬───────┘
           ┌───────────────┼───────────────┬──────────────┐
           v               v               v              v
    ┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌───────────┐
    │phase_vocoder│ │  realtime   │ │ ml_voice   │ │secure_audio│
    │  STFT core  │ │ live stream │ │  kNN-VC    │ │  AES vault │
    └──────┬──────┘ └──────┬──────┘ └────────────┘ └───────────┘
           │               │
           v               v
    ┌─────────────┐ ┌─────────────┐
    │  audio_io   │ │ sounddevice │
    │ read/write  │ │  mic / spk  │
    └─────────────┘ └─────────────┘

    ┌──────────────────────────────────┐
    │     Song Recognition Pipeline    │
    │  fingerprint → db → ingest/match │
    │         (PostgreSQL)             │
    └──────────────────────────────────┘
```

---

## Sharing the database

To let collaborators use the same song library:

1. Host PostgreSQL on [Neon](https://neon.tech) (free tier available).
2. Share the connection string with collaborators.
3. For read-only access, create a restricted role:

```sql
CREATE ROLE friend WITH LOGIN PASSWORD 'a_strong_password';
GRANT CONNECT ON DATABASE voxshield TO friend;
GRANT USAGE ON SCHEMA public TO friend;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO friend;
```

For full read-write access (can ingest new songs):

```sql
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO friend;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO friend;
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT
