"""Create listening comparisons for the phase-vocoder project."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase_vocoder import anonymize_voice, StftConfig, naive_time_stretch, pitch_shift, time_stretch
from audio_io import read_audio, write_output_wav


parser = argparse.ArgumentParser(description="Compare naive resampling and a phase-vocoder voice transform.")
parser.add_argument("input", type=Path, help="audio input (WAV always works; more formats need optional audio packages)")
parser.add_argument("--stretch", type=float, default=1.5, help="duration factor; 1.5 is 50%% longer")
parser.add_argument("--pitch-semitones", type=float, help="also write a fixed-duration pitch-shifted version")
parser.add_argument(
    "--anonymize-semitones",
    type=float,
    help="write a phase-locked voice-transformation version; choose a non-zero offset",
)
parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
args = parser.parse_args()

if args.stretch <= 0:
    parser.error("--stretch must be positive")

signal, sample_rate = read_audio(args.input)
args.output_dir.mkdir(parents=True, exist_ok=True)
stem = args.input.stem
config = StftConfig(sample_rate=sample_rate)

outputs = {
    f"{stem}_naive_{args.stretch:g}x.wav": naive_time_stretch(signal, args.stretch),
    f"{stem}_phase_vocoder_{args.stretch:g}x.wav": time_stretch(signal, args.stretch, config),
    f"{stem}_phase_locked_{args.stretch:g}x.wav": time_stretch(
        signal, args.stretch, config, phase_locking=True
    ),
}
if args.pitch_semitones is not None:
    outputs[f"{stem}_pitch_{args.pitch_semitones:+g}st.wav"] = pitch_shift(signal, args.pitch_semitones, config)
if args.anonymize_semitones is not None:
    if args.anonymize_semitones == 0:
        parser.error("--anonymize-semitones must be non-zero")
    outputs[f"{stem}_voice_transform_{args.anonymize_semitones:+g}st.wav"] = anonymize_voice(
        signal, args.anonymize_semitones, config
    )

for filename, transformed in outputs.items():
    destination = args.output_dir / filename
    clipped = write_output_wav(destination, transformed, sample_rate)
    note = f" ({clipped} clipped samples)" if clipped else ""
    print(f"wrote {destination}: {len(transformed)} samples{note}")
