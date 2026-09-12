"""Near-real-time pitch/formant shifting for a live microphone feed.

Honesty up front: this is *not* a sample-accurate streaming phase vocoder.
A true streaming phase vocoder needs persistent phase-accumulation state
across an unbounded stream, which would mean rewriting `phase_vocoder.py`'s
batch functions into stateful ones. Instead, this module takes short
overlapping blocks of the incoming audio, runs each one through the existing
*batch* pitch/formant pipeline (`shift_voice_character`), and stitches the
results back together with a windowed overlap-add — the same idea the rest
of this project already uses for synthesis, just applied one block at a
time. The trade is latency (roughly one block, a few hundred milliseconds)
for a live-feeling demo without a DSP rewrite.

To actually route this into a call (Zoom, Discord, etc.), the app can't
inject audio into another process directly — instead, install a virtual
audio cable (e.g. VB-CABLE on Windows), set this module's output device to
that cable's playback endpoint, and select the cable as the microphone
inside the call app. `list_output_devices()` below is what the Live page
uses to populate that choice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from phase_vocoder import StftConfig, linear_resample, shift_voice_character


class _SampleQueue:
    """A small thread-safe FIFO of float samples (producer: audio thread)."""

    def __init__(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def extend(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.concatenate([self._buffer, np.asarray(chunk, dtype=np.float32)])

    def pop(self, count: int) -> np.ndarray:
        with self._lock:
            take = min(count, len(self._buffer))
            chunk = self._buffer[:take]
            self._buffer = self._buffer[take:]
            return chunk

    def clear(self) -> None:
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


@dataclass
class LiveStats:
    blocks_processed: int = 0
    input_buffered: int = 0
    output_queued: int = 0
    underruns: int = 0


class BlockOverlapProcessor:
    """Chunk a continuous input stream into blocks, transform, and overlap-add.

    `push_input(samples)` accepts any number of new samples at a time (audio
    callbacks rarely align with block boundaries) and returns zero or more
    ready output chunks of `hop` samples each, in order.
    """

    def __init__(
        self,
        sample_rate: int,
        config: StftConfig | None = None,
        *,
        block_seconds: float = 0.25,
        phase_locking: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.config = config or StftConfig(sample_rate=sample_rate, frame_size=1024, hop_size=256)
        # The block must comfortably contain several analysis frames or the
        # pitch/formant estimate inside it has nothing to work with.
        self.block_size = max(self.config.frame_size * 3, int(sample_rate * block_seconds))
        self.hop = max(1, self.block_size // 2)  # 50% hop keeps the Hann window's COLA property exact
        self.phase_locking = phase_locking
        self._window = np.hanning(self.block_size)
        # A Hann window at exactly 50% hop sums to a per-sample constant
        # across overlapping blocks (the constant-overlap-add property), so
        # dividing by that one measured constant reconstructs the signal —
        # much safer than dividing by a per-sample weight that is near zero
        # at a block's untouched edges (that's what caused this to blow up
        # during testing: a tiny denominator amplifying noise into a spike).
        overlap_sum = self._window[: self.hop] + self._window[self.hop : 2 * self.hop]
        self._cola_scale = float(np.mean(overlap_sum)) if len(overlap_sum) else 1.0
        self._input_buffer = np.zeros(0)
        self._output_accum = np.zeros(self.block_size)
        self._lock = threading.Lock()
        self.semitones = 0.0
        self.formant_ratio = 1.0
        self.stats = LiveStats()

    def set_parameters(self, semitones: float, formant_ratio: float) -> None:
        with self._lock:
            self.semitones = float(semitones)
            self.formant_ratio = float(formant_ratio)

    def reset(self) -> None:
        self._input_buffer = np.zeros(0)
        self._output_accum = np.zeros(self.block_size)
        self.stats = LiveStats()

    def push_input(self, samples: np.ndarray) -> list[np.ndarray]:
        self._input_buffer = np.concatenate([self._input_buffer, np.asarray(samples, dtype=np.float64)])
        ready: list[np.ndarray] = []
        while len(self._input_buffer) >= self.block_size:
            block = self._input_buffer[: self.block_size]
            self._input_buffer = self._input_buffer[self.hop :]
            ready.append(self._process_block(block))
        self.stats.input_buffered = len(self._input_buffer)
        return ready

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        with self._lock:
            semitones, formant_ratio = self.semitones, self.formant_ratio
        if semitones == 0 and abs(formant_ratio - 1.0) < 1e-6:
            processed = block
        else:
            processed = shift_voice_character(
                block, semitones, formant_ratio, self.config, phase_locking=self.phase_locking
            )
            if len(processed) != self.block_size:
                # Integer-hop rounding inside the batch pipeline can leave the
                # block a handful of samples off; force it back so overlap-add
                # keeps working with a fixed window length.
                processed = linear_resample(processed, self.block_size)

        windowed = processed * self._window
        self._output_accum += windowed

        ready = self._output_accum[: self.hop].copy() / self._cola_scale

        self._output_accum = np.concatenate(
            [self._output_accum[self.hop :], np.zeros(self.hop)]
        )
        self.stats.blocks_processed += 1
        return ready.astype(np.float32)


class LiveVoiceChanger:
    """Wires `BlockOverlapProcessor` to a live microphone in/speaker out pair."""

    def __init__(
        self,
        sample_rate: int = 44_100,
        config: StftConfig | None = None,
        *,
        block_seconds: float = 0.25,
        phase_locking: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.processor = BlockOverlapProcessor(
            sample_rate, config, block_seconds=block_seconds, phase_locking=phase_locking
        )
        self._output_queue = _SampleQueue()
        self._input_stream = None
        self._output_stream = None

    def set_parameters(self, semitones: float, formant_ratio: float) -> None:
        self.processor.set_parameters(semitones, formant_ratio)

    @property
    def running(self) -> bool:
        return self._input_stream is not None

    @property
    def latency_seconds(self) -> float:
        return self.processor.block_size / self.sample_rate

    def start(self, *, output_device: int | str | None = None, input_device: int | str | None = None) -> None:
        if self.running:
            return
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as error:
            raise ValueError("Live mode needs sounddevice. Run: py -3 -m pip install sounddevice") from error

        self.processor.reset()
        self._output_queue.clear()

        def in_callback(indata, frames, time, status) -> None:
            if status:
                self.processor.stats.underruns += 1
            mono = indata[:, 0].astype(np.float64, copy=True)
            for chunk in self.processor.push_input(mono):
                self._output_queue.extend(chunk)
            self.processor.stats.output_queued = len(self._output_queue)

        def out_callback(outdata, frames, time, status) -> None:
            chunk = self._output_queue.pop(frames)
            outdata.fill(0)
            if len(chunk):
                outdata[: len(chunk), 0] = chunk

        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=input_device, callback=in_callback,
        )
        self._output_stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=output_device, callback=out_callback,
        )
        self._input_stream.start()
        self._output_stream.start()

    def stop(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                stream.stop()
                stream.close()
        self._input_stream = None
        self._output_stream = None
        self._output_queue.clear()


def list_output_devices() -> list[tuple[int, str]]:
    """List (index, name) for playback-capable devices, or [] if unavailable.

    On Windows, a virtual audio cable installed for call-routing shows up
    here alongside real speakers/headphones once its driver is installed.
    """
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        devices = sd.query_devices()
    except Exception:
        return []
    return [
        (index, device.get("name", f"Device {index}"))
        for index, device in enumerate(devices)
        if device.get("max_output_channels", 0) > 0
    ]


def list_input_devices() -> list[tuple[int, str]]:
    """List (index, name) for recording-capable devices, or [] if unavailable."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        devices = sd.query_devices()
    except Exception:
        return []
    return [
        (index, device.get("name", f"Device {index}"))
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]


# Substrings seen in the playback-endpoint name of the common free/paid
# virtual audio cables people install for exactly this purpose: making one
# app's output show up as another app's microphone. Matching loosely on
# name is the only "detection" available — there's no OS API that labels a
# device "this one is virtual and meant for call routing".
_VIRTUAL_CABLE_SIGNATURES = (
    "cable input", "vb-audio", "vb-cable", "voicemeeter", "virtual audio cable", "vac",
)


def detect_virtual_cable() -> tuple[int, str] | None:
    """Return the first output device that looks like a virtual audio cable.

    Used to make call routing (WhatsApp, Messenger, Discord, Zoom, ...) a
    one-click "Start call mode" instead of a manual device-picking exercise:
    if something like VB-CABLE is already installed, this finds it so the
    app can select it automatically and just tell the person which name to
    pick inside the call app's microphone settings.
    """
    for index, name in list_output_devices():
        lowered = name.lower()
        if any(signature in lowered for signature in _VIRTUAL_CABLE_SIGNATURES):
            return index, name
    return None
