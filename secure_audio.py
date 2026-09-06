"""Decoy-playable WAV containers with AES-GCM protected audio payloads."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import struct
import wave

import numpy as np

from audio_io import read_audio

CHUNK_ID = b"pva0"
MAGIC = b"PVLT1"


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM
    except ImportError as error:
        raise ValueError("Vault encryption needs cryptography. Run: py -3 -m pip install cryptography") from error


def _pcm16(samples: np.ndarray) -> bytes:
    return np.round(np.clip(np.asarray(samples, dtype=np.float64), -1, 1) * 32767).astype("<i2").tobytes()


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(_pcm16(samples))
    return buffer.getvalue()


def _key(passphrase: str, decoy_pcm: bytes, salt: bytes) -> bytes:
    if len(passphrase) < 8:
        raise ValueError("Use a passphrase of at least 8 characters.")
    # The decoy binds this payload to its audible container; the passphrase is
    # the secret. The decoy alone is intentionally not treated as a secret.
    material = passphrase.encode("utf-8") + hashlib.sha256(decoy_pcm).digest()
    return hashlib.scrypt(material, salt=salt, n=2**14, r=8, p=1, dklen=32)


def _find_chunk(container: bytes, identifier: bytes) -> tuple[int, bytes]:
    if container[:4] != b"RIFF" or container[8:12] != b"WAVE":
        raise ValueError("Not a WAV vault container.")
    position = 12
    while position + 8 <= len(container):
        chunk_id, size = container[position : position + 4], struct.unpack_from("<I", container, position + 4)[0]
        data_start, data_end = position + 8, position + 8 + size
        if data_end > len(container):
            raise ValueError("The WAV container is truncated.")
        if chunk_id == identifier:
            return position, container[data_start:data_end]
        position = data_end + (size % 2)
    raise ValueError("This WAV does not contain a Voice Lab vault payload.")


def create_vault(real_samples: np.ndarray, real_sample_rate: int, decoy_path: str | Path, passphrase: str, destination: str | Path) -> None:
    """Write a normal-playable decoy WAV with a hidden encrypted real clip."""
    decoy_samples, decoy_rate = read_audio(decoy_path)
    decoy_wav = bytearray(_wav_bytes(decoy_samples, decoy_rate))
    _, decoy_pcm = _find_chunk(decoy_wav, b"data")
    salt, nonce = os.urandom(16), os.urandom(12)
    plaintext = struct.pack("<I", real_sample_rate) + _pcm16(real_samples)
    ciphertext = _aesgcm()(_key(passphrase, decoy_pcm, salt)).encrypt(nonce, plaintext, MAGIC)
    payload = MAGIC + salt + nonce + ciphertext
    decoy_wav.extend(CHUNK_ID + struct.pack("<I", len(payload)) + payload)
    if len(payload) % 2:
        decoy_wav.extend(b"\0")
    struct.pack_into("<I", decoy_wav, 4, len(decoy_wav) - 8)
    Path(destination).write_bytes(decoy_wav)


def unlock_vault(path: str | Path, passphrase: str) -> tuple[np.ndarray, int]:
    """Decrypt a Voice Lab vault and return the hidden clip as a NumPy array."""
    container = Path(path).read_bytes()
    _, payload = _find_chunk(container, CHUNK_ID)
    if len(payload) < len(MAGIC) + 28 or payload[:5] != MAGIC:
        raise ValueError("Unsupported or damaged Voice Lab vault payload.")
    _, decoy_pcm = _find_chunk(container, b"data")
    salt, nonce, ciphertext = payload[5:21], payload[21:33], payload[33:]
    try:
        plaintext = _aesgcm()(_key(passphrase, decoy_pcm, salt)).decrypt(nonce, ciphertext, MAGIC)
    except Exception as error:
        raise ValueError("Could not unlock this vault. Check the passphrase and decoy integrity.") from error
    if len(plaintext) < 4 or (len(plaintext) - 4) % 2:
        raise ValueError("Decrypted audio payload is invalid.")
    sample_rate = struct.unpack_from("<I", plaintext)[0]
    return np.frombuffer(plaintext[4:], dtype="<i2").astype(np.float64) / 32768.0, sample_rate
