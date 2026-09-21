"""Shared audio decoding for HTTP, ASR, diarization, and speaker identity."""
from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class DecodedAudio:
    pcm: np.ndarray
    duration: float
    transcoded: bool


def wav_bytes(pcm16: bytes) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm16))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm16))
        + pcm16
    )


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _fast_pcm_wav(data: bytes) -> np.ndarray | None:
    """Return zero-copy-compatible PCM for the legacy WAV passthrough case."""
    if not is_wav(data):
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getframerate() != SAMPLE_RATE
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                return None
            frames = source.readframes(source.getnframes())
    except (EOFError, wave.Error):
        return None
    if not frames:
        raise ValueError("audio stream decoded to zero samples")
    return (np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0).copy()


def decode_audio(data: bytes) -> DecodedAudio:
    """Decode a supported container to mono, 16-kHz, contiguous float32 PCM."""
    pcm = _fast_pcm_wav(data)
    if pcm is not None:
        return DecodedAudio(pcm, len(pcm) / SAMPLE_RATE, False)

    import av
    from av.audio.resampler import AudioResampler

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise ValueError("file contains no audio stream")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                chunks.append(np.frombuffer(output.planes[0], dtype=np.float32, count=output.samples).copy())
        for output in resampler.resample(None):
            chunks.append(np.frombuffer(output.planes[0], dtype=np.float32, count=output.samples).copy())
    if not chunks:
        raise ValueError("audio stream decoded to zero samples")
    pcm = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    return DecodedAudio(pcm, len(pcm) / SAMPLE_RATE, True)


def pcm16_audio(pcm16: bytes) -> DecodedAudio:
    if len(pcm16) % 2:
        raise ValueError("PCM16 byte count must be even")
    pcm = (np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0).copy()
    return DecodedAudio(pcm, len(pcm) / SAMPLE_RATE, False)
