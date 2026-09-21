"""Native speech backends and normalized internal result types."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    def as_dict(self) -> dict:
        out = {"word": self.word, "start": self.start, "end": self.end}
        if self.confidence is not None:
            # Keep both names: legacy clients used conf; OpenAI-shaped clients use confidence.
            out["conf"] = self.confidence
            out["confidence"] = self.confidence
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    identity: dict | None = None

    def as_dict(self) -> dict:
        out = {"start": self.start, "end": self.end, "speaker": self.speaker, "confidence": self.confidence}
        if self.identity is not None:
            out["identity"] = self.identity
        return out


@dataclass
class TranscriptionResult:
    text: str
    duration: float
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    native_timings: dict[str, float] = field(default_factory=dict)

    def as_legacy(self, include_words: bool = True) -> dict:
        out = {"text": self.text, "duration": self.duration}
        if include_words:
            out["language"] = self.language or "en"
            out["words"] = [word.as_dict() for word in self.words]
        return out


class ASRBackend(Protocol):
    model_id: str
    device: str | None

    async def transcribe(self, pcm: np.ndarray, duration: float) -> TranscriptionResult: ...
    async def close(self) -> None: ...


class DiarizationBackend(Protocol):
    device: str | None

    async def diarize(self, pcm: np.ndarray) -> list[SpeakerSegment]: ...
    async def close(self) -> None: ...


class SpeakerEmbeddingBackend(Protocol):
    dimension: int

    async def embed(self, pcm: np.ndarray) -> np.ndarray: ...
    async def close(self) -> None: ...


def model_identity(path: str | Path) -> str:
    name = Path(path).name.lower()
    return "parakeet-tdt-0.6b-v2" if "tdt-0.6b-v2" in name else "parakeet-unified-en-0.6b"


def resolve_device(module, selector: str | None):
    if not selector or selector == "auto":
        return None
    devices = module.backends()
    if selector.isdigit():
        index = int(selector)
        return next((device for device in devices if device.index == index), None) or _missing_device(selector)
    lowered = selector.lower()
    return next(
        (
            device
            for device in devices
            if lowered in {str(device.device_id).lower(), device.name.lower(), device.kind.lower()}
        ),
        None,
    ) or _missing_device(selector)


def _missing_device(selector: str):
    raise RuntimeError(f"transcribe.cpp device {selector!r} is not available")


class TranscribeCppASR:
    """One persistent native model/session serialized behind an async lock."""

    def __init__(self, path: str | Path, *, device_selector: str | None = None, threads: int = 0):
        import transcribe_cpp

        self._module = transcribe_cpp
        self.model_id = model_identity(path)
        device = resolve_device(transcribe_cpp, device_selector)
        kwargs = {"device": device} if device is not None else {"backend": "auto"}
        self._model = transcribe_cpp.Model(str(path), **kwargs)
        self._session = self._model.session(n_threads=threads)
        native_device = self._model.device
        self.device = native_device.device_id or native_device.name or native_device.kind
        self._lock = asyncio.Lock()

    @property
    def native_identity(self) -> dict:
        return {
            "version": self._module.native_version(),
            "commit": self._module.native_commit(),
            "arch": self._model.arch,
            "variant": self._model.variant,
            "backend": self._model.backend,
        }

    async def transcribe(self, pcm: np.ndarray, duration: float) -> TranscriptionResult:
        async with self._lock:
            # Request token precision even though the gateway exposes words:
            # transcribe.cpp intentionally elides the token table for
            # timestamps="word", and token entropy is needed for confidence.
            native = await asyncio.to_thread(self._session.run, pcm, timestamps="token", language="en")
        words: list[Word] = []
        tokens = native.tokens
        for item in native.words:
            probabilities = [
                float(tokens[index].p)
                for index in range(item.first_token, min(item.first_token + item.n_tokens, len(tokens)))
                if math.isfinite(float(tokens[index].p))
            ]
            score = min(probabilities) if probabilities else None
            words.append(Word(item.text, item.t0_ms / 1000, item.t1_ms / 1000, score))
        timings = {
            "load_ms": float(native.timings.load_ms),
            "mel_ms": float(native.timings.mel_ms),
            "encode_ms": float(native.timings.encode_ms),
            "decode_ms": float(native.timings.decode_ms),
        }
        return TranscriptionResult(native.text, duration, words, native.language or "en", timings)

    async def warm(self) -> None:
        await self.transcribe(np.zeros(8_000, dtype=np.float32), 0.5)

    async def close(self) -> None:
        await asyncio.to_thread(self._session.close)
        await asyncio.to_thread(self._model.close)


class TranscribeCppDiarizer:
    def __init__(self, path: str | Path, *, device_selector: str | None = None, threads: int = 0):
        import transcribe_cpp

        self._module = transcribe_cpp
        device = resolve_device(transcribe_cpp, device_selector)
        kwargs = {"device": device} if device is not None else {"backend": "auto"}
        self._model = transcribe_cpp.Model(str(path), **kwargs)
        self._session = self._model.session(n_threads=threads)
        native_device = self._model.device
        self.device = native_device.device_id or native_device.name or native_device.kind
        self._options = transcribe_cpp.SortformerStreamOptions(preset="very_high_latency")
        self._lock = asyncio.Lock()

    async def diarize(self, pcm: np.ndarray) -> list[SpeakerSegment]:
        async with self._lock:
            # Sortformer has no transcript timestamp axis; AUTO resolves to its
            # speaker-segment output while explicit text segment timestamps are
            # correctly rejected by the native API.
            result = await asyncio.to_thread(self._session.run, pcm, timestamps="auto", family=self._options)
        labels: dict[int, str] = {}
        output: list[SpeakerSegment] = []
        for segment in sorted(result.speaker_segments, key=lambda value: (value.t0_ms, value.t1_ms)):
            if segment.speaker_id not in labels:
                labels[segment.speaker_id] = f"speaker_{len(labels)}"
            probability = float(segment.p)
            output.append(
                SpeakerSegment(
                    segment.t0_ms / 1000,
                    segment.t1_ms / 1000,
                    labels[segment.speaker_id],
                    probability if math.isfinite(probability) else None,
                )
            )
        return output

    async def warm(self) -> None:
        await self.diarize(np.zeros(8_000, dtype=np.float32))

    async def close(self) -> None:
        await asyncio.to_thread(self._session.close)
        await asyncio.to_thread(self._model.close)
