"""Native speech backends and normalized internal result types."""
from __future__ import annotations

import asyncio
import math
import time
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
    speaker_slot: str | None = None

    def as_dict(self) -> dict:
        out = {"word": self.word, "start": self.start, "end": self.end}
        if self.confidence is not None:
            # Keep both names: legacy clients used conf; OpenAI-shaped clients use confidence.
            out["conf"] = self.confidence
            out["confidence"] = self.confidence
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.speaker_slot is not None:
            out["speaker_slot"] = self.speaker_slot
        return out


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    identity: dict | None = None
    speaker_slot: str | None = None
    native_speaker_id: int | None = None

    def as_dict(self) -> dict:
        out = {"start": self.start, "end": self.end, "speaker": self.speaker, "confidence": self.confidence}
        if self.identity is not None:
            out["identity"] = self.identity
        if self.speaker_slot is not None:
            out["speaker_slot"] = self.speaker_slot
        if self.native_speaker_id is not None:
            out["native_speaker_id"] = self.native_speaker_id
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

    async def diarize(self, pcm: np.ndarray, session_id: str | None = None) -> list[SpeakerSegment]: ...
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
        raise RuntimeError("bare numeric device selectors are unstable; use a stable device_id or cuda:N")
    lowered = selector.lower()
    if lowered.startswith("cuda:") and lowered[5:].isdigit():
        cuda_devices = [device for device in devices if _device_kind(device) == "cuda"]
        index = int(lowered[5:])
        return cuda_devices[index] if index < len(cuda_devices) else _missing_device(selector)
    return next(
        (
            device
            for device in devices
            if lowered in {str(device.device_id).lower(), str(device.name).lower(), _device_kind(device)}
        ),
        None,
    ) or _missing_device(selector)


def _missing_device(selector: str):
    raise RuntimeError(f"transcribe.cpp device {selector!r} is not available")


def _device_kind(device) -> str:
    return str(getattr(device, "kind", getattr(device, "device_type", ""))).lower().split(".")[-1]


def device_description(device) -> dict[str, str | None]:
    return {"kind": _device_kind(device), "name": str(device.name), "device_id": str(device.device_id) if device.device_id else None}


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
        self.device_description = device_description(native_device)
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


@dataclass
class StatefulSpeakerSession:
    """Bounded, in-memory Sortformer state for one logical speaker session.

    This holds native AOSC/FIFO state and semantic bindings only.  It never
    stores PCM, transcripts, or prior result segments.
    """

    session_id: str
    native_session: object
    created_at: float
    last_used_at: float
    identity_bindings: dict[str, dict] = field(default_factory=dict)


class TranscribeCppDiarizer:
    def __init__(
        self,
        path: str | Path,
        *,
        device_selector: str | None = None,
        threads: int = 0,
        session_ttl_seconds: float = 3600,
        session_max: int = 16,
    ):
        import transcribe_cpp

        self._module = transcribe_cpp
        device = resolve_device(transcribe_cpp, device_selector)
        kwargs = {"device": device} if device is not None else {"backend": "auto"}
        self._model = transcribe_cpp.Model(str(path), **kwargs)
        self._session = self._model.session(n_threads=threads)
        native_device = self._model.device
        self.device = native_device.device_id or native_device.name or native_device.kind
        self.device_description = device_description(native_device)
        self._options = transcribe_cpp.SortformerStreamOptions(preset="very_high_latency")
        self._stateful_options = transcribe_cpp.SortformerStreamOptions(
            preset="very_high_latency", preserve_state=True
        )
        self._threads = threads
        self._session_ttl_seconds = max(0.0, float(session_ttl_seconds))
        self._session_max = max(1, int(session_max))
        self._speaker_sessions: dict[str, StatefulSpeakerSession] = {}
        self._lock = asyncio.Lock()

    @property
    def speaker_sessions_active(self) -> int:
        return len(self._speaker_sessions)

    async def _close_native_session(self, session: object) -> None:
        await asyncio.to_thread(session.close)

    async def _expire_sessions_locked(self, now: float) -> None:
        if self._session_ttl_seconds <= 0:
            expired = list(self._speaker_sessions.values())
            self._speaker_sessions.clear()
        else:
            expired = [
                value for value in self._speaker_sessions.values()
                if now - value.last_used_at >= self._session_ttl_seconds
            ]
            for value in expired:
                self._speaker_sessions.pop(value.session_id, None)
        for value in expired:
            await self._close_native_session(value.native_session)

    async def _get_stateful_session_locked(self, session_id: str) -> StatefulSpeakerSession:
        now = time.monotonic()
        await self._expire_sessions_locked(now)
        current = self._speaker_sessions.get(session_id)
        if current is not None:
            current.last_used_at = now
            return current
        if len(self._speaker_sessions) >= self._session_max:
            # Deterministic LRU eviction keeps the registry bounded even when
            # callers create many session IDs.
            oldest = min(self._speaker_sessions.values(), key=lambda value: value.last_used_at)
            self._speaker_sessions.pop(oldest.session_id, None)
            await self._close_native_session(oldest.native_session)
        native_session = await asyncio.to_thread(self._model.session, n_threads=self._threads)
        current = StatefulSpeakerSession(session_id, native_session, now, now)
        self._speaker_sessions[session_id] = current
        return current

    async def diarize(self, pcm: np.ndarray, session_id: str | None = None) -> list[SpeakerSegment]:
        async with self._lock:
            # Sortformer has no transcript timestamp axis; AUTO resolves to its
            # speaker-segment output while explicit text segment timestamps are
            # correctly rejected by the native API.
            if session_id is None:
                native_session, options = self._session, self._options
            else:
                state = await self._get_stateful_session_locked(session_id)
                native_session, options = state.native_session, self._stateful_options
            result = await asyncio.to_thread(native_session.run, pcm, timestamps="auto", family=options)
        output: list[SpeakerSegment] = []
        for segment in sorted(result.speaker_segments, key=lambda value: (value.t0_ms, value.t1_ms)):
            native_id = int(segment.speaker_id)
            # Sortformer slots are one-based.  Preserve them directly rather
            # than renumbering speakers by first arrival in each request.
            slot = f"speaker_{native_id - 1}"
            probability = float(segment.p)
            output.append(
                SpeakerSegment(
                    segment.t0_ms / 1000,
                    segment.t1_ms / 1000,
                    slot,
                    probability if math.isfinite(probability) else None,
                    speaker_slot=slot,
                    native_speaker_id=native_id,
                )
            )
        return output

    async def apply_identity_bindings(self, session_id: str, observations: dict[str, dict]) -> dict[str, dict]:
        """Apply only authoritative CAM++ observations and return bindings.

        A known result replaces the slot's current identity and moves that
        identity from any other slot.  All non-known outcomes intentionally
        leave established session identity untouched.
        """
        async with self._lock:
            state = self._speaker_sessions.get(session_id)
            if state is None:
                return {}
            state.last_used_at = time.monotonic()
            for slot, observation in observations.items():
                if observation.get("status") != "known" or not observation.get("speaker_id"):
                    continue
                speaker_id = observation["speaker_id"]
                for bound_slot, binding in list(state.identity_bindings.items()):
                    if bound_slot != slot and binding.get("speaker_id") == speaker_id:
                        del state.identity_bindings[bound_slot]
                state.identity_bindings[slot] = dict(observation)
            return {slot: dict(binding) for slot, binding in state.identity_bindings.items()}

    async def get_identity_bindings(self, session_id: str) -> dict[str, dict]:
        async with self._lock:
            state = self._speaker_sessions.get(session_id)
            if state is None:
                return {}
            state.last_used_at = time.monotonic()
            return {slot: dict(binding) for slot, binding in state.identity_bindings.items()}

    async def reset_speaker_session(self, session_id: str) -> bool:
        async with self._lock:
            state = self._speaker_sessions.pop(session_id, None)
            if state is None:
                return False
            await self._close_native_session(state.native_session)
            return True

    async def expire_speaker_sessions(self) -> None:
        async with self._lock:
            await self._expire_sessions_locked(time.monotonic())

    async def warm(self) -> None:
        await self.diarize(np.zeros(8_000, dtype=np.float32))

    async def close(self) -> None:
        async with self._lock:
            stateful = list(self._speaker_sessions.values())
            self._speaker_sessions.clear()
        for state in stateful:
            await self._close_native_session(state.native_session)
        await asyncio.to_thread(self._session.close)
        await asyncio.to_thread(self._model.close)
