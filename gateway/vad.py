"""Stateful CPU-only Silero ONNX VAD for PCM16 WebSocket audio."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64


@dataclass
class VadEvent:
    kind: str
    audio: bytes | None = None


@dataclass
class StreamingVad:
    """Turn detector; each WebSocket connection owns one instance."""

    model_path: Path
    threshold: float = 0.5
    min_silence_ms: int = 350
    speech_pad_ms: int = 120
    max_utterance_ms: int = 8_000
    _session: ort.InferenceSession = field(init=False)
    _state: np.ndarray = field(init=False)
    _context: np.ndarray = field(init=False)
    _pending: bytearray = field(default_factory=bytearray, init=False)
    _pre_roll: bytearray = field(default_factory=bytearray, init=False)
    _turn: bytearray = field(default_factory=bytearray, init=False)
    _triggered: bool = field(default=False, init=False)
    _silence_samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._reset_model()

    @property
    def _pad_bytes(self) -> int:
        return self.speech_pad_ms * SAMPLE_RATE * 2 // 1000

    @property
    def _silence_limit(self) -> int:
        return self.min_silence_ms * SAMPLE_RATE // 1000

    @property
    def _max_samples(self) -> int:
        return self.max_utterance_ms * SAMPLE_RATE // 1000

    def _reset_model(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def _reset_turn(self) -> None:
        self._turn.clear()
        self._pre_roll.clear()
        self._triggered = False
        self._silence_samples = 0
        self._reset_model()

    def _probability(self, pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        audio = np.concatenate((self._context, samples))[None, :]
        output, self._state = self._session.run(
            None,
            {"input": audio, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)},
        )
        self._context = audio[0, -CONTEXT_SAMPLES:]
        return float(output[0, 0])

    def _append_pre_roll(self, pcm: bytes) -> None:
        self._pre_roll.extend(pcm)
        overflow = len(self._pre_roll) - self._pad_bytes
        if overflow > 0:
            del self._pre_roll[:overflow]

    def _finish(self, *, trim_silence: bool) -> bytes | None:
        if not self._triggered:
            return None
        audio = bytes(self._turn)
        if trim_silence and self._silence_samples:
            end = len(audio) - self._silence_samples * 2 + self._pad_bytes
            audio = audio[:max(0, min(len(audio), end))]
        self._reset_turn()
        return audio if audio else None

    def feed(self, data: bytes) -> list[VadEvent]:
        """Accept arbitrary PCM16 frame boundaries and emit VAD lifecycle events."""
        self._pending.extend(data)
        events: list[VadEvent] = []
        window_bytes = WINDOW_SAMPLES * 2
        while len(self._pending) >= window_bytes:
            window = bytes(self._pending[:window_bytes])
            del self._pending[:window_bytes]
            probability = self._probability(window)
            speaking = probability >= self.threshold
            if not self._triggered:
                if speaking:
                    self._triggered = True
                    self._turn.extend(self._pre_roll)
                    self._turn.extend(window)
                    self._silence_samples = 0
                    events.append(VadEvent("speech_started"))
                else:
                    self._append_pre_roll(window)
                continue

            self._turn.extend(window)
            if speaking:
                self._silence_samples = 0
            elif probability < max(self.threshold - 0.15, 0.01):
                self._silence_samples += WINDOW_SAMPLES

            if self._silence_samples >= self._silence_limit:
                audio = self._finish(trim_silence=True)
                if audio:
                    events.append(VadEvent("turn", audio))
            elif len(self._turn) // 2 >= self._max_samples:
                audio = self._finish(trim_silence=False)
                if audio:
                    events.append(VadEvent("turn", audio))
        return events

    def flush(self) -> bytes | None:
        """Return an in-progress voice turn for a client-requested commit."""
        self._pending.clear()
        return self._finish(trim_silence=True)
