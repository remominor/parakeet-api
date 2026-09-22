"""OpenAI-compatible local speech-intelligence gateway."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from .audio import SAMPLE_RATE as TARGET_RATE, DecodedAudio, decode_audio, pcm16_audio, wav_bytes
from .backends import SpeakerSegment, TranscriptionResult
from .fusion import attribute_words, interval_statistics
from .manager import ModelManager, StackConfig
from .speakers import SPEAKER_ID_RE, match_embedding
from .vad import StreamingVad, VadEvent

logging.basicConfig(level=os.getenv("PARAKEET_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("parakeet-api")
VERSION = "2.0.0"
TRANSCRIBE_CPP_VERSION = "0.2.3"
TRANSCRIBE_CPP_COMMIT = "63a44d9239d610b3908e8a66b384924cd4a77217"
FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
SPEECH_CONTEXTS = {"none", "diarization", "full"}
SPEAKER_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SENTENCE_RE = re.compile(r"[.?!][\"')\]]*$")
STACK_CONFIG = StackConfig.load()
MODEL_ID = STACK_CONFIG.model_id


def env_list(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    keys: list[str]
    aliases: set[str]
    limit: int
    timeout: float
    cors: list[str]
    log_text: bool
    ws_vad_threshold: float
    ws_min_silence_ms: int
    ws_speech_pad_ms: int
    ws_max_utterance_ms: int
    ws_max_frame_bytes: int
    low_confidence: float
    segment_max_duration_ms: int
    segment_max_chars: int
    segment_pause_ms: int
    url_connect_timeout: float
    url_total_timeout: float
    url_allowed_hosts: set[str]
    url_allow_private: bool
    metrics_enabled: bool
    webui_enabled: bool

    @classmethod
    def load(cls):
        aliases = {MODEL_ID, "parakeet", "parakeet-en", "whisper-1", *env_list("PARAKEET_MODEL_ALIASES")}
        return cls(
            env_list("PARAKEET_API_KEYS"), aliases,
            int(os.getenv("PARAKEET_MAX_UPLOAD_MB", "64")) * 1048576,
            float(os.getenv("PARAKEET_UPSTREAM_TIMEOUT", "300")), env_list("PARAKEET_CORS_ORIGINS"),
            enabled("PARAKEET_LOG_TRANSCRIPTS"), float(os.getenv("PARAKEET_WS_VAD_THRESHOLD", "0.5")),
            int(os.getenv("PARAKEET_WS_MIN_SILENCE_MS", "350")), int(os.getenv("PARAKEET_WS_SPEECH_PAD_MS", "120")),
            int(os.getenv("PARAKEET_WS_MAX_UTTERANCE_MS", "8000")), int(os.getenv("PARAKEET_WS_MAX_FRAME_BYTES", "65536")),
            float(os.getenv("PARAKEET_LOW_CONFIDENCE_THRESHOLD", "0.70")), int(os.getenv("PARAKEET_SEGMENT_MAX_DURATION_MS", "6000")),
            int(os.getenv("PARAKEET_SEGMENT_MAX_CHARS", "100")), int(os.getenv("PARAKEET_SEGMENT_PAUSE_MS", "700")),
            float(os.getenv("PARAKEET_URL_CONNECT_TIMEOUT", "5")), float(os.getenv("PARAKEET_URL_TOTAL_TIMEOUT", "30")),
            {item.lower() for item in env_list("PARAKEET_URL_ALLOWED_HOSTS")}, enabled("PARAKEET_URL_ALLOW_PRIVATE"),
            enabled("PARAKEET_METRICS_ENABLED"), enabled("PARAKEET_WEBUI_ENABLED"),
        )


SETTINGS = Settings.load()


@dataclass
class Stats:
    requests_total: int = 0
    requests_failed: int = 0
    requests_active: int = 0
    requests_queued: int = 0
    transcoded_total: int = 0
    passthrough_total: int = 0
    audio_seconds_total: float = 0.0
    total_ms_sum: float = 0.0
    engine_ms_sum: float = 0.0
    diarization_total: int = 0
    diarization_failed: int = 0
    identity_total: int = 0
    identity_failed: int = 0
    websocket_connections_total: int = 0
    websocket_connections_active: int = 0
    websocket_turns_total: int = 0
    request_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    ws_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    engine_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    decode_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    diarization_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    identity_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))


STATS = Stats()


def credentials_valid(authz: str | None, key: str | None) -> bool:
    if not SETTINGS.keys:
        return True
    value = authz[7:].strip() if authz and authz.lower().startswith("bearer ") else key
    return bool(value) and any(hmac.compare_digest(value, expected) for expected in SETTINGS.keys)


def auth(authz: str | None, key: str | None) -> None:
    if not credentials_valid(authz, key):
        raise HTTPException(401, "missing bearer token" if not authz and not key else "invalid API key")


def valid_model(model: str | None) -> None:
    if model and model.strip() not in SETTINGS.aliases:
        raise HTTPException(404, f"unknown model {model!r}; available: {MODEL_ID}")


def validate_options(language: str | None, prompt: str | None, temperature: float | None) -> None:
    if language and language.strip().lower() not in {"en", "eng", "en-us"}:
        raise HTTPException(422, "The loaded Parakeet model supports English transcription only.")
    if prompt and prompt.strip():
        raise HTTPException(422, "prompt is not supported by the loaded Parakeet model.")
    if temperature is not None and temperature != 0:
        raise HTTPException(422, "non-zero temperature is not supported; Parakeet uses deterministic greedy decoding.")


def request_id(value: str | None) -> str:
    return value if value and RID_RE.fullmatch(value) else str(uuid.uuid4())


def confidence(result: dict[str, Any]) -> dict[str, float | int] | None:
    values = [float(word.get("conf", word.get("confidence"))) for word in result.get("words", []) if word.get("conf", word.get("confidence")) is not None]
    return {"mean": round(sum(values) / len(values), 3), "min": round(min(values), 3), "low_word_count": sum(value < SETTINGS.low_confidence for value in values)} if values else None


def make_segment(index: int, words: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": index, "start": float(words[0].get("start", 0)), "end": float(words[-1].get("end", 0)), "text": " ".join(str(item.get("word", "")).strip() for item in words).strip()}


def segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    words = result.get("words") or []
    if not words:
        return result.get("segments") or []
    output: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current:
            previous = current[-1]
            text = " ".join(str(item.get("word", "")).strip() for item in current + [word]).strip()
            duration_ms = (float(word.get("end", 0)) - float(current[0].get("start", 0))) * 1000
            pause_ms = (float(word.get("start", 0)) - float(previous.get("end", 0))) * 1000
            if pause_ms >= SETTINGS.segment_pause_ms or duration_ms > SETTINGS.segment_max_duration_ms or len(text) > SETTINGS.segment_max_chars:
                output.append(make_segment(len(output), current)); current = []
        current.append(word)
        if SENTENCE_RE.search(str(word.get("word", "")).strip()):
            output.append(make_segment(len(output), current)); current = []
    if current:
        output.append(make_segment(len(output), current))
    return output


def enrich(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("words"):
        result["segments"] = segments(result)
    return result


def stamp(seconds: float, comma: bool) -> str:
    milliseconds = int(round(seconds * 1000)); hours, milliseconds = divmod(milliseconds, 3600000); minutes, milliseconds = divmod(milliseconds, 60000); seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{',' if comma else '.'}{milliseconds:03d}"


def subtitle(result: dict[str, Any], kind: str) -> str:
    cues = segments(result) or [{"start": 0, "end": result.get("duration", 0), "text": result.get("text", "")}]
    output = ["WEBVTT\n"] if kind == "vtt" else []
    for index, cue in enumerate(cues, 1):
        if kind == "srt":
            output.append(str(index))
        output += [f"{stamp(float(cue['start']), kind == 'srt')} --> {stamp(float(cue['end']), kind == 'srt')}", str(cue["text"]), ""]
    return "\n".join(output)


async def read_upload(file: UploadFile) -> bytes:
    chunks, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > SETTINGS.limit:
            raise HTTPException(413, f"file exceeds {SETTINGS.limit // 1048576} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def validate_url(value: str) -> None:
    try:
        parsed = urlparse(value); host = (parsed.hostname or "").lower()
    except ValueError as exc:
        raise HTTPException(422, "malformed audio_url") from exc
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise HTTPException(422, "audio_url must be a valid http or https URL without credentials")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise HTTPException(422, "audio_url host could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified) and not (SETTINGS.url_allow_private or host in SETTINGS.url_allowed_hosts):
            raise HTTPException(422, "audio_url resolves to a private or local address")


async def download_audio(url: str) -> bytes:
    validate_url(url)
    timeout = httpx.Timeout(SETTINGS.url_total_timeout, connect=SETTINGS.url_connect_timeout)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream("GET", url, headers={"Accept": "audio/*,application/octet-stream"}) as response:
                if 300 <= response.status_code < 400:
                    raise HTTPException(422, "audio_url redirects are not allowed")
                if response.status_code != 200:
                    raise HTTPException(422, f"audio_url returned HTTP {response.status_code}")
                length = response.headers.get("content-length")
                if length and int(length) > SETTINGS.limit:
                    raise HTTPException(413, f"audio_url exceeds {SETTINGS.limit // 1048576} MB limit")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > SETTINGS.limit:
                        raise HTTPException(413, f"audio_url exceeds {SETTINGS.limit // 1048576} MB limit")
                    chunks.append(chunk)
                return b"".join(chunks)
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(422, f"could not download audio_url: {exc.__class__.__name__}") from exc


async def get_audio(file: UploadFile | None, audio_url: str | None) -> DecodedAudio:
    if bool(file and file.filename) == bool(audio_url and audio_url.strip()):
        raise HTTPException(400, "provide exactly one of file or audio_url")
    data = await (read_upload(file) if file and file.filename else download_audio(audio_url.strip()))
    if not data:
        raise HTTPException(400, "empty upload")
    started = time.perf_counter()
    try:
        decoded = await asyncio.to_thread(decode_audio, data)
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc
    finally:
        STATS.decode_ms.append((time.perf_counter() - started) * 1000)
    STATS.transcoded_total += int(decoded.transcoded); STATS.passthrough_total += int(not decoded.transcoded)
    return decoded


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started = time.monotonic()
    app.state.model = ModelManager(STACK_CONFIG)
    await app.state.model.request_load()
    try:
        yield
    finally:
        await app.state.model.shutdown()


async def engine_transcribe(app: FastAPI, payload: bytes | DecodedAudio | np.ndarray, form: dict[str, Any] | None = None, *, parse_json: bool = True, admitted: bool = False) -> tuple[dict[str, Any] | str, float]:
    if isinstance(payload, bytes):
        decoded = await asyncio.to_thread(decode_audio, payload)
    elif isinstance(payload, DecodedAudio):
        decoded = payload
    else:
        pcm = np.asarray(payload, dtype=np.float32); decoded = DecodedAudio(pcm, len(pcm) / TARGET_RATE, False)
    STATS.requests_queued += 1
    queued = True
    try:
        async def execute():
            nonlocal queued
            STATS.requests_queued -= 1; queued = False
            started = time.perf_counter()
            result: TranscriptionResult = await app.state.model.asr.transcribe(decoded.pcm, decoded.duration)
            return result, (time.perf_counter() - started) * 1000
        if admitted:
            result, engine_ms = await execute()
        else:
            async with app.state.model.admit():
                result, engine_ms = await execute()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"speech engine unavailable: {exc.__class__.__name__}: {exc}") from exc
    finally:
        if queued:
            STATS.requests_queued -= 1
    if not parse_json:
        return result.text, engine_ms
    needs_words = not form or form.get("response_format") == "verbose_json" or "word" in form.get("timestamp_granularities[]", [])
    return result.as_legacy(include_words=needs_words), engine_ms


app = FastAPI(title="Parakeet OpenAI API", version=VERSION, lifespan=lifespan)
if SETTINGS.cors:
    app.add_middleware(CORSMiddleware, allow_origins=SETTINGS.cors, allow_credentials=False, allow_methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"])


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    request.state.request_id = request_id(request.headers.get("x-request-id")); response = await call_next(request)
    if request.url.path.startswith("/v1/audio/"):
        response.headers.setdefault("X-Request-ID", request.state.request_id)
    return response


@app.exception_handler(HTTPException)
async def errors(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": str(exc.detail), "type": "invalid_request_error" if exc.status_code < 500 else "server_error", "code": exc.status_code}})


@app.get("/health")
async def health(request: Request):
    body = await request.app.state.model.health(); return JSONResponse(status_code=200 if body["model_state"] == "loaded" else 503, content=body)


@app.get("/readyz")
async def readyz(request: Request):
    body = await request.app.state.model.health(); body["ready"] = body["model_state"] == "loaded"; return JSONResponse(status_code=200 if body["ready"] else 503, content=body)


async def model_lifecycle(request: Request, action: str, authorization: str | None, x_api_key: str | None):
    auth(authorization, x_api_key); state, pending = await (request.app.state.model.request_load() if action == "load" else request.app.state.model.request_unload())
    return JSONResponse(status_code=202 if pending else 200, content={"model": request.app.state.model.model_id, "model_state": state, "status": "accepted" if pending else "complete"})


@app.post("/internal/model/load")
@app.post("/v1/model/load")
async def load_model(request: Request, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    return await model_lifecycle(request, "load", authorization, x_api_key)


@app.post("/internal/model/unload")
@app.post("/v1/model/unload")
async def unload_model(request: Request, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    return await model_lifecycle(request, "unload", authorization, x_api_key)


@app.get("/v1/models")
async def models(authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "transcribe.cpp"}]}


def percentiles(values: deque[float]) -> dict[str, float | int]:
    data = sorted(values)
    def pick(quantile): return data[min(len(data) - 1, int(quantile * (len(data) - 1)))] if data else 0.0
    return {"samples": len(data), "p50": round(pick(0.5), 1), "p95": round(pick(0.95), 1), "p99": round(pick(0.99), 1)}


@app.get("/stats")
async def stats(authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key)
    return {"requests_total": STATS.requests_total, "requests_failed": STATS.requests_failed, "requests_active": STATS.requests_active, "requests_queued": STATS.requests_queued, "passthrough_total": STATS.passthrough_total, "transcoded_total": STATS.transcoded_total, "audio_seconds_total": round(STATS.audio_seconds_total, 2), "diarization_total": STATS.diarization_total, "diarization_failed": STATS.diarization_failed, "identity_total": STATS.identity_total, "identity_failed": STATS.identity_failed, "websocket_connections_total": STATS.websocket_connections_total, "websocket_connections_active": STATS.websocket_connections_active, "websocket_turns_total": STATS.websocket_turns_total, "http_latency_ms": percentiles(STATS.request_ms), "websocket_latency_ms": percentiles(STATS.ws_ms), "engine_ms": percentiles(STATS.engine_ms), "decode_ms": percentiles(STATS.decode_ms), "diarization_ms": percentiles(STATS.diarization_ms), "identity_ms": percentiles(STATS.identity_ms)}


@app.get("/info")
async def info(request: Request, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key)
    manager = request.app.state.model
    native = getattr(manager.asr, "native_identity", {}) if manager.asr else {}
    identity_ready = (
        getattr(manager, "identity", None) is not None
        and getattr(manager, "components", {}).get("identity", {}).get("status") == "ready"
    )
    diarization_ready = (
        getattr(manager, "diarizer", None) is not None
        and getattr(manager, "components", {}).get("diarization", {}).get("status") == "ready"
    )
    return {"service": "parakeet-api", "version": VERSION, "model": manager.model_id, "engine": "transcribe.cpp", "engine_version": TRANSCRIBE_CPP_VERSION, "engine_commit": TRANSCRIBE_CPP_COMMIT, "native": native, "language": ["en"], "uptime_seconds": round(time.monotonic() - request.app.state.started, 1), "confidence_semantics": "minimum native entropy-based token confidence per word", "capabilities": {"word_timestamps": True, "word_confidence": True, "segments": True, "srt": True, "vtt": True, "diarization": diarization_ready, "diarization_max_speakers": 4, "speaker_enrollment": identity_ready, "speaker_identification": identity_ready, "stateful_diarization_sessions": diarization_ready, "websocket_turn_endpointing": True, "realtime_transcription": True, "stateful_websocket_diarization": False, "partial_transcription": False, "translation": False, "prompt": False, "temperature_sampling": False}, "limits": {"max_upload_mb": SETTINGS.limit // 1048576, "websocket_max_frame_bytes": SETTINGS.ws_max_frame_bytes, "websocket_max_utterance_ms": SETTINGS.ws_max_utterance_ms, "websocket_completed_turn_queue": 2, "speaker_session_ttl_seconds": STACK_CONFIG.speaker_session_ttl_seconds, "speaker_session_max": STACK_CONFIG.speaker_session_max}}


@app.get("/metrics")
async def metrics():
    if not SETTINGS.metrics_enabled: raise HTTPException(404, "metrics endpoint is disabled")
    def summary(name, values):
        values_ms = percentiles(values); return [f"# TYPE {name} summary", f'{name}{{quantile="0.5"}} {values_ms["p50"] / 1000}', f'{name}{{quantile="0.95"}} {values_ms["p95"] / 1000}', f'{name}{{quantile="0.99"}} {values_ms["p99"] / 1000}', f"{name}_count {values_ms['samples']}"]
    lines = ["# TYPE parakeet_requests_total counter", f"parakeet_requests_total {STATS.requests_total}", f"parakeet_requests_failed_total {STATS.requests_failed}", "# TYPE parakeet_requests_active gauge", f"parakeet_requests_active {STATS.requests_active}", "# TYPE parakeet_requests_queued gauge", f"parakeet_requests_queued {STATS.requests_queued}", f"parakeet_audio_seconds_total {STATS.audio_seconds_total}", f"parakeet_transcoded_total {STATS.transcoded_total}", f"parakeet_passthrough_total {STATS.passthrough_total}", f"parakeet_diarization_total {STATS.diarization_total}", f"parakeet_diarization_failed_total {STATS.diarization_failed}", f"parakeet_identity_total {STATS.identity_total}", f"parakeet_identity_failed_total {STATS.identity_failed}", f"parakeet_ws_connections_active {STATS.websocket_connections_active}", f"parakeet_ws_turns_total {STATS.websocket_turns_total}"] + summary("parakeet_engine_duration_seconds", STATS.engine_ms) + summary("parakeet_request_duration_seconds", STATS.request_ms) + summary("parakeet_diarization_duration_seconds", STATS.diarization_ms) + summary("parakeet_identity_duration_seconds", STATS.identity_ms)
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/audio/translations")
async def translations():
    raise HTTPException(501, "translation is not supported; use /v1/audio/transcriptions")


def _semantic_diarization_response(diarization: list[SpeakerSegment], observations: dict[str, dict], bindings: dict[str, dict], identify: bool) -> tuple[dict[str, dict], dict[str, str]]:
    """Apply session semantic labels while preserving Sortformer slot metadata."""
    status: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for item in diarization:
        slot = item.speaker_slot or item.speaker
        binding = bindings.get(slot)
        public = binding.get("speaker_id") if binding else slot
        labels[slot] = public
        item.speaker_slot = slot
        item.speaker = public
        item.identity = dict(binding or (observations.get(slot, {"status": "unavailable"}) if identify else {"status": "unavailable"}))
        item.identity["speaker_slot"] = slot
        status[public] = dict(item.identity)
    return status, labels


def _semantic_statistics(statistics: dict, labels: dict[str, str]) -> dict:
    output = dict(statistics)
    output["dominant_speaker"] = labels.get(statistics.get("dominant_speaker"), statistics.get("dominant_speaker"))
    output["speaker_durations"] = {labels.get(name, name): value for name, value in statistics.get("speaker_durations", {}).items()}
    return output


async def diarize_audio(manager: ModelManager, decoded: DecodedAudio, *, identify: bool, session_id: str | None = None) -> dict:
    if manager.diarizer is None: raise HTTPException(503, "diarization_unavailable")
    STATS.diarization_total += 1; started = time.perf_counter()
    diarization_elapsed = 0.0; identity_elapsed = 0.0
    try:
        diarization: list[SpeakerSegment] = await manager.diarizer.diarize(decoded.pcm) if session_id is None else await manager.diarizer.diarize(decoded.pcm, session_id=session_id)
    except Exception as exc: STATS.diarization_failed += 1; raise HTTPException(502, f"diarization_failed: {exc.__class__.__name__}: {exc}") from exc
    finally: diarization_elapsed = (time.perf_counter() - started) * 1000; STATS.diarization_ms.append(diarization_elapsed)
    identities: dict[str, dict] = {}; identity_error = None; identity_component = "not_requested"
    if identify:
        STATS.identity_total += 1; identity_started = time.perf_counter()
        if manager.identity is None:
            identity_component = "unavailable"; identities = {speaker: {"status": "unavailable"} for speaker in dict.fromkeys(item.speaker for item in diarization)}; identity_error = {"component": "identity", "code": "unavailable", "message": "speaker identity is unavailable"}
        else:
            try: identities = await manager.identity.identify(decoded.pcm, diarization); identity_component = "ready"
            except Exception as exc:
                identity_component = "degraded"; STATS.identity_failed += 1; identities = {speaker: {"status": "unavailable"} for speaker in dict.fromkeys(item.speaker for item in diarization)}; identity_error = {"component": "identity", "code": "inference_failed", "message": f"{exc.__class__.__name__}: {exc}"}
        identity_elapsed = (time.perf_counter() - identity_started) * 1000; STATS.identity_ms.append(identity_elapsed)
    bindings: dict[str, dict] = {}
    if session_id is not None:
        apply = getattr(manager.diarizer, "apply_identity_bindings", None)
        generation = diarization[0].session_generation if diarization else None
        if identify and callable(apply): bindings = await apply(session_id, identities, generation)
        else:
            get_bindings = getattr(manager.diarizer, "get_identity_bindings", None)
            if callable(get_bindings): bindings = await get_bindings(session_id)
    raw_statistics = interval_statistics(diarization)
    speaker_status, labels = _semantic_diarization_response(diarization, identities, bindings, identify)
    body = {"segments": [item.as_dict() for item in diarization], "statistics": _semantic_statistics(raw_statistics, labels), "speaker_status": speaker_status, "components": {"diarization": "ready", "identity": identity_component}, "timings": {"diarization_ms": round(diarization_elapsed, 1), "identity_ms": round(identity_elapsed, 1)}}
    if session_id is not None: body["speaker_session_id"] = session_id
    if identity_error: body["errors"] = [identity_error]
    return body


async def add_speech_context(manager: ModelManager, result: dict, decoded: DecodedAudio, mode: str, session_id: str | None = None) -> None:
    started = time.perf_counter(); errors: list[dict] = []
    context: dict[str, Any] = {"requested": mode, "status": "complete", "segments": [], "speakers": [], "speaker_status": {}, "statistics": interval_statistics([]), "components": {"asr": "ready", "diarization": "unavailable", "identity": "not_requested"}, "timings": {"diarization_ms": 0.0, "identity_ms": 0.0}, "errors": errors}
    if session_id is not None: context["speaker_session_id"] = session_id
    try:
        detail = await diarize_audio(manager, decoded, identify=mode == "full", session_id=session_id)
        diarization = [SpeakerSegment(item["start"], item["end"], item.get("speaker_slot", item["speaker"]), item.get("confidence"), speaker_slot=item.get("speaker_slot")) for item in detail["segments"]]
        from .backends import Word
        normalized = [Word(str(item.get("word", "")), float(item.get("start", 0)), float(item.get("end", 0)), item.get("confidence", item.get("conf"))) for item in result.get("words", [])]
        attribute_words(normalized, diarization)
        labels = {item.get("speaker_slot", item["speaker"]): item["speaker"] for item in detail["segments"]}
        for public, word in zip(result.get("words", []), normalized):
            if word.speaker not in {None, "overlap", "unattributed"}:
                public["speaker_slot"] = word.speaker
                public["speaker"] = labels.get(word.speaker, word.speaker)
            else: public["speaker"] = word.speaker
        context["segments"] = detail["segments"]; context["statistics"] = detail["statistics"]; context["speaker_status"] = detail["speaker_status"]; context["speakers"] = [{"speaker": speaker, **status} for speaker, status in detail["speaker_status"].items()]; context["components"].update(detail["components"]); context["timings"].update(detail["timings"]); errors.extend(detail.get("errors", []))
        if errors: context["status"] = "degraded"
    except HTTPException as exc:
        context["status"] = "degraded"; context["errors"].append({"component": "diarization", "code": "unavailable" if exc.status_code == 503 else "inference_failed", "message": str(exc.detail)})
    except Exception:
        LOG.exception("unexpected optional speech-context failure")
        context["status"] = "degraded"; context["errors"].append({"component": "enrichment", "code": "unexpected_error", "message": "optional speech-context processing failed"})
    context["timings"]["total_enrichment_ms"] = round((time.perf_counter() - started) * 1000, 1); result["speech_context"] = context


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request, file: UploadFile | None = File(None), audio_url: str | None = Form(None), model: str | None = Form(None), response_format: str = Form("json"), language: str | None = Form(None), prompt: str | None = Form(None), temperature: float | None = Form(None), timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"), speech_context: str | None = Form(None), speaker_session_id: str | None = Form(None), authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key"), x_request_id: str | None = Header(None, alias="X-Request-ID")):
    auth(authorization, x_api_key); valid_model(model); validate_options(language, prompt, temperature)
    if not await request.app.state.model.is_loaded(): raise HTTPException(503, "model_unavailable")
    output = (response_format or "json").lower(); context_mode = (speech_context or "none").lower()
    if output not in FORMATS: raise HTTPException(400, f"response_format must be one of {', '.join(sorted(FORMATS))}")
    if context_mode not in SPEECH_CONTEXTS: raise HTTPException(422, "speech_context must be none, diarization, or full")
    if speaker_session_id and not SPEAKER_SESSION_ID_RE.fullmatch(speaker_session_id): raise HTTPException(422, "speaker_session_id must match [a-zA-Z0-9_-]{1,64}")
    if speaker_session_id and context_mode == "none": raise HTTPException(422, "speaker_session_id requires diarization or full speech_context")
    if context_mode != "none" and output not in {"json", "verbose_json"}: raise HTTPException(422, "speech_context enrichment requires json or verbose_json")
    decoded = await get_audio(file, audio_url); STATS.requests_total += 1; STATS.requests_active += 1; started = time.perf_counter()
    granularities = list(timestamp_granularities or []); needs_words = output in {"verbose_json", "srt", "vtt"} or "word" in granularities or context_mode != "none"
    if needs_words and "word" not in granularities: granularities.append("word")
    form = {"response_format": "verbose_json" if needs_words else output, "timestamp_granularities[]": granularities}; engine_ms = 0.0
    try:
        async with request.app.state.model.admit():
            result, engine_ms = await engine_transcribe(request.app, decoded, form, parse_json=output != "text", admitted=True)
            if context_mode != "none": assert isinstance(result, dict); await add_speech_context(request.app.state.model, result, decoded, context_mode, speaker_session_id)
    except HTTPException: STATS.requests_failed += 1; raise
    finally: STATS.requests_active -= 1
    total_ms = (time.perf_counter() - started) * 1000; STATS.request_ms.append(total_ms); STATS.engine_ms.append(engine_ms); STATS.total_ms_sum += total_ms; STATS.engine_ms_sum += engine_ms; STATS.audio_seconds_total += decoded.duration
    headers = {"X-Request-ID": request.state.request_id, "X-Parakeet-Model": request.app.state.model.model_id, "X-Parakeet-Engine-Ms": f"{engine_ms:.1f}", "X-Parakeet-Total-Ms": f"{total_ms:.1f}", "X-Parakeet-Transcoded": "1" if decoded.transcoded else "0"}
    if output == "text": return PlainTextResponse(result, headers=headers)
    assert isinstance(result, dict); LOG.info("request_id=%s total=%.1fms engine=%.1fms %s", request.state.request_id, total_ms, engine_ms, repr(result.get("text", "")) if SETTINGS.log_text else f"chars={len(result.get('text', ''))}")
    if output in {"verbose_json", "srt", "vtt"} or context_mode != "none": enrich(result)
    if output in {"srt", "vtt"}: return PlainTextResponse(subtitle(result, output), media_type="application/x-subrip" if output == "srt" else "text/vtt", headers=headers)
    return JSONResponse(result, headers=headers)


@app.post("/v1/audio/diarizations")
async def diarizations(request: Request, file: UploadFile | None = File(None), audio_url: str | None = Form(None), identify: bool = Form(False), speaker_session_id: str | None = Form(None), authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key)
    if not await request.app.state.model.is_loaded(): raise HTTPException(503, "model_unavailable")
    if speaker_session_id and not SPEAKER_SESSION_ID_RE.fullmatch(speaker_session_id): raise HTTPException(422, "speaker_session_id must match [a-zA-Z0-9_-]{1,64}")
    decoded = await get_audio(file, audio_url)
    async with request.app.state.model.admit(): body = await diarize_audio(request.app.state.model, decoded, identify=identify, session_id=speaker_session_id)
    body["duration"] = decoded.duration; return body


@app.delete("/v1/audio/diarization-sessions/{session_id}", status_code=204)
async def reset_diarization_session(request: Request, session_id: str, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key)
    if not SPEAKER_SESSION_ID_RE.fullmatch(session_id): raise HTTPException(422, "speaker_session_id must match [a-zA-Z0-9_-]{1,64}")
    reset = getattr(getattr(request.app.state.model, "diarizer", None), "reset_speaker_session", None)
    if not callable(reset): raise HTTPException(503, "diarization_unavailable")
    await reset(session_id)
    return Response(status_code=204)


def require_identity(request: Request):
    service = request.app.state.model.identity
    if service is None: raise HTTPException(503, "speaker_identity_unavailable")
    return service


@app.post("/v1/speakers/enroll", status_code=201)
async def enroll_speaker(request: Request, speaker_id: str = Form(...), display_name: str | None = Form(None), files: list[UploadFile] = File(...), authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); service = require_identity(request)
    if not SPEAKER_ID_RE.fullmatch(speaker_id): raise HTTPException(422, "speaker_id must match [a-z0-9][a-z0-9_-]{0,63}")
    if service.store.exists(speaker_id): raise HTTPException(409, "speaker_id already exists")
    embeddings = []
    for upload in files:
        data = await read_upload(upload)
        try: decoded = await asyncio.to_thread(decode_audio, data)
        except Exception as exc: raise HTTPException(400, f"could not decode enrollment audio: {exc}") from exc
        if decoded.duration < STACK_CONFIG.identity_minimum_audio or float(np.sqrt(np.mean(np.square(decoded.pcm)))) < 1e-4: raise HTTPException(422, "insufficient_audio")
        embeddings.append(await service.backend.embed(decoded.pcm[:15 * TARGET_RATE]))
    try: record = service.store.create(speaker_id, display_name, embeddings)
    except FileExistsError: raise HTTPException(409, "speaker_id already exists") from None
    return record.public()


@app.get("/v1/speakers")
async def list_speakers(request: Request, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); service = require_identity(request); return {"object": "list", "data": [record.public() for record in service.store.list()]}


@app.get("/v1/speakers/{speaker_id}")
async def get_speaker(request: Request, speaker_id: str, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); service = require_identity(request)
    try: return service.store.get(speaker_id).public()
    except (KeyError, ValueError): raise HTTPException(404, "speaker not found") from None


@app.delete("/v1/speakers/{speaker_id}", status_code=204)
async def delete_speaker(request: Request, speaker_id: str, authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); service = require_identity(request)
    try:
        if not service.store.delete(speaker_id): raise HTTPException(404, "speaker not found")
    except ValueError: raise HTTPException(404, "speaker not found") from None
    return Response(status_code=204)


@app.post("/v1/speakers/verify")
async def verify_speaker(request: Request, file: UploadFile = File(...), speaker_id: str | None = Form(None), authorization: str | None = Header(None), x_api_key: str | None = Header(None, alias="X-API-Key")):
    auth(authorization, x_api_key); service = require_identity(request); decoded = await get_audio(file, None)
    if decoded.duration < STACK_CONFIG.identity_minimum_audio or float(np.sqrt(np.mean(np.square(decoded.pcm)))) < 1e-4: return {"status": "insufficient_audio"}
    embedding = await service.backend.embed(decoded.pcm[:15 * TARGET_RATE])
    if speaker_id:
        try: service.store.get(speaker_id)
        except (KeyError, ValueError): raise HTTPException(404, "speaker not found") from None
    return match_embedding(embedding, service.store.list(), threshold=service.threshold, margin=service.margin, target=speaker_id)


def vad_values(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {"threshold", "min_silence_ms", "speech_pad_ms", "max_utterance_ms"}
    if not isinstance(value, dict) or set(value) - allowed: raise ValueError("unsupported VAD configuration field")
    output = {}
    for key, item in value.items():
        if key == "threshold" and isinstance(item, (int, float)) and 0.05 <= float(item) <= 0.95: output[key] = float(item)
        elif key == "min_silence_ms" and isinstance(item, int) and 50 <= item <= 5000: output[key] = item
        elif key == "speech_pad_ms" and isinstance(item, int) and 0 <= item <= 2000: output[key] = item
        elif key == "max_utterance_ms" and isinstance(item, int) and 250 <= item <= 60000: output[key] = item
        else: raise ValueError(f"invalid VAD {key}")
    return output


def realtime_error(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}}


def pcm24_to_16(resampler, data: bytes) -> bytes:
    if len(data) % 2: raise ValueError("audio must be PCM16")
    if not data: return b""
    import av
    frame = av.AudioFrame(format="s16", layout="mono", samples=len(data) // 2); frame.sample_rate = 24000; frame.planes[0].update(data)
    return b"".join(bytes(memoryview(output.planes[0])[:output.samples * 2]) for output in resampler.resample(frame))


@dataclass
class CompletedTurn:
    turn_id: str
    audio: bytes
    start_ms: float
    end_ms: float
    item_id: str | None = None


async def run_turn_worker(app: FastAPI, queue: asyncio.Queue[CompletedTurn], emit, session_id: str):
    while True:
        turn = await queue.get(); STATS.requests_total += 1; STATS.requests_active += 1; STATS.websocket_turns_total += 1; started = time.perf_counter()
        try: result, engine_ms = await engine_transcribe(app, pcm16_audio(turn.audio), {"response_format": "verbose_json", "timestamp_granularities[]": ["word"]})
        except HTTPException as exc: STATS.requests_failed += 1; await emit(realtime_error("transcription_failed", str(exc.detail)) | {"turn_id": turn.turn_id}); queue.task_done(); continue
        finally: STATS.requests_active -= 1
        assert isinstance(result, dict); total_ms = (time.perf_counter() - started) * 1000; STATS.ws_ms.append(total_ms); STATS.engine_ms.append(engine_ms); STATS.engine_ms_sum += engine_ms; STATS.total_ms_sum += total_ms; STATS.audio_seconds_total += len(turn.audio) / 32000
        await emit((turn, result, engine_ms, total_ms)); LOG.info("request_id=%s turn_id=%s total=%.1fms engine=%.1fms chars=%s", session_id, turn.turn_id, total_ms, engine_ms, len(result.get("text", ""))); queue.task_done()


async def run_vad_events(detector: StreamingVad, events: list[VadEvent], turn_state: dict[str, Any], queue: asyncio.Queue[CompletedTurn], emit, lifecycle):
    for event in events:
        if event.kind == "speech_started": turn_state["id"] = str(int(turn_state.get("next", 0)) + 1); turn_state["next"] = int(turn_state["id"]); await lifecycle("speech_started", turn_state["id"], event)
        elif event.kind == "speech_stopped" and turn_state.get("id"): await lifecycle("speech_stopped", turn_state["id"], event)
        elif event.kind == "turn" and event.audio and turn_state.get("id"):
            turn = CompletedTurn(turn_state["id"], event.audio, event.audio_start_ms or 0, event.audio_end_ms or 0, "item_" + uuid.uuid4().hex); event.item_id = turn.item_id; turn_state["id"] = None
            if queue.full(): await emit(realtime_error("turn_queue_full", "completed-turn queue is full; turn was discarded") | {"turn_id": turn.turn_id})
            else: await queue.put(turn); await lifecycle("committed", turn.turn_id, event)


def enabled_value(value: str | None) -> bool:
    return bool(value and value.lower() in {"1", "true", "yes", "on"})


@app.websocket("/v1/audio/transcriptions/ws")
async def stream_transcriptions(websocket: WebSocket):
    if not credentials_valid(websocket.headers.get("authorization"), websocket.query_params.get("api_key")): await websocket.close(code=1008, reason="invalid API key"); return
    if not await websocket.app.state.model.is_loaded(): await websocket.close(code=1013, reason="model unavailable"); return
    await websocket.accept(); STATS.websocket_connections_total += 1; STATS.websocket_connections_active += 1; verbose = enabled_value(websocket.query_params.get("verbose")); session_id = request_id(websocket.headers.get("x-request-id")); send_lock = asyncio.Lock()
    async def emit(value):
        async with send_lock:
            if isinstance(value, tuple):
                turn, result, engine_ms, total_ms = value; output = {"type": "final", "turn_id": turn.turn_id, "text": result.get("text", ""), "duration_ms": round(len(turn.audio) / 32, 1), "engine_ms": round(engine_ms, 1), "total_ms": round(total_ms, 1)}
                if aggregate := confidence(result): output["confidence"] = aggregate
                if verbose: output["words"] = result.get("words", [])
                await websocket.send_json(output)
            else: await websocket.send_json(value)
    async def lifecycle(kind, turn_id, event):
        if kind == "speech_started": await emit({"type": "speech_started", "turn_id": turn_id, "audio_start_ms": round(event.audio_start_ms or 0, 1)})
        elif kind == "speech_stopped": await emit({"type": "speech_stopped", "turn_id": turn_id, "audio_end_ms": round(event.audio_end_ms or 0, 1)})
    try:
        detector = await asyncio.to_thread(StreamingVad, Path(__file__).with_name("silero_vad.onnx"), SETTINGS.ws_vad_threshold, SETTINGS.ws_min_silence_ms, SETTINGS.ws_speech_pad_ms, SETTINGS.ws_max_utterance_ms); queue = asyncio.Queue(maxsize=2); state = {"next": 0, "id": None}; worker = asyncio.create_task(run_turn_worker(websocket.app, queue, emit, session_id)); await emit({"type": "ready", "sample_rate": TARGET_RATE, "model": websocket.app.state.model.model_id, "request_id": session_id})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect": return
            if message.get("text") is not None:
                try: control = json.loads(message["text"])
                except ValueError: await emit(realtime_error("invalid_control", "control message must be JSON")); continue
                if control.get("type") == "commit":
                    event = await asyncio.to_thread(detector.flush)
                    if event: await run_vad_events(detector, [VadEvent("speech_stopped", audio_end_ms=event.audio_end_ms), event], state, queue, emit, lifecycle)
                elif control.get("type") == "clear": await asyncio.to_thread(detector.clear); state["id"] = None
                elif control.get("type") == "config":
                    try: values = vad_values(control.get("vad")); await asyncio.to_thread(detector.configure, **values); await emit({"type": "config", "vad": values})
                    except ValueError as exc: await emit(realtime_error("invalid_vad_config", str(exc)))
                else: await emit(realtime_error("unsupported_control", "unsupported native WebSocket control event"))
                continue
            data = message.get("bytes")
            if not data or len(data) % 2 or len(data) > SETTINGS.ws_max_frame_bytes: await emit(realtime_error("invalid_audio_frame", "send non-empty PCM16 mono binary frames no larger than the configured limit")); await websocket.close(code=1003); return
            await run_vad_events(detector, await asyncio.to_thread(detector.feed, data), state, queue, emit, lifecycle)
    except WebSocketDisconnect: return
    except Exception as exc: LOG.exception("WebSocket VAD failure"); await emit(realtime_error("vad_unavailable", str(exc)))
    finally:
        if "worker" in locals(): worker.cancel()
        STATS.websocket_connections_active -= 1


@app.websocket("/v1/realtime")
async def realtime_transcriptions(websocket: WebSocket):
    if not credentials_valid(websocket.headers.get("authorization"), websocket.query_params.get("api_key")): await websocket.close(code=1008, reason="invalid API key"); return
    if websocket.query_params.get("intent", "transcription") != "transcription" or (websocket.query_params.get("model") and websocket.query_params.get("model") not in SETTINGS.aliases): await websocket.close(code=1008, reason="transcription-only realtime endpoint"); return
    if not await websocket.app.state.model.is_loaded(): await websocket.close(code=1013, reason="model unavailable"); return
    await websocket.accept(); STATS.websocket_connections_total += 1; STATS.websocket_connections_active += 1; session_id = "sess_" + uuid.uuid4().hex; send_lock = asyncio.Lock()
    session = {"id": session_id, "object": "realtime.transcription_session", "type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": websocket.app.state.model.model_id, "language": "en"}, "turn_detection": {"type": "server_vad", "threshold": SETTINGS.ws_vad_threshold, "silence_duration_ms": SETTINGS.ws_min_silence_ms, "prefix_padding_ms": SETTINGS.ws_speech_pad_ms}}}}
    async def emit(value):
        async with send_lock:
            if isinstance(value, tuple):
                turn, result, _, _ = value; item_id = turn.item_id or "item_" + uuid.uuid4().hex
                await websocket.send_json({"type": "conversation.item.created", "event_id": "event_" + uuid.uuid4().hex, "item": {"id": item_id, "type": "message", "role": "user", "status": "completed"}}); await websocket.send_json({"type": "conversation.item.input_audio_transcription.completed", "event_id": "event_" + uuid.uuid4().hex, "item_id": item_id, "content_index": 0, "transcript": result.get("text", ""), "usage": {"type": "duration", "seconds": round(len(turn.audio) / 32000, 3)}})
            else: await websocket.send_json(value)
    async def lifecycle(kind, _turn_id, event):
        if kind == "speech_started": await emit({"type": "input_audio_buffer.speech_started", "event_id": "event_" + uuid.uuid4().hex, "audio_start_ms": round(event.audio_start_ms or 0, 1)})
        elif kind == "speech_stopped": await emit({"type": "input_audio_buffer.speech_stopped", "event_id": "event_" + uuid.uuid4().hex, "audio_end_ms": round(event.audio_end_ms or 0, 1)})
        elif kind == "committed": await emit({"type": "input_audio_buffer.committed", "event_id": "event_" + uuid.uuid4().hex, "item_id": getattr(event, "item_id", None), "audio_start_ms": round(event.audio_start_ms or 0, 1), "audio_end_ms": round(event.audio_end_ms or 0, 1)})
    try:
        from av.audio.resampler import AudioResampler
        detector = await asyncio.to_thread(StreamingVad, Path(__file__).with_name("silero_vad.onnx"), SETTINGS.ws_vad_threshold, SETTINGS.ws_min_silence_ms, SETTINGS.ws_speech_pad_ms, SETTINGS.ws_max_utterance_ms); resampler = AudioResampler(format="s16", layout="mono", rate=16000); queue = asyncio.Queue(maxsize=2); state = {"next": 0, "id": None}; manual = bytearray(); worker = asyncio.create_task(run_turn_worker(websocket.app, queue, emit, session_id)); await emit({"type": "session.created", "event_id": "event_" + uuid.uuid4().hex, "session": session})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect": return
            if message.get("bytes") is not None: await emit(realtime_error("unsupported_event", "realtime endpoint accepts JSON events only")); continue
            try: event = json.loads(message.get("text") or "")
            except ValueError: await emit(realtime_error("invalid_event", "event must be JSON")); continue
            kind = event.get("type")
            if kind == "input_audio_buffer.append":
                if set(event) - {"type", "audio"}: await emit(realtime_error("unsupported_event", "unsupported append fields")); continue
                try: pcm = base64.b64decode(event.get("audio", ""), validate=True); pcm = pcm24_to_16(resampler, pcm)
                except (ValueError, binascii.Error): await emit(realtime_error("invalid_audio", "audio must be base64 PCM16 at 24 kHz")); continue
                if session["audio"]["input"]["turn_detection"] is None: manual.extend(pcm)
                else: await run_vad_events(detector, await asyncio.to_thread(detector.feed, pcm), state, queue, emit, lifecycle)
            elif kind == "input_audio_buffer.commit":
                if session["audio"]["input"]["turn_detection"] is None:
                    tail = b"".join(bytes(memoryview(output.planes[0])[:output.samples * 2]) for output in resampler.resample(None)); manual.extend(tail)
                    if manual:
                        state["next"] += 1; turn = CompletedTurn(str(state["next"]), bytes(manual), 0, len(manual) / 32, "item_" + uuid.uuid4().hex); manual.clear()
                        if queue.full(): await emit(realtime_error("turn_queue_full", "completed-turn queue is full; turn was discarded"))
                        else: await queue.put(turn); await emit({"type": "input_audio_buffer.committed", "event_id": "event_" + uuid.uuid4().hex, "item_id": turn.item_id, "audio_start_ms": 0, "audio_end_ms": round(turn.end_ms, 1)})
                else:
                    event_out = await asyncio.to_thread(detector.flush)
                    if event_out: await run_vad_events(detector, [VadEvent("speech_stopped", audio_end_ms=event_out.audio_end_ms), event_out], state, queue, emit, lifecycle)
            elif kind == "input_audio_buffer.clear": await asyncio.to_thread(detector.clear); manual.clear(); state["id"] = None
            elif kind == "session.update":
                update = event.get("session")
                try:
                    if not isinstance(update, dict) or set(update) - {"type", "audio"} or update.get("type", "transcription") != "transcription": raise ValueError("only transcription session.audio.input is supported")
                    audio = update.get("audio", {}); inp = audio.get("input") if isinstance(audio, dict) else None
                    if not isinstance(inp, dict) or set(audio) != {"input"} or set(inp) - {"format", "transcription", "turn_detection"}: raise ValueError("unsupported session audio configuration")
                    if "format" in inp and inp["format"] != {"type": "audio/pcm", "rate": 24000}: raise ValueError("only 24 kHz audio/pcm input is supported")
                    if "transcription" in inp:
                        transcription = inp["transcription"]
                        if not isinstance(transcription, dict) or set(transcription) - {"model", "language"}: raise ValueError("unsupported transcription configuration")
                        if transcription.get("model", MODEL_ID) not in SETTINGS.aliases or transcription.get("language", "en").lower() not in {"en", "eng", "en-us"}: raise ValueError("Parakeet supports English transcription only")
                    if "turn_detection" not in inp: raise ValueError("session.audio.input.turn_detection is required")
                    turn_detection = inp["turn_detection"]
                    if turn_detection is None: session["audio"]["input"]["turn_detection"] = None; manual.clear(); await asyncio.to_thread(detector.clear)
                    else:
                        mapping = {"threshold": "threshold", "silence_duration_ms": "min_silence_ms", "prefix_padding_ms": "speech_pad_ms"}
                        if not isinstance(turn_detection, dict) or turn_detection.get("type") != "server_vad" or set(turn_detection) - ({"type"} | set(mapping)): raise ValueError("only server_vad turn detection is supported")
                        values = vad_values({mapping[key]: value for key, value in turn_detection.items() if key in mapping}); await asyncio.to_thread(detector.configure, **values); current = session["audio"]["input"]["turn_detection"] or {"type": "server_vad"}; current.update(turn_detection); session["audio"]["input"]["turn_detection"] = current
                    await emit({"type": "session.updated", "event_id": "event_" + uuid.uuid4().hex, "session": session})
                except ValueError as exc: await emit(realtime_error("invalid_session_update", str(exc)))
            else: await emit(realtime_error("unsupported_event", "unsupported realtime event"))
    except WebSocketDisconnect: return
    except Exception as exc: LOG.exception("Realtime VAD failure"); await emit(realtime_error("realtime_unavailable", str(exc)))
    finally:
        if "worker" in locals(): worker.cancel()
        STATS.websocket_connections_active -= 1


@app.get("/", response_class=HTMLResponse)
async def webui():
    if not SETTINGS.webui_enabled: raise HTTPException(404, "web UI is disabled")
    return HTMLResponse(Path(__file__).with_name("webui.html").read_text())
