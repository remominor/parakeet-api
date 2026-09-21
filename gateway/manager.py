"""Lifecycle manager for the in-process speech-intelligence stack."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from .artifacts import verify_artifact
from .backends import TranscribeCppASR, TranscribeCppDiarizer, model_identity
from .speakers import CampPlusONNX, IdentityService, SpeakerStore


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in {None, ""} else None


def _models_root() -> Path:
    configured = Path(os.getenv("PARAKEET_MODELS_DIR", "/models"))
    if configured.exists():
        return configured
    local = Path.cwd() / "models"
    return local if local.exists() else configured


@dataclass(frozen=True)
class StackConfig:
    asr_model: Path
    diarization_model: Path
    campp_model: Path
    speaker_store: Path
    diarization_enabled: bool = False
    identity_enabled: bool = False
    asr_device: str | None = None
    diarization_device: str | None = None
    native_threads: int = 0
    campp_intra_threads: int = 0
    campp_inter_threads: int = 1
    campp_concurrency: int = 1
    identity_threshold: float | None = None
    identity_margin: float | None = None
    identity_minimum_audio: float = 1.5

    @property
    def model_id(self) -> str:
        return model_identity(self.asr_model)

    @classmethod
    def load(cls) -> "StackConfig":
        root = _models_root()

        def resolve(value: str, component: str) -> Path:
            path = Path(value)
            if path.is_absolute():
                return path
            nested = root / component / path
            return nested if nested.exists() else root / path

        asr_name = os.getenv("PARAKEET_ASR_MODEL_FILE") or os.getenv("PARAKEET_MODEL_FILE") or "parakeet-unified-en-0.6b-Q8_0.gguf"
        return cls(
            asr_model=resolve(asr_name, "asr"),
            diarization_model=resolve(os.getenv("PARAKEET_DIARIZATION_MODEL_FILE", "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf"), "diarization"),
            campp_model=resolve(os.getenv("PARAKEET_CAMPP_MODEL_FILE", "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"), "campp"),
            speaker_store=Path(os.getenv("PARAKEET_SPEAKER_STORE", "/data/speakers")),
            diarization_enabled=env_bool("PARAKEET_DIARIZATION_ENABLED"),
            identity_enabled=env_bool("PARAKEET_IDENTITY_ENABLED"),
            asr_device=os.getenv("PARAKEET_ASR_DEVICE"),
            diarization_device=os.getenv("PARAKEET_DIARIZATION_DEVICE"),
            native_threads=int(os.getenv("PARAKEET_NATIVE_THREADS", "0")),
            campp_intra_threads=int(os.getenv("PARAKEET_CAMPP_INTRA_THREADS", "0")),
            campp_inter_threads=int(os.getenv("PARAKEET_CAMPP_INTER_THREADS", "1")),
            campp_concurrency=int(os.getenv("PARAKEET_CAMPP_CONCURRENCY", "1")),
            identity_threshold=env_optional_float("PARAKEET_IDENTITY_THRESHOLD"),
            identity_margin=env_optional_float("PARAKEET_IDENTITY_MARGIN"),
            identity_minimum_audio=float(os.getenv("PARAKEET_IDENTITY_MINIMUM_AUDIO_MS", "1500")) / 1000,
        )


class ModelManager:
    """Loads ASR first, drains work before unload, and retains CAM++ on CPU."""

    def __init__(
        self,
        config: StackConfig | None = None,
        *,
        asr_factory: Callable[..., Any] = TranscribeCppASR,
        diarizer_factory: Callable[..., Any] = TranscribeCppDiarizer,
        embedding_factory: Callable[..., Any] = CampPlusONNX,
    ):
        self.config = config or StackConfig.load()
        self.model_id = self.config.model_id
        self.state = "unloaded"
        self.lock = asyncio.Lock()
        self.idle = asyncio.Event()
        self.idle.set()
        self.admitted = 0
        self.task: asyncio.Task | None = None
        self.load_after_unload = False
        self.asr = None
        self.diarizer = None
        self.embedding = None
        self.identity: IdentityService | None = None
        self.components: dict[str, dict] = {}
        self._asr_factory = asr_factory
        self._diarizer_factory = diarizer_factory
        self._embedding_factory = embedding_factory
        # Kept as a no-op compatibility attribute for older operational tests.
        self.process = None
        self.telemetry_at = 0.0
        self.telemetry: tuple[str | None, int | None] = (None, 0)

    async def is_loaded(self) -> bool:
        async with self.lock:
            return self.state == "loaded"

    async def request_load(self, _app=None) -> tuple[str, bool]:
        async with self.lock:
            if self.state == "loaded":
                return self.state, False
            if self.state == "loading":
                return self.state, True
            if self.state == "unloading":
                self.load_after_unload = True
                return self.state, True
            self.state = "loading"
            self.task = asyncio.create_task(self._load())
            return self.state, True

    async def request_unload(self) -> tuple[str, bool]:
        async with self.lock:
            if self.state == "unloaded":
                return self.state, False
            if self.state == "unloading":
                return self.state, True
            self.state = "unloading"
            self.task = asyncio.create_task(self._unload())
            return self.state, True

    @asynccontextmanager
    async def admit(self):
        async with self.lock:
            if self.state != "loaded":
                raise HTTPException(503, "model_unavailable")
            self.admitted += 1
            self.idle.clear()
        try:
            yield
        finally:
            async with self.lock:
                self.admitted -= 1
                if not self.admitted:
                    self.idle.set()

    async def _load(self) -> None:
        asr = diarizer = None
        try:
            self.components = {"asr": {"status": "loading", "required": True}}
            await asyncio.to_thread(verify_artifact, self.config.asr_model, allow_unpinned=env_bool("PARAKEET_ALLOW_UNPINNED_MODELS"))
            asr = await asyncio.to_thread(
                self._asr_factory,
                self.config.asr_model,
                device_selector=self.config.asr_device,
                threads=self.config.native_threads,
            )
            if hasattr(asr, "warm"):
                await asr.warm()
            self.asr = asr
            self.components["asr"] = {"status": "ready", "required": True, "device": getattr(asr, "device", None), "vram_mb": None}

            if self.config.diarization_enabled:
                self.components["diarization"] = {"status": "loading", "required": False}
                try:
                    await asyncio.to_thread(verify_artifact, self.config.diarization_model, allow_unpinned=env_bool("PARAKEET_ALLOW_UNPINNED_MODELS"))
                    diarizer = await asyncio.to_thread(
                        self._diarizer_factory,
                        self.config.diarization_model,
                        device_selector=self.config.diarization_device,
                        threads=self.config.native_threads,
                    )
                    if hasattr(diarizer, "warm"):
                        await diarizer.warm()
                    self.diarizer = diarizer
                    self.components["diarization"] = {"status": "ready", "required": False, "device": getattr(diarizer, "device", None), "vram_mb": None}
                except Exception as exc:
                    self.components["diarization"] = {"status": "degraded", "required": False, "error": str(exc), "vram_mb": None}

            if self.config.identity_enabled:
                self.components["identity"] = {"status": "loading", "required": False, "device": "cpu"}
                try:
                    # CAM++ deliberately remains resident across GPU unloads.
                    # Reuse it and restore component reporting on reload.
                    if self.embedding is None:
                        self.embedding = await asyncio.to_thread(
                            self._embedding_factory,
                            self.config.campp_model,
                            intra_threads=self.config.campp_intra_threads,
                            inter_threads=self.config.campp_inter_threads,
                            concurrency=self.config.campp_concurrency,
                        )
                    if self.identity is None:
                        store = SpeakerStore(self.config.speaker_store)
                        self.identity = IdentityService(
                            self.embedding, store,
                            threshold=self.config.identity_threshold,
                            margin=self.config.identity_margin,
                            minimum_audio=self.config.identity_minimum_audio,
                        )
                    self.components["identity"] = {"status": "ready", "required": False, "device": "cpu", "vram_mb": 0}
                except Exception as exc:
                    self.components["identity"] = {"status": "degraded", "required": False, "device": "cpu", "error": str(exc), "vram_mb": 0}

            async with self.lock:
                if self.state == "loading":
                    self.state = "loaded"
        except asyncio.CancelledError:
            if diarizer:
                await diarizer.close()
            if asr:
                await asr.close()
            raise
        except Exception as exc:
            self.components["asr"] = {"status": "error", "required": True, "error": str(exc)}
            if diarizer:
                await diarizer.close()
            if asr:
                await asr.close()
            self.asr = self.diarizer = None
            async with self.lock:
                if self.state == "loading":
                    self.state = "error"

    async def _unload(self) -> None:
        await self.idle.wait()
        diarizer, asr = self.diarizer, self.asr
        self.diarizer = self.asr = None
        if diarizer is not None:
            await diarizer.close()
        if asr is not None:
            await asr.close()
        follow_up = self.load_after_unload
        self.load_after_unload = False
        async with self.lock:
            if self.state == "unloading":
                self.state = "unloaded"
        if self.components:
            self.components["asr"] = {"status": "unloaded", "required": True}
            if "diarization" in self.components:
                self.components["diarization"] = {"status": "unloaded", "required": False}
        if follow_up:
            await self.request_load()

    async def shutdown(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.diarizer is not None:
            await self.diarizer.close()
        if self.asr is not None:
            await self.asr.close()
        if self.embedding is not None:
            await self.embedding.close()
        self.diarizer = self.asr = self.embedding = self.identity = None
        self.state = "unloaded"

    async def gpu_metrics(self) -> tuple[str | None, int | None]:
        if self.state != "loaded":
            return None, 0
        if time.monotonic() - self.telemetry_at < 1:
            return self.telemetry
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=0.75)
            rows = [line.split(",") for line in stdout.decode().splitlines()]
            matches = [(parts[1].strip(), int(float(parts[2]))) for parts in rows if len(parts) == 3 and parts[0].strip() == str(os.getpid())]
            used = sum(value for _, value in matches) if matches else None
            device = getattr(self.asr, "device", None)
            self.telemetry = (device, used)
        except (FileNotFoundError, TimeoutError, ValueError):
            self.telemetry = (getattr(self.asr, "device", None), None)
        self.telemetry_at = time.monotonic()
        return self.telemetry

    async def health(self) -> dict[str, Any]:
        async with self.lock:
            state = self.state
        device, used = await self.gpu_metrics()
        body: dict[str, Any] = {
            "status": "ready" if state == "loaded" else "unavailable",
            "model": self.model_id,
            "model_state": state,
            "device": device,
            "vram_allocated_mb": used,
            "vram_reserved_mb": used,
        }
        if self.components:
            body["components"] = self.components
        return body
