"""Lifecycle manager for the in-process speech-intelligence stack."""
from __future__ import annotations

import asyncio
import logging
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

LOG = logging.getLogger("parakeet-api")


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
            asr_device=os.getenv("PARAKEET_ASR_DEVICE") or None,
            diarization_device=os.getenv("PARAKEET_DIARIZATION_DEVICE") or os.getenv("PARAKEET_ASR_DEVICE") or None,
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
        self.task: asyncio.Task | None = None  # compatibility alias for the latest lifecycle task
        self.load_task: asyncio.Task | None = None
        self.unload_task: asyncio.Task | None = None
        self.generation = 0
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
        self.telemetry: dict[str, Any] = {"gpu_memory": [], "total_mb": 0}

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
            self.generation += 1
            generation = self.generation
            self.state = "loading"
            self.components = {"asr": {"status": "loading", "required": True}}
            self.load_task = asyncio.create_task(self._load(generation))
            self.task = self.load_task
            return self.state, True

    async def request_unload(self) -> tuple[str, bool]:
        async with self.lock:
            if self.state == "unloaded" and not (self.load_task and not self.load_task.done()):
                return self.state, False
            if self.state == "unloading":
                return self.state, True
            self.generation += 1
            self.state = "unloading"
            self.unload_task = asyncio.create_task(self._unload(self.load_task))
            self.task = self.unload_task
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

    @staticmethod
    async def _close(value) -> None:
        if value is not None:
            await value.close()

    async def _load(self, generation: int) -> None:
        asr = diarizer = None
        embedding = self.embedding
        identity = self.identity
        new_embedding = False
        components: dict[str, dict] = {"asr": {"status": "loading", "required": True}}
        try:
            await asyncio.to_thread(verify_artifact, self.config.asr_model, allow_unpinned=env_bool("PARAKEET_ALLOW_UNPINNED_MODELS"))
            asr = await asyncio.to_thread(
                self._asr_factory,
                self.config.asr_model,
                device_selector=self.config.asr_device,
                threads=self.config.native_threads,
            )
            if hasattr(asr, "warm"):
                await asr.warm()
            components["asr"] = {"status": "ready", "required": True, "device": getattr(asr, "device", None), "vram_mb": None}

            if self.config.diarization_enabled:
                components["diarization"] = {"status": "loading", "required": False}
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
                    components["diarization"] = {"status": "ready", "required": False, "device": getattr(diarizer, "device", None), "vram_mb": None}
                except Exception as exc:
                    await self._close(diarizer); diarizer = None
                    components["diarization"] = {"status": "degraded", "required": False, "error": str(exc), "vram_mb": None}

            if self.config.identity_enabled:
                components["identity"] = {"status": "loading", "required": False, "device": "cpu"}
                try:
                    # CAM++ deliberately remains resident across GPU unloads.
                    # Reuse it and restore component reporting on reload.
                    if embedding is None:
                        embedding = await asyncio.to_thread(
                            self._embedding_factory,
                            self.config.campp_model,
                            intra_threads=self.config.campp_intra_threads,
                            inter_threads=self.config.campp_inter_threads,
                            concurrency=self.config.campp_concurrency,
                        )
                        new_embedding = True
                    if identity is None:
                        store = SpeakerStore(self.config.speaker_store)
                        identity = IdentityService(
                            embedding, store,
                            threshold=self.config.identity_threshold,
                            margin=self.config.identity_margin,
                            minimum_audio=self.config.identity_minimum_audio,
                        )
                    components["identity"] = {"status": "ready", "required": False, "device": "cpu", "vram_mb": 0}
                except Exception as exc:
                    if new_embedding:
                        await self._close(embedding); embedding = None; identity = None; new_embedding = False
                    LOG.error("identity component degraded at startup: %s", exc)
                    components["identity"] = {"status": "degraded", "required": False, "device": "cpu", "error": str(exc), "vram_mb": 0}

            committed = False
            async with self.lock:
                if self.state == "loading" and self.generation == generation:
                    self.asr, self.diarizer = asr, diarizer
                    self.embedding, self.identity = embedding, identity
                    self.components = components
                    self.state = "loaded"
                    self.telemetry_at = 0.0
                    committed = True
            if committed:
                self._log_device("ASR", asr)
                if diarizer is not None:
                    self._log_device("Diarization", diarizer)
                return
            await self._close(diarizer)
            await self._close(asr)
            if new_embedding:
                await self._close(embedding)
        except asyncio.CancelledError:
            await self._close(diarizer); await self._close(asr)
            if new_embedding: await self._close(embedding)
            raise
        except Exception as exc:
            await self._close(diarizer); await self._close(asr)
            if new_embedding: await self._close(embedding)
            async with self.lock:
                if self.state == "loading" and self.generation == generation:
                    self.components = {"asr": {"status": "error", "required": True, "error": str(exc)}}
                    self.state = "error"

    @staticmethod
    def _log_device(component: str, backend) -> None:
        detail = getattr(backend, "device_description", None)
        if isinstance(detail, dict):
            LOG.info("%s device: kind=%s name=%s device_id=%s", component, detail.get("kind"), detail.get("name"), detail.get("device_id"))
        else:
            LOG.info("%s device: device_id=%s", component, getattr(backend, "device", None))

    async def _unload(self, pending_load: asyncio.Task | None = None) -> None:
        if pending_load is not None and not pending_load.done():
            try:
                await asyncio.shield(pending_load)
            except asyncio.CancelledError:
                if not pending_load.cancelled():
                    raise
            except Exception:
                pass
        await self.idle.wait()
        async with self.lock:
            diarizer, asr = self.diarizer, self.asr
            self.diarizer = self.asr = None
        await self._close(diarizer); await self._close(asr)
        async with self.lock:
            follow_up = self.load_after_unload
            self.load_after_unload = False
            self.state = "unloaded"
            self.telemetry_at = 0.0
            if self.components:
                self.components["asr"] = {"status": "unloaded", "required": True}
                if "diarization" in self.components:
                    self.components["diarization"] = {"status": "unloaded", "required": False}
        if follow_up:
            await self.request_load()

    async def shutdown(self) -> None:
        async with self.lock:
            self.generation += 1
            self.load_after_unload = False
            self.state = "unloading"
            pending_load = self.load_task
            if self.unload_task is None or self.unload_task.done():
                self.unload_task = asyncio.create_task(self._unload(pending_load))
            task = self.unload_task
        await task
        embedding = self.embedding
        self.embedding = self.identity = None
        await self._close(embedding)
        self.state = "unloaded"

    async def gpu_metrics(self) -> dict[str, Any]:
        if self.state != "loaded":
            return {"gpu_memory": [], "total_mb": 0}
        if time.monotonic() - self.telemetry_at < 1:
            return self.telemetry
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=0.75)
            rows = [line.split(",") for line in stdout.decode().splitlines()]
            matches = [{"uuid": parts[1].strip(), "used_mb": int(float(parts[2]))} for parts in rows if len(parts) == 3 and parts[0].strip() == str(os.getpid())]
            self.telemetry = {"gpu_memory": matches, "total_mb": sum(item["used_mb"] for item in matches) if matches else None}
        except (FileNotFoundError, TimeoutError, ValueError):
            self.telemetry = {"gpu_memory": [], "total_mb": None}
        self.telemetry_at = time.monotonic()
        return self.telemetry

    async def health(self) -> dict[str, Any]:
        async with self.lock:
            state = self.state
        telemetry = await self.gpu_metrics(); used = telemetry["total_mb"]
        body: dict[str, Any] = {
            "status": "ready" if state == "loaded" else "unavailable",
            "model": self.model_id,
            "model_state": state,
            "device": getattr(self.asr, "device", None),
            "vram_allocated_mb": used,
            "vram_reserved_mb": used,
            "gpu_memory": telemetry["gpu_memory"],
            "gpu_memory_scope": "process_total",
        }
        if self.components:
            body["components"] = self.components
        return body
