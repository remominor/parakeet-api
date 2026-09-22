import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from gateway.app import MODEL_ID, ModelManager
from gateway.manager import StackConfig


class ControlledBackend:
    device = "cpu"

    def __init__(self, *, warm_entered=None, warm_release=None, warm_error=None):
        self.warm_entered = warm_entered
        self.warm_release = warm_release
        self.warm_error = warm_error
        self.closed = 0

    async def warm(self):
        if self.warm_entered:
            self.warm_entered.set()
        if self.warm_release:
            await self.warm_release.wait()
        if self.warm_error:
            raise self.warm_error

    async def close(self):
        self.closed += 1


class FakeEmbedding(ControlledBackend):
    pass


def config(root: Path, *, diarization=False, identity=False):
    return StackConfig(root / "asr.gguf", root / "diar.gguf", root / "campp.onnx", root / "speakers", diarization_enabled=diarization, identity_enabled=identity)


class ModelManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.verify = patch("gateway.manager.verify_artifact")
        self.verify.start()

    async def asyncTearDown(self):
        self.verify.stop()
        self.temporary.cleanup()

    async def wait_thread_event(self, event):
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def test_unloaded_health_is_unavailable(self):
        manager = ModelManager()
        health = await manager.health()
        self.assertEqual(health["status"], "unavailable")
        self.assertEqual(health["model"], MODEL_ID)
        self.assertEqual(health["vram_allocated_mb"], 0)
        self.assertEqual(health["gpu_memory"], [])

    async def test_loaded_health_reports_process_total_per_gpu(self):
        manager = ModelManager(); manager.state = "loaded"; manager.asr = ControlledBackend(); manager.asr.device = "0000:01:00.0"
        async def metrics(): return {"gpu_memory": [{"uuid": "GPU-a", "used_mb": 1000}, {"uuid": "GPU-b", "used_mb": 42}], "total_mb": 1042}
        manager.gpu_metrics = metrics
        health = await manager.health()
        self.assertEqual(health["device"], "0000:01:00.0")
        self.assertEqual(health["vram_allocated_mb"], 1042)
        self.assertEqual(len(health["gpu_memory"]), 2)
        self.assertEqual(health["gpu_memory_scope"], "process_total")

    async def test_unloaded_model_cannot_admit_inference(self):
        manager = ModelManager()
        with self.assertRaises(HTTPException) as raised:
            async with manager.admit(): pass
        self.assertEqual(raised.exception.status_code, 503)

    async def test_diarization_device_defaults_to_asr_device(self):
        with patch.dict("os.environ", {"PARAKEET_MODELS_DIR": str(self.root), "PARAKEET_ASR_DEVICE": "0000:01:00.0"}, clear=False):
            loaded = StackConfig.load()
        self.assertEqual(loaded.asr_device, "0000:01:00.0")
        self.assertEqual(loaded.diarization_device, "0000:01:00.0")

    async def test_unload_while_asr_constructor_is_blocked(self):
        entered, release, made = threading.Event(), threading.Event(), []
        def factory(*_args, **_kwargs): entered.set(); release.wait(); made.append(ControlledBackend()); return made[-1]
        manager = ModelManager(config(self.root), asr_factory=factory)
        await manager.request_load(); await self.wait_thread_event(entered); await manager.request_unload(); unload = manager.unload_task
        self.assertFalse(unload.done()); release.set(); await unload
        self.assertEqual(manager.state, "unloaded"); self.assertIsNone(manager.asr); self.assertEqual(made[0].closed, 1)

    async def test_unload_while_asr_warmup_is_blocked(self):
        entered, release = asyncio.Event(), asyncio.Event(); backend = ControlledBackend(warm_entered=entered, warm_release=release)
        manager = ModelManager(config(self.root), asr_factory=lambda *_a, **_k: backend)
        await manager.request_load(); await entered.wait(); await manager.request_unload(); unload = manager.unload_task
        release.set(); await unload
        self.assertEqual((manager.state, manager.asr, backend.closed), ("unloaded", None, 1))

    async def test_unload_while_sortformer_constructor_is_blocked(self):
        entered, release, made = threading.Event(), threading.Event(), []
        def diarizer_factory(*_args, **_kwargs): entered.set(); release.wait(); made.append(ControlledBackend()); return made[-1]
        manager = ModelManager(config(self.root, diarization=True), asr_factory=lambda *_a, **_k: ControlledBackend(), diarizer_factory=diarizer_factory)
        await manager.request_load(); await self.wait_thread_event(entered); await manager.request_unload(); unload = manager.unload_task
        release.set(); await unload
        self.assertEqual(manager.state, "unloaded"); self.assertIsNone(manager.diarizer); self.assertEqual(made[0].closed, 1)

    async def test_unload_while_sortformer_warmup_is_blocked(self):
        entered, release = asyncio.Event(), asyncio.Event(); diarizer = ControlledBackend(warm_entered=entered, warm_release=release)
        manager = ModelManager(config(self.root, diarization=True), asr_factory=lambda *_a, **_k: ControlledBackend(), diarizer_factory=lambda *_a, **_k: diarizer)
        await manager.request_load(); await entered.wait(); await manager.request_unload(); unload = manager.unload_task
        release.set(); await unload
        self.assertEqual(manager.state, "unloaded"); self.assertIsNone(manager.diarizer); self.assertEqual(diarizer.closed, 1)

    async def test_load_requested_during_unload_reloads_after_drain(self):
        made = []
        def factory(*_a, **_k): made.append(ControlledBackend()); return made[-1]
        manager = ModelManager(config(self.root), asr_factory=factory)
        await manager.request_load(); await manager.load_task
        lease = manager.admit(); await lease.__aenter__(); await manager.request_unload(); unload = manager.unload_task
        self.assertEqual(await manager.request_load(), ("unloading", True)); self.assertFalse(unload.done())
        await lease.__aexit__(None, None, None); await unload; await manager.load_task
        self.assertEqual(manager.state, "loaded"); self.assertEqual(made[0].closed, 1); self.assertIs(manager.asr, made[1])
        await manager.shutdown()

    async def test_rapid_load_unload_load_does_not_leak(self):
        entered, release, made = asyncio.Event(), asyncio.Event(), []
        def factory(*_a, **_k):
            backend = ControlledBackend(warm_entered=entered if not made else None, warm_release=release if not made else None); made.append(backend); return backend
        manager = ModelManager(config(self.root), asr_factory=factory)
        await manager.request_load(); await entered.wait(); await manager.request_unload(); unload = manager.unload_task; await manager.request_load()
        release.set(); await unload; await manager.load_task
        self.assertEqual(manager.state, "loaded"); self.assertEqual(made[0].closed, 1); self.assertEqual(made[1].closed, 0)
        await manager.shutdown(); self.assertEqual(made[1].closed, 1)

    async def test_shutdown_during_load_closes_discarded_objects(self):
        entered, release = asyncio.Event(), asyncio.Event(); backend = ControlledBackend(warm_entered=entered, warm_release=release)
        manager = ModelManager(config(self.root), asr_factory=lambda *_a, **_k: backend)
        await manager.request_load(); await entered.wait(); shutdown = asyncio.create_task(manager.shutdown()); release.set(); await shutdown
        self.assertEqual(manager.state, "unloaded"); self.assertIsNone(manager.asr); self.assertEqual(backend.closed, 1)

    async def test_cancelled_load_can_be_unloaded_without_leak(self):
        entered, release = asyncio.Event(), asyncio.Event(); backend = ControlledBackend(warm_entered=entered, warm_release=release)
        manager = ModelManager(config(self.root), asr_factory=lambda *_a, **_k: backend)
        await manager.request_load(); await entered.wait(); manager.load_task.cancel()
        with self.assertRaises(asyncio.CancelledError): await manager.load_task
        await manager.request_unload(); await manager.unload_task
        self.assertEqual(manager.state, "unloaded"); self.assertIsNone(manager.asr); self.assertEqual(backend.closed, 1)

    async def test_optional_component_failure_closes_it_and_loads_asr(self):
        asr = ControlledBackend(); diarizer = ControlledBackend(warm_error=RuntimeError("broken"))
        manager = ModelManager(config(self.root, diarization=True), asr_factory=lambda *_a, **_k: asr, diarizer_factory=lambda *_a, **_k: diarizer)
        await manager.request_load(); await manager.load_task
        self.assertEqual(manager.state, "loaded"); self.assertIs(manager.asr, asr); self.assertIsNone(manager.diarizer)
        self.assertEqual(diarizer.closed, 1); self.assertEqual(manager.components["diarization"]["status"], "degraded")
        await manager.shutdown()

    async def test_unusable_speaker_store_degrades_identity_without_blocking_asr(self):
        asr, embedding = ControlledBackend(), FakeEmbedding()
        manager = ModelManager(config(self.root, identity=True), asr_factory=lambda *_a, **_k: asr, embedding_factory=lambda *_a, **_k: embedding)
        with patch("gateway.manager.SpeakerStore", side_effect=RuntimeError("speaker store /data/speakers is not writable by uid 10001; fix volume ownership to 10001:10001")):
            await manager.request_load(); await manager.load_task
        self.assertEqual(manager.state, "loaded")
        self.assertIs(manager.asr, asr)
        self.assertIsNone(manager.identity)
        self.assertEqual(manager.components["identity"]["status"], "degraded")
        self.assertEqual(embedding.closed, 1)
        await manager.shutdown()

    async def test_cpu_identity_is_reused_and_reported_after_reload(self):
        embeddings = []
        def embedding_factory(*_args, **_kwargs): embeddings.append(FakeEmbedding()); return embeddings[-1]
        manager = ModelManager(config(self.root, identity=True), asr_factory=lambda *_a, **_k: ControlledBackend(), embedding_factory=embedding_factory)
        await manager.request_load(); await manager.load_task; resident = manager.embedding
        await manager.request_unload(); await manager.unload_task; await manager.request_load(); await manager.load_task
        self.assertIs(manager.embedding, resident); self.assertEqual(len(embeddings), 1); self.assertEqual(manager.components["identity"]["status"], "ready")
        await manager.shutdown(); self.assertEqual(resident.closed, 1)


if __name__ == "__main__": unittest.main()
