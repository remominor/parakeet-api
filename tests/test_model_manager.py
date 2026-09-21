import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from gateway.app import MODEL_ID, ModelManager
from gateway.manager import StackConfig


class DeadProcess:
    pid = 1234
    returncode = 0


class FakeBackend:
    device = "cpu"
    async def warm(self): pass
    async def close(self): pass


class FakeEmbedding:
    async def close(self): pass


class ModelManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unloaded_health_is_unavailable(self):
        manager = ModelManager()

        self.assertEqual(await manager.health(), {
            "status": "unavailable",
            "model": MODEL_ID,
            "model_state": "unloaded",
            "device": None,
            "vram_allocated_mb": 0,
            "vram_reserved_mb": 0,
        })

    async def test_loaded_health_reports_engine_gpu_memory(self):
        manager = ModelManager()
        manager.state = "loaded"

        async def metrics():
            return "cuda:1", 1842

        manager.gpu_metrics = metrics
        self.assertEqual((await manager.health())["status"], "ready")
        self.assertEqual((await manager.health())["device"], "cuda:1")
        self.assertEqual((await manager.health())["vram_allocated_mb"], 1842)

    async def test_unloaded_model_cannot_admit_inference(self):
        manager = ModelManager()

        with self.assertRaises(HTTPException) as raised:
            async with manager.admit():
                pass
        self.assertEqual(raised.exception.status_code, 503)

    async def test_unload_of_an_already_stopped_engine_finishes(self):
        manager = ModelManager()
        manager.state = "loaded"
        manager.process = DeadProcess()

        state, pending = await manager.request_unload()
        self.assertEqual((state, pending), ("unloading", True))
        await manager.task
        self.assertEqual(manager.state, "unloaded")

    async def test_cpu_identity_is_reused_and_reported_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = StackConfig(root / "asr.gguf", root / "diar.gguf", root / "campp.onnx", root / "speakers", identity_enabled=True)
            embeddings = []
            def embedding_factory(*_args, **_kwargs):
                embeddings.append(FakeEmbedding())
                return embeddings[-1]
            manager = ModelManager(config, asr_factory=lambda *_args, **_kwargs: FakeBackend(), embedding_factory=embedding_factory)
            with patch("gateway.manager.verify_artifact"):
                await manager.request_load(); await manager.task
                resident = manager.embedding
                await manager.request_unload(); await manager.task
                await manager.request_load(); await manager.task
            self.assertIs(manager.embedding, resident)
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(manager.components["identity"]["status"], "ready")
            await manager.shutdown()


if __name__ == "__main__":
    unittest.main()
