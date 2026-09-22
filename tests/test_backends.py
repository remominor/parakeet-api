import unittest
from types import SimpleNamespace

import numpy as np

from gateway.backends import TranscribeCppASR, model_identity, resolve_device


class FakeSession:
    def run(self, _pcm, **_kwargs):
        return SimpleNamespace(
            text="hello world", language="en",
            words=(SimpleNamespace(text="hello", t0_ms=0, t1_ms=500, first_token=0, n_tokens=2),),
            tokens=(SimpleNamespace(p=.9), SimpleNamespace(p=.7)),
            timings=SimpleNamespace(load_ms=1, mel_ms=2, encode_ms=3, decode_ms=4),
        )


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def test_model_identity(self):
        self.assertEqual(model_identity("parakeet-unified-en-0.6b-Q8_0.gguf"), "parakeet-unified-en-0.6b")
        self.assertEqual(model_identity("parakeet-tdt-0.6b-v2-Q8_0.gguf"), "parakeet-tdt-0.6b-v2")

    async def test_minimum_token_probability_is_word_confidence(self):
        backend = object.__new__(TranscribeCppASR)
        backend._lock = __import__("asyncio").Lock()
        backend._session = FakeSession()
        result = await backend.transcribe(np.zeros(16000, dtype=np.float32), 1.0)
        self.assertEqual(result.words[0].confidence, .7)
        self.assertEqual(result.native_timings["decode_ms"], 4)

    def test_device_selection_rejects_registry_index_and_supports_cuda_ordinal(self):
        devices = (
            SimpleNamespace(index=0, kind="cpu", name="CPU", device_id="cpu"),
            SimpleNamespace(index=7, kind="cuda", name="GPU A", device_id="0000:01:00.0"),
            SimpleNamespace(index=2, kind="cuda", name="GPU B", device_id="0000:02:00.0"),
        )
        module = SimpleNamespace(backends=lambda: devices)
        with self.assertRaisesRegex(RuntimeError, "bare numeric"):
            resolve_device(module, "1")
        self.assertIs(resolve_device(module, "cuda:1"), devices[2])
        self.assertIs(resolve_device(module, "0000:01:00.0"), devices[1])


if __name__ == "__main__":
    unittest.main()
