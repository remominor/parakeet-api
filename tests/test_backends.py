import asyncio
import time
import unittest
from types import SimpleNamespace

import numpy as np

from gateway.backends import TranscribeCppASR, TranscribeCppDiarizer, model_identity, resolve_device


class FakeSession:
    def run(self, _pcm, **_kwargs):
        return SimpleNamespace(
            text="hello world", language="en",
            words=(SimpleNamespace(text="hello", t0_ms=0, t1_ms=500, first_token=0, n_tokens=2),),
            tokens=(SimpleNamespace(p=.9), SimpleNamespace(p=.7)),
            timings=SimpleNamespace(load_ms=1, mel_ms=2, encode_ms=3, decode_ms=4),
        )


class FakeDiarSession:
    def __init__(self, native_id=1): self.native_id = native_id; self.closed = 0; self.calls = []
    def run(self, _pcm, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(speaker_segments=(SimpleNamespace(t0_ms=0, t1_ms=1000, speaker_id=self.native_id, p=.8),))
    def close(self): self.closed += 1


class FakeDiarModel:
    def __init__(self): self.created = []; self.closed = 0
    def session(self, **_kwargs):
        value = FakeDiarSession(2)
        self.created.append(value)
        return value
    def close(self): self.closed += 1


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

    async def test_stateful_sessions_reuse_native_state_and_keep_native_slots(self):
        backend = object.__new__(TranscribeCppDiarizer)
        backend._lock = asyncio.Lock()
        backend._model = FakeDiarModel()
        backend._threads = 0
        backend._session = FakeDiarSession(1)
        backend._options = object()
        backend._stateful_options = object()
        backend._threads = 0
        backend._session_ttl_seconds = 3600
        backend._session_max = 2
        backend._speaker_sessions = {}
        stateless = await backend.diarize(np.zeros(8, dtype=np.float32))
        first = await backend.diarize(np.zeros(8, dtype=np.float32), session_id="home")
        second = await backend.diarize(np.zeros(8, dtype=np.float32), session_id="home")
        other = await backend.diarize(np.zeros(8, dtype=np.float32), session_id="other")
        self.assertEqual(stateless[0].speaker, "speaker_0")
        self.assertEqual(first[0].speaker, "speaker_1")
        self.assertEqual(first[0].speaker_slot, "speaker_1")
        self.assertEqual(len(backend._model.created), 2)
        self.assertEqual(len(backend._model.created[0].calls), 2)
        self.assertIsNot(first, second)
        self.assertEqual(other[0].native_speaker_id, 2)

    async def test_session_binding_rules_reset_and_ttl(self):
        backend = object.__new__(TranscribeCppDiarizer)
        backend._lock = asyncio.Lock()
        backend._model = FakeDiarModel()
        backend._threads = 0
        backend._session_ttl_seconds = .001
        backend._session_max = 2
        backend._speaker_sessions = {}
        state = await backend._get_stateful_session_locked("home")
        bindings = await backend.apply_identity_bindings("home", {"speaker_0": {"status": "known", "speaker_id": "remo"}})
        self.assertEqual(bindings["speaker_0"]["speaker_id"], "remo")
        # Moving a known identity to another slot makes the mapping one-to-one.
        bindings = await backend.apply_identity_bindings("home", {"speaker_1": {"status": "known", "speaker_id": "remo"}})
        self.assertNotIn("speaker_0", bindings)
        self.assertEqual(bindings["speaker_1"]["speaker_id"], "remo")
        await backend.apply_identity_bindings("home", {"speaker_1": {"status": "unknown"}})
        self.assertIn("speaker_1", await backend.get_identity_bindings("home"))
        self.assertTrue(await backend.reset_speaker_session("home"))
        self.assertEqual(state.native_session.closed, 1)
        state = await backend._get_stateful_session_locked("stale")
        state.last_used_at = time.monotonic() - 1
        await backend.expire_speaker_sessions()
        self.assertEqual(state.native_session.closed, 1)

    async def test_close_releases_all_persistent_sessions(self):
        backend = object.__new__(TranscribeCppDiarizer)
        backend._lock = asyncio.Lock(); backend._model = FakeDiarModel(); backend._threads = 0
        backend._session = FakeDiarSession(); backend._session_ttl_seconds = 3600; backend._session_max = 2; backend._speaker_sessions = {}
        first = await backend._get_stateful_session_locked("one")
        second = await backend._get_stateful_session_locked("two")
        await backend.close()
        self.assertEqual((first.native_session.closed, second.native_session.closed, backend._session.closed, backend._model.closed), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
