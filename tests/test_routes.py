import asyncio
import dataclasses
import threading
import unittest
import tempfile
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.audio import wav_bytes
import numpy as np

from gateway.backends import SpeakerSegment, TranscriptionResult, Word
from gateway.speakers import IdentityService, SpeakerStore


class FakeASR:
    async def transcribe(self, _pcm, duration):
        return TranscriptionResult("hello", duration, [Word("hello", 0, min(duration, .5), .8)])


class FakeDiarizer:
    async def diarize(self, _pcm):
        return [SpeakerSegment(0, 1, "speaker_0")]


class SessionDiarizer:
    def __init__(self): self.sessions = {}; self.reset_calls = []
    async def diarize(self, _pcm, session_id=None):
        self.sessions.setdefault(session_id, 0); self.sessions[session_id] += 1
        return [SpeakerSegment(0, 1, "speaker_1", speaker_slot="speaker_1", native_speaker_id=2)]
    async def apply_identity_bindings(self, session_id, observations, _generation=None):
        bindings = self.sessions.setdefault(("bindings", session_id), {})
        for slot, value in observations.items():
            if value.get("status") == "known": bindings[slot] = dict(value)
        return dict(bindings)
    async def get_identity_bindings(self, session_id): return dict(self.sessions.get(("bindings", session_id), {}))
    async def reset_speaker_session(self, session_id): self.reset_calls.append(session_id); return True


class FakeEmbedding:
    async def embed(self, _pcm):
        return np.ones(512, dtype=np.float32) / np.sqrt(512)


class BrokenIdentity:
    async def identify(self, _pcm, _segments):
        raise RuntimeError("broken")


class BlockingDiarizer:
    def __init__(self): self.entered = threading.Event(); self.release = threading.Event(); self.closed = False
    async def diarize(self, _pcm):
        self.entered.set(); await asyncio.to_thread(self.release.wait)
        if self.closed: raise RuntimeError("closed during request")
        return [SpeakerSegment(0, 1, "speaker_0")]


class LeaseManager:
    model_id = "parakeet-unified-en-0.6b"
    identity = None
    def __init__(self): self.asr = FakeASR(); self.diarizer = BlockingDiarizer(); self.active = 0; self.idle = threading.Event(); self.idle.set()
    async def is_loaded(self): return True
    @asynccontextmanager
    async def admit(self):
        self.active += 1; self.idle.clear()
        try: yield
        finally:
            self.active -= 1
            if not self.active: self.idle.set()
    async def unload(self):
        await asyncio.to_thread(self.idle.wait); self.diarizer.closed = True
    async def shutdown(self): pass


class FakeManager:
    model_id = "parakeet-unified-en-0.6b"
    asr = FakeASR()
    diarizer = None
    identity = None
    async def request_load(self): return "loading", True
    async def shutdown(self): pass
    async def is_loaded(self): return True
    @asynccontextmanager
    async def admit(self): yield
    async def health(self): return {"status": "ready", "model": self.model_id, "model_state": "loaded", "device": None, "vram_allocated_mb": None, "vram_reserved_mb": None}


class RouteTests(unittest.TestCase):
    def setUp(self):
        import gateway.app as module
        self.module = module
        self.patch = patch.object(module, "ModelManager", return_value=FakeManager())
        self.patch.start()
        self.client = TestClient(module.app)
        self.client.__enter__()
        self.audio = wav_bytes(b"\0\0" * 16000)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()

    def test_legacy_json_has_no_enrichment_fields(self):
        response = self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"text", "duration"})
        self.assertEqual(response.headers["x-parakeet-transcoded"], "0")

    def test_webui_contains_complete_manual_console(self):
        with patch.object(self.module, "SETTINGS", dataclasses.replace(self.module.SETTINGS, webui_enabled=True)):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        for marker in ("speech_context", "speaker_session_id", "Reset session", "/v1/audio/diarization-sessions/", "/v1/speakers/enroll", "/v1/speakers/verify", "Record enrollment sample", "Raw speech context", "sessionStorage", "candidate_score", "stream?.getTracks()"):
            self.assertIn(marker, response.text)

    def test_metrics_exposes_exact_queue_gauge(self):
        previous = self.module.STATS.requests_queued; self.module.STATS.requests_queued = 3
        try:
            with patch.object(self.module, "SETTINGS", dataclasses.replace(self.module.SETTINGS, metrics_enabled=True)):
                response = self.client.get("/metrics")
            self.assertIn("# TYPE parakeet_requests_queued gauge", response.text)
            self.assertIn("parakeet_requests_queued 3", response.text)
        finally:
            self.module.STATS.requests_queued = previous

    def test_ui_api_routes_return_structured_auth_error(self):
        with patch.object(self.module, "SETTINGS", dataclasses.replace(self.module.SETTINGS, keys=["secret"])):
            response = self.client.get("/v1/speakers")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], 401)
        self.assertIsInstance(response.json()["error"]["message"], str)

    def test_info_hides_identity_capabilities_when_component_is_degraded(self):
        manager = self.client.app.state.model
        manager.identity = object()
        manager.components = {"identity": {"status": "degraded", "required": False}}
        with patch.object(self.module, "STACK_CONFIG", dataclasses.replace(self.module.STACK_CONFIG, identity_enabled=True)):
            response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        capabilities = response.json()["capabilities"]
        self.assertFalse(capabilities["speaker_enrollment"])
        self.assertFalse(capabilities["speaker_identification"])

    def test_info_hides_diarization_session_capabilities_when_component_is_degraded(self):
        manager = self.client.app.state.model
        manager.diarizer = object()
        manager.components = {"diarization": {"status": "degraded", "required": False}}
        with patch.object(self.module, "STACK_CONFIG", dataclasses.replace(self.module.STACK_CONFIG, diarization_enabled=True)):
            response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        capabilities = response.json()["capabilities"]
        self.assertFalse(capabilities["diarization"])
        self.assertFalse(capabilities["stateful_diarization_sessions"])

    def test_enrichment_failure_fails_open(self):
        response = self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", self.audio, "audio/wav")}, data={"speech_context": "diarization"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["speech_context"]["status"], "degraded")
        self.assertIn("words", response.json())

    def test_enrichment_rejects_text(self):
        response = self.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", self.audio, "audio/wav")}, data={"speech_context": "full", "response_format": "text"})
        self.assertEqual(response.status_code, 422)

    def test_standalone_diarization_failure_is_503(self):
        response = self.client.post("/v1/audio/diarizations", files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 503)

    def test_standalone_diarization_success(self):
        self.client.app.state.model.diarizer = FakeDiarizer()
        response = self.client.post("/v1/audio/diarizations", files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["segments"][0]["speaker"], "speaker_0")
        self.assertEqual(response.json()["statistics"]["dominant_speaker"], "speaker_0")

    def test_session_id_validation_and_idempotent_reset(self):
        self.client.app.state.model.diarizer = SessionDiarizer()
        invalid = self.client.post("/v1/audio/diarizations", files={"file": ("a.wav", self.audio, "audio/wav")}, data={"speaker_session_id": "has space"})
        self.assertEqual(invalid.status_code, 422)
        response = self.client.post("/v1/audio/diarizations", files={"file": ("a.wav", self.audio, "audio/wav")}, data={"speaker_session_id": "household-main"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["segments"][0]["speaker_slot"], "speaker_1")
        reset = self.client.delete("/v1/audio/diarization-sessions/household-main")
        self.assertEqual(reset.status_code, 204)
        self.assertEqual(self.client.app.state.model.diarizer.reset_calls, ["household-main"])

    def test_session_binding_is_public_but_slot_remains_available(self):
        diarizer = SessionDiarizer(); self.client.app.state.model.diarizer = diarizer
        self.client.app.state.model.identity = SimpleNamespace(identify=lambda *_args: None)
        # Direct helper isolates semantic display policy from model implementation.
        segment = SpeakerSegment(0, 1, "speaker_1", speaker_slot="speaker_1", native_speaker_id=2)
        status, labels = self.module._semantic_diarization_response([segment], {"speaker_1": {"status": "unknown"}}, {"speaker_1": {"status": "known", "speaker_id": "remo", "display_name": "Remo", "score": .8}}, True)
        self.assertEqual(segment.speaker, "remo")
        self.assertEqual(segment.speaker_slot, "speaker_1")
        self.assertEqual(status["remo"]["status"], "known")
        self.assertEqual(labels["speaker_1"], "remo")

    def test_identity_failure_fails_open_with_component_degraded(self):
        self.client.app.state.model.diarizer = FakeDiarizer()
        self.client.app.state.model.identity = BrokenIdentity()
        response = self.client.post("/v1/audio/transcriptions", data={"speech_context": "full"}, files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        context = response.json()["speech_context"]
        self.assertEqual(context["status"], "degraded")
        self.assertEqual(context["components"]["identity"], "degraded")
        self.assertEqual(context["speaker_status"]["speaker_0"]["status"], "unavailable")

    def test_unexpected_fusion_failure_fails_open_without_internal_detail(self):
        self.client.app.state.model.diarizer = FakeDiarizer()
        with patch("gateway.app.attribute_words", side_effect=RuntimeError("private stack detail")):
            response = self.client.post("/v1/audio/transcriptions", data={"speech_context": "diarization"}, files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        context = response.json()["speech_context"]
        self.assertEqual(context["status"], "degraded")
        self.assertEqual(context["errors"][-1]["code"], "unexpected_error")
        self.assertNotIn("private stack detail", response.text)

    def test_full_enrichment_holds_admission_until_diarization_finishes(self):
        manager = LeaseManager(); self.client.app.state.model = manager; response_box = {}
        request_thread = threading.Thread(target=lambda: response_box.setdefault("response", self.client.post("/v1/audio/transcriptions", data={"speech_context": "full"}, files={"file": ("a.wav", self.audio, "audio/wav")})))
        request_thread.start(); self.assertTrue(manager.diarizer.entered.wait(2))
        unload_done = threading.Event()
        unload_thread = threading.Thread(target=lambda: (asyncio.run(manager.unload()), unload_done.set()))
        unload_thread.start(); self.assertFalse(unload_done.wait(.05)); self.assertFalse(manager.diarizer.closed)
        manager.diarizer.release.set(); request_thread.join(2); unload_thread.join(2)
        self.assertTrue(unload_done.is_set()); self.assertTrue(manager.diarizer.closed); self.assertEqual(response_box["response"].status_code, 200)

    def test_speaker_enrollment_privacy_duplicate_verify_and_delete(self):
        loud_audio = wav_bytes((1000).to_bytes(2, "little", signed=True) * 32000)
        with tempfile.TemporaryDirectory() as directory:
            service = IdentityService(FakeEmbedding(), SpeakerStore(directory), threshold=None, margin=None)
            self.client.app.state.model.identity = service
            files = [("files", ("a.wav", loud_audio, "audio/wav"))]
            created = self.client.post("/v1/speakers/enroll", data={"speaker_id": "alice", "display_name": "Alice"}, files=files)
            self.assertEqual(created.status_code, 201)
            self.assertNotIn("embedding", created.json())
            duplicate = self.client.post("/v1/speakers/enroll", data={"speaker_id": "alice"}, files=files)
            self.assertEqual(duplicate.status_code, 409)
            listed = self.client.get("/v1/speakers").json()["data"]
            self.assertEqual([item["speaker_id"] for item in listed], ["alice"])
            self.assertNotIn("embedding", listed[0])
            verified = self.client.post("/v1/speakers/verify", data={"speaker_id": "alice"}, files={"file": ("a.wav", loud_audio, "audio/wav")})
            self.assertEqual(verified.json()["status"], "calibration_required")
            self.assertEqual(verified.json()["speaker_id"], "alice")
            self.assertEqual(verified.json()["display_name"], "Alice")
            self.assertAlmostEqual(verified.json()["score"], 1.0)
            self.assertEqual(self.client.delete("/v1/speakers/alice").status_code, 204)
            self.assertEqual(self.client.get("/v1/speakers/alice").status_code, 404)


class QueueAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_admission_failure_decrements_queue_exactly_once(self):
        import gateway.app as module
        class RejectManager:
            @asynccontextmanager
            async def admit(self):
                raise HTTPException(503, "no admission")
                yield
        application = SimpleNamespace(state=SimpleNamespace(model=RejectManager()))
        before = module.STATS.requests_queued
        with self.assertRaises(HTTPException):
            await module.engine_transcribe(application, np.zeros(16000, dtype=np.float32))
        self.assertEqual(module.STATS.requests_queued, before)


if __name__ == "__main__":
    unittest.main()
