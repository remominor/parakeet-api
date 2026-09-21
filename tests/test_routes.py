import unittest
import tempfile
from contextlib import asynccontextmanager
from unittest.mock import patch

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


class FakeEmbedding:
    async def embed(self, _pcm):
        return np.ones(512, dtype=np.float32) / np.sqrt(512)


class BrokenIdentity:
    async def identify(self, _pcm, _segments):
        raise RuntimeError("broken")


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

    def test_identity_failure_fails_open_with_component_degraded(self):
        self.client.app.state.model.diarizer = FakeDiarizer()
        self.client.app.state.model.identity = BrokenIdentity()
        response = self.client.post("/v1/audio/transcriptions", data={"speech_context": "full"}, files={"file": ("a.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        context = response.json()["speech_context"]
        self.assertEqual(context["status"], "degraded")
        self.assertEqual(context["components"]["identity"], "degraded")
        self.assertEqual(context["speaker_status"]["speaker_0"]["status"], "unavailable")

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
            self.assertEqual(self.client.delete("/v1/speakers/alice").status_code, 204)
            self.assertEqual(self.client.get("/v1/speakers/alice").status_code, 404)


if __name__ == "__main__":
    unittest.main()
