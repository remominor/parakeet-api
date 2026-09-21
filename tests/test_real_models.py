import os
import unittest

import numpy as np


class RealModelTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.getenv("PARAKEET_REAL_ASR_MODEL"), "set PARAKEET_REAL_ASR_MODEL for opt-in native tests")
    async def test_asr_load_and_run(self):
        from gateway.backends import TranscribeCppASR
        backend = TranscribeCppASR(os.environ["PARAKEET_REAL_ASR_MODEL"], device_selector=os.getenv("PARAKEET_ASR_DEVICE"))
        try:
            result = await backend.transcribe(np.zeros(16000, dtype=np.float32), 1.0)
            self.assertIsInstance(result.text, str)
        finally:
            await backend.close()

    @unittest.skipUnless(os.getenv("PARAKEET_REAL_DIARIZATION_MODEL"), "set PARAKEET_REAL_DIARIZATION_MODEL for opt-in native tests")
    async def test_diarizer_load_and_run(self):
        from gateway.backends import TranscribeCppDiarizer
        backend = TranscribeCppDiarizer(os.environ["PARAKEET_REAL_DIARIZATION_MODEL"], device_selector=os.getenv("PARAKEET_DIARIZATION_DEVICE"))
        try:
            result = await backend.diarize(np.zeros(32000, dtype=np.float32))
            self.assertIsInstance(result, list)
        finally:
            await backend.close()

    @unittest.skipUnless(os.getenv("PARAKEET_REAL_CAMPP_MODEL"), "set PARAKEET_REAL_CAMPP_MODEL for opt-in native tests")
    async def test_campp_load_and_run(self):
        from gateway.speakers import CampPlusONNX
        backend = CampPlusONNX(os.environ["PARAKEET_REAL_CAMPP_MODEL"])
        try:
            time = np.arange(32000, dtype=np.float32) / 16000
            embedding = await backend.embed((.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32))
            self.assertEqual(embedding.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
        finally:
            await backend.close()


if __name__ == "__main__":
    unittest.main()
