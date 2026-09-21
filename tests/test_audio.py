import io
import unittest
import wave

import numpy as np

from gateway.audio import decode_audio, wav_bytes


class AudioTests(unittest.TestCase):
    def test_pcm16_mono_16k_is_passthrough(self):
        decoded = decode_audio(wav_bytes(b"\0\0" * 1600))
        self.assertFalse(decoded.transcoded)
        self.assertEqual(decoded.pcm.dtype, np.float32)
        self.assertAlmostEqual(decoded.duration, 0.1)

    def test_non_16k_wav_is_resampled(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(1); target.setsampwidth(2); target.setframerate(8000)
            target.writeframes(b"\0\0" * 8000)
        decoded = decode_audio(output.getvalue())
        self.assertTrue(decoded.transcoded)
        self.assertGreater(len(decoded.pcm), 15900)


if __name__ == "__main__":
    unittest.main()
