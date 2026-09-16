import unittest

from gateway.app import pcm24_to_16, vad_values


class RealtimeHelperTests(unittest.TestCase):
    def test_vad_validation(self):
        self.assertEqual(vad_values({"threshold": .5, "min_silence_ms": 350}), {"threshold": .5, "min_silence_ms": 350})
        with self.assertRaises(ValueError): vad_values({"threshold": 2})
        with self.assertRaises(ValueError): vad_values({"unknown": 1})

    def test_filtered_24khz_resample(self):
        from av.audio.resampler import AudioResampler
        output = pcm24_to_16(AudioResampler(format="s16", layout="mono", rate=16000), b"\0\0" * 24_000)
        self.assertGreater(len(output), 30_000)
        self.assertLessEqual(len(output), 32_000)


if __name__ == "__main__":
    unittest.main()
