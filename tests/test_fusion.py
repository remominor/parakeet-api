import unittest

from gateway.backends import SpeakerSegment, Word
from gateway.fusion import attribute_words, clean_regions, interval_statistics


class FusionTests(unittest.TestCase):
    def test_nested_overlap_union_and_transitions(self):
        segments = [
            SpeakerSegment(0, 5, "speaker_0"),
            SpeakerSegment(1, 2, "speaker_1"),
            SpeakerSegment(3, 4, "speaker_1"),
            SpeakerSegment(5, 6, "speaker_1"),
        ]
        stats = interval_statistics(segments)
        self.assertEqual(stats["speech_duration"], 6)
        self.assertEqual(stats["overlap_duration"], 2)
        self.assertAlmostEqual(stats["overlap_ratio"], 1 / 3, places=6)
        self.assertEqual(stats["dominant_speaker"], "speaker_0")
        self.assertEqual(stats["speaker_changes"], 1)
        self.assertEqual(clean_regions(segments, "speaker_0"), [(0, 1), (2, 3), (4, 5)])

    def test_word_thresholds_overlap_and_point_containment(self):
        segments = [SpeakerSegment(0, 2, "speaker_0"), SpeakerSegment(1.85, 3, "speaker_1")]
        words = [Word("lead", 0, 1), Word("ok", 1, 2), Word("mixed", 1.8, 2.2), Word("point", 2.5, 2.5), Word("none", 4, 4)]
        attribute_words(words, segments)
        self.assertEqual([word.speaker for word in words], ["speaker_0", "speaker_0", "overlap", "speaker_1", "unattributed"])

    def test_dominant_tie_uses_first_arrival(self):
        stats = interval_statistics([SpeakerSegment(1, 2, "speaker_1"), SpeakerSegment(2, 3, "speaker_0")])
        self.assertEqual(stats["dominant_speaker"], "speaker_1")

    def test_duplicate_same_speaker_intervals_do_not_double_count_word_coverage(self):
        segments = [
            SpeakerSegment(0, .3, "speaker_0"),
            SpeakerSegment(0, .3, "speaker_0"),
            SpeakerSegment(.2, 1, "speaker_1"),
        ]
        word = Word("mixed", 0, 1)
        attribute_words([word], segments)
        self.assertEqual(word.speaker, "overlap")


if __name__ == "__main__":
    unittest.main()
