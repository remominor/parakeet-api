import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gateway.speakers import SpeakerStore, match_embedding


class SpeakerStoreTests(unittest.TestCase):
    def test_atomic_private_record_and_public_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory)
            record = store.create("alice-1", "Alice", [np.ones(512), np.ones(512) * 2])
            self.assertNotIn("embedding", record.public())
            path = Path(directory) / "alice-1.json"
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(store.get("alice-1").embedding.shape, (512,))
            with self.assertRaises(FileExistsError):
                store.create("alice-1", None, [np.ones(512)])

    def test_existing_private_writable_mount_allows_denied_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            with patch("gateway.speakers.os.chmod", side_effect=PermissionError("mounted volume")) as chmod:
                store = SpeakerStore(directory)
            chmod.assert_called_once_with(store.directory, 0o700)
            self.assertEqual(store.directory, Path(directory))

    def test_nonprivate_or_unwritable_mount_has_explicit_ownership_error(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o755)
            with patch("gateway.speakers.os.chmod", side_effect=PermissionError("mounted volume")):
                with self.assertRaisesRegex(RuntimeError, r"speaker store .*not writable by uid .*fix volume ownership to"):
                    SpeakerStore(directory)

    def test_unreadable_or_unwritable_mount_has_explicit_ownership_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("gateway.speakers.os.chmod", side_effect=PermissionError("mounted volume")), patch("gateway.speakers.os.access", return_value=False):
                with self.assertRaisesRegex(RuntimeError, r"speaker store .*not writable by uid .*fix volume ownership to"):
                    SpeakerStore(directory)

    def test_corrupt_and_incompatible_records_are_listed_not_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory)
            (Path(directory) / "broken.json").write_text("not json")
            record = store.list()[0]
            self.assertEqual(record.compatibility, "incompatible")
            self.assertEqual(match_embedding(np.ones(512), [record], threshold=0.5, margin=0.1)["status"], "unknown")

    def test_calibration_and_match_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory)
            alice = store.create("alice", None, [np.r_[1, np.zeros(511)]])
            bob = store.create("bob", None, [np.r_[0.8, 0.6, np.zeros(510)]])
            query = np.r_[1, np.zeros(511)]
            targeted = match_embedding(query, [alice], threshold=None, margin=None, target="alice")
            self.assertEqual(targeted, {"status": "calibration_required", "speaker_id": "alice", "display_name": None, "score": 1.0})
            untargeted = match_embedding(query, [alice, bob], threshold=None, margin=None)
            self.assertEqual(untargeted, {"status": "calibration_required", "candidate_speaker_id": "alice", "candidate_display_name": None, "candidate_score": 1.0})
            missing_margin = match_embedding(query, [alice, bob], threshold=0.5, margin=None)
            self.assertEqual(missing_margin["status"], "calibration_required")
            self.assertEqual(missing_margin["candidate_speaker_id"], "alice")
            self.assertEqual(missing_margin["candidate_score"], 1.0)
            self.assertEqual(match_embedding(query, [alice], threshold=1.0, margin=None, target="alice")["status"], "known")
            self.assertEqual(match_embedding(query, [alice, bob], threshold=0.5, margin=0.25)["status"], "ambiguous")
            self.assertEqual(match_embedding(query, [alice, bob], threshold=0.5, margin=0.1)["status"], "known")

    def test_single_candidate_does_not_require_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory)
            alice = store.create("alice", None, [np.r_[1, np.zeros(511)]])
            self.assertEqual(match_embedding(np.r_[1, np.zeros(511)], [alice], threshold=.5, margin=None)["status"], "known")
            self.assertEqual(match_embedding(np.r_[0, 1, np.zeros(510)], [alice], threshold=.5, margin=None)["status"], "unknown")

    def test_concurrent_create_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory); barrier = threading.Barrier(2); outcomes = []
            def create(vector):
                barrier.wait()
                try: store.create("alice", None, [vector]); outcomes.append("created")
                except FileExistsError: outcomes.append("conflict")
            threads = [threading.Thread(target=create, args=(np.r_[1, np.zeros(511)],)), threading.Thread(target=create, args=(np.r_[0, 1, np.zeros(510)],))]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertCountEqual(outcomes, ["created", "conflict"])
            self.assertEqual(store.get("alice").embedding.shape, (512,))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_dimension_and_id_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerStore(directory)
            with self.assertRaises(ValueError): store.create("Upper", None, [np.ones(512)])
            with self.assertRaises(ValueError): store.create("short", None, [np.ones(10)])


if __name__ == "__main__":
    unittest.main()
