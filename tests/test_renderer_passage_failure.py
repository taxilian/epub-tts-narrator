"""A failed passage in the run shell: its fresh-takes reset and a person's direction change saved to the book."""

from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.take_review import TakeReview
from epub_tts_narrator.renderer.workflow import _failure_passage


HASH = "a" * 64


class FreshTakesResetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / "render-work"
        self.folder = self.work / "clips" / HASH
        self.folder.mkdir(parents=True)
        self.review = TakeReview(self.work, {}, None, HASH, directory=self.folder)

    def test_a_saved_choice_and_repair_are_set_aside_never_deleted(self):
        write_json(self.folder / "exhaustion-selection.json", {"audit": {}})
        write_json(self.folder / "clip-repair.json", {"phase": "failed"})
        (self.folder / "clip-repair-parts").mkdir()
        (self.folder / "clip-repair-parts" / "part-001.json").write_text("{}")
        moved = self.review.reset_for_fresh_takes()
        self.assertEqual(len(moved), 3)
        for name in ("exhaustion-selection.json", "clip-repair.json", "clip-repair-parts"):
            self.assertFalse((self.folder / name).exists())
        self.assertTrue(all((self.folder / name).exists() and ".fresh-retry-" in name for name in moved))
        record = read_json(next(self.folder.glob("fresh-retry-*.json")))
        self.assertEqual((record["evidence"]["reason"], sorted(record["retired"])),
                         ("person_requested_fresh_takes", sorted(moved)))
        self.assertEqual(self.review.reset_for_fresh_takes(), [])  # nothing left to set aside

    def test_an_accepted_take_is_never_touched(self):
        write_json(self.folder / "accepted.json", {"attempt": 1})
        write_json(self.folder / "exhaustion-selection.json", {"audit": {}})
        self.assertEqual(self.review.reset_for_fresh_takes(), [])
        self.assertTrue((self.folder / "exhaustion-selection.json").exists())

    def test_attempts_list_takes_without_audio_with_their_errors(self):
        request = {"kind": "speech", "chunk": {"chunk_id": "c", "spoken_text": "Hello there.",
                                               "segment": {"id": "s1", "delivery": {"direction": "softly"}}}}
        write_json(self.folder / "attempt-001.json", {
            "request_hash": HASH, "request": request, "attempt": 1, "audio_file": None,
            "report": {"passed": False, "reasons": ["generation_or_verification_error"],
                       "error": "Drama 3 did not finish the text: 'silence'", "error_type": "EngineError"}})
        write_json(self.folder / "attempt-002.json", {
            "request_hash": HASH, "request": request, "attempt": 2, "audio_file": None,
            "report": {"passed": False, "reasons": ["generation_incomplete"], "retry_on_resume": True}})
        write_json(self.folder / "attempt-003.json", {"request_hash": "b" * 64, "attempt": 3, "report": {}})
        rows = self.review.attempts()
        self.assertEqual([(row["attempt"], row["audio"], row["pending"]) for row in rows],
                         [(1, False, False), (2, False, True)])
        self.assertEqual(rows[0]["error"], "EngineError: Drama 3 did not finish the text: 'silence'")
        self.assertEqual(self.review.passage(), {"text": "Hello there.", "kind": "speech", "segment_ids": ["s1"],
                                                 "directions": ["softly"]})


class FailurePassageTests(unittest.TestCase):
    def test_block_and_speakers_come_from_the_chapter_and_cast(self):
        bundle = {"cast": {"roles": [{"id": "r1", "kind": "dialogue", "character_id": "c1"},
                                     {"id": "rn", "kind": "narration", "character_id": None}],
                           "characters": [{"id": "c1", "name": "Mara"}]}}
        chapter = {"chunks": [{"chunk_id": "x", "segment": {"id": "s0", "role_id": "rn"}},
                              {"chunk_id": "y", "paragraph_turns": [
                                  {"segment": {"id": "s1", "role_id": "rn"}},
                                  {"segment": {"id": "s2", "role_id": "r1"}}]}]}
        self.assertEqual(_failure_passage(bundle, chapter, {"chunk_id": "y", "segment_ids": ["s1", "s2"]}),
                         {"block": 2, "speakers": ["Narrator", "Mara"]})
        self.assertEqual(_failure_passage(bundle, chapter, {}), {})


class DirectionEditTests(unittest.TestCase):
    def test_a_new_direction_is_saved_to_the_book_as_a_person_s_choice(self):
        from epub_tts_narrator.analysis_directory import _completed_directory
        from epub_tts_narrator.renderer.passage_edit import check_direction, save_direction
        from test_review_tui import publish_book
        with tempfile.TemporaryDirectory() as root:
            directory, document = publish_book(root)
            segment = next(item for item in document["segments"] if item["kind"] != "heading")
            save_direction(directory, [segment["id"]], "  quietly, tired  ", progress=lambda _: None)
            saved = next(item for item in _completed_directory(directory)[0]["segments"]
                         if item["id"] == segment["id"])
            self.assertEqual((saved["delivery"]["direction"], saved["delivery"]["basis"]), ("quietly, tired", "user"))
            with self.assertRaisesRegex(ValueError, "no longer has this passage"):
                save_direction(directory, ["missing-segment"], "softly", progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "at most 240 characters"):
            check_direction("x" * 241)


if __name__ == "__main__":
    unittest.main()
