"""Interactive speaker decisions checkpoint to disk and replay on a rerun."""

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.speaker_decisions import (DECISION_CHECKPOINT_VERSION, decision_checkpoint_key,
                                                 prompt_speaker_decisions)
from shared_fixtures import Terminal
from test_speaker_decisions import missing


def decide(document, answers, cache):
    output = StringIO()
    result = prompt_speaker_decisions(document, book_path=Path("books/A Book"),
                                      input_stream=Terminal(answers), output_stream=output,
                                      decisions_cache=cache)
    return result, output.getvalue()


def journals(cache):
    folder = Path(cache) / "speaker-decisions"
    return list(folder.glob("*.json")) if folder.is_dir() else []


class SpeakerDecisionResumeTests(unittest.TestCase):
    def test_decisions_checkpoint_written_and_replayed_byte_identically_without_prompting(self):
        with tempfile.TemporaryDirectory() as cache:
            document = missing()
            first, _ = decide(document, "1\n", cache)
            self.assertEqual(validate_document(first), [])
            files = journals(cache)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, decision_checkpoint_key(document) + ".json")
            record = json.loads(files[0].read_text())
            self.assertEqual(record["scope"], "speaker-decisions")
            self.assertEqual(record["version"], DECISION_CHECKPOINT_VERSION)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(len(record["decisions"]), 1)
            # A rerun with no answers available must replay, not prompt.
            second, second_out = decide(missing(), "", cache)
            self.assertEqual(second, first)
            self.assertIn("Replaying 1 saved speaker decision", second_out)
            self.assertNotIn("Most likely speakers", second_out)

    def test_defined_voice_replays_identical_records_without_reasking(self):
        with tempfile.TemporaryDirectory() as cache:
            first, _ = decide(missing(), "n\nKessa\nA wary voice.\n", cache)
            second, second_out = decide(missing(), "", cache)
            self.assertEqual(second, first)
            self.assertNotIn("New character name", second_out)
            self.assertEqual([c["name"] for c in second["cast"]["characters"]].count("Kessa"), 1)

    def test_partial_answer_resumes_and_prompts_only_the_remainder(self):
        with tempfile.TemporaryDirectory() as cache:
            first, _ = decide(missing((1, 2)), "1\nq\n", cache)
            self.assertIsNotNone(first["segments"][1]["role_id"])
            self.assertIsNone(first["segments"][2]["role_id"])
            record = json.loads(journals(cache)[0].read_text())
            self.assertEqual(record["status"], "interrupted")
            second, second_out = decide(missing((1, 2)), "1\n", cache)
            self.assertEqual(second_out.count("PASSAGE TO ASSIGN"), 1)
            self.assertIn("Replaying 1 saved speaker decision", second_out)
            self.assertIsNotNone(second["segments"][1]["role_id"])
            self.assertIsNotNone(second["segments"][2]["role_id"])
            self.assertEqual(validate_document(second), [])

    def test_changed_input_discards_checkpoint_and_prompts_fresh(self):
        with tempfile.TemporaryDirectory() as cache:
            decide(missing((1,)), "1\n", cache)
            # A rerun over a different pending set has a different checkpoint key.
            second, second_out = decide(missing((2,)), "1\n", cache)
            self.assertNotIn("Replaying", second_out)
            self.assertIn("Most likely speakers", second_out)

    def test_malformed_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as cache:
            path = Path(cache) / "speaker-decisions" / (decision_checkpoint_key(missing()) + ".json")
            path.parent.mkdir(parents=True)
            path.write_text("not json")
            result, out = decide(missing(), "1\n", cache)
            self.assertNotIn("Replaying", out)
            self.assertIn("Most likely speakers", out)
            self.assertIsNotNone(result["segments"][1]["role_id"])

    def test_replay_applies_without_a_terminal(self):
        with tempfile.TemporaryDirectory() as cache:
            decide(missing(), "1\n", cache)
            output = StringIO()
            result = prompt_speaker_decisions(missing(), book_path=Path("books/A Book"),
                                              input_stream=StringIO(""), output_stream=output,
                                              decisions_cache=cache)
            self.assertIsNotNone(result["segments"][1]["role_id"])


if __name__ == "__main__":
    unittest.main()
