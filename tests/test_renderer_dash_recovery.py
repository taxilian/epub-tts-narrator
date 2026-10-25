from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.speech_recovery import split_recovery_text
from legacy_recovery_fixtures import LegacySpeechResolver as SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
import test_renderer_speech_recovery_integration as fixtures
from test_renderer_recurring_resolution import PlanClient


TEXT = "Several people wait—then others return—before daylight comes!"
PARTS = ["Several people wait—", "then others return—", "before daylight comes!"]


class DashSplitTests(unittest.TestCase):
    def test_emdash_clauses_are_exact_contiguous_fallback_parts(self):
        parts = split_recovery_text(TEXT)
        self.assertIsNotNone(parts)
        self.assertEqual([row["text"] for row in parts], PARTS)
        self.assertEqual("".join(row["text"] for row in parts), TEXT)
        self.assertEqual(parts[0]["start"], 0)
        self.assertEqual(parts[-1]["end"], len(TEXT))
        self.assertTrue(all(left["end"] == right["start"] for left, right in zip(parts, parts[1:])))

    def test_existing_sentence_strategy_wins_over_new_dash_boundaries(self):
        text = "Several people wait—then return. Before daylight comes!"
        self.assertEqual([row["text"] for row in split_recovery_text(text)],
                         ["Several people wait—then return. ", "Before daylight comes!"])

    def test_tiny_fragments_hyphens_ranges_and_addresses_are_not_dash_split(self):
        for text in ("Stop—then others return.", "Several people wait—stop!",
                     "Several well-known people waited.", "Read pages 1–3 before bed.",
                     "Read https://example.test/a—b before bed.",
                     "Write to reader@exam—ple.test before bed.",
                     "Read www.example.test/a—b before bed."):
            with self.subTest(text=text):
                self.assertIsNone(split_recovery_text(text))


class DashEngine(fixtures.RecoveryEngine):
    def _write(self, text, path, seed):
        metadata = super()._write(text, path, seed)
        if text == TEXT:
            import hashlib
            self.registry[hashlib.sha256(fixtures.pcm(path)).hexdigest()] = TEXT.replace("others ", "")
        return metadata


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class DashRecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.SpeechRecoveryIntegrationTests.setUp(self)
        self.chunk["spoken_text"] = TEXT
        self.chunk["source_ranges"][0]["end"] = self.chunk["source_ranges"][0]["start"] + len(TEXT)
        self.engine = DashEngine(self.registry)
        self.client = PlanClient([{"action": "stop", "edits": [], "reason": "No whole-take strategy appears useful."}])
        self.base = self.verifier
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))

    def render(self):
        return fixtures.SpeechRecoveryIntegrationTests.render_chunk(self)

    def test_model_stop_uses_untried_safe_split_and_reuses_verified_result(self):
        result, acceptance = self.render()
        self.assertTrue(acceptance["report"]["passed"])
        self.assertEqual([row["text"] for row in self.engine.calls], [TEXT] * 3 + PARTS)
        self.assertEqual(acceptance["report"]["expected_text"], TEXT)
        self.assertEqual(acceptance["report"]["metrics"]["duration_seconds"], 3)
        self.assertTrue(Path(result["wav_path"]).is_file())
        marker = next((self.work / "clips").glob("*/speech-recovery.json"))
        record = read_json(marker.parent / "attempt-003.json")
        self.assertEqual(record["report"]["planner_resolution_action"], "stop")
        self.assertEqual(record["report"]["resolution_action"], "split")
        state = read_json(marker.parent / "speech-resolution.json")
        self.assertEqual(state["history"][-1]["audit"]["proposal"]["action"], "stop")
        before = len(self.engine.calls), len(self.client.plans)
        self.assertEqual(self.render(), (result, acceptance))
        self.assertEqual((len(self.engine.calls), len(self.client.plans)), before)

    def test_old_unsplittable_stop_resumes_without_another_whole_take(self):
        with patch("epub_tts_narrator.renderer.speech_recovery.split_recovery_text", return_value=None):
            with self.assertRaises(SpeechVerificationFailed):
                self.render()
        self.assertEqual(len(self.engine.calls), 3)

        _, accepted = self.render()

        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual([row["text"] for row in self.engine.calls], [TEXT] * 3 + PARTS)
        marker = next((self.work / "clips").glob("*/speech-recovery.json"))
        history = read_json(marker.parent / "speech-resolution.json")["history"]
        self.assertEqual(len(history), 2)
        self.assertNotIn("split", history[0]["audit"]["payload"]["permitted_actions"])
        self.assertIn("split", history[1]["audit"]["payload"]["permitted_actions"])

    def test_interrupted_split_resumes_verified_parts_without_more_whole_takes(self):
        self.engine.interrupt_part = PARTS[1]
        with self.assertRaises(KeyboardInterrupt):
            self.render()
        accepted_parts = {path: path.read_bytes() for path in (self.work / "clips").glob("*/parts/*/accepted.json")}
        self.assertEqual(len(accepted_parts), 1)
        self.engine.interrupt_part = None

        _, accepted = self.render()

        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual([row["text"] for row in self.engine.calls], [TEXT] * 3 + PARTS[:2] + PARTS[1:])
        for path, contents in accepted_parts.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_split_fallback_respects_small_shared_allowance(self):
        self.config["verification"].update(max_resolution_attempts=4, recurrence_threshold=1)
        _, accepted = self.render()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual([row["text"] for row in self.engine.calls], [TEXT] + PARTS)

    def test_stop_remains_terminal_when_budget_cannot_cover_all_parts(self):
        self.config["verification"].update(max_resolution_attempts=2, recurrence_threshold=1)
        with self.assertRaises(SpeechVerificationFailed) as caught:
            self.render()
        self.assertEqual(caught.exception.report["resolution_action"], "stop")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertFalse(list((self.work / "clips").glob("*/speech-recovery.json")))

    def test_rejected_join_never_publishes_or_repeats_exhausted_split(self):
        self.base.fail_assembly = True
        with self.assertRaises(SpeechVerificationFailed):
            self.render()
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))
        self.assertEqual([row["text"] for row in self.engine.calls if row["text"] != TEXT], PARTS)
        markers = list((self.work / "clips").glob("*/speech-recovery.json"))
        self.assertEqual(len(markers), 1)
        state = read_json(markers[0].parent / "speech-resolution.json")
        self.assertFalse(state["binding"]["can_split"])


if __name__ == "__main__":
    unittest.main()
