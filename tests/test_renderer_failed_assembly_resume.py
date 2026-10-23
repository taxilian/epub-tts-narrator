"""A definitively rejected join must not trap every later render in the same join."""

from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, VerificationUnavailable, _render_chunk,
)
import epub_tts_narrator.renderer.speech_recovery as speech_recovery
from test_renderer_plain_delivery_integration import DirectedFixtureEngine, DIRECTION
from test_renderer_recurring_resolution import PlanClient
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryVerifier
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


TEXT = "Arrgh the pain has returned. Everyone waited outside."


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for rejected-join recovery")
class FailedAssemblyResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        registry = {}
        self.engine = DirectedFixtureEngine(registry)
        self.engine.heard_override = lambda given, direction, number: (
            given.replace("Arrgh", "Oh") if given == TEXT and direction else given)
        self.base = RecoveryVerifier(registry)
        self.base.fail_assembly = True
        self.client = PlanClient([{"action": "plain_delivery", "edits": [],
                                   "reason": "Try the whole passage without the obstructing acting direction."}])
        self.verifier = self.base
        self.voice = reference("character")
        voice_path = self.work / "reference.wav"
        write_tone(voice_path)
        self.voice.update(reference_audio=str(voice_path), reference_sha256=file_hash(voice_path))
        self.chunk = speech_request(self.voice, TEXT)["chunk"]
        self.chunk["segment"]["delivery"]["direction"] = DIRECTION
        self.messages = []

    def adaptive(self):
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))

    def render_chunk(self, budget=None):
        return _render_chunk(self.chunk, 1, 1, {self.voice["voice_id"]: self.voice},
                             self.engine, self.engine.identity, self.verifier, self.config,
                             self.work, {}, Lock(), notify=self.messages.append,
                             retry_budget=budget or SpeechRetryBudget())

    def verified_parts(self):
        paths = list((self.work / "clips").glob("*/parts/*/accepted.json"))
        self.assertEqual(len(paths), 2)
        return {path: path.read_bytes() for path in paths}

    def check_unchanged(self, files):
        for path, contents in files.items():
            self.assertEqual(path.read_bytes(), contents, str(path))

    def test_saved_rejected_join_can_resume_with_a_new_verified_plain_whole_take(self):
        # Produce the state older versions left behind: three rejected whole
        # takes, two accepted parts, and one definitely rejected joined WAV.
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        self.assertEqual(sum(call["text"] == TEXT for call in self.engine.calls), 3)
        self.assertEqual(sum(call["text"] != TEXT for call in self.engine.calls), 2)
        parts = self.verified_parts()
        marker_path = next((self.work / "clips").glob("*/speech-recovery.json"))
        marker_bytes = marker_path.read_bytes()
        joined_dir = self.work / "clips" / read_json(marker_path)["recovery_request_hash"]
        rejected_join = {path: path.read_bytes() for path in joined_dir.glob("attempt-001.*")}
        self.assertFalse(read_json(joined_dir / "attempt-001.json")["report"]["passed"])
        self.adaptive()
        result, accepted = self.render_chunk()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        self.assertEqual(self.engine.calls[-1]["text"], TEXT)
        self.assertEqual(self.engine.calls[-1]["direction"], "")
        self.assertEqual(len(self.engine.calls), 6, "Resume needs one new whole take, not another join")
        self.assertEqual(len(self.client.plans), 1)
        feedback = [row for row in self.client.plans[0]["previous_decisions"]
                    if row.get("error_code") == "split_full_verification_failed"]
        self.assertEqual(len(feedback), 1)
        joined_transcript = read_json(joined_dir / "attempt-001.json")["report"]["transcript"]
        self.assertIn(joined_transcript, feedback[0]["validation_error"],
                      "The planner must see why the actual joined recording failed")
        accepted_record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertIn("delivery_recovery", accepted_record["engine_metadata"])
        self.check_unchanged(parts)
        self.check_unchanged(rejected_join)
        self.assertEqual(marker_path.read_bytes(), marker_bytes)
        before = len(self.engine.calls), len(self.base.calls), len(self.client.plans)
        self.assertEqual(self.render_chunk(), (result, accepted))
        self.assertEqual((len(self.engine.calls), len(self.base.calls), len(self.client.plans)), before)

    def test_failed_join_and_fresh_whole_recovery_share_one_invocation_budget(self):
        self.client.actions = [{"action": "retry", "edits": [], "reason": "Try another complete performance."}]
        self.adaptive()
        budget = SpeechRetryBudget()
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(budget)
        self.assertEqual(len(self.engine.calls), 12, "Returning to whole synthesis cannot reset the shared cap")
        self.assertEqual(sum(call["text"] != TEXT for call in self.engine.calls), 2)
        self.assertEqual(sum(call["text"] == TEXT for call in self.engine.calls), 10)
        parts = self.verified_parts()
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk(budget)
        self.assertEqual(len(self.engine.calls), 12)
        self.check_unchanged(parts)
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))

    def test_saved_explicit_split_decision_cannot_replay_the_already_failed_strategy(self):
        self.client.actions = [
            {"action": "split", "edits": [], "reason": "Try independent sentences."},
            {"action": "plain_delivery", "edits": [], "reason": "The join failed; try clear whole speech."},
        ]
        self.adaptive()
        recover = speech_recovery.recover_speech

        def interrupt_after_definite_join_failure(*args, **kwargs):
            try:
                return recover(*args, **kwargs)
            except SpeechVerificationFailed as exc:
                if "recovery" in exc.request:
                    raise KeyboardInterrupt("Saved explicit split and rejected assembly") from exc
                raise

        with patch.object(speech_recovery, "recover_speech", side_effect=interrupt_after_definite_join_failure):
            with self.assertRaises(KeyboardInterrupt):
                self.render_chunk()
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(len(self.client.plans), 1)
        parts = self.verified_parts()
        _, accepted = self.render_chunk()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.engine.calls), 6)
        self.assertEqual(self.engine.calls[-1]["direction"], "")
        self.assertEqual(len(self.client.plans), 2)
        self.assertNotIn("split", self.client.plans[-1]["permitted_actions"],
                         "A definitely failed persisted split is not an available new strategy")
        self.check_unchanged(parts)

    def test_unavailable_join_verification_retains_parts_without_fresh_whole_synthesis(self):
        self.client.actions = [{"action": "retry", "edits": [], "reason": "Try a new take."}]
        self.adaptive()
        transcribe = self.base.transcribe

        def unavailable_only_for_join(path):
            with wave.open(str(path), "rb") as audio:
                joined = audio.getnframes() > 44100
            if joined:
                raise RuntimeError("Fixture verification service unavailable")
            return transcribe(path)

        with patch.object(self.base, "transcribe", side_effect=unavailable_only_for_join):
            with self.assertRaises(VerificationUnavailable):
                self.render_chunk()
        self.assertEqual(len(self.engine.calls), 8, "Unavailable ASR must not be treated as a rejected assembly")
        self.assertEqual(sum(call["text"] == TEXT for call in self.engine.calls), 6)
        self.verified_parts()
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))

    def test_child_rejection_is_not_mistaken_for_an_exhausted_join(self):
        self.client.actions = [{"action": "retry", "edits": [], "reason": "Try a new take."}]
        self.adaptive()
        self.engine.heard_override = lambda given, direction, number: (
            given.replace("Arrgh", "Oh") if given == TEXT else given.replace("pain ", ""))
        with self.assertRaises(SpeechVerificationFailed) as failure:
            self.render_chunk()
        self.assertEqual(failure.exception.request["kind"], "speech_recovery_part")
        self.assertEqual(sum(call["text"] == TEXT for call in self.engine.calls), 6)
        self.assertEqual(len(self.engine.calls), 11)
        self.assertFalse(list((self.work / "clips").glob("*/parts/*/accepted.json")))
        self.assertFalse(list((self.work / "speech-cache").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
