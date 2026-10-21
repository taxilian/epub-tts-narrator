"""A new render invocation retries failed speech without losing accepted work."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, SpeechRetryBudget, SpeechVerificationFailed,
    VerificationUnavailable, verified_take,
)
import epub_tts_narrator.renderer.workflow as workflow
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for retry integration")
class SpeechRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
            keep_intermediate_audio=True,
        )
        self.config["quality"] = validate_policy({})
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)
        self.text = "Hello there."
        self.request = {"kind": "speech", "text": self.text, "engine": self.engine.identity}
        self.calls = []
        self.messages = []

    def generate(self, path, seed):
        self.calls.append({"path": path, "seed": seed})
        return self.engine._write(self.text, path, seed)

    def take(self, budget=None, *, verifier=None, generate=None, purpose="speech", request=None):
        return verified_take(
            self.text, generate or self.generate, self.root / "takes",
            request or self.request, verifier or self.verifier, self.config,
            notify=self.messages.append, purpose=purpose, retry_budget=budget,
        )

    def snapshot(self):
        return {path: path.read_bytes() for path in (self.root / "takes").iterdir()
                if path.is_file()}

    def assert_snapshot_unchanged(self, before):
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents, path.name)

    def test_rerun_generates_fresh_seeds_retains_failures_and_resumes_new_acceptance(self):
        self.verifier.wrong_texts.add(self.text)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(SpeechRetryBudget())
        before = self.snapshot()
        self.assertEqual(len(self.calls), 2)

        def succeed_on_fourth(path, seed):
            if len(self.calls) == 3:
                self.verifier.wrong_texts.clear()
            return self.generate(path, seed)

        wav, accepted = self.take(SpeechRetryBudget(), generate=succeed_on_fourth)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len({call["seed"] for call in self.calls}), 4)
        self.assertEqual(accepted["attempt"], 4)
        self.assertEqual(wav.name, "attempt-004.wav")
        self.assert_snapshot_unchanged(before)
        self.assertEqual(
            [read_json(self.root / "takes" / f"attempt-{number:03d}.json")["report"]["passed"]
             for number in range(1, 5)],
            [False, False, False, True],
        )
        count = len(self.verifier.calls)
        with patch.object(self.engine, "_write", side_effect=AssertionError("regenerated accepted speech")):
            self.assertEqual(self.take(SpeechRetryBudget()), (wav, accepted))
        self.assertEqual(len(self.verifier.calls), count)

    def test_same_invocation_cannot_reset_exhausted_request_budget(self):
        self.verifier.wrong_texts.add(self.text)
        budget = SpeechRetryBudget()
        for _ in range(3):
            with self.assertRaises(SpeechVerificationFailed):
                self.take(budget)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 2)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(SpeechRetryBudget())
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.verifier.calls), 4)

    def test_ledger_without_explicit_render_invocation_retains_legacy_behavior(self):
        self.verifier.wrong_texts.add(self.text)
        for _ in range(2):
            with self.assertRaises(SpeechVerificationFailed):
                self.take()
        self.assertEqual(len(self.calls), 2)

    def test_changed_verifier_rechecks_saved_audio_before_generating_anything(self):
        self.verifier.wrong_texts.add(self.text)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(SpeechRetryBudget())
        replacement_verifier = FixtureVerifier(self.fixtures, version=2)
        with patch.object(self.engine, "_write", side_effect=AssertionError("regenerated before rechecking")):
            wav, accepted = self.take(SpeechRetryBudget(), verifier=replacement_verifier)
        self.assertEqual(wav.name, "attempt-001.wav")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(replacement_verifier.calls), 1)
        self.assertEqual(len(self.calls), 2)

    def test_interrupted_verification_reuses_saved_waveform(self):
        with patch.object(workflow, "validate_audio", side_effect=KeyboardInterrupt("verification interrupted")):
            with self.assertRaises(KeyboardInterrupt):
                self.take(SpeechRetryBudget())
        wav = self.root / "takes" / "attempt-001.wav"
        original_bytes = wav.read_bytes()
        pending_path = wav.with_suffix(".json")
        pending = read_json(pending_path)
        self.assertEqual(pending["report"]["reasons"], ["generation_incomplete"])
        self.assertTrue(pending["report"]["retry_on_resume"])
        orphan = self.root / "takes" / "attempt-002-raw.wav"
        self.engine._write(self.text, orphan, 42)
        before = {path: contents for path, contents in self.snapshot().items() if path != pending_path}
        resumed, accepted = self.take(SpeechRetryBudget())
        self.assertEqual(resumed, wav)
        self.assertEqual(resumed.read_bytes(), original_bytes)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 1)
        self.assert_snapshot_unchanged(before)

    def test_raw_only_orphans_are_removed_before_fresh_generation_with_next_sparse_id(self):
        directory = self.root / "takes"
        directory.mkdir()
        old_seeds = []
        for attempt, suffix in ((1, "raw"), (1, "recovery-raw"), (1000, "raw"),
                                (1000, "recovery-raw")):
            seed = workflow._seed(self.config["render"]["seed"], self.request, attempt)
            old_seeds.append(seed)
            self.engine._write(self.text, directory / f"attempt-{attempt:03d}-{suffix}.wav", seed)
        metadata = directory / "attempt-1000-generation.json"
        write_json(metadata, {"interrupted": True, "diagnostic": "unfinished generation"})
        before = self.snapshot()

        def generate_after_discard(path, seed):
            # The abandoned audio is discarded before spending time on a new take.
            self.assertTrue(all(not old.exists() for old in before if old.suffix == ".wav"))
            return self.generate(path, seed)

        wav, accepted = self.take(SpeechRetryBudget(), generate=generate_after_discard)
        self.assertEqual(wav.name, "attempt-1001.wav")
        self.assertEqual(accepted["attempt"], 1001)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(self.calls[0]["seed"], old_seeds)
        self.assertEqual(self.calls[0]["path"].name, "attempt-1001-raw.wav")
        self.assertEqual(metadata.read_bytes(), before[metadata])
        self.assertTrue(all(not old.exists() for old in before if old.suffix == ".wav"))
        self.assertEqual(len(self.verifier.calls), 1)
        self.assertFalse((directory / "attempt-001.json").exists())
        self.assertFalse((directory / "attempt-1000.json").exists())

    def test_interruption_after_discard_before_output_keeps_attempt_number_and_seed_reserved(self):
        directory = self.root / "takes"
        directory.mkdir()
        orphan = directory / "attempt-1000-raw.wav"
        self.engine._write(self.text, orphan, 42)

        def interrupt_before_writing(path, seed):
            self.assertFalse(orphan.exists())
            self.calls.append({"path": path, "seed": seed})
            raise KeyboardInterrupt("generation interrupted before output")

        with self.assertRaises(KeyboardInterrupt):
            self.take(SpeechRetryBudget(), generate=interrupt_before_writing)
        pending_path = directory / "attempt-1001.json"
        pending = read_json(pending_path)
        pending_bytes = pending_path.read_bytes()
        self.assertEqual(pending["attempt"], 1001)
        self.assertEqual(pending["seed"], self.calls[0]["seed"])
        self.assertEqual(pending["report"]["reasons"], ["generation_incomplete"])
        self.assertTrue(pending["report"]["retry_on_resume"])
        self.assertIsNone(pending["audio_file"])
        self.assertFalse((directory / "attempt-1001-raw.wav").exists())
        wav, accepted = self.take(SpeechRetryBudget())
        self.assertEqual(wav.name, "attempt-1002.wav")
        self.assertEqual(accepted["attempt"], 1002)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(self.calls[0]["seed"], self.calls[1]["seed"])
        self.assertEqual(pending_path.read_bytes(), pending_bytes)

    def test_reusing_accepted_take_does_not_discard_raw_orphans(self):
        wav, accepted = self.take(SpeechRetryBudget())
        directory = self.root / "takes"
        for suffix in ("raw", "recovery-raw"):
            self.engine._write(self.text, directory / f"attempt-002-{suffix}.wav", 42)
        before = self.snapshot()
        self.assertEqual(self.take(SpeechRetryBudget()), (wav, accepted))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assert_snapshot_unchanged(before)

    def test_jev_outage_retries_verification_without_spending_new_generation_budget(self):
        validate = workflow.validate_audio

        def unavailable(*args, **kwargs):
            report = validate(*args, **kwargs)
            report.update(passed=False, reasons=["text_mismatch"],
                          text_judge={"decision": "unavailable", "error_code": "fixture_outage"})
            return report

        with patch.object(workflow, "validate_audio", side_effect=unavailable):
            for _ in range(2):
                with self.assertRaises(VerificationUnavailable):
                    self.take(SpeechRetryBudget())
        self.assertEqual(len(self.calls), 1)
        wav, accepted = self.take(SpeechRetryBudget())
        self.assertEqual(wav.name, "attempt-001.wav")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 1)

    def test_cumulative_attempt_ids_above_ten_allow_acceptance_and_missing_audio_replacement(self):
        self.verifier.wrong_texts.add(self.text)
        for _ in range(6):
            with self.assertRaises(SpeechVerificationFailed):
                self.take(SpeechRetryBudget())
        self.assertEqual(len(self.calls), 12)
        failures = self.snapshot()
        self.verifier.wrong_texts.clear()
        wav, accepted = self.take(SpeechRetryBudget())
        self.assertEqual(accepted["attempt"], 13)
        self.assertEqual(wav.name, "attempt-013.wav")
        self.assert_snapshot_unchanged(failures)
        wav.unlink()
        replacement, replacement_record = self.take(SpeechRetryBudget())
        self.assertTrue(replacement.is_file())
        self.assertTrue(replacement_record["report"]["passed"])
        self.assertEqual(len(self.calls), 14)
        self.assert_snapshot_unchanged(failures)

    def test_references_never_acquire_an_automatic_new_casting_budget(self):
        self.verifier.wrong_texts.add(self.text)
        request = {**self.request, "kind": "reference"}
        for _ in range(2):
            with self.assertRaises(SpeechVerificationFailed):
                self.take(SpeechRetryBudget(), purpose="reference", request=request)
        self.assertEqual(len(self.calls), 2)

    def test_failed_paragraphs_keep_their_exhausted_budget_for_split_recovery(self):
        self.verifier.wrong_texts.add(self.text)
        request = {**self.request, "kind": "paragraph_speech"}
        for _ in range(2):
            with self.assertRaises(SpeechVerificationFailed):
                self.take(SpeechRetryBudget(), request=request)
        self.assertEqual(len(self.calls), 2)

    def test_repaired_transport_stops_retrying_as_soon_as_one_take_passes(self):
        class Disconnected(RuntimeError):
            transport_error = True
            retryable = True

        def disconnected(path, seed):
            self.calls.append({"path": path, "seed": seed})
            raise Disconnected("fixture renderer disconnected")

        with self.assertRaisesRegex(SpeechVerificationFailed, "fixture renderer disconnected"):
            self.take(SpeechRetryBudget(), generate=disconnected)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.verifier.calls, [])
        _, accepted = self.take(SpeechRetryBudget())
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 3)

    def test_permanent_transport_error_stops_immediately_but_new_invocation_can_recover(self):
        class Unauthorized(RuntimeError):
            transport_error = True
            retryable = False

        def unauthorized(path, seed):
            self.calls.append({"path": path, "seed": seed})
            raise Unauthorized("fixture credential rejected")

        with self.assertRaisesRegex(RenderBlocked, "fixture credential rejected"):
            self.take(SpeechRetryBudget(), generate=unauthorized)
        self.assertEqual(len(self.calls), 1)
        _, accepted = self.take(SpeechRetryBudget())
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 2)

    def test_changed_policy_after_transport_failure_preserves_old_reports_and_appends_take(self):
        class Disconnected(RuntimeError):
            transport_error = True
            retryable = True

        def disconnected(path, seed):
            self.calls.append({"path": path, "seed": seed})
            raise Disconnected("fixture renderer disconnected")

        with self.assertRaises(SpeechVerificationFailed):
            self.take(SpeechRetryBudget(), generate=disconnected)
        before = self.snapshot()
        self.assertEqual(len(before), 2)
        self.assertTrue(all(read_json(path)["audio_file"] is None for path in before))
        self.config["quality"]["min_duration_seconds"] = 0.10
        wav, accepted = self.take(SpeechRetryBudget())
        self.assertEqual(wav.name, "attempt-003.wav")
        self.assertEqual(accepted["attempt"], 3)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len({call["seed"] for call in self.calls}), 3)
        self.assert_snapshot_unchanged(before)
        self.assertEqual(len(self.verifier.calls), 1)


if __name__ == "__main__":
    unittest.main()
