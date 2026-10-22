"""Sentence recovery retains exact text and verifies parts plus the full assembly."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.speech_recovery import (
    recover_speech, recovery_available, split_recovery_text,
)
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, SpeechRetryBudget, SpeechVerificationFailed, VerificationUnavailable,
)
from epub_tts_narrator.renderer.assembly import normalize_clip
from test_renderer_jev_verification import Judge
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, timed_transcript


TEXT = "Insang Mortuary Services! What does it—"


class SplitPlanTests(unittest.TestCase):
    def test_target_and_whitespace_have_exact_contiguous_offsets(self):
        text = "Insang Mortuary Services!\n\tWhat does it—"
        parts = split_recovery_text(text)
        self.assertEqual([part["text"] for part in parts],
                         ["Insang Mortuary Services!\n\t", "What does it—"])
        self.assertEqual("".join(part["text"] for part in parts), text)
        self.assertEqual(parts[0]["start"], 0)
        self.assertEqual(parts[0]["end"], parts[1]["start"])
        self.assertEqual(parts[-1]["end"], len(text))

    def test_abbreviations_decimals_ellipses_and_tiny_parts_are_not_split(self):
        for text in ("Ask Dr. Jones now.", "Find X. Xavier now.", "It costs 3.14 today.",
                     "We waited... Then left quietly.", "Stop! What happens next?", "This is all.", ""):
            with self.subTest(text=text):
                self.assertIsNone(split_recovery_text(text))
        parts = split_recovery_text("Ask Dr. Jones now. Then come home.")
        self.assertEqual([part["text"] for part in parts], ["Ask Dr. Jones now. ", "Then come home."])
        parts = split_recovery_text("One person; two people\nthree people! four people? five people.")
        self.assertEqual(len(parts), 4)
        self.assertEqual("".join(part["text"] for part in parts),
                         "One person; two people\nthree people! four people? five people.")


class AudioVerifier:
    identity = {"backend": "independent-recovery-fixture"}

    def __init__(self, fixtures):
        self.fixtures = fixtures
        self.calls = []
        self.final_transcript = None
        self.final_error = None

    def transcribe(self, path):
        self.calls.append(Path(path))
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            data = source.readframes(frames)
        if frames > 44100 and self.final_error is not None:
            raise self.final_error
        # The fixture generator writes one-second PCM tones. Decode each tone
        # by its actual bytes; do not read requests or expected-text files.
        words = [self.fixtures[hashlib.sha256(data[start:start + 88200]).hexdigest()]
                 for start in range(0, len(data), 88200)]
        heard = "".join(words)
        if frames > 44100 and self.final_transcript is not None:
            heard = self.final_transcript
        return timed_transcript(heard)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for PCM recovery tests")
class SpeechRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
                                     keep_intermediate_audio=True)
        self.voice = {"asset_hash": "fixture-voice", "kind": "reference",
                      "reference_text": "The same character voice."}
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = AudioVerifier(self.fixtures)
        self.request = {"kind": "speech", "engine": self.engine.identity,
                        "voice_asset_hash": self.voice["asset_hash"], "base_seed": 1729,
                        "sample_rate": 44100, "chunk": {"spoken_text": TEXT,
                        "segment": {"delivery": {"direction": "A tired receptionist."}},
                        "source_ranges": [{"start": 10, "end": 10 + len(TEXT)}]}}
        self.parent = self.root / "clips" / digest(self.request)
        self.calls, self.messages = [], []
        self.bad_part = None
        self.interrupt_on = None

    def synthesize(self, text, path, seed):
        self.calls.append((text, seed))
        if len(self.calls) == self.interrupt_on:
            raise KeyboardInterrupt()
        spoken = "What it" if text == self.bad_part else text
        return self.engine._write(spoken, path, seed)

    def recover(self, *, verifier=None, retry_budget=None):
        return recover_speech(TEXT, self.synthesize, self.parent, self.request,
                              verifier or self.verifier, self.config, notify=self.messages.append,
                              retry_budget=retry_budget or SpeechRetryBudget())

    def test_parts_and_full_audio_verified_with_bound_audit_and_normal_speech_cache(self):
        self.assertFalse(recovery_available(self.parent, self.request, TEXT, self.verifier, self.config))
        wav, accepted = self.recover()
        self.assertTrue(recovery_available(self.parent, self.request, TEXT, self.verifier, self.config))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 3)
        self.assertEqual("".join(text for text, _ in self.calls), TEXT)
        self.assertEqual(accepted["request"]["chunk"], self.request["chunk"])
        self.assertEqual(wav.parent.name, accepted["request_hash"])
        self.assertEqual(digest(accepted["request"]), accepted["request_hash"])
        self.assertEqual(file_hash(wav), accepted["audio_sha256"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        audit = accepted["engine_metadata"]["recovery"]
        self.assertEqual(audit["parent_request_hash"], digest(self.request))
        self.assertEqual("".join(part["text"] for part in audit["parts"]), TEXT)
        for part in audit["parts"]:
            audio = wav.parent / part["audio_file"]
            record = read_json(audio.parent / "accepted.json")
            self.assertEqual(part["audio_sha256"], file_hash(audio))
            self.assertEqual(part["request_hash"], digest(record["request"]))
            self.assertTrue(record["report"]["passed"])
        with wave.open(str(wav), "rb") as audio:
            self.assertEqual(audio.getnframes(), 88200)
        cache = SpeechCache(self.root, {"voice": self.voice})
        identity = cache.identity(self.request)
        cache.remember(identity, accepted)
        self.assertEqual(cache.find(identity, request=self.request), (wav, accepted))
        self.assertEqual(self.recover(), (wav, accepted))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 3)

    def test_interruption_resumes_existing_plan_without_repeating_good_part(self):
        self.interrupt_on = 2
        with self.assertRaises(KeyboardInterrupt):
            self.recover()
        self.assertTrue(recovery_available(self.parent, self.request, TEXT, self.verifier, self.config))
        self.interrupt_on = None
        wav, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(sum(text.startswith("Insang") for text, _ in self.calls), 1)
        self.assertTrue(wav.is_file())

    def test_missing_word_in_child_cannot_publish_but_new_run_retries_only_failed_part(self):
        self.bad_part = "What does it—"
        budget = SpeechRetryBudget()
        for _ in range(2):
            with self.assertRaises(SpeechVerificationFailed):
                self.recover(retry_budget=budget)
        self.assertEqual(len(self.calls), 3)  # One good part and two failed takes.
        marker = read_json(self.parent / "speech-recovery.json")
        final = self.parent.parent / marker["recovery_request_hash"]
        self.assertFalse((final / "accepted.json").exists())
        self.assertFalse((final / "attempt-001.wav").exists())
        self.assertTrue(next((final / "parts").glob("*/accepted.json")).is_file())
        self.bad_part = None
        wav, accepted = self.recover(retry_budget=SpeechRetryBudget())
        self.assertTrue(accepted["report"]["passed"])
        self.assertTrue(wav.is_file())
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(sum(text.startswith("Insang") for text, _ in self.calls), 1)
        self.assertEqual(len({seed for text, seed in self.calls if text == "What does it—"}), 3)

    def test_changed_attempt_allowance_keeps_saved_plan_and_good_parts(self):
        self.bad_part = "What does it—"
        with self.assertRaises(SpeechVerificationFailed):
            self.recover()
        marker = read_json(self.parent / "speech-recovery.json")
        self.config["render"]["max_attempts"] = 3
        self.assertTrue(recovery_available(self.parent, self.request, TEXT, self.verifier, self.config))
        self.bad_part = None
        _, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(read_json(self.parent / "speech-recovery.json"), marker)
        self.assertEqual(sum(text.startswith("Insang") for text, _ in self.calls), 1)

    def test_full_assembly_mismatch_cannot_publish_or_repeat_same_concat(self):
        self.verifier.final_transcript = "Insang Mortuary Services! What it—"
        for _ in range(2):
            with self.assertRaisesRegex(SpeechVerificationFailed, "final full-text verification"):
                self.recover()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 3)
        marker = read_json(self.parent / "speech-recovery.json")
        final = self.parent.parent / marker["recovery_request_hash"]
        self.assertFalse((final / "accepted.json").exists())
        self.assertEqual(read_json(final / "attempt-001.json")["report"]["reasons"], ["text_mismatch"])

    def test_pending_final_confirmation_rechecks_saved_waveform_without_regeneration(self):
        self.verifier.final_transcript = TEXT.replace("Mortuary", "Mórtuary")
        judge = Judge("unavailable")
        wrapped = ReferenceTextJudgingVerifier(self.verifier, judge)
        with self.assertRaises(VerificationUnavailable):
            self.recover(verifier=wrapped)
        marker = read_json(self.parent / "speech-recovery.json")
        wav = self.parent.parent / marker["recovery_request_hash"] / "attempt-001.wav"
        previous = wav.read_bytes()
        self.verifier.final_transcript = None
        reused, accepted = self.recover(verifier=wrapped)
        self.assertEqual(reused, wav)
        self.assertEqual(wav.read_bytes(), previous)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 4)
        self.assertTrue(accepted["report"]["passed"])

    def test_transient_final_asr_failure_rechecks_saved_waveform_without_regeneration(self):
        self.verifier.final_error = RuntimeError("Recognizer temporarily unavailable")
        with self.assertRaisesRegex(VerificationUnavailable, "rerun to retry verification"):
            self.recover()
        marker = read_json(self.parent / "speech-recovery.json")
        directory = self.parent.parent / marker["recovery_request_hash"]
        wav = directory / "attempt-001.wav"
        previous = wav.read_bytes()
        record = read_json(directory / "attempt-001.json")
        self.assertTrue(record["report"]["retry_on_resume"])
        self.assertFalse((directory / "accepted.json").exists())
        self.verifier.final_error = None
        reused, accepted = self.recover()
        self.assertEqual(reused, wav)
        self.assertEqual(wav.read_bytes(), previous)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 4)
        self.assertTrue(accepted["report"]["passed"])

    def test_repeated_interrupted_normalization_discards_only_orphan_assembly_raw(self):
        normalization_calls = []

        def fail_final_normalization(source, target, **kwargs):
            with wave.open(str(source), "rb") as audio:
                is_final = audio.getnframes() > 44100
            if is_final:
                normalization_calls.append(Path(source))
                raise RuntimeError("Normalizer temporarily unavailable")
            return normalize_clip(source, target, **kwargs)

        with patch("epub_tts_narrator.renderer.workflow.normalize_clip", fail_final_normalization):
            for _ in range(3):
                with self.assertRaisesRegex(VerificationUnavailable, "rerun to retry verification"):
                    self.recover()
                marker = read_json(self.parent / "speech-recovery.json")
                directory = self.parent.parent / marker["recovery_request_hash"]
                self.assertFalse((directory / "attempt-001.wav").exists())
                self.assertFalse((directory / "accepted.json").exists())
                self.assertEqual([path.name for path in directory.glob("*.wav")],
                                 ["attempt-001-raw.wav"])
                self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(normalization_calls), 3)
        wav, accepted = self.recover()
        self.assertEqual(wav.parent, directory)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 3)
        self.assertFalse((directory / "attempt-001-recovery-raw.wav").exists())

    def test_marker_tampering_and_changed_audio_are_rejected(self):
        wav, accepted = self.recover()
        marker_path = self.parent / "speech-recovery.json"
        marker = read_json(marker_path)
        changed = deepcopy(marker)
        changed["recovery_request_hash"] = "../unrelated"
        write_json(marker_path, changed)
        with self.assertRaises(RenderBlocked):
            recovery_available(self.parent, self.request, TEXT, self.verifier, self.config)
        with self.assertRaises(RenderBlocked):
            self.recover()
        write_json(marker_path, marker)
        wav.write_bytes(wav.read_bytes() + b"changed")
        with self.assertRaisesRegex(RenderBlocked, "changed"):
            self.recover()
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
