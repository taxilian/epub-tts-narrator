"""Bounded pronunciation consensus resolves uncertainty without resynthesizing audio."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.quality import text_confirmation_unavailable, validation_key
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier, create_verifier
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, VerificationUnavailable, verified_take,
)
from test_renderer_jev_verification import Judge
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import PrimaryVerifier
from test_renderer_verification_factory import fake_primary
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


REQUESTED = "Derrins walked through the station and quietly closed the door behind him."
RECOGNIZED = REQUESTED.replace("Derrins", "Darin's")


class PairClient:
    """Provider double; the production reviewer still validates and counts votes."""

    def __init__(self, votes=(True, False, True), model="phonetic-fixture"):
        self.identity = {"model": model, "base_url": "https://offline.invalid/v1"}
        self.votes = votes
        self.calls = []
        self.error = None

    def complete(self, prompt, payload):
        self.calls.append((prompt, deepcopy(payload)))
        if self.error:
            raise self.error
        answer = self.votes[(len(self.calls) - 1) % len(self.votes)]
        return {"pairs": [{"id": pair["id"], "plausible": answer,
                           "reason": "Plausible alternate spelling of the spoken name."
                           if answer else "The pronunciations seem different."}
                          for pair in payload["pairs"]]}


class PhoneticFallbackTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=RECOGNIZED, *, decision="uncertain", votes=(True, False, True)):
        client = PairClient(votes)
        judge = Judge(decision)
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier(actual), judge, PronunciationReviewer(client))
        return verifier, judge, client

    def test_majority_resolves_uncertainty_for_speech_and_reference_with_original_audit(self):
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                verifier, judge, client = self.wrapped()
                report = self.check(verifier, expected=REQUESTED, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "phonetic_primary")
                self.assertEqual(report["transcript"], RECOGNIZED)
                self.assertEqual(report["expected_text"], REQUESTED)
                self.assertEqual(report["text_judge"]["decision"], "uncertain")
                self.assertEqual(report["text_judge"]["probabilities"]["roughly_same"], .6)
                self.assertGreater(report["metrics"]["wer"], 0)
                self.assertEqual(len(judge.calls), 1)
                self.assertEqual(len(client.calls), 3)
                audit = report["phonetic_reviews"][0]
                self.assertEqual(audit["decision"], "accepted")
                self.assertEqual(audit["source"], "primary")
                self.assertEqual([vote["pairs"][0]["plausible"] for vote in audit["votes"]],
                                 [True, False, True])
                self.assertEqual(audit["pairs"][0]["expected_word"], "Derrins")
                self.assertEqual(audit["pairs"][0]["recognized_word"], "Darin's")
                if purpose == "reference":
                    self.assertEqual(report["reference_text"], RECOGNIZED)
                else:
                    self.assertNotIn("reference_text", report)

    def test_one_positive_vote_does_not_resolve_uncertainty(self):
        verifier, _, client = self.wrapped(votes=(False, True, False))
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "rejected")
        self.assertEqual(len(client.calls), 3)
        self.assertFalse(text_confirmation_unavailable(report))
        from epub_tts_narrator.renderer.workflow import _verification_failure_detail
        self.assertIn("phonetic reviewers rejected", _verification_failure_detail(report))

    def test_exact_transcript_needs_neither_judge_nor_fallback(self):
        verifier, judge, client = self.wrapped(actual=REQUESTED)
        report = self.check(verifier, expected=REQUESTED)
        self.assertTrue(report["passed"], report)
        self.assertEqual(judge.calls, [])
        self.assertEqual(client.calls, [])
        self.assertNotIn("phonetic_reviews", report)

    def test_definite_jev_rejection_cannot_be_overridden(self):
        verifier, _, client = self.wrapped(decision="different")
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(client.calls, [])
        self.assertNotIn("phonetic_reviews", report)

    def test_word_coverage_negation_and_quantity_guards_precede_fallback(self):
        cases = (
            (REQUESTED, RECOGNIZED + " Thank you."),
            (REQUESTED, RECOGNIZED.replace("quietly ", "")),
            ("Derrins did not close the door.", "Darin's did close the door."),
            ("Derrins did not close the door.", "Darin's did now close the door."),
            ("Derrins found fifteen coins inside the box.",
             "Darin's found fifty coins inside the box."),
        )
        for expected, actual in cases:
            with self.subTest(actual=actual):
                verifier, _, client = self.wrapped(actual=actual)
                report = self.check(verifier, expected=expected, purpose="speech")
                self.assertFalse(report["passed"], report)
                self.assertEqual(client.calls, [])
                self.assertNotIn("phonetic_reviews", report)

    def test_provider_failure_is_unavailable_with_no_secret_in_report(self):
        verifier, _, client = self.wrapped()
        client.error = TimeoutError("PRIVATE_PROVIDER_CREDENTIAL")
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertTrue(text_confirmation_unavailable(report))
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "unavailable")
        self.assertNotIn("PRIVATE_PROVIDER_CREDENTIAL", json.dumps(report))


class PhoneticFallbackFactoryTests(unittest.TestCase):
    settings = {"model": "/unused/whisper", "language": "en"}
    jev = {"enabled": True, "api_key": "private-jev-fixture"}
    provider = {"model": "analysis-model", "base_url": "https://offline.invalid/v1",
                "api_key": "private-analysis-fixture"}

    def make(self, *, settings=None, provider=None):
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier",
                   return_value=fake_primary()):
            return create_verifier(settings or self.settings, self.jev, provider)

    def test_configured_analysis_provider_is_reused_without_credentials_in_identity(self):
        original = self.make(provider=self.provider)
        self.assertIsInstance(original.phonetic_reviewer, PronunciationReviewer)
        provider = original.phonetic_review_identity["provider"]
        self.assertEqual(provider["model"], self.provider["model"])
        self.assertEqual(provider["base_url"], self.provider["base_url"])
        self.assertNotIn("private-", json.dumps(original.identity))
        changed = self.make(provider={**self.provider, "model": "analysis-model-v2"})
        self.assertNotEqual(validation_key(REQUESTED, original), validation_key(REQUESTED, changed))
        rotated = self.make(provider={**self.provider, "api_key": "other-private-key"})
        self.assertEqual(validation_key(REQUESTED, original), validation_key(REQUESTED, rotated))

    def test_no_provider_and_explicit_off_do_not_create_fallback(self):
        for provider, settings in ((None, self.settings),
                                   (self.provider, {**self.settings, "phonetic_fallback": "off"})):
            with self.subTest(provider=provider is not None):
                verifier = self.make(provider=provider, settings=settings)
                self.assertIsNone(verifier.phonetic_reviewer)
                self.assertNotIn("phonetic_review", verifier.identity)

    def test_renderer_reads_shared_provider_section_from_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renderer.toml"
            path.write_text('[provider]\nmodel = "analysis-model"\n'
                            'base_url = "https://offline.invalid/v1"\n'
                            'api_key = "private-analysis-fixture"\n'
                            '[engine]\ncheckpoint = "/unused/drama-3"\n'
                            '[verification]\nphonetic_fallback = "off"\n', encoding="utf-8")
            config = load_config(path, require_verifier=False)
        self.assertEqual(config["provider"]["model"], self.provider["model"])
        self.assertEqual(config["provider"]["api_key"], self.provider["api_key"])
        self.assertEqual(config["verification"]["phonetic_fallback"], "off")

    def test_invalid_fallback_setting_reports_a_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renderer.toml"
            for setting in ('[]', '{}', 'true', '"unknown"'):
                with self.subTest(setting=setting):
                    path.write_text('[verification]\nphonetic_fallback = ' + setting + '\n')
                    with self.assertRaisesRegex(ValueError, "verification.phonetic_fallback"):
                        load_config(path, require_verifier=False)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for saved-take integration")
class PhoneticFallbackResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.calls = []
        self.request = {"kind": "speech", "text": REQUESTED, "engine": "fixture"}

    def generate(self, path, seed):
        self.calls.append((path, seed))
        write_tone(path)
        return {"fixture": True, "seed": seed}

    def take(self, verifier):
        return verified_take(REQUESTED, self.generate, self.root / "takes", self.request,
                             verifier, self.config, notify=lambda _: None,
                             retry_budget=SpeechRetryBudget())

    def test_new_policy_rechecks_saved_failure_without_new_synthesis(self):
        judge = Judge("uncertain")
        before = ReferenceTextJudgingVerifier(PrimaryVerifier(RECOGNIZED), judge)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(before)
        wav = self.root / "takes" / "attempt-001.wav"
        old_bytes = wav.read_bytes()
        old_report = read_json(wav.with_suffix(".json"))
        reviewer = PronunciationReviewer(PairClient())
        after = ReferenceTextJudgingVerifier(PrimaryVerifier(RECOGNIZED), judge, reviewer)
        reused, accepted = self.take(after)
        self.assertEqual(reused, wav)
        self.assertEqual(reused.read_bytes(), old_bytes)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["text_confirmation_method"], "phonetic_primary")
        self.assertNotEqual(accepted["validation_key"], old_report["validation_key"])

    def test_provider_outage_pauses_without_new_takes_and_resume_rechecks_same_waveform(self):
        client = PairClient()
        client.error = TimeoutError("temporary fixture outage")
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier(RECOGNIZED), Judge("uncertain"), PronunciationReviewer(client))
        for _ in range(2):
            with self.assertRaises(VerificationUnavailable):
                self.take(verifier)
        self.assertEqual(len(self.calls), 1)
        wav = self.root / "takes" / "attempt-001.wav"
        old_bytes = wav.read_bytes()
        client.error = None
        reused, accepted = self.take(verifier)
        self.assertEqual(reused, wav)
        self.assertEqual(reused.read_bytes(), old_bytes)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(accepted["report"]["passed"])


if __name__ == "__main__":
    unittest.main()
