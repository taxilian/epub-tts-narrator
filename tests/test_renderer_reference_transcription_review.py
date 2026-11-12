import unittest
from pathlib import Path
import shutil
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.quality import validation_key
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier, assemble_verifier
from epub_tts_narrator.renderer.workflow import RenderBlocked, verified_take
from shared_fixtures import renderer_config
from test_renderer_workflow import FFMPEG
from test_renderer_jev_coverage import CONTEXT
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_review import Client
from test_renderer_reference_consensus import IndependentVerifier, PrimaryVerifier
import test_renderer_reference_consensus as consensus


class ReferenceTranscriptionReviewTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_unaligned_reference_substitutions_reach_sound_review_before_rejection(self):
        examples = (
            ("Dalley checked the records.", "Darly checked the records."),
            ("Dalley checked the records.", "Darling checked the records."),
            ("I assume you're here about the records.", "I assume you hear about the records."),
            ("You and probably the captain knew.", "You I'm probably the captain knew."),
            ("The firepower she carried was impressive.", "The fire tower she carried was impressive."),
        )
        for expected, heard in examples:
            with self.subTest(expected=expected, heard=heard):
                client = Client((True, False, True))
                verifier = ReferenceTextJudgingVerifier(
                    PrimaryVerifier(heard + CONTEXT), Judge(), PronunciationReviewer(client))

                report = self.check(verifier, expected=expected + CONTEXT)

                self.assertTrue(report["passed"], (report["reasons"], report.get("text_judge_skipped")))
                self.assertEqual(len(client.calls), 3)
                self.assertEqual(report["reference_text"], heard + CONTEXT)
                self.assertEqual(report["expected_text"], expected + CONTEXT)
                self.assertEqual(report["phonetic_reviews"][-1]["decision"], "accepted")
                self.assertEqual(len(report["phonetic_reviews"][-1]["votes"]), 3)

    def test_negative_sound_votes_reject_the_reference_and_keep_the_audit(self):
        client = Client((False, True, False))
        expected = "Dalley checked the records." + CONTEXT
        heard = "Darling checked the records." + CONTEXT
        verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(heard), Judge(), PronunciationReviewer(client))

        report = self.check(verifier, expected=expected)

        self.assertFalse(report["passed"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(report["phonetic_reviews"][-1]["decision"], "rejected")
        self.assertNotIn("reference_text", report)

    def test_reference_nominations_do_not_apply_to_audiobook_speech(self):
        client = Client()
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier("Darling checked the records." + CONTEXT), Judge(), PronunciationReviewer(client))

        report = self.check(verifier, expected="Dalley checked the records." + CONTEXT, purpose="speech")

        self.assertFalse(report["passed"])
        self.assertEqual(client.calls, [])

    def test_reference_sound_review_works_without_jev_and_does_not_change_speech_keys(self):
        expected = "Dalley checked the records." + CONTEXT
        heard = "Darling checked the records." + CONTEXT
        primary = PrimaryVerifier(heard)

        def approve(prompt, payload):
            return {"pairs": [{"id": pair["id"], "plausible": True, "reason": "A plausible ASR spelling."}
                              for pair in payload["pairs"]]}

        with patch("epub_tts_narrator.provider.OpenAICompatibleClient.complete", side_effect=approve) as complete:
            verifier = assemble_verifier(
                primary, {"phonetic_fallback": "analysis", "recurring_resolution": "off"}, None,
                {"model": "fixture", "base_url": "https://offline.invalid/v1"},
                lambda: self.fail("No secondary model should be constructed"))
            report = self.check(verifier, expected=expected)

        self.assertTrue(report["passed"], report.get("reasons"))
        self.assertEqual(complete.call_count, 3)
        self.assertEqual(report["reference_text"], heard)
        self.assertEqual(verifier.identity, primary.identity)
        self.assertEqual(validation_key(expected, verifier), validation_key(expected, primary))
        speech_key = validation_key(expected, verifier)
        reference_key = validation_key(expected, verifier, purpose="reference")
        verifier.reference_phonetic_review_identity = dict(verifier.reference_phonetic_review_identity, version=99)
        self.assertEqual(validation_key(expected, verifier), speech_key)
        self.assertNotEqual(validation_key(expected, verifier, purpose="reference"), reference_key)

    def test_an_existing_negative_phonetic_vote_is_not_retried_as_a_reference_nomination(self):
        client = Client((False, False, False))
        expected = "Rigallo Lucrani's records were on the desk." + CONTEXT
        heard = "Regalo Lucani's records were on the desk." + CONTEXT
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier(heard), Judge("different"), PronunciationReviewer(client))

        report = self.check(verifier, expected=expected)

        self.assertFalse(report["passed"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(report["phonetic_reviews"]), 1)
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "rejected")

    def test_unavailable_reference_review_is_retained_as_pending_confirmation(self):
        from epub_tts_narrator.renderer.quality import text_confirmation_unavailable
        client = Client((RuntimeError("private provider details"),))
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier("Darling checked the records." + CONTEXT), Judge(), PronunciationReviewer(client))

        report = self.check(verifier, expected="Dalley checked the records." + CONTEXT)

        self.assertFalse(report["passed"])
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(text_confirmation_unavailable(report))
        self.assertEqual(report["phonetic_reviews"][-1]["decision"], "unavailable")

    def test_missing_or_disabled_reference_reviewer_keeps_the_existing_rejection(self):
        expected = "Dalley checked the records." + CONTEXT
        heard = "Darling checked the records." + CONTEXT
        missing = ReferenceTextJudgingVerifier(PrimaryVerifier(heard), Judge())
        disabled = assemble_verifier(
            PrimaryVerifier(heard), {"phonetic_fallback": "off", "recurring_resolution": "off"}, None,
            {"model": "fixture", "base_url": "https://offline.invalid/v1"},
            lambda: self.fail("No secondary model should be constructed"))

        for verifier in (missing, disabled):
            with self.subTest(verifier=type(verifier).__name__):
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"])
                self.assertNotIn("phonetic_reviews", report)

    def test_reviewed_secondary_transcript_is_saved_without_hiding_the_primary_omission(self):
        expected = "Dalley checked the records." + CONTEXT
        primary = "Checked the records." + CONTEXT
        secondary = "Darling checked the records." + CONTEXT
        for use_jev in (False, True):
            with self.subTest(use_jev=use_jev):
                client = Client((True, False, True))
                verifier = ReferenceTextJudgingVerifier(
                    IndependentVerifier(primary, secondary), Judge() if use_jev else None,
                    PronunciationReviewer(client))

                report = self.check(verifier, expected=expected)

                self.assertTrue(report["passed"], report["reasons"])
                self.assertEqual(len(client.calls), 3)
                self.assertEqual(report["transcript"], primary)
                self.assertEqual(report["reference_text"], secondary)
                self.assertEqual(report["verified_transcript"], secondary)
                self.assertEqual(report["phonetic_reviews"][-1]["source"], "secondary")
                self.assertEqual(report["corroboration"]["decision"], "phonetic_confirmed_variations")

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for retained-reference normalization")
    def test_new_reference_policy_reviews_saved_rejection_without_new_generation(self):
        root = Path(self.temp.name)
        expected = "Dalley checked the records." + CONTEXT
        heard = "Darling checked the records." + CONTEXT
        client = Client((True, False, True))
        reviewer = PronunciationReviewer(client)
        legacy = ReferenceTextJudgingVerifier(PrimaryVerifier(heard), Judge(), reviewer)
        legacy.reference_phonetic_review_identity = None
        config = renderer_config(root, series_id="reference-review", max_attempts=1)
        request = {"kind": "voice_design", "voice_id": "reference-review", "text": expected,
                   "engine": {"engine": "fixture", "model": "fixture"}}
        directory = root / "saved-reference"

        def generate(path, seed):
            shutil.copyfile(self.audio, path)
            return {"seed": seed, "description_applied": True}

        with patch("epub_tts_narrator.renderer.quality.REFERENCE_VARIATION_VERSION", 6):
            with self.assertRaises(RenderBlocked):
                verified_take(expected, generate, directory, request, legacy, config,
                              purpose="reference", notify=lambda _: None)
        self.assertEqual(client.calls, [])
        waveforms = {path.name: path.read_bytes() for path in directory.glob("*.wav")}
        current = ReferenceTextJudgingVerifier(PrimaryVerifier(heard), Judge(), reviewer)
        no_generation = Mock(side_effect=AssertionError("Retained audio must be reviewed before new TTS"))

        wav, accepted = verified_take(expected, no_generation, directory, request, current, config,
                                     purpose="reference", notify=lambda _: None)

        no_generation.assert_not_called()
        self.assertEqual(wav.read_bytes(), waveforms[wav.name])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["reference_text"], heard)
        self.assertEqual(len(client.calls), 3)

    def test_missing_extra_reordered_and_protected_words_do_not_reach_sound_review(self):
        examples = (
            ("Please wait here.", "Please wait."),
            ("She had had enough.", "She had enough."),
            ("The captain opened the gate.", "The captain opened the gate and called inside."),
            ("The captain checked the door.", "The door checked the captain."),
            ("I will not open the door.", "I will open the door."),
            ("Bring seventeen maps.", "Bring seventy maps."),
        )
        for expected, heard in examples:
            with self.subTest(expected=expected, heard=heard):
                client = Client()
                verifier = ReferenceTextJudgingVerifier(
                    PrimaryVerifier(heard + CONTEXT), Judge(), PronunciationReviewer(client))

                report = self.check(verifier, expected=expected + CONTEXT)

                self.assertFalse(report["passed"], report)
                self.assertEqual(client.calls, [])
