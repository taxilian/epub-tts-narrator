"""Real verification flow resolves bounded sound disputes without waiving coverage."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.quality import text_confirmation_unavailable
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, VerificationUnavailable, verified_take,
)
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_fallback import PairClient
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import IndependentVerifier, PrimaryVerifier
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


REQUESTED = "Insang Mortuary Services! What does it—"
RECOGNIZED = "In saying Mortuary Services! What does it—"


class AdaptiveVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=RECOGNIZED, *, secondary=None, votes=(True, False, True)):
        recognizer = (PrimaryVerifier(actual) if secondary is None else
                      IndependentVerifier(primary=actual, secondary=secondary))
        client = PairClient(votes)
        judge = Judge("different")
        verifier = ReferenceTextJudgingVerifier(
            recognizer, judge, PronunciationReviewer(client))
        return verifier, recognizer, judge, client

    def test_rejected_split_name_gets_three_independent_sound_votes_and_retains_audit(self):
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                verifier, recognizer, judge, client = self.wrapped()
                report = self.check(verifier, expected=REQUESTED, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "phonetic_primary")
                self.assertEqual(report["expected_text"], REQUESTED)
                self.assertEqual(report["transcript"], RECOGNIZED)
                self.assertEqual(report["raw_transcript"], RECOGNIZED)
                self.assertEqual(report["verified_transcript"], RECOGNIZED)
                self.assertGreater(report["metrics"]["wer"], 0)
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertEqual(report["text_judge"]["probabilities"]["roughly_same"], .01)
                self.assertEqual(len(judge.calls), 1)
                self.assertEqual(len(client.calls), 3)
                self.assertEqual(client.calls[0], client.calls[1])
                self.assertEqual(client.calls[1], client.calls[2])
                self.assertEqual(set(client.calls[0][1]), {"pairs"})
                pair = client.calls[0][1]["pairs"][0]
                self.assertEqual(pair["expected_word"], "Insang")
                self.assertEqual(pair["recognized_word"], "In saying")
                self.assertEqual((pair["expected_start"], pair["expected_end"]), (0, 1))
                self.assertEqual((pair["actual_start"], pair["actual_end"]), (0, 2))
                audit = report["phonetic_reviews"][0]
                self.assertEqual(audit["decision"], "accepted")
                self.assertEqual(audit["source"], "primary")
                self.assertEqual([vote["pairs"][0]["plausible"] for vote in audit["votes"]],
                                 [True, False, True])
                self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                if purpose == "reference":
                    self.assertEqual(report["reference_text"], RECOGNIZED)
                else:
                    self.assertNotIn("reference_text", report)

    def test_one_sound_vote_cannot_override_judge_rejection(self):
        verifier, _, _, client = self.wrapped(votes=(False, True, False))
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "rejected")
        self.assertEqual(len(client.calls), 3)
        self.assertFalse(text_confirmation_unavailable(report))

    def test_changed_verb_and_coverage_failures_never_reach_even_unanimous_jury(self):
        cases = (
            (REQUESTED, RECOGNIZED.replace("does", "is")),
            (REQUESTED, RECOGNIZED.replace("Mortuary ", "")),
            (REQUESTED, RECOGNIZED + " Thank you."),
            (REQUESTED, RECOGNIZED.replace("Mortuary", "Mortuary Mortuary")),
            ("Insang did not close the door.", "In saying did close the door."),
            ("Insang did not close the door.", "In saying did now close the door."),
            ("Insang found fifteen coins in the box.",
             "In saying found fifty coins in the box."),
            ("Insang carried the second box.", "In saying carried the third box."),
        )
        for expected, actual in cases:
            with self.subTest(expected=expected, actual=actual):
                verifier, _, judge, client = self.wrapped(actual, votes=(True, True, True))
                report = self.check(verifier, expected=expected, purpose="speech")
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertEqual(judge.calls, [])
                self.assertEqual(client.calls, [])
                self.assertNotIn("phonetic_reviews", report)
                self.assertFalse(text_confirmation_unavailable(report))

    def test_secondary_split_name_rescues_failed_primary_and_discards_primary_timings(self):
        primary = RECOGNIZED.replace("does", "is")
        verifier, recognizer, judge, client = self.wrapped(primary, secondary=RECOGNIZED)
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["text_confirmation_method"], "phonetic_secondary")
        self.assertEqual(report["expected_text"], REQUESTED)
        self.assertEqual(report["transcript"], primary)
        self.assertEqual(report["raw_transcript"], primary)
        self.assertEqual(report["verified_transcript"], RECOGNIZED)
        self.assertGreater(report["metrics"]["wer"], 0)
        self.assertEqual(report["words"], [])
        self.assertEqual(report["word_timing_status"], "unavailable")
        self.assertEqual(report["phonetic_reviews"][0]["source"], "secondary")
        self.assertEqual(report["corroboration"]["transcript"], RECOGNIZED)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(recognizer.calls, [("speech", self.audio), ("corroboration", self.audio)])

    def test_secondary_recognition_cannot_rescue_a_still_missing_word(self):
        verifier, _, judge, client = self.wrapped(
            RECOGNIZED.replace("does", "is"), secondary=RECOGNIZED.replace("Mortuary ", ""))
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(judge.calls, [])
        self.assertEqual(client.calls, [])
        self.assertNotIn("phonetic_reviews", report)

    def test_sound_reviewer_outage_is_unavailable_even_after_definite_jev_rejection(self):
        verifier, _, _, client = self.wrapped()
        client.error = TimeoutError("temporary fixture outage")
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertTrue(text_confirmation_unavailable(report))
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "unavailable")
        self.assertEqual(len(client.calls), 1)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for saved-take integration")
class AdaptiveVerificationResumeTests(unittest.TestCase):
    def test_model_outage_pauses_then_resolves_the_same_audio_without_resynthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = deepcopy(DEFAULTS)
            config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE)
            generations = []

            def generate(path, seed):
                generations.append((path, seed))
                write_tone(path)
                return {"fixture": True, "seed": seed}

            client = PairClient()
            client.error = TimeoutError("temporary fixture outage")
            verifier = ReferenceTextJudgingVerifier(
                PrimaryVerifier(RECOGNIZED), Judge("different"), PronunciationReviewer(client))

            def take():
                return verified_take(
                    REQUESTED, generate, root / "takes",
                    {"kind": "speech", "text": REQUESTED, "engine": "fixture"},
                    verifier, config, notify=lambda _: None, retry_budget=SpeechRetryBudget())

            for _ in range(2):
                with self.assertRaises(VerificationUnavailable):
                    take()
            self.assertEqual(len(generations), 1)
            wav = root / "takes" / "attempt-001.wav"
            before = wav.read_bytes()
            client.error = None
            reused, accepted = take()
            self.assertEqual(reused, wav)
            self.assertEqual(wav.read_bytes(), before)
            self.assertEqual(len(generations), 1)
            self.assertTrue(accepted["report"]["passed"])
            self.assertEqual(accepted["report"]["text_confirmation_method"], "phonetic_primary")


if __name__ == "__main__":
    unittest.main()
