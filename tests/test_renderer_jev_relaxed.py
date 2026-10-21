"""Actual judge routing, with deterministic JEV responses and real PCM checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.verification import ReferenceTextJudgingVerifier
from audiobook_renderer.workflow import VerificationUnavailable, verified_take
from epub_audiobook.jev import JevTranscriptJudge
import test_renderer_reference_consensus as consensus


SCRIPT = (
    "Mmm... still warm here. Grandmother, is it morning yet? No? Good. "
    "I want to watch the light come through the snow-clouds a little longer. "
    "Everything in this place hums softly, like it's purring back at me. "
    "Wake me when the others arrive... maybe."
)


def answer(score):
    return {"model": "jev-fixture", "answers": {"roughly_same": {"type": "noul", "noul": score}},
            "usage": {"input_tokens": 30, "output_tokens": 2}}


class RelaxedJevTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_reported_hum_cases_reach_jev_for_samples_and_chapters(self):
        cases = [
            (SCRIPT.replace("Mmm... ", ""), SCRIPT.replace("Mmm... ", "")),
            (SCRIPT.replace("Mmm", "Hmm"), SCRIPT.replace("Mmm... still", "Hmm. It's still")),
        ]
        for purpose in ("reference", "speech"):
            for primary, secondary in cases:
                for score in (.99, .4):
                    with self.subTest(purpose=purpose, primary=primary, score=score):
                        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
                        recognizer = consensus.IndependentVerifier(primary, secondary)
                        verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                        with patch.object(judge._client, "evaluate", return_value=answer(score)) as evaluate:
                            report = self.check(verifier, expected=SCRIPT, purpose=purpose)
                        self.assertGreater(evaluate.call_count, 0)
                        self.assertEqual(report["passed"], score == .99, report)
                        self.assertEqual(report["text_judge"]["probabilities"]["roughly_same"], score)
                        self.assertEqual(report["transcript"], primary)
                        self.assertEqual(report["expected_text"], SCRIPT)
                        self.assertGreater(report["metrics"]["wer"], 0)
                        if purpose == "speech":
                            self.assertTrue(report["words"])
                            self.assertNotIn("reference_text", report)
                        elif score == .99:
                            self.assertEqual(report["reference_text"], primary)

    def test_missing_secondary_does_not_prevent_jev_decision(self):
        primary = SCRIPT.replace("Mmm... ", "")
        for recognizer in (consensus.PrimaryVerifier(primary), consensus.IndependentVerifier(primary, primary)):
            if isinstance(recognizer, consensus.IndependentVerifier):
                recognizer.secondary = RuntimeError("optional recognizer failed")
            judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
            verifier = ReferenceTextJudgingVerifier(recognizer, judge)
            with patch.object(judge._client, "evaluate", return_value=answer(.99)) as evaluate:
                report = self.check(verifier, expected=SCRIPT)
            self.assertTrue(report["passed"], report)
            self.assertGreater(evaluate.call_count, 0)

    def test_source_hyphen_is_not_proof_of_equivalent_spoken_words(self):
        source = "I will re-sign the agreement tomorrow."
        heard = "I will resign the agreement tomorrow."
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(heard), judge)
        with patch.object(judge._client, "evaluate", return_value=answer(.01)) as evaluate:
            report = self.check(verifier, expected=source, purpose="speech")
        self.assertFalse(report["passed"], report)
        evaluate.assert_called_once()
        self.assertEqual(report["text_judge"]["decision"], "different")

    def test_only_exact_text_and_acoustic_failures_skip_jev(self):
        for primary, score in ((SCRIPT, .99), ("Thank you.", .01),
                               (SCRIPT.removesuffix(" arrive... maybe."), .01)):
            judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
            verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(primary), judge)
            with patch.object(judge._client, "evaluate", return_value=answer(score)) as evaluate:
                report = self.check(verifier, expected=SCRIPT, purpose="speech")
            self.assertEqual(report["passed"], primary == SCRIPT, report)
            self.assertEqual(evaluate.call_count, 0 if primary == SCRIPT else 1)
        self.write_audio(amplitude=0)
        with patch.object(judge._client, "evaluate") as evaluate:
            report = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertFalse(report["passed"])
        evaluate.assert_not_called()

    def test_names_go_directly_to_jev_and_bad_secondary_cannot_veto(self):
        for heard in ("Chase!", "Jazz!", "Jace!", "A completely different phrase."):
            judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
            recognizer = consensus.IndependentVerifier(heard, "Chessed.")
            verifier = ReferenceTextJudgingVerifier(recognizer, judge)
            with patch.object(judge._client, "evaluate", return_value=answer(.99)) as evaluate:
                report = self.check(verifier, expected="Jas!", purpose="speech")
            self.assertTrue(report["passed"], report)
            evaluate.assert_called_once()
            self.assertEqual(recognizer.calls, [("speech", self.audio)])
            self.assertEqual(evaluate.call_args.args[0]["recognized_passage"], heard)
            self.assertNotIn("judgment_eligibility", report)

    def test_many_small_differences_need_only_one_whole_passage_judgment(self):
        source = " ".join(f"mmm word{index}" for index in range(40))
        heard = source.replace("mmm ", "")
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(heard), judge)
        with patch.object(judge._client, "evaluate", return_value=answer(.99)) as evaluate:
            report = self.check(verifier, expected=source, purpose="speech")
        self.assertTrue(report["passed"], report)
        evaluate.assert_called_once()
        self.assertEqual(report["text_judge"]["decision"], "equivalent")

    def test_jev_outage_resumes_same_chapter_audio_without_generating_again(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            def generate(path, seed):
                generated.append(path)
                Path(path).write_bytes(self.audio.read_bytes())
                return {"seed": seed}
            config = {"quality": {}, "render": {"max_attempts": 3, "seed": 42,
                       "sample_rate": 44100, "ffmpeg": "unused"}}
            judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
            verifier = ReferenceTextJudgingVerifier(
                consensus.PrimaryVerifier(SCRIPT.replace("Mmm... ", "")), judge)
            with patch("audiobook_renderer.workflow.normalize_clip",
                       side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())), \
                    patch.object(judge._client, "evaluate", side_effect=[TimeoutError(), answer(.99)]) as evaluate:
                with self.assertRaises(VerificationUnavailable):
                    verified_take(SCRIPT, generate, directory, {"fixture": True}, verifier, config, purpose="speech")
                self.assertEqual(len(generated), 1)
                saved = Path(directory, "attempt-001.wav").read_bytes()
                path, record = verified_take(SCRIPT, generate, directory, {"fixture": True},
                                             verifier, config, purpose="speech")
            self.assertEqual(len(generated), 1)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(path.read_bytes(), saved)
            self.assertTrue(record["report"]["passed"])


if __name__ == "__main__":
    unittest.main()
