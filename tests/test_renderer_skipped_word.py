"""A source word every take leaves out passes only when the text judge calls it a source error."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.common import write_json
from epub_tts_narrator.renderer.skipped_word import POLICY, apply, find
from epub_tts_narrator.renderer.workflow import RenderBlocked, verified_take


EXPECTED = "Sal built a realised train, a labour of love, when it he finally could."
HEARD = "Sal built a realized train, a labor of love, when he finally could."
IDENTITY = {"provider": "fake", "policy_version": 1, "threshold": 0.8}


def failed_report(transcript=HEARD, expected=EXPECTED):
    return {"passed": False, "purpose": "speech", "expected_text": expected, "transcript": transcript,
            "reasons": ["text_mismatch"], "metrics": {}, "words": [],
            "text_judge": {"decision": "different", "error_code": "incomplete_spoken_coverage"},
            "text_judge_skipped": "incomplete_spoken_coverage"}


class Judge:
    reference_judge_identity = IDENTITY

    def __init__(self, decision="source_error"):
        self.decision, self.calls = decision, []

    def judge_skipped_word(self, expected, transcript, word, context):
        self.calls.append((expected, transcript, word, context))
        return {"decision": self.decision, "probability": .95, "identity": IDENTITY}


class FindTests(unittest.TestCase):
    def test_finds_the_one_left_out_word_while_spellings_elsewhere_match_by_sound(self):
        found = find(EXPECTED, HEARD)
        self.assertEqual(found["word"], "it")
        self.assertEqual(found["context"], "labour of love when [it] he finally could")
        self.assertEqual(found["method"], "double_metaphone")

    def test_numbers_negations_and_more_than_one_missing_word_never_qualify(self):
        self.assertIsNone(find("He did not go home.", "He did go home."))
        self.assertIsNone(find("She ate three apples.", "She ate apples."))
        self.assertIsNone(find("We went down to the old mill.", "We went to the mill."))
        self.assertIsNone(find("We went home.", "We went home."))
        self.assertIsNone(find("We went home.", "They ran away."))

    def test_a_doubled_word_is_one_candidate(self):
        self.assertEqual(find("We went to the the store.", "We went to the store.")["word"], "the")


class ApplyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def earlier(self, number, transcript=HEARD, audio=None, request_hash="request"):
        write_json(self.directory / f"attempt-{number:03d}.json",
                   {"request_hash": request_hash, "attempt": number, "audio_sha256": audio or f"{number:064x}",
                    "report": failed_report(transcript)})

    def run_apply(self, judge, report=None, audio="f" * 64):
        return apply(report or failed_report(), directory=self.directory, verifier=judge,
                     request_hash="request", audio_sha256=audio)

    def test_two_takes_skipping_the_same_word_and_a_source_error_judgment_accept(self):
        self.earlier(1)
        judge = Judge()
        report, outcome = self.run_apply(judge)
        self.assertTrue(report["passed"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["previous_reasons"], ["text_mismatch"])
        self.assertEqual(report["text_confirmation_method"], "skipped_source_word")
        self.assertEqual(report["skipped_source_word"]["word"], "it")
        self.assertEqual(len(report["skipped_source_word"]["takes"]), 2)
        self.assertIn("accepted", outcome)
        self.assertEqual(judge.calls, [(EXPECTED, HEARD, "it", "labour of love when [it] he finally could")])

    def test_one_take_alone_waits_without_asking_the_judge(self):
        judge = Judge()
        report, outcome = self.run_apply(judge)
        self.assertFalse(report["passed"])
        self.assertIn("pending", outcome)
        self.assertEqual(judge.calls, [])

    def test_a_take_that_read_the_word_or_another_request_does_not_count(self):
        self.earlier(1, transcript=EXPECTED.replace("realised", "realized"))
        judge = Judge()
        report, outcome = self.run_apply(judge)
        self.assertFalse(report["passed"])
        self.assertIsNone(outcome)
        (self.directory / "attempt-001.json").unlink()
        self.earlier(1, request_hash="another")
        self.earlier(2, audio="f" * 64)  # the same audio again is not another take
        report, outcome = self.run_apply(judge)
        self.assertIn("pending", outcome)
        self.assertEqual(judge.calls, [])

    def test_clip_repair_takes_neither_count_nor_contradict(self):
        self.earlier(1)
        record = json.loads((self.directory / "attempt-001.json").read_text())
        for number, transcript in ((2, EXPECTED), (3, HEARD)):
            write_json(self.directory / f"attempt-00{number}.json",
                       dict(record, attempt=number, audio_sha256=f"{number:064x}",
                            report=failed_report(transcript), engine_metadata={"clip_repair": {}}))
        report, outcome = self.run_apply(Judge())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["skipped_source_word"]["takes"]), 2)
        untouched = failed_report()
        self.assertEqual(apply(untouched, directory=self.directory, verifier=Judge(), request_hash="request",
                               audio_sha256="e" * 64, engine_metadata={"speech_resolution": {}}),
                         (untouched, None))

    def test_a_word_the_judge_keeps_or_an_invalid_judgment_still_fails(self):
        self.earlier(1)
        for judge in (Judge("belongs"), Judge("uncertain"), Judge("unavailable")):
            with self.subTest(decision=judge.decision):
                report, outcome = self.run_apply(judge)
                self.assertFalse(report["passed"])
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertIn("rejected", outcome)
        stranger = Judge()
        stranger.reference_judge_identity = dict(IDENTITY, provider="other")
        report, outcome = self.run_apply(stranger)
        self.assertFalse(report["passed"])
        self.assertEqual(report["skipped_source_word"]["judgment"]["decision"], "unavailable")

    def test_other_failures_and_verifiers_without_a_judge_are_untouched(self):
        self.earlier(1)
        for change in ({"reasons": ["text_mismatch", "audio_too_quiet"]}, {"history_echo": {}},
                       {"text_judge_skipped": None}, {"passed": True}):
            with self.subTest(change=change):
                report = dict(failed_report(), **change)
                self.assertEqual(self.run_apply(Judge(), report), (report, None))
        self.assertEqual(self.run_apply(object()), (failed_report(), None))


class JevSkippedWordTests(unittest.TestCase):
    def judge(self, probability):
        judge = JevTranscriptJudge({"enabled": True, "api_key": "test-key", "threshold": .8})
        state = {}

        def evaluate(sent, questions):
            state.update(sent=sent, questions=questions)
            return {"model": "jev-latest", "answers": {"source_error": {"type": "noul", "noul": probability}},
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        judge._client.evaluate = evaluate
        return judge.judge_skipped_word(EXPECTED, HEARD, "it", "when [it] he"), state

    def test_scores_map_to_decisions_and_the_state_carries_only_text(self):
        for probability, decision in ((.9, "source_error"), (.1, "belongs"), (.5, "uncertain")):
            with self.subTest(probability=probability):
                record, state = self.judge(probability)
                self.assertEqual(record["decision"], decision)
                self.assertEqual(set(state["sent"]), {"requested_passage", "skipped_word",
                                                      "skipped_context", "recognized_passage"})
                self.assertEqual(list(state["questions"]), ["source_error"])

    def test_disabled_judge_never_approves(self):
        record = JevTranscriptJudge({"enabled": False}).judge_skipped_word(EXPECTED, HEARD, "it", "x")
        self.assertEqual(record["decision"], "unavailable")


class TakeLoopTests(unittest.TestCase):
    def test_the_second_take_skipping_the_same_typo_is_accepted_with_an_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            messages, judge = [], Judge()

            def generate(path, seed):
                Path(path).write_bytes(f"take {seed}".encode())
                return {"seed": seed}

            def normalize(raw, wav, **kwargs):
                Path(wav).write_bytes(Path(raw).read_bytes())

            config = {"quality": {}, "render": {"max_attempts": 3, "seed": 42, "sample_rate": 44100,
                                                 "ffmpeg": "unused"}}
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("epub_tts_narrator.renderer.workflow.validate_audio",
                          side_effect=lambda *args, **kwargs: deepcopy(failed_report())), \
                    patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize):
                verified_take(EXPECTED, generate, temp, {"test": True}, judge, config, notify=messages.append)
            accepted = json.loads((Path(temp) / "accepted.json").read_text())
            self.assertEqual(accepted["attempt"], 2)
            self.assertEqual(accepted["report"]["text_confirmation_method"], "skipped_source_word")
            self.assertEqual(len(judge.calls), 1)
            self.assertTrue(any("Left-out source word noted" in line for line in messages), messages)
            self.assertTrue(any("Accepted with one source word left out (labour of love when [it] he finally could)" in line
                                for line in messages), messages)

    def test_without_a_source_error_judgment_every_take_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            def generate(path, seed):
                Path(path).write_bytes(f"take {seed}".encode())
                return {"seed": seed}

            def normalize(raw, wav, **kwargs):
                Path(wav).write_bytes(Path(raw).read_bytes())

            config = {"quality": {}, "render": {"max_attempts": 2, "seed": 42, "sample_rate": 44100,
                                                 "ffmpeg": "unused"}}
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("epub_tts_narrator.renderer.workflow.validate_audio",
                          side_effect=lambda *args, **kwargs: deepcopy(failed_report())), \
                    patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize):
                with self.assertRaises(RenderBlocked):
                    verified_take(EXPECTED, generate, temp, {"test": True}, Judge("belongs"), config,
                                  notify=lambda message: None)
            self.assertFalse((Path(temp) / "accepted.json").exists())


if __name__ == "__main__":
    unittest.main()
