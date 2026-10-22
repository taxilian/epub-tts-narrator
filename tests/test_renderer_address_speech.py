"""Address reading conventions reach JEV without hiding missing content."""

import unittest
import subprocess
import sys
from unittest.mock import patch

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.quality import validate_audio
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.address_speech import address_spans, address_spelling_equivalent
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
import test_renderer_reference_consensus as consensus


EMAIL = "author.sample.writer.email@example.com"
URL = "https://www.example.com/gp/product/B07KJSRVMS/"
SPELLED = ("h t t p s colon slash slash w w w dot example dot com slash g p slash product slash "
           "bee zero seven kay jay ess are vee em ess slash")


class AddressCoverageTests(unittest.TestCase):
    def test_whole_address_helper_rejects_partial_spans_and_added_prose(self):
        self.assertTrue(address_spelling_equivalent(URL, "  " + SPELLED + "  "))
        self.assertTrue(address_spelling_equivalent(EMAIL, "author.sample.writer.email at example.com"))
        for source, replacement in (
            ("Visit " + URL, "Visit " + SPELLED),
            (URL + " today", SPELLED + " today"),
            (URL, "Visit " + SPELLED),
            (URL, SPELLED + " today"),
            (URL, "www.example.com.gp.product.bo7kjsrvms.com"),
            ("B07KJSRVMS", "bee zero seven kay jay ess are vee em ess"),
            (URL, SPELLED.replace("zero", "one")),
            ("https://example.com/" + "a" * 1024, "a"),
            (URL, "a" * 16_385),
        ):
            with self.subTest(source=source, replacement=replacement[:100]):
                self.assertFalse(address_spelling_equivalent(source, replacement))

    def test_address_spans_preserve_original_offsets_and_separate_prose(self):
        text = "Email " + EMAIL + ", then visit " + URL + ". Thank you."
        spans = address_spans(text)
        self.assertEqual([row["source"] for row in spans], [EMAIL, URL])
        for row in spans:
            self.assertEqual(text[row["start"]:row["end"]], row["source"])
        self.assertEqual(address_spans("Ordinary text with no address."), [])
        self.assertEqual(address_spans("x" * 100_001), [])

    def test_long_ambiguous_spellings_and_late_failures_are_bounded(self):
        # Isolate the adversarial match so a regression cannot hang the suite.
        code = '''
from epub_tts_narrator.renderer.address_speech import _pattern, address_spans
assert address_spans("a." * 49_999 + "@") == []
cases = [
 ("https://www.example.com/" + "be" * 200 + "/end",
  "https://www.example.com/" + "bee ee " * 200 + "slash wrong"),
 ("https://www.example.com/" + "be/" * 100 + "end",
  "https://www.example.com/" + "be/" * 100 + "wrong"),
]
for source, heard in cases:
 pattern = _pattern(source)
 for _ in range(20):
  assert pattern.match(heard) is None
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_complete_address_readings_are_reviewable(self):
        examples = (
            (EMAIL, "author.sample.writer.email at example.com"),
            (EMAIL, "author dot sample dot writer dot email at example dot com"),
            (URL, SPELLED),
            (URL, "www.example.com/gp/product/B07KJSRVMS"),
            (URL, "www.example.com.gp.product.B07KJSRVMS"),
            (URL, SPELLED.replace("zero", "oh")),
            ("https://www.facebook.com/groups/indiefantasyaddicts/",
             "h t t p s colon slash slash w w w dot facebook dot com slash groups slash indie fantasy addicts slash"),
            ("https://example.com/code?id=A09&part=2",
             "https colon slash slash example dot com slash code question mark id equals a zero nine and part equals two"),
        )
        for expected, heard in examples:
            with self.subTest(expected=expected, heard=heard):
                report = speech_coverage("Please visit " + expected + " today.", "Please visit " + heard + " today.")
                self.assertTrue(report["passed"], report)
                self.assertTrue(any(row["operation"] == "address_pronunciation" for row in report["alignment"]))

    def test_address_match_does_not_hide_changes_or_missing_prose(self):
        examples = (
            (URL, "www.example.com.gp.product.bo7kjsrvms.com"),
            (URL, SPELLED.replace("zero", "one")),
            (URL, SPELLED.replace("vee em", "em vee")),
            (URL, SPELLED.replace(" em", "")),
            (URL, SPELLED.replace(" slash g p", "")),
            (URL, SPELLED.replace("example", "otherhost")),
            (EMAIL, "author.sample.email at example.com"),
            (EMAIL, "author.writer.sample.email at example.com"),
            (EMAIL, "author.sample.writer.email at example.org"),
            (EMAIL, "author.sample.writer.email at example.com.com"),
            ("Please do not visit " + URL, "Please do visit " + SPELLED),
            ("Visit " + URL + " and return tomorrow.", "Visit " + SPELLED + " and return."),
            ("Visit " + URL + " today.", "Visit " + SPELLED + " twice today."),
            ("https://example.com/code?id=A09&part=2",
             "https colon slash slash example dot com slash code question mark id equals a zero nine and part equals three"),
        )
        for expected, heard in examples:
            with self.subTest(expected=expected, heard=heard):
                self.assertFalse(speech_coverage(expected, heard)["passed"])

    def test_spelling_and_separator_conventions_do_not_leak_into_prose(self):
        for expected, heard in (("Write A07KJSRVMS.", "Write ay zero seven kay jay ess are vee em ess."),
                                ("Go from home/work.", "Go from home slash work."),
                                ("Read the writer.email.", "Read the writer dot email.")):
            with self.subTest(expected=expected):
                self.assertFalse(speech_coverage(expected, heard)["passed"])


class AddressJevTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_nonexact_address_readings_require_the_actual_judge_for_both_purposes(self):
        for purpose in ("reference", "speech"):
            for expected, heard in ((EMAIL, "author.sample.writer.email at example.com"), (URL, SPELLED)):
                for score, passed in ((.99, True), (.01, False)):
                    with self.subTest(purpose=purpose, expected=expected, score=score):
                        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
                        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(heard), judge)
                        answer = {"model": "jev-fixture", "answers": {"roughly_same": {"type": "noul", "noul": score}},
                                  "usage": {"input_tokens": 20, "output_tokens": 2}}
                        with patch.object(judge._client, "evaluate", return_value=answer) as evaluate:
                            report = self.check(verifier, expected=expected, purpose=purpose)
                        evaluate.assert_called_once()
                        self.assertEqual(report["passed"], passed, report)
                        self.assertEqual(evaluate.call_args.args[0]["requested_passage"], expected)
                        self.assertEqual(evaluate.call_args.args[0]["recognized_passage"], heard)

    def test_live_invented_domain_suffix_cannot_reach_or_be_approved_by_jev(self):
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
        heard = "www.example.com.gp.product.bo7kjsrvms.com"
        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(heard), judge)
        with patch.object(judge._client, "evaluate") as evaluate:
            report = self.check(verifier, expected=URL, purpose="speech")
        self.assertFalse(report["passed"])
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
