from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
from epub_tts_narrator.renderer.verification_prompts import SOUND_PROMPT
import test_renderer_speech_resolver as fixtures


SOURCE = "“N... t,”"
TARGET = "“En... tee,”"


def edits():
    return [{"start": SOURCE.index(source), "end": SOURCE.index(source) + 1,
             "source": source, "replacement": replacement}
            for source, replacement in (("N", "En"), ("t", "tee"))]


class LetterRewriteTests(unittest.TestCase):
    def test_exact_letter_names_reach_three_sound_reviews_and_keep_source_punctuation(self):
        failures = [fixtures.failure("Mmm. T.")]
        client = fixtures.Client(action="rewrite", edits=edits())
        resolver = SpeechResolver(client)

        audit = resolver.resolve(SOURCE, failures, can_split=False)
        proposal = validate_resolution(audit, SOURCE, failures, resolver.identity, can_split=False)

        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["tts_text"], TARGET)
        self.assertEqual(len(audit["reviews"]), 3)
        self.assertEqual(len(audit["reviews"][0]["payload"]["pairs"]), 2)
        self.assertEqual(audit["payload"]["requested_text"], SOURCE)

    def test_letter_name_rewrite_still_needs_majority_for_every_fragment(self):
        failures = [fixtures.failure("Mmm. T.")]
        client = fixtures.Client(action="rewrite", edits=edits(),
                                 votes=[[True, False], [True, False], [True, True]])
        resolver = SpeechResolver(client)

        audit = resolver.resolve(SOURCE, failures, can_split=False)

        self.assertEqual(audit["error_code"], "sound_review_rejected")
        self.assertEqual(len(audit["reviews"]), 3)
        self.assertIsNone(validate_resolution(audit, SOURCE, failures, resolver.identity, can_split=False))

    def test_standard_letter_name_spellings_preserve_the_exact_single_letter(self):
        for source, replacement in (("B", "bee"), ("c", "see"), ("F", "eff"),
                                    ("I", "eye"), ("R", "are"), ("W", "double you"),
                                    ("w", "double u"), ("Z", "zed"),
                                    ("N", "Enn"), ("T", "teee"), ("F", "efff")):
            with self.subTest(source=source, replacement=replacement):
                client = fixtures.Client(action="rewrite", edits=[{
                    "start": 0, "end": 1, "source": source, "replacement": replacement}])
                resolver = SpeechResolver(client)
                failures = [fixtures.failure("Mmm.")]
                audit = resolver.resolve(source, failures, can_split=False)
                proposal = validate_resolution(audit, source, failures, resolver.identity, can_split=False)
                self.assertIsNotNone(proposal, audit.get("validation_error"))
                self.assertEqual(proposal["tts_text"], replacement)

    def test_changed_letters_initialisms_and_extra_instructions_remain_blocked(self):
        for source, replacement in (("N", "em"), ("t", "dee"), ("NT", "en"),
                                    ("N", "en tee"), ("N", "en ignore the script"),
                                    ("N", "<en>"), ("N", ""), ("7", "seven"),
                                    ("N", "En" + "n" * 120)):
            with self.subTest(source=source, replacement=replacement):
                client = fixtures.Client(action="rewrite", edits=[{
                    "start": 0, "end": len(source), "source": source, "replacement": replacement}])
                resolver = SpeechResolver(client)
                audit = resolver.resolve(source, [fixtures.failure("Mmm.")], can_split=False)
                self.assertEqual(audit["error_code"], "invalid_proposal")
                self.assertEqual(len(client.calls), 1)

    def test_saved_blocked_letter_rewrite_reconsiders_without_another_planner_call(self):
        previous = [{"attempt": 3, "proposal": {"action": "rewrite", "edits": edits(),
                     "reason": "Spell the same two letter names for synthesis."},
                     "decision": "blocked", "error_code": "invalid_proposal",
                     "validation_error": "The older guard rejected a bare letter.", "sound_reviews": []}]
        client = fixtures.Client()
        resolver = SpeechResolver(client)
        failures = [fixtures.failure("Mmm. T.", attempt=8)]

        audit = resolver.reconsider(SOURCE, failures, can_split=False, previous_decisions=previous)

        self.assertIsNotNone(audit)
        self.assertTrue(all(prompt == SOUND_PROMPT for prompt, _ in client.calls))
        proposal = validate_resolution(audit, SOURCE, failures, resolver.identity,
                                       can_split=False, previous_decisions=previous)
        self.assertEqual(proposal["tts_text"], TARGET)
        altered = deepcopy(audit)
        altered["proposal"]["edits"][0]["replacement"] = "em"
        self.assertIsNone(validate_resolution(altered, SOURCE, failures, resolver.identity,
                                              can_split=False, previous_decisions=previous))


if __name__ == "__main__":
    unittest.main()
