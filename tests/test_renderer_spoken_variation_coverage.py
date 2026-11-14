"""Coverage v15: ordinary spoken variation reaches the judges; lost words still do not.

The cases follow takes a listener marked good after the earlier coverage gate
rejected them before any judge was asked, plus takes they marked bad.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.phonetic_review import substitution_pairs
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, verified_take
import test_renderer_reference_consensus as consensus


def rules(expected, heard):
    coverage = speech_coverage(expected, heard)
    if not coverage["passed"]:
        return None
    return [(row["operation"], row.get("rule")) for row in coverage["alignment"]]


class SpokenVariationCoverageTests(unittest.TestCase):
    def assertEligible(self, expected, heard, rule):
        found = rules(expected, heard)
        self.assertIsNotNone(found, (expected, heard))
        self.assertIn(rule, {name for _, name in found}, found)

    def test_a_contraction_of_any_word_may_be_heard_spelled_out(self):
        for expected, heard in (("No one else's seen you!", "No one else has seen you."),
                                ("They ought to've known.", "They ought to have known."),
                                ("This'll do nicely.", "This will do nicely.")):
            with self.subTest(expected=expected):
                found = rules(expected, heard)
                self.assertIsNotNone(found)
                self.assertEqual({operation for operation, _ in found}, {"contraction"})
        # The host word itself must still be heard.
        self.assertIsNone(rules("No one else's seen you!", "No one has seen you."))

    def test_elided_letters_may_be_read_in_full(self):
        for expected, heard in (("Tell 'em to wait.", "Tell them to wait."),
                                ("G'night.", "Good night."), ("G'night.", "Goodnight.")):
            with self.subTest(expected=expected, heard=heard):
                self.assertEligible(expected, heard, "elided_spelling")
        # The apostrophe marks at most three left-out letters, in order.
        self.assertIsNone(rules("Tell 'em to wait.", "Tell problem to wait."))

    def test_dictionary_homophones_and_non_rhotic_readings_go_to_the_judges(self):
        self.assertEligible('"High, Lanzecki,"', "Hi, Lanzecki.", "dictionary_homophone")
        self.assertEligible("Just the two of us.", "Just the to of us.", "dictionary_homophone")
        self.assertEligible("A crack in the flaw.", "A crack in the floor.", "non_rhotic")
        self.assertEligible("Come on, Killa!", "Come on, killer!", "non_rhotic")
        # Different dictionary vowels stay different words.
        self.assertIsNone(rules("He pulled the cord.", "He pulled the cod."))

    def test_roman_numerals_symbols_and_units_have_their_spoken_forms(self):
        self.assertEqual(rules("Carigana, Sector VI.", "Carigana, Sector six."),
                         [("roman_numeral", "uppercase_roman_numeral")])
        self.assertIsNone(rules("Carigana, Sector vi.", "Carigana, Sector six."))
        self.assertEqual(rules("It affects the DNA/ RNA pattern.", "It affects the DNA-RNA pattern."),
                         [("letter_names", "exact_english_letter_names"), ("symbol_reading", "unspoken_symbol"),
                          ("letter_names", "exact_english_letter_names")])
        self.assertEligible("It affects the DNA/ RNA pattern.", "It affects the DNA slash RNA pattern.",
                            "symbol_name")
        self.assertEligible("A bath at 35° C, please.", "A bath at 35 degrees Celsius, please.", "degree_unit")
        self.assertEligible("Dated 23/4/3308 by the clerk.", "Dated 23-4-3308 by the clerk.", "joining_hyphen")
        # A spaced minus sign is still a spoken word.
        self.assertIsNone(rules("It fell to -4 degrees.", "It fell to 4 degrees."))

    def test_a_few_vocal_sounds_and_restarts_may_differ(self):
        self.assertEligible("Probably. I don't remember.", "Oh, probably. I don't remember.", "added_vocalization")
        self.assertEligible("Ha! I shall have to tip the agent.", "I shall have to tip the agent.",
                            "omitted_vocalization")
        self.assertEligible("Oh, pharrellin.", "Huh, pharrellin.", "changed_vocalization")
        self.assertEligible('"Trag would never make it."', "T-Trag would never make it.", "hyphenated_restart")
        # Bounded per passage, and a written hum is never left out.
        self.assertIsNone(rules("We should go now.", "Oh, we, uh, should go, um, now."))
        self.assertIsNone(rules("Mmm... still warm here.", "Still warm here."))
        # A restart must be hyphen-joined to the word it begins.
        self.assertIsNone(rules("Trag would never make it.", "T Trag would never make it."))

    def test_an_invented_name_may_be_heard_across_new_word_breaks(self):
        for heard in ("Kila Chandra Rhee, at last.", "Kilachandrari, at last.", "Kill a Chandra Ree, at last."):
            with self.subTest(heard=heard):
                self.assertEligible("Killashandra Ree, at last.", heard, "invented_name_sound")
        # Only a capitalized word no dictionary knows; never a spelled letter.
        self.assertIsNone(rules("killashandra ree, at last.", "kila chandra rhee, at last."))
        self.assertIsNone(rules('"Rimbol C-hen-stal-az"', "Rimboel hen stahlaz!"))
        self.assertIsNone(rules("Killashandra Ree, at last.", "Kila Chandra not at last."))

    def test_listener_rejected_takes_stay_outside_the_judges(self):
        for expected, heard in (
                ("The survivors waited by the sled.", "The waited by the sled."),
                ("No, not that crystal.", "Here, not that crystal."),
                ("Killashandra now looked up.", "Kilachandra looked up."),
                ("Rimbol?", "I'm alright, I thought there'd be some fever."),
                ("Ah!", "If I should drop."),
                ("Carrik peered at the label.",
                 "He glanced at her skeptically just as a chilled bottle slid from the service panel.")):
            with self.subTest(expected=expected):
                self.assertIsNone(rules(expected, heard))


class SoundReviewBoundaryTests(unittest.TestCase):
    def test_sound_votes_cannot_approve_a_passage_around_a_one_sided_sound(self):
        # The name pair alone is reviewable by sound votes...
        self.assertIsNotNone(substitution_pairs("Killashandra Ree waited.", "Kilachandrari waited."))
        # ...but an added vocal sound in the same passage needs the text judge.
        self.assertIsNone(substitution_pairs("Killashandra Ree waited.", "Oh, Kilachandrari waited."))
        # An unspoken symbol carries no words and does not block the votes.
        pairs = substitution_pairs("Killashandra Ree waited at home/work.", "Kilachandrari waited at home work.")
        self.assertEqual([pair["expected_word"] for pair in pairs], ["Killashandra Ree"])


def answer(score):
    return {"model": "jev-fixture", "answers": {"roughly_same": {"type": "noul", "noul": score}},
            "usage": {"input_tokens": 30, "output_tokens": 2}}


class PolicyRecheckTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio

    EXPECTED = '"I was told no one else\'s seen you! Are you all right?"'
    HEARD = "I was told no one else has seen you. Are you all right?"

    def render(self, directory, score, generated):
        def generate(path, seed):
            generated.append(path)
            Path(path).write_bytes(self.audio.read_bytes())
            return {"seed": seed}
        config = {"quality": {}, "render": {"max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused"}}
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(self.HEARD), judge)
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                   side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())), \
                patch.object(judge._client, "evaluate", return_value=answer(score)) as evaluate:
            try:
                return verified_take(self.EXPECTED, generate, directory, {"fixture": True}, verifier, config,
                                     purpose="speech"), evaluate.call_count
            except SpeechVerificationFailed:
                return None, evaluate.call_count

    def rejected_take(self, directory, generated):
        result, _ = self.render(directory, .01, generated)
        self.assertIsNone(result)
        record = read_json(Path(directory, "attempt-001.json"))
        self.assertEqual(record["report"]["reasons"], ["text_mismatch"])
        return record

    def test_a_take_rejected_under_an_earlier_policy_is_rechecked_not_remade(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            record = self.rejected_take(directory, generated)
            current_key = record["validation_key"]
            # As saved by an earlier policy whose coverage gate stopped the judge.
            record["validation_key"] = "0" * 64
            write_json(Path(directory, "attempt-001.json"), record)
            result, judged = self.render(directory, .99, generated)
            self.assertIsNotNone(result)
            path, accepted = result
            self.assertEqual(len(generated), 1)
            self.assertEqual(judged, 1)
            self.assertTrue(accepted["report"]["passed"])
            self.assertEqual(accepted["validation_key"], current_key)
            saved = read_json(Path(directory, "attempt-001.json"))
            self.assertEqual(saved["policy_recheck"]["previous_validation_key"], "0" * 64)
            self.assertEqual(saved["policy_recheck"]["previous_report"]["reasons"], ["text_mismatch"])

    def test_a_take_rejected_under_the_current_policy_is_not_checked_again(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            self.rejected_take(directory, generated)
            result, judged = self.render(directory, .99, generated)
            # The saved rejection stands: no recognition or judgment repeats.
            self.assertIsNone(result)
            self.assertEqual((len(generated), judged), (1, 0))
            self.assertNotIn("policy_recheck", read_json(Path(directory, "attempt-001.json")))

    def test_acoustic_failures_are_never_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = []
            record = self.rejected_take(directory, generated)
            record["validation_key"] = "0" * 64
            record["report"]["reasons"] = ["internal_silence_stall"]
            write_json(Path(directory, "attempt-001.json"), record)
            result, judged = self.render(directory, .99, generated)
            self.assertIsNone(result)
            self.assertEqual((len(generated), judged), (1, 0))


if __name__ == "__main__":
    unittest.main()
