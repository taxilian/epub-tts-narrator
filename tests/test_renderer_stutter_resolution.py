"""Stutter onset recovery preserves every full word and requires sound review."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from epub_tts_narrator.provider import OutputTruncatedError
from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
from test_renderer_speech_resolver import Client, failure


class StutterResolutionTests(unittest.TestCase):
    def resolve(self, source, transcript, *, replacement=None, approve=True):
        edits = None
        if replacement is not None:
            start, end = 1, source.index(" Mother")
            edits = [{"start": start, "end": end, "source": source[start:end],
                      "replacement": replacement}]
        client = Client("accept" if replacement is None else "rewrite", edits=edits,
                        votes=[[approve]] * 3)
        resolver = SpeechResolver(client)
        evidence = [failure(transcript, attempt=number) for number in (1, 2, 3)]
        audit = resolver.resolve(source, evidence)
        proposal = validate_resolution(audit, source, evidence, resolver.identity)
        return client, resolver, evidence, audit, proposal

    def test_saved_passage_can_accept_recognition_without_incomplete_prefix(self):
        source = "“B-but Mother—” "
        for transcript in (" But mother...", " But... Mother...", "Buh-but mother"):
            with self.subTest(transcript=transcript):
                client, _, _, audit, proposal = self.resolve(source, transcript)
                self.assertIsNotNone(proposal, audit)
                self.assertEqual(proposal["action"], "accept")
                self.assertEqual(proposal["acceptance_transcript"], transcript)
                self.assertEqual(len(client.calls), 4)
                pair = proposal["validated_pairs"][0]
                self.assertEqual(pair["expected_word"], "B-but")
                self.assertEqual((pair["expected_start"], pair["expected_end"]), (1, 6))
                self.assertEqual(pair["requested_context"], "B-but Mother")

    def test_saved_pronunciation_rewrite_can_preserve_final_function_word(self):
        source = "“B-but Mother—” "
        for replacement in ("Buh-but", "But"):
            with self.subTest(replacement=replacement):
                client, _, _, audit, proposal = self.resolve(source, "But mother", replacement=replacement)
                self.assertIsNotNone(proposal, audit)
                self.assertEqual(proposal["action"], "rewrite")
                self.assertEqual(proposal["tts_text"], f"“{replacement} Mother—” ")
                self.assertEqual(len(client.calls), 4)

    def test_preserves_negation_and_repeated_incomplete_prefixes(self):
        for source, transcript in (("N-no Mother", "No mother"),
                                   ("Th-th-there is Mother", "There is mother"),
                                   ("B-but Mother and f-father", "But mother and father")):
            with self.subTest(source=source):
                client, _, _, audit, proposal = self.resolve(source, transcript)
                self.assertIsNotNone(proposal, audit)
                self.assertEqual(len(client.calls), 4)

    def test_omissions_extra_words_and_changed_whole_words_stay_rejected(self):
        for transcript in ("Mother", "But", "But but Mother", "B-but-but Mother",
                           "But another Mother", "But Father", "Not Mother", "But not Mother"):
            with self.subTest(transcript=transcript):
                client, _, _, audit, proposal = self.resolve("“B-but Mother—” ", transcript)
                self.assertIsNone(proposal, audit)
                self.assertEqual(len(client.calls), 1)

    def test_other_mismatches_cannot_piggyback_on_stutter_exception(self):
        _, _, _, audit, proposal = self.resolve("B-but Mother", "But Mutha")
        self.assertIsNone(proposal, audit)
        _, _, _, audit, proposal = self.resolve("N-no Mother", "Yes Mother")
        self.assertIsNone(proposal, audit)

    def test_complete_repeats_compounds_and_initialisms_are_not_stutter_fragments(self):
        for source, transcript in (("But-but Mother", "But Mother"),
                                   ("Go-gone Mother", "Gone Mother"),
                                   ("Car-carpet Mother", "Carpet Mother"),
                                   ("Ship-woman Mother", "Woman Mother"),
                                   ("B-BBC reported", "BBC reported")):
            with self.subTest(source=source):
                _, _, _, audit, proposal = self.resolve(source, transcript)
                self.assertIsNone(proposal, audit)

    def test_rewrites_cannot_replace_full_word_or_insert_full_words(self):
        for replacement in ("But but", "Buh-not", "Not", "Buh", "Buh-but now", "Buh but please",
                            "Bee-but", "[say but]", "Buh\nbut", "Buh-but-but"):
            with self.subTest(replacement=replacement):
                client, _, _, audit, proposal = self.resolve(
                    "“B-but Mother—” ", "But Mother", replacement=replacement)
                self.assertIsNone(proposal, audit)
                self.assertEqual(len(client.calls), 1)

    def test_stutter_decision_still_needs_independent_sound_consensus(self):
        for replacement in (None, "Buh-but"):
            with self.subTest(replacement=replacement):
                client, _, _, audit, proposal = self.resolve(
                    "“B-but Mother—” ", "But Mother", replacement=replacement, approve=False)
                self.assertIsNone(proposal)
                self.assertEqual(audit["error_code"], "sound_review_rejected")
                self.assertEqual(len(client.calls), 4)

    def test_previous_feedback_policy_rewrite_remains_valid_after_prompt_upgrade(self):
        saved = json.loads((Path(__file__).parent / "fixtures/speech-rewrite-policy-v3.json").read_text())
        for version in (5, 6, 7):
            with self.subTest(version=version):
                audit = deepcopy(saved["audit"])
                identity = audit["identity"]
                identity["coverage_version"] = version
                identity["policy_sha256"] = digest({key: value for key, value in identity.items()
                                                     if key != "policy_sha256"})
                proposal = validate_resolution(audit, saved["expected_text"], saved["failures"], identity)
                self.assertIsNotNone(proposal)
                self.assertEqual(proposal["action"], "rewrite")
                self.assertEqual(proposal["tts_text"], "Where is she")
                altered = deepcopy(audit)
                altered["payload"]["previous_decisions"] = []
                self.assertIsNone(validate_resolution(altered, saved["expected_text"], saved["failures"],
                                                      identity, previous_decisions=audit["payload"]["previous_decisions"]))
                altered = deepcopy(audit)
                altered["reviews"][0]["response"]["pairs"][0]["plausible"] = False
                altered["reviews"][1]["response"]["pairs"][0]["plausible"] = False
                self.assertIsNone(validate_resolution(altered, saved["expected_text"], saved["failures"], identity))

    def test_output_truncation_is_identified_for_planner_and_sound_review(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                class TruncatedClient(Client):
                    def complete(self, prompt, payload):
                        if len(self.calls) + 1 == fail_at:
                            raise OutputTruncatedError("Output budget exhausted")
                        return super().complete(prompt, payload)
                resolver = SpeechResolver(TruncatedClient())
                audit = resolver.resolve("B-but Mother", [failure("But Mother")])
                self.assertEqual(audit["decision"], "unavailable")
                self.assertEqual(audit["error_code"], "output_truncated")


if __name__ == "__main__":
    unittest.main()
