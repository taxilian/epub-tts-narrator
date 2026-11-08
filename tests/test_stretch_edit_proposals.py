"""Deterministic stretch proposals and their narrator-read publication path."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.narrator_read import narrated_analyze
from epub_tts_narrator.performance_plan import validate_performance
from epub_tts_narrator.pipeline import base_document
from epub_tts_narrator.stretch_edits import attach_stretch_proposals
from test_attribution_performance import document, segment
from test_narrator_read import NarratorClient, make_source


def stretch_document(text, *corpus, characters=None, proposals=None):
    row = segment("b1", text, proposals=proposals)
    result = document([row], block_text=text, characters=characters)
    for number, block_text in enumerate(corpus, 2):
        result["source"]["blocks"].append({"id": f"b{number}", "chapter_id": "ch1",
                                           "text": block_text, "source": {}})
    return result


def attached_edits(payload):
    attach_stretch_proposals(payload)
    return [(edit["original"], edit["replacement"])
            for edit in payload["segments"][0].get("inline_proposals", {}).get("edits", [])]


class StretchEditProposalTests(unittest.TestCase):
    def test_attested_variants_use_corpus_frequency_then_minimal_change(self):
        cases = [
            ("messsage", "message", "message"),
            ("Hmmm", "Hm Hm Hmm", "Hm"),
            ("Hmmm", "Hm Hmm", "Hmm"),
        ]
        for word, corpus, expected in cases:
            with self.subTest(word=word, corpus=corpus):
                self.assertEqual(attached_edits(stretch_document(word, corpus)), [(word, expected)])

    def test_unattested_fallback_uses_run_length(self):
        long_sibilant = "s" * 14 + "hi"
        text = f"Hmmm messsage oooold Ahhh Wheerrre issss sheeee {long_sibilant}"
        self.assertEqual(attached_edits(stretch_document(text)), [
            ("Hmmm", "Hmm"),
            ("messsage", "message"),
            ("oooold", "old"),
            ("Ahhh", "Ahh"),
            ("Wheerrre", "Wheerre"),
            ("issss", "is"),
            ("sheeee", "she"),
            (long_sibilant, "shi"),
        ])

    def test_uppercase_roman_numerals_are_not_elongations(self):
        self.assertEqual(attached_edits(stretch_document("VIII XXIII Hmmm")), [("Hmmm", "Hmm")])

    def test_an_established_name_cannot_evade_protection_through_a_partial_collapse(self):
        characters = [{"id": "c1", "name": "Mara", "aliases": [], "description": "A speaker.",
                       "voice_id": "v1"}]
        self.assertEqual(attached_edits(stretch_document("Maraaa Hmmm", characters=characters)),
                         [("Hmmm", "Hmm")])

    def test_an_overlapping_model_edit_is_not_duplicated(self):
        proposals = {"edits": [{"original": "Hmmm", "replacement": "Hm",
                                "direction": "model direction", "reason": "Model proposal."}],
                     "controls": []}
        payload = stretch_document("Hmmm messsage", "message", proposals=proposals)
        self.assertEqual(attached_edits(payload), [("Hmmm", "Hm"), ("messsage", "message")])
        edits = payload["segments"][0]["inline_proposals"]["edits"]
        self.assertEqual(edits[0]["direction"], "model direction")

    def test_only_dialogue_thought_and_letter_rows_gain_proposals(self):
        for kind in ("dialogue", "thought", "letter", "narration"):
            with self.subTest(kind=kind):
                payload = stretch_document("Hmmm")
                payload["segments"][0]["kind"] = kind
                attach_stretch_proposals(payload)
                if kind == "narration":
                    self.assertNotIn("inline_proposals", payload["segments"][0])
                else:
                    self.assertEqual(payload["segments"][0]["inline_proposals"]["edits"][0]["replacement"], "Hmm")

    def test_consensus_document_assembly_is_unchanged(self):
        payload = stretch_document("Hmmm")
        consensus = base_document(payload["source"], payload["cast"], deepcopy(payload["segments"]))
        self.assertNotIn("inline_proposals", consensus["segments"][0])
        narrator = base_document(payload["source"], payload["cast"], deepcopy(payload["segments"]),
                                 provenance={"strategy": "narrator_read"})
        self.assertEqual(narrator["segments"][0]["inline_proposals"]["edits"][0]["replacement"], "Hmm")

    def test_narrator_read_publishes_mechanical_edits_without_a_review_request(self):
        source = make_source([[
            ('"Hmmm, the messsage feels oooold," Mara said. "Wheerrre issss sheeee?"', "p"),
            ("The message is old, and she is here.", "p"),
        ]])
        with tempfile.TemporaryDirectory() as directory:
            client = NarratorClient()
            result = narrated_analyze(source, client, Path(directory) / "cache", workers=2)
        rows = {row["text"]: row for row in result["segments"]}
        expected = {
            '"Hmmm, the messsage feels oooold,"': [("Hmmm", "Hmm"), ("messsage", "message"),
                                                   ("oooold", "old")],
            '"Wheerrre issss sheeee?"': [("Wheerrre", "Wheerre"), ("issss", "is"), ("sheeee", "she")],
        }
        for text, pairs in expected.items():
            with self.subTest(text=text):
                row = rows[text]
                self.assertEqual([(edit["original"], edit["replacement"])
                                  for edit in row["speech_normalization"]["edits"]], pairs)
                self.assertEqual(validate_performance(row["performance_controls"], row["spoken_text"]),
                                 row["performance_controls"])
                self.assertNotIn("inline_proposals", row)
        self.assertFalse(any(call.get("mode") == "inline_performance_review" for call in client.calls))
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"mechanical": 6})
        self.assertEqual(result["provenance"]["inline_performance"]["batches"], [])


if __name__ == "__main__":
    unittest.main()
