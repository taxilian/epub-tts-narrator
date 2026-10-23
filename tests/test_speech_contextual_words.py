from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.speech_normalization import normalize_document_speech
from test_speech_normalization import document, edit


class Client:
    identity = {"model": "offline-contextual-spelling"}

    def __init__(self, changes, verdicts=None, *, pair_verdicts=None, omit_pair_votes=False):
        self.changes = changes
        self.verdicts = verdicts or {}
        self.pair_verdicts = pair_verdicts or {}
        self.omit_pair_votes = omit_pair_votes
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == "speech_spelling_scan":
            result = []
            for row in payload["target_segments"]:
                edits = [edit(row["text"], old, new) for old, new in self.changes.items() if old in row["text"]]
                result.append({"segment_id": row["segment_id"], "edits": sorted(edits, key=lambda item: item["start"])})
            return {"segments": result}
        votes = []
        for proposal in payload["proposals"]:
            verdict = self.verdicts.get(payload["reviewer_id"], "approve")
            vote = {"edit_id": proposal["edit_id"], "verdict": verdict, "reason": "Fixture contextual judgment."}
            requirement = proposal.get("word_preservation")
            if requirement and not self.omit_pair_votes:
                vote["word_verdicts"] = [
                    {"word_id": pair["word_id"],
                     "verdict": self.pair_verdicts.get((payload["reviewer_id"], pair["word_id"]), verdict)}
                    for pair in requirement["pairs"]]
            votes.append(vote)
        return {"votes": votes}


class SpeechContextualWordsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)

    def normalize(self, source, client):
        return normalize_document_speech(source, CachedRequests(client, self.cache, repair_attempts=0), workers=1)

    def test_hissing_normalization_reaches_three_reviews_without_scanner_repairs(self):
        text = ' “Pay more attenssshun! Thisss isss already unusssual!”'
        source = document(text)
        before = deepcopy(source)
        client = Client({"attenssshun": "attention", "Thisss": "This", " isss ": " is ", "unusssual": "unusual"})
        result = self.normalize(source, client)
        self.assertEqual(result["segments"][0]["spoken_text"], ' “Pay more attention! This is already unusual!”')
        self.assertEqual(source, before)
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(all("correction" not in payload for payload in client.calls))
        reviews = client.calls[1:]
        self.assertEqual(len({row["reviewer_id"] for row in reviews}), 3)
        for payload in reviews:
            self.assertNotIn("votes", payload)
            marked = [row for row in payload["proposals"] if row.get("word_preservation")]
            self.assertEqual([row["original"] for row in marked], ["attenssshun"])
            self.assertEqual(marked[0]["word_preservation"]["pairs"][0]["original"], ["attenssshun"])
            self.assertEqual(marked[0]["word_preservation"]["pairs"][0]["replacement"], ["attention"])
        for change in result["segments"][0]["speech_normalization"]["edits"]:
            self.assertNotIn("word_preservation", change)
        audit = result["provenance"]["speech_normalization"]["decisions"]
        marked_audit = next(row for row in audit if row["proposal"]["original"] == "attenssshun")
        self.assertTrue(all("word_verdicts" in vote for vote in marked_audit["votes"]))
        resumed = self.normalize(source, client)
        self.assertEqual(resumed, result)
        self.assertEqual(len(client.calls), 4)

    def test_unrelated_substitution_is_reviewed_and_rejected_without_aborting_analysis(self):
        source = document("red")
        client = Client({"red": "blue"}, {f"speech_spelling_reviewer_{i}": "reject" for i in range(1, 4)})
        result = self.normalize(source, client)
        self.assertNotIn("spoken_text", result["segments"][0])
        audit = result["provenance"]["speech_normalization"]
        self.assertFalse(audit["decisions"][0]["applied"])
        self.assertEqual(audit["unresolved_edits"], 1)
        self.assertEqual(len(client.calls), 4)

    def test_recorded_three_edit_proposal_preserves_source_bounds_and_reaches_review(self):
        text = ' “Pay more attenssshun! Thisss isss already unusssual!”'
        client = Client({"attenssshun": "attention", "Thisss isss": "This is", "unusssual": "unusual"})
        result = self.normalize(document(text), client)
        changes = result["segments"][0]["speech_normalization"]["edits"]
        self.assertEqual([(row["start"], row["end"], row["original"], row["replacement"]) for row in changes],
                         [(11, 22, "attenssshun", "attention"), (24, 35, "Thisss isss", "This is"),
                          (44, 53, "unusssual", "unusual")])
        self.assertEqual(result["segments"][0]["spoken_text"], ' “Pay more attention! This is already unusual!”')
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(all("correction" not in payload for payload in client.calls))

    def test_uncertain_review_cannot_apply_unrecognized_spelling(self):
        client = Client({"attenssshun": "attention"},
                        {"speech_spelling_reviewer_2": "uncertain", "speech_spelling_reviewer_3": "reject"})
        result = self.normalize(document("attenssshun"), client)
        self.assertNotIn("speech_normalization", result["segments"][0])
        self.assertEqual(result["provenance"]["speech_normalization"]["unresolved_edits"], 1)

    def test_two_complete_approvals_apply_mixed_known_and_unknown_word_pairs(self):
        client = Client({"wull attenssshun": "well attention"}, {"speech_spelling_reviewer_3": "reject"})
        result = self.normalize(document("wull attenssshun"), client)
        self.assertEqual(result["segments"][0]["spoken_text"], "well attention")
        decision = result["provenance"]["speech_normalization"]["decisions"][0]
        self.assertEqual([(pair["original"], pair["replacement"])
                          for pair in decision["proposal"]["word_preservation"]["pairs"]],
                         [(["wull"], ["well"]), (["attenssshun"], ["attention"])])
        self.assertEqual(result["provenance"]["speech_normalization"]["unresolved_edits"], 0)

    def test_overall_approval_does_not_override_rejected_word_pairs(self):
        client = Client({"attenssshun": "attention"}, pair_verdicts={
            ("speech_spelling_reviewer_2", "w0"): "reject",
            ("speech_spelling_reviewer_3", "w0"): "uncertain"})
        result = self.normalize(document("attenssshun"), client)
        self.assertNotIn("speech_normalization", result["segments"][0])
        self.assertFalse(result["provenance"]["speech_normalization"]["decisions"][0]["applied"])

    def test_missing_word_evidence_cannot_apply_proposal(self):
        source = document("attenssshun")
        with self.assertRaises(ValueError):
            self.normalize(source, Client({"attenssshun": "attention"}, omit_pair_votes=True))
        self.assertNotIn("speech_normalization", source["provenance"])

    def test_partial_word_approvals_from_different_reviewers_cannot_be_combined(self):
        client = Client({"attenssshun reezun": "attention reason"}, pair_verdicts={
            ("speech_spelling_reviewer_1", "w1"): "reject",
            ("speech_spelling_reviewer_2", "w0"): "reject"})
        result = self.normalize(document("attenssshun reezun"), client)
        self.assertNotIn("speech_normalization", result["segments"][0])
        self.assertEqual(result["provenance"]["speech_normalization"]["unresolved_edits"], 1)

    def test_word_verdicts_must_cover_every_pair_once_in_order(self):
        mutations = (
            lambda words: words[:-1],
            lambda words: [words[0], words[0]],
            lambda words: list(reversed(words)),
            lambda words: [{"word_id": "unknown", "verdict": "approve"}, words[1]],
            lambda words: [{"word_id": "w0", "verdict": {}}, words[1]],
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                class Malformed(Client):
                    def complete(self, system, payload):
                        result = super().complete(system, payload)
                        if payload["mode"] == "speech_spelling_review":
                            result["votes"][0]["word_verdicts"] = mutate(result["votes"][0]["word_verdicts"])
                        return result
                source = document("attenssshun reezun")
                with self.assertRaises(ValueError):
                    normalize_document_speech(
                        source, CachedRequests(Malformed({"attenssshun reezun": "attention reason"}),
                                               self.cache / str(index), repair_attempts=0), workers=1)
                self.assertNotIn("speech_normalization", source["provenance"])

    def test_clear_word_loss_addition_and_reordering_still_fail_before_jury(self):
        for original, replacement in (("Pay more attenssshun", "Pay attention"),
                                      ("Pay attenssshun", "Pay attention now"),
                                      ("attenssshun now", "now attention"), ("Ⅳ", "six")):
            with self.subTest(original=original):
                client = Client({original: replacement})
                with self.assertRaises(ValueError) as caught:
                    self.normalize(document(original), client)
                self.assertEqual(len(client.calls), 1)
                self.assertIn("original=", str(caught.exception))
                self.assertIn("replacement=", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
