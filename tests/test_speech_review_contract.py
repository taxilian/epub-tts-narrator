import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.speech_normalization import _validate_votes, normalize_document_speech
from test_speech_contextual_words import Client
from test_speech_normalization import document, edit


class UniformEvidenceClient(Client):
    def complete(self, system, payload):
        result = super().complete(system, payload)
        if payload["mode"] == "speech_spelling_review":
            for vote, proposal in zip(result["votes"], payload["proposals"]):
                if "word_verdicts" not in vote:
                    vote["word_verdicts"] = [
                        {"word_id": f"w{index}",
                         "verdict": self.pair_verdicts.get((payload["reviewer_id"], f"w{index}"), "approve")}
                        for index, _ in enumerate(proposal["original"].split())]
        return result


class SpeechReviewContractTests(unittest.TestCase):
    def test_mixed_batch_accepts_complete_word_evidence_for_every_edit_without_repair(self):
        source = document(' “Pay more attenssshun! Thisss isss already unusssual!”')
        client = UniformEvidenceClient({"attenssshun": "attention", "Thisss isss": "This is", "unusssual": "unusual"})
        with tempfile.TemporaryDirectory() as cache:
            result = normalize_document_speech(source, CachedRequests(client, cache, repair_attempts=0), workers=1)
            self.assertEqual(result["segments"][0]["spoken_text"], ' “Pay more attention! This is already unusual!”')
            self.assertEqual(len(client.calls), 4)
            self.assertTrue(all("correction" not in payload for payload in client.calls))
            replay = normalize_document_speech(source, CachedRequests(client, cache), workers=1)
            self.assertEqual(replay, result)
            self.assertEqual(len(client.calls), 4)
        for decision in result["provenance"]["speech_normalization"]["decisions"]:
            self.assertTrue(all("word_verdicts" in vote for vote in decision["votes"]))

    def test_optional_negative_word_evidence_is_not_ignored(self):
        client = UniformEvidenceClient({"Thisss isss": "This is"}, pair_verdicts={
            ("speech_spelling_reviewer_2", "w1"): "reject",
            ("speech_spelling_reviewer_3", "w0"): "uncertain"})
        with tempfile.TemporaryDirectory() as cache:
            result = normalize_document_speech(document("Thisss isss"),
                                               CachedRequests(client, cache, repair_attempts=0), workers=1)
        self.assertNotIn("speech_normalization", result["segments"][0])
        self.assertFalse(result["provenance"]["speech_normalization"]["decisions"][0]["applied"])

    def test_bad_optional_word_evidence_identifies_vote_and_edit(self):
        proposal = {"edit_id": "fixture-edit", **edit("Thisss isss", "Thisss isss", "This is")}
        response = {"votes": [{"edit_id": "fixture-edit", "verdict": "approve", "reason": "Same words.",
                               "word_verdicts": [{"word_id": "w0", "verdict": "approve"}]}]}
        with self.assertRaises(ValueError) as caught:
            _validate_votes(response, [proposal])
        self.assertIn("votes[0]", str(caught.exception))
        self.assertIn("fixture-edit", str(caught.exception))
        self.assertIn("word_verdicts", str(caught.exception))

    def test_invalid_fields_have_specific_diagnostics(self):
        proposal = {"edit_id": "fixture-edit", **edit("wull", "wull", "well")}
        for field, value in (("edit_id", "wrong"), ("verdict", {}), ("reason", "")):
            with self.subTest(field=field):
                vote = {"edit_id": "fixture-edit", "verdict": "approve", "reason": "Same word."}
                vote[field] = value
                with self.assertRaises(ValueError) as caught:
                    _validate_votes({"votes": [vote]}, [proposal])
                self.assertIn("votes[0]", str(caught.exception))
                self.assertIn(field, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
