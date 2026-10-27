"""Real dictionary integration; only the model boundary is replaced."""

import json
import gzip
import hashlib
from importlib.resources import files
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.pronunciation_discovery import discover_pronunciations
from epub_tts_narrator.pronunciation_proposals import propose_pronunciation
from epub_tts_narrator.pronunciation_dictionary import dictionary_readings
from epub_tts_narrator.performance_plan import validate_phonemes
from test_contracts import valid_document
from test_pronunciation_discovery import DiscoveryClient
from test_pronunciation_proposals import ProposalClient


class PronunciationDictionaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.document = valid_document()

    def test_packaged_dictionary_is_exact_upstream_data_with_all_valid_readings(self):
        resource = files("epub_tts_narrator").joinpath("data")

        data = gzip.decompress(resource.joinpath("cmudict.dict.gz").read_bytes())

        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22")
        self.assertEqual(len(data), 3618488)
        self.assertEqual(len(data.splitlines()), 135166)
        self.assertEqual(hashlib.sha256(resource.joinpath("CMUDICT-LICENSE").read_bytes()).hexdigest(),
                         "bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a")
        for line in data.decode("utf-8").splitlines():
            phones = " ".join(line.partition("#")[0].split()[1:])
            self.assertEqual(validate_phonemes(phones), phones)

    def test_lookup_keeps_alternates_apostrophes_and_strips_inline_comments(self):
        cases = {"Read": {"R EH1 D", "R IY1 D"}, "BASS": {"B AE1 S", "B EY1 S"},
                 "don't": {"D OW1 N T", "D OW1 N"}, "aalborg": {"AO1 L B AO0 R G", "AA1 L B AO0 R G"}}

        readings = {word: set(dictionary_readings(word)) for word in cases}

        self.assertEqual(readings, cases)
        self.assertEqual(dictionary_readings("Zyrrq"), ())

    def test_discovery_filters_known_words_even_when_model_calls_them_names(self):
        # Given dictionary words, deterministic cast names, and an unknown word.
        self.document["source"]["blocks"][0]["text"] = "May Will Rose Read Mark read bass ordinary softly Zyrrq"
        self.document["cast"]["characters"] = [
            {"name": word, "aliases": []} for word in ("May", "Will", "Rose", "Read", "Mark")]
        client = DiscoveryClient(words=("read", "bass", "ordinary", "softly", "Zyrrq"),
                                 mutation=lambda response: response["candidates"][0].update(
                                     reason="Manually selected for pronunciation review."))
        requests = CachedRequests(client, self.root, repair_attempts=0)

        entries = discover_pronunciations(self.document, requests)

        self.assertEqual({row["word"] for row in entries}, {"May", "Will", "Rose", "Read", "Mark", "Zyrrq"})
        self.assertTrue(all(row["status"] == "pending" for row in entries))

    def test_proposal_supplies_all_readings_without_replacing_owner_hint(self):
        # Given a cast heteronym and an explicit correction unlike either reading.
        self.document["source"]["blocks"][0]["text"] = "Read arrived."
        self.document["cast"]["characters"] = [{"name": "Read", "aliases": []}]
        client = ProposalClient()
        requests = CachedRequests(client, self.root, repair_attempts=0)

        propose_pronunciation(self.document, {"word": "Read"}, requests, respelling="RAHD")

        system, payload = client.calls[0]
        self.assertEqual(set(payload["dictionary"]["readings"]), {"R EH1 D", "R IY1 D"})
        self.assertEqual(payload["respelling"], "RAHD")
        self.assertEqual(payload["word"], "Read")
        self.assertEqual(payload["cast_context"], [{"names": ["Read"]}])
        self.assertLessEqual(len(system) + len(json.dumps(payload, ensure_ascii=False)), 16000)

    def test_multiple_primary_stresses_remain_valid_proposals(self):
        self.document["source"]["blocks"][0]["text"] = "Read arrived."
        client = ProposalClient([{"phonemes": "R EH1 D R IY1 D"}])
        requests = CachedRequests(client, self.root, repair_attempts=0)

        result = propose_pronunciation(self.document, {"word": "Read"}, requests)

        self.assertEqual(result, {"alphabet": "cmu_arpabet", "value": "R EH1 D R IY1 D"})


if __name__ == "__main__":
    unittest.main()
