from copy import deepcopy
from json import dumps
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity import audit_identities
from epub_tts_narrator.pipeline import fingerprint, make_units
from tests.test_identity import case, fixture
from tests.test_identity_adaptive import Bounded


class TargetedIdentityContextTests(unittest.TestCase):
    def test_jury_omits_unrelated_narrator_and_chapters_but_keeps_all_affected_evidence(self):
        source, cast = fixture(("Mara", "Red", "Quillon"))
        cast["characters"][1]["evidence"] = [{"block_id": "b2", "quote": "Red returned after a year"}]
        source["blocks"][1]["chapter_id"] = "ch2"
        source["chapters"][0]["block_ids"] = ["b1"]
        source["chapters"].append({"id": "ch2", "title": "Two", "source_href": "two.xhtml", "block_ids": ["b2"]})
        third = deepcopy(source["blocks"][1])
        third.update(id="b3", chapter_id="ch3", text="Quillon retained UNRELATED_CAST_CONTEXT in another chapter.")
        source["blocks"].append(third)
        source["chapters"].append({"id": "ch3", "title": "Three", "source_href": "three.xhtml", "block_ids": ["b3"]})
        cast["characters"][2].update(description="UNRELATED_CAST_CONTEXT " * 100,
                                      evidence=[{"block_id": "b3", "quote": "Quillon"}])
        cast["roles"][0]["character_id"] = "c3"

        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded([case()], cache=Path(directory))
            updated, _, provenance = audit_identities(source, cast, requests, make_units(source), adaptive=True)

        votes = [payload for payload in requests.calls if payload["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertTrue(all({unit["block_id"] for unit in payload["source_units"]} == {"b1", "b2"} for payload in votes))
        self.assertTrue(all({row["id"] for row in payload["original_cast"]["characters"]} == {"c1", "c2"} for payload in votes))
        self.assertTrue(all({row["id"] for row in payload["proposed_cast"]["characters"]} == {"c1"} for payload in votes))
        self.assertTrue(all("UNRELATED_CAST_CONTEXT" not in dumps(payload) for payload in votes))
        self.assertTrue(all(payload["original_cast_fingerprint"] == fingerprint(cast) for payload in votes))
        self.assertEqual({row["id"] for row in updated["characters"]}, {"c1", "c3"})
        self.assertEqual(provenance["final_cast_fingerprint"], fingerprint(updated))
