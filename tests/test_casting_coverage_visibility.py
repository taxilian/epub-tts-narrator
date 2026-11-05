from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.casting_coverage import verify_coverage
from epub_tts_narrator.pipeline import CachedRequests, DEFAULT_POLICY, make_units
from test_identity import fixture


class CoverageVisibilityTests(unittest.TestCase):
    def test_binding_to_a_canonical_character_absent_from_the_supplied_view_is_rejected(self):
        names = ("Mara",) + tuple(f"Outlander{index}" for index in range(32))
        source, cast = fixture(names)
        source["blocks"][1]["text"] = " ".join(name + " returned." for name in names[1:])
        for character in cast["characters"][1:]:
            character["description"] = "A distinct later source character. " * 20
            character["evidence"] = [{"block_id": "b2", "quote": character["name"]}]

        class Client:
            identity = {"model": "offline-visibility"}

            def __init__(self, variant):
                self.calls = []
                self.variant = variant

            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                response = {"characters": [], "roles": [], "voices": [], "merges": [], "issues": [],
                        "summary": "A deliberately unsupported binding.", "certified": True,
                        "findings": [{"block_id": "b1", "start": 0, "end": 4, "quote": "Mara",
                                      "character_id": "c33", "reason": "An unsupported existing identity."}]}
                if self.variant == "echo":
                    response["characters"] = [deepcopy(cast["characters"][-1])]
                if self.variant == "merge":
                    response["findings"] = []
                    response["merges"] = [{"from_character_id": "c33", "into_character_id": "c1",
                                            "reason": "An unsupported identity link.",
                                            "evidence": [{"block_id": "b1", "quote": "Mara"}]}]
                return response

        for variant in ("binding", "echo", "merge"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                client = Client(variant)
                requests = CachedRequests(client, directory, max_request_chars=15000, repair_attempts=0)

                with self.assertRaisesRegex(ValueError, "character_id|cast_view"):
                    verify_coverage(source, cast, requests, make_units(source), policy=DEFAULT_POLICY)

                self.assertEqual(len(client.calls), 1)
                supplied = client.calls[0]["cast_view"]["records"]["characters"]
                self.assertNotIn("c33", {row["id"] for row in supplied})
                self.assertFalse(list(Path(directory).glob("*.json")))

    def test_merge_cannot_cite_a_passage_outside_the_supplied_region(self):
        source, cast = fixture()

        class Client:
            identity = {"model": "offline-merge-visibility"}

            def complete(self, system, payload):
                return {"characters": [], "roles": [], "voices": [], "issues": [], "findings": [],
                        "certified": True, "summary": "A source-bound proposal.",
                        "merges": [{"from_character_id": "c2", "into_character_id": "c1",
                                    "reason": "A proposed identity link.",
                                    "evidence": [{"block_id": "b2", "quote": "Red returned after a year"}]}]}

        first = next(unit for batch in make_units(source) for unit in batch if unit["block_id"] == "b1")
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(Client(), directory, repair_attempts=0)
            with self.assertRaisesRegex(ValueError, "visible"):
                verify_coverage(source, cast, requests, [[first]], policy=DEFAULT_POLICY)
            self.assertFalse(list(Path(directory).glob("*.json")))
