"""Cast repair feedback names the actual record and citation without rewriting it."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_audiobook.pipeline import CachedRequests, check_cast, check_evidence, merge_patch
from epub_audiobook.series import validate_series_proposals
from test_contracts import valid_document
from test_series_identity_scan import fixture


class CastEvidenceDiagnosticTests(unittest.TestCase):
    def test_series_failures_report_both_character_paths_ids_and_referenced_block(self):
        source, cast, registry = fixture()
        for character in cast["characters"]:
            character["series_identity"] = {
                "character_id": "prior_mara", "basis": "context", "review_required": False,
                "reason": "A continuity hypothesis needing validation.",
                "evidence": [{"block_id": "b1", "quote": "Mara and Bex returned together."}],
            }
        before = deepcopy((source, cast, registry))
        with self.assertRaises(ValueError) as raised:
            validate_series_proposals(source, cast, registry)
        message = str(raised.exception)
        for index in range(2):
            self.assertIn(f"cast.characters[{index}].series_identity.evidence[0]", message)
            self.assertIn(f'record_id="c{index + 1}"', message)
        self.assertIn('block_id="b1"', message)
        self.assertIn("exact substring", message)
        self.assertNotIn("Mara and Bex returned together.", message)
        self.assertLessEqual(len(message), 1800)
        self.assertEqual((source, cast, registry), before)

    def test_character_and_voice_trait_failures_identify_records_and_source_blocks(self):
        document = valid_document()
        source, cast = document["source"], document["cast"]
        cast["roles"][0]["id"] = "narrator_main"
        cast["characters"][0]["evidence"][0]["quote"] = "MARIA"
        cast["voices"][1]["traits"][0]["evidence"][0]["quote"] = "SOFTLY"
        before = deepcopy((source, cast))
        with self.assertRaises(ValueError) as raised:
            check_cast(source, cast)
        message = str(raised.exception)
        self.assertIn("cast.characters[0].evidence[0].quote", message)
        self.assertIn('record_id="c1"; block_id="b1"', message)
        self.assertIn("cast.voices[1].traits[0].evidence[0].quote", message)
        self.assertIn('record_id="v1"; block_id="b1"', message)
        self.assertIn("exact source spelling starts with 'softly'", message)
        self.assertIn("Copy evidence quotations verbatim", message)
        self.assertLessEqual(len(message), 1800)
        self.assertEqual((source, cast), before)

    def test_merge_evidence_uses_response_path_and_preserves_every_input(self):
        source, cast, _ = fixture()
        patch = {"characters": [], "roles": [], "voices": [], "issues": [], "summary": "No changes.",
                 "merges": [{"from_character_id": "c1", "into_character_id": "c2",
                             "evidence": [{"block_id": "b01", "quote": "Mara"}]}]}
        available = {"characters": [], "roles": [], "voices": []}
        before = deepcopy((source, cast, patch))
        with self.assertRaises(ValueError) as raised:
            merge_patch(source, cast, patch, available)
        message = str(raised.exception)
        self.assertIn("merges[0].evidence[0]", message)
        self.assertIn('block_id="b01"', message)
        self.assertIn("unknown source block ID", message)
        self.assertIn("leading zero", message)
        self.assertEqual((source, cast, patch), before)

    def test_many_invalid_citations_and_arbitrary_values_produce_bounded_feedback(self):
        source, cast, _ = fixture()
        evidence = [{"block_id": "x\n" * 10000, "quote": "SECRET QUOTATION"} for _ in range(50)]
        evidence[0] = {"block_id": {"secret": "DO NOT ECHO"}, "quote": "wrong"}
        evidence[1] = {"block_id": "b1", "quote": {"secret": "DO NOT ECHO"}}
        with self.assertRaises(ValueError) as raised:
            check_evidence(evidence, source, label="characters[7].series_identity.evidence", record_id="c8")
        message = str(raised.exception)
        self.assertLessEqual(len(message), 1800)
        self.assertIn("evidence[0]", message)
        self.assertIn("evidence[1]", message)
        self.assertIn("additional invalid location(s) omitted", message)
        self.assertNotIn("\n", message)
        self.assertNotIn("DO NOT ECHO", message)
        self.assertNotIn("SECRET QUOTATION", message)

        cast["characters"][0]["evidence"] = evidence[2:]
        with self.assertRaises(ValueError) as raised:
            check_cast(source, cast)
        message = str(raised.exception)
        self.assertLessEqual(len(message), 1800)
        self.assertIn("cast.characters[0].evidence[0].block_id", message)
        self.assertIn('record_id="c1"', message)
        self.assertIn("additional invalid location(s) omitted", message)
        self.assertNotIn("SECRET QUOTATION", message)
        self.assertNotIn("\n", message)

    def test_accepted_evidence_extra_metadata_and_whitespace_rules_are_unchanged(self):
        source, cast, registry = fixture()
        evidence = [{"block_id": "b1", "quote": " ", "annotation": "Allowed extra metadata"},
                    {"block_id": "b1", "quote": "Captain Mara"}]
        before = deepcopy((evidence, source, cast, registry))
        self.assertIsNone(check_evidence(evidence, source, nonempty=True))
        self.assertIsNone(check_evidence([], source))
        self.assertIsNone(check_cast(source, cast))
        self.assertIsNone(validate_series_proposals(source, cast, registry))
        self.assertEqual((evidence, source, cast, registry), before)
        with self.assertRaises(ValueError):
            check_evidence([], source, nonempty=True)
        with self.assertRaises(ValueError):
            check_evidence([{"block_id": "b1", "quote": "captain mara"}], source)

    def test_cached_repair_gets_both_series_citation_locations_and_valid_replay(self):
        source, cast, registry = fixture()
        for character in cast["characters"]:
            character["series_identity"] = {
                "character_id": "prior_mara", "basis": "context", "review_required": False,
                "reason": "A proposal; this test checks source validation only.",
                "evidence": [{"block_id": "b1", "quote": "Mara and Bex returned together."}],
            }
        payload = {"cast": cast}
        class Client:
            identity = {"model": "offline-fixture", "base_url": "https://offline.invalid"}
            def __init__(self):
                self.calls = []
            def complete(self, system, request):
                self.calls.append(deepcopy(request))
                response = deepcopy(request["cast"])
                if "correction" in request:
                    for character in response["characters"]:
                        character["series_identity"]["evidence"][0]["quote"] = "Captain Mara"
                return response
        def validate(response):
            validate_series_proposals(source, response, registry)
            return response
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=1)
            result = requests.run("Fixture unchanged prompt", payload, validate)
            replay = requests.run("Fixture unchanged prompt", payload, validate)
        self.assertEqual(len(client.calls), 2)
        feedback = client.calls[1]["correction"]["validator_error"]
        self.assertIn("cast.characters[0].series_identity.evidence[0]", feedback)
        self.assertIn("cast.characters[1].series_identity.evidence[0]", feedback)
        self.assertLessEqual(len(feedback), 1800)
        self.assertEqual(replay, result)
        self.assertEqual(payload["cast"], cast)
