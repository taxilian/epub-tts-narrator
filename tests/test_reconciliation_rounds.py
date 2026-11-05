"""Finding-driven source coverage preserves candidates without configured rounds."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import analyze
from epub_tts_narrator.provider import ProviderError
from test_contracts import valid_document
from test_pipeline import FakeClient


class CumulativeClient(FakeClient):
    def reconcile(self, payload):
        known = payload["known_cast"]
        name = next((name for name in ("Jules", "Quinn", "Rowan")
                     if name not in {c["name"] for c in known["characters"]}), None)
        if name is None:
            return self.patch()
        ids = payload["available_ids"]
        character = {"id": ids["characters"][0], "name": name, "aliases": [],
                     "description": "Another separately named performer.",
                     "evidence": [{"block_id": "b2", "quote": name}],
                     "voice_id": ids["voices"][0]}
        role = {"id": ids["roles"][0], "kind": "dialogue",
                "character_id": character["id"], "voice_id": character["voice_id"],
                "description": f"Only {name}'s direct words."}
        voice = {"id": character["voice_id"], "description": f"A distinct voice for {name}.",
                 "traits": [{"name": "pace", "value": "measured", "basis": "casting", "evidence": []}]}
        # Deliberately omit prior records: patch semantics must retain them.
        return self.patch(characters=[character], roles=[role], voices=[voice])


class ReconciliationRoundsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "cache"
        self.source = deepcopy(valid_document()["source"])
        self.source["blocks"][1]["text"] = "Jules, Quinn, and Rowan called out."

    def run_analysis(self, client, **kwargs):
        kwargs.setdefault("identity_consensus", False)
        return analyze(self.source, client, self.cache, **kwargs)

    def test_default_and_explicit_one_round_have_identical_requests_and_replay(self):
        client = FakeClient()
        implicit = self.run_analysis(client)
        self.assertEqual([p.get("mode", "attribute") for _, p in client.calls],
                         ["casting_discovery", "casting_coverage", "voice_ensemble", "attribute"])
        self.assertEqual(implicit["provenance"]["casting"]["strategy"], "adaptive_evidence_first")
        replay = FakeClient()
        self.assertEqual(self.run_analysis(replay, cast_reconciliation_rounds=1), implicit)
        self.assertEqual(replay.calls, [])
        fresh = FakeClient()
        analyze(self.source, fresh, self.cache / "explicit", identity_consensus=False,
                cast_reconciliation_rounds=1)
        self.assertEqual(fresh.calls, client.calls)
        self.assertTrue(all("cast_reconciliation_rounds" not in payload for _, payload in client.calls))

    def test_distinct_findings_finish_even_when_the_legacy_round_setting_is_one(self):
        for whole_book in (False, True):
            with self.subTest(whole_book=whole_book):
                self.cache = Path(self.temp.name) / str(whole_book)
                first = CumulativeClient()
                original = deepcopy(self.source)
                baseline = self.run_analysis(first, whole_book_discovery=whole_book)
                self.assertEqual({c["name"] for c in baseline["cast"]["characters"]},
                                 {"Mara", "Jules", "Quinn", "Rowan"})
                checks = [payload for _, payload in first.calls if payload.get("mode") == "casting_coverage"]
                self.assertEqual(len(checks), 4)
                self.assertEqual({unit["block_id"] for unit in checks[0]["target_units"]}, {"b1", "b2"})
                self.assertTrue(all({unit["block_id"] for unit in check["target_units"]} == {"b2"}
                                    for check in checks[1:]))
                second = CumulativeClient()
                result = self.run_analysis(second, whole_book_discovery=whole_book,
                                           cast_reconciliation_rounds=3)
                self.assertEqual(second.calls, [])
                self.assertEqual(result, baseline)
                self.assertEqual(self.source, original)
                self.assertEqual(validate_document(result), [])
                replay = CumulativeClient()
                self.assertEqual(self.run_analysis(replay, whole_book_discovery=whole_book,
                                                   cast_reconciliation_rounds=2), result)
                self.assertEqual(replay.calls, [])

    def test_initial_and_new_candidates_reach_identity_review_before_voice_design(self):
        class MergeClient(CumulativeClient):
            def reconcile(self, payload):
                result = super().reconcile(payload)
                if not payload["pending_identity_merges"]:
                    a, b = payload["known_cast"]["characters"][:2]
                    if all(any(item["block_id"] == unit["block_id"] and item["quote"] in unit["text"]
                               for unit in payload["target_units"]) for item in a["evidence"]):
                        result["merges"] = [{"from_character_id": b["id"], "into_character_id": a["id"],
                                             "evidence": deepcopy(a["evidence"])}]
                return result

        client = MergeClient(duplicate=True)
        seen = []

        def audit(source, cast, requests, batches, **kwargs):
            seen.append((deepcopy(cast), deepcopy(kwargs["pending_merges"]), len(client.calls)))
            return cast, [], {"cases": []}

        with patch("epub_tts_narrator.identity.audit_identities", side_effect=audit) as identity:
            document = self.run_analysis(client, identity_consensus=True, cast_reconciliation_rounds=3,
                                         cast_reconciliation_notes=["", "", "Verify remaining source identities."])
        self.assertEqual(identity.call_count, 2)
        reconciliations = [p for _, p in client.calls if p.get("mode") == "casting_coverage"]
        self.assertEqual(len(reconciliations), 4)
        self.assertEqual([len(p["cast_view"]["records"]["characters"]) for p in reconciliations], [2, 3, 4, 5])
        self.assertTrue(seen[0][1])
        self.assertTrue(seen[1][1])
        self.assertEqual(seen[0][2], 1)
        self.assertEqual(seen[1][2], 5)
        self.assertEqual(seen[1][0], document["cast"])
        self.assertEqual(len(document["cast"]["characters"]), 5, "Deferred merge was not applied prematurely")
        self.assertEqual(validate_document(document), [])

    def test_repeated_response_local_ids_allocate_distinct_canonical_identities(self):
        for block in self.source["blocks"]:
            block["text"] += " Extra source context." * 24
        client = CumulativeClient()
        document = self.run_analysis(client, chunk_chars=256, cast_reconciliation_rounds=3)
        reconciliations = [p for _, p in client.calls if p.get("mode") == "casting_coverage"]
        self.assertEqual(len(reconciliations), 4)
        self.assertEqual({p["available_ids"]["characters"][0] for p in reconciliations}, {"c1"})
        self.assertEqual(len({row["id"] for row in document["cast"]["characters"]}), 4)
        self.assertEqual(len({row["voice_id"] for row in document["cast"]["characters"]}), 4)
        self.assertEqual(len(document["cast"]["characters"]), 4)
        self.assertEqual(validate_document(document), [])

    def test_invalid_bounds_fail_before_any_requests(self):
        for value in (None, True, False, 0, -1, 4, 1.0, "2", []):
            client = CumulativeClient()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cast_reconciliation_rounds"):
                self.run_analysis(client, cast_reconciliation_rounds=value)
            self.assertEqual(client.calls, [])
        self.assertFalse(self.cache.exists())

    def test_failure_in_a_dirty_check_blocks_publication_and_preserves_accepted_work(self):
        class FailLater(CumulativeClient):
            def reconcile(self, payload):
                if any(c["name"] == "Jules" for c in payload["known_cast"]["characters"]):
                    raise ProviderError("Fixture transport failure.")
                return super().reconcile(payload)

        client = FailLater()
        with patch("epub_tts_narrator.identity.audit_identities",
                   side_effect=lambda source, cast, *args, **kwargs: (cast, [], {"cases": []})) as identity:
            with self.assertRaises(ProviderError):
                self.run_analysis(client, identity_consensus=True, cast_reconciliation_rounds=3)
        identity.assert_called_once()
        self.assertEqual([p.get("mode", "attribute") for _, p in client.calls],
                         ["casting_discovery", "casting_coverage", "casting_coverage"])
        first_cast = client.calls[-1][1]["cast_view"]["records"]
        self.assertEqual({c["name"] for c in first_cast["characters"]}, {"Mara", "Jules"})
        # Successful responses remain cached, and a normal resume can finish.
        resumed = CumulativeClient()
        with patch("epub_tts_narrator.identity.audit_identities",
                   side_effect=lambda source, cast, *args, **kwargs: (cast, [], {"cases": []})):
            result = self.run_analysis(resumed, cast_reconciliation_rounds=3, identity_consensus=True)
        self.assertEqual([p.get("mode", "attribute") for _, p in resumed.calls],
                         ["casting_coverage", "casting_coverage", "casting_coverage", "voice_ensemble", "attribute"])
        self.assertEqual({c["name"] for c in result["cast"]["characters"]},
                         {"Mara", "Jules", "Quinn", "Rowan"})
        self.assertEqual(validate_document(result), [])
