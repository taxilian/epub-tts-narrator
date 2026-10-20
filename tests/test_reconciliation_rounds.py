"""Cumulative reconciliation preserves accepted cast and legacy request identity."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_audiobook.contracts import validate_document
from epub_audiobook.pipeline import analyze
from epub_audiobook.provider import ProviderError
from test_contracts import valid_document
from test_pipeline import FakeClient


class CumulativeClient(FakeClient):
    def reconcile(self, payload):
        known = payload["known_cast"]
        name = next(name for name in ("Jules", "Quinn", "Rowan")
                    if name not in {c["name"] for c in known["characters"]})
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
                         ["discover", "reconcile", "attribute"])
        self.assertEqual(implicit["provenance"]["cast_reconciliation_rounds"], 1)
        replay = FakeClient()
        self.assertEqual(self.run_analysis(replay, cast_reconciliation_rounds=1), implicit)
        self.assertEqual(replay.calls, [])
        fresh = FakeClient()
        analyze(self.source, fresh, self.cache / "explicit", identity_consensus=False,
                cast_reconciliation_rounds=1)
        self.assertEqual(fresh.calls, client.calls)
        self.assertTrue(all("cast_reconciliation_rounds" not in payload for _, payload in client.calls))

    def test_extra_round_reuses_first_round_cache_and_keeps_prior_additions(self):
        for whole_book in (False, True):
            with self.subTest(whole_book=whole_book):
                self.cache = Path(self.temp.name) / str(whole_book)
                first = CumulativeClient()
                baseline = self.run_analysis(first, whole_book_discovery=whole_book,
                                             cast_review_notes="Check overlooked distinct speakers.")
                second = CumulativeClient()
                original = deepcopy(self.source)
                result = self.run_analysis(second, whole_book_discovery=whole_book,
                                           cast_review_notes="Check overlooked distinct speakers.",
                                           cast_reconciliation_rounds=2)
                self.assertEqual([p.get("mode", "attribute") for _, p in second.calls],
                                 ["reconcile", "attribute"])
                later_system, later = second.calls[0]
                first_system, first_payload = next((s, p) for s, p in first.calls if p.get("mode") == "reconcile")
                self.assertEqual(later["known_cast"], baseline["cast"])
                self.assertEqual(later["target_units"], first_payload["target_units"])
                self.assertEqual(later["context_units"], first_payload["context_units"])
                self.assertEqual(later["scene_summaries"], first_payload["scene_summaries"])
                self.assertEqual(later["review_notes"], first_payload["review_notes"])
                self.assertEqual(later["available_ids"]["characters"][0], "char_0003_001")
                self.assertIn("later cumulative", later_system)
                self.assertNotIn("later cumulative", first_system)
                self.assertEqual({c["name"] for c in result["cast"]["characters"]}, {"Mara", "Jules", "Quinn"})
                self.assertEqual(result["provenance"]["cast_reconciliation_rounds"], 2)
                self.assertEqual(self.source, original)
                self.assertEqual(validate_document(result), [])
                replay = CumulativeClient()
                self.assertEqual(self.run_analysis(replay, whole_book_discovery=whole_book,
                                                   cast_review_notes="Check overlooked distinct speakers.",
                                                   cast_reconciliation_rounds=2), result)
                self.assertEqual(replay.calls, [])

    def test_all_rounds_precede_identity_audit_and_preserve_pending_merges(self):
        class MergeClient(CumulativeClient):
            def reconcile(self, payload):
                result = super().reconcile(payload)
                if not payload["pending_identity_merges"]:
                    a, b = payload["known_cast"]["characters"][:2]
                    result["merges"] = [{"from_character_id": b["id"], "into_character_id": a["id"],
                                         "evidence": deepcopy(a["evidence"])}]
                return result

        client = MergeClient(duplicate=True)
        seen = []

        def audit(source, cast, requests, batches, **kwargs):
            seen.append((deepcopy(cast), deepcopy(kwargs["pending_merges"]), len(client.calls)))
            return cast, [], {"cases": []}

        with patch("epub_audiobook.identity.audit_identities", side_effect=audit) as identity:
            document = self.run_analysis(client, identity_consensus=True, cast_reconciliation_rounds=3,
                                         cast_reconciliation_notes=["", "", "Verify remaining source identities."])
        identity.assert_called_once()
        reconciliations = [p for _, p in client.calls if p.get("mode") == "reconcile"]
        self.assertEqual(len(reconciliations), 3)
        self.assertEqual([len(p["known_cast"]["characters"]) for p in reconciliations], [2, 3, 4])
        self.assertEqual(reconciliations[0]["pending_identity_merges"], [])
        self.assertEqual(reconciliations[1]["pending_identity_merges"], seen[0][1])
        self.assertEqual(reconciliations[2]["pending_identity_merges"], seen[0][1])
        self.assertEqual(len(seen[0][1]), 1)
        self.assertEqual(seen[0][2], 4, "Identity runs after discovery and all three reconciliations")
        self.assertEqual(seen[0][0], document["cast"])
        self.assertEqual(len(document["cast"]["characters"]), 5, "Deferred merge was not applied prematurely")
        self.assertEqual(validate_document(document), [])

    def test_later_round_available_ids_follow_all_discovery_batches(self):
        for block in self.source["blocks"]:
            block["text"] += " Extra source context." * 24
        client = CumulativeClient()
        document = self.run_analysis(client, chunk_chars=256, cast_reconciliation_rounds=3)
        discoveries = [p for _, p in client.calls if p.get("mode") == "discover"]
        reconciliations = [p for _, p in client.calls if p.get("mode") == "reconcile"]
        self.assertGreater(len(discoveries), 1)
        self.assertEqual([p["available_ids"]["characters"][0] for p in reconciliations],
                         [f"char_{len(discoveries) + round_:04d}_001" for round_ in (1, 2, 3)])
        self.assertEqual(len(document["cast"]["characters"]), 4)
        self.assertEqual(validate_document(document), [])

    def test_invalid_bounds_fail_before_any_requests(self):
        for value in (None, True, False, 0, -1, 4, 1.0, "2", []):
            client = CumulativeClient()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cast_reconciliation_rounds"):
                self.run_analysis(client, cast_reconciliation_rounds=value)
            self.assertEqual(client.calls, [])
        self.assertFalse(self.cache.exists())

    def test_failure_in_later_round_stops_before_identity_or_attribution(self):
        class FailLater(CumulativeClient):
            def reconcile(self, payload):
                if any(c["name"] == "Jules" for c in payload["known_cast"]["characters"]):
                    raise ProviderError("Fixture transport failure.")
                return super().reconcile(payload)

        client = FailLater()
        with patch("epub_audiobook.identity.audit_identities") as identity:
            with self.assertRaises(ProviderError):
                self.run_analysis(client, identity_consensus=True, cast_reconciliation_rounds=3)
        identity.assert_not_called()
        self.assertEqual([p.get("mode", "attribute") for _, p in client.calls],
                         ["discover", "reconcile", "reconcile"])
        first_cast = client.calls[-1][1]["known_cast"]
        self.assertEqual({c["name"] for c in first_cast["characters"]}, {"Mara", "Jules"})
        # Successful responses remain cached, and a normal resume can finish.
        resumed = CumulativeClient()
        with patch("epub_audiobook.identity.audit_identities",
                   side_effect=lambda source, cast, *args, **kwargs: (cast, [], {"cases": []})):
            result = self.run_analysis(resumed, cast_reconciliation_rounds=3, identity_consensus=True)
        self.assertEqual([p.get("mode", "attribute") for _, p in resumed.calls],
                         ["reconcile", "reconcile", "attribute"])
        self.assertEqual({c["name"] for c in result["cast"]["characters"]},
                         {"Mara", "Jules", "Quinn", "Rowan"})
        self.assertEqual(validate_document(result), [])
