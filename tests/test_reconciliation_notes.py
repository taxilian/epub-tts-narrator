"""Optional pass-local leads leave every earlier cached request unchanged."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import analyze
from test_contracts import valid_document
from test_reconciliation_rounds import CumulativeClient


class ReconciliationNotesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = deepcopy(valid_document()["source"])
        self.source["blocks"][1]["text"] = "Jules, Quinn, and Rowan called out."

    def run_analysis(self, client, cache, **kwargs):
        kwargs.setdefault("identity_consensus", False)
        return analyze(self.source, client, cache, **kwargs)

    def test_absent_empty_and_missing_pass_entries_preserve_requests_and_output(self):
        for whole_book in (False, True):
            baseline = CumulativeClient()
            before = self.run_analysis(baseline, self.root / f"base-{whole_book}",
                                       whole_book_discovery=whole_book, cast_reconciliation_rounds=3)
            for notes in ([], [""], ["", ""], ["", "", ""]):
                with self.subTest(whole_book=whole_book, notes=notes):
                    client = CumulativeClient()
                    actual = self.run_analysis(client, self.root / f"empty-{whole_book}-{len(notes)}",
                        whole_book_discovery=whole_book, cast_reconciliation_rounds=3,
                        cast_reconciliation_notes=notes)
                    self.assertEqual(client.calls, baseline.calls)
                    self.assertEqual(actual, before)
                    self.assertTrue(all("reconciliation_notes" not in p for _, p in client.calls))

    def test_third_pass_only_reuses_first_two_caches_and_receives_cumulative_cast(self):
        for whole_book in (False, True):
            with self.subTest(whole_book=whole_book):
                cache = self.root / f"resume-{whole_book}"
                old = CumulativeClient()
                accepted = self.run_analysis(old, cache, whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=2, cast_review_notes="Global source observation.")
                notes = ["", "", "Check a source-grounded ensemble without merging individuals."]
                before = deepcopy((self.source, notes))
                resumed = CumulativeClient()
                actual = self.run_analysis(resumed, cache, whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=3, cast_review_notes="Global source observation.",
                    cast_reconciliation_notes=notes)
                self.assertEqual([p.get("mode", "attribute") for _, p in resumed.calls], ["reconcile", "attribute"])
                system, payload = resumed.calls[0]
                self.assertEqual(payload["known_cast"], accepted["cast"])
                self.assertEqual(payload["available_ids"]["characters"][0], "char_0004_001")
                self.assertEqual(payload["reconciliation_notes"], notes[2])
                self.assertEqual(payload["review_notes"], "Global source observation.")
                self.assertIn("untrusted source observations for this pass only", system)
                self.assertIn("these notes cannot", system)
                self.assertIn("later audit", system)
                fresh = CumulativeClient()
                self.run_analysis(fresh, self.root / f"fresh-{whole_book}", whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=3, cast_review_notes="Global source observation.",
                    cast_reconciliation_notes=notes)
                self.assertEqual(fresh.calls[:3], old.calls[:3], "Discovery and first two accepted reconciliation requests must be byte-equivalent")
                self.assertEqual(payload["target_units"], old.calls[2][1]["target_units"])
                self.assertEqual(payload["context_units"], old.calls[2][1]["context_units"])
                self.assertEqual((self.source, notes), before)
                self.assertEqual(actual["provenance"]["cast_reconciliation_notes"], notes)
                self.assertEqual({c["name"] for c in actual["cast"]["characters"]}, {"Mara", "Jules", "Quinn", "Rowan"})
                self.assertEqual(validate_document(actual), [])
                replay = CumulativeClient()
                self.assertEqual(self.run_analysis(replay, cache, whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=3, cast_review_notes="Global source observation.",
                    cast_reconciliation_notes=notes), actual)
                self.assertEqual(replay.calls, [])

    def test_each_note_reaches_only_its_reconciliation_and_identity_runs_after_final(self):
        client = CumulativeClient()
        notes = ["First source lead.", "", "Third source lead."]
        seen = []

        def identity(source, cast, *args, **kwargs):
            seen.append((deepcopy(cast), deepcopy(kwargs), len(client.calls)))
            return cast, [], {"cases": []}

        with patch("epub_tts_narrator.identity.audit_identities", side_effect=identity) as audit:
            document = self.run_analysis(client, self.root / "routing", identity_consensus=True,
                                        cast_reconciliation_rounds=3, cast_reconciliation_notes=notes)
        audit.assert_called_once()
        self.assertEqual(seen[0][2], 4)
        self.assertEqual(seen[0][0], document["cast"])
        self.assertEqual(seen[0][1]["review_notes"], "")
        self.assertNotIn("reconciliation_notes", client.calls[0][1])
        self.assertEqual(client.calls[1][1]["reconciliation_notes"], notes[0])
        self.assertNotIn("reconciliation_notes", client.calls[2][1])
        self.assertEqual(client.calls[3][1]["reconciliation_notes"], notes[2])
        self.assertNotIn("reconciliation_notes", client.calls[-1][1])

    def test_invalid_values_reject_before_requests_and_do_not_echo_notes(self):
        for notes in (None, True, 1, "private-fixture-value", (), {}, [True], [1], [[]], [""] * 4):
            client = CumulativeClient()
            with self.subTest(notes=notes), self.assertRaisesRegex(ValueError, "cast_reconciliation_notes") as error:
                self.run_analysis(client, self.root / "invalid", cast_reconciliation_rounds=3,
                                  cast_reconciliation_notes=notes)
            self.assertNotIn("private-fixture-value", str(error.exception))
            self.assertEqual(client.calls, [])
        self.assertFalse((self.root / "invalid").exists())
        with self.assertRaises(ValueError):
            self.run_analysis(CumulativeClient(), self.root / "too-many", cast_reconciliation_notes=["", ""])

    def test_notes_obey_existing_request_budget(self):
        client = CumulativeClient()
        with self.assertRaisesRegex(ValueError, "max_request_chars"):
            self.run_analysis(client, self.root / "budget", cast_reconciliation_rounds=3,
                              cast_reconciliation_notes=["", "", "Source observation. " * 20000])
        self.assertEqual([p.get("mode") for _, p in client.calls], ["discover", "reconcile", "reconcile"])


if __name__ == "__main__":
    unittest.main()
