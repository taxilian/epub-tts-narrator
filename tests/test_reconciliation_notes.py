"""Source-review observations affect verification, never discovery or work quotas."""
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

    def test_changed_observations_reuse_discovery_and_preserve_the_resolved_cast(self):
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
                self.assertEqual([p.get("mode") for _, p in resumed.calls], ["casting_coverage"] * 4)
                expected_notes = "Global source observation.\n" + notes[2]
                self.assertTrue(all(payload["review_notes"] == expected_notes for _, payload in resumed.calls))
                self.assertEqual(actual["cast"], accepted["cast"])
                self.assertTrue(all(payload["cast_view"]["records"]["characters"] for _, payload in resumed.calls))
                fresh = CumulativeClient()
                self.run_analysis(fresh, self.root / f"fresh-{whole_book}", whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=3, cast_review_notes="Global source observation.",
                    cast_reconciliation_notes=notes)
                self.assertEqual(fresh.calls[0], old.calls[0], "Discovery is independent of later review observations")
                self.assertEqual((self.source, notes), before)
                self.assertEqual(actual["provenance"]["cast_reconciliation_notes"], notes)
                self.assertEqual({c["name"] for c in actual["cast"]["characters"]}, {"Mara", "Jules", "Quinn", "Rowan"})
                self.assertEqual(validate_document(actual), [])
                replay = CumulativeClient()
                self.assertEqual(self.run_analysis(replay, cache, whole_book_discovery=whole_book,
                    cast_reconciliation_rounds=3, cast_review_notes="Global source observation.",
                    cast_reconciliation_notes=notes), actual)
                self.assertEqual(replay.calls, [])

    def test_all_observations_reach_source_review_and_both_identity_stages(self):
        client = CumulativeClient()
        notes = ["First source lead.", "", "Third source lead."]
        seen = []

        def identity(source, cast, *args, **kwargs):
            seen.append((deepcopy(cast), deepcopy(kwargs), len(client.calls)))
            return cast, [], {"cases": []}

        with patch("epub_tts_narrator.identity.audit_identities", side_effect=identity) as audit:
            document = self.run_analysis(client, self.root / "routing", identity_consensus=True,
                                        cast_reconciliation_rounds=3, cast_reconciliation_notes=notes)
        self.assertEqual(audit.call_count, 2)
        self.assertEqual([row[2] for row in seen], [1, 5])
        self.assertEqual(seen[-1][0], document["cast"])
        expected = "First source lead.\nThird source lead."
        self.assertTrue(all(row[1]["review_notes"] == expected for row in seen))
        self.assertNotIn("review_notes", client.calls[0][1])
        checks = [payload for _, payload in client.calls if payload.get("mode") == "casting_coverage"]
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(payload["review_notes"] == expected for payload in checks))
        self.assertNotIn("review_notes", client.calls[-1][1])

    def test_invalid_values_reject_before_requests_and_do_not_echo_notes(self):
        for notes in (None, True, 1, "private-fixture-value", (), {}, [True], [1], [[]]):
            client = CumulativeClient()
            with self.subTest(notes=notes), self.assertRaisesRegex(ValueError, "cast_reconciliation_notes") as error:
                self.run_analysis(client, self.root / "invalid", cast_reconciliation_rounds=3,
                                  cast_reconciliation_notes=notes)
            self.assertNotIn("private-fixture-value", str(error.exception))
            self.assertEqual(client.calls, [])
        self.assertFalse((self.root / "invalid").exists())
        result = self.run_analysis(CumulativeClient(), self.root / "many-notes",
                                   cast_reconciliation_notes=["", "", "", "", ""])
        self.assertEqual({character["name"] for character in result["cast"]["characters"]},
                         {"Mara", "Jules", "Quinn", "Rowan"})

    def test_notes_obey_existing_request_budget(self):
        client = CumulativeClient()
        with self.assertRaisesRegex(ValueError, "max_request_chars") as caught:
            self.run_analysis(client, self.root / "budget", cast_reconciliation_rounds=3,
                              cast_reconciliation_notes=["", "", "Source observation. " * 20000])
        self.assertEqual([p.get("mode") for _, p in client.calls], ["casting_discovery"])
        self.assertNotIn("Source observation.", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
