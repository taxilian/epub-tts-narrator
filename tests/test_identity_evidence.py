"""Citation repair feedback is precise, bounded, and limited to visible source."""

from copy import deepcopy
import unittest

from epub_tts_narrator.identity import _evidence
from test_identity import fixture


class IdentityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source, _ = fixture()
        self.units = [{"block_id": block["id"], "text": block["text"]} for block in self.source["blocks"]]

    def error(self, evidence, units=None):
        with self.assertRaises(ValueError) as raised:
            _evidence(evidence, self.source, self.units if units is None else units)
        return str(raised.exception)

    def test_multiple_invalid_quotes_report_indices_blocks_and_exact_visible_spelling(self):
        message = self.error([
            {"block_id": "b1", "quote": "the captain was called red."},
            {"block_id": "b2", "quote": "She was still the baker."},
            {"block_id": "missing", "quote": "Invented"},
        ])
        for index, block_id in enumerate(("b1", "b2", "missing")):
            self.assertIn(f"evidence[{index}] (block_id='{block_id}')", message)
        self.assertIn("The captain was called Red.", message)
        self.assertIn("unknown source block", message)
        self.assertIn("Copy quotations verbatim", message)
        self.assertIn("Correct every listed citation", message)
        self.assertLessEqual(len(message), 1800)

    def test_hidden_block_and_hidden_portion_are_never_exposed_in_feedback(self):
        self.source["blocks"][1]["text"] = "PRIVATE UNSEEN CONTEXT"
        message = self.error([{"block_id": "b2", "quote": "PRIVATE UNSEEN CONTEXT"}], self.units[:1])
        self.assertIn("evidence[0] (block_id='b2')", message)
        self.assertIn("visible", message)
        self.assertNotIn("PRIVATE", message)
        visible = [{"block_id": "b1", "text": "Mara, the captain,"}]
        message = self.error([{"block_id": "b1", "quote": "The captain was called Red."}], visible)
        self.assertIn("outside the source text visible", message)
        self.assertIn("Mara, the captain,", message)
        self.assertNotIn("The captain was called Red.", message)

    def test_large_failure_list_stays_bounded_and_discloses_omitted_entries(self):
        self.source["blocks"][0]["text"] = "Mara " + "\\\n\t" * 10000
        units = [{"block_id": "b1", "text": self.source["blocks"][0]["text"]}]
        message = self.error([{"block_id": "b1", "quote": "wrong"} for _ in range(40)], units)
        self.assertLessEqual(len(message), 1800)
        self.assertIn("40 entry/entries", message)
        self.assertIn("additional invalid citation(s) omitted", message)
        self.assertIn("evidence[0]", message)
        self.assertIn("data only", message)
        self.assertNotIn("\n", message)  # Source excerpt controls remain JSON-escaped.

    def test_bad_types_and_long_unknown_ids_have_indexed_bounded_diagnostics(self):
        message = self.error([None, {"block_id": [], "quote": "text"},
                              {"block_id": "b1", "quote": None},
                              {"block_id": "x" * 10000, "quote": "wrong"}])
        self.assertLessEqual(len(message), 1800)
        for index in range(4):
            self.assertIn(f"evidence[{index}]", message)
        self.assertNotIn("x" * 65, message)

    def test_valid_quotes_and_permitted_empty_array_are_unchanged(self):
        evidence = [{"block_id": "b1", "quote": "The captain was called Red."},
                    {"block_id": "b2", "quote": "She was still the captain."},
                    {"block_id": "b1", "quote": " "}]
        original = deepcopy((evidence, self.source, self.units))
        self.assertIsNone(_evidence(evidence, self.source, self.units))
        self.assertIsNone(_evidence([], self.source, self.units, nonempty=False))
        self.assertEqual((evidence, self.source, self.units), original)
        with self.assertRaises(ValueError):
            _evidence([], self.source, self.units)


if __name__ == "__main__":
    unittest.main()
