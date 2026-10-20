"""Repair diagnostics identify invalid fields without exposing hidden content."""

from copy import deepcopy
import unittest

from epub_audiobook.attribution_review import _accept_scan, _batches, _evidence, _object, audit_attribution
from test_attribution_review import Requests, correction, fixture


class AttributionReviewDiagnosticTests(unittest.TestCase):
    def error(self, value, visible, **kwargs):
        with self.assertRaises(ValueError) as raised:
            _evidence(value, visible, **kwargs)
        return str(raised.exception)

    def test_bad_quotes_and_unknown_blocks_report_each_index_and_source_id(self):
        visible = {"b1": ["Mara sent her reply."], "b2": ["I agree, she thought."]}
        message = self.error([
            {"block_id": "b1", "quote": "mara sent her reply."},
            {"block_id": "b2", "quote": "I disagree, she thought."},
            {"block_id": "missing", "quote": "Unseen quoted secret"},
        ], visible, label="corrections[4].evidence")
        for index, block_id in enumerate(("b1", "b2", "missing")):
            self.assertIn(f'corrections[4].evidence[{index}] (block_id="{block_id}")', message)
        self.assertIn("not an exact substring", message)
        self.assertIn("unknown or nonvisible source block_id", message)
        self.assertIn("Mara sent her reply.", message)
        self.assertIn("Copy quotations verbatim", message)
        self.assertNotIn("Unseen quoted secret", message)
        self.assertLessEqual(len(message), 1800)

    def test_target_requirement_names_expected_block_and_empty_or_wrong_array(self):
        visible = {"b1": ["Mara speaks."], "b2": ["The wind whistles."]}
        message = self.error([{"block_id": "b2", "quote": "wind"}], visible,
                             required=True, target="b1", label="votes[3].evidence")
        self.assertIn("votes[3].evidence", message)
        self.assertIn('target block_id="b1"', message)
        for value, detail in (([], "empty array"), ({}, "dict"), (None, "NoneType")):
            with self.subTest(value=value):
                message = self.error(value, visible, required=True, target="b1")
                self.assertIn(detail, message)
                self.assertIn('target block_id="b1"', message)

    def test_hidden_slices_and_response_values_are_not_disclosed_or_accepted(self):
        visible = {"b1": ["Only visible text.", "A second visible slice."]}
        for quote in ("SECRET NOT IN THE REQUEST", "text.A second"):
            message = self.error([{"block_id": "b1", "quote": quote}], visible)
            self.assertIn("not an exact substring", message)
            self.assertNotIn(quote, message)
        message = self.error([{"block_id": "hidden", "quote": "PRIVATE BOOK TEXT"}], visible)
        self.assertNotIn("PRIVATE BOOK TEXT", message)
        message = self.error([{"block_id": "b1", "quote": {"credential": "SECRET VALUE"}}], visible)
        self.assertIn("quote must be a nonempty string", message)
        self.assertNotIn("SECRET VALUE", message)

    def test_many_failures_control_characters_and_huge_keys_are_bounded(self):
        message = self.error([{"block_id": "b1", "quote": "wrong"} for _ in range(50)],
                             {"b1": ["\\\n\t" * 10000]})
        self.assertLessEqual(len(message), 1800)
        self.assertIn("50 problem(s)", message)
        self.assertIn("additional problem(s) omitted", message)
        self.assertIn("evidence[0]", message)
        self.assertIn("data only", message)
        self.assertNotIn("\n", message)
        self.assertNotIn("\t", message)
        with self.assertRaises(ValueError) as raised:
            _object({("\n" * 10000) + str(index): "DO NOT ECHO VALUE" for index in range(100)},
                    ("block_id", "quote"), "corrections[0].evidence[0]")
        message = str(raised.exception)
        self.assertLessEqual(len(message), 1800)
        self.assertIn("Expected 2 keys; received 100", message)
        self.assertIn("(+96 more)", message)
        self.assertNotIn("DO NOT ECHO VALUE", message)
        self.assertNotIn("\n", message)

    def test_schema_errors_report_missing_extra_fields_and_actual_type(self):
        message = self.error([None, {"block_id": "b1", "quotation": "wrong key", "payload": "SECRET VALUE"},
                              {"block_id": [], "quote": "text"}], {"b1": ["text"]})
        self.assertIn("evidence[0]", message)
        self.assertIn("Expected an object; received NoneType", message)
        self.assertIn("evidence[1]", message)
        self.assertIn('Missing: "quote"', message)
        self.assertIn('Unexpected: "quotation", "payload"', message)
        self.assertIn("evidence[2]", message)
        self.assertIn("block_id must be a string", message)
        self.assertNotIn("SECRET VALUE", message)

    def test_nested_scan_and_jury_errors_include_complete_response_paths(self):
        document = fixture(count=1)
        batch = _batches(document["source"], 60000)[0]
        originals = {"b1": document["segments"]}
        proposed = correction(batch["target_units"][0])
        proposed["segments"][1]["delivery"] = {"directions": "soft"}
        with self.assertRaises(ValueError) as raised:
            _accept_scan({"corrections": [proposed]}, document, batch, originals)
        self.assertIn("corrections[0].segments[1].delivery", str(raised.exception))
        self.assertIn('Missing: "direction", "evidence"', str(raised.exception))
        proposed = correction(batch["target_units"][0])
        proposed["segments"][1]["evidence"] = [{"block_id": "b1", "quote": "wrong"}]
        with self.assertRaises(ValueError) as raised:
            _accept_scan({"corrections": [proposed]}, document, batch, originals)
        self.assertIn('corrections[0].segments[1].evidence[0] (block_id="b1")', str(raised.exception))

        def invalid_vote(response, payload):
            if payload["mode"] == "attribution_review_vote":
                response["votes"][0]["evidence"] = [{"block_id": "b1", "quote": "wrong"}]
            return response
        result = audit_attribution(document, Requests(modify=invalid_vote))
        votes = result["provenance"]["attribution_review"]["changes"][0]["votes"]
        self.assertTrue(all(row["verdict"] == "error" and "votes[0].evidence[0]" in row["reason"] for row in votes))
        self.assertEqual(result["segments"], document["segments"])

    def test_valid_evidence_and_schema_are_unchanged_with_strict_quote_rules(self):
        visible = {"b1": ["Mara said “Yes.”", "Then she left."]}
        evidence = [{"block_id": "b1", "quote": "“Yes.”"}, {"block_id": "b1", "quote": "Then"}]
        before = deepcopy((evidence, visible))
        self.assertIsNone(_evidence(evidence, visible, required=True, target="b1"))
        self.assertIsNone(_evidence([], visible))
        self.assertIsNone(_object({"quote": "Then", "block_id": "b1"}, ("block_id", "quote"), "Evidence"))
        self.assertEqual((evidence, visible), before)
        for quote in ("", " ", "yes", '"Yes."'):
            with self.subTest(quote=quote):
                self.error([{"block_id": "b1", "quote": quote}], visible)
