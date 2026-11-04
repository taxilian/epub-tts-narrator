from copy import deepcopy
import unittest

from epub_tts_narrator.analysis_plan import plan_analysis
from epub_tts_narrator.config import DEFAULTS
from test_contracts import valid_document


class AnalysisPlanTests(unittest.TestCase):
    def test_twenty_million_character_book_uses_the_same_automatic_policy(self):
        source = valid_document()["source"]
        source["blocks"][0]["text"] = "word " * 4_000_000

        result = plan_analysis(source, DEFAULTS["analysis"], DEFAULTS["performance_policy"])

        self.assertEqual(result["text_characters"], 20_000_010)
        self.assertEqual(result["source_units"], 1668)
        self.assertEqual(result["adaptive_source_tree_node_bound"], 3335)
        self.assertEqual(result["casting_strategy"], "adaptive_evidence_first")

    def test_work_bound_scales_with_book_without_a_partition_setting(self):
        source = valid_document()["source"]
        block = source["blocks"][0]
        source["blocks"] = [dict(deepcopy(block), id=f"b{index}") for index in range(17)]
        source["chapters"][0]["block_ids"] = [row["id"] for row in source["blocks"]]

        result = plan_analysis(source, DEFAULTS["analysis"], DEFAULTS["performance_policy"])

        self.assertEqual(result["source_units"], 17)
        self.assertEqual(result["adaptive_source_tree_node_bound"], 33)
        self.assertEqual(result["work_limits"], "derived_from_source")
        self.assertTrue(result["request_count_is_estimate"])

    def test_old_binary_mode_and_round_count_do_not_change_automatic_work(self):
        source = valid_document()["source"]
        settings = deepcopy(DEFAULTS["analysis"])
        before = plan_analysis(source, settings, DEFAULTS["performance_policy"])
        settings.update(whole_book_discovery=True, cast_reconciliation_rounds=3)

        after = plan_analysis(source, settings, DEFAULTS["performance_policy"])

        self.assertEqual(after, before)
        self.assertEqual(after["baseline_model_requests"], 5)

    def test_plan_preserves_inputs_and_distinguishes_optional_identity_work(self):
        source = valid_document()["source"]
        original = deepcopy(source)
        settings = deepcopy(DEFAULTS["analysis"])
        settings["identity_consensus"] = False

        result = plan_analysis(source, settings, DEFAULTS["performance_policy"])

        self.assertEqual(source, original)
        self.assertEqual(result["identity_scan_requests_minimum"], 0)
        self.assertEqual(result["baseline_model_requests"], 4)
