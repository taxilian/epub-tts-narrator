import json
import unittest

from epub_tts_narrator.adaptive_requests import run_partitioned
from epub_tts_narrator.provider import OutputTruncatedError


class AdaptiveRequestScaleTests(unittest.TestCase):
    def test_large_input_uses_all_capacity_without_a_partition_or_depth_quota(self):
        items = list(range(16_384))
        executed = []

        def build(selected):
            return "Capacity fixture", {"items": selected}

        def execute(system, payload):
            selected = list(payload["items"])
            executed.append(selected)
            return selected

        leaves = run_partitioned(items, build, execute,
                                 fits=lambda system, payload: len(payload["items"]) <= 16)

        self.assertEqual(len(executed), 1024)
        self.assertTrue(all(len(group) == 16 for group in executed))
        self.assertEqual([item for group, _ in leaves for item in group], items)
        self.assertEqual([item for _, group in leaves for item in group], items)

    def test_observations_distinguish_local_sizing_from_provider_failures_without_source_text(self):
        events = []

        def build(items):
            return "Sizing fixture", {"items": items}

        def execute(system, payload):
            if len(payload["items"]) > 1:
                raise OutputTruncatedError("The model output was truncated. Increase the output token limit or reduce the input.")
            return payload["items"]

        result = run_partitioned(["private-source-a", "private-source-b", "private-source-c"], build, execute,
                                 fits=lambda system, payload: len(payload["items"]) <= 2, observe=events.append)

        self.assertEqual(len(result), 3)
        self.assertEqual([row["reason"] for row in events], ["input_budget", "provider_capacity"])
        self.assertEqual([row["target_count"] for row in events], [3, 2])
        self.assertNotIn("private-source", json.dumps(events))
        self.assertTrue(all(len(row["request_fingerprint"]) == 64 for row in events))
