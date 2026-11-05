from collections import Counter
from copy import deepcopy
import json
import unittest

from epub_tts_narrator.pipeline import DEFAULT_POLICY, fingerprint, seed_cast
from epub_tts_narrator.voice_contrast import contrast_findings
from epub_tts_narrator.voice_ensemble import _contrast_context, design_ensemble
from test_contracts import valid_document
from test_voice_ensemble import RecordingRequests


class VoiceEnsembleContextTests(unittest.TestCase):
    def test_large_cast_fits_budget_without_dropping_any_contrast_comparison(self):
        # Given a cast whose pairwise findings exceed its request budget.
        source = valid_document()["source"]
        cast = seed_cast()
        cast["voices"].extend({"id": f"series_voice_{index:04d}_" + "x" * 48,
                               "description": "A distinct local voice.", "traits": []}
                              for index in range(137))
        original = deepcopy(cast)
        findings = contrast_findings(cast["voices"])
        requests = RecordingRequests()
        requests.max_request_chars = 300000
        self.assertGreater(len(json.dumps(findings)), requests.max_request_chars)

        # When the complete cast is sent through the production ensemble boundary.
        updated, provenance = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY)

        # Then every original finding can be reconstructed and nothing else is trimmed.
        system, payload = requests.calls[0]
        compact = payload["contrast_findings"]
        expanded = [dict(kind=group["kind"],
                         voice_ids=[compact["voice_ids"][index] for index in pair],
                         matching_dimensions=group["matching_dimensions"],
                         differing_dimensions=group["differing_dimensions"])
                    for group in compact["groups"] for pair in group["pairs"]]
        self.assertEqual(Counter(map(fingerprint, expanded)), Counter(map(fingerprint, findings)))
        self.assertEqual(len(expanded), len(cast["voices"]) * (len(cast["voices"]) - 1) // 2)
        self.assertEqual(payload["cast"], original)
        self.assertEqual(provenance["contrast_findings_before"], findings)
        self.assertEqual(provenance["contrast_findings_after"], findings)
        size = len(system) + len(json.dumps(payload, ensure_ascii=False))
        self.assertLessEqual(size, requests.max_request_chars)
        self.assertEqual(provenance["request_characters"], size)
        self.assertEqual(cast, original)
        self.assertEqual(updated, original)

    def test_grouping_preserves_distinct_kinds_dimensions_and_repeated_pairs(self):
        # Given varied known profiles and a repeated finding that must not be deduplicated.
        cast = seed_cast()
        for index in range(24):
            cast["voices"].append({"id": f"voice_{index}", "description": "A local voice.",
                                   "traits": [{"name": "pitch", "value": "high" if index % 2 else "low",
                                               "basis": "casting", "evidence": []}]})
        findings = contrast_findings(cast["voices"])
        findings.append(deepcopy(findings[0]))
        original = deepcopy(findings)

        # When the findings are prepared as ensemble request context.
        compact = _contrast_context(findings)

        # Then grouping retains pair multiplicity, order within pairs, kinds and dimensions.
        expanded = [dict(kind=group["kind"],
                         voice_ids=[compact["voice_ids"][index] for index in pair],
                         matching_dimensions=group["matching_dimensions"],
                         differing_dimensions=group["differing_dimensions"])
                    for group in compact["groups"] for pair in group["pairs"]]
        self.assertEqual(Counter(map(fingerprint, expanded)), Counter(map(fingerprint, original)))
        self.assertEqual(findings, original)


if __name__ == "__main__":
    unittest.main()
