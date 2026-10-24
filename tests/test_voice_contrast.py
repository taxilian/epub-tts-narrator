"""The deterministic contrast audit compares declared dimensions conservatively."""

import unittest

from epub_tts_narrator.voice_contrast import contrast_findings, known_dimensions


def voice(identifier, traits=(), **extra):
    return dict({"id": identifier, "description": "A clear voice.",
                 "traits": [{"name": name, "value": value, "basis": "casting", "evidence": []}
                            for name, value in traits]}, **extra)


class VoiceContrastTests(unittest.TestCase):
    def test_identical_known_profiles_are_reported_with_ids_and_dimension_names(self):
        finding = contrast_findings([
            voice("a", (("pitch", "low"), ("pace", "slow"))),
            voice("b", (("pitch", "low"), ("pace", "slow"))),
        ])
        self.assertEqual(len(finding), 1)
        self.assertEqual(finding[0]["kind"], "identical_known_profile")
        self.assertEqual(finding[0]["voice_ids"], ["a", "b"])
        self.assertEqual(finding[0]["matching_dimensions"], ["pace", "pitch"])
        self.assertEqual(finding[0]["differing_dimensions"], [])

    def test_fewer_than_two_demonstrably_different_dimensions_are_reported(self):
        finding = contrast_findings([
            voice("a", (("pitch", "low"), ("pace", "slow"), ("timbre", "bright"))),
            voice("b", (("pitch", "high"), ("pace", "slow"), ("timbre", "bright"))),
        ])
        self.assertEqual(len(finding), 1)
        self.assertEqual(finding[0]["kind"], "insufficient_known_contrast")
        self.assertEqual(finding[0]["differing_dimensions"], ["pitch"])

    def test_two_demonstrably_different_dimensions_produce_no_finding(self):
        self.assertEqual(contrast_findings([
            voice("a", (("pitch", "low"), ("pace", "slow"))),
            voice("b", (("pitch", "high"), ("pace", "quick"))),
        ]), [])

    def test_missing_or_unrecognized_dimensions_are_unknown_never_differences(self):
        self.assertEqual(known_dimensions(voice("a", (("pitch", "low"), ("Gravity", "heavy")))),
                         {"pitch": "low"})
        self.assertEqual(contrast_findings([
            voice("a", (("pitch", "low"),)),
            voice("b", (("pace", "slow"),)),
        ]), [{"kind": "insufficient_known_contrast", "voice_ids": ["a", "b"],
              "matching_dimensions": [], "differing_dimensions": []}])

    def test_spelling_and_synonym_normalization_count_justified_matches(self):
        finding = contrast_findings([
            voice("a", (("Tempo", "slow"), ("Timbre.", "bright"))),
            voice("b", (("pace", "slow"), ("tone color", "bright"))),
        ])
        self.assertEqual(finding[0]["kind"], "identical_known_profile")
        self.assertEqual(finding[0]["matching_dimensions"], ["pace", "timbre"])
        self.assertEqual(known_dimensions(voice("a", (("Pitches", "low"),))),
                         {"pitch": "low"})

    def test_case_and_whitespace_only_value_differences_do_not_count(self):
        self.assertEqual(contrast_findings([
            voice("a", (("pitch", "Low  resonant"),)),
            voice("b", (("pitch", "low resonant"),)),
        ])[0]["kind"], "identical_known_profile")

    def test_aliases_and_shared_voice_duplicates_never_pair(self):
        profiles = [voice("a", (("pitch", "low"),)), voice("b", (("pitch", "high"),))]
        self.assertEqual(contrast_findings(profiles, canonical_id=lambda value: "same"), [])
        self.assertEqual(contrast_findings(profiles, canonical_id=lambda value: value),
                         [{"kind": "insufficient_known_contrast", "voice_ids": ["a", "b"],
                           "matching_dimensions": [], "differing_dimensions": ["pitch"]}])

    def test_alias_group_comparisons_use_the_primary_profile(self):
        finding = contrast_findings(
            [voice("a1", (("pitch", "low"),)), voice("a2", (("pitch", "high"),)), voice("b", (("pitch", "low"),))],
            canonical_id=lambda value: "a" if value.startswith("a") else value)
        self.assertEqual([row["voice_ids"] for row in finding], [["a1", "b"]])

    def test_findings_carry_no_values_quotations_or_secrets(self):
        finding = contrast_findings([voice("a", (("accent", "Glaswegian"),)),
                                     voice("b", (("accent", "Glaswegian"),))])
        self.assertEqual(set(finding[0]), {"kind", "voice_ids", "matching_dimensions", "differing_dimensions"})
        self.assertNotIn("Glaswegian", str(finding))


if __name__ == "__main__":
    unittest.main()
