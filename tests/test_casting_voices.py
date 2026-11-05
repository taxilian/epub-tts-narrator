from copy import deepcopy
import unittest

from epub_tts_narrator.casting_voices import consolidate_merged_voices
from epub_tts_narrator.pipeline import check_cast, merge_patch
from test_identity import fixture


class CastingVoiceTests(unittest.TestCase):
    def setUp(self):
        self.source, self.before = fixture()
        self.source["blocks"][0]["text"] += " Her voice had a rough edge."
        self.trait = {"name": "timbre", "value": "rough edge", "basis": "text",
                      "evidence": [{"block_id": "b1", "quote": "rough edge"}]}
        self.before["voices"][2]["traits"] = [deepcopy(self.trait)]
        self.merge = {"from_character_id": "c2", "into_character_id": "c1",
                      "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}

    def merged(self):
        patch = {"characters": [], "roles": [], "voices": [], "issues": [],
                 "summary": "One captain.", "merges": [self.merge]}
        return merge_patch(self.source, self.before, patch,
                           {key: [] for key in ("characters", "roles", "voices")})[0]

    def test_superseded_voice_is_archived_and_its_text_traits_reach_the_canonical_voice(self):
        merged = self.merged()
        untouched = deepcopy(merged)

        result, archive, issues = consolidate_merged_voices(merged, [self.before], [self.merge])

        self.assertEqual({voice["id"] for voice in result["voices"]}, {"vn", "v1"})
        self.assertEqual(next(voice for voice in result["voices"] if voice["id"] == "v1")["traits"], [self.trait])
        self.assertEqual(archive[0]["voice"], self.before["voices"][2])
        self.assertEqual(archive[0]["canonical_voice_id"], "v1")
        self.assertEqual(issues, [])
        self.assertEqual(merged, untouched)
        check_cast(self.source, result)

    def test_unrelated_unused_profiles_and_inherited_profiles_are_not_discarded(self):
        self.before["voices"].append({"id": "v_future", "description": "An unassigned proposal.", "traits": []})
        self.before["voices"][2]["series_origin"] = {"kind": "series_casting_reuse"}
        merged = self.merged()

        result, archive, _ = consolidate_merged_voices(merged, [self.before], [self.merge])

        self.assertEqual(result, merged)
        self.assertEqual(archive, [])

    def test_conflicting_text_traits_remain_exact_and_create_a_review_issue(self):
        original_trait = dict(deepcopy(self.trait), value="smooth edge")
        self.before["voices"][1]["traits"] = [original_trait]
        merged = self.merged()

        result, archive, issues = consolidate_merged_voices(merged, [self.before], [self.merge])

        target = next(voice for voice in result["voices"] if voice["id"] == "v1")
        self.assertEqual(target["traits"], [original_trait, self.trait])
        self.assertEqual(issues[0]["kind"], "casting_traits")
        self.assertEqual(issues[0]["target"], "v1")
        self.assertEqual(len(archive), 1)

    def test_repeated_same_fact_combines_all_evidence_without_repeating_voice_direction(self):
        existing = deepcopy(self.trait)
        existing["evidence"] = [{"block_id": "b1", "quote": "Her voice had a rough edge."}]
        self.before["voices"][1]["traits"] = [existing]
        merged = self.merged()

        result, archive, issues = consolidate_merged_voices(merged, [self.before], [self.merge])

        traits = next(voice for voice in result["voices"] if voice["id"] == "v1")["traits"]
        self.assertEqual(len(traits), 1)
        self.assertEqual(traits[0]["value"], "rough edge")
        self.assertEqual(traits[0]["evidence"], existing["evidence"] + self.trait["evidence"])
        self.assertEqual(archive[0]["voice"]["traits"], [self.trait])
        self.assertEqual(issues, [])
        check_cast(self.source, result)

    def test_transitive_identity_merges_archive_each_superseded_voice_to_the_final_owner(self):
        source, before = fixture(("Mara", "Red", "Captain"))
        merges = [{"from_character_id": "c1", "into_character_id": "c2",
                   "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]},
                  {"from_character_id": "c2", "into_character_id": "c3",
                   "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}]
        patch = {"characters": [], "roles": [], "voices": [], "issues": [],
                 "summary": "One captain.", "merges": merges}
        merged = merge_patch(source, before, patch, {key: [] for key in ("characters", "roles", "voices")})[0]

        result, archive, issues = consolidate_merged_voices(merged, [before], merges)

        self.assertEqual({voice["id"] for voice in result["voices"]}, {"vn", "v3"})
        self.assertEqual({row["voice"]["id"] for row in archive}, {"v1", "v2"})
        self.assertEqual({row["canonical_voice_id"] for row in archive}, {"v3"})
        self.assertEqual(issues, [])
        check_cast(source, result)
