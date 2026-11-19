"""Collective voices in the analysis format: groups, ensembles, pool voices, line overrides."""

from copy import deepcopy
from pathlib import Path
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.ensembles import lanes_of, proposed_voices, recipe_for
from epub_tts_narrator.renderer.common import read_json

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "reviewed.json"


def group_document(*, ensemble=None, pool=2):
    """The example with Lena's line moved to a chorus group (and, optionally, its ensemble)."""
    document = read_json(EXAMPLE)
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    line = next(row for row in document["segments"] if row["id"] == "b000004:40:64")
    quote = blocks[line["block_id"]]["text"][:12]
    cast = document["cast"]
    cast["characters"].append({
        "id": "chorus", "name": "chorus", "aliases": [], "description": "Voices answering together",
        "evidence": [], "voice_id": "voice_chorus",
        "group": {"size": None, "scale": "several", "members": [], "proposed_voices": 4,
                  "proposed_mode": "unison", "basis": "analysis",
                  "evidence": [{"block_id": line["block_id"], "quote": quote}]}})
    cast["roles"].append({"id": "chorus_dialogue", "kind": "dialogue", "character_id": "chorus",
                          "voice_id": "voice_chorus", "description": "Dialogue role for the chorus."})
    cast["voices"].append({"id": "voice_chorus", "description": "Several low voices", "traits": []})
    for index in range(1, pool + 1):
        cast["voices"].append({"id": f"voice_chorus_{index}", "description": "Another low voice", "traits": [],
                               **({"member_of": "voice_chorus"} if ensemble is not None else {})})
    if ensemble is not None:
        cast["voices"][-pool - 1]["ensemble"] = ensemble
    line.update(role_id="chorus_dialogue", voice_id="voice_chorus")
    return document, line, quote


class EnsembleFormatTests(unittest.TestCase):
    def test_a_group_proposal_with_source_words_is_valid_and_renders_as_today(self):
        document, _, _ = group_document()
        self.assertEqual(validate_document(document), [])

    def test_analysis_never_states_a_group_without_the_sources_words(self):
        document, _, _ = group_document()
        document["cast"]["characters"][-1]["group"]["evidence"] = []
        self.assertTrue(any("group.evidence" in error for error in validate_document(document)))
        document["cast"]["characters"][-1]["group"]["basis"] = "user"
        self.assertEqual(validate_document(document), [])

    def test_a_person_set_ensemble_leads_with_the_groups_own_voice(self):
        members = ["voice_chorus_1", "voice_chorus_2"]
        document, _, _ = group_document(ensemble={"members": members, "mode": "unison", "mix": "loose",
                                                  "basis": "user"})
        self.assertEqual(validate_document(document), [])
        lead = next(voice for voice in document["cast"]["voices"] if voice["id"] == "voice_chorus")
        self.assertEqual(lanes_of(lead), ["voice_chorus", *members])

    def test_only_a_person_sets_up_an_ensemble(self):
        document, _, _ = group_document(ensemble={"members": ["voice_chorus_1"], "mode": "unison",
                                                  "basis": "analysis"})
        self.assertTrue(any("ensemble.basis" in error for error in validate_document(document)))

    def test_ensemble_rules(self):
        cases = {
            "members": ({"members": ["voice_chorus_1", "voice_chorus_1"], "mode": "unison", "basis": "user"},
                        "distinct voices"),
            "empty": ({"members": [], "mode": "unison", "basis": "user"}, "only a declined ensemble"),
            "recipe": ({"members": ["voice_chorus_1"], "mode": "unison", "mix": "stadium", "basis": "user"},
                       "ensemble.mix"),
            "mode": ({"members": ["voice_chorus_1"], "mode": "chorus", "basis": "user"}, "ensemble.mode"),
        }
        for name, (ensemble, message) in cases.items():
            document, _, _ = group_document(ensemble=ensemble)
            with self.subTest(name):
                self.assertTrue(any(message in error for error in validate_document(document)),
                                validate_document(document))
        declined, _, _ = group_document(ensemble={"members": [], "mode": "single", "basis": "user"}, pool=0)
        self.assertEqual(validate_document(declined), [])

    def test_at_most_eight_voices_lead_included(self):
        members = [f"voice_chorus_{index}" for index in range(1, 9)]
        document, _, _ = group_document(ensemble={"members": members, "mode": "unison", "basis": "user"}, pool=8)
        self.assertTrue(any("at most 8 voices" in error for error in validate_document(document)))
        document, _, _ = group_document(ensemble={"members": members[:7], "mode": "unison", "basis": "user"}, pool=7)
        self.assertEqual(validate_document(document), [])

    def test_the_narration_voice_cannot_lead_an_ensemble_and_pool_voices_have_no_character(self):
        document, _, _ = group_document()
        mara = next(voice for voice in document["cast"]["voices"] if voice["id"] == "voice_mara")
        mara["ensemble"] = {"members": ["voice_chorus_1"], "mode": "unison", "basis": "user"}
        self.assertTrue(any("narration voice" in error for error in validate_document(document)))
        document, _, _ = group_document(ensemble={"members": ["voice_chorus_1", "voice_chorus_2"],
                                                  "mode": "unison", "basis": "user"})
        lena = next(voice for voice in document["cast"]["voices"] if voice["id"] == "voice_lena")
        lena["member_of"] = "voice_chorus"
        errors = validate_document(document)
        self.assertTrue(any("lists this voice" in error for error in errors))
        self.assertTrue(any("belongs to no character" in error for error in errors))

    def test_named_characters_can_be_members_of_a_group_and_of_its_ensemble(self):
        document, _, _ = group_document(ensemble={"members": ["voice_lena"], "mode": "alternate",
                                                  "basis": "user"}, pool=0)
        document["cast"]["characters"][-1]["group"]["members"] = ["lena"]
        self.assertEqual(validate_document(document), [])
        document["cast"]["characters"][-1]["group"]["members"] = ["chorus"]
        self.assertTrue(any("its own member" in error for error in validate_document(document)))

    def test_line_overrides_need_the_lines_own_words_unless_a_person_chose(self):
        document, line, quote = group_document()
        line["ensemble"] = {"lanes": 2, "basis": "explicit", "evidence": []}
        self.assertTrue(any("ensemble.evidence" in error for error in validate_document(document)))
        line["ensemble"]["evidence"] = [{"block_id": line["block_id"], "quote": quote}]
        self.assertEqual(validate_document(document), [])
        line["ensemble"] = {"members": ["voice_chorus_1"], "basis": "user"}
        self.assertTrue(any("needs the group's ensemble" in error for error in validate_document(document)))

    def test_line_overrides_only_on_group_lines_and_members_within_the_ensemble(self):
        document, line, _ = group_document(ensemble={"members": ["voice_chorus_1", "voice_chorus_2"],
                                                     "mode": "unison", "basis": "user"})
        line["ensemble"] = {"members": ["voice_chorus", "voice_chorus_2"], "slot": 1, "basis": "user"}
        self.assertEqual(validate_document(document), [])
        line["ensemble"]["members"] = ["voice_lena"]
        self.assertTrue(any("of this group's ensemble" in error for error in validate_document(document)))
        other = next(row for row in document["segments"] if row["id"] == "b000006:0:20")
        other["ensemble"] = {"mode": "unison", "basis": "user"}
        self.assertTrue(any("only to a group's lines" in error for error in validate_document(document)))

    def test_simultaneous_mode_belongs_to_simultaneous_rows(self):
        document = read_json(EXAMPLE)
        line = next(row for row in document["segments"] if row["id"] == "b000006:0:20")
        line["simultaneous_mode"] = "staggered"
        self.assertTrue(any("simultaneous_speakers rows" in error for error in validate_document(document)))

    def test_voice_count_and_recipe_defaults(self):
        self.assertEqual(proposed_voices(7, None), 7)
        self.assertEqual(proposed_voices(24, "crowd"), 8)
        self.assertEqual([proposed_voices(None, scale) for scale in ("pair", "few", "several", "many", "crowd")],
                         [2, 3, 4, 6, 8])
        self.assertIsNone(proposed_voices(None, None))
        self.assertEqual((recipe_for("unison"), recipe_for("staggered"), recipe_for("layered"),
                          recipe_for("unison", "crowd")), ("loose", "babble", "lead", "crowd"))

    def test_the_format_names_exactly_the_renderers_recipes(self):
        from epub_tts_narrator.ensembles import RECIPES
        from epub_tts_narrator.renderer.lane_mix import RECIPES as RENDERER_RECIPES
        self.assertEqual(set(RECIPES), set(RENDERER_RECIPES))


if __name__ == "__main__":
    unittest.main()
