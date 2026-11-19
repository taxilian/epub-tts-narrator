"""A person-cast collective voice: which voices speak each group line, and how it renders."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from audition_fixtures import prepare_bundle
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import segment
from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle, make_plan
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.ensemble_plan import lane_recipe, resolve_chapter, validate_resolution
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import render
from test_renderer_simultaneous import LaneEngine, LaneVerifier
from test_renderer_workflow import EXAMPLE


def cast_voices(members, mode="unison", mix=None):
    lead = {"id": "voice_group", "ensemble": {"members": list(members), "mode": mode, "basis": "user",
                                              **({"mix": mix} if mix else {})}}
    return {"voice_group": lead, **{member: {"id": member, "member_of": "voice_group"} for member in members}}


def line(identifier, *, voice="voice_group", role="group_dialogue", **ensemble):
    row = {"id": identifier, "kind": "dialogue", "role_id": role, "voice_id": voice}
    if ensemble:
        row["ensemble"] = dict(ensemble, basis="explicit", evidence=[])
    return row


def voices_of(resolution):
    return [performer["voice_id"] for performer in resolution["performers"]]


class EnsembleResolutionTests(unittest.TestCase):
    def test_unison_lines_take_every_voice_lead_first_and_a_lanes_hint_keeps_the_lead(self):
        voices = cast_voices(["v1", "v2", "v3"])
        resolved = resolve_chapter([line("a"), line("b", lanes=2)], voices, "chapter-0001")
        self.assertEqual(voices_of(resolved["a"]), ["voice_group", "v1", "v2", "v3"])
        self.assertEqual(resolved["a"]["recipe"], "loose")
        self.assertEqual(len(voices_of(resolved["b"])), 2)
        self.assertEqual(voices_of(resolved["b"])[0], "voice_group")

    def test_the_cast_recipe_applies_to_its_own_mode_and_other_modes_use_their_default(self):
        voices = cast_voices(["v1", "v2"], mix="crowd")
        resolved = resolve_chapter([line("a"), line("b", mode="staggered"), line("c", mode="layered")],
                                   voices, "chapter-0001")
        self.assertEqual([resolved[key]["recipe"] for key in "abc"], ["crowd", "babble", "lead"])

    def test_pool_slots_keep_one_voice_per_anonymous_individual(self):
        voices = cast_voices(["v1", "v2", "v3"], mode="pool")
        rows = [line("a", slot=1), line("b", slot=2), line("c", slot=1), line("d", slot=3), line("e", slot=2)]
        resolved = resolve_chapter(rows, voices, "chapter-0001")
        picks = {key: voices_of(resolved[key]) for key in "abcde"}
        self.assertTrue(all(len(pick) == 1 for pick in picks.values()))
        self.assertEqual(picks["a"], picks["c"])
        self.assertEqual(picks["b"], picks["e"])
        self.assertEqual(len({picks["a"][0], picks["b"][0], picks["d"][0]}), 3, "three individuals, three voices")
        self.assertIsNone(resolved["a"]["recipe"])

    def test_pool_lines_without_a_slot_never_repeat_the_previous_voice(self):
        voices = cast_voices(["v1"], mode="pool")
        resolved = resolve_chapter([line(str(index)) for index in range(6)], voices, "chapter-0001")
        picks = [voices_of(resolved[str(index)])[0] for index in range(6)]
        self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])), picks)

    def test_adding_a_pool_member_moves_only_slots_that_prefer_it(self):
        rows = [line(str(slot), slot=slot) for slot in range(1, 4)]
        before = resolve_chapter(rows, cast_voices([f"v{n}" for n in range(1, 8)], mode="pool"), "chapter-0001")
        after = resolve_chapter(rows, cast_voices([f"v{n}" for n in range(1, 9)], mode="pool"), "chapter-0001")
        moved = [slot for slot in "123" if voices_of(before[slot]) != voices_of(after[slot])]
        self.assertTrue(all(voices_of(after[slot]) == ["v8"] for slot in moved), moved)

    def test_alternate_takes_turns_through_a_run_and_restarts_after_another_speaker(self):
        voices = cast_voices(["v1"], mode="alternate")
        rows = [line("a"), {"id": "n", "kind": "narration", "role_id": "narrator_main", "voice_id": "narrator"},
                line("b"), line("c"), line("x", voice="voice_other", role="other"), line("d")]
        resolved = resolve_chapter(rows, voices, "chapter-0001")
        self.assertEqual([voices_of(resolved[key])[0] for key in "abcd"],
                         ["voice_group", "v1", "voice_group", "voice_group"])

    def test_single_is_the_lead_alone_and_unensembled_lines_are_left_as_they_are(self):
        voices = cast_voices(["v1"], mode="single")
        resolved = resolve_chapter([line("a"), line("b", voice="voice_other", role="other")], voices, "c")
        self.assertEqual(voices_of(resolved["a"]), ["voice_group"])
        self.assertNotIn("b", resolved)

    def test_named_characters_in_unison_mix_tight_unless_their_line_says_otherwise(self):
        row = {"id": "s", "kind": "dialogue", "role_id": None, "voice_id": None,
               "simultaneous_speakers": [{"role_id": "a", "voice_id": "va"}, {"role_id": "b", "voice_id": "vb"}]}
        self.assertEqual(lane_recipe({"segment": row}), "tight")
        self.assertEqual(lane_recipe({"segment": dict(row, simultaneous_mode="staggered")}), "babble")
        self.assertEqual(lane_recipe({"segment": dict(row, simultaneous_mode="layered")}), "lead")
        self.assertEqual(resolve_chapter([row], {}, "c"), {})

    def test_a_chapter_service_checks_a_resolution_without_the_cast(self):
        voices = cast_voices(["v1", "v2"])
        row = line("a")
        chunk = {"segment": row, "ensemble": resolve_chapter([row], voices, "c")["a"]}
        self.assertEqual(chunk_speakers(chunk), chunk["ensemble"]["performers"])
        self.assertEqual(lane_recipe(chunk), "loose")
        for broken in ({"recipe": None}, {"voice_id": "voice_other"}, {"mode": "chorus"},
                       {"performers": [{"role_id": "other", "voice_id": "v1"}] * 2}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_resolution(dict(chunk, ensemble=dict(chunk["ensemble"], **broken)))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class EnsembleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def bundle(self, ensemble):
        document = read_json(EXAMPLE / "reviewed.json")
        group_text = "“We are ready. Let us begin,”"
        tag = " the chorus said in unison."
        block_id = "b000007"
        document["source"]["chapters"].append({"id": "ch0002", "title": "Chorus Pilot",
            "source_href": "chorus.xhtml", "block_ids": [block_id]})
        document["source"]["blocks"].append({"id": block_id, "chapter_id": "ch0002",
            "text": group_text + tag, "source": {"href": "chorus.xhtml", "element_id": None, "tag": "p"}})
        cast = document["cast"]
        cast["characters"].append({"id": "chorus", "name": "chorus", "aliases": [], "evidence": [],
            "description": "Voices answering together", "voice_id": "voice_chorus",
            "group": {"size": None, "scale": "few", "members": [], "proposed_voices": 3, "proposed_mode": "unison",
                      "basis": "analysis", "evidence": [{"block_id": block_id, "quote": "the chorus said in unison"}]}})
        role = {"id": "chorus_dialogue", "kind": "dialogue", "character_id": "chorus", "voice_id": "voice_chorus",
                "description": "Dialogue role for the chorus."}
        cast["roles"].append(role)
        cast["voices"].append({"id": "voice_chorus", "description": "Low chanting voices", "traits": [],
                               "ensemble": ensemble})
        for member in ensemble["members"]:
            cast["voices"].append({"id": member, "description": "Another low voice", "traits": [],
                                   "member_of": "voice_chorus"})
        narrator = next(item for item in cast["roles"] if item["id"] == "narrator_main")
        document["segments"].extend([
            segment(document["source"], block_id, 0, group_text, role, kind="dialogue", basis="explicit",
                    reason="The chorus speaks.", evidence=[{"block_id": block_id, "quote": "the chorus said"}],
                    candidates=[], review_required=False),
            segment(document["source"], block_id, len(group_text), tag, narrator, reason="Narrative speech tag.")])
        export_bundle(document, self.root / "bundle")
        self.group_spoken = "We are ready. Let us begin,"
        return prepare_bundle(load_bundle(self.root / "bundle"))

    def config(self):
        config = deepcopy(DEFAULTS)
        config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        config["verification"]["model"] = str(self.root / "unused-asr")
        config["voices"].update(directory=str(self.root / "voices"), series_id="chorus-fixture")
        config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                max_attempts=2, chapter_numbers=[2], require_cover=False)
        config["quality"] = validate_policy({})
        return config

    def test_the_plan_names_every_member_voice_so_casting_covers_them(self):
        bundle = self.bundle({"members": ["voice_chorus_1", "voice_chorus_2"], "mode": "unison", "basis": "user"})
        plan = make_plan(bundle, self.config())
        self.assertEqual(plan["errors"], [])
        self.assertTrue({"voice_chorus", "voice_chorus_1", "voice_chorus_2"} <= set(plan["voice_ids"]))
        group = plan["chapters"][0]["chunks"][0]
        self.assertEqual([speaker["voice_id"] for speaker in chunk_speakers(group)],
                         ["voice_chorus", "voice_chorus_1", "voice_chorus_2"])

    def test_a_unison_group_line_is_three_verified_takes_mixed_by_its_recipe(self):
        bundle = self.bundle({"members": ["voice_chorus_1", "voice_chorus_2"], "mode": "unison",
                              "mix": "crowd", "basis": "user"})
        engine = LaneEngine()
        verifier = LaneVerifier(engine)
        render(bundle, self.config(), engine, verifier, notify=lambda _: None)
        spoken = [call for call in engine.speech_calls if call["text"] == self.group_spoken]
        self.assertEqual(len({call["voice_hash"] for call in spoken}), 3)
        acceptance = read_json(self.root / "output" / "verification.json")[0]
        self.assertEqual(acceptance["report"]["method"], "independently_verified_simultaneous_lanes")
        self.assertEqual([lane["voice_id"] for lane in acceptance["speaker_lanes"]],
                         ["voice_chorus", "voice_chorus_1", "voice_chorus_2"])
        self.assertTrue(all(lane["report"]["passed"] for lane in acceptance["speaker_lanes"]))
        self.assertEqual(acceptance["mix"]["recipe"], "crowd")
        group = read_json(self.root / "output" / "sync.json")["segments"][0]
        self.assertEqual(group["ensemble"]["mode"], "unison")
        self.assertFalse(any("mixes" in path for path in verifier.calls), "the mix is never recognized")

    def test_a_pool_line_is_one_members_ordinary_take(self):
        bundle = self.bundle({"members": ["voice_chorus_1", "voice_chorus_2"], "mode": "pool", "basis": "user"})
        engine = LaneEngine()
        render(bundle, self.config(), engine, LaneVerifier(engine), notify=lambda _: None)
        acceptance = read_json(self.root / "output" / "verification.json")[0]
        self.assertNotIn("speaker_lanes", acceptance)
        spoken = [call for call in engine.speech_calls if call["text"] == self.group_spoken]
        self.assertEqual(len({call["voice_hash"] for call in spoken}), 1)


if __name__ == "__main__":
    unittest.main()
