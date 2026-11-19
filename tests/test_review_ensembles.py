"""A person changes how one group line's voices speak, in review-ui; saved as an attribution edit."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.analysis_revision import related_directories, revise_directories
from epub_tts_narrator.attribution_edits import edit_ensemble, edit_simultaneous_mode, edit_speaker
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle, make_plan
from epub_tts_narrator.renderer.ensemble_command import apply, plan_ensemble
from epub_tts_narrator.review_tui import ReviewApp
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_character_merge import AUDITION, build

PARAGRAPHS = [
    [("Mara waved. ", None), ("“Hello,” ", "mara"), ("she said.", None)],
    [("“We are here,” ", "chorus"), ("they said in unison.", None)],
    [("“Together!” ", ("mara", "lena")), ("they cried.", None)],
]


class GroupLineEditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"
        document = build("chorus-book", "The Chorus", [("mara", "Mara", ()), ("lena", "Lena", ()),
                                                       ("chorus", "the chorus", ())], PARAGRAPHS)
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)
        self.chorus = next(item["voice_id"] for item in linked["cast"]["characters"] if item["name"] == "the chorus")
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="unison"), progress=lambda _: None)
        self.document = _completed_directory(self.path)[0]
        self.voices = {voice["id"]: voice for voice in self.document["cast"]["voices"]}
        self.pool = self.voices[self.chorus]["ensemble"]["members"]
        self.line = next(row for row in self.document["segments"] if row["id"] == "b2:0")

    def test_a_person_pins_a_voice_or_chooses_voices_and_clearing_restores_the_default(self):
        pinned = edit_ensemble(self.line, self.voices, mode="pool", members=[self.pool[1]])
        self.assertEqual(pinned["ensemble"], {"mode": "pool", "members": [self.pool[1]], "basis": "user"})
        self.assertNotIn("ensemble", edit_ensemble(pinned, self.voices))
        with self.assertRaisesRegex(ValueError, "this group's ensemble"):
            edit_ensemble(self.line, self.voices, members=["not-a-member"])
        mara = next(row for row in self.document["segments"] if row["id"] == "b1:1")
        with self.assertRaisesRegex(ValueError, "cast screen"):
            edit_ensemble(mara, self.voices, mode="unison")

    def test_a_new_speaker_drops_the_old_groups_override(self):
        pinned = edit_ensemble(self.line, self.voices, mode="pool", members=[self.pool[0]])
        roles = {role["id"]: role for role in self.document["cast"]["roles"]}
        mara = next(row["role_id"] for row in self.document["segments"] if row["id"] == "b1:1")
        moved = edit_speaker(pinned, [mara], roles, {role_id: role_id for role_id in roles})
        self.assertNotIn("ensemble", moved)

    def test_named_characters_together_can_enter_one_after_another(self):
        together = next(row for row in self.document["segments"] if row["id"] == "b3:0")
        staggered = edit_simultaneous_mode(together, "staggered")
        self.assertEqual(staggered["simultaneous_mode"], "staggered")
        self.assertNotIn("simultaneous_mode", edit_simultaneous_mode(staggered, "unison"))
        with self.assertRaisesRegex(ValueError, "several named characters"):
            edit_simultaneous_mode(self.line, "staggered")

    def test_review_ui_saves_a_pinned_voice_and_the_plan_uses_it(self):
        app = ReviewApp(self.path, self.document, revise_directories=revise_directories,
                        related_directories=related_directories)
        app.info_segment_id = "b2:0"
        self.assertEqual(app.voices_kind(), "group")
        self.assertEqual(app.voices_text(), "Several voices: unison (the group's default)")
        self.assertEqual([label for _, label in app.group_choices()["voices"]],
                         ["the chorus", "the chorus, voice 2", "the chorus, voice 3"])
        self.assertTrue(app.set_line_voices(mode="pool", members=[self.pool[0]]))
        app.info_segment_id = "b3:0"
        self.assertEqual(app.voices_kind(), "simultaneous")
        self.assertTrue(app.set_simultaneous_mode("layered"))
        app.save()
        saved = _completed_directory(self.path)[0]
        self.assertEqual(validate_document(saved), [])
        plan = make_plan(load_bundle(self.path), renderer_config())
        chunks = {chunk["segment"]["id"]: chunk for chapter in plan["chapters"] for chunk in chapter["chunks"]}
        self.assertEqual([speaker["voice_id"] for speaker in chunk_speakers(chunks["b2:0"])], [self.pool[0]])
        self.assertEqual(chunks["b3:0"]["segment"]["simultaneous_mode"], "layered")


if __name__ == "__main__":
    unittest.main()
