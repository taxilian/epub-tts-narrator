"""A voice's accent cue opens every line in that voice, and only that voice's lines change."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle, chunk_speakers
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.performance_runtime import synthesis_direction
from epub_tts_narrator.renderer.speech_traits import (
    ACCENTS_FILENAME, _book_terms, accent_cues, accent_overrides, save_accent, speech_trait_overrides, suggest_accent,
    validate_accent,
)
from epub_tts_narrator.renderer.workflow import _preflight, make_render_plan
import test_book_context


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
ACCENT = "heavy Brooklyn accent"


def turns(plan):
    for chapter in plan["chapters"]:
        for chunk in chapter["chunks"]:
            yield from chunk.get("paragraph_turns", [chunk])


class AccentCueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))
        self.session = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(self.session.close)
        plan = make_render_plan(self.bundle, self.config, multi_speaker=False)
        spoken = {speaker["voice_id"] for turn in turns(plan) for speaker in chunk_speakers(turn)}
        self.voice_id, self.other_id = sorted(spoken)[:2]

    def test_a_cue_opens_every_line_of_its_voice_ahead_of_the_line_direction(self):
        before = make_render_plan(self.bundle, self.config, multi_speaker=False)
        self.session.update_prompt(self.voice_id, accent=ACCENT)
        after = make_render_plan(self.bundle, self.config, multi_speaker=False)
        for old, new in zip(turns(before), turns(after)):
            voices = {speaker["voice_id"] for speaker in chunk_speakers(new)}
            if self.voice_id in voices:
                direction = synthesis_direction(old, self.voice_id)
                self.assertEqual(synthesis_direction(new, self.voice_id),
                                 "; ".join(value for value in (ACCENT, direction) if value))
            else:
                # Other voices' lines are byte-identical, so their takes are reused.
                self.assertEqual(new, old)

    def test_the_render_preflight_freezes_the_cue_and_clearing_it_restores_the_plan(self):
        original = make_render_plan(self.bundle, self.config, multi_speaker=False)
        self.session.update_prompt(self.voice_id, accent=ACCENT)
        config, preflight, _ = _preflight(self.bundle, self.config, chapters_only=True, multi_speaker=False)
        self.assertEqual(config["_accent_snapshot"], {self.voice_id: ACCENT})
        self.assertTrue(any(turn.get("speech_traits") == {self.voice_id: ACCENT} for turn in turns(preflight)))
        self.session.update_prompt(self.voice_id, accent="")
        self.assertEqual(make_render_plan(self.bundle, self.config, multi_speaker=False), original)

    def test_a_cue_is_saved_beside_the_book_without_a_sample_model_or_draft(self):
        with patch.object(self.session, "_draft", side_effect=AssertionError("no sample needed")):
            self.session.update_prompt(self.voice_id, accent=ACCENT)
        record = read_json(self.directory / ACCENTS_FILENAME)
        self.assertEqual((record["format"], record["accents"]), ("audiobook-book-voice-accents", {self.voice_id: ACCENT}))
        row = next(row for row in self.session.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual(row["accent"], ACCENT)
        # Voice traits are a separate choice and stay unset.
        self.assertEqual(speech_trait_overrides(self.bundle, self.config), {})
        self.assertIsNone(self.session._engine)

    def test_invalid_cues_are_refused_before_anything_is_saved(self):
        for value in (" ", "a\nb", "[shouting] accent", "x" * 81, 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.session.update_prompt(self.voice_id, accent=value)
        self.assertFalse((self.directory / ACCENTS_FILENAME).exists())

    def test_rows_suggest_a_cue_from_the_voice_description(self):
        self.session.update_prompt(self.voice_id, prompt="A gruff baritone with an over-the-top old-time New York accent: broad vowels.")
        row = next(row for row in self.session.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual(row["accent_suggestion"], "over-the-top old-time New York accent")
        self.assertEqual(row["accent"], "")


    def test_a_voice_whose_description_names_an_accent_gets_that_cue_unless_a_person_chose_otherwise(self):
        profile = next(voice for voice in self.bundle["cast"]["voices"] if voice["id"] == self.voice_id)
        profile["description"] = "A gruff baritone, close-miked, with a thick Welsh accent."
        plan = make_render_plan(self.bundle, self.config, multi_speaker=False)
        lines = [turn for turn in turns(plan) if self.voice_id in {s["voice_id"] for s in chunk_speakers(turn)}]
        self.assertTrue(lines)
        for turn in lines:
            self.assertEqual(turn["speech_traits"], {self.voice_id: "thick Welsh accent"})
        self.session = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(self.session.close)
        row = next(row for row in self.session.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual((row["accent"], row["accent_automatic"]), ("thick Welsh accent", True))
        # A person's cue replaces it, and an empty choice means no cue at all.
        self.session.update_prompt(self.voice_id, accent=ACCENT)
        self.assertEqual(accent_cues(self.bundle, self.config)[self.voice_id], ACCENT)
        row = next(row for row in self.session.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual((row["accent"], row["accent_automatic"]), (ACCENT, False))
        self.session.update_prompt(self.voice_id, accent="")
        self.assertNotIn(self.voice_id, accent_cues(self.bundle, self.config))
        cleared = make_render_plan(self.bundle, self.config, multi_speaker=False)
        self.assertFalse(any(self.voice_id in turn.get("speech_traits", {}) for turn in turns(cleared)))

    def test_a_saved_voice_uses_the_description_that_designed_its_recording(self):
        class Store:
            def load(self, voice_id):
                return ({"creation": {"requested_description": "Smooth voice with a soft Irish accent"}}
                        if voice_id == self.target else None)
        store = Store()
        store.target = self.voice_id
        profile = next(voice for voice in self.bundle["cast"]["voices"] if voice["id"] == self.voice_id)
        profile["description"] = "A voice with a heavy Brooklyn accent"
        self.assertEqual(accent_cues(self.bundle, self.config, store=store), {self.voice_id: "soft Irish accent"})

    def test_an_accent_named_after_the_books_own_people_gives_no_cue(self):
        # A character's name is the book's own term; a speech model cannot know its "accent".
        profile = next(voice for voice in self.bundle["cast"]["voices"] if voice["id"] == self.voice_id)
        self.assertIn("mara", _book_terms(self.bundle))
        profile["description"] = "A clipped voice with a strong Mara accent"
        self.assertNotIn(self.voice_id, accent_cues(self.bundle, self.config))


class SuggestionTests(unittest.TestCase):
    def test_the_accent_phrase_a_description_names_is_suggested(self):
        for description, expected in (
                ("Pronounced italian accent", "Pronounced italian accent"),
                ("a heavy kinda dimwitted voice with a hugely exaggerated Brooklyn accent", "hugely exaggerated Brooklyn accent"),
                ("Mid-range smooth neutral General American voice", ""),
                ("measured pace in a General American accent", "General American accent"),
                ("close-miked voice with flat Transatlantic accent", "flat Transatlantic accent"),
                ("exaggerated old-Hollywood gumshoe accent", "exaggerated old-Hollywood gumshoe accent"),
                ("", "")):
            with self.subTest(description=description):
                self.assertEqual(suggest_accent(description), expected)
                if expected:
                    validate_accent(expected)

    def test_no_cue_without_a_real_place_or_people(self):
        for description in ("Clear voice with no accent", "speaks without any Welsh accent",
                            "unkempt male voice with heavy accent", "foreign accent", "Heavy accent.",
                            "menacing General American with thick accent"):
            with self.subTest(description=description):
                self.assertEqual(suggest_accent(description), "")
        self.assertEqual(suggest_accent("strong Murgo accent", frozenset({"murgo"})), "")
        self.assertEqual(suggest_accent("strong Murgo accent"), "strong Murgo accent")


class DesignTests(unittest.TestCase):
    def test_a_voice_is_designed_naming_exactly_its_cued_accent(self):
        from epub_tts_narrator.renderer.speech_traits import with_accent
        for description, cue, expected in (
                ("Sturdy baritone with a neutral General American accent and a warm edge.", "Yorkshire accent",
                 "Sturdy baritone with a Yorkshire accent and a warm edge."),
                ("Low husky voice, unhurried", "Received Pronunciation accent",
                 "Low husky voice, unhurried. Speaks with a Received Pronunciation accent."),
                ("Bright voice.", "Irish accent", "Bright voice. Speaks with an Irish accent."),
                ("Rough voice with a thick Welsh accent", "thick Welsh accent", "Rough voice with a thick Welsh accent"),
                ("Bright voice.", "", "Bright voice.")):
            with self.subTest(description=description):
                self.assertEqual(with_accent(description, cue), expected)


class AccentInheritanceTests(unittest.TestCase):
    make_book = test_book_context.BookContextTests.make_book

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first = self.make_book("first")
        self.second = self.make_book("second", self.first)

    def test_a_later_book_inherits_a_cue_and_can_clear_it_without_changing_the_earlier_book(self):
        books = [load_bundle(path) for path in (self.first, self.second)]
        configs = [resolve_book_config(book, deepcopy(DEFAULTS)) for book in books]
        voice_id = books[0]["cast"]["characters"][0]["voice_id"]
        save_accent(books[0], configs[0], voice_id, ACCENT)
        self.assertEqual(accent_overrides(books[1], configs[1]), {voice_id: ACCENT})
        save_accent(books[1], configs[1], voice_id, "")
        self.assertEqual(accent_overrides(books[1], configs[1]), {})
        self.assertEqual(accent_overrides(books[1], configs[1], include_cleared=True), {voice_id: ""})
        self.assertEqual(accent_overrides(books[0], configs[0]), {voice_id: ACCENT})


if __name__ == "__main__":
    unittest.main()
