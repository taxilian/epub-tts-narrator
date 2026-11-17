"""A person's own notes about a character reach its accent and voice description, and are inherited by later books."""

from copy import deepcopy
from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import accent_assignment, accent_readiness, character_notes as notes_module
from epub_tts_narrator.renderer.accent_assignment import assigned_accents, request_assignment, voice_entries
from epub_tts_narrator.renderer.accent_prompts import ASSIGN, KNOWN
from epub_tts_narrator.renderer.accent_review import AccentPlanner
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cast_tui_model import CastController, detail_sections
from epub_tts_narrator.renderer.character_notes import (
    character_notes, description_conflict, from_text, own_character_notes, save_character_notes, stale_assignments,
    to_text, validate_notes,
)
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.description_revision import revise
from epub_tts_narrator.renderer.description_revision_prompts import FOLLOW_NOTES, KNOWN as DESCRIPTION_KNOWN, REVISE
from epub_tts_narrator.voice_prompt_refresh import refresh_voice_descriptions
from epub_tts_narrator.voice_refresh_prompts import LISTENER_NOTES, SYSTEM
from cast_tui_fixtures import FakePlayer, FakeSession
from test_renderer_accent_assignment import BookTestCase, FakeRequests, everyone
import test_book_context


ZANDRAMAS_TEXT = """names: Zandramas
gender: female
origin: Mallorea
notes: Revealed later as Zandramas.
A more exotic accent than other Malloreans."""
ZANDRAMAS = {"names": ["Zandramas"], "gender": "female", "origin": "Mallorea",
             "notes": "Revealed later as Zandramas.\nA more exotic accent than other Malloreans."}


class NotesTextTests(unittest.TestCase):
    def test_the_edited_lines_become_a_record_and_back(self):
        self.assertEqual(from_text(ZANDRAMAS_TEXT), ZANDRAMAS)
        self.assertEqual(from_text(to_text(ZANDRAMAS)), ZANDRAMAS)
        # Unlabelled lines are free notes; labels may repeat and names may be comma-separated.
        self.assertEqual(from_text("She is the villain: later revealed.\nalso: Zandramas, the Child of Dark\n"),
                         {"names": ["Zandramas", "the Child of Dark"], "notes": "She is the villain: later revealed."})
        self.assertEqual(from_text(to_text({})), {})

    def test_notes_are_plain_bounded_words(self):
        validate_notes(ZANDRAMAS)
        for bad in ({"names": ["<Zandramas>"]}, {"names": ["A", "a"]}, {"gender": ""}, {"age": "40"},
                    {"names": [f"n{index}" for index in range(9)]}, {"origin": "line\nbreak"},
                    {"notes": "x" * 1501}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_notes(bad)

    def test_a_description_contradicting_the_notes_gender_is_flagged(self):
        self.assertIn("female", description_conflict("Male, hollow cold voice, slow and menacing.", ZANDRAMAS))
        self.assertEqual(description_conflict("A cold, hollow female voice.", ZANDRAMAS), "")
        self.assertEqual(description_conflict("A cold, hollow voice.", ZANDRAMAS), "")
        self.assertEqual(description_conflict("Male, hollow voice.", {"notes": "a woman"}), "")


class InheritanceTests(unittest.TestCase):
    make_book = test_book_context.BookContextTests.make_book

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.books = [load_bundle(path) for path in (self.make_book("first"),)]
        self.books.append(load_bundle(self.make_book("second", Path(self.books[0]["directory"]))))
        self.configs = [resolve_book_config(book, deepcopy(DEFAULTS)) for book in self.books]
        self.voice_id = self.books[0]["cast"]["characters"][0]["voice_id"]

    def test_a_later_book_inherits_notes_and_can_clear_them_without_changing_the_earlier_book(self):
        first, second = self.books
        save_character_notes(first, self.configs[0], self.voice_id, ZANDRAMAS)
        self.assertEqual(character_notes(second, self.configs[1]), {self.voice_id: ZANDRAMAS})
        self.assertEqual(own_character_notes(second, self.configs[1]), {})
        save_character_notes(second, self.configs[1], self.voice_id, {})
        self.assertEqual(character_notes(second, self.configs[1]), {})
        self.assertEqual(character_notes(first, self.configs[0]), {self.voice_id: ZANDRAMAS})
        save_character_notes(second, self.configs[1], self.voice_id, None)  # inherit again
        self.assertEqual(character_notes(second, self.configs[1]), {self.voice_id: ZANDRAMAS})

    def test_notes_added_in_a_later_book_make_its_inherited_accent_undecided_there_only(self):
        first, second = self.books
        accent_assignment.save_assignments(first, self.configs[0], {self.voice_id: "Persian accent"})
        self.assertEqual(assigned_accents(second, self.configs[1])[self.voice_id], "Persian accent")
        save_character_notes(second, self.configs[1], self.voice_id, ZANDRAMAS)
        self.assertEqual(stale_assignments(second, self.configs[1]), {self.voice_id})
        self.assertNotIn(self.voice_id, assigned_accents(second, self.configs[1]))
        self.assertEqual(assigned_accents(first, self.configs[0])[self.voice_id], "Persian accent")
        # Reassigned in the later book with these notes: decided again.
        accent_assignment.save_assignments(second, self.configs[1], {self.voice_id: "Turkish accent"})
        self.assertEqual(assigned_accents(second, self.configs[1])[self.voice_id], "Turkish accent")
        self.assertEqual(stale_assignments(second, self.configs[1]), set())


class AccentNotesTests(BookTestCase):
    def test_without_notes_the_request_is_unchanged(self):
        requests = FakeRequests(everyone("Welsh"))
        self.prepare(requests)
        system, payload = requests.calls[0]
        self.assertEqual(system, ASSIGN)
        self.assertNotIn("known", payload)

    def test_notes_go_with_the_character_and_their_later_names_find_their_passages(self):
        voice_id = self.voices[0]
        self.bundle["source"]["blocks"].append({"id": "b-later", "chapter_id": "x",
                                                "text": "Beware Zandramas, the sorceress of Darshiva in Mallorea."})
        save_character_notes(self.bundle, self.config, voice_id, ZANDRAMAS)
        entries = voice_entries(self.bundle, (), character_notes(self.bundle, self.config))
        self.assertTrue(any("sorceress of Darshiva" in text for text in
                            entries[voice_id]["intro"] + entries[voice_id]["more"]["passages"]))
        requests = FakeRequests(lambda payload: {"g": [["Mallorea", "Persian", [payload["known"][0][0]]],
                                                        ["the town", "Welsh", [row[0] for row in payload["voices"]
                                                                               if row[0] != payload["known"][0][0]]]]})
        self.prepare(requests)
        system, payload = requests.calls[0]
        self.assertEqual(system, ASSIGN + KNOWN)
        row = next(row for row in payload["voices"] if row[1] == self.entries[voice_id]["names"])
        self.assertEqual(payload["known"], [[row[0], ["Zandramas"], "female", "Mallorea",
                                             "Revealed later as Zandramas. A more exotic accent than other Malloreans."]])
        self.assertEqual(assigned_accents(self.bundle, self.config)[voice_id], "Persian accent")

    def test_changed_notes_make_that_accent_undecided_and_the_next_assignment_asks_only_about_it(self):
        voice_id, other = self.voices[:2]
        self.prepare(FakeRequests(everyone("Welsh")))
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config), [])
        save_character_notes(self.bundle, self.config, voice_id, ZANDRAMAS)
        self.assertNotIn(voice_id, assigned_accents(self.bundle, self.config))
        self.assertEqual(assigned_accents(self.bundle, self.config)[other], "Welsh accent")
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config), [voice_id])
        again = FakeRequests(everyone("Persian", "Mallorea"))
        self.prepare(again)
        payload = again.calls[0][1]
        self.assertEqual([row[1] for row in payload["voices"]], [self.entries[voice_id]["names"]])
        self.assertEqual(len(payload["known"]), 1)
        self.assertEqual(assigned_accents(self.bundle, self.config)[voice_id], "Persian accent")
        self.assertEqual(accent_readiness.undecided(self.bundle, self.config), [])
        # The same notes saved again change nothing; edited notes undecide it again.
        save_character_notes(self.bundle, self.config, voice_id, deepcopy(ZANDRAMAS))
        self.assertEqual(stale_assignments(self.bundle, self.config), set())
        save_character_notes(self.bundle, self.config, voice_id, dict(ZANDRAMAS, origin="Darshiva, in Mallorea"))
        self.assertEqual(stale_assignments(self.bundle, self.config), {voice_id})

    def test_the_accent_review_shows_changed_notes_and_saving_it_records_them(self):
        voice_id = self.voices[0]
        self.prepare(FakeRequests(everyone("Welsh")))
        save_character_notes(self.bundle, self.config, voice_id, ZANDRAMAS)
        requests = FakeRequests(everyone("Persian", "Mallorea"))
        planner = AccentPlanner(self.bundle, self.config, None, requests=requests, progress=lambda _: None)
        self.assertEqual(planner.notes_changed, [voice_id])
        self.assertTrue(planner.load_saved())
        self.assertEqual(planner.accents()[voice_id], "Welsh accent")  # still in its group, for review
        self.assertIn("your notes changed", planner.about(voice_id))
        self.assertIn("Zandramas", planner.about(voice_id))
        planner.revise("Place the characters I wrote notes about")
        self.assertEqual(requests.calls[0][1]["known"][0][1:3], [["Zandramas"], "female"])
        planner.save()
        self.assertEqual(assigned_accents(self.bundle, self.config)[voice_id], "Persian accent")
        self.assertEqual(stale_assignments(self.bundle, self.config), set())

    def test_the_accents_command_says_whose_notes_changed(self):
        voice_id = self.voices[0]
        self.prepare(FakeRequests(everyone("Welsh")))
        save_character_notes(self.bundle, self.config, voice_id, ZANDRAMAS)
        with patch("epub_tts_narrator.tui_support.wants_tui", return_value=False), \
                patch("sys.stdin", io.StringIO("")):
            code, output = self.run_command(FakeRequests(everyone("Persian")))
        self.assertEqual(code, 0)
        self.assertIn(f"Your notes about {self.entries[voice_id]['label']}", output)

    def test_a_person_chosen_cue_still_wins_over_notes(self):
        from epub_tts_narrator.renderer.speech_traits import accent_cues, save_accent
        voice_id = self.voices[0]
        save_accent(self.bundle, self.config, voice_id, "heavy Brooklyn accent")
        save_character_notes(self.bundle, self.config, voice_id, ZANDRAMAS)
        requests = FakeRequests(everyone("Welsh"))
        self.prepare(requests)
        self.assertNotIn("known", requests.calls[0][1])
        self.assertEqual(accent_cues(self.bundle, self.config)[voice_id], "heavy Brooklyn accent")


class DescriptionNotesTests(unittest.TestCase):
    def test_a_revision_sends_the_notes_only_when_there_are_some(self):
        replies = []

        class Requests:
            def run(self, system, payload, validate):
                replies.append((system, deepcopy(payload)))
                return validate({"d": [["v1", "A cold, hollow female voice, slow and menacing."]]})

        voices = [{"voice_id": "v", "name": "cowled shadow", "accent": "Persian accent",
                   "description": "Male, hollow cold voice, slow and menacing."}]
        revise(Requests(), voices, "lower")
        self.assertEqual(replies[-1][0], REVISE)
        self.assertEqual(len(replies[-1][1]["voices"][0]), 4)  # id, name, description, accent: as before
        proposal = revise(Requests(), [dict(voices[0], known=ZANDRAMAS)], FOLLOW_NOTES)
        system, payload = replies[-1]
        self.assertEqual(system, REVISE + DESCRIPTION_KNOWN)
        self.assertEqual(payload["voices"][0][-1][:3], [["Zandramas"], "female", "Mallorea"])
        self.assertEqual(proposal, {"v": "A cold, hollow female voice, slow and menacing."})

    def test_a_refresh_sends_the_notes_for_that_voice_only(self):
        from test_audition_scripts import bundle
        from test_voice_prompt_refresh import Requests
        book = bundle()
        first, second = [voice["id"] for voice in book["cast"]["voices"]][:2]
        requests = Requests()
        refresh_voice_descriptions(book, requests, voice_ids=[first, second], listener_notes={first: ZANDRAMAS})
        calls = {payload["voice"]["id"]: (system, payload) for system, payload in requests.calls}
        self.assertEqual(calls[first][0], SYSTEM + LISTENER_NOTES)
        self.assertEqual(calls[first][1]["listener_notes"], ZANDRAMAS)
        self.assertEqual(calls[second][0], SYSTEM)
        self.assertNotIn("listener_notes", calls[second][1])


class CastScreenNotesTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.controller = CastController(self.session, FakePlayer(), describe_error=str)
        self.addCleanup(self.controller.close)
        self.controller.refresh()

    def test_notes_are_saved_for_the_voice_and_shown_and_found(self):
        row = self.controller.row("v-ben")
        self.controller.save_notes(row, ZANDRAMAS_TEXT)
        self.controller.jobs.wait_idle(5)
        self.assertEqual(self.session.calls[-1], ("save_notes", ("v-ben", ZANDRAMAS), {}))
        self.controller.refresh()
        row = self.controller.row("v-ben")
        sections = {title: value for title, value, _ in detail_sections(row, self.controller.rows, "missing", "")}
        self.assertIn("also Zandramas; female; Mallorea", sections["Your notes"])
        self.assertNotIn("Check", sections)
        self.controller.filter.text = "zandramas"
        self.assertEqual([item[0]["voice_id"] for item in self.controller.visible()], ["v-ben"])
        row["prompt"] = "Male, hollow cold voice."
        sections = {title: value for title, value, _ in detail_sections(row, self.controller.rows, "missing", "")}
        self.assertIn("Your notes say female", sections["Check"])

    def test_a_changed_voice_takes_no_notes_of_its_own(self):
        with self.assertRaisesRegex(ValueError, "main voice"):
            self.controller.parse_notes(self.controller.row("v-ava-old"), ZANDRAMAS_TEXT)

    def test_following_the_notes_asks_for_a_description_that_agrees_with_them(self):
        asked = []

        class Requests:
            def run(self, system, payload, validate):
                asked.append((system, deepcopy(payload)))
                return validate({"d": [["v1", "A cold, hollow female voice."]]})

        self.controller.requests_factory = Requests
        ready = []
        self.controller.follow_notes(self.controller.row("v-ben"), ZANDRAMAS, ready.append)
        self.controller.jobs.wait_idle(5)
        system, payload = asked[0]
        self.assertTrue(system.endswith(DESCRIPTION_KNOWN))
        self.assertEqual(payload["instructions"], FOLLOW_NOTES)
        self.assertEqual(payload["voices"][0][-1][0], ["Zandramas"])
        self.assertEqual(ready, [{"v-ben": ("A calm voice for Ben.", "A cold, hollow female voice.")}])
        self.assertFalse(any(call[0] == "update_prompt" for call in self.session.calls))  # nothing saved


class CastingSessionNotesTests(unittest.TestCase):
    def test_saved_notes_reach_the_cast_rows(self):
        from epub_tts_narrator.renderer.casting import CastingSession
        from test_renderer_accent_assignment import EXAMPLE
        import shutil
        from audition_fixtures import prepare_bundle
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name) / "book"
        shutil.copytree(EXAMPLE, directory)
        bundle = prepare_bundle(load_bundle(directory))
        config = resolve_book_config(bundle, deepcopy(DEFAULTS))
        session = CastingSession(bundle, config, notify=lambda _: None)
        self.addCleanup(session.close)
        voice_id = bundle["cast"]["characters"][0]["voice_id"]
        session.save_notes(voice_id, ZANDRAMAS)
        row = next(row for row in session.rows() if row["voice_id"] == voice_id)
        self.assertEqual((row["notes"], row["notes_source"]), (ZANDRAMAS, "this book"))
        self.assertTrue((directory / notes_module.NOTES_FILENAME).is_file())


if __name__ == "__main__":
    unittest.main()
