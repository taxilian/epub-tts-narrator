"""Rerunning analyze finds cast entries that are one person and merges them, asking first when a voice would change."""

from copy import deepcopy
import contextlib
import io
import json
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator import cast_review
from epub_tts_narrator.tui_support import screen_host
from epub_tts_narrator.config import DEFAULTS
from shared_fixtures import ScreenHost
from test_attribution_verification_cli import AttributionCLIHarness


class ReviewClient:
    """Answers the whole-cast request with ``pairs`` (cast labels) and every same-person check with ``verdict``."""

    def __init__(self, pairs=(), verdict="y", same=(), apart=()):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.pairs, self.verdict = [list(pair) for pair in pairs], verdict
        self.same, self.apart = [list(pair) for pair in same], [list(pair) for pair in apart]
        self.calls, self.lock = [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] == "cast_review_nominate":
            if "instructions" in payload:
                return {"r": self.pairs, "same": self.same, "apart": self.apart}
            return {"r": self.pairs}
        if payload["mode"] == "narrator_duplicate_check":
            return {"r": [[item["id"], self.verdict] for item in payload["items"]]}
        raise AssertionError(f"Unexpected request mode: {payload['mode']}")

    def modes(self):
        return [call["mode"] for call in self.calls]


class Store:
    """A voice store where ``voiced`` voice IDs have an accepted recording."""

    def __init__(self, voiced=()):
        self.voiced = set(voiced)

    def canonical_id(self, voice_id):
        return voice_id

    def load(self, voice_id):
        return {"inherited": False} if voice_id in self.voiced else None


def renamed(name):
    def prepare(document):
        document["cast"]["characters"][1]["name"] = name
    return prepare


class CastReviewTests(AttributionCLIHarness):
    def review(self, client, *, voiced=(), notes=None, interactive=False, answer="", instructions=None):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        analysis = json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))
        voice_of = {row["name"]: row["voice_id"] for row in analysis["cast"]["characters"]}
        notes = {voice_of[name]: record for name, record in (notes or {}).items()}
        with (patch.object(cast_review, "_renderer_context",
                           return_value={"notes": notes, "store": Store({voice_of[name] for name in voiced})}),
              patch("sys.stdin", io.StringIO(answer)),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO())):
            changed = cast_review.review_cast(self.book, config, None, lambda settings: client,
                                              lambda message: None, interactive=interactive,
                                              instructions=instructions)
        return changed, output.getvalue()

    def cast(self):
        analysis = json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))
        return [row["name"] for row in analysis["cast"]["characters"]]

    def test_overlapping_names_the_text_shows_are_one_person_merge_when_neither_has_a_voice(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        changed, output = self.review(ReviewClient())
        self.assertTrue(changed, output)
        self.assertEqual(len(self.cast()), 1)
        self.assertIn("merged", output)

    def test_an_epithet_before_the_reveal_is_found_by_the_whole_cast_review(self):
        self.make_book(risk=False, prepare=renamed("the cowled shadow"))
        client = ReviewClient(pairs=[["c1", "c2"]])
        changed, output = self.review(client)
        self.assertTrue(changed, output)
        self.assertEqual(len(self.cast()), 1)
        self.assertEqual(client.modes().count("cast_review_nominate"), 1)

    def test_two_people_stay_two_entries(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        changed, output = self.review(ReviewClient(verdict="n"))
        self.assertFalse(changed)
        self.assertEqual(len(self.cast()), 2)

    def test_a_merge_that_would_change_a_voice_is_only_listed_without_a_terminal(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        manifest = (self.book / "manifest.json").read_bytes()
        changed, output = self.review(ReviewClient(), voiced={"Mara Vell"})
        self.assertFalse(changed)
        self.assertIn("a voice would change", output)
        self.assertIn("merge-characters", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_a_person_asked_in_a_terminal_can_merge(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        changed, output = self.review(ReviewClient(), voiced={"Mara"}, interactive=True, answer="m\n")
        self.assertTrue(changed, output)
        self.assertIn("[M] merge them", output)
        self.assertEqual(len(self.cast()), 1)

    def test_kept_apart_is_remembered_and_not_asked_again(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        changed, output = self.review(ReviewClient(), voiced={"Mara"}, interactive=True, answer="k\n")
        self.assertFalse(changed)
        self.assertEqual(len(self.cast()), 2)
        self.assertTrue((self.book / cast_review.REVIEW_FILENAME).is_file())
        client = ReviewClient()
        changed, output = self.review(client, voiced={"Mara"}, interactive=True)
        self.assertFalse(changed)
        self.assertNotIn("[M] merge them", output)
        self.assertNotIn("narrator_duplicate_check", client.modes())

    def test_a_later_name_in_a_persons_notes_is_asked_about_even_when_the_text_does_not_say(self):
        self.make_book(risk=False, prepare=renamed("the cowled shadow"))
        notes = {"the cowled shadow": {"names": ["Mara"]}}
        changed, output = self.review(ReviewClient(verdict="?"), notes=notes, interactive=True, answer="m\n")
        self.assertTrue(changed, output)
        self.assertIn("your notes say the cowled shadow is also called Mara", output)
        self.assertEqual(len(self.cast()), 1)

    def test_analyze_notes_saying_two_entries_are_one_person_are_asked_about(self):
        self.make_book(risk=False)
        client = ReviewClient(verdict="?", same=[["c1", "c2"]])
        changed, output = self.review(client, voiced={"Mara", "Tom"}, interactive=True, answer="m\n",
                                      instructions="Tom and Mara are the same person.")
        self.assertTrue(changed, output)
        self.assertIn("your notes say Mara and Tom are one person", output)
        self.assertEqual(len(self.cast()), 1)
        nominate = next(call for call in client.calls if call["mode"] == "cast_review_nominate")
        self.assertEqual(nominate["instructions"], "Tom and Mara are the same person.")

    def test_analyze_notes_saying_two_entries_are_different_people_keep_them_apart(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        client = ReviewClient(apart=[["c1", "c2"]])
        changed, output = self.review(client, instructions="Mara and Mara Vell are different people.")
        self.assertFalse(changed, output)
        self.assertIn("your notes say Mara and Mara Vell are different people", output)
        self.assertEqual(len(self.cast()), 2)

    def test_without_analyze_notes_the_whole_cast_request_is_unchanged(self):
        self.make_book(risk=False)
        client = ReviewClient()
        self.review(client)
        nominate = next(call for call in client.calls if call["mode"] == "cast_review_nominate")
        self.assertNotIn("instructions", nominate)

    def test_in_the_run_shell_the_question_opens_its_own_screen_with_both_entries(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        host = ScreenHost("k")
        with screen_host(host):
            changed, output = self.review(ReviewClient(), voiced={"Mara"}, interactive=True)
        self.assertFalse(changed)
        self.assertNotIn("[M] merge them", output)
        self.assertTrue((self.book / cast_review.REVIEW_FILENAME).is_file())
        [screen] = host.screens
        self.assertEqual(type(screen).__name__, "MergeScreen")
        view = screen.view
        self.assertEqual(view["title"], "Are these the same character?")
        self.assertEqual(view["why"], "The book's text shows they are one person.")
        kept, absorbed = view["entries"]
        self.assertEqual((kept["name"], kept["voice"], absorbed["name"], absorbed["voice"]),
                         ("Mara", "Accepted voice", "Mara Vell", "No accepted voice yet"))
        self.assertTrue(kept["from"].startswith("New in "))
        self.assertEqual((kept["lines"], absorbed["lines"]), (2, 0))
        self.assertEqual(view["summary"], "Merging keeps Mara, because it already has an accepted voice. Mara Vell "
                                          "has no lines in this book, so nothing is rendered again.")
        # The raw IDs are only in the details the person may open.
        character_ids = [row["id"] for row in json.loads((self.book / "analysis.json").read_text())["cast"]["characters"]]
        shown = json.dumps({key: value for key, value in view.items() if key != "details"})
        self.assertFalse(any(character_id in shown for character_id in character_ids))
        self.assertTrue(all(character_id in view["details"] for character_id in character_ids))

    def test_merging_on_the_run_shells_screen_merges(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        with screen_host(ScreenHost("m")):
            changed, output = self.review(ReviewClient(), voiced={"Mara"}, interactive=True)
        self.assertTrue(changed, output)
        self.assertEqual(self.cast(), ["Mara"])

    def test_an_unsure_pair_is_reported_and_left(self):
        self.make_book(risk=False, prepare=renamed("Mara Vell"))
        changed, output = self.review(ReviewClient(verdict="?"))
        self.assertFalse(changed)
        self.assertIn("the text does not say", output)
        self.assertEqual(len(self.cast()), 2)


if __name__ == "__main__":
    unittest.main()
