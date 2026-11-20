"""Rerunning analyze fixes a voice description whose gender the book or a person's notes contradicts."""

from copy import deepcopy
import contextlib
import io
import json
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator import cast_review, gender_review
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.tui_support import screen_host
from shared_fixtures import ScreenHost
from test_attribution_verification_cli import AttributionCLIHarness
from test_cast_review import Store


class GenderClient:
    """Answers each item for a character named in ``shown`` with that gender and an exact quote."""

    def __init__(self, shown):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.shown, self.calls, self.lock = shown, [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        assert payload["mode"] == "gender_review", payload["mode"]
        answers = []
        for item in payload["items"]:
            name = item["who"].split("|")[0]
            if name in self.shown:
                number, text = item["text"].splitlines()[0].split("|", 1)
                answers.append([item["id"], self.shown[name], int(number), text[:12]])
            else:
                answers.append([item["id"], "?"])
        return {"r": answers}


def male_mara(document):
    voice = next(voice for voice in document["cast"]["voices"] if voice["id"] == "v1")
    voice["description"] = "Male, warm and low, sounds about forty."


class GenderReviewTests(AttributionCLIHarness):
    def review(self, client, *, voiced=(), notes=None, interactive=False, answer=""):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        analysis = self.analysis()
        voice_of = {row["name"]: row["voice_id"] for row in analysis["cast"]["characters"]}
        notes = {voice_of[name]: record for name, record in (notes or {}).items()}
        output = io.StringIO()
        with (patch.object(cast_review, "_renderer_context",
                           return_value={"notes": notes, "store": Store({voice_of[name] for name in voiced})}),
              patch("sys.stdin", io.StringIO(answer)),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO())):
            changed = gender_review.review_genders(self.book, config, None, lambda settings: client,
                                                   lambda message: None, interactive=interactive)
        return changed, output.getvalue()

    def analysis(self):
        return json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))

    def description(self, name):
        analysis = self.analysis()
        voice_id = next(row["voice_id"] for row in analysis["cast"]["characters"] if row["name"] == name)
        return next(voice["description"] for voice in analysis["cast"]["voices"] if voice["id"] == voice_id)

    def test_a_wrong_gender_is_fixed_when_the_character_has_no_voice_yet(self):
        self.make_book(risk=False, prepare=male_mara)
        changed, output = self.review(GenderClient({"Mara": "Female"}))
        self.assertTrue(changed, output)
        self.assertEqual(self.description("Mara"), "Female, warm and low, sounds about forty.")
        self.assertIn("the book says", output)

    def test_an_existing_voice_is_only_reported_without_a_terminal(self):
        self.make_book(risk=False, prepare=male_mara)
        manifest = (self.book / "manifest.json").read_bytes()
        changed, output = self.review(GenderClient({"Mara": "Female"}), voiced={"Mara"})
        self.assertFalse(changed)
        self.assertIn("recast it", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_a_person_may_fix_the_description_of_an_existing_voice(self):
        self.make_book(risk=False, prepare=male_mara)
        changed, output = self.review(GenderClient({"Mara": "Female"}), voiced={"Mara"}, interactive=True,
                                      answer="f\n")
        self.assertTrue(changed, output)
        self.assertTrue(self.description("Mara").startswith("Female"))

    def test_in_the_run_shell_the_question_shows_the_description_before_and_after(self):
        self.make_book(risk=False, prepare=male_mara)
        host = ScreenHost("f")
        with screen_host(host):
            changed, output = self.review(GenderClient({"Mara": "Female"}), voiced={"Mara"}, interactive=True)
        self.assertTrue(changed, output)
        self.assertNotIn("[F] fix the description", output)
        self.assertEqual(self.description("Mara"), "Female, warm and low, sounds about forty.")
        [screen] = host.screens
        self.assertEqual(type(screen).__name__, "GenderScreen")
        view = screen.view
        self.assertEqual((view["name"], view["current"], view["wanted"]), ("Mara", "Male", "Female"))
        self.assertEqual((view["description"], view["fixed"]),
                         ("Male, warm and low, sounds about forty.", "Female, warm and low, sounds about forty."))
        self.assertTrue(view["why"].startswith("The book says: “"))

    def test_leaving_it_on_the_run_shells_screen_changes_nothing(self):
        self.make_book(risk=False, prepare=male_mara)
        manifest = (self.book / "manifest.json").read_bytes()
        with screen_host(ScreenHost("l")):
            changed, _ = self.review(GenderClient({"Mara": "Female"}), voiced={"Mara"}, interactive=True)
        self.assertFalse(changed)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_a_persons_notes_win_over_the_text(self):
        self.make_book(risk=False, prepare=male_mara)
        changed, output = self.review(GenderClient({"Mara": "Male"}), notes={"Mara": {"gender": "woman"}})
        self.assertTrue(changed, output)
        self.assertTrue(self.description("Mara").startswith("Female"))
        self.assertIn("your notes say so", output)

    def test_an_agreeing_cast_changes_nothing(self):
        self.make_book(risk=False, prepare=male_mara)
        manifest = (self.book / "manifest.json").read_bytes()
        changed, output = self.review(GenderClient({"Mara": "Male"}))
        self.assertFalse(changed)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
