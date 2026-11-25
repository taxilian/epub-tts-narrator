"""Rerunning analyze gives a line the scene's characters say together ("the voices all shouted") to all of them."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.joint_lines import candidates, kept, review_joint_lines
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import link_analysis
from test_character_merge import AUDITION, build

CHARACTERS = [("first", "First Voice", ()), ("second", "Second Voice", ()), ("third", "Third Voice", ()),
              ("mara", "Mara", ())]
PARAGRAPHS = [
    [("“Here,” ", "first"), ("the voice behind the first door said.", None)],
    [("“She is here,” ", "second"), ("the wall whispered.", None)],
    [("“Found her,” ", "third"), ("a third voice said.", None)],
    [("“Her!” ", "first"), ("the voices all shouted as one.", None)],
    [("“Hello,” ", "mara"), ("she said.", None)],
]


class JointClient:
    """Answers with ``speakers`` (numbers in each item) and the line's tag as the quote, or "n"."""

    def __init__(self, speakers=None, *, quote=None):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.speakers, self.quote, self.calls, self.lock = speakers, quote, [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        assert payload["mode"] == "joint_line_review", payload["mode"]
        answers = []
        for item in payload["items"]:
            if self.speakers is None:
                answers.append([item["id"], "n"])
                continue
            number = int(item["text"].splitlines()[-1].split("|", 1)[0])
            answers.append([item["id"], self.speakers, number, self.quote or item["tag"]])
        return {"r": answers}


class JointLineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"

    def publish(self, *, by_person=False):
        document = build("voices", "Voices", CHARACTERS, PARAGRAPHS)
        if by_person:
            line = next(row for row in document["segments"] if row["text"].strip() == "“Her!”")
            line["attribution"].update(basis="user", reason="A person chose this speaker.", evidence=[])
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)

    def run_check(self, client, **options):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_joint_lines(self.path, config, lambda settings: client, lambda message: None, **options)
        return changed, output.getvalue()

    def line(self):
        return next(row for row in _completed_directory(self.path)[0]["segments"] if row["text"].strip() == "“Her!”")

    def names(self, row):
        document = _completed_directory(self.path)[0]
        roles = {role["id"]: role["character_id"] for role in document["cast"]["roles"]}
        names = {character["id"]: character["name"] for character in document["cast"]["characters"]}
        return [names[roles[speaker["role_id"]]] for speaker in row.get("simultaneous_speakers", [])]

    def test_the_flagged_line_is_asked_about_with_the_scenes_speakers(self):
        self.publish()
        found = candidates(_completed_directory(self.path)[0])
        self.assertEqual(len(found), 1)
        row, clause, speakers, _ = found[0]
        self.assertEqual((row["text"].strip(), clause), ("“Her!”", "the voices all shouted as one."))
        self.assertEqual(len(speakers), 3, "the three voices of the scene, the line's own speaker first")

    def test_a_line_the_scene_says_together_is_given_to_all_of_them_with_the_quote(self):
        self.publish()
        changed, output = self.run_check(JointClient([1, 2, 3]))
        self.assertTrue(changed, output)
        row = self.line()
        self.assertEqual(self.names(row), ["First Voice", "Third Voice", "Second Voice"])
        self.assertEqual((row["role_id"], row["attribution"]["basis"]), (None, "explicit"))
        self.assertEqual(row["attribution"]["evidence"][0]["quote"], "the voices all shouted as one.")
        self.assertEqual(validate_document(_completed_directory(self.path)[0]), [])
        self.assertIn("Second Voice now say line b4:0 together.", output)
        self.assertNotIn("shouted", output, "no book text in the log")
        self.assertEqual(self.run_check(JointClient([1, 2, 3]))[0], False, "a joint line is no longer flagged")

    def test_one_speaker_or_a_quote_not_in_the_book_is_refused_and_nothing_changes(self):
        self.publish()
        before = self.line()
        for client in (JointClient([1]), JointClient([1, 2], quote="words nowhere in it")):
            with self.subTest(client.speakers), self.assertRaises(Exception):
                self.run_check(client)
        self.assertEqual(self.line(), before)

    def test_a_no_changes_nothing(self):
        self.publish()
        changed, output = self.run_check(JointClient(None))
        self.assertFalse(changed)
        self.assertIn("no flagged line", output)
        self.assertNotIn("simultaneous_speakers", self.line())

    def test_a_line_a_person_assigned_is_never_changed_without_them(self):
        self.publish(by_person=True)
        changed, output = self.run_check(JointClient([1, 2, 3]), interactive=False)
        self.assertFalse(changed)
        self.assertIn("which you assigned", output)
        self.assertNotIn("simultaneous_speakers", self.line())

    def test_a_person_who_says_together_gets_a_joint_line_marked_as_theirs(self):
        self.publish(by_person=True)
        with patch("epub_tts_narrator.joint_lines._decide", return_value="j") as decide:
            changed, _ = self.run_check(JointClient([1, 2, 3]), interactive=True)
        self.assertTrue(changed)
        view = decide.call_args.args[0]
        self.assertEqual((view["now"], view["together"]),
                         ("First Voice alone (you chose)", "First Voice, Third Voice, Second Voice"))
        self.assertEqual(self.line()["attribution"]["basis"], "user")
        self.assertEqual(len(self.line()["simultaneous_speakers"]), 3)

    def test_a_kept_line_is_remembered_and_not_asked_again(self):
        self.publish(by_person=True)
        with patch("epub_tts_narrator.joint_lines._decide", return_value="k"):
            changed, _ = self.run_check(JointClient([1, 2, 3]), interactive=True)
        self.assertFalse(changed)
        self.assertIn(self.line()["id"], kept(self.path))
        client = JointClient([1, 2, 3])
        with patch("epub_tts_narrator.joint_lines._decide") as decide:
            self.assertFalse(self.run_check(client, interactive=True)[0])
        self.assertFalse(decide.called)
        self.assertEqual(client.calls, [], "nothing left to ask the model about")

    def test_later_asks_again_next_time(self):
        self.publish(by_person=True)
        with patch("epub_tts_narrator.joint_lines._decide", return_value="l"):
            self.run_check(JointClient([1, 2, 3]), interactive=True)
        self.assertEqual(kept(self.path), {})


class JointLineScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_screen_shows_the_line_and_answers_with_a_letter(self):
        from textual.app import App
        from epub_tts_narrator.rerun_review_tui import JointLineScreen
        view = {"title": "Do they say this together?", "line": "“Her!”", "paragraph": "“Her!” the voices all shouted.",
                "now": "First Voice alone (you chose)", "together": "First Voice, Second Voice", "speakers": [],
                "why": "The book: “the voices all shouted”.", "afterwards": "Only this line changes."}
        answers = []

        class Host(App):
            def on_mount(self):
                self.push_screen(JointLineScreen(view), answers.append)
        async with Host().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("j")
            await pilot.pause()
        self.assertEqual(answers, ["j"])


if __name__ == "__main__":
    unittest.main()
