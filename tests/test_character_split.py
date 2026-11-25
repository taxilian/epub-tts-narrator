"""Rerunning analyze with instructions splits a cast entry that holds two characters' lines."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.character_split import kept, review_splits, split_document, validate_character_split
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import link_analysis
from test_character_merge import AUDITION, build

CHARACTERS = [("eater", "Third Eaterling", ("third voice",)), ("first", "First Voice", ()),
              ("second", "Second Voice", ())]
PARAGRAPHS = [
    [("“Rush her now!” ", "eater"), ("the third eaterling hissed to the pack.", None)],
    [("“Hurry!” ", "eater"), ("the pack's third said.", None)],
    [("“Here,” ", "first"), ("the voice behind the first door whispered.", None)],
    [("“She's here,” ", "second"), ("the wall whispered.", None)],
    [("“Found her,” ", "eater"), ("a third voice crawled out from behind the other door.", None)],
    [("“Give,” ", "eater"), ("the third voice said, over its own slamming.", None)],
]
INSTRUCTIONS = "There are three whispering voices behind the doors; the third is not the eaterling from the pack."


class SplitClient:
    """Nominates the eaterling, then gives the whispering paragraphs to "Third Voice"."""

    def __init__(self, *, answer="split"):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.answer, self.calls = answer, []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == "character_split_nominate":
            label = next(line.split("|")[0] for line in payload["cast"].splitlines() if "|Third Eaterling|" in line)
            return {"r": [[label, "the third whispering voice"]]}
        assert payload["mode"] == "character_split_check", payload["mode"]
        item = payload["items"][0]
        if self.answer == "n":
            return {"r": [[item["id"], "n"]]}
        rows = [line.split("|", 1) for line in item["text"].splitlines()]
        whisper = [int(number) for number, text in rows if "third voice" in text]
        return {"r": [[item["id"], "Third Voice", "Male, a hissing whisper from behind a door.", whisper,
                       whisper[0], "a third voice crawled out from behind the other door"]]}


class CharacterSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"

    def publish(self, *, chosen=False):
        document = build("whispers", "Whispers", CHARACTERS, PARAGRAPHS)
        if chosen:
            row = next(row for row in document["segments"] if row["text"].startswith("“Give"))
            row["attribution"].update(basis="user", reason="A person chose this speaker.", evidence=[])
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)

    def run_review(self, client, *, answer="s", **options):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        with patch("epub_tts_narrator.character_split._decide", return_value=answer) as decide, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_splits(self.path, config, lambda settings: client, lambda message: None,
                                    instructions=INSTRUCTIONS, interactive=True, **options)
        return changed, output.getvalue(), decide

    def speakers(self):
        document = _completed_directory(self.path)[0]
        roles = {role["id"]: role["character_id"] for role in document["cast"]["roles"]}
        names = {character["id"]: character["name"] for character in document["cast"]["characters"]}
        return {row["text"].strip(): names.get(roles.get(row["role_id"])) for row in document["segments"]
                if row["kind"] == "dialogue"}

    def test_the_whispering_lines_go_to_a_new_character_and_the_pack_lines_stay(self):
        self.publish()
        changed, output, decide = self.run_review(SplitClient())
        self.assertTrue(changed, output)
        view = decide.call_args.args[0]
        self.assertEqual(view["summary"], "2 lines of Third Eaterling would go to a new character, Third Voice.")
        self.assertEqual(self.speakers(), {"“Rush her now!”": "Third Eaterling", "“Hurry!”": "Third Eaterling",
                                           "“Here,”": "First Voice", "“She's here,”": "Second Voice",
                                           "“Found her,”": "Third Voice", "“Give,”": "Third Voice"})
        document = _completed_directory(self.path)[0]
        self.assertEqual(validate_document(document), [])
        new = next(row for row in document["cast"]["characters"] if row["name"] == "Third Voice")
        self.assertTrue(new["description"].startswith("Male"))
        self.assertIn("Cast the new voice", output)

    def test_nothing_happens_without_instructions_or_when_the_text_does_not_show_it(self):
        self.publish()
        client = SplitClient()
        config = {"analysis": deepcopy(DEFAULTS["analysis"]), "provider": {"model": "test-model"}}
        self.assertFalse(review_splits(self.path, config, lambda settings: client, lambda message: None))
        self.assertEqual(client.calls, [])
        self.assertFalse(self.run_review(SplitClient(answer="n"))[0])

    def test_keep_is_remembered_and_not_asked_again(self):
        self.publish()
        self.assertFalse(self.run_review(SplitClient(), answer="k")[0])
        self.assertIn(next(row["id"] for row in _completed_directory(self.path)[0]["cast"]["characters"]
                           if row["name"] == "Third Eaterling"), kept(self.path))
        changed, _, decide = self.run_review(SplitClient())
        self.assertFalse(changed)
        self.assertFalse(decide.called)

    def test_a_line_a_person_chose_moves_only_with_reconsider_choices(self):
        self.publish(chosen=True)
        changed, _, decide = self.run_review(SplitClient())
        self.assertTrue(changed)
        self.assertIn("--reconsider-choices", " ".join(decide.call_args.args[0]["notes"]))
        self.assertEqual(self.speakers()["“Give,”"], "Third Eaterling")

        self.setUp()
        self.publish(chosen=True)
        self.assertTrue(self.run_review(SplitClient(), reconsider_choices=True)[0])
        self.assertEqual(self.speakers()["“Give,”"], "Third Voice")

    def test_a_split_revision_may_only_give_lines_to_one_other_character(self):
        self.publish()
        document = _completed_directory(self.path)[0]
        eater = next(row["id"] for row in document["cast"]["characters"] if row["name"] == "Third Eaterling")
        found = {"name": "First Voice", "description": "x", "blocks": ["b5"],
                 "evidence": {"block_id": "b5", "quote": "a third voice"}}
        candidate, moved, _, target, created = split_document(document, eater, found)
        self.assertEqual((target["name"], created, len(moved)), ("First Voice", False, 1),
                         "a name the cast already has takes the lines without a new character")
        self.assertEqual(validate_character_split(document, candidate), [])
        changed_text = deepcopy(candidate)
        changed_text["segments"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "only change who speaks"):
            validate_character_split(document, changed_text)
        with self.assertRaisesRegex(ValueError, "at least one line"):
            validate_character_split(document, deepcopy(document))


class SplitScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_screen_lists_the_lines_and_answers_with_a_letter(self):
        from textual.app import App
        from epub_tts_narrator.rerun_review_tui import SplitScreen
        view = {"title": "Are some of Third Eaterling's lines someone else's?",
                "summary": "2 lines of Third Eaterling would go to a new character, Third Voice.",
                "why": "The book: “a third voice”.", "from": "Third Eaterling", "to": "Third Voice",
                "lines": ["“Found her,”", "“Give,”"], "notes": ["Keeping them together is remembered."]}
        answers = []

        class Host(App):
            def on_mount(self):
                self.push_screen(SplitScreen(view), answers.append)
        async with Host().run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        self.assertEqual(answers, ["s"])


if __name__ == "__main__":
    unittest.main()
