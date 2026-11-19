"""Rerunning analyze asks the model about characters the speaker tags leave unclear."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.group_check import review_group_tags, unclear
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import link_analysis
from test_character_merge import AUDITION, build


class GroupClient:
    """Answers "y" with an exact quote for the names in ``groups``, else "n"."""

    def __init__(self, groups, *, bad_quote=False):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.groups, self.bad_quote, self.calls, self.lock = groups, bad_quote, [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        assert payload["mode"] == "group_review", payload["mode"]
        answers = []
        for item in payload["items"]:
            if item["who"].split("|")[0] in self.groups:
                number, text = item["text"].splitlines()[0].split("|", 1)
                quote = "words nowhere in it" if self.bad_quote else text.split("“")[-1].split("”")[-1].strip()
                answers.append([item["id"], "y", int(number), quote])
            else:
                answers.append([item["id"], "n"])
        return {"r": answers}


class GroupCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"
        document = build("nobles", "Nobles", [("nobles", "Arendish noble", ()), ("mara", "Mara", ())], [
            [("“Who now shall rule us?” ", "nobles"), ("the nobles asked.", None)],
            [("“Hello,” ", "mara"), ("she said.", None)]])
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)

    def run_check(self, client):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_group_tags(self.path, config, lambda settings: client, lambda message: None)
        return changed, output.getvalue()

    def nobles(self):
        return next(row for row in _completed_directory(self.path)[0]["cast"]["characters"]
                    if row["name"] == "Arendish noble")

    def test_only_unclear_non_personal_names_are_asked_about(self):
        document = _completed_directory(self.path)[0]
        self.assertEqual([character["name"] for character, _ in unclear(document)], ["Arendish noble"])

    def test_a_confirmed_group_is_proposed_with_the_models_exact_quote(self):
        changed, output = self.run_check(GroupClient({"Arendish noble"}))
        self.assertTrue(changed, output)
        group = self.nobles()["group"]
        self.assertEqual((group["basis"], group["evidence"][0]["quote"]), ("analysis", "the nobles asked."))
        self.assertIn("cast screen (C)", output)
        changed, _ = self.run_check(GroupClient({"Arendish noble"}))
        self.assertFalse(changed, "a proposed group is not asked about again")

    def test_one_person_changes_nothing(self):
        changed, _ = self.run_check(GroupClient(set()))
        self.assertFalse(changed)
        self.assertNotIn("group", self.nobles())

    def test_a_quote_that_is_not_in_the_paragraph_is_refused(self):
        client = GroupClient({"Arendish noble"}, bad_quote=True)
        with self.assertRaisesRegex(Exception, "exact text"):
            self.run_check(client)
        self.assertNotIn("group", self.nobles())
        self.assertGreater(len(client.calls), 1, "the model was asked to repair its answer")

    def test_nothing_is_asked_when_no_character_is_unclear(self):
        document = build("plain", "Plain", [("mara", "Mara", ())], [[("“Hello,” ", "mara"), ("she said.", None)]])
        self.assertEqual(unclear(document), [])


if __name__ == "__main__":
    unittest.main()
