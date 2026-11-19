"""``rename-character``: a cast character gets a new name; nothing else in the book changes."""

from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace
import unittest

from epub_tts_narrator.analysis_directory import continuation_registry, publish_directory
from epub_tts_narrator.analysis_revision import revise_directories
from epub_tts_narrator.character_rename import rename_document, validate_character_rename
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.renderer.character_rename_command import apply_rename, plan_rename, run_rename_command
from epub_tts_narrator.series import link_analysis
from test_attribution_verification import risk_document
from test_attribution_verification_cli import AttributionCLIHarness


class TerminalIO(io.StringIO):
    def isatty(self):
        return True


def misspelled(document):
    """Tom was typed as "Tmo"; the book's text never says Tmo."""
    tom = next(row for row in document["cast"]["characters"] if row["name"] == "Tom")
    tom.update(name="Tmo", description="Tmo, another speaker.")
    next(role for role in document["cast"]["roles"] if role["character_id"] == tom["id"])["description"] = \
        "Tmo speaking."


class CharacterRenameTests(AttributionCLIHarness):
    def analysis(self, book=None):
        return json.loads(((book or self.book) / "analysis.json").read_text(encoding="utf-8"))

    def character(self, name, book=None):
        return next(row for row in self.analysis(book)["cast"]["characters"] if row["name"] == name)

    def test_a_misspelled_name_is_replaced_and_nothing_else_changes(self):
        self.make_book(risk=False, prepare=misspelled)
        before = self.analysis()
        apply_rename(plan_rename(self.book, "Tmo", "Tom"), progress=lambda _: None)
        after = self.analysis()
        tom = self.character("Tom")
        self.assertEqual((tom["aliases"], tom["description"]), ([], "Tom, another speaker."))
        self.assertEqual(next(role for role in after["cast"]["roles"] if role["character_id"] == tom["id"])
                         ["description"], "Tom speaking.")
        self.assertEqual(after["segments"], before["segments"])
        self.assertEqual(after["cast"]["voices"], before["cast"]["voices"])
        self.assertEqual(after["provenance"]["character_renames"][-1]["from"], "Tmo")

    def test_a_name_the_book_uses_stays_as_an_alias(self):
        self.make_book(risk=False)
        plan = plan_rename(self.book, "Mara", "Mara Venn")
        self.assertIn("stays as an alias", plan.text)
        apply_rename(plan, progress=lambda _: None)
        self.assertEqual(self.character("Mara Venn")["aliases"], ["Mara"])

    def test_another_entrys_name_is_refused(self):
        self.make_book(risk=False)
        with self.assertRaisesRegex(ValueError, "merge-characters"):
            plan_rename(self.book, "Tom", "mara")

    def test_the_rename_scope_refuses_other_changes(self):
        self.make_book(risk=False)
        document = self.analysis()
        tom = self.character("Tom")
        renamed = rename_document(document, tom["id"], "Thomas")
        validate_character_rename(document, renamed)
        tampered = deepcopy(renamed)
        tampered["cast"]["voices"][0]["description"] = "Something else."
        with self.assertRaisesRegex(ValueError, "only the character's name"):
            validate_character_rename(document, tampered)
        hashes = {str(self.book): hashlib.sha256((self.book / "manifest.json").read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "record each rename"):
            revise_directories({self.book: document}, directories=[self.book], expected_manifest_hashes=hashes,
                               scope="character-rename", progress=lambda _: None)

    def test_later_books_follow_and_accent_groups_show_the_new_name(self):
        self.make_book(risk=False)
        first = self.analysis()
        document = risk_document(False)
        document["source"]["book"].update(id="book-two", title="The Return")
        for row, earlier in zip(document["cast"]["characters"], first["cast"]["characters"]):
            row["series_identity"] = {"character_id": earlier["id"], "basis": "context", "review_required": False,
                                      "reason": "Same person.", "evidence": [{"block_id": "b1", "quote": "Mara"}]}
        later = self.root / "book-two"
        registry = continuation_registry(self.book, document["source"])
        publish_directory(link_analysis(document, registry), registry, later, export_bundle, continue_from=self.book)
        tom = self.character("Tom")
        self.assertEqual(self.character("Tom", later)["id"], tom["id"])
        write_json(self.book / "voice-accent-groups.json",
                   {"version": 1, "groups": [{"voice_ids": [tom["voice_id"]], "names": ["Tom"]}]})
        output = io.StringIO()
        args = SimpleNamespace(character="Tom", name="Thomas", yes=False, dry_run=False)
        self.assertEqual(run_rename_command(self.book, args, input_stream=io.StringIO("y\n"), output_stream=output), 0)
        self.assertIn("Nothing was changed", output.getvalue())
        self.assertEqual(run_rename_command(self.book, args, input_stream=TerminalIO("y\n"), output_stream=output), 0)
        self.assertEqual(self.character("Thomas", later)["id"], tom["id"])
        history = json.loads((later / "series-cast.json").read_text(encoding="utf-8"))["books"][0]
        self.assertIn("Thomas", [row["name"] for row in history["cast"]["characters"]])
        groups = json.loads((self.book / "voice-accent-groups.json").read_text(encoding="utf-8"))
        self.assertEqual(groups["groups"][0]["names"], ["Thomas"])


if __name__ == "__main__":
    unittest.main()
