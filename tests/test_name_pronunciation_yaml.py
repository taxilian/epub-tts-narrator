"""Confirmed pronunciations move between books through a hand-editable YAML file."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.name_pronunciation_yaml import dump, entries_from, export_file, import_file
from epub_tts_narrator.name_pronunciations import effective_definitions, load_local, save_definition
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_series import fixture, propose


class TransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def book(self, name, previous=None):
        directory = self.root / name
        document = fixture(sequel=previous is not None)
        if previous is None:
            document["source"]["book"].update(id=name, title=name)
            registry = fresh_registry(directory)
        else:
            document["source"]["book"].update(id=name, title=name)
            registry = continuation_registry(previous, document["source"])
            prior = json.loads((previous / "analysis.json").read_text())
            propose(document, prior["cast"]["characters"][0]["id"])
        publish_directory(link_analysis(document, registry), registry, directory, export_bundle,
                          continue_from=previous)
        return directory

    def test_a_series_choices_round_trip_into_a_new_book(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual",
                        parts_of_speech=["proper_noun"])
        second = self.book("second", previous=first)
        save_definition(second, "Yarran", None, say_as="yar-ran", status="confirmed", source="manual")
        save_definition(second, "Quorn", "K W AO1 R N", status="draft", source="model")
        path = self.root / "series.yaml"

        self.assertEqual(export_file(second, path), 2)
        text = path.read_text()
        self.assertIn('phonemes: "M AA1 R AH0"', text)
        self.assertIn('say_as: "yar-ran"', text)
        self.assertIn("parts_of_speech: [proper noun]", text)
        self.assertNotIn("Quorn", text)  # Drafts never travel.
        self.assertEqual(export_file(second, self.root / "own.yaml", inherited=False), 1)

        fresh = self.book("fresh")
        outcome = import_file(fresh, path)
        self.assertEqual(outcome["added"], ["Mara", "Yarran"])
        definitions = load_local(fresh)["definitions"]
        self.assertEqual((definitions["mara"]["phonemes"], definitions["mara"]["parts_of_speech"]),
                         ("M AA1 R AH0", ["proper_noun"]))
        self.assertEqual(definitions["yarran"]["say_as"], "yar-ran")
        self.assertEqual(definitions["mara"]["history"][-1]["source"], "imported")
        self.assertEqual(load_local(fresh)["mentions"], {})
        # Importing the same file again changes nothing.
        before = (fresh / "name-pronunciations.json").read_bytes()
        self.assertEqual(import_file(fresh, path)["unchanged"], ["Mara", "Yarran"])
        self.assertEqual(json.loads((fresh / "name-pronunciations.json").read_bytes()), json.loads(before))

    def test_this_book_s_own_different_choice_is_kept_unless_replaced_and_an_inherited_one_is_overridden(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        path = self.root / "other.yaml"
        path.write_text(dump({"mara": {"name": "Mara", "phonemes": "M EH1 R AH0", "respelling": None,
                                       "parts_of_speech": ["proper_noun"], "history": []}}))
        self.assertEqual(import_file(second, path)["overrode_inherited"], ["Mara"])
        self.assertEqual(effective_definitions(second)["mara"]["phonemes"], "M EH1 R AH0")
        self.assertEqual(effective_definitions(first)["mara"]["phonemes"], "M AA1 R AH0")

        self.assertEqual(import_file(first, path)["conflict"], ["Mara"])
        self.assertEqual(load_local(first)["definitions"]["mara"]["phonemes"], "M AA1 R AH0")
        self.assertEqual(import_file(first, path, replace=True)["replaced"], ["Mara"])
        self.assertEqual(load_local(first)["definitions"]["mara"]["phonemes"], "M EH1 R AH0")

    def test_a_hand_edited_file_is_read_and_mistakes_name_their_line(self):
        entries = entries_from("""# my edits
format: epub-tts-pronunciations
version: 1
pronunciations:
  - word: 'Belgarath'   # the old man
    phonemes: b eh1 l g er0 ae2 th
    parts_of_speech:
      - proper noun
      - adjective
  - word: Garion
    phonemes: "G EH1 R IY0 | AH0 N"
""")
        self.assertEqual([(entry["name"], entry["reading"], entry["parts"]) for entry in entries], [
            ("Belgarath", {"phonemes": "B EH1 L G ER0 AE2 TH"}, ["proper_noun", "adjective"]),
            ("Garion", {"phonemes": "G EH1 R IY0 | AH0 N"}, ["proper_noun", "common_noun", "adjective", "verb",
                                                           "adverb", "interjection"])])
        header = "format: epub-tts-pronunciations\nversion: 1\npronunciations:\n"
        for body, message in (("  - word: Mara\n    sound: x\n", "line 5: unknown field 'sound'"),
                              ("  - word: Mara\n", "line 4: give exactly one of phonemes or say_as"),
                              ("  - word: Mara\n    phonemes: M AA R AH\n", "line 4: Name phonemes must mark"),
                              ("  - word: Mara\n    say_as: Mar\n  - word: mara\n    say_as: Mur\n", None)):
            with self.subTest(body=body):
                if message is None:
                    path = self.root / "twice.yaml"
                    path.write_text(header + body)
                    with self.assertRaisesRegex(ValueError, "more than once"):
                        import_file(self.book("twice"), path)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        entries_from(header + body)
        with self.assertRaisesRegex(ValueError, "Not a pronunciation file"):
            entries_from("pronunciations: []\n")

    def test_the_pronounce_command_exports_and_imports_then_tags_the_uses(self):
        from epub_tts_narrator.renderer.cli import main
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        other = self.book("other")
        path = self.root / "out.yaml"
        config = renderer_config(paragraph_synthesis=False)
        output = io.StringIO()
        with (patch("epub_tts_narrator.renderer.cli.load_config", return_value=deepcopy(config)),
              contextlib.redirect_stdout(output)):
            self.assertEqual(main(["pronounce", str(first), "--export", str(path)]), 0)
        self.assertIn("Exported 1 confirmed pronunciation(s)", output.getvalue())
        output = io.StringIO()
        with (patch("epub_tts_narrator.renderer.cli.load_config", return_value=deepcopy(config)),
              patch("epub_tts_narrator.renderer.name_pronunciation_tool.run_pronounce_name", return_value=0) as tag,
              contextlib.redirect_stdout(output)):
            self.assertEqual(main(["pronounce", str(other), "--import", str(path)]), 0)
        self.assertIn("1 added: Mara", output.getvalue())
        self.assertIsNone(tag.call_args.args[2].name)  # Then tags every confirmed word's uses.
        self.assertEqual(load_local(other)["definitions"]["mara"]["phonemes"], "M AA1 R AH0")


if __name__ == "__main__":
    unittest.main()
