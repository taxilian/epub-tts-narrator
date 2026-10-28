import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main as analysis_main
from epub_tts_narrator.name_script_workflow import prepare_name_script, resolve_mention
from epub_tts_narrator.pronunciations import set_pronunciation
from epub_tts_narrator.renderer.bundle import make_plan
from epub_tts_narrator.renderer.cli import main as renderer_main
from epub_tts_narrator.scoped_pronunciations import FILENAME, load, local, reset, scope_for, set_choice
from name_script_fixtures import NameClient, PHONES, TEXT, book, requests
from shared_fixtures import renderer_config


class ScopedSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = book(self.root / "first")
        self.directory = Path(self.bundle["directory"])

    def prepare(self, *, disagree=False):
        return prepare_name_script(self.directory, {}, requests=requests(self.root, NameClient(disagree=disagree)))

    def test_copied_registry_rejects_other_source_even_with_matching_word_and_block(self):
        other = book(self.root / "other", text=TEXT + " More softly.")
        set_choice(self.directory, self.bundle, "softly", PHONES, scope={"class": "special"})
        for empty in (False, True):
            if empty:
                reset(self.directory, "softly", {"class": "special"})
            Path(other["directory"], FILENAME).write_bytes((self.directory / FILENAME).read_bytes())
            with self.subTest(empty=empty), self.assertRaisesRegex(ValueError, "source|book|owner"):
                load(other["directory"], {"class": "special"})

    def test_owner_book_id_is_validated_even_for_migration_only_registry(self):
        set_choice(self.directory, self.bundle, "softly", PHONES, scope={"class": "special"})
        reset(self.directory, "softly", {"class": "special"})
        path = self.directory / FILENAME
        record = json.loads(path.read_text())
        record["book_id"] = "wrong-book"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "book|owner"):
            local(self.directory)

    def test_inheritance_validates_migration_only_owner_and_rejects_unbound_records(self):
        set_choice(self.directory, self.bundle, "softly", PHONES, scope={"class": "special"})
        reset(self.directory, "softly", {"class": "special"})
        child = book(self.root / "child", previous=self.directory)
        path = self.directory / FILENAME
        record = json.loads(path.read_text())
        record["source_fingerprint"] = "wrong"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "source"):
            load(child["directory"], {"class": "special"})
        record.pop("source_fingerprint")
        record.pop("book_id")
        path.write_text(json.dumps(record))
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "ownership"):
            local(self.directory)
        self.assertEqual(path.read_bytes(), before)

    def test_parent_migration_cannot_suppress_child_explicit_legacy_choice(self):
        for kind in ("name", "special"):
            parent = book(self.root / (kind + "-parent"))
            prepare_name_script(parent["directory"], {}, requests=requests(self.root, NameClient()))
            scope = scope_for(parent, "N1") if kind == "name" else {"class": "special"}
            set_choice(parent["directory"], parent, "Ally", PHONES, scope=scope, migrate=True)
            child = book(self.root / kind, previous=parent["directory"])
            prepare_name_script(child["directory"], {}, requests=requests(self.root, NameClient()))
            set_pronunciation(child["directory"], child, "Ally", {"alphabet": "cmu_arpabet", "value": "AH0 L AY1"})
            before = Path(child["directory"], "pronunciations.json").read_bytes()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "migrate"):
                make_plan(child, renderer_config(paragraph_synthesis=False))
            set_choice(child["directory"], child, "Ally", PHONES, scope=scope_for(child, "N1"), migrate=True)
            make_plan(child, renderer_config(paragraph_synthesis=False))
            reset(child["directory"], "Ally", scope_for(child, "N1"))
            make_plan(child, renderer_config(paragraph_synthesis=False))
            self.assertEqual(Path(child["directory"], "pronunciations.json").read_bytes(), before)

    def test_cli_special_confirmation_rejects_unresolved_without_registry_mutation(self):
        snapshot = self.prepare(disagree=True)
        set_choice(self.directory, self.bundle, "softly", status="ignored", scope={"class": "special"})
        before = (self.directory / FILENAME).read_bytes()
        arguments = ["pronunciations", str(self.directory), "--class", "special", "--word", "Ally",
                     "--action", "set", "--phonemes", PHONES["value"]]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = renderer_main(arguments)
        self.assertNotEqual(code, 0)
        self.assertEqual((self.directory / FILENAME).read_bytes(), before)
        set_choice(self.directory, self.bundle, "Ally", status="ignored", scope={"class": "special"})
        for occurrence in snapshot["unresolved"]:
            resolve_mention(self.directory, occurrence["mention_id"], "ordinary")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(renderer_main(arguments), 0)

    def test_parent_migration_cannot_override_newer_legacy_default_choice(self):
        set_choice(self.directory, self.bundle, "softly", PHONES, scope={"class": "special"})
        child = book(self.root / "child-default", previous=self.directory)
        set_pronunciation(child["directory"], child, "softly", status="ignored")
        with self.assertRaisesRegex(ValueError, "migrate"):
            make_plan(child, renderer_config(paragraph_synthesis=False))
        set_choice(child["directory"], child, "softly", status="ignored", scope={"class": "special"}, migrate=True)
        plan = make_plan(child, renderer_config(paragraph_synthesis=False))
        self.assertTrue(all("expressive_performance" not in chunk for chunk in plan["chapters"][0]["chunks"]))

    def test_reset_special_does_not_revive_migrated_legacy_without_name_script(self):
        set_pronunciation(self.directory, self.bundle, "softly", PHONES)
        legacy = (self.directory / "pronunciations.json").read_bytes()
        set_choice(self.directory, self.bundle, "softly", PHONES, scope={"class": "special"}, migrate=True)
        reset(self.directory, "softly", {"class": "special"})
        plan = make_plan(self.bundle, renderer_config(paragraph_synthesis=False))
        self.assertTrue(all("expressive_performance" not in chunk for chunk in plan["chapters"][0]["chunks"]))
        self.assertEqual((self.directory / "pronunciations.json").read_bytes(), legacy)

    def test_special_confirmation_requires_a_resolved_ordinary_occurrence(self):
        self.bundle = book(self.root / "names-only", text="Ally spoke softly.")
        self.directory = Path(self.bundle["directory"])
        self.prepare()
        with self.assertRaisesRegex(ValueError, "ordinary"):
            set_choice(self.directory, self.bundle, "Ally", PHONES, scope={"class": "special"})
        set_choice(self.directory, self.bundle, "Ally", status="ignored", scope={"class": "special"})

    def test_cli_can_choose_default_while_name_preparation_is_pending(self):
        from epub_tts_narrator.name_script_state import read_state, save_state
        self.prepare()
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope={"class": "special"})
        state = read_state(self.directory)
        state["status"] = "pending"
        save_state(self.directory, state)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = renderer_main(["pronunciations", str(self.directory), "--class", "special",
                                  "--word", "Ally", "--action", "ignore"])
        self.assertEqual(code, 0)
        self.assertEqual(load(self.directory, {"class": "special"})[0]["status"], "ignored")

    def test_cli_output_projects_identity_without_descriptions_or_evidence(self):
        self.prepare()
        for action in ("list", "prepare"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(analysis_main(["prepare-name-script", str(self.directory), "--action", action]), 0)
            result = json.loads(output.getvalue())
            for name in result["names"].values():
                self.assertEqual(set(name), {"character_id", "aliases"})
            for occurrence in result["occurrences"]:
                self.assertEqual(set(occurrence), {"mention_id", "block_id", "start", "end", "word", "names", "decision"})

    def test_annotation_capacity_is_checked_before_model_work(self):
        client = NameClient()
        with patch("epub_tts_narrator.name_script_state.MAX_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "size|capacity"):
                prepare_name_script(self.directory, {}, requests=requests(self.root, client))
        self.assertEqual(client.calls, [])
