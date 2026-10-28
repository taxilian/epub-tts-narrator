from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.renderer.bundle import make_plan
from epub_tts_narrator.scoped_pronunciations import load, scope_for, set_choice
from name_script_fixtures import NameClient, PHONES, book, requests
from shared_fixtures import renderer_config


class ScopedPronunciationInheritanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = book(self.root / "first")
        prepare_name_script(self.first["directory"], {}, requests=requests(self.root, NameClient()))
        self.scope = scope_for(self.first, "N1")
        set_choice(self.first["directory"], self.first, "Ally", PHONES, scope=self.scope)

    def test_same_numeric_label_and_spelling_do_not_identify_a_sequel_character(self):
        second = book(self.root / "second", previous=self.first["directory"], reuse=False)
        prepare_name_script(second["directory"], {}, requests=requests(self.root, NameClient()))
        scope = scope_for(second, "N1")
        self.assertNotEqual(scope, self.scope)
        self.assertEqual(load(second["directory"], scope), [])
        plan = make_plan(second, renderer_config(paragraph_synthesis=False))
        self.assertTrue(all("expressive_performance" not in chunk for chunk in plan["chapters"][0]["chunks"]))

    def test_actual_series_identity_inherits_late_choices_but_not_mentions(self):
        second = book(self.root / "second", previous=self.first["directory"])
        scope_entries = load(second["directory"], self.scope)
        self.assertEqual(scope_entries[0]["pronunciation"], PHONES)
        with self.assertRaises(ValueError):
            make_plan(second, renderer_config(paragraph_synthesis=False))
        prepare_name_script(second["directory"], {}, requests=requests(self.root, NameClient()))
        self.assertEqual(scope_for(second, "N1"), self.scope)
        plan = make_plan(second, renderer_config(paragraph_synthesis=False))
        self.assertEqual(len(plan["chapters"][0]["chunks"][0]["expressive_performance"]["controls"]), 2)
        set_choice(self.first["directory"], self.first, "Ally", status="ignored", scope=self.scope)
        plan = make_plan(second, renderer_config(paragraph_synthesis=False))
        self.assertNotIn("expressive_performance", plan["chapters"][0]["chunks"][0])

    def test_unrelated_book_never_inherits_by_numeric_label(self):
        second = book(self.root / "unrelated")
        prepare_name_script(second["directory"], {}, requests=requests(self.root, NameClient()))
        self.assertEqual(load(second["directory"], scope_for(second, "N1")), [])

    def test_unmatched_entity_cannot_fall_back_to_unmigrated_legacy_name_choice(self):
        from epub_tts_narrator.pronunciations import set_pronunciation
        set_pronunciation(self.first["directory"], self.first, "Ally", PHONES)
        second = book(self.root / "second", previous=self.first["directory"], reuse=False)
        with self.assertRaisesRegex(ValueError, "migrate"):
            make_plan(second, renderer_config(paragraph_synthesis=False))

    def test_explicit_descendant_default_needs_no_stale_annotation_for_ancestor_override(self):
        from epub_tts_narrator.name_script_state import read_state, save_state
        second = book(self.root / "second", previous=self.first["directory"])
        prepare_name_script(second["directory"], {}, requests=requests(self.root, NameClient()))
        set_choice(second["directory"], second, "Ally", status="ignored", scope=scope_for(second, "N1"))
        state = read_state(second["directory"])
        state["binding"]["identities"] = "stale"
        save_state(second["directory"], state)
        plan = make_plan(second, renderer_config(paragraph_synthesis=False))
        self.assertNotIn("expressive_performance", plan["chapters"][0]["chunks"][0])
