from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.pronunciations import save_candidates, set_pronunciation
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.scoped_pronunciations import reset, scope_for, set_choice
from epub_tts_narrator.series import link_analysis
from name_script_fixtures import requests
from shared_fixtures import renderer_config
from test_pronunciations import candidate
from test_series import fixture, propose


NAME = {"alphabet": "cmu_arpabet", "value": "AE1 L IY0"}
ORDINARY = {"alphabet": "cmu_arpabet", "value": "AH0 L AY1"}


class FamilyClient:
    identity = {"model": "migration-family-fixture"}

    def complete(self, system, payload):
        return {"decisions": [item["names"][0] if item["start"] == 0 or item["word"].isupper() else "ordinary"
                              for item in payload["window"]["candidates"]]}


class MigrationFamilyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def book(self, label, name, text, previous=None):
        directory = self.root / label
        document = fixture()
        document["source"]["book"].update(id=label, title=label)
        document["source"]["blocks"][0]["text"] = text
        document["cast"]["characters"][0].update(name=name, aliases=[],
            evidence=[{"block_id": "b1", "quote": text.split()[0]}])
        base = document["segments"][0]
        document["segments"] = [dict(deepcopy(base), id=block["id"], block_id=block["id"], text=block["text"],
                                    start=0, end=len(block["text"])) for block in document["source"]["blocks"]]
        registry = fresh_registry(directory) if previous is None else continuation_registry(previous, document["source"])
        if previous is not None:
            propose(document, load_bundle(previous)["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle, continue_from=previous)
        bundle = load_bundle(directory)
        prepare_name_script(directory, {}, requests=requests(self.root, FamilyClient()))
        return bundle

    def controls(self, bundle):
        plan = make_plan(bundle, renderer_config(paragraph_synthesis=False))
        return plan["chapters"][0]["chunks"][0]["expressive_performance"]["controls"]

    def test_migrating_name_preserves_separate_capitonym_and_initialism_rules(self):
        for name, ordinary in (("Polish", "polish"), ("US", "us")):
            with self.subTest(name=name):
                text = f"{name} spoke softly about {ordinary}. {name.upper()} waved."
                bundle = self.book(name, name, text)
                directory = bundle["directory"]
                set_pronunciation(directory, bundle, name, NAME)
                set_pronunciation(directory, bundle, ordinary, ORDINARY)
                raw = Path(directory, "pronunciations.json").read_bytes()
                set_choice(directory, bundle, name, NAME, scope=scope_for(bundle, "N1"), migrate=True)
                controls = self.controls(bundle)
                self.assertEqual([(item["text"], item["phonemes"]) for item in controls],
                                 [(name, NAME["value"]), (ordinary, ORDINARY["value"]), (name.upper(), NAME["value"])])
                self.assertEqual(Path(directory, "pronunciations.json").read_bytes(), raw)

    def test_normal_capitalization_family_migration_still_removes_ordinary_override(self):
        bundle = self.book("ally", "Ally", "Ally spoke softly about ally. ALLY waved.")
        set_pronunciation(bundle["directory"], bundle, "Ally", NAME)
        set_choice(bundle["directory"], bundle, "Ally", NAME, scope=scope_for(bundle, "N1"), migrate=True)
        self.assertEqual([item["text"] for item in self.controls(bundle)], ["Ally", "ALLY"])

    def test_inherited_short_shouted_name_keeps_original_legacy_name_family(self):
        parent = self.book("parent", "Al", "Al spoke softly about al. AL waved.")
        save_candidates(parent["directory"], parent, [candidate("Al")])
        set_pronunciation(parent["directory"], parent, "Al", NAME)
        set_choice(parent["directory"], parent, "Al", NAME, scope=scope_for(parent, "N1"), migrate=True)
        child = self.book("child", "Al", "AL spoke softly.", previous=parent["directory"])
        self.assertEqual([(item["text"], item["phonemes"]) for item in self.controls(child)], [("AL", NAME["value"])])

    def test_parent_name_migration_does_not_conflict_with_child_ordinary_family(self):
        text = "Polish spoke softly about polish. POLISH waved."
        parent = self.book("parent", "Polish", text)
        set_pronunciation(parent["directory"], parent, "Polish", NAME)
        set_pronunciation(parent["directory"], parent, "polish", ORDINARY)
        set_choice(parent["directory"], parent, "Polish", NAME, scope=scope_for(parent, "N1"), migrate=True)
        child = self.book("child", "Polish", text, previous=parent["directory"])
        newer = {"alphabet": "cmu_arpabet", "value": "P AA1 L IH0 SH"}
        set_pronunciation(child["directory"], child, "polish", newer)
        self.assertEqual([(item["text"], item["phonemes"]) for item in self.controls(child)],
                         [("Polish", NAME["value"]), ("polish", newer["value"]), ("POLISH", NAME["value"])])

    def test_reset_name_default_does_not_borrow_separate_ordinary_rule_at_marked_mention(self):
        from epub_tts_narrator.name_script_state import mention_id, read_state
        from epub_tts_narrator.name_script_workflow import resolve_mention
        text = "Polish spoke softly about polish."
        bundle = self.book("polish", "Polish", text)
        directory = bundle["directory"]
        set_pronunciation(directory, bundle, "Polish", NAME)
        set_pronunciation(directory, bundle, "polish", ORDINARY)
        scope = scope_for(bundle, "N1")
        set_choice(directory, bundle, "Polish", NAME, scope=scope, migrate=True)
        state = read_state(directory)
        start = text.index("polish")
        resolve_mention(directory, mention_id(state, "b1", start, start + len("polish")), "N1")
        reset(directory, "Polish", scope)
        plan = make_plan(bundle, renderer_config(paragraph_synthesis=False))
        self.assertNotIn("expressive_performance", plan["chapters"][0]["chunks"][0])
        resolve_mention(directory, mention_id(state, "b1", start, start + len("polish")), "unresolved")
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            make_plan(bundle, renderer_config(paragraph_synthesis=False))
        resolve_mention(directory, mention_id(state, "b1", start, start + len("polish")), "ordinary")
        self.assertEqual([(item["text"], item["phonemes"]) for item in self.controls(bundle)], [("polish", ORDINARY["value"])])
