import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.pronunciations import set_pronunciation
from epub_tts_narrator.renderer.bundle import make_plan
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.scoped_pronunciations import scope_for, set_choice
from name_script_fixtures import NameClient, PHONES, TEXT, book, requests
from shared_fixtures import renderer_config


class ScopedPronunciationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "book"
        self.bundle = book(self.directory, two=True)
        self.config = renderer_config(paragraph_synthesis=False)
        self.before = make_plan(self.bundle, self.config)
        prepare_name_script(self.directory, {}, requests=requests(self.root, NameClient()))

    def chunks(self, config=None):
        return make_plan(self.bundle, config or self.config)["chapters"][0]["chunks"]

    def test_no_override_keeps_exact_legacy_plan(self):
        self.assertEqual(make_plan(self.bundle, self.config), self.before)

    def test_entity_rule_never_leaks_to_ordinary_or_second_entity(self):
        scope = scope_for(self.bundle, "N1")
        set_pronunciation(self.directory, self.bundle, "Ally", PHONES, scope=scope)
        chunks = self.chunks()
        self.assertEqual([item["start"] for item in chunks[0]["expressive_performance"]["controls"]], [0, TEXT.index("ALLY")])
        self.assertNotIn("expressive_performance", chunks[1])
        for original, changed in zip(self.before["chapters"][0]["chunks"], chunks):
            for key in ("segment", "spoken_text", "source_ranges", "chunk_id"):
                self.assertEqual(changed[key], original[key])
        engine_text = prepare_synthesis(chunks[0]["spoken_text"], performance=chunks[0]["expressive_performance"])[1]
        self.assertIn("her ally", engine_text)
        self.assertIn("Ally with", engine_text)
        self.assertNotIn("(N1:", engine_text)

    def test_two_entities_same_spelling_have_independent_readings(self):
        second = {"alphabet": "cmu_arpabet", "value": "AH0 L AY1"}
        for number, phones in (("N1", PHONES), ("N2", second)):
            set_choice(self.directory, self.bundle, "Ally", phones, scope=scope_for(self.bundle, number))
        chunks = self.chunks()
        self.assertEqual(chunks[0]["expressive_performance"]["controls"][0]["phonemes"], PHONES["value"])
        self.assertEqual(chunks[1]["expressive_performance"]["controls"][0]["phonemes"], second["value"])

    def test_special_rule_excludes_names_even_when_names_have_no_override(self):
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope={"class": "special"})
        chunks = self.chunks()
        controls = chunks[0]["expressive_performance"]["controls"]
        self.assertEqual([item["start"] for item in controls], [TEXT.index("ally"), TEXT.index("Ally with")])
        self.assertNotIn("expressive_performance", chunks[1])

    def test_legacy_collision_requires_explicit_migration_and_preserves_raw_registry(self):
        set_pronunciation(self.directory, self.bundle, "Ally", PHONES)
        raw = (self.directory / "pronunciations.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "migrate"):
            self.chunks()
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope_for(self.bundle, "N1"), migrate=True)
        self.assertEqual(len(self.chunks()[0]["expressive_performance"]["controls"]), 2)
        self.assertEqual((self.directory / "pronunciations.json").read_bytes(), raw)

    def test_frozen_snapshot_survives_new_choices_during_recovery(self):
        from epub_tts_narrator.renderer.workflow import _preflight, make_render_plan
        from copy import deepcopy
        rows = self.bundle["chapters"][0]["rows"]
        original = rows[0]
        split = original["text"].index("ALLY")
        rows[:1] = [dict(deepcopy(original), id="left", text=original["text"][:split], end=split),
                    dict(deepcopy(original), id="right", text=original["text"][split:], start=split)]
        self.config["render"]["paragraph_synthesis"] = True
        scope = scope_for(self.bundle, "N1")
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope)
        config, before, _ = _preflight(self.bundle, self.config, chapters_only=True, voice_store_override=object())
        self.assertTrue(any("paragraph_turns" in chunk for chunk in before["chapters"][0]["chunks"]))
        set_choice(self.directory, self.bundle, "Ally", status="ignored", scope=scope)
        self.assertEqual(make_render_plan(self.bundle, config), before)
        self.assertNotIn("expressive_performance", self.chunks()[0])

    def test_real_renderer_cli_set_and_ignore_scoped_rule(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["pronunciations", str(self.directory), "--action", "set", "--class", "name",
                         "--name-id", "N1", "--word", "Ally", "--phonemes", PHONES["value"]])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.chunks()[0]["expressive_performance"]["controls"]), 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["pronunciations", str(self.directory), "--action", "ignore", "--class", "name",
                         "--name-id", "N1", "--word", "Ally"])
        self.assertEqual(code, 0)
        self.assertNotIn("expressive_performance", self.chunks()[0])

    def test_closed_remote_contract_carries_only_existing_controls(self):
        from epub_tts_narrator.renderer.chapter_protocol import _chunk_shape, _leaf
        from epub_tts_narrator.renderer.workflow import make_render_plan
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope_for(self.bundle, "N1"))
        plan = make_render_plan(self.bundle, self.config, multi_speaker=False)
        bindings = {key: "a" * 64 for key in plan["voice_ids"]}
        for chunk in plan["chapters"][0]["chunks"]:
            _chunk_shape(chunk, set())
            _leaf(chunk, bindings)
        first = plan["chapters"][0]["chunks"][0]
        self.assertEqual(first["spoken_text"], TEXT)
        self.assertEqual([item["text"] for item in first["expressive_performance"]["controls"]], ["Ally", "ALLY"])

    def test_stale_annotations_block_overrides_but_not_default_rendering(self):
        from epub_tts_narrator.name_script_state import read_state, save_state
        scope = scope_for(self.bundle, "N1")
        state = read_state(self.directory)
        state["binding"]["identities"] = "changed"
        save_state(self.directory, state)
        self.assertEqual(make_plan(self.bundle, self.config), self.before)
        with self.assertRaises(ValueError):
            set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope)

    def test_word_only_cli_cannot_choose_one_of_multiple_entities(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(["pronunciations", str(self.directory), "--action", "set",
                         "--word", "Ally", "--phonemes", PHONES["value"]])
        self.assertEqual(code, 2)
        self.assertIn("--name-id", output.getvalue())
        self.assertEqual(make_plan(self.bundle, self.config), self.before)

    def test_contextually_identified_capitonym_name_groups_lowercase_mentions(self):
        from epub_tts_narrator.pipeline import write_json
        from epub_tts_narrator.name_script_state import new_state, save_state
        from epub_tts_narrator.scoped_pronunciations import load
        from copy import deepcopy
        document = deepcopy(self.bundle)
        document["source"]["blocks"][0]["text"] = "Job greeted job."
        document["source"]["blocks"][1]["text"] = "Job waited."
        document["cast"]["characters"][0].update(name="Job", aliases=[])
        document["cast"]["characters"][1].update(name="Job", aliases=[])
        directory = self.root / "minimal"
        directory.mkdir()
        document["directory"] = directory
        write_json(directory / "source.json", document["source"], overwrite=True)
        state = new_state(document)
        state["windows"] = [
            {"block_id": "b1", "start": 0, "end": 16, "text": "(N1:Job) greeted (N1:job).", "unresolved": []},
            {"block_id": "b2", "start": 0, "end": 11, "text": "(N2:Job) waited.", "unresolved": []}]
        state["status"] = "ready"
        save_state(directory, state)
        scope = scope_for(document, "N1")
        set_choice(directory, document, "Job", PHONES, scope=scope)
        self.assertEqual(set(load(directory, scope)[0]["variants"]), {"Job", "job"})

    def test_scoped_name_maps_normalization_without_changing_expected_words_or_ranges(self):
        row = self.bundle["chapters"][0]["rows"][0]
        row["speech_normalization"] = {"version": 1, "edits": [{
            "start": 0, "end": 4, "original": "Ally", "replacement": "Alli",
            "direction": "Pronunciation hint", "reason": "Approved spelling."}]}
        before = self.chunks()[0]
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope_for(self.bundle, "N1"))
        after = self.chunks()[0]
        for key in ("spoken_text", "source_ranges", "segment", "chunk_id"):
            self.assertEqual(after[key], before[key])
        controls = [item for item in after["expressive_performance"]["controls"] if item["kind"] == "pronunciation"]
        self.assertEqual([item["text"] for item in controls], ["Alli", "ALLY"])
        row["speech_normalization"]["edits"][0].update(end=12, original="Ally greeted", replacement="Alli greeted")
        with self.assertRaisesRegex(ValueError, "multiword"):
            self.chunks()

    def test_name_rule_cannot_silently_drop_a_mention_split_across_script_rows(self):
        from copy import deepcopy
        rows = self.bundle["chapters"][0]["rows"]
        original = rows[0]
        rows[:1] = [dict(deepcopy(original), id="left", text=original["text"][:2], end=2),
                    dict(deepcopy(original), id="right", text=original["text"][2:], start=2)]
        set_choice(self.directory, self.bundle, "Ally", PHONES, scope=scope_for(self.bundle, "N1"))
        with self.assertRaisesRegex(ValueError, "segment boundary"):
            self.chunks()
