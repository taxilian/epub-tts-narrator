"""User-defined name pronunciations: choose by ear, tag real uses, render tags.

Model and synthesis boundaries are fakes: no network or model calls.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.name_pronunciation_model import guess, tag
from epub_tts_narrator.name_pronunciations import (
    FILENAME, book_occurrences, effective_definitions, form_phonemes, load_local, local_draft, occurrences,
    render_snapshot, save_decisions, save_definition, tagging_state, validate_name_phonemes,
)
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.renderer.name_pronunciation_tool import run_pronounce_name, sentence_controls, test_sentence
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_renderer_pronunciation_review import EXAMPLE, HAS_FFMPEG, PreviewEngine
from test_renderer_workflow import write_tone
from test_series import fixture, propose


ROSE = "S AE1 L"  # Any valid reading; the tests check where it goes, not how it sounds.


class FakeModel:
    """Stands in for the provider: scripted guesses, rule-based tag decisions."""
    identity = {"model": "name-pronunciation-fixture"}
    model, base_url = "fixture-model", "http://fixture.invalid"

    def __init__(self, guesses=(), decide=None, fail_after=None):
        self.guesses = list(guesses)
        self.decide = decide or (lambda text: "name")
        self.fail_after = fail_after
        self.payloads = []

    def complete(self, system, payload):
        self.payloads.append(deepcopy(payload))
        if payload["mode"] == "name_pronunciation_guess":
            return self.guesses.pop(0)
        tagging = [item for item in self.payloads if item["mode"] == "name_use_tagging"]
        if self.fail_after is not None and len(tagging) > self.fail_after:
            raise RuntimeError("provider stopped")
        return {"decisions": [self.decide(item["text"]) for item in payload["occurrences"]]}


def homonym(text):
    """Fake judgement: the lowercase flower is not the name; everything else is."""
    return "other" if "⟦rose⟧" in text else "name"


def rose_document():
    document = valid_document()
    first = "Rose said softly, “Hello.”"
    second = "They left a rose on Rose’s desk, and ROSE laughed with the Roses."
    block1, block2 = document["source"]["blocks"]
    block1["text"], block2["text"] = first, second
    quote = first.index("“")
    s1, s2, s3 = document["segments"]
    s1.update(text=first[:quote], end=quote)
    s2.update(text=first[quote:], start=quote, end=len(first))
    s3.update(text=second, end=len(second))
    document["cast"]["characters"][0].update(name="Rose", evidence=[{"block_id": "b1", "quote": "Rose"}])
    return document


class FormsAndOccurrencesTests(unittest.TestCase):
    def test_inflections_follow_the_final_sound(self):
        cases = (("Salvatore", "S AE2 L V AH0 T AO1 R IY0", "Z"), ("James", "JH EY1 M Z", "IH0 Z"),
                 ("Mark", "M AA1 R K", "S"), ("Rich", "R IH1 CH", "IH0 Z"))
        for name, phonemes, suffix in cases:
            with self.subTest(name=name):
                forms = form_phonemes(name, phonemes)
                self.assertEqual(forms[name.casefold()], ("base", phonemes))
                self.assertEqual(forms[(name + "’s").casefold()], ("possessive", phonemes + " " + suffix))
                self.assertEqual(forms[(name + "'s").casefold()][1], phonemes + " " + suffix)
        self.assertIn("jameses", form_phonemes("James", "JH EY1 M Z"))

    def test_occurrences_cover_case_inflections_and_continuations_but_not_longer_words(self):
        text = ("Salvatore met SALVATORE’s twin; Salvatore's coat, the Salvatores, Salvatore'll go, "
                "a Salvatore-like grin, Salvatoresque, and Sal.")
        found = [(item["word"], item["form"]) for item in occurrences(text, "Salvatore")]
        self.assertEqual(found, [("Salvatore", "base"), ("SALVATORE’s", "possessive"), ("Salvatore's", "possessive"),
                                 ("Salvatores", "plural"), ("Salvatore", "base"), ("Salvatore", "base")])
        for item in occurrences(text, "Salvatore"):
            self.assertEqual(text[item["start"]:item["end"]], item["word"])

    def test_phonemes_must_follow_cmu_stress_rules(self):
        self.assertEqual(validate_name_phonemes("S AE2 L V AH0 T AO1 R IY0"), "S AE2 L V AH0 T AO1 R IY0")
        for value in ("S AE L", "S AE0 L", "s ae1 l", "S Æ1 L", "<|phoneme_start|>S AE1 L", "S AE1  L"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_name_phonemes(value)

    def test_test_sentence_uses_the_name_several_times_with_the_books_possessive(self):
        found = occurrences("Salvatore’s map. The Salvatores.", "Salvatore")
        sentence = test_sentence("Salvatore", found)
        self.assertEqual(sentence.count("Salvatore"), 4)
        self.assertIn("Salvatore’s name", sentence)
        self.assertIn("Salvatores are here", sentence)
        controls = sentence_controls("Salvatore", sentence, "S AE2 L V AH0 T AO1 R IY0")
        self.assertEqual(len(controls), 4)
        self.assertEqual(controls[1]["phonemes"], "S AE2 L V AH0 T AO1 R IY0 Z")
        compiled = prepare_synthesis(sentence, performance={
            "version": 1, "text_sha256": text_sha256(sentence), "controls": controls})[1]
        self.assertEqual(compiled.count("<|phoneme_start|>"), 4)
        self.assertNotIn("Salvatore", compiled)


class StorageTaggingAndRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(rose_document(), self.root / "book", allow_review=True)
        self.directory = self.root / "book"
        self.bundle = load_bundle(self.directory)
        self.config = renderer_config(self.root, series_id="fixture", paragraph_synthesis=False)

    def requests(self, model):
        return CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=0)

    def tag_with(self, model, name="Rose"):
        state = tagging_state(self.directory, self.bundle["source"], name)
        tag(self.bundle, name, self.requests(model), state["pending"],
            on_batch=lambda decisions: save_decisions(self.directory, self.bundle["source"], name, decisions),
            progress=lambda message: None)
        return tagging_state(self.directory, self.bundle["source"], name)

    def controls(self, plan):
        return [(chunk["spoken_text"][control["start"]:control["end"]], control["phonemes"])
                for chapter in plan["chapters"] for chunk in chapter["chunks"]
                for control in chunk.get("expressive_performance", {}).get("controls", [])
                if control["kind"] == "pronunciation"]

    def test_draft_checkpoints_never_change_rendering_until_confirmed(self):
        before = make_plan(self.bundle, self.config)
        save_definition(self.directory, "Rose", ROSE, status="draft", source="model", respelling="SAL")
        self.assertEqual(effective_definitions(self.directory), {})
        self.assertEqual(local_draft(self.directory, "Rose")["phonemes"], ROSE)
        self.assertEqual(make_plan(self.bundle, self.config), before)
        save_definition(self.directory, "Rose", "R OW1 Z", status="confirmed", source="manual")
        save_definition(self.directory, "Rose", ROSE, status="draft", source="model_revision", instruction="like Sal")
        self.assertEqual(effective_definitions(self.directory)["rose"]["phonemes"], "R OW1 Z")
        record = load_local(self.directory)
        self.assertEqual([step["source"] for step in record["drafts"]["rose"]["history"]],
                         ["model", "manual", "model_revision"])

    def test_tagging_excludes_homonyms_and_render_tags_only_name_uses(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        with self.assertRaisesRegex(ValueError, "tagged before rendering"):
            make_plan(self.bundle, self.config)
        model = FakeModel(decide=homonym)
        state = self.tag_with(model)
        self.assertTrue(state["ready"])
        prompt = model.payloads[0]
        self.assertEqual(prompt["mode"], "name_use_tagging")
        self.assertEqual(prompt["characters"][0]["name"], "Rose")
        self.assertIn("a ⟦rose⟧ on", prompt["occurrences"][1]["text"])
        decisions = {(item["block_id"], item["word"]): item["decision"] for item in state["decided"]}
        self.assertEqual(decisions[("b2", "rose")], "other")
        self.assertEqual(self.controls(make_plan(self.bundle, self.config)),
                         [("Rose", ROSE), ("Rose’s", ROSE + " Z"), ("ROSE", ROSE), ("Roses", ROSE + " Z")])
        chunk = next(chunk for chunk in make_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                     if "a rose" in chunk["spoken_text"])
        compiled = prepare_synthesis(chunk["spoken_text"], performance=chunk["expressive_performance"])[1]
        self.assertIn("a rose on <|phoneme_start|>S AE1 L Z<|phoneme_end|> desk", compiled)
        self.assertNotIn("[", compiled.replace("<|phoneme_start|>", "").replace("<|phoneme_end|>", ""))

    def test_unsure_blocks_render_until_a_person_decides(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        state = self.tag_with(FakeModel(decide=lambda text: "unsure" if "ROSE" in text.split("⟦")[1][:5] else "name"))
        self.assertEqual([item["word"] for item in state["unsure"]], ["ROSE"])
        with self.assertRaisesRegex(ValueError, "1 unsure"):
            render_snapshot(self.directory, self.bundle["source"])
        item = state["unsure"][0]
        save_decisions(self.directory, self.bundle["source"], "Rose", [
            {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
            | {"decision": "other", "basis": "user"}])
        self.assertNotIn("ROSE", [word for word, _ in self.controls(make_plan(self.bundle, self.config))])
        with self.assertRaises(ValueError):
            save_decisions(self.directory, self.bundle["source"], "Rose", [
                {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
                | {"decision": "unsure", "basis": "user"}])

    def test_tagging_resumes_after_a_failed_batch_and_retags_only_changed_blocks(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        from epub_tts_narrator import name_pronunciation_model
        original = name_pronunciation_model.BATCH_OCCURRENCES
        name_pronunciation_model.BATCH_OCCURRENCES = 2
        self.addCleanup(setattr, name_pronunciation_model, "BATCH_OCCURRENCES", original)
        with self.assertRaisesRegex(RuntimeError, "provider stopped"):
            self.tag_with(FakeModel(decide=homonym, fail_after=1))
        state = tagging_state(self.directory, self.bundle["source"], "Rose")
        self.assertEqual((len(state["decided"]), len(state["pending"])), (2, 3))
        resumed = FakeModel(decide=homonym)
        self.assertTrue(self.tag_with(resumed)["ready"])
        self.assertEqual(sum(len(payload["occurrences"]) for payload in resumed.payloads), 3)
        changed = deepcopy(self.bundle["source"])
        changed["blocks"][1]["text"] = changed["blocks"][1]["text"].replace("laughed", "smiled")
        state = tagging_state(self.directory, changed, "Rose")
        self.assertEqual({item["block_id"] for item in state["pending"]}, {"b2"})
        self.assertEqual({item["block_id"] for item in state["decided"]}, {"b1"})

    def test_guess_sends_context_as_data_and_a_revision_never_repeats_a_heard_reading(self):
        found = book_occurrences(self.bundle["source"], "Rose")
        model = FakeModel(guesses=[{"phonemes": "r ow1 z", "respelling": "ROHZ"},
                                   {"phonemes": "R OW1 Z"}, {"phonemes": "R OW1 Z IY0", "respelling": "ROH-zee"}])
        requests = self.requests(model)
        first = guess(self.bundle, "Rose", requests, found)
        self.assertEqual(first, {"phonemes": "R OW1 Z", "respelling": "ROHZ"})
        self.assertIn("⟦Rose⟧ said softly", model.payloads[0]["excerpts"][0])
        self.assertEqual(model.payloads[0]["dictionary_readings"], ["R OW1 Z"])
        second = guess(self.bundle, "Rose", CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=1),
                       found, current="R OW1 Z", instruction="like ROH-zee", rejected=["R OW1 Z"])
        self.assertEqual(second["phonemes"], "R OW1 Z IY0")
        self.assertEqual(model.payloads[1]["listener_instruction"], "like ROH-zee")
        self.assertIn("already heard", json.dumps(model.payloads[2]["correction"]))

    def test_a_record_for_another_book_or_symlink_is_rejected(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        path = self.directory / FILENAME
        record = json.loads(path.read_text())
        record["book_id"] = "another-book"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "different book"):
            load_local(self.directory)
        target = self.root / "elsewhere.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular file"):
            load_local(self.directory)


class SeriesInheritanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def book(self, name, previous=None):
        directory = self.root / name
        document = fixture(sequel=previous is not None)
        if previous is None:
            registry = fresh_registry(directory)
        else:
            document["source"]["book"].update(id=name, title=name)
            text = "Mara’s lamp lit the bay while Mara watched."
            document["source"]["blocks"][1]["text"] = text
            document["segments"][1].update(text=text, end=len(text))
            registry = continuation_registry(previous, document["source"])
            prior = json.loads((previous / "analysis.json").read_text())
            propose(document, prior["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle, continue_from=previous)
        return directory

    def test_a_continued_book_inherits_the_choice_but_tags_its_own_uses(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        bundle = load_bundle(second)
        inherited = effective_definitions(second)["mara"]
        self.assertTrue(inherited["inherited"])
        self.assertEqual(inherited["origin_directory"], str(first))
        config = renderer_config(paragraph_synthesis=False)
        with self.assertRaisesRegex(ValueError, r"Mara \(inherited\): 3 untagged"):
            make_plan(bundle, config)
        model = FakeModel()
        state = tagging_state(second, bundle["source"], "Mara")
        tag(bundle, "Mara", CachedRequests(model, second / ".analysis-cache", repair_attempts=0), state["pending"],
            on_batch=lambda decisions: save_decisions(second, bundle["source"], "Mara", decisions),
            progress=lambda message: None)
        phones = [control["phonemes"] for chapter in make_plan(bundle, config)["chapters"]
                  for chunk in chapter["chunks"] for control in chunk.get("expressive_performance", {}).get("controls", [])]
        self.assertEqual(phones, ["M AA1 R AH0", "M AA1 R AH0 Z", "M AA1 R AH0"])
        self.assertNotIn("mara", load_local(first)["mentions"])
        save_definition(second, "Mara", "M EH1 R AH0", status="confirmed", source="manual")
        self.assertEqual(effective_definitions(second)["mara"]["phonemes"], "M EH1 R AH0")
        self.assertEqual(effective_definitions(first)["mara"]["phonemes"], "M AA1 R AH0")


@unittest.skipUnless(HAS_FFMPEG, "FFmpeg and FFprobe are required for listening previews")
class InteractiveToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(EXAMPLE, self.root / "book")
        self.bundle = load_bundle(self.root / "book")
        self.directory = Path(self.bundle["directory"])
        self.config = renderer_config(self.root, series_id="name-pronunciation")
        self.config["engine"]["type"] = "fixture"
        self.config["engine"]["options"] = {"api_key": "DO_NOT_ECHO"}
        self.engine = PreviewEngine()
        self.factory = Mock(return_value=self.engine)
        self.store = voice_store(self.bundle, resolve_book_config(self.bundle, self.config))
        reference = self.root / "reference.wav"
        write_tone(reference)
        for profile in self.bundle["cast"]["voices"]:
            if self.store.load(profile["id"]) is None:
                self.store.publish(profile["id"], profile, reference, "An existing character voice.",
                                   self.engine.identity, {"passed": True}, {})
        self.output, self.played = [], []
        self.player = Mock()

    def run_tool(self, commands, model, **arguments):
        answers = iter(commands)
        def answer(prompt):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None
        args = SimpleNamespace(name="Mara", phonemes=None, voice_id=None, tag_only=False, review_tags=False,
                               list=False)
        for key, value in arguments.items():
            setattr(args, key, value)
        return run_pronounce_name(
            self.bundle, self.config, args, analysis_config=lambda: {"analysis": {"attribution_workers": 1}},
            input_fn=answer, output=self.output.append, engine_factory=self.factory, store=self.store,
            player=self.player, playback_fn=lambda player, path: self.played.append(path) or True,
            requests=CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=0),
            interactive=True)

    def test_guess_revise_edit_and_accept_then_tag(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0", "respelling": "MAR-uh"},
                                   {"phonemes": "M AA1 R AH0", "respelling": "MAH-ruh"}])
        self.assertEqual(self.run_tool(["I more like MAH-ruh", "B", "E m aa1 r ah0 z", "A"], model), 0)
        texts = [call["text"] for call in self.engine.calls]
        self.assertTrue(all(text == "The name is Mara. It is Mara’s name. Give it to Mara." or
                            text.startswith("The name is Mara. It is Mara") for text in texts))
        compiled = [call["compiled"] for call in self.engine.calls]
        self.assertIn("<|phoneme_start|>M AE1 R AH0<|phoneme_end|>", compiled[0])
        self.assertIn("<|phoneme_start|>M AA1 R AH0 Z<|phoneme_end|>", compiled[1])
        self.assertEqual(compiled[2].count("<|phoneme_start|>"), 0)  # B: the default reading
        self.assertIn("<|phoneme_start|>M AA1 R AH0 Z IH0 Z<|phoneme_end|>", compiled[3])
        self.assertEqual(len(self.played), 4)
        self.assertTrue(all(Path(path).is_file() and "name-pronunciation-test" in Path(path).name
                            for path in self.played))
        definition = effective_definitions(self.directory)["mara"]
        self.assertEqual(definition["phonemes"], "M AA1 R AH0 Z")
        self.assertEqual([step["source"] for step in definition["history"]], ["model", "model_revision", "manual"])
        self.assertEqual(definition["history"][1]["instruction"], "more like MAH-ruh")
        revision = next(payload for payload in model.payloads if payload.get("listener_instruction"))
        self.assertEqual(revision["current"], "M AE1 R AH0")
        self.assertNotIn("rejected", revision)  # The heard reading is "current", excluded from the reply.
        self.assertTrue(tagging_state(self.directory, self.bundle["source"], "Mara")["ready"])
        self.assertFalse(any("DO_NOT_ECHO" in line for line in self.output))

    def test_quit_keeps_the_draft_and_rerun_resumes_without_a_new_guess(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["Q"], model), 0)
        self.assertEqual(effective_definitions(self.directory), {})
        self.assertEqual(local_draft(self.directory, "Mara")["phonemes"], "M AE1 R AH0")
        again = FakeModel()
        self.assertEqual(self.run_tool(["A"], again), 0)
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")
        self.assertEqual([payload["mode"] for payload in again.payloads], ["name_use_tagging"])
        self.assertIsNone(local_draft(self.directory, "Mara"))

    def test_invalid_manual_phonemes_are_refused_and_the_loop_continues(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["E M AE R AH", "A"], model), 0)
        self.assertTrue(any("Action stopped" in line and "stress" in line for line in self.output))
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")

    def test_without_a_name_tags_every_untagged_definition_and_lists_status(self):
        save_definition(self.directory, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        model = FakeModel()
        self.assertEqual(self.run_tool([], model, name=None), 0)
        self.assertEqual(self.engine.calls, [])
        self.output.clear()
        self.assertEqual(self.run_tool([], FakeModel(), name=None, list=True), 0)
        self.assertTrue(any("Mara: M AA1 R AH0 (this book); 1 tagged uses; tagged" in line for line in self.output))


if __name__ == "__main__":
    unittest.main()
