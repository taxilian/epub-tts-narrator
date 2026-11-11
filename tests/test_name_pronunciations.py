"""User-defined name pronunciations: choose by ear, tag real uses, render tags.

Model and synthesis boundaries are fakes: no network or model calls.
"""

import contextlib
import re
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.name_pronunciation_model import guess, label
from epub_tts_narrator import name_pronunciations
from epub_tts_narrator.name_pronunciations import (
    FILENAME, PARTS_OF_SPEECH, book_occurrences, default_parts, describe_parts, effective_definitions, form_phonemes,
    load_local, local_draft, normalize_phonemes, occurrences, parse_parts, render_snapshot, save_decisions,
    save_definition, tagging_state, validate_name_phonemes,
)
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.engines import EngineError, prepare_synthesis
from epub_tts_narrator.renderer.name_pronunciation_tool import (book_sentences, run_pronounce_name, sentence_controls,
                                                                test_sentence, with_plural_audition)
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_renderer_pronunciation_review import EXAMPLE, HAS_FFMPEG, PreviewEngine
from test_renderer_workflow import write_tone
from test_series import fixture, propose


ROSE = "S AE1 L"  # Any valid reading; the tests check where it goes, not how it sounds.


class FakeModel:
    """Stands in for the provider: scripted guesses, rule-based part-of-speech labels."""
    identity = {"model": "name-pronunciation-fixture"}
    model, base_url = "fixture-model", "http://fixture.invalid"

    def __init__(self, guesses=(), decide=None, fail_after=None):
        self.guesses = list(guesses)
        self.decide = decide or (lambda text: "proper_noun")
        self.fail_after = fail_after
        self.payloads = []

    def complete(self, system, payload):
        self.payloads.append(deepcopy(payload))
        if payload["mode"] == "name_pronunciation_guess":
            return self.guesses.pop(0)
        labelling = [item for item in self.payloads if item["mode"] == "word_part_of_speech"]
        if self.fail_after is not None and len(labelling) > self.fail_after:
            raise RuntimeError("provider stopped")
        return {"labels": [self.decide(item["text"]) for item in payload["occurrences"]]}


def homonym(text):
    """Fake judgement: the lowercase flower is a common noun; everything else a name."""
    return "common_noun" if "⟦rose⟧" in text else "proper_noun"


def rose_document(first="Rose said softly, “Hello.”",
                  second="They left a rose on Rose’s desk, and ROSE laughed with the Roses.", name="Rose"):
    document = valid_document()
    block1, block2 = document["source"]["blocks"]
    block1["text"], block2["text"] = first, second
    quote = first.index("“")
    s1, s2, s3 = document["segments"]
    s1.update(text=first[:quote], end=quote)
    s2.update(text=first[quote:], start=quote, end=len(first))
    s3.update(text=second, end=len(second))
    document["cast"]["characters"][0].update(name=name, evidence=[{"block_id": "b1", "quote": name}])
    return document


YARRAN = "Y AA1 R AH0 N"


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

    def test_a_hyphenated_name_matches_whole_never_by_its_parts(self):
        text = "King Cho-Hag's horse; Cho‐Hag rode, two Cho-Hags, Cho-Hag'll go, a Cho-Hag-like grin. Cho alone, Hag too."
        found = [(item["word"], item["form"]) for item in occurrences(text, "Cho-Hag")]
        self.assertEqual(found, [("Cho-Hag's", "possessive"), ("Cho‐Hag", "base"), ("Cho-Hags", "plural"),
                                 ("Cho-Hag", "base"), ("Cho-Hag", "base")])
        for item in occurrences(text, "Cho-Hag"):
            self.assertEqual(text[item["start"]:item["end"]], item["word"])
        # Its parts are not uses of it, and a part defined alone still matches only alone.
        self.assertEqual([item["word"] for item in occurrences(text, "Cho")], ["Cho", "Cho", "Cho", "Cho", "Cho", "Cho"])

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

    def test_test_sentence_always_auditions_the_plural_even_without_a_book_plural(self):
        found = occurrences("Salvatore met Sal.", "Salvatore")
        sentence = test_sentence("Salvatore", found)
        self.assertIn("Salvatores are here", sentence)
        controls = sentence_controls("Salvatore", sentence, "S AE2 L V AH0 T AO1 R IY0")
        self.assertEqual([control["phonemes"] for control in controls if control["text"] == "Salvatores"],
                         ["S AE2 L V AH0 T AO1 R IY0 Z"])

    def test_auditions_append_the_derived_plural_when_the_sentence_lacks_one(self):
        sentence, spans = with_plural_audition("Mara", "My name is Mara.", [(11, 15, "Mara")])
        self.assertEqual(sentence, "My name is Mara. The Maras are here.")
        self.assertEqual(spans, [(11, 15, "Mara"), (21, 26, "Maras")])
        controls = sentence_controls("Mara", sentence, "M AA1 R AH0", spans)
        self.assertEqual([control["phonemes"] for control in controls], ["M AA1 R AH0", "M AA1 R AH0 Z"])
        # A sentence that already auditions the plural is kept verbatim.
        self.assertEqual(with_plural_audition("Mara", "The Maras are here.", None), ("The Maras are here.", None))


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

    def tag_with(self, model, name="Rose", parts=None):
        state = tagging_state(self.directory, self.bundle["source"], name, parts)
        label(self.bundle, name, self.requests(model), state["pending"],
              on_batch=lambda decisions: save_decisions(self.directory, self.bundle["source"], name, decisions),
              progress=lambda message: None)
        return tagging_state(self.directory, self.bundle["source"], name, parts)

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
        self.assertEqual(prompt["mode"], "word_part_of_speech")
        self.assertEqual(prompt["characters"][0]["name"], "Rose")
        self.assertIn("a ⟦rose⟧ on", prompt["occurrences"][1]["text"])
        decisions = {(item["block_id"], item["word"]): (item["use"], item["label"]) for item in state["decided"]}
        self.assertEqual(decisions[("b2", "rose")], (False, "common_noun"))
        self.assertEqual(state["parts_of_speech"], ["proper_noun"])  # "Rose" is also an ordinary word.
        self.assertEqual(self.controls(make_plan(self.bundle, self.config)),
                         [("Rose", ROSE), ("Rose’s", ROSE + " Z"), ("ROSE", ROSE), ("Roses", ROSE + " Z")])
        chunk = next(chunk for chunk in make_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                     if "a rose" in chunk["spoken_text"])
        compiled = prepare_synthesis(chunk["spoken_text"], performance=chunk["expressive_performance"])[1]
        self.assertIn("a rose on <|phoneme_start|>S AE1 L Z<|phoneme_end|> desk", compiled)
        self.assertNotIn("[", compiled.replace("<|phoneme_start|>", "").replace("<|phoneme_end|>", ""))

    def test_unsure_blocks_render_until_a_person_decides(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        state = self.tag_with(FakeModel(decide=lambda text: "unsure" if "ROSE" in text.split("⟦")[1][:5] else "proper_noun"))
        self.assertEqual([item["word"] for item in state["unsure"]], ["ROSE"])
        with self.assertRaisesRegex(ValueError, "1 unsure"):
            render_snapshot(self.directory, self.bundle["source"])
        item = state["unsure"][0]
        save_decisions(self.directory, self.bundle["source"], "Rose", [
            {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
            | {"use": False, "basis": "user"}])
        self.assertNotIn("ROSE", [word for word, _ in self.controls(make_plan(self.bundle, self.config))])
        saved = next(item for item in load_local(self.directory)["mentions"]["rose"] if item["word"] == "ROSE")
        self.assertEqual((saved["use"], saved["label"]), (False, "unsure"))  # The model's label stays for display.
        with self.assertRaises(ValueError):
            save_decisions(self.directory, self.bundle["source"], "Rose", [
                {key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
                | {"use": "no", "basis": "user"}])

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

    def test_parts_of_speech_parse_and_default_by_dictionary_membership(self):
        self.assertEqual(parse_parts("Proper nouns, adjective"), ["proper_noun", "adjective"])
        self.assertEqual(parse_parts("name and adj"), ["proper_noun", "adjective"])
        self.assertEqual(parse_parts("common-noun/verb"), ["common_noun", "verb"])
        self.assertEqual(parse_parts("all"), list(PARTS_OF_SPEECH))
        with self.assertRaisesRegex(ValueError, "Unknown part of speech"):
            parse_parts("pronoun")
        self.assertEqual(default_parts("Rose"), ["proper_noun"])  # An ordinary word shares the spelling.
        self.assertEqual(default_parts("Yarran"), list(PARTS_OF_SPEECH))
        self.assertEqual(describe_parts(["proper_noun", "adjective"]), "proper noun, adjective")

    def yarran_book(self):
        directory = self.root / "yarran"
        export_bundle(rose_document("The Yarran said softly, “Hello.”",
                                    "They drank Yarran beer with the Yarrans, and YARRAN songs rang.", name="Yarran"),
                      directory, allow_review=True)
        return directory, load_bundle(directory)

    def test_an_invented_word_tags_every_use_without_asking_the_model(self):
        directory, bundle = self.yarran_book()
        save_definition(directory, "Yarran", YARRAN, status="confirmed", source="manual")
        state = tagging_state(directory, bundle["source"], "Yarran")
        self.assertEqual((state["ready"], state["labels_needed"], {item["basis"] for item in state["decided"]}),
                         (True, False, {"rule"}))
        self.assertEqual(self.controls(make_plan(bundle, self.config)),
                         [("Yarran", YARRAN), ("Yarran", YARRAN), ("Yarrans", YARRAN + " Z"), ("YARRAN", YARRAN)])

    def test_restricted_parts_are_labelled_once_and_changing_them_needs_no_requests(self):
        directory, bundle = self.yarran_book()
        save_definition(directory, "Yarran", YARRAN, status="confirmed", source="manual", parts_of_speech=["adjective"])
        with self.assertRaisesRegex(ValueError, "4 untagged"):
            make_plan(bundle, self.config)
        model = FakeModel(decide=lambda text: "adjective" if re.search(r"⟦\w+⟧ (beer|songs)", text) else "proper_noun")
        state = tagging_state(directory, bundle["source"], "Yarran")
        label(bundle, "Yarran", CachedRequests(model, directory / ".analysis-cache", repair_attempts=0), state["pending"],
              on_batch=lambda decisions: save_decisions(directory, bundle["source"], "Yarran", decisions),
              progress=lambda message: None)
        self.assertEqual(self.controls(make_plan(bundle, self.config)), [("Yarran", YARRAN), ("YARRAN", YARRAN)])
        requests = len(model.payloads)
        save_definition(directory, "Yarran", YARRAN, status="confirmed", source="manual",
                        parts_of_speech=["proper_noun", "adjective"])
        self.assertEqual(len(self.controls(make_plan(bundle, self.config))), 4)
        self.assertEqual(len(model.payloads), requests)

    def test_outdated_model_labels_are_relabelled_but_a_persons_decisions_are_kept(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        state = self.tag_with(FakeModel(decide=homonym))
        first = state["decided"][0]
        save_decisions(self.directory, self.bundle["source"], "Rose", [
            {key: first[key] for key in ("block_id", "start", "end", "word", "block_sha256")} | {"use": True, "basis": "user"}])
        original = name_pronunciations.LABEL_VERSION
        name_pronunciations.LABEL_VERSION = original + 1
        self.addCleanup(setattr, name_pronunciations, "LABEL_VERSION", original)
        state = tagging_state(self.directory, self.bundle["source"], "Rose")
        self.assertEqual(len(state["pending"]), 4)
        self.assertEqual([(item["word"], item["basis"]) for item in state["decided"]], [(first["word"], "user")])

    def test_a_version_1_record_keeps_a_persons_decisions_and_drops_model_verdicts(self):
        occurrence = book_occurrences(self.bundle["source"], "Rose")
        spans = [{key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")} for item in occurrence]
        definition = {"name": "Rose", "phonemes": ROSE, "respelling": None,
                      "history": [{"phonemes": ROSE, "source": "manual"}]}
        (self.directory / FILENAME).write_text(json.dumps({
            "format": "epub-tts-name-pronunciations", "version": 1, "book_id": self.bundle["source"]["book"]["id"],
            "definitions": {"rose": definition}, "drafts": {},
            "mentions": {"rose": [dict(spans[0], decision="name", basis="model"),
                                  dict(spans[1], decision="other", basis="user")]}}))
        record = load_local(self.directory)
        self.assertEqual(record["version"], 2)
        self.assertEqual(record["definitions"]["rose"]["parts_of_speech"], ["proper_noun"])
        self.assertEqual(record["mentions"]["rose"], [dict(spans[1], basis="user", use=False)])
        state = tagging_state(self.directory, self.bundle["source"], "Rose")
        self.assertEqual((len(state["decided"]), len(state["pending"])), (1, 4))

    def test_book_sentences_use_only_sentences_whose_every_use_gets_the_reading(self):
        save_definition(self.directory, "Rose", ROSE, status="confirmed", source="manual")
        state = self.tag_with(FakeModel(decide=homonym))
        text, spans = book_sentences(self.bundle["source"], "Rose", state)
        self.assertEqual(text, "Rose said softly, “Hello.”")  # The other sentence also has "a rose".
        self.assertEqual(spans, [(0, 4, "Rose")])

    def milekey_book(self):
        directory = self.root / "milekey"
        export_bundle(rose_document("Milekey said softly, “Hello.”", "They saw Milekey’s map, and MILEKEY waved.",
                                    name="Milekey"), directory, allow_review=True)
        return directory, load_bundle(directory)

    def test_grouped_phonemes_reach_the_engine_as_consecutive_tags(self):
        self.assertEqual(normalize_phonemes("m ay1 l|k iy1"), "M AY1 L | K IY1")
        directory, bundle = self.milekey_book()
        save_definition(directory, "Milekey", normalize_phonemes("M AY1 L | K IY1"), status="confirmed", source="manual")
        plan = make_plan(bundle, self.config)
        self.assertEqual(self.controls(plan), [("Milekey", "M AY1 L | K IY1"), ("Milekey’s", "M AY1 L | K IY1 Z"),
                                               ("MILEKEY", "M AY1 L | K IY1")])
        chunk = next(chunk for chapter in plan["chapters"] for chunk in chapter["chunks"] if "map" in chunk["spoken_text"])
        compiled = prepare_synthesis(chunk["spoken_text"], performance=chunk["expressive_performance"])[1]
        self.assertIn("saw <|phoneme_start|>M AY1 L<|phoneme_end|> <|phoneme_start|>K IY1 Z<|phoneme_end|> map",
                      compiled)
        with self.assertRaisesRegex(ValueError, "separate groups"):
            validate_name_phonemes("M AY1 L || K IY1")

    def test_a_hyphenated_name_gets_one_reading_over_the_whole_name(self):
        directory = self.root / "cho-hag"
        export_bundle(rose_document("Cho-Hag said softly, “Hello.”", "They saw Cho‐Hag’s horse, and Cho waited.",
                                    name="Cho-Hag"), directory, allow_review=True)
        bundle = load_bundle(directory)
        save_definition(directory, "Cho-Hag", "CH OW1 HH AE2 G", status="confirmed", source="manual")
        plan = make_plan(bundle, self.config)
        self.assertEqual(self.controls(plan), [("Cho-Hag", "CH OW1 HH AE2 G"), ("Cho‐Hag’s", "CH OW1 HH AE2 G Z")])

    def test_a_say_as_spelling_is_spoken_in_place_of_each_use(self):
        directory, bundle = self.milekey_book()
        save_definition(directory, "Milekey", None, say_as="mile-key", status="confirmed", source="manual")
        plan = make_plan(bundle, self.config)
        compiled = []
        for chapter in plan["chapters"]:
            for chunk in chapter["chunks"]:
                if chunk.get("expressive_performance"):
                    compiled.append(prepare_synthesis(chunk["spoken_text"], performance=chunk["expressive_performance"])[1])
        text = " ".join(compiled)
        self.assertIn("mile-key said softly", text)
        self.assertIn("saw mile-key’s map, and MILE-KEY waved", text)
        self.assertNotIn("phoneme_start", text)
        with self.assertRaises(ValueError):
            save_definition(directory, "Milekey", "M AY1 L", say_as="mile-key", status="confirmed", source="manual")

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

    def book(self, name, previous=None, sequel_text="Mara’s lamp lit the bay while Mara watched."):
        directory = self.root / name
        document = fixture(sequel=previous is not None)
        if previous is None:
            registry = fresh_registry(directory)
        else:
            document["source"]["book"].update(id=name, title=name)
            text = sequel_text
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
        label(bundle, "Mara", CachedRequests(model, second / ".analysis-cache", repair_attempts=0), state["pending"],
              on_batch=lambda decisions: save_decisions(second, bundle["source"], "Mara", decisions),
              progress=lambda message: None)
        phones = [control["phonemes"] for chapter in make_plan(bundle, config)["chapters"]
                  for chunk in chapter["chunks"] for control in chunk.get("expressive_performance", {}).get("controls", [])]
        self.assertEqual(phones, ["M AA1 R AH0", "M AA1 R AH0 Z", "M AA1 R AH0"])
        self.assertNotIn("mara", load_local(first)["mentions"])
        save_definition(second, "Mara", "M EH1 R AH0", status="confirmed", source="manual")
        self.assertEqual(effective_definitions(second)["mara"]["phonemes"], "M EH1 R AH0")
        self.assertEqual(effective_definitions(first)["mara"]["phonemes"], "M AA1 R AH0")

    def test_a_plural_first_used_in_a_later_book_inherits_the_saved_reading(self):
        first = self.book("first")  # Only the singular occurs here.
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first, sequel_text="Mara’s lamp lit the bay while the Maras watched.")
        inherited = effective_definitions(second)["mara"]
        self.assertTrue(inherited["inherited"])
        self.assertEqual(inherited["phonemes"], "M AA1 R AH0")
        model = FakeModel()
        status, output, _ = self.analyze_again(second, model)
        self.assertEqual(status, 0, output)
        self.assertEqual({payload["mode"] for payload in model.payloads}, {"word_part_of_speech"})  # No new guess.
        self.assertIn("Pronunciation of Mara: every use is decided (3 use(s) get the reading)", output)
        words = [(chunk["spoken_text"][control["start"]:control["end"]], control["phonemes"])
                 for chapter in make_plan(load_bundle(second), renderer_config(paragraph_synthesis=False))["chapters"]
                 for chunk in chapter["chunks"] for control in chunk.get("expressive_performance", {}).get("controls", [])
                 if control["kind"] == "pronunciation"]
        self.assertEqual(words, [("Mara", "M AA1 R AH0"), ("Mara’s", "M AA1 R AH0 Z"), ("Maras", "M AA1 R AH0 Z")])

    def remove(self, directory, name="Mara"):
        output = []
        args = SimpleNamespace(name=name, remove=True, sound=None, phonemes=None, tag_only=False,
                               review_tags=False, list=False)
        status = run_pronounce_name(load_bundle(directory), {}, args, analysis_config=None, output=output.append)
        return status, "\n".join(output)

    def plan_phonemes(self, directory):
        return [control["phonemes"] for chapter in make_plan(load_bundle(directory), renderer_config(paragraph_synthesis=False))["chapters"]
                for chunk in chapter["chunks"] for control in chunk.get("expressive_performance", {}).get("controls", [])]

    def test_removing_a_choice_made_in_the_wrong_book_stops_its_inheritance(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        self.analyze_again(second, FakeModel())
        self.assertEqual(len(self.plan_phonemes(second)), 3)
        with self.assertRaisesRegex(ValueError, f"inherited from {first}, so remove it there"):
            self.remove(second)
        status, output = self.remove(first)
        self.assertEqual(status, 0)
        self.assertIn('--phonemes "M AA1 R AH0"', output)
        self.assertIn("default reading", output)
        self.assertEqual(effective_definitions(second), {})
        self.assertEqual(self.plan_phonemes(second), [])
        # Defined again in the right book, the saved tagging is reused without new requests.
        save_definition(second, "Mara", "M EH1 R AH0", status="confirmed", source="manual")
        again = FakeModel()
        self.assertEqual(self.analyze_again(second, again)[0], 0)
        self.assertEqual(again.payloads, [])
        self.assertEqual(self.plan_phonemes(second), ["M EH1 R AH0", "M EH1 R AH0 Z", "M EH1 R AH0"])

    def test_removing_a_local_override_falls_back_to_the_inherited_choice(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        save_definition(second, "Mara", "M EH1 R AH0", status="confirmed", source="manual")
        save_definition(second, "Mara", "M EY1 R AH0", status="draft", source="manual")
        status, output = self.remove(second)
        self.assertEqual(status, 0)
        self.assertIn(f"now inherits Mara (M AA1 R AH0) from {first}", output)
        self.assertIsNone(local_draft(second, "Mara"))
        self.assertEqual(effective_definitions(second)["mara"]["phonemes"], "M AA1 R AH0")
        with self.assertRaisesRegex(ValueError, "no pronunciation defined in this book"):
            self.remove(first, "Lamp")

    def analyze_again(self, directory, model):
        config = self.root / "analysis.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="http://offline.invalid/v1"\n'
                          '[analysis]\nattribution_workers=1\nrepair_attempts=0\n')
        output, error = StringIO(), StringIO()
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(error),
              patch("epub_tts_narrator.cli.analyze", side_effect=AssertionError("Must reuse saved analysis")),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=model) as provider):
            status = main(["analyze", str(directory), "--config", str(config)])
        return status, output.getvalue() + error.getvalue(), provider

    def test_rerunning_analysis_of_a_continued_book_tags_every_inherited_name(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        model = FakeModel(decide=lambda text: "common_noun" if "⟦Mara⟧ watched" in text else "proper_noun")
        status, output, _ = self.analyze_again(second, model)
        self.assertEqual(status, 0, output)
        self.assertEqual({payload["mode"] for payload in model.payloads}, {"word_part_of_speech"})
        self.assertIn("Pronunciation of Mara: every use is decided (2 use(s) get the reading)", output)
        bundle = load_bundle(second)
        phones = [control["phonemes"] for chapter in make_plan(bundle, renderer_config(paragraph_synthesis=False))["chapters"]
                  for chunk in chapter["chunks"] for control in chunk.get("expressive_performance", {}).get("controls", [])]
        self.assertEqual(phones, ["M AA1 R AH0", "M AA1 R AH0 Z"])  # The excluded use keeps the default reading.
        again = FakeModel()
        status, _, provider = self.analyze_again(second, again)
        self.assertEqual((status, again.payloads), (0, []))  # Decided uses are never re-sent.
        provider.assert_not_called()

    def test_unsure_inherited_uses_leave_analysis_needing_a_person(self):
        first = self.book("first")
        save_definition(first, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        second = self.book("second", previous=first)
        status, output, _ = self.analyze_again(second, FakeModel(decide=lambda text: "unsure"))
        self.assertEqual(status, 2)
        self.assertIn("3 unsure use(s) need your decision before rendering", output)
        self.assertIn("audiobook-render pronounce ", output)
        with self.assertRaisesRegex(ValueError, r"Mara \(inherited\): 0 untagged, 3 unsure"):
            make_plan(load_bundle(second), renderer_config(paragraph_synthesis=False))

    def test_a_book_without_defined_names_makes_no_name_requests(self):
        first = self.book("first")
        status, _, provider = self.analyze_again(first, FakeModel())
        self.assertEqual(status, 0)
        provider.assert_not_called()


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
        args = SimpleNamespace(name="Mara", sound=None, phonemes=None, voice_id=None, tag_only=False,
                               review_tags=False, list=False)
        for key, value in arguments.items():
            setattr(args, key, value)
        return run_pronounce_name(
            self.bundle, self.config, args, analysis_config=lambda: {"analysis": {"attribution_workers": 1}},
            input_fn=answer, output=self.output.append, engine_factory=self.factory, store=self.store,
            player=self.player, playback_fn=lambda player, path: self.played.append(path) or True,
            requests=CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=0),
            interactive=True)

    def test_lowercase_command_auditions_capitalized_script_rows_and_default_reading(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0", "respelling": "MAR-uh"}])
        self.assertEqual(self.run_tool(["B", "A"], model, name="mara"), 0)
        self.assertEqual(len(self.engine.calls), 2)
        compiled = [call["compiled"] for call in self.engine.calls]
        self.assertIn("<|phoneme_start|>M AE1 R AH0<|phoneme_end|>", compiled[0])
        self.assertEqual(compiled[1].count("<|phoneme_start|>"), 0)  # B: the default reading
        voice_ids = {call["voice"]["voice_id"] for call in self.engine.calls}
        self.assertEqual(len(voice_ids), 1)
        self.assertIsNotNone(self.store.load(next(iter(voice_ids))))
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")

    def test_resuming_a_saved_lowercase_definition_auditions_the_capitalized_rows(self):
        """A finished, fully cast book resumed by its lowercase command spelling.

        The saved record is version 1 exactly as persisted before the
        parts-of-speech rework: the confirmed definition's ``name`` is the
        lowercase command spelling while every script row says ``Mara`` (the
        ``milekey`` book stored ``Milekey`` rows). The automatic tagged play
        and the default ``B`` reading must select the rows' real saved voice
        instead of claiming no cast speaker voice contains the name.
        """
        uses = book_occurrences(self.bundle["source"], "mara")
        record = {"format": "epub-tts-name-pronunciations", "version": 1,
                  "book_id": self.bundle["source"]["book"]["id"],
                  "definitions": {"mara": {"name": "mara", "phonemes": "M AE1 R AH0", "respelling": None,
                                           "history": [{"phonemes": "M AE1 R AH0", "source": "model"},
                                                       {"phonemes": "M AE1 R AH0", "source": "model_revision"}]}},
                  "drafts": {},
                  "mentions": {"mara": [{key: item[key] for key in
                                         ("block_id", "start", "end", "word", "block_sha256")}
                                       | {"decision": "name", "basis": "model"} for item in uses]}}
        (self.directory / FILENAME).write_text(json.dumps(record), encoding="utf-8")
        # The v1 model verdicts are dropped by the upgrade, so the dictionary
        # spelling is relabelled once; no guess is asked for.
        model = FakeModel(decide=lambda text: "proper_noun")
        self.assertEqual(self.run_tool(["B", "A"], model, name="mara"), 0)
        self.assertEqual([payload["mode"] for payload in model.payloads], ["word_part_of_speech"])
        self.assertTrue(any("is already defined (in this book)" in line for line in self.output))
        self.assertFalse(any("Playback stopped" in line or "No spoken line with a saved voice" in line
                             for line in self.output))
        self.assertEqual(len(self.engine.calls), 2)
        compiled = [call["compiled"] for call in self.engine.calls]
        self.assertIn("<|phoneme_start|>M AE1 R AH0<|phoneme_end|>", compiled[0])  # the saved reading
        self.assertEqual(compiled[1].count("<|phoneme_start|>"), 0)  # B: the default reading
        voice_ids = {call["voice"]["voice_id"] for call in self.engine.calls}
        self.assertEqual(len(voice_ids), 1)
        self.assertIsNotNone(self.store.load(next(iter(voice_ids))))  # a real saved voice, not a fallback
        self.assertEqual(len(self.played), 2)
        definition = effective_definitions(self.directory)["mara"]
        self.assertEqual((definition["name"], definition["phonemes"]), ("mara", "M AE1 R AH0"))

    def test_guess_revise_edit_and_accept_then_tag(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0", "respelling": "MAR-uh"},
                                   {"phonemes": "M AA1 R AH0", "respelling": "MAH-ruh"}])
        self.assertEqual(self.run_tool(["I more like MAH-ruh", "B", "E m aa1 r ah0 z", "A"], model), 0)
        texts = [call["text"] for call in self.engine.calls]
        self.assertEqual(set(texts), {"My name is Mara. The Maras are here."})  # The book's sentence plus the derived plural.
        compiled = [call["compiled"] for call in self.engine.calls]
        self.assertIn("<|phoneme_start|>M AE1 R AH0<|phoneme_end|>", compiled[0])
        self.assertEqual(compiled[1], "My name is <|phoneme_start|>M AA1 R AH0<|phoneme_end|>. "
                         "The <|phoneme_start|>M AA1 R AH0 Z<|phoneme_end|> are here.")
        self.assertEqual(compiled[2].count("<|phoneme_start|>"), 0)  # B: the default reading
        self.assertEqual(compiled[3], "My name is <|phoneme_start|>M AA1 R AH0 Z<|phoneme_end|>. "
                         "The <|phoneme_start|>M AA1 R AH0 Z IH0 Z<|phoneme_end|> are here.")
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

    def test_a_description_on_the_command_line_drives_the_first_guess_even_over_a_draft(self):
        save_definition(self.directory, "Mara", "M AE1 R AH0", status="draft", source="model")
        model = FakeModel(guesses=[{"phonemes": "M AA1 R AH0", "respelling": "MAH-ruh"}])
        self.assertEqual(self.run_tool(["A"], model, sound="  MAH-ruh,   like   the start of marathon "), 0)
        first = model.payloads[0]
        self.assertEqual(first["mode"], "name_pronunciation_guess")
        self.assertEqual(first["listener_instruction"], "MAH-ruh, like the start of marathon")
        self.assertNotIn("current", first)
        self.assertIn("<|phoneme_start|>M AA1 R AH0<|phoneme_end|>", self.engine.calls[0]["compiled"])
        definition = effective_definitions(self.directory)["mara"]
        self.assertEqual(definition["phonemes"], "M AA1 R AH0")
        self.assertIn({"phonemes": "M AA1 R AH0", "source": "model_revision", "respelling": "MAH-ruh",
                       "instruction": "MAH-ruh, like the start of marathon"}, definition["history"])

    def test_s_chooses_parts_of_speech_and_an_ordinary_word_starts_with_proper_nouns(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["S adjective, proper noun", "A"], model), 0)
        self.assertTrue(any("also an ordinary English word" in line for line in self.output))
        definition = effective_definitions(self.directory)["mara"]
        self.assertEqual(definition["parts_of_speech"], ["proper_noun", "adjective"])
        self.assertEqual([payload["mode"] for payload in model.payloads].count("word_part_of_speech"), 1)

    def test_as_on_the_command_line_sets_the_parts_before_the_first_guess(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["A"], model, parts_of_speech="common noun"), 0)
        self.assertEqual(model.payloads[0]["used_as"], "common noun")
        self.assertEqual(effective_definitions(self.directory)["mara"]["parts_of_speech"], ["common_noun"])
        self.assertTrue(any("The word is Mara." in line for line in self.output))  # No book use is a common noun.

    def test_r_says_the_word_as_another_spelling_and_e_accepts_phoneme_groups(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["R mar ah", "E m aa1 r | ah0", "R mar-ah", "A"], model), 0)
        texts = [call["compiled"] for call in self.engine.calls]
        self.assertIn("My name is mar-ah. The mar-ahs are here.", texts)
        self.assertIn("My name is <|phoneme_start|>M AA1 R<|phoneme_end|> <|phoneme_start|>AH0<|phoneme_end|>."
                      " The <|phoneme_start|>M AA1 R<|phoneme_end|> <|phoneme_start|>AH0 Z<|phoneme_end|> are here.",
                      texts)
        definition = effective_definitions(self.directory)["mara"]
        self.assertEqual(definition["say_as"], "mar-ah")
        self.assertTrue(any("said as mar-ah" in line for line in self.output))

    def test_a_description_cannot_be_combined_with_phonemes_or_tag_only(self):
        for extra in ({"phonemes": "M AA1 R AH0"}, {"tag_only": True}):
            with self.assertRaisesRegex(ValueError, "not more than one"):
                self.run_tool([], FakeModel(), sound="MAH-ruh", **extra)
        self.assertEqual(self.engine.calls, [])

    def test_quit_keeps_the_draft_and_rerun_resumes_without_a_new_guess(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["Q"], model), 0)
        self.assertEqual(effective_definitions(self.directory), {})
        self.assertEqual(local_draft(self.directory, "Mara")["phonemes"], "M AE1 R AH0")
        again = FakeModel()
        self.assertEqual(self.run_tool(["A"], again), 0)
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")
        self.assertEqual(again.payloads, [])  # Labels made before listening are reused.
        self.assertIsNone(local_draft(self.directory, "Mara"))

    def test_invalid_manual_phonemes_are_refused_and_the_loop_continues(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["E M AE R AH", "A"], model), 0)
        self.assertTrue(any("Action stopped" in line and "stress" in line for line in self.output))
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")

    def test_a_take_that_stops_before_the_end_moves_on_to_new_seeds_instead_of_repeating(self):
        stopped, synthesize = set(), self.engine.synthesize
        def unfinished(text, voice, path, *, seed, **options):
            if len(stopped) < 3:
                stopped.add(seed)
                self.engine.calls.append({"seed": seed})
                raise EngineError("Drama 3 did not finish the text: 'silence'")
            return synthesize(text, voice, path, seed=seed, **options)
        self.engine.synthesize = unfinished
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0"}])
        self.assertEqual(self.run_tool(["P", "A"], model), 0)
        seeds = [call["seed"] for call in self.engine.calls]
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(set(seeds)), 4)  # P never replays a seed that already stopped.
        self.assertTrue(any("3 takes in a row stopped before finishing" in line for line in self.output))
        self.assertEqual(len(self.played), 1)
        self.assertIn("take-004", Path(self.played[0]).name)
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AE1 R AH0")

    def test_without_a_name_tags_every_untagged_definition_and_lists_status(self):
        save_definition(self.directory, "Mara", "M AA1 R AH0", status="confirmed", source="manual")
        model = FakeModel()
        self.assertEqual(self.run_tool([], model, name=None), 0)
        self.assertEqual(self.engine.calls, [])
        self.output.clear()
        self.assertEqual(self.run_tool([], FakeModel(), name=None, list=True), 0)
        self.assertTrue(any("Mara: M AA1 R AH0 for proper noun (this book); 1 tagged uses; tagged" in line for line in self.output))


if __name__ == "__main__":
    unittest.main()
