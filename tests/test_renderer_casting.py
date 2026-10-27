"""Casting tests: verified samples become active and failures preserve old voices."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession, MAX_PROMPT_CHARACTERS
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voice_prompts import AUDIOBOOK_VOICE_CONTEXT, audiobook_voice_prompt
from epub_tts_narrator.renderer.workflow import RenderBlocked, ensure_voice, render
from epub_tts_narrator.cli import export_bundle
from audition_fixtures import prepare_bundle
from shared_fixtures import pcm_hash


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
FFMPEG = shutil.which("ffmpeg")


class CastingEngine:
    identity = {"engine": "casting_fixture", "model": "fixture-v1"}
    capabilities = {"reference_voice": True, "voice_design": True}

    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.calls = []
        self.speech_calls = []
        self.closed = False
        self.interrupt = False

    def create_voice(self, prompt, text, path, *, seed):
        self.calls.append({"prompt": prompt, "text": text, "seed": seed})
        if self.interrupt:
            raise KeyboardInterrupt()
        return self._write(text, path, seed)

    def _write(self, text, path, seed):
        rate = 44100
        frequency = 200 + seed % 3000
        samples = [round(4000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(rate)]
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack("<" + "h" * rate, *samples))
        self.transcripts[pcm_hash(path)] = text
        return {"description_applied": self.capabilities.get("voice_design", False)}

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.speech_calls.append({"voice_id": voice["voice_id"], "asset_hash": voice["asset_hash"]})
        return self._write(text, path, seed)

    def close(self):
        self.closed = True


class CastingVerifier:
    identity = {"backend": "fixture-asr", "version": 1}

    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.calls = []
        self.wrong = False

    def transcribe(self, path):
        self.calls.append(str(path))
        text = self.transcripts[pcm_hash(path)]
        if self.wrong:
            text = "Invented unrelated words."
        return {"text": text, "segments": [{
            "text": text, "start": 0.0, "end": 1.0, "avg_logprob": -0.1, "no_speech_prob": 0.001,
            "words": [{"word": word, "start": index / len(text.split()),
                       "end": (index + 1) / len(text.split()), "probability": 0.99}
                      for index, word in enumerate(text.split())],
        }]}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting acceptance tests")
class CastingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = prepare_bundle(load_bundle(EXAMPLE))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "tts-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="casting-series")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=2, ffmpeg=FFMPEG, require_cover=False,
        )
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = CastingEngine(self.transcripts)
        self.verifier = CastingVerifier(self.transcripts)
        self.engine_factories = []
        self.verifier_factories = []
        self.session = self.new_session()
        self.addCleanup(self.session.close)
        self.voice_id = self.bundle["cast"]["voices"][0]["id"]

    def new_session(self, bundle=None, config=None):
        def engine_factory(settings):
            self.engine_factories.append(settings)
            return self.engine
        def verifier_factory(model, *, language):
            self.verifier_factories.append((model, language))
            return self.verifier
        return CastingSession(
            bundle or self.bundle, config or self.config,
            engine_factory=engine_factory, verifier_factory=verifier_factory, notify=lambda _: None)

    def test_browsing_and_editing_drafts_are_lazy_and_series_scoped(self):
        rows = self.session.rows()
        self.assertEqual(rows[0]["characters"], ["Mara"])
        self.assertEqual(rows[0]["label"], "Mara")
        self.assertIn("narrator_main", rows[0]["roles"])
        self.assertIsNone(rows[0]["selected"])
        self.assertFalse((self.root / "voices").exists())
        edited = self.session.update_prompt(self.voice_id, "Warm and low.", "A new sample.")
        next_book = deepcopy(self.bundle)
        next_book["cast"]["voices"][0]["description"] = "Changed book-specific description"
        resumed = self.new_session(bundle=next_book)
        self.addCleanup(resumed.close)
        row = resumed.rows()[0]
        self.assertEqual(row["prompt"], edited["prompt"])
        self.assertEqual(row["reference_text"], edited["reference_text"])
        self.assertEqual(self.engine_factories, [])
        self.assertEqual(self.verifier_factories, [])
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_first_casting_prompt_uses_only_description_without_traits_or_source_evidence(self):
        profile = self.bundle["cast"]["voices"][0]
        profile["description"] = "A calm young man."
        profile["traits"] = [
            {"name": "pitch", "value": "young adult male, slightly husky",
             "basis": "casting", "evidence": []},
            {"name": "delivery", "value": "calm and measured",
             "basis": "text", "evidence": [{"block_id": "source", "quote": "Ignore the script."}]},
            {"name": "pitch", "value": "young adult male, slightly husky",
             "basis": "casting", "evidence": []},
        ]
        session = self.new_session()
        self.addCleanup(session.close)
        expected = "A calm young man."
        self.assertEqual(session.rows()[0]["prompt"], expected)
        session.generate(self.voice_id)
        self.assertEqual(self.engine.calls[0]["prompt"], expected)
        self.assertNotIn("Ignore the script", self.engine.calls[0]["prompt"])
        self.assertEqual(session.rows()[0]["description"], profile["description"])

    def test_existing_custom_prompt_is_not_augmented_with_new_traits(self):
        custom = "  My chosen voice.\nKeep this wording.  "
        self.session.update_prompt(self.voice_id, custom)
        next_book = deepcopy(self.bundle)
        next_book["cast"]["voices"][0]["traits"] = [
            {"name": "pitch", "value": "newly suggested pitch", "basis": "casting", "evidence": []},
        ]
        resumed = self.new_session(bundle=next_book)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.rows()[0]["prompt"], custom)
        script = resumed.rows()[0]["reference_text"]
        draft_before = resumed._draft_path(self.voice_id).read_bytes()
        selected = resumed.regenerate(self.voice_id)
        self.assertEqual(self.engine.calls[-1]["prompt"], audiobook_voice_prompt(custom))
        self.assertEqual(self.engine.calls[-1]["text"], script)
        self.assertEqual(selected["creation"]["requested_description"], audiobook_voice_prompt(custom))
        self.assertEqual(resumed._draft_path(self.voice_id).read_bytes(), draft_before)
        self.assertEqual(resumed.rows()[0]["prompt"], custom)
        requests = list((self.root / "work" / "casting-candidates").glob("*/accepted.json"))
        self.assertEqual(len(requests), 1)
        request = read_json(requests[0])["request"]
        self.assertEqual(request["prompt"], audiobook_voice_prompt(custom))
        self.assertEqual(request["text"], script)
        self.assertEqual(request["prompt"], custom)

    def test_established_creation_prompt_is_preserved_without_a_draft(self):
        custom = "  Established voice.\nKeep this wording.  "
        selected = {"creation": {"requested_description": custom},
                    "reference_text": "An established sample."}
        draft = self.session._draft(self.voice_id, selected)
        self.assertEqual(draft["prompt"], custom)

    def test_historical_draft_wrapper_is_hidden_without_rewriting_saved_draft(self):
        description = "An adult woman portraying a girl with a light, bright tone.\nKeep this  spacing."
        legacy = AUDIOBOOK_VOICE_CONTEXT + " " + description
        self.session.update_prompt(self.voice_id, legacy)
        path = self.session._draft_path(self.voice_id)
        original = path.read_bytes()
        self.assertEqual(self.session.rows()[0]["prompt"], description)
        self.assertEqual(path.read_bytes(), original)
        selected = self.session.generate(self.voice_id)
        self.assertEqual(self.engine.calls[-1]["prompt"], description)
        self.assertEqual(selected["creation"]["requested_description"], description)
        self.assertEqual(path.read_bytes(), original)

    def test_historical_selected_description_is_hidden_without_changing_provenance(self):
        description = "An adult woman portraying a girl with a light, bright tone."
        legacy = AUDIOBOOK_VOICE_CONTEXT + " " + description
        selected = {"creation": {"requested_description": legacy},
                    "reference_text": "An established sample."}
        before = deepcopy(selected)
        self.assertEqual(self.session._draft(self.voice_id, selected)["prompt"], description)
        self.assertEqual(selected, before)

    def test_missing_script_listing_removes_legacy_context_without_editing_selected_voice(self):
        from epub_tts_narrator.renderer.samples import MissingSampleScript
        description = "An adult performer with clear diction and warm resonance."
        selected = {"creation": {"requested_description": AUDIOBOOK_VOICE_CONTEXT + " " + description,
                                 "sample_source": {"kind": "book_excerpt"}},
                    "reference_text": "Too short."}
        before = deepcopy(selected)
        with patch.object(self.session.store, "load", return_value=selected), \
                patch.object(self.session, "_sample_default", side_effect=MissingSampleScript("No script")):
            row = self.session.rows()[0]
        self.assertEqual(row["prompt"], description)
        self.assertEqual(row["sample_source"]["kind"], "unprepared")
        self.assertEqual(selected, before)
        self.assertEqual(self.engine_factories, [])

    def test_overlong_default_stops_before_loading_engine_or_creating_draft(self):
        self.bundle["cast"]["voices"][0]["description"] = "A rounded resonant voice. " * 500
        session = self.new_session()
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError, "shorten the description"):
            session.generate(self.voice_id)
        self.assertEqual(self.engine_factories, [])
        self.assertEqual(self.engine.calls, [])
        self.assertFalse(self.session._draft_path(self.voice_id).exists())

    def test_complete_casting_description_reaches_designer_without_trait_list(self):
        description = "An elderly masculine voice with a harsh dry rasp and precise consonants."
        self.bundle["cast"]["voices"][0].update(description=description, traits=[
            {"name": "presentation", "value": "masculine voice", "basis": "casting", "evidence": []},
            {"name": "texture", "value": "harsh dry rasp", "basis": "casting", "evidence": []},
        ])
        session = self.new_session()
        self.addCleanup(session.close)
        script = session.rows()[0]["reference_text"]
        selected = session.generate(self.voice_id)
        expected = description
        self.assertEqual(self.engine.calls[0]["prompt"], expected)
        self.assertEqual(self.engine.calls[0]["text"], script)
        self.assertEqual(selected["creation"]["requested_description"], expected)

    def test_narration_and_noncharacter_voices_have_readable_labels(self):
        bundle = deepcopy(self.bundle)
        bundle["cast"]["voices"].extend([
            {"id": "narration_voice", "description": "A calm voice.", "traits": []},
            {"id": "afterword_voice", "description": "Author's afterword voice: Warm and friendly.", "traits": []},
            {"id": "other_narrator", "description": "Warm and friendly.", "traits": []},
        ])
        bundle["cast"]["roles"][0]["voice_id"] = "narration_voice"
        bundle["cast"]["roles"].extend([
            {"id": "afterword_role", "kind": "narration", "voice_id": "afterword_voice", "character_id": None},
            {"id": "other_narration", "kind": "narration", "voice_id": "other_narrator", "character_id": None},
        ])
        session = self.new_session(bundle=bundle)
        self.addCleanup(session.close)
        labels = {row["voice_id"]: row["label"] for row in session.rows()}
        self.assertEqual(labels["narration_voice"], "Mara (narration)")
        self.assertEqual(labels["afterword_voice"], "Author's afterword")
        self.assertEqual(labels["other_narrator"], "Narration")
        self.assertEqual(self.engine_factories, [])
        self.assertEqual(self.verifier_factories, [])

    def test_author_narration_labels_use_book_author_without_description_colon(self):
        descriptions = [
            "Author afterword and acknowledgments (ch0014) and front/back matter; nonstory text distinct from the main narrative frame.",
            "Author afterword narration by Nathan Thompson: the post-story address, acknowledgments, and promotional links; a distinct narration scope from the story frame.",
            "Author afterword and acknowledgments by Nathan Thompson; a distinct narration scope separate from the story frame.",
            "Author back-matter narration: the afterword, acknowledgments, and the Advancement stages appendix, read in the author's own voice rather than the story narrator's.",
            "Author narration for the copyright page, dedication, the Advancement Stages and Substages appendix, and the afterword; nonstory text outside Jasper's narrative frame.",
            "Author afterword narration scope for the book's closing chapter; covers the author's direct address, acknowledgments, and promotional notes. Distinct from Jasper's first-person narrative frame, which remains narrator_main.",
        ]
        for number, description in enumerate(descriptions, 1):
            with self.subTest(book=number):
                bundle = deepcopy(self.bundle)
                bundle["source"]["book"]["authors"] = ["Nathan Thompson"]
                bundle["cast"]["voices"].append({"id": "author_voice", "description":
                    "A warm, friendly authorial voice with a natural mid register and relaxed resonance.", "traits": []})
                bundle["cast"]["roles"].append({"id": "afterword", "kind": "narration", "voice_id": "author_voice",
                                                 "character_id": None, "description": description})
                if number == 5:
                    bundle["cast"]["roles"].append({"id": "editor_notes", "kind": "narration", "voice_id": "author_voice",
                        "character_id": None, "description": "Editorial margin-note commentary ([1]-[23]) accompanying the text; nonstory author/editor asides distinct from the narrative frame and from the formal afterword."})
                session = self.new_session(bundle=bundle)
                self.addCleanup(session.close)
                labels = {row["voice_id"]: row["label"] for row in session.rows()}
                self.assertEqual(labels["author_voice"], "Nathan Thompson (author)")
                self.assertEqual(labels[self.voice_id], "Mara")
        self.assertEqual(self.engine_factories, [])
        self.assertEqual(self.verifier_factories, [])

    def test_author_label_is_metadata_driven_and_retains_voice_alias_identity(self):
        bundle = deepcopy(self.bundle)
        bundle["source"]["book"]["authors"] = ["C. Mantis"]
        bundle["cast"]["voices"].append({"id": "author_voice", "description": "Author voice: Warm and friendly.", "traits": []})
        bundle["cast"]["roles"].append({"id": "afterword", "kind": "narration", "voice_id": "author_voice",
                                         "character_id": None, "description": "Author afterword and acknowledgments."})
        session = self.new_session(bundle=bundle)
        self.addCleanup(session.close)
        with patch.object(session.store, "canonical_id", side_effect=lambda voice_id: self.voice_id if voice_id == "author_voice" else voice_id):
            row = next(row for row in session.rows() if row["voice_id"] == "author_voice")
        self.assertEqual(row["label"], "C. Mantis (author)")
        self.assertEqual(row["canonical_voice_id"], self.voice_id)
        self.assertEqual(row["roles"], ["afterword"])

    def test_draft_corruption_and_wrong_voice_are_rejected(self):
        self.session.update_prompt(self.voice_id, "New voice.")
        path = self.session._draft_path(self.voice_id)
        draft = read_json(path)
        draft["prompt"] = "Changed without a new checksum"
        write_json(path, draft)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.session.rows()
        with self.assertRaisesRegex(ValueError, "not in"):
            self.session.update_prompt("another-voice", "New prompt.")

    def test_invalid_draft_inputs_have_no_side_effects(self):
        for fields in ({"prompt": ""}, {"reference_text": " "}, {"prompt": "x" * 4001},
                       {"reference_text": "x" * 1001}, {"reference_text": "hello\x00there"}):
            with self.subTest(fields=tuple(fields)):
                with self.assertRaises(ValueError):
                    self.session.update_prompt(self.voice_id, **fields)
        self.assertFalse((self.root / "voices").exists())

    def test_complete_description_budget_reaches_generation_without_trimming_custom_draft(self):
        prompt = "x" * MAX_PROMPT_CHARACTERS
        self.session.update_prompt(self.voice_id, prompt)
        before = self.session._draft_path(self.voice_id).read_bytes()
        candidate = self.session.generate(self.voice_id)
        self.assertEqual(self.engine.calls[0]["prompt"], prompt)
        self.assertEqual(candidate["creation"]["requested_description"], prompt)
        self.assertIsNone(self.session.store.load(self.voice_id))
        self.assertEqual(self.session._draft_path(self.voice_id).read_bytes(), before)
        self.assertEqual(self.session.rows()[0]["prompt"], prompt)

    def test_prompt_and_text_edits_reach_engine_without_mutating_cast_profile(self):
        original = deepcopy(self.bundle["cast"]["voices"][0])
        self.session.update_prompt(self.voice_id, "Warm and low.", "Listen to this sentence.")
        candidate = self.session.generate(self.voice_id)
        self.assertEqual(self.engine.calls[0]["prompt"], audiobook_voice_prompt("Warm and low."))
        self.assertEqual(self.engine.calls[0]["text"], "Listen to this sentence.")
        self.assertEqual(candidate["profile"], original)
        self.assertEqual(candidate["creation"]["requested_description"], audiobook_voice_prompt("Warm and low."))
        self.assertTrue(candidate["description_applied"])
        self.assertIsNone(self.session.store.load(self.voice_id))
        self.assertEqual(len(self.session.rows()[0]["candidates"]), 1)

    def test_each_generate_creates_a_new_candidate_and_seed(self):
        first = self.session.generate(self.voice_id)
        second = self.session.generate(self.voice_id)
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertNotEqual(first["creation"]["seed"], second["creation"]["seed"])
        self.assertNotEqual(first["asset_hash"], second["asset_hash"])
        self.assertEqual(len(self.session.rows()[0]["candidates"]), 2)
        self.assertIsNone(self.session.store.load(self.voice_id))

    def test_accept_replaces_only_selection_and_keeps_history(self):
        first = self.session.generate(self.voice_id)
        before_accept = len(self.verifier.calls)
        selected = self.session.accept(self.voice_id, first["candidate_id"])
        self.assertEqual(selected["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.verifier.calls), before_accept)
        second = self.session.generate(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.session.accept(self.voice_id, second["candidate_id"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], second["asset_hash"])
        self.assertEqual(self.session.store.load_candidate(self.voice_id, first["candidate_id"])["asset_hash"],
                         first["asset_hash"])
        self.session.accept(self.voice_id, first["candidate_id"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(list((self.session.store.directory / "casting-acceptance").glob("*.json"))), 3)

    def test_pending_candidate_blocks_implicit_renderer_voice_creation(self):
        self.session.generate(self.voice_id)
        with self.assertRaisesRegex(RenderBlocked, "cast command"):
            ensure_voice(self.session.store, self.voice_id, self.bundle["cast"]["voices"][0],
                         self.engine, self.verifier, self.config, self.config["render"]["work_dir"],
                         notify=lambda _: None)
        self.assertEqual(len(self.engine.calls), 1)

    def test_edit_only_draft_blocks_implicit_creation_but_keeps_an_accepted_voice(self):
        self.session.update_prompt(self.voice_id, "A carefully chosen voice.", "Please use this sample.")
        with self.assertRaisesRegex(RenderBlocked, "cast command"):
            ensure_voice(self.session.store, self.voice_id, self.bundle["cast"]["voices"][0],
                         self.engine, self.verifier, self.config, self.config["render"]["work_dir"],
                         notify=lambda _: None)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.verifier.calls, [])
        candidate = self.session.generate(self.voice_id)
        self.session.accept(self.voice_id, candidate["candidate_id"])
        self.session.update_prompt(self.voice_id, "An alternative idea for the next audition.")
        generation_count = len(self.engine.calls)
        saved = ensure_voice(
            self.session.store, self.voice_id, self.bundle["cast"]["voices"][0],
            self.engine, self.verifier, self.config, self.config["render"]["work_dir"],
            notify=lambda _: None)
        self.assertEqual(saved["asset_hash"], candidate["asset_hash"])
        self.assertEqual(len(self.engine.calls), generation_count)

    def test_generate_missing_selects_new_voices_and_skips_selected_voices(self):
        first = self.session.generate(self.voice_id)
        self.session.accept(self.voice_id, first["candidate_id"])
        generated = self.session.generate_missing()
        self.assertEqual(len(generated), 1)
        self.assertNotEqual(generated[0]["voice_id"], self.voice_id)
        self.assertEqual(self.session.store.load(generated[0]["voice_id"])["asset_hash"],
                         generated[0]["asset_hash"])
        self.assertEqual(self.session.generate_missing(), [])
        self.assertEqual(len(self.engine.calls), 2)

    def test_missing_voice_is_generated_verified_and_selected_immediately(self):
        selected = self.session.ensure_ready(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], selected["asset_hash"])
        self.assertTrue(selected["verification"]["passed"])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_ready_voice_is_reused_without_loading_models_even_with_newer_candidate(self):
        selected = self.session.ensure_ready(self.voice_id)
        newer = self.session.generate(self.voice_id)
        self.assertNotEqual(selected["asset_hash"], newer["asset_hash"])
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        with patch.object(resumed, "_get_engine", side_effect=AssertionError("Engine loaded")), \
                patch.object(resumed, "_get_verifier", side_effect=AssertionError("Verifier loaded")):
            reused = resumed.ensure_ready(self.voice_id)
        self.assertEqual(reused["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.engine.calls), 2)

    def test_existing_pending_samples_adopt_latest_without_regenerating(self):
        older = self.session.generate(self.voice_id)
        latest = self.session.generate(self.voice_id)
        self.assertNotEqual(older["asset_hash"], latest["asset_hash"])
        before = len(self.verifier.calls)
        selected = self.session.ensure_ready(self.voice_id)
        self.assertEqual(selected["asset_hash"], latest["asset_hash"])
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(len(self.verifier.calls), before)
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 2)

    def test_generate_missing_adopts_pending_and_generates_only_missing(self):
        pending = self.session.generate(self.voice_id)
        selected = self.session.generate_missing()
        self.assertEqual(len(selected), 2)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], pending["asset_hash"])
        self.assertTrue(all(row["selected"] is not None for row in self.session.rows()))
        self.assertEqual(len(self.engine.calls), 2)

    def test_regenerate_selects_replacement_and_keeps_previous_sample(self):
        first = self.session.ensure_ready(self.voice_id)
        self.session.update_prompt(self.voice_id, "Soft and measured.", "The next sample is ready.")
        replacement = self.session.regenerate(self.voice_id)
        self.assertNotEqual(replacement["asset_hash"], first["asset_hash"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.engine.calls[-1]["prompt"], audiobook_voice_prompt("Soft and measured."))
        self.assertEqual(self.engine.calls[-1]["text"], "The next sample is ready.")
        self.assertEqual(self.session.store.load_candidate(self.voice_id, first["candidate_id"])["asset_hash"],
                         first["asset_hash"])
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_failed_regeneration_keeps_current_voice_and_releases_lock(self):
        first = self.session.ensure_ready(self.voice_id)
        self.verifier.wrong = True
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.session.regenerate(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 1)
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_interrupted_regeneration_keeps_current_voice_and_releases_lock(self):
        first = self.session.ensure_ready(self.voice_id)
        self.engine.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.session.regenerate(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_failed_replacement_publication_keeps_current_voice_and_saved_alternative(self):
        first = self.session.ensure_ready(self.voice_id)
        with patch.object(self.session.store, "select_candidate",
                          side_effect=OSError("Selection publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.session.regenerate(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 2)
        self.assertFalse((self.session.store.directory / ".lock").exists())
        # Reopening the tool never replaces the already active voice with an
        # unselected candidate left by an interrupted replacement.
        self.assertEqual(self.session.ensure_ready(self.voice_id)["asset_hash"], first["asset_hash"])

    def test_startup_resumes_after_candidate_saved_but_selection_interrupted(self):
        with patch.object(self.session.store, "select_candidate", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.session.ensure_ready(self.voice_id)
        self.assertIsNone(self.session.store.load(self.voice_id))
        candidates = self.session.store.list_candidates(self.voice_id)
        self.assertEqual(len(candidates), 1)
        self.assertFalse((self.session.store.directory / ".lock").exists())
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        selected = resumed.ensure_ready(self.voice_id)
        self.assertEqual(selected["asset_hash"], candidates[0]["asset_hash"])
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)

    def test_changed_quality_or_verifier_rechecks_before_accept(self):
        candidate = self.session.generate(self.voice_id)
        before = len(self.verifier.calls)
        self.session.config["quality"]["max_clipped_fraction"] = 0.1
        self.session.accept(self.voice_id, candidate["candidate_id"])
        self.assertEqual(len(self.verifier.calls), before + 1)
        self.verifier.identity = {"backend": "fixture-asr", "version": 2}
        self.session.accept(self.voice_id, candidate["candidate_id"])
        self.assertEqual(len(self.verifier.calls), before + 2)

    def test_failed_reverification_preserves_previous_selection(self):
        first = self.session.generate(self.voice_id)
        self.session.accept(self.voice_id, first["candidate_id"])
        alternative = self.session.generate(self.voice_id)
        self.verifier.identity = {"backend": "fixture-asr", "version": 2}
        self.verifier.wrong = True
        with self.assertRaisesRegex(RenderBlocked, "current verification"):
            self.session.accept(self.voice_id, alternative["candidate_id"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_different_engine_cannot_accept_a_candidate(self):
        candidate = self.session.generate(self.voice_id)
        self.engine.identity = {"engine": "other", "model": "fixture-v1"}
        with self.assertRaisesRegex(RenderBlocked, "different engine"):
            self.session.accept(self.voice_id, candidate["candidate_id"])
        self.assertIsNone(self.session.store.load(self.voice_id))

    def test_different_actual_model_weights_cannot_accept_candidate(self):
        self.engine.identity = {**self.engine.identity, "checkpoint_files_sha256": {"weights": "old"}}
        candidate = self.session.generate(self.voice_id)
        self.engine.identity = {**self.engine.identity, "checkpoint_files_sha256": {"weights": "new"}}
        with self.assertRaisesRegex(RenderBlocked, "different model weights"):
            self.session.accept(self.voice_id, candidate["candidate_id"])

    def test_failed_generation_retains_attempts_but_never_selects_voice(self):
        self.verifier.wrong = True
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.session.generate(self.voice_id)
        self.assertEqual(self.session.rows()[0]["candidates"], [])
        self.assertIsNone(self.session.store.load(self.voice_id))
        self.assertEqual(len(list((self.root / "work" / "casting-candidates").glob("*/attempt-*.json"))), 2)
        self.assertFalse((self.session.store.directory / ".lock").exists())

    def test_interrupt_releases_lock_and_preserves_accepted_voice(self):
        first = self.session.generate(self.voice_id)
        self.session.accept(self.voice_id, first["candidate_id"])
        self.engine.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.session.generate(self.voice_id)
        self.assertFalse((self.session.store.directory / ".lock").exists())
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])

    def test_drama_style_sampling_cannot_silently_ignore_description(self):
        self.engine.capabilities = {"reference_voice": True, "voice_design": False,
                                    "unconditioned_voice_creation": True}
        self.session.update_prompt(self.voice_id, "Warm and low.")
        with self.assertRaisesRegex(RenderBlocked, "voice designer"):
            self.session.generate(self.voice_id)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.session.store.list_candidates(self.voice_id), [])

    def test_series_mismatch_and_path_overlap_rejected_before_side_effects(self):
        bundle = deepcopy(self.bundle)
        bundle["manifest"]["provenance"]["series"] = {"series_id": "another-series"}
        with self.assertRaisesRegex(ValueError, "series_id"):
            self.new_session(bundle=bundle)
        for section, key, bad_path in (
            ("voices", "directory", self.bundle["directory"]),
            ("render", "work_dir", self.config["engine"]["checkpoint"]),
            ("voices", "directory", self.config["render"]["work_dir"]),
            ("render", "output_dir", self.root),
        ):
            config = deepcopy(self.config)
            config[section][key] = str(bad_path)
            with self.subTest(section=section, key=key):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.new_session(config=config)
        self.assertFalse((self.root / "voices").exists())

    def test_legacy_binding_can_be_auditioned_and_restored(self):
        reference = self.root / "legacy.wav"
        text = self.config["voices"]["reference_text"]
        self.engine.create_voice("Original voice.", text, reference, seed=1)
        baseline = self.session.store.publish(
            self.voice_id, self.bundle["cast"]["voices"][0], reference, text,
            self.engine.identity, {"passed": True},
            {"method": "existing_reference", "requested_description": "Original voice."})
        self.assertEqual(self.session.rows()[0]["prompt"], "Original voice.")
        candidate = self.session.generate(self.voice_id)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], baseline["asset_hash"])
        self.session.accept(self.voice_id, candidate["candidate_id"])
        self.session.accept(self.voice_id, "legacy")
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], baseline["asset_hash"])

    def test_close_is_lazy_idempotent_and_closes_initialized_engine(self):
        self.session.close()
        self.session.close()
        self.assertEqual(self.engine_factories, [])
        with self.assertRaisesRegex(ValueError, "closed"):
            self.session.rows()
        session = self.new_session()
        session.generate(self.voice_id)
        session.close()
        session.close()
        self.assertTrue(self.engine.closed)

    @unittest.skipUnless(shutil.which("ffprobe"), "FFprobe is required for complete renderer validation")
    def test_render_cache_changes_only_after_accept_and_future_books_reuse_selection(self):
        for candidate in self.session.generate_missing():
            self.session.accept(candidate["voice_id"], candidate["candidate_id"])
        self.config["render"]["ffprobe"] = shutil.which("ffprobe")

        def book_render(name, bundle=None, *, new_work=False):
            config = deepcopy(self.config)
            config["render"]["output_dir"] = str(self.root / name)
            if new_work:
                config["render"]["work_dir"] = str(self.root / (name + "-work"))
            return render(bundle or self.bundle, config, self.engine, self.verifier, notify=lambda _: None)

        first = book_render("first-output")
        speech_count = len(self.engine.speech_calls)
        affected_count = sum(call["voice_id"] == self.voice_id for call in self.engine.speech_calls)
        candidate = self.session.generate(self.voice_id)
        pending = book_render("pending-output")
        self.assertEqual(first["voice_assets"], pending["voice_assets"])
        self.assertEqual(len(self.engine.speech_calls), speech_count)
        self.session.accept(self.voice_id, candidate["candidate_id"])
        changed = book_render("selected-output")
        self.assertEqual(changed["voice_assets"][self.voice_id], candidate["asset_hash"])
        self.assertNotEqual(first["voice_assets"][self.voice_id], changed["voice_assets"][self.voice_id])
        self.assertEqual(len(self.engine.speech_calls), speech_count + affected_count)
        for call in self.engine.speech_calls[speech_count:]:
            self.assertEqual(call["asset_hash"], candidate["asset_hash"])
        # A new book has a different source, but the same stable series/voice IDs.
        following = read_json(EXAMPLE.parent / "reviewed.json")
        following["source"]["book"]["id"] = hashlib.sha256(b"casting sequel").hexdigest()
        following["source"]["book"]["title"] = "Another Lantern"
        following_dir = self.root / "following-bundle"
        export_bundle(following, following_dir)
        voice_generation_count = len(self.engine.calls)
        sequel = book_render("following-output", load_bundle(following_dir), new_work=True)
        self.assertEqual(len(self.engine.calls), voice_generation_count)
        self.assertEqual(changed["voice_assets"], sequel["voice_assets"])
        # Finished artifacts still contain the original immutable cast decision.
        self.assertEqual(read_json(self.root / "first-output" / "render.json")["voice_assets"],
                         first["voice_assets"])


if __name__ == "__main__":
    unittest.main()
