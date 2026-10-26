"""Confirmed series pronunciations change only affected model inputs."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.chapter_protocol import _chunk_shape, _leaf
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.renderer.performance_runtime import require_engine_support
from epub_tts_narrator.renderer.pronunciation_runtime import merge_row_pronunciations
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.workflow import make_render_plan, render
from epub_tts_narrator.spoken_text import build_spoken
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_renderer_speech_identity import reference, speech_request


def entry(word="Mara", value="MAH-rah", *, status="confirmed", alphabet="respelling"):
    result = {"id": "test-" + word, "word": word, "status": status}
    if status == "confirmed":
        result["pronunciation"] = {"alphabet": alphabet, "value": value}
    return result


def row(text="Mara spoke.", **changes):
    return {"id": "s1", "text": text, "start": 40, "end": 40 + len(text),
            "delivery": {"direction": "", "evidence": []}, **changes}


def record(text, controls=()):
    return {"version": 1, "text_sha256": text_sha256(text), "controls": list(controls)}


class PronunciationRowTests(unittest.TestCase):
    def test_whole_case_sensitive_words_leave_similar_names_and_possessives_untouched(self):
        source = row("Mara greeted Maran, mara, and Mara's sister. Mara!")
        before = deepcopy(source)
        result = merge_row_pronunciations(source, None, {"Mara": entry()})
        phones = [item for item in result["controls"] if item["kind"] == "pronunciation"]
        self.assertEqual([item["start"] for item in phones], [0, 45])
        self.assertEqual([item["text"] for item in phones], ["Mara", "Mara"])
        self.assertEqual(source, before)
        self.assertEqual(prepare_synthesis(source["text"], performance=result)[1],
                         "MAH-rah greeted Maran, mara, and Mara's sister. MAH-rah!")

    def test_source_name_maps_through_approved_replacement_without_editing_speech(self):
        source = row("Mara spoke.")
        source["speech_normalization"] = {"version": 1, "edits": [{
            "start": 0, "end": 4, "original": "Mara", "replacement": "Marra",
            "direction": "A precise initial consonant", "reason": "Recorded pronunciation."}]}
        text = build_spoken(source)[0]
        result = merge_row_pronunciations(source, None, {"Mara": entry()})
        phone = next(item for item in result["controls"] if item["kind"] == "pronunciation")
        self.assertEqual((phone["start"], phone["end"], phone["text"]), (0, 5, "Marra"))
        self.assertEqual(text, "Marra spoke.")
        self.assertEqual(result["controls"], [phone])
        self.assertEqual(prepare_synthesis(text, performance=result)[2]["expressive_performance"]["expected_text"], text)
        self.assertEqual(source["text"], "Mara spoke.")
        manual = record(text, [{"kind": "delivery", "start": 0, "end": 5, "text": "Marra",
                                "tag": "style", "direction": "deliberately harsh"}])
        combined = merge_row_pronunciations(source, manual, {"Mara": entry()})
        directions = [item["direction"] for item in combined["controls"] if item["kind"] == "delivery"]
        self.assertEqual(directions, ["deliberately harsh"])

    def test_ambiguous_normalization_never_guesses_a_word_scope(self):
        for original, replacement, end in (("Mara spoke", "Marra spoke", 10), ("Mara", "Mah rah", 4)):
            with self.subTest(replacement=replacement):
                source = row("Mara spoke.")
                source["speech_normalization"] = {"version": 1, "edits": [{
                    "start": 0, "end": end, "original": original, "replacement": replacement,
                    "direction": "Pronunciation hint", "reason": "Recorded pronunciation."}]}
                with self.assertRaisesRegex(ValueError, "normalization"):
                    merge_row_pronunciations(source, None, {"Mara": entry()})

    def test_manual_and_confirmed_controls_deduplicate_or_fail_on_conflict(self):
        source = row()
        first = merge_row_pronunciations(source, None, {"Mara": entry()})
        self.assertEqual(merge_row_pronunciations(source, first, {"Mara": entry()}), first)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            merge_row_pronunciations(source, first, {"Mara": entry(value="MAR-ah")})
        existing = record(source["text"], [{"kind": "delivery", "start": 0, "end": 4,
                                          "text": "Mara", "tag": "whisper", "direction": ""}])
        combined = merge_row_pronunciations(source, existing, {
            "Mara": entry(value="M AA1 R AH0", alphabet="cmu_arpabet")})
        self.assertIn("<whisper><|phoneme_start|>M AA1 R AH0<|phoneme_end|></whisper>",
                      prepare_synthesis(source["text"], performance=combined)[1])

    def test_registry_unicode_hyphens_and_apostrophes_share_word_scope(self):
        for word in ("Source‐Vow", "Source‑Vow", "Mara’s"):
            with self.subTest(word=word):
                source = row(f"{word} beside {word}ish.")
                result = merge_row_pronunciations(source, None, {word: entry(word=word, value="Quarm")})
                phones = [item for item in result["controls"] if item["kind"] == "pronunciation"]
                self.assertEqual([item["text"] for item in phones], [word])
                self.assertEqual(prepare_synthesis(source["text"], performance=result)[1],
                                 f"Quarm beside {word}ish.")


class PronunciationPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        export_bundle(valid_document(), self.root / "book", allow_review=True)
        self.bundle = load_bundle(self.root / "book")
        self.config = renderer_config(self.root, series_id="fixture", paragraph_synthesis=False)
        self.entries = []
        patcher = patch("epub_tts_narrator.pronunciations.load_pronunciations", side_effect=lambda _: deepcopy(self.entries))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unconfirmed_absent_or_unmatched_entries_leave_legacy_plan_identical(self):
        before = make_plan(self.bundle, self.config)
        for entries in ([entry(status="pending")], [entry(status="ignored")], [entry(word="Nobody")], []):
            with self.subTest(entries=entries):
                self.entries = entries
                self.assertEqual(make_plan(self.bundle, self.config), before)

    def test_confirmed_correction_is_sparse_and_dynamically_rereads_registry(self):
        before = make_render_plan(self.bundle, self.config, multi_speaker=False)
        self.entries = [entry()]
        after = make_render_plan(self.bundle, self.config, multi_speaker=False)
        old, new = before["chapters"][0]["chunks"], after["chapters"][0]["chunks"]
        self.assertNotIn("expressive_performance", old[0])
        self.assertEqual(new[1:], old[1:])
        for key in ("spoken_text", "source_ranges", "chunk_id", "segment"):
            self.assertEqual(old[0][key], new[0][key])
        self.assertEqual(new[0]["expressive_traits"], new[0].get("speech_traits", {}))
        voice = reference(new[0]["segment"]["voice_id"])
        identities = [generation_identity(speech_request(voice, chunk=chunk), {voice["asset_hash"]: voice})
                      for chunk in (old[0], new[0])]
        self.assertNotEqual(identities[0]["text"], identities[1]["text"])
        self.assertIn("MAH-rah", identities[1]["text"])
        self.entries = [entry(value="MAR-ah")]
        updated = make_plan(self.bundle, self.config)
        self.assertEqual(updated["chapters"][0]["chunks"][0]["expressive_performance"]["controls"][0]["replacement"], "MAR-ah")

    def test_empty_slices_of_a_corrected_row_keep_legacy_fields_and_identity(self):
        source = self.bundle["chapters"][0]["rows"][0]
        source.update(text="Mara spoke. Ordinary words follow. More words follow.", start=0,
                      end=len("Mara spoke. Ordinary words follow. More words follow."))
        self.config["render"]["max_chunk_chars"] = 20
        before = make_plan(self.bundle, self.config)
        self.entries = [entry()]
        after = make_plan(self.bundle, self.config)
        old = [chunk for chunk in before["chapters"][0]["chunks"] if chunk["segment"]["id"] == source["id"]]
        new = [chunk for chunk in after["chapters"][0]["chunks"] if chunk["segment"]["id"] == source["id"]]
        self.assertGreater(len(new), 1)
        self.assertIn("expressive_performance", new[0])
        self.assertEqual(new[1:], old[1:])

    def test_corrected_plan_crosses_remote_contract_and_rejects_unsupported_engine(self):
        self.entries = [entry()]
        plan = make_render_plan(self.bundle, self.config, multi_speaker=False)
        bindings = {key: "a" * 64 for key in plan["voice_ids"]}
        for chunk in plan["chapters"][0]["chunks"]:
            _chunk_shape(chunk, set())
            _leaf(chunk, bindings)
        with self.assertRaisesRegex(ValueError, "does not support"):
            require_engine_support(plan, type("Unsupported", (), {"capabilities": {}})())
        require_engine_support(plan, type("Supported", (), {"capabilities": {"structured_performance": True}})())

    def test_existing_empty_expressive_plan_keeps_same_untouched_generation_identity(self):
        source = self.bundle["chapters"][0]["rows"][0]
        text = build_spoken(source)[0]
        legacy = make_render_plan(self.bundle, self.config, multi_speaker=False)["chapters"][0]["chunks"][0]
        existing = record(text)
        merged = merge_row_pronunciations(source, existing, {})
        self.assertEqual(merged, existing)
        from epub_tts_narrator.renderer.performance_runtime import bind_chunk_performance
        expressive = deepcopy(legacy)
        bind_chunk_performance(expressive, merged)
        voice = reference(legacy["segment"]["voice_id"])
        self.assertEqual(generation_identity(speech_request(voice, chunk=legacy), {voice["asset_hash"]: voice}),
                         generation_identity(speech_request(voice, chunk=expressive), {voice["asset_hash"]: voice}))

    def _paragraph_preflight(self):
        from epub_tts_narrator.renderer.workflow import _preflight
        self.config["render"]["paragraph_synthesis"] = True
        rows = self.bundle["chapters"][0]["rows"]
        rows[1].update(voice_id=rows[0]["voice_id"], role_id=rows[0]["role_id"])
        self.entries = [entry()]
        config, plan, _ = _preflight(self.bundle, self.config, chapters_only=True,
                                     voice_store_override=object())
        chapter = plan["chapters"][0]
        self.assertIn("paragraph_turns", chapter["chunks"][0])
        voices = {identity: reference(identity) for identity in plan["voice_ids"]}
        return config, plan, chapter, voices

    def _replacements(self, chapter):
        return [control["replacement"] for chunk in chapter["chunks"]
                for turn in chunk.get("paragraph_turns", [chunk])
                for control in turn.get("expressive_performance", {}).get("controls", [])
                if control["kind"] == "pronunciation"]

    def test_preflight_snapshot_is_run_local_and_next_invocation_reloads_decisions(self):
        from epub_tts_narrator.renderer.workflow import _preflight
        config, plan, chapter, _ = self._paragraph_preflight()
        self.assertNotIn("_pronunciation_snapshot", self.config)
        self.entries = [entry(value="MAR-ah")]
        self.assertEqual(self._replacements(make_render_plan(self.bundle, config)["chapters"][0]), ["MAH-rah"])
        refreshed, next_plan, _ = _preflight(self.bundle, config, chapters_only=True,
                                            voice_store_override=object())
        self.assertEqual(self._replacements(next_plan["chapters"][0]), ["MAR-ah"])
        self.assertEqual(self._replacements(plan["chapters"][0]), ["MAH-rah"])
        self.assertEqual(config["_pronunciation_snapshot"]["Mara"]["pronunciation"]["value"], "MAH-rah")

    def test_local_paragraph_recovery_uses_initial_snapshot_after_midrun_edit(self):
        from epub_tts_narrator.renderer.workflow import (
            SpeechVerificationFailed, _paragraph_request, _render_chapter_with_recovery,
        )
        config, _, chapter, voices = self._paragraph_preflight()
        identity = {"engine": "fixture", "model": "fixture"}
        request = _paragraph_request(chapter["chunks"][0], voices, identity, config)
        calls = []
        def synthesize(current, *args, **kwargs):
            calls.append(current)
            if len(calls) == 1:
                self.entries = [entry(value="MAR-ah")]
                raise SpeechVerificationFailed("split", request=request, report={"reasons": ["text_mismatch"]})
            self.assertFalse(any("paragraph_turns" in chunk for chunk in current["chunks"]))
            self.assertEqual(self._replacements(current), ["MAH-rah"])
            return [], []
        with patch("epub_tts_narrator.renderer.workflow._render_chapter_chunks", side_effect=synthesize):
            revised, _, _ = _render_chapter_with_recovery(
                self.bundle, chapter, voices, object(), identity, object(), config,
                config["render"]["work_dir"], notify=lambda _: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self._replacements(revised), ["MAH-rah"])

    def test_remote_paragraph_recovery_uses_same_frozen_policy_as_initial_job(self):
        from epub_tts_narrator.renderer.chapter_client import RemoteChapterFailed, _recover_failed_paragraph
        from epub_tts_narrator.renderer.common import digest
        from epub_tts_narrator.renderer.workflow import _paragraph_request
        config, plan, chapter, voices = self._paragraph_preflight()
        identity = {"engine": "fixture", "model": "fixture"}
        chunk = chapter["chunks"][0]
        request = _paragraph_request(chunk, voices, identity, config)
        failure = RemoteChapterFailed("split", job_id="job", record={"selection": {"chapter": chapter}},
                                      paragraph={"request_hash": digest(request), "chunk_id": chunk["chunk_id"]})
        self.entries = [entry(value="MAR-ah")]
        _, replacement = _recover_failed_paragraph(
            self.bundle, config, plan, voices, {"engine": {"engine": identity}}, failure, lambda _: None)
        self.assertFalse(any("paragraph_turns" in item for item in replacement["chunks"]))
        self.assertEqual(self._replacements(replacement), ["MAH-rah"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
    def test_confirmed_revision_regenerates_only_affected_take_and_original_can_be_reused(self):
        from audition_fixtures import prepare_bundle
        from test_renderer_workflow import FixtureEngine, FixtureVerifier
        class StructuredFixtureEngine(FixtureEngine):
            capabilities = {**FixtureEngine.capabilities, "structured_performance": True}
            identity = {"engine": "drama3_mlx", "model": "fixture-model-v1",
                        "checkpoint_files_sha256": {"weights.safetensors": "fixture-weights"},
                        "temperature": 1.0, "top_p": .9, "top_k": 30}
            voice_binding_identity = FixtureEngine.identity

            def synthesize(self, text, voice, path, *, seed, direction="", performance=None, traits=""):
                prepared, compiled, metadata = prepare_synthesis(
                    text, direction, performance=performance, traits=traits)
                return {**super().synthesize(prepared, voice, path, seed=seed, direction=direction),
                        **metadata, "synthesis_text": compiled}

        engine = StructuredFixtureEngine({})
        verifier = FixtureVerifier(engine.fixture_transcripts)
        bundle = prepare_bundle(self.bundle)
        original = render(bundle, self.config, engine, verifier, chapters_only=True, notify=lambda _: None)
        calls, voice_calls = len(engine.speech_calls), len(engine.voice_calls)
        self.entries = [entry()]
        revised = render(bundle, self.config, engine, verifier, chapters_only=True, notify=lambda _: None)
        self.assertEqual(len(engine.speech_calls), calls + 1)
        self.assertEqual(len(engine.voice_calls), voice_calls)
        self.assertNotEqual(original["chapters"][0]["cache_directory"], revised["chapters"][0]["cache_directory"])
        with patch.object(engine, "synthesize", side_effect=AssertionError("accepted speech regenerated")):
            repeated = render(bundle, self.config, engine, verifier, chapters_only=True, notify=lambda _: None)
            self.assertEqual(repeated["chapters"], revised["chapters"])
            self.entries = []
            restored = render(bundle, self.config, engine, verifier, chapters_only=True, notify=lambda _: None)
        self.assertEqual(restored["chapters"], original["chapters"])


class PronunciationInheritancePlanTests(unittest.TestCase):
    def test_late_ancestor_correction_applies_without_reloading_child_or_reanalysis(self):
        from epub_tts_narrator.pronunciations import set_pronunciation, save_candidates, save_proposal, reset_pronunciation
        from epub_tts_narrator.renderer.config import DEFAULTS
        from test_pronunciations import PronunciationsTests, candidate
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PronunciationsTests()
            fixture.root = Path(temporary).resolve()
            first, first_doc = fixture.book()
            second, second_doc = fixture.book("second", first)
            child = load_bundle(second)
            config = deepcopy(DEFAULTS)
            baseline = make_plan(child, config)
            save_candidates(second, second_doc, [candidate()])
            save_proposal(second, second_doc, "Mara", {"alphabet": "cmu_arpabet", "value": "M AA1 R AH0"})
            self.assertEqual(make_plan(child, config), baseline)
            set_pronunciation(first, first_doc, "Mara", {"alphabet": "respelling", "value": "MAH-ruh"})
            updated = make_plan(child, config)
            def replacements(plan):
                return [control["replacement"] for chapter in plan["chapters"] for chunk in chapter["chunks"]
                        for control in chunk.get("expressive_performance", {}).get("controls", [])
                        if control["kind"] == "pronunciation"]
            self.assertTrue(replacements(updated))
            self.assertEqual(set(replacements(updated)), {"MAH-ruh"})
            set_pronunciation(second, second_doc, "Mara", {"alphabet": "respelling", "value": "MAIR-uh"})
            self.assertEqual(set(replacements(make_plan(child, config))), {"MAIR-uh"})
            set_pronunciation(first, first_doc, "Mara", {"alphabet": "respelling", "value": "MAR-ah"})
            self.assertEqual(set(replacements(make_plan(child, config))), {"MAIR-uh"})
            set_pronunciation(second, second_doc, "Mara", status="ignored")
            self.assertEqual(make_plan(child, config), baseline)
            reset_pronunciation(second, "Mara")
            self.assertEqual(set(replacements(make_plan(child, config))), {"MAR-ah"})
