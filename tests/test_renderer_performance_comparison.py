"""Isolated comparison snapshots, restart safety, and playable report artifacts."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.spoken_text import build_spoken
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.chapter_cache import prepare_chapter
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.performance_comparison import compare_chapter, prepare_comparison
from epub_tts_narrator.renderer.voices import VoiceStore
from shared_fixtures import renderer_config
from test_renderer_assembly import clip
from test_renderer_workflow import FixtureEngine, FixtureVerifier, write_tone


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


def empty_plan(bundle):
    rows = bundle["chapters"][0]["rows"]
    return {"format": "epub-tts-performance-plan", "version": 1,
            "book_id": bundle["source"]["book"]["id"],
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "status": "complete", "chapters": [1],
            "segments": {row["id"]: {"version": 1, "text_sha256": text_sha256(build_spoken(row)[0]),
                                       "controls": []} for row in rows},
            "provenance": {"script_sha256": fingerprint(rows), "cast_sha256": fingerprint(bundle["cast"])}}


def inventory(root):
    return {str(path.relative_to(root)): file_hash(path) for path in root.rglob("*") if path.is_file()}


class PerformanceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        shutil.copytree(EXAMPLE, self.root / "book")
        self.bundle = load_bundle(self.root / "book")
        self.config = renderer_config(self.root, series_id="comparison-fixture")
        self.config["engine"]["type"] = "fixture"
        self.config["engine"]["max_new_tokens"] = 2048
        self.config["jev"]["api_key"] = "DO_NOT_ECHO"
        self.config["render"].update(normalize_loudness=False, normalize_pauses=False)
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier({})
        self.resolved = resolve_book_config(self.bundle, self.config)
        self.store = voice_store(self.bundle, self.resolved)
        self.reference = self.root / "reference.wav"
        write_tone(self.reference)
        profiles = {item["id"]: item for item in self.bundle["cast"]["voices"]}
        for voice_id in make_plan(self.bundle, self.resolved)["voice_ids"]:
            if self.store.load(voice_id) is None:
                self.store.publish(voice_id, profiles[voice_id], self.reference, "Existing saved voice.",
                                   self.engine.identity, {"passed": True}, {"description_applied": True})
        self.directory = self.root / "comparison"
        self.plan = empty_plan(self.bundle)
        self.calls = []
        self.fail = None

    def prepare(self, config=None):
        return prepare_comparison(self.bundle, config or self.config, 1, self.directory)

    def render_fixture(self, bundle, config, engine, verifier, *, chapters_only, voice_store_override, notify):
        mode = config["render"]["performance_mode"]
        self.calls.append(mode)
        self.assertIs(engine, self.engine)
        self.assertIs(verifier, self.verifier)
        self.assertTrue(chapters_only)
        self.assertEqual(config["render"]["chapter_numbers"], [1])
        self.assertEqual(Path(bundle["directory"]), self.directory / "snapshot")
        for voice_id in make_plan(bundle, config)["voice_ids"]:
            self.assertEqual(voice_store_override.load(voice_id)["asset_hash"], self.store.load(voice_id)["asset_hash"])
            self.assertTrue(Path(voice_store_override.load(voice_id)["reference_audio"]).is_relative_to(self.directory))
        if self.fail == mode:
            raise RuntimeError("DO_NOT_ECHO and source text must not enter state")
        chapter = {"id": "chapter-0001", "number": 1, "title": bundle["chapters"][0]["entry"]["title"],
                   "script_path": "scripts/chapter-0001.jsonl", "_ordinal": 1, "_total_chapters": 1}
        ready = prepare_chapter(chapter, [clip(self.reference)], Path(config["render"]["work_dir"]) / "chapters",
                                book=bundle["source"]["book"], render_settings=config["render"],
                                context={"mode": mode}, notify=lambda _: None)
        return {"status": "chapters_complete", "chapters": [{"number": 1, "title": chapter["title"],
                 "audio_file": ready["encoded_path"], "cache_directory": ready["cache_directory"]}]}

    def compare(self):
        return compare_chapter(self.bundle, self.config, self.engine, self.verifier,
                               chapter_number=1, directory=self.directory, performance_plan=self.plan,
                               render_method=self.render_fixture, notify=lambda _: None)

    def test_preparation_freezes_byte_identical_source_and_voice_assets_without_touching_originals(self):
        original_book, original_voices = inventory(self.root / "book"), inventory(self.root / "voices")
        prepared = self.prepare()
        for name, checksum in self.bundle["manifest"]["sha256"].items():
            self.assertEqual(file_hash(prepared["directory"] / "snapshot" / name), checksum)
        for voice_id in make_plan(self.bundle, self.resolved)["voice_ids"]:
            self.assertEqual(prepared["voice_store"].load(voice_id)["asset_hash"], self.store.load(voice_id)["asset_hash"])
        self.assertEqual(inventory(self.root / "book"), original_book)
        self.assertEqual(inventory(self.root / "voices"), original_voices)
        raw = (self.directory / "comparison.json").read_text()
        self.assertNotIn("DO_NOT_ECHO", raw)
        self.assertEqual(read_json(self.directory / "comparison.json")["inputs"]["settings"]["engine"]["max_new_tokens"], 2048)
        self.assertEqual(self.prepare()["manifest"]["input_sha256"], prepared["manifest"]["input_sha256"])

    def test_missing_voice_stops_before_creating_comparison_or_regenerating_any_voice(self):
        voice_id = make_plan(self.bundle, self.resolved)["voice_ids"][0]
        shutil.rmtree(self.store.folder(self.store.canonical_id(voice_id)))
        with self.assertRaisesRegex(ValueError, "existing selected voice"):
            self.prepare()
        self.assertFalse(self.directory.exists())
        self.assertFalse(self.engine.voice_calls)

    def test_changed_seed_or_voice_selection_cannot_mix_comparison_inputs(self):
        self.prepare()
        config = deepcopy(self.config)
        config["render"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.prepare(config)
        voice_id = make_plan(self.bundle, self.resolved)["voice_ids"][0]
        current = self.store.load(voice_id)
        candidate = "1" * 32
        self.store.publish_candidate(voice_id, current["profile"], self.reference,
                                     current["reference_text"], self.engine.identity,
                                     {"passed": True}, {}, candidate_id=candidate)
        self.store.select_candidate(voice_id, candidate)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.prepare()

    def test_global_expressive_mode_does_not_require_a_preexisting_plan_to_prepare_comparison(self):
        config = deepcopy(self.config)
        config["render"].update(performance_mode="expressive", performance_plan=str(self.root / "missing-plan.json"))
        prepared = self.prepare(config)
        self.assertEqual(prepared["config"]["render"]["performance_mode"], "legacy")
        self.assertIsNone(prepared["config"]["render"]["performance_plan"])
        self.assertEqual(config["render"]["performance_mode"], "expressive")

    def test_frozen_asset_corruption_fails_and_does_not_replace_it(self):
        prepared = self.prepare()
        voice_id = make_plan(self.bundle, self.resolved)["voice_ids"][0]
        target = Path(prepared["voice_store"].load(voice_id)["reference_audio"])
        target.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()
        self.assertEqual(target.read_bytes(), b"modified")

    def test_snapshot_retains_selected_candidate_not_legacy_reference(self):
        voice_id = make_plan(self.bundle, self.resolved)["voice_ids"][0]
        current = self.store.load(voice_id)
        candidate = "2" * 32
        self.store.publish_candidate(voice_id, current["profile"], self.reference,
                                     current["reference_text"], self.engine.identity,
                                     {"passed": True}, {}, candidate_id=candidate)
        self.store.select_candidate(voice_id, candidate)
        prepared = self.prepare()
        self.assertEqual(prepared["voice_store"].load(voice_id)["candidate_id"], candidate)
        self.assertEqual(prepared["voice_store"].load(voice_id)["asset_hash"], self.store.load(voice_id)["asset_hash"])

    def test_invalid_chapter_overlap_and_symlink_are_rejected(self):
        for number in (0, 2, True):
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "exactly one valid"):
                prepare_comparison(self.bundle, self.config, number, self.directory)
        for directory in (self.root / "book" / "comparison", self.root / "voices", self.root):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "separate"):
                prepare_comparison(self.bundle, self.config, 1, directory)
        target = self.root / "redirected"
        target.mkdir()
        self.directory.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.prepare()

    def test_ancestor_only_alias_target_freezes_without_requiring_it_in_current_cast(self):
        ids = make_plan(self.bundle, self.resolved)["voice_ids"]
        parent = VoiceStore(self.root / "ancestor-voices", self.resolved["voices"]["series_id"])
        parent.publish("ancestor-voice", {}, self.reference, "Ancestor voice.", self.engine.identity, {"passed": True}, {})
        original = VoiceStore(self.root / "voices", self.resolved["voices"]["series_id"],
                              aliases={ids[0]: "ancestor-voice"}, parents=[parent])
        with patch("epub_tts_narrator.renderer.performance_comparison.voice_store", return_value=original):
            prepared = self.prepare()
        self.assertEqual(prepared["voice_store"].load(ids[0])["asset_hash"], parent.load("ancestor-voice")["asset_hash"])
        self.assertEqual(prepared["voice_store"].canonical_id(ids[0]), "ancestor-voice")

    def test_incomplete_performance_plan_fails_before_rendering_either_half(self):
        self.plan["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.compare()
        self.assertEqual(self.calls, [])

    def test_compare_cli_prepares_once_without_models_and_resumes_complete_plan_without_provider(self):
        from epub_tts_narrator.renderer.cli import main
        calls = []
        def prepare(bundle, analysis_config, path, *, chapters):
            calls.append(chapters)
            write_json(path, self.plan)
        arguments = ["compare-performance", str(self.root / "book"), "--chapter", "1", "--directory",
                     str(self.directory), "--prepare-only", "--local"]
        configured = deepcopy(self.config)
        configured["render_pool"]["workers"] = [{"name": "external", "type": "fish_http"}]
        configured["remote_chapters"]["url"] = "https://example.invalid"
        with patch("epub_tts_narrator.renderer.cli.load_config", side_effect=lambda *_args, **_kwargs: deepcopy(configured)), \
                patch("epub_tts_narrator.config.load_config", return_value={}), \
                patch("epub_tts_narrator.performance_planning.prepare_performance", side_effect=prepare), \
                patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure", side_effect=AssertionError("loaded a model")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 0)
            with patch("epub_tts_narrator.config.load_config", side_effect=AssertionError("unneeded provider")):
                self.assertEqual(main(arguments), 0)
        self.assertEqual(calls, [[1]])
        self.assertEqual(len(configured["render_pool"]["workers"]), 1)
        self.assertEqual(read_json(self.directory / "comparison.json")["inputs"]["settings"]["render_pool"]["workers"], [])

    def test_prepare_cli_uses_same_config_for_analysis_and_needs_no_tts_or_selected_voices(self):
        from epub_tts_narrator.renderer.cli import main
        destination = self.root / "chosen-plan.json"
        shutil.rmtree(self.root / "voices")
        with patch("epub_tts_narrator.renderer.cli.load_config", return_value=self.config) as renderer_load, \
                patch("epub_tts_narrator.config.load_config", return_value={"provider": {"model": "fixture"}}) as analysis_load, \
                patch("epub_tts_narrator.performance_planning.prepare_performance") as prepare, \
                patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure", side_effect=AssertionError("loaded a model")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["prepare-performance", str(self.root / "book"), "--chapters", "1",
                                   "--plan-file", str(destination), "--config", "settings.toml"]), 0)
        renderer_load.assert_called_once_with(Path("settings.toml"), require_verifier=False, require_engine=False)
        analysis_load.assert_called_once_with(Path("settings.toml"))
        self.assertEqual(prepare.call_args.args[2], destination)
        self.assertEqual(prepare.call_args.kwargs["chapters"], [1])

    def test_compare_cli_requires_exactly_one_chapter_and_explicit_destination(self):
        from epub_tts_narrator.renderer.cli import parser
        for arguments in (["compare-performance", "book", "--directory", "comparison"],
                          ["compare-performance", "book", "--chapter", "1"],
                          ["compare-performance", "book", "--chapter", "1-3", "--directory", "comparison"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(arguments)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
    def test_real_renderer_uses_frozen_voices_and_both_verified_modes_without_creating_voices(self):
        from epub_tts_narrator.renderer.engines import prepare_synthesis
        class StructuredFixtureEngine(FixtureEngine):
            capabilities = {**FixtureEngine.capabilities, "structured_performance": True}
            identity = {"engine": "drama3_mlx", "model": "fixture-model-v1",
                        "checkpoint_files_sha256": {"weights.safetensors": "fixture-weights"},
                        "temperature": 1.0, "top_p": .9, "top_k": 30}
            voice_binding_identity = FixtureEngine.identity

            def __init__(self, transcripts):
                super().__init__(transcripts)
                self.performances = []

            def synthesize(self, text, voice, path, *, seed, direction="", performance=None, traits=""):
                prepared, compiled, delivery = prepare_synthesis(text, direction, performance=performance, traits=traits)
                self.performances.append(performance)
                return {**super().synthesize(prepared, voice, path, seed=seed, direction=direction),
                        **delivery, "synthesis_text": compiled}

        transcripts = {}
        self.engine = StructuredFixtureEngine(transcripts)
        self.verifier = FixtureVerifier(transcripts)
        row = self.bundle["chapters"][0]["rows"][0]
        text = build_spoken(row)[0]
        self.plan["segments"][row["id"]]["controls"] = [{"kind": "delivery", "tag": "emphasis",
                                                          "start": 0, "end": len(text), "text": text}]
        original = inventory(self.root / "book"), inventory(self.root / "voices")
        result = compare_chapter(self.bundle, self.config, self.engine, self.verifier,
                                 chapter_number=1, directory=self.directory, performance_plan=self.plan,
                                 notify=lambda _: None)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(self.engine.voice_calls)
        self.assertIn(None, self.engine.performances)
        self.assertTrue(any(item is not None for item in self.engine.performances))
        self.assertEqual((inventory(self.root / "book"), inventory(self.root / "voices")), original)
        calls = len(self.engine.speech_calls)
        compare_chapter(self.bundle, self.config, self.engine, self.verifier,
                        chapter_number=1, directory=self.directory, performance_plan=self.plan,
                        notify=lambda _: None)
        self.assertEqual(len(self.engine.speech_calls), calls)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
    def test_both_methods_share_inputs_and_publish_playable_report_with_independent_sync(self):
        original = inventory(self.root / "book"), inventory(self.root / "voices")
        result = self.compare()
        self.assertEqual(self.calls, ["legacy", "expressive"])
        self.assertEqual(result["status"], "complete")
        report = Path(result["html"]).read_text()
        self.assertEqual(report.count("<audio "), 2)
        self.assertNotIn("DO_NOT_ECHO", Path(result["manifest"]).read_text())
        for mode in ("legacy", "expressive"):
            record = result["modes"][mode]
            audio = self.directory / record["audio"]
            sync = read_json(self.directory / record["sync"])
            self.assertEqual(file_hash(audio), sync["audio_sha256"])
            self.assertGreater(record["duration_seconds"], 0)
            self.assertGreater(record["elapsed_seconds"], 0)
        self.assertEqual((inventory(self.root / "book"), inventory(self.root / "voices")), original)
        self.compare()
        self.assertEqual(self.calls, ["legacy", "expressive"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
    def test_failed_expressive_half_retains_legacy_and_resumes_only_incomplete_half(self):
        self.fail = "expressive"
        with self.assertRaises(RuntimeError):
            self.compare()
        state = read_json(self.directory / "comparison.json")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["modes"]["legacy"]["status"], "complete")
        self.assertNotIn("DO_NOT_ECHO", (self.directory / "comparison.json").read_text())
        self.fail = None
        self.compare()
        self.assertEqual(self.calls, ["legacy", "expressive", "expressive"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
    def test_missing_output_restores_from_renderer_cache_and_changed_audio_is_not_silently_replaced(self):
        result = self.compare()
        audio = self.directory / result["modes"]["legacy"]["audio"]
        audio.unlink()
        self.compare()
        self.assertEqual(self.calls, ["legacy", "expressive", "legacy"])
        audio.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.compare()
