import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.cli import chapter_numbers, doctor, main
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.cli import export_bundle
from test_contracts import valid_document
from audition_fixtures import prepare_document


class RendererCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "renderer.toml"
        self.config.write_text(
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            '[voices]\nseries_id="test-series"\n'
        )
        self.bundle = self.root / "bundle"
        export_bundle(prepare_document(valid_document()), self.bundle)

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(a) for a in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_plan_never_initializes_model(self):
        with patch("epub_tts_narrator.renderer.engines.create_engine") as factory:
            status, output, error = self.run_cli("plan", self.bundle, "--config", self.config)
        self.assertEqual(status, 0, error)
        factory.assert_not_called()
        self.assertEqual(json.loads(output)["voices"], 2)

    def test_chapter_ranges_are_deduplicated_in_source_order(self):
        self.assertEqual(chapter_numbers("5, 1-3,3,2-4", 5), [1, 2, 3, 4, 5])
        self.assertEqual(chapter_numbers(" 3 ", 5), [3])
        for value in ("", "0", "-1", "2-1", "1,", "1,,2", "x", "1.5", "1-9999999999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                chapter_numbers(value, 5)

    def test_plan_chapter_override_does_not_edit_global_config(self):
        before = self.config.read_bytes()
        status, output, error = self.run_cli("plan", self.bundle, "--config", self.config,
                                             "--chapters", "1,1-1")
        self.assertEqual(status, 0, error)
        self.assertEqual(json.loads(output)["chapter_titles"][0]["number"], 1)
        self.assertEqual(self.config.read_bytes(), before)

    def test_invalid_chapter_selection_stops_before_loading_models(self):
        with patch("epub_tts_narrator.renderer.engines.create_engine") as engine:
            status, _, error = self.run_cli("render", self.bundle, "--config", self.config,
                                             "--chapters", "2")
        self.assertEqual(status, 2)
        self.assertIn("--chapters", error)
        engine.assert_not_called()

    def test_chapters_only_passes_per_run_options_and_closes_models(self):
        engine, verifier = Mock(), Mock()
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render") as render):
            render.return_value = {"status": "chapters_complete", "chapter_count": 1,
                                   "chapters": [{"number": 1, "title": "Chapter One",
                                                 "audio_file": "/fixture/chapter.mp3"}]}
            status, output, error = self.run_cli("render", self.bundle, "--config", self.config,
                "--chapters", "1", "--chapters-only", "--chapter-format", "mp3", "--allow-review")
        self.assertEqual(status, 0, error)
        self.assertIn("Completed chapter audio", output)
        self.assertIn("Chapter 1: /fixture/chapter.mp3", output)
        config = render.call_args.args[1]
        self.assertEqual(config["render"]["chapter_numbers"], [1])
        self.assertEqual(config["render"]["chapter_format"], "mp3")
        self.assertIs(config["render"]["allow_review"], True)
        self.assertEqual(render.call_args.kwargs, {"chapters_only": True})
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_assemble_does_not_create_any_model(self):
        with (patch("epub_tts_narrator.renderer.engines.create_engine") as engine,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier,
              patch("epub_tts_narrator.renderer.workflow.assemble", create=True) as assemble):
            status, output, error = self.run_cli("assemble", self.bundle, "--config", self.config,
                                               "--chapters", "1", "--chapter-format", "mp3")
        self.assertEqual(status, 0, error)
        self.assertIn("Completed audiobook", output)
        self.assertEqual(assemble.call_args.args[1]["render"]["chapter_format"], "mp3")
        engine.assert_not_called()
        verifier.assert_not_called()

    def test_review_notes_need_no_override_and_missing_speech_still_blocks(self):
        document = valid_document()
        document["segments"][1]["attribution"]["review_required"] = True
        flagged = self.root / "flagged"
        export_bundle(document, flagged, allow_review=True)
        with patch("epub_tts_narrator.renderer.workflow.assemble", create=True) as assemble:
            status, _, error = self.run_cli("assemble", flagged, "--config", self.config)
            self.assertEqual(status, 0, error)
            assemble.assert_called_once()
            status, _, error = self.run_cli("assemble", flagged, "--config", self.config, "--allow-review")
            self.assertEqual(status, 0, error)
        document["segments"][1].update(role_id=None, voice_id=None)
        document["segments"][1]["attribution"].update(basis="unresolved")
        unassigned = self.root / "unassigned"
        export_bundle(document, unassigned, allow_review=True)
        with patch("epub_tts_narrator.renderer.workflow.assemble", create=True) as assemble:
            status, _, error = self.run_cli("assemble", unassigned, "--config", self.config, "--allow-review")
        self.assertEqual(status, 2)
        self.assertIn("Missing voice", error)
        assemble.assert_not_called()

    def test_omitted_review_flag_preserves_global_setting(self):
        with self.config.open("a") as stream:
            stream.write('[render]\nallow_review=true\n')
        with patch("epub_tts_narrator.renderer.workflow.assemble", create=True) as assemble:
            status, _, error = self.run_cli("assemble", self.bundle, "--config", self.config)
        self.assertEqual(status, 0, error)
        self.assertIs(assemble.call_args.args[1]["render"]["allow_review"], True)

    def test_failed_render_closes_verifier_and_engine_even_if_cleanup_fails(self):
        engine, verifier = Mock(), Mock()
        verifier.close.side_effect = RuntimeError("cleanup fixture")
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=RuntimeError("render fixture"))):
            status, _, error = self.run_cli("render", self.bundle, "--config", self.config)
        self.assertEqual(status, 2)
        self.assertIn("render fixture", error)
        self.assertIn("cleanup fixture", error)
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_render_loudness_and_encoding_defaults_and_validation(self):
        settings = load_config(self.config)["render"]
        self.assertIs(settings["normalize_loudness"], True)
        self.assertEqual((settings["target_lufs"], settings["true_peak_db"]), (-20.0, -1.5))
        self.assertEqual((settings["aac_bitrate"], settings["mp3_bitrate"]), ("128k", "128k"))
        before = self.config.read_text()
        invalid = {"normalize_loudness": ("1", '"true"'),
                   "target_lufs": ("nan", "inf", "true", "-41", "-9"),
                   "true_peak_db": ("nan", "true", "-10", "1"),
                   "aac_bitrate": ('""', '"0k"', '"128"', '"1M"', "128"),
                   "mp3_bitrate": ('"-1k"', '"128k -y"', '"1000k"', "false")}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.config.write_text(before + f"[render]\n{key}={value}\n")
                    with self.assertRaisesRegex(ValueError, key):
                        load_config(self.config)
        self.config.write_text(before + '[render]\nnormalize_loudness=false\ntarget_lufs=-24\n'
                               'true_peak_db=-2\naac_bitrate="96k"\nmp3_bitrate="192k"\n')
        self.assertEqual(load_config(self.config)["render"]["aac_bitrate"], "96k")

    def test_series_mismatch_blocks_voice_creation_before_model(self):
        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["series"] = {"series_id": "another-series"}
        manifest_path.write_text(json.dumps(manifest))
        with patch("epub_tts_narrator.renderer.engines.create_engine") as factory:
            status, output, error = self.run_cli("create-voices", self.bundle, "--config", self.config)
        self.assertEqual(status, 2)
        self.assertIn("series", error)
        factory.assert_not_called()
        self.assertFalse((self.root / "voices").exists())

    def test_pause_normalization_settings_have_defaults_and_strict_types(self):
        settings = load_config(self.config)["render"]
        self.assertIs(settings["normalize_pauses"], True)
        self.assertEqual((settings["sentence_pause_ms"], settings["clause_pause_ms"]), (220, 100))
        before = self.config.read_text()
        invalid = {
            "paragraph_synthesis": ("1", '"true"'),
            "max_paragraph_chars": ("319", "1501", "1000.0", "true"),
            "normalize_pauses": ("1", '"true"'),
            "sentence_pause_ms": ("-1", "5001", "220.0", "true", "nan"),
            "clause_pause_ms": ("-1", "5001", "100.0", "false", "inf"),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.config.write_text(before + f"[render]\n{key}={value}\n")
                    with self.assertRaisesRegex(ValueError, key):
                        load_config(self.config)
        self.config.write_text(before + '[render]\nnormalize_pauses=false\n'
                               'sentence_pause_ms=300\nclause_pause_ms=0\n')
        settings = load_config(self.config)["render"]
        self.assertIs(settings["normalize_pauses"], False)
        self.assertEqual((settings["sentence_pause_ms"], settings["clause_pause_ms"]), (300, 0))
        # These controls also remain valid in the one shared global file when
        # the analyzer reads it; no book-specific configuration is required.
        from epub_tts_narrator.config import load_config as load_analyzer_config
        load_analyzer_config(self.config)

    def test_config_paths_are_relative_to_file_not_cwd_or_environment(self):
        with patch.dict("os.environ", {"AUDIOBOOK_OUTPUT_DIR": "/wrong", "FISH_CHECKPOINT": "/wrong"}):
            config = load_config(self.config)
        self.assertEqual(config["engine"]["checkpoint"], str(self.root / "tts-model"))
        self.assertEqual(config["render"]["output_dir"], str(self.root / "audiobook"))

    def test_config_rejects_pauses_outside_assembly_contract(self):
        with self.config.open("a") as stream:
            stream.write('[render]\nparagraph_pause_ms=5001\n')
        with self.assertRaisesRegex(ValueError, "paragraph_pause"):
            load_config(self.config)

    def test_both_tools_reject_global_voice_aliases_including_empty_table(self):
        from epub_tts_narrator.config import load_config as load_analyzer_config
        original = self.config.read_text()
        for policy in ('[voices.aliases]\n', '[voices.aliases]\nsource="target"\n'):
            self.config.write_text(original + policy)
            for loader in (load_config, load_analyzer_config):
                with self.subTest(policy=policy, loader=loader.__module__), self.assertRaisesRegex(
                        ValueError, "book-specific.*global configuration"):
                    loader(self.config)

    def test_config_rejects_model_and_work_overlap(self):
        with self.config.open("a") as stream:
            stream.write('[render]\nwork_dir="./tts-model/work"\n')
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_config(self.config)

    def test_doctor_reports_missing_models_without_loading_them(self):
        with patch("epub_tts_narrator.renderer.engines.create_engine") as factory:
            status, output, _ = self.run_cli("doctor", "--config", self.config)
        self.assertEqual(status, 2)
        self.assertFalse(json.loads(output)["ready"])
        factory.assert_not_called()

    def test_preset_plugin_can_have_options_without_a_checkpoint(self):
        self.config.write_text(
            '[engine]\ntype="future-plugin"\n[engine.options]\nvoice="stock-a"\n'
            '[verification]\nmodel="./asr-model"\n[voices]\nseries_id="test"\n'
        )
        config = load_config(self.config)
        self.assertIsNone(config["engine"]["checkpoint"])
        self.assertEqual(config["engine"]["options"], {"voice": "stock-a"})

    def set_reference_model(self, toml_value):
        self.config.write_text(
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            f'reference_model={toml_value}\n'
            '[voices]\nseries_id="test-series"\n')

    def test_reference_model_is_optional_and_relative_to_config_file(self):
        self.assertIsNone(load_config(self.config)["verification"]["reference_model"])
        self.set_reference_model('"./parakeet-model"')
        with patch.dict("os.environ", {"REFERENCE_MODEL": "/wrong"}):
            config = load_config(self.config)
        self.assertEqual(config["verification"]["reference_model"], str(self.root / "parakeet-model"))

    def test_reference_model_rejects_blank_or_nonpath_values(self):
        for value in ('""', '"   "', "true", "42", "[]"):
            with self.subTest(value=value):
                self.set_reference_model(value)
                with self.assertRaisesRegex(ValueError, "verification.reference_model"):
                    load_config(self.config)

    def test_reference_model_cannot_overlap_writable_stores(self):
        for value in ('"./voices"', '"./voices/model"', '"./render-work/model"',
                      '"./audiobook/model"', '"."'):
            with self.subTest(value=value):
                self.set_reference_model(value)
                with self.assertRaisesRegex(ValueError, "distinct|overlap"):
                    load_config(self.config)

    def test_doctor_reports_optional_reference_model_files_without_loading_models(self):
        self.set_reference_model('"./parakeet-model"')
        config = load_config(self.config)
        with (patch("epub_tts_narrator.renderer.engines.create_engine") as engine,
              patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="test"),
              patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/usr/bin/fixture")):
            missing = doctor(config)
            reference = self.root / "parakeet-model"
            reference.mkdir()
            (reference / "config.json").write_text("{}")
            config_only = doctor(config)
            (reference / "model.safetensors").write_bytes(b"fixture weights")
            available = doctor(config)
        self.assertFalse(missing["reference_verification"]["available"])
        self.assertFalse(missing["checks"]["reference_asr_config"])
        self.assertFalse(missing["checks"]["reference_asr_weights"])
        self.assertFalse(config_only["reference_verification"]["available"])
        self.assertTrue(available["reference_verification"]["available"])
        self.assertEqual(available["reference_verification"]["model"], str(reference))
        self.assertTrue(available["checks"]["reference_asr_config"])
        self.assertTrue(available["checks"]["reference_asr_weights"])
        engine.assert_not_called()

    def test_doctor_does_not_require_unconfigured_reference_model(self):
        report = doctor(load_config(self.config))
        self.assertEqual(report["reference_verification"],
                         {"configured": False, "model": None, "available": None})
        self.assertNotIn("reference_asr_config", report["checks"])
        self.assertNotIn("reference_asr_weights", report["checks"])
