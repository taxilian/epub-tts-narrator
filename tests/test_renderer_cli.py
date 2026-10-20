import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.cli import doctor, main
from audiobook_renderer.config import load_config
from epub_audiobook.cli import export_bundle
from test_contracts import valid_document


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
        export_bundle(valid_document(), self.bundle)

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(a) for a in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_plan_never_initializes_model(self):
        with patch("audiobook_renderer.engines.create_engine") as factory:
            status, output, error = self.run_cli("plan", self.bundle, "--config", self.config)
        self.assertEqual(status, 0, error)
        factory.assert_not_called()
        self.assertEqual(json.loads(output)["voices"], 2)

    def test_series_mismatch_blocks_voice_creation_before_model(self):
        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["series"] = {"series_id": "another-series"}
        manifest_path.write_text(json.dumps(manifest))
        with patch("audiobook_renderer.engines.create_engine") as factory:
            status, output, error = self.run_cli("create-voices", self.bundle, "--config", self.config)
        self.assertEqual(status, 2)
        self.assertIn("series", error)
        factory.assert_not_called()
        self.assertFalse((self.root / "voices").exists())

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
        from epub_audiobook.config import load_config as load_analyzer_config
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
        with patch("audiobook_renderer.engines.create_engine") as factory:
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
        with (patch("audiobook_renderer.engines.create_engine") as engine,
              patch("audiobook_renderer.cli.metadata.version", return_value="test"),
              patch("audiobook_renderer.cli.shutil.which", return_value="/usr/bin/fixture")):
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
