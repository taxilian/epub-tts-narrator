"""Model selection coexists with the usual checkout's newer configuration."""

from contextlib import redirect_stderr
from importlib.metadata import PackageNotFoundError
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.cli import doctor
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.engines import DRAMA3_RUNTIME_VERSION
from test_renderer_model_selection import FOUR_BIT, write_quantized
from test_renderer_verifier_dispatch import converted_model


class PrimaryModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "renderer.toml"

    def config(self, *, model=FOUR_BIT.repo, engine_extra="", extra=""):
        self.path.write_text(
            '[engine]\n'
            f'model="{model}"\n'
            'checkpoint="./checkpoint"\n'
            + engine_extra
            + '[verification]\n'
            'type="faster_whisper"\nmodel="./whisper"\n'
            + extra
        )
        return load_config(self.path)

    def test_model_selection_and_chapter_storage_load_without_unknown_key_warning(self):
        diagnostics = io.StringIO()

        with redirect_stderr(diagnostics):
            config = self.config(extra='[chapter_storage]\nbucket="audio-fixture"\nprefix="chapters"\n')

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertEqual(config["engine"]["model"], FOUR_BIT.repo)
        self.assertEqual(config["engine"]["checkpoint"], str(self.root / "checkpoint"))
        self.assertEqual(config["chapter_storage"]["bucket"], "audio-fixture")
        self.assertEqual(config["chapter_storage"]["prefix"], "chapters")
        self.assertEqual(config["remote_chapters"]["url"], "")
        self.assertEqual(config["chapter_server"]["data_dir"], str(self.root / "chapter-server-data"))

    def test_unknown_engine_parameter_still_warns_and_is_ignored(self):
        diagnostics = io.StringIO()

        with redirect_stderr(diagnostics):
            config = self.config(engine_extra='temperatur=0.4\n')

        self.assertIn("ignoring unknown configuration key engine.temperatur", diagnostics.getvalue())
        self.assertNotIn("temperatur", config["engine"])
        self.assertEqual(config["engine"]["model"], FOUR_BIT.repo)
        self.assertEqual(config["engine"]["temperature"], 1.0)

    def test_cache_alias_loads_without_replacing_selected_model_or_checkpoint(self):
        diagnostics = io.StringIO()

        with redirect_stderr(diagnostics):
            config = self.config(engine_extra='cache_model_alias=" fishaudio/drama-3 "\n')

        self.assertEqual(diagnostics.getvalue(), "")
        self.assertEqual(config["engine"]["cache_model_alias"], "fishaudio/drama-3")
        self.assertEqual(config["engine"]["model"], FOUR_BIT.repo)
        self.assertEqual(config["engine"]["checkpoint"], str(self.root / "checkpoint"))
        self.assertEqual(config["render_pool"]["model_revision"], FOUR_BIT.revision)

    def test_cache_alias_defaults_to_strict_identity_without_environment_settings(self):
        with patch.dict("os.environ", {"CACHE_MODEL_ALIAS": "fishaudio/drama-3"}):
            config = self.config()

        self.assertIsNone(config["engine"]["cache_model_alias"])

    def test_empty_malformed_or_control_character_cache_alias_is_rejected(self):
        for value in ("", "  ", True, 42, [], "group\nsecond", "group\x1b", "group\x7f"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "engine.cache_model_alias"):
                self.config(engine_extra=f'cache_model_alias={json.dumps(value)}\n')

    def test_custom_plugin_cannot_silently_claim_builtin_cache_alias_support(self):
        with self.assertRaisesRegex(ValueError, "cache_model_alias.*drama3_mlx"):
            self.config(model="plugin/model", engine_extra=(
                'type="custom_plugin"\ncache_model_alias="fishaudio/drama-3"\n'
            ))

    def test_invalid_model_is_rejected_instead_of_warned_and_replaced_with_default(self):
        diagnostics = io.StringIO()

        with redirect_stderr(diagnostics), self.assertRaisesRegex(ValueError, "engine.model"):
            self.config(model="mlx-community/fish-audio-drama-3-typo")

        self.assertEqual(diagnostics.getvalue(), "")

    def ready_config(self):
        write_quantized(self.root / "checkpoint")
        converted_model(self.root / "whisper")
        return self.config()

    def original_config(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type":"fish_omni"}')
        (checkpoint / "codec.pth").write_bytes(b"fixture codec")
        (checkpoint / "model.safetensors").write_bytes(b"fixture model")
        (checkpoint / "model.safetensors.index.json").write_text(
            '{"weight_map":{"fixture":"model.safetensors"}}'
        )
        converted_model(self.root / "whisper")
        return self.config(model="fishaudio/drama-3")

    def test_original_doctor_rejects_upstream_runtime_that_the_loader_cannot_use(self):
        config = self.original_config()

        def version(name):
            return "0.1.1" if name == "drama3-mlx" else "available"

        with patch("epub_tts_narrator.renderer.cli.metadata.version", side_effect=version), \
             patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"):
            result = doctor(config)

        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["drama_runtime_version"])
        self.assertEqual(result["packages"]["drama3-mlx"], "0.1.1")

    def test_original_doctor_accepts_shared_runtime(self):
        config = self.original_config()

        def version(name):
            return DRAMA3_RUNTIME_VERSION if name == "drama3-mlx" else "available"

        with patch("epub_tts_narrator.renderer.cli.metadata.version", side_effect=version), \
             patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"):
            result = doctor(config)

        self.assertTrue(result["ready"], result)
        self.assertTrue(result["checks"]["drama_runtime_version"])

    def test_doctor_accepts_quantized_runtime_with_faster_whisper_without_sogni(self):
        config = self.ready_config()

        for runtime in ("0.6.1", "0.6.2", "0.6.1.post1"):
            with self.subTest(runtime=runtime):
                def version(name):
                    if name in {"drama3-mlx", "mlx-whisper"}:
                        raise PackageNotFoundError(name)
                    return runtime if name == "mlx-audio" else "available"

                with patch("epub_tts_narrator.renderer.cli.metadata.version", side_effect=version), \
                     patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"):
                    result = doctor(config)

                self.assertTrue(result["ready"], result)
                self.assertTrue(result["checks"]["mlx_audio_drama3_version"])
                self.assertNotIn("drama3-mlx", result["packages"])
                self.assertNotIn("mlx-whisper", result["packages"])
                self.assertIn("faster-whisper", result["packages"])

    def test_doctor_reports_missing_or_wrong_quantized_runtime_without_crashing(self):
        config = self.ready_config()

        for runtime in (None, "", "invalid", "0.5.7", "0.6.0", "0.6.0.post1",
                        "0.6.1rc1", "0.6.2a1", "0.6.2.dev1", "0.7.0", "1.0.0"):
            with self.subTest(runtime=runtime):
                def version(name):
                    if name in {"drama3-mlx", "mlx-whisper"} or (name == "mlx-audio" and runtime is None):
                        raise PackageNotFoundError(name)
                    return runtime if name == "mlx-audio" else "available"

                with patch("epub_tts_narrator.renderer.cli.metadata.version", side_effect=version), \
                     patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"):
                    result = doctor(config)

                self.assertFalse(result["ready"])
                self.assertFalse(result["checks"]["mlx_audio_drama3_version"])
                self.assertEqual(result["packages"]["mlx-audio"], runtime)


if __name__ == "__main__":
    unittest.main()
