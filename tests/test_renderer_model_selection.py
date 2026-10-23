"""Model selection, local-file readiness and download routing without inference."""

from importlib.metadata import PackageNotFoundError
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.cli import doctor
from epub_tts_narrator.renderer.config import load_config, resolve_voice_design
from epub_tts_narrator.renderer.drama3_models import (
    DEFAULT_DRAMA3_MODEL, DRAMA3_MODELS, get_drama3_model,
)
from epub_tts_narrator.renderer.model_setup import ModelSetup, ModelSetupError
from shared_fixtures import Terminal


QUANTIZED = [model for model in DRAMA3_MODELS.values() if model.runtime == "mlx_audio"]
FOUR_BIT = get_drama3_model("mlx-community/fish-audio-drama-3-4bit")
EIGHT_BIT = get_drama3_model("mlx-community/fish-audio-drama-3-8bit")


def write_quantized(path, model=FOUR_BIT):
    path.mkdir(parents=True, exist_ok=True)
    bits = 4 if model.repo.endswith("-4bit") else 8 if model.repo.endswith("-8bit") else 3
    (path / "config.json").write_text(json.dumps({
        "model_type": "fish_drama3", "architectures": ["FishOmniModel"],
        "speech_template_version": "reference-bracket-v1",
        "quantization": {
            "bits": 8, "group_size": 64,
            "backbone.model.layers.0.mlp.switch_mlp.gate_proj": {"bits": bits, "group_size": 64},
        },
    }))
    for name in ("tokenizer.json", "codec.safetensors", *model.weight_files):
        (path / name).write_bytes(b"offline fixture")


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "renderer.toml"

    def config(self, engine="", extra=""):
        self.path.write_text('[engine]\ncheckpoint="./checkpoint"\n' + engine
                             + '[verification]\nmodel="./whisper"\n' + extra)
        return load_config(self.path)

    def test_omitted_model_keeps_existing_engine_and_designer_defaults(self):
        config = self.config()

        self.assertEqual(config["engine"]["model"], DEFAULT_DRAMA3_MODEL)
        self.assertEqual(config["engine"]["experts"], "q8")
        self.assertEqual(resolve_voice_design(config), {"type": "engine"})

    def test_supported_selection_preserves_explicit_directory_and_legacy_experts(self):
        for model in QUANTIZED:
            with self.subTest(model=model.repo):
                config = self.config(f'model="{model.repo}"\nexperts="q8"\n')

                self.assertEqual(config["engine"]["checkpoint"], str(self.root / "checkpoint"))
                self.assertEqual(config["engine"]["model"], model.repo)
                self.assertEqual(config["render_pool"]["model_revision"], model.revision)
                # Existing saved voices may render without any voice designer.
                self.assertEqual(resolve_voice_design(config), {"type": "engine"})

    def test_unknown_or_malformed_model_is_rejected_during_config_loading(self):
        for value in ('"unknown/drama"', '""', '[]', 'true'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "engine.model"):
                self.config(f'model={value}\n')

    def test_plugin_model_names_are_not_interpreted_as_builtin_checkpoints(self):
        config = self.config('type="custom_plugin"\nmodel="custom/model"\n')
        self.assertEqual(config["engine"]["model"], "custom/model")

    def test_remote_worker_inherits_selected_model_but_cannot_claim_a_different_one(self):
        worker = '[[render_pool.workers]]\nname="remote"\ntype="fish_http"\nurl="https://example.test/v1/tts"\n'
        config = self.config(f'model="{FOUR_BIT.repo}"\n', worker)

        self.assertEqual(config["render_pool"]["workers"][0]["model"], FOUR_BIT.repo)
        with self.assertRaisesRegex(ValueError, "match.*engine.model"):
            self.config(f'model="{FOUR_BIT.repo}"\n', worker + 'model="drama-3"\n')

    def test_explicit_pool_revision_remains_an_operator_setting(self):
        config = self.config(f'model="{FOUR_BIT.repo}"\n',
                             '[render_pool]\nmodel_revision="evaluated-release"\n')
        self.assertEqual(config["render_pool"]["model_revision"], "evaluated-release")


class SelectedModelSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.downloads = []

    def setup(self, model=FOUR_BIT, path=None):
        def download(**settings):
            self.downloads.append(settings)
            destination = Path(settings["local_dir"])
            write_quantized(destination, get_drama3_model(settings["repo_id"]))

        return ModelSetup(
            {"engine": {"type": "drama3_mlx", "model": model.repo,
                        "checkpoint": str(path or self.checkpoint)}},
            input_stream=Terminal("yes\n"), output_stream=io.StringIO(), downloader=download,
        )

    def test_missing_models_download_selected_repository_and_all_expected_shards(self):
        for model in QUANTIZED:
            with self.subTest(model=model.repo):
                path = self.root / model.repo.rsplit("/", 1)[1]
                self.setup(model, path).ensure("engine")
                self.assertEqual(self.downloads[-1]["repo_id"], model.repo)
                self.assertEqual(self.downloads[-1]["revision"], model.revision)
                self.assertEqual(self.downloads[-1]["local_dir"], str(path))
                self.assertTrue(all((path / name).is_file() for name in model.weight_files))

    def test_missing_final_shard_is_repaired_even_without_a_weight_index(self):
        write_quantized(self.checkpoint)
        missing = self.checkpoint / FOUR_BIT.weight_files[-1]
        missing.unlink()

        self.setup().ensure("engine")

        self.assertEqual(len(self.downloads), 1)
        self.assertTrue(missing.is_file())
        self.assertFalse((self.checkpoint / "model.safetensors.index.json").exists())

    def test_complete_model_is_read_only_and_needs_no_download(self):
        write_quantized(self.checkpoint)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.checkpoint.iterdir()}

        self.setup().ensure("engine")

        self.assertEqual(self.downloads, [])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in self.checkpoint.iterdir()})

    def test_original_checkpoint_is_not_overwritten_with_quantized_files(self):
        self.checkpoint.mkdir()
        config = self.checkpoint / "config.json"
        config.write_text('{"model_type":"fish_omni"}')

        with self.assertRaisesRegex(ModelSetupError, "different model"):
            self.setup().ensure("engine")

        self.assertEqual(self.downloads, [])
        self.assertEqual(config.read_text(), '{"model_type":"fish_omni"}')

    def test_files_without_compatible_model_metadata_do_not_look_ready(self):
        write_quantized(self.checkpoint)
        config = self.checkpoint / "config.json"
        for metadata in ({}, {"model_type": "fish_drama3"}):
            with self.subTest(metadata=metadata):
                config.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ModelSetupError, "does not identify|speech template"):
                    self.setup().ensure("engine")
        self.assertEqual(self.downloads, [])

    def test_known_other_conversion_revision_blocks_download_before_mixing_files(self):
        write_quantized(self.checkpoint, EIGHT_BIT)
        # A partial download can have fewer shards than the selected recipe.
        for name in EIGHT_BIT.weight_files[1:]:
            (self.checkpoint / name).unlink()
        metadata = self.checkpoint / ".cache/huggingface/download/config.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(EIGHT_BIT.revision + "\netag\n0\n")

        with self.assertRaisesRegex(ModelSetupError, "files downloaded from"):
            self.setup().ensure("engine")

        self.assertEqual(self.downloads, [])

    def test_renamed_checkpoint_with_wrong_expert_recipe_is_not_filled_in(self):
        write_quantized(self.checkpoint, FOUR_BIT)

        with self.assertRaisesRegex(ModelSetupError, "quantization recipe"):
            self.setup(EIGHT_BIT).ensure("engine")

        self.assertEqual(self.downloads, [])

    def test_missing_or_malformed_quantization_recipe_does_not_look_ready(self):
        write_quantized(self.checkpoint)
        path = self.checkpoint / "config.json"
        metadata = json.loads(path.read_text())
        expert = "backbone.model.layers.0.mlp.switch_mlp.gate_proj"
        invalid_recipes = (
            None, [], {}, {"bits": 8, "group_size": 64},
            {expert: {"bits": 4, "group_size": 64}},
            {"bits": True, "group_size": 64, expert: {"bits": 4, "group_size": 64}},
            {"bits": 8, "group_size": 0, expert: {"bits": 4, "group_size": 64}},
            {"bits": 8, "group_size": 64, expert: {"bits": "4", "group_size": 64}},
        )
        for recipe in invalid_recipes:
            with self.subTest(recipe=recipe):
                if recipe is None:
                    metadata.pop("quantization", None)
                else:
                    metadata["quantization"] = recipe
                path.write_text(json.dumps(metadata))

                with self.assertRaisesRegex(ModelSetupError, "quantization recipe"):
                    self.setup().ensure("engine")

                self.assertEqual(json.loads(path.read_text()), metadata)
        self.assertEqual(self.downloads, [])

    def test_unexpected_shards_block_mixing_even_without_download_metadata(self):
        write_quantized(self.checkpoint)
        (self.checkpoint / "model-00010.safetensors").write_bytes(b"another recipe")

        with self.assertRaisesRegex(ModelSetupError, "incompatible shard"):
            self.setup().ensure("engine")

        self.assertEqual(self.downloads, [])

    def test_complete_snapshot_in_another_repository_is_still_rejected(self):
        path = self.root / ("models--" + EIGHT_BIT.repo.replace("/", "--")) / "snapshots" / EIGHT_BIT.revision
        write_quantized(path, FOUR_BIT)

        with self.assertRaisesRegex(ModelSetupError, "different repository"):
            self.setup(FOUR_BIT, path).ensure("engine")

        self.assertEqual(self.downloads, [])


class SelectedModelDoctorTests(unittest.TestCase):
    def test_quantized_readiness_requires_supported_runtime_and_all_shards_without_sogni(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_quantized(root / "model")
            whisper = root / "whisper"
            whisper.mkdir()
            (whisper / "config.json").write_text("{}")
            (whisper / "weights.safetensors").write_bytes(b"weights")
            config_file = root / "config.toml"
            config_file.write_text(f'[engine]\nmodel="{FOUR_BIT.repo}"\ncheckpoint="./model"\n'
                                   '[verification]\nmodel="./whisper"\n')
            config = load_config(config_file)

            def version(name):
                if name == "drama3-mlx":
                    raise PackageNotFoundError(name)
                return "0.6.1" if name == "mlx-audio" else "available"

            with patch("epub_tts_narrator.renderer.cli.metadata.version", side_effect=version), \
                 patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"):
                ready = doctor(config)
                (root / "model" / FOUR_BIT.weight_files[-1]).unlink()
                incomplete = doctor(config)
                write_quantized(root / "model")
                with patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="0.5.7"):
                    old_runtime = doctor(config)

            self.assertTrue(ready["ready"], ready)
            self.assertNotIn("drama3-mlx", ready["packages"])
            self.assertFalse(incomplete["ready"])
            self.assertIn(FOUR_BIT.weight_files[-1], incomplete["model_problem"])
            self.assertFalse(old_runtime["ready"])
            self.assertFalse(old_runtime["checks"]["mlx_audio_drama3_version"])


if __name__ == "__main__":
    unittest.main()
