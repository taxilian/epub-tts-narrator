"""Model setup tests validate checkpoint layouts without downloading weights."""

import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.model_setup import (
    DRAMA, PARAKEET, QWEN, WHISPER, ModelSetup, ModelSetupError,
)
from shared_fixtures import Terminal


def files(path, names):
    path.mkdir(parents=True, exist_ok=True)
    for name, content in names.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(content) if isinstance(content, dict) else content)


def complete(path, kind="drama"):
    if kind == "drama":
        files(path, {"config.json": {"model_type": "fish_omni"},
                     "tokenizer.json": "tokenizer", "codec.pth": "codec",
                     "model.safetensors.index.json": {"weight_map": {"one": "model-1.safetensors", "two": "model-2.safetensors"}},
                     "model-1.safetensors": "first shard", "model-2.safetensors": "second shard"})
    elif kind == "whisper":
        files(path, {"config.json": {"model_type": "whisper"}, "weights.safetensors": "weights"})
    elif kind == "parakeet":
        files(path, {"config.json": {"encoder": {}}, "model.safetensors": "weights"})
    elif kind == "qwen":
        files(path, {"config.json": {"model_type": "qwen3_tts", "tts_model_type": "voice_design"},
                     "model.safetensors": "weights", "vocab.json": "vocab", "merges.txt": "merges",
                     "tokenizer_config.json": "tokenizer", "speech_tokenizer/config.json": "config",
                     "speech_tokenizer/model.safetensors": "weights"})


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "drama-3"
        self.config = {
            "engine": {"type": "drama3_mlx", "checkpoint": str(self.checkpoint)},
            "verification": {"type": "mlx_whisper", "model": str(self.root / "whisper-large-v3-turbo")},
            "voice_design": {"type": "engine"},
        }
        self.output = io.StringIO()
        self.downloader = Mock(side_effect=lambda **kw: complete(Path(kw["local_dir"])))

    def setup(self, answer="y\n", **kwargs):
        return ModelSetup(self.config, input_stream=Terminal(answer), output_stream=self.output,
                          downloader=self.downloader, **kwargs)

    def test_accept_downloads_pinned_model_and_reports_destination(self):
        self.setup().ensure("engine")
        self.downloader.assert_called_once_with(repo_id=DRAMA.repo, revision=DRAMA.revision,
                                                local_dir=str(self.checkpoint), token=False, max_workers=5)
        for text in (DRAMA.repo, "41.5 GB", str(self.checkpoint), "[y/N]", "is ready"):
            self.assertIn(text, self.output.getvalue())

    def test_file_configured_api_key_is_only_passed_to_downloader(self):
        self.config["huggingface"] = {"api_key": "hf_secret/private+key"}
        self.setup().ensure("engine")
        self.assertEqual(self.downloader.call_args.kwargs["token"], "hf_secret/private+key")
        self.assertNotIn("hf_secret", self.output.getvalue())

    def test_complete_checkpoint_never_prompts_downloads_or_writes(self):
        complete(self.checkpoint)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.checkpoint.iterdir()}
        setup = ModelSetup(self.config, input_stream=io.StringIO(), output_stream=self.output,
                           downloader=self.downloader)
        setup.ensure("engine")
        setup.ensure("engine")
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.checkpoint.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(self.output.getvalue(), "")
        self.downloader.assert_not_called()

    def test_decline_default_eof_and_invalid_answer_never_download(self):
        for answer in ("n\n", "\n", "", "sure\n"):
            with self.subTest(answer=answer), self.assertRaisesRegex(ModelSetupError, "declined") as error:
                self.setup(answer).ensure("engine")
            self.assertIn("hf download fishaudio/drama-3", str(error.exception))
            self.assertIn(DRAMA.revision, str(error.exception))
        self.downloader.assert_not_called()

    def test_explicit_noninteractive_stream_does_not_consume_input_or_open_terminal(self):
        source = io.StringIO("yes\n")
        setup = ModelSetup(self.config, input_stream=source, output_stream=self.output, downloader=self.downloader)
        with patch("builtins.open") as opened, self.assertRaisesRegex(ModelSetupError, "No interactive"):
            setup.ensure("engine")
        self.assertEqual(source.tell(), 0)
        opened.assert_not_called()
        self.downloader.assert_not_called()

    def test_redirected_stdin_can_ask_on_controlling_terminal(self):
        setup = ModelSetup(self.config, output_stream=self.output, downloader=self.downloader)
        terminal = Terminal("yes\n")
        with patch("sys.stdin", io.StringIO("piped data")), patch("builtins.open", return_value=terminal) as opened:
            setup.ensure("engine")
        opened.assert_called_once_with("/dev/tty", "r", encoding="utf-8")
        self.downloader.assert_called_once()

    def test_missing_controlling_terminal_fails_with_manual_instructions(self):
        setup = ModelSetup(self.config, output_stream=self.output, downloader=self.downloader)
        with patch("sys.stdin", io.StringIO()), patch("builtins.open", side_effect=OSError("no tty")):
            with self.assertRaisesRegex(ModelSetupError, "interactive terminal") as error:
                setup.ensure("engine")
        self.assertIn("--local-dir", str(error.exception))
        self.downloader.assert_not_called()

    def test_decline_cached_but_manual_install_recovers_in_same_session(self):
        setup = self.setup("n\ny\n")
        for _ in range(2):
            with self.assertRaises(ModelSetupError):
                setup.ensure("engine")
        self.assertEqual(self.output.getvalue().count("[y/N]"), 1)
        complete(self.checkpoint)
        setup.ensure("engine")
        self.downloader.assert_not_called()

    def test_missing_and_empty_required_assets_are_repaired(self):
        for name in ("model-2.safetensors", "tokenizer.json", "codec.pth", "config.json"):
            with self.subTest(name=name):
                complete(self.checkpoint)
                (self.checkpoint / name).write_bytes(b"")
                self.downloader.reset_mock()
                self.setup().ensure("engine")
                self.downloader.assert_called_once()
                self.assertGreater((self.checkpoint / name).stat().st_size, 0)

    def test_partial_download_failure_retains_files_and_retries_in_new_session(self):
        def interrupted(**kwargs):
            files(Path(kwargs["local_dir"]), {"model-1.safetensors": "saved shard"})
            raise RuntimeError("connection reset while fetching shard 2")
        self.downloader.side_effect = interrupted
        setup = self.setup()
        with self.assertRaisesRegex(ModelSetupError, "connection reset") as error:
            setup.ensure("engine")
        self.assertIn("partial files remain", str(error.exception))
        with self.assertRaises(ModelSetupError):
            setup.ensure("engine")
        self.downloader.assert_called_once()
        self.assertEqual((self.checkpoint / "model-1.safetensors").read_text(), "saved shard")
        self.downloader.side_effect = lambda **kw: complete(Path(kw["local_dir"]))
        self.setup().ensure("engine")
        self.assertEqual(self.downloader.call_count, 2)

    def test_error_details_are_preserved_but_keys_and_controls_are_redacted(self):
        self.config["huggingface"] = {"api_key": "hf_private+key"}
        self.downloader.side_effect = RuntimeError("HTTP 403: gated repository; token=hf_private+key; encoded=hf_private%2Bkey\x1b[31m denied")
        with self.assertRaises(ModelSetupError) as error:
            self.setup().ensure("engine")
        message = str(error.exception)
        self.assertIn("HTTP 403: gated repository", message)
        self.assertIn("[REDACTED]", message)
        self.assertNotIn("hf_private", message)
        self.assertNotIn("\x1b", message)

    def test_download_must_leave_complete_checkpoint(self):
        self.downloader.side_effect = lambda **kw: files(Path(kw["local_dir"]), {"config.json": {"model_type": "fish_omni"}})
        with self.assertRaisesRegex(ModelSetupError, "still incomplete"):
            self.setup().ensure("engine")

    def test_keyboard_interrupt_preserves_resume_guidance_and_no_retry_in_session(self):
        self.downloader.side_effect = KeyboardInterrupt
        setup = self.setup()
        with self.assertRaises(KeyboardInterrupt):
            setup.ensure("engine")
        self.assertIn("download interrupted", self.output.getvalue())
        with self.assertRaisesRegex(ModelSetupError, "download interrupted"):
            setup.ensure("engine")
        self.downloader.assert_called_once()

    def test_different_model_metadata_is_never_overwritten(self):
        files(self.checkpoint, {"config.json": {"model_type": "qwen3_tts"}})
        with self.assertRaisesRegex(ModelSetupError, "different model"):
            self.setup().ensure("engine")
        self.assertNotIn("[y/N]", self.output.getvalue())
        self.downloader.assert_not_called()

    def test_different_architecture_without_model_type_is_never_overwritten(self):
        files(self.checkpoint, {"config.json": {"architectures": ["LlamaForCausalLM"]}})
        with self.assertRaisesRegex(ModelSetupError, "incompatible architecture"):
            self.setup().ensure("engine")
        self.downloader.assert_not_called()

    def test_malformed_model_type_has_actionable_error_without_download(self):
        for value in ([], {}, 5):
            with self.subTest(model_type=value):
                files(self.checkpoint, {"config.json": {"model_type": value}})
                with self.assertRaisesRegex(ModelSetupError, "choose the correct directory"):
                    self.setup().ensure("engine")
        self.downloader.assert_not_called()

    def test_downloader_package_is_imported_only_after_acceptance(self):
        setup = ModelSetup(self.config, input_stream=Terminal("y\n"), output_stream=self.output)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with self.assertRaisesRegex(ModelSetupError, "downloader is not installed"):
                setup.ensure("engine")

    def test_raised_eof_is_treated_as_decline(self):
        source = Terminal()
        source.readline = Mock(side_effect=EOFError)
        setup = ModelSetup(self.config, input_stream=source, output_stream=self.output,
                           downloader=self.downloader)
        with self.assertRaisesRegex(ModelSetupError, "declined"):
            setup.ensure("engine")
        self.downloader.assert_not_called()

    def test_invalid_index_path_is_rejected(self):
        complete(self.checkpoint)
        files(self.checkpoint, {"model.safetensors.index.json": {"weight_map": {"weight": "../secret.safetensors"}}})
        with self.assertRaisesRegex(ModelSetupError, "Unsafe model shard"):
            self.setup().ensure("engine")
        self.downloader.assert_not_called()

    def test_converted_complete_drama_is_accepted_but_never_filled_with_release_files(self):
        files(self.checkpoint, {"config.json": {"model_type": "fish_omni", "drama3_mlx": {"version": 1}},
                               "tokenizer.json": "tokens", "codec.safetensors": "codec", "model.safetensors": "model"})
        self.setup().ensure("engine")
        (self.checkpoint / "model.safetensors").unlink()
        with self.assertRaisesRegex(ModelSetupError, "Converted Drama"):
            self.setup().ensure("engine")
        self.downloader.assert_not_called()

    def test_configured_model_components_are_downloaded_lazily(self):
        self.config["verification"]["reference_model"] = str(self.root / "parakeet-tdt-0.6b-v3")
        self.config["voice_design"] = {"type": "qwen3_mlx", "checkpoint": str(self.root / "qwen-design")}
        specs = {m.repo: m for m in (DRAMA, WHISPER, PARAKEET, QWEN)}
        self.downloader.side_effect = lambda **kw: complete(Path(kw["local_dir"]), specs[kw["repo_id"]].kind)
        setup = self.setup("y\ny\ny\n")
        setup.ensure("verification")
        self.assertEqual([call.kwargs["repo_id"] for call in self.downloader.call_args_list], [WHISPER.repo, PARAKEET.repo])
        self.assertFalse(self.checkpoint.exists())
        setup.ensure("voice_design")
        self.assertEqual(self.downloader.call_args.kwargs["repo_id"], QWEN.repo)
        self.assertFalse(self.checkpoint.exists())

    def test_custom_complete_asr_is_accepted_but_missing_unknown_asr_is_not_guessed(self):
        custom = self.root / "my-custom-asr"
        self.config["verification"]["model"] = str(custom)
        with self.assertRaisesRegex(ModelSetupError, "cannot be inferred"):
            self.setup().ensure("verification")
        complete(custom, "whisper")
        self.setup().ensure("verification")
        self.downloader.assert_not_called()

    def cache_snapshot(self, model=WHISPER, revision=None, *, repo=None):
        hub = self.root / "huggingface hub"
        return (hub / ("models--" + (repo or model.repo).replace("/", "--"))
                / "snapshots" / (revision or model.revision))

    def configure_snapshot(self, model, path):
        if model == DRAMA:
            self.config["engine"]["checkpoint"] = str(path)
            return "engine"
        if model == QWEN:
            self.config["voice_design"] = {"type": "qwen3_mlx", "checkpoint": str(path)}
            return "voice_design"
        if model == PARAKEET:
            primary = self.root / "existing-whisper"
            complete(primary, "whisper")
            self.config["verification"]["model"] = str(primary)
            self.config["verification"]["reference_model"] = str(path)
        else:
            self.config["verification"]["model"] = str(path)
        return "verification"

    def download_snapshot(self, **kwargs):
        model = next(m for m in (DRAMA, WHISPER, PARAKEET, QWEN) if m.repo == kwargs["repo_id"])
        path = (Path(kwargs["cache_dir"]) / ("models--" + model.repo.replace("/", "--"))
                / "snapshots" / kwargs["revision"])
        complete(path, model.kind)
        return str(path)

    def test_missing_known_hub_snapshots_are_downloaded_through_cache(self):
        self.downloader.side_effect = self.download_snapshot
        for model in (WHISPER, DRAMA, PARAKEET, QWEN):
            with self.subTest(model=model.repo):
                path = self.cache_snapshot(model)
                component = self.configure_snapshot(model, path)
                self.downloader.reset_mock()
                self.setup().ensure(component)
                self.downloader.assert_called_once_with(
                    repo_id=model.repo, revision=model.revision,
                    cache_dir=str(path.parents[2]), token=False, max_workers=5)
                self.assertTrue((path / "config.json").is_file())
                self.assertFalse((path / ".cache").exists())
                self.assertIn(str(path), self.output.getvalue())

    def test_snapshot_uses_explicit_commit_and_file_configured_token(self):
        revision = "b" * 40
        self.assertNotEqual(revision, WHISPER.revision)
        path = self.cache_snapshot(revision=revision)
        self.configure_snapshot(WHISPER, path)
        self.config["huggingface"] = {"api_key": "hf_snapshot_private_key"}
        self.downloader.side_effect = self.download_snapshot
        self.setup().ensure("verification")
        self.downloader.assert_called_once_with(
            repo_id=WHISPER.repo, revision=revision,
            cache_dir=str(path.parents[2]), token="hf_snapshot_private_key", max_workers=5)
        self.assertFalse(self.cache_snapshot().exists())
        self.assertNotIn("hf_snapshot_private_key", self.output.getvalue())

    def test_missing_hub_blobs_are_repaired_without_replacing_snapshot_symlinks(self):
        path = self.cache_snapshot()
        self.configure_snapshot(WHISPER, path)
        path.mkdir(parents=True)
        (path / "config.json").symlink_to("../../blobs/config-blob")
        (path / "weights.safetensors").symlink_to("../../blobs/weights-blob")
        self.assertFalse((path / "config.json").exists())

        def repair_blobs(**kwargs):
            self.assertEqual(kwargs["cache_dir"], str(path.parents[2]))
            self.assertNotIn("local_dir", kwargs)
            files(path.parents[1] / "blobs", {
                "config-blob": {"model_type": "whisper"}, "weights-blob": "weights"})
            return str(path)

        self.downloader.side_effect = repair_blobs
        self.setup().ensure("verification")
        self.downloader.assert_called_once()
        self.assertTrue((path / "config.json").is_symlink())
        self.assertEqual((path / "weights.safetensors").read_text(), "weights")

    def test_complete_hub_snapshot_never_prompts_downloads_or_writes(self):
        path = self.cache_snapshot(revision="c" * 40)
        self.configure_snapshot(WHISPER, path)
        complete(path, "whisper")
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        self.setup().ensure("verification")
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(self.output.getvalue(), "")
        self.downloader.assert_not_called()

    def test_declining_hub_download_preserves_exact_manual_cache_command(self):
        revision = "d" * 40
        path = self.cache_snapshot(revision=revision)
        self.configure_snapshot(WHISPER, path)
        with self.assertRaisesRegex(ModelSetupError, "declined") as error:
            self.setup("n\n").ensure("verification")
        command = shlex.split(str(error.exception).splitlines()[-1])
        self.assertEqual(command, [".venv-renderer/bin/hf", "download", WHISPER.repo,
                                   "--revision", revision, "--cache-dir", str(path.parents[2])])
        self.assertFalse(path.parents[2].exists())
        self.downloader.assert_not_called()

    def test_wrong_hub_repository_is_never_guessed_or_written(self):
        for model in (WHISPER, DRAMA, PARAKEET, QWEN):
            with self.subTest(model=model.repo):
                path = self.cache_snapshot(model, repo="another-owner/" + model.repo.split("/")[-1])
                component = self.configure_snapshot(model, path)
                with self.assertRaises(ModelSetupError):
                    self.setup().ensure(component)
                self.assertFalse(path.parents[2].exists())
        self.assertNotIn("[y/N]", self.output.getvalue())
        self.downloader.assert_not_called()

    def test_malformed_hub_snapshot_revision_is_never_guessed_or_written(self):
        for model in (WHISPER, DRAMA, PARAKEET, QWEN):
            for revision in ("main", "a" * 39, "A" * 40, "g" * 40):
                with self.subTest(model=model.repo, revision=revision):
                    path = self.cache_snapshot(model, revision)
                    component = self.configure_snapshot(model, path)
                    with self.assertRaises(ModelSetupError):
                        self.setup().ensure(component)
                    self.assertFalse(path.parents[2].exists())
        self.assertNotIn("[y/N]", self.output.getvalue())
        self.downloader.assert_not_called()

    def test_hub_snapshot_download_failure_retains_files_and_uses_cache_on_retry(self):
        path = self.cache_snapshot()
        self.configure_snapshot(WHISPER, path)

        def interrupted(**kwargs):
            self.assertNotIn("local_dir", kwargs)
            files(path, {"config.json": {"model_type": "whisper"}})
            raise RuntimeError("connection reset downloading cached weights")

        self.downloader.side_effect = interrupted
        setup = self.setup()
        with self.assertRaisesRegex(ModelSetupError, "connection reset") as error:
            setup.ensure("verification")
        self.assertIn("--cache-dir", str(error.exception))
        self.assertTrue((path / "config.json").is_file())
        with self.assertRaises(ModelSetupError):
            setup.ensure("verification")
        self.downloader.assert_called_once()
        self.downloader.side_effect = self.download_snapshot
        self.setup().ensure("verification")
        self.assertEqual(self.downloader.call_count, 2)
        self.assertTrue((path / "weights.safetensors").is_file())

    def test_temporary_designer_settings_are_respected(self):
        setup = self.setup()
        setup.ensure("voice_design", settings={"type": "gemini"})
        setup.ensure("engine", settings={"type": "third-party"})
        self.downloader.assert_not_called()

    def test_existing_file_path_is_not_replaced_with_directory(self):
        self.checkpoint.write_text("keep me")
        with self.assertRaisesRegex(ModelSetupError, "not a directory"):
            self.setup().ensure("engine")
        self.assertEqual(self.checkpoint.read_text(), "keep me")
        self.downloader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
