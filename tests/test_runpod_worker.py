"""The deployment launcher must not download implicitly or strand CUDA workers."""

from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
import wave
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.config import DEFAULTS, load_config


_path = Path(__file__).resolve().parents[1] / "deploy/runpod/worker.py"
_spec = importlib.util.spec_from_file_location("runpod_worker", _path)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def fake_download(**kwargs):
    directory = Path(kwargs["local_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if kwargs["repo_id"] == "fishaudio/drama-3":
        files = ["codec.pth", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "chat_template.jinja", "model-00001-of-00001.safetensors"]
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"fixture.weight": "model-00001-of-00001.safetensors"}}))
    else:
        files = ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"]
    for name in files:
        (directory / name).write_bytes(b"fixture model")
    return str(directory)


class RunpodWorkerTests(unittest.TestCase):
    def setUp(self):
        # Subprocess assertions should never capture the developer's ambient
        # credentials in a unittest failure. Tests opt into needed values.
        patch.dict(os.environ, {"PATH": os.defpath}, clear=True).start()
        self.addCleanup(patch.stopall)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama")
        self.config["verification"].update(type="faster_whisper", device="cpu",
                                            model=str(self.root / "whisper"))
        self.config["chapter_server"].update(data_dir=str(self.root / "data"), api_key="test-service-key")
        self.config["render_pool"]["workers"] = [{
            "name": "native", "type": "fish_http", "protocol": "fish_native_drama3",
            "url": worker.FISH_URL, "concurrency": 1, "api_key": ""}]
        self.downloader = Mock(side_effect=fake_download)
        self.real_check_audio_decoder = worker.check_audio_decoder
        self.decoder_check = patch.object(worker, "check_audio_decoder").start()
        self.addCleanup(patch.stopall)

    def prepare(self):
        with patch("sys.stdout", new=io.StringIO()):
            worker.prepare(self.config, downloader=self.downloader)

    def native_image(self):
        native = self.root / "fish"
        for name in (".venv/bin/python", "tools/api_server.py"):
            path = native / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        application = self.root / "app/.venv/bin/python"
        application.parent.mkdir(parents=True, exist_ok=True)
        application.touch()
        return {"fish_source": native, "application_python": application}

    def test_prepare_pins_both_repositories_and_never_uses_ambient_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "ambient-secret", "HF_ENDPOINT": "https://invalid.example"}):
            self.prepare()
        self.assertEqual(self.downloader.call_count, 2)
        fish, whisper = [call.kwargs for call in self.downloader.call_args_list]
        self.assertEqual(fish["revision"], self.config["render_pool"]["model_revision"])
        self.assertEqual(whisper["revision"], worker.WHISPER_REVISION)
        self.assertEqual(whisper["repo_id"], worker.WHISPER_REPOSITORY)
        for args in (fish, whisper):
            self.assertIs(args["token"], False)
            self.assertEqual(args["endpoint"], "https://huggingface.co")
        worker.check(self.config)

    def test_configured_download_key_is_passed_explicitly(self):
        self.config["huggingface"]["api_key"] = "file-key"
        self.prepare()
        self.assertTrue(all(call.kwargs["token"] == "file-key" for call in self.downloader.call_args_list))

    def test_audio_decoder_preflight_uses_real_pcm_without_loading_a_model(self):
        def decode(audio, sampling_rate):
            self.assertEqual(sampling_rate, 16000)
            with wave.open(audio, "rb") as reader:
                self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 16000))
                frames = reader.readframes(reader.getnframes())
            return [int.from_bytes(frames[i:i + 2], "little", signed=True) / 32768
                    for i in range(0, len(frames), 2)]
        self.real_check_audio_decoder(decoder=decode)

    def test_audio_decoder_dependency_failure_is_actionable(self):
        decoder = Mock(side_effect=TypeError("open() got an unexpected keyword argument 'metadata_errors'"))
        with self.assertRaisesRegex(RuntimeError, "full worker update.*metadata_errors"):
            self.real_check_audio_decoder(decoder=decoder)

    def test_worker_check_tests_audio_decoding_before_loading_models(self):
        self.prepare()
        self.decoder_check.side_effect = RuntimeError("incompatible decoder")
        with self.assertRaisesRegex(RuntimeError, "incompatible decoder"), \
                patch.object(worker, "required_model_files") as files:
            worker.check(self.config)
        self.decoder_check.assert_called_once_with()
        files.assert_not_called()

    def test_installed_whisper_can_decode_pcm_without_a_model(self):
        if importlib.util.find_spec("faster_whisper") is None:
            self.skipTest("faster-whisper is an optional worker dependency")
        self.real_check_audio_decoder()

    def test_prepared_models_are_reused_without_download_requests(self):
        self.prepare()
        self.downloader.reset_mock()
        self.prepare()
        self.downloader.assert_not_called()

    def test_launcher_paths_follow_the_installed_checkout(self):
        self.assertEqual(worker.APPLICATION_PYTHON, _path.parents[2] / ".venv/bin/python")
        self.assertEqual(worker.FISH_SOURCE, _path.parents[3] / "fish-speech")

    def test_runtime_installer_reuses_environments_without_package_resolution(self):
        commands = self.root / "commands"
        commands.mkdir()
        log = self.root / "runtime.log"
        fixtures = {
            "uname": 'case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac',
            "id": "echo 0",
            "dpkg-query": "printf installed",
            "apt-get": "exit 91",
            "uv": "exit 92",
        }
        for name, source in fixtures.items():
            executable = commands / name
            executable.write_text("#!/bin/sh\n" + source + "\n")
            executable.chmod(0o755)
        app, fish = self.root / "app", self.root / "fish"
        for directory in (app, fish):
            python = directory / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"" + str(log) + "\"\n")
            python.chmod(0o755)
        (app / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        result = subprocess.run(
            ["/bin/sh", str(_path.parent / "install.sh"), str(app), "runtime",
             str(self.root / "runtime"), str(fish)], capture_output=True, text=True,
            env={**os.environ, "PATH": str(commands) + ":" + os.environ.get("PATH", "")})
        self.assertEqual(result.returncode, 0, result.stderr)
        imports = log.read_text().splitlines()
        self.assertEqual(len(imports), 2)
        self.assertIn("faster_whisper", imports[0])
        self.assertIn("torch", imports[1])
        self.assertFalse((self.root / "runtime").exists())
        # SGLang uses its own existing environment; native is not imported or
        # loaded as a second copy when the batching backend is selected.
        result = subprocess.run(
            ["/bin/sh", str(_path.parent / "install.sh"), str(app), "runtime",
             str(self.root / "runtime"), str(self.root / "missing-native"), "sglang", str(fish)],
            capture_output=True, text=True,
            env={**os.environ, "PATH": str(commands) + ":" + os.environ.get("PATH", "")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sglang_omni", log.read_text().splitlines()[-1])
        self.assertFalse((self.root / "runtime").exists())
        # Runtime startup cannot silently use an old CUDA verifier environment.
        (app / ".venv/bin/python").write_text("#!/bin/sh\nexit 1\n")
        result = subprocess.run(
            ["/bin/sh", str(_path.parent / "install.sh"), str(app), "runtime",
             str(self.root / "runtime"), str(fish)], capture_output=True, text=True,
            env={**os.environ, "PATH": str(commands) + ":" + os.environ.get("PATH", "")})
        self.assertEqual(result.returncode, 1)
        self.assertIn("full worker update", result.stderr)

    def test_example_config_loads_but_requires_replacing_predictable_application_key(self):
        template = _path.parent / "epub-tts-narrator.example.toml"
        config = load_config(template)
        with self.assertRaisesRegex(ValueError, "new random application key"):
            worker.validate_worker(config)
        config["chapter_server"]["api_key"] = "random-fixture-token"
        worker.validate_worker(config)

    def test_blank_and_placeholder_application_keys_cannot_expose_service(self):
        for value in ("", "REPLACE-WITH-YOUR-KEY", "replace-with-another-key"):
            with self.subTest(value=value):
                config = deepcopy(self.config)
                config["chapter_server"]["api_key"] = value
                with self.assertRaisesRegex(ValueError, "new random application key"):
                    worker.validate_worker(config)

    def test_check_is_offline_and_detects_changed_model_files(self):
        self.prepare()
        before = self.downloader.call_count
        model = Path(self.config["engine"]["checkpoint"]) / "codec.pth"
        model.write_bytes(b"changed codec")
        with self.assertRaisesRegex(ValueError, "files or revision changed"):
            worker.check(self.config)
        self.assertEqual(self.downloader.call_count, before)

    def test_revision_change_requires_fresh_directory(self):
        self.prepare()
        self.downloader.reset_mock()
        self.config["render_pool"]["model_revision"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "new model directory"):
            worker.prepare(self.config, downloader=self.downloader)
        self.downloader.assert_not_called()

    def test_failed_download_has_no_completion_receipt_and_can_resume(self):
        def failing(**kwargs):
            fake_download(**kwargs)
            raise RuntimeError("network failed")
        with patch("sys.stdout", new=io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "partial downloads remain"):
                worker.prepare(self.config, downloader=failing)
        self.assertTrue((self.root / "drama/codec.pth").exists())
        self.assertFalse((self.root / "drama" / worker.RECEIPT).exists())
        self.prepare()
        worker.check(self.config)

    def test_missing_model_never_launches_or_downloads_during_serve(self):
        with patch.object(worker.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "not prepared"):
                worker.serve(self.config, self.root / "config.toml")
        popen.assert_not_called()

    def test_incomplete_download_is_not_marked_prepared(self):
        with patch("sys.stdout", new=io.StringIO()):
            with self.assertRaises((ValueError, OSError)):
                worker.prepare(self.config, downloader=lambda **kwargs: None)
        self.assertFalse((self.root / "drama" / worker.RECEIPT).exists())

    def test_untrusted_index_cannot_escape_checkpoint_directory(self):
        self.prepare()
        index = self.root / "drama/model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"x": "../outside"}}))
        with self.assertRaisesRegex(ValueError, "invalid shard"):
            worker.required_model_files("fishaudio/drama-3", index.parent)

    def test_deployment_profile_rejects_remote_backend_duplicate_gpu_and_mlx(self):
        variants = []
        for key, value in (("url", "https://external.example/v1/tts"), ("concurrency", 2),
                           ("protocol", "fish_compatible"), ("api_key", "unexpected")):
            item = deepcopy(self.config)
            item["render_pool"]["workers"][0][key] = value
            variants.append(item)
        for key, value in (("type", "mlx_whisper"), ("device", "mps"), ("reference_model", "/model")):
            item = deepcopy(self.config)
            item["verification"][key] = value
            variants.append(item)
        for item in variants:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    worker.validate_worker(item)

    def test_worker_lease_prevents_two_model_processes_for_same_store(self):
        with worker.worker_lease(self.config):
            with self.assertRaisesRegex(ValueError, "already uses"):
                with worker.worker_lease(self.config):
                    self.fail("second worker acquired the same lease")

    def test_ready_requires_native_drama3_model_family(self):
        process = Mock()
        process.poll.return_value = None
        fetch = Mock(side_effect=[{"status": "ok"}, {"model_family": "drama3"}])
        worker.wait_ready(process, threading.Event(), fetch=fetch)
        self.assertEqual(fetch.call_count, 2)
        fetch = Mock(side_effect=[{"status": "ok"}, {"model_family": "other"}])
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        with self.assertRaises(worker.WorkerStopped):
            worker.wait_ready(process, stop, fetch=fetch)

    def test_early_model_exit_is_detected_before_starting_chapter_service(self):
        process = Mock()
        process.poll.return_value = 2
        fetch = Mock()
        with self.assertRaisesRegex(RuntimeError, "during startup"):
            worker.wait_ready(process, threading.Event(), fetch=fetch)
        fetch.assert_not_called()

    def batched_image(self):
        self.config["render_pool"]["workers"][0].update(
            protocol="fish_sglang_drama3", url=worker.SGLANG_URL,
            concurrency=4, multi_speaker=False)
        root = self.root / "sglang"
        executable = root / ".venv/bin/sgl-omni"
        executable.parent.mkdir(parents=True)
        executable.touch()
        return root

    def test_batched_server_is_one_process_with_loopback_and_local_codec(self):
        root = self.batched_image()
        worker.validate_worker(self.config)
        command, cwd = worker.inference_command(self.config, sglang_source=root)
        self.assertEqual(cwd, root)
        self.assertEqual(command[0], str(root / ".venv/bin/sgl-omni"))
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--tts_engine.factory.max_running_requests") + 1], "4")
        self.assertEqual(command[command.index("--tts_engine.gpu_memory_fraction") + 1], "0.70")
        self.assertEqual(command[command.index("--vocoder.factory.codec_path") + 1],
                         str(self.root / "drama/codec.pth"))
        self.assertNotIn("--workers", command)
        self.config["verification"].update(device="cuda", compute_type="int8_float16")
        worker.validate_worker(self.config)
        cuda_command, _ = worker.inference_command(self.config, sglang_source=root)
        self.assertEqual(cuda_command[cuda_command.index("--tts_engine.gpu_memory_fraction") + 1], "0.60")
        self.assertNotIn("--tts_engine.engine.mem_fraction_static", cuda_command)
        for update in ({"concurrency": 9}, {"multi_speaker": True}, {"url": worker.FISH_URL}):
            with self.subTest(update=update):
                invalid = deepcopy(self.config)
                invalid["render_pool"]["workers"][0].update(update)
                with self.assertRaises(ValueError):
                    worker.validate_worker(invalid)

    def test_batched_readiness_checks_health_and_loaded_model(self):
        process = Mock()
        process.poll.return_value = None
        fetch = Mock(side_effect=[{"status": "healthy", "running": True},
                                  {"data": [{"id": "drama-3"}]}])
        worker.wait_ready(process, threading.Event(), protocol="fish_sglang_drama3", fetch=fetch)
        self.assertEqual([call.args[0] for call in fetch.call_args_list],
                         ["http://127.0.0.1:8080/health", "http://127.0.0.1:8080/v1/models"])
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        wrong = Mock(side_effect=[{"status": "healthy", "running": True},
                                  {"data": [{"id": "other"}]}])
        with self.assertRaises(worker.WorkerStopped):
            worker.wait_ready(process, stop, protocol="fish_sglang_drama3", fetch=wrong)

    def test_backend_environment_finds_venv_tools_and_preserves_compile_caches(self):
        root = self.batched_image()
        for name in ("ninja", "python"):
            executable = root / ".venv/bin" / name
            executable.write_text(f"#!/bin/sh\nprintf 'selected-{name}'\n")
            executable.chmod(0o755)
        inherited = {"PATH": "/usr/bin:/bin", "HOME": "/unchanged-home",
                     "TORCHINDUCTOR_CACHE_DIR": "/ephemeral-cache", "KEEP_ME": "fixture"}
        config_before = deepcopy(self.config)
        with patch.dict(os.environ, inherited, clear=True):
            environment = worker.inference_environment(self.config, root)
            self.assertEqual(dict(os.environ), inherited)
        self.assertEqual(environment["PATH"], str(root / ".venv/bin") + ":/usr/bin:/bin")
        self.assertEqual(environment["HOME"], "/unchanged-home")
        self.assertEqual(environment["KEEP_ME"], "fixture")
        self.assertEqual(self.config, config_before)
        for name in ("ninja", "python"):
            result = subprocess.run([name], env=environment, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, "selected-" + name)
        cache = self.root / "data/runtime-cache/sglang"
        expected = {
            "SGLANG_CACHE_DIR": cache, "SGLANG_JIT_CACHE_DIR": cache / "jit",
            "TORCHINDUCTOR_CACHE_DIR": cache / "inductor", "TRITON_CACHE_DIR": cache / "triton",
            "FLASHINFER_WORKSPACE_BASE": cache, "CUDA_CACHE_PATH": cache / "nv"}
        for key, directory in expected.items():
            self.assertEqual(environment[key], str(directory))
            self.assertTrue(directory.is_dir())
        marker = cache / "triton/compiled-fixture"
        marker.write_bytes(b"keep across launches")
        worker.inference_environment(self.config, root)
        self.assertEqual(marker.read_bytes(), b"keep across launches")

    def test_native_backend_environment_selects_its_venv_without_sglang_cache_controls(self):
        root = self.native_image()["fish_source"]
        with patch.dict(os.environ, {}, clear=True):
            environment = worker.inference_environment(self.config, root)
        self.assertEqual(environment["PATH"], str(root / ".venv/bin") + os.pathsep + os.defpath)
        self.assertEqual(set(environment), {"PATH"})
        self.assertFalse((self.root / "data/runtime-cache").exists())

    def cuda_runtime(self):
        python = self.native_image()["application_python"]
        directory = python.parent.parent / "lib/python3.12/site-packages/nvidia/cublas/lib"
        directory.mkdir(parents=True)
        for name in ("libcublas.so.12", "libcublasLt.so.12"):
            (directory / name).touch()
        return python, directory

    def test_cuda_library_probe_requires_both_pinned_libraries_in_application_venv(self):
        python, directory = self.cuda_runtime()
        result = Mock(returncode=0, stdout=json.dumps(str(directory)))
        with patch.object(worker.subprocess, "run", return_value=result) as probe:
            self.assertEqual(worker.cuda_library_directory(python), directory)
            self.assertEqual(probe.call_args.args[0][0], str(python))
            self.assertIn("4.8.3", probe.call_args.args[0][2])
            self.assertIn("12.8.4.1", probe.call_args.args[0][2])
            (directory / "libcublasLt.so.12").unlink()
            with self.assertRaisesRegex(RuntimeError, "full worker update"):
                worker.cuda_library_directory(python)
            result.stdout = json.dumps("/usr/local/cuda/lib64")
            with self.assertRaisesRegex(RuntimeError, "full worker update"):
                worker.cuda_library_directory(python)
            result.returncode = 1
            result.stdout = ""
            with self.assertRaisesRegex(RuntimeError, "pinned CTranslate2"):
                worker.cuda_library_directory(python)

    def test_cuda_loader_path_is_application_only_and_does_not_mutate_parent(self):
        root = self.batched_image()
        python, directory = self.cuda_runtime()
        self.config["verification"].update(device="cuda", compute_type="int8_float16")
        inherited = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/driver/lib:/other/lib", "KEEP": "value"}
        with (patch.dict(os.environ, inherited, clear=True),
              patch.object(worker, "cuda_library_directory", return_value=directory)):
            application = worker.application_environment(self.config, application_python=python)
            self.assertEqual(dict(os.environ), inherited)
        self.assertEqual(application["LD_LIBRARY_PATH"], str(directory) + ":/driver/lib:/other/lib")
        with patch.dict(os.environ, application, clear=True):
            backend = worker.inference_environment(self.config, root, application_python=python)
            self.assertEqual(dict(os.environ), application)
        self.assertEqual(backend["LD_LIBRARY_PATH"], "/driver/lib:/other/lib")
        self.assertEqual(backend["KEEP"], "value")
        # An existing correctly prepared application launch is idempotent.
        with (patch.dict(os.environ, application, clear=True),
              patch.object(worker, "cuda_library_directory", return_value=directory)):
            self.assertEqual(worker.application_environment(self.config, application_python=python), application)
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(directory)}, clear=True):
            self.assertNotIn("LD_LIBRARY_PATH", worker.inference_environment(
                self.config, root, application_python=python))

    def test_cuda_preflight_checks_hardware_without_loading_whisper_or_cpu_fallback(self):
        self.config["verification"].update(device="cuda", compute_type="int8_float16")
        environment = {"LD_LIBRARY_PATH": "/application/cublas/lib"}
        result = Mock(returncode=0, stdout=json.dumps(["float16", "int8_float16"]), stderr="")
        with (patch.object(worker, "application_environment", return_value=environment),
              patch.object(worker.subprocess, "run", return_value=result) as probe):
            worker.check_cuda_verification(self.config)
            command = probe.call_args.args[0]
            self.assertIn("get_supported_compute_types('cuda', 0)", command[-1])
            self.assertNotIn("WhisperModel", command[-1])
            self.assertEqual(probe.call_args.kwargs["env"], environment)
            self.config["verification"]["compute_type"] = "bfloat16"
            with self.assertRaisesRegex(ValueError, "does not support.*bfloat16"):
                worker.check_cuda_verification(self.config)
            result.returncode, result.stderr = 1, "RuntimeError: No visible CUDA device"
            with self.assertRaisesRegex(RuntimeError, "No visible CUDA device.*CPU fallback is disabled"):
                worker.check_cuda_verification(self.config)
            self.assertEqual(self.config["verification"]["device"], "cuda")
        self.config["verification"]["device"] = "cpu"
        with patch.object(worker.subprocess, "run") as probe:
            worker.check_cuda_verification(self.config)
        probe.assert_not_called()

    def test_application_wrapper_preserves_argument_boundaries_and_executes_selected_python(self):
        self.config["verification"]["device"] = "cuda"
        arguments = ["/workspace/script.py", "--config", "path with spaces", "literal $(value)"]
        environment = {"LD_LIBRARY_PATH": "/application/cublas/lib"}
        with (patch.object(worker, "application_environment", return_value=environment),
              patch.object(worker.os, "execve") as execute):
            worker.run_application(self.config, arguments, application_python=Path("/app/python"))
        execute.assert_called_once_with("/app/python", ["/app/python", *arguments], environment)
        with self.assertRaisesRegex(ValueError, "requires --"):
            worker.run_application(self.config, [])

    def test_application_cli_forwards_only_arguments_after_separator(self):
        arguments = ["/workspace/benchmark.py", "verify", "manifest.json", "--config", "cuda.toml"]
        with (patch.object(worker, "load_config", return_value=self.config),
              patch.object(worker, "run_application", return_value=17) as launch):
            self.assertEqual(worker.main(["--config", "cuda.toml", "run-application", "--", *arguments]), 17)
        launch.assert_called_once_with(self.config, arguments)

    def test_cuda_smoke_reexecutes_once_with_application_loader_environment(self):
        self.config["verification"]["device"] = "cuda"
        environment = {"LD_LIBRARY_PATH": "/application/cublas/lib"}
        with (patch.object(worker, "load_config", return_value=self.config),
              patch.object(worker, "application_environment", return_value=environment),
              patch.object(worker, "run_application", return_value=0) as launch,
              patch.object(worker, "smoke") as smoke,
              patch.dict(os.environ, {}, clear=True)):
            self.assertEqual(worker.main(["--config", "cuda.toml", "smoke"]), 0)
            launch.assert_called_once_with(self.config, [str(_path), "--config", "cuda.toml", "smoke"])
            smoke.assert_not_called()
            launch.reset_mock()
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(worker.main(["--config", "cuda.toml", "smoke"]), 0)
            launch.assert_not_called()
            smoke.assert_called_once_with(self.config)

    def test_cuda_service_gets_loader_environment_but_tts_does_not(self):
        self.prepare()
        root = self.batched_image()
        python, directory = self.cuda_runtime()
        self.config["verification"].update(device="cuda", compute_type="int8_float16")
        native, chapter = Mock(pid=100001), Mock(pid=100002)
        native.poll.return_value, chapter.poll.return_value = None, 0
        with (patch.object(worker, "check_cuda_verification") as preflight,
              patch.object(worker, "cuda_library_directory", return_value=directory),
              patch.object(worker.subprocess, "Popen", side_effect=[native, chapter]) as popen,
              patch.object(worker, "wait_ready"), patch.object(worker, "stop_children"),
              patch("sys.stdout", new=io.StringIO()),
              patch.dict(os.environ, {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/driver/lib"}, clear=True)):
            self.assertEqual(worker.serve(self.config, self.root / "cuda.toml",
                                         sglang_source=root, application_python=python), 0)
        preflight.assert_called_once_with(self.config, application_python=python)
        self.assertEqual(popen.call_args_list[0].kwargs["env"]["LD_LIBRARY_PATH"], "/driver/lib")
        self.assertEqual(popen.call_args_list[1].kwargs["env"]["LD_LIBRARY_PATH"], str(directory) + ":/driver/lib")

    def test_isolated_tts_mode_never_starts_or_resumes_chapter_jobs(self):
        root = self.batched_image()
        self.prepare()
        inference = Mock(pid=100001)
        inference.poll.return_value = 2
        with (patch.object(worker.subprocess, "Popen", return_value=inference) as popen,
              patch.object(worker, "wait_ready") as ready,
              patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "exited"):
                worker.serve(self.config, self.root / "config.toml", sglang_source=root, chapters=False)
        self.assertEqual(popen.call_count, 1)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(root / ".venv/bin"))
        self.assertEqual(environment["TORCHINDUCTOR_CACHE_DIR"], str(self.root / "data/runtime-cache/sglang/inductor"))
        self.assertEqual(ready.call_args.kwargs["protocol"], "fish_sglang_drama3")
        cleanup.assert_called_once_with([inference])

    def test_process_supervision_uses_one_model_and_cleans_up_both_groups(self):
        self.prepare()
        image = self.native_image()
        native, chapter = Mock(pid=100001), Mock(pid=100002)
        native.poll.return_value = None
        chapter.poll.return_value = 7
        with (patch.object(worker.subprocess, "Popen", side_effect=[native, chapter]) as popen,
              patch.object(worker, "wait_ready") as ready,
              patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            status = worker.serve(self.config, self.root / "config.toml", **image)
        self.assertEqual(status, 7)
        self.assertEqual(popen.call_count, 2)
        native_command, chapter_command = [call.args[0] for call in popen.call_args_list]
        self.assertNotIn("--compile", native_command)
        self.assertEqual(native_command[native_command.index("--workers") + 1], "1")
        self.assertEqual(native_command[native_command.index("--device") + 1], "cuda:0")
        self.assertIn("serve-chapters", chapter_command)
        self.assertNotIn("--maintenance", chapter_command)
        self.assertEqual(popen.call_args_list[0].kwargs["env"]["PATH"].split(os.pathsep)[0],
                         str(image["fish_source"] / ".venv/bin"))
        self.assertEqual(popen.call_args_list[1].kwargs["env"], dict(os.environ))
        self.assertTrue(all(call.kwargs["start_new_session"] for call in popen.call_args_list))
        ready.assert_called_once()
        cleanup.assert_called_once_with([native, chapter])

    def test_maintenance_worker_serves_api_without_starting_job_consumers(self):
        self.prepare()
        image = self.native_image()
        native, chapter = Mock(pid=100001), Mock(pid=100002)
        native.poll.return_value, chapter.poll.return_value = None, 0
        with (patch.object(worker.subprocess, "Popen", side_effect=[native, chapter]) as popen,
              patch.object(worker, "wait_ready"), patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            self.assertEqual(worker.serve(self.config, self.root / "config.toml", maintenance=True, **image), 0)
        self.assertEqual(popen.call_count, 2)
        self.assertIn("--maintenance", popen.call_args_list[1].args[0])
        cleanup.assert_called_once_with([native, chapter])

    def test_cli_maintenance_command_preserves_normal_and_isolated_startup_modes(self):
        for command in ("serve", "serve-maintenance", "serve-tts"):
            with self.subTest(command=command), patch.object(worker, "load_config", return_value=self.config), \
                    patch.object(worker, "serve", return_value=0) as serve:
                self.assertEqual(worker.main(["--config", "fixture.toml", command]), 0)
                self.assertEqual(serve.call_args.kwargs["maintenance"], command == "serve-maintenance")
                self.assertEqual(serve.call_args.kwargs["chapters"], command != "serve-tts")

    def test_failed_model_startup_still_terminates_native_process(self):
        self.prepare()
        image = self.native_image()
        native = Mock(pid=100001)
        with (patch.object(worker.subprocess, "Popen", return_value=native) as popen,
              patch.object(worker, "wait_ready", side_effect=RuntimeError("startup failed")),
              patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                worker.serve(self.config, self.root / "config.toml", **image)
        self.assertEqual(popen.call_count, 1)
        cleanup.assert_called_once_with([native])

    def test_signal_during_startup_prevents_chapter_launch_and_restores_handlers(self):
        self.prepare()
        image = self.native_image()
        native = Mock(pid=100001)
        original = signal.getsignal(signal.SIGTERM)
        def interrupted(process, stop, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            raise worker.WorkerStopped()
        with (patch.object(worker.subprocess, "Popen", return_value=native) as popen,
              patch.object(worker, "wait_ready", side_effect=interrupted),
              patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            status = worker.serve(self.config, self.root / "config.toml", **image)
        self.assertEqual(status, 143)
        self.assertEqual(popen.call_count, 1)
        cleanup.assert_called_once_with([native])
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_shutdown_kills_surviving_inference_group_after_grace(self):
        process = Mock(pid=100001)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 0), 0]
        with patch.object(worker.os, "killpg") as kill:
            worker.stop_children([process], grace_seconds=0)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(100001, signal.SIGTERM), (100001, signal.SIGKILL)])

    def test_cli_redacts_configured_credentials_in_download_failure(self):
        self.config["huggingface"]["api_key"] = "hf-file-secret"
        self.config["chapter_storage"]["secret_access_key"] = "storage-file-secret"
        output = io.StringIO()
        with (patch.object(worker, "load_config", return_value=self.config),
              patch.object(worker, "prepare", side_effect=RuntimeError("hf-file-secret storage-file-secret")),
              patch("sys.stderr", new=output)):
            self.assertEqual(worker.main(["prepare"]), 1)
        self.assertNotIn("hf-file-secret", output.getvalue())
        self.assertNotIn("storage-file-secret", output.getvalue())
        self.assertIn("REDACTED", output.getvalue())

    def publisher(self):
        publisher = Mock()
        publisher.bucket = "fixture-bucket"
        missing = RuntimeError("not found")
        missing.response = {"Error": {"Code": "NoSuchKey"}}
        publisher.client.head_object.side_effect = missing
        return publisher

    def test_storage_smoke_removes_only_unique_object_after_hash_failure(self):
        audio = self.root / "chapter.flac"
        audio.write_bytes(b"compressed fixture")
        publisher = self.publisher()
        publisher.publish.return_value = {"size": audio.stat().st_size}
        publisher.url.return_value = "https://storage.example/private-signature"
        factory = Mock(return_value=publisher)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            worker.storage_smoke(self.config, audio, publisher_factory=factory,
                                 download=lambda *_: "0" * 64)
        self.assertTrue(factory.call_args.args[0]["prefix"].endswith("/deployment-smoke"))
        job_id, name, _, checksum = publisher.publish.call_args.args
        key = publisher.client.delete_object.call_args.kwargs["Key"]
        self.assertEqual(key, "/".join((factory.call_args.args[0]["prefix"], job_id, checksum, name)))
        publisher.client.head_object.assert_called_once_with(Bucket="fixture-bucket", Key=key)

    def test_storage_smoke_cleans_up_an_upload_that_failed_head_validation(self):
        audio = self.root / "chapter.m4a"
        audio.write_bytes(b"aac fixture")
        publisher = self.publisher()
        publisher.publish.side_effect = RuntimeError("head check failed")
        with self.assertRaisesRegex(RuntimeError, "head check failed"):
            worker.storage_smoke(self.config, audio, publisher_factory=lambda _: publisher)
        publisher.client.delete_object.assert_called_once()
        publisher.url.assert_not_called()

    def test_storage_smoke_reports_cleanup_failure(self):
        audio = self.root / "chapter.flac"
        audio.write_bytes(b"compressed fixture")
        publisher = self.publisher()
        publisher.publish.return_value = {"size": audio.stat().st_size}
        publisher.client.delete_object.side_effect = RuntimeError("private storage detail")
        with self.assertRaisesRegex(RuntimeError, "confirm deletion") as caught:
            worker.storage_smoke(self.config, audio, publisher_factory=lambda _: publisher,
                                 download=lambda *_: worker.file_hash(audio))
        self.assertNotIn("private storage detail", str(caught.exception))

    def test_smoke_authenticates_then_checks_speech_encodes_and_roundtrips_compressed_audio(self):
        self.prepare()
        from epub_tts_narrator.renderer.chapter_protocol import PROTOCOL
        fetch = Mock(return_value={"protocol": PROTOCOL})
        engine = Mock()
        def generate(text, references, destination, seed):
            self.assertEqual(references, [])
            with wave.open(str(destination), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x10\x10" * 24000)
        engine._request.side_effect = generate
        verifier, roundtrip = Mock(), Mock()
        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b"fixture m4a")
            return Mock(returncode=0)
        with (patch("epub_tts_narrator.renderer.quality.validate_audio", return_value={"passed": True}),
              patch.object(worker.subprocess, "run", side_effect=encode),
              patch("sys.stdout", new=io.StringIO())):
            worker.smoke(self.config, fetch=fetch, engine_factory=lambda *_: engine,
                         verifier_factory=lambda *_: verifier, roundtrip=roundtrip)
        self.assertEqual(fetch.call_args.kwargs["headers"], {"Authorization": "Bearer test-service-key"})
        self.assertEqual([call.args[1].suffix for call in roundtrip.call_args_list], [".flac", ".m4a"])
        self.assertEqual(list((self.root / "data/deployment-smoke").iterdir()), [])
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_smoke_rejects_incorrect_service_before_synthesis(self):
        self.prepare()
        engine_factory = Mock()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            worker.smoke(self.config, fetch=lambda *_, **__: {}, engine_factory=engine_factory)
        engine_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
