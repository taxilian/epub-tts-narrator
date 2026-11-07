"""Several Drama 3 copies on one Runpod GPU, each a single-request history worker.

Offline: subprocesses, nvidia-smi, MPS, and HTTP are all stubbed.
"""

from copy import deepcopy
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
import wave
from unittest.mock import Mock, patch

from shared_fixtures import FFMPEG
from test_runpod_worker import fake_download

from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.runpod import Deployment
from epub_tts_narrator.renderer.runpod_config import (
    DEPLOYMENT_DEFAULTS, build_remote_config, inference_ports, inference_workers, load_settings, write_private_toml,
)


_path = Path(__file__).resolve().parents[1] / "deploy/runpod/worker.py"
_spec = importlib.util.spec_from_file_location("runpod_worker_replicas", _path)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


class ReplicaConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.path = self.directory / "epub-tts-narrator.toml"
        write_private_toml(self.path, {
            "engine": {"checkpoint": "models/local"},
            "verification": {"model": "models/whisper"},
            "chapter_server": {"runpod_api_key": "file-runpod-key"},
            "chapter_storage": {"bucket": "books", "endpoint_url": "https://storage.example.test",
                                "region": "default", "access_key_id": "ceph-key", "secret_access_key": "ceph-secret"},
        })

    def settings(self, **runpod):
        write_private_toml(self.directory / "runpod.local.toml", {"runpod": runpod})
        return load_settings(self.path)

    def remote(self, settings):
        return build_remote_config(settings["renderer_config"], settings["override_raw"], "pod123",
                                   "application-key", deployment=settings["deployment"])

    def test_one_copy_by_default_keeps_the_original_worker_and_port(self):
        settings = self.settings()
        self.assertEqual((settings["deployment"]["inference_replicas"], settings["deployment"]["inference_mps"]), (1, False))
        workers = self.remote(settings)["render_pool"]["workers"]
        self.assertEqual([(w["name"], w["url"], w["concurrency"]) for w in workers],
                         [("drama3-cuda", "http://127.0.0.1:8080/v1/tts", 1)])

    def test_each_copy_is_a_single_request_worker_on_its_own_loopback_port(self):
        settings = self.settings(inference_replicas=3, prefer_h100=False, inference_mps=True)
        workers = self.remote(settings)["render_pool"]["workers"]
        self.assertEqual([w["url"] for w in workers], ["http://127.0.0.1:8080/v1/tts",
                                                       "http://127.0.0.1:8090/v1/tts", "http://127.0.0.1:8091/v1/tts"])
        self.assertEqual([w["name"] for w in workers], ["drama3-cuda", "drama3-cuda-2", "drama3-cuda-3"])
        self.assertTrue(all(w["concurrency"] == 1 and w["api_key"] == "" and w["protocol"] == "fish_inline"
                            for w in workers))
        # The worker's own profile check accepts exactly this projection.
        config = deepcopy(DEFAULTS)
        config["engine"]["checkpoint"] = "/workspace/models/drama-3"
        config["verification"]["type"] = "faster_whisper"
        config["chapter_server"]["api_key"] = "service-key"
        config["render_pool"].update(workers=workers, model_revision="a" * 40)
        worker.validate_worker(config)
        config["render_pool"]["workers"] = list(reversed(workers))
        with self.assertRaisesRegex(ValueError, "ports 8080, 8090, 8091"):
            worker.validate_worker(config)

    def test_invalid_replica_settings_are_rejected_before_any_api_call(self):
        for values, message in (({"inference_replicas": 0}, "inference_replicas"),
                                ({"inference_replicas": 5}, "inference_replicas"),
                                ({"inference_replicas": True}, "inference_replicas"),
                                ({"inference_mps": "yes"}, "inference_mps"),
                                ({"inference_replicas": 2}, "prefer_h100 = false"),
                                ({"service_port": 8090}, "private speech ports"),
                                ({"service_port": 8092}, "private speech ports")):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                self.settings(**values)

    def test_ports_are_stable(self):
        self.assertEqual(inference_ports(1), [8080])
        self.assertEqual(inference_ports(4), [8080, 8090, 8091, 8092])
        self.assertEqual(len(inference_workers(4)), 4)

    def test_mps_reaches_every_worker_command(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        renderer = deepcopy(DEFAULTS)
        renderer["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        settings = {"deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=temporary.name),
                    "runpod_api_key": "fixture-provider-secret", "renderer_config": renderer, "override_raw": {}}
        deployment = Deployment(settings, api=Mock(), notify=Mock())
        self.assertNotIn("--mps", deployment.worker_command("serve"))
        settings["deployment"]["inference_mps"] = True
        self.assertIn(" --mps serve", Deployment(settings, api=Mock(), notify=Mock()).worker_command("serve"))


class ReplicaWorkerTests(unittest.TestCase):
    def setUp(self):
        patch.dict(os.environ, {"PATH": os.defpath}, clear=True).start()
        self.addCleanup(patch.stopall)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama")
        self.config["verification"].update(type="faster_whisper", device="cpu", model=str(self.root / "whisper"))
        self.config["chapter_server"].update(data_dir=str(self.root / "data"), api_key="test-service-key")
        self.config["render_pool"]["workers"] = inference_workers(2)
        patch.object(worker, "check_audio_decoder").start()
        with patch("sys.stdout", new=io.StringIO()):
            worker.prepare(self.config, downloader=Mock(side_effect=fake_download))
        fish = self.root / "fish/.venv/bin/python"
        application = self.root / "app/.venv/bin/python"
        for path in (fish, application):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.image = {"fish_source": self.root / "fish", "application_python": application}

    def serve(self, *, natives, mps=False, gpu_memory=None, mps_control=None):
        chapter = Mock(pid=200001)
        chapter.poll.return_value = 0
        options = dict(mps=mps, gpu_memory=gpu_memory or Mock(return_value=98304))
        if mps_control is not None:
            options["mps_control"] = mps_control
        with (patch.object(worker.subprocess, "Popen", side_effect=[*natives, chapter]) as popen,
              patch.object(worker, "wait_ready") as ready,
              patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            status = worker.serve(self.config, self.root / "config.toml", **self.image, **options)
        return status, popen, ready, cleanup, chapter

    def test_every_copy_starts_on_its_port_before_the_chapter_service(self):
        natives = [Mock(pid=100001 + index) for index in range(2)]
        for native in natives:
            native.poll.return_value = None
        gpu_memory = Mock(return_value=98304)
        status, popen, ready, cleanup, chapter = self.serve(natives=natives, gpu_memory=gpu_memory)
        self.assertEqual(status, 0)
        gpu_memory.assert_called_once_with(2)
        commands = [call.args[0] for call in popen.call_args_list]
        self.assertEqual(len(commands), 3)
        ports = [tomllib.loads(Path(command[command.index("--config") + 1]).read_text())["server"]["port"]
                 for command in commands[:2]]
        self.assertEqual(ports, [8080, 8090])
        self.assertIn("serve-chapters", commands[2])
        self.assertEqual([call.kwargs["worker"]["url"] for call in ready.call_args_list],
                         ["http://127.0.0.1:8080/v1/tts", "http://127.0.0.1:8090/v1/tts"])
        self.assertTrue(all("CUDA_MPS_PIPE_DIRECTORY" not in call.kwargs["env"] for call in popen.call_args_list))
        cleanup.assert_called_once_with([*natives, chapter])

    def test_any_copy_exiting_stops_the_whole_worker(self):
        natives = [Mock(pid=100001), Mock(pid=100002)]
        natives[0].poll.return_value = None
        natives[1].poll.return_value = 1
        chapter = Mock(pid=200001)
        chapter.poll.return_value = None
        with (patch.object(worker.subprocess, "Popen", side_effect=[*natives, chapter]),
              patch.object(worker, "wait_ready"), patch.object(worker, "stop_children") as cleanup,
              patch("sys.stdout", new=io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "exited"):
                worker.serve(self.config, self.root / "config.toml", gpu_memory=Mock(), **self.image)
        cleanup.assert_called_once_with([*natives, chapter])

    def test_insufficient_gpu_memory_fails_before_loading_any_copy(self):
        with self.assertRaisesRegex(ValueError, "2 Drama 3 copies need about 88 GB.*has 79 GB"):
            worker.check_gpu_memory(2, query=lambda: "81559\n")
        self.assertEqual(worker.check_gpu_memory(2, query=lambda: "97887\n"), 97887)
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            worker.check_gpu_memory(2, query=lambda: "N/A\n")
        with (patch.object(worker.subprocess, "Popen") as popen, patch("sys.stdout", new=io.StringIO()),
              self.assertRaisesRegex(ValueError, "GPU memory")):
            worker.serve(self.config, self.root / "config.toml", **self.image,
                         gpu_memory=lambda replicas: worker.check_gpu_memory(replicas, query=lambda: "81559"))
        popen.assert_not_called()

    def test_one_copy_skips_the_memory_check(self):
        self.config["render_pool"]["workers"] = inference_workers(1)
        native = Mock(pid=100001)
        native.poll.return_value = None
        gpu_memory = Mock()
        self.serve(natives=[native], gpu_memory=gpu_memory)
        gpu_memory.assert_not_called()

    def test_mps_daemon_serves_only_the_copies_and_always_stops(self):
        variables = worker.mps_environment(self.root / "mps")
        start, stop = Mock(return_value=("/usr/bin/nvidia-cuda-mps-control", variables)), Mock()
        natives = [Mock(pid=100001), Mock(pid=100002)]
        for native in natives:
            native.poll.return_value = None
        _status, popen, _ready, _cleanup, _chapter = self.serve(natives=natives, mps=True, mps_control=(start, stop))
        envs = [call.kwargs["env"] for call in popen.call_args_list]
        self.assertTrue(all(env["CUDA_MPS_PIPE_DIRECTORY"] == variables["CUDA_MPS_PIPE_DIRECTORY"] for env in envs[:2]))
        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", envs[2])
        stop.assert_called_once_with("/usr/bin/nvidia-cuda-mps-control", variables)
        # A failed start still stops the daemon after cleaning up children.
        start.reset_mock(), stop.reset_mock()
        failing = [Mock(pid=100003), Mock(pid=100004)]
        failing[0].poll.return_value, failing[1].poll.return_value = None, 3
        chapter = Mock(pid=200002)
        chapter.poll.return_value = None
        with (patch.object(worker.subprocess, "Popen", side_effect=[*failing, chapter]),
              patch.object(worker, "wait_ready"), patch.object(worker, "stop_children"),
              patch("sys.stdout", new=io.StringIO()), self.assertRaises(RuntimeError)):
            worker.serve(self.config, self.root / "config.toml", **self.image, mps=True,
                         gpu_memory=Mock(), mps_control=(start, stop))
        stop.assert_called_once()

    def test_mps_start_requires_the_control_binary_and_checks_its_exit(self):
        with self.assertRaisesRegex(ValueError, "nvidia-cuda-mps-control"):
            worker.start_mps(root=self.root / "mps", which=lambda name: None)
        run = Mock(return_value=Mock(returncode=0))
        binary, variables = worker.start_mps(root=self.root / "mps", which=lambda name: "/bin/mps", run=run)
        self.assertEqual(run.call_args.args[0], ["/bin/mps", "-d"])
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_MPS_PIPE_DIRECTORY"], variables["CUDA_MPS_PIPE_DIRECTORY"])
        self.assertTrue(Path(variables["CUDA_MPS_LOG_DIRECTORY"]).is_dir())
        with self.assertRaisesRegex(RuntimeError, "MPS control daemon"):
            worker.start_mps(root=self.root / "mps", which=lambda name: "/bin/mps", run=Mock(return_value=Mock(returncode=1)))
        stop = Mock()
        worker.stop_mps(binary, variables, run=stop)
        self.assertEqual(stop.call_args.kwargs["input"], b"quit\n")

    def test_smoke_validates_normalized_pcm_not_the_raw_float32_take(self):
        # Regression: the CUDA engine writes IEEE-float WAV (format 3), which the
        # PCM WAV reader rejects; smoke must check the normalized clip instead.
        import numpy as np
        import soundfile as sf
        ffmpeg = FFMPEG
        from epub_tts_narrator.renderer.chapter_protocol import PROTOCOL
        self.assertIsNotNone(ffmpeg, "FFmpeg is required by the renderer test environment")
        self.config["render_pool"]["workers"] = inference_workers(1)
        self.config["render"]["ffmpeg"] = ffmpeg
        engine = Mock()
        engine._request.side_effect = lambda text, refs, history, destination, seed: sf.write(
            destination, (0.2 * np.sin(np.linspace(0, 800, 44100))).astype(np.float32), 44100, subtype="FLOAT")
        checked = []

        def validate(path, text, verifier, quality, purpose):
            with wave.open(str(path), "rb") as stream:  # raises "unknown format: 3" on the raw take
                checked.append((Path(path).name, stream.getsampwidth()))
            return {"passed": True}

        real_run = worker.subprocess.run

        def run(command, **kwargs):
            if command[0] == ffmpeg and command[-1].endswith(".m4a"):
                Path(command[-1]).write_bytes(b"fixture m4a")
                return Mock(returncode=0)
            return real_run(command, **kwargs)
        with (patch("epub_tts_narrator.renderer.quality.validate_audio", side_effect=validate),
              patch.object(worker.subprocess, "run", side_effect=run), patch("sys.stdout", new=io.StringIO())):
            worker.smoke(self.config, fetch=Mock(return_value={"protocol": PROTOCOL}), engine_factory=lambda *_: engine,
                         verifier_factory=lambda *_: Mock(), roundtrip=Mock())
        self.assertEqual(checked, [("sample-1-normalized.wav", 2)])

    def test_smoke_checks_every_copy(self):
        from epub_tts_narrator.renderer.chapter_protocol import PROTOCOL
        engines = []

        def factory(settings, engine_config):
            engine = Mock(url=settings["url"])

            def generate(text, references, history, destination, seed):
                with wave.open(str(destination), "wb") as stream:
                    stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    stream.writeframes(b"\x10\x10" * 24000)
            engine._request.side_effect = generate
            engines.append(engine)
            return engine

        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b"fixture m4a")
            return Mock(returncode=0)
        verify = Mock(side_effect=[{"passed": True}, {"passed": False, "reasons": ["missing words"]}])
        with (patch("epub_tts_narrator.renderer.quality.validate_audio", verify),
              patch.object(worker.subprocess, "run", side_effect=encode), patch("sys.stdout", new=io.StringIO()),
              self.assertRaisesRegex(RuntimeError, "copy 2 did not pass verification")):
            worker.smoke(self.config, fetch=Mock(return_value={"protocol": PROTOCOL}), engine_factory=factory,
                         verifier_factory=lambda *_: Mock(), roundtrip=Mock(),
                         normalize=lambda source, target, **_: Path(target).write_bytes(Path(source).read_bytes()))
        self.assertEqual([engine.url for engine in engines],
                         ["http://127.0.0.1:8080/v1/tts", "http://127.0.0.1:8090/v1/tts"])
        self.assertTrue(all(engine._request.call_count == 1 and engine.close.call_count == 1 for engine in engines))


if __name__ == "__main__":
    unittest.main()
