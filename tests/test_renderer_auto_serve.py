"""Auto-serve config keys and small-job discover-or-spawn resolution, offline.

The serve idle-exit behavior itself is covered by test_renderer_server_idle.py.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from epub_tts_narrator.renderer.auto_serve import (
    SPAWN_READY_SECONDS, _ABSENT, _WARMING, AutoServeError, resolve_small_job_engine,
)
from epub_tts_narrator.renderer.config import DEFAULTS, validate_distributed_settings
from epub_tts_narrator.renderer.fish_http import PROTOCOL, FishHTTPEngine

AUTO_SERVE = "epub_tts_narrator.renderer.auto_serve"


class ServerAutoServeConfigTests(unittest.TestCase):
    def config(self):
        return deepcopy(DEFAULTS)

    def test_defaults_keep_idle_exit_and_auto_start_off(self):
        config = self.config()
        validate_distributed_settings(config)
        self.assertEqual(config["server"]["idle_timeout_seconds"], 0.0)
        self.assertIs(config["server"]["auto_start"], False)

    def test_idle_timeout_accepts_disabled_zero_and_the_full_day_bound(self):
        for value in (0, 0.0, 1, 1.5, 300, 86400):
            with self.subTest(value=value):
                config = self.config()
                config["server"]["idle_timeout_seconds"] = value
                validate_distributed_settings(config)

    def test_idle_timeout_rejects_fractional_nonzero_and_out_of_range_values(self):
        for value in (-1, 0.5, 86400.01, 86401, "300", True, None):
            with self.subTest(value=value):
                config = self.config()
                config["server"]["idle_timeout_seconds"] = value
                with self.assertRaises(ValueError) as raised:
                    validate_distributed_settings(config)
                self.assertIn("idle_timeout_seconds", str(raised.exception))

    def test_auto_start_requires_a_boolean(self):
        for value, valid in ((True, True), (False, True), (1, False), ("yes", False), (None, False)):
            with self.subTest(value=value, valid=valid):
                config = self.config()
                config["server"]["auto_start"] = value
                if valid:
                    validate_distributed_settings(config)
                else:
                    with self.assertRaises(ValueError) as raised:
                        validate_distributed_settings(config)
                    self.assertIn("auto_start", str(raised.exception))


class ResolverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config_path = self.root / "epub-tts-narrator.toml"
        self.config_path.write_text("[engine]\ncheckpoint = './models/drama-3'\n")
        self.checkpoint = self.root / "models" / "drama-3"
        self.checkpoint.mkdir(parents=True)
        self.cache = self.root / "cache"

    def config(self, *, with_path=True, checkpoint=True, **server):
        settings = {"host": "127.0.0.1", "port": 8080, "api_key": "", "auto_start": True, **server}
        config = {"server": settings, "engine": {"type": "drama3_mlx", "model": "fishaudio/drama-3"}}
        if checkpoint:
            config["engine"]["checkpoint"] = str(self.checkpoint)
        if with_path:
            config["_config_path"] = str(self.config_path)
        return config

    def health(self, **overrides):
        return {"status": "ready", "model": "drama-3", "protocol": PROTOCOL,
                "reference_reuse": True, "roles": ["tts"], "concurrent_requests": 1, **overrides}

    def test_a_healthy_local_serve_returns_a_client_without_spawning(self):
        with patch(f"{AUTO_SERVE}._probe", return_value=self.health()) as probe, \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            engine = resolve_small_job_engine(self.config())
        self.assertIsInstance(engine, FishHTTPEngine)
        self.assertEqual(engine.url, "http://127.0.0.1:8080/v1/tts")
        self.assertEqual(engine.model, "drama-3")
        self.assertEqual(engine.name, "local-serve")
        self.assertEqual(engine.protocol, "fish_inline")
        self.assertEqual(engine.sampling, {"temperature": 0.7, "top_p": 0.7,
                                           "top_k": 30, "max_new_tokens": 2048})
        spawn.assert_not_called()
        probe.assert_called_once_with(8080, api_key="")

    def test_the_client_model_follows_the_configured_engine_model(self):
        config = self.config()
        config["engine"]["model"] = "mlx-community/fish-audio-drama-3-4bit"
        with patch(f"{AUTO_SERVE}._probe",
                   return_value=self.health(model="mlx-community/fish-audio-drama-3-4bit")):
            engine = resolve_small_job_engine(config)
        self.assertEqual(engine.model, "mlx-community/fish-audio-drama-3-4bit")

    def test_a_serve_running_a_different_model_is_refused(self):
        with patch(f"{AUTO_SERVE}._probe",
                   return_value=self.health(model="mlx-community/fish-audio-drama-3-8bit")), \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(self.config())
        self.assertIn("engine.model", str(raised.exception))
        self.assertIn("server.port", str(raised.exception))
        spawn.assert_not_called()

    def test_a_foreign_protocol_on_the_port_fails_closed_without_spawning(self):
        with patch(f"{AUTO_SERVE}._probe", return_value=self.health(protocol="someone-else")), \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(self.config())
        self.assertIn("free the port", str(raised.exception))
        self.assertIn("server.port", str(raised.exception))
        spawn.assert_not_called()

    def test_a_foreign_http_answer_fails_closed(self):
        for body, code in ((b'{"status": "ok"}', 200), (b"<html>not our server</html>", 200),
                           (b"denied", 401), (b"moved", 503)):
            with self.subTest(body=body, code=code):
                opener = Mock()
                if code == 200:
                    response = io.BytesIO(body)
                    response.status = 200
                    opener.open.return_value = response
                else:
                    opener.open.side_effect = HTTPError("http://127.0.0.1:8080/health", code,
                                                        "HTTP error", {}, io.BytesIO(body))
                with patch(f"{AUTO_SERVE}._PROBE_OPENER", opener), \
                     patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
                    with self.assertRaises(AutoServeError) as raised:
                        resolve_small_job_engine(self.config())
                self.assertIn("free the port", str(raised.exception))
                spawn.assert_not_called()

    def test_spawn_runs_serve_detached_with_the_config_path_and_port(self):
        process = Mock(pid=4242)
        answers = iter([_ABSENT, _ABSENT, self.health()])
        with patch(f"{AUTO_SERVE}._probe", side_effect=lambda port, **kwargs: next(answers)), \
             patch(f"{AUTO_SERVE}._wait_ready", return_value=self.health()) as wait, \
             patch(f"{AUTO_SERVE}.subprocess.Popen", return_value=process) as spawn, \
             patch(f"{AUTO_SERVE}._cache_root", return_value=self.cache):
            engine = resolve_small_job_engine(self.config())
        self.assertEqual(engine.url, "http://127.0.0.1:8080/v1/tts")
        command = spawn.call_args.args[0]
        self.assertEqual(command, [sys.executable, "-m", "epub_tts_narrator.renderer", "serve",
                                   "--config", str(self.config_path), "--port", "8080"])
        kwargs = spawn.call_args.kwargs
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["start_new_session"], True)
        self.assertEqual(kwargs["stdout"].mode, "ab")
        self.assertTrue(kwargs["stdout"].name.endswith("serve-8080.log"))
        self.assertEqual(kwargs["stderr"].name, kwargs["stdout"].name)
        wait.assert_called_once()
        self.assertEqual(wait.call_args.args, (8080, SPAWN_READY_SECONDS))
        note = json.loads((self.cache / "auto-serve-8080.json").read_text())
        self.assertEqual(note["pid"], 4242)
        self.assertEqual(note["port"], 8080)
        self.assertIn("started_at", note)
        self.assertTrue((self.cache / "auto-serve-8080.lock").exists())

    def test_a_spawn_that_never_becomes_ready_reports_the_log_tail(self):
        self.cache.mkdir(parents=True)
        (self.cache / "serve-8080.log").write_bytes(b"Downloading model weights failed\n")
        with patch(f"{AUTO_SERVE}._probe", return_value=_ABSENT), \
             patch(f"{AUTO_SERVE}._wait_ready", return_value=None), \
             patch(f"{AUTO_SERVE}.subprocess.Popen", return_value=Mock(pid=7)), \
             patch(f"{AUTO_SERVE}._cache_root", return_value=self.cache):
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(self.config())
        message = str(raised.exception)
        self.assertIn("serve-8080.log", message)
        self.assertIn("Downloading model weights failed", message)
        self.assertIn(f"{SPAWN_READY_SECONDS:g}", message)

    def test_auto_start_without_the_config_file_path_fails_clearly(self):
        with patch(f"{AUTO_SERVE}._probe", return_value=_ABSENT), \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(self.config(with_path=False))
        self.assertIn("configuration file", str(raised.exception))
        spawn.assert_not_called()

    def test_auto_start_without_the_model_checkpoint_refuses_to_spawn(self):
        with patch(f"{AUTO_SERVE}._probe", return_value=_ABSENT), \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(self.config(checkpoint=False))
        self.assertIn("engine.checkpoint", str(raised.exception))
        spawn.assert_not_called()

    def test_auto_start_refuses_engines_that_cannot_serve(self):
        config = self.config()
        config["engine"]["type"] = "gemini"
        with patch(f"{AUTO_SERVE}._probe") as probe:
            with self.assertRaises(AutoServeError) as raised:
                resolve_small_job_engine(config)
        self.assertIn("drama3_mlx", str(raised.exception))
        probe.assert_not_called()

    def test_a_warming_worker_is_waited_for_instead_of_spawned_twice(self):
        answers = iter([_WARMING, _WARMING])
        with patch(f"{AUTO_SERVE}._probe", side_effect=lambda port, **kwargs: next(answers)), \
             patch(f"{AUTO_SERVE}._wait_ready", return_value=self.health()) as wait, \
             patch(f"{AUTO_SERVE}.subprocess.Popen") as spawn:
            engine = resolve_small_job_engine(self.config())
        self.assertEqual(engine.url, "http://127.0.0.1:8080/v1/tts")
        wait.assert_called_once()
        spawn.assert_not_called()

    def test_concurrent_resolvers_spawn_only_one_server(self):
        ready = threading.Event()

        def probe(port, **kwargs):
            return self.health() if ready.is_set() else _ABSENT

        def wait(port, budget, **kwargs):
            time.sleep(0.2)  # the lock holder stays inside the flock during its wait
            ready.set()
            return self.health()

        spawn = Mock(return_value=Mock(pid=11))
        with patch(f"{AUTO_SERVE}._probe", side_effect=probe), \
             patch(f"{AUTO_SERVE}._wait_ready", side_effect=wait), \
             patch(f"{AUTO_SERVE}.subprocess.Popen", spawn), \
             patch(f"{AUTO_SERVE}._cache_root", return_value=self.cache):
            results, failures = [], []

            def run():
                try:
                    results.append(resolve_small_job_engine(self.config()))
                except BaseException as exc:
                    failures.append(exc)

            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10.0)
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(spawn.call_count, 1, "the flock must serialize spawn attempts")
        self.assertTrue((self.cache / "auto-serve-8080.lock").exists())
