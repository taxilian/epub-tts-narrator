"""The local speech server says which process it is and whether its code is current, and can be stopped."""

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from epub_tts_narrator.renderer.auto_serve import (
    _ABSENT, AutoServeError, older_code, resolve_small_job_engine, stop_server)
from epub_tts_narrator.renderer.code_freshness import PACKAGE_ROOT, changed_since
from epub_tts_narrator.renderer.local_server_command import describe
from test_renderer_auto_serve import ResolverTests
from test_renderer_server_idle import FakeServeEngine, wav_bytes

AUTO_SERVE = "epub_tts_narrator.renderer.auto_serve"


class CodeFreshnessTests(unittest.TestCase):
    def test_only_loaded_package_files_changed_after_the_start_count(self):
        inside = PACKAGE_ROOT / "renderer" / "code_freshness.py"
        modules = {"in": SimpleNamespace(__file__=str(inside)),
                   "outside": SimpleNamespace(__file__=__file__), "builtin": ModuleType("builtin")}
        modified = inside.stat().st_mtime
        self.assertEqual(changed_since(modified + 1, modules=modules), [])
        self.assertEqual(changed_since(modified - 1, modules=modules),
                         ["epub_tts_narrator/renderer/code_freshness.py"])

    def test_a_server_too_old_to_report_counts_as_older(self):
        from epub_tts_narrator.renderer.fish_http import PROTOCOL
        self.assertTrue(older_code({"status": "ready", "protocol": PROTOCOL}))
        self.assertTrue(older_code({"protocol": PROTOCOL, "code_changed": ["epub_tts_narrator/performance_plan.py"]}))
        self.assertFalse(older_code({"protocol": PROTOCOL, "code_changed": []}))
        self.assertFalse(older_code({"status": "ok"}))  # not ours: refused elsewhere, never stopped


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class ServerHealthAndStopTests(unittest.TestCase):
    def start(self, engine):
        from epub_tts_narrator.renderer.server import make_server
        config = {"server": {"host": "127.0.0.1", "port": 0, "max_pending_requests": 4},
                  "engine": {"type": "drama3_mlx"}}
        server = make_server(config, engine, notify=lambda _: None)
        self.addCleanup(server.server_close)
        def run():  # as serve() does: stop listening once serve_forever returns
            server.serve_forever(poll_interval=0.01)
            server.server_close()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(lambda: thread.is_alive() and server.shutdown())
        self.config = {"server": {"port": server.server_address[1], "api_key": ""}}
        return server, thread

    def health(self):
        with urlopen(f"http://127.0.0.1:{self.config['server']['port']}/health", timeout=5) as response:
            return json.loads(response.read())

    def test_health_names_the_process_its_start_and_current_code(self):
        import os
        server, _ = self.start(FakeServeEngine())
        health = self.health()
        self.assertEqual(health["pid"], os.getpid())
        self.assertIn("started_at", health)
        self.assertEqual(health["code_changed"], [])
        self.assertGreaterEqual(health["idle_seconds"], 0)
        server.started_at = 0  # every loaded renderer file is newer than this "start"
        self.assertIn("epub_tts_narrator/renderer/server.py", self.health()["code_changed"])

    def test_a_working_server_refuses_to_stop_unless_forced_and_an_idle_one_stops(self):
        entered, release = threading.Event(), threading.Event()
        server, thread = self.start(FakeServeEngine(gate=(entered, release)))
        port = self.config["server"]["port"]
        body = {"text": "Hello.", "format": "wav", "normalize": False, "temperature": 0.7, "top_p": 0.7,
                "max_new_tokens": 2048,
                "references": [{"audio": base64.b64encode(wav_bytes()).decode(), "text": "Reference."}]}
        request = Request(f"http://127.0.0.1:{port}/v1/tts", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Audiobook-Seed": "5",
                                   "X-Audiobook-Top-K": "30"})
        client = threading.Thread(target=lambda: urlopen(request, timeout=5).read(), daemon=True)
        client.start()
        self.assertTrue(entered.wait(5))
        with self.assertRaisesRegex(AutoServeError, "working on 1 request"):
            stop_server(self.config)
        release.set()
        client.join(5)
        self.assertTrue(stop_server(self.config, wait=10))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(stop_server(self.config))  # nothing running now


class ReplaceOlderServerTests(ResolverTests):
    """The resolver replaces an idle server running older code and keeps a busy one."""

    def resolve(self, health, after):
        answers = iter([health, health, *after])
        said = []
        with patch(f"{AUTO_SERVE}._probe", side_effect=lambda port, **kwargs: next(answers)), \
             patch(f"{AUTO_SERVE}.stop_server", return_value=True) as stop, \
             patch(f"{AUTO_SERVE}._wait_ready", return_value=self.health(code_changed=[])), \
             patch(f"{AUTO_SERVE}.subprocess.Popen", return_value=SimpleNamespace(pid=4242)) as spawn, \
             patch(f"{AUTO_SERVE}._cache_root", return_value=self.cache):
            resolve_small_job_engine(self.config(), notify=said.append)
        return stop, spawn, said

    def test_an_idle_server_running_older_code_is_replaced(self):
        older = self.health(code_changed=["epub_tts_narrator/performance_plan.py"], idle_seconds=300,
                            work={"requests": 0})
        stop, spawn, said = self.resolve(older, [_ABSENT])
        stop.assert_called_once()
        spawn.assert_called_once()
        self.assertTrue(any("older code" in line for line in said))

    def test_a_busy_server_running_older_code_is_used_as_it_is(self):
        older = self.health(code_changed=["epub_tts_narrator/performance_plan.py"], idle_seconds=0,
                            work={"requests": 3})
        stop, spawn, said = self.resolve(older, [])
        stop.assert_not_called()
        spawn.assert_not_called()
        self.assertTrue(any("in use" in line for line in said))


class DescribeTests(unittest.TestCase):
    def test_status_says_whether_it_runs_and_whether_its_code_is_current(self):
        absent = describe({"port": 8080, "state": "absent", "log": "serve-8080.log", "idle_timeout_seconds": 600})
        self.assertIn("not running", absent[0])
        health = {"pid": 4242, "started_at": "2026-11-14T11:07:04-07:00", "model": "drama-3",
                  "concurrent_requests": 8, "work": {"requests": 0}, "idle_seconds": 75,
                  "code_changed": ["epub_tts_narrator/performance_plan.py"]}
        lines = "\n".join(describe({"port": 8080, "state": "ready", "health": health, "log": "serve-8080.log",
                                    "idle_timeout_seconds": 600}))
        self.assertIn("process 4242", lines)
        self.assertIn("idle for 1m 15s", lines)
        self.assertIn("older than this checkout (1 file(s) changed since it started: performance_plan.py)", lines)
        self.assertIn("just server restart", lines)


if __name__ == "__main__":
    unittest.main()
