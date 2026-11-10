"""serve exits itself after server.idle_timeout_seconds without requests."""

import base64
import importlib.util
import io
import json
import re
import struct
import tempfile
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
import wave

from epub_tts_narrator.renderer.server import make_server, serve


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<h", 1000) * 4000)
    return output.getvalue()


class FakeServeEngine:
    """A model-free engine for make_server; optional gate for in-flight tests."""

    def __init__(self, gate=None):
        self.gate = gate
        self.calls = []
        self.error = None
        self.temperature = 0.7
        self.top_p = 0.7
        self.top_k = 30
        self.max_new_tokens = 2048

    def synthesize_fish(self, text, voices, path, *, seed, history=()):
        self.calls.append(text)
        if self.gate is not None:
            entered, release = self.gate
            entered.set()
            release.wait(5)
        path.write_bytes(wav_bytes())
        return {"seed": seed, "codes": [[1] * 3 for _ in range(10)],
                "codebook_size": 1024, "history_turns": len(history), "history_sha256": "0" * 64}

    def close(self):
        pass

    def warmup(self):
        pass


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class IdleExitTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.ready = threading.Condition()
        self.captured = {}

    def notify(self, message):
        with self.ready:
            self.events.append(message)
            self.ready.notify_all()

    def wait_for(self, marker, timeout=10):
        with self.ready:
            ready = self.ready.wait_for(
                lambda: any(marker in event for event in self.events), timeout)
        self.assertTrue(ready, self.events)

    def run_server(self, idle_timeout):
        self.engine = FakeServeEngine()
        config = {"server": {"host": "127.0.0.1", "port": 0, "idle_timeout_seconds": idle_timeout},
                  "engine": {"type": "drama3_mlx"}}
        real_make_server = make_server

        def capturing(config, engine=None, **kwargs):
            self.captured["server"] = real_make_server(config, engine, **kwargs)
            return self.captured["server"]

        # The patches must stay active for the server thread's lifetime: the
        # thread calls make_server after start(), not before.
        create_patcher = patch("epub_tts_narrator.renderer.server.create_engine", return_value=self.engine)
        make_patcher = patch("epub_tts_narrator.renderer.server.make_server", side_effect=capturing)
        create_patcher.start()
        self.addCleanup(create_patcher.stop)
        make_patcher.start()
        self.addCleanup(make_patcher.stop)
        thread = threading.Thread(target=serve, args=(config,), kwargs={"notify": self.notify}, daemon=True)
        thread.start()
        self.addCleanup(self.stop, thread)
        self.wait_for("listening")
        return thread

    def stop(self, thread):
        server = self.captured.get("server")
        if server is not None:
            server.shutdown()
        thread.join(timeout=10)

    def port(self):
        listening = next(event for event in self.events if "listening" in event)
        return int(re.search(r":(\d+)/v1/tts", listening).group(1))

    def synthesize_once(self, port):
        body = {"text": "Hello.", "format": "wav", "normalize": False,
                "temperature": 0.7, "top_p": 0.7, "max_new_tokens": 2048,
                "references": [{"audio": base64.b64encode(wav_bytes()).decode(), "text": "Reference."}]}
        request = Request(f"http://127.0.0.1:{port}/v1/tts", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Audiobook-Seed": "5",
                                   "X-Audiobook-Top-K": "30"})
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_idle_server_exits_within_a_bounded_time(self):
        thread = self.run_server(0.5)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any("stopping the renderer server" in event for event in self.events))

    def test_a_served_request_resets_the_idle_clock(self):
        thread = self.run_server(1.0)
        port = self.port()
        time.sleep(0.8)
        self.synthesize_once(port)
        time.sleep(0.8)
        # Without the reset the server would have exited by now (1.0s timeout,
        # sub-second shutdown latency); the request bought a fresh window.
        self.assertTrue(thread.is_alive())
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any("stopping the renderer server" in event for event in self.events))

    def test_disabled_idle_timeout_keeps_the_server_running(self):
        thread = self.run_server(0)
        time.sleep(1.5)
        self.assertTrue(thread.is_alive())
        self.assertFalse(any("Idle exit enabled" in event for event in self.events))
        self.assertFalse(any("stopping the renderer server" in event for event in self.events))


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class InFlightIdleClockTests(unittest.TestCase):
    def test_in_flight_requests_hold_the_idle_clock(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        entered, release = threading.Event(), threading.Event()
        engine = FakeServeEngine(gate=(entered, release))
        config = {"server": {"host": "127.0.0.1", "port": 0, "max_pending_requests": 4},
                  "engine": {"type": "drama3_mlx"}}
        server = make_server(config, engine, notify=lambda _: None)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        # Cleanup runs last-to-first: shutdown ends serve_forever, then join,
        # then server_close drains the engine.
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        body = {"text": "Hello.", "format": "wav", "normalize": False,
                "temperature": 0.7, "top_p": 0.7, "max_new_tokens": 2048,
                "references": [{"audio": base64.b64encode(wav_bytes()).decode(), "text": "Reference."}]}
        request = Request(f"http://127.0.0.1:{port}/v1/tts", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Audiobook-Seed": "5",
                                   "X-Audiobook-Top-K": "30"})

        def send():
            try:
                urlopen(request, timeout=5).read()
            except Exception as error:  # surfaced via the engine call count below
                engine.error = error

        client = threading.Thread(target=send, daemon=True)
        client.start()
        self.assertTrue(entered.wait(5), "the gated synthesis never started")
        self.assertEqual(server.idle_seconds(), 0.0)
        release.set()
        client.join(timeout=5)
        time.sleep(0.05)
        self.assertGreater(server.idle_seconds(), 0.0)
        self.assertIsNone(getattr(engine, "error", None))
