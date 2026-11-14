"""A render whose auto-started local speech server disappears starts it again instead of failing every take."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.auto_serve import AutoServeError
from epub_tts_narrator.renderer.fish_http import FishHTTPError
from epub_tts_narrator.renderer.pool import MAX_SERVER_RESTARTS, RenderPool
from epub_tts_narrator.renderer.workflow import RenderServiceUnavailable, VerificationUnavailable
from test_renderer_pool import pool_config, worker

REFUSED = "Fish HTTP request failed (local; endpoint=http://127.0.0.1:8080/v1/tts): <urlopen error [Errno 61] Connection refused>"


def routed(name="local", concurrency=1):
    """A local slot as shared_worker_config routes it to the auto-started loopback server."""
    return {**worker(name, concurrency=concurrency), "url": "http://127.0.0.1:8080/v1/tts", "auto_serve": True}


class Engine:
    """Fails with ``failures`` (one per call) before answering."""

    identity = {"engine": "fixture"}

    def __init__(self, *failures):
        self.failures, self.calls, self.lock = list(failures), 0, Lock()

    def synthesize(self, text, *args, **kwargs):
        with self.lock:
            self.calls += 1
            failure = self.failures.pop(0) if self.failures else None
        if failure is not None:
            raise failure
        return {"text": text}


class LostServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.restarts = 0
        self.absent = True

    def pool(self, *workers, auto=True):
        config = pool_config(*workers)
        if auto:
            config["_auto_serve"] = {"server": {"port": 8080}}
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        return pool

    def patched(self, restart=None):
        def start(settings, notify=None):
            self.restarts += 1
            self.absent = False
            if restart is not None:
                restart()
        return (patch("epub_tts_narrator.renderer.auto_serve.server_absent", side_effect=lambda settings: self.absent),
                patch("epub_tts_narrator.renderer.auto_serve.resolve_small_job_engine", side_effect=start))

    def synthesize(self, pool, text="Hello."):
        return pool.synthesize(text, {}, self.root / "take", seed=1)

    def test_a_server_that_is_gone_is_started_again_and_the_request_retried(self):
        pool = self.pool(routed())
        engine = Engine(FishHTTPError(REFUSED))
        pool._workers[0]._engine = engine
        absent, start = self.patched()
        with absent, start:
            result = self.synthesize(pool)
        self.assertEqual((result["text"], engine.calls, self.restarts), ("Hello.", 2, 1))

    def test_a_running_server_keeps_the_failure_an_ordinary_failed_attempt(self):
        pool = self.pool(routed())
        pool._workers[0]._engine = Engine(FishHTTPError(REFUSED))
        self.absent = False
        absent, start = self.patched()
        with absent, start, self.assertRaises(FishHTTPError):
            self.synthesize(pool)
        self.assertEqual(self.restarts, 0)

    def test_every_slot_failing_at_once_restarts_the_server_only_once(self):
        slots = 8
        pool = self.pool(routed(concurrency=slots))
        together = Barrier(slots)

        class Together(Engine):
            def synthesize(self, text, *args, **kwargs):
                with self.lock:
                    self.calls += 1
                    first = self.calls <= slots
                if first:
                    together.wait(5)  # all eight requests reached the dead server
                    raise FishHTTPError(REFUSED)
                return {"text": text}
        engine = Together()
        pool._workers[0]._engine = engine
        absent, start = self.patched()
        with absent, start, ThreadPoolExecutor(max_workers=slots) as executor:
            results = list(executor.map(lambda n: self.synthesize(pool, f"line {n}"), range(slots)))
        self.assertEqual(len(results), slots)
        self.assertEqual(self.restarts, 1)
        self.assertEqual(engine.calls, 2 * slots)

    def test_a_server_that_cannot_be_started_stops_the_render_as_an_outage(self):
        pool = self.pool(routed())
        pool._workers[0]._engine = Engine(FishHTTPError(REFUSED))

        def refuse(settings, notify=None):
            raise AutoServeError("did not become ready")
        with patch("epub_tts_narrator.renderer.auto_serve.server_absent", return_value=True), \
                patch("epub_tts_narrator.renderer.auto_serve.resolve_small_job_engine", side_effect=refuse), \
                self.assertRaises(RenderServiceUnavailable) as caught:
            self.synthesize(pool)
        self.assertIsInstance(caught.exception, VerificationUnavailable)  # run-fatal, not a failed take
        self.assertIn("could not be started again", str(caught.exception))
        self.assertIn("lack of memory", str(caught.exception))

    def test_a_server_that_keeps_disappearing_stops_the_render_after_the_restart_limit(self):
        pool = self.pool(routed())
        pool._workers[0]._engine = Engine(*[FishHTTPError(REFUSED) for _ in range(2 * MAX_SERVER_RESTARTS + 2)])

        def start(settings, notify=None):
            self.restarts += 1  # it comes up, then is gone again by the next request
        with patch("epub_tts_narrator.renderer.auto_serve.server_absent", return_value=True), \
                patch("epub_tts_narrator.renderer.auto_serve.resolve_small_job_engine", side_effect=start):
            for _ in range(MAX_SERVER_RESTARTS):
                with self.assertRaises(FishHTTPError):  # the retry after each restart failed too
                    self.synthesize(pool)
            with self.assertRaisesRegex(RenderServiceUnavailable, "stopped again"):
                self.synthesize(pool)
        self.assertEqual(self.restarts, MAX_SERVER_RESTARTS)

    def test_an_http_answer_or_a_remote_worker_never_restarts_anything(self):
        absent, start = self.patched()
        with absent as probe, start:
            pool = self.pool(routed())
            pool._workers[0]._engine = Engine(FishHTTPError("Fish HTTP 500: model error", status=500))
            with self.assertRaises(FishHTTPError):
                self.synthesize(pool)
            remote = self.pool(worker("second-mac"))
            remote._workers[0]._engine = Engine(FishHTTPError(REFUSED))
            with self.assertRaises(FishHTTPError):
                self.synthesize(remote)
        probe.assert_not_called()
        self.assertEqual(self.restarts, 0)


if __name__ == "__main__":
    unittest.main()
