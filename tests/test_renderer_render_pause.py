"""Pausing a render: nothing new starts, and a lost connection is not a failed take."""

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from epub_tts_narrator.jev import JevError
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.renderer import render_pause
from epub_tts_narrator.renderer.fish_http import FishHTTPError
from epub_tts_narrator.renderer.interrupts import RenderInterrupted, keyboard_interrupts
from epub_tts_narrator.renderer.render_pause import PauseGate, network_failure
from epub_tts_narrator.renderer.render_resource_context import run_review_leaf
from epub_tts_narrator.run_shell_model import RunModel
from test_renderer_lost_server import Engine
from test_renderer_pool import pool_config, worker


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def flaky(failures, result="spoken"):
    """A request failing with each of ``failures`` in turn, then answering; ``calls`` counts the sends."""
    queue = list(failures)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        if queue:
            failure = queue.pop(0)
            if callable(failure):
                failure = failure()
            raise failure
        return result
    return request, calls


class NetworkFailureTests(unittest.TestCase):
    def test_only_requests_that_got_no_answer_are_network_failures(self):
        self.assertTrue(network_failure(FishHTTPError("connection refused")))
        self.assertTrue(network_failure(FishHTTPError("timed out", timed_out=True)))
        self.assertTrue(network_failure(JevError("down", error_code="transport_error")))
        self.assertTrue(network_failure(ProviderError("down", retryable=True)))
        # An answer from the server is the server's verdict, paused or not.
        self.assertFalse(network_failure(FishHTTPError("busy", status=503)))
        self.assertFalse(network_failure(JevError("bad", error_code="http_error", status_code=500)))
        self.assertFalse(network_failure(ProviderError("rate", status_code=429, retryable=True)))
        self.assertFalse(network_failure(ProviderError("malformed", retryable=False)))
        self.assertFalse(network_failure(ValueError("text mismatch")))


class PauseGateTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gate = PauseGate(clock=self.clock, sleep=self.clock.sleep)

    def test_nothing_starts_while_paused(self):
        self.gate.pause()
        request, calls = flaky([])
        results = []
        worker = threading.Thread(target=lambda: results.append(self.gate.call(request, "line", seed=7)))
        worker.start()
        time.sleep(0.3)
        self.assertEqual(calls, [])  # still waiting
        self.gate.resume()
        worker.join(5)
        self.assertEqual(results, ["spoken"])
        self.assertEqual(calls, [(("line",), {"seed": 7})])  # the same request, sent once

    def test_a_request_the_pause_cut_off_is_sent_again_after_resume(self):
        def lose_connection():
            self.gate.pause()  # the laptop was paused while the request ran
            threading.Timer(0.3, self.gate.resume).start()
            return FishHTTPError("network is unreachable")
        request, calls = flaky([lose_connection])
        self.assertEqual(self.gate.call(request, "line", seed=7), "spoken")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])

    def test_a_lost_connection_without_a_pause_still_fails(self):
        request, calls = flaky([FishHTTPError("network is unreachable")])
        with self.assertRaises(FishHTTPError):
            self.gate.call(request)
        self.assertEqual(len(calls), 1)

    def test_an_answer_from_the_server_still_fails_while_paused(self):
        def refused():
            self.gate.pause()
            return FishHTTPError("bad request", status=400, retryable=False)
        request, calls = flaky([refused])
        with self.assertRaises(FishHTTPError):
            self.gate.call(request)
        self.assertEqual(len(calls), 1)

    def test_just_after_resume_the_network_may_still_be_coming_back(self):
        self.gate.pause()
        self.gate.resume()
        down = FishHTTPError("network is unreachable")
        request, calls = flaky([down, down])
        self.assertEqual(self.gate.call(request), "spoken")
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.clock.slept, [render_pause.RECONNECT_RETRY_SECONDS] * 2)
        # Long after resuming, a lost connection is an ordinary failure again.
        self.clock.now += render_pause.RESUME_GRACE_SECONDS
        request, calls = flaky([down])
        with self.assertRaises(FishHTTPError):
            self.gate.call(request)

    def test_stopping_while_paused_stops(self):
        self.gate.pause()
        request, calls = flaky([])
        with keyboard_interrupts(force_exit=lambda: None) as controller:
            threading.Timer(0.2, controller.request).start()
            with self.assertRaises(RenderInterrupted):
                self.gate.call(request)
        self.assertEqual(calls, [])

    def test_review_requests_wait_for_resume(self):
        self.gate.pause()
        request, calls = flaky([])
        results = []
        with mock.patch.object(render_pause, "PAUSE", self.gate):
            worker = threading.Thread(target=lambda: results.append(run_review_leaf(request, "payload")))
            worker.start()
            time.sleep(0.3)
            self.assertEqual(calls, [])
            self.gate.resume()
            worker.join(5)
        self.assertEqual(results, ["spoken"])


class PausedPoolTests(unittest.TestCase):
    def test_a_take_whose_connection_the_pause_lost_is_sent_again_not_failed(self):
        gate = PauseGate()
        with tempfile.TemporaryDirectory() as root:
            from epub_tts_narrator.renderer.pool import RenderPool
            pool = RenderPool(pool_config(worker("second-mac")))
            self.addCleanup(pool.close)

            def lose_connection():
                gate.pause()
                threading.Timer(0.3, gate.resume).start()
                return FishHTTPError("network is unreachable", timed_out=True)
            engine = Engine(lose_connection)

            def synthesize(text, *args, **kwargs):
                failure = engine.failures.pop(0) if engine.failures else None
                engine.calls += 1
                if failure is not None:
                    raise failure()
                return {"text": text}
            engine.synthesize = synthesize
            pool._workers[0]._engine = engine
            with mock.patch.object(render_pause, "PAUSE", gate):
                result = pool.synthesize("Hello.", {}, Path(root) / "take", seed=1)
        self.assertEqual((result["text"], engine.calls), ("Hello.", 2))


class PausedShellTests(unittest.TestCase):
    def test_the_glance_says_paused_and_the_pause_is_never_a_stall(self):
        clock = Clock()
        model = RunModel("render", clock=clock)
        model.set_paused(True)
        level, text = model.health()
        self.assertEqual(level, "warning")
        self.assertIn("Paused", text)
        clock.now += 3600
        model.set_paused(False)
        self.assertEqual(model.health(), ("ok", "Running normally"))


if __name__ == "__main__":
    unittest.main()
