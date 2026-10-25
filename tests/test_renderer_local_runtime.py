"""Local MLX ownership without loading Metal or any model weights."""

from concurrent.futures import ThreadPoolExecutor
import sys
from threading import Event, Timer, get_ident
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition, stop_if_requested
from epub_tts_narrator.renderer.local_runtime import LocalRuntime, requires_mlx
from epub_tts_narrator.renderer.pool import RenderPool
from test_renderer_pool import pool_config, worker


class LocalRuntimeTests(unittest.TestCase):
    def test_scoped_stop_conditions_reach_the_persistent_thread_with_live_predicates(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        stopped = []

        def operation():
            try:
                stop_if_requested()
            except RenderPaused:
                stopped.append("paused")
                return "paused"
            return "ran"

        self.assertEqual(runtime.run(operation), "ran")
        with scoped_stop_condition(lambda: True, lambda: RenderPaused("owner gone")):
            self.assertEqual(runtime.run(operation), "paused")
        # The predicate is evaluated lazily on the inference thread: flipping
        # it after submission still stops the queued operation.
        live = []
        with scoped_stop_condition(lambda: live and live[0], lambda: RenderPaused("owner gone")):
            live.append(False)
            self.assertEqual(runtime.run(operation), "ran")
            live[0] = True
            self.assertEqual(runtime.run(operation), "paused")
        self.assertEqual(stopped, ["paused", "paused"])
    def test_operations_from_different_callers_use_one_persistent_thread_and_reenter(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        main = get_ident()
        with ThreadPoolExecutor(max_workers=4) as callers:
            identities = list(callers.map(lambda _: runtime.run(get_ident), range(10)))
        self.assertEqual(len(set(identities)), 1)
        self.assertNotEqual(identities[0], main)
        self.assertEqual(runtime.run(lambda: runtime.run(get_ident)), identities[0])
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.run(get_ident)

    def test_fake_operations_do_not_initialize_mlx_and_failures_do_not_break_executor(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        with patch.object(runtime, "initialize_mlx", side_effect=AssertionError("unexpected MLX")):
            self.assertEqual(runtime.run(lambda: 3), 3)
            def fail():
                raise ValueError("original failure")
            with self.assertRaisesRegex(ValueError, "original failure"):
                runtime.run(fail)
            self.assertEqual(runtime.run(lambda: 4), 4)
        self.assertFalse(requires_mlx(SimpleNamespace(identity={"engine": "drama3_mlx"})))

    def test_explicit_streams_are_initialized_once_on_owned_thread(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        calls = []
        core = ModuleType("mlx.core")
        core.cpu, core.gpu = "cpu", "gpu"
        core.new_stream = lambda device: calls.append(("create", device, get_ident())) or device
        core.set_default_stream = lambda stream: calls.append(("default", stream, get_ident()))
        mlx = ModuleType("mlx")
        mlx.core = core
        with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core}):
            first = runtime.run(get_ident, initialize_mlx=True)
            second = runtime.run(get_ident, initialize_mlx=True)
        self.assertEqual(first, second)
        self.assertEqual(calls, [("create", "cpu", first), ("create", "gpu", first),
                                 ("default", "cpu", first), ("default", "gpu", first)])
        with self.assertRaisesRegex(RuntimeError, "inference thread"):
            runtime.initialize_mlx()

    def test_interrupted_wait_drains_queued_work_before_propagating(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        started, release = Event(), Event()
        finished = []

        def blocking_operation():
            started.set()
            if not release.wait(5):
                raise AssertionError("blocker was never released")
            finished.append("first")

        # Occupy the owned thread before queuing the interrupted operation.
        with ThreadPoolExecutor(max_workers=1) as caller:
            first = caller.submit(runtime.run, blocking_operation)
            self.assertTrue(started.wait(5))
            original_submit = runtime._executor.submit
            queued = []

            def interrupted_submit(*args, **kwargs):
                future = original_submit(*args, **kwargs)
                queued.append(future)
                future.result = lambda: (_ for _ in ()).throw(KeyboardInterrupt("fixture Ctrl-C"))
                return future

            timer = Timer(.03, release.set)
            timer.start()
            try:
                with patch.object(runtime._executor, "submit", side_effect=interrupted_submit):
                    with self.assertRaisesRegex(KeyboardInterrupt, "fixture Ctrl-C"):
                        runtime.run(lambda: finished.append("second"))
                # Returning early here would release render locks while the
                # queued model still had permission to write an audio file.
                self.assertEqual(finished, ["first", "second"])
                self.assertTrue(queued[0].done())
            finally:
                release.set()
                timer.join()
            first.result(5)

    def test_stream_initialization_failure_preserves_error_and_can_retry(self):
        runtime = LocalRuntime()
        self.addCleanup(runtime.close)
        with patch.object(runtime, "initialize_mlx", side_effect=ValueError("Metal setup failed")):
            with self.assertRaisesRegex(ValueError, "Metal setup failed"):
                runtime.run(lambda: None, initialize_mlx=True)
        self.assertEqual(runtime.run(lambda: 7), 7)

    def test_pool_model_initialization_synthesis_verification_and_close_share_owner(self):
        pool = RenderPool(pool_config(worker("local", local=True)))
        self.addCleanup(pool.close)
        calls = []

        class Engine:
            identity = {"engine": "fixture", "model": "fixture"}
            def __init__(self):
                calls.append(("load", get_ident()))
            def synthesize(self, *args, **kwargs):
                calls.append(("speech", get_ident()))
                return {}
            def create_voice(self, *args, **kwargs):
                calls.append(("create_voice", get_ident()))
                return {"description_applied": True}
            def close(self):
                calls.append(("engine_close", get_ident()))

        class Verifier:
            identity = {"backend": "fixture"}
            def transcribe(self, path):
                calls.append(("asr", get_ident()))
                # Reentrant delegation should not submit to itself and deadlock.
                return pool._local_runtime.run(lambda: {"text": "words"})
            def close(self):
                calls.append(("verifier_close", get_ident()))

        with patch("epub_tts_narrator.renderer.pool.create_engine", side_effect=lambda _: Engine()), \
             patch.object(pool._local_runtime, "initialize_mlx", side_effect=AssertionError("fixture used GPU")):
            wrapped = pool.wrap_verifier(Verifier())
            with ThreadPoolExecutor(max_workers=3) as callers:
                work = [callers.submit(pool.synthesize, "words", {}, "unused", seed=1),
                        callers.submit(wrapped.transcribe, "unused"),
                        callers.submit(pool.create_voice, "voice", "words", "unused", seed=1)]
                for future in work:
                    future.result(5)
            wrapped.close()
            pool.close()
        self.assertEqual({name for name, _ in calls},
                         {"load", "speech", "create_voice", "asr", "verifier_close", "engine_close"})
        self.assertEqual(len({identity for _, identity in calls}), 1)
        self.assertNotEqual(calls[0][1], get_ident())


if __name__ == "__main__":
    unittest.main()
