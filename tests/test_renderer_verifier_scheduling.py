"""Text review overlaps inference without changing verifier ownership or policy."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event, Timer, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.recovery_client import RecoveryModelClient
from epub_tts_narrator.renderer.verification import (
    RecurringSpeechVerifier, ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier,
)


class Recognizer:
    identity = {"backend": "fixture-asr"}

    def __init__(self, events):
        self.events = events

    def transcribe(self, path):
        self.events.append(("asr", get_ident()))
        return {"text": "words"}

    def transcribe_reference(self, path):
        self.events.append(("reference", get_ident()))
        return {"text": "reference words"}

    def close(self):
        self.events.append(("asr_close", get_ident()))


class VerifierSchedulingTests(unittest.TestCase):
    def setUp(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"] = {"workers": [{"name": "local", "type": "local"}]}
        self.pool = RenderPool(config)
        self.addCleanup(self.pool.close)
        self.events = []
        events = self.events

        class Engine:
            identity = {"engine": "fixture"}
            def synthesize(self, *args, **kwargs):
                events.append(("speech", get_ident()))
                return {}
            def close(self):
                events.append(("engine_close", get_ident()))

        self.pool._workers[0]._engine = Engine()
        self.judge = JevTranscriptJudge({})
        self.reviewer = PronunciationReviewer(RecoveryModelClient({
            "model": "fixture", "base_url": "https://example.invalid/v1",
        }))
        self.base = Recognizer(events)
        self.reference = Recognizer(events)
        self.text_verifier = ReferenceTextJudgingVerifier(
            ReferenceCorroboratingVerifier(self.base, self.reference), self.judge, self.reviewer)
        self.verifier = RecurringSpeechVerifier(
            self.text_verifier, SimpleNamespace(identity={"recovery": "fixture"}))
        self.wrapped = self.pool.wrap_verifier(self.verifier)

    def blocker(self, name, started, release):
        def run(*args, **kwargs):
            self.events.append((name, get_ident()))
            started.set()
            if not release.wait(5):
                raise AssertionError("text review was never released")
            self.events.append((name + "_done", get_ident()))
            return {"decision": "fixture"}
        return run

    def test_builtin_text_reviews_overlap_inference_and_keep_model_thread_affinity(self):
        for method_name, component, underlying in (
            ("judge_reference", self.judge, "judge"),
            ("review_phonetics", self.reviewer, "review"),
        ):
            with self.subTest(method=method_name):
                self.events.clear()
                started, release = Event(), Event()
                with patch.object(component, underlying, side_effect=self.blocker("review", started, release)), \
                     patch.object(self.pool._review_runtime, "initialize_mlx", side_effect=AssertionError("review used MLX")), \
                     ThreadPoolExecutor(max_workers=2) as callers:
                    review = callers.submit(getattr(self.wrapped, method_name), "expected", "heard")
                    self.assertTrue(started.wait(5))
                    try:
                        self.assertEqual(callers.submit(self.wrapped.transcribe, "unused").result(5), {"text": "words"})
                        callers.submit(self.wrapped.transcribe_reference, "unused").result(5)
                        callers.submit(self.wrapped.corroborate_reference, "unused").result(5)
                        callers.submit(self.pool.synthesize, "words", {}, "unused", seed=1).result(5)
                        self.assertFalse(review.done())
                    finally:
                        release.set()
                    self.assertEqual(review.result(5), {"decision": "fixture"})
                inference = {thread for name, thread in self.events if name in {"asr", "reference", "speech"}}
                review_threads = {thread for name, thread in self.events if name.startswith("review")}
                self.assertEqual(len(inference), 1)
                self.assertEqual(len(review_threads), 1)
                self.assertTrue(inference.isdisjoint(review_threads))

    def test_text_clients_share_one_bounded_review_lane(self):
        started, release, second_queued, all_queued = Event(), Event(), Event(), Event()
        submitted = []
        original_submit = self.pool._review_runtime._executor.submit

        def submit(*args, **kwargs):
            result = original_submit(*args, **kwargs)
            submitted.append(result)
            if len(submitted) == 2:
                second_queued.set()
            if len(submitted) == 3:
                all_queued.set()
            return result

        with patch.object(self.judge, "judge", side_effect=self.blocker("judge", started, release)), \
             patch.object(self.reviewer, "review", side_effect=self.blocker("phonetic", Event(), release)), \
             patch.object(self.pool._review_runtime._executor, "submit", side_effect=submit), \
             ThreadPoolExecutor(max_workers=3) as callers:
            first = callers.submit(self.wrapped.judge_reference, "expected", "heard")
            self.assertTrue(started.wait(5))
            second = callers.submit(self.wrapped.review_phonetics, "expected", "heard")
            try:
                self.assertTrue(second_queued.wait(5))
                third = callers.submit(self.wrapped.judge_reference, "expected", "heard")
                self.assertTrue(all_queued.wait(5))
                self.assertEqual([name for name, _ in self.events], ["judge"])
                self.assertEqual(self.wrapped.transcribe("unused"), {"text": "words"})
            finally:
                release.set()
            for future in (first, second, third):
                future.result(5)
        reviews = [(name, thread) for name, thread in self.events if name != "asr"]
        self.assertEqual(len({thread for _, thread in reviews}), 1)
        self.assertEqual([name for name, _ in reviews],
                         ["judge", "judge_done", "phonetic", "phonetic_done", "judge", "judge_done"])

    def test_subclasses_custom_clients_and_plugin_methods_remain_model_affine(self):
        class CustomJudge(JevTranscriptJudge):
            def judge(self, *args):
                return get_ident()

        class CustomTextVerifier(ReferenceTextJudgingVerifier):
            def judge_reference(self, *args):
                return get_ident()

        variants = [
            ReferenceTextJudgingVerifier(self.base, CustomJudge({})),
            CustomTextVerifier(self.base, self.judge),
            SimpleNamespace(judge_reference=lambda *args: get_ident()),
        ]
        custom_reviewer = PronunciationReviewer(SimpleNamespace(identity={"custom": True}))
        variants.append(ReferenceTextJudgingVerifier(self.base, self.judge, custom_reviewer))
        owner = self.pool._local_runtime.run(get_ident)
        with patch.object(custom_reviewer, "review", side_effect=lambda *args: get_ident()):
            for verifier in variants:
                wrapped = self.pool.wrap_verifier(verifier)
                method = wrapped.review_phonetics if verifier is variants[-1] else wrapped.judge_reference
                self.assertEqual(method("expected", "heard"), owner)

    def test_close_drains_review_rejects_new_calls_and_closes_once_on_model_lane(self):
        started, release, closing = Event(), Event(), Event()
        original_wait = self.wrapped._state.wait

        def wait(*args, **kwargs):
            closing.set()
            return original_wait(*args, **kwargs)

        owner = self.pool._local_runtime.run(get_ident)
        with patch.object(self.judge, "judge", side_effect=self.blocker("review", started, release)), \
             patch.object(self.wrapped._state, "wait", side_effect=wait), \
             ThreadPoolExecutor(max_workers=3) as callers:
            review = callers.submit(self.wrapped.judge_reference, "expected", "heard")
            self.assertTrue(started.wait(5))
            first_close = callers.submit(self.wrapped.close)
            try:
                self.assertTrue(closing.wait(5))
                second_close = callers.submit(self.wrapped.close)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    self.wrapped.transcribe("unused")
                self.assertFalse(first_close.done())
                self.assertNotIn("asr_close", [name for name, _ in self.events])
            finally:
                release.set()
            review.result(5)
            first_close.result(5)
            second_close.result(5)
        self.assertEqual([name for name, _ in self.events], ["review", "review_done", "asr_close", "asr_close"])
        self.assertEqual({thread for name, thread in self.events if name == "asr_close"}, {owner})
        self.wrapped.close()
        self.assertEqual(len(self.events), 4)

    def test_interrupt_during_verifier_close_still_drains_and_closes(self):
        started, release = Event(), Event()
        original_wait = self.wrapped._state.wait
        waits = []

        def interrupted_wait(*args, **kwargs):
            waits.append(True)
            if len(waits) == 1:
                raise KeyboardInterrupt("fixture cancellation")
            return original_wait(*args, **kwargs)

        with patch.object(self.judge, "judge", side_effect=self.blocker("review", started, release)), \
             ThreadPoolExecutor(max_workers=1) as callers:
            review = callers.submit(self.wrapped.judge_reference, "expected", "heard")
            self.assertTrue(started.wait(5))
            timer = Timer(.03, release.set)
            timer.start()
            try:
                with patch.object(self.wrapped._state, "wait", side_effect=interrupted_wait):
                    with self.assertRaisesRegex(KeyboardInterrupt, "fixture cancellation"):
                        self.wrapped.close()
                self.assertEqual([name for name, _ in self.events],
                                 ["review", "review_done", "asr_close", "asr_close"])
                self.assertTrue(self.wrapped._closed)
            finally:
                release.set()
                timer.join()
            review.result(5)

    def test_pool_close_drains_review_even_when_calling_wait_is_interrupted(self):
        started, release = Event(), Event()
        original_submit = self.pool._local_runtime._executor.submit
        submissions = []

        def interrupted_submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            submissions.append(future)
            if len(submissions) == 1:
                def interrupt():
                    raise KeyboardInterrupt("fixture cancellation")
                future.result = interrupt
            return future

        with patch.object(self.judge, "judge", side_effect=self.blocker("review", started, release)), \
             ThreadPoolExecutor(max_workers=1) as callers:
            review = callers.submit(self.wrapped.judge_reference, "expected", "heard")
            self.assertTrue(started.wait(5))
            timer = Timer(.03, release.set)
            timer.start()
            try:
                with patch.object(self.pool._local_runtime._executor, "submit", side_effect=interrupted_submit):
                    with self.assertRaisesRegex(KeyboardInterrupt, "fixture cancellation"):
                        self.pool.close()
                self.assertEqual([name for name, _ in self.events], ["review", "review_done", "engine_close"])
                self.assertTrue(self.pool._review_runtime._closed)
                self.assertTrue(self.pool._local_runtime._closed)
                self.assertTrue(all(future.done() for future in submissions))
            finally:
                release.set()
                timer.join()
            review.result(5)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.wrapped.judge_reference("expected", "heard")

    def test_failures_leave_both_lanes_usable_and_preserve_identities(self):
        original_pool_identity = deepcopy(self.pool.identity)
        original_verifier_identity = deepcopy(self.verifier.identity)
        self.assertIs(self.pool.wrap_verifier(self.wrapped), self.wrapped)
        self.assertFalse(hasattr(self.wrapped, "unsupported_plugin_method"))
        with patch.object(self.judge, "judge", side_effect=ValueError("fixture review failed")):
            with self.assertRaisesRegex(ValueError, "fixture review failed"):
                self.wrapped.judge_reference("expected", "heard")
        self.assertEqual(self.wrapped._active, 0)
        self.assertEqual(self.wrapped.transcribe("unused"), {"text": "words"})
        self.assertEqual(self.wrapped.judge_reference("expected", "heard")["error_code"], "disabled")
        self.assertEqual(self.pool.identity, original_pool_identity)
        self.assertEqual(self.wrapped.identity, original_verifier_identity)

    def test_close_from_a_worker_is_rejected_instead_of_waiting_on_itself(self):
        for runtime in (self.pool._local_runtime, self.pool._review_runtime):
            with self.subTest(runtime=runtime):
                with self.assertRaisesRegex(RuntimeError, "from its owner"):
                    runtime.run(self.wrapped.close)
        self.assertFalse(self.wrapped._closing)
        self.assertEqual(self.wrapped.transcribe("unused"), {"text": "words"})


if __name__ == "__main__":
    unittest.main()
