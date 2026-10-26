"""Attempt-scoped untranscribed-audio backpressure (standalone primitive)."""

from __future__ import annotations

import hashlib
import threading
import time
import unittest
import wave
from copy import deepcopy
from pathlib import Path
import tempfile

from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_audio, validate_policy
from epub_tts_narrator.renderer.quality import validation_key as make_validation_key
from epub_tts_narrator.renderer.render_resource_context import resource_scope
from epub_tts_narrator.renderer.render_resources import RenderResources
from epub_tts_narrator.renderer.speech_recovery import recover_speech
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, verified_take,
)
from shared_fixtures import timed_transcript
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier

from epub_tts_narrator.renderer.render_backlog import (
    DEFAULT_MAX_MATERIALIZED_BYTES,
    BacklogBudget,
    BacklogCancelled,
    BacklogClosed,
    BacklogError,
    BacklogReleased,
    begin_attempt_ticket,
    current_ticket,
)

WORKER_WAIT = 10  # deadlock watchdog for every Event wait, join, and future

GAUGE_KEYS = {
    "pending_count",
    "materialized_bytes",
    "max_pending",
    "max_materialized_bytes",
    "low_water_bytes",
    "new_generation_blocked",
    "waiters",
    "closed",
}


def wait_until(predicate, timeout=WORKER_WAIT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class Producer:
    """One attempt thread driven through the minimal public API."""

    def __init__(self, budget, *, new_generation=True, initial_bytes=0,
                 bytes_to_materialize=None, cancel=None, phase_ok=None,
                 hold_until=None, fail_inside=False, defer_materialize=False,
                 post_materialize_hold=None):
        self.budget = budget
        self.new_generation = new_generation
        self.initial_bytes = initial_bytes
        self.bytes_to_materialize = bytes_to_materialize
        self.cancel = cancel
        self.phase_ok = phase_ok
        self.hold_until = hold_until
        self.fail_inside = fail_inside
        self.defer_materialize = defer_materialize
        self.post_materialize_hold = post_materialize_hold
        self.acquired = threading.Event()
        self.failure = None
        self.ticket = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            with self.budget.reserve(
                new_generation=self.new_generation,
                initial_bytes=self.initial_bytes,
                cancel=self.cancel,
                phase_ok=self.phase_ok,
            ) as ticket:
                self.ticket = ticket
                self.acquired.set()
                if self.defer_materialize:
                    # Generation output lands only when the hold releases.
                    if not self.hold_until.wait(WORKER_WAIT):
                        raise AssertionError("hold watchdog timed out")
                    ticket.materialize(self.bytes_to_materialize)
                    if self.post_materialize_hold is not None:
                        if not self.post_materialize_hold.wait(WORKER_WAIT):
                            raise AssertionError(
                                "post-materialize watchdog timed out")
                else:
                    if self.bytes_to_materialize is not None:
                        ticket.materialize(self.bytes_to_materialize)
                    if self.hold_until is not None:
                        if not self.hold_until.wait(WORKER_WAIT):
                            raise AssertionError("hold watchdog timed out")
                if self.fail_inside:
                    raise RuntimeError("synthesis failed")
        except BaseException as error:  # surfaced to the joining test
            self.failure = error

    def join(self):
        self.thread.join(WORKER_WAIT)
        if self.thread.is_alive():
            raise AssertionError("producer thread hung")


class BacklogBudgetContractTests(unittest.TestCase):
    def test_defaults_pin_the_soft_high_water_at_256_mib_with_half_low_water(self):
        budget = BacklogBudget(max_pending=4)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["max_materialized_bytes"],
                         DEFAULT_MAX_MATERIALIZED_BYTES)
        self.assertEqual(snapshot["low_water_bytes"],
                         DEFAULT_MAX_MATERIALIZED_BYTES // 2)

    def test_invalid_bounds_are_rejected_with_field_identifying_errors(self):
        for kwargs in ({"max_pending": 0}, {"max_pending": -3},
                       {"max_pending": "four"},
                       {"max_pending": 2, "max_materialized_bytes": 0},
                       {"max_pending": 2, "max_materialized_bytes": -1},
                       {"max_pending": 2, "max_materialized_bytes": None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BacklogBudget(**kwargs)

    def test_invalid_initial_bytes_are_rejected_without_leaking_a_slot(self):
        budget = BacklogBudget(max_pending=2, max_materialized_bytes=100)
        for initial_bytes in (-1, "many", None):
            with self.subTest(initial_bytes=initial_bytes):
                with self.assertRaises(ValueError):
                    with budget.reserve(initial_bytes=initial_bytes):
                        pass
                self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_snapshot_is_payload_free_with_only_fixed_numeric_gauges(self):
        budget = BacklogBudget(max_pending=2, max_materialized_bytes=100)
        with budget.reserve(initial_bytes=60) as ticket:
            ticket.materialize(80)
            live = budget.snapshot()
        self.assertEqual(set(live), GAUGE_KEYS)
        self.assertTrue(all(isinstance(value, (int, bool)) for value
                            in live.values()))
        self.assertEqual(live["pending_count"], 1)
        self.assertEqual(live["materialized_bytes"], 80)
        settled = budget.snapshot()
        self.assertEqual(settled["pending_count"], 0)
        self.assertEqual(settled["materialized_bytes"], 0)

    def test_reservation_context_is_single_use_and_use_after_release_raises(self):
        budget = BacklogBudget(max_pending=1)
        reservation = budget.reserve()
        with reservation as ticket:
            ticket.materialize(5)
            ticket.release()
            with self.assertRaises(BacklogReleased):
                ticket.materialize(6)
            with self.assertRaises(BacklogReleased):
                ticket.update_bytes(6)
        with self.assertRaises(BacklogError):
            with reservation:
                pass
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_duplicate_release_keeps_gauges_exact_and_never_negative(self):
        budget = BacklogBudget(max_pending=1, max_materialized_bytes=100)
        with budget.reserve(initial_bytes=40) as ticket:
            ticket.release()
            ticket.release()
        ticket.release()
        barrier = threading.Barrier(4, timeout=WORKER_WAIT)

        def hammer(shared):
            barrier.wait()
            shared.release()

        with budget.reserve(initial_bytes=90) as shared:
            threads = [threading.Thread(target=hammer, args=(shared,))
                       for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(WORKER_WAIT)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["materialized_bytes"], 0)

    def test_byte_updates_are_absolute_and_shrinking_can_clear_the_block(self):
        budget = BacklogBudget(max_pending=4, max_materialized_bytes=100)
        with budget.reserve(initial_bytes=100) as ticket:
            self.assertTrue(budget.snapshot()["new_generation_blocked"])
            ticket.update_bytes(30)
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["materialized_bytes"], 30)
            self.assertFalse(snapshot["new_generation_blocked"])


class BacklogBoundTests(unittest.TestCase):
    def test_one_permit_is_released_before_child_recursion_so_join_recursion_completes(self):
        budget = BacklogBudget(max_pending=1, max_materialized_bytes=10 ** 9)
        visits = []

        def attempt(depth):
            with budget.reserve() as ticket:
                ticket.materialize(1)
                # Release BEFORE children and join: no parent ticket is held
                # across child recovery or split/join recursion.
                ticket.release()
                if depth:
                    children = [attempt(depth - 1) for _ in range(2)]
                    del children
                visits.append(depth)

        attempt(3)
        self.assertEqual(len(visits), 15)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["materialized_bytes"], 0)

    def test_pending_count_bound_is_exact_across_generation_and_adopted_tickets(self):
        budget = BacklogBudget(max_pending=2, max_materialized_bytes=10 ** 9)
        hold = threading.Event()
        first = Producer(budget, hold_until=hold).start()
        self.assertTrue(wait_until(lambda: first.acquired.is_set()))
        with budget.reserve(new_generation=False, initial_bytes=7):
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["pending_count"], 2)
            self.assertEqual(snapshot["materialized_bytes"], 7)
            blocked = Producer(budget).start()
            self.assertTrue(wait_until(
                lambda: budget.snapshot()["waiters"] == 1))
            self.assertFalse(blocked.acquired.is_set())
        hold.set()
        first.join()
        self.assertIsNone(first.failure)
        self.assertTrue(blocked.acquired.wait(WORKER_WAIT))
        blocked.join()
        self.assertIsNone(blocked.failure)
        self.assertEqual(budget.snapshot()["pending_count"], 0)
        self.assertEqual(budget.snapshot()["materialized_bytes"], 0)

    def test_many_failed_but_retained_takes_cannot_exhaust_the_budget(self):
        budget = BacklogBudget(max_pending=2, max_materialized_bytes=256)
        retained_on_disk = []
        for attempt_index in range(50):
            with budget.reserve() as ticket:
                ticket.materialize(200)
                retained_on_disk.append(attempt_index)  # evidence file kept
        self.assertEqual(len(retained_on_disk), 50)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["materialized_bytes"], 0)
        with budget.reserve() as ticket:
            ticket.materialize(256)
            self.assertEqual(budget.snapshot()["materialized_bytes"], 256)


class BacklogHighWaterTests(unittest.TestCase):
    def test_high_water_overshoot_from_inflight_outputs_does_not_deadlock_the_drain(self):
        budget = BacklogBudget(max_pending=4, max_materialized_bytes=100)
        first_hold = threading.Event()
        early_hold = threading.Event()
        second_hold = threading.Event()
        first = Producer(budget, bytes_to_materialize=60,
                         hold_until=first_hold).start()
        self.assertTrue(wait_until(lambda: first.acquired.is_set()))
        # Granted while unblocked; its output lands only after the mark is
        # already exceeded.
        early_finish = threading.Event()
        early = Producer(budget, bytes_to_materialize=40, hold_until=early_hold,
                         defer_materialize=True,
                         post_materialize_hold=early_finish).start()
        self.assertTrue(wait_until(lambda: early.acquired.is_set()))
        second = Producer(budget, bytes_to_materialize=90,
                          hold_until=second_hold).start()
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["materialized_bytes"] == 150))
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 3)
        self.assertTrue(snapshot["new_generation_blocked"])
        blocked = Producer(budget, bytes_to_materialize=10).start()
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["waiters"] >= 1))
        self.assertFalse(blocked.acquired.is_set())
        # In-flight output lands while the mark is exceeded: never blocked.
        early_hold.set()
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["materialized_bytes"] == 190))
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["materialized_bytes"], 190)
        self.assertTrue(snapshot["new_generation_blocked"])
        self.assertFalse(blocked.acquired.is_set())
        early_finish.set()
        early.join()
        self.assertIsNone(early.failure)
        self.assertEqual(budget.snapshot()["materialized_bytes"], 150)
        # Half of the held bytes releasing is still not below the low water.
        first_hold.set()
        first.join()
        self.assertIsNone(first.failure)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["materialized_bytes"], 90)
        self.assertTrue(snapshot["new_generation_blocked"])
        self.assertFalse(blocked.acquired.is_set())
        # The last release drops below half; the blocked drain resumes.
        second_hold.set()
        second.join()
        self.assertIsNone(second.failure)
        self.assertTrue(blocked.acquired.wait(WORKER_WAIT))
        blocked.join()
        self.assertIsNone(blocked.failure)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["materialized_bytes"], 0)
        self.assertFalse(snapshot["new_generation_blocked"])

    def test_block_resume_hysteresis_requires_dropping_below_half_the_mark(self):
        budget = BacklogBudget(max_pending=3, max_materialized_bytes=100)
        first_hold = threading.Event()
        second_hold = threading.Event()
        first = Producer(budget, bytes_to_materialize=50,
                         hold_until=first_hold).start()
        self.assertTrue(wait_until(lambda: first.acquired.is_set()))
        second = Producer(budget, bytes_to_materialize=50,
                          hold_until=second_hold).start()
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["materialized_bytes"] == 100))
        self.assertTrue(budget.snapshot()["new_generation_blocked"])
        blocked = Producer(budget).start()
        self.assertTrue(wait_until(lambda: budget.snapshot()["waiters"] == 1))
        self.assertFalse(blocked.acquired.is_set())
        # First ASR return leaves 50 bytes: at the low water, not below it.
        first_hold.set()
        first.join()
        self.assertIsNone(first.failure)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["materialized_bytes"], 50)
        self.assertTrue(snapshot["new_generation_blocked"])
        self.assertFalse(blocked.acquired.is_set())
        # Second ASR return drops below half; the waiter resumes.
        second_hold.set()
        second.join()
        self.assertIsNone(second.failure)
        self.assertTrue(blocked.acquired.wait(WORKER_WAIT))
        self.assertFalse(budget.snapshot()["new_generation_blocked"])
        blocked.join()
        self.assertIsNone(blocked.failure)
        self.assertEqual(budget.snapshot()["pending_count"], 0)


class BacklogCancellationTests(unittest.TestCase):
    def test_cancel_while_waiting_wakes_promptly_and_leaks_nothing(self):
        budget = BacklogBudget(max_pending=1)
        holder = Producer(budget, hold_until=threading.Event()).start()
        self.assertTrue(wait_until(lambda: holder.acquired.is_set()))
        cancel_flag = threading.Event()
        waiter = Producer(budget, cancel=cancel_flag.is_set).start()
        self.assertTrue(wait_until(lambda: budget.snapshot()["waiters"] == 1))
        cancel_flag.set()
        waiter.join()
        self.assertIsInstance(waiter.failure, BacklogCancelled)
        self.assertEqual(waiter.failure.reason, "cancel")
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 1)
        self.assertEqual(snapshot["waiters"], 0)
        holder.hold_until.set()
        holder.join()
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_phase_predicate_false_aborts_before_any_waiting(self):
        budget = BacklogBudget(max_pending=2)
        with self.assertRaises(BacklogCancelled) as caught:
            with budget.reserve(phase_ok=lambda: False):
                pass
        self.assertEqual(caught.exception.reason, "phase")
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_close_wakes_every_waiter_and_rejects_new_reservations(self):
        budget = BacklogBudget(max_pending=1)
        holder = Producer(budget, hold_until=threading.Event()).start()
        self.assertTrue(wait_until(lambda: holder.acquired.is_set()))
        waiters = [Producer(budget).start() for _ in range(3)]
        self.assertTrue(wait_until(lambda: budget.snapshot()["waiters"] == 3))
        budget.close()
        for waiter in waiters:
            waiter.join()
            self.assertIsInstance(waiter.failure, BacklogClosed)
        self.assertTrue(budget.snapshot()["closed"])
        self.assertEqual(budget.snapshot()["waiters"], 0)
        with self.assertRaises(BacklogClosed):
            with budget.reserve():
                pass
        # Tickets already granted still release normally after close.
        holder.hold_until.set()
        holder.join()
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_close_is_idempotent_and_can_be_called_again(self):
        budget = BacklogBudget(max_pending=1)
        budget.close()
        budget.close()
        self.assertTrue(budget.snapshot()["closed"])


class BacklogConcurrencyTests(unittest.TestCase):
    def test_cross_thread_asr_release_returns_the_exact_slot(self):
        budget = BacklogBudget(max_pending=1)
        next_hold = threading.Event()
        with budget.reserve() as ticket:
            ticket.materialize(64)
            self.assertEqual(budget.snapshot()["pending_count"], 1)
            next_attempt = Producer(budget, hold_until=next_hold).start()
            self.assertTrue(wait_until(
                lambda: budget.snapshot()["waiters"] == 1))
            asr = threading.Thread(target=ticket.release)
            asr.start()
            asr.join(WORKER_WAIT)
            self.assertTrue(next_attempt.acquired.wait(WORKER_WAIT))
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["pending_count"], 1)
            self.assertEqual(snapshot["materialized_bytes"], 0)
        next_hold.set()
        next_attempt.join()
        self.assertIsNone(next_attempt.failure)
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_concurrent_reserve_materialize_release_respects_the_bound_and_bounds_memory(self):
        budget = BacklogBudget(max_pending=3, max_materialized_bytes=4096)
        threads = 8
        cycles = 60
        barrier = threading.Barrier(threads, timeout=WORKER_WAIT)
        observed_peak = []

        def work():
            barrier.wait()
            peak = 0
            for _ in range(cycles):
                with budget.reserve() as ticket:
                    ticket.materialize(33)
                    peak = max(peak, budget.snapshot()["pending_count"])
                    ticket.update_bytes(17)
            observed_peak.append(peak)

        workers = [threading.Thread(target=work) for _ in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(WORKER_WAIT)
            self.assertFalse(worker.is_alive())
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 0)
        self.assertEqual(snapshot["materialized_bytes"], 0)
        self.assertEqual(snapshot["waiters"], 0)
        self.assertEqual(len(observed_peak), threads)
        self.assertLessEqual(max(observed_peak), 3)

    def test_nested_attempt_token_reset_never_releases_the_wrong_ticket(self):
        budget = BacklogBudget(max_pending=4)
        seen = []
        with budget.reserve() as outer:
            seen.append(("outer-enter", current_ticket() is outer))
            with budget.reserve(new_generation=False,
                                initial_bytes=4) as inner:
                seen.append(("inner-enter", current_ticket() is inner))
                inner.release()
            seen.append(("after-inner", current_ticket() is outer))
            seen.append(("outer-still-live", not outer.released))
        seen.append(("after-outer", current_ticket() is None))
        self.assertEqual(seen, [
            ("outer-enter", True),
            ("inner-enter", True),
            ("after-inner", True),
            ("outer-still-live", True),
            ("after-outer", True),
        ])
        self.assertEqual(budget.snapshot()["pending_count"], 0)


class BacklogDriverTests(unittest.TestCase):
    """Real-thread driver over the minimal public API; gauge-only output."""

    def test_driver_blocked_producer_asr_drain_and_hysteresis_recovery(self):
        budget = BacklogBudget(max_pending=2, max_materialized_bytes=600)

        def report(phase):
            snapshot = budget.snapshot()
            print(f"driver: {phase}: pending={snapshot['pending_count']} "
                  f"bytes={snapshot['materialized_bytes']} "
                  f"blocked={snapshot['new_generation_blocked']} "
                  f"waiters={snapshot['waiters']}")

        report("empty")
        first_hold = threading.Event()
        second_hold = threading.Event()
        third_hold = threading.Event()
        take_one = Producer(budget, bytes_to_materialize=400,
                            hold_until=first_hold).start()
        self.assertTrue(wait_until(lambda: take_one.acquired.is_set()))
        take_two = Producer(budget, bytes_to_materialize=500,
                            hold_until=second_hold).start()
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["materialized_bytes"] == 900))
        report("two_takes_held_soft_mark_overshot")
        snapshot = budget.snapshot()
        self.assertTrue(snapshot["new_generation_blocked"])
        take_three = Producer(budget, bytes_to_materialize=100,
                              hold_until=third_hold).start()
        self.assertTrue(wait_until(lambda: budget.snapshot()["waiters"] == 1))
        self.assertFalse(take_three.acquired.is_set())
        report("third_take_blocked_above_high_water")
        # ASR returns for the second take: 400 remain, not below half.
        second_hold.set()
        take_two.join()
        self.assertIsNone(take_two.failure)
        report("second_asr_returned_still_blocked")
        self.assertTrue(budget.snapshot()["new_generation_blocked"])
        self.assertFalse(take_three.acquired.is_set())
        # ASR returns for the first take: 0 < 300, so the drain resumes the
        # third take; its output lands and its ticket stays with the runner.
        first_hold.set()
        take_one.join()
        self.assertIsNone(take_one.failure)
        self.assertTrue(take_three.acquired.wait(WORKER_WAIT))
        self.assertTrue(wait_until(
            lambda: budget.snapshot()["materialized_bytes"] == 100))
        report("drain_resumed_third_take")
        self.assertIsNone(take_three.failure)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["pending_count"], 1)
        self.assertEqual(snapshot["materialized_bytes"], 100)
        self.assertFalse(snapshot["new_generation_blocked"])
        # Primary ASR returns for the third take on the runner's thread.
        take_three.ticket.release()
        report("third_asr_returned")
        self.assertEqual(budget.snapshot()["pending_count"], 0)
        self.assertEqual(budget.snapshot()["materialized_bytes"], 0)
        third_hold.set()
        take_three.join()

    def test_driver_failed_generation_cleans_up_without_throttling_the_budget(self):
        budget = BacklogBudget(max_pending=1, max_materialized_bytes=1000)
        attempt = Producer(budget, fail_inside=True).start()
        self.assertTrue(wait_until(lambda: attempt.acquired.is_set()))
        attempt.join()
        self.assertIsInstance(attempt.failure, RuntimeError)
        report = budget.snapshot()
        print(f"driver: failure_cleanup: pending={report['pending_count']} "
              f"bytes={report['materialized_bytes']} "
              f"blocked={report['new_generation_blocked']}")
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["materialized_bytes"], 0)
        # The cleaned-up failure does not throttle the next generation.
        retry = Producer(budget, bytes_to_materialize=50).start()
        self.assertTrue(wait_until(lambda: retry.acquired.is_set()))
        retry.join()
        self.assertIsNone(retry.failure)
        self.assertEqual(budget.snapshot()["pending_count"], 0)


class ScopedRuntime:
    """Pipeline-runtime stand-in: backlog budget plus direct leaf execution."""

    def __init__(self, backlog):
        self.backlog = backlog

    def run_preparation(self, function, *args, owner=(None, None), **kwargs):
        return function(*args, **kwargs)

    def run_recognition(self, function, *args, owner=(None, None), **kwargs):
        return function(*args, **kwargs)

    def run_review_leaf(self, function, *args, owner=(None, None), **kwargs):
        return function(*args, **kwargs)

    def run_review_parent(self, function, *args, owner=(None, None), **kwargs):
        return function(*args, **kwargs)


class GaugeProbe:
    """Verifier wrapper recording backlog gauges at every ASR boundary."""

    def __init__(self, inner, budget, samples):
        self.inner = inner
        self.budget = budget
        self.samples = samples
        self.identity = inner.identity

    def transcribe(self, path):
        self.samples.append(("asr", self.budget.snapshot()))
        return self.inner.transcribe(path)


def write_silence(path):
    """A silent waveform fixture; acoustic gates reject it before recognition."""
    count, rate = 44100, 44100
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x00" * count)


class BacklogHookUnitTests(unittest.TestCase):
    def test_hooks_are_noops_without_a_pipeline_scope(self):
        self.assertIsNone(begin_attempt_ticket(True))
        self.assertIsNone(begin_attempt_ticket(False, 12))
        self.assertIsNone(current_ticket())

    def test_attempt_tickets_are_visible_to_adopt_and_nested_attempts(self):
        budget = BacklogBudget(max_pending=2)
        with resource_scope(ScopedRuntime(budget), ("run", 1)):
            handle = begin_attempt_ticket(True)
            self.assertIsNotNone(handle)
            self.assertIs(current_ticket(), handle.ticket)
            self.assertIsNone(begin_attempt_ticket(False, 5))
            nested = begin_attempt_ticket(True)
            self.assertIsNotNone(nested)
            self.assertIs(current_ticket(), nested.ticket)
            nested.release()
            self.assertIs(current_ticket(), handle.ticket)
            handle.release()
        self.assertEqual(budget.snapshot()["pending_count"], 0)

    def test_render_resources_owns_the_budget_for_its_pool_lifetime(self):
        resources = RenderResources(tts_slots=1, synthesis_backlog=1,
                                    review_concurrency=1, max_inflight=1,
                                    pending_audio_mb=64)
        self.addCleanup(resources.close)
        backlog = resources.backlog
        snapshot = backlog.snapshot()
        self.assertEqual(snapshot["max_pending"], 1)
        self.assertEqual(snapshot["max_materialized_bytes"], 64 * 1024 * 1024)
        self.assertEqual(snapshot["low_water_bytes"], 32 * 1024 * 1024)

    def test_resource_close_wakes_a_blocked_attempt_waiter(self):
        resources = RenderResources(tts_slots=1, synthesis_backlog=1,
                                    review_concurrency=1, max_inflight=1,
                                    pending_audio_mb=64)
        self.addCleanup(resources.close)
        held = resources.backlog.acquire(new_generation=True)
        outcome = {}

        def waiter():
            with resource_scope(resources, ("run", 1)):
                try:
                    begin_attempt_ticket(True)
                    outcome["result"] = "acquired"
                except BacklogClosed as error:
                    outcome["result"] = f"closed: {error}"

        thread = threading.Thread(target=waiter)
        thread.start()
        self.assertTrue(wait_until(
            lambda: resources.backlog.snapshot()["waiters"] == 1))
        resources.close()
        thread.join(WORKER_WAIT)
        self.assertFalse(thread.is_alive())
        self.assertTrue(str(outcome.get("result", "")).startswith("closed:"),
                        outcome)
        self.assertTrue(resources.backlog.snapshot()["closed"])


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for backlog integration")
class BacklogAttemptHookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
            keep_intermediate_audio=True,
        )
        self.config["quality"] = validate_policy({})
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)
        self.text = "Hello there."
        self.request = {"kind": "speech", "text": self.text,
                        "engine": self.engine.identity}
        self.budget = BacklogBudget(max_pending=2,
                                    max_materialized_bytes=10 * 1024 * 1024)
        self.samples = []
        self.probe = GaugeProbe(self.verifier, self.budget, self.samples)

    def take(self, generate=None, *, budget=None, verifier=None, attempts=None):
        config = self.config
        if attempts is not None:
            config = deepcopy(self.config)
            config["render"]["max_attempts"] = attempts
        with resource_scope(ScopedRuntime(self.budget), ("run", 1)):
            return verified_take(
                self.text, generate or self.generate, self.root / "takes",
                dict(self.request), verifier or self.probe, config,
                notify=lambda _: None, retry_budget=SpeechRetryBudget(),
            )

    def generate(self, path, seed):
        self.samples.append(("generate", self.budget.snapshot()))
        return self.engine._write(self.text, path, seed)

    def asr_samples(self):
        return [snapshot for phase, snapshot in self.samples if phase == "asr"]

    def generation_samples(self):
        return [snapshot for phase, snapshot in self.samples if phase == "generate"]

    def test_fresh_generation_reserves_before_synthesis_and_releases_after_primary_asr(self):
        wav, accepted = self.take()
        self.assertTrue(accepted["report"]["passed"], accepted)
        generation = self.generation_samples()
        self.assertEqual(len(generation), 1)
        self.assertEqual(generation[0]["pending_count"], 1)
        self.assertEqual(generation[0]["materialized_bytes"], 0)
        landing = self.asr_samples()
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0]["pending_count"], 1)
        expected_bytes = (self.root / "takes" / "attempt-001.wav").stat().st_size
        self.assertEqual(landing[0]["materialized_bytes"], expected_bytes)
        settled = self.budget.snapshot()
        self.assertEqual(settled["pending_count"], 0)
        self.assertEqual(settled["materialized_bytes"], 0)
        self.assertIsNone(current_ticket())

    def test_acoustic_failure_releases_before_recovery_and_keeps_the_failed_wave(self):
        failures = []

        def silent_generation(path, seed):
            failures.append(path)
            write_silence(path)
            return {"fixture": True, "seed": seed}

        with self.assertRaises(SpeechVerificationFailed) as caught:
            self.take(generate=silent_generation, attempts=1)
        self.assertEqual(self.verifier.calls, [])
        self.assertTrue(self.probe.samples == [] or all(
            snapshot["pending_count"] == 1
            for _, snapshot in self.probe.samples))
        self.assertTrue((self.root / "takes" / "attempt-001.wav").exists())
        settled = self.budget.snapshot()
        self.assertEqual(settled["pending_count"], 0)
        self.assertEqual(settled["materialized_bytes"], 0)
        self.assertIn("silent_audio", caught.exception.report.get("reasons", [])
                      + [caught.exception.report.get("error", "")])

    def test_saved_unverified_resume_adopts_without_spending_generation_budget(self):
        takes = self.root / "takes"
        takes.mkdir(parents=True)
        wav = takes / "attempt-001.wav"
        self.engine._write(self.text, wav, 7)
        resume_budget = SpeechRetryBudget()
        request_hash = digest(self.request)
        write_json(takes / "attempt-001.json", {
            "request_hash": request_hash, "request": self.request,
            "attempt": 1, "seed": 7,
            "validation_key": make_validation_key(
                self.text, self.verifier, self.config["quality"]),
            "report": {"passed": False, "reasons": ["generation_incomplete"],
                       "retry_on_resume": True},
            "engine_metadata": {}, "audio_file": wav.name,
            "audio_sha256": file_hash(wav),
        })
        spent = []

        def refuse_generation(path, seed):
            spent.append(path)
            raise AssertionError("a saved-unverified resume must not generate")

        reused, accepted = self.take(generate=refuse_generation,
                                     budget=resume_budget)
        self.assertEqual(reused, wav)
        self.assertTrue(accepted["report"]["passed"], accepted)
        self.assertEqual(spent, [])
        self.assertEqual(resume_budget.started(takes, request_hash), 0)
        landing = self.asr_samples()
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0]["pending_count"], 1)
        self.assertEqual(landing[0]["materialized_bytes"], wav.stat().st_size)
        self.assertEqual(self.budget.snapshot()["pending_count"], 0)

    def test_adopt_waits_for_a_count_slot_and_bypasses_the_byte_high_water(self):
        takes = self.root / "takes"
        takes.mkdir(parents=True)
        wav = takes / "attempt-001.wav"
        self.engine._write(self.text, wav, 11)
        self.budget = BacklogBudget(max_pending=1,
                                    max_materialized_bytes=1000)
        self.probe = GaugeProbe(self.verifier, self.budget, self.samples)
        blocker = self.budget.acquire(new_generation=True)
        blocker.materialize(1000)
        self.assertTrue(self.budget.snapshot()["new_generation_blocked"])
        outcome = {}

        def adopted_verification():
            try:
                with resource_scope(ScopedRuntime(self.budget), ("run", 1)):
                    outcome["report"] = validate_audio(
                        wav, self.text, self.probe, self.config["quality"],
                        purpose="speech")
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=adopted_verification)
        thread.start()
        self.assertTrue(wait_until(
            lambda: self.budget.snapshot()["waiters"] == 1))
        self.assertFalse(outcome)
        blocker.release()
        thread.join(WORKER_WAIT)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome, outcome)
        self.assertTrue(outcome["report"]["passed"], outcome)
        landing = self.asr_samples()
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0]["pending_count"], 1)
        self.assertEqual(self.budget.snapshot()["pending_count"], 0)
        self.assertEqual(self.budget.snapshot()["materialized_bytes"], 0)
    def test_validate_audio_does_not_double_count_while_an_attempt_ticket_is_held(self):
        takes = self.root / "takes"
        takes.mkdir(parents=True)
        wav = takes / "attempt-001.wav"
        self.engine._write(self.text, wav, 3)
        with resource_scope(ScopedRuntime(self.budget), ("run", 1)):
            held = begin_attempt_ticket(True)
            held.materialize(100)
            report = validate_audio(wav, self.text, self.probe,
                                    self.config["quality"], purpose="speech")
            held.release()
        self.assertTrue(report["passed"], report)
        landing = self.asr_samples()
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0]["pending_count"], 1)
        self.assertEqual(landing[0]["materialized_bytes"], 100)
        held.release()
        self.assertEqual(self.budget.snapshot()["pending_count"], 0)
        self.assertEqual(self.budget.snapshot()["materialized_bytes"], 0)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for recovery backlog integration")
class BacklogSplitRecoveryTests(unittest.TestCase):
    """One permit must admit split recovery: parts then the joined assembly."""

    TEXT = "Insang Mortuary Services! What does it—"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG,
                                     ffprobe=FFPROBE, keep_intermediate_audio=True)
        self.voice = {"asset_hash": "fixture-voice", "kind": "reference",
                      "reference_text": "The same character voice."}
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = ChunkedVerifier(self.fixtures)
        self.request = {"kind": "speech", "engine": self.engine.identity,
                        "voice_asset_hash": self.voice["asset_hash"],
                        "base_seed": 1729, "sample_rate": 44100,
                        "chunk": {"spoken_text": self.TEXT,
                                  "segment": {"delivery": {"direction": "A tired receptionist."}},
                                  "source_ranges": [{"start": 10, "end": 10 + len(self.TEXT)}]}}
        self.parent = self.root / "clips" / digest(self.request)
        self.budget = BacklogBudget(max_pending=1,
                                    max_materialized_bytes=150 * 1024)
        self.samples = []
        self.probe = GaugeProbe(self.verifier, self.budget, self.samples)
        self.calls = []

    def synthesize(self, text, path, seed):
        self.calls.append((text, seed))
        return self.engine._write(text, path, seed)

    def test_parts_and_join_recurse_under_a_single_permit_without_deadlock(self):
        with resource_scope(ScopedRuntime(self.budget), ("run", 1)):
            wav, accepted = recover_speech(
                self.TEXT, self.synthesize, self.parent, self.request,
                self.probe, self.config, notify=lambda _: None,
                retry_budget=SpeechRetryBudget())
        self.assertTrue(accepted["report"]["passed"], accepted)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.verifier.calls), 3)
        landing = [snapshot for _, snapshot in self.samples]
        self.assertEqual(len(landing), 3)
        self.assertEqual(max(snapshot["pending_count"] for snapshot in landing), 1)
        self.assertTrue(all(snapshot["materialized_bytes"] > 0
                            for snapshot in landing))
        settled = self.budget.snapshot()
        self.assertEqual(settled["pending_count"], 0)
        self.assertEqual(settled["materialized_bytes"], 0)
        self.assertFalse(settled["new_generation_blocked"])
        self.assertIsNone(current_ticket())


class ChunkedVerifier:
    """Decodes concatenated one-second fixture tones; assembly-safe ASR fake."""

    identity = {"backend": "chunked-recovery-fixture"}

    def __init__(self, fixtures):
        self.fixtures = fixtures
        self.calls = []

    def transcribe(self, path):
        self.calls.append(Path(path))
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            data = source.readframes(frames)
        words = [self.fixtures[hashlib.sha256(data[start:start + 88200]).hexdigest()]
                 for start in range(0, len(data), 88200)]
        return timed_transcript("".join(words))


if __name__ == "__main__":
    unittest.main()
