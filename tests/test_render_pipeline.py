"""Fair logical-clip scheduling: bounded, chapter-fair, cancellable, leak-free."""

from concurrent.futures import CancelledError
from contextvars import ContextVar
from threading import Event, Lock, Thread
import threading
import unittest

from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition
from epub_tts_narrator.renderer.render_pipeline import LogicalClipScheduler, chapter_share


class Starts:
    """Records body starts under a lock with one Event per submission."""

    def __init__(self, test):
        self.test = test
        self.lock = Lock()
        self.order = []
        self.events = {}

    def record(self, name):
        with self.lock:
            self.order.append(name)
            self.events.setdefault(name, Event()).set()

    def expect(self, name, timeout=10.0):
        self.events.setdefault(name, Event())
        self.test.assertTrue(self.events[name].wait(timeout),
                             f"{name} never started; order={self.order}")

    def expect_absent(self, *names):
        with self.lock:
            present = [name for name in names if name in self.order]
        self.test.assertEqual(present, [], f"started too early; order={self.order}")


class SchedulerTestCase(unittest.TestCase):
    def setUp(self):
        self.starts = Starts(self)
        self.scheduler = None

    def start_scheduler(self, limit, chapters, **kwargs):
        self.scheduler = LogicalClipScheduler(limit, chapters, **kwargs)
        self.addCleanup(self.scheduler.close)
        return self.scheduler

    def body(self, name, gate=None, result=None, error=None):
        def run():
            self.starts.record(name)
            if gate is not None:
                self.assertTrue(gate.wait(30.0), f"{name} gate never opened")
            if error is not None:
                raise error
            return result if result is not None else name
        return run

    def submit(self, owner, name, **kwargs):
        future = self.scheduler.submit(owner, self.body(name, **kwargs))
        return future


class ShareTests(unittest.TestCase):
    def test_share_floors_and_never_drops_below_one(self):
        self.assertEqual(chapter_share(128, 4), 32)
        self.assertEqual(chapter_share(8, 4), 2)
        self.assertEqual(chapter_share(2, 4), 1)
        self.assertEqual(chapter_share(3, 4), 1)
        self.assertEqual(chapter_share(1, 16), 1)
        for limit, chapters in ((0, 4), (-1, 4), (4, 0), (4, -2), (True, 4), (4, "4"), (None, 4)):
            with self.subTest(limit=limit, chapters=chapters):
                with self.assertRaises(ValueError):
                    chapter_share(limit, chapters)


class FairnessTests(SchedulerTestCase):
    def test_late_chapter_starts_while_first_chapter_fills_its_own_share(self):
        scheduler = self.start_scheduler(2, 2)  # share = 1
        first_gate, second_gate = Event(), Event()
        first = self.submit("book:1", "1-a", gate=first_gate)
        self.submit("book:1", "1-b")
        self.submit("book:2", "2-a", gate=second_gate)
        self.submit("book:2", "2-b")
        self.starts.expect("1-a")
        self.starts.expect("2-a")  # a reserved slot remained for the later chapter
        self.starts.expect_absent("1-b", "2-b")  # each chapter's share is non-borrowable
        first_gate.set()
        self.starts.expect("1-b")
        self.starts.expect_absent("2-b")
        second_gate.set()
        self.starts.expect("2-b")
        self.assertEqual(first.result(10), "1-a")

    def test_one_worker_slot_rotates_round_robin_across_chapters(self):
        scheduler = self.start_scheduler(1, 2, pending_limit=8)  # share = 1
        gate = Event()
        priming = self.submit("book:0", "0-a", gate=gate)
        futures = [self.submit("book:1", "1-a", gate=gate), self.submit("book:2", "2-a", gate=gate),
                   self.submit("book:1", "1-b", gate=gate), self.submit("book:2", "2-b", gate=gate)]
        self.starts.expect("0-a")
        self.assertEqual(self.starts.order, ["0-a"])
        gate.set()
        self.assertEqual(priming.result(10), "0-a")
        for future in futures:
            future.result(10)
        # Once both chapters hold queued work, the single slot strictly
        # alternates owners instead of draining one chapter first.
        self.assertEqual(self.starts.order, ["0-a", "1-a", "2-a", "1-b", "2-b"])

    def test_global_limit_never_grows_with_registered_or_configured_chapters(self):
        scheduler = self.start_scheduler(2, 4)  # share = 1
        gates = {"1-a": Event(), "2-a": Event()}
        first = self.submit("book:1", "1-a", gate=gates["1-a"])
        second = self.submit("book:2", "2-a", gate=gates["2-a"])
        self.starts.expect("2-a")
        third = self.submit("book:3", "3-a")
        self.starts.expect_absent("3-a")  # both global slots are occupied
        gates["1-a"].set()
        self.starts.expect("3-a")  # the freed slot rotates past the idle chapters
        self.assertEqual(first.result(10), "1-a")
        gates["2-a"].set()
        self.assertEqual(second.result(10), "2-a")
        self.assertEqual(third.result(10), "3-a")

    def test_each_chapter_keeps_source_fifo_order(self):
        scheduler = self.start_scheduler(2, 2)
        gate = Event()
        futures = [self.submit("book:1", f"1-{index}", gate=gate) for index in "abcd"]
        self.submit("book:2", "2-a")
        self.starts.expect("2-a")
        self.starts.expect_absent("1-b", "1-c", "1-d")
        gate.set()
        self.assertEqual([future.result(10) for future in futures], ["1-a", "1-b", "1-c", "1-d"])


class BoundedPendingTests(SchedulerTestCase):
    def test_submit_backpressures_when_the_pending_bound_is_full(self):
        scheduler = self.start_scheduler(2, 1, pending_limit=1)  # share = 2
        gate = Event()
        running = [self.submit("book:1", "1-a", gate=gate), self.submit("book:1", "1-b", gate=gate)]
        queued = self.submit("book:1", "1-c", gate=gate)  # pending = 1, the bound
        completed = Event()
        accepted = []

        def submit_fourth():
            scheduler.submit("book:1", self.body("1-d", gate=gate))
            accepted.append(True)
            completed.set()

        helper = Thread(target=submit_fourth)
        helper.start()
        self.assertFalse(completed.wait(0.5), "submit returned while pending was full")
        gate.set()
        for future in running + [queued]:
            future.result(10)
        self.assertTrue(completed.wait(10))
        helper.join(10)
        self.assertEqual(accepted, [True])
        self.starts.expect("1-d")

    def test_cancelling_queued_work_releases_pending_bound(self):
        scheduler = self.start_scheduler(1, 1, pending_limit=1)
        gate = Event()
        running = self.submit("book:1", "1-a", gate=gate)
        self.starts.expect("1-a")
        queued = self.submit("book:1", "1-b", gate=gate)
        accepted = Event()

        def submit_third():
            scheduler.submit("book:1", self.body("1-c", gate=gate))
            accepted.set()

        helper = Thread(target=submit_third)
        helper.start()
        self.assertFalse(accepted.wait(0.5))
        self.assertEqual(scheduler.cancel_owner("book:1"), 1)
        with self.assertRaises(CancelledError):
            queued.result(10)
        self.assertTrue(accepted.wait(10))
        helper.join(10)
        gate.set()
        self.assertEqual(running.result(10), "1-a")


class CancellationAndCloseTests(SchedulerTestCase):
    def test_cancel_owner_cancels_only_queued_siblings_of_that_owner(self):
        scheduler = self.start_scheduler(2, 2)
        gate = Event()
        running = self.submit("book:1", "1-a", gate=gate)
        self.starts.expect("1-a")
        sibling = self.submit("book:1", "1-b")
        other = self.submit("book:2", "2-a", gate=gate)
        self.assertEqual(scheduler.cancel_owner("book:1"), 1)
        self.assertTrue(sibling.cancelled())
        gate.set()
        self.assertEqual(running.result(10), "1-a")  # already-started work drains
        self.assertEqual(other.result(10), "2-a")

    def test_cancel_of_unknown_or_active_owner_cancels_nothing(self):
        scheduler = self.start_scheduler(1, 1)
        gate = Event()
        running = self.submit("book:1", "1-a", gate=gate)
        self.starts.expect("1-a")
        self.assertEqual(scheduler.cancel_owner("missing"), 0)
        self.assertEqual(scheduler.cancel_owner("book:1"), 0)
        gate.set()
        self.assertEqual(running.result(10), "1-a")

    def test_close_cancels_queued_work_drains_running_and_stops_submission(self):
        baseline = threading.active_count()
        scheduler = LogicalClipScheduler(2, 2)
        gate = Event()
        running = scheduler.submit("book:1", self.body("1-a", gate=gate))
        self.starts.expect("1-a")
        queued = scheduler.submit("book:1", self.body("1-b"))
        scheduler.close(wait=False)
        self.assertTrue(queued.cancelled())
        with self.assertRaises(RuntimeError):
            scheduler.submit("book:1", self.body("1-c"))
        scheduler.close()  # idempotent
        gate.set()
        self.assertEqual(running.result(10), "1-a")
        self.settle_threads(baseline)

    def settle_threads(self, baseline):
        settled = Event()
        while threading.active_count() > baseline and not settled.wait(0.05):
            pass
        self.assertLessEqual(threading.active_count(), baseline)

    def test_errors_propagate_unchanged_and_close_only_that_owner(self):
        scheduler = self.start_scheduler(2, 2)
        failing = self.submit("book:1", "1-a", error=ValueError("boom"))
        with self.assertRaises(ValueError):
            failing.result(10)
        # The failed chapter's admission is closed; other owners keep running.
        with self.assertRaises(RuntimeError):
            self.submit("book:1", "1-b")
        healthy = self.submit("book:2", "2-a")
        self.assertEqual(healthy.result(10), "2-a")
        # Retiring the drained owner reopens it for a fresh generation.
        self.assertTrue(scheduler.retire_owner("book:1"))
        retried = self.submit("book:1", "1-b")
        self.assertEqual(retried.result(10), "1-b")

    def test_close_with_pending_limit_waiters_wakes_blocked_submitters(self):
        baseline = threading.active_count()
        scheduler = LogicalClipScheduler(1, 1, pending_limit=1)
        gate = Event()
        running = scheduler.submit("book:1", self.body("1-a", gate=gate))
        self.starts.expect("1-a")
        scheduler.submit("book:1", self.body("1-b", gate=gate))
        errors = []

        def blocked():
            try:
                scheduler.submit("book:1", self.body("1-c", gate=gate))
            except RuntimeError as error:
                errors.append(error)

        helper = Thread(target=blocked)
        helper.start()
        helper.join(0.5)
        scheduler.close(wait=False)
        helper.join(10)
        self.assertEqual(len(errors), 1)
        gate.set()
        running.result(10)
        self.settle_threads(baseline)


class ContextAndStopTests(SchedulerTestCase):
    def test_each_submission_gets_a_fresh_copied_context(self):
        scheduler = self.start_scheduler(1, 1)
        scope = ContextVar("pipeline-test-scope", default=None)
        scope.set("outside")
        observed = scheduler.submit("book:1", scope.get)
        self.assertEqual(observed.result(10), "outside")
        scope.set("changed")

        def mutator():
            scope.set("inside")
            return scope.get()

        inside = scheduler.submit("book:1", mutator)
        self.assertEqual(inside.result(10), "inside")
        self.assertEqual(scope.get(), "changed")  # worker writes never leak back

    def test_scoped_stop_cancels_queued_work_before_its_body_starts(self):
        scheduler = self.start_scheduler(1, 1)
        gate = Event()
        running = self.submit("book:1", "1-a", gate=gate)
        self.starts.expect("1-a")
        with scoped_stop_condition(lambda: True, RenderPaused("owner lost")):
            queued = self.submit("book:1", "1-b")
        gate.set()
        self.assertEqual(running.result(10), "1-a")
        with self.assertRaises(RenderPaused):
            queued.result(10)
        self.starts.expect_absent("1-b")
        # The owner-loss stop closed this chapter's admission; retiring the
        # drained owner opens a fresh scheduling generation.
        with self.assertRaises(RuntimeError):
            self.submit("book:1", "1-c")
        self.assertTrue(scheduler.retire_owner("book:1"))
        recovered = self.submit("book:1", "1-c")
        self.assertEqual(recovered.result(10), "1-c")

    def test_chapter_coordinator_context_is_never_the_logical_executor(self):
        scheduler = self.start_scheduler(1, 1)
        marker = {}
        done = Event()

        def coordinate():
            marker["coordinator"] = scheduler.on_logical_executor()
            observed = scheduler.submit("book:1", scheduler.on_logical_executor).result(10)
            marker["clip"] = observed
            done.set()

        thread = Thread(target=coordinate)
        thread.start()
        self.assertTrue(done.wait(10))
        thread.join(10)
        self.assertFalse(marker["coordinator"])
        self.assertTrue(marker["clip"])

    def test_invalid_construction_fails_before_creating_any_threads(self):
        baseline = threading.active_count()
        for limit, chapters in ((0, 4), (4, 0), (2, 0), ("8", 4), (8, 2.5)):
            with self.subTest(limit=limit, chapters=chapters):
                with self.assertRaises(ValueError):
                    LogicalClipScheduler(limit, chapters)
        for pending in (0, -1, "4", True):
            with self.subTest(pending=pending):
                with self.assertRaises(ValueError):
                    LogicalClipScheduler(4, 4, pending_limit=pending)
        self.assertEqual(threading.active_count(), baseline)


class CollectTests(SchedulerTestCase):
    def test_collect_returns_results_in_source_order_not_completion_order(self):
        scheduler = self.start_scheduler(2, 2)
        gate = Event()
        slow = self.submit("book:1", "1-a", gate=gate, result="slow")
        fast = self.submit("book:2", "2-a", result="fast")
        self.assertEqual(fast.result(10), "fast")
        gate.set()
        self.assertEqual(scheduler.collect([slow, fast]), ["slow", "fast"])


if __name__ == "__main__":
    unittest.main()
