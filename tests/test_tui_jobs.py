"""The serial background job queue shared by full-screen interfaces."""

import threading
import unittest

from epub_tts_narrator.tui_jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING, JobQueue
from epub_tts_narrator.tui_support import INSTALL_HINT, add_plain_option, wants_tui


class FakeTerminal:
    def __init__(self, tty=True):
        self.tty = tty

    def isatty(self):
        return self.tty


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.changes = []
        self.queue = JobQueue(on_change=lambda job: self.changes.append((job.job_id, job.state)),
                              describe_error=lambda error: "described: " + str(error))
        self.addCleanup(self.queue.close)

    def test_jobs_run_one_at_a_time_in_submission_order(self):
        order, gate = [], threading.Event()
        self.queue.submit("first", lambda: (gate.wait(5), order.append(1)))
        self.queue.submit("second", lambda: order.append(2))
        self.assertEqual([job.state for job in self.queue.jobs()][1], QUEUED)
        gate.set()
        self.assertTrue(self.queue.wait_idle(5))
        self.assertEqual(order, [1, 2])
        self.assertEqual([job.state for job in self.queue.jobs()], [DONE, DONE])

    def test_a_failure_is_recorded_with_described_text_and_later_jobs_still_run(self):
        def boom():
            raise ValueError("disk full")
        failed = self.queue.submit("fails", boom)
        later = self.queue.submit("later", lambda: "ok")
        self.queue.wait_idle(5)
        self.assertEqual((failed.state, failed.error, failed.error_type), (FAILED, "described: disk full", "ValueError"))
        self.assertEqual((later.state, later.result), (DONE, "ok"))
        self.assertEqual(self.queue.failures(), [failed])

    def test_retry_runs_the_same_work_again_and_supersedes_the_failure(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("try again")
            return len(attempts)
        failed = self.queue.submit("flaky", flaky, scope={"voice": ["v1"]})
        self.queue.wait_idle(5)
        retry = self.queue.retry(failed.job_id)
        self.queue.wait_idle(5)
        self.assertEqual((retry.state, retry.result, retry.retry_of, retry.scope), (DONE, 2, failed.job_id, {"voice": ["v1"]}))
        self.assertEqual(self.queue.failures(), [])
        with self.assertRaisesRegex(ValueError, "Only a failed"):
            self.queue.retry(retry.job_id)

    def test_queued_jobs_can_be_cancelled_but_the_running_one_finishes(self):
        gate, started, ran = threading.Event(), threading.Event(), []
        running = self.queue.submit("running", lambda: (started.set(), gate.wait(5), ran.append("running")))
        self.assertTrue(started.wait(5))
        queued = self.queue.submit("queued", lambda: ran.append("queued"))
        self.assertEqual(self.queue.cancel_queued(), [queued])
        gate.set()
        self.queue.wait_idle(5)
        self.assertEqual((running.state, queued.state, ran), (DONE, CANCELLED, ["running"]))

    def test_active_scopes_report_running_over_queued(self):
        gate = threading.Event()
        started = threading.Event()
        self.queue.submit("a", lambda: (started.set(), gate.wait(5)), scope={"voice": ["v1"]})
        self.queue.submit("b", lambda: None, scope={"voice": ["v1", "v2"]})
        started.wait(5)
        self.assertEqual(self.queue.active_scopes("voice"), {"v1": RUNNING, "v2": QUEUED})
        gate.set()
        self.queue.wait_idle(5)
        self.assertEqual(self.queue.active_scopes("voice"), {})

    def test_a_closed_queue_refuses_new_work(self):
        self.queue.close()
        with self.assertRaises(RuntimeError):
            self.queue.submit("late", lambda: None)


class WantsTuiTests(unittest.TestCase):
    def decide(self, *, plain=False, tty=True, term="xterm-256color", available=True):
        warnings = []
        parser = add_plain_option(__import__("argparse").ArgumentParser())
        args = parser.parse_args(["--plain"] if plain else [])
        result = wants_tui(args, stdin=FakeTerminal(tty), stdout=FakeTerminal(tty),
                           environ={"TERM": term} if term else {}, available=available, warn=warnings.append)
        return result, warnings

    def test_only_an_interactive_terminal_with_textual_and_without_plain_gets_the_tui(self):
        self.assertEqual(self.decide(), (True, []))
        self.assertEqual(self.decide(plain=True), (False, []))
        self.assertEqual(self.decide(tty=False), (False, []))
        self.assertEqual(self.decide(term="dumb"), (False, []))
        self.assertEqual(self.decide(term=None), (False, []))

    def test_a_terminal_without_textual_gets_one_install_hint_and_the_line_interface(self):
        self.assertEqual(self.decide(available=False), (False, [INSTALL_HINT]))


class LaneTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.queue = JobQueue(describe_error=str)
        self.addCleanup(self.queue.close)
        self.queue.add_lane("generate", 3, opening=lambda: self.events.append("open"),
                            closing=lambda: self.events.append("close"))

    def test_lane_jobs_run_together_up_to_the_limit_and_never_beside_other_jobs(self):
        gate, started, lock = threading.Event(), [], threading.Lock()
        running = {"now": 0, "most": 0}

        def work(name):
            def run():
                with lock:
                    running["now"] += 1
                    running["most"] = max(running["most"], running["now"])
                    started.append(name)
                gate.wait(5)
                with lock:
                    running["now"] -= 1
            return run
        for name in "abcd":
            self.queue.submit(name, work(name), lane="generate")
        self.queue.submit("save", lambda: self.events.append(("save", running["now"])))
        deadline = threading.Event()
        for _ in range(100):
            if len(started) == 3:
                break
            deadline.wait(0.02)
        self.assertEqual(sorted(started), ["a", "b", "c"])  # the fourth waits for a free slot
        gate.set()
        self.assertTrue(self.queue.wait_idle(5))
        self.assertEqual(running["most"], 3)
        self.assertEqual(self.events, ["open", "close", ("save", 0)])  # the lane closes before other work

    def test_a_failed_opening_fails_the_waiting_job_and_the_queue_goes_on(self):
        def refuse():
            raise RuntimeError("no designer")
        self.queue.add_lane("broken", 2, opening=refuse)
        failed = self.queue.submit("voice", lambda: "never", lane="broken")
        later = self.queue.submit("later", lambda: "ok")
        self.assertTrue(self.queue.wait_idle(5))
        self.assertEqual((failed.state, failed.error), (FAILED, "no designer"))
        self.assertEqual(later.state, DONE)
        self.assertEqual(self.queue.retry(failed.job_id).lane, "broken")

    def test_an_independent_lane_does_not_wait_for_long_work(self):
        self.queue.add_independent_lane("save")
        gate, started = threading.Event(), threading.Event()
        long_job = self.queue.submit("generate", lambda: (started.set(), gate.wait(5)))
        self.assertTrue(started.wait(2))
        saved = threading.Event()
        save = self.queue.submit("save", saved.set, lane="save")
        self.assertTrue(saved.wait(2))
        self.assertEqual(long_job.state, RUNNING)
        gate.set()
        self.assertTrue(self.queue.wait_idle(5))
        self.assertEqual(save.state, DONE)


if __name__ == "__main__":
    unittest.main()
