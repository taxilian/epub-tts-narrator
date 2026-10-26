"""Connected-client session contracts for the chapter service.

Regression coverage for automatic queue execution: a server restart, a stale
queue, an observer, or an unrelated book's connection must never generate for
a client that is not explicitly connected. A lost client's in-flight take
drains at a durable boundary and its run metadata is then removed; accepted
audio stays on disk, and a rerun is a fresh run of the connected client.
"""

from pathlib import Path
import tempfile
import threading
import time
import unittest

from epub_tts_narrator.renderer.chapter_client import ChapterClientError, ClientHeartbeat
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_sessions import ClientSessions
from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition, stop_if_requested
from shared_fixtures import service_session
from test_chapter_jobs import Executor, specification


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def tick(self, seconds=1000.0):
        self.now += seconds

    def __call__(self):
        return self.now


class ClientSessionsTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.sessions = ClientSessions(30, clock=self.clock)

    def test_open_sessions_stay_live_until_the_ttl_passes(self):
        session = "a" * 64
        self.sessions.open(session)
        self.assertTrue(self.sessions.live(session))
        self.clock.tick(29)
        self.assertTrue(self.sessions.live(session))
        self.clock.tick(2)
        self.assertFalse(self.sessions.live(session))

    def test_invalid_session_ids_are_rejected(self):
        for value in ("nothex", "A" * 64, "b" * 63, 42, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.sessions.open(value)
        self.assertFalse(self.sessions.live("c" * 64))

    def test_heartbeat_renews_only_an_existing_live_session(self):
        session = "d" * 64
        self.sessions.open(session)
        self.clock.tick(20)
        self.sessions.heartbeat(session)
        self.clock.tick(20)
        self.assertTrue(self.sessions.live(session))
        with self.assertRaises(KeyError):
            self.sessions.heartbeat("e" * 64)
        self.clock.tick(11)
        self.assertFalse(self.sessions.live(session))
        # A late or forged heartbeat can never resurrect an expired session.
        with self.assertRaisesRegex(KeyError, "expired"):
            self.sessions.heartbeat(session)
        self.assertFalse(self.sessions.live(session))

    def test_revoke_is_immediate_and_idempotent(self):
        session = "f" * 64
        self.sessions.open(session)
        self.assertTrue(self.sessions.revoke(session))
        self.assertFalse(self.sessions.live(session))
        self.assertFalse(self.sessions.revoke(session))


class PausingExecutor(Executor):
    """Blocks inside synthesis and honors scoped stop conditions like an engine."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def synthesize(self, spec, voices, work, notify, retry_budget=None):
        self.generated.append(spec["chapter"]["number"])
        self.budgets.append(retry_budget)
        self.entered.set()
        while not self.release.wait(0.01):
            stop_if_requested()
        if self.fail:
            raise ValueError("fixture failure")
        return {}


class ConnectedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = PausingExecutor()
        self.service = ChapterJobs(self.root / "server", self.engine, {"assembly_workers": 2})
        self.addCleanup(self.service.close)
        self.engine.release.set()

    def submit(self, number=1, submission=None, service=None, session=None):
        service = service if service is not None else self.service
        session = session if session is not None else service_session(service)
        spec = specification(service, number)
        return service.submit(spec, digest(spec), submission, session)

    def wait_for(self, job_id, states):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            status = self.service.status(job_id)
            if status["state"] in states:
                return status
            time.sleep(0.01)
        self.fail(f"Job did not reach {states}: {status}")

    def test_capabilities_advertise_connected_client_sessions(self):
        capabilities = self.service.capabilities()
        self.assertEqual(capabilities["client_sessions"], 1)
        self.assertEqual(capabilities["session_ttl_seconds"], 30)

    def test_execution_requires_an_explicitly_opened_live_session(self):
        with self.assertRaisesRegex(ValueError, "connected client"):
            spec = specification(self.service)
            self.service.submit(spec, digest(spec))
        session = service_session(self.service)
        job = self.submit(1, session=session)
        self.service.sessions.revoke(session)
        with self.assertRaisesRegex(ValueError, "unknown or expired"):
            spec = specification(self.service)
            self.service.submit(spec, digest(spec), None, session)
        with self.assertRaisesRegex(ValueError, "unknown or expired"):
            self.service.retry(job["id"], session=session)

    def test_old_pending_jobs_without_owner_never_execute_for_a_new_book_client(self):
        old_client = service_session(self.service)
        stale_job = self.submit(1, session=old_client)
        other_stale = self.submit(2, session=old_client)
        self.service.sessions.revoke(old_client)
        self.engine.release.clear()
        self.service.start()
        # The connected new book gets all generation; detached runs were
        # pruned at revoke and can never execute or linger.
        live_job = self.submit(3)
        deadline = time.monotonic() + 3
        while self.engine.generated != [3] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.engine.generated, [3])
        for stale in (stale_job, other_stale):
            with self.assertRaises(KeyError):
                self.service.status(stale["id"])
        self.engine.release.set()
        self.wait_for(live_job["id"], {"complete"})
        self.assertEqual(self.engine.generated, [3])
        # An explicit connected resubmission recreates the chapter identity
        # and executes it as this new client's own run.
        reattached = self.submit(1, service=self.service, session=service_session(self.service))
        self.assertEqual(reattached["id"], stale_job["id"])
        self.wait_for(stale_job["id"], {"complete"})

    def test_restart_generates_nothing_until_a_client_reattaches(self):
        job = self.submit()
        self.service.close()
        self.engine = PausingExecutor()
        self.engine.release.set()
        restarted = ChapterJobs(self.root / "server", self.engine, {"assembly_workers": 2})
        self.addCleanup(restarted.close)
        self.service = restarted
        self.service.start()
        time.sleep(0.3)
        self.assertEqual(self.engine.generated, [])
        # A restart clears every session and removes past-run rows.
        with self.assertRaises(KeyError):
            self.service.status(job["id"])
        self.submit(1, service=restarted, session=service_session(restarted))
        self.wait_for(job["id"], {"complete"})

    def test_lost_client_drains_at_a_boundary_and_its_run_metadata_is_removed(self):
        self.engine.release.clear()
        job = self.submit(1, "d" * 64, session=service_session(self.service))
        self.service.start()
        self.assertTrue(self.engine.entered.wait(3))
        session = self.service._owners[job["id"]]
        self.service.sessions.revoke(session)
        # The in-flight take drains at its durable boundary; the abandoned
        # run's row is then removed while accepted audio stays on disk.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                self.service.status(job["id"])
            except KeyError:
                break
            time.sleep(0.01)
        else:
            self.fail("The abandoned run's row was not removed")
        self.assertEqual(self.engine.generated, [1])
        self.engine.release.set()
        # A rerun is a fresh run: its own submission budget and job identity.
        resumed = self.submit(1, "e" * 64)
        self.assertEqual(resumed["id"], job["id"])
        self.wait_for(job["id"], {"complete"})
        self.assertIsNot(self.engine.budgets[0], self.engine.budgets[1])
        self.assertEqual(self.engine.generated, [1, 1])

    def test_disconnect_before_an_assembly_slot_opens_removes_the_run(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.service.start()
        entered = [threading.Event(), threading.Event()]

        def occupy_slot(ready):
            ready.set()
            gate.wait(5)

        for ready in entered:
            self.service._assembly.submit(occupy_slot, ready)
        self.assertTrue(all(ready.wait(3) for ready in entered))
        session = service_session(self.service)
        job = self.submit(session=session)
        self.wait_for(job["id"], {"assembling"})
        self.service.revoke_session(session)
        gate.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                self.service.status(job["id"])
            except KeyError:
                break
            time.sleep(0.01)
        else:
            self.fail("The abandoned run's row was not removed")

    def test_live_owner_conflicts_and_expired_owner_rebinds_safely(self):
        self.engine.release.clear()
        first = service_session(self.service)
        job = self.submit(1, session=first)
        self.service.start()
        self.assertTrue(self.engine.entered.wait(3))
        rival = service_session(self.service)
        spec = specification(self.service)
        with self.assertRaisesRegex(RuntimeError, "Another connected client"):
            self.service.submit(spec, digest(spec), None, rival)
        self.service.sessions.revoke(first)
        # An explicit reattach while the old take is still draining keeps the
        # new live owner's grant: the drained generation never deletes it.
        rebound = self.service.submit(spec, digest(spec), None, rival)
        self.assertEqual(rebound["id"], job["id"])
        self.engine.release.set()
        self.assertEqual(self.wait_for(job["id"], {"complete"})["state"], "complete")
        self.assertEqual(self.engine.generated, [1, 1])
        self.assertEqual(self.service.status(job["id"])["owner_connected"], True)

    def test_removed_run_does_not_affect_a_sibling_connected_job(self):
        first = service_session(self.service)
        removed = self.submit(1, session=first)
        self.engine.release.clear()
        self.service.start()
        self.assertTrue(self.engine.entered.wait(3))
        sibling = self.submit(2, session=service_session(self.service))
        self.service.sessions.revoke(first)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                self.service.status(removed["id"])
            except KeyError:
                break
            time.sleep(0.01)
        else:
            self.fail("The abandoned run's row was not removed")
        self.engine.release.set()
        self.assertEqual(self.wait_for(sibling["id"], {"complete"})["state"], "complete")
        self.assertEqual(self.engine.generated, [1, 2])

    def test_status_and_list_reports_flags_without_leaking_sessions(self):
        session = service_session(self.service)
        job = self.submit(1, session=session)
        status = self.service.status(job["id"])
        self.assertTrue(status["owner_connected"])
        self.assertFalse(status["paused"])
        listed = self.service.list_jobs()
        self.assertEqual([entry["id"] for entry in listed], [job["id"]])
        self.assertEqual(set(listed[0]), {"id", "state", "paused", "owner_connected", "chapter_number",
                                          "total_chapters", "source_fingerprint", "updated", "last_event"})
        self.service.revoke_session(session)
        # A detached run's row is pruned; nothing paused lingers for observers.
        with self.assertRaises(KeyError):
            self.service.status(job["id"])
        self.assertEqual(self.service.list_jobs(), [])
        for view in (self.service.list_jobs(),):
            self.assertNotIn(session, repr(view))

    def test_list_jobs_orders_active_first_and_validates_limit(self):
        done = self.submit(1)
        self.service._set_state(done["id"], "complete")
        pending = self.submit(2, session=service_session(self.service))
        listed = self.service.list_jobs()
        self.assertEqual([entry["id"] for entry in listed], [pending["id"], done["id"]])
        self.assertEqual(listed[0]["chapter_number"], 2)
        self.assertEqual(listed[0]["total_chapters"], 3)
        self.assertEqual(listed[1]["state"], "complete")
        self.assertEqual(self.service.list_jobs(limit=1), [listed[0]])
        for limit in (0, 257, "10", None):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.service.list_jobs(limit=limit)


class ClientHeartbeatTests(unittest.TestCase):
    def test_isolated_connection_failure_recovers_and_resets_failure_streak(self):
        renewed = threading.Event()
        calls = []

        class IntermittentTransport:
            def request(self, method, path, payload=None, timeout=None):
                calls.append(path)
                if len(calls) in {1, 3}:
                    raise ChapterClientError("connection failed")
                if len(calls) >= 4:
                    renewed.set()
                return {"renewed": True}

        heartbeat = ClientHeartbeat(IntermittentTransport(), "d" * 64, 0.01, timeout=1.0)
        heartbeat.start()
        try:
            self.assertTrue(renewed.wait(2), "Heartbeat stopped instead of renewing after a transient failure")
        finally:
            heartbeat.stop()
        heartbeat.check()
        self.assertGreaterEqual(len(calls), 4)

    def test_expired_session_is_not_retried_or_reopened(self):
        calls = []

        class ExpiredTransport:
            def request(self, method, path, payload=None, timeout=None):
                calls.append((method, path))
                raise ChapterClientError("expired session", status=404)

        heartbeat = ClientHeartbeat(ExpiredTransport(), "e" * 64, 0.01, timeout=1.0)
        heartbeat.start()
        heartbeat._thread.join(2)
        heartbeat.stop()
        with self.assertRaises(ChapterClientError):
            heartbeat.check()
        self.assertEqual(calls, [("POST", f"v1/sessions/{heartbeat.session_id}/heartbeat")])

    def make_client(self, calls, *, fail_after=None):
        class FakeTransport:
            settings = {"timeout_seconds": 5}
            request_failures = fail_after

            def request(self, method, path, payload=None, timeout=None):
                calls.append((method, path, timeout))
                if fail_after is not None and len(calls) > fail_after:
                    raise ChapterClientError("connection failed")
                return {"renewed": True}

        return FakeTransport()

    def test_heartbeat_beats_until_stopped_and_joins_before_returning(self):
        calls = []
        heartbeat = ClientHeartbeat(self.make_client(calls), "a" * 64, 0.01, timeout=1.0)
        heartbeat.start()
        deadline = time.monotonic() + 2
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        heartbeat.stop()
        beats = len(calls)
        time.sleep(0.05)
        self.assertEqual(len(calls), beats)
        self.assertGreaterEqual(beats, 3)
        heartbeat.check()

    def test_heartbeat_failure_is_visible_to_the_foreground_check(self):
        calls = []
        heartbeat = ClientHeartbeat(self.make_client(calls, fail_after=0), "b" * 64, 0.01, timeout=1.0)
        heartbeat.start()
        deadline = time.monotonic() + 2
        while heartbeat.failure is None and time.monotonic() < deadline:
            time.sleep(0.005)
        heartbeat.stop()
        with self.assertRaisesRegex(ChapterClientError, "heartbeat failed"):
            heartbeat.check()
        self.assertEqual(len(calls), 2)

    def test_stopped_heartbeat_never_renews_a_revoked_session(self):
        sessions = ClientSessions(30)
        calls = []

        class RevokingTransport:
            settings = {"timeout_seconds": 5}

            def request(self, method, path, payload=None, timeout=None):
                calls.append(path)
                if method == "POST":
                    sessions.heartbeat(path.split("/")[2])
                return {"renewed": True}

        heartbeat = ClientHeartbeat(RevokingTransport(), "c" * 64, 0.01, timeout=1.0)
        sessions.open(heartbeat.session_id)
        heartbeat.start()
        deadline = time.monotonic() + 2
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        heartbeat.stop()
        self.assertTrue(sessions.revoke(heartbeat.session_id))
        with self.assertRaises(KeyError):
            sessions.heartbeat(heartbeat.session_id)
        self.assertFalse(sessions.live(heartbeat.session_id))


class ScopedStopConditionTests(unittest.TestCase):
    def test_scope_raises_at_boundaries_and_threads_only_through_copied_context(self):
        with self.assertRaises(RenderPaused):
            with scoped_stop_condition(lambda: True, lambda: RenderPaused("owner gone")):
                stop_if_requested()
        # Without a copied context, sibling threads are unaffected.
        results = []
        def worker():
            try:
                stop_if_requested()
                results.append("ran")
            except RenderPaused:
                results.append("paused")
        with scoped_stop_condition(lambda: True, lambda: RenderPaused("owner gone")):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(2)
        self.assertEqual(results, ["ran"])
        stop_if_requested()

    def test_nested_scopes_chain_and_restore(self):
        with scoped_stop_condition(lambda: True, lambda: RenderPaused("outer")):
            with scoped_stop_condition(lambda: False, lambda: RenderPaused("inner")):
                # The inner scope does not fire, so the inherited outer stops.
                with self.assertRaisesRegex(RenderPaused, "outer"):
                    stop_if_requested()
            with self.assertRaisesRegex(RenderPaused, "outer"):
                stop_if_requested()
        stop_if_requested()


if __name__ == "__main__":
    unittest.main()
