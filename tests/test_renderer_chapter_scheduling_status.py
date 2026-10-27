"""Chapter job queue scheduling status over the real HTTP surface.

A uvicorn server, the real urllib ChapterClient, and a model-free executor
prove the read-only scheduling observer: queue counts per state, the
additive ``scheduling`` block on ``GET /v1/jobs``, and an observer that
never renews a session, starts work, or grows any registry.
"""

from pathlib import Path
import tempfile
import threading
import time
import unittest

from epub_tts_narrator.renderer.chapter_client import ChapterClient
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_server import create_app
from epub_tts_narrator.renderer.common import digest


class QuietExecutor:
    """A plugin executor without any scheduling contract: conservative."""
    identity = {"engine": "scheduling-fixture", "version": 1}

    def __init__(self):
        self.generated = []
        self.release = threading.Event()

    def synthesize(self, spec, voices, work, notify, retry_budget=None):
        self.generated.append(spec["chapter"]["number"])
        self.release.wait(5)
        return {}

    def assemble(self, spec, prepared, work, notify):
        folder = Path(work) / "scheduling-artifact"
        folder.mkdir(exist_ok=True)
        path = folder / "chapter.m4a"
        path.write_bytes(b"scheduling fixture audio")
        return {"chapter.m4a": path}


def specification(service, number=1):
    return {"version": 1, "book": {"id": "scheduling", "title": "Scheduling Book"},
            "source_fingerprint": "b" * 64, "renderer_fingerprint": service.fingerprint,
            "chapter": {"id": f"chapter-{number:04d}", "number": number, "title": "Chapter",
                        "chunks": []},
            "total_chapters": 2, "voice_bindings": {}, "render": {}}


class ChapterJobsSchedulingStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executor = QuietExecutor()
        self.service = ChapterJobs(self.root / "jobs", self.executor,
                                   {"api_key": "", "event_poll_seconds": 0.01,
                                    "idle_timeout_seconds": 0})
        self.addCleanup(self.service.close)

    def test_queue_counts_are_numeric_and_state_keyed(self):
        session = digest({"scheduling": "counts"})
        self.service.sessions.open(session)
        spec = specification(self.service, 1)
        self.service.submit(spec, digest(spec), session=session)
        status = self.service.scheduling_status()
        self.assertEqual(status["jobs"],
                         {"queued": 1, "generating": 0, "assembling": 0,
                          "complete": 0, "failed": 0, "cancelled": 0})
        self.assertEqual(status["active_workers"], 0)
        self.assertGreaterEqual(status["coordinators"], 1)
        self.assertIsInstance(status["take_identities"], int)
        self.assertIsNone(status["engine"])  # conservative for unknown executors

    def test_capabilities_advertise_the_scheduling_block(self):
        self.assertEqual(self.service.capabilities()["scheduling_status"], 1)

    def test_take_identities_counts_executor_dedup_registrations(self):
        from threading import Lock

        class LockCountingExecutor(QuietExecutor):
            def __init__(self):
                super().__init__()
                self._take_locks = {"a": Lock(), "b": Lock()}
                self._take_locks_guard = Lock()

            def take_identity_count(self):
                with self._take_locks_guard:
                    return len(self._take_locks)

        counting = LockCountingExecutor()
        service = ChapterJobs(self.root / "locks", counting,
                              {"api_key": "", "event_poll_seconds": 0.01,
                               "idle_timeout_seconds": 0})
        self.addCleanup(service.close)
        self.assertEqual(service.scheduling_status()["take_identities"], 2)

    def test_a_broken_executor_probe_reports_zero_instead_of_failing(self):
        class ExplodingExecutor(QuietExecutor):
            def take_identity_count(self):
                raise RuntimeError("boom")

        broken = ExplodingExecutor()
        service = ChapterJobs(self.root / "broken", broken,
                              {"api_key": "", "event_poll_seconds": 0.01,
                               "idle_timeout_seconds": 0})
        self.addCleanup(service.close)
        self.assertEqual(service.scheduling_status()["take_identities"], 0)


class SchedulingHttpObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        settings = {"api_key": "", "event_poll_seconds": 0.01, "idle_timeout_seconds": 0}
        self.executor = QuietExecutor()
        self.service = ChapterJobs(self.root / "http", self.executor, settings)
        self.app = create_app(self.service, settings)
        import uvicorn
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        self.client = ChapterClient({"url": f"http://127.0.0.1:{self.port}", "api_key": "",
                                     "timeout_seconds": 5, "websocket": False,
                                     "poll_seconds": 0.02})

    def tearDown(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.executor.release.set()

    def test_jobs_response_carries_the_scheduling_block(self):
        overview = self.client.request("GET", "v1/jobs")
        self.assertEqual(overview["jobs"], [])
        scheduling = overview["scheduling"]
        self.assertEqual(scheduling["jobs"]["queued"], 0)
        self.assertTrue(all(isinstance(value, int) for value in scheduling["jobs"].values()))

    def test_observer_polling_renews_starts_and_grows_nothing(self):
        session = digest({"scheduling": "observer"})
        self.client.request("POST", "v1/sessions", {"id": session})
        deadline_before = self.service.sessions._deadlines[session]
        spec = specification(self.service)
        self.client.request("POST", "v1/jobs",
                            {"spec": spec, "idempotency_key": digest(spec), "session": session})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self.executor.generated:
            time.sleep(0.02)
        self.assertEqual(self.executor.generated, [1])
        budgets_before = dict(self.service._submission_budgets)
        for _ in range(5):
            overview = self.client.request("GET", "v1/jobs")
            self.assertEqual(overview["scheduling"]["jobs"]["generating"], 1)
        # No renewal: the session's exact expiry is untouched by observation.
        self.assertEqual(self.service.sessions._deadlines[session], deadline_before)
        # No new execution, no budget or epoch growth from the observer.
        self.assertEqual(self.executor.generated, [1])
        self.assertEqual(self.service._submission_budgets, budgets_before)

    def test_maintenance_observation_executes_nothing_and_creates_no_session(self):
        settings = {"api_key": "", "event_poll_seconds": 0.01, "idle_timeout_seconds": 0}
        executor = QuietExecutor()
        service = ChapterJobs(self.root / "maintenance", executor, settings, maintenance=True)
        app = create_app(service, settings)
        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.addCleanup(lambda: setattr(server, "should_exit", True))
        self.addCleanup(thread.join, timeout=10)
        self.addCleanup(executor.release.set)
        observer = ChapterClient({"url": f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}",
                                  "api_key": "", "timeout_seconds": 5, "websocket": False,
                                  "poll_seconds": 0.02})
        for _ in range(5):
            overview = observer.request("GET", "v1/jobs")
            self.assertEqual(overview["scheduling"]["jobs"],
                             {"queued": 0, "generating": 0, "assembling": 0,
                              "complete": 0, "failed": 0, "cancelled": 0})
        self.assertEqual(executor.generated, [])
        self.assertEqual(service.sessions._deadlines, {})


class RunpodProgressSchedulingTests(unittest.TestCase):
    def render(self, response, *, json_mode=False):
        import argparse
        from contextlib import redirect_stdout
        from io import StringIO
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from epub_tts_narrator.renderer.runpod import add_parser, deployment_lock
        from epub_tts_narrator.renderer.runpod_progress import run_progress

        class Client:
            def __init__(self, settings):
                pass

            def request(self, method, path):
                self.last = path
                return response

        output = StringIO()
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "state.json").write_text(
                json.dumps({"pod_id": "pod", "api_key": "private-token"}))
            settings = {"deployment": {"state_dir": folder, "service_port": 8082}}
            parser = argparse.ArgumentParser()
            add_parser(parser.add_subparsers(dest="command"))
            args = parser.parse_args(
                ["runpod", "progress"] + (["--json"] if json_mode else []))
            with deployment_lock(Path(folder)), redirect_stdout(output), \
                    patch("epub_tts_narrator.renderer.runpod_config.load_settings",
                          return_value=settings), \
                    patch("epub_tts_narrator.renderer.runpod_progress.ChapterClient", Client):
                code = run_progress(settings, args)
        self.assertEqual(code, 0)
        return json.loads(output.getvalue()) if json_mode else output.getvalue()

    def scheduling_block(self):
        return {"mode": "pipeline",
                "jobs": {"queued": 2, "generating": 1, "assembling": 0,
                         "complete": 3, "failed": 0, "cancelled": 0},
                "capacity": {"tts_slots": 8, "max_inflight": 16,
                             "inflight_basis": "pipeline_global_transactions",
                             "chapter_concurrency": 4},
                "logical": {"pending": 1, "inflight": 3, "owners": 2,
                            "global_limit": 16, "chapter_share": 4, "closed": False},
                "backlog": {"pending_count": 4, "peak_pending_count": 9,
                            "materialized_bytes": 1048576,
                            "peak_materialized_bytes": 2097152,
                            "max_materialized_bytes": 268435456,
                            "new_generation_blocked": False, "waiters": 1},
                "stages": {"synthesis": {"queued_now": 1, "active_now": 2}}}

    def service_block(self):
        """The chapter-service wrapper shape: queue counts plus nested engine."""
        return {"jobs": self.scheduling_block()["jobs"],
                "active_workers": 1, "coordinators": 2, "take_identities": 5,
                "engine": self.scheduling_block()}

    def test_json_mode_projects_the_scheduling_block(self):
        projected = self.render({"jobs": [], "scheduling": self.scheduling_block()},
                                json_mode=True)
        self.assertEqual(projected["scheduling"]["mode"], "pipeline")
        self.assertEqual(projected["scheduling"]["backlog"]["peak_pending_count"], 9)
        self.assertNotIn("spec", projected)

    def test_json_mode_projects_the_nested_service_wrapper(self):
        projected = self.render({"jobs": [], "scheduling": self.service_block()},
                                json_mode=True)
        self.assertEqual(projected["scheduling"]["mode"], "pipeline")
        self.assertEqual(projected["scheduling"]["jobs"]["generating"], 1)
        self.assertEqual(projected["scheduling"]["capacity"]["tts_slots"], 8)

    def test_text_mode_prints_one_scheduling_line(self):
        text = self.render({"jobs": [], "scheduling": self.service_block()})
        self.assertIn("scheduling:", text)
        self.assertIn("pipeline", text)
        self.assertIn("peak", text)
        self.assertIn("jobs queued 2 generating 1", text)

    def test_an_older_server_without_scheduling_stays_quiet(self):
        text = self.render({"jobs": []})
        self.assertNotIn("scheduling:", text)
        projected = self.render({"jobs": []}, json_mode=True)
        self.assertNotIn("scheduling", projected)


if __name__ == "__main__":
    unittest.main()
