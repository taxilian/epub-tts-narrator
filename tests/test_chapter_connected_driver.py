"""Real HTTP driver for connected-client execution over an actual socket.

A uvicorn server, the real urllib ChapterClient, and a real ChapterJobs queue
with a model-free executor prove the run-scoped lifecycle over the wire: a
connected run generates, downloads, and acknowledges verified delivery, the
acknowledged run's rows disappear while its files remain, a disconnect drains
the in-flight take and removes the run without orphan rows, a restart leaves
no history and never executes past runs, and observers never execute.
"""

from pathlib import Path
import tempfile
import threading
import time
import unittest

from epub_tts_narrator.renderer.chapter_client import ChapterClient, ChapterClientError
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_server import create_app
from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.interrupts import stop_if_requested


class DriverExecutor:
    identity = {"engine": "driver-fixture", "version": 1}

    def __init__(self):
        self.generated = []
        self.entered = threading.Event()
        # Generation stays in-flight (checking its stop condition) until a
        # test releases it, so the disconnect lands mid-generation.
        self.release = threading.Event()

    def synthesize(self, spec, voices, work, notify, retry_budget=None):
        self.generated.append(spec["chapter"]["number"])
        self.entered.set()
        while not self.release.wait(0.01):
            stop_if_requested()
        return {}

    def assemble(self, spec, prepared, work, notify):
        folder = Path(work) / "driver-artifact"
        folder.mkdir(exist_ok=True)
        path = folder / "chapter.m4a"
        path.write_bytes(b"driver compressed chapter audio")
        return {"chapter.m4a": path}


def specification(service, number=1):
    return {"version": 1, "book": {"id": "driver", "title": "Driver Book"},
            "source_fingerprint": "a" * 64, "renderer_fingerprint": service.fingerprint,
            "chapter": {"id": f"chapter-{number:04d}", "number": number, "title": "Chapter", "chunks": []},
            "total_chapters": 2, "voice_bindings": {}, "render": {}}


class ConnectedDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.servers, self.threads = [], []

    def start_server(self, name, *, maintenance=False):
        settings = {"api_key": "", "event_poll_seconds": 0.01, "idle_timeout_seconds": 0}
        executor = DriverExecutor()
        service = ChapterJobs(self.root / name, executor, settings, maintenance=maintenance)
        app = create_app(service, settings)
        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(server.started)
        port = server.servers[0].sockets[0].getsockname()[1]

        def shutdown():
            server.should_exit = True
            thread.join(timeout=10)

        self.addCleanup(shutdown)
        self.servers.append(server)
        self.threads.append(thread)
        client = ChapterClient({"url": f"http://127.0.0.1:{port}", "api_key": "",
                                "timeout_seconds": 5, "websocket": False, "poll_seconds": 0.02})
        return service, executor, client

    def test_maintenance_activation_reuses_the_same_live_model_and_server(self):
        service, engine, client = self.start_server("warm", maintenance=True)
        self.addCleanup(engine.release.set)
        self.assertTrue(client.request("GET", "v1/capabilities")["maintenance"])
        self.assertFalse(client.request("POST", "v1/server/activate", {})["maintenance"])
        self.assertIs(service.executor, engine)
        self.assertEqual(engine.generated, [])
        session = digest({"warm": "client"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        job = client.request("POST", "v1/jobs", {"spec": spec, "idempotency_key": digest(spec), "session": session})
        self.wait_for(client, job["id"], "generating")
        engine.release.set()
        self.wait_for(client, job["id"], "complete")
        client.request("POST", f"v1/jobs/{job['id']}/ack", {"session": session})
        client.request("DELETE", f"v1/sessions/{session}")
        self.assertEqual(client.request("GET", "v1/jobs")["jobs"], [])
        next_session = digest({"warm": "next-book"})
        client.request("POST", "v1/sessions", {"id": next_session})
        next_spec = specification(service, number=2)
        next_job = client.request("POST", "v1/jobs", {"spec": next_spec, "idempotency_key": digest(next_spec),
                                                      "session": next_session})
        self.wait_for(client, next_job["id"], "complete")
        client.request("POST", f"v1/jobs/{next_job['id']}/ack", {"session": next_session})
        client.request("DELETE", f"v1/sessions/{next_session}")
        self.assertIs(service.executor, engine)
        self.assertEqual(engine.generated, [1, 2])
        self.assertEqual(client.request("GET", "v1/jobs")["jobs"], [])

    def stop_server(self):
        server, thread = self.servers.pop(), self.threads.pop()
        # The lifespan close joins the workers; every driver worker is either
        # paused at a boundary or completed before shutdown is requested.
        server.should_exit = True
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())

    def wait_for(self, client, job_id, state):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.request("GET", f"v1/jobs/{job_id}")
            if status["state"] == state:
                return status
            time.sleep(0.02)
        self.fail(f"Job did not reach {state}: {status}")

    def wait_until_gone(self, client, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                client.request("GET", f"v1/jobs/{job_id}")
            except ChapterClientError as exc:
                if exc.status == 404:
                    return
                raise
            time.sleep(0.02)
        self.fail("The abandoned run's row was not removed")

    def test_connected_run_generates_downloads_and_ack_cleans_up(self):
        service, engine, client = self.start_server("driver-ack")
        self.addCleanup(service.close)
        capabilities = client.request("GET", "v1/capabilities")
        self.assertEqual(capabilities["client_sessions"], 1)
        self.assertEqual(capabilities["ephemeral_runs"], 1)
        session = digest({"driver": "ack"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        job = client.request("POST", "v1/jobs",
                             {"spec": spec, "idempotency_key": digest(spec), "session": session})
        self.wait_for(client, job["id"], "generating")
        engine.release.set()
        self.wait_for(client, job["id"], "complete")
        artifacts = client.request("GET", f"v1/jobs/{job['id']}")["artifacts"]
        target = self.root / "chapter.m4a"
        client.download(artifacts[0], target)
        self.assertEqual(target.read_bytes(), b"driver compressed chapter audio")
        # The verified delivery is acknowledged: the run's rows disappear
        # while its artifact files remain in the server store.
        client.request("POST", f"v1/jobs/{job['id']}/ack", {"session": session})
        self.wait_until_gone(client, job["id"])
        self.assertEqual(client.request("GET", "v1/jobs?limit=256")["jobs"], [])
        self.assertTrue(any((service.root / "render-work").rglob("chapter.m4a")))
        client.request("DELETE", f"v1/sessions/{session}")

    def test_disconnect_drains_and_removes_the_run_without_orphan_rows(self):
        service, engine, client = self.start_server("driver-drain")
        self.addCleanup(service.close)
        session = digest({"driver": "drain"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        job = client.request("POST", "v1/jobs",
                             {"spec": spec, "idempotency_key": digest(spec), "session": session})
        self.wait_for(client, job["id"], "generating")
        engine.entered.wait(3)
        client.request("DELETE", f"v1/sessions/{session}")
        # The in-flight take drains at its durable boundary; the abandoned
        # run's metadata is then removed and no orphan rows linger.
        self.wait_until_gone(client, job["id"])
        self.assertEqual(client.request("GET", "v1/jobs?limit=256")["jobs"], [])
        self.assertEqual(engine.generated, [1])

    def test_restart_leaves_no_history_and_never_executes_past_runs(self):
        service, engine, client = self.start_server("driver-restart")
        self.addCleanup(service.close)
        session = digest({"driver": "restart"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        job = client.request("POST", "v1/jobs",
                             {"spec": spec, "idempotency_key": digest(spec), "session": session})
        self.wait_for(client, job["id"], "generating")
        client.request("DELETE", f"v1/sessions/{session}")
        self.wait_until_gone(client, job["id"])
        generation_count = len(engine.generated)
        self.stop_server()
        restarted_service, restarted_engine, restarted_client = self.start_server("driver-restart")
        self.addCleanup(restarted_service.close)
        time.sleep(0.4)
        # A restart clears every session and every past-run row: nothing is
        # listed, nothing executes, and the old job stays unknown.
        self.assertEqual(restarted_engine.generated, [])
        self.assertEqual(restarted_client.request("GET", "v1/jobs?limit=256")["jobs"], [])
        with self.assertRaises(ChapterClientError):
            restarted_client.request("GET", f"v1/jobs/{job['id']}")
        with self.assertRaises(ChapterClientError):
            restarted_client.request("POST", f"v1/sessions/{session}/heartbeat")
        # A fresh connected submission executes as a new run.
        restarted_engine.release.set()
        fresh = digest({"driver": "after-restart"})
        restarted_client.request("POST", "v1/sessions", {"id": fresh})
        created = restarted_client.request(
            "POST", "v1/jobs", {"spec": spec, "idempotency_key": digest(spec), "session": fresh})
        self.wait_for(restarted_client, created["id"], "complete")
        self.assertEqual(restarted_engine.generated, [1])
        self.assertEqual(generation_count, 1)
        restarted_client.request("DELETE", f"v1/sessions/{fresh}")


if __name__ == "__main__":
    unittest.main()
