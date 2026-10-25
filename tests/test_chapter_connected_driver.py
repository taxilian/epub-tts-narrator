"""Real HTTP driver for connected-client execution over an actual socket.

A uvicorn server, the real urllib ChapterClient, and a real ChapterJobs queue
with a model-free executor prove the boundary the automated queue lost:
disconnect pauses at a durable boundary, observers never execute, restart
never resumes unowned work, and an explicit reconnect finishes the job.
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

    def start_server(self, name):
        settings = {"api_key": "", "event_poll_seconds": 0.01, "idle_timeout_seconds": 0}
        executor = DriverExecutor()
        service = ChapterJobs(self.root / name, executor, settings)
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

    def test_disconnect_pauses_reconnect_resumes_and_observers_cannot_execute(self):
        service, engine, client = self.start_server("driver-a")
        self.addCleanup(service.close)
        capabilities = client.request("GET", "v1/capabilities")
        self.assertEqual(capabilities["client_sessions"], 1)
        session = digest({"driver": "first"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        payload = {"spec": spec, "idempotency_key": digest(spec), "session": session}
        job = client.request("POST", "v1/jobs", payload)
        status = self.wait_for(client, job["id"], "generating")
        self.assertTrue(status["owner_connected"])
        engine.entered.wait(3)
        # Explicit disconnect: the lease is revoked, the server must stop
        # dispatching further GPU work and pause losslessly.
        client.request("DELETE", f"v1/sessions/{session}")
        paused = self.wait_for(client, job["id"], "queued")
        self.assertTrue(paused["paused"])
        self.assertFalse(paused["owner_connected"])
        self.assertIsNone(paused["error"])
        generation_count = len(engine.generated)
        time.sleep(0.2)
        self.assertEqual(len(engine.generated), generation_count)
        # A heartbeat can never recreate a revoked session.
        with self.assertRaises(ChapterClientError):
            client.request("POST", f"v1/sessions/{session}/heartbeat")
        # The read-only queue overview identifies the paused book without
        # leaking anything, and observing it executes nothing.
        listed = client.request("GET", "v1/jobs?limit=256")["jobs"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["chapter_number"], 1)
        self.assertEqual(listed[0]["total_chapters"], 2)
        self.assertTrue(listed[0]["paused"])
        self.assertEqual(len(engine.generated), generation_count)
        # Explicit reconnect resumes the accepted work on the same job.
        engine.release.set()
        resumed_session = digest({"driver": "second"})
        client.request("POST", "v1/sessions", {"id": resumed_session})
        resumed = client.request("POST", "v1/jobs", dict(payload, session=resumed_session))
        self.assertEqual(resumed["id"], job["id"])
        self.wait_for(client, job["id"], "generating")
        self.wait_for(client, job["id"], "complete")
        self.assertEqual(engine.generated, [1, 1])
        artifacts = client.request("GET", f"v1/jobs/{job['id']}")["artifacts"]
        target = self.root / "chapter.m4a"
        client.download(artifacts[0], target)
        self.assertEqual(target.read_bytes(), b"driver compressed chapter audio")
        client.request("DELETE", f"v1/sessions/{resumed_session}")

    def test_restart_never_executes_unowned_jobs_until_an_explicit_reattach(self):
        service, engine, client = self.start_server("driver-b")
        self.addCleanup(service.close)
        session = digest({"driver": "restart"})
        client.request("POST", "v1/sessions", {"id": session})
        spec = specification(service)
        job = client.request("POST", "v1/jobs",
                             {"spec": spec, "idempotency_key": digest(spec), "session": session})
        self.wait_for(client, job["id"], "generating")
        client.request("DELETE", f"v1/sessions/{session}")
        self.wait_for(client, job["id"], "queued")
        generation_count = len(engine.generated)
        self.stop_server()
        restarted_service, restarted_engine, restarted_client = self.start_server("driver-b")
        self.addCleanup(restarted_service.close)
        time.sleep(0.4)
        self.assertEqual(restarted_engine.generated, [])
        status = restarted_client.request("GET", f"v1/jobs/{job['id']}")
        self.assertEqual(status["state"], "queued")
        self.assertTrue(status["paused"])
        with self.assertRaises(ChapterClientError):
            restarted_client.request("POST", f"v1/sessions/{session}/heartbeat")
        restarted_engine.release.set()
        fresh = digest({"driver": "after-restart"})
        restarted_client.request("POST", "v1/sessions", {"id": fresh})
        reattached = restarted_client.request(
            "POST", "v1/jobs", {"spec": spec, "idempotency_key": digest(spec), "session": fresh})
        self.assertEqual(reattached["id"], job["id"])
        self.wait_for(restarted_client, job["id"], "complete")
        self.assertEqual(restarted_engine.generated, [1])
        self.assertEqual(generation_count, 1)
        restarted_client.request("DELETE", f"v1/sessions/{fresh}")


if __name__ == "__main__":
    unittest.main()
