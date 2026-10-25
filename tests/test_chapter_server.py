"""Chapter API contract tests; no speech models, paid services, or GPU required."""

import asyncio
import base64
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.chapter_server import create_app, serve
from epub_tts_narrator.renderer.chapter_sessions import ClientSessions
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from shared_fixtures import service_session


HAS_SERVER_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))
JOB = "a" * 64
VOICE = "b" * 64
KEY = "local-test-credential"
AUTH = {"Authorization": f"Bearer {KEY}"}
SESSION = hashlib.sha256(b"connected-client-fixture").hexdigest()


class FakeJobs:
    def __init__(self, root):
        self.root = root
        self.started = False
        self.closed = False
        self.voices = {}
        self.submit_calls = []
        self.retry_calls = []
        self.ack_calls = []
        self.events_requested = []
        self.idle_checks = []
        self.cancel_count = 0
        self.shutdown_ready = False
        self.sessions = ClientSessions()
        self.job = {
            "id": JOB, "state": "queued", "artifacts": [], "last_event": 1,
        }
        self.log = [{"sequence": 1, "type": "queued", "message": "Chapter queued"}]

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def capabilities(self):
        return {"version": 1, "chapter_formats": ["m4a"], "voice_format": "flac"}

    def open_session(self, session_id):
        self.sessions.open(session_id)
        return {"opened": True, "session_ttl_seconds": self.sessions.ttl}

    def heartbeat_session(self, session_id):
        self.sessions.heartbeat(session_id)
        return {"renewed": True, "session_ttl_seconds": self.sessions.ttl}

    def revoke_session(self, session_id):
        return {"revoked": self.sessions.revoke(session_id)}

    def ensure_voices(self, voices):
        return [voice for voice in voices if voice not in self.voices]

    def register_voice(self, asset_id, manifest, audio):
        self.voices[asset_id] = {"manifest": manifest, "audio": audio}
        return {"id": asset_id}

    def submit(self, spec, idempotency_key, submission=None, session=None):
        self.submit_calls.append((spec, idempotency_key, submission, session))
        return deepcopy(self.job)

    def status(self, job_id):
        if job_id != JOB:
            raise KeyError(job_id)
        return deepcopy(self.job)

    def events(self, job_id, after):
        self.status(job_id)
        self.events_requested.append(after)
        return [event for event in self.log if event["sequence"] > after]

    def retry(self, job_id, submission=None, session=None):
        self.retry_calls.append((job_id, submission, session))
        return self.status(job_id)

    def acknowledge(self, job_id, session):
        self.ack_calls.append((job_id, session))
        return {"acknowledged": True}

    def list_jobs(self, limit=100):
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Job list limit must be an integer from 1 through 256")
        return [{"id": JOB, "state": self.job["state"], "paused": False,
                 "owner_connected": True, "chapter_number": 2, "total_chapters": 5,
                 "source_fingerprint": "a" * 64, "updated": 1.0, "last_event": 1}][:limit]

    def artifact(self, job_id, name):
        status = self.status(job_id)
        if name not in {item["name"] for item in status["artifacts"]}:
            raise FileNotFoundError(name)
        return self.root / name

    def begin_shutdown_if_idle(self, timeout):
        self.idle_checks.append(timeout)
        return self.shutdown_ready

    def cancel_shutdown(self):
        self.cancel_count += 1
        self.shutdown_ready = False


class ChapterServerSettingsTests(unittest.TestCase):
    def test_public_bind_requires_auth_before_importing_optional_dependencies(self):
        for host in ("0.0.0.0", "::", "192.0.2.1", "tts.example.test"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "requires.*api_key"):
                create_app(None, {"host": host})

    def test_chapter_server_model_and_checkpoint_requirements_are_enforced_at_load_time(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "global.toml"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verification.model must be a nonempty string"):
                load_config(path)
            path.write_text('[verification]\nmodel = "/fixture/local-asr"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engine.checkpoint is required for Drama 3"):
                load_config(path)

    @unittest.skipUnless(HAS_SERVER_DEPS and importlib.util.find_spec("uvicorn"), "requires server dependencies")
    def test_configured_ffprobe_is_passed_to_voice_registry_without_mutating_config(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"type": "fish_http"}]
        config["verification"]["model"] = "/fixture/local-asr"
        config["render"]["ffprobe"] = "/custom/ffprobe"
        executor, jobs = Mock(), Mock()
        with (patch("epub_tts_narrator.renderer.chapter_executor.ChapterExecutor", return_value=executor),
              patch("epub_tts_narrator.renderer.chapter_jobs.ChapterJobs", return_value=jobs) as create_jobs,
              patch("uvicorn.run") as run):
            serve(config)
        self.assertEqual(create_jobs.call_args.args[2]["ffprobe"], "/custom/ffprobe")
        self.assertNotIn("ffprobe", config["chapter_server"])
        self.assertEqual(run.call_args.kwargs["workers"], 1)

    @unittest.skipUnless(HAS_SERVER_DEPS and importlib.util.find_spec("uvicorn"), "requires server dependencies")
    def test_maintenance_is_passed_to_job_service_without_changing_config(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"type": "fish_http"}]
        config["verification"]["model"] = "/fixture/local-asr"
        before = deepcopy(config)
        with (patch("epub_tts_narrator.renderer.chapter_executor.ChapterExecutor"),
              patch("epub_tts_narrator.renderer.chapter_jobs.ChapterJobs") as jobs,
              patch("uvicorn.run")):
            serve(config, maintenance=True)
        self.assertIs(jobs.call_args.kwargs["maintenance"], True)
        self.assertEqual(config, before)

    @unittest.skipUnless(HAS_SERVER_DEPS and importlib.util.find_spec("uvicorn"), "requires server dependencies")
    def test_runpod_idle_shutdown_stops_compute_through_the_managed_api_client(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"type": "fish_http"}]
        config["verification"]["model"] = "/fixture/local-asr"
        config["chapter_server"].update(idle_action="runpod_stop", persistent_data=True,
                                        runpod_pod_id="pod_fixture", runpod_api_key="fixture-key")
        config["chapter_storage"]["bucket"] = "fixture-bucket"
        api = Mock()
        with (patch("epub_tts_narrator.renderer.chapter_storage.S3ArtifactPublisher", return_value=Mock()),
              patch("epub_tts_narrator.renderer.chapter_executor.ChapterExecutor"),
              patch("epub_tts_narrator.renderer.chapter_jobs.ChapterJobs"),
              patch("epub_tts_narrator.renderer.runpod_api.RunpodAPI", return_value=api) as client,
              patch("epub_tts_narrator.renderer.chapter_server.create_app", return_value=Mock()) as create_app,
              patch("uvicorn.run")):
            serve(config)
        client.assert_called_once_with("fixture-key")
        api.stop_pod.assert_not_called()
        create_app.call_args.kwargs["shutdown_callback"]()
        api.stop_pod.assert_called_once_with("pod_fixture")


@unittest.skipUnless(HAS_SERVER_DEPS, "requires FastAPI and httpx")
class ChapterServerTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.jobs = FakeJobs(Path(self.temp.name))
        self.settings = {"api_key": KEY, "event_poll_seconds": 0.01}
        self.client_type = TestClient

    def client(self, **settings):
        return self.client_type(create_app(self.jobs, self.settings | settings))

    def complete(self, *, external=False):
        audio = b"compressed AAC fixture"
        (self.jobs.root / "chapter.m4a").write_bytes(audio)
        artifact = {"name": "chapter.m4a", "sha256": hashlib.sha256(audio).hexdigest(), "size": len(audio)}
        if external:
            artifact["url"] = "https://downloads.example.test/chapter.m4a?signature=fixture"
        self.jobs.job.update(state="complete", artifacts=[artifact], last_event=2)
        self.jobs.log.append({"sequence": 2, "type": "complete", "message": "Chapter complete"})

    def test_http_requires_bearer_even_for_capabilities_and_downloads(self):
        with self.client() as client:
            for path in ("/v1/capabilities", f"/v1/jobs/{JOB}", f"/v1/jobs/{JOB}/artifacts/chapter.m4a"):
                for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic " + KEY}):
                    with self.subTest(path=path, headers=headers):
                        self.assertEqual(client.get(path, headers=headers).status_code, 401)
            self.assertEqual(client.get("/v1/capabilities", headers=AUTH).status_code, 200)
            self.assertTrue(self.jobs.started)
            self.assertFalse(self.jobs.closed)
        self.assertTrue(self.jobs.closed)

    def test_missing_voice_manifest_and_flac_upload(self):
        audio = b"fLaCcompressed-reference-fixture"
        manifest = {"version": 1, "reference_text": "Hello there.", "audio_sha256": hashlib.sha256(audio).hexdigest()}
        with self.client() as client:
            response = client.post("/v1/voices/ensure", headers=AUTH, json={"voices": [VOICE, VOICE]})
            self.assertEqual(response.json(), {"missing": [VOICE]})
            response = client.put(f"/v1/voices/{VOICE}", headers=AUTH, json={
                "manifest": manifest, "audio_base64": base64.b64encode(audio).decode(),
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.jobs.voices[VOICE]["audio"], audio)
            self.assertEqual(client.post("/v1/voices/ensure", headers=AUTH, json={"voices": [VOICE]}).json(), {"missing": []})

    def test_uncompressed_voice_upload_is_rejected(self):
        with self.client() as client:
            response = client.put(f"/v1/voices/{VOICE}", headers=AUTH, json={
                "manifest": {}, "audio_base64": base64.b64encode(b"RIFF-WAV").decode(),
            })
            self.assertEqual(response.status_code, 400)
            self.assertFalse(self.jobs.voices)

    def test_request_limit_applies_to_content_length_and_chunked_body(self):
        with self.client(max_request_bytes=40) as client:
            response = client.post("/v1/voices/ensure", headers=AUTH, json={"voices": [VOICE]})
            self.assertEqual(response.status_code, 413)
            response = client.post("/v1/voices/ensure", headers=AUTH | {"Content-Type": "application/json"},
                                   content=iter([b'{"voices": ["', b"b" * 64, b'"]}']))
            self.assertEqual(response.status_code, 413)

    def test_invalid_json_paths_and_identifiers_are_rejected(self):
        with self.client() as client:
            for content in (b'{"voices": [], "voices": []}', b'{"voices": NaN}', b'[]', b'null'):
                with self.subTest(content=content):
                    response = client.post("/v1/voices/ensure", headers=AUTH | {"Content-Type": "application/json"}, content=content)
                    self.assertEqual(response.status_code, 400)
            self.assertEqual(client.get("/v1/jobs/not-a-hash", headers=AUTH).status_code, 400)
            self.assertEqual(client.get(f"/v1/jobs/{'c' * 64}", headers=AUTH).status_code, 404)
            self.assertEqual(client.get(f"/v1/jobs/{JOB}/artifacts/reference.wav", headers=AUTH).status_code, 400)
            self.assertEqual(client.get(f"/v1/jobs/{JOB}/artifacts/%2E%2E", headers=AUTH).status_code, 400)

    def test_submit_returns_immediately_and_keeps_idempotency_key(self):
        with self.client() as client:
            body = {"spec": {"chapter": {"number": 2}}, "idempotency_key": "c" * 64,
                    "session": "f" * 64}
            result = client.post("/v1/jobs", headers=AUTH, json=body)
            self.assertEqual(result.status_code, 202)
            self.assertEqual(result.json()["id"], JOB)
            self.assertEqual(self.jobs.submit_calls, [(body["spec"], body["idempotency_key"], None, "f" * 64)])
            retry = client.post(f"/v1/jobs/{JOB}/retry", headers=AUTH,
                                json={"session": "f" * 64})
            self.assertEqual(retry.status_code, 202)
            self.assertEqual(self.jobs.retry_calls, [(JOB, None, "f" * 64)])

    def test_sessions_open_renew_only_and_revoke(self):
        with self.client() as client:
            opened = client.post("/v1/sessions", headers=AUTH, json={"id": "1" * 64})
            self.assertEqual(opened.status_code, 201, opened.text)
            self.assertTrue(opened.json()["opened"])
            renewed = client.post(f"/v1/sessions/{'1' * 64}/heartbeat", headers=AUTH)
            self.assertEqual(renewed.status_code, 200, renewed.text)
            # A heartbeat never recreates a revoked session.
            self.assertEqual(client.delete(f"/v1/sessions/{'1' * 64}", headers=AUTH).json(), {"revoked": True})
            self.assertEqual(client.post(f"/v1/sessions/{'1' * 64}/heartbeat", headers=AUTH).status_code, 404)
            self.assertEqual(client.delete(f"/v1/sessions/{'1' * 64}", headers=AUTH).json(), {"revoked": False})
            self.assertEqual(client.post(f"/v1/sessions/{'1' * 64}/heartbeat", headers=AUTH).status_code, 404)
            for payload in ({}, {"id": "nothex"}, {"id": "2" * 64, "extra": True}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post("/v1/sessions", headers=AUTH, json=payload).status_code, 400)

    def test_job_list_is_read_only_and_validates_limit(self):
        with self.client() as client:
            listed = client.get("/v1/jobs", headers=AUTH)
            self.assertEqual(listed.status_code, 200, listed.text)
            self.assertEqual(listed.json()["jobs"][0]["id"], JOB)
            self.assertNotIn("spec", listed.json()["jobs"][0])
            self.assertEqual(client.get("/v1/jobs?limit=0", headers=AUTH).status_code, 400)
            self.assertEqual(client.get("/v1/jobs?limit=257", headers=AUTH).status_code, 400)
            self.assertEqual(client.get("/v1/jobs?limit=nope", headers=AUTH).status_code, 400)

    def test_submission_identity_passes_through_and_malformed_values_are_rejected(self):
        with self.client() as client:
            body = {"spec": {"chapter": {"number": 2}}, "idempotency_key": "c" * 64,
                    "submission": "d" * 64, "session": "f" * 64}
            self.assertEqual(client.post("/v1/jobs", headers=AUTH, json=body).status_code, 202)
            self.assertEqual(client.post(f"/v1/jobs/{JOB}/retry", headers=AUTH,
                                         json={"submission": "e" * 64, "session": "f" * 64}).status_code, 202)
            self.assertEqual(self.jobs.submit_calls[-1], (body["spec"], body["idempotency_key"], "d" * 64, "f" * 64))
            self.assertEqual(self.jobs.retry_calls[-1], (JOB, "e" * 64, "f" * 64))
            for payload in ({"submission": "not-a-hash"}, {"session": "not-a-hash"}, {"unknown": True}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post("/v1/jobs", headers=AUTH,
                                                 json={**body, **payload}).status_code, 400)
                    self.assertEqual(client.post(f"/v1/jobs/{JOB}/retry", headers=AUTH,
                                                 json=payload).status_code, 400)

    def test_acknowledgement_requires_a_session_and_passes_it_to_the_service(self):
        with self.client() as client:
            self.assertEqual(client.post(f"/v1/jobs/{JOB}/ack", headers=AUTH,
                                         json={"session": "f" * 64}).status_code, 200)
            self.assertEqual(self.jobs.ack_calls, [(JOB, "f" * 64)])
            for payload in ({}, {"session": "not-a-hash"}, {"session": "f" * 64, "extra": True}, {"no": "session"}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.post(f"/v1/jobs/{JOB}/ack", headers=AUTH,
                                                 json=payload).status_code, 400)
            self.assertEqual(len(self.jobs.ack_calls), 1)

    def test_poll_replays_only_later_events(self):
        self.complete()
        with self.client() as client:
            result = client.get(f"/v1/jobs/{JOB}/events?after=1", headers=AUTH)
            self.assertEqual(result.status_code, 200)
            self.assertEqual([event["sequence"] for event in result.json()["events"]], [2])
            self.assertEqual(client.get(f"/v1/jobs/{JOB}/events?after=-1", headers=AUTH).status_code, 400)

    def test_websocket_auth_replay_and_terminal_close(self):
        from starlette.websockets import WebSocketDisconnect

        self.complete()
        with self.client() as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect(f"/v1/jobs/{JOB}/events?token={KEY}"):
                    pass
            with client.websocket_connect(f"/v1/jobs/{JOB}/events?after=1", headers=AUTH) as socket:
                result = socket.receive_json()
                self.assertEqual(result["job"]["state"], "complete")
                self.assertEqual([event["sequence"] for event in result["events"]], [2])
                with self.assertRaises(WebSocketDisconnect) as raised:
                    socket.receive_json()
                self.assertEqual(raised.exception.code, 1000)

    def test_websocket_disconnect_does_not_cancel_job_or_service(self):
        with self.client() as client:
            with client.websocket_connect(f"/v1/jobs/{JOB}/events", headers=AUTH) as socket:
                self.assertEqual(socket.receive_json()["job"]["state"], "queued")
            self.assertFalse(self.jobs.closed)
            self.assertEqual(client.get(f"/v1/jobs/{JOB}", headers=AUTH).json()["state"], "queued")

    def test_terminal_websocket_drains_every_page_before_close(self):
        from starlette.websockets import WebSocketDisconnect

        self.jobs.log = [{"sequence": number, "message": str(number)} for number in range(1, 8)]
        self.jobs.job.update(state="complete", last_event=7)
        original = self.jobs.events
        self.jobs.events = lambda job_id, after: original(job_id, after)[:2]
        with self.client() as client:
            with client.websocket_connect(f"/v1/jobs/{JOB}/events?after=1", headers=AUTH) as socket:
                received = []
                try:
                    while True:
                        received.extend(event["sequence"] for event in socket.receive_json()["events"])
                except WebSocketDisconnect as exc:
                    self.assertEqual(exc.code, 1000)
                self.assertEqual(received, list(range(2, 8)))

    def test_untrusted_browser_cannot_read_unauthenticated_loopback_websocket(self):
        from starlette.websockets import WebSocketDisconnect

        with self.client(api_key="") as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect(f"/v1/jobs/{JOB}/events", headers={"Origin": "https://unrelated.example"}):
                    pass
            with client.websocket_connect(f"/v1/jobs/{JOB}/events") as socket:
                self.assertEqual(socket.receive_json()["job"]["state"], "queued")

    def test_local_artifacts_use_download_endpoint_and_publish_urls_survive(self):
        self.complete()
        with self.client() as client:
            status = client.get(f"/v1/jobs/{JOB}", headers=AUTH).json()
            url = status["artifacts"][0]["url"]
            self.assertEqual(url, f"/v1/jobs/{JOB}/artifacts/chapter.m4a")
            downloaded = client.get(url, headers=AUTH)
            self.assertEqual(downloaded.content, b"compressed AAC fixture")
            self.assertEqual(downloaded.headers["content-type"], "audio/mp4")
            self.assertNotIn("url", self.jobs.job["artifacts"][0])
            self.complete(external=True)
            status = client.get(f"/v1/jobs/{JOB}", headers=AUTH).json()
            self.assertTrue(status["artifacts"][0]["url"].startswith("https://downloads.example.test/"))

    def test_idle_shutdown_uses_core_quiescence_and_recovers_provider_failure(self):
        calls = []

        def shutdown():
            calls.append("stop")
            raise RuntimeError("private-provider-key-must-not-be-logged")

        app = create_app(self.jobs, self.settings | {"idle_timeout_seconds": 0.01}, shutdown_callback=shutdown)
        with self.assertLogs("epub_tts_narrator.renderer.chapter_server", level="ERROR") as logs:
            with self.client_type(app) as client:
                client.get(f"/v1/jobs/{JOB}", headers=AUTH)
                time.sleep(0.03)
                self.assertFalse(calls)
                self.jobs.shutdown_ready = True
                deadline = time.monotonic() + 2
                while self.jobs.cancel_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(calls, ["stop"])
                self.assertEqual(self.jobs.cancel_count, 1)
        self.assertNotIn("private-provider-key", " ".join(logs.output))

    def test_no_idle_callback_means_no_shutdown_monitor(self):
        self.jobs.shutdown_ready = True
        with self.client(idle_timeout_seconds=0.01):
            time.sleep(0.03)
        self.assertEqual(self.jobs.idle_checks, [])

    def test_idle_shutdown_waits_for_incoming_upload_body(self):
        import httpx

        calls = []
        app = create_app(self.jobs, self.settings | {"idle_timeout_seconds": 0.01},
                         shutdown_callback=lambda: calls.append("stop"))

        async def exercise():
            started, release = asyncio.Event(), asyncio.Event()

            async def upload():
                started.set()
                yield b'{"voices": ['
                await release.wait()
                yield b'"' + VOICE.encode() + b'"]}'

            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
                    task = asyncio.create_task(client.post("/v1/voices/ensure", content=upload(),
                        headers=AUTH | {"Content-Type": "application/json"}))
                    try:
                        await asyncio.wait_for(started.wait(), timeout=2)
                        self.jobs.shutdown_ready = True
                        await asyncio.sleep(0.04)
                        self.assertEqual(calls, [])
                    finally:
                        self.jobs.shutdown_ready = False
                        release.set()
                    response = await asyncio.wait_for(task, timeout=2)
                    self.assertEqual(response.status_code, 200, response.text)

        asyncio.run(exercise())

    def test_partial_startup_failure_still_closes_service_resources(self):
        def fail_start():
            raise RuntimeError("Unable to start fixture")

        self.jobs.start = fail_start
        with self.assertRaisesRegex(RuntimeError, "Unable to start fixture"):
            with self.client():
                pass
        self.assertTrue(self.jobs.closed)

    def test_unexpected_storage_error_is_retryable_without_leaking_credentials(self):
        def fail_status(job_id):
            raise OSError("remote storage failed: secret-provider-credential")

        self.jobs.status = fail_status
        with self.assertLogs("epub_tts_narrator.renderer.chapter_server", level="ERROR") as logs:
            with self.client() as client:
                response = client.get(f"/v1/jobs/{JOB}", headers=AUTH)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret-provider", response.text + " ".join(logs.output))


class FixtureExecutor:
    identity = {"engine": "model-free-http-fixture", "version": 1}

    def __init__(self):
        self.synthesis_calls = 0
        self.assembly_calls = 0
        self.fail_next = False
        self.closed = False
        self.release = threading.Event()
        self.release.set()

    def synthesize(self, spec, voices, work, notify, retry_budget=None):
        self.synthesis_calls += 1
        notify("Generating paragraph 1 / 1")
        if not self.release.wait(5):
            raise RuntimeError("Fixture generation wait timed out")
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Fixture speech check failed")
        return {"voice_count": len(voices)}

    def assemble(self, spec, prepared, work, notify):
        self.assembly_calls += 1
        notify("Encoding chapter")
        output = Path(work) / "http-fixture"
        output.mkdir(exist_ok=True)
        result = {}
        for name, data in {"chapter.m4a": b"compressed chapter fixture", "sync.json": b'{"precision":"paragraph"}'}.items():
            path = output / name
            path.write_bytes(data)
            result[name] = path
        return result

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_SERVER_DEPS, "requires FastAPI and httpx")
class RealChapterCoreHTTPTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executor = FixtureExecutor()
        self.jobs = ChapterJobs(self.root, self.executor)
        self.client_type = TestClient
        self.app = create_app(self.jobs, {"api_key": KEY, "event_poll_seconds": 0.01})
        self.spec = {
            "version": 1, "book": {"title": "HTTP chapter fixture"},
            "source_fingerprint": "d" * 64, "renderer_fingerprint": self.jobs.fingerprint,
            "chapter": {"number": 1, "id": "chapter-0001", "title": "One", "chunks": []},
            "total_chapters": 1, "voice_bindings": {}, "render": {},
        }
        self.payload = {"spec": self.spec, "idempotency_key": "e" * 64}

    def connect(self, client):
        response = client.post("/v1/sessions", headers=AUTH, json={"id": SESSION})
        self.assertEqual(response.status_code, 201, response.text)
        return SESSION

    def wait_for(self, client, job_id, state):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/v1/jobs/{job_id}", headers=AUTH)
            self.assertEqual(response.status_code, 200, response.text)
            status = response.json()
            if status["state"] == state:
                return status
            if state != "failed" and status["state"] == "failed":
                self.fail(str(status))
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not reach {state}")

    def submit(self, client):
        payload = dict(self.payload, session=self.connect(client))
        response = client.post("/v1/jobs", json=payload, headers=AUTH)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["id"]

    def test_real_core_completion_ack_removes_rows_and_restart_never_replays(self):
        from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs

        with self.client_type(self.app) as client:
            identifier = self.submit(client)
            status = self.wait_for(client, identifier, "complete")
            self.assertEqual(self.submit(client), identifier)
            self.assertEqual(self.executor.synthesis_calls, 1)
            self.assertEqual(self.executor.assembly_calls, 1)
            for artifact in status["artifacts"]:
                response = client.get(artifact["url"], headers=AUTH)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(hashlib.sha256(response.content).hexdigest(), artifact["sha256"])
                self.assertEqual(len(response.content), artifact["size"])
            # Only the connected owner can acknowledge the verified delivery;
            # the acknowledged run's rows disappear while its files remain.
            self.assertEqual(client.post(f"/v1/jobs/{identifier}/ack", headers=AUTH,
                                         json={"session": SESSION}).status_code, 200)
            self.assertEqual(client.get(f"/v1/jobs/{identifier}", headers=AUTH).status_code, 404)
        self.assertTrue(self.executor.closed)
        self.assertTrue(any((self.root / "render-work").rglob("chapter.m4a")))
        replacement = FixtureExecutor()
        restarted = ChapterJobs(self.root, replacement)
        with self.client_type(create_app(restarted, {"api_key": KEY})) as client:
            # A restart clears past-run rows; the old job stays unknown until
            # a fresh connected submission re-renders it.
            self.assertEqual(client.get(f"/v1/jobs/{identifier}", headers=AUTH).status_code, 404)
            self.assertEqual(self.submit(client), identifier)
            self.wait_for(client, identifier, "complete")
            self.assertEqual(replacement.synthesis_calls, 1)

    def test_real_maintenance_service_discards_old_runs_and_rejects_new_job_writes(self):
        from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs

        job = self.jobs.submit(self.spec, self.payload["idempotency_key"],
                               session=service_session(self.jobs))
        self.jobs._set_state(job["id"], "generating")
        self.jobs.close()
        engine = FixtureExecutor()
        maintenance = ChapterJobs(self.root, engine, maintenance=True)
        with self.client_type(create_app(maintenance, {"api_key": KEY})) as client:
            caps = client.get("/v1/capabilities", headers=AUTH)
            self.assertEqual(caps.status_code, 200)
            self.assertTrue(caps.json()["maintenance"])
            self.assertEqual(client.get(f"/v1/jobs/{job['id']}", headers=AUTH).status_code, 404)
            self.assertEqual(client.post("/v1/voices/ensure", headers=AUTH,
                                         json={"voices": []}).status_code, 200)
            for endpoint, payload in (("/v1/jobs", self.payload), (f"/v1/jobs/{job['id']}/retry", None)):
                response = client.post(endpoint, json=payload, headers=AUTH)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("maintenance mode", response.text)
            self.assertEqual(engine.synthesis_calls, 0)
            self.assertEqual(engine.assembly_calls, 0)
            self.assertEqual(client.get("/v1/jobs", headers=AUTH).json()["jobs"], [])
        self.assertTrue(engine.closed)

    def test_real_core_failure_retry_reuses_job_and_retains_event_history(self):
        self.executor.fail_next = True
        with self.client_type(self.app) as client:
            identifier = self.submit(client)
            failed = self.wait_for(client, identifier, "failed")
            self.assertIn("Fixture speech check failed", failed["error"])
            session = self.connect(client)
            response = client.post(f"/v1/jobs/{identifier}/retry", headers=AUTH,
                                   json={"session": session})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["id"], identifier)
            completed = self.wait_for(client, identifier, "complete")
            self.assertGreater(completed["last_event"], failed["last_event"])
            self.assertEqual(self.executor.synthesis_calls, 2)
            events = client.get(f"/v1/jobs/{identifier}/events", headers=AUTH).json()["events"]
            self.assertTrue(any(event.get("state") == "failed" for event in events))
            self.assertTrue(any("Retry queued" in event["message"] for event in events))

    def test_real_core_terminal_websocket_replays_over_500_events(self):
        from starlette.websockets import WebSocketDisconnect

        with self.client_type(self.app) as client:
            identifier = self.submit(client)
            self.wait_for(client, identifier, "complete")
            with self.jobs._connect() as db:
                for index in range(1001):
                    self.jobs._event(db, identifier, f"Retained progress {index}")
            last = self.jobs.status(identifier)["last_event"]
            with client.websocket_connect(f"/v1/jobs/{identifier}/events", headers=AUTH) as socket:
                received = []
                try:
                    while True:
                        received.extend(event["sequence"] for event in socket.receive_json()["events"])
                except WebSocketDisconnect as exc:
                    self.assertEqual(exc.code, 1000)
            self.assertEqual(received, list(range(1, last + 1)))

    def test_real_core_keeps_generation_running_after_socket_disconnect(self):
        self.executor.release.clear()
        with self.client_type(self.app) as client:
            try:
                identifier = self.submit(client)
                self.wait_for(client, identifier, "generating")
                with client.websocket_connect(f"/v1/jobs/{identifier}/events", headers=AUTH) as socket:
                    self.assertEqual(socket.receive_json()["job"]["state"], "generating")
                self.assertEqual(client.get(f"/v1/jobs/{identifier}", headers=AUTH).json()["state"], "generating")
            finally:
                self.executor.release.set()
            self.wait_for(client, identifier, "complete")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_real_core_voice_registration_validates_hash_and_reuses_flac(self):
        from epub_tts_narrator.renderer.common import digest

        path = self.root / "reference.flac"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
             "-ac", "1", "-c:a", "flac", str(path)],
            capture_output=True, check=True, timeout=10,
        )
        encoded = path.read_bytes()
        manifest = {"version": 1, "audio_sha256": hashlib.sha256(encoded).hexdigest(), "reference_text": "Fixture voice."}
        asset = digest(manifest)
        payload = {"manifest": manifest, "audio_base64": base64.b64encode(encoded).decode()}
        with self.client_type(self.app) as client:
            self.assertEqual(client.post("/v1/voices/ensure", headers=AUTH, json={"voices": [asset]}).json(), {"missing": [asset]})
            for _ in range(2):
                response = client.put(f"/v1/voices/{asset}", headers=AUTH, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.post("/v1/voices/ensure", headers=AUTH, json={"voices": [asset]}).json(), {"missing": []})
            bad = payload | {"audio_base64": base64.b64encode(encoded + b"changed").decode()}
            self.assertEqual(client.put(f"/v1/voices/{asset}", headers=AUTH, json=bad).status_code, 400)
            self.assertEqual((self.root / "voices" / asset / "reference.flac").read_bytes(), encoded)


if __name__ == "__main__":
    unittest.main()
