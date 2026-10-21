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
from epub_tts_narrator.renderer.config import DEFAULTS


HAS_SERVER_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))
JOB = "a" * 64
VOICE = "b" * 64
KEY = "local-test-credential"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeJobs:
    def __init__(self, root):
        self.root = root
        self.started = False
        self.closed = False
        self.voices = {}
        self.submit_calls = []
        self.retry_calls = []
        self.events_requested = []
        self.idle_checks = []
        self.cancel_count = 0
        self.shutdown_ready = False
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

    def ensure_voices(self, voices):
        return [voice for voice in voices if voice not in self.voices]

    def register_voice(self, asset_id, manifest, audio):
        self.voices[asset_id] = {"manifest": manifest, "audio": audio}
        return {"id": asset_id}

    def submit(self, spec, idempotency_key):
        self.submit_calls.append((spec, idempotency_key))
        return deepcopy(self.job)

    def status(self, job_id):
        if job_id != JOB:
            raise KeyError(job_id)
        return deepcopy(self.job)

    def events(self, job_id, after):
        self.status(job_id)
        self.events_requested.append(after)
        return [event for event in self.log if event["sequence"] > after]

    def retry(self, job_id):
        self.retry_calls.append(job_id)
        return self.status(job_id)

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

    def test_client_only_configuration_cannot_start_an_unconfigured_server(self):
        config = deepcopy(DEFAULTS)
        config["remote_chapters"]["url"] = "https://fixture.invalid"
        with patch("epub_tts_narrator.renderer.chapter_executor.ChapterExecutor") as executor:
            with self.assertRaisesRegex(ValueError, "serve-chapters requires verification.model"):
                serve(config)
        executor.assert_not_called()

    def test_server_without_remote_workers_needs_a_local_speech_checkpoint(self):
        config = deepcopy(DEFAULTS)
        config["verification"]["model"] = "/fixture/local-asr"
        with patch("epub_tts_narrator.renderer.chapter_executor.ChapterExecutor") as executor:
            with self.assertRaisesRegex(ValueError, "serve-chapters requires engine.checkpoint"):
                serve(config)
        executor.assert_not_called()

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
            body = {"spec": {"chapter": {"number": 2}}, "idempotency_key": "c" * 64}
            result = client.post("/v1/jobs", headers=AUTH, json=body)
            self.assertEqual(result.status_code, 202)
            self.assertEqual(result.json()["id"], JOB)
            self.assertEqual(self.jobs.submit_calls, [(body["spec"], body["idempotency_key"])])
            retry = client.post(f"/v1/jobs/{JOB}/retry", headers=AUTH)
            self.assertEqual(retry.status_code, 202)
            self.assertEqual(self.jobs.retry_calls, [JOB])

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

    def synthesize(self, spec, voices, work, notify):
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
        response = client.post("/v1/jobs", json=self.payload, headers=AUTH)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["id"]

    def test_real_core_completion_download_and_restart_preserve_idempotency(self):
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
        self.assertTrue(self.executor.closed)
        replacement = FixtureExecutor()
        restarted = ChapterJobs(self.root, replacement)
        with self.client_type(create_app(restarted, {"api_key": KEY})) as client:
            self.assertEqual(self.submit(client), identifier)
            self.assertEqual(self.wait_for(client, identifier, "complete")["id"], identifier)
            self.assertEqual(replacement.synthesis_calls, 0)

    def test_real_core_failure_retry_reuses_job_and_retains_event_history(self):
        self.executor.fail_next = True
        with self.client_type(self.app) as client:
            identifier = self.submit(client)
            failed = self.wait_for(client, identifier, "failed")
            self.assertIn("Fixture speech check failed", failed["error"])
            response = client.post(f"/v1/jobs/{identifier}/retry", headers=AUTH)
            self.assertEqual(response.status_code, 202)
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
