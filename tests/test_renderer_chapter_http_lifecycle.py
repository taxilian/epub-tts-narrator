"""Real-HTTP chapter lifecycle over an actual RenderPool.

A uvicorn server, the real urllib ChapterClient, a real ChapterJobs queue,
and a real ChapterExecutor whose engine is a pipeline RenderPool with a
gated fixture TTS prove the run-scoped lifecycle end to end: maintenance
activation, a disconnected owner's queued work never starting synthesis
while the started take drains into durable acceptance, a reattached
client's fresh epoch resuming accepted units without any new generation,
and a warm server serving sequential books and sixty generations with no
run history or owner-registry residue.
"""

from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle
from epub_tts_narrator.renderer.chapter_client import ChapterClient, ChapterClientError
from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_server import create_app
from epub_tts_narrator.renderer.common import digest, read_json
from epub_tts_narrator.renderer.engines import DRAMA3_MODEL_REVISION
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.speech_text import trim_speech_text
from epub_tts_narrator.renderer.workflow import make_render_plan
from audition_fixtures import prepare_bundle
from shared_fixtures import FFMPEG, FFPROBE, renderer_config
from test_renderer_workflow import EXAMPLE, FixtureEngine, FixtureVerifier, write_tone

WATCHDOG = 90
SERVER_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx", "uvicorn"))


class GatedEngine(FixtureEngine):
    """Fixture engine that holds synthesis calls from an index until released."""

    def __init__(self, fixture_transcripts, block_from=1):
        super().__init__(fixture_transcripts)
        self.lock = threading.Lock()
        self.started = []
        self.gate = threading.Event()
        self.gate.set()
        self.block_from = block_from
        self.blocked_entered = threading.Event()

    def synthesize(self, text, voice, path, *, seed, direction=""):
        with self.lock:
            index = len(self.started)
            self.started.append(text)
        if index >= self.block_from:
            self.blocked_entered.set()
            if not self.gate.wait(WATCHDOG):
                raise RuntimeError("gate watchdog: synthesis was never released")
        return FixtureEngine.synthesize(self, text, voice, path, seed=seed, direction=direction)


def exported_bundle(root):
    export_bundle(read_json(EXAMPLE / "reviewed.json"), root / "bundle")
    return load_bundle(root / "bundle")


@unittest.skipUnless(SERVER_DEPS and FFMPEG and FFPROBE, "requires server dependencies and FFmpeg/FFprobe")
class ChapterHttpLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = prepare_bundle(exported_bundle(self.root))
        self.executor_config = renderer_config(self.root, series_id="http-lifecycle",
                                               checkpoint="unused-tts", asr_model="unused-asr",
                                               max_attempts=2, paragraph_synthesis=False)
        self.executor_config["render_pool"] = {
            # A remote-type worker keeps the fixture gate off the shared
            # LocalRuntime thread, so an indefinitely blocked synthesis
            # cannot deadlock model-affine recognition.
            "workers": [{"name": "lane", "type": "remote", "multi_speaker": True,
                         "concurrency": 1}],
            "model_revision": DRAMA3_MODEL_REVISION, "scheduler": "pipeline",
            "chapter_concurrency": 2,
        }
        self.fixtures = {}
        self.engine = GatedEngine(self.fixtures, block_from=1)
        self.verifier = FixtureVerifier(self.fixtures)
        self.pool = RenderPool(deepcopy(self.executor_config))
        self.addCleanup(self.pool.close)
        self.pool._workers[0]._engine = self.engine
        self.executor = ChapterExecutor(deepcopy(self.executor_config),
                                        engine=self.pool, verifier=self.verifier)
        self.service, self.client = self.start_server()

    def start_server(self):
        settings = {"api_key": "", "event_poll_seconds": 0.01, "idle_timeout_seconds": 0,
                    "assembly_workers": 2}
        import uvicorn
        service = ChapterJobs(self.root / "server", self.executor, settings, maintenance=True)
        app = create_app(service, settings)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(server.started)
        port = server.servers[0].sockets[0].getsockname()[1]

        def shutdown():
            server.should_exit = True
            thread.join(timeout=20)

        self.addCleanup(shutdown)
        self.addCleanup(service.close)
        client = ChapterClient({"url": f"http://127.0.0.1:{port}", "api_key": "",
                                "timeout_seconds": 30, "websocket": False, "poll_seconds": 0.02})
        return service, client

    def activate(self):
        self.assertTrue(self.client.request("GET", "v1/capabilities")["maintenance"])
        self.assertFalse(self.client.request("POST", "v1/server/activate", {})["maintenance"])

    def plan_chunks(self):
        plan = make_render_plan(self.bundle, self.executor_config)
        return [chunk for chunk in plan["chapters"][0]["chunks"] if not chunk["silent_control"]]

    def expected_texts(self, chunks):
        # The engine receives trimmed delivery text, and concurrent chunk
        # transactions race for the synthesis lane, so expectations compare
        # trimmed text as sets, never source order.
        return sorted(trim_speech_text(chunk["spoken_text"]) for chunk in chunks)

    def register_voices(self, chunks):
        bindings = {}
        needed = {speaker["voice_id"] for chunk in chunks for speaker in chunk_speakers(chunk)}
        for voice_id in sorted(needed):
            wav = self.root / f"{voice_id}-ref.wav"
            write_tone(wav)
            flac = wav.with_suffix(".flac")
            subprocess.run([FFMPEG, "-v", "error", "-i", str(wav), "-c:a", "flac", str(flac)],
                           check=True, timeout=60)
            audio = flac.read_bytes()
            manifest = {"version": 1, "audio_sha256": sha256(audio).hexdigest(),
                        "reference_text": "Reference words."}
            identifier = digest(manifest)
            self.service.register_voice(identifier, manifest, audio)
            bindings[voice_id] = identifier
        return bindings

    def specification(self, number, chunks, bindings, fingerprint="a" * 64, book="http-lifecycle",
                      total=1):
        return {"version": 1, "book": {"id": book, "title": "Lifecycle Book"},
                "source_fingerprint": fingerprint, "renderer_fingerprint": self.service.fingerprint,
                "chapter": {"id": f"chapter-{number:04d}", "number": number,
                            "title": f"Chapter {number}", "chunks": chunks},
                "total_chapters": total, "voice_bindings": bindings, "render": {}}

    def submit(self, session, spec):
        return self.client.request("POST", "v1/jobs",
                                   {"spec": spec, "idempotency_key": digest(spec), "session": session})

    def accepted_count(self):
        return len(list((self.service.root / "render-work" / "clips").glob("*/accepted.json")))

    def wait_for(self, session, job_id, states, timeout=WATCHDOG, heartbeat=True):
        deadline = time.monotonic() + timeout
        status = {}
        while time.monotonic() < deadline:
            try:
                if heartbeat:
                    self.client.request("POST", f"v1/sessions/{session}/heartbeat", {})
                status = self.client.request("GET", f"v1/jobs/{job_id}")
                if status["state"] in states:
                    return status
            except ChapterClientError as exc:
                if exc.status == 404 and "gone" in states:
                    return {"state": "gone"}
                raise
            time.sleep(0.02)
        self.fail(f"Job did not reach {states}: {status}")

    def wait_until(self, condition, timeout=WATCHDOG, message="condition never held"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.02)
        self.fail(message)

    def test_reattached_epoch_completes_accepted_units_while_old_cancel_stays_unaffected(self):
        self.activate()
        self.engine.gate.clear()
        chunks = self.plan_chunks()[:2]
        bindings = self.register_voices(chunks)
        spec = self.specification(1, chunks, bindings)
        session_a = digest({"epoch": "old"})
        self.client.request("POST", "v1/sessions", {"id": session_a})
        job = self.submit(session_a, spec)["id"]
        self.wait_for(session_a, job, ("generating",))
        self.wait_until(lambda: self.engine.blocked_entered.is_set()
                        and self.accepted_count() == 1,
                        message="first take never reached durable acceptance")

        self.client.request("DELETE", f"v1/sessions/{session_a}")
        session_b = digest({"epoch": "new"})
        self.client.request("POST", "v1/sessions", {"id": session_b})
        self.submit(session_b, spec)
        self.engine.gate.set()
        self.wait_for(session_b, job, ("complete",))
        self.assertEqual(sorted(self.engine.started), self.expected_texts(chunks))

        with self.assertRaises(ChapterClientError):
            self.client.request("POST", f"v1/jobs/{job}/ack", {"session": session_a})
        with self.assertRaises(ChapterClientError):
            self.client.request("POST", f"v1/sessions/{session_a}/heartbeat", {})
        self.client.request("POST", f"v1/jobs/{job}/ack", {"session": session_b})
        self.assertEqual(self.client.request("GET", "v1/jobs")["jobs"], [])

    def test_queued_work_never_starts_synthesis_after_owner_disconnect(self):
        self.activate()
        self.engine.gate.clear()
        self.engine.block_from = 0
        chunks = self.plan_chunks()[:3]
        bindings = self.register_voices(chunks)
        session = digest({"queue": "owner"})
        self.client.request("POST", "v1/sessions", {"id": session})
        first = self.submit(session, self.specification(1, chunks[:2], bindings, total=2))["id"]
        second = self.submit(session, self.specification(2, chunks[2:3], bindings, total=2))["id"]
        self.wait_until(lambda: len(self.engine.started) == 1,
                        message="the first take never started")
        self.wait_for(session, second, ("generating",))

        self.client.request("DELETE", f"v1/sessions/{session}")
        self.engine.gate.set()
        self.wait_for(session, first, ("gone",), heartbeat=False)
        self.wait_for(session, second, ("gone",), heartbeat=False)
        # Exactly one take ever reached the engine; it finished verification
        # and saved acceptance. Every other chapter and sibling chunk stopped
        # without generating.
        self.assertEqual(len(self.engine.started), 1)
        self.assertEqual(self.accepted_count(), 1)

        fresh = digest({"queue": "fresh"})
        self.client.request("POST", "v1/sessions", {"id": fresh})
        resumed_first = self.submit(fresh, self.specification(1, chunks[:2], bindings, total=2))["id"]
        resumed_second = self.submit(fresh, self.specification(2, chunks[2:3], bindings, total=2))["id"]
        self.wait_for(fresh, resumed_first, ("complete",))
        self.wait_for(fresh, resumed_second, ("complete",))
        # Every chunk is generated at most once across the disconnect: the
        # accepted unit is reused, only never-started chunks generate.
        self.assertEqual(len(self.engine.started), 3)
        self.assertEqual(sorted(self.engine.started), self.expected_texts(chunks))
        self.client.request("POST", f"v1/jobs/{resumed_first}/ack", {"session": fresh})
        self.client.request("POST", f"v1/jobs/{resumed_second}/ack", {"session": fresh})
        self.assertEqual(self.client.request("GET", "v1/jobs")["jobs"], [])

    def test_warm_server_serves_sequential_books_and_generations_without_registry_leaks(self):
        self.activate()
        self.assertIs(self.service.executor, self.executor)
        self.assertIs(self.executor._engine, self.pool)
        chunks = self.plan_chunks()[:2]
        bindings = self.register_voices(chunks)
        session = digest({"warm": "books"})
        self.client.request("POST", "v1/sessions", {"id": session})
        for fingerprint, book in (("a" * 64, "book-one"), ("b" * 64, "book-two")):
            for number in (1, 2):
                spec = self.specification(number, chunks[number - 1:number], bindings,
                                          fingerprint=fingerprint, book=book, total=2)
                job = self.submit(session, spec)["id"]
                self.wait_for(session, job, ("complete",))
                self.client.request("POST", f"v1/jobs/{job}/ack", {"session": session})
            self.assertEqual(self.client.request("GET", "v1/jobs")["jobs"], [])

        single = self.plan_chunks()[:1]
        for generation in range(60):
            spec = self.specification(1, single, bindings,
                                      fingerprint=f"{generation:064x}", book=f"gen-{generation}")
            job = self.submit(session, spec)["id"]
            self.wait_for(session, job, ("complete",))
            self.client.request("POST", f"v1/jobs/{job}/ack", {"session": session})
        self.assertEqual(self.client.request("GET", "v1/jobs")["jobs"], [])
        self.assertEqual(self.service._owners, {})
        self.assertEqual(self.service._epochs, {})
        self.assertEqual(self.service._job_budgets, {})
        self.assertEqual(self.service._submission_budgets, {})
        self.assertEqual(self.pool.logical._owners, {})


if __name__ == "__main__":
    unittest.main()
