"""Shared model scheduling crosses chapter boundaries without racing take files."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import tempfile
from threading import Event, Lock, get_ident
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle
from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.common import digest, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import make_render_plan
from test_chapter_jobs import Executor, specification
from test_renderer_paragraph_workflow import ParagraphFixtureEngine
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureVerifier, pcm_key, write_tone


def settings(concurrency=1):
    config = deepcopy(DEFAULTS)
    config["quality"] = validate_policy({})
    config["render"].update(ffmpeg=FFMPEG, ffprobe=FFPROBE, normalize_loudness=False, max_attempts=1)
    config["render_pool"]["workers"] = [{"name": "shared-gpu", "type": "fish_http",
        "url": "http://localhost:8081/v1/tts", "api_key": "", "model": "drama-3",
        "timeout_seconds": 30, "concurrency": concurrency, "multi_speaker": True}]
    return config


def chapter_spec(executor, number, text, voice_hash):
    row = {"id": f"s{number}", "block_id": f"b{number}", "chapter_id": "original-epub-chapter",
           "start": 0, "end": len(text), "text": text, "spoken_text": text,
           "voice_id": "voice", "role_id": "narrator", "break_after": "paragraph"}
    chunk = {"chunk_id": digest(row)[:32], "segment": row, "spoken_text": text,
             "spoken_start": 0, "spoken_end": len(text), "source_ranges": [{"start": 0, "end": len(text)}],
             "silent_control": False}
    return {"version": 1, "book": {"id": "fixture", "title": "Fixture Book"},
            "source_fingerprint": "a" * 64, "renderer_fingerprint": digest(executor.identity),
            "chapter": {"number": number, "id": f"chapter-{number:04d}", "title": f"Chapter {number}",
                        "chunks": [chunk]}, "total_chapters": 2, "voice_bindings": {"voice": voice_hash},
            "render": {}}


class CountingEngine(ParagraphFixtureEngine):
    def __init__(self):
        super().__init__()
        self.next_speech = Event()
        self.generation_entered = Event()
        self.generation_release = None
        self.guard = Lock()
        self.active = self.maximum = 0
        self.thread_ids = []

    def _write(self, text, path, seed):
        with self.guard:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.thread_ids.append(get_ident())
        self.generation_entered.set()
        if text.startswith("Second"):
            self.next_speech.set()
        try:
            if self.generation_release is not None and not self.generation_release.wait(5):
                raise AssertionError("Synthesis was not released")
            return super()._write(text, path, seed)
        finally:
            with self.guard:
                self.active -= 1


class BlockingVerifier(FixtureVerifier):
    def __init__(self, fixtures):
        super().__init__(fixtures)
        self.entered, self.release = Event(), Event()

    def transcribe(self, path):
        if self.fixture_transcripts[pcm_key(path)].startswith("First"):
            self.entered.set()
            if not self.release.wait(5):
                raise AssertionError("Verification was not released")
        return super().transcribe(path)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ChapterPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.reference = self.root / "reference.wav"
        write_tone(self.reference)
        flac = self.root / "reference.flac"
        subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(self.reference),
                        "-c:a", "flac", str(flac)], check=True, timeout=30)
        self.audio = flac.read_bytes()
        self.manifest = {"version": 1, "audio_sha256": hashlib.sha256(self.audio).hexdigest(),
                         "reference_text": "Saved reference."}
        self.voice_hash = digest(self.manifest)

    def terminal(self, service, identifier):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            status = service.status(identifier)
            if status["state"] in {"complete", "failed"}:
                return status
            time.sleep(.01)
        self.fail("Chapter did not finish")

    def test_next_chapter_uses_free_gpu_during_previous_chapter_verification(self):
        config, backend = settings(), CountingEngine()
        verifier = BlockingVerifier(backend.fixture_transcripts)
        self.addCleanup(verifier.release.set)
        executor = ChapterExecutor(config, verifier=verifier)
        service = ChapterJobs(self.root / "server", executor)
        self.addCleanup(service.close)
        service.register_voice(self.voice_hash, self.manifest, self.audio)
        first = chapter_spec(executor, 1, "First chapter finishes speech before verification.", self.voice_hash)
        second = chapter_spec(executor, 2, "Second chapter can start speech immediately.", self.voice_hash)
        with patch("epub_tts_narrator.renderer.fish_http.FishHTTPEngine", return_value=backend) as factory:
            job1 = service.submit(first, digest(first))
            service.start()
            try:
                self.assertTrue(verifier.entered.wait(4))
                job2 = service.submit(second, digest(second))
                self.assertTrue(backend.next_speech.wait(4), "next chapter waited for previous verification")
                self.assertEqual(service.status(job1["id"])["state"], "generating")
                self.assertFalse(service.begin_shutdown_if_idle(.001))
                self.assertEqual(backend.maximum, 1)
                factory.assert_called_once()
            finally:
                verifier.release.set()
            self.assertEqual(self.terminal(service, job1["id"])["state"], "complete")
            self.assertEqual(self.terminal(service, job2["id"])["state"], "complete")
        self.assertEqual(service.capabilities()["parallel_chapters"], 2)
        service.close()
        self.assertTrue(all(not thread.is_alive() for thread in service._threads))

    def test_concurrent_identical_paragraphs_share_one_accepted_take(self):
        config, backend = settings(2), CountingEngine()
        release = backend.generation_release = Event()
        verifier = FixtureVerifier(backend.fixture_transcripts)
        executor = ChapterExecutor(config, verifier=verifier)
        self.addCleanup(executor.close)
        plan = make_render_plan(load_bundle(EXAMPLE / "bundle"), config)
        chunk = next(chunk for chunk in plan["chapters"][0]["chunks"] if "paragraph_turns" in chunk)
        first = chapter_spec(executor, 1, "unused", self.voice_hash)
        first["chapter"]["chunks"] = [chunk]
        first["voice_bindings"] = {speaker["voice_id"]: self.voice_hash for speaker in chunk_speakers(chunk)}
        second = deepcopy(first)
        second["chapter"].update(id="chapter-0002", number=2)
        voices = {voice_id: {"asset_hash": self.voice_hash, "reference_audio": str(self.reference),
                            "reference_text": "Saved reference."} for voice_id in first["voice_bindings"]}
        with patch("epub_tts_narrator.renderer.fish_http.FishHTTPEngine", return_value=backend) as factory:
            with ThreadPoolExecutor(max_workers=2) as workers:
                a = workers.submit(executor.synthesize, first, voices, self.root / "work", lambda _: None)
                self.assertTrue(backend.generation_entered.wait(3))
                b = workers.submit(executor.synthesize, second, voices, self.root / "work", lambda _: None)
                try:
                    time.sleep(.05)
                    self.assertEqual(len(backend.dialogue_calls), 1)
                finally:
                    release.set()
                prepared_a, prepared_b = a.result(5), b.result(5)
            factory.assert_called_once()
        self.assertEqual(len(backend.dialogue_calls), 1)
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(prepared_a["acceptance"], prepared_b["acceptance"])
        accepted = list((self.root / "work" / "clips").glob("*/accepted.json"))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(read_json(accepted[0])["attempt"], 1)

    def test_default_local_backend_has_one_persistent_model_thread(self):
        config, backend = settings(), CountingEngine()
        config["render_pool"]["workers"] = []
        verifier = FixtureVerifier(backend.fixture_transcripts)
        executor = ChapterExecutor(config, verifier=verifier)
        self.addCleanup(executor.close)
        initialization_threads = []

        def create_model(*args, **kwargs):
            initialization_threads.append(get_ident())
            return backend

        with patch("epub_tts_narrator.renderer.pool.create_engine", side_effect=create_model) as factory:
            self.assertEqual(executor.parallel_chapters, 2)
            self.assertIsInstance(executor._engine, RenderPool)
            self.assertEqual(executor._engine.max_parallel, 1)
            factory.assert_not_called()
            voices = {"voice": {"asset_hash": self.voice_hash, "reference_audio": str(self.reference),
                                 "reference_text": "Saved reference."}}
            specs = [chapter_spec(executor, number, f"Local chapter {number} shares the same model.", self.voice_hash)
                     for number in (1, 2)]
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [workers.submit(executor.synthesize, spec, voices, self.root / "work", lambda _: None)
                           for spec in specs]
                for future in futures:
                    future.result(5)
            factory.assert_called_once()
        self.assertEqual(len(set(initialization_threads + backend.thread_ids)), 1)
        self.assertNotEqual(initialization_threads[0], get_ident())
        self.assertEqual(backend.maximum, 1)

    def test_nonpool_plugin_retains_serial_chapter_contract(self):
        backend = CountingEngine()
        executor = ChapterExecutor(settings(), engine=backend, verifier=FixtureVerifier(backend.fixture_transcripts))
        self.addCleanup(executor.close)
        self.assertEqual(executor.parallel_chapters, 1)


class AtomicChapterClaimTests(unittest.TestCase):
    def test_two_coordinators_claim_every_job_exactly_once_and_drain(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = Executor()
            engine.parallel_chapters = 2
            service = ChapterJobs(Path(temporary), engine)
            self.addCleanup(service.close)
            identifiers = []
            for number in range(1, 31):
                spec = specification(service, number)
                spec["total_chapters"] = 30
                identifiers.append(service.submit(spec, digest(spec))["id"])
            service.start()
            until = time.monotonic() + 8
            while time.monotonic() < until:
                if all(service.status(identifier)["state"] == "complete" for identifier in identifiers):
                    break
                time.sleep(.01)
            service.close()
            self.assertEqual(sorted(engine.generated), list(range(1, 31)))
            self.assertEqual(service._active, 0)
            self.assertEqual(len(service._threads), 2)
            self.assertTrue(all(not thread.is_alive() for thread in service._threads))


if __name__ == "__main__":
    unittest.main()
