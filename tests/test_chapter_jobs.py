"""Durability, admission, pipelining and immutable compressed asset contracts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_protocol import validate_spec
from epub_tts_narrator.renderer.common import digest, write_json
from test_renderer_workflow import write_tone


class Executor:
    identity = {"engine": "fixture"}
    multi_speaker = False

    def __init__(self):
        self.generated = []
        self.fail = False
        self.assembly_entered = threading.Event()
        self.assembly_release = None

    def synthesize(self, spec, voices, work, notify):
        self.generated.append(spec["chapter"]["number"])
        notify("Paragraph complete")
        if self.fail:
            raise ValueError("fixture failure")
        return {}

    def assemble(self, spec, prepared, work, notify):
        self.assembly_entered.set()
        if self.assembly_release is not None:
            if not self.assembly_release.wait(5):
                raise RuntimeError("fixture assembly timeout")
        folder = work / digest(spec)
        folder.mkdir(exist_ok=True)
        write_json(folder / "chapter.json", {"chapter": spec["chapter"]["number"]})
        return {"chapter.json": folder / "chapter.json"}


class Publisher:
    def __init__(self):
        self.fail = False
        self.counter = 0

    def publish(self, job_id, name, path, sha256):
        if self.fail:
            raise RuntimeError("upload failed")
        return {"key": job_id + "/" + name}

    def url(self, record):
        self.counter += 1
        return "https://storage.invalid/" + record["key"] + f"?signature={self.counter}"


def specification(service, number=1):
    return {"version": 1, "book": {"id": "fixture", "title": "A Book"},
            "source_fingerprint": "a" * 64, "renderer_fingerprint": service.fingerprint,
            "chapter": {"id": f"chapter-{number:04d}", "number": number, "title": "Chapter", "chunks": []},
            "total_chapters": 3, "voice_bindings": {}, "render": {"chapter_format": "m4a"}}


class ChapterJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = Executor()
        self.service = ChapterJobs(self.root / "server", self.engine, {"assembly_workers": 2})
        self.addCleanup(self.service.close)

    def submit(self, number=1):
        spec = specification(self.service, number)
        return self.service.submit(spec, digest(spec))

    def terminal(self, job_id):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            status = self.service.status(job_id)
            if status["state"] in {"complete", "failed"}:
                return status
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_idempotent_submission_survives_restart(self):
        first = self.submit()
        self.assertEqual(self.submit()["id"], first["id"])
        self.service.close()
        other = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(other.close)
        spec = specification(other)
        self.assertEqual(other.submit(spec, digest(spec))["id"], first["id"])
        other.start()
        self.service = other
        self.assertEqual(self.terminal(first["id"])["state"], "complete")

    def test_changed_payload_cannot_reuse_idempotency_key(self):
        spec = specification(self.service)
        key = digest(spec)
        self.service.submit(spec, key)
        spec["chapter"]["title"] = "Different chapter"
        with self.assertRaisesRegex(ValueError, "different chapter"):
            self.service.submit(spec, key)

    def test_generation_of_next_chapter_overlaps_assembly(self):
        release = threading.Event()
        self.engine.assembly_release = release
        self.addCleanup(release.set)
        first, second = self.submit(1), self.submit(2)
        self.service.start()
        self.assertTrue(self.engine.assembly_entered.wait(3))
        until = time.monotonic() + 3
        while len(self.engine.generated) < 2 and time.monotonic() < until:
            time.sleep(.01)
        self.assertEqual(self.engine.generated, [1, 2])
        self.assertEqual(self.service.status(first["id"])["state"], "assembling")
        self.assertFalse(self.service.begin_shutdown_if_idle(.001))
        release.set()
        self.assertEqual(self.terminal(first["id"])["state"], "complete")
        self.assertEqual(self.terminal(second["id"])["state"], "complete")

    def test_failed_job_can_retry_and_replay_events(self):
        self.engine.fail = True
        job = self.submit()
        self.service.start()
        self.assertEqual(self.terminal(job["id"])["state"], "failed")
        events = self.service.events(job["id"], 0)
        last = events[-1]["sequence"]
        self.engine.fail = False
        self.service.retry(job["id"])
        self.assertEqual(self.terminal(job["id"])["state"], "complete")
        resumed = self.service.events(job["id"], last)
        self.assertTrue(all(event["sequence"] > last for event in resumed))
        self.assertEqual(resumed[-1]["state"], "complete")

    def test_pending_jobs_and_draining_control_admission(self):
        self.service._last_activity -= 1000
        self.submit()
        self.assertFalse(self.service.begin_shutdown_if_idle(600))
        self.service.start()
        self.terminal(self.submit()["id"])
        self.service._last_activity -= 1000
        self.assertTrue(self.service.begin_shutdown_if_idle(600))
        with self.assertRaisesRegex(RuntimeError, "draining"):
            self.submit(2)
        self.service.cancel_shutdown()
        self.assertEqual(self.submit(2)["state"], "queued")

    def test_reading_completed_status_does_not_reset_idle_timer(self):
        job = self.submit()
        self.service.start()
        self.terminal(job["id"])
        self.service._last_activity -= 1000
        self.service.status(job["id"])
        self.service.events(job["id"], 0)
        self.assertGreater(self.service.idle_seconds(), 900)

    def test_shutdown_requires_durable_results_and_urls_refresh(self):
        self.service.publisher = Publisher()
        self.service.settings.update(idle_action="runpod_stop", persistent_data=True)
        job = self.submit()
        self.service.start()
        first = self.terminal(job["id"])
        self.assertEqual(first["state"], "complete")
        second = self.service.status(job["id"])
        self.assertNotEqual(first["artifacts"][0]["url"], second["artifacts"][0]["url"])
        self.service._last_activity -= 1000
        self.assertTrue(self.service.begin_shutdown_if_idle(600))

    def test_failed_publication_does_not_mark_complete(self):
        publisher = Publisher()
        publisher.fail = True
        self.service.publisher = publisher
        job = self.submit()
        self.service.start()
        status = self.terminal(job["id"])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["artifacts"], [])

    def test_artifact_integrity_and_wav_exclusion(self):
        job = self.submit()
        self.service.start()
        self.terminal(job["id"])
        path = self.service.artifact(job["id"], "chapter.json")
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.service.artifact(job["id"], "chapter.json")
        with self.assertRaises(KeyError):
            self.service.artifact(job["id"], "chapter.wav")

    def test_recover_inflight_state_and_single_process_lease(self):
        job = self.submit()
        self.service._set_state(job["id"], "generating")
        self.service.start()
        other = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(other.close)
        with self.assertRaisesRegex(RuntimeError, "Another chapter service"):
            other.start()
        self.assertEqual(self.terminal(job["id"])["state"], "complete")
        self.assertTrue(any("Recovered" in event["message"] for event in self.service.events(job["id"])))

    def test_server_settings_are_not_client_controlled(self):
        spec = specification(self.service)
        spec["render"]["ffmpeg"] = "/arbitrary/program"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.service.submit(spec, digest(spec))
        spec["render"] = {}
        spec["renderer_fingerprint"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "renderer changed"):
            self.service.submit(spec, digest(spec))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_voice_upload_is_lossless_compressed_and_immutable(self):
        wav, flac = self.root / "voice.wav", self.root / "voice.flac"
        write_tone(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "flac", str(flac)], check=True)
        audio = flac.read_bytes()
        manifest = {"version": 1, "audio_sha256": hashlib.sha256(audio).hexdigest(), "reference_text": "Test voice."}
        identifier = digest(manifest)
        self.assertEqual(self.service.ensure_voices([identifier]), [identifier])
        self.service.register_voice(identifier, manifest, audio)
        self.service.register_voice(identifier, manifest, audio)
        self.assertEqual(self.service.ensure_voices([identifier]), [])
        with self.assertRaisesRegex(ValueError, "compressed FLAC"):
            self.service.register_voice(identifier, manifest, wav.read_bytes())
        with self.assertRaisesRegex(ValueError, "hash"):
            self.service.register_voice(identifier, dict(manifest, reference_text="Changed."), audio)


if __name__ == "__main__":
    unittest.main()
