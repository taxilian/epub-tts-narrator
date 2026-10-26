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
from unittest.mock import patch

from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_protocol import validate_spec
from epub_tts_narrator.renderer.common import digest, write_json
from test_renderer_workflow import write_tone
from shared_fixtures import service_session


class Executor:
    identity = {"engine": "fixture"}
    multi_speaker = False

    def __init__(self):
        self.generated = []
        self.budgets = []
        self.fail = False
        self.assembly_entered = threading.Event()
        self.assembly_release = None

    def synthesize(self, spec, voices, work, notify, retry_budget=None):
        self.generated.append(spec["chapter"]["number"])
        self.budgets.append(retry_budget)
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
        self.session = service_session(self.service)

    def submit(self, number=1, submission=None, service=None, session=None):
        service = service if service is not None else self.service
        session = session if session is not None else self.session
        spec = specification(service, number)
        return service.submit(spec, digest(spec), submission, session)

    def terminal(self, job_id):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            status = self.service.status(job_id)
            if status["state"] in {"complete", "failed"}:
                return status
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_forwarding_wrappers_receive_the_exact_existing_call(self):
        calls = []

        class Target:
            identity = {"engine": "wrapped-target"}

            def synthesize(self, spec, voices, work, notify, retry_budget=None):
                notify("generated")
                return {}

            def assemble(self, spec, prepared, work, notify):
                folder = work / digest(spec)
                folder.mkdir(exist_ok=True)
                (folder / "chapter.m4a").write_bytes(b"x")
                return {"chapter.m4a": folder / "chapter.m4a"}

        class RejectingTarget(Target):
            def synthesize(self, spec, voices, work, notify, retry_budget=None,
                           **forbidden):
                if "scheduler_owner" in forbidden:
                    raise TypeError(f"unexpected keyword scheduler_owner")
                return Target.synthesize(self, spec, voices, work, notify,
                                         retry_budget=retry_budget)

        class ForwardingWrapper:
            identity = {"engine": "forwarding-wrapper"}

            def __init__(self, target):
                self.target = target
                self.multi_speaker = False

            def synthesize(self, spec, *args, **kwargs):
                if "scheduler_owner" in kwargs:
                    raise TypeError("scheduler_owner must never reach a wrapper")
                calls.append(dict(kwargs))
                return self.target.synthesize(spec, *args, **kwargs)

            def assemble(self, spec, prepared, work, notify):
                return self.target.assemble(spec, prepared, work, notify)

        self.service = ChapterJobs(self.root / "wrapped",
                                   ForwardingWrapper(RejectingTarget()),
                                   {"assembly_workers": 2})
        self.addCleanup(self.service.close)
        self.service.start()
        self.session = service_session(self.service)
        done = self.submit()
        self.assertEqual(self.terminal(done["id"])["state"], "complete")
        # The wrapper saw the exact legacy call: retry budget only, never an
        # unknown scheduling keyword it would forward to its target.
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {"notify", "retry_budget"})

    def test_named_owner_executors_receive_job_and_epoch_owners(self):
        observed = []

        class OwnerAwareExecutor:
            identity = {"engine": "owner-aware"}
            multi_speaker = False

            def synthesize(self, spec, voices, work, notify, retry_budget=None,
                           scheduler_owner=None):
                observed.append(scheduler_owner)
                notify("done")
                return {}

            def assemble(self, spec, prepared, work, notify):
                folder = work / digest(spec)
                folder.mkdir(exist_ok=True)
                (folder / "chapter.m4a").write_bytes(b"x")
                return {"chapter.m4a": folder / "chapter.m4a"}

        self.service = ChapterJobs(self.root / "owner-aware", OwnerAwareExecutor(),
                                   {"assembly_workers": 2})
        self.addCleanup(self.service.close)
        self.service.start()
        self.session = service_session(self.service)
        first = self.submit(1)
        second = self.submit(2)
        self.assertEqual(self.terminal(first["id"])["state"], "complete")
        self.assertEqual(self.terminal(second["id"])["state"], "complete")
        self.assertEqual(len(observed), 2)
        self.assertEqual([owner[0] for owner in observed],
                         [first["id"], second["id"]])
        # Both first claims carry the same epoch number with distinct job
        # identities: the owner pair is run-unique per chapter job.
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertNotEqual(observed[0][0], observed[1][0])

    def test_restart_clears_unowned_rows_and_never_autoexecutes_them(self):
        first = self.submit()
        self.assertEqual(self.submit()["id"], first["id"])
        self.service.close()
        other = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(other.close)
        other.start()
        self.service = other
        # Past-run rows are scheduling history: a restart removes them and
        # they never execute on their own.
        with self.assertRaises(KeyError):
            other.status(first["id"])
        self.assertEqual(other.executor.generated, [])
        # A fresh run-scoped submission re-renders instead of replaying history.
        spec = specification(other)
        created = other.submit(spec, digest({"run": "fresh"}), None, service_session(other))
        self.assertNotEqual(created["id"], first["id"])
        self.assertEqual(self.terminal(created["id"])["state"], "complete")
        self.assertEqual(other.executor.generated, [1])

    def test_changed_payload_cannot_reuse_idempotency_key(self):
        spec = specification(self.service)
        key = digest(spec)
        self.service.submit(spec, key, None, self.session)
        spec["chapter"]["title"] = "Different chapter"
        with self.assertRaisesRegex(ValueError, "different chapter"):
            self.service.submit(spec, key, None, self.session)

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
        self.service.retry(job["id"], session=self.session)
        self.assertEqual(self.terminal(job["id"])["state"], "complete")
        resumed = self.service.events(job["id"], last)
        self.assertTrue(all(event["sequence"] > last for event in resumed))
        self.assertEqual(resumed[-1]["state"], "complete")

    def test_pending_jobs_and_draining_control_admission(self):
        self.service._last_activity -= 1000
        self.submit()
        self.assertFalse(self.service.begin_shutdown_if_idle(600))
        self.service.start()
        job = self.submit()
        self.terminal(job["id"])
        self.service.acknowledge(job["id"], self.session)
        self.service._last_activity -= 1000
        self.assertTrue(self.service.begin_shutdown_if_idle(600))
        with self.assertRaisesRegex(RuntimeError, "draining"):
            self.submit(2)
        self.service.cancel_shutdown()
        self.assertEqual(self.submit(2)["state"], "queued")

    def test_completed_delivery_keeps_the_service_busy_until_acknowledged(self):
        job = self.submit()
        self.service.start()
        self.terminal(job["id"])
        self.service._last_activity -= 1000
        self.service.status(job["id"])
        self.service.events(job["id"], 0)
        self.assertEqual(self.service.idle_seconds(), 0)
        self.service.acknowledge(job["id"], self.session)
        self.service._last_activity -= 1000
        self.service.list_jobs()
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
        self.assertFalse(self.service.begin_shutdown_if_idle(600))
        self.service.acknowledge(job["id"], self.session)
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

    def test_submission_budget_is_shared_across_jobs_and_refreshed_per_submission(self):
        self.assertTrue(self.service.capabilities()["submission_budget"])
        first, second = self.submit(1, "d" * 64), self.submit(2, "d" * 64)
        self.service.start()
        self.assertEqual(self.terminal(first["id"])["state"], "complete")
        self.assertEqual(self.terminal(second["id"])["state"], "complete")
        shared = self.engine.budgets[0]
        self.assertIsNotNone(shared)
        self.assertIs(self.engine.budgets[1], shared)
        self.assertFalse(shared.retry_paragraphs)
        third = self.submit(3, "e" * 64)
        self.assertEqual(self.terminal(third["id"])["state"], "complete")
        self.assertIsNot(self.engine.budgets[2], shared)

    def test_retry_spends_the_retried_submission_budget_and_keeps_it_without_one(self):
        self.engine.fail = True
        job = self.submit(1, "d" * 64)
        self.service.start()
        self.assertEqual(self.terminal(job["id"])["state"], "failed")
        original = self.engine.budgets[0]
        self.service.retry(job["id"], session=self.session)
        self.assertEqual(self.terminal(job["id"])["state"], "failed")
        self.assertIs(self.engine.budgets[1], original)
        self.service.retry(job["id"], "e" * 64, self.session)
        self.assertEqual(self.terminal(job["id"])["state"], "failed")
        self.assertIsNot(self.engine.budgets[2], original)
        self.engine.fail = False
        self.service.retry(job["id"], session=self.session)
        self.assertEqual(self.terminal(job["id"])["state"], "complete")
        self.assertIs(self.engine.budgets[3], self.engine.budgets[2])

    def test_failed_paragraph_verification_records_a_hash_only_wire_suffix(self):
        from epub_tts_narrator.renderer.chapter_protocol import FAILED_PARAGRAPH_FIELD
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
        request = {"kind": "paragraph_speech", "chunk": {
            "chunk_id": "c" * 32,
            "paragraph_turns": [{"chunk_id": "a" * 32}, {"chunk_id": "b" * 32}]}}
        job = self.submit()
        for failure, label in ((SpeechVerificationFailed(
                "Speech verification failed after 2 attempts. Inspect /work/clips/x",
                request=request, report={}), "identified"),
                               (SpeechVerificationFailed("word" * 1000, request=request, report={}),
                                "truncated")):
            with self.subTest(label=label):
                self.service._failure(job["id"], failure)
                error = self.service.status(job["id"])["error"]
                self.assertIn(FAILED_PARAGRAPH_FIELD, error)
                detail = json.loads(error.rsplit(FAILED_PARAGRAPH_FIELD, 1)[1])
                self.assertEqual(detail, {"request_hash": digest(request), "chunk_id": "c" * 32})
                self.assertNotIn("paragraph_turns", error)
                if label == "truncated":
                    self.assertEqual(len(error), 2000)
        self.service._failure(job["id"], ValueError("plain fixture failure"))
        self.assertNotIn(FAILED_PARAGRAPH_FIELD, self.service.status(job["id"])["error"])

    def test_interrupted_state_is_cleared_at_start_and_single_process_lease_holds(self):
        job = self.submit()
        self.service._set_state(job["id"], "generating")
        self.service.start()
        other = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(other.close)
        with self.assertRaisesRegex(RuntimeError, "Another chapter service"):
            other.start()
        self.service.close()
        restarted = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(restarted.close)
        restarted.start()
        self.service = restarted
        # The interrupted generation is past-run history: a restart clears
        # every session and removes the row instead of recovering it into a
        # paused queue entry.
        with self.assertRaises(KeyError):
            self.service.status(job["id"])
        self.assertEqual(self.engine.generated, [])

    def test_maintenance_clears_old_run_history_without_deleting_accepted_audio(self):
        jobs = [self.submit(number) for number in range(1, 4)]
        self.service._set_state(jobs[1]["id"], "generating")
        self.service._set_state(jobs[2]["id"], "assembling")
        accepted = self.service.root / "render-work" / "accepted.flac"
        accepted.write_bytes(b"accepted take must remain unchanged")
        self.service.close()
        engine = Executor()
        maintenance = ChapterJobs(self.root / "server", engine, maintenance=True)
        self.addCleanup(maintenance.close)
        self.assertEqual(maintenance.fingerprint, self.service.fingerprint)
        self.assertTrue(maintenance.capabilities()["maintenance"])
        maintenance.start()
        maintenance.start()
        self.assertEqual(engine.generated, [])
        self.assertEqual(maintenance._threads, [])
        self.assertIsNone(maintenance._assembly)
        for job in jobs:
            with self.assertRaises(KeyError):
                maintenance.status(job["id"])
        contender = ChapterJobs(self.root / "server", Executor())
        self.addCleanup(contender.close)
        with self.assertRaisesRegex(RuntimeError, "Another chapter service"):
            contender.start()
        maintenance.close()
        self.assertEqual(accepted.read_bytes(), b"accepted take must remain unchanged")
        contender.start()
        self.service = contender
        # The restarted service never executes on its own; each job resumes
        # only when its client explicitly reattaches with a live session.
        contender_session = service_session(contender)
        for job, number in zip(jobs, range(1, 4)):
            spec = specification(contender, number)
            contender.submit(spec, digest(spec), None, contender_session)
        for job in jobs:
            self.assertEqual(self.terminal(job["id"])["state"], "complete")
        self.assertEqual(contender.executor.generated, [1, 2, 3])
        self.assertFalse(contender.capabilities()["maintenance"])

    def test_maintenance_rejects_submissions_and_retries_after_discarding_old_run_history(self):
        job = self.submit()
        self.service._set_state(job["id"], "failed", error="retained failure")
        self.service.close()
        maintenance = ChapterJobs(self.root / "server", Executor(), maintenance=True)
        self.addCleanup(maintenance.close)
        maintenance.start()
        for number in (1, 2):
            spec = specification(maintenance, number)
            with self.assertRaisesRegex(RuntimeError, "maintenance mode"):
                maintenance.submit(spec, digest(spec), None, service_session(maintenance))
        with self.assertRaisesRegex(RuntimeError, "maintenance mode"):
            maintenance.retry(job["id"], session=service_session(maintenance))
        self.assertEqual(maintenance.list_jobs(), [])

    def test_maintenance_can_activate_without_reloading_its_executor(self):
        engine = Executor()
        warm = ChapterJobs(self.root / "warm", engine, maintenance=True)
        self.addCleanup(warm.close)
        warm.start()
        self.assertTrue(warm.capabilities()["maintenance"])
        self.assertFalse(warm.activate()["maintenance"])
        count = len(warm._threads)
        warm.activate()
        self.assertEqual(len(warm._threads), count)
        self.assertIs(warm.executor, engine)
        self.assertEqual(engine.generated, [])

    def test_shutdown_begins_only_at_the_five_minute_idle_boundary(self):
        self.service._last_activity = 100.0
        with patch("epub_tts_narrator.renderer.chapter_jobs.time.monotonic", return_value=399.0):
            self.assertFalse(self.service.begin_shutdown_if_idle(300))
        with patch("epub_tts_narrator.renderer.chapter_jobs.time.monotonic", return_value=400.0):
            self.assertTrue(self.service.begin_shutdown_if_idle(300))

    def test_idle_window_starts_after_busy_delivery_not_before_it(self):
        job = self.submit()
        self.service._set_state(job["id"], "complete")
        self.service._last_activity = 100.0
        with patch("epub_tts_narrator.renderer.chapter_jobs.time.monotonic", return_value=399.0):
            self.assertEqual(self.service.idle_seconds(), 0)
        self.service.sessions.revoke(self.session)
        with patch("epub_tts_narrator.renderer.chapter_jobs.time.monotonic", return_value=698.0):
            self.assertFalse(self.service.begin_shutdown_if_idle(300))
        with patch("epub_tts_narrator.renderer.chapter_jobs.time.monotonic", return_value=699.0):
            self.assertTrue(self.service.begin_shutdown_if_idle(300))

    def test_maintenance_paused_jobs_do_not_block_idle_shutdown_but_uploads_do(self):
        self.submit()
        self.service.close()
        maintenance = ChapterJobs(self.root / "server", Executor(), maintenance=True,
                                 settings={"idle_action": "runpod_stop", "persistent_data": True},
                                 publisher=Publisher())
        self.addCleanup(maintenance.close)
        maintenance.start()
        maintenance._last_activity -= 1000
        self.assertGreater(maintenance.idle_seconds(), 900)
        maintenance._active = 1
        try:
            self.assertFalse(maintenance.begin_shutdown_if_idle(600))
        finally:
            maintenance._active = 0
        maintenance._last_activity -= 1000
        self.assertTrue(maintenance.begin_shutdown_if_idle(600))

    def test_live_setup_heartbeat_prevents_shutdown_during_a_long_smoke_check(self):
        maintenance = ChapterJobs(self.root / "setup", Executor(), maintenance=True)
        self.addCleanup(maintenance.close)
        maintenance.start()
        session = service_session(maintenance)
        maintenance._last_activity -= 1000
        self.assertEqual(maintenance.idle_seconds(), 0)
        self.assertFalse(maintenance.begin_shutdown_if_idle(300))
        maintenance.revoke_session(session)
        maintenance._last_activity -= 301
        self.assertTrue(maintenance.begin_shutdown_if_idle(300))

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
