"""Malformed chapter maps and worker interruptions must fail before publication."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_protocol import validate_spec
from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.workflow import make_render_plan
from test_chapter_jobs import Executor, specification
from test_renderer_workflow import EXAMPLE


class ChapterProtocolAuditTests(unittest.TestCase):
    def setUp(self):
        config = deepcopy(DEFAULTS)
        bundle = load_bundle(EXAMPLE / "bundle")
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        plan = make_render_plan(bundle, config)
        self.spec = {"version": 1, "book": bundle["source"]["book"],
                     "source_fingerprint": bundle["manifest"]["source_fingerprint"],
                     "renderer_fingerprint": "f" * 64, "chapter": plan["chapters"][0],
                     "total_chapters": len(bundle["chapters"]), "render": {},
                     "voice_bindings": {key: digest(key) for key in plan["voice_ids"]}}
        self.group = next(chunk for chunk in self.spec["chapter"]["chunks"] if "paragraph_turns" in chunk)

    def test_real_plan_keeps_original_epub_chapter_identity(self):
        self.assertEqual(validate_spec(self.spec), self.spec)
        # Logical output chapters and EPUB source chapters are separate IDs.
        self.assertNotEqual(self.group["segment"]["chapter_id"], self.spec["chapter"]["id"])

    def test_forged_group_source_ranges_are_rejected(self):
        self.group["source_ranges"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "source mapping"):
            validate_spec(self.spec)

    def test_forged_group_spoken_offsets_are_rejected(self):
        self.group["spoken_end"] -= 1
        with self.assertRaisesRegex(ValueError, "source mapping"):
            validate_spec(self.spec)

    def test_group_cannot_overlap_or_reorder_source_turns(self):
        turns = self.group["paragraph_turns"]
        turns[0], turns[1] = turns[1], turns[0]
        self.group["spoken_text"] = "".join(turn["spoken_text"] for turn in turns)
        self.group["segment"]["text"] = self.group["spoken_text"]
        with self.assertRaisesRegex(ValueError, "source mapping"):
            validate_spec(self.spec)

    def test_group_turn_ids_are_required_and_unique(self):
        self.group["paragraph_turns"][1]["chunk_id"] = self.group["paragraph_turns"][0]["chunk_id"]
        with self.assertRaisesRegex(ValueError, "duplicate chapter passage"):
            validate_spec(self.spec)

    def test_group_cannot_cross_original_source_chapters(self):
        self.group["paragraph_turns"][1]["segment"]["chapter_id"] = "another-original-chapter"
        with self.assertRaisesRegex(ValueError, "crosses source"):
            validate_spec(self.spec)

    def test_runtime_audio_fields_and_malformed_performers_are_rejected(self):
        self.group["paragraph_turns"][0]["wav_path"] = "/private/file.wav"
        with self.assertRaisesRegex(ValueError, "chapter passage"):
            validate_spec(self.spec)
        self.group["paragraph_turns"][0].pop("wav_path")
        self.group["paragraph_turns"][0]["segment"]["voice_id"] = []
        with self.assertRaisesRegex(ValueError, "performers"):
            validate_spec(self.spec)

    def test_bad_audio_metadata_is_rejected_before_model_work(self):
        self.spec["book"]["authors"] = "Not an author array"
        with self.assertRaisesRegex(ValueError, "authors"):
            validate_spec(self.spec)


class ChapterLifetimeAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def terminal(self, service, identifier):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            status = service.status(identifier)
            if status["state"] in {"complete", "failed"}:
                return status
            time.sleep(.01)
        self.fail("Chapter job did not finish")

    def test_model_baseexceptions_release_leases_and_keep_coordinator_alive(self):
        for stage in ("synthesize", "assemble"):
            with self.subTest(stage=stage):
                engine = Executor()
                original = getattr(engine, stage)

                def interrupt_first(spec, *args, **kwargs):
                    if spec["chapter"]["number"] == 1:
                        raise KeyboardInterrupt("model worker interrupted")
                    return original(spec, *args, **kwargs)

                setattr(engine, stage, interrupt_first)
                service = ChapterJobs(self.root / stage, engine)
                self.addCleanup(service.close)
                first = specification(service, 1)
                second = specification(service, 2)
                a = service.submit(first, digest(first))
                b = service.submit(second, digest(second))
                service.start()
                status = self.terminal(service, a["id"])
                self.assertEqual(status["state"], "failed")
                self.assertIn("KeyboardInterrupt", status["error"])
                self.assertEqual(self.terminal(service, b["id"])["state"], "complete")
                self.assertTrue(service._thread.is_alive())
                service.close()
                self.assertEqual(service._active, 0)

    def test_completed_idempotent_job_survives_renderer_configuration_change(self):
        engine = Executor()
        service = ChapterJobs(self.root / "server", engine)
        self.addCleanup(service.close)
        spec = specification(service)
        key = digest(spec)
        old = service.submit(spec, key)
        service.start()
        self.assertEqual(self.terminal(service, old["id"])["state"], "complete")
        service.close()
        new_engine = Executor()
        new_engine.identity = {"engine": "changed-fixture"}
        replacement = ChapterJobs(self.root / "server", new_engine)
        self.addCleanup(replacement.close)
        self.assertNotEqual(service.fingerprint, replacement.fingerprint)
        resumed = replacement.submit(spec, key)
        self.assertEqual(resumed["id"], old["id"])
        self.assertEqual(resumed["state"], "complete")
        self.assertTrue(resumed["artifacts"])
        self.assertEqual(new_engine.generated, [])
        with self.assertRaisesRegex(ValueError, "renderer changed"):
            replacement.submit(spec, digest("a brand-new idempotency key"))

    def test_failure_redaction_precedes_truncation(self):
        secret = "api_key_fixture_" + "a" * 100
        service = ChapterJobs(self.root / "server", Executor(), {"api_key": secret})
        self.addCleanup(service.close)
        spec = specification(service)
        job = service.submit(spec, digest(spec))
        service._failure(job["id"], RuntimeError("x" * 1970 + secret))
        message = service.status(job["id"])["error"]
        self.assertNotIn("api_key_fixture", message)
        self.assertIn("[redacted]", message)

    def test_close_waits_for_admitted_voice_upload_and_closes_executor_once(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        engine = Executor()
        engine.close = Mock()
        service = ChapterJobs(self.root / "server", engine)
        self.addCleanup(service.close)
        service.start()
        audio = b"fLaC" + b"fixture" * 20
        manifest = {"version": 1, "audio_sha256": hashlib.sha256(audio).hexdigest(),
                    "reference_text": "Saved voice."}
        identifier = digest(manifest)
        errors = []

        def inspect(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Test upload was not released")
            return SimpleNamespace(stdout=json.dumps({"streams": [{"codec_name": "flac", "channels": 1}],
                                                        "format": {"duration": "1"}}).encode())

        def upload():
            try:
                service.register_voice(identifier, manifest, audio)
            except BaseException as exc:
                errors.append(exc)

        def close():
            try:
                service.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        with patch("epub_tts_narrator.renderer.chapter_jobs.subprocess.run", side_effect=inspect):
            worker = threading.Thread(target=upload)
            worker.start()
            self.assertTrue(entered.wait(2))
            closer = threading.Thread(target=close)
            closer.start()
            self.assertFalse(finished.wait(.05))
            self.assertFalse(service._file_lock.closed)
            other_closer = threading.Thread(target=close)
            other_closer.start()
            release.set()
            worker.join(3)
            closer.join(3)
            other_closer.join(3)
        self.assertFalse(worker.is_alive() or closer.is_alive() or other_closer.is_alive())
        self.assertEqual(errors, [])
        engine.close.assert_called_once_with()
        self.assertEqual(service.ensure_voices([identifier]), [])
        self.assertTrue(service._file_lock.closed)


if __name__ == "__main__":
    unittest.main()
