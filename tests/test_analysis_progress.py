"""Pending analysis progress must checkpoint durably and never fake completion."""

from copy import deepcopy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from epub_tts_narrator.analysis_progress import (
    PENDING_DIRNAME, PROGRESS_VERSION, STAGES, create_pending, discard_pending,
    inspect_pending, is_pending_directory, pending_identity, resume_document)
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import fingerprint, write_json
from test_contracts import valid_document

CLIENT_IDENTITY = {"model": "fixture-model", "base_url": "http://127.0.0.1:8000/v1"}
OPTIONS_FINGERPRINT = fingerprint({"chunk_chars": 2400})


def run_identity(source=None):
    return pending_identity(source if source is not None else valid_document(),
                            CLIENT_IDENTITY, OPTIONS_FINGERPRINT)


def state_path(output):
    return Path(output) / PENDING_DIRNAME / "state.json"


class AnalysisProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "book-01"

    def advance(self, handle, count):
        document = valid_document()
        for stage in STAGES[:count]:
            handle.begin_stage(stage)
            handle.complete_stage(stage, deepcopy(document))
        return document

    def test_pending_directory_records_stage_checkpoints_and_resumes_from_the_last_completed_stage(self):
        document = valid_document()
        with create_pending(self.output, run_identity()) as handle:
            self.assertTrue(is_pending_directory(self.output))
            pending = self.output / PENDING_DIRNAME
            self.assertTrue((pending / "state.json").is_file())
            self.assertTrue((pending / "snapshots").is_dir())
            self.assertTrue((pending / "lease").is_file())
            owner = json.loads((pending / "owner.json").read_text(encoding="utf-8"))
            self.assertEqual(owner["format"], "analysis-progress-owner")
            self.assertEqual(owner["pid"], os.getpid())
            self.assertTrue(owner["token"])
            handle.begin_stage("analyzed")
            record = inspect_pending(self.output)
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["current_stage"], "analyzed")
            self.assertEqual(record["completed_stages"], [])
            handle.complete_stage("analyzed", deepcopy(document))
        record = inspect_pending(self.output)
        self.assertEqual(record["scope"], "analysis-progress")
        self.assertEqual(record["version"], PROGRESS_VERSION)
        self.assertEqual(record["completed_stages"], ["analyzed"])
        self.assertIsNone(record["current_stage"])
        entry = record["snapshots"]["analyzed"]
        snapshot = self.output / PENDING_DIRNAME / entry["path"]
        self.assertEqual(entry["path"], "snapshots/analyzed.json")
        self.assertEqual(entry["sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
        resumed, next_stage = resume_document(self.output, run_identity())
        self.assertEqual(resumed, document)
        self.assertEqual(next_stage, "speech-normalized-1")

    def test_tampered_progress_record_fails_closed_naming_the_state_file(self):
        for mutate in ("completed_stages", "non-finite"):
            with self.subTest(mutate=mutate):
                output = Path(self.temporary.name) / f"book-{mutate}"
                with create_pending(output, run_identity()) as handle:
                    handle.begin_stage("analyzed")
                    handle.complete_stage("analyzed", valid_document())
                path = state_path(output)
                record = json.loads(path.read_text(encoding="utf-8"))
                if mutate == "completed_stages":
                    record["completed_stages"].append("speakers-decided")
                    text = json.dumps(record)
                else:
                    text = path.read_text(encoding="utf-8").replace(": null", ": NaN")
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as raised:
                    inspect_pending(output)
                self.assertIn("state.json", str(raised.exception))
                with self.assertRaises(ValueError):
                    resume_document(output, run_identity())

    def test_stage_sequence_enforces_ordered_begin_and_single_completion(self):
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", valid_document())
            with self.assertRaises(ValueError):
                handle.begin_stage("samples-prepared-1")
            with self.assertRaises(ValueError):
                handle.begin_stage("analyzed")
            with self.assertRaises(ValueError):
                handle.complete_stage("analyzed", valid_document())
            with self.assertRaises(ValueError):
                handle.begin_stage("nonsense-stage")
            with self.assertRaises(ValueError):
                handle.complete_stage("nonsense-stage", valid_document())
            # Completing without a begin is the crash-recovery path: begin only
            # records intent, completion is authoritative.
            handle.complete_stage("speech-normalized-1", valid_document())
            for stage in STAGES[2:]:
                handle.begin_stage(stage)
                handle.complete_stage(stage, valid_document())
            with self.assertRaises(ValueError):
                handle.begin_stage(STAGES[0])
            with self.assertRaises(ValueError):
                handle.complete_stage(STAGES[-1], valid_document())

    def test_snapshot_written_without_a_state_update_is_redone_on_resume(self):
        document = valid_document()
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", deepcopy(document))
        # A crash here leaves the snapshot but not the state update.
        orphan = self.output / PENDING_DIRNAME / "snapshots" / "speech-normalized-1.json"
        write_json(orphan, valid_document())
        resumed, next_stage = resume_document(self.output, run_identity())
        self.assertEqual(resumed, document)
        self.assertEqual(next_stage, "speech-normalized-1")
        with create_pending(self.output, run_identity()) as handle:
            record = inspect_pending(self.output)
            self.assertEqual(record["completed_stages"], ["analyzed"])
            handle.begin_stage("speech-normalized-1")
            handle.complete_stage("speech-normalized-1", valid_document())
        record = inspect_pending(self.output)
        self.assertEqual(record["completed_stages"], ["analyzed", "speech-normalized-1"])

    def test_resume_refuses_progress_from_a_different_run_identity(self):
        other_source = valid_document()
        other_source["source"]["book"]["title"] = "A Different Book"
        for field, alternate in (("source", run_identity(other_source)),
                                 ("options", pending_identity(valid_document(), CLIENT_IDENTITY,
                                                              fingerprint({"chunk_chars": 1200})))):
            with self.subTest(field=field):
                output = Path(self.temporary.name) / f"book-{field}"
                with create_pending(output, run_identity()) as handle:
                    handle.begin_stage("analyzed")
                    handle.complete_stage("analyzed", valid_document())
                with self.assertRaises(ValueError) as raised:
                    resume_document(output, alternate)
                message = str(raised.exception)
                self.assertIn("different run configuration", message)
                self.assertIn(field, message)
                self.assertIn("state.json", message)

    def test_foreign_content_prevents_pending_directory_recognition(self):
        cases = {
            "manifest": {"manifest.json": "{}"},
            "sibling-file": {"notes.txt": "unrelated"},
            "sibling-dir": {"scripts": {}},
        }
        for name, extra in cases.items():
            with self.subTest(name=name):
                output = Path(self.temporary.name) / f"book-{name}"
                with create_pending(output, run_identity()):
                    pass
                for relative, content in extra.items():
                    path = output / relative
                    if isinstance(content, dict):
                        path.mkdir()
                    else:
                        path.write_text(content, encoding="utf-8")
                self.assertFalse(is_pending_directory(output))
                self.assertIsNone(inspect_pending(output))

    @unittest.skipUnless(os.name == "posix", "POSIX fcntl required")
    def test_concurrent_analysis_is_refused_until_the_lease_is_released(self):
        handle = create_pending(self.output, run_identity())
        try:
            with self.assertRaises(FileExistsError) as raised:
                create_pending(self.output, run_identity())
            self.assertIn("Another analysis is running", str(raised.exception))
        finally:
            handle.close()
        with create_pending(self.output, run_identity()) as resumed:
            record = inspect_pending(self.output)
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["completed_stages"], [])
            resumed.begin_stage("analyzed")
            resumed.complete_stage("analyzed", valid_document())
        self.assertEqual(inspect_pending(self.output)["completed_stages"], ["analyzed"])

    def test_resume_rejects_a_snapshot_that_is_not_a_valid_analysis_document(self):
        self.assertEqual(validate_document(valid_document()), [])
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", {"format": "bogus"})
        with self.assertRaises(ValueError) as raised:
            resume_document(self.output, run_identity())
        self.assertIn("snapshots/analyzed.json", str(raised.exception))

    def test_recorded_snapshots_are_immutable_after_completion(self):
        document = valid_document()
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", deepcopy(document))
        snapshot = self.output / PENDING_DIRNAME / "snapshots" / "analyzed.json"
        before = snapshot.read_bytes()
        other = valid_document()
        other["source"]["book"]["title"] = "Rewritten"
        with create_pending(self.output, run_identity()) as handle:
            with self.assertRaises(ValueError):
                handle.complete_stage("analyzed", other)
        self.assertEqual(snapshot.read_bytes(), before)

    def test_redone_stage_replaces_a_divergent_orphaned_snapshot_and_reuses_identical_content(self):
        document = valid_document()
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", deepcopy(document))
        orphan = self.output / PENDING_DIRNAME / "snapshots" / "speech-normalized-1.json"
        divergent = valid_document()
        divergent["source"]["book"]["title"] = "From A Dead Attempt"
        write_json(orphan, divergent)
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("speech-normalized-1")
            handle.complete_stage("speech-normalized-1", deepcopy(document))
        self.assertNotIn("From A Dead Attempt", orphan.read_text(encoding="utf-8"))
        record = inspect_pending(self.output)
        self.assertEqual(record["completed_stages"], ["analyzed", "speech-normalized-1"])
        self.assertEqual(hashlib.sha256(orphan.read_bytes()).hexdigest(),
                         record["snapshots"]["speech-normalized-1"]["sha256"])
        # An orphan with identical content is reused instead of rewritten.
        reused = Path(self.temporary.name) / "book-reuse"
        with create_pending(reused, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", deepcopy(document))
        reused_orphan = reused / PENDING_DIRNAME / "snapshots" / "speech-normalized-1.json"
        write_json(reused_orphan, valid_document())
        identical = reused_orphan.read_bytes()
        with create_pending(reused, run_identity()) as handle:
            handle.complete_stage("speech-normalized-1", valid_document())
        self.assertEqual(reused_orphan.read_bytes(), identical)

    def test_inspect_returns_none_when_nothing_is_pending(self):
        self.assertIsNone(inspect_pending(self.output))
        self.output.mkdir()
        self.assertIsNone(inspect_pending(self.output))
        manifest_only = Path(self.temporary.name) / "completed-book"
        manifest_only.mkdir()
        (manifest_only / "manifest.json").write_text("{}", encoding="utf-8")
        self.assertFalse(is_pending_directory(manifest_only))
        self.assertIsNone(inspect_pending(manifest_only))

    def test_resume_with_no_completed_stages_starts_from_the_first_stage(self):
        create_pending(self.output, run_identity()).close()
        self.assertEqual(resume_document(self.output, run_identity()), (None, "analyzed"))

    def test_resume_after_every_stage_reports_no_next_stage(self):
        document = valid_document()
        with create_pending(self.output, run_identity()) as handle:
            for stage in STAGES:
                handle.begin_stage(stage)
                handle.complete_stage(stage, deepcopy(document))
        resumed, next_stage = resume_document(self.output, run_identity())
        self.assertEqual(resumed, document)
        self.assertIsNone(next_stage)

    def test_discard_removes_only_the_pending_directory_and_refuses_foreign_content(self):
        with create_pending(self.output, run_identity()) as handle:
            handle.begin_stage("analyzed")
            handle.complete_stage("analyzed", valid_document())
        discard_pending(self.output)
        self.assertFalse((self.output / PENDING_DIRNAME).exists())
        self.assertTrue(self.output.is_dir())
        discard_pending(self.output)  # Absent pending state is a no-op.
        foreign = Path(self.temporary.name) / "foreign-book"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("not ours", encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            discard_pending(foreign)
        self.assertIn("Refusing to remove", str(raised.exception))
        self.assertTrue((foreign / "keep.txt").exists())

    def test_failure_status_is_recorded_and_resume_still_offers_the_next_stage(self):
        for status, kwargs, message in (("failed", {}, "disk went away"),
                                        ("interrupted", {"interrupted": True}, "")):
            with self.subTest(status=status):
                output = Path(self.temporary.name) / f"book-{status}"
                with create_pending(output, run_identity()) as handle:
                    handle.begin_stage("analyzed")
                    handle.complete_stage("analyzed", valid_document())
                    handle.begin_stage("speech-normalized-1")
                    handle.mark_failed(message or "x" * 5000, **kwargs)
                record = inspect_pending(output)
                self.assertEqual(record["status"], status)
                self.assertIsNotNone(record["error"])
                self.assertLessEqual(len(record["error"]), 2012)
                resumed, next_stage = resume_document(output, run_identity())
                self.assertIsNotNone(resumed)
                self.assertEqual(next_stage, "speech-normalized-1")

    def test_identity_accepts_a_document_or_its_source_and_validates_fields(self):
        document = valid_document()
        from_document = run_identity(document)
        from_source = pending_identity(document["source"], CLIENT_IDENTITY, OPTIONS_FINGERPRINT)
        self.assertEqual(from_document, from_source)
        self.assertEqual(from_document["source_fingerprint"], fingerprint(document["source"]))
        self.assertEqual(from_document["prompt_version"], "7")
        self.assertEqual(from_document["analyzer_version"], "0.1.0")
        self.assertEqual(from_document["provider"], CLIENT_IDENTITY)
        self.assertEqual(from_document["options_fingerprint"], OPTIONS_FINGERPRINT)
        cases = {
            "source": ({"not": "a source"}, CLIENT_IDENTITY, OPTIONS_FINGERPRINT,
                       "analysis progress identity source"),
            "provider": (document["source"], "not-an-object", OPTIONS_FINGERPRINT,
                         "analysis progress identity provider"),
            "options": (document["source"], CLIENT_IDENTITY, "nothex",
                        "analysis progress identity options_fingerprint"),
        }
        for name, args in cases.items():
            with self.subTest(field=name):
                with self.assertRaises(ValueError) as raised:
                    pending_identity(*args[:3])
                self.assertIn(args[3], str(raised.exception))


if __name__ == "__main__":
    unittest.main()
