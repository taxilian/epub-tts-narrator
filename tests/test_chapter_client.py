"""Remote chapter orchestration preserves jobs, voice references and compressed wire data."""

import base64
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.chapter_client import (
    ChapterClient, ChapterClientError, PROTOCOL, _cached_artifacts_present,
    _migrate_legacy_job, _read_cache, _read_job, _render_remote_locked, _selection, _submit_chapter,
    _voice_assets, _wait_jobs, _validate_metadata, assemble_remote,
)
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import make_render_plan
from audition_fixtures import prepare_document
from test_contracts import valid_document
from test_renderer_workflow import write_tone, FFMPEG, FFPROBE


def fixture_artifacts(spec, mutate=None, mutate_manifest=None):
    """A complete transport fixture; real codec coverage uses ChapterExecutor below."""
    chapter = spec["chapter"]
    segments, cursor = [], 0
    for chunk in chapter["chunks"]:
        row = chunk["segment"]
        duration = 0 if chunk["silent_control"] else 4410
        timing = {"chunk_id": chunk["chunk_id"], "segment_id": row["id"],
                  "block_id": row["block_id"], "voice_id": row.get("voice_id"), "role_id": row.get("role_id"),
                  "source_text": row["text"], "source_start": row["start"], "source_end": row["end"],
                  "source_ranges": deepcopy(chunk["source_ranges"]), "spoken_text": chunk["spoken_text"],
                  "spoken_start": chunk["spoken_start"], "spoken_end": chunk["spoken_end"],
                  "no_speech": chunk["silent_control"], "chapter_start_frame": cursor,
                  "chapter_end_frame": cursor + duration}
        if duration:
            timing.update(source_audio_start_frame=0, source_audio_end_frame=duration)
        if "paragraph_turns" in chunk:
            timing["precision"] = "paragraph"
            timing["source_segments"] = [{
                "chunk_id": turn["chunk_id"], "segment_id": turn["segment"]["id"],
                "block_id": turn["segment"]["block_id"], "voice_id": turn["segment"]["voice_id"],
                "role_id": turn["segment"]["role_id"], "source_ranges": deepcopy(turn["source_ranges"]),
                "spoken_text": turn["spoken_text"], "spoken_start": turn["spoken_start"],
                "spoken_end": turn["spoken_end"],
            } for turn in chunk["paragraph_turns"]]
        if row.get("performance"):
            timing["performance"] = deepcopy(row["performance"])
        segments.append(timing)
        cursor += duration
    encoded = b"compressed chapter fixture"
    audio_hash = hashlib.sha256(encoded).hexdigest() if cursor else None
    metadata = {"format": "epub-tts-narrator-chapter", "version": 1, "spec_hash": digest(spec),
                "source_fingerprint": spec["source_fingerprint"], "voice_bindings": spec["voice_bindings"],
                "id": chapter["id"], "number": chapter["number"], "title": chapter["title"],
                "frames": cursor, "channels": 1, "sample_rate": 44100, "segments": segments,
                "status": "complete" if cursor else "empty", "audio_file": "chapter.m4a" if cursor else None,
                "lossless_file": "chapter.flac" if cursor else None,
                "encoded_sha256": audio_hash, "pcm_data_sha256": "a" * 64 if cursor else None}
    context = {"spec_hash": digest(spec), "source_fingerprint": spec["source_fingerprint"],
               "voice_assets": spec["voice_bindings"], "engine": {"engine": "fixture"},
               "verifier": {"verifier": "fixture"}, "quality_version": 1, "quality": {}}
    verification = {"format": "audiobook-renderer-chapter-verification", "version": 1,
                    "spec_hash": digest(spec), "passed": True,
                    **{key: context[key] for key in ("engine", "verifier", "quality_version", "quality")},
                    "acceptance": [{"chunk_id": chunk["chunk_id"], "report": {"passed": True}}
                                   for chunk in chapter["chunks"] if not chunk["silent_control"]]}
    sync = {"format": "audiobook-renderer-chapter-sync", "version": 1, "book_id": spec["book"].get("id"),
            "source_fingerprint": spec["source_fingerprint"], "spec_hash": digest(spec),
            "chapter_id": chapter["id"], "chapter_number": chapter["number"], "sample_rate": 44100,
            "timebase": "1/44100", "range_end": "exclusive", "frames": cursor,
            "timing_basis": "PCM sample frames after trimming and inserted pauses",
            "precision": "paragraph" if any("source_segments" in row for row in segments) else "chunk",
            "audio_file": metadata["audio_file"], "audio_sha256": audio_hash,
            "playback_note": "Use codec delay/padding-aware playback; word alignment is not provided.",
            "segments": deepcopy(segments)}
    files = {"chapter.json": metadata, "sync.json": sync, "verification.json": verification}
    if cursor:
        files.update({"chapter.m4a": encoded, "chapter.flac": b"fLaCfixture"})
    if mutate:
        mutate(files)
    data = {name: value if isinstance(value, bytes) else json.dumps(value).encode() for name, value in files.items()}
    manifest = {"format": "epub-tts-narrator-chapter-artifacts", "version": 1,
                "identity": {"spec_hash": digest(spec), "context": context, "prepared_sha256": "b" * 64},
                "spec_hash": digest(spec), "source_fingerprint": spec["source_fingerprint"],
                "status": metadata["status"], "artifacts": {
                    name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)} for name, value in data.items()}}
    if mutate_manifest:
        mutate_manifest(manifest)
    data["manifest.json"] = json.dumps({**manifest, "record_sha256": digest(manifest)}).encode()
    return data


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = ChapterClient({"url": "https://worker.example", "api_key": "secret-fixture",
                                     "timeout_seconds": 5, "poll_seconds": .1, "websocket": False})
        self.body = b"compressed fixture"
        self.artifact = {"name": "chapter.flac", "sha256": hashlib.sha256(self.body).hexdigest(),
                         "size": len(self.body), "url": "https://objects.example/chapter.flac?signature=fixture"}
        self.client.opener = Mock()
        self.client.opener.open.side_effect = lambda *args, **kwargs: BytesIO(self.body)

    def test_signed_external_download_does_not_receive_service_credentials_and_reuses_cache(self):
        target = self.root / "chapter.flac"
        self.client.download(self.artifact, target)
        request = self.client.opener.open.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(target.read_bytes(), self.body)
        self.client.download(self.artifact, target)
        self.assertEqual(self.client.opener.open.call_count, 1)

    def test_same_origin_download_uses_authentication(self):
        self.artifact["url"] = "/v1/jobs/job/artifacts/chapter.flac"
        self.client.download(self.artifact, self.root / "chapter.flac")
        self.assertEqual(self.client.opener.open.call_args.args[0].get_header("Authorization"),
                         "Bearer secret-fixture")

    def test_never_downloads_wav_or_untrusted_url(self):
        for changed in ({"name": "chapter.wav"}, {"name": "../chapter.flac"},
                        {"url": "file:///tmp/chapter.flac"}, {"url": "http://objects.example/chapter.flac"}):
            with self.subTest(changed=changed), self.assertRaises(ChapterClientError):
                self.client.download({**self.artifact, **changed}, self.root / "target")
        self.client.opener.open.assert_not_called()

    def test_bad_checksum_is_not_published_and_existing_modified_file_is_not_overwritten(self):
        target = self.root / "chapter.flac"
        with self.assertRaisesRegex(ChapterClientError, "integrity"):
            self.client.download({**self.artifact, "sha256": "0" * 64}, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        target.write_bytes(b"manual edit")
        with self.assertRaisesRegex(ChapterClientError, "changed"):
            self.client.download(self.artifact, target)
        self.assertEqual(target.read_bytes(), b"manual edit")

    def test_websocket_does_not_read_proxy_environment_settings(self):
        self.client.settings["websocket"] = True
        connect = Mock()
        with patch.dict("sys.modules", {"websockets.sync.client": Mock(connect=connect)}):
            self.client.websocket("job", 0)
        self.assertIsNone(connect.call_args.kwargs["proxy"])


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ChapterOrchestrationTests(unittest.TestCase):
    def test_observing_another_clients_paused_job_does_not_wait_forever(self):
        client = Mock()
        client.settings = {"poll_seconds": 0}
        identifier = "a" * 64
        client.request.side_effect = [
            {"events": [], "job": {"id": identifier, "state": "queued", "paused": True}},
            AssertionError("Observer kept waiting after the owner disconnected")]
        client.websocket.return_value = None
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ChapterClientError, "lost this run's session"):
            _wait_jobs(client, {identifier: (Path(folder), {"last_event": 0})}, lambda message: None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                     ffmpeg=FFMPEG, ffprobe=FFPROBE, paragraph_synthesis=False)
        self.config["quality"] = validate_policy({})
        self.config["remote_chapters"].update(url="https://worker.example", websocket=False, poll_seconds=.1)
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="fixture-series")
        document = prepare_document(valid_document())
        export_bundle(document, self.root / "book")
        self.bundle = load_bundle(self.root / "book")
        self.plan = make_render_plan(self.bundle, self.config)
        # A second chapter exercises queue submission before progress waiting.
        first = self.plan["chapters"][0]
        second = deepcopy(first)
        second.update(number=2, id="chapter-0002", title="Second chapter")
        self.plan["chapters"].append(second)
        self.bundle["chapters"].append(deepcopy(self.bundle["chapters"][0]))
        reference = self.root / "reference.wav"
        write_tone(reference)
        self.reference_before = reference.read_bytes()
        self.voice = {"reference_audio": str(reference), "reference_text": "A saved voice reference.",
                      "asset_hash": "a" * 64}
        self.store = Mock()
        self.store.load.return_value = self.voice

    def fake_client(self):
        owner = self
        class FakeClient:
            settings = owner.config["remote_chapters"]
            def __init__(self, settings):
                self.calls, self.jobs, self.artifacts, self.uploads = [], {}, {}, []
                self.sessions_opened, self.sessions_revoked = 0, 0
            def websocket(self, *args):
                return None
            def request(self, method, path, payload=None, timeout=None):
                self.calls.append((method, path, payload))
                if path == "v1/capabilities":
                    return {"protocol": PROTOCOL, "multi_speaker": True, "renderer_fingerprint": "f" * 64,
                            "client_sessions": 1, "session_ttl_seconds": 30, "ephemeral_runs": 1}
                if path == "v1/sessions":
                    self.sessions_opened += 1
                    return {"opened": True}
                if path.startswith("v1/sessions/"):
                    if method == "POST":
                        return {"renewed": True}
                    self.sessions_revoked += 1
                    return {"revoked": True}
                if path == "v1/voices/ensure":
                    return {"missing": payload["voices"]}
                if method == "PUT":
                    self.uploads.append(payload)
                    return {"registered": True}
                if path == "v1/jobs":
                    spec = payload["spec"]
                    job_id = digest(spec)
                    files = fixture_artifacts(spec)
                    artifacts = []
                    for name, data in files.items():
                        url = f"/{job_id}/{name}"
                        self.artifacts[url] = data
                        artifacts.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                                          "size": len(data), "url": url})
                    self.jobs[job_id] = {"id": job_id, "state": "completed", "artifacts": artifacts, "last_event": 1}
                    return {"id": job_id, "state": "queued"}
                if "/events?" in path:
                    job_id = path.split("/")[2]
                    return {"events": [{"sequence": 1, "message": "Ready"}], "job": self.jobs[job_id]}
                if path.endswith("/ack"):
                    return {"acknowledged": True}
                raise AssertionError((method, path))
            def download(self, artifact, target):
                target.write_bytes(self.artifacts[artifact["url"]])
        return FakeClient

    def test_registers_flac_once_queues_all_chapters_and_resumes_without_service(self):
        client = self.fake_client()(self.config["remote_chapters"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            result = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                           notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(client.uploads), 1)
        self.assertTrue(base64.b64decode(client.uploads[0]["audio_base64"]).startswith(b"fLaC"))
        first_event = next(index for index, (_, path, _) in enumerate(client.calls) if "/events?" in path)
        self.assertEqual(sum(path == "v1/jobs" for _, path, _ in client.calls[:first_event]), 2)
        self.assertEqual(len(result["chapters"]), 2)
        self.assertTrue(all(Path(item["audio_file"]).is_file() for item in result["chapters"]))
        for path in (self.root / "work" / "remote-chapters").glob("*/cache.json"):
            descriptor = json.loads(path.read_text())
            for artifact in descriptor["artifacts"]:
                if artifact["name"] == "chapter.flac":
                    self.assertTrue((path.parent / "chapter.flac").is_file())
        self.assertEqual(Path(self.voice["reference_audio"]).read_bytes(), self.reference_before)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        for path in (self.root / "work" / "remote-chapters").glob("*/cache.json"):
            self.assertNotIn("api_key", path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            descriptor = read_json(path)
            self.assertNotIn("last_event", descriptor)
            self.assertNotIn("id", descriptor)
            self.assertNotIn("state", descriptor)

    def test_single_speaker_worker_accepts_unchanged_uniform_paragraphs(self):
        from test_renderer_workflow import EXAMPLE
        bundle = load_bundle(EXAMPLE / "bundle")
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        config = deepcopy(self.config)
        config["render"]["paragraph_synthesis"] = True
        plan = make_render_plan(bundle, config)
        self.assertTrue(any(chunk.get("paragraph_turns") for chapter in plan["chapters"]
                            for chunk in chapter["chunks"]))
        client = self.fake_client()(config["remote_chapters"])
        request = client.request

        def single_voice_capabilities(method, path, payload=None, timeout=None):
            result = request(method, path, payload)
            if path == "v1/capabilities":
                result.update(multi_speaker=False, single_speaker_paragraph=True)
            return result

        with patch.object(client, "request", side_effect=single_voice_capabilities), \
             patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            result = _render_remote_locked(bundle, config, plan, self.store,
                                           notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(result["chapters"]), len(plan["chapters"]))
        submitted = [payload["spec"]["chapter"] for _, path, payload in client.calls if path == "v1/jobs"]
        self.assertEqual(submitted, plan["chapters"])

    def test_recovery_never_reobserves_a_chapter_already_downloaded_and_acknowledged(self):
        base = self.fake_client()

        class Client(base):
            def __init__(self, settings):
                super().__init__(settings)
                self.retired = set()
                self.failed = False

            def request(self, method, path, payload=None, timeout=None):
                if "/events?" in path and path.split("/")[2] in self.retired:
                    raise AssertionError("Reobserved an acknowledged completed run")
                result = super().request(method, path, payload, timeout=timeout)
                if path.endswith("/ack"):
                    self.retired.add(path.split("/")[2])
                if path == "v1/jobs" and payload["spec"]["chapter"]["number"] == 2 and not self.failed:
                    self.failed = True
                    self.jobs[result["id"]].update(state="failed", error="SpeechVerificationFailed: fixture", artifacts=[])
                return result

        client = Client(self.config["remote_chapters"])
        revised = deepcopy(self.plan)
        replacement = revised["chapters"][1]
        replacement["chunks"][0]["segment"]["delivery"]["direction"] = "gently"
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client), \
                patch("epub_tts_narrator.renderer.chapter_client._recover_failed_paragraph",
                      return_value=(revised, replacement)):
            result = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                           notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(result["chapters"]), 2)
        self.assertEqual(len(client.retired), 2)

    def test_missing_voice_fails_before_contacting_service(self):
        self.store.load.return_value = None
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(ChapterClientError, "cast first"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()

    def test_late_invalid_chapter_fails_before_compression_or_service_requests(self):
        self.plan["chapters"][-1]["chunks"][0]["segment"]["delivery"] = {
            "direction": "<|speaker:1|>", "evidence": []}
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory, \
                patch("epub_tts_narrator.renderer.chapter_client._voice_assets") as assets:
            with self.assertRaisesRegex(ChapterClientError, "Chapter 2 / 2.*delivery"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        assets.assert_not_called()

    def fresh_server_client(self, notes, *, event_messages=("Chapter generating", "Encoding chapter")):
        """A fake service backed by a freshly created database: its event
        counter starts at 1, with no memory of any past run."""
        owner = self
        base = owner.fake_client()

        class FreshServerClient(base):
            def __init__(self, settings):
                super().__init__(settings)
                self.event_cursors, self.idempotency_keys, self.sessions = [], [], []

            def request(self, method, path, payload=None, timeout=None):
                self.calls.append((method, path, payload))
                if path == "v1/capabilities":
                    return {"protocol": PROTOCOL, "multi_speaker": True, "renderer_fingerprint": "f" * 64,
                            "client_sessions": 1, "session_ttl_seconds": 30, "ephemeral_runs": 1}
                if path == "v1/sessions":
                    self.sessions.append(payload["id"])
                    return {"opened": True}
                if path.startswith("v1/sessions/"):
                    if method == "POST":
                        return {"renewed": True}
                    return {"revoked": True}
                if path == "v1/voices/ensure":
                    return {"missing": payload["voices"]}
                if method == "PUT":
                    self.uploads.append(payload)
                    return {"registered": True}
                if path == "v1/jobs":
                    spec = payload["spec"]
                    self.idempotency_keys.append(payload["idempotency_key"])
                    job_id = digest({"fresh": len(self.idempotency_keys)})
                    files = fixture_artifacts(spec)
                    artifacts = []
                    for name, data in files.items():
                        url = f"/{job_id}/{name}"
                        self.artifacts[url] = data
                        artifacts.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                                          "size": len(data), "url": url})
                    self.jobs[job_id] = {"id": job_id, "state": "completed", "artifacts": artifacts,
                                         "last_event": len(event_messages)}
                    return {"id": job_id, "state": "queued"}
                if "/events?" in path:
                    job_id = path.split("/")[2]
                    self.event_cursors.append(int(path.split("after=")[1]))
                    events = [{"sequence": number, "message": message}
                              for number, message in enumerate(event_messages, 1)]
                    return {"events": events, "job": self.jobs[job_id]}
                if path.endswith("/ack"):
                    notes.append("acked")
                    return {"acknowledged": True}
                raise AssertionError((method, path))

        return FreshServerClient(owner.config["remote_chapters"])

    def test_old_server_cursor_and_receipt_cannot_suppress_a_fresh_run(self):
        notes = []
        voices = {voice_id: self.voice for voice_id in self.plan["voice_ids"]}
        bindings, _ = _voice_assets(voices, self.root / "regression-work", FFMPEG)
        chapter = self.plan["chapters"][0]
        selection = _selection(self.bundle, chapter, self.config, bindings)
        folder = self.root / "work" / "remote-chapters" / digest(selection)
        folder.mkdir(parents=True)
        stale = {"format": PROTOCOL, "version": 1, "id": "c" * 64, "selection": selection,
                 "spec": {**selection, "renderer_fingerprint": "f" * 64},
                 "last_event": 16740, "state": "generating"}
        write_json(folder / "job.json", dict(stale, record_sha256=digest(stale)))
        client = self.fresh_server_client(notes)
        seen = []
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            result = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                           notify=seen.append, chapters_only=True)
        # The fresh run polls from this run's own beginning, never from the
        # dead server's 16740, so new progress stays visible.
        self.assertTrue(client.event_cursors and all(cursor == 0 for cursor in client.event_cursors))
        printed = "\n".join(seen)
        self.assertIn("Chapter generating", printed)
        self.assertIn("Encoding chapter", printed)
        self.assertIn("downloading verified chapter artifacts", printed)
        # Run-scoped idempotency: the key binds this invocation's session, so a
        # fresh database cannot collide with the old deterministic job identity.
        self.assertEqual(len(client.idempotency_keys), 2)
        self.assertTrue(all(key != digest(payload["spec"])
                            for key, (_, _, payload) in zip(client.idempotency_keys,
                                                            [call for call in client.calls if call[1] == "v1/jobs"])))
        posted = next(payload for _, path, payload in client.calls if path == "v1/jobs")
        self.assertNotEqual(posted["idempotency_key"], digest(posted["spec"]))
        self.assertEqual(notes, ["acked", "acked"])
        self.assertFalse((folder / "job.json").exists())
        self.assertTrue((folder / "cache.json").is_file())
        self.assertNotIn("last_event", read_json(folder / "cache.json"))
        self.assertEqual(len(result["chapters"]), 2)
        # A second invocation reuses the validated cache entirely offline.
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            rerun = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                          notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        self.assertEqual(len(rerun["chapters"]), 2)

    def metadata_fixture(self, *, spec=None, mutate=None, mutate_manifest=None):
        if spec is None:
            bindings = {key: "a" * 64 for key in self.plan["voice_ids"]}
            spec = _selection(self.bundle, self.plan["chapters"][0], self.config, bindings)
            spec["renderer_fingerprint"] = "f" * 64
        folder = self.root / "metadata"
        folder.mkdir(exist_ok=True)
        for path in folder.iterdir():
            path.unlink()
        artifacts = []
        for name, data in fixture_artifacts(spec, mutate, mutate_manifest).items():
            (folder / name).write_bytes(data)
            artifacts.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
        record = {"spec": spec, "selection": {"chapter": spec["chapter"]}, "artifacts": artifacts}
        return folder, record

    def legacy_fixture(self, *, state="failed", last_event=16740, corrupt_receipt=False):
        selection = _selection(self.bundle, self.plan["chapters"][0], self.config,
                               {key: "a" * 64 for key in self.plan["voice_ids"]})
        spec = {**deepcopy(selection), "renderer_fingerprint": "f" * 64}
        folder = self.root / "legacy"
        folder.mkdir(exist_ok=True)
        record = {"format": PROTOCOL, "version": 1, "id": digest({"legacy": last_event}),
                  "selection": selection, "spec": spec, "last_event": last_event, "state": state}
        if corrupt_receipt:
            (folder / "job.json").write_bytes(b"{not json")
        else:
            write_json(folder / "job.json", dict(record, record_sha256=digest(record)))
        return folder, selection, spec, record

    def test_submission_uses_a_run_scoped_idempotency_key_and_retries_in_run(self):
        selection = _selection(self.bundle, self.plan["chapters"][0], self.config,
                               {key: "a" * 64 for key in self.plan["voice_ids"]})
        spec = {**deepcopy(selection), "renderer_fingerprint": "f" * 64}
        session = "d" * 64
        client = Mock()
        client.request.side_effect = [{"id": "e" * 64, "state": "failed"},
                                      {"id": "e" * 64, "state": "queued"}]
        record, status = _submit_chapter(client, selection, spec, submission="a" * 64,
                                         session=session, notify=Mock())
        submitted, retried = client.request.call_args_list
        self.assertEqual(submitted.args[:2], ("POST", "v1/jobs"))
        payload = submitted.args[2]
        self.assertEqual(payload["idempotency_key"], digest({"spec": digest(spec), "run": session}))
        self.assertEqual(payload["submission"], "a" * 64)
        self.assertEqual(payload["session"], session)
        self.assertEqual(retried.args[:2], ("POST", f"v1/jobs/{'e' * 64}/retry"))
        self.assertEqual(retried.args[2], {"session": session, "submission": "a" * 64})
        self.assertEqual(record["id"], "e" * 64)
        self.assertEqual(record["last_event"], 0)
        self.assertEqual(status["state"], "queued")
        other_session = "1" * 64
        self.assertNotEqual(digest({"spec": digest(spec), "run": other_session}), payload["idempotency_key"])

    def test_completed_legacy_receipt_migrates_to_cache_descriptor_and_reuses_offline(self):
        folder, selection, spec, record = self.legacy_fixture(state="completed", last_event=16795)
        files = fixture_artifacts(spec)
        record["artifacts"] = [{"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                                "size": len(data), "url": f"https://objects.example/{name}"}
                               for name, data in files.items()]
        write_json(folder / "job.json", dict(record, record_sha256=digest(record)))
        for name, data in files.items():
            (folder / name).write_bytes(data)
        descriptor = _migrate_legacy_job(folder, selection)
        self.assertIsNotNone(descriptor)
        self.assertFalse((folder / "job.json").exists())
        self.assertEqual(_read_cache(folder, selection)["spec"], spec)
        self.assertTrue(_cached_artifacts_present(folder, descriptor, require_masters=False))
        self.assertTrue(_cached_artifacts_present(folder, descriptor))
        for key in ("id", "state", "last_event"):
            self.assertNotIn(key, descriptor)
        # Migration is consumed once: only the descriptor remains.
        self.assertIsNone(_migrate_legacy_job(folder, selection))

    def test_incomplete_legacy_receipts_and_leftovers_cannot_steer_a_fresh_run(self):
        voices = {voice_id: self.voice for voice_id in self.plan["voice_ids"]}
        bindings, _ = _voice_assets(voices, self.root / "legacy-assets", FFMPEG)
        chapter = self.plan["chapters"][0]
        selection = _selection(self.bundle, chapter, self.config, bindings)
        for state, corrupt in (("generating", False), ("completed", False), ("failed", True)):
            with self.subTest(state=state, corrupt_receipt=corrupt):
                config = deepcopy(self.config)
                config["render"]["work_dir"] = str(self.root / f"work-{state}-{int(corrupt)}")
                folder = Path(config["render"]["work_dir"]) / "remote-chapters" / digest(selection)
                folder.mkdir(parents=True)
                spec = {**deepcopy(selection), "renderer_fingerprint": "f" * 64}
                record = {"format": PROTOCOL, "version": 1, "id": "c" * 64, "selection": selection,
                          "spec": spec, "last_event": 16740, "state": state}
                if corrupt:
                    (folder / "job.json").write_bytes(b"{not json")
                else:
                    write_json(folder / "job.json", dict(record, record_sha256=digest(record)))
                stale = folder / "chapter.m4a"
                stale.write_bytes(b"stale past-run audio")
                client = self.fake_client()(config["remote_chapters"])
                with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
                    result = _render_remote_locked(self.bundle, config, self.plan, self.store,
                                                   notify=lambda _: None, chapters_only=True)
                submitted = [payload["spec"]["chapter"]["number"] for _, path, payload in client.calls
                             if path == "v1/jobs"]
                self.assertIn(1, submitted)
                self.assertFalse((folder / "job.json").exists())
                self.assertTrue((folder / "cache.json").is_file())
                self.assertEqual(stale.read_bytes(), fixture_artifacts(spec)["chapter.m4a"])
                self.assertEqual(len(result["chapters"]), 2)

    def test_corrupted_legacy_downloads_are_cleared_and_regenerated_not_trusted(self):
        voices = {voice_id: self.voice for voice_id in self.plan["voice_ids"]}
        bindings, _ = _voice_assets(voices, self.root / "corrupt-assets", FFMPEG)
        chapter = self.plan["chapters"][0]
        selection = _selection(self.bundle, chapter, self.config, bindings)
        folder = self.root / "work" / "remote-chapters" / digest(selection)
        folder.mkdir(parents=True)
        spec = {**deepcopy(selection), "renderer_fingerprint": "f" * 64}
        files = fixture_artifacts(spec)
        record = {"format": PROTOCOL, "version": 1, "id": "c" * 64, "selection": selection,
                  "spec": spec, "last_event": 16737, "state": "completed",
                  "artifacts": [{"name": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                                for name, data in files.items()]}
        write_json(folder / "job.json", dict(record, record_sha256=digest(record)))
        (folder / "chapter.m4a").write_bytes(b"corrupted past download")
        self.assertIsNone(_migrate_legacy_job(folder, selection))
        client = self.fake_client()(self.config["remote_chapters"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
        self.assertFalse((folder / "job.json").exists())
        self.assertTrue((folder / "cache.json").is_file())
        self.assertEqual((folder / "chapter.m4a").read_bytes(), files["chapter.m4a"])

    def test_service_without_ephemeral_run_support_fails_closed(self):
        client = self.fake_client()(self.config["remote_chapters"])
        original = client.request

        def legacy_capabilities(method, path, payload=None, timeout=None):
            result = original(method, path, payload)
            if path == "v1/capabilities":
                result.pop("ephemeral_runs")
            return result

        with patch.object(client, "request", side_effect=legacy_capabilities), \
                patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            with self.assertRaisesRegex(ChapterClientError, "update the worker"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        self.assertFalse(any(path == "v1/jobs" for _, path, _ in client.calls))

    def test_http_errors_carry_status_and_transport_failures_do_not(self):
        from urllib.error import HTTPError, URLError
        cases = ((HTTPError("https://worker.example/v1/jobs", 404, "missing", None, BytesIO(b"")), 404),
                 (URLError("unreachable"), None))
        for error, expected in cases:
            with self.subTest(expected=expected):
                opener = Mock()
                opener.open.side_effect = error
                with patch("epub_tts_narrator.renderer.chapter_client.build_opener", return_value=opener):
                    client = ChapterClient(self.config["remote_chapters"])
                    with self.assertRaises(ChapterClientError) as caught:
                        client.request("GET", "v1/jobs/" + "a" * 64)
                self.assertEqual(caught.exception.status, expected)

    def test_downloaded_metadata_requires_complete_speech_and_exact_source_mapping(self):
        def both(files, change):
            change(files["chapter.json"])
            files["sync.json"]["segments"] = deepcopy(files["chapter.json"]["segments"])
        mutations = {
            "zero frames despite spoken chunks": lambda files: files["chapter.json"].update(frames=0),
            "changed source text": lambda files: both(files, lambda value: value["segments"][0].update(source_text="Another passage")),
            "changed spoken text": lambda files: both(files, lambda value: value["segments"][0].update(spoken_text="Another passage")),
            "changed source offsets": lambda files: both(files, lambda value: value["segments"][0].update(source_start=12345)),
            "changed source ranges": lambda files: both(files, lambda value: value["segments"][0].update(source_ranges=[{"start": 7, "end": 8}])),
            "changed voice": lambda files: both(files, lambda value: value["segments"][0].update(voice_id="different-voice")),
            "reversed source order": lambda files: both(files, lambda value: value["segments"].reverse()),
            "missing speech marked silent": lambda files: both(files, lambda value: value["segments"][0].update(no_speech=True)),
            "overlapping frame ranges": lambda files: both(files, lambda value: value["segments"][1].update(chapter_start_frame=1)),
            "frames outside chapter": lambda files: both(files, lambda value: value["segments"][-1].update(chapter_end_frame=value["frames"] + 1)),
            "empty synchronization": lambda files: files.update({"sync.json": {}}),
            "different synchronization audio": lambda files: files["sync.json"].update(audio_sha256="0" * 64),
            "different synchronization timeline": lambda files: files["sync.json"].update(frames=1),
            "different audio checksum": lambda files: files["chapter.json"].update(encoded_sha256="0" * 64),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                folder, record = self.metadata_fixture(mutate=mutate)
                # Every changed file has a matching transport hash and manifest:
                # rejection must come from source/timeline consistency checks.
                with self.assertRaises(ChapterClientError):
                    _validate_metadata(folder, record)

    def test_downloaded_manifest_binds_job_inventory_and_file_bytes(self):
        mutations = {
            "different job": lambda value: value.update(spec_hash="0" * 64),
            "different identity": lambda value: value["identity"].update(spec_hash="0" * 64),
            "different source": lambda value: value["identity"]["context"].update(source_fingerprint="0" * 64),
            "different voices": lambda value: value["identity"]["context"].update(voice_assets={}),
            "missing master": lambda value: value["artifacts"].pop("chapter.flac"),
            "mismatched checksum": lambda value: value["artifacts"]["chapter.m4a"].update(sha256="0" * 64),
            "mismatched size": lambda value: value["artifacts"]["chapter.m4a"].update(bytes=1),
            "unexpected artifact": lambda value: value["artifacts"].update({"chapter.wav": {"sha256": "0" * 64, "bytes": 1}}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                folder, record = self.metadata_fixture(mutate_manifest=mutate)
                with self.assertRaises(ChapterClientError):
                    _validate_metadata(folder, record)
        folder, record = self.metadata_fixture()
        manifest = read_json(folder / "manifest.json")
        manifest["record_sha256"] = "0" * 64
        write_json(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ChapterClientError, "manifest"):
            _validate_metadata(folder, record)

    def test_legacy_chapter_only_metadata_allows_deferred_lossless_master_but_not_missing_audio(self):
        folder, record = self.metadata_fixture()
        (folder / "chapter.flac").unlink()
        _validate_metadata(folder, record)
        (folder / "chapter.m4a").unlink()
        with self.assertRaises((ChapterClientError, ValueError, FileNotFoundError)):
            _validate_metadata(folder, record)

    def test_empty_chapter_metadata_remains_valid(self):
        _, record = self.metadata_fixture()
        spec = deepcopy(record["spec"])
        spec["chapter"]["chunks"] = []
        spec["voice_bindings"] = {}
        folder, record = self.metadata_fixture(spec=spec)
        self.assertEqual(_validate_metadata(folder, record)["frames"], 0)

    def test_paragraph_source_turns_must_match_the_submitted_script(self):
        from test_renderer_workflow import EXAMPLE
        bundle = load_bundle(EXAMPLE / "bundle")
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        config = deepcopy(self.config)
        config["render"]["paragraph_synthesis"] = True
        plan = make_render_plan(bundle, config)
        spec = _selection(bundle, plan["chapters"][0], config, {key: "a" * 64 for key in plan["voice_ids"]})
        spec["renderer_fingerprint"] = "f" * 64
        self.assertTrue(any("paragraph_turns" in chunk for chunk in spec["chapter"]["chunks"]))
        folder, record = self.metadata_fixture(spec=spec)
        _validate_metadata(folder, record)
        def change_turn(files):
            timing = next(row for row in files["chapter.json"]["segments"] if "source_segments" in row)
            timing["source_segments"][0]["voice_id"] = "different-voice"
            files["sync.json"]["segments"] = deepcopy(files["chapter.json"]["segments"])
        folder, record = self.metadata_fixture(spec=spec, mutate=change_turn)
        with self.assertRaisesRegex(ChapterClientError, "source turns"):
            _validate_metadata(folder, record)

    def test_same_audio_different_transcript_has_distinct_registered_identity(self):
        bindings, assets = _voice_assets({"one": self.voice,
                                         "two": {**self.voice, "reference_text": "Another transcript."}},
                                        self.root / "work", FFMPEG)
        self.assertNotEqual(bindings["one"], bindings["two"])
        self.assertEqual(len({str(value["path"]) for value in assets.values()}), 1)

    def test_partial_local_speech_blocks_unintended_remote_regeneration(self):
        chunk = self.plan["chapters"][0]["chunks"][0]
        folder = self.root / "work" / "clips" / ("b" * 64)
        folder.mkdir(parents=True)
        audio = folder / "accepted.wav"
        shutil.copyfile(self.voice["reference_audio"], audio)
        write_json(folder / "accepted.json", {
            "request": {"kind": "speech", "chunk": chunk}, "audio_file": audio.name,
            "audio_sha256": file_hash(audio), "report": {"passed": True}})
        before = {path: path.read_bytes() for path in folder.iterdir()}
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(ChapterClientError, "Partial local clip migration"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in folder.iterdir()})

    def _render_local_revision_fixture(self):
        from epub_tts_narrator.renderer.book_context import voice_store
        from epub_tts_narrator.renderer.workflow import render
        from test_renderer_workflow import FixtureEngine, FixtureVerifier
        engine = FixtureEngine({})
        render(self.bundle, self.config, engine, FixtureVerifier(engine.fixture_transcripts),
               notify=lambda _: None, chapters_only=True)
        self.store = voice_store(self.bundle, self.config)
        self.plan = make_render_plan(self.bundle, self.config)
        self.plan["chapters"][0]["chunks"][0]["segment"]["delivery"]["direction"] = "gently"

    def test_changed_local_chapter_renders_remote_revision_and_reuses_unchanged_local_chapter(self):
        self._render_local_revision_fixture()
        retained = {path: path.read_bytes() for name in ("clips", "chapters", "completed-chapters")
                    for path in (self.root / "work" / name).rglob("*") if path.is_file()}
        client = self.fake_client()(self.config["remote_chapters"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            result = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                           notify=lambda _: None, chapters_only=True)
        submitted = [payload["spec"]["chapter"]["number"] for _, path, payload in client.calls
                     if path == "v1/jobs"]
        self.assertEqual(submitted, [1])
        self.assertEqual(len(result["chapters"]), 2)
        self.assertEqual(retained, {path: path.read_bytes() for path in retained})
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            resumed = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                            notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        self.assertEqual(result, resumed)

    def test_stale_local_checkpoint_does_not_hide_corrupted_chapter_audio(self):
        self._render_local_revision_fixture()
        checkpoint = read_json(self.root / "work" / "completed-chapters" / "chapter-0001.json")
        audio = self.root / "work" / "chapters" / checkpoint["cache_key"] / "chapter.wav"
        audio.write_bytes(audio.read_bytes()[:-2] + b"xx")
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(ChapterClientError, "PCM failed integrity"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()

    def test_stale_local_checkpoint_does_not_hide_corrupted_accepted_clip(self):
        self._render_local_revision_fixture()
        accepted_path = next((self.root / "work" / "clips").glob("*/accepted.json"))
        accepted = read_json(accepted_path)
        audio = accepted_path.parent / accepted["audio_file"]
        audio.write_bytes(audio.read_bytes()[:-2] + b"xx")
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(ChapterClientError, "Cached audio changed"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()

    def paragraph_plan(self):
        from test_renderer_workflow import EXAMPLE
        bundle = load_bundle(EXAMPLE / "bundle")
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        config = deepcopy(self.config)
        config["render"]["paragraph_synthesis"] = True
        plan = make_render_plan(bundle, config)
        paragraph = next(chunk for chunk in plan["chapters"][0]["chunks"] if "paragraph_turns" in chunk)
        return bundle, config, plan, paragraph

    def splitting_client(self, paragraph, voices, config, *, include_detail=True, request_hash=None):
        from epub_tts_narrator.renderer.chapter_protocol import FAILED_PARAGRAPH_FIELD
        from epub_tts_narrator.renderer.workflow import _paragraph_request
        request = _paragraph_request(paragraph, voices, {"engine": "fixture"}, config)
        detail = {"request_hash": request_hash or digest(request), "chunk_id": paragraph["chunk_id"]}
        base = self.fake_client()

        class SplittingClient(base):
            def __init__(self, settings):
                super().__init__(settings)
                self.specs = []

            def request(self, method, path, payload=None, timeout=None):
                if path == "v1/capabilities":
                    return {"protocol": PROTOCOL, "multi_speaker": True,
                            "renderer_fingerprint": "f" * 64, "submission_budget": True,
                            "client_sessions": 1, "session_ttl_seconds": 30, "ephemeral_runs": 1,
                            "engine": {"engine": {"engine": "fixture"}}}
                if path == "v1/sessions" or path.startswith("v1/sessions/"):
                    return super().request(method, path, payload)
                if path == "v1/jobs":
                    self.specs.append(deepcopy(payload["spec"]))
                    if len(self.specs) == 1:
                        job_id = digest(payload["spec"])
                        error = "SpeechVerificationFailed: failed after 2 attempts"
                        if include_detail:
                            error += "\n" + FAILED_PARAGRAPH_FIELD + json.dumps(detail, sort_keys=True)
                        self.jobs[job_id] = {"id": job_id, "state": "failed", "artifacts": [],
                                             "last_event": 1, "error": error}
                        return {"id": job_id, "state": "queued"}
                    return super().request(method, path, payload)
                if "/events?" in path:
                    job = self.jobs[path.split("/")[2]]
                    if job["state"] == "failed":
                        return {"events": [{"sequence": 1, "message": "Chapter failed"}], "job": job}
                    return super().request(method, path, payload)
                return super().request(method, path, payload)

        return SplittingClient(self.config["remote_chapters"])

    def test_failed_paragraph_is_recorded_and_the_split_chapter_is_resubmitted(self):
        from epub_tts_narrator.renderer.paragraph_recovery import load_paragraph_fallbacks
        bundle, config, plan, paragraph = self.paragraph_plan()
        voices = {voice_id: self.voice for voice_id in plan["voice_ids"]}
        client = self.splitting_client(paragraph, voices, config)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            result = _render_remote_locked(bundle, config, plan, self.store,
                                           notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(client.specs), 2)
        grouped, split = client.specs
        self.assertTrue(any("paragraph_turns" in chunk for chunk in grouped["chapter"]["chunks"]))
        self.assertTrue(all("paragraph_turns" not in chunk for chunk in split["chapter"]["chunks"]))
        self.assertEqual(grouped["chapter"]["id"], split["chapter"]["id"])
        self.assertEqual("".join(chunk["spoken_text"] for chunk in grouped["chapter"]["chunks"]),
                         "".join(chunk["spoken_text"] for chunk in split["chapter"]["chunks"]))
        submissions = [payload["submission"] for _, path, payload in client.calls if path == "v1/jobs"]
        self.assertEqual(submissions, [submissions[0]])
        self.assertRegex(submissions[0], r"[0-9a-f]{64}")
        self.assertEqual(load_paragraph_fallbacks(config["render"]["work_dir"]),
                         {turn["chunk_id"] for turn in paragraph["paragraph_turns"]})
        self.assertEqual(len(result["chapters"]), len(plan["chapters"]))
        self.assertTrue(all(Path(item["audio_file"]).is_file() for item in result["chapters"]))

    def test_unidentified_paragraph_failure_stops_without_replanning(self):
        bundle, config, plan, paragraph = self.paragraph_plan()
        voices = {voice_id: self.voice for voice_id in plan["voice_ids"]}
        client = self.splitting_client(paragraph, voices, config, include_detail=False)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            with self.assertRaisesRegex(ChapterClientError, "Remote chapter 1 failed"):
                _render_remote_locked(bundle, config, plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(client.specs), 1)
        self.assertEqual(list((Path(config["render"]["work_dir"]) / "paragraph-fallbacks").glob("*.json")), [])

    def test_paragraph_failure_with_a_mismatched_request_stops_without_recording(self):
        bundle, config, plan, paragraph = self.paragraph_plan()
        voices = {voice_id: self.voice for voice_id in plan["voice_ids"]}
        client = self.splitting_client(paragraph, voices, config, request_hash="0" * 64)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            with self.assertRaisesRegex(ChapterClientError, "does not match the service's"):
                _render_remote_locked(bundle, config, plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(client.specs), 1)
        self.assertEqual(list((Path(config["render"]["work_dir"]) / "paragraph-fallbacks").glob("*.json")), [])

    def test_submission_is_sent_only_when_the_service_advertises_budget_support(self):
        client = self.fake_client()(self.config["remote_chapters"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", return_value=client):
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
        payloads = [payload for _, path, payload in client.calls if path == "v1/jobs"]
        self.assertTrue(payloads)
        self.assertTrue(all("submission" not in payload for payload in payloads))

    def test_assembly_does_not_automatically_render_stale_local_chapter(self):
        from epub_tts_narrator.renderer.chapter_client import _local_chapter
        self._render_local_revision_fixture()
        voices = {key: self.store.load(key) for key in self.plan["voice_ids"]}
        with self.assertRaisesRegex(ChapterClientError, "is stale"):
            _local_chapter(self.bundle, self.plan["chapters"][0], self.config, voices)

    def test_existing_flac_is_copied_byte_for_byte_without_reencoding(self):
        original_ids, assets = _voice_assets({"voice": self.voice}, self.root / "first", FFMPEG)
        source = assets[original_ids["voice"]]["path"]
        voice = {**self.voice, "reference_audio": str(source)}
        with patch("epub_tts_narrator.renderer.chapter_client._run_audio", side_effect=AssertionError("reencoded")):
            copied_ids, copied = _voice_assets({"voice": voice}, self.root / "second", FFMPEG)
        self.assertEqual(copied_ids, original_ids)
        self.assertEqual(copied[copied_ids["voice"]]["path"].read_bytes(), source.read_bytes())

    def test_changed_samples_during_flac_conversion_fail_before_upload(self):
        wrong = self.root / "wrong.wav"
        write_tone(wrong, frequency=440)
        _, assets = _voice_assets({"voice": {**self.voice, "reference_audio": str(wrong)}},
                                 self.root / "wrong-flac", FFMPEG)
        encoded_wrong = next(iter(assets.values()))["path"]
        def corrupt_conversion(arguments):
            shutil.copyfile(encoded_wrong, arguments[-1])
        with patch("epub_tts_narrator.renderer.chapter_client._run_audio", side_effect=corrupt_conversion):
            with self.assertRaisesRegex(ChapterClientError, "original PCM samples"):
                _voice_assets({"voice": self.voice}, self.root / "damaged-conversion", FFMPEG)
        self.assertEqual(Path(self.voice["reference_audio"]).read_bytes(), self.reference_before)

    def test_real_queue_executor_download_and_final_m4b_use_only_compressed_transfers(self):
        from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
        from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
        from epub_tts_narrator.renderer.assembly import probe_audio
        from test_renderer_workflow import FixtureEngine, FixtureVerifier
        fixtures = {}
        engine = FixtureEngine(fixtures)
        executor = ChapterExecutor(self.config, engine=engine, verifier=FixtureVerifier(fixtures))

        class ObjectStore:
            """Independent compressed-artifact storage whose links outlive job rows."""
            def __init__(self):
                self.objects = {}
            def publish(self, job_id, name, path, sha256):
                self.objects[self.url({"key": f"{job_id}/{name}"})] = Path(path).read_bytes()
                return {"key": f"{job_id}/{name}"}
            def url(self, record):
                return f"https://objects.example/{record['key']}?signature=fixture"

        store = ObjectStore()
        service = ChapterJobs(self.root / "server", executor,
                              {"ffprobe": FFPROBE, "assembly_workers": 2, "max_pending_jobs": 10},
                              publisher=store)
        self.addCleanup(service.close)
        service.start()
        downloads, uploads = [], []
        class InProcessClient:
            settings = self.config["remote_chapters"]
            def __init__(self, settings):
                pass
            def websocket(self, *args):
                return None
            def request(self, method, path, payload=None, timeout=None):
                if path == "v1/capabilities":
                    return service.capabilities()
                if path == "v1/sessions":
                    service.sessions.open(payload["id"])
                    return {"opened": True, "session_ttl_seconds": service.sessions.ttl}
                if path.startswith("v1/sessions/"):
                    identifier = path.split("/")[2]
                    if method == "POST":
                        service.sessions.heartbeat(identifier)
                        return {"renewed": True}
                    return {"revoked": service.sessions.revoke(identifier)}
                if path == "v1/voices/ensure":
                    return {"missing": service.ensure_voices(payload["voices"])}
                if method == "PUT":
                    audio = base64.b64decode(payload["audio_base64"])
                    uploads.append(audio)
                    return service.register_voice(path.rsplit("/", 1)[-1], payload["manifest"], audio)
                if path == "v1/jobs":
                    return service.submit(payload["spec"], payload["idempotency_key"],
                                          payload.get("submission"), payload.get("session"))
                if "/events?after=" in path:
                    identifier = path.split("/")[2]
                    return {"events": service.events(identifier, int(path.split("after=")[1])),
                            "job": service.status(identifier)}
                if path.endswith("/ack"):
                    return service.acknowledge(path.split("/")[2], payload["session"])
                if method == "GET" and path.startswith("v1/jobs/"):
                    return service.status(path.rsplit("/", 1)[-1])
                raise AssertionError((method, path))
            def download(self, artifact, target):
                if target.exists():
                    if file_hash(target) != artifact["sha256"]:
                        raise ChapterClientError("fixture cached artifact differs")
                    return target
                downloads.append(artifact["name"])
                target.write_bytes(store.objects[artifact["url"]])
                if file_hash(target) != artifact["sha256"]:
                    raise AssertionError("transfer changed bytes")
                return target
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", InProcessClient):
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
            self.assertIn("chapter.flac", downloads)
            self.assertIn("chapter.m4a", downloads)
            calls_before_assembly = len(engine.speech_calls)
            # Cached durable artifact URLs suffice even after the job API stops.
            with patch.object(InProcessClient, "request", side_effect=AssertionError("service offline")):
                result = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                               notify=lambda _: None, chapters_only=False)
        self.assertEqual(len(engine.speech_calls), calls_before_assembly)
        self.assertTrue(all(audio.startswith(b"fLaC") for audio in uploads))
        self.assertTrue({"chapter.m4a", "chapter.flac"}.issubset(downloads))
        self.assertFalse(any(name.endswith(".wav") for name in downloads))
        self.assertEqual(result["status"], "complete")
        book = self.root / "output" / "book.m4b"
        self.assertTrue(book.is_file())
        self.assertEqual(len(probe_audio(book, ffprobe=FFPROBE)["chapters"]), 2)
        sync = read_json(self.root / "output" / "sync.json")
        self.assertEqual(len(sync["chapters"]), 2)
        self.assertEqual(list((self.root / "work" / "remote-chapters").rglob("*.wav")), [])
        self.assertEqual(list(self.root.glob(".remote-audiobook-*")), [])
        self.assertEqual(engine.voice_calls, [])

        # A revised script updates the same complete audiobook, retaining the
        # unchanged remote chapter and the server's accepted unchanged clips.
        previous = read_json(self.root / "output" / "render.json")
        self.config["_publication_expected"] = digest(previous)
        unchanged_chapter = file_hash(self.root / "output" / "chapters" / "chapter-0002.m4a")
        self.plan["chapters"][0]["chunks"][0]["segment"]["delivery"]["direction"] = "gently"
        self.bundle["bundle_hash"] = digest({"old": self.bundle["bundle_hash"], "revision": 1})
        self.plan["bundle_hash"] = self.bundle["bundle_hash"]
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", InProcessClient):
            revised = _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                            notify=lambda _: None, chapters_only=False)
        self.assertEqual(len(engine.speech_calls), calls_before_assembly + 1)
        self.assertNotEqual(revised["chapter_identities"]["chapter-0001"],
                            previous["chapter_identities"]["chapter-0001"])
        self.assertEqual(revised["chapter_identities"]["chapter-0002"],
                         previous["chapter_identities"]["chapter-0002"])
        self.assertEqual(revised, read_json(self.root / "output" / "render.json"))
        self.assertEqual(revised["bundle_hash"], self.bundle["bundle_hash"])
        self.assertEqual(unchanged_chapter, file_hash(self.root / "output" / "chapters" / "chapter-0002.m4a"))
        self.assertEqual(len(probe_audio(book, ffprobe=FFPROBE)["chapters"]), 2)
        self.assertEqual(list(self.root.glob(".output.previous-*")), [])

        # A caller that observed an earlier publication must not replace a newer
        # result, even when every remote chapter is already downloaded.
        from epub_tts_narrator.renderer.workflow import RenderBlocked
        preserved = {path: path.read_bytes() for path in (self.root / "output").rglob("*") if path.is_file()}
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(RenderBlocked, "changed during rendering"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=False)
        factory.assert_not_called()
        self.assertEqual(preserved, {path: path.read_bytes() for path in preserved})
        self.assertEqual(list(self.root.glob(".remote-audiobook-*")), [])


class ChapterCLITests(unittest.TestCase):
    def test_remote_render_routes_before_any_model_setup_or_verifier_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "global.toml"
            config.write_text('[remote_chapters]\nurl="https://worker.example"\n')
            export_bundle(prepare_document(valid_document()), root / "book")
            with (patch("epub_tts_narrator.renderer.chapter_client.render_remote", return_value={"chapters": []}) as remote,
                  patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
                  patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier,
                  redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error):
                status = main(["render", str(root / "book"), "--config", str(config), "--chapters-only"])
            self.assertEqual(status, 0, error.getvalue())
            remote.assert_called_once()
            setup.assert_not_called()
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
