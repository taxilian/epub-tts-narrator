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
    _read_job, _render_remote_locked, _save_job, _selection, _submit_chapter, _voice_assets, _wait_jobs,
    _validate_metadata, assemble_remote,
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
            "precision": "paragraph" if any("source_segments" in row for row in segments) else "chunk",
            "audio_file": metadata["audio_file"], "audio_sha256": audio_hash, "segments": deepcopy(segments)}
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
            def websocket(self, *args):
                return None
            def request(self, method, path, payload=None):
                self.calls.append((method, path, payload))
                if path == "v1/capabilities":
                    return {"protocol": PROTOCOL, "multi_speaker": True, "renderer_fingerprint": "f" * 64}
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
        self.assertEqual(Path(self.voice["reference_audio"]).read_bytes(), self.reference_before)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()
        for path in (self.root / "work" / "remote-chapters").glob("*/job.json"):
            self.assertNotIn("api_key", path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_voice_fails_before_contacting_service(self):
        self.store.load.return_value = None
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as factory:
            with self.assertRaisesRegex(ChapterClientError, "cast first"):
                _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                      notify=lambda _: None, chapters_only=True)
        factory.assert_not_called()

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

    def resume_fixture(self, *, changed=True):
        selection = _selection(self.bundle, self.plan["chapters"][0], self.config,
                               {key: "a" * 64 for key in self.plan["voice_ids"]})
        old = {**deepcopy(selection), "renderer_fingerprint": "e" * 64}
        current = {**deepcopy(selection), "renderer_fingerprint": ("f" if changed else "e") * 64}
        folder = self.root / "resume"
        folder.mkdir()
        saved = {"format": PROTOCOL, "version": 1, "id": digest(old), "selection": selection,
                 "spec": old, "last_event": 7, "state": "failed"}
        _save_job(folder, saved)
        return folder, selection, saved, current

    def test_unchanged_renderer_retries_same_job_without_archiving(self):
        folder, selection, saved, current = self.resume_fixture(changed=False)
        client = Mock()
        client.request.side_effect = [{"id": saved["id"], "state": "failed"},
                                      {"id": saved["id"], "state": "queued"}]
        record, status = _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
        self.assertEqual(record["id"], saved["id"])
        self.assertEqual(status["state"], "queued")
        self.assertEqual(client.request.call_args_list[0].args,
                         ("POST", "v1/jobs", {"spec": saved["spec"], "idempotency_key": digest(saved["spec"])}))
        self.assertEqual(client.request.call_args_list[1].args[:2], ("POST", f"v1/jobs/{saved['id']}/retry"))
        self.assertFalse((folder / "history").exists())

    def test_changed_failed_renderer_preserves_history_and_voice_selection(self):
        folder, selection, saved, current = self.resume_fixture()
        (folder / "chapter.m4a").write_bytes(b"previous partial download")
        client = Mock()
        client.request.side_effect = [
            {"id": saved["id"], "state": "failed", "spec_hash": digest(saved["spec"])},
            {"id": digest(current), "state": "queued", "spec_hash": digest(current)},
        ]
        record, _ = _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
        self.assertEqual(record["spec"], current)
        self.assertEqual(record["selection"], selection)
        self.assertEqual(record["spec"]["voice_bindings"], saved["spec"]["voice_bindings"])
        history = folder / record["previous_job"]["archive"]
        self.assertEqual((history / "chapter.m4a").read_bytes(), b"previous partial download")
        archived = _read_job(history, selection)
        self.assertEqual(archived["id"], saved["id"])
        self.assertEqual(archived["last_event"], 7)
        self.assertFalse((folder / "chapter.m4a").exists())
        self.assertEqual(_read_job(folder, selection)["id"], record["id"])
        self.assertEqual(len(client.request.call_args_list), 2)

    def test_completed_old_renderer_downloads_without_resubmission(self):
        folder, selection, saved, current = self.resume_fixture()
        files = fixture_artifacts(saved["spec"])
        artifacts = [{"name": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                      "url": f"https://objects.example/{name}"} for name, data in files.items()]
        client = Mock()
        client.request.return_value = {"id": saved["id"], "state": "completed",
                                       "spec_hash": digest(saved["spec"]), "artifacts": artifacts}
        client.download.side_effect = lambda artifact, target: target.write_bytes(files[artifact["name"]])
        record, _ = _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
        client.request.assert_called_once_with("GET", f"v1/jobs/{saved['id']}")
        self.assertEqual(record["spec"], saved["spec"])
        self.assertTrue(_cached_artifacts_present(folder, record, require_masters=False))
        self.assertFalse((folder / "chapter.flac").exists())
        self.assertFalse((folder / "history").exists())

    def test_unconfirmed_old_job_never_replaces_receipt_or_partial_downloads(self):
        folder, selection, saved, current = self.resume_fixture()
        partial = folder / "chapter.m4a"
        partial.write_bytes(b"retain old audio")
        receipt = (folder / "job.json").read_bytes()
        for response in [ChapterClientError("HTTP 400"),
                         {"id": saved["id"], "state": "unknown", "spec_hash": digest(saved["spec"])},
                         {"id": saved["id"], "state": "failed", "spec_hash": "0" * 64}]:
            with self.subTest(response=response):
                client = Mock()
                client.request.side_effect = [response]
                with self.assertRaises(ChapterClientError):
                    _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
                client.request.assert_called_once_with("GET", f"v1/jobs/{saved['id']}")
                self.assertEqual((folder / "job.json").read_bytes(), receipt)
                self.assertEqual(partial.read_bytes(), b"retain old audio")
                self.assertFalse((folder / "history").exists())

    def test_lost_replacement_response_reuses_persisted_spec_before_archiving(self):
        folder, selection, saved, current = self.resume_fixture()
        partial = folder / "chapter.m4a"
        partial.write_bytes(b"retain until acknowledgement")
        client = Mock()
        client.request.side_effect = [
            {"id": saved["id"], "state": "failed", "spec_hash": digest(saved["spec"])},
            ChapterClientError("lost response"),
        ]
        with self.assertRaisesRegex(ChapterClientError, "lost response"):
            _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
        pending = _read_job(folder, selection)
        self.assertEqual(pending["id"], saved["id"])
        self.assertEqual(pending["replacement_spec"], current)
        self.assertEqual(partial.read_bytes(), b"retain until acknowledgement")
        self.assertFalse((folder / "history").exists())
        first_payload = client.request.call_args.args[2]
        client.request.reset_mock()
        client.request.side_effect = [{"id": digest(current), "state": "queued", "spec_hash": digest(current)}]
        record, _ = _submit_chapter(client, folder, selection, pending, current, Mock(), include_masters=False)
        client.request.assert_called_once_with("POST", "v1/jobs", first_payload)
        self.assertEqual((folder / record["previous_job"]["archive"] / "chapter.m4a").read_bytes(),
                         b"retain until acknowledgement")

    def test_active_old_renderer_is_observed_and_migrated_only_after_failure(self):
        folder, selection, saved, current = self.resume_fixture()
        client = Mock()
        old_status = {"id": saved["id"], "spec_hash": digest(saved["spec"])}
        client.request.side_effect = [{**old_status, "state": "generating"},
                                      {**old_status, "state": "failed"},
                                      {"id": digest(current), "state": "queued", "spec_hash": digest(current)}]
        with patch("epub_tts_narrator.renderer.chapter_client._wait_jobs",
                   side_effect=ChapterClientError("old worker failed")) as wait:
            record, _ = _submit_chapter(client, folder, selection, saved, current, Mock(), include_masters=False)
        self.assertEqual(wait.call_args.args[1], {saved["id"]: (folder, saved)})
        self.assertEqual(record["id"], digest(current))
        self.assertEqual([call.args[:2] for call in client.request.call_args_list],
                         [("GET", f"v1/jobs/{saved['id']}"), ("GET", f"v1/jobs/{saved['id']}"), ("POST", "v1/jobs")])

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

    def test_chapter_only_metadata_allows_deferred_lossless_master_but_not_missing_audio(self):
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
        service = ChapterJobs(self.root / "server", executor,
                              {"ffprobe": FFPROBE, "assembly_workers": 2, "max_pending_jobs": 10})
        self.addCleanup(service.close)
        service.start()
        downloads, uploads = [], []
        class InProcessClient:
            settings = self.config["remote_chapters"]
            def __init__(self, settings):
                pass
            def websocket(self, *args):
                return None
            def request(self, method, path, payload=None):
                if path == "v1/capabilities":
                    return service.capabilities()
                if path == "v1/voices/ensure":
                    return {"missing": service.ensure_voices(payload["voices"])}
                if method == "PUT":
                    audio = base64.b64decode(payload["audio_base64"])
                    uploads.append(audio)
                    return service.register_voice(path.rsplit("/", 1)[-1], payload["manifest"], audio)
                if path == "v1/jobs":
                    return service.submit(payload["spec"], payload["idempotency_key"])
                if "/events?after=" in path:
                    identifier = path.split("/")[2]
                    return {"events": service.events(identifier, int(path.split("after=")[1])),
                            "job": service.status(identifier)}
                if method == "GET" and path.startswith("v1/jobs/"):
                    return service.status(path.rsplit("/", 1)[-1])
                raise AssertionError((method, path))
            def download(self, artifact, target):
                if target.exists():
                    if file_hash(target) != artifact["sha256"]:
                        raise ChapterClientError("fixture cached artifact differs")
                    return target
                downloads.append(artifact["name"])
                identifier = artifact["url"].split("/")[3]
                source = service.artifact(identifier, artifact["name"])
                shutil.copyfile(source, target)
                if file_hash(target) != artifact["sha256"]:
                    raise AssertionError("transfer changed bytes")
                return target
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", InProcessClient):
            _render_remote_locked(self.bundle, self.config, self.plan, self.store,
                                  notify=lambda _: None, chapters_only=True)
            self.assertNotIn("chapter.flac", downloads)
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
        self.assertNotEqual(revised["remote_jobs"]["chapter-0001"], previous["remote_jobs"]["chapter-0001"])
        self.assertEqual(revised["remote_jobs"]["chapter-0002"], previous["remote_jobs"]["chapter-0002"])
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
