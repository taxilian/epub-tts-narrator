"""Run-scoped remote chapter rendering with compressed references and downloads.

The remote service owns paragraph synthesis, verification, and chapter assembly.
The client owns source provenance, voice selections, downloaded checkpoints, and
final book packaging. Each invocation opens one ephemeral connected session and
keeps it live with a heartbeat; work without a live session pauses losslessly
instead of executing unattended.

Scheduling state belongs only to the current invocation: jobs are submitted
under run-scoped idempotency keys, progress is followed from a fresh event
cursor, and nothing about the run's queue position is persisted. Resumability
comes solely from validated chapter caches and the service's speech cache: a
rerun starts a fresh run and reuses verified downloads. Once every artifact of
a chapter is downloaded and hash-checked, the client records a durable cache
descriptor and acknowledges the job so the server can discard that run's rows.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from .bundle import chunk_speakers
from .common import USER_AGENT, RejectRedirects, digest, file_hash, pcm_sha256, read_json, regular_file, write_json, verify_flac_pcm
from .chapter_protocol import (
    PROTOCOL, ARTIFACT_NAMES, FAILED_PARAGRAPH_FIELD, SAFE_RENDER_KEYS, validate_spec,
)
from .chapter_records import (
    CHAPTER_FORMAT, MANIFEST_FORMAT, VERIFICATION_FORMAT, audio_pointers,
    expected_artifact_names, signed_record, sync_record,
)

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK = re.compile(r"[0-9a-f]{32,64}\Z")
_JOB = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ChapterClientError(RuntimeError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        # Set only for definite HTTP answers; its absence means transport
        # uncertainty, which never authorizes fallback behavior.
        self.status = status


class RemoteChapterFailed(ChapterClientError):
    """A remote job reached a terminal failed state on this client's watch."""

    def __init__(self, message, *, job_id, record, paragraph=None, status=None):
        super().__init__(message, status=status)
        super().__init__(message)
        self.job_id = job_id
        self.record = record
        self.paragraph = paragraph


def _failed_paragraph(error):
    """Parse the service's hash-only failed-paragraph identity from an error."""
    if not isinstance(error, str):
        return None
    marker = FAILED_PARAGRAPH_FIELD
    if marker not in error:
        return None
    try:
        detail = json.loads(error.rsplit(marker, 1)[1].splitlines()[0].strip())
    except (ValueError, IndexError):
        return None
    if (not isinstance(detail, dict) or set(detail) != {"request_hash", "chunk_id"}
            or not isinstance(detail["request_hash"], str) or not _HASH.fullmatch(detail["request_hash"])
            or not isinstance(detail["chunk_id"], str) or not _CHUNK.fullmatch(detail["chunk_id"])):
        return None
    return detail


def _redirect_refused(request, fp, code, headers):
    return ChapterClientError("Chapter service redirects are disabled; configure its direct URL")


def _safe_id(value):
    if not isinstance(value, str) or not _JOB.fullmatch(value):
        raise ChapterClientError("Chapter service returned an invalid job ID")
    return value


def _regular_directory(path):
    path = Path(path)
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ChapterClientError("Chapter cache directories must not use symlinks")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_audio(arguments):
    result = subprocess.run(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=3600)
    if result.returncode:
        raise ChapterClientError("Local lossless audio conversion failed")


def _reference_pcm_hash(path):
    return pcm_sha256(path, format_error=ChapterClientError("Reference WAV must use the saved mono 16-bit cloning format"))


def _verify_flac_pcm(path, expected, ffmpeg):
    error = ChapterClientError("Compressed voice transfer copy does not preserve the original PCM samples")
    verify_flac_pcm(path, expected, ffmpeg, decode_error=error, mismatch_error=error)


class ChapterClient:
    """Small authenticated transport; credentials never enter saved cache records."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings["url"].rstrip("/") + "/"
        self.headers = {"Authorization": f"Bearer {settings['api_key']}"} if settings["api_key"] else {}
        self.opener = build_opener(ProxyHandler({}), RejectRedirects(_redirect_refused))

    def _url(self, path):
        return self.base_url + path.lstrip("/")

    def request(self, method, path, payload=None, *, timeout=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        headers = {**self.headers, "Accept": "application/json", "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout or self.settings["timeout_seconds"]) as response:
                body = response.read(100 * 1024 * 1024 + 1)
            if len(body) > 100 * 1024 * 1024:
                raise ChapterClientError("Chapter service JSON response is too large")
            result = json.loads(body, parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
        except HTTPError as exc:
            raise ChapterClientError(f"Chapter service returned HTTP {exc.code}", status=exc.code) from None
        except (URLError, TimeoutError, ConnectionError, OSError):
            raise ChapterClientError("Chapter service connection failed; rerun the same command to start a "
                                     "fresh run (verified chapter caches are reused automatically)") from None
        except (ValueError, UnicodeError):
            raise ChapterClientError("Chapter service returned invalid JSON") from None
        if not isinstance(result, dict):
            raise ChapterClientError("Chapter service response must be an object")
        return result

    def websocket(self, job_id, after):
        if not self.settings.get("websocket", True):
            return None
        try:
            from websockets.sync.client import connect
        except ImportError:
            return None
        parts = urlsplit(self._url(f"v1/jobs/{_safe_id(job_id)}/events?after={after}"))
        url = urlunsplit(("wss" if parts.scheme == "https" else "ws", *parts[1:]))
        try:
            return connect(url, additional_headers=self.headers,
                           open_timeout=min(10, self.settings["timeout_seconds"]),
                           close_timeout=1, max_size=2 * 1024 * 1024, proxy=None)
        except (OSError, TimeoutError, ValueError):
            return None
        except Exception:
            # Handshake/version/proxy errors must not disable durable HTTP status.
            return None

    def download(self, artifact, target):
        name, expected = artifact.get("name"), artifact.get("sha256")
        size = artifact.get("size")
        if (not isinstance(name, str) or name != Path(name).name or name in {"", ".", ".."}
                or name not in ARTIFACT_NAMES
                or not isinstance(expected, str) or not _HASH.fullmatch(expected)
                or type(size) is not int or not 0 <= size <= 32 * 1024**3):
            raise ChapterClientError("Invalid compressed chapter artifact metadata")
        target = Path(target)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise ChapterClientError("Chapter download destination must be a regular file")
            if target.stat().st_size == size and file_hash(target) == expected:
                return target
            raise ChapterClientError(f"Saved chapter artifact changed: {target}. Remove it explicitly to download again")
        supplied = artifact.get("url")
        if not isinstance(supplied, str) or not supplied:
            raise ChapterClientError("Chapter artifact has no download URL")
        url = urljoin(self.base_url, supplied)
        parsed, origin = urlsplit(url), urlsplit(self.base_url)
        same_origin = (parsed.scheme, parsed.hostname, parsed.port) == (origin.scheme, origin.hostname, origin.port)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.fragment
                or (not same_origin and parsed.scheme != "https")
                or any(ord(char) < 32 or char.isspace() for char in url)):
            raise ChapterClientError("Chapter artifact URL must use HTTPS or the configured service origin")
        headers = self.headers if same_origin else {}
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
        received, checksum = 0, hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as stream:
                try:
                    with self.opener.open(Request(url, headers=headers), timeout=self.settings["timeout_seconds"]) as response:
                        while data := response.read(1024 * 1024):
                            received += len(data)
                            if received > size:
                                raise ChapterClientError("Chapter download exceeds its declared size")
                            checksum.update(data)
                            stream.write(data)
                except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
                    raise ChapterClientError("Chapter artifact download failed; rerun to obtain a fresh link") from None
                stream.flush()
                os.fsync(stream.fileno())
            if received != size or checksum.hexdigest() != expected:
                raise ChapterClientError("Chapter download failed its size or SHA-256 integrity check")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target


class ClientHeartbeat:
    """Keep this invocation's session live on one dedicated heartbeat thread.

    The heartbeat spans every submission, wait, and download of the rendering
    run. It is explicitly bounded (short timeout), stops before the caller
    revokes the session, and the server never lets a heartbeat recreate an
    expired or revoked session, so no late beat can outlive a delete. The
    session identifier is a hash and never enters logs or notifications.
    """

    def __init__(self, client, session_id, interval, *, timeout=5.0):
        self.client = client
        self.session_id = session_id
        self.interval = interval
        self.timeout = timeout
        self._stop = threading.Event()
        self._thread = None
        self.failure = None

    def start(self):
        self._thread = threading.Thread(target=self._beat, name="chapter-session-heartbeat", daemon=True)
        self._thread.start()

    def _beat(self):
        while not self._stop.wait(self.interval):
            try:
                self.client.request("POST", f"v1/sessions/{self.session_id}/heartbeat", timeout=self.timeout)
            except ChapterClientError as exc:
                self.failure = str(exc)
                return

    def check(self):
        """Raise promptly in the foreground once the heartbeat thread died."""
        if self._thread is not None and not self._thread.is_alive() and self.failure is not None:
            raise ChapterClientError(f"The rendering session heartbeat failed: {self.failure}")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 5)


def _voice_assets(voices, work, ffmpeg):
    """Create lossless transfer copies; never mutate original cloning references."""
    directory = _regular_directory(Path(work) / "remote-voice-assets")
    bindings, assets = {}, {}
    for voice_id, voice in sorted(voices.items()):
        source = regular_file(voice["reference_audio"])
        original_hash = file_hash(source)
        with source.open("rb") as stream:
            already_flac = stream.read(4) == b"fLaC"
        expected_pcm = None if already_flac else _reference_pcm_hash(source)
        conversion_key = digest({"audio": original_hash, "encoding": "flac-v1"})
        target, record = directory / f"{conversion_key}.flac", directory / f"{conversion_key}.json"
        if record.exists() and target.exists():
            saved = read_json(regular_file(record))
            if file_hash(regular_file(target)) != saved.get("sha256"):
                raise ChapterClientError("Saved compressed voice transfer copy changed")
            if already_flac and file_hash(target) != original_hash:
                raise ChapterClientError("Saved FLAC voice transfer copy differs from the original")
            if not already_flac and saved.get("pcm_sha256") != expected_pcm:
                _verify_flac_pcm(target, expected_pcm, ffmpeg)
                write_json(record, {**saved, "pcm_sha256": expected_pcm})
        else:
            if target.exists() or target.is_symlink():
                raise ChapterClientError("Incomplete compressed voice cache; remove the orphaned transfer copy")
            with tempfile.TemporaryDirectory(prefix=".voice-flac-", dir=directory) as temporary:
                converted = Path(temporary) / "reference.flac"
                if already_flac:
                    shutil.copyfile(source, converted)
                else:
                    _run_audio([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                                "-protocol_whitelist", "file,pipe", "-i", str(source),
                                "-map", "0:a:0", "-map_metadata", "-1",
                                "-c:a", "flac", "-compression_level", "8", str(converted)])
                    _verify_flac_pcm(converted, expected_pcm, ffmpeg)
                if file_hash(source) != original_hash:
                    raise ChapterClientError("Voice reference changed during compression")
                target.hardlink_to(converted)
            write_json(record, {"sha256": file_hash(target), "source_sha256": original_hash,
                                "pcm_sha256": expected_pcm})
        manifest = {"version": 1, "audio_sha256": file_hash(target),
                    "reference_text": voice["reference_text"]}
        asset_id = digest(manifest)
        bindings[voice_id] = asset_id
        assets[asset_id] = {"manifest": manifest, "path": target}
    return bindings, assets


def _register_voices(client, assets, notify):
    if not assets:
        return
    response = client.request("POST", "v1/voices/ensure", {"voices": sorted(assets)})
    missing = response.get("missing")
    if (not isinstance(missing, list) or len(set(missing)) != len(missing)
            or any(item not in assets for item in missing)):
        raise ChapterClientError("Chapter service returned invalid missing voice IDs")
    for index, asset_id in enumerate(missing, 1):
        asset = assets[asset_id]
        data = regular_file(asset["path"]).read_bytes()
        if not data.startswith(b"fLaC"):
            raise ChapterClientError("Voice uploads must use lossless FLAC")
        if hashlib.sha256(data).hexdigest() != asset["manifest"]["audio_sha256"]:
            raise ChapterClientError("Voice transfer copy changed before upload")
        notify(f"Registering compressed voice {index}/{len(missing)}")
        client.request("PUT", f"v1/voices/{asset_id}",
                       {"manifest": asset["manifest"], "audio_base64": base64.b64encode(data).decode("ascii")})


def _safe_render(settings):
    return {key: deepcopy(settings[key]) for key in SAFE_RENDER_KEYS if key in settings}


def _selection(bundle, chapter, config, bindings):
    used = {speaker["voice_id"] for chunk in chapter["chunks"] if not chunk["silent_control"]
            for speaker in chunk_speakers(chunk)}
    return {"version": 1, "book": bundle["source"]["book"],
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "chapter": chapter, "total_chapters": len(bundle["chapters"]),
            "voice_bindings": {key: bindings[key] for key in sorted(used)},
            "render": _safe_render(config["render"])}


def validate_remote_plan(bundle, config, plan):
    """Validate every chapter before starting compute or sending any assets.

    Real voice hashes and the worker fingerprint are bound at submission. Their
    placeholders let us check the complete source/transport contract locally,
    without compressing voices or making any request to the chapter service.
    """
    bindings = {voice_id: "0" * 64 for voice_id in plan["voice_ids"]}
    for chapter in plan["chapters"]:
        try:
            spec = _selection(bundle, chapter, config, bindings)
            spec["renderer_fingerprint"] = "0" * 64
            validate_spec(spec)
        except ValueError as exc:
            raise ChapterClientError(
                f"Chapter {chapter['number']} / {len(bundle['chapters'])} "
                f"cannot be rendered remotely: {exc}") from exc


CACHE_FORMAT = "epub-tts-narrator-remote-chapter-cache"


def _cache_descriptor(selection, spec, artifacts):
    """Build the durable validated-artifact descriptor for one chapter.

    It binds the exact source/voice/render selection and every artifact hash,
    so completed downloads stay verifiable and assemblable after the service
    stops. It intentionally carries no job id, event cursor, queue state,
    session, or scheduling permission of any kind.
    """
    record = {"format": CACHE_FORMAT, "version": 1, "selection": deepcopy(selection),
              "spec": deepcopy(spec),
              "artifacts": [{key: item[key] for key in ("name", "sha256", "size", "url") if key in item}
                            for item in artifacts]}
    return dict(record, record_sha256=digest(record))


def _write_cache(folder, descriptor):
    write_json(Path(folder) / "cache.json", descriptor)


def _clear_legacy_receipt_history(folder):
    for receipt in Path(folder).glob("history/*/job.json"):
        _regular_directory(receipt.parent)
        regular_file(receipt).unlink()


def _read_cache(folder, selection):
    """Load the validated-artifact cache descriptor for this exact selection."""
    _clear_legacy_receipt_history(folder)
    path = Path(folder) / "cache.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = read_json(regular_file(path))
    if not isinstance(record, dict):
        raise ChapterClientError("Saved remote chapter cache descriptor must be an object")
    checksum = record.pop("record_sha256", None)
    spec = record.get("spec")
    artifacts = record.get("artifacts")
    names = [item.get("name") for item in artifacts if isinstance(item, dict)] if isinstance(artifacts, list) else []
    if (checksum != signed_record(record)
            or record.get("format") != CACHE_FORMAT or record.get("version") != 1
            or record.get("selection") != selection or not isinstance(spec, dict)
            or {key: value for key, value in spec.items() if key != "renderer_fingerprint"} != selection
            or not artifacts or len(names) != len(artifacts) or len(set(names)) != len(names)
            or any(name not in ARTIFACT_NAMES for name in names)
            or any(not isinstance(item.get("sha256"), str) or not _HASH.fullmatch(item["sha256"])
                   or type(item.get("size")) is not int or item["size"] < 0
                   for item in artifacts)):
        raise ChapterClientError("Saved remote chapter cache descriptor failed integrity checks")
    return record


def _read_job(folder, selection):
    """Read a legacy run receipt; it can only vouch for downloaded artifacts."""
    path = Path(folder) / "job.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = read_json(regular_file(path))
    if not isinstance(record, dict):
        raise ChapterClientError("Invalid saved remote chapter job")
    checksum = record.pop("record_sha256", None)
    if (checksum != signed_record(record) or record.get("selection") != selection
            or record.get("format") != PROTOCOL or record.get("version") != 1):
        raise ChapterClientError("Saved remote chapter job failed integrity checks")
    _safe_id(record.get("id"))
    return record


def _migrate_legacy_job(folder, selection):
    """Consume a legacy run receipt only to retain its verified downloads.

    A past run's job.json may vouch for already-downloaded artifacts; they are
    validated and re-recorded as a plain cache descriptor. It can never steer a
    new run, grant resumption, or supply its event cursor. Receipts and leftover
    files that fail validation are ignored here; unvalidated leftovers are
    cleared before this run's fresh submission instead.
    """
    path = Path(folder) / "job.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        record = _read_job(folder, selection)
        if record is None or not record.get("artifacts"):
            return None
        if not _cached_artifacts_present(folder, record, require_masters=False):
            return None
    except (ChapterClientError, KeyError, TypeError, ValueError, OSError):
        return None
    descriptor = _cache_descriptor(record["selection"], record["spec"], record["artifacts"])
    _write_cache(folder, descriptor)
    regular_file(path).unlink()
    return descriptor


def _clear_stale_chapter_folder(folder):
    """Remove unvalidated past-run leftovers before a fresh submission.

    Files without a validated descriptor are not accepted outputs; keeping
    them would either obstruct this run's downloads or silently mix old
    bytes. Accepted local chapter audio lives elsewhere and is never touched.
    """
    for name in ("job.json", *ARTIFACT_NAMES):
        path = Path(folder) / name
        if path.exists() or path.is_symlink():
            regular_file(path).unlink()


def _submit_chapter(client, selection, spec, *, submission, session, notify):
    """Submit this run's own chapter job under a run-scoped idempotency key.

    The key binds the chapter specification to this invocation's session, so a
    fresh run never collides with a past server job — even on a recreated
    database with the same deterministic identity — while retries within the
    run stay idempotent. The returned record lives only in memory; scheduling
    state is never persisted client-side.
    """
    validate_spec(spec)
    payload = {"spec": spec, "idempotency_key": digest({"spec": digest(spec), "run": session})}
    if submission is not None:
        payload["submission"] = submission
    payload["session"] = session
    status = client.request("POST", "v1/jobs", payload)
    job_id = _safe_id(status.get("id"))
    record = {"format": PROTOCOL, "version": 1, "id": job_id, "selection": deepcopy(selection),
              "spec": deepcopy(spec), "last_event": 0, "state": status.get("state")}
    if status.get("state") in {"failed", "cancelled"}:
        retry = {"session": session}
        if submission is not None:
            retry["submission"] = submission
        status = client.request("POST", f"v1/jobs/{job_id}/retry", retry)
    record["state"] = status.get("state")
    return record, status


def _event(record, event, notify):
    if not isinstance(event, dict):
        raise ChapterClientError("Invalid chapter progress event")
    sequence = event.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ChapterClientError("Invalid chapter progress sequence")
    if sequence <= record.get("last_event", 0):
        return
    record["last_event"] = sequence
    message = event.get("message")
    if isinstance(message, str):
        safe_message = "".join(char for char in message if ord(char) >= 32 or char == "\t")
        chapter = record["selection"]["chapter"]
        notify(f"Chapter {chapter['number']} / {record['selection']['total_chapters']}: {safe_message}")


def _completed_download(client, folder, record, status, *, include_masters=True, notify=None, session=None):
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ChapterClientError("Completed chapter job has no artifacts")
    names = [artifact.get("name") for artifact in artifacts if isinstance(artifact, dict)]
    if (len(names) != len(artifacts) or len(set(names)) != len(names)
            or not {"chapter.json", "sync.json", "verification.json", "manifest.json"}.issubset(names)):
        raise ChapterClientError("Completed chapter job lacks its timing/verification metadata")
    folder = _regular_directory(folder)
    chapter = record["selection"]["chapter"]
    if notify is not None:
        notify(f"Chapter {chapter['number']} / {record['selection']['total_chapters']}: "
               "downloading verified chapter artifacts")
    for artifact in artifacts:
        if artifact["name"] == "chapter.flac" and not include_masters:
            continue
        client.download(artifact, folder / artifact["name"])
    _validate_metadata(folder, record, artifacts=artifacts)
    # The durable cache descriptor binds these validated files to their exact
    # selection and hashes. Keep expiring signed links privately in it, so
    # masters can be fetched directly from durable storage after GPU shutdown.
    _write_cache(folder, _cache_descriptor(record["selection"], record["spec"], artifacts))
    record["artifacts"] = [{key: artifact[key] for key in ("name", "sha256", "size", "url")} for artifact in artifacts]
    record["state"] = "completed"
    legacy = folder / "job.json"
    if legacy.exists() or legacy.is_symlink():
        regular_file(legacy).unlink()
    if notify is not None:
        notify(f"Chapter {chapter['number']} / {record['selection']['total_chapters']}: verified download complete")
    if session is not None:
        try:
            client.request("POST", f"v1/jobs/{record['id']}/ack", {"session": session})
        except ChapterClientError:
            # A lost acknowledgement is safe: the validated local cache is
            # already authoritative, so the run must not regenerate or lose
            # files. The server prunes the row when the session detaches.
            pass


def _source_timing(chunk, *, paragraph_member=False):
    """The source portion of a timing row is fully determined by the script."""
    row = chunk["segment"]
    expected = {"chunk_id": chunk["chunk_id"], "segment_id": row["id"],
                "block_id": row["block_id"], "voice_id": row.get("voice_id"),
                "role_id": row.get("role_id"), "source_ranges": chunk["source_ranges"],
                "spoken_text": chunk["spoken_text"], "spoken_start": chunk["spoken_start"],
                "spoken_end": chunk["spoken_end"]}
    if not paragraph_member:
        expected.update(source_text=row["text"], source_start=row["start"], source_end=row["end"])
    return expected


def _validate_timing_sources(metadata, chunks):
    from .chapter_cache import _timings
    try:
        _timings(metadata)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ChapterClientError("Downloaded chapter has invalid synchronization frame ranges") from exc
    for timing, chunk in zip(metadata["segments"], chunks):
        expected = _source_timing(chunk)
        if any(timing.get(key) != value for key, value in expected.items()):
            raise ChapterClientError("Downloaded chapter timing differs from its submitted source text")
        silent = chunk["silent_control"]
        duration = timing["chapter_end_frame"] - timing["chapter_start_frame"]
        if timing.get("no_speech") is not silent or (duration == 0) is not silent:
            raise ChapterClientError("Downloaded chapter timing omits or invents speech audio")
        if not silent and not {"source_audio_start_frame", "source_audio_end_frame"} <= timing.keys():
            raise ChapterClientError("Downloaded chapter lacks its original audio frame ranges")
        turns = chunk.get("paragraph_turns")
        if turns is not None:
            if (timing.get("precision") != "paragraph"
                    or timing.get("source_segments") != [_source_timing(turn, paragraph_member=True) for turn in turns]):
                raise ChapterClientError("Downloaded paragraph timing differs from its submitted source turns")
        elif "source_segments" in timing or timing.get("precision") == "paragraph":
            raise ChapterClientError("Downloaded chapter invents paragraph source turns")
        row = chunk["segment"]
        for key in ("performance", "simultaneous_speakers"):
            if timing.get(key, {} if key == "performance" else []) != row.get(key, {} if key == "performance" else []):
                raise ChapterClientError("Downloaded chapter timing differs from its submitted performers")
        lanes = timing.get("speaker_lanes")
        speakers = row.get("simultaneous_speakers", [])
        if speakers and not silent:
            if (not isinstance(lanes, list)
                    or [{key: lane.get(key) for key in ("role_id", "voice_id")} for lane in lanes] != speakers):
                raise ChapterClientError("Downloaded simultaneous timing differs from its submitted speakers")
        elif lanes is not None or "mix" in timing:
            raise ChapterClientError("Downloaded chapter invents simultaneous speaker lanes")


def _validate_artifact_manifest(folder, spec, metadata, verification, artifacts):
    manifest = read_json(regular_file(folder / "manifest.json"))
    if not isinstance(manifest, dict):
        raise ChapterClientError("Downloaded chapter artifact manifest must be an object")
    identity = manifest.get("identity")
    context = identity.get("context") if isinstance(identity, dict) else None
    if (manifest.get("format") != MANIFEST_FORMAT or manifest.get("version") != 1
            or manifest.get("record_sha256") != signed_record(manifest)
            or manifest.get("spec_hash") != digest(spec)
            or manifest.get("source_fingerprint") != spec["source_fingerprint"]
            or manifest.get("status") != metadata["status"]
            or not isinstance(identity, dict) or identity.get("spec_hash") != digest(spec)
            or not isinstance(identity.get("prepared_sha256"), str) or not _HASH.fullmatch(identity["prepared_sha256"])
            or not isinstance(context, dict) or context.get("spec_hash") != digest(spec)
            or context.get("source_fingerprint") != spec["source_fingerprint"]
            or context.get("voice_assets") != spec["voice_bindings"]
            or any(context.get(key) != verification.get(key) for key in ("engine", "verifier", "quality_version", "quality"))):
        raise ChapterClientError("Downloaded chapter artifact manifest does not match its verified job")
    expected = expected_artifact_names(metadata["frames"])
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, dict) or set(inventory) != expected:
        raise ChapterClientError("Downloaded chapter artifact inventory is incomplete or unexpected")
    records = {}
    for name, item in inventory.items():
        if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
                or not _HASH.fullmatch(item["sha256"]) or type(item.get("bytes")) is not int or item["bytes"] <= 0):
            raise ChapterClientError("Downloaded chapter artifact inventory is invalid")
        path = folder / name
        # Chapter-only downloads intentionally defer the lossless master.
        if name != "chapter.flac" or path.exists() or path.is_symlink():
            if file_hash(regular_file(path)) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ChapterClientError("Downloaded chapter artifact differs from its manifest")
        records[name] = {"sha256": item["sha256"], "size": item["bytes"]}
    if metadata["frames"] and metadata.get("encoded_sha256") != inventory["chapter.m4a"]["sha256"]:
        raise ChapterClientError("Downloaded chapter audio differs from its synchronization checksum")
    if artifacts is not None:
        if (not isinstance(artifacts, list) or len(artifacts) != len(expected) + 1
                or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in artifacts)
                or {item.get("name") for item in artifacts} != expected | {"manifest.json"}):
            raise ChapterClientError("Chapter service artifact inventory differs from its manifest")
        path = folder / "manifest.json"
        records["manifest.json"] = {"sha256": file_hash(path), "size": path.stat().st_size}
        if any(any(item.get(key) != value for key, value in records[item["name"]].items()) for item in artifacts):
            raise ChapterClientError("Chapter service artifact checksums differ from its manifest")


def _validate_metadata(folder, record, *, artifacts=None):
    folder = Path(folder)
    metadata = read_json(regular_file(folder / "chapter.json"))
    spec, chapter = record["spec"], record["selection"]["chapter"]
    expected = [chunk["chunk_id"] for chunk in chapter["chunks"] if not chunk["silent_control"]]
    if (not isinstance(metadata, dict)
            or metadata.get("format") != CHAPTER_FORMAT or metadata.get("version") != 1
            or metadata.get("source_fingerprint") != spec["source_fingerprint"]
            or metadata.get("spec_hash") != digest(spec)
            or metadata.get("voice_bindings") != spec["voice_bindings"]
            or metadata.get("id") != chapter["id"] or metadata.get("number") != chapter["number"]
            or type(metadata.get("frames")) is not int or metadata["frames"] < 0
            or bool(metadata["frames"]) != bool(expected)
            or metadata.get("sample_rate") != spec["render"].get("sample_rate", 44100)
            or metadata.get("channels") != 1 or not isinstance(metadata.get("segments"), list)
            or any(not isinstance(item, dict) for item in metadata["segments"])
            or [item.get("chunk_id") for item in metadata["segments"]]
               != [chunk["chunk_id"] for chunk in chapter["chunks"]]):
        raise ChapterClientError("Downloaded chapter metadata does not match its submitted source and timeline")
    audio = bool(metadata["frames"])
    pointers = audio_pointers(metadata["frames"])
    if (metadata.get("status") != pointers["status"]
            or metadata.get("audio_file") != pointers["audio_file"]
            or metadata.get("lossless_file") != pointers["lossless_file"]
            or any(not isinstance(metadata.get(key), str) or not _HASH.fullmatch(metadata[key])
                   for key in ("encoded_sha256", "pcm_data_sha256") if audio)
            or (not audio and (metadata.get("encoded_sha256") is not None or metadata.get("pcm_data_sha256") is not None))):
        raise ChapterClientError("Downloaded chapter audio metadata is incomplete")
    _validate_timing_sources(metadata, chapter["chunks"])
    verification = read_json(regular_file(folder / "verification.json"))
    acceptance = verification.get("acceptance") if isinstance(verification, dict) else None
    if (not isinstance(verification, dict)
            or verification.get("format") != VERIFICATION_FORMAT or verification.get("version") != 1
            or verification.get("spec_hash") != digest(spec) or verification.get("passed") is not True
            or not isinstance(acceptance, list)
            or any(not isinstance(entry, dict) or not isinstance(entry.get("report"), dict) for entry in acceptance)
            or [entry.get("chunk_id") for entry in acceptance] != expected
            or any(entry.get("report", {}).get("passed") is not True for entry in acceptance)):
        raise ChapterClientError("Downloaded chapter lacks complete accepted speech verification")
    sync = read_json(regular_file(folder / "sync.json"))
    expected_sync = sync_record(
        book_id=spec["book"].get("id"), source_fingerprint=spec["source_fingerprint"],
        spec_hash=digest(spec), chapter_id=chapter["id"], chapter_number=chapter["number"],
        sample_rate=metadata["sample_rate"], frames=metadata["frames"],
        precision="paragraph" if any("paragraph_turns" in chunk for chunk in chapter["chunks"]) else "chunk",
        audio_file=metadata["audio_file"], audio_sha256=metadata["encoded_sha256"],
        segments=metadata["segments"])
    if not isinstance(sync, dict) or any(sync.get(key) != value for key, value in expected_sync.items()):
        raise ChapterClientError("Downloaded synchronization metadata differs from its chapter")
    _validate_artifact_manifest(folder, spec, metadata, verification,
                                artifacts if artifacts is not None else record.get("artifacts"))
    return metadata


def _cached_artifacts_present(folder, record, *, require_masters=True):
    if not record.get("artifacts"):
        return False
    present = True
    for item in record["artifacts"]:
        name = item["name"]
        if name == "chapter.flac" and not require_masters:
            continue
        if name != Path(name).name:
            raise ChapterClientError("Invalid saved chapter artifact name")
        path = folder / name
        if not path.exists() and not path.is_symlink():
            present = False
        elif file_hash(regular_file(path)) != item["sha256"] or path.stat().st_size != item["size"]:
            raise ChapterClientError(f"Saved remote chapter artifact changed: {path}")
    return present


def _wait_jobs(client, jobs, notify, *, include_masters=True, session=None, watchdog=None):
    """Watch this run's own jobs from a fresh cursor until each completes.

    ``session`` is the run's session id, used to acknowledge verified
    deliveries; ``watchdog`` is the invocation's heartbeat, checked while
    waiting. Records exist only in memory, so every cursor starts at zero and
    no past run's events can be replayed or suppressed.
    """
    pending = dict(jobs)
    total, finished = len(pending), 0
    connection, watching = None, None
    try:
        while pending:
            if watchdog is not None:
                watchdog.check()
            for job_id, (folder, record) in list(pending.items()):
                response = client.request("GET", f"v1/jobs/{job_id}/events?after={record.get('last_event', 0)}")
                for event in response.get("events", []):
                    _event(record, event, notify)
                status = response.get("job") or client.request("GET", f"v1/jobs/{job_id}")
                if status.get("id") != job_id:
                    raise ChapterClientError("Chapter service returned another job's status")
                if status.get("paused") is True:
                    raise ChapterClientError("The rendering service lost this run's session; rerun the same "
                                             "command to start a fresh run (verified chapter caches are reused)")
                state = status.get("state")
                record["state"] = state
                if state in {"completed", "complete"}:
                    if record.get("last_event", 0) < status.get("last_event", 0):
                        continue
                    _completed_download(client, folder, record, status, include_masters=include_masters,
                                        notify=notify, session=session)
                    del pending[job_id]
                    finished += 1
                    chapter = record["selection"]["chapter"]
                    notify(f"Chapter {chapter['number']} / {record['selection']['total_chapters']} finished; "
                           f"{finished} of {total} chapters complete in this run")
                elif state in {"failed", "cancelled"}:
                    error = status.get("error")
                    detail = str(error)[:600] if error else "no diagnostic provided"
                    raise RemoteChapterFailed(
                        f"Remote chapter {record['selection']['chapter']['number']} {state}: {detail}. "
                        "Accepted paragraphs remain in the service's speech cache; rerun to render this chapter again",
                        job_id=job_id, record=record, paragraph=_failed_paragraph(error))
            if not pending:
                break
            first = next(iter(pending))
            if first != watching:
                if connection is not None:
                    connection.close()
                watching = first
                connection = client.websocket(first, pending[first][1].get("last_event", 0))
            interval = client.settings["poll_seconds"]
            if connection is None:
                time.sleep(interval)
            else:
                try:
                    payload = json.loads(connection.recv(timeout=interval))
                    if isinstance(payload, dict):
                        for event in payload.get("events", []):
                            _event(pending[first][1], event, notify)
                except TimeoutError:
                    pass
                except Exception:
                    connection.close()
                    connection, watching = None, None
    finally:
        if connection is not None:
            connection.close()


def _local_chapter(bundle, chapter, config, voices, *, allow_stale=False):
    """Retain already accepted local chapters when existing validation agrees."""
    from .workflow import (
        _chapter_context, _read_checkpoint, _restore_chapter,
        _same_chapter_audio_context, _saved_chapter_clips_present, RenderBlocked,
    )
    record = _read_checkpoint(config["render"]["work_dir"], chapter)
    if record is None:
        return None
    context = _chapter_context(bundle, chapter, config, voices,
                               record["context"]["engine"], record["context"]["verifier"])
    try:
        if allow_stale and not _same_chapter_audio_context(record["context"], context):
            # A deliberate script/voice/settings revision can replace a complete
            # local chapter remotely. Inspect its retained files first so the
            # revision does not silently hide corrupted local audio.
            from .chapter_cache import load_chapter
            _saved_chapter_clips_present(config["render"]["work_dir"], record)
            composed = load_chapter(Path(config["render"]["work_dir"]) / "chapters" / record["cache_key"],
                                    ffprobe=config["render"]["ffprobe"])
            if (composed["context"] != record["context"]
                    or composed["settings"] != record["context"]["settings"]):
                raise ValueError("Chapter cache and checkpoint disagree")
            return {"stale": True}
        composed, acceptance = _restore_chapter(config["render"]["work_dir"], chapter,
                                                record, context, voices, config)
        return {"composed": composed, "acceptance": acceptance}
    except (RenderBlocked, ValueError, KeyError, TypeError, OSError) as exc:
        raise ChapterClientError(f"Existing local chapter {chapter['number']} cannot be reused: {exc}. "
                                 "Keep or restore its accepted cache before choosing a remote rerender") from exc


def _partial_local_chunk_ids(work):
    """Identify retained local speech that this protocol cannot yet import."""
    identifiers = set()
    for path in (Path(work) / "clips").glob("*/accepted.json"):
        record = read_json(regular_file(path))
        if not isinstance(record, dict) or not isinstance(record.get("request"), dict):
            raise ChapterClientError("Invalid accepted local speech record; local audio remains untouched")
        request = record["request"]
        if request.get("kind") not in {"speech", "paragraph_speech"}:
            continue
        chunk, filename = request.get("chunk"), record.get("audio_file")
        if (not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str)
                or not isinstance(filename, str) or filename != Path(filename).name):
            raise ChapterClientError("Invalid accepted local speech binding; local audio remains untouched")
        audio = path.parent / filename
        if audio.exists() or audio.is_symlink():
            regular_file(audio)
            identifiers.add(chunk["chunk_id"])
    return identifiers


def _has_partial_local_speech(chapter, identifiers):
    for chunk in chapter["chunks"]:
        if chunk["chunk_id"] in identifiers:
            return True
        if any(turn["chunk_id"] in identifiers for turn in chunk.get("paragraph_turns", [])):
            return True
    return False


def _recover_failed_paragraph(bundle, config, plan, voices, capabilities, failure, notify):
    """Record one failed remote paragraph and rebuild the chapter with its turns.

    The chapter protocol already validates pre-split paragraph turns, so the
    resubmitted spec expresses the failed paragraph as its source turns and the
    server needs no splitting behavior. Only hash identities crossed the wire;
    the client matches them against its own plan and confirms the exact
    synthesis request before recording the immutable fallback marker.
    """
    detail = failure.paragraph
    if detail is None:
        return None
    chapter_id = failure.record["selection"]["chapter"]["id"]
    chapter = next((item for item in plan["chapters"] if item["id"] == chapter_id), None)
    if chapter is None:
        return None
    chunk = next((item for item in chapter["chunks"] if item["chunk_id"] == detail["chunk_id"]), None)
    if chunk is None or "paragraph_turns" not in chunk:
        return None
    identity = capabilities.get("engine")
    if not isinstance(identity, dict) or not isinstance(identity.get("engine"), dict):
        return None
    from .workflow import _paragraph_request, make_render_plan
    request = _paragraph_request(chunk, voices, identity["engine"], config)
    if digest(request) != detail["request_hash"]:
        raise ChapterClientError(
            f"Chapter {chapter['number']} paragraph {chunk['chunk_id']} does not match the service's "
            "synthesis request; the paragraph fallback was not recorded. Reconcile renderer "
            "configuration between this client and the chapter service, then rerun")
    from .paragraph_recovery import record_paragraph_fallback
    try:
        record_paragraph_fallback(config["render"]["work_dir"], request)
    except (ValueError, OSError) as exc:
        raise ChapterClientError(f"Could not save remote paragraph recovery: {exc}") from exc
    new_plan = make_render_plan(bundle, config, multi_speaker=True)
    if new_plan["errors"]:
        raise ChapterClientError("Paragraph recovery produced an invalid plan: "
                                 + "; ".join(new_plan["errors"]))
    replacement = next(item for item in new_plan["chapters"] if item["id"] == chapter_id)
    if (any(item["chunk_id"] == chunk["chunk_id"] for item in replacement["chunks"])
            or "".join(item["spoken_text"] for item in chapter["chunks"])
            != "".join(item["spoken_text"] for item in replacement["chunks"])):
        raise ChapterClientError("Paragraph recovery did not preserve the complete remote speech plan")
    notify(f"  Chapter {chapter['number']} paragraph failed verification; resubmitting its "
           f"{len(chunk['paragraph_turns'])} passages separately with their assigned voices. "
           "Verified clips are retained.")
    return new_plan, replacement


def render_remote(bundle, config, *, notify=print, chapters_only=False):
    from .workflow import _preflight, _lock_book_audio
    validated = config.pop("_validated_remote_plan", None)
    if validated is None:
        config, plan, store = _preflight(bundle, config, chapters_only=chapters_only, multi_speaker=True)
    else:
        plan, store = validated
    lock = _lock_book_audio(bundle)
    try:
        return _render_remote_locked(bundle, config, plan, store, notify=notify, chapters_only=chapters_only,
                                     validated=validated is not None)
    finally:
        lock.rmdir()


def _render_remote_locked(bundle, config, plan, store, *, notify, chapters_only, validated=False):
    if config["render"]["chapter_format"] != "m4a":
        raise ChapterClientError("Remote chapter jobs currently produce M4A; use --chapter-format m4a")
    voices = {voice_id: store.load(voice_id) for voice_id in plan["voice_ids"]}
    missing = [key for key, value in voices.items() if value is None]
    if missing:
        raise ChapterClientError("Remote chapter rendering requires saved voices. Run cast first for: " + ", ".join(missing))
    if not validated:
        validate_remote_plan(bundle, config, plan)
    work = _regular_directory(config["render"]["work_dir"])
    root = _regular_directory(work / "remote-chapters")
    bindings, assets = _voice_assets(voices, work, config["render"]["ffmpeg"])
    local_clip_ids = _partial_local_chunk_ids(work)
    cached, unfinished, local = {}, [], {}
    for chapter in plan["chapters"]:
        local_result = _local_chapter(bundle, chapter, config, voices, allow_stale=True)
        stale_local = local_result is not None and local_result.get("stale") is True
        if local_result is not None and not stale_local:
            local[chapter["id"]] = local_result
            notify(f"Reusing accepted local chapter {chapter['number']} / {len(bundle['chapters'])}")
            continue
        selection = _selection(bundle, chapter, config, bindings)
        folder = _regular_directory(root / digest(selection))
        descriptor = _read_cache(folder, selection)
        if descriptor is None:
            descriptor = _migrate_legacy_job(folder, selection)
        if descriptor is None and not stale_local and _has_partial_local_speech(chapter, local_clip_ids):
            raise ChapterClientError(
                f"Chapter {chapter['number']} has accepted local speech but no completed local chapter. "
                "Partial local clip migration to the chapter service is not supported yet; the local cache is retained. "
                "Finish this chapter with local rendering first, or deliberately choose a new --work-dir to render it remotely")
        if stale_local:
            notify(f"Chapter {chapter['number']} changed; rendering the revision remotely. "
                   "Previous local audio is retained.")
        if (descriptor is not None
                and _cached_artifacts_present(folder, descriptor, require_masters=False)
                and not _cached_artifacts_present(folder, descriptor)):
            # A durable storage link may still work after the GPU service stops.
            # Consult it only when a missing/expired link requires the service.
            client = ChapterClient(config["remote_chapters"])
            try:
                for artifact in descriptor["artifacts"]:
                    client.download(artifact, folder / artifact["name"])
            except ChapterClientError:
                pass
        if descriptor is not None and _cached_artifacts_present(folder, descriptor):
            notify(f"Reusing downloaded chapter {chapter['number']} / {len(bundle['chapters'])}")
            cached[chapter["id"]] = (folder, descriptor)
        else:
            unfinished.append((chapter, selection, folder, descriptor))
    if unfinished:
        client = ChapterClient(config["remote_chapters"])
        capabilities = client.request("GET", "v1/capabilities")
        if capabilities.get("protocol") != PROTOCOL:
            raise ChapterClientError("Chapter service uses an incompatible protocol")
        if capabilities.get("client_sessions") != 1:
            raise ChapterClientError("The chapter service does not support connected client sessions; "
                                     "update the worker to the matching renderer before rendering")
        if capabilities.get("ephemeral_runs") != 1:
            raise ChapterClientError("The chapter service does not support run-scoped ephemeral jobs with "
                                     "acknowledged cleanup; update the worker to the matching renderer before rendering")
        from .paragraphs import paragraph_supported
        if any(not paragraph_supported(capabilities, chunk)
               for chapter, *_ in unfinished for chunk in chapter["chunks"]):
            raise ChapterClientError("Chapter service does not support this plan's multi-speaker paragraphs")
        needed = {asset_id for _, selection, _, _ in unfinished for asset_id in selection["voice_bindings"].values()}
        _register_voices(client, {key: assets[key] for key in needed}, notify)
        # One retry allowance spans every job of this submission and their
        # retries; the next render_remote invocation starts a fresh budget.
        submission = os.urandom(32).hex() if capabilities.get("submission_budget") is True else None
        # A new explicit invocation always opens a new ephemeral session;
        # sessions are never persisted permissions.
        session_id = os.urandom(32).hex()
        client.request("POST", "v1/sessions", {"id": session_id})
        ttl = capabilities.get("session_ttl_seconds")
        interval = max(1.0, min(10.0, ttl / 6)) if type(ttl) in (int, float) and ttl > 0 else 5.0
        heartbeat = ClientHeartbeat(client, session_id, interval)
        heartbeat.start()
        pending = {}

        def submit(chapter, selection, folder, descriptor):
            heartbeat.check()
            spec = deepcopy(selection)
            fingerprint = capabilities.get("renderer_fingerprint")
            if fingerprint is not None:
                spec["renderer_fingerprint"] = fingerprint
            try:
                validate_spec(spec)
            except ValueError as exc:
                raise ChapterClientError(
                    f"Chapter {chapter['number']} / {len(bundle['chapters'])} "
                    f"cannot be rendered remotely: {exc}") from exc
            if descriptor is None:
                _clear_stale_chapter_folder(folder)
            record, status = _submit_chapter(client, selection, spec, submission=submission,
                                             session=session_id, notify=notify)
            cached[chapter["id"]] = (folder, record)
            if _cached_artifacts_present(folder, record):
                return
            pending[record["id"]] = (folder, record)
            notify(f"Scheduled chapter {chapter['number']} / {len(bundle['chapters'])}: {record['id']}")

        try:
            # Submit every chapter before watching any, so synthesis can immediately
            # enter the next queued chapter while previous chapters are encoded.
            for chapter, selection, folder, descriptor in unfinished:
                submit(chapter, selection, folder, descriptor)
            while pending:
                try:
                    _wait_jobs(client, pending, notify, include_masters=True,
                               session=session_id, watchdog=heartbeat)
                    break
                except RemoteChapterFailed as failure:
                    heartbeat.check()
                    pending = {identifier: value for identifier, value in pending.items()
                               if value[1].get("state") != "completed"}
                    recovered = _recover_failed_paragraph(bundle, config, plan, voices, capabilities,
                                                          failure, notify)
                    if recovered is None:
                        raise
                    pending.pop(failure.job_id, None)
                    plan, replacement = recovered
                    selection = _selection(bundle, replacement, config, bindings)
                    folder = _regular_directory(root / digest(selection))
                    submit(replacement, selection, folder, _read_cache(folder, selection))
        finally:
            heartbeat.stop()
            try:
                client.request("DELETE", f"v1/sessions/{session_id}")
            except ChapterClientError:
                pass
    if chapters_only:
        completed = []
        for chapter in plan["chapters"]:
            if chapter["id"] in local:
                audio = local[chapter["id"]]["composed"].get("encoded_path")
            else:
                folder, record = cached[chapter["id"]]
                metadata = _validate_metadata(folder, record)
                audio = str(folder / "chapter.m4a") if metadata["frames"] else None
            completed.append({"number": chapter["number"], "audio_file": audio})
        return {"chapters": completed}
    return _assemble_downloads(bundle, config, plan, cached, local, notify=notify)


def has_remote_chapters(config):
    directory = Path(config["render"]["work_dir"]) / "remote-chapters"
    return directory.is_dir() and (any(directory.glob("*/cache.json")) or any(directory.glob("*/job.json")))


def assemble_remote(bundle, config, *, notify=print):
    """Package accepted chapters, fetching compressed masters only when needed."""
    from .workflow import _preflight, _lock_book_audio
    config, plan, store = _preflight(bundle, config, chapters_only=False, multi_speaker=True)
    lock = _lock_book_audio(bundle)
    try:
        voices = {voice_id: store.load(voice_id) for voice_id in plan["voice_ids"]}
        if any(value is None for value in voices.values()):
            raise ChapterClientError("A selected voice is missing; restore it before assembly")
        bindings, _ = _voice_assets(voices, config["render"]["work_dir"], config["render"]["ffmpeg"])
        cached, local = {}, {}
        for chapter in plan["chapters"]:
            result = _local_chapter(bundle, chapter, config, voices)
            if result is not None:
                local[chapter["id"]] = result
                continue
            selection = _selection(bundle, chapter, config, bindings)
            folder = Path(config["render"]["work_dir"]) / "remote-chapters" / digest(selection)
            descriptor = _read_cache(folder, selection)
            if descriptor is None:
                descriptor = _migrate_legacy_job(folder, selection)
            if descriptor is None or not _cached_artifacts_present(folder, descriptor, require_masters=False):
                raise ChapterClientError(f"Chapter {chapter['number']} is not downloaded; run render --chapters-only to download it")
            if not _cached_artifacts_present(folder, descriptor):
                if not config["remote_chapters"]["url"]:
                    raise ChapterClientError("Final assembly needs the remote service URL to download lossless chapter masters")
                client = ChapterClient(config["remote_chapters"])
                notify(f"Downloading lossless master for chapter {chapter['number']} / {len(bundle['chapters'])}")
                try:
                    for artifact in descriptor["artifacts"]:
                        client.download(artifact, folder / artifact["name"])
                except ChapterClientError:
                    raise ChapterClientError(
                        "A legacy cache is missing its lossless master and its download link has expired. "
                        "Restore the master from durable storage, or render with a new --work-dir against a "
                        "live service. Existing verified files are retained; new runs download masters "
                        "before acknowledging delivery.") from None
            cached[chapter["id"]] = (folder, descriptor)
        return _assemble_downloads(bundle, config, plan, cached, local, notify=notify)
    finally:
        lock.rmdir()


def _assemble_downloads(bundle, config, plan, cached, local, *, notify):
    from .assembly import package_audio
    from .book_metadata import publication_assets
    from .workflow import _publish
    output = Path(config["render"]["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".remote-audiobook-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        stage = scratch / "ready"
        stage.mkdir()
        chapters, acceptance = [], []
        for chapter in plan["chapters"]:
            if chapter["id"] in local:
                composed = dict(local[chapter["id"]]["composed"])
                acceptance.extend(local[chapter["id"]]["acceptance"])
            else:
                folder, record = cached[chapter["id"]]
                metadata = _validate_metadata(folder, record)
                composed = deepcopy(metadata.get("chapter", metadata))
                if (metadata.get("source_fingerprint") != plan["source_fingerprint"]
                        or metadata.get("spec_hash") != digest(record["spec"])
                        or metadata.get("voice_bindings") != record["spec"]["voice_bindings"]
                        or composed.get("id") != chapter["id"]
                        or composed.get("number") != chapter["number"]
                        or type(composed.get("frames")) is not int or composed["frames"] < 0
                        or composed.get("sample_rate") != config["render"]["sample_rate"]
                        or not isinstance(composed.get("segments"), list)
                        or [item.get("chunk_id") for item in composed["segments"]]
                           != [chunk["chunk_id"] for chunk in chapter["chunks"]]):
                    raise ChapterClientError("Downloaded chapter metadata does not match its submitted source and timeline")
                wav = scratch / f"{chapter['id']}.wav"
                if composed["frames"]:
                    _run_audio([config["render"]["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                                "-i", str(folder / "chapter.flac"), "-map", "0:a:0", "-c:a", "pcm_s16le", str(wav)])
                    if pcm_sha256(wav) != metadata.get("pcm_data_sha256"):
                        raise ChapterClientError("Decoded chapter master differs from the server's lossless PCM")
                else:
                    with wave.open(str(wav), "wb") as stream:
                        stream.setparams((1, 2, config["render"]["sample_rate"], 0, "NONE", "not compressed"))
                composed["wav_path"] = str(wav)
                # WAV container hashes need not equal the server's transient WAV.
                composed.pop("pcm_sha256", None)
                encoded = folder / f"chapter.{config['render']['chapter_format']}"
                if encoded.exists():
                    composed["encoded_path"] = str(encoded)
                    composed["encoded_sha256"] = file_hash(encoded)
                acceptance.append(read_json(regular_file(folder / "verification.json")))
            composed.update(number=chapter["number"], title=chapter["title"], script_path=chapter["script_path"],
                            total_chapters=len(bundle["chapters"]))
            chapters.append(composed)
        book, cover, artwork = publication_assets(bundle, config, stage / "artwork", notify=notify)
        audio = package_audio(chapters, stage, book=book, cover_path=cover,
                              chapter_format=config["render"]["chapter_format"],
                              aac_bitrate=config["render"]["aac_bitrate"], mp3_bitrate=config["render"]["mp3_bitrate"],
                              ffmpeg=config["render"]["ffmpeg"], ffprobe=config["render"]["ffprobe"],
                              source_fingerprint=plan["source_fingerprint"], notify=notify)
        if artwork is not None:
            audio["audiobook"]["artwork"] = artwork
            write_json(stage / "audio-manifest.json", audio)
        write_json(stage / "source.json", bundle["source"])
        write_json(stage / "cast.json", bundle["cast"])
        write_json(stage / "verification.json", acceptance)
        result = {"format": "audiobook-render", "version": 1, "status": "complete",
                  "bundle_hash": bundle["bundle_hash"], "source_fingerprint": plan["source_fingerprint"],
                  "partial_book": plan["partial_book"], "series_id": plan["series_id"], "audio": audio,
                  "chapter_identities": {key: digest(value[1]["spec"]) for key, value in cached.items()},
                  "verification_policy": "preserve_successful_acceptance",
                  "files_sha256": {path.relative_to(stage).as_posix(): file_hash(path)
                                   for path in sorted(stage.rglob("*")) if path.is_file()}}
        write_json(stage / "render.json", result)
        publication = ({"expected": config["_publication_expected"]}
                       if "_publication_expected" in config else {})
        _publish(stage, output, **publication)
    notify(f"Audiobook ready: {output / 'book.m4b'}")
    return result
