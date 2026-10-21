#!/usr/bin/env python3
"""Prepare and supervise our chapter service plus the private Fish CUDA process.

Run with the project's Python environment. Application settings and credentials
come exclusively from the supplied global TOML file. Only ``prepare`` downloads
models; ``serve`` uses prepared local files and never starts a cloud resource.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text


FISH_SOURCE = Path("/opt/fish-speech")
APPLICATION_PYTHON = Path("/opt/epub-tts-narrator/.venv/bin/python")
WHISPER_REPOSITORY = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
RECEIPT = ".epub-tts-prepared.json"
FISH_URL = "http://127.0.0.1:8080/v1/tts"


def validate_worker(config):
    """Check this image's deliberately narrow, single-model deployment profile."""
    if config["remote_chapters"]["url"]:
        raise ValueError("The worker config must not set remote_chapters.url")
    workers = config["render_pool"]["workers"]
    if len(workers) != 1 or not (
        workers[0]["type"] == "fish_http"
        and workers[0].get("protocol") == "fish_native_drama3"
        and workers[0]["url"] == FISH_URL
        and workers[0]["concurrency"] == 1
        and not workers[0].get("api_key")
    ):
        raise ValueError("Use one fish_native_drama3 worker at http://127.0.0.1:8080/v1/tts "
                         "with concurrency = 1 and no internal API key")
    if not config["engine"].get("checkpoint"):
        raise ValueError("engine.checkpoint must name the native Drama 3 model directory")
    if not re.fullmatch(r"[0-9a-f]{40}", config["render_pool"]["model_revision"]):
        raise ValueError("render_pool.model_revision must be a pinned 40-character commit SHA")
    verification = config["verification"]
    if verification["type"] != "faster_whisper" or verification["device"] != "cpu":
        raise ValueError("This worker requires verification.type = faster_whisper and device = cpu")
    if verification.get("reference_model"):
        raise ValueError("This worker prepares one Whisper model; omit verification.reference_model")
    api_key = config["chapter_server"]["api_key"]
    if not api_key or api_key.upper().startswith("REPLACE-WITH"):
        raise ValueError("Set chapter_server.api_key to a new random application key before starting the worker")
    if config["chapter_server"]["port"] == 8080:
        raise ValueError("chapter_server.port must differ from the private Fish port 8080")


def model_specs(config):
    return (
        ("fishaudio/drama-3", config["render_pool"]["model_revision"],
         Path(config["engine"]["checkpoint"])),
        (WHISPER_REPOSITORY, WHISPER_REVISION, Path(config["verification"]["model"])),
    )


def required_model_files(repo, directory):
    if repo == "fishaudio/drama-3":
        names = {"codec.pth", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json"}
        index = read_json(directory / "model.safetensors.index.json")
        weights = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Drama 3 model index has no weight map")
        for name in weights.values():
            if not isinstance(name, str) or not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name):
                raise ValueError("Drama 3 model index contains an invalid shard filename")
            names.add(name)
    else:
        names = {"model.bin", "config.json", "tokenizer.json", "preprocessor_config.json",
                 "vocabulary.json"}
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Model is incomplete: missing or empty {name}; run prepare")
    return sorted(names)


def file_state(directory, names):
    return {name: {"size": (state := (directory / name).stat()).st_size,
                   "mtime_ns": state.st_mtime_ns} for name in names}


@contextmanager
def worker_lease(config):
    directory = Path(config["chapter_server"]["data_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runpod-worker.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another worker prepare/serve process already uses this data directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(config, *, downloader=None):
    """An explicit, restartable download step; never invoked by serve."""
    validate_worker(config)
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("Install huggingface-hub in the worker environment before prepare") from None
        downloader = snapshot_download
    with worker_lease(config):
        for repo, revision, directory in model_specs(config):
            receipt_path = directory / RECEIPT
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if (not isinstance(receipt, dict) or receipt.get("repository") != repo
                        or receipt.get("revision") != revision):
                    raise ValueError("Prepared model revision changed; configure a new model directory")
            print(f"Preparing {repo} at revision {revision} in {directory}", flush=True)
            try:
                downloader(repo_id=repo, revision=revision, local_dir=str(directory),
                           token=config["huggingface"]["api_key"] or False,
                           endpoint="https://huggingface.co", max_workers=5)
            except Exception as exc:
                raise RuntimeError(f"Could not prepare {repo}: {exc}. "
                                   "Completed and partial downloads remain for a retry.") from None
            names = required_model_files(repo, directory)
            write_json(receipt_path, {"version": 1, "repository": repo, "revision": revision,
                                      "files": file_state(directory, names)})
    print("Models prepared. Start serve when ready; no service was started.", flush=True)


def check(config):
    validate_worker(config)
    for repo, revision, directory in model_specs(config):
        try:
            names = required_model_files(repo, directory)
            receipt = read_json(directory / RECEIPT)
        except (OSError, ValueError):
            raise ValueError(f"{repo} is not prepared; run worker.py prepare with this config") from None
        if receipt != {"version": 1, "repository": repo, "revision": revision,
                       "files": file_state(directory, names)}:
            raise ValueError(f"{repo} files or revision changed; run prepare before serving")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get_json(url):
    with build_opener(ProxyHandler({}), _NoRedirect()).open(
            Request(url, headers={"Accept": "application/json"}), timeout=3) as response:
        if response.status != 200:
            raise ValueError("Unexpected health status")
        body = response.read(16385)
        if len(body) > 16384:
            raise ValueError("Oversized health response")
        return json.loads(body)


class WorkerStopped(Exception):
    pass


def wait_ready(process, stop, *, timeout=600, fetch=_get_json):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop.is_set():
            raise WorkerStopped()
        if process.poll() is not None:
            raise RuntimeError("Fish CUDA process exited during startup; inspect its log above")
        try:
            health = fetch("http://127.0.0.1:8080/v1/health")
            defaults = fetch("http://127.0.0.1:8080/v1/tts/defaults")
            if (health == {"status": "ok"} and isinstance(defaults, dict)
                    and defaults.get("model_family") == "drama3"):
                return
        except (OSError, HTTPError, URLError, ValueError):
            pass
        stop.wait(1)
    raise RuntimeError("Fish CUDA startup timed out; inspect its log above")


def stop_children(children, *, grace_seconds=15):
    """Signal the process groups too, including inference subprocesses."""
    for process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for process in reversed(children):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A parent may have exited while an inference child stayed alive. Kill any
    # surviving members of our own dedicated groups before leaving the pod.
    for process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def serve(config, config_path, *, fish_source=FISH_SOURCE,
          application_python=APPLICATION_PYTHON, startup_timeout=600):
    check(config)
    fish_python = fish_source / ".venv/bin/python"
    for path in (fish_python, fish_source / "tools/api_server.py", application_python):
        if not path.is_file():
            raise ValueError(f"Worker image is missing {path}")
    native_command = [str(fish_python), "-u", "tools/api_server.py", "--listen", "127.0.0.1:8080",
                      "--llama-checkpoint-path", config["engine"]["checkpoint"],
                      "--device", "cuda:0", "--workers", "1"]
    chapter_command = [str(application_python), "-u", "-m", "epub_tts_narrator.renderer.cli",
                       "serve-chapters", "--config", str(Path(config_path).resolve())]
    stop = threading.Event()
    received_signal = []

    def on_signal(number, frame):
        received_signal.append(number)
        stop.set()

    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    children = []
    with worker_lease(config):
        try:
            for number in handlers:
                signal.signal(number, on_signal)
            print("Starting the private Fish CUDA process; model loading may take several minutes.", flush=True)
            native = subprocess.Popen(native_command, cwd=fish_source, start_new_session=True)
            children.append(native)
            wait_ready(native, stop, timeout=startup_timeout)
            if stop.is_set():
                raise WorkerStopped()
            print("Fish CUDA is ready. Starting the chapter service.", flush=True)
            chapter = subprocess.Popen(chapter_command, start_new_session=True)
            children.append(chapter)
            while not stop.wait(0.5):
                native_status, chapter_status = native.poll(), chapter.poll()
                if native_status is not None:
                    raise RuntimeError("Fish CUDA process exited; stopping the chapter service")
                if chapter_status is not None:
                    return chapter_status if chapter_status >= 0 else 128 - chapter_status
        except WorkerStopped:
            pass
        finally:
            try:
                stop_children(children)
            finally:
                for number, handler in handlers.items():
                    signal.signal(number, handler)
    return 128 + received_signal[0] if received_signal else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/workspace/epub-tts-narrator.toml"))
    parser.add_argument("--startup-timeout", type=float, default=600,
                        help="Seconds to allow the private CUDA process to load (1–3600)")
    parser.add_argument("command", choices=("prepare", "check", "serve"))
    args = parser.parse_args(argv)
    config = None
    try:
        if not 1 <= args.startup_timeout <= 3600:
            raise ValueError("--startup-timeout must be from 1 through 3600 seconds")
        config = load_config(args.config)
        if args.command == "prepare":
            prepare(config)
        elif args.command == "check":
            check(config)
            print("Worker config and prepared model files are ready.", flush=True)
        else:
            return serve(config, args.config, startup_timeout=args.startup_timeout)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        # Downloader/provider exceptions may contain request headers or URLs.
        # Include the storage key names not collected by the shared helper.
        secrets = configured_secrets(config or {})
        if config:
            secrets.update(config["chapter_storage"].get(key, "")
                           for key in ("access_key_id", "secret_access_key"))
        print("Worker stopped: " + safe_error_text(exc, secrets), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
