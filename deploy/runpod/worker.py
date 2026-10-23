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
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text


APPLICATION_ROOT = Path(__file__).resolve().parents[2]
FISH_SOURCE = APPLICATION_ROOT.parent / "fish-speech"
SGLANG_SOURCE = APPLICATION_ROOT.parent / "sglang-omni"
APPLICATION_PYTHON = APPLICATION_ROOT / ".venv/bin/python"
WHISPER_REPOSITORY = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
RECEIPT = ".epub-tts-prepared.json"
FISH_URL = "http://127.0.0.1:8080/v1/tts"
SGLANG_URL = "http://127.0.0.1:8080/v1/audio/speech"
SMOKE_TEXT = ("The little boat crossed the quiet lake. We reached the far shore before sunset, "
              "then walked home together beneath the stars.")


def validate_worker(config):
    """Check this image's deliberately narrow, single-model deployment profile."""
    if config["remote_chapters"]["url"]:
        raise ValueError("The worker config must not set remote_chapters.url")
    workers = config["render_pool"]["workers"]
    if len(workers) != 1:
        raise ValueError("Use exactly one private CUDA worker")
    worker = workers[0]
    native = (worker.get("protocol") == "fish_native_drama3"
              and worker["url"] == FISH_URL and worker["concurrency"] == 1)
    batched = (worker.get("protocol") == "fish_sglang_drama3"
               and worker["url"] == SGLANG_URL
               and type(worker["concurrency"]) is int and 1 <= worker["concurrency"] <= 8
               and not worker.get("multi_speaker", False))
    if worker["type"] != "fish_http" or worker.get("api_key") or not (native or batched):
        raise ValueError("Use one loopback Fish native worker (concurrency 1) or SGLang worker "
                         "(concurrency 1–8, multi_speaker = false), with no internal API key")
    if not config["engine"].get("checkpoint"):
        raise ValueError("engine.checkpoint must name the native Drama 3 model directory")
    if not re.fullmatch(r"[0-9a-f]{40}", config["render_pool"]["model_revision"]):
        raise ValueError("render_pool.model_revision must be a pinned 40-character commit SHA")
    verification = config["verification"]
    if verification["type"] != "faster_whisper" or verification["device"] not in {"cpu", "cuda"}:
        raise ValueError("This worker requires verification.type = faster_whisper and device = cpu or cuda")
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
                try:
                    names = required_model_files(repo, directory)
                    unchanged = receipt.get("files") == file_state(directory, names)
                except (OSError, ValueError):
                    unchanged = False
                if unchanged:
                    print(f"Prepared {repo} is unchanged; reusing local model files.", flush=True)
                    continue
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


def check_audio_decoder(*, decoder=None):
    """Exercise the installed Whisper/PyAV combination without loading models."""
    try:
        if decoder is None:
            from faster_whisper.audio import decode_audio
            decoder = decode_audio
        audio = io.BytesIO()
        with wave.open(audio, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\xe8\x03" * 1600)
        audio.seek(0)
        decoded = decoder(audio, sampling_rate=16000)
        if len(decoded) != 1600 or not any(decoded):
            raise ValueError("the decoder did not preserve the PCM test samples")
    except Exception as exc:
        raise RuntimeError("Whisper audio decoder check failed; run a full worker update to repair "
                           f"its dependencies: {exc}") from None


def cuda_library_directory(application_python=APPLICATION_PYTHON):
    """Locate the pinned application cuBLAS, without importing CUDA or a model."""
    probe = (
        "import json; from pathlib import Path; import nvidia.cublas; "
        "from importlib.metadata import version; "
        "assert version('ctranslate2') == '4.8.3'; "
        "assert version('nvidia-cublas-cu12') == '12.8.4.1'; "
        "print(json.dumps(str(Path(nvidia.cublas.__path__[0]) / 'lib')))"
    )
    try:
        result = subprocess.run([str(application_python), "-c", probe],
                                capture_output=True, text=True, timeout=30, check=False)
        directory = Path(json.loads(result.stdout)) if result.returncode == 0 else None
        app_venv = Path(application_python).absolute().parent.parent.resolve()
        if (directory is None or not directory.is_absolute()
                or not directory.resolve().is_relative_to(app_venv)
                or not all((directory / name).is_file()
                           for name in ("libcublas.so.12", "libcublasLt.so.12"))):
            raise ValueError("missing pinned cuBLAS libraries")
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        raise RuntimeError("CUDA Whisper requires the pinned CTranslate2 4.8.3 and cuBLAS "
                           "12.8.4.1 application packages; run a full worker update") from None
    return directory


def application_environment(config, *, application_python=APPLICATION_PYTHON):
    """Set loader paths before the application interpreter starts, never globally."""
    environment = os.environ.copy()
    if config["verification"]["device"] == "cuda":
        directory = str(cuda_library_directory(application_python))
        inherited = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [directory, *(entry for entry in inherited.split(os.pathsep)
                          if entry and entry != directory)])
    return environment


def run_application(config, arguments, *, application_python=APPLICATION_PYTHON):
    """Replace this launcher with application Python, preserving signals/status."""
    if not arguments:
        raise ValueError("run-application requires -- followed by a Python script or -m module")
    environment = application_environment(config, application_python=application_python)
    os.execve(str(application_python), [str(application_python), *arguments], environment)


def check_cuda_verification(config, *, application_python=APPLICATION_PYTHON):
    """Check CUDA libraries and hardware support without loading Whisper weights."""
    if config["verification"]["device"] != "cuda":
        return
    environment = application_environment(config, application_python=application_python)
    probe = (
        "import ctypes, json, ctranslate2; "
        "ctypes.CDLL('libcublasLt.so.12', mode=ctypes.RTLD_GLOBAL); "
        "ctypes.CDLL('libcublas.so.12', mode=ctypes.RTLD_GLOBAL); "
        "assert ctranslate2.get_cuda_device_count() > 0, 'No visible CUDA device'; "
        "print(json.dumps(sorted(ctranslate2.get_supported_compute_types('cuda', 0))))"
    )
    try:
        result = subprocess.run([str(application_python), "-c", probe], env=environment,
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            detail = " ".join(result.stderr.splitlines()[-3:])[:600]
            raise ValueError("CUDA libraries or device initialization failed" +
                             (f": {detail}" if detail else ""))
        supported = json.loads(result.stdout)
        if not isinstance(supported, list) or not supported or not all(
                isinstance(item, str) for item in supported):
            raise ValueError("CUDA capability probe returned no supported compute types")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"CUDA Whisper preflight failed: {exc}. Check the NVIDIA driver "
                           "and run a full worker update; CPU fallback is disabled") from None
    requested = config["verification"]["compute_type"]
    if requested not in {"default", "auto"} and requested not in supported:
        raise ValueError(f"CUDA Whisper does not support verification.compute_type={requested!r} "
                         "on GPU 0; supported types: " + ", ".join(supported))


def check(config, *, application_python=APPLICATION_PYTHON):
    validate_worker(config)
    check_audio_decoder()
    check_cuda_verification(config, application_python=application_python)
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


def _get_json(url, *, headers=None):
    with build_opener(ProxyHandler({}), _NoRedirect()).open(
            Request(url, headers={"Accept": "application/json", **(headers or {})}), timeout=3) as response:
        if response.status != 200:
            raise ValueError("Unexpected health status")
        body = response.read(16385)
        if len(body) > 16384:
            raise ValueError("Oversized health response")
        return json.loads(body)


def _download_hash(url, expected_size):
    """Validate a presigned download without exposing its URL in diagnostics."""
    try:
        checksum, size = hashlib.sha256(), 0
        with build_opener(ProxyHandler({}), _NoRedirect()).open(url, timeout=60) as response:
            if response.status != 200:
                raise ValueError("Unexpected artifact response")
            while blob := response.read(min(1024 * 1024, expected_size - size + 1)):
                size += len(blob)
                if size > expected_size:
                    raise ValueError("Oversized artifact response")
                checksum.update(blob)
        if size != expected_size:
            raise ValueError("Truncated artifact response")
        return checksum.hexdigest()
    except Exception:
        raise RuntimeError("The smoke artifact could not be downloaded and checked") from None


def storage_smoke(config, path, *, publisher_factory=None, download=_download_hash):
    """Round-trip only our unique temporary object; never touch book exports."""
    if publisher_factory is None:
        from epub_tts_narrator.renderer.chapter_storage import S3ArtifactPublisher
        publisher_factory = S3ArtifactPublisher
    settings = dict(config["chapter_storage"])
    settings["prefix"] = "/".join(part for part in (
        settings.get("prefix", "epub-tts-narrator").strip("/"), "deployment-smoke") if part)
    publisher = publisher_factory(settings)
    checksum = file_hash(path)
    job_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    # Compute the unique key before upload so a partially successful upload is
    # cleaned up even when its HEAD verification fails.
    name = "chapter" + path.suffix
    key = "/".join((settings["prefix"], job_id, checksum, name))
    try:
        record = publisher.publish(job_id, name, path, checksum)
        if download(publisher.url(record), record["size"]) != checksum:
            raise RuntimeError("The smoke artifact download checksum does not match")
    finally:
        try:
            publisher.client.delete_object(Bucket=publisher.bucket, Key=key)
            try:
                publisher.client.head_object(Bucket=publisher.bucket, Key=key)
            except Exception as exc:
                error = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if str(error) not in {"404", "NoSuchKey", "NotFound"}:
                    raise
            else:
                raise RuntimeError("Temporary smoke artifact still exists")
        except Exception:
            raise RuntimeError("Could not confirm deletion of the temporary smoke artifact") from None


def smoke(config, *, fetch=_get_json, engine_factory=None, verifier_factory=None,
          roundtrip=storage_smoke):
    """Exercise synthesis, recognition, encoding and Ceph using invented text."""
    from epub_tts_narrator.renderer.chapter_protocol import PROTOCOL
    from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
    from epub_tts_narrator.renderer.quality import validate_audio
    from epub_tts_narrator.renderer.verification import create_verifier
    import soundfile as sf

    check(config)
    service = config["chapter_server"]
    capabilities = fetch(f"http://127.0.0.1:{service['port']}/v1/capabilities",
                         headers={"Authorization": f"Bearer {service['api_key']}"})
    if not isinstance(capabilities, dict) or capabilities.get("protocol") != PROTOCOL:
        raise RuntimeError("The authenticated chapter service is not ready")
    print("Smoke test: authenticated chapter service is ready; generating a short GPU sample.", flush=True)
    settings = {**config["render_pool"]["workers"][0], "timeout_seconds": 300}
    engine = (engine_factory or FishHTTPEngine)(settings, config["engine"])
    verifier = None
    scratch = Path(service["data_dir"]) / "deployment-smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="check-", dir=scratch) as temporary:
            directory = Path(temporary)
            audio = directory / "sample.wav"
            # A self-contained smoke check cannot require any book's cast. The
            # native transport accepts an empty reference list for this probe.
            engine._request(SMOKE_TEXT, [], audio, 4101)
            device = config["verification"]["device"].upper()
            print(f"Smoke test: checking generated words with {device} Whisper.", flush=True)
            verifier = (verifier_factory or create_verifier)(
                config["verification"], config.get("jev"), config.get("provider"))
            report = validate_audio(audio, SMOKE_TEXT, verifier, config["quality"], purpose="reference")
            if not report["passed"]:
                raise RuntimeError("GPU smoke speech did not pass verification: " + ", ".join(report["reasons"]))
            samples, rate = sf.read(audio, dtype="int16", always_2d=True)
            compressed = directory / "chapter.flac"
            sf.write(compressed, samples, rate, subtype="PCM_16", format="FLAC")
            print("Smoke test: encoding M4A and checking compressed Ceph downloads.", flush=True)
            chapter = directory / "chapter.m4a"
            result = subprocess.run(
                [config["render"]["ffmpeg"], "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-i", str(compressed), "-c:a", "aac", "-b:a", config["render"]["aac_bitrate"],
                 "-movflags", "+faststart", str(chapter)],
                capture_output=True, timeout=60, check=False)
            if result.returncode or not chapter.is_file() or chapter.stat().st_size == 0:
                raise RuntimeError("Smoke M4A encoding failed")
            for artifact in (compressed, chapter):
                roundtrip(config, artifact)
            print("Smoke test passed: GPU speech, Whisper, M4A/FLAC and Ceph round trips; temporary objects deleted.", flush=True)
    finally:
        engine.close()
        if verifier is not None:
            verifier.close()


class WorkerStopped(Exception):
    pass


def wait_ready(process, stop, *, timeout=600, fetch=_get_json, protocol="fish_native_drama3"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop.is_set():
            raise WorkerStopped()
        if process.poll() is not None:
            raise RuntimeError("Fish CUDA process exited during startup; inspect its log above")
        try:
            if protocol == "fish_sglang_drama3":
                health = fetch("http://127.0.0.1:8080/health")
                models = fetch("http://127.0.0.1:8080/v1/models")
                if (isinstance(health, dict) and health.get("status") == "healthy"
                        and health.get("running") is True and isinstance(models, dict)
                        and any(isinstance(item, dict) and item.get("id") == "drama-3"
                                for item in models.get("data", []))):
                    return
            else:
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


def inference_command(config, *, fish_source=FISH_SOURCE, sglang_source=SGLANG_SOURCE):
    """Select one stock server. Parallel slots are requests, never model replicas."""
    worker = config["render_pool"]["workers"][0]
    if worker.get("protocol") == "fish_sglang_drama3":
        executable = sglang_source / ".venv/bin/sgl-omni"
        if not executable.is_file():
            raise ValueError(f"Worker image is missing {executable}; run a full update")
        return [str(executable), "serve", "--model-path", config["engine"]["checkpoint"],
                "--model-name", "drama-3", "--host", "127.0.0.1", "--port", "8080",
                "--tts_engine.gpu_memory_fraction",
                "0.60" if config["verification"]["device"] == "cuda" else "0.70",
                "--tts_engine.factory.max_running_requests", str(worker["concurrency"]),
                "--tts_engine.factory.first_frame_cuda_graph_max_requests", str(worker["concurrency"]),
                "--vocoder.factory.codec_path", str(Path(config["engine"]["checkpoint"]) / "codec.pth")], sglang_source
    fish_python = fish_source / ".venv/bin/python"
    for path in (fish_python, fish_source / "tools/api_server.py"):
        if not path.is_file():
            raise ValueError(f"Worker image is missing {path}")
    return [str(fish_python), "-u", "tools/api_server.py", "--listen", "127.0.0.1:8080",
            "--llama-checkpoint-path", config["engine"]["checkpoint"],
            "--device", "cuda:0", "--workers", "1"], fish_source


def inference_environment(config, backend_directory, *, application_python=APPLICATION_PYTHON):
    """Resolve compiler tools and caches before the backend imports its libraries."""
    environment = os.environ.copy()
    # The application may have been launched with its CUDA 12 loader path.
    # Never pass that managed path to the independent SGLang CUDA 13 process.
    app_venv = Path(application_python).absolute().parent.parent.resolve()
    if "LD_LIBRARY_PATH" in environment:
        remaining = [entry for entry in environment["LD_LIBRARY_PATH"].split(os.pathsep)
                     if entry and not (Path(entry).resolve().is_relative_to(app_venv)
                                       and Path(entry).parts[-3:] == ("nvidia", "cublas", "lib"))]
        if remaining:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(remaining)
        else:
            environment.pop("LD_LIBRARY_PATH", None)
    backend_bin = str(Path(backend_directory) / ".venv/bin")
    inherited_path = environment.get("PATH", os.defpath)
    environment["PATH"] = os.pathsep.join(
        [backend_bin, *(entry for entry in inherited_path.split(os.pathsep)
                        if entry and entry != backend_bin)])
    if config["render_pool"]["workers"][0].get("protocol") == "fish_sglang_drama3":
        # These are stock upstream cache controls, not user configuration via
        # environment variables. Derive their location from the configured
        # persistent store and override ambient ephemeral paths before imports.
        # Omni otherwise selects /root/.cache/sglang-omni/torchinductor early.
        cache = Path(config["chapter_server"]["data_dir"]) / "runtime-cache" / "sglang"
        directories = {
            "SGLANG_CACHE_DIR": cache,
            "SGLANG_JIT_CACHE_DIR": cache / "jit",
            "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
            "TRITON_CACHE_DIR": cache / "triton",
            # FlashInfer appends .cache/flashinfer/<version>/<architecture>.
            "FLASHINFER_WORKSPACE_BASE": cache,
            "CUDA_CACHE_PATH": cache / "nv",
        }
        for key, directory in directories.items():
            directory.mkdir(parents=True, exist_ok=True)
            environment[key] = str(directory)
    return environment


def serve(config, config_path, *, fish_source=FISH_SOURCE, sglang_source=SGLANG_SOURCE,
          application_python=APPLICATION_PYTHON, startup_timeout=600, chapters=True, maintenance=False):
    check(config, application_python=application_python)
    native_command, native_directory = inference_command(
        config, fish_source=fish_source, sglang_source=sglang_source)
    if chapters and not application_python.is_file():
        raise ValueError(f"Worker image is missing {application_python}")
    chapter_command = [str(application_python), "-u", "-m", "epub_tts_narrator.renderer.cli",
                       "serve-chapters", "--config", str(Path(config_path).resolve())]
    if maintenance:
        chapter_command.append("--maintenance")
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
            native = subprocess.Popen(native_command, cwd=native_directory, start_new_session=True,
                                      env=inference_environment(config, native_directory,
                                                                application_python=application_python))
            children.append(native)
            wait_ready(native, stop, timeout=startup_timeout,
                       protocol=config["render_pool"]["workers"][0]["protocol"])
            if stop.is_set():
                raise WorkerStopped()
            chapter = None
            if chapters:
                mode = " in maintenance mode; queued books remain paused" if maintenance else ""
                print(f"Fish CUDA is ready. Starting the chapter service{mode}.", flush=True)
                chapter = subprocess.Popen(
                    chapter_command, start_new_session=True,
                    env=application_environment(config, application_python=application_python))
                children.append(chapter)
            else:
                print("Fish CUDA is ready for isolated benchmarking; chapter jobs are not running.", flush=True)
            while not stop.wait(0.5):
                native_status = native.poll()
                if native_status is not None:
                    raise RuntimeError("Fish CUDA process exited; stopping the chapter service")
                chapter_status = chapter.poll() if chapter is not None else None
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
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    application_args = arguments[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/workspace/epub-tts-narrator.remote.toml"))
    parser.add_argument("--startup-timeout", type=float, default=600,
                        help="Seconds to allow the private CUDA process to load (1–3600)")
    parser.add_argument("--fish-source", type=Path, default=FISH_SOURCE,
                        help="Persistent, pinned Fish Speech checkout (default: beside the application)")
    parser.add_argument("--sglang-source", type=Path, default=SGLANG_SOURCE,
                        help="Persistent, pinned SGLang-Omni checkout")
    parser.add_argument("command", choices=("prepare", "check", "serve", "serve-maintenance", "serve-tts", "smoke", "run-application"))
    args = parser.parse_args(arguments[:separator])
    config = None
    try:
        if not 1 <= args.startup_timeout <= 3600:
            raise ValueError("--startup-timeout must be from 1 through 3600 seconds")
        if separator < len(arguments) and args.command != "run-application":
            raise ValueError("Only run-application accepts arguments after --")
        config = load_config(args.config)
        if args.command == "run-application":
            return run_application(config, application_args)
        if args.command == "prepare":
            prepare(config)
        elif args.command == "check":
            check(config)
            print("Worker config and prepared model files are ready.", flush=True)
        elif args.command == "smoke":
            # dlopen reads the process's startup search path. Restart this CLI
            # once with the same environment used by the chapter service.
            if config["verification"]["device"] == "cuda":
                environment = application_environment(config)
                if environment.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH"):
                    return run_application(config, [str(Path(__file__).resolve()), *arguments])
            def smoke_timeout(number, frame):
                raise RuntimeError("Worker smoke test exceeded its ten-minute limit")
            previous_handler = signal.signal(signal.SIGALRM, smoke_timeout)
            previous_alarm = signal.alarm(600)
            try:
                smoke(config)
            finally:
                signal.alarm(previous_alarm)
                signal.signal(signal.SIGALRM, previous_handler)
        else:
            return serve(config, args.config, startup_timeout=args.startup_timeout,
                         fish_source=args.fish_source, sglang_source=args.sglang_source,
                         chapters=args.command != "serve-tts", maintenance=args.command == "serve-maintenance")
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
