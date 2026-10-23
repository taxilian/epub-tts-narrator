"""File-configured, stop-by-default Runpod worker lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import secrets
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from .runpod_api import RunpodAPI, RunpodAmbiguousOutcome, RunpodAPIError
from .diagnostics import configured_secrets, safe_error_text


def add_parser(commands):
    group = commands.add_parser("runpod", help="Bootstrap, update, and encode with a stop-by-default Runpod worker")
    actions = group.add_subparsers(dest="runpod_action", required=True)
    for action in ("bootstrap", "update", "encode", "status", "stop"):
        command = actions.add_parser(action)
        command.add_argument("--config", type=Path, help="Normal local TOML configuration")
        command.add_argument("--override-config", type=Path, help="Remote overrides (default: runpod.local.toml beside local config)")
        if action in {"bootstrap", "update"}:
            command.add_argument("--source", choices=("github", "local"), default="github",
                                 help="Deploy GitHub git_ref, or this checkout's committed HEAD")
        if action in {"bootstrap", "update", "encode"}:
            command.add_argument("--stop-delay", type=int, choices=range(0, 301), metavar="SECONDS", default=0,
                                 help="Wait up to 300 seconds before stopping on completion/error (default: immediately; Ctrl-C stops now)")
        if action == "encode":
            command.add_argument("bundle", type=Path)
            command.add_argument("--chapters", help="Manifest chapter numbers, e.g. 1-3,5")
            command.add_argument("--chapters-only", action="store_true")
            command.add_argument("--chapter-format", choices=("m4a", "mp3"))
            command.add_argument("--work-dir", type=Path)
            command.add_argument("-o", "--output", type=Path)
            command.add_argument("--epub", type=Path)


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def deployment_lock(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "operation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another Runpod operation is using this deployment; wait for it to finish") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ssh_endpoint(pod):
    host = pod.get("publicIp")
    mappings = pod.get("portMappings")
    if not isinstance(host, str) or not isinstance(mappings, dict):
        return None
    try:
        ipaddress.ip_address(host)
        supplied = mappings.get("22", 0)
        if type(supplied) not in (str, int) or (isinstance(supplied, str) and not supplied.isdigit()):
            return None
        port = int(supplied)
        if not 1 <= port <= 65535:
            return None
    except (ValueError, TypeError):
        return None
    return host, port


class Deployment:
    def __init__(self, settings, *, api=None, notify=print):
        self.settings = settings
        self.cfg = settings["deployment"]
        self.directory = Path(self.cfg["state_dir"])
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.directory / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.api = api or RunpodAPI(settings["runpod_api_key"])
        self.notify = notify
        self.endpoint = None
        self._replacement_attempted = False
        self.secrets = configured_secrets(settings["renderer_config"]) | {settings["runpod_api_key"]}
        self.secrets.update(configured_secrets(settings["override_raw"]))
        for source in (settings["renderer_config"], settings["override_raw"]):
            self.secrets.update(value for key, value in source.get("chapter_storage", {}).items()
                                if key in {"secret_access_key", "access_key_id"} and value)
        if self.cfg.get("pod_id"):
            if self.state.get("pod_id") and self.state["pod_id"] != self.cfg["pod_id"]:
                raise ValueError("Configured pod_id differs from saved deployment; use a separate state_dir")
            self.state["pod_id"] = self.cfg["pod_id"]
        if not self.state.get("deployment_id"):
            self.state["deployment_id"] = uuid.uuid4().hex[:12]
        if not self.state.get("api_key"):
            self.state["api_key"] = secrets.token_urlsafe(32)
        self.secrets.add(self.state["api_key"])
        self.save()

    def save(self):
        private_json(self.state_path, self.state)

    @property
    def pod_id(self):
        return self.state.get("pod_id")

    @property
    def url(self):
        return f"https://{self.pod_id}-{self.cfg['service_port']}.proxy.runpod.net"

    def _run(self, command, *, input=None, timeout=1800):
        # Captured output is redacted before display, including failed command
        # diagnostics. Credentials are sent only over SSH stdin, never argv.
        process = subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = queue.Queue()
        def read_output():
            for line in iter(process.stdout.readline, b""):
                lines.put(line)
            lines.put(None)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        def write_input():
            try:
                process.stdin.write(input)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if input is not None:
            threading.Thread(target=write_input, daemon=True).start()
        output = bytearray()
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Remote operation timed out; stopping the worker. Saved files remain resumable.")
                try:
                    line = lines.get(timeout=min(1, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if line is None:
                    break
                output.extend(line)
                # Keep diagnostics bounded even for a verbose package installer.
                if len(output) > 4 * 1024 * 1024:
                    del output[:-2 * 1024 * 1024]
                # Preserve protocol markers for launch(), without exposing
                # its repetitive internal process probes as progress output.
                if line.rstrip(b"\r\n") in {b"EPUB_TTS_WORKER_STATUS=RUNNING", b"EPUB_TTS_WORKER_STATUS=EXITED"}:
                    continue
                text = safe_error_text(line.decode(errors="replace"), self.secrets).rstrip()
                if text:
                    self.notify(text)
            process.wait(timeout=max(1, deadline - time.monotonic()))
            if process.returncode:
                raise RuntimeError(f"Worker command failed (exit {process.returncode}); see diagnostics above")
            return bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()

    def ssh(self, command, *, input=None, timeout=1800):
        if self.endpoint is None:
            raise RuntimeError("Worker SSH is not ready")
        host, port = self.endpoint
        return self._run(["ssh", "-T", "-i", str(self.directory / "id_ed25519"),
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={json.dumps(str(self.directory / 'known_hosts'))}",
            "-p", str(port), f"{self.cfg['ssh_user']}@{host}", command], input=input, timeout=timeout)

    def _public_key(self):
        key = self.directory / "id_ed25519"
        if not key.exists():
            self._run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "epub-tts-narrator-worker", "-f", str(key)])
        return key.with_suffix(".pub").read_text().strip()

    def quote(self):
        """Read live price/stock before creating either paid resource."""
        if not self.cfg["image"]:
            raise ValueError("Set runpod.image to a verified Runpod SSH image in the override file")
        rows = self.api.catalog_gpus(include="AVAILABILITY", product="POD", cloud="SECURE", count=1,
                                     minCudaVersion="13.0" if self.cfg["tts_backend"] == "sglang" else "12.9").get("gpus", [])
        gpu = next((row for row in rows if row.get("id") == self.cfg["gpu_type_id"]), None)
        if not gpu or gpu.get("availability") in {None, "NONE"}:
            raise ValueError(f"Configured GPU {self.cfg['gpu_type_id']} is unavailable; no resources were created")
        price = gpu.get("price", {}).get("secure")
        if (not isinstance(price, (int, float)) or price <= 0 or price > self.cfg["max_hourly_price"]):
            raise ValueError("Current GPU catalog price exceeds runpod.max_hourly_price or is unavailable")
        centers = {row["id"] for row in gpu.get("dataCenters", []) if row.get("availability") != "NONE"}
        center = self.cfg["data_center_id"]
        if not center or center not in centers:
            raise ValueError("Set runpod.data_center_id to an available datacenter for the configured GPU")
        self.notify(f"Runpod: {gpu.get('name', gpu['id'])} in {center}, catalog GPU rate ${price:.2f}/hour; persistent storage is billed separately.")
        return price

    def provision(self):
        if self.pod_id:
            return
        name = self.cfg["name"] + "-" + self.state["deployment_id"]
        # Reconcile a previous uncertain creation before making any mutation.
        if self.state.get("pod_creation_pending"):
            matches = [p for p in self.api.list_pods() if p.get("name") == name]
            if len(matches) != 1:
                raise RuntimeError("Previous Pod creation is unconfirmed; inspect Runpod and saved state before retrying. No replacement created.")
            self.state["pod_id"] = matches[0]["id"]
            self.state.pop("pod_creation_pending", None)
            self.save()
            return
        price = self.quote()
        volume_id = self.state.get("volume_id") or self.cfg.get("volume_id")
        if not volume_id:
            if self.state.get("volume_creation_pending"):
                matches = [v for v in self.api.list_volumes() if v.get("name") == name]
                if len(matches) != 1:
                    raise RuntimeError("Previous volume creation is unconfirmed; inspect Runpod before retrying. No duplicate created.")
                volume = matches[0]
            else:
                self.state["volume_creation_pending"] = True
                self.save()
                try:
                    volume = self.api.create_volume({"name": name, "size": self.cfg["volume_size_gb"],
                                                     "dataCenterId": self.cfg["data_center_id"]})
                except RunpodAPIError as exc:
                    if not exc.ambiguous:
                        self.state.pop("volume_creation_pending", None)
                        self.save()
                    raise
            volume_id = volume["id"]
            self.state["volume_id"] = volume_id
            self.state.pop("volume_creation_pending", None)
            self.save()
        else:
            volume = self.api.get_volume(volume_id)
            if volume.get("dataCenterId") != self.cfg["data_center_id"]:
                raise ValueError("Persistent volume must be in the selected GPU datacenter")
            self.state["volume_id"] = volume_id
            self.save()
        # v1 validates datacenter IDs against a stale enum. v2 accepts the
        # current catalog's IDs and retains explicit placement beside the
        # existing network volume. Secure on-demand Pods expose a public IP.
        spec = self._pod_spec(name, volume_id)
        self.state["pod_creation_pending"] = True
        self.save()
        try:
            pod = self.api.create_pod(spec)
        except RunpodAmbiguousOutcome:
            matches = [p for p in self.api.list_pods() if p.get("name") == name]
            if len(matches) != 1:
                raise
            pod = matches[0]
        except RunpodAPIError:
            self.state.pop("pod_creation_pending", None)
            self.save()
            raise
        self.state["pod_id"] = pod["id"]
        self.state.pop("pod_creation_pending", None)
        self.state["catalog_hourly_price"] = price
        self.save()
        # Provider pricing is authoritative; stop before installing if it differs
        # from the user's configured spending ceiling.
        actual = pod.get("cost", pod.get("costPerHr"))
        if actual is not None and float(actual) > self.cfg["max_hourly_price"]:
            raise RuntimeError("Created worker's rate exceeds configured cap; stopping immediately")
        self.notify(f"Created worker {self.pod_id}; persistent volume {volume_id}.")

    def _pod_spec(self, name, volume_id):
        return {"name": name, "image": self.cfg["image"], "cloud": "SECURE",
                "gpu": {"id": self.cfg["gpu_type_id"], "count": 1,
                        "minCudaVersion": "13.0" if self.cfg["tts_backend"] == "sglang" else "12.9"},
                "dataCenterIds": [self.cfg["data_center_id"]],
                "mounts": {"network": [{"volumeId": volume_id, "path": "/workspace"}]},
                "disk": self.cfg["container_disk_gb"], "ports": self.service_ports(),
                "env": {"PUBLIC_KEY": self._public_key()}, "cmd": self.startup_command()}

    @staticmethod
    def _check_replacement_placement(pod, spec, *, candidate):
        """Fail closed on missing placement data; a name alone cannot prove identity."""
        gpu, mounts = pod.get("gpu"), pod.get("mounts")
        if (not isinstance(gpu, dict) or not isinstance(mounts, dict)
                or gpu.get("id") != spec["gpu"]["id"] or type(gpu.get("count")) is not int
                or gpu["count"] != 1 or pod.get("cloud") != "SECURE"
                or pod.get("dataCenterId") != spec["dataCenterIds"][0]
                or mounts.get("network") != spec["mounts"]["network"]
                or mounts.get("persistent") or mounts.get("global")
                or pod.get("image") != spec["image"]
                or (candidate and pod.get("name") != spec["name"])):
            raise RuntimeError("Pod replacement placement does not match its saved GPU, datacenter, image, or existing volume; no worker adopted")

    def _check_replacement_source(self, intent):
        spec = intent["spec"]
        volume_id = spec["mounts"]["network"][0]["volumeId"]
        if (self.cfg.get("pod_id") or self.pod_id != intent["previous_pod_id"]
                or self.state.get("volume_id") != volume_id
                or (self.cfg.get("volume_id") and self.cfg["volume_id"] != volume_id)
                or self.cfg["gpu_type_id"] != spec["gpu"]["id"]
                or self.cfg["data_center_id"] != spec["dataCenterIds"][0]
                or self.cfg["image"] != spec["image"]):
            raise RuntimeError("Saved Pod replacement differs from the configured deployment; inspect saved state before continuing")
        volume = self.api.get_volume(volume_id)
        if volume.get("id") != volume_id or volume.get("dataCenterId") != spec["dataCenterIds"][0]:
            raise RuntimeError("Pod replacement requires the same existing network volume in the configured datacenter")
        previous = self.api.get_pod_details(intent["previous_pod_id"])
        if previous.get("status") not in {"EXITED", "STOPPED"}:
            raise RuntimeError("Previous Pod is not confirmed stopped; no replacement will be created or adopted")
        self._check_replacement_placement(previous, spec, candidate=False)

    def _replacement_candidate(self):
        """Reconcile a unique saved creation intent, never issue another POST."""
        intent = self.state["pod_replacement"]
        if not intent.get("candidate_id"):
            matches = [pod for pod in self.api.list_pods() if pod.get("name") == intent["spec"]["name"]]
            if len(matches) != 1 or not isinstance(matches[0].get("id"), str) or not matches[0]["id"]:
                raise RuntimeError("POD REPLACEMENT UNCONFIRMED: inspect Runpod for " + intent["spec"]["name"] +
                                   "; GPU billing may continue. No duplicate will be created automatically.")
            if matches[0]["id"] == intent["previous_pod_id"]:
                raise RuntimeError("Pod replacement resolved to the previous worker; inspect saved state")
            # This unpredictable name was persisted before our only POST. Save
            # its unique returned ID before validation so cleanup can stop it.
            intent["candidate_id"] = matches[0]["id"]
            self.save()
        if intent["candidate_id"] == intent["previous_pod_id"]:
            raise RuntimeError("Pod replacement resolved to the previous worker; inspect saved state")
        return intent["candidate_id"]

    def _adopt_replacement(self):
        intent = self.state["pod_replacement"]
        self._check_replacement_source(intent)
        candidate_id = self._replacement_candidate()
        pod = self.api.get_pod_details(candidate_id)
        self._check_replacement_placement(pod, intent["spec"], candidate=True)
        price = pod.get("cost")
        cap = min(intent["max_hourly_price"], self.cfg["max_hourly_price"])
        if type(price) not in (float, int) or not math.isfinite(price) or not 0 < price <= cap:
            raise RuntimeError("Replacement worker's rate is unavailable or exceeds the configured cap; stopping it")
        previous = intent["previous_pod_id"]
        retired = self.state.setdefault("previous_pod_ids", [])
        if previous not in retired:
            retired.append(previous)
        self.state["pod_id"] = candidate_id
        self.state["catalog_hourly_price"] = intent["catalog_hourly_price"]
        self.state.pop("pod_replacement")
        self.endpoint = None
        self.save()
        self.notify(f"Using replacement worker {candidate_id}; existing volume retained and previous worker {previous} remains stopped.")

    def _replace_capacity_blocked_pod(self):
        if self.cfg.get("pod_id"):
            raise RunpodAPIError("The configured runpod.pod_id has no free GPU on its host. It is fixed and will not be replaced automatically.",
                                 status=400, reason="host_capacity")
        if self._replacement_attempted:
            raise RunpodAPIError("Replacement worker also has no free GPU on its host; only one replacement is attempted per operation.",
                                 status=400, reason="host_capacity")
        self._replacement_attempted = True
        volume_id = self.state.get("volume_id")
        if not volume_id:
            raise RuntimeError("Capacity replacement requires a saved existing network volume; no new volume will be created")
        intent = {"previous_pod_id": self.pod_id,
                  "spec": self._pod_spec(self.cfg["name"] + "-" + self.state["deployment_id"] + "-r" + uuid.uuid4().hex[:12], volume_id),
                  "max_hourly_price": self.cfg["max_hourly_price"]}
        self._check_replacement_source(intent)
        intent["catalog_hourly_price"] = self.quote()
        self.state["pod_replacement"] = intent
        self.save()
        self.notify("The stopped worker's host has no free GPU. Creating one replacement in the same datacenter with the existing volume…")
        try:
            pod = self.api.create_pod(intent["spec"])
        except RunpodAmbiguousOutcome:
            pass  # Observe our durable unique name; never retry creation.
        except RunpodAPIError:
            self.state.pop("pod_replacement")
            self.save()
            raise
        else:
            intent["candidate_id"] = pod["id"]
            self.save()
        self._adopt_replacement()

    def service_ports(self):
        return ["22/tcp", f"{self.cfg['service_port']}/http"]

    @staticmethod
    def startup_command():
        # Network storage may restore permissive modes even when the original
        # keys were private. Fix the local copies before sshd reads them.
        startup = ("set -eu; keys=/workspace/.epub-tts-runtime/ssh-host-keys; "
                   "mkdir -p \"$keys\"; chmod 700 \"$keys\"; "
                   "if test -f \"$keys/ssh_host_ed25519_key\"; then cp -p \"$keys\"/ssh_host_* /etc/ssh/; "
                   "else ssh-keygen -A; cp -p /etc/ssh/ssh_host_* \"$keys\"/; fi; "
                   "chmod 600 /etc/ssh/ssh_host_*_key; chmod 644 /etc/ssh/ssh_host_*_key.pub; "
                   "exec /start.sh")
        return ["bash", "-lc", startup]

    def configure_runtime(self, pod):
        """Reconcile startup and service ports without replacing persistent storage."""
        ports, command = self.service_ports(), self.startup_command()
        changes = {}
        if set(pod.get("ports") or []) != set(ports):
            changes["ports"] = ports
        if pod.get("dockerStartCmd") != command:
            changes["cmd"] = command
        if not changes:
            return pod
        if pod.get("desiredStatus") not in {"EXITED", "STOPPED"}:
            self.stop()
        self.notify(f"Configuring worker startup and chapter service on port {self.cfg['service_port']}…")
        try:
            self.api.update_pod(self.pod_id, changes)
        except RunpodAmbiguousOutcome:
            pass  # Read back before deciding whether another operation is safe.
        updated = self.api.get_pod(self.pod_id)
        if (set(updated.get("ports") or []) != set(ports)
                or updated.get("dockerStartCmd") != command):
            raise RuntimeError("Worker runtime update was not confirmed; no replacement worker was created")
        if updated.get("desiredStatus") not in {"EXITED", "STOPPED"}:
            self.stop()
            updated = self.api.get_pod(self.pod_id)
        return updated

    def start(self):
        if not self.pod_id:
            raise ValueError("No saved worker; run runpod bootstrap first")
        if self.state.get("pod_replacement"):
            self._replacement_attempted = True
            self._adopt_replacement()
        # At most the original worker and one replacement. Capacity errors on
        # a recovered/replacement allocation cannot cause a creation loop.
        for _ in range(2):
            pod = self.api.get_pod(self.pod_id)
            actual = pod.get("costPerHr")
            if actual is not None and float(actual) > self.cfg["max_hourly_price"]:
                raise ValueError("Worker's current rate exceeds runpod.max_hourly_price")
            pod = self.configure_runtime(pod)
            if pod.get("desiredStatus") != "RUNNING":
                self.notify(f"Starting worker {self.pod_id}…")
                try:
                    self.api.start_pod(self.pod_id)
                except RunpodAmbiguousOutcome:
                    if self.api.get_pod(self.pod_id).get("desiredStatus") != "RUNNING":
                        raise
                except RunpodAPIError as exc:
                    if exc.reason != "host_capacity":
                        raise
                    self._replace_capacity_blocked_pod()
                    continue
            break
        self.notify("Waiting for the worker's SSH connection…")
        deadline = time.monotonic() + self.cfg["startup_timeout_seconds"]
        while time.monotonic() < deadline:
            pod = self.api.get_pod(self.pod_id)
            endpoint = ssh_endpoint(pod)
            if endpoint:
                self.endpoint = endpoint
                try:
                    self.ssh("true", timeout=15)
                except RuntimeError:
                    time.sleep(5)
                    continue
                # v1's accepted CUDA enum lags the live GPU catalog. Check
                # the allocated host rather than excluding newer drivers.
                minimum = (13, 0) if self.cfg["tts_backend"] == "sglang" else (12, 9)
                check = ("import re,subprocess; s=subprocess.check_output(['nvidia-smi'],text=True); "
                         "m=re.search(r'CUDA Version: (\\d+)\\.(\\d+)',s); "
                         f"assert m and tuple(map(int,m.groups())) >= {minimum!r}, "
                         f"'Worker driver must support CUDA {minimum[0]}.{minimum[1]} or newer'")
                self.ssh("python3 -c " + shlex.quote(check), timeout=30)
                return
            time.sleep(5)
        raise RuntimeError("Worker did not become ready before startup_timeout_seconds; GPU capacity may be unavailable")

    def stop(self):
        # A failed validation or lost creation response must not leave a paid
        # candidate running merely because the old Pod is still the active ID.
        targets, unresolved = [], None
        if self.state.get("pod_replacement"):
            try:
                targets.append(self._replacement_candidate())
            except Exception as exc:
                unresolved = exc
        if self.pod_id and self.pod_id not in targets:
            targets.append(self.pod_id)
        failures = []
        for pod_id in targets:
            try:
                self._stop_pod(pod_id)
            except Exception as exc:
                failures.append(safe_error_text(exc, self.secrets))
        if unresolved is not None:
            failures.append(safe_error_text(unresolved, self.secrets))
        if failures:
            raise RuntimeError("; ".join(failures))

    def _stop_pod(self, pod_id):
        self.notify(f"Stopping worker {pod_id}…")
        for attempt in range(3):
            pod = self.api.get_pod(pod_id)
            if pod.get("desiredStatus") in {"EXITED", "STOPPED", "TERMINATED"}:
                break
            try:
                self.api.stop_pod(pod_id)
            except RunpodAmbiguousOutcome:
                pass  # Observe status before any repeat stop request.
            time.sleep(3)
        else:
            pod = self.api.get_pod(pod_id)
        if pod.get("desiredStatus") not in {"EXITED", "STOPPED", "TERMINATED"}:
            raise RuntimeError(f"STOP NOT CONFIRMED for Runpod worker {pod_id}; GPU billing may continue. Stop it in the Runpod console.")
        if pod_id == self.pod_id:
            self.state["last_status"] = pod["desiredStatus"]
            self.save()
        self.notify("Runpod confirms the worker is stopped. Persistent storage remains billed.")

    def upload(self, local_path, remote_path):
        target = shlex.quote(remote_path)
        self.ssh(f"umask 077; cat > {target}.upload && chmod 600 {target}.upload && mv {target}.upload {target}",
                 input=Path(local_path).read_bytes())

    def deploy_code(self, source):
        app, url, ref = (self.cfg[key] for key in ("app_dir", "github_url", "git_ref"))
        q = shlex.quote
        self.notify("Installing the selected project revision…")
        self.ssh(f"test -d {q(app + '/.git')} || git clone -- {q(url)} {q(app)}")
        origin = self.ssh(f"git -C {q(app)} remote get-url origin").decode().strip()
        if origin.removesuffix(".git") != url.removesuffix(".git"):
            raise ValueError("Remote checkout origin differs from runpod.github_url; use a separate app_dir")
        dirty = self.ssh(f"git -C {q(app)} status --porcelain --untracked-files=no").decode().strip()
        if dirty:
            raise ValueError("Remote checkout has tracked changes; refusing to overwrite them")
        if source == "local":
            root = Path(__file__).resolve().parents[2]
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            with tempfile.TemporaryDirectory(dir=self.directory) as temp:
                bundle = Path(temp) / "source.bundle"
                self._run(["git", "-C", str(root), "bundle", "create", str(bundle), "HEAD"])
                self.upload(bundle, app + "/source.bundle")
            self.ssh(f"git -C {q(app)} fetch {q(app + '/source.bundle')} HEAD && git -C {q(app)} checkout --detach {q(revision)} && rm {q(app + '/source.bundle')}")
        else:
            if ref.startswith("-"):
                raise ValueError("git_ref must not start with a dash")
            self.ssh(f"git -C {q(app)} fetch -- origin {q(ref)} && git -C {q(app)} checkout --detach FETCH_HEAD")
            revision = self.ssh(f"git -C {q(app)} rev-parse HEAD").decode().strip()
        self.state["revision"] = revision
        self.state["bootstrapped"] = False
        self.save()

    def stop_service(self):
        """Replace only the supervisor recorded by this deployment, never pkill."""
        script = '''import os, pathlib, signal, sys, time
pidfile = pathlib.Path(sys.argv[1])
worker = sys.argv[2].encode()
if pidfile.exists():
    pid = int(pidfile.read_text())
    proc = pathlib.Path('/proc') / str(pid)
    def ours():
        try:
            return worker in (proc / 'cmdline').read_bytes().split(b'\\0')
        except FileNotFoundError:
            return False
    if ours():
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while ours() and time.monotonic() < deadline:
            time.sleep(1)
        if ours():
            raise SystemExit('Previous worker did not shut down; refusing to load another model')
    pidfile.unlink(missing_ok=True)
'''
        q = shlex.quote
        self.ssh(f"python3 - {q(self.cfg['app_dir'] + '/worker.pid')} {q(self.cfg['app_dir'] + '/deploy/runpod/worker.py')}",
                 input=script.encode(), timeout=90)

    def configure(self):
        from .runpod_config import build_remote_config, write_private_toml
        data = build_remote_config(self.settings["renderer_config"], self.settings["override_raw"],
                                   self.pod_id, self.state["api_key"], deployment=self.cfg)
        local = self.directory / "remote.toml"
        write_private_toml(local, data)
        self.upload(local, self.cfg["remote_config"])

    def install(self, *, full):
        self.notify("Installing persistent dependencies…" if full else "Checking the persistent runtime…")
        q = shlex.quote
        self.ssh(f"sh {q(self.cfg['app_dir'] + '/deploy/runpod/install.sh')} {q(self.cfg['app_dir'])} "
                 f"{'full' if full else 'runtime'} /workspace/.epub-tts-runtime {q(self.cfg['fish_source'])} "
                 f"{q(self.cfg['tts_backend'])} {q(self.cfg['sglang_source'])}", timeout=2400)

    def worker_command(self, action):
        q = shlex.quote
        return (f"{q(self.cfg['app_dir'] + '/.venv/bin/python')} -u "
                f"{q(self.cfg['app_dir'] + '/deploy/runpod/worker.py')} --config {q(self.cfg['remote_config'])} "
                f"--fish-source {q(self.cfg['fish_source'])} "
                f"--sglang-source {q(self.cfg['sglang_source'])} {action}")

    def launch(self, *, prepare=False):
        # Bootstrap/Update test the installed engine without resuming durable
        # chapter jobs. Encode is the only managed operation that runs them.
        action = "serve-maintenance" if prepare else "serve"
        if prepare:
            self.notify("Preparing model downloads on persistent storage (completed downloads are reused)…")
            self.ssh(self.worker_command("prepare"), timeout=3600)
        self.ssh(self.worker_command("check"))
        log = shlex.quote(self.cfg["app_dir"] + "/worker.log")
        pid = shlex.quote(self.cfg["app_dir"] + "/worker.pid")
        self.ssh(f"nohup {self.worker_command(action)} > {log} 2>&1 < /dev/null & echo $! > {pid}", timeout=30)
        self.notify("Loading Drama 3 and waiting for the authenticated chapter service…")
        from .chapter_client import ChapterClient, ChapterClientError
        client = ChapterClient({"url": self.url, "api_key": self.state["api_key"], "timeout_seconds": 15})
        started = time.monotonic()
        deadline = started + self.cfg["startup_timeout_seconds"]
        next_update = started + 30
        while time.monotonic() < deadline:
            try:
                capabilities = client.request("GET", "v1/capabilities")
                if capabilities.get("maintenance", False) is not prepare:
                    raise RuntimeError("Chapter service started in the wrong maintenance mode")
                self.notify(f"Chapter service ready ({max(0, time.monotonic() - started):.0f}s startup).")
                return
            except ChapterClientError:
                # A dead supervisor cannot recover merely by waiting for the
                # HTTP timeout. Surface its diagnostics and stop paid compute.
                check = '''import pathlib, sys
pidfile, worker = pathlib.Path(sys.argv[1]), sys.argv[2].encode()
try:
    proc = pathlib.Path('/proc') / str(int(pidfile.read_text()))
    running = worker in (proc / 'cmdline').read_bytes().split(b'\\0')
    running = running and (proc / 'stat').read_text().rpartition(') ')[2].split()[0] != 'Z'
except (OSError, ValueError, IndexError):
    running = False
print('EPUB_TTS_WORKER_STATUS=RUNNING' if running else 'EPUB_TTS_WORKER_STATUS=EXITED')
'''
                probe = self.ssh("python3 -c " + shlex.quote(check) + " " + pid + " " +
                                 shlex.quote(self.cfg["app_dir"] + "/deploy/runpod/worker.py"), timeout=30)
                # SSH diagnostics share this stream with the probe's stdout.
                # Only our exact status line identifies an exited supervisor.
                if b"EPUB_TTS_WORKER_STATUS=EXITED" in probe.splitlines():
                    self.ssh(f"tail -n 60 {log}")
                    raise RuntimeError("Chapter worker exited during startup; see diagnostics above")
                now = time.monotonic()
                if now >= next_update:
                    self.notify(f"Still waiting for Drama 3 and the chapter service ({max(0, now - started):.0f}s elapsed)…")
                    next_update = now + 30
                time.sleep(5)
        self.ssh(f"tail -n 60 {log}")
        raise RuntimeError("Chapter service failed to start before startup_timeout_seconds")

    def smoke(self):
        self.notify("Testing GPU synthesis, Whisper transcription, and compressed Ceph publication…")
        self.ssh(self.worker_command("smoke"), timeout=1200)
        self.state["bootstrapped"] = True
        self.save()

    def encode(self, args):
        bundle, config = self.prepare_encode(args)
        from .chapter_client import render_remote
        result = render_remote(bundle, config, notify=self.notify, chapters_only=args.chapters_only)
        if args.chapters_only:
            self.notify("Completed chapter audio:")
            for chapter in result["chapters"]:
                location = chapter.get("audio_file") or "empty chapter (no speech)"
                self.notify(f"  Chapter {chapter['number']} / {len(bundle['chapters'])}: {location}")
        else:
            self.notify(f"Completed audiobook: {config['render']['output_dir']}")
        return result

    def prepare_encode(self, args):
        from .runpod_config import build_client_config
        from .bundle import load_bundle
        from .book_context import resolve_book_config
        from .cli import chapter_numbers
        from .workflow import _preflight
        from .chapter_client import validate_remote_plan
        if args.chapter_format not in (None, "m4a"):
            raise ValueError("Runpod chapter rendering produces M4A; use --chapter-format m4a")
        config = build_client_config(self.settings["renderer_config"], self.url, self.state["api_key"],
                                     override_raw=self.settings["override_raw"])
        bundle = load_bundle(args.bundle)
        config = resolve_book_config(bundle, config, output=args.output, work_dir=args.work_dir, epub=args.epub)
        if args.chapters is not None:
            config["render"]["chapter_numbers"] = chapter_numbers(args.chapters, len(bundle["chapters"]))
        if args.chapter_format:
            config["render"]["chapter_format"] = args.chapter_format
        config, plan, store = _preflight(bundle, config, chapters_only=args.chapters_only, multi_speaker=True)
        missing = [voice_id for voice_id in plan["voice_ids"] if store.load(voice_id) is None]
        if missing:
            raise ValueError("Remote rendering requires saved voices; run cast first for: " + ", ".join(missing))
        validate_remote_plan(bundle, config, plan)
        # encode() hands this to render_remote, which then skips its own preflight.
        config["_validated_remote_plan"] = (plan, store)
        return bundle, config


def run_command(args):
    from .runpod_config import load_settings, build_remote_config
    settings = load_settings(args.config, args.override_config)
    directory = Path(settings["deployment"]["state_dir"])
    with deployment_lock(directory):
        deployment = Deployment(settings)
        if args.runpod_action == "status":
            pod = deployment.api.get_pod(deployment.pod_id) if deployment.pod_id else {}
            summary = {"pod_id": deployment.pod_id, "status": pod.get("desiredStatus", "not created"),
                       "volume_id": deployment.state.get("volume_id"), "revision": deployment.state.get("revision"),
                       "bootstrapped": deployment.state.get("bootstrapped", False)}
            if deployment.state.get("pod_replacement"):
                intent = deployment.state["pod_replacement"]
                summary["pending_replacement"] = {"name": intent["spec"]["name"],
                                                   "pod_id": intent.get("candidate_id"),
                                                   "status": "unconfirmed; may still be running"}
            print(json.dumps(summary, indent=2))
            return 0
        if args.runpod_action == "stop":
            deployment.stop()
            return 0
        if args.runpod_action != "bootstrap" and not deployment.pod_id:
            raise ValueError("No saved worker; run runpod bootstrap first")
        if args.runpod_action == "encode" and not deployment.state.get("bootstrapped"):
            raise ValueError("Worker has not passed its smoke test; run runpod bootstrap or update first")
        # Fail configuration checks before allocating or starting paid compute.
        build_remote_config(settings["renderer_config"], settings["override_raw"],
                            deployment.pod_id or "preflight", deployment.state["api_key"],
                            deployment=settings["deployment"])
        if args.runpod_action == "encode":
            deployment.prepare_encode(args)
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        interrupted = False
        operation_error = None
        try:
            if args.runpod_action == "bootstrap":
                deployment.provision()
            deployment.start()
            deployment.stop_service()
            if args.runpod_action in {"bootstrap", "update"}:
                deployment.deploy_code(args.source)
            deployment.configure()
            deployment.install(full=args.runpod_action != "encode")
            deployment.launch(prepare=args.runpod_action != "encode")
            if args.runpod_action == "encode":
                deployment.encode(args)
            else:
                deployment.smoke()
            return 0
        except KeyboardInterrupt:
            interrupted = True
            raise
        except Exception as exc:
            operation_error = exc
            raise
        finally:
            try:
                if deployment.pod_id:
                    # A failed operation skips the delay; paid time is not spent beside stopped work.
                    if args.stop_delay and not interrupted and operation_error is None:
                        print(f"Stopping in {args.stop_delay} seconds; press Ctrl-C to stop now.", flush=True)
                        try:
                            time.sleep(args.stop_delay)
                        except KeyboardInterrupt:
                            pass
                    saved_interrupt = signal.getsignal(signal.SIGINT)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    try:
                        deployment.stop()
                    except Exception as exc:
                        message = ("Could not confirm shutdown of the Runpod deployment; GPU billing may continue. "
                                   "Stop it in the Runpod console. " + safe_error_text(exc, deployment.secrets))
                        if operation_error is not None:
                            message += (f" Operation failure ({type(operation_error).__name__}): "
                                        + safe_error_text(operation_error, deployment.secrets))
                        raise RuntimeError(message) from None
                    finally:
                        signal.signal(signal.SIGINT, saved_interrupt)
                elif deployment.state.get("pod_creation_pending"):
                    print("POD CREATION UNCONFIRMED: GPU billing may continue. Inspect Runpod for "
                          f"{deployment.cfg['name']}-{deployment.state['deployment_id']}; no duplicate will be created automatically.", flush=True)
            finally:
                signal.signal(signal.SIGTERM, previous)
