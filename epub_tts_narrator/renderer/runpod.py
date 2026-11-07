"""File-configured, stop-by-default Runpod worker lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .runpod_api import RunpodAPI, RunpodAmbiguousOutcome, RunpodAPIError
from .diagnostics import configured_secrets, safe_error_text
from .runpod_gpu import CLOUD_LABELS, H100_CANDIDATE_IDS, PREFERRED_H100_HOURLY_LIMIT, select_placement


WORKSPACE = "/workspace"


def _local_head():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def add_parser(commands):
    group = commands.add_parser("runpod", help="Bootstrap, update, and encode with an idle-shutdown Runpod worker")
    actions = group.add_subparsers(dest="runpod_action", required=True)
    for action in ("bootstrap", "update", "encode", "status", "stop", "progress", "cleanup"):
        command = actions.add_parser(action)
        if action == "cleanup":
            command.add_argument("--yes", action="store_true",
                                 help="Delete without the interactive confirmation")
        command.add_argument("--config", type=Path, help="Normal local TOML configuration")
        command.add_argument("--override-config", type=Path, help="Remote overrides (default: runpod.local.toml beside local config)")
        if action in {"bootstrap", "update"}:
            command.add_argument("--source", choices=("github", "local"), default="github",
                                 help="Deploy GitHub git_ref, or this checkout's committed HEAD")
        if action in {"bootstrap", "update", "encode"}:
            command.add_argument("--stop-delay", type=int, choices=range(0, 301), metavar="SECONDS", default=0,
                                 help="Deprecated: shutdown is controlled by runpod.idle_timeout_seconds")
        if action == "encode":
            command.add_argument("bundle", type=Path)
            command.add_argument("--chapters", help="Manifest chapter numbers, e.g. 1-3,5")
            command.add_argument("--chapters-only", action="store_true")
            command.add_argument("--chapter-format", choices=("m4a", "mp3"))
            command.add_argument("--work-dir", type=Path)
            command.add_argument("-o", "--output", type=Path)
            command.add_argument("--epub", type=Path)
            command.add_argument("--no-update", action="store_true",
                                 help="Encode with the deployed code as-is; by default a stale worker is updated first without stopping")
        if action == "progress":
            command.add_argument("--job", help="Show only this chapter job ID")
            command.add_argument("--watch", action="store_true", help="Follow progress until Ctrl-C without resuming jobs")
            command.add_argument("--poll-seconds", type=float, default=2.0,
                                 help="Seconds between refreshes while watching (default: 2)")
            command.add_argument("--limit", type=int, default=100,
                                 help="Maximum number of queued jobs to show (default: 100)")
            command.add_argument("--json", action="store_true", help="Print machine-readable JSON")


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


def _mounts_network_volume(spec):
    """True for an older request or Pod that mounts a network volume."""
    mounts = spec.get("mounts") if isinstance(spec, dict) else None
    return isinstance(mounts, dict) and isinstance(mounts.get("network"), list) and bool(mounts["network"])


def _mounts_match(actual, expected):
    """Compare a Pod's storage with its request: one network volume or one Pod disk."""
    if expected.get("network"):
        return actual.get("network") == expected["network"] and not actual.get("persistent")
    disk, wanted = actual.get("persistent"), expected.get("persistent")
    return (isinstance(disk, dict) and isinstance(wanted, dict) and not actual.get("network")
            and disk.get("path") == wanted["path"] and disk.get("size") == wanted["size"])


def _resource_missing(exc):
    """A confirmed 404 answer: the Runpod resource no longer exists."""
    return isinstance(exc, RunpodAPIError) and exc.status == 404


class Deployment:
    def __init__(self, settings, *, api=None, notify=print):
        self.settings = settings
        self.cfg = settings["deployment"]
        self.directory = Path(self.cfg["state_dir"])
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.directory / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        # The saved identity is the cloud, GPU, and region of the current
        # worker; cfg holds only the selection fallback, the cloud preference
        # order, and an optional location hint. A chosen placement never
        # rewrites them. Workers saved before cloud selection are Secure.
        self.gpu_type_id = self.state.get("gpu_type_id") or self.cfg["gpu_type_id"]
        self.data_center_id = self.state.get("data_center_id")
        self.cloud_types = tuple(cloud.upper() for cloud in self.cfg["cloud_types"])
        self.cloud = self.state.get("cloud") or "SECURE"
        self.api = api or RunpodAPI(settings["runpod_api_key"])
        self.notify = notify
        self.endpoint = None
        self._replacement_attempted = False
        self._cloud_fallback_attempted = False
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

    def quote(self, *, clouds=None):
        """Read live price/stock and fix the concrete cloud, GPU, and US datacenter.

        Clouds are tried in runpod.cloud_types order (or the given subset);
        the first with an eligible placement wins. Within a cloud the
        configured data_center_id is only a tie-break hint: empty, stale, or
        non-US hints are ignored, never errors.
        """
        if not self.cfg["image"]:
            raise ValueError("Set runpod.image to a verified Runpod SSH image in the override file")
        failures = []
        for cloud in clouds or self.cloud_types:
            rows = self.api.catalog_gpus(include="AVAILABILITY", product="POD", cloud=cloud, count=1,
                                         minCudaVersion="12.9").get("gpus", [])
            try:
                placement = select_placement(rows, cloud=cloud, fallback_id=self.cfg["gpu_type_id"],
                                             prefer_h100=self.cfg["prefer_h100"],
                                             max_hourly_price=self.cfg["max_hourly_price"],
                                             preferred_data_center=self.cfg["data_center_id"])
            except ValueError as exc:
                failures.append(exc)
                continue
            self.cloud = placement.cloud
            self.gpu_type_id = placement.gpu_type_id
            self.data_center_id = placement.data_center_id
            hint = self.cfg["data_center_id"]
            if hint and hint != placement.data_center_id:
                self.notify(f"runpod.data_center_id {hint} is only a hint; selected {placement.data_center_id} "
                            "by GPU availability, price, and stock.")
            gpu = next((row for row in rows if isinstance(row, dict) and row.get("id") == placement.gpu_type_id), {})
            preferred = "preferred H100 " if placement.gpu_type_id in H100_CANDIDATE_IDS else ""
            self.notify(f"Runpod {CLOUD_LABELS[self.cloud]}: {gpu.get('name', placement.gpu_type_id)} in "
                        f"{placement.data_center_id}, catalog {preferred}rate ${placement.hourly_price:.2f}/hour; "
                        "the Pod disk is billed separately.")
            return placement.hourly_price
        if len(failures) == 1:
            raise failures[0]
        raise ValueError("No cloud in runpod.cloud_types has an eligible placement; no resources were created. "
                         + " ".join(str(failure) for failure in failures))

    def _later_clouds(self):
        """Clouds after the current one in the configured preference order."""
        order = self.cloud_types
        return order[order.index(self.cloud) + 1:] if self.cloud in order else ()

    def _fallback_clouds(self, exc):
        """One fallback per operation, only after a confirmed placement refusal.

        Runpod reports no capacity (400) and an inaccessible pool (403)
        without a machine-readable code; both confirm that nothing was
        created, so one request on the next preferred cloud cannot duplicate
        a worker. Ambiguous outcomes and other errors never fall back.
        """
        if exc.ambiguous or exc.status not in {400, 403} or self._cloud_fallback_attempted:
            return ()
        remaining = self._later_clouds()
        if remaining:
            self._cloud_fallback_attempted = True
        return remaining

    def _retire_network_volume(self):
        """Move a legacy network volume to the retained list; never delete it."""
        volume_id = self.state.pop("volume_id", None)
        if volume_id:
            retained = self.state.setdefault("retained_network_volumes", [])
            if volume_id not in retained:
                retained.append(volume_id)
            self.notify(f"Network volume {volume_id} is no longer used; it stays in Runpod and keeps billing "
                        "for storage until you delete it in the Runpod console.")

    def provision(self):
        if self.state.get("pod_replacement"):
            self._adopt_replacement()
            return
        if self.pod_id and not self.state.get("pod_creation_pending"):
            try:
                self.api.get_pod(self.pod_id)
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    # 401/403/5xx and transport failures never prove absence;
                    # no resource is created or replaced on an uncertain API.
                    raise
                if self.cfg.get("pod_id"):
                    raise ValueError(
                        f"The configured worker {self.pod_id} does not exist in Runpod; "
                        "correct or remove runpod.pod_id in the configuration.") from None
                # The saved worker was deleted (for example in the Runpod
                # console) together with its Pod disk; create a new one.
                self.notify(f"Saved worker {self.pod_id} no longer exists in Runpod; creating a new one.")
                self.state.pop("pod_id", None)
                self.state.pop("last_status", None)
                self.save()
            else:
                # A healthy saved worker is never replaced merely to change
                # its cloud, GPU, region, or storage type.
                return
        allocation = self.state.get("allocation")
        saved_spec = self.state.get("pod_creation_spec")
        name = (saved_spec["name"] if isinstance(saved_spec, dict) else
                allocation.get("pod_name") if isinstance(allocation, dict) else None)
        name = name or self.cfg["name"] + "-" + self.state["deployment_id"]
        # Reconcile a previous uncertain creation before making any mutation.
        if self.state.get("pod_creation_pending"):
            matches = [p for p in self.api.list_pods() if p.get("name") == name]
            if len(matches) > 1:
                raise RuntimeError("Previous Pod creation is unconfirmed; multiple Pods share its name. "
                                   "Inspect Runpod and delete the duplicates before retrying. No replacement created.")
            if matches:
                self.state["pod_id"] = matches[0]["id"]
                self.save()
                if isinstance(allocation, dict):
                    self._restore_allocation(allocation)
                self._validate_recovered_creation(matches[0])
                if isinstance(saved_spec, dict):
                    self.data_center_id = saved_spec["dataCenterIds"][0]
                    self.cloud = saved_spec.get("cloud", "SECURE")
                    self.state["data_center_id"] = self.data_center_id
                self._adopt_new_worker(saved_spec)
                self.state.pop("pod_creation_pending", None)
                self.state.pop("pod_creation_spec", None)
                self.state.pop("allocation", None)
                self.save()
                return
            # The inventory shows the unconfirmed creation made nothing, so a
            # retry cannot duplicate it.
            self.notify("The unconfirmed worker creation made nothing; creating a new worker.")
            self.notify("Repeated unconfirmed creations mean Runpod's service is failing; check the "
                        "Runpod console or try a different runpod.data_center_id.")
            self.state.pop("pod_creation_pending", None)
            self.save()
        if isinstance(saved_spec, dict) and _mounts_network_volume(saved_spec):
            # An older request would mount a network volume. Nothing was
            # created for it, so select a new Pod-disk request instead.
            self.notify("Discarding an earlier unsent worker request that used a network volume; "
                        "workers now use a host-local Pod disk.")
            saved_spec = allocation = None
            self.state.pop("pod_creation_spec", None)
            self.state.pop("allocation", None)
            self.save()
        clouds = None
        while True:
            # Read the live price and fix the concrete cloud, GPU, and US
            # datacenter before creating the paid resource; an unavailable or
            # over-budget catalog must create nothing.
            if isinstance(allocation, dict):
                price = self._restore_allocation(allocation)
            elif isinstance(saved_spec, dict):
                self.gpu_type_id = saved_spec["gpu"]["id"]
                self.data_center_id = saved_spec["dataCenterIds"][0]
                self.cloud = saved_spec.get("cloud", "SECURE")
                price = self.state.get("catalog_hourly_price", self.cfg["max_hourly_price"])
                self._save_allocation(name, price)
            else:
                price = self.quote(clouds=clouds)
                self._save_allocation(name, price)
            spec = saved_spec if isinstance(saved_spec, dict) else self._pod_spec(name)
            if spec["dataCenterIds"] != [self.data_center_id] or spec.get("cloud", "SECURE") != self.cloud:
                raise RuntimeError("The saved worker request disagrees with its saved allocation")
            # The chosen identity is durable before the only POST, so an
            # ambiguous create or restart can never re-resolve it.
            self.state["gpu_type_id"] = self.gpu_type_id
            self.state["data_center_id"] = self.data_center_id
            self.state["pod_creation_spec"] = spec
            self.state["catalog_hourly_price"] = price
            self.state["pod_creation_pending"] = True
            self.save()
            try:
                pod = self.api.create_pod(spec)
            except RunpodAmbiguousOutcome:
                matches = [p for p in self.api.list_pods() if p.get("name") == name]
                if len(matches) != 1:
                    raise
                pod = matches[0]
                self.state["pod_id"] = pod["id"]
                self.save()
                self._validate_recovered_creation(pod)
            except RunpodAPIError as exc:
                self.state.pop("pod_creation_pending", None)
                self.state.pop("pod_creation_spec", None)
                self.state.pop("allocation", None)
                self.save()
                fallback = self._fallback_clouds(exc)
                if not fallback:
                    raise
                self.notify(f"Runpod {CLOUD_LABELS[self.cloud]} refused the worker request "
                            f"(HTTP {exc.status}); trying {CLOUD_LABELS[fallback[0]]} once.")
                clouds, allocation, saved_spec = fallback, None, None
                continue
            break
        self.state["pod_id"] = pod["id"]
        self._adopt_new_worker(spec)
        self.state.pop("pod_creation_pending", None)
        self.state.pop("pod_creation_spec", None)
        self.state["catalog_hourly_price"] = price
        cap = self._creation_cap()
        self.state.pop("allocation", None)
        self.save()
        # Provider pricing is authoritative; stop before installing if it differs
        # from the user's configured spending ceiling.
        if self._actual_rate_over_limit(pod.get("cost", pod.get("costPerHr")),
                                        cap=cap, gpu_id=self.gpu_type_id):
            raise RuntimeError("Created worker's rate exceeds configured cap; stopping immediately")
        self.notify(f"Created worker {self.pod_id} on Runpod {CLOUD_LABELS[self.cloud]} with an empty "
                    f"{self.cfg['volume_size_gb']} GB Pod disk; it needs a full setup before encoding.")

    def _adopt_new_worker(self, spec):
        """Record a newly created worker; its empty disk needs a full setup."""
        self.state["cloud"] = self.cloud
        if not (isinstance(spec, dict) and _mounts_network_volume(spec)):
            self._retire_network_volume()
        # A new Pod disk has none of the saved revision, models, or smoke
        # result; the marker clears only after a successful smoke check.
        self.require_bootstrap(save=False)

    def _save_allocation(self, name, price):
        if not isinstance(self.data_center_id, str) or not self.data_center_id.startswith("US-"):
            raise RuntimeError("A saved allocation outside the US cannot authorize a new worker request")
        self.state["allocation"] = {"pod_name": name, "gpu_type_id": self.gpu_type_id,
                                    "data_center_id": self.data_center_id, "cloud": self.cloud,
                                    "catalog_hourly_price": price,
                                    "max_hourly_price": self.cfg["max_hourly_price"], "image": self.cfg["image"],
                                    "fallback_gpu_id": self.cfg["gpu_type_id"], "prefer_h100": self.cfg["prefer_h100"]}
        self.save()

    def _restore_allocation(self, allocation):
        cloud = allocation.get("cloud", "SECURE")
        if (allocation.get("image") != self.cfg["image"]
                or allocation.get("fallback_gpu_id") != self.cfg["gpu_type_id"]
                or allocation.get("prefer_h100") != self.cfg["prefer_h100"]
                or cloud not in self.cloud_types):
            raise RuntimeError("Saved allocation policy differs from the configured deployment; inspect saved state")
        self.data_center_id = allocation["data_center_id"]
        self.gpu_type_id = allocation["gpu_type_id"]
        self.cloud = cloud
        if not isinstance(self.data_center_id, str) or not self.data_center_id.startswith("US-"):
            raise RuntimeError("Saved allocation is not in a US datacenter")
        price = allocation["catalog_hourly_price"]
        if self._actual_rate_over_limit(price, cap=self._creation_cap(), gpu_id=self.gpu_type_id, required=True):
            raise ValueError("The recorded allocation price exceeds the current spending policy")
        return price

    def _creation_cap(self):
        allocation = self.state.get("allocation", {})
        saved = allocation.get("max_hourly_price", self.cfg["max_hourly_price"])
        if type(saved) not in (int, float) or not math.isfinite(saved) or saved <= 0:
            raise RuntimeError("Saved allocation has an invalid hourly cap")
        return min(saved, self.cfg["max_hourly_price"])

    def require_bootstrap(self, *, save=True):
        """Durably mark new storage: full setup and smoke before any reuse."""
        self.state["bootstrap_required"] = True
        self.state.pop("revision", None)
        self.state.pop("bootstrapped", None)
        if save:
            self.save()

    def _validate_recovered_creation(self, pod):
        spec = self.state.get("pod_creation_spec")
        if spec is not None:
            if not spec["dataCenterIds"][0].startswith("US-"):
                raise RuntimeError("The recovered worker request is outside the US and cannot be used")
            pod = self.api.get_pod_details(pod["id"])
            self._check_replacement_placement(pod, spec, candidate=True)
        if self._actual_rate_over_limit(pod.get("cost", pod.get("costPerHr")),
                                        cap=self._creation_cap(), gpu_id=self.gpu_type_id,
                                        required=spec is not None):
            raise RuntimeError("Created worker's rate exceeds configured cap or is unavailable; stopping immediately")

    def _pod_spec(self, name):
        # Secure and Community workers both keep /workspace on a host-local
        # Pod disk: it survives stop/start on the same host and is discarded
        # with the Pod. Network volumes are Secure-only and are not used.
        return {"name": name, "image": self.cfg["image"], "cloud": self.cloud,
                "gpu": {"id": self.gpu_type_id, "count": 1,
                        "minCudaVersion": "12.9"},
                "dataCenterIds": [self.data_center_id],
                "mounts": {"persistent": {"size": self.cfg["volume_size_gb"], "path": WORKSPACE}},
                "disk": self.cfg["container_disk_gb"], "ports": self.service_ports(),
                "env": {"PUBLIC_KEY": self._public_key()}, "cmd": self.startup_command()}

    def _actual_rate_over_limit(self, actual, *, cap, gpu_id, required=False, prefer_h100=None):
        """True when the provider's authoritative rate is unusable for this GPU.

        A preferred H100 must be strictly under the shared hourly preference
        limit, not merely under the configured cap. Missing data is tolerated
        only when required=False; malformed or non-finite data always fails.
        """
        if actual is None:
            return required
        if isinstance(actual, bool):
            return True
        try:
            rate = float(actual)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(rate):
            return True
        preference = self.cfg["prefer_h100"] if prefer_h100 is None else prefer_h100
        if preference and gpu_id in H100_CANDIDATE_IDS:
            return not 0 < rate < PREFERRED_H100_HOURLY_LIMIT or rate > cap
        return not 0 < rate <= cap

    @staticmethod
    def _check_replacement_placement(pod, spec, *, candidate, gpu_id=None):
        """Fail closed on missing placement data; a name alone cannot prove identity."""
        expected_gpu = spec["gpu"]["id"] if gpu_id is None else gpu_id
        gpu, mounts = pod.get("gpu"), pod.get("mounts")
        if (not isinstance(gpu, dict) or not isinstance(mounts, dict)
                or gpu.get("id") != expected_gpu or type(gpu.get("count")) is not int
                or gpu["count"] != 1 or pod.get("cloud") != spec.get("cloud", "SECURE")
                or pod.get("dataCenterId") != spec["dataCenterIds"][0]
                or not _mounts_match(mounts, spec["mounts"]) or mounts.get("global")
                or pod.get("image") != spec["image"]
                or (candidate and pod.get("name") != spec["name"])):
            raise RuntimeError("Pod replacement placement does not match its saved cloud, GPU, datacenter, image, or storage; no worker adopted")

    @staticmethod
    def _source_spec(previous, source_gpu_id):
        """Immutable placement snapshot of the old Pod, independent of the target."""
        gpu, mounts = previous.get("gpu"), previous.get("mounts")
        dc, image, cloud = previous.get("dataCenterId"), previous.get("image"), previous.get("cloud")
        if (not isinstance(gpu, dict) or gpu.get("id") != source_gpu_id or gpu.get("count") != 1
                or not isinstance(mounts, dict)
                or not (_mounts_network_volume({"mounts": mounts}) or isinstance(mounts.get("persistent"), dict))
                or not isinstance(dc, str) or not dc or cloud not in CLOUD_LABELS
                or not isinstance(image, str) or not image):
            raise RuntimeError("Runpod returned incomplete placement for the stopped worker; "
                               "no replacement will be created")
        return {"gpu": {"id": source_gpu_id, "count": 1}, "dataCenterIds": [dc], "cloud": cloud,
                "mounts": deepcopy(mounts), "image": image}

    def _check_replacement_source(self, intent, previous=None):
        """Validate the old worker against its immutable source snapshot; a
        changed location hint or cloud preference is never a mismatch."""
        source = intent.get("source_spec")
        spec = intent["spec"] if isinstance(intent.get("spec"), dict) else source
        if (self.cfg.get("pod_id") or self.pod_id != intent["previous_pod_id"]
                or self.cfg["gpu_type_id"] != intent.get("fallback_gpu_id", spec["gpu"]["id"])
                or ("prefer_h100" in intent and self.cfg["prefer_h100"] != intent["prefer_h100"])
                or self.cfg["image"] != spec["image"]):
            raise RuntimeError("Saved Pod replacement differs from the configured deployment; inspect saved state before continuing")
        if previous is None:
            try:
                previous = self.api.get_pod_details(intent["previous_pod_id"])
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                return
        if previous.get("status") not in {"EXITED", "STOPPED"}:
            raise RuntimeError("Previous Pod is not confirmed stopped; no replacement will be created or adopted")
        # The old Pod is validated against its immutable source placement even
        # when the replacement target selects a different cloud, region, or GPU.
        self._check_replacement_placement(previous, source or spec, candidate=False,
                                          gpu_id=intent.get("source_gpu_id", spec["gpu"]["id"]))

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

    def _prepare_replacement(self, intent):
        spec = intent.get("spec")
        if isinstance(spec, dict):
            if intent.get("creation_attempted") is not False or not _mounts_network_volume(spec):
                return
            # An older unsent request would mount a network volume; rebuild
            # it as a Pod-disk request for the same recorded target.
            intent.pop("spec")
        intent.setdefault("creation_attempted", False)
        intent.setdefault("pod_name", self.cfg["name"] + "-" + self.state["deployment_id"] + "-r" + uuid.uuid4().hex[:12])
        self.gpu_type_id = intent["target_gpu_type_id"]
        self.data_center_id = intent["target_data_center_id"]
        self.cloud = intent.get("target_cloud", "SECURE")
        if self.cloud not in self.cloud_types:
            raise RuntimeError("Saved Pod replacement targets a cloud that runpod.cloud_types no longer allows; "
                               "inspect saved state before continuing")
        cap = min(intent["max_hourly_price"], self.cfg["max_hourly_price"])
        if self._actual_rate_over_limit(intent["catalog_hourly_price"], cap=cap,
                                        gpu_id=self.gpu_type_id, required=True,
                                        prefer_h100=intent.get("prefer_h100", False)):
            raise ValueError("Saved replacement price exceeds the current spending policy")
        intent["spec"] = self._pod_spec(intent["pod_name"])
        self.save()

    def _post_replacement(self, intent):
        intent["creation_attempted"] = True
        self.save()
        try:
            pod = self.api.create_pod(intent["spec"])
        except RunpodAmbiguousOutcome:
            return
        except RunpodAPIError:
            self.state.pop("pod_replacement", None)
            self.save()
            raise
        intent["candidate_id"] = pod["id"]
        self.save()

    def _adopt_replacement(self):
        intent = self.state["pod_replacement"]
        self._check_replacement_source(intent)
        self._prepare_replacement(intent)
        if intent.get("creation_attempted") is False:
            self._post_replacement(intent)
        candidate_id = self._replacement_candidate()
        pod = self.api.get_pod_details(candidate_id)
        self._check_replacement_placement(pod, intent["spec"], candidate=True)
        cap = min(intent["max_hourly_price"], self.cfg["max_hourly_price"])
        if self._actual_rate_over_limit(pod.get("cost"), cap=cap,
                                        gpu_id=intent["spec"]["gpu"]["id"], required=True,
                                        prefer_h100=intent.get("prefer_h100", False)):
            raise RuntimeError("Replacement worker's rate is unavailable or exceeds the configured cap; stopping it")
        previous = intent["previous_pod_id"]
        retired = self.state.setdefault("previous_pod_ids", [])
        if previous not in retired:
            retired.append(previous)
        spec = intent["spec"]
        # Atomic switch of worker, cloud, GPU, and region in one save; the
        # previous Pod ID is retained for accounting.
        self.state["pod_id"] = candidate_id
        self.gpu_type_id = spec["gpu"]["id"]
        self.state["gpu_type_id"] = self.gpu_type_id
        self.data_center_id = spec["dataCenterIds"][0]
        self.state["data_center_id"] = self.data_center_id
        self.cloud = spec.get("cloud", "SECURE")
        self.state["catalog_hourly_price"] = intent["catalog_hourly_price"]
        self._adopt_new_worker(spec)
        self.state.pop("pod_replacement")
        self.endpoint = None
        self.save()
        self.notify(f"Using replacement worker {candidate_id} on Runpod {CLOUD_LABELS[self.cloud]}; its empty "
                    "Pod disk needs a full setup before encoding. Previous worker "
                    f"{previous} remains stopped, and its disk keeps billing until it is deleted (runpod cleanup deletes "
                    "every Pod this deployment created).")

    _REPLACEMENT_REASONS = {
        "host_capacity": ("The configured runpod.pod_id has no free GPU on its host. It is fixed and will not be replaced automatically.",
                          "Replacement worker also has no free GPU on its host; only one replacement is attempted per operation.",
                          "The stopped worker's host has no free GPU. Creating one replacement worker…"),
        "no_direct_ssh": ("The configured runpod.pod_id never published a direct SSH port. It is fixed and will not be replaced automatically.",
                          "Replacement worker also never published a direct SSH port; only one replacement is attempted per operation.",
                          "The Community Cloud worker never published a direct SSH port. Creating one replacement worker…"),
    }

    def _replace_capacity_blocked_pod(self, *, reason="host_capacity", clouds=None):
        pinned, repeated, starting = self._REPLACEMENT_REASONS[reason]
        if self.cfg.get("pod_id"):
            raise RunpodAPIError(pinned, status=400, reason=reason)
        if self._replacement_attempted:
            raise RunpodAPIError(repeated, status=400, reason=reason)
        self._replacement_attempted = True
        previous = self.api.get_pod_details(self.pod_id)
        source_gpu_id = self.state.get("gpu_type_id") or previous.get("gpu", {}).get("id")
        allowed = H100_CANDIDATE_IDS | {self.cfg["gpu_type_id"]} if self.cfg["prefer_h100"] else {self.cfg["gpu_type_id"]}
        if not isinstance(source_gpu_id, str) or source_gpu_id not in allowed:
            raise RuntimeError("Previous worker's GPU does not match the configured deployment")
        source = self._source_spec(previous, source_gpu_id)
        intent = {"previous_pod_id": self.pod_id,
                  "source_gpu_id": source_gpu_id,
                  "fallback_gpu_id": self.cfg["gpu_type_id"],
                  "prefer_h100": bool(self.cfg["prefer_h100"]),
                  "source_spec": source,
                  "max_hourly_price": self.cfg["max_hourly_price"]}
        self._check_replacement_source(intent, previous)
        price = self.quote(clouds=clouds)
        intent.update(target_data_center_id=self.data_center_id, target_gpu_type_id=self.gpu_type_id,
                      target_cloud=self.cloud, catalog_hourly_price=price, creation_attempted=False,
                      pod_name=self.cfg["name"] + "-" + self.state["deployment_id"] + "-r" + uuid.uuid4().hex[:12])
        self.state["pod_replacement"] = intent
        self.save()
        self.notify(starting)
        self._adopt_replacement()

    def service_ports(self):
        return ["22/tcp", f"{self.cfg['service_port']}/http"]

    @staticmethod
    def startup_command():
        # Persistent storage may restore permissive modes even when the original
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

    def _require_pod(self):
        try:
            return self.api.get_pod(self.pod_id)
        except RunpodAPIError as exc:
            if not _resource_missing(exc):
                raise
            raise RuntimeError(
                f"The saved worker {self.pod_id} no longer exists in Runpod (deleted outside this tool). "
                "Its Pod disk was deleted with it; run runpod bootstrap to create and fully set up a new worker."
            ) from None

    def start(self):
        if not self.pod_id:
            raise ValueError("No saved worker; run runpod bootstrap first")
        if self.state.get("pod_replacement"):
            self._replacement_attempted = True
            self._adopt_replacement()
        # At most the original worker and one replacement. Capacity errors on
        # a recovered/replacement allocation cannot cause a creation loop.
        for _ in range(2):
            pod = self._require_pod()
            if self._actual_rate_over_limit(pod.get("costPerHr"), cap=self.cfg["max_hourly_price"],
                                            gpu_id=self.gpu_type_id):
                preferred = " or the preferred H100 hourly limit" if self.gpu_type_id in H100_CANDIDATE_IDS else ""
                raise ValueError("Worker's current rate exceeds runpod.max_hourly_price" + preferred)
            pod = self.configure_runtime(pod)
            if pod.get("desiredStatus") != "RUNNING":
                self.notify(f"Starting worker {self.pod_id}…")
                try:
                    self.api.start_pod(self.pod_id)
                except RunpodAmbiguousOutcome:
                    if self._require_pod().get("desiredStatus") != "RUNNING":
                        raise
                except RunpodAPIError as exc:
                    if exc.reason != "host_capacity":
                        raise
                    self._replace_capacity_blocked_pod()
                    continue
            break
        self.notify("Waiting for the worker's SSH connection…")
        deadline = time.monotonic() + self.cfg["startup_timeout_seconds"]
        endpoint_seen = False
        while time.monotonic() < deadline:
            pod = self._require_pod()
            endpoint = ssh_endpoint(pod)
            if endpoint:
                endpoint_seen = True
                self.endpoint = endpoint
                try:
                    self.ssh("true", timeout=15)
                except RuntimeError:
                    time.sleep(5)
                    continue
                # v1's accepted CUDA enum lags the live GPU catalog. Check
                # the allocated host rather than excluding newer drivers.
                minimum = (12, 9)
                check = ("import re,subprocess; s=subprocess.check_output(['nvidia-smi'],text=True); "
                         "m=re.search(r'CUDA Version: (\\d+)\\.(\\d+)',s); "
                         f"assert m and tuple(map(int,m.groups())) >= {minimum!r}, "
                         f"'Worker driver must support CUDA {minimum[0]}.{minimum[1]} or newer'")
                self.ssh("python3 -c " + shlex.quote(check), timeout=30)
                return
            time.sleep(5)
        if self.cloud == "COMMUNITY" and not endpoint_seen:
            # Community hosts may lack a public IP; Runpod's proxy SSH carries
            # only an interactive shell, which cannot run this deployment.
            # Stop the worker (confirmed) before one replacement, preferring
            # the next configured cloud.
            self.notify(f"Community Cloud worker {self.pod_id} never published a direct SSH port "
                        "(22/tcp on a public IP); stopping it.")
            self._stop_pod(self.pod_id)
            self._replace_capacity_blocked_pod(reason="no_direct_ssh", clouds=self._later_clouds() or (self.cloud,))
            return self.start()
        raise RuntimeError("Worker did not become ready before startup_timeout_seconds; GPU capacity may be unavailable")

    def stop(self):
        # A failed validation or lost creation response must not leave a paid
        # candidate running merely because the old Pod is still the active ID.
        targets, unresolved = [], None
        intent = self.state.get("pod_replacement")
        if intent and (intent.get("candidate_id") or
                       (isinstance(intent.get("spec"), dict) and intent.get("creation_attempted") is not False)):
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

        def missing():
            self.notify(f"Worker {pod_id} no longer exists in Runpod; nothing to stop.")

        for attempt in range(3):
            try:
                pod = self.api.get_pod(pod_id)
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                missing()
                return
            if pod.get("desiredStatus") in {"EXITED", "STOPPED", "TERMINATED"}:
                break
            try:
                self.api.stop_pod(pod_id)
            except RunpodAmbiguousOutcome:
                pass  # Observe status before any repeat stop request.
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                missing()
                return
            time.sleep(3)
        else:
            try:
                pod = self.api.get_pod(pod_id)
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                missing()
                return
        if pod.get("desiredStatus") not in {"EXITED", "STOPPED", "TERMINATED"}:
            raise RuntimeError(f"STOP NOT CONFIRMED for Runpod worker {pod_id}; GPU billing may continue. Stop it in the Runpod console.")
        if pod_id == self.pod_id:
            self.state["last_status"] = pod["desiredStatus"]
            self.save()
        self.notify("Runpod confirms the worker is stopped. Its Pod disk remains billed.")

    def created_pods(self):
        """Pods this deployment created that still exist, as (pod, reason) rows.

        A Pod qualifies by this deployment's unique generated name (the
        original ``NAME-DEPLOYMENT_ID`` or a ``-rHEX`` replacement) or by an ID
        recorded in its state. A Pod pinned by runpod.pod_id was not created
        by this tool and is never included.
        """
        base = self.cfg["name"] + "-" + self.state["deployment_id"]
        generated = re.compile(re.escape(base) + r"(?:-r[0-9a-f]{12})?")
        intent = self.state.get("pod_replacement") or {}
        recorded = {self.state.get("pod_id"), intent.get("candidate_id"), *self.state.get("previous_pod_ids", [])}
        rows = []
        for pod in self.api.list_pods():
            pod_id, name = pod.get("id"), pod.get("name")
            if not isinstance(pod_id, str) or not pod_id or pod_id == self.cfg.get("pod_id"):
                continue
            if (isinstance(name, str) and generated.fullmatch(name)) or pod_id in recorded:
                rows.append(pod)
        return rows

    def _forget_pod(self, pod_id, name):
        """Drop every saved reference to a deleted Pod in one durable save."""
        if self.state.get("pod_id") == pod_id:
            self.state.pop("pod_id", None)
            self.state.pop("last_status", None)
            self.require_bootstrap(save=False)
        self.state["previous_pod_ids"] = [kept for kept in self.state.get("previous_pod_ids", []) if kept != pod_id]
        intent = self.state.get("pod_replacement")
        if isinstance(intent, dict) and (intent.get("candidate_id") == pod_id or
                                         (intent.get("spec") or {}).get("name") == name):
            self.state.pop("pod_replacement", None)
        spec = self.state.get("pod_creation_spec")
        if isinstance(spec, dict) and spec.get("name") == name:
            for key in ("pod_creation_pending", "pod_creation_spec", "allocation"):
                self.state.pop(key, None)
        self.save()

    def delete_pod(self, pod):
        """Delete one Pod and confirm it is gone; a 404 means already deleted."""
        pod_id = pod["id"]
        self.notify(f"Deleting worker {pod_id}…")
        try:
            self.api.delete_pod(pod_id)
        except RunpodAmbiguousOutcome:
            pass  # Read back before deciding anything.
        except RunpodAPIError as exc:
            if not _resource_missing(exc):
                raise
        for attempt in range(4):
            try:
                current = self.api.get_pod(pod_id)
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                break
            if current.get("desiredStatus") == "TERMINATED":
                break
            if attempt == 3:
                raise RuntimeError(f"DELETE NOT CONFIRMED for Runpod worker {pod_id}; it may still bill. "
                                   "Rerun runpod cleanup or delete it in the Runpod console.")
            time.sleep(3)
        self._forget_pod(pod_id, pod.get("name"))
        self.notify(f"Deleted worker {pod_id} and its Pod disk.")

    def cleanup(self, *, assume_yes=False, confirm=input):
        """Delete every Pod this deployment created, after explicit confirmation."""
        pods = self.created_pods()
        if self.cfg.get("pod_id"):
            self.notify(f"Worker {self.cfg['pod_id']} is pinned by runpod.pod_id and is never deleted by cleanup.")
        if not pods:
            self.notify("No Runpod Pods created by this deployment exist.")
            return 0
        self.notify("Runpod Pods created by this deployment:")
        for pod in pods:
            status = pod.get("desiredStatus") or "unknown status"
            self.notify(f"  {pod['id']}  {pod.get('name', '')}  ({status})")
        if not assume_yes:
            answer = confirm(f"Delete these {len(pods)} Pods and their Pod disks? This cannot be undone. "
                             "Type 'delete' to confirm: ")
            if answer.strip() != "delete":
                self.notify("Nothing was deleted.")
                return 1
        failures = []
        for pod in pods:
            try:
                self.delete_pod(pod)
            except Exception as exc:
                failures.append(safe_error_text(exc, self.secrets))
        if failures:
            raise RuntimeError("; ".join(failures))
        self.notify("Network volumes and Ceph objects were not touched; the next bootstrap or encode "
                    "creates and fully sets up a new worker.")
        return 0

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
            revision = _local_head()
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
        self.state["configuration_stamp"] = self.configuration_stamp()
        self.save()

    def configuration_stamp(self):
        stamp = []
        for key in ("config_path", "overrides_path"):
            path = self.settings.get(key)
            if path is None or not Path(path).is_file():
                stamp.append(None)
            else:
                info = Path(path).stat()
                stamp.append([str(path), info.st_mtime_ns, info.st_size])
        return stamp

    def activate(self):
        from .chapter_client import ChapterClient
        client = ChapterClient({"url": self.url, "api_key": self.state["api_key"], "timeout_seconds": 10})
        capabilities = client.request("POST", "v1/server/activate", {})
        if capabilities.get("maintenance") is not False or capabilities.get("ephemeral_runs") != 1:
            raise RuntimeError("The warm chapter service did not confirm normal connected-client operation")

    def reuse_service(self):
        if (not self.state.get("bootstrapped") or "configuration_stamp" not in self.state
                or self.state["configuration_stamp"] != self.configuration_stamp()):
            return False
        from .chapter_client import ChapterClient, ChapterClientError
        client = ChapterClient({"url": self.url, "api_key": self.state["api_key"], "timeout_seconds": 10})
        try:
            capabilities = client.request("GET", "v1/capabilities")
            if capabilities.get("ephemeral_runs") != 1 or capabilities.get("warm_activation") != 1:
                return False
            if capabilities.get("maintenance") is True:
                self.activate()
        except ChapterClientError:
            return False
        self.notify("Reusing the running chapter service and its loaded inference model.")
        return True

    def install(self, *, full):
        self.notify("Installing persistent dependencies…" if full else "Checking the persistent runtime…")
        q = shlex.quote
        self.ssh(f"sh {q(self.cfg['app_dir'] + '/deploy/runpod/install.sh')} {q(self.cfg['app_dir'])} "
                 f"{'full' if full else 'runtime'} /workspace/.epub-tts-runtime {q(self.cfg['fish_source'])}",
                 timeout=2400)

    def worker_command(self, action):
        q = shlex.quote
        return (f"{q(self.cfg['app_dir'] + '/.venv/bin/python')} -u "
                f"{q(self.cfg['app_dir'] + '/deploy/runpod/worker.py')} --config {q(self.cfg['remote_config'])} "
                f"--fish-source {q(self.cfg['fish_source'])} {action}")

    def launch(self, *, prepare=False):
        # A cold worker must prove run-scoped ownership support in maintenance
        # before activation; even --no-update cannot resume a legacy queue.
        action = "serve-maintenance"
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
                if capabilities.get("maintenance") is not True:
                    raise RuntimeError("Chapter service started in the wrong maintenance mode")
                if capabilities.get("ephemeral_runs") != 1 or capabilities.get("warm_activation") != 1:
                    raise RuntimeError("The deployed worker lacks run-scoped execution; update it before rendering")
                if not prepare:
                    self.activate()
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
        from .chapter_client import ChapterClient, ChapterClientError, ClientHeartbeat
        client = ChapterClient({"url": self.url, "api_key": self.state["api_key"], "timeout_seconds": 10})
        session = secrets.token_hex(32)
        client.request("POST", "v1/sessions", {"id": session})
        heartbeat = ClientHeartbeat(client, session, 5)
        try:
            heartbeat.start()
            self.ssh(self.worker_command("smoke"), timeout=1200)
            heartbeat.check()
        finally:
            heartbeat.stop()
            try:
                client.request("DELETE", f"v1/sessions/{session}")
            except ChapterClientError:
                pass
        self.state["bootstrapped"] = True
        # The full-setup requirement is cleared only by a successful smoke
        # check on the new storage; every earlier failure keeps it durable.
        self.state.pop("bootstrap_required", None)
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
        # Before provisioning there is no worker URL, and a saved pod_id may be
        # stale. The preflight never contacts this URL; encode() re-prepares
        # with the real worker's URL before submitting anything.
        url = self.url if self.pod_id else f"https://pending-{self.cfg['service_port']}.proxy.runpod.net"
        config = build_client_config(self.settings["renderer_config"], url, self.state["api_key"],
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
    if args.runpod_action == "progress":
        # Read-only observation: no deployment lock, no deployment state, and
        # no resource or paid-compute work of any kind.
        from .runpod_progress import run_progress
        return run_progress(settings, args)
    directory = Path(settings["deployment"]["state_dir"])
    with deployment_lock(directory):
        deployment = Deployment(settings)
        if args.runpod_action == "status":
            try:
                pod = deployment.api.get_pod(deployment.pod_id) if deployment.pod_id else {}
            except RunpodAPIError as exc:
                if not _resource_missing(exc):
                    raise
                pod = {"desiredStatus": "deleted in Runpod; run bootstrap to create a new worker"}
            legacy_volumes = list(deployment.state.get("retained_network_volumes", []))
            for volume_id in [deployment.state.get("volume_id"),
                              *deployment.state.get("volumes_by_center", {}).values()]:
                if volume_id and volume_id not in legacy_volumes:
                    legacy_volumes.append(volume_id)
            summary = {"pod_id": deployment.pod_id, "status": pod.get("desiredStatus", "not created"),
                       "cloud": deployment.state.get("cloud", "SECURE") if deployment.pod_id else None,
                       "storage": ("network volume " + deployment.state["volume_id"]
                                   if deployment.state.get("volume_id") else "Pod disk") if deployment.pod_id else None,
                       "data_center_id": deployment.state.get("data_center_id"),
                       "revision": deployment.state.get("revision"),
                       "bootstrapped": deployment.state.get("bootstrapped", False),
                       "bootstrap_required": deployment.state.get("bootstrap_required", False),
                       "network_volumes": {"ids": legacy_volumes, "in_use": bool(deployment.state.get("volume_id")),
                                           "billed_separately": True},
                       "previous_pod_ids": deployment.state.get("previous_pod_ids", []),
                       "previous_pods_billed_for_disk_until_terminated": True}
            if deployment.state.get("volume_creation_pending"):
                request = deployment.state.get("volume_creation") or {}
                summary["unconfirmed_network_volume_creation"] = {
                    "name": request.get("name"), "status": "an older run may have created it; inspect Runpod"}
            if deployment.state.get("pod_replacement"):
                intent = deployment.state["pod_replacement"]
                spec = intent.get("spec") or {}
                summary["pending_replacement"] = {"name": spec.get("name"),
                                                   "pod_id": intent.get("candidate_id"),
                                                   "status": "unconfirmed; may still be running"}
            print(json.dumps(summary, indent=2))
            return 0
        if args.runpod_action == "stop":
            deployment.stop()
            return 0
        if args.runpod_action == "cleanup":
            if not args.yes and not sys.stdin.isatty():
                raise ValueError("runpod cleanup needs an interactive confirmation; pass --yes to delete without one")
            return deployment.cleanup(assume_yes=args.yes)
        if args.runpod_action not in {"bootstrap", "encode"} and not deployment.pod_id:
            raise ValueError("No saved worker; run runpod bootstrap first")
        # Fail configuration checks before allocating or starting paid compute.
        build_remote_config(settings["renderer_config"], settings["override_raw"],
                            deployment.pod_id or "preflight", deployment.state["api_key"],
                            deployment=settings["deployment"])
        if args.runpod_action == "encode":
            # The complete local preflight runs before provisioning or starting
            # any paid resource, so preflight failures create and start nothing.
            # A deleted or never-created deployment has no worker URL yet;
            # encode() re-prepares with the real worker's URL before submitting.
            deployment.prepare_encode(args)
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        interrupted = False
        operation_error = None
        service_ready = False
        try:
            forced_bootstrap = False
            saved_pod = deployment.pod_id
            if args.runpod_action in {"bootstrap", "encode"}:
                deployment.provision()
                if args.runpod_action == "encode" and (
                        deployment.pod_id != saved_pod
                        or not deployment.state.get("revision")
                        or not deployment.state.get("bootstrapped")):
                    # The saved revision and smoke flags describe a disk that was
                    # deleted or never existed. Mark it durably so a crash
                    # after this point still forces the full local setup.
                    deployment.require_bootstrap()
            deployment.start()
            # A replacement can switch worker, cloud, GPU, and region DURING
            # start(), after the pre-start flag was calculated; re-check the
            # durable flag, then the same worker conditions.
            forced_bootstrap = bool(deployment.state.get("bootstrap_required"))
            if args.runpod_action == "encode" and not forced_bootstrap and (
                    deployment.pod_id != saved_pod
                    or not deployment.state.get("revision")
                    or not deployment.state.get("bootstrapped")):
                forced_bootstrap = True
            if forced_bootstrap and args.runpod_action == "encode" and args.no_update:
                deployment.notify("--no-update ignored: this worker's Pod disk is new or was never "
                                  "bootstrapped; installing code, models, and the smoke check before encoding.")
            updated = False
            needs_update = args.runpod_action in {"bootstrap", "update"} or forced_bootstrap
            if args.runpod_action == "encode" and not args.no_update and not needs_update:
                needs_update = deployment.state.get("revision") != _local_head()
            if args.runpod_action == "encode" and not needs_update and deployment.reuse_service() is True:
                service_ready = True
                deployment.encode(args)
                return 0
            deployment.stop_service()
            if args.runpod_action in {"bootstrap", "update"}:
                deployment.deploy_code(args.source)
                updated = True
            elif forced_bootstrap:
                deployment.deploy_code("local")
                updated = True
            elif args.runpod_action == "encode" and not args.no_update:
                head = _local_head()
                if deployment.state.get("revision") != head:
                    deployment.notify(f"Deployed code is not the committed HEAD ({head[:12]}); updating the worker first…")
                    deployment.deploy_code("local")
                    updated = True
            deployment.configure()
            deployment.install(full=args.runpod_action != "encode" or updated)
            deployment.launch(prepare=args.runpod_action != "encode" or updated)
            service_ready = args.runpod_action == "encode" and not updated
            if args.runpod_action == "encode":
                if updated:
                    deployment.smoke()
                    deployment.activate()
                    service_ready = True
                deployment.encode(args)
            else:
                deployment.smoke()
                deployment.activate()
                service_ready = True
            return 0
        except KeyboardInterrupt:
            interrupted = True
            raise
        except Exception as exc:
            operation_error = exc
            raise
        finally:
            try:
                if deployment.pod_id and service_ready:
                    if args.stop_delay:
                        deployment.notify("--stop-delay is ignored; the server's idle timeout controls shutdown.")
                    deployment.notify(f"Worker left running; compute stops after {deployment.cfg['idle_timeout_seconds']} "
                                      "idle seconds. The next book can reuse the loaded service.")
                elif deployment.pod_id:
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
