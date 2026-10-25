"""Deletion recovery: encode recovers deleted or never-created Runpod resources.

Uses the real Deployment and run_command flow against a stateful fake provider
API; only SSH, the local git transfer, the chapter client, and the local book
preflight are stubbed, like the rest of the lifecycle coverage.
"""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.runpod import Deployment, add_parser, run_command
from epub_tts_narrator.renderer.runpod_api import RunpodAPIError, RunpodAmbiguousOutcome
from epub_tts_narrator.renderer.runpod_config import DEPLOYMENT_DEFAULTS
from epub_tts_narrator.renderer.config import DEFAULTS


def not_found():
    return RunpodAPIError("Runpod could not find the requested resource (HTTP 404)", status=404)


class FakeRunpodAPI:
    """In-memory provider: confirmed deletions, durable creates, priced catalog."""

    def __init__(self, gpu_row):
        self.pods, self.volumes = {}, {}
        self.catalog = {"gpus": [gpu_row]}
        self.calls = []
        self.get_pod_error = None
        self.crash_after_pod_create = False

    def catalog_gpus(self, **query):
        return deepcopy(self.catalog)

    def datacenters(self):
        return {"dataCenters": [{"id": center["id"], "networkVolumeTypes": ["STANDARD"]}
                                for row in self.catalog["gpus"] for center in row.get("dataCenters", [])]}

    def get_pod(self, pod_id, **expansions):
        self.calls.append(("get_pod", pod_id))
        if self.get_pod_error is not None:
            raise self.get_pod_error
        pod = self.pods.get(pod_id)
        if pod is None:
            raise not_found()
        return {"id": pod_id, "desiredStatus": pod["status"], "costPerHr": pod["cost"],
                "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}

    def get_volume(self, volume_id):
        self.calls.append(("get_volume", volume_id))
        if volume_id not in self.volumes:
            raise not_found()
        return deepcopy(self.volumes[volume_id])

    def list_pods(self):
        return [{"id": pod_id, "name": pod["name"], "costPerHr": pod["cost"]}
                for pod_id, pod in self.pods.items()]

    def list_volumes(self):
        return [{"id": volume_id, "name": volume["name"]} for volume_id, volume in self.volumes.items()]

    def create_volume(self, spec):
        self.calls.append(("create_volume", spec["name"]))
        self.volumes["volume_new"] = {"id": "volume_new", "name": spec["name"],
                                      "dataCenterId": spec["dataCenterId"], "size": spec["size"]}
        return deepcopy(self.volumes["volume_new"])

    def create_pod(self, spec):
        self.calls.append(("create_pod", spec["name"]))
        count = sum(1 for key in self.pods if key.startswith("pod_new"))
        pod_id = f"pod_new{count + 1}"
        self.pods[pod_id] = {"name": spec["name"], "image": spec["image"], "status": "EXITED",
                             "cost": 2.0, "gpu": deepcopy(spec["gpu"]), "cloud": "SECURE",
                             "dataCenterId": spec["dataCenterIds"][0], "spec": deepcopy(spec),
                             "mounts": deepcopy(spec["mounts"])}
        if self.crash_after_pod_create:
            self.crash_after_pod_create = False
            raise KeyboardInterrupt()
        return {"id": pod_id, "cost": 2.0, "status": "PROVISIONING"}

    def get_pod_details(self, pod_id):
        self.calls.append(("get_pod_details", pod_id))
        pod = self.pods.get(pod_id)
        if pod is None:
            raise not_found()
        return {"id": pod_id, "name": pod["name"], "image": pod["image"], "gpu": deepcopy(pod["gpu"]),
                "mounts": deepcopy(pod["mounts"]), "cloud": "SECURE", "dataCenterId": pod["dataCenterId"],
                "cost": pod["cost"], "status": pod["status"]}

    def start_pod(self, pod_id):
        self.calls.append(("start_pod", pod_id))
        self.pods[pod_id]["status"] = "RUNNING"
        return {"id": pod_id}

    def stop_pod(self, pod_id):
        self.calls.append(("stop_pod", pod_id))
        self.pods[pod_id]["status"] = "EXITED"
        return {"id": pod_id}

    def created(self, kind):
        return [call for call in self.calls if call[0] == kind]


class DeletionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory / "state"),
                               image="fixture/image@sha256:" + "a" * 64, data_center_id="US-FIXTURE-1"),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": deepcopy(DEFAULTS),
            "override_raw": {},
        }
        self.settings["renderer_config"]["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        self.settings["renderer_config"]["chapter_storage"].update(
            bucket="fixture-bucket", region="us-east-1", endpoint_url="https://fixture-storage.invalid",
            access_key_id="fixture-access-key", secret_access_key="fixture-ceph-secret")
        gpu_row = {"id": self.settings["deployment"]["gpu_type_id"], "name": "Fixture GPU",
                   "availability": "HIGH", "price": {"secure": 2.0},
                   "dataCenters": [{"id": "US-FIXTURE-1", "availability": "HIGH"}]}
        self.api = FakeRunpodAPI(gpu_row)
        self.notify = Mock()
        self.phase_calls = []
        self.service_mode = {"prepare": False}
        for patcher in (patch("epub_tts_narrator.renderer.runpod.time.sleep"),
                        patch("epub_tts_narrator.renderer.runpod._local_head", return_value="fixture-head")):
            patcher.start()
        self.addCleanup(patch.stopall)

    def isolate(self, label):
        self.settings["deployment"]["state_dir"] = str(self.directory / f"state-{label}")

    def live_pod(self, pod_id="pod_old", status="RUNNING"):
        self.api.pods[pod_id] = {"name": "old-worker", "image": self.settings["deployment"]["image"],
                                 "status": status, "cost": 2.0, "gpu": {"id": "gpu", "count": 1},
                                 "cloud": "SECURE", "dataCenterId": "US-FIXTURE-1", "spec": {}, "mounts": {}}

    def retained_volume(self, volume_id="volume_kept"):
        self.api.volumes[volume_id] = {"id": volume_id, "name": "old",
                                       "dataCenterId": "US-FIXTURE-1", "size": 150}

    def write_state(self, **values):
        state_dir = Path(self.settings["deployment"]["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(json.dumps(values))

    def read_state(self):
        return json.loads((Path(self.settings["deployment"]["state_dir"]) / "state.json").read_text())

    def build_deployment(self, settings, *, real_preflight=False):
        deployment = Deployment(settings, api=self.api, notify=self.notify)
        deployment._public_key = Mock(return_value="ssh-ed25519 fixture-public-key")
        deployment.configure_runtime = Mock(side_effect=lambda pod: pod)
        deployment._run = Mock(return_value=b"")
        deployment.upload = Mock()
        service_mode = self.service_mode

        def ssh(command, *, input=None, timeout=1800):
            if "serve-maintenance >" in command:
                service_mode["prepare"] = True
            elif "serve >" in command:
                service_mode["prepare"] = False
            if "remote get-url" in command:
                return f"{deployment.cfg['github_url']}\n".encode()
            if command.startswith("git -C") and "rev-parse HEAD" in command:
                return b"fixture-remote-revision\n"
            return b""

        deployment.ssh = Mock(side_effect=ssh)

        def recorded(name, method):
            def wrapper(*args, **kwargs):
                parts = [repr(value) for value in args] + [f"{key}={value}" for key, value in kwargs.items()]
                self.phase_calls.append(f"{name}({', '.join(parts)})")
                return method(*args, **kwargs)
            return wrapper

        for name in ("deploy_code", "install", "launch", "smoke", "encode"):
            setattr(deployment, name, recorded(name, getattr(deployment, name)))
        if not real_preflight:
            deployment.prepare_encode = Mock(return_value=({"chapters": [{}]},
                                                           {"render": {"output_dir": "/unused/audiobook"}}))
        return deployment

    def run_driver(self, args, *, real_preflight=False):
        real = Deployment
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
             patch("epub_tts_narrator.renderer.runpod.Deployment",
                   side_effect=lambda settings: self.build_deployment(settings, real_preflight=real_preflight)), \
             patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client, \
             patch("epub_tts_narrator.renderer.chapter_client.render_remote",
                   return_value={"chapters": [{"number": 1, "audio_file": "/unused/chapter.m4a"}]}) as render, \
             redirect_stdout(StringIO()):
            def request(method, path, payload=None, timeout=None):
                if path == "v1/server/activate":
                    self.service_mode["prepare"] = False
                    self.phase_calls.append("activate()")
                return {"maintenance": self.service_mode["prepare"], "ephemeral_runs": 1, "warm_activation": 1}
            client.return_value.request.side_effect = request
            result = run_command(args)
        return result, render

    def actions(self):
        return " | ".join(self.phase_calls)

    def args(self, *extra):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args(["runpod", *extra])

    def test_encode_with_no_saved_state_bootstraps_everything_before_rendering(self):
        result, render = self.run_driver(self.args("encode", "book"))
        self.assertEqual(result, 0)
        render.assert_called_once()
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        state = self.read_state()
        self.assertEqual(state["volume_id"], "volume_new")
        self.assertEqual(state["pod_id"], "pod_new1")
        self.assertEqual(state["revision"], "fixture-head")
        self.assertTrue(state["bootstrapped"])
        for expected in ("deploy_code('local')", "install(full=True)", "smoke()",
                         "launch(prepare=True)", "activate()", "encode("):
            self.assertIn(expected, self.actions())
        self.assertEqual([call[1] for call in self.api.created("start_pod")], ["pod_new1"])
        self.assertEqual(self.api.pods["pod_new1"]["status"], "RUNNING")

    def test_encode_with_deleted_pod_and_retained_volume_keeps_the_fast_path(self):
        self.retained_volume()
        self.write_state(pod_id="pod_old", volume_id="volume_kept", revision="fixture-head",
                         bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
        result, render = self.run_driver(self.args("encode", "book"))
        self.assertEqual(result, 0)
        render.assert_called_once()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(len(self.api.created("create_pod")), 1)
        state = self.read_state()
        self.assertEqual(state["pod_id"], "pod_new1")
        self.assertEqual(state["volume_id"], "volume_kept")
        self.assertEqual(state["revision"], "fixture-head")
        self.assertTrue(state["bootstrapped"])
        self.assertIn("install(full=False)", self.actions())
        self.assertIn("launch(prepare=False)", self.actions())
        self.assertNotIn("smoke()", self.actions())
        self.assertNotIn("deploy_code", self.actions())
        self.assertEqual(self.api.pods["pod_new1"]["spec"]["mounts"]["network"][0]["volumeId"], "volume_kept")
        notices = " ".join(call.args[0] for call in self.notify.call_args_list)
        self.assertIn("Saved worker pod_old no longer exists", notices)

    def test_encode_with_pod_and_volume_deleted_fully_bootstraps_replacement_storage(self):
        self.write_state(pod_id="pod_old", volume_id="volume_old", revision="stale-revision",
                         bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
        result, render = self.run_driver(self.args("encode", "book"))
        self.assertEqual(result, 0)
        render.assert_called_once()
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        state = self.read_state()
        self.assertEqual(state["volume_id"], "volume_new")
        self.assertEqual(state["revision"], "fixture-head")
        self.assertTrue(state["bootstrapped"])
        for expected in ("deploy_code('local')", "install(full=True)", "smoke()"):
            self.assertIn(expected, self.actions())
        notices = " ".join(call.args[0] for call in self.notify.call_args_list)
        self.assertIn("persistent volume volume_old no longer exists", notices)
        self.assertIn("Saved worker pod_old no longer exists", notices)

    def test_encode_with_deleted_volume_stops_the_live_pod_before_any_replacement(self):
        for previous_status in ("RUNNING", "EXITED"):
            with self.subTest(previous_status=previous_status):
                self.setUp()
                self.live_pod(status=previous_status)
                self.write_state(pod_id="pod_old", volume_id="volume_old", revision="stale-revision",
                                 bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
                result, render = self.run_driver(self.args("encode", "book"))
                self.assertEqual(result, 0)
                render.assert_called_once()
                if previous_status == "RUNNING":
                    # The confirmed stop of the old worker precedes every paid
                    # replacement, and the old pod is never started again.
                    first_create = next(index for index, call in enumerate(self.api.calls)
                                        if call[0] in {"create_volume", "create_pod"})
                    self.assertLess(self.api.calls.index(("stop_pod", "pod_old")), first_create)
                self.assertNotIn(("start_pod", "pod_old"), self.api.calls)
                self.assertEqual(self.api.pods["pod_old"]["status"], "EXITED")
                self.assertEqual(self.api.pods["pod_new1"]["status"], "RUNNING")
                self.assertEqual([call[1] for call in self.api.created("start_pod")], ["pod_new1"])
                state = self.read_state()
                self.assertNotEqual(state["pod_id"], "pod_old")
                self.assertEqual(state["volume_id"], "volume_new")
                self.assertEqual(len(self.api.created("create_volume")), 1)
                self.assertEqual(len(self.api.created("create_pod")), 1)
                self.assertIn("deploy_code('local')", self.actions())
                self.assertIn("smoke()", self.actions())

    def test_healthy_existing_deployment_encodes_without_any_full_bootstrap(self):
        self.retained_volume()
        self.live_pod(pod_id="pod_kept", status="EXITED")
        self.write_state(pod_id="pod_kept", volume_id="volume_kept", revision="fixture-head",
                         bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
        result, render = self.run_driver(self.args("encode", "book", "--no-update"))
        self.assertEqual(result, 0)
        render.assert_called_once()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])
        self.assertIn("install(full=False)", self.actions())
        self.assertNotIn("deploy_code", self.actions())
        self.assertNotIn("smoke()", self.actions())

    def test_missing_configured_pins_fail_before_any_resource_creation_or_stopping(self):
        self.settings["deployment"]["pod_id"] = "pod_pin"
        deployment = self.build_deployment(self.settings)
        with self.assertRaisesRegex(ValueError, "runpod.pod_id"):
            deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])
        self.assertEqual(self.api.created("stop_pod"), [])

        del self.settings["deployment"]["pod_id"]
        self.isolate("volume-pin")
        self.settings["deployment"]["volume_id"] = "volume_pin"
        self.live_pod()
        deployment = self.build_deployment(self.settings)
        with self.assertRaisesRegex(ValueError, "runpod.volume_id"):
            deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])
        self.assertEqual(self.api.created("stop_pod"), [])

    def test_deleted_volume_cannot_silently_replace_an_explicit_pod_pin(self):
        self.live_pod(pod_id="pod_pinned", status="RUNNING")
        self.settings["deployment"]["pod_id"] = "pod_pinned"
        self.write_state(pod_id="pod_pinned", volume_id="volume_deleted", revision="fixture-head",
                         bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
        with self.assertRaisesRegex(ValueError, "runpod.pod_id"):
            self.build_deployment(self.settings).provision()
        self.assertEqual(self.api.created("stop_pod"), [])
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])

    def test_uncertain_provider_answers_never_allocate_replacement_resources(self):
        for error in (RunpodAPIError("Runpod denied permission for this operation (HTTP 403)", status=403),
                      RunpodAPIError("Runpod rejected the request (HTTP 500)", status=500),
                      RunpodAmbiguousOutcome("Runpod did not confirm GET /pods.")):
            with self.subTest(error=type(error).__name__ + str(getattr(error, "status", None))):
                self.setUp()
                self.write_state(pod_id="pod_old", volume_id="volume_old", revision="fixture-head",
                                 bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12)
                self.api.get_pod_error = error
                deployment = self.build_deployment(self.settings)
                with self.assertRaises(RunpodAPIError):
                    deployment.provision()
                self.assertEqual(self.api.created("create_volume"), [])
                self.assertEqual(self.api.created("create_pod"), [])
                self.assertEqual(self.api.created("stop_pod"), [])

    def test_unsafe_book_fails_before_any_resource_creation(self):
        with patch("epub_tts_narrator.renderer.bundle.load_bundle",
                   side_effect=ValueError("fixture invalid book")):
            with self.assertRaisesRegex(ValueError, "invalid book"):
                self.run_driver(self.args("encode", "book"), real_preflight=True)
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])

    def test_lost_pod_creation_response_rerun_adopts_and_never_duplicates(self):
        self.api.crash_after_pod_create = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_driver(self.args("encode", "book"))
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        pending = self.read_state()
        self.assertTrue(pending["pod_creation_pending"])
        self.assertNotIn("pod_id", pending)

        result, render = self.run_driver(self.args("encode", "book"))
        self.assertEqual(result, 0)
        render.assert_called_once()
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        state = self.read_state()
        self.assertEqual(state["pod_id"], "pod_new1")
        self.assertNotIn("pod_creation_pending", state)
        self.assertTrue(state["bootstrapped"])
        self.assertIn("encode(", self.actions())

    def test_progress_action_parses_observation_options(self):
        parsed = self.args("progress", "--watch", "--job", "a" * 64, "--poll-seconds", "0.5",
                           "--limit", "10", "--json")
        self.assertEqual(parsed.runpod_action, "progress")
        self.assertEqual(parsed.job, "a" * 64)
        self.assertTrue(parsed.watch)
        self.assertEqual(parsed.poll_seconds, 0.5)
        self.assertEqual(parsed.limit, 10)
        self.assertTrue(parsed.json)
        defaults = self.args("progress")
        self.assertEqual(defaults.poll_seconds, 2.0)
        self.assertEqual(defaults.limit, 100)
        self.assertFalse(defaults.watch)
        self.assertFalse(defaults.json)
        self.assertFalse(hasattr(defaults, "stop_delay"))


if __name__ == "__main__":
    unittest.main()
