"""US-wide placement: hints are not locks, storage is reused weakly, and
cross-region creation is durable. Uses the real Deployment and run_command
flow against the shared stateful fake provider API; only SSH, the local git
transfer, the chapter client, and the local book preflight are stubbed.
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

from test_runpod_deletion_recovery import FakeRunpodAPI, not_found

from epub_tts_narrator.renderer.runpod import Deployment, add_parser, run_command
from epub_tts_narrator.renderer.runpod_api import RunpodAPIError, RunpodAmbiguousOutcome
from epub_tts_narrator.renderer.runpod_config import DEPLOYMENT_DEFAULTS
from epub_tts_narrator.renderer.config import DEFAULTS

RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
H100 = "NVIDIA H100 PCIe"
NC, CO = "US-NC-1", "US-CO-1"


def gpu_row(gpu_id, price=2.0, centers=(NC,), availability="HIGH"):
    return {"id": gpu_id, "name": gpu_id, "availability": availability, "price": {"secure": price},
            "dataCenters": [{"id": center, "availability": availability} for center in centers]}


class PlacementFakeAPI(FakeRunpodAPI):
    """Shared fake plus multiple regions, unique volume IDs, and hooks."""

    def __init__(self, rows, centers):
        super().__init__(rows[0])
        self.catalog["gpus"] = deepcopy(rows)
        self.fixed_centers = deepcopy(centers)
        self.get_volume_error = None
        self.pod_specs = []

    def create_pod(self, spec):
        self.pod_specs.append(deepcopy(spec))
        return super().create_pod(spec)

    def datacenters(self):
        return {"dataCenters": deepcopy(self.fixed_centers)}

    def get_volume(self, volume_id):
        self.calls.append(("get_volume", volume_id))
        if self.get_volume_error is not None:
            raise self.get_volume_error
        volume = self.volumes.get(volume_id)
        if volume is None:
            raise not_found()
        return deepcopy(volume)

    def create_volume(self, spec):
        self.calls.append(("create_volume", spec["name"], spec["dataCenterId"]))
        volume_id = f"volume_new{sum(1 for call in self.calls if call[0] == 'create_volume')}"
        self.volumes[volume_id] = {"id": volume_id, "name": spec["name"],
                                   "dataCenterId": spec["dataCenterId"], "size": spec["size"]}
        return deepcopy(self.volumes[volume_id])


class FullHostAPI(PlacementFakeAPI):
    """The source worker's host has no free GPU; every other host schedules."""

    def start_pod(self, pod_id):
        self.calls.append(("start_pod", pod_id))
        if pod_id == "pod_old":
            raise RunpodAPIError("The Pod's host has no free GPUs (HTTP 400)", status=400,
                                 reason="host_capacity")
        self.pods[pod_id]["status"] = "RUNNING"
        return {"id": pod_id}


class USPlacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory / "state"),
                               image="fixture/image@sha256:" + "a" * 64, data_center_id=NC),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": deepcopy(DEFAULTS),
            "override_raw": {},
        }
        self.settings["renderer_config"]["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        self.settings["renderer_config"]["chapter_storage"].update(
            bucket="fixture-bucket", region="us-east-1", endpoint_url="https://fixture-storage.invalid",
            access_key_id="fixture-access-key", secret_access_key="fixture-ceph-secret")
        self.centers = [{"id": NC, "networkVolumeTypes": ["STANDARD"]},
                        {"id": CO, "networkVolumeTypes": ["STANDARD"]}]
        self.api = PlacementFakeAPI([gpu_row(RTX)], self.centers)
        self.notify = Mock()
        self.phase_calls = []
        self.service_mode = {"prepare": False}
        self.fail_phase = None
        for patcher in (patch("epub_tts_narrator.renderer.runpod.time.sleep"),
                        patch("epub_tts_narrator.renderer.runpod._local_head", return_value="fixture-head")):
            patcher.start()
        self.addCleanup(patch.stopall)

    def isolate(self, label):
        self.settings["deployment"]["state_dir"] = str(self.directory / f"state-{label}")

    def write_state(self, **values):
        state_dir = Path(self.settings["deployment"]["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "state.json").write_text(json.dumps(values))

    def read_state(self):
        return json.loads((Path(self.settings["deployment"]["state_dir"]) / "state.json").read_text())

    def live_pod(self, pod_id="pod_old", status="EXITED", data_center_id=NC):
        self.api.pods[pod_id] = {"name": "old-worker", "image": self.settings["deployment"]["image"],
                                 "status": status, "cost": 2.0, "gpu": {"id": RTX, "count": 1},
                                 "cloud": "SECURE", "dataCenterId": data_center_id, "spec": {},
                                 "mounts": {"network": [{"volumeId": "volume_kept", "path": "/workspace"}]}}

    def retained_volume(self, volume_id="volume_kept", data_center_id=NC):
        self.api.volumes[volume_id] = {"id": volume_id, "name": "retained",
                                       "dataCenterId": data_center_id, "size": 150}

    def regional_name(self, data_center_id, legacy=False):
        suffix = "" if legacy else f"-{data_center_id.lower()}"
        return f"epub-tts-narrator-{'d' * 12}{suffix}"

    def build_deployment(self, settings):
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
                self.phase_calls.append(name)
                return method(*args, **kwargs)
            return wrapper

        for name in ("deploy_code", "install", "launch", "smoke", "encode"):
            setattr(deployment, name, recorded(name, getattr(deployment, name)))
        deployment.prepare_encode = Mock(return_value=({"chapters": [{}]},
                                                       {"render": {"output_dir": "/unused/audiobook"}}))
        if self.fail_phase:
            deployment.install = Mock(side_effect=RuntimeError(f"fixture {self.fail_phase} failure"))
        return deployment

    def run_driver(self, args):
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
             patch("epub_tts_narrator.renderer.runpod.Deployment",
                   side_effect=lambda settings: self.build_deployment(settings)), \
             patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client, \
             patch("epub_tts_narrator.renderer.chapter_client.render_remote",
                   return_value={"chapters": [{"number": 1, "audio_file": "/unused/chapter.m4a"}]}), \
             redirect_stdout(StringIO()):
            def request(method, path, payload=None, timeout=None):
                if path == "v1/server/activate":
                    self.service_mode["prepare"] = False
                    self.phase_calls.append("activate")
                return {"maintenance": self.service_mode["prepare"], "ephemeral_runs": 1, "warm_activation": 1}
            client.return_value.request.side_effect = request
            return run_command(args)

    def actions(self):
        return " | ".join(self.phase_calls)

    def args(self, *extra):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args(["runpod", *extra])

    def notices(self):
        return " ".join(str(call.args[0]) for call in self.notify.call_args_list)

    def capacity_setup(self, rows=None, mapping=None):
        self.api = FullHostAPI(rows or [gpu_row(RTX, 2.49, (NC,)), gpu_row(H100, 2.59, (CO,))],
                               self.centers)
        self.retained_volume()
        self.live_pod(status="EXITED")
        state = dict(pod_id="pod_old", volume_id="volume_kept", revision="fixture-head",
                     bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12,
                     gpu_type_id=RTX, data_center_id=NC)
        if mapping:
            state["volumes_by_center"] = mapping
        self.write_state(**state)

    def test_configured_region_is_only_a_hint_and_stock_selects_the_region(self):
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(deployment.data_center_id, NC)
        self.assertEqual(self.settings["deployment"]["data_center_id"], NC)
        self.assertEqual(deployment.state["data_center_id"], NC)

        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": NC, "availability": "LOW"},
                                                      {"id": CO, "availability": "HIGH"}]
        self.isolate("stock-drop")
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(deployment.data_center_id, CO)
        state = self.read_state()
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["dataCenterIds"], [CO])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], state["volume_id"])
        self.assertIn("only a hint", self.notices())

    def test_empty_stale_or_non_us_hints_still_allocate_in_eligible_us_regions(self):
        for hint in ("", "CA-MTL-3", "US-RETIRED-9"):
            with self.subTest(hint=hint):
                self.setUp()
                self.isolate(f"hint-{hint or 'empty'}")
                self.settings["deployment"]["data_center_id"] = hint
                deployment = self.build_deployment(self.settings)
                deployment.provision()
                self.assertEqual(deployment.data_center_id, NC)
                self.assertEqual([call[1] for call in self.api.created("create_pod")],
                                 ["epub-tts-narrator-" + deployment.state["deployment_id"]])

    def test_existing_volume_region_wins_ties_without_new_storage(self):
        self.retained_volume()
        self.write_state(volume_id="volume_kept", api_key="fixture-worker-secret", deployment_id="d" * 12)
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(deployment.data_center_id, NC)
        self.assertEqual(self.api.created("create_volume"), [])
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_kept")
        self.assertEqual(spec["dataCenterIds"], [NC])
        state = self.read_state()
        self.assertEqual(state["volumes_by_center"], {NC: "volume_kept"})
        self.assertEqual(state["data_center_id"], NC)

    def test_returning_to_a_known_region_reuses_verified_mapped_storage(self):
        self.retained_volume()
        self.retained_volume("volume_returned", CO)
        self.write_state(volume_id="volume_kept", data_center_id=NC,
                         volumes_by_center={NC: "volume_kept", CO: "volume_returned"},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": CO, "availability": "HIGH"}]
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_returned")
        self.assertEqual(spec["dataCenterIds"], [CO])
        self.assertEqual(self.read_state()["volumes_by_center"], {NC: "volume_kept", CO: "volume_returned"})

    def test_local_shortage_creates_new_colocated_storage_and_retains_the_old_volume(self):
        self.retained_volume()
        self.write_state(volume_id="volume_kept", api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": CO, "availability": "HIGH"}]
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual([call[1:] for call in self.api.created("create_volume")],
                         [(self.regional_name(CO), CO)])
        self.assertEqual(deployment.data_center_id, CO)
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["dataCenterIds"], [CO])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_new1")
        state = self.read_state()
        self.assertEqual(state["volumes_by_center"], {NC: "volume_kept", CO: "volume_new1"})
        self.assertIn("volume_kept", self.api.volumes)
        notices = self.notices()
        self.assertIn("remains billed for storage", notices)
        self.assertIn("no cache or data is migrated", notices)

    def test_affordable_h100_elsewhere_beats_reusing_the_local_rtx_volume(self):
        self.retained_volume()
        self.write_state(volume_id="volume_kept", api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.catalog["gpus"] = [gpu_row(RTX, 2.49, (NC,)), gpu_row(H100, 2.59, (CO,))]
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(deployment.gpu_type_id, H100)
        self.assertEqual(deployment.data_center_id, CO)
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["gpu"]["id"], H100)
        self.assertEqual(spec["dataCenterIds"], [CO])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_new1")
        self.assertEqual(self.read_state()["volumes_by_center"][NC], "volume_kept")

    def test_pending_volume_request_reconciles_before_today_catalog_or_hint(self):
        name = self.regional_name(NC)
        self.api.volumes["volume_pending"] = {"id": "volume_pending", "name": name,
                                              "dataCenterId": NC, "size": 150}
        self.write_state(volume_creation_pending=True,
                         volume_creation={"name": name, "size": 150, "data_center_id": NC},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.settings["deployment"]["data_center_id"] = CO
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        state = self.read_state()
        self.assertEqual(state["volume_id"], "volume_pending")
        self.assertNotIn("volume_creation_pending", state)
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_pending")
        self.assertEqual(spec["dataCenterIds"], [NC])

    def test_legacy_pending_bool_and_empty_inventory_retry_the_exact_saved_request(self):
        name = self.regional_name(NC, legacy=True)
        self.api.volumes["volume_pending"] = {"id": "volume_pending", "name": name,
                                              "dataCenterId": NC, "size": 150}
        self.isolate("legacy-pending")
        self.write_state(volume_creation_pending=True, api_key="fixture-worker-secret",
                         deployment_id="d" * 12)
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.read_state()["volume_id"], "volume_pending")

        self.setUp()
        self.isolate("empty-inventory")
        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": NC, "availability": "HIGH"},
                                                      {"id": CO, "availability": "HIGH"}]
        saved = self.regional_name(CO)
        self.write_state(volume_creation_pending=True,
                         volume_creation={"name": saved, "size": 150, "data_center_id": CO},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.assertEqual([call[1:] for call in self.api.created("create_volume")],
                         [(saved, CO)])
        state = self.read_state()
        self.assertEqual(state["volume_id"], "volume_new1")
        self.assertEqual(self.api.pod_specs[-1]["mounts"]["network"][0]["volumeId"], "volume_new1")
        self.assertEqual(self.api.pod_specs[-1]["dataCenterIds"], [CO])
        self.assertNotIn("volume_creation_pending", state)
        self.assertIn("made nothing", self.notices())

    def test_uncertain_volume_answers_never_discard_mappings_or_duplicate(self):
        self.retained_volume()
        self.retained_volume("volume_returned", CO)
        self.write_state(volume_id="volume_kept", data_center_id=NC,
                         volumes_by_center={NC: "volume_kept", CO: "volume_returned"},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": CO, "availability": "HIGH"}]
        self.api.get_volume_error = RunpodAPIError("Runpod denied permission for this operation (HTTP 403)",
                                                   status=403)
        deployment = self.build_deployment(self.settings)
        with self.assertRaises(RunpodAPIError):
            deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])
        self.assertEqual(self.read_state()["volumes_by_center"], {NC: "volume_kept", CO: "volume_returned"})

    def test_explicit_volume_pin_restricts_the_region_and_requires_us(self):
        self.retained_volume("volume_pinned", CO)
        self.write_state(api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.settings["deployment"]["volume_id"] = "volume_pinned"
        self.settings["deployment"]["data_center_id"] = NC
        self.api.catalog["gpus"][0]["dataCenters"] = [{"id": NC, "availability": "HIGH"},
                                                      {"id": CO, "availability": "HIGH"}]
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        spec = self.api.pod_specs[-1]
        self.assertEqual(spec["dataCenterIds"], [CO])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_pinned")
        self.assertEqual(self.api.created("create_volume"), [])

        self.setUp()
        self.isolate("non-us-pin")
        self.retained_volume("volume_pinned", "CA-MTL-3")
        self.write_state(api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.settings["deployment"]["volume_id"] = "volume_pinned"
        deployment = self.build_deployment(self.settings)
        with self.assertRaisesRegex(ValueError, "non-US datacenter"):
            deployment.provision()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])

    def test_cross_region_replacement_bootstraps_fully_despite_no_update(self):
        self.capacity_setup()
        result = self.run_driver(self.args("encode", "book", "--no-update"))
        self.assertEqual(result, 0)
        self.assertEqual([call[1:] for call in self.api.created("create_volume")],
                         [(self.regional_name(CO), CO)])
        self.assertEqual(len(self.api.created("create_pod")), 1)
        state = self.read_state()
        self.assertEqual(state["pod_id"], "pod_new1")
        self.assertEqual(state["gpu_type_id"], H100)
        self.assertEqual(state["data_center_id"], CO)
        self.assertEqual(state["volume_id"], "volume_new1")
        self.assertEqual(state["volumes_by_center"], {NC: "volume_kept", CO: "volume_new1"})
        self.assertTrue(state["bootstrapped"])
        self.assertNotIn("bootstrap_required", state)
        self.assertEqual(state["revision"], "fixture-head")
        self.assertEqual(self.api.pods["pod_old"]["status"], "EXITED")
        self.assertIn("volume_kept", self.api.volumes)
        for expected in ("deploy_code", "install", "smoke", "launch", "encode", "activate"):
            self.assertIn(expected, self.actions())
        self.assertIn("--no-update ignored", self.notices())

    def test_failed_setup_after_adoption_keeps_the_bootstrap_requirement_durable(self):
        self.capacity_setup()
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        deployment.start()
        state = self.read_state()
        self.assertTrue(state["bootstrap_required"])
        self.assertNotIn("bootstrapped", state)
        self.assertNotIn("revision", state)

        self.fail_phase = "install"
        with self.assertRaisesRegex(RuntimeError, "fixture install failure"):
            self.run_driver(self.args("encode", "book", "--no-update"))
        self.assertTrue(self.read_state()["bootstrap_required"])

        self.fail_phase = None
        self.assertEqual(self.run_driver(self.args("encode", "book", "--no-update")), 0)
        self.assertIn("deploy_code", self.actions())
        self.assertIn("smoke", self.actions())
        state = self.read_state()
        self.assertTrue(state["bootstrapped"])
        self.assertNotIn("bootstrap_required", state)

    def test_replacement_adopts_after_a_region_hint_change_without_reselection(self):
        self.capacity_setup()
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        real_create = self.api.create_pod
        self.api.create_pod = Mock(side_effect=RunpodAmbiguousOutcome("lost response"))
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        intent = self.read_state()["pod_replacement"]
        self.assertEqual(intent["spec"]["dataCenterIds"], [CO])
        self.assertEqual(intent["source_spec"]["dataCenterIds"], [NC])
        self.assertIsNone(intent.get("candidate_id"))
        self.api.create_pod = real_create
        self.api.create_pod(intent["spec"])

        self.settings["deployment"]["data_center_id"] = "US-MO-9"
        restored = self.build_deployment(self.settings)
        restored.start()
        self.assertEqual(restored.pod_id, "pod_new1")
        self.assertEqual(len(self.api.created("create_pod")), 1)
        self.assertEqual(restored.state["gpu_type_id"], H100)
        self.assertEqual(restored.data_center_id, CO)
        self.assertEqual(restored.state["volume_id"], "volume_new1")
        self.assertTrue(restored.state["bootstrap_required"])
        self.assertEqual(self.api.pods["pod_old"]["status"], "EXITED")

    def test_replacement_reuses_a_retained_volume_in_the_target_region(self):
        rtx = gpu_row(RTX, 2.0, (NC,))
        rtx["dataCenters"] = [{"id": NC, "availability": "LOW"}, {"id": CO, "availability": "HIGH"}]
        self.capacity_setup(rows=[rtx], mapping={NC: "volume_kept", CO: "volume_returned"})
        self.retained_volume("volume_returned", CO)
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        deployment.start()
        self.assertEqual(self.api.created("create_volume"), [])
        state = self.read_state()
        self.assertEqual(state["volume_id"], "volume_returned")
        self.assertEqual(state["data_center_id"], CO)
        self.assertTrue(state["bootstrap_required"])
        self.assertEqual(state["volumes_by_center"], {NC: "volume_kept", CO: "volume_returned"})
        self.assertEqual(self.api.pods["pod_old"]["status"], "EXITED")

    def test_healthy_worker_stays_fast_when_only_the_hint_changes(self):
        self.retained_volume()
        self.live_pod(status="EXITED")
        self.write_state(pod_id="pod_old", volume_id="volume_kept", revision="fixture-head",
                         bootstrapped=True, api_key="fixture-worker-secret", deployment_id="d" * 12,
                         data_center_id=NC, volumes_by_center={NC: "volume_kept"})
        self.settings["deployment"]["data_center_id"] = CO
        self.api.catalog_gpus = Mock(wraps=self.api.catalog_gpus)
        deployment = self.build_deployment(self.settings)
        deployment.provision()
        self.api.catalog_gpus.assert_not_called()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.created("create_pod"), [])
        self.assertEqual(self.read_state()["pod_id"], "pod_old")

    def test_lost_initial_volume_response_resumes_the_original_placement(self):
        self.api.catalog["gpus"] = [gpu_row(RTX, centers=(NC, CO))]
        original = self.api.create_volume

        def uncertain(spec):
            original(spec)
            raise RunpodAmbiguousOutcome("lost volume response")

        self.api.create_volume = uncertain
        deployment = self.build_deployment(self.settings)
        with self.assertRaises(RunpodAmbiguousOutcome):
            deployment.provision()
        self.api.create_volume = original
        self.settings["deployment"]["data_center_id"] = CO
        self.api.catalog["gpus"] = [gpu_row(H100, 2.59, (CO,))]
        restored = self.build_deployment(self.settings)
        restored.provision()
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(self.api.pod_specs[-1]["dataCenterIds"], [NC])
        self.assertEqual(restored.state["volume_id"], "volume_new1")

    def test_volume_inventory_adoption_persists_its_mapping_before_clearing_intent(self):
        name = self.regional_name(NC)
        self.api.volumes["volume_pending"] = {"id": "volume_pending", "name": name,
                                               "dataCenterId": NC, "size": 150}
        self.write_state(volume_creation_pending=True,
                         volume_creation={"name": name, "size": 150, "data_center_id": NC},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.build_deployment(self.settings)._ensure_volume(None)
        state = self.read_state()
        self.assertNotIn("volume_creation_pending", state)
        self.assertEqual(state["volumes_by_center"][NC], "volume_pending")

    def test_interrupted_replacement_volume_creation_preserves_source_and_target(self):
        self.capacity_setup()
        original = self.api.create_volume

        def interrupted(spec):
            original(spec)
            raise KeyboardInterrupt()

        self.api.create_volume = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.build_deployment(self.settings).start()
        state = self.read_state()
        self.assertEqual(state["pod_id"], "pod_old")
        self.assertEqual(state["volume_id"], "volume_kept")
        self.assertEqual(state["data_center_id"], NC)
        self.assertEqual(state["pod_replacement"]["target_data_center_id"], CO)
        self.api.create_volume = original
        self.api.catalog_gpus = Mock(side_effect=AssertionError("reselected a saved target"))
        restored = self.build_deployment(self.settings)
        restored.start()
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        self.assertEqual(restored.state["data_center_id"], CO)
        self.assertEqual(restored.state["volume_id"], "volume_new1")

    def test_deleted_source_during_volume_creation_resumes_the_replacement(self):
        self.capacity_setup()
        original = self.api.create_volume

        def interrupted(spec):
            original(spec)
            raise KeyboardInterrupt()

        self.api.create_volume = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.build_deployment(self.settings).start()
        del self.api.pods["pod_old"]
        self.api.create_volume = original
        self.api.catalog_gpus = Mock(side_effect=AssertionError("reselected a saved target"))

        restored = self.build_deployment(self.settings)
        restored.provision()
        restored.start()

        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertEqual(len(self.api.created("create_pod")), 1)
        self.assertEqual(restored.state["volume_id"], "volume_new1")
        self.assertEqual(restored.state["data_center_id"], CO)
        self.assertNotIn("pod_replacement", self.read_state())

    def test_deleted_source_after_uncertain_pod_creation_adopts_existing_candidate(self):
        self.capacity_setup()
        self.api.crash_after_pod_create = True
        with self.assertRaises(KeyboardInterrupt):
            self.build_deployment(self.settings).start()
        del self.api.pods["pod_old"]
        self.api.crash_after_pod_create = False

        restored = self.build_deployment(self.settings)
        restored.provision()
        restored.start()

        self.assertEqual(restored.pod_id, "pod_new1")
        self.assertEqual(len(self.api.created("create_pod")), 1)
        self.assertNotIn("pod_replacement", self.read_state())

    def test_legacy_pending_volume_uses_actual_us_location_after_hint_change(self):
        self.retained_volume("volume_pending", NC)
        self.api.volumes["volume_pending"]["name"] = self.regional_name(NC, legacy=True)
        self.write_state(volume_creation_pending=True, api_key="fixture-worker-secret",
                         deployment_id="d" * 12)
        self.settings["deployment"]["data_center_id"] = CO

        deployment = self.build_deployment(self.settings)
        deployment.provision()

        self.assertEqual(deployment.state["volume_id"], "volume_pending")
        self.assertEqual(self.api.pod_specs[-1]["dataCenterIds"], [NC])
        self.assertEqual(self.api.created("create_volume"), [])

    def test_pending_non_us_volume_is_not_adopted_from_inventory(self):
        self.retained_volume("volume_pending", "CA-MTL-3")
        self.api.volumes["volume_pending"]["name"] = "old-request"
        self.write_state(volume_creation_pending=True,
                         volume_creation={"name": "old-request", "size": 150, "data_center_id": "CA-MTL-3"},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)

        with self.assertRaisesRegex(RuntimeError, "US datacenter"):
            self.build_deployment(self.settings).provision()

        self.assertNotIn("volume_id", self.read_state())
        self.assertEqual(self.api.created("create_pod"), [])

    def test_cross_region_adoption_is_one_durable_state_switch(self):
        self.capacity_setup()
        deployment = self.build_deployment(self.settings)
        snapshots = []
        original = deployment.save

        def save():
            snapshots.append(deepcopy(deployment.state))
            original()

        deployment.save = save
        deployment.start()
        self.assertFalse(any(state.get("pod_id") == "pod_new1" and "pod_replacement" in state
                             for state in snapshots))
        self.assertTrue(self.read_state()["bootstrap_required"])

    def test_explicit_volume_pin_also_constrains_capacity_replacement(self):
        self.capacity_setup()
        self.settings["deployment"]["volume_id"] = "volume_kept"
        deployment = self.build_deployment(self.settings)
        deployment.start()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.pod_specs[-1]["dataCenterIds"], [NC])
        self.assertEqual(deployment.state["volume_id"], "volume_kept")

    def test_legacy_capacity_replacement_prefers_actual_volume_location_on_a_tie(self):
        self.capacity_setup(rows=[gpu_row(RTX, 2.49, (NC, CO))])
        self.settings["deployment"]["data_center_id"] = CO
        deployment = self.build_deployment(self.settings)
        deployment.start()
        self.assertEqual(self.api.created("create_volume"), [])
        self.assertEqual(self.api.pod_specs[-1]["dataCenterIds"], [NC])

    def test_initial_storage_switch_marks_bootstrap_before_an_interrupted_pod_post(self):
        self.retained_volume()
        self.write_state(volume_id="volume_kept", revision="old-revision", bootstrapped=True,
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.catalog["gpus"] = [gpu_row(RTX, centers=(CO,))]
        self.api.crash_after_pod_create = True
        with self.assertRaises(KeyboardInterrupt):
            self.build_deployment(self.settings).provision()
        state = self.read_state()
        self.assertTrue(state["bootstrap_required"])
        self.assertNotIn("revision", state)
        self.assertNotIn("bootstrapped", state)
        self.api.catalog["gpus"] = [gpu_row(RTX, centers=(NC,))]
        self.build_deployment(self.settings).provision()
        self.assertEqual(len(self.api.created("create_pod")), 1)
        self.assertEqual(len(self.api.created("create_volume")), 1)
        self.assertTrue(self.read_state()["bootstrap_required"])

    def test_mismatched_volume_identity_cannot_overwrite_the_region_map(self):
        self.retained_volume()
        self.write_state(volume_id="volume_kept", volumes_by_center={NC: "volume_kept"},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        self.api.get_volume = Mock(return_value={"id": "other_volume", "dataCenterId": NC})
        with self.assertRaisesRegex(RuntimeError, "volume identity"):
            self.build_deployment(self.settings).provision()
        self.assertEqual(self.read_state()["volumes_by_center"], {NC: "volume_kept"})
        self.assertEqual(self.api.created("create_volume"), [])

    def test_non_us_pending_volume_request_never_allocates_non_us_storage(self):
        self.write_state(volume_creation_pending=True,
                         volume_creation={"name": "old-request", "size": 150, "data_center_id": "CA-MTL-3"},
                         api_key="fixture-worker-secret", deployment_id="d" * 12)
        with self.assertRaisesRegex(RuntimeError, "US datacenter"):
            self.build_deployment(self.settings)._ensure_volume(None)
        self.assertEqual(self.api.created("create_volume"), [])


if __name__ == "__main__":
    unittest.main()
