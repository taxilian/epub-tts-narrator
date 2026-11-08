"""Cloud preference order: Secure/Community selection, one confirmed-refusal
fallback, and replacement of a Community worker without direct SSH.

Uses the real Deployment against the shared stateful fake provider extended
with per-cloud catalogs; only SSH, the SSH key, and runtime reconfiguration
are stubbed. No network or model calls are made.
"""

from copy import deepcopy
from itertools import count
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_runpod_deletion_recovery import FakeRunpodAPI

from epub_tts_narrator.renderer.runpod import Deployment
from epub_tts_narrator.renderer.runpod_api import RunpodAPIError, RunpodAmbiguousOutcome
from epub_tts_narrator.renderer.runpod_config import DEPLOYMENT_DEFAULTS
from epub_tts_narrator.renderer.config import DEFAULTS

RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
CO, TX = "US-CO-1", "US-TX-3"
POD_DISK = {"persistent": {"size": 150, "path": "/workspace"}}


def gpu_row(secure=2.0, community=1.2, centers=(CO,), gpu_id=RTX):
    return {"id": gpu_id, "name": gpu_id, "manufacturer": "NVIDIA", "memory": 96, "availability": "HIGH",
            "price": {"secure": secure, "community": community},
            "dataCenters": [{"id": center, "availability": "HIGH"} for center in centers]}


class CloudFakeAPI(FakeRunpodAPI):
    """Per-cloud catalogs, per-cloud creation refusals, and hosts without public IPs."""

    def __init__(self, rows_by_cloud):
        super().__init__(gpu_row())
        self.rows_by_cloud = deepcopy(rows_by_cloud)
        self.catalog_clouds = []
        self.refusals = {}
        self.clouds_without_public_ip = set()
        self.pod_specs = []

    def catalog_gpus(self, **query):
        self.catalog_clouds.append(query["cloud"])
        return {"gpus": deepcopy(self.rows_by_cloud.get(query["cloud"], []))}

    def datacenters(self):
        raise AssertionError("Pod-disk placement must not need the datacenter storage catalog")

    def create_pod(self, spec):
        self.pod_specs.append(deepcopy(spec))
        refusal = self.refusals.get(spec["cloud"])
        if refusal is not None:
            self.calls.append(("create_pod", spec["name"]))
            raise refusal
        return super().create_pod(spec)

    def get_pod(self, pod_id, **expansions):
        result = super().get_pod(pod_id, **expansions)
        if self.pods[pod_id]["cloud"] in self.clouds_without_public_ip:
            result.pop("publicIp")
            result["portMappings"] = {}
        return result


class CloudPlacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory / "state"),
                               image="fixture/image@sha256:" + "a" * 64, data_center_id=CO,
                               cloud_types=["community", "secure"], startup_timeout_seconds=60),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": deepcopy(DEFAULTS),
            "override_raw": {},
        }
        self.settings["renderer_config"]["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        self.api = CloudFakeAPI({"SECURE": [gpu_row()], "COMMUNITY": [gpu_row(centers=(TX,))]})
        self.notify = Mock()
        clock = count(step=7)
        for patcher in (patch("epub_tts_narrator.renderer.runpod.time.sleep"),
                        patch("epub_tts_narrator.renderer.runpod.time.monotonic", side_effect=lambda: next(clock))):
            patcher.start()
        self.addCleanup(patch.stopall)

    def deployment(self):
        deployment = Deployment(self.settings, api=self.api, notify=self.notify)
        deployment._public_key = Mock(return_value="ssh-ed25519 fixture-public-key")
        deployment.configure_runtime = Mock(side_effect=lambda pod: pod)
        deployment.ssh = Mock(return_value=b"")
        return deployment

    def state(self):
        return json.loads((Path(self.settings["deployment"]["state_dir"]) / "state.json").read_text())

    def notices(self):
        return " ".join(str(call.args[0]) for call in self.notify.call_args_list)

    def test_secure_only_default_never_reads_or_requests_community(self):
        self.settings["deployment"]["cloud_types"] = ["secure"]
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(self.api.catalog_clouds, ["SECURE"])
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["SECURE"])
        self.assertEqual(self.state()["cloud"], "SECURE")

    def test_preferred_community_placement_uses_its_price_region_and_a_pod_disk(self):
        saved = []
        original = self.api.create_pod
        self.api.create_pod = lambda spec: (saved.append(self.state()), original(spec))[1]
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(self.api.catalog_clouds, ["COMMUNITY"])
        spec = self.api.pod_specs[-1]
        self.assertEqual((spec["cloud"], spec["dataCenterIds"], spec["mounts"]), ("COMMUNITY", [TX], POD_DISK))
        # The chosen cloud is durable before the only POST.
        self.assertEqual(saved[0]["allocation"]["cloud"], "COMMUNITY")
        self.assertEqual(saved[0]["pod_creation_spec"]["cloud"], "COMMUNITY")
        self.assertEqual(saved[0]["catalog_hourly_price"], 1.2)
        state = self.state()
        self.assertEqual((state["cloud"], state["pod_id"]), ("COMMUNITY", "pod_new1"))
        self.assertTrue(state["bootstrap_required"])
        self.assertIn("Runpod Community Cloud", self.notices())

    def test_community_without_eligible_placement_selects_secure_before_any_post(self):
        for rows in ([], [gpu_row(community=9.0, centers=(TX,))], [gpu_row(centers=("CA-MTL-3",))]):
            with self.subTest(rows=rows):
                self.setUp()
                self.api.rows_by_cloud["COMMUNITY"] = rows
                self.deployment().provision()
                self.assertEqual(self.api.catalog_clouds, ["COMMUNITY", "SECURE"])
                self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["SECURE"])
                self.assertEqual(self.api.pod_specs[0]["dataCenterIds"], [CO])
                self.assertEqual(self.state()["cloud"], "SECURE")

    def test_confirmed_community_refusal_falls_back_once_to_secure(self):
        for status in (400, 403):
            with self.subTest(status=status):
                self.setUp()
                self.api.refusals["COMMUNITY"] = RunpodAPIError("could not be placed", status=status)
                deployment = self.deployment()
                deployment.provision()
                self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY", "SECURE"])
                self.assertEqual(self.api.catalog_clouds, ["COMMUNITY", "SECURE"])
                state = self.state()
                self.assertEqual((state["cloud"], state["pod_id"], state["data_center_id"]), ("SECURE", "pod_new1", CO))
                self.assertNotIn("pod_creation_pending", state)
                self.assertIn(f"refused the worker request (HTTP {status}); trying Secure Cloud once", self.notices())

    def test_other_failures_never_fall_back(self):
        for error in (RunpodAPIError("insufficient balance", status=402),
                      RunpodAPIError("conflict", status=409)):
            with self.subTest(status=error.status):
                self.setUp()
                self.api.refusals["COMMUNITY"] = error
                with self.assertRaises(RunpodAPIError):
                    self.deployment().provision()
                self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY"])
                self.assertNotIn("pod_creation_pending", self.state())

    def test_ambiguous_community_create_is_reconciled_never_fallen_back(self):
        self.api.refusals["COMMUNITY"] = RunpodAmbiguousOutcome("lost response")
        with self.assertRaises(RunpodAmbiguousOutcome):
            self.deployment().provision()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY"])
        state = self.state()
        self.assertTrue(state["pod_creation_pending"])
        self.assertEqual(state["pod_creation_spec"]["cloud"], "COMMUNITY")
        # The rerun reconciles by name; an empty inventory resends the same request.
        self.api.refusals.clear()
        self.deployment().provision()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY", "COMMUNITY"])
        self.assertEqual(self.api.pod_specs[0], self.api.pod_specs[1])
        self.assertEqual(self.state()["cloud"], "COMMUNITY")

    def test_only_one_fallback_and_never_past_the_last_cloud(self):
        self.api.refusals = {"COMMUNITY": RunpodAPIError("no capacity", status=400),
                             "SECURE": RunpodAPIError("no capacity", status=400)}
        with self.assertRaises(RunpodAPIError):
            self.deployment().provision()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY", "SECURE"])
        self.assertNotIn("pod_creation_pending", self.state())

        self.setUp()
        self.settings["deployment"]["cloud_types"] = ["community"]
        self.api.refusals["COMMUNITY"] = RunpodAPIError("no capacity", status=400)
        with self.assertRaises(RunpodAPIError):
            self.deployment().provision()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY"])

    def test_no_cloud_with_a_placement_creates_nothing_and_names_each_cloud(self):
        self.api.rows_by_cloud = {"SECURE": [], "COMMUNITY": []}
        with self.assertRaisesRegex(ValueError, "Community Cloud.*Secure Cloud"):
            self.deployment().provision()
        self.assertEqual(self.api.pod_specs, [])

    def test_recovered_creation_must_match_its_saved_cloud(self):
        self.api.crash_after_pod_create = True
        with self.assertRaises(KeyboardInterrupt):
            self.deployment().provision()
        self.api.pods["pod_new1"]["cloud"] = "SECURE"
        with self.assertRaisesRegex(RuntimeError, "does not match its saved cloud"):
            self.deployment().provision()
        self.api.pods["pod_new1"]["cloud"] = "COMMUNITY"
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual((deployment.pod_id, self.state()["cloud"]), ("pod_new1", "COMMUNITY"))
        self.assertTrue(self.state()["bootstrap_required"])
        self.assertEqual(len(self.api.pod_specs), 1)

    def test_saved_allocation_for_a_cloud_no_longer_allowed_is_refused(self):
        self.api.refusals["COMMUNITY"] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.deployment().provision()
        state = self.state()
        state.pop("pod_creation_pending")
        state.pop("pod_creation_spec")
        (Path(self.settings["deployment"]["state_dir"]) / "state.json").write_text(json.dumps(state))
        self.settings["deployment"]["cloud_types"] = ["secure"]
        with self.assertRaisesRegex(RuntimeError, "policy differs"):
            self.deployment().provision()
        self.assertEqual(len(self.api.pod_specs), 1)

    def test_healthy_secure_worker_is_not_replaced_when_community_becomes_preferred(self):
        self.settings["deployment"]["cloud_types"] = ["secure"]
        self.deployment().provision()
        self.settings["deployment"]["cloud_types"] = ["community", "secure"]
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(len(self.api.pod_specs), 1)
        self.assertEqual((deployment.pod_id, deployment.cloud), ("pod_new1", "SECURE"))

    def test_unsent_legacy_network_volume_request_is_rebuilt_as_a_pod_disk(self):
        deployment = self.deployment()
        legacy = deployment._pod_spec("epub-tts-narrator-" + deployment.state["deployment_id"])
        legacy.update(cloud="SECURE", mounts={"network": [{"volumeId": "volume_old", "path": "/workspace"}]})
        deployment.state.update(pod_creation_pending=True, pod_creation_spec=legacy, volume_id="volume_old")
        deployment.save()
        self.deployment().provision()
        self.assertEqual(len(self.api.pod_specs), 1)
        self.assertEqual(self.api.pod_specs[0]["mounts"], POD_DISK)
        self.assertEqual(self.api.pod_specs[0]["cloud"], "COMMUNITY")
        state = self.state()
        self.assertEqual(state["retained_network_volumes"], ["volume_old"])
        self.assertIn("made nothing", self.notices())
        self.assertIn("Discarding an earlier unsent worker request", self.notices())

    def test_community_worker_without_direct_ssh_is_stopped_and_replaced_on_secure(self):
        self.api.clouds_without_public_ip.add("COMMUNITY")
        deployment = self.deployment()
        deployment.provision()
        deployment.start()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY", "SECURE"])
        self.assertEqual(self.api.pods["pod_new1"]["status"], "EXITED")
        self.assertEqual(self.api.pods["pod_new2"]["status"], "RUNNING")
        self.assertLess(self.api.calls.index(("stop_pod", "pod_new1")),
                        self.api.calls.index(("create_pod", self.api.pod_specs[1]["name"])))
        state = self.state()
        self.assertEqual((state["pod_id"], state["cloud"]), ("pod_new2", "SECURE"))
        self.assertEqual(state["previous_pod_ids"], ["pod_new1"])
        self.assertTrue(state["bootstrap_required"])
        self.assertNotIn("pod_replacement", state)
        self.assertEqual(deployment.endpoint, ("203.0.113.7", 10341))
        self.assertIn("never published a direct SSH port", self.notices())

    def test_community_only_replacement_stays_on_community_and_happens_once(self):
        self.settings["deployment"]["cloud_types"] = ["community"]
        self.api.clouds_without_public_ip.add("COMMUNITY")
        deployment = self.deployment()
        deployment.provision()
        with self.assertRaisesRegex(RunpodAPIError, "only one replacement"):
            deployment.start()
        self.assertEqual([spec["cloud"] for spec in self.api.pod_specs], ["COMMUNITY", "COMMUNITY"])
        self.assertEqual(self.api.pods["pod_new1"]["status"], "EXITED")
        self.assertEqual(self.api.pods["pod_new2"]["status"], "EXITED")
        self.assertEqual(self.state()["pod_id"], "pod_new2")

    def test_pinned_community_worker_without_direct_ssh_is_never_replaced(self):
        self.api.clouds_without_public_ip.add("COMMUNITY")
        self.deployment().provision()
        self.settings["deployment"]["pod_id"] = "pod_new1"
        deployment = self.deployment()
        with self.assertRaisesRegex(RunpodAPIError, "fixed"):
            deployment.start()
        self.assertEqual(len(self.api.pod_specs), 1)
        self.assertEqual(self.api.pods["pod_new1"]["status"], "EXITED")

    def test_secure_worker_without_ssh_times_out_without_replacement(self):
        self.settings["deployment"]["cloud_types"] = ["secure"]
        self.api.clouds_without_public_ip.add("SECURE")
        deployment = self.deployment()
        deployment.provision()
        with self.assertRaisesRegex(RuntimeError, "startup_timeout_seconds"):
            deployment.start()
        self.assertEqual(len(self.api.pod_specs), 1)


if __name__ == "__main__":
    unittest.main()
