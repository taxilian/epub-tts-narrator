"""`runpod cleanup` deletes only Pods this deployment created, after confirmation.

Uses the real Deployment and run_command against the shared stateful fake
provider; no network calls are made.
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

DEPLOYMENT_ID = "d" * 12
BASE = "epub-tts-narrator-" + DEPLOYMENT_ID


class CleanupFakeAPI(FakeRunpodAPI):
    def __init__(self):
        super().__init__({"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "availability": "HIGH",
                          "manufacturer": "NVIDIA", "memory": 96,
                          "price": {"secure": 2.0}, "dataCenters": [{"id": "US-CO-1", "availability": "HIGH"}]})
        self.delete_errors = {}
        self.survives_delete = set()

    def delete_pod(self, pod_id):
        self.calls.append(("delete_pod", pod_id))
        error = self.delete_errors.pop(pod_id, None)
        if error is not None and getattr(error, "status", None) == 404:
            self.pods.pop(pod_id, None)  # deleted elsewhere after the listing
            raise error
        if error is not None and not isinstance(error, RunpodAmbiguousOutcome):
            raise error
        if pod_id not in self.pods:
            raise not_found()
        if pod_id not in self.survives_delete:
            del self.pods[pod_id]
        if error is not None:
            raise error

    def add(self, pod_id, name, status="EXITED"):
        self.pods[pod_id] = {"name": name, "image": "fixture", "status": status, "cost": 2.0,
                             "gpu": {"id": "gpu", "count": 1}, "cloud": "SECURE",
                             "dataCenterId": "US-CO-1", "spec": {}, "mounts": {}}


class RunpodCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory / "state"),
                               image="fixture/image@sha256:" + "a" * 64),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": deepcopy(DEFAULTS),
            "override_raw": {},
        }
        self.settings["renderer_config"]["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        self.api = CleanupFakeAPI()
        self.api.add("pod_current", BASE, status="RUNNING")
        self.api.add("pod_retired", BASE + "-r" + "a" * 12)
        self.api.add("pod_lost_reply", BASE + "-r" + "b" * 12)  # created, ID never recorded
        self.api.add("pod_renamed", "renamed-in-console")       # recorded ID, different name
        self.api.add("pod_other_deployment", "epub-tts-narrator-" + "e" * 12)
        self.api.add("pod_lookalike", BASE + "-extra")
        self.api.add("pod_unrelated", "someone-elses-notebook", status="RUNNING")
        self.api.volumes["volume_kept"] = {"id": "volume_kept", "name": "old", "dataCenterId": "US-CO-1"}
        self.write_state(pod_id="pod_current", previous_pod_ids=["pod_retired", "pod_renamed", "pod_gone"],
                         revision="fixture-head", bootstrapped=True, retained_network_volumes=["volume_kept"])
        self.notify = Mock()
        patch("epub_tts_narrator.renderer.runpod.time.sleep").start()
        self.addCleanup(patch.stopall)

    def write_state(self, **values):
        state_dir = Path(self.settings["deployment"]["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True)
        values.setdefault("api_key", "fixture-worker-secret")
        values.setdefault("deployment_id", DEPLOYMENT_ID)
        (state_dir / "state.json").write_text(json.dumps(values))

    def state(self):
        return json.loads((Path(self.settings["deployment"]["state_dir"]) / "state.json").read_text())

    def deployment(self):
        return Deployment(self.settings, api=self.api, notify=self.notify)

    def deleted(self):
        return [call[1] for call in self.api.calls if call[0] == "delete_pod"]

    def notices(self):
        return " ".join(str(call.args[0]) for call in self.notify.call_args_list)

    OURS = ["pod_current", "pod_retired", "pod_lost_reply", "pod_renamed"]
    NOT_OURS = ["pod_other_deployment", "pod_lookalike", "pod_unrelated"]

    def test_deletes_only_pods_this_deployment_created_and_forgets_them(self):
        confirm = Mock(return_value="delete")
        self.assertEqual(self.deployment().cleanup(confirm=confirm), 0)
        confirm.assert_called_once()
        self.assertIn("cannot be undone", confirm.call_args.args[0])
        self.assertEqual(sorted(self.deleted()), sorted(self.OURS))
        for pod_id in self.NOT_OURS:
            self.assertIn(pod_id, self.api.pods)
        state = self.state()
        self.assertNotIn("pod_id", state)
        self.assertEqual(state["previous_pod_ids"], ["pod_gone"])  # never existed in the listing; kept as history
        self.assertTrue(state["bootstrap_required"])
        self.assertNotIn("revision", state)
        self.assertEqual(state["deployment_id"], DEPLOYMENT_ID)
        self.assertEqual(state["retained_network_volumes"], ["volume_kept"])
        self.assertIn("volume_kept", self.api.volumes)
        self.assertIn("Network volumes and Ceph objects were not touched", self.notices())

    def test_anything_but_typing_delete_deletes_nothing(self):
        for answer in ("", "y", "yes", "DELETE"):
            with self.subTest(answer=answer):
                self.assertEqual(self.deployment().cleanup(confirm=Mock(return_value=answer)), 1)
                self.assertEqual(self.deleted(), [])
                self.assertEqual(self.state()["pod_id"], "pod_current")

    def test_pinned_pod_is_never_deleted(self):
        self.settings["deployment"]["pod_id"] = "pod_current"
        self.deployment().cleanup(assume_yes=True)
        self.assertNotIn("pod_current", self.deleted())
        self.assertIn("pod_current", self.api.pods)
        self.assertEqual(self.state()["pod_id"], "pod_current")
        self.assertIn("pinned by runpod.pod_id", self.notices())

    def test_nothing_to_delete_is_reported_without_prompting(self):
        self.api.pods = {"pod_unrelated": self.api.pods["pod_unrelated"]}
        confirm = Mock()
        self.assertEqual(self.deployment().cleanup(confirm=confirm), 0)
        confirm.assert_not_called()
        self.assertEqual(self.deleted(), [])
        self.assertIn("No Runpod Pods created by this deployment exist", self.notices())

    def test_ambiguous_or_missing_delete_is_confirmed_by_reading_back(self):
        self.api.delete_errors["pod_current"] = RunpodAmbiguousOutcome("lost reply")
        self.api.delete_errors["pod_retired"] = not_found()
        self.deployment().cleanup(assume_yes=True)
        self.assertNotIn("pod_current", self.api.pods)
        state = self.state()
        self.assertNotIn("pod_id", state)
        self.assertNotIn("pod_retired", state["previous_pod_ids"])

    def test_unconfirmed_delete_keeps_the_id_and_other_deletions_are_durable(self):
        self.api.survives_delete.add("pod_current")
        self.api.delete_errors["pod_retired"] = RunpodAPIError("Runpod denied permission (HTTP 403)", status=403)
        with self.assertRaisesRegex(RuntimeError, "DELETE NOT CONFIRMED for Runpod worker pod_current.*HTTP 403"):
            self.deployment().cleanup(assume_yes=True)
        state = self.state()
        self.assertEqual(state["pod_id"], "pod_current")
        self.assertIn("pod_retired", state["previous_pod_ids"])
        self.assertNotIn("pod_renamed", state["previous_pod_ids"])
        self.assertNotIn("pod_lost_reply", self.api.pods)
        # Rerunning resumes with only the remaining Pods.
        self.api.survives_delete.clear()
        self.api.calls.clear()
        self.deployment().cleanup(assume_yes=True)
        self.assertEqual(sorted(self.deleted()), ["pod_current", "pod_retired"])

    def test_deleting_pending_creations_lets_the_next_run_create_a_fresh_worker(self):
        replacement = BASE + "-r" + "c" * 12
        self.api.add("pod_candidate", replacement)
        self.write_state(pod_id="pod_current", pod_creation_pending=True,
                         pod_creation_spec={"name": BASE}, allocation={"pod_name": BASE},
                         pod_replacement={"previous_pod_id": "pod_current", "spec": {"name": replacement},
                                          "creation_attempted": True})
        self.deployment().cleanup(assume_yes=True)
        state = self.state()
        for key in ("pod_id", "pod_creation_pending", "pod_creation_spec", "allocation", "pod_replacement"):
            self.assertNotIn(key, state)
        self.assertNotIn("pod_candidate", self.api.pods)
        deployment = self.deployment()
        deployment._public_key = Mock(return_value="ssh-ed25519 fixture-public-key")
        deployment.provision()
        self.assertEqual(deployment.pod_id, "pod_new1")

    def run_cli(self, *extra, tty=False):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest="command", required=True))
        args = parser.parse_args(["runpod", "cleanup", *extra])
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment",
                      side_effect=lambda settings: Deployment(settings, api=self.api, notify=self.notify)), \
                patch("epub_tts_narrator.renderer.runpod.sys.stdin") as stdin, redirect_stdout(StringIO()):
            stdin.isatty.return_value = tty
            return run_command(args)

    def test_cli_requires_yes_without_a_terminal_and_never_starts_compute(self):
        with self.assertRaisesRegex(ValueError, "--yes"):
            self.run_cli()
        self.assertEqual(self.deleted(), [])
        self.assertEqual(self.run_cli("--yes"), 0)
        self.assertEqual(sorted(self.deleted()), sorted(self.OURS))
        self.assertFalse([call for call in self.api.calls if call[0] in {"start_pod", "create_pod", "stop_pod"}])

    def test_cli_holds_the_deployment_lock(self):
        from epub_tts_narrator.renderer.runpod import deployment_lock
        with deployment_lock(Path(self.settings["deployment"]["state_dir"])):
            with self.assertRaisesRegex(ValueError, "Another Runpod operation"):
                self.run_cli("--yes")
        self.assertEqual(self.deleted(), [])


if __name__ == "__main__":
    unittest.main()
