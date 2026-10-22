"""Exercise stop-by-default deployment without SSH, providers, or paid resources."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.runpod import Deployment, add_parser, run_command, ssh_endpoint
from epub_tts_narrator.renderer.runpod_api import RunpodAPIError, RunpodAmbiguousOutcome
from epub_tts_narrator.renderer.runpod_config import DEPLOYMENT_DEFAULTS
from epub_tts_narrator.renderer.config import DEFAULTS


class RunpodLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        renderer = deepcopy(DEFAULTS)
        renderer["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        renderer["chapter_storage"].update(bucket="fixture-bucket", region="us-east-1",
                                            endpoint_url="https://fixture-storage.invalid",
                                            access_key_id="fixture-access-key", secret_access_key="fixture-ceph-secret")
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory),
                               image="fixture/image@sha256:" + "a" * 64, data_center_id="US-FIXTURE-1"),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": renderer,
            "override_raw": {},
        }
        self.api = Mock()
        self.api.catalog_gpus.return_value = {"gpus": [{
            "id": self.settings["deployment"]["gpu_type_id"], "name": "Fixture GPU",
            "availability": "HIGH", "price": {"secure": 2.0},
            "dataCenters": [{"id": "US-FIXTURE-1", "availability": "HIGH"}],
        }]}
        self.api.create_volume.return_value = {"id": "volume_fixture", "dataCenterId": "US-FIXTURE-1"}
        self.api.create_pod.return_value = {"id": "pod_fixture", "cost": 2.0, "status": "PROVISIONING"}
        self.notify = Mock()
        self.sleep = patch("epub_tts_narrator.renderer.runpod.time.sleep").start()
        self.addCleanup(patch.stopall)

    def deployment(self):
        result = Deployment(self.settings, api=self.api, notify=self.notify)
        result._public_key = Mock(return_value="ssh-ed25519 fixture-public-key")
        return result

    def args(self, action, *extra):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args(["runpod", action, *extra])

    def fake_deployment(self):
        result = Mock()
        result.pod_id = "pod_fixture"
        result.state = {"bootstrapped": True, "api_key": "fixture-worker-secret", "volume_id": "volume_fixture"}
        result.secrets = {"fixture-provider-secret", "fixture-worker-secret", "fixture-ceph-secret"}
        return result

    def run_fake(self, fake, args):
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=fake), \
                patch("epub_tts_narrator.renderer.runpod.signal.signal"), \
                redirect_stdout(StringIO()):
            return run_command(args)

    def test_every_operation_stops_on_success(self):
        for action in ("bootstrap", "update", "encode"):
            with self.subTest(action=action):
                fake = self.fake_deployment()
                args = self.args(action, *(["book"] if action == "encode" else []))
                self.assertEqual(self.run_fake(fake, args), 0)
                fake.stop.assert_called_once()
                fake.start.assert_called_once()
                fake.configure.assert_called_once()
                fake.install.assert_called_once_with(full=action != "encode")
                fake.launch.assert_called_once_with(prepare=action != "encode")
                if action == "encode":
                    fake.encode.assert_called_once_with(args)
                    fake.deploy_code.assert_not_called()
                else:
                    fake.smoke.assert_called_once()
                    fake.deploy_code.assert_called_once_with("github")

    def test_errors_in_each_phase_still_stop_existing_or_created_pod(self):
        for action in ("bootstrap", "update", "encode"):
            phases = ["start", "stop_service", "configure", "install", "launch", "encode" if action == "encode" else "smoke"]
            if action != "encode":
                phases.append("deploy_code")
            if action == "bootstrap":
                phases.append("provision")
            for phase in phases:
                with self.subTest(action=action, phase=phase):
                    fake = self.fake_deployment()
                    getattr(fake, phase).side_effect = RuntimeError("fixture phase failure")
                    args = self.args(action, *(["book"] if action == "encode" else []))
                    with self.assertRaisesRegex(RuntimeError, "fixture phase failure"):
                        self.run_fake(fake, args)
                    fake.stop.assert_called_once()

    def test_ctrl_c_during_work_stops_before_propagating(self):
        fake = self.fake_deployment()
        fake.encode.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(fake, self.args("encode", "book", "--stop-delay", "300"))
        fake.stop.assert_called_once()
        self.sleep.assert_not_called()

    def test_ctrl_c_during_shutdown_delay_stops_immediately(self):
        fake = self.fake_deployment()
        self.sleep.side_effect = KeyboardInterrupt()
        self.assertEqual(self.run_fake(fake, self.args("update", "--stop-delay", "300")), 0)
        self.sleep.assert_called_once_with(300)
        fake.stop.assert_called_once()

    def test_shutdown_ignores_interrupt_signals_then_restores_handlers(self):
        fake = self.fake_deployment()
        originals = {signal.SIGINT: object(), signal.SIGTERM: object()}
        handlers = dict(originals)
        def check_cleanup_handlers():
            self.assertEqual(handlers[signal.SIGINT], signal.SIG_IGN)
            self.assertEqual(handlers[signal.SIGTERM], signal.SIG_IGN)
        fake.stop.side_effect = check_cleanup_handlers
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=fake), \
                patch("epub_tts_narrator.renderer.runpod.signal.getsignal", side_effect=handlers.get), \
                patch("epub_tts_narrator.renderer.runpod.signal.signal", side_effect=lambda sig, handler: handlers.update({sig: handler})), \
                redirect_stdout(StringIO()):
            self.assertEqual(run_command(self.args("update")), 0)
        self.assertEqual(handlers, originals)

    def test_previous_service_is_stopped_before_updating_code_or_config(self):
        fake = self.fake_deployment()
        self.run_fake(fake, self.args("update"))
        names = [entry[0] for entry in fake.method_calls]
        self.assertLess(names.index("stop_service"), names.index("deploy_code"))
        self.assertLess(names.index("stop_service"), names.index("configure"))
        self.assertLess(names.index("stop_service"), names.index("launch"))

    def test_invalid_remote_configuration_fails_before_paid_operations(self):
        fake = self.fake_deployment()
        with patch("epub_tts_narrator.renderer.runpod_config.build_remote_config", side_effect=ValueError("invalid remote config")):
            with self.assertRaisesRegex(ValueError, "invalid remote config"):
                self.run_fake(fake, self.args("bootstrap"))
        fake.provision.assert_not_called()
        fake.start.assert_not_called()

    def test_invalid_book_plan_fails_before_starting_paid_compute(self):
        fake = self.fake_deployment()
        fake.prepare_encode.side_effect = ValueError("fixture invalid book plan")
        with self.assertRaisesRegex(ValueError, "invalid book plan"):
            self.run_fake(fake, self.args("encode", "book"))
        fake.start.assert_not_called()
        fake.launch.assert_not_called()
        fake.stop.assert_not_called()

    def test_shutdown_failure_warns_about_billing_and_redacts_secrets(self):
        fake = self.fake_deployment()
        fake.stop.side_effect = RuntimeError("fixture-provider-secret fixture-ceph-secret fixture-worker-secret")
        with self.assertRaisesRegex(RuntimeError, "billing may continue") as caught:
            self.run_fake(fake, self.args("update"))
        for secret in fake.secrets:
            self.assertNotIn(secret, str(caught.exception))

    def test_status_only_prints_selected_fields_without_credentials(self):
        fake = self.fake_deployment()
        fake.api.get_pod.return_value = {
            "desiredStatus": "EXITED", "env": {"PASSWORD": "fixture-provider-secret"},
            "privateRuntime": "fixture-ceph-secret", "id": "pod_fixture",
        }
        output = StringIO()
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=fake), redirect_stdout(output):
            self.assertEqual(run_command(self.args("status")), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "EXITED")
        self.assertEqual(record["pod_id"], "pod_fixture")
        for secret in fake.secrets:
            self.assertNotIn(secret, output.getvalue())
        fake.start.assert_not_called()
        fake.stop.assert_not_called()

    def test_cli_flags_select_local_commits_and_remote_chapter_rendering(self):
        bootstrap = self.args("bootstrap", "--source", "local", "--config", "local.toml",
                              "--override-config", "remote.toml", "--stop-delay", "300")
        self.assertEqual(bootstrap.source, "local")
        self.assertEqual(bootstrap.config, Path("local.toml"))
        self.assertEqual(bootstrap.override_config, Path("remote.toml"))
        encoded = self.args("encode", "book", "--chapters", "2-3", "--chapters-only",
                            "--chapter-format", "m4a", "--work-dir", "work", "-o", "audio")
        self.assertEqual(encoded.bundle, Path("book"))
        self.assertEqual(encoded.chapters, "2-3")
        self.assertTrue(encoded.chapters_only)
        self.assertEqual(encoded.stop_delay, 0)
        for delay in ("-1", "301"):
            with redirect_stdout(StringIO()), patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
                self.args("update", "--stop-delay", delay)

    def test_create_spec_uses_persistent_volume_and_exposes_only_required_ports(self):
        deployment = self.deployment()
        deployment.provision()
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["mounts"], {"network": [{"volumeId": "volume_fixture", "path": "/workspace"}]})
        self.assertEqual(spec["dataCenterIds"], ["US-FIXTURE-1"])
        self.assertEqual(spec["image"], self.settings["deployment"]["image"])
        self.assertEqual(spec["disk"], self.settings["deployment"]["container_disk_gb"])
        self.assertEqual(spec["cloud"], "SECURE")
        self.assertEqual(spec["ports"], ["22/tcp", "8082/http"])
        self.assertNotIn("interruptible", spec)
        self.assertNotIn("supportPublicIp", spec)
        self.assertEqual(spec["gpu"], {"id": self.settings["deployment"]["gpu_type_id"],
                                       "count": 1, "minCudaVersion": "12.9"})
        self.assertEqual(spec["cmd"][:2], ["bash", "-lc"])
        self.assertIn("exec /start.sh", spec["cmd"][2])
        self.assertEqual(spec["env"], {"PUBLIC_KEY": "ssh-ed25519 fixture-public-key"})
        self.assertNotIn("fixture-provider-secret", json.dumps(spec))
        self.assertNotIn("fixture-ceph-secret", json.dumps(spec))
        self.assertEqual(deployment.state["pod_id"], "pod_fixture")
        self.assertEqual(deployment.state_path.stat().st_mode & 0o777, 0o600)

    def test_unavailable_or_over_budget_gpu_creates_nothing(self):
        for changes in ({"availability": "NONE"}, {"price": {"secure": 10.0}}):
            with self.subTest(changes=changes):
                deployment = self.deployment()
                self.api.catalog_gpus.return_value["gpus"][0].update(changes)
                with self.assertRaises(ValueError):
                    deployment.provision()
                self.api.create_volume.assert_not_called()
                self.api.create_pod.assert_not_called()

    def test_ambiguous_create_recovers_exactly_one_named_pod(self):
        deployment = self.deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        name = deployment.cfg["name"] + "-" + deployment.state["deployment_id"]
        self.api.list_pods.return_value = [{"name": name, "id": "recovered_pod", "costPerHr": "2.0"}]
        deployment.provision()
        self.assertEqual(deployment.pod_id, "recovered_pod")
        self.assertNotIn("pod_creation_pending", deployment.state)
        self.api.create_pod.assert_called_once()

    def test_unconfirmed_create_is_not_blindly_repeated(self):
        deployment = self.deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        self.api.list_pods.return_value = []
        with self.assertRaises(RunpodAmbiguousOutcome):
            deployment.provision()
        self.assertTrue(deployment.state["pod_creation_pending"])
        with self.assertRaisesRegex(RuntimeError, "No replacement"):
            deployment.provision()
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_called_once()

    def test_definite_create_rejection_clears_pending_flag(self):
        deployment = self.deployment()
        self.api.create_pod.side_effect = RunpodAPIError("settings rejected", status=400)
        with self.assertRaises(RunpodAPIError):
            deployment.provision()
        self.assertNotIn("pod_creation_pending", deployment.state)
        self.assertEqual(deployment.state["volume_id"], "volume_fixture")

    def test_retry_with_current_datacenter_reuses_saved_volume(self):
        self.settings["deployment"]["data_center_id"] = "US-CO-1"
        self.api.catalog_gpus.return_value["gpus"][0]["dataCenters"] = [
            {"id": "US-CO-1", "availability": "HIGH"}]
        self.api.get_volume.return_value = {"id": "saved_volume", "dataCenterId": "US-CO-1"}
        deployment = self.deployment()
        deployment.state["volume_id"] = "saved_volume"
        deployment.provision()
        self.api.create_volume.assert_not_called()
        self.api.get_volume.assert_called_once_with("saved_volume")
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["dataCenterIds"], ["US-CO-1"])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "saved_volume")

    def test_volume_timeout_is_reconciled_before_another_volume_creation(self):
        deployment = self.deployment()
        self.api.create_volume.side_effect = RunpodAmbiguousOutcome("uncertain")
        with self.assertRaises(RunpodAmbiguousOutcome):
            deployment.provision()
        self.assertTrue(deployment.state["volume_creation_pending"])
        name = deployment.cfg["name"] + "-" + deployment.state["deployment_id"]
        self.api.list_volumes.return_value = [{"name": name, "id": "recovered_volume", "dataCenterId": "US-FIXTURE-1"}]
        deployment.provision()
        self.api.create_volume.assert_called_once()
        self.assertEqual(self.api.create_pod.call_args.args[0]["mounts"]["network"][0]["volumeId"], "recovered_volume")

    def test_pending_pod_recovery_does_not_depend_on_current_stock(self):
        deployment = self.deployment()
        deployment.state["pod_creation_pending"] = True
        deployment.save()
        name = deployment.cfg["name"] + "-" + deployment.state["deployment_id"]
        self.api.list_pods.return_value = [{"name": name, "id": "recovered_pod", "costPerHr": "2.0"}]
        self.api.catalog_gpus.return_value = {"gpus": []}
        deployment.provision()
        self.assertEqual(deployment.pod_id, "recovered_pod")
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()

    def test_actual_create_price_above_cap_retains_id_for_cleanup(self):
        deployment = self.deployment()
        self.api.create_pod.return_value["cost"] = 100.0
        with self.assertRaisesRegex(RuntimeError, "rate exceeds"):
            deployment.provision()
        self.assertEqual(deployment.pod_id, "pod_fixture")
        self.assertNotIn("pod_creation_pending", deployment.state)

    def test_v1_reconciled_create_price_still_enforces_cap(self):
        deployment = self.deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        name = deployment.cfg["name"] + "-" + deployment.state["deployment_id"]
        self.api.list_pods.return_value = [{"name": name, "id": "recovered_pod", "costPerHr": "100.0"}]
        with self.assertRaisesRegex(RuntimeError, "rate exceeds"):
            deployment.provision()
        self.assertEqual(deployment.pod_id, "recovered_pod")
        self.assertNotIn("pod_creation_pending", deployment.state)

    def test_stop_polls_after_ambiguous_response_before_retrying(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        self.api.get_pod.side_effect = [{"desiredStatus": "RUNNING"}, {"desiredStatus": "EXITED"}]
        self.api.stop_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        deployment.stop()
        self.api.stop_pod.assert_called_once_with("pod_fixture")
        self.assertEqual(deployment.state["last_status"], "EXITED")

    def test_stop_never_reports_success_until_provider_confirms(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        self.api.get_pod.return_value = {"desiredStatus": "RUNNING"}
        with self.assertRaisesRegex(RuntimeError, "STOP NOT CONFIRMED"):
            deployment.stop()
        self.assertEqual(self.api.stop_pod.call_count, 3)
        self.assertNotIn("last_status", deployment.state)

    def test_existing_worker_price_is_checked_before_starting(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        self.api.get_pod.return_value = {"desiredStatus": "EXITED", "costPerHr": "100.0"}
        with patch("epub_tts_narrator.renderer.runpod.time.monotonic", side_effect=[0, 10000]):
            with self.assertRaisesRegex((ValueError, RuntimeError), "rate|price|configured cap"):
                deployment.start()
        self.api.start_pod.assert_not_called()

    def test_custom_service_port_sets_proxy_and_create_request(self):
        self.settings["deployment"]["service_port"] = 9001
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(deployment.url, "https://pod_fixture-9001.proxy.runpod.net")
        self.assertEqual(self.api.create_pod.call_args.args[0]["ports"], ["22/tcp", "9001/http"])

    def test_changed_port_updates_stopped_pod_without_replacing_volume(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        pod = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8081/http"]}
        updated = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8082/http"]}
        self.api.get_pod.return_value = updated
        self.assertEqual(deployment.configure_ports(pod), updated)
        self.api.update_pod.assert_called_once_with("pod_fixture", {"ports": ["22/tcp", "8082/http"]})
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()
        self.api.start_pod.assert_not_called()
        self.api.stop_pod.assert_not_called()

    def test_unchanged_port_does_not_mutate_running_worker(self):
        deployment = self.deployment()
        pod = {"desiredStatus": "RUNNING", "ports": ["8082/http", "22/tcp"]}
        self.assertIs(deployment.configure_ports(pod), pod)
        self.api.update_pod.assert_not_called()
        self.api.stop_pod.assert_not_called()

    def test_uncertain_port_update_is_read_back_without_retrying(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        self.api.update_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        old = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8081/http"]}
        self.api.get_pod.return_value = old
        with self.assertRaisesRegex(RuntimeError, "port update was not confirmed"):
            deployment.configure_ports(old)
        self.assertEqual(self.api.update_pod.call_count, 1)
        self.api.start_pod.assert_not_called()

    def test_dead_supervisor_fails_launch_without_waiting_full_timeout(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        deployment.ssh = Mock(side_effect=[b"", b"", b"Worker supervisor exited.\n", b"Bind failed.\n"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
            client.return_value.request.side_effect = ChapterClientError("unavailable")
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                deployment.launch()
        self.assertEqual(client.return_value.request.call_count, 1)
        self.assertIn("tail -n 60", deployment.ssh.call_args.args[0])
        self.sleep.assert_not_called()

    def test_missing_or_invalid_ssh_endpoint_is_not_used(self):
        self.assertEqual(ssh_endpoint({"publicIp": "203.0.113.7", "portMappings": {"22": 10341}}),
                         ("203.0.113.7", 10341))
        for pod in ({}, {"publicIp": "-oProxyCommand=bad", "portMappings": {"22": 22}},
                    {"publicIp": "203.0.113.7", "portMappings": {"22": 0}},
                    {"publicIp": "203.0.113.7", "portMappings": {"22": 65536}},
                    {"publicIp": "203.0.113.7", "portMappings": None},
                    {"publicIp": "203.0.113.7", "portMappings": {"22": True}},
                    {"publicIp": "203.0.113.7", "portMappings": {"22": 22.5}},
                    {"publicIp": True, "portMappings": {"22": 22}}):
            with self.subTest(pod=pod):
                self.assertIsNone(ssh_endpoint(pod))


if __name__ == "__main__":
    unittest.main()
