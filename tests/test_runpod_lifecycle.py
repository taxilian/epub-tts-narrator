"""Exercise stop-by-default deployment without SSH, providers, or paid resources."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
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
        self.rtx_row = deepcopy(self.api.catalog_gpus.return_value["gpus"][0])
        self.api.create_volume.return_value = {"id": "volume_fixture", "dataCenterId": "US-FIXTURE-1"}
        self.api.create_pod.return_value = {"id": "pod_fixture", "cost": 2.0, "status": "PROVISIONING"}
        def creation_details(pod_id):
            spec = deepcopy(self.api.create_pod.call_args.args[0])
            summary = self.api.list_pods.return_value[0]
            return dict(spec, id=pod_id, dataCenterId=spec["dataCenterIds"][0],
                        cost=summary.get("costPerHr", 2.0))
        self.api.get_pod_details.side_effect = creation_details
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
        result.state = {"bootstrapped": True, "api_key": "fixture-worker-secret", "volume_id": "volume_fixture",
                        "revision": "fixture-revision"}
        result.secrets = {"fixture-provider-secret", "fixture-worker-secret", "fixture-ceph-secret"}
        return result

    def run_fake(self, fake, args, head="fixture-revision"):
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=fake), \
                patch("epub_tts_narrator.renderer.runpod._local_head", return_value=head), \
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

    def test_encode_skips_the_update_when_the_worker_matches_head(self):
        fake = self.fake_deployment()
        self.assertEqual(self.run_fake(fake, self.args("encode", "book")), 0)
        fake.deploy_code.assert_not_called()
        fake.smoke.assert_not_called()
        fake.install.assert_called_once_with(full=False)
        fake.launch.assert_called_once_with(prepare=False)
        self.assertEqual(fake.stop_service.call_count, 1)
        fake.encode.assert_called_once()
        fake.stop.assert_called_once()

    def test_encode_updates_a_stale_worker_and_keeps_it_running(self):
        fake = self.fake_deployment()
        fake.state["revision"] = "stale-revision"
        self.assertEqual(self.run_fake(fake, self.args("encode", "book"), head="current-head"), 0)
        fake.deploy_code.assert_called_once_with("local")
        fake.install.assert_called_once_with(full=True)
        fake.smoke.assert_called_once()
        self.assertEqual([call.kwargs for call in fake.launch.call_args_list],
                         [{"prepare": True}, {"prepare": False}])
        self.assertEqual(fake.stop_service.call_count, 2)
        fake.encode.assert_called_once()
        fake.stop.assert_called_once()
        order = [name for name, _, _ in fake.mock_calls if name in
                 {"deploy_code", "install", "smoke", "encode", "stop"}]
        self.assertEqual(order, ["deploy_code", "install", "smoke", "encode", "stop"])

    def test_encode_no_update_uses_the_deployed_code_as_is(self):
        fake = self.fake_deployment()
        fake.state["revision"] = "stale-revision"
        args = self.args("encode", "book", "--no-update")
        self.assertEqual(self.run_fake(fake, args, head="current-head"), 0)
        fake.deploy_code.assert_not_called()
        fake.smoke.assert_not_called()
        fake.install.assert_called_once_with(full=False)
        fake.encode.assert_called_once()

    def test_encode_no_update_still_requires_a_smoke_tested_worker(self):
        fake = self.fake_deployment()
        fake.state["bootstrapped"] = False
        args = self.args("encode", "book", "--no-update")
        with self.assertRaisesRegex(ValueError, "has not passed its smoke test"):
            self.run_fake(fake, args)
        fake.start.assert_not_called()

    def test_encode_without_deployed_code_requires_bootstrap(self):
        fake = self.fake_deployment()
        del fake.state["revision"]
        fake.state["bootstrapped"] = False
        with self.assertRaisesRegex(ValueError, "no deployed code"):
            self.run_fake(fake, self.args("encode", "book"))
        fake.start.assert_not_called()

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

    def transport_preflight_fixture(self):
        config = deepcopy(self.settings["renderer_config"])
        chapters = []
        for number in range(1, 22):
            chapter_id = f"chapter-{number:04d}"
            row = {"id": f"segment-{number}", "chapter_id": chapter_id, "block_id": f"block-{number}",
                   "text": "Hello.", "start": 0, "end": 6, "kind": "dialogue",
                   "voice_id": "voice_fixture", "role_id": "role_fixture",
                   "delivery": {"direction": "", "evidence": []}}
            chunks = [{"chunk_id": f"{number:032x}", "segment": row, "spoken_text": "Hello.",
                       "spoken_start": 0, "spoken_end": 6, "source_ranges": [{"start": 0, "end": 6}],
                       "silent_control": False}]
            chapters.append({"id": chapter_id, "number": number, "title": f"Chapter {number}", "chunks": chunks})
        bundle = {"source": {"book": {"id": "b" * 64, "title": "Fixture"}},
                  "manifest": {"source_fingerprint": "a" * 64}, "chapters": deepcopy(chapters)}
        plan = {"chapters": chapters, "voice_ids": ["voice_fixture"]}
        store = Mock()
        store.load.return_value = {"reference_audio": "/never-read-during-transport-validation.wav"}
        return bundle, config, plan, store

    def test_late_invalid_transport_chapter_fails_before_starting_paid_compute(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError

        deployment = self.deployment()
        deployment.state.update(pod_id="pod_fixture", bootstrapped=True, revision="fixture-revision")
        deployment.start = Mock()
        deployment.stop = Mock()
        bundle, config, plan, store = self.transport_preflight_fixture()
        plan["chapters"][-1]["chunks"][0]["segment"]["delivery"]["direction"] = "<|speaker_id_0|>"
        with patch("epub_tts_narrator.renderer.runpod_config.build_client_config", return_value=config), \
                patch("epub_tts_narrator.renderer.bundle.load_bundle", return_value=bundle), \
                patch("epub_tts_narrator.renderer.book_context.resolve_book_config", return_value=config), \
                patch("epub_tts_narrator.renderer.workflow._preflight", return_value=(config, plan, store)), \
                patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client, \
                patch("epub_tts_narrator.renderer.chapter_client._voice_assets") as conversions:
            with self.assertRaisesRegex(ChapterClientError, "[Cc]hapter 21"):
                self.run_fake(deployment, self.args("encode", "book"))
        deployment.start.assert_not_called()
        deployment.stop.assert_not_called()
        client.assert_not_called()
        conversions.assert_not_called()
        self.assertEqual(self.api.mock_calls, [])

    def test_transport_preflight_checks_full_valid_plan_without_audio_conversion(self):
        deployment = self.deployment()
        bundle, config, plan, store = self.transport_preflight_fixture()
        before = deepcopy(plan)
        with patch("epub_tts_narrator.renderer.runpod_config.build_client_config", return_value=config), \
                patch("epub_tts_narrator.renderer.bundle.load_bundle", return_value=bundle), \
                patch("epub_tts_narrator.renderer.book_context.resolve_book_config", return_value=config), \
                patch("epub_tts_narrator.renderer.workflow._preflight", return_value=(config, plan, store)), \
                patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client, \
                patch("epub_tts_narrator.renderer.chapter_client._voice_assets") as conversions:
            self.assertEqual(deployment.prepare_encode(self.args("encode", "book")), (bundle, config))
        self.assertEqual(plan, before)
        store.load.assert_called_once_with("voice_fixture")
        client.assert_not_called()
        conversions.assert_not_called()
        self.assertEqual(self.api.mock_calls, [])

    def test_shutdown_failure_warns_about_billing_and_redacts_secrets(self):
        fake = self.fake_deployment()
        fake.stop.side_effect = RuntimeError("fixture-provider-secret fixture-ceph-secret fixture-worker-secret")
        with self.assertRaisesRegex(RuntimeError, "billing may continue") as caught:
            self.run_fake(fake, self.args("update"))
        for secret in fake.secrets:
            self.assertNotIn(secret, str(caught.exception))

    def test_operation_failure_skips_stop_delay_and_still_stops(self):
        fake = self.fake_deployment()
        fake.encode.side_effect = RuntimeError("encode fixture failure")
        with self.assertRaisesRegex(RuntimeError, "encode fixture failure"):
            self.run_fake(fake, self.args("encode", "book", "--stop-delay", "300"))
        self.sleep.assert_not_called()
        fake.stop.assert_called_once()

    def test_double_failure_retains_operation_and_stop_diagnostics(self):
        fake = self.fake_deployment()
        fake.install.side_effect = RuntimeError("install fixture failure")
        fake.stop.side_effect = RuntimeError("stop fixture failure fixture-provider-secret")
        with self.assertRaises(RuntimeError) as caught:
            self.run_fake(fake, self.args("update", "--stop-delay", "300"))
        message = str(caught.exception)
        self.assertIn("install fixture failure", message)
        self.assertIn("stop fixture failure", message)
        self.assertIn("billing may continue", message)
        self.assertNotIn("fixture-provider-secret", message)
        self.sleep.assert_not_called()
        fake.stop.assert_called_once()

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

    def test_encode_chapters_only_reports_actual_downloaded_paths(self):
        deployment = self.deployment()
        bundle = {"chapters": [{}, {}, {}]}
        config = {"render": {"output_dir": "/unused/audiobook"}}
        deployment.prepare_encode = Mock(return_value=(bundle, config))
        result = {"chapters": [{"number": 1, "audio_file": None},
                               {"number": 3, "audio_file": "/work/remote-chapters/fixture/chapter.m4a"}]}
        args = self.args("encode", "book", "--chapters-only")
        with patch("epub_tts_narrator.renderer.chapter_client.render_remote", return_value=result) as render:
            self.assertIs(deployment.encode(args), result)
        render.assert_called_once_with(bundle, config, notify=self.notify, chapters_only=True)
        self.assertEqual([call.args[0] for call in self.notify.call_args_list], [
            "Completed chapter audio:", "  Chapter 1 / 3: empty chapter (no speech)",
            "  Chapter 3 / 3: /work/remote-chapters/fixture/chapter.m4a"])

    def test_encode_audiobook_still_reports_published_output_directory(self):
        deployment = self.deployment()
        deployment.prepare_encode = Mock(return_value=({"chapters": [{}]},
                                                       {"render": {"output_dir": "/audio/book"}}))
        result = {"status": "complete"}
        with patch("epub_tts_narrator.renderer.chapter_client.render_remote", return_value=result):
            self.assertIs(deployment.encode(self.args("encode", "book")), result)
        self.notify.assert_called_once_with("Completed audiobook: /audio/book")

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

    def test_batched_backend_requests_cuda13_without_extra_gpu_replicas(self):
        self.settings["deployment"]["tts_backend"] = "sglang"
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(self.api.catalog_gpus.call_args.kwargs["minCudaVersion"], "13.0")
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["gpu"]["minCudaVersion"], "13.0")
        self.assertEqual(spec["gpu"]["count"], 1)
        self.assertEqual(deployment.state["pod_id"], "pod_fixture")
        self.assertEqual(deployment.state_path.stat().st_mode & 0o777, 0o600)

    def capacity_deployment(self):
        """Fake provider with a stopped old host and a separately schedulable Pod."""
        deployment = self.deployment()
        deployment.state.update(pod_id="old_pod", volume_id="volume_fixture", bootstrapped=True,
                                revision="fixture-revision")
        deployment.save()
        old = {"id": "old_pod", "name": "original-worker", "image": deployment.cfg["image"],
               "gpu": {"id": deployment.cfg["gpu_type_id"], "count": 1}, "cloud": "SECURE",
               "dataCenterId": "US-FIXTURE-1", "status": "EXITED", "cost": 2.0,
               "mounts": {"network": [{"volumeId": "volume_fixture", "path": "/workspace"}]}}
        pods = {"old_pod": old}
        self.api.get_volume.return_value = {"id": "volume_fixture", "dataCenterId": "US-FIXTURE-1"}
        self.api.get_pod_details.side_effect = lambda pod_id: deepcopy(pods[pod_id])
        def read(pod_id):
            return {"id": pod_id, "desiredStatus": pods[pod_id]["status"], "costPerHr": pods[pod_id]["cost"],
                    "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}
        self.api.get_pod.side_effect = read
        def create(spec):
            pods["new_pod"] = dict(deepcopy(old), id="new_pod", name=spec["name"], status="RUNNING",
                                   image=spec["image"], mounts=deepcopy(spec["mounts"]),
                                   gpu={"id": spec["gpu"]["id"], "count": 1})
            return deepcopy(pods["new_pod"])
        self.api.create_pod.side_effect = create
        self.api.list_pods.side_effect = lambda: deepcopy(list(pods.values()))
        def start(pod_id):
            if pod_id == "old_pod":
                raise RunpodAPIError("host unavailable", status=400, reason="host_capacity")
            pods[pod_id]["status"] = "RUNNING"
        self.api.start_pod.side_effect = start
        self.api.stop_pod.side_effect = lambda pod_id: pods[pod_id].update(status="EXITED")
        deployment.configure_runtime = Mock(side_effect=lambda pod: pod)
        deployment.ssh = Mock(return_value=b"")
        return deployment, pods, create

    def test_capacity_replacement_preserves_volume_revision_and_application_identity(self):
        deployment, pods, _ = self.capacity_deployment()
        identity = {key: deployment.state[key] for key in ("deployment_id", "api_key", "revision", "bootstrapped")}
        deployment.start()
        self.assertEqual(deployment.pod_id, "new_pod")
        self.assertEqual(deployment.state["previous_pod_ids"], ["old_pod"])
        self.assertEqual(pods["old_pod"]["status"], "EXITED")
        self.assertEqual(identity, {key: deployment.state[key] for key in identity})
        self.assertNotIn("pod_replacement", deployment.state)
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["dataCenterIds"], ["US-FIXTURE-1"])
        self.assertEqual(spec["gpu"]["id"], deployment.cfg["gpu_type_id"])
        self.assertEqual(spec["mounts"], pods["old_pod"]["mounts"])
        self.assertNotEqual(spec["name"], deployment.cfg["name"] + "-" + deployment.state["deployment_id"])
        self.api.start_pod.assert_called_once_with("old_pod")

    def test_fixed_pod_capacity_failure_never_replaces(self):
        self.settings["deployment"]["pod_id"] = "old_pod"
        deployment, _, _ = self.capacity_deployment()
        with self.assertRaisesRegex(RunpodAPIError, "fixed"):
            deployment.start()
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()
        self.assertEqual(deployment.pod_id, "old_pod")

    def test_unknown_start_failure_never_replaces(self):
        for failure in (RunpodAPIError("bad settings", status=400), RunpodAmbiguousOutcome("timeout")):
            with self.subTest(failure=failure):
                deployment, _, _ = self.capacity_deployment()
                self.api.start_pod.side_effect = failure
                with self.assertRaises(RunpodAPIError):
                    deployment.start()
                self.api.create_pod.assert_not_called()
                self.api.create_volume.assert_not_called()

    def test_replacement_requires_confirmed_stopped_matching_source_and_volume(self):
        for change in ("running", "gpu", "image", "datacenter", "mount", "missing_volume", "volume_datacenter"):
            with self.subTest(change=change):
                deployment, pods, _ = self.capacity_deployment()
                if change == "running":
                    pods["old_pod"]["status"] = "RUNNING"
                elif change == "gpu":
                    pods["old_pod"]["gpu"]["id"] = "another-gpu"
                elif change == "image":
                    pods["old_pod"]["image"] = "another-image"
                elif change == "datacenter":
                    pods["old_pod"]["dataCenterId"] = "another-datacenter"
                elif change == "mount":
                    pods["old_pod"]["mounts"]["network"][0]["volumeId"] = "another-volume"
                elif change == "missing_volume":
                    deployment.state.pop("volume_id")
                else:
                    self.api.get_volume.return_value["dataCenterId"] = "another-datacenter"
                with self.assertRaises(RuntimeError):
                    deployment._replace_capacity_blocked_pod()
                self.api.create_pod.assert_not_called()
                self.api.create_volume.assert_not_called()
                self.assertEqual(deployment.pod_id, "old_pod")

    def test_replacement_checks_current_price_before_creating(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.catalog_gpus.return_value["gpus"][0]["price"]["secure"] = 100
        with self.assertRaisesRegex(ValueError, "price"):
            deployment.start()
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()

    def test_lost_replacement_response_recovers_by_unique_intent_without_duplicate(self):
        deployment, _, create = self.capacity_deployment()
        def uncertain(spec):
            saved = json.loads(deployment.state_path.read_text())
            self.assertEqual(saved["pod_id"], "old_pod")
            self.assertEqual(saved["pod_replacement"]["spec"], spec)
            create(spec)
            raise RunpodAmbiguousOutcome("lost response")
        self.api.create_pod.side_effect = uncertain
        deployment.start()
        self.assertEqual(deployment.pod_id, "new_pod")
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()

    def test_pending_replacement_is_reconciled_on_next_start_without_another_post(self):
        deployment, _, create = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        pending = deepcopy(deployment.state["pod_replacement"])
        create(pending["spec"])
        restored = self.deployment()
        restored.configure_runtime = Mock(side_effect=lambda pod: pod)
        restored.ssh = Mock(return_value=b"")
        restored.start()
        self.assertEqual(restored.pod_id, "new_pod")
        self.assertNotIn("pod_replacement", restored.state)
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()

    def test_pending_replacement_missing_or_duplicate_match_never_creates_again(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        for matches in ([], [{"id": "one"}, {"id": "two"}]):
            with self.subTest(matches=matches):
                name = deployment.state["pod_replacement"]["spec"]["name"]
                self.api.list_pods.side_effect = None
                self.api.list_pods.return_value = [dict(row, name=name) for row in matches]
                restored = self.deployment()
                with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
                    restored.start()
                self.assertEqual(restored.pod_id, "old_pod")
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()

    def test_definite_replacement_rejection_keeps_old_id_and_clears_intent(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAPIError("rejected", status=400)
        with self.assertRaises(RunpodAPIError):
            deployment.start()
        self.assertEqual(deployment.pod_id, "old_pod")
        self.assertNotIn("pod_replacement", deployment.state)
        self.api.create_pod.assert_called_once()

    def test_only_one_replacement_is_allowed_per_lifecycle(self):
        deployment, pods, create = self.capacity_deployment()
        def stopped_create(spec):
            result = create(spec)
            pods["new_pod"]["status"] = "EXITED"
            return result
        self.api.create_pod.side_effect = stopped_create
        self.api.start_pod.side_effect = RunpodAPIError("host unavailable", status=400, reason="host_capacity")
        with self.assertRaisesRegex(RunpodAPIError, "only one replacement"):
            deployment.start()
        self.api.create_pod.assert_called_once()
        self.assertEqual(deployment.pod_id, "new_pod")

    def test_candidate_mismatch_or_price_failure_stops_candidate_and_keeps_old_id(self):
        for field, value in (("dataCenterId", "wrong-datacenter"), ("image", "wrong-image"), ("cost", 100),
                             ("gpu", {"id": "wrong-gpu", "count": 1}),
                             ("mounts", {"network": [{"volumeId": "wrong-volume", "path": "/workspace"}]})):
            with self.subTest(field=field):
                deployment, pods, create = self.capacity_deployment()
                def wrong_create(spec):
                    result = create(spec)
                    pods["new_pod"][field] = value
                    return result
                self.api.create_pod.side_effect = wrong_create
                with self.assertRaises(RuntimeError):
                    self.run_fake(deployment, self.args("update"))
                self.assertEqual(deployment.pod_id, "old_pod")
                self.assertEqual(deployment.state["pod_replacement"]["candidate_id"], "new_pod")
                self.assertEqual(pods["new_pod"]["status"], "EXITED")
                self.api.stop_pod.assert_called_with("new_pod")
                self.api.create_volume.assert_not_called()
                # Each subcase owns a new intent, as separate operations would.
                deployment.state.pop("pod_replacement")
                deployment.save()

    def test_resuming_stopped_candidate_requires_a_positive_current_rate(self):
        deployment, pods, create = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        create(deployment.state["pod_replacement"]["spec"])
        pods["new_pod"].update(status="EXITED", cost=0)
        restored = self.deployment()
        with self.assertRaisesRegex(RuntimeError, "rate is unavailable"):
            restored.start()
        self.assertEqual(restored.pod_id, "old_pod")
        self.assertEqual(restored.state["pod_replacement"]["candidate_id"], "new_pod")
        self.api.start_pod.assert_called_once_with("old_pod")
        self.api.create_pod.assert_called_once()

    def test_interrupt_after_replacement_creation_stops_known_candidate(self):
        deployment, pods, create = self.capacity_deployment()
        def interrupted_create(spec):
            create(spec)
            raise KeyboardInterrupt()
        self.api.create_pod.side_effect = interrupted_create
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(deployment, self.args("update"))
        self.assertEqual(pods["new_pod"]["status"], "EXITED")
        self.assertEqual(deployment.pod_id, "old_pod")
        self.api.create_pod.assert_called_once()

    def test_unknown_candidate_shutdown_still_reports_pending_billing(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "billing may continue"):
            self.run_fake(deployment, self.args("update"))
        self.assertIn("pod_replacement", deployment.state)
        self.api.stop_pod.assert_not_called()
        self.api.create_pod.assert_called_once()

    def test_status_reports_pending_candidate_even_when_previous_pod_stopped(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaises(RuntimeError):
            deployment.start()
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=deployment), \
                redirect_stdout(StringIO()) as output:
            run_command(self.args("status"))
        status = json.loads(output.getvalue())
        self.assertEqual(status["status"], "EXITED")
        self.assertIn("unconfirmed", status["pending_replacement"]["status"])
        self.assertEqual(status["pending_replacement"]["name"], deployment.state["pod_replacement"]["spec"]["name"])
        for secret in deployment.secrets:
            self.assertNotIn(secret, output.getvalue())

    def test_encode_after_replacement_uses_runtime_install_and_current_worker_configuration(self):
        deployment, pods, _ = self.capacity_deployment()
        deployment.prepare_encode = Mock()
        deployment.stop_service = Mock()
        deployment.configure = Mock()
        deployment.install = Mock()
        deployment.launch = Mock()
        deployment.encode = Mock()
        deployment.deploy_code = Mock()
        self.assertEqual(self.run_fake(deployment, self.args("encode", "book")), 0)
        deployment.install.assert_called_once_with(full=False)
        deployment.launch.assert_called_once_with(prepare=False)
        deployment.deploy_code.assert_not_called()
        self.assertIn("new_pod-8082", deployment.url)
        self.assertEqual(pods["new_pod"]["status"], "EXITED")

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

    def test_deleted_saved_pod_is_replaced_by_bootstrap_reusing_the_volume(self):
        self.api.get_volume.return_value = {"id": "volume_fixture", "dataCenterId": "US-FIXTURE-1"}
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_deleted_elsewhere"
        deployment.state["volume_id"] = "volume_fixture"
        deployment.save()
        self.api.get_pod.side_effect = RunpodAPIError(
            "Runpod could not find the requested resource (HTTP 404)", status=404)

        deployment.provision()

        self.assertEqual(deployment.pod_id, "pod_fixture")
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()
        self.assertEqual(self.api.create_pod.call_args.args[0]["mounts"]["network"][0]["volumeId"],
                         "volume_fixture")
        notices = " ".join(call.args[0] for call in self.notify.call_args_list)
        self.assertIn("no longer exists", notices)

    def test_stop_treats_a_pod_deleted_elsewhere_as_nothing_to_stop(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_deleted_elsewhere"
        deployment.save()
        self.api.get_pod.side_effect = RunpodAPIError(
            "Runpod could not find the requested resource (HTTP 404)", status=404)

        deployment.stop()

        self.api.stop_pod.assert_not_called()
        notices = " ".join(call.args[0] for call in self.notify.call_args_list)
        self.assertIn("nothing to stop", notices)

    def test_start_reports_a_pod_deleted_elsewhere_with_bootstrap_guidance(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_deleted_elsewhere"
        deployment.save()
        self.api.get_pod.side_effect = RunpodAPIError(
            "Runpod could not find the requested resource (HTTP 404)", status=404)

        with self.assertRaisesRegex(RuntimeError, "runpod bootstrap"):
            deployment.start()

    def test_status_reports_a_pod_deleted_elsewhere(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_deleted_elsewhere"
        deployment.save()
        self.api.get_pod.side_effect = RunpodAPIError(
            "Runpod could not find the requested resource (HTTP 404)", status=404)
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=deployment), \
                redirect_stdout(StringIO()) as output:
            run_command(self.args("status"))
        status = json.loads(output.getvalue())
        self.assertIn("deleted", status["status"])
        self.assertIn("bootstrap", status["status"])

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
        self.assertTrue(deployment.state["pod_creation_pending"])
        with self.assertRaisesRegex(RuntimeError, "rate exceeds"):
            self.deployment().provision()
        self.api.create_pod.assert_called_once()

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

    def test_startup_repairs_restored_ssh_key_permissions_before_launch(self):
        deployment = self.deployment()
        saved, restored = self.directory / "saved host keys", self.directory / "local ssh"
        saved.mkdir()
        restored.mkdir()
        for algorithm in ("ed25519", "rsa"):
            for suffix in ("", ".pub"):
                key = saved / f"ssh_host_{algorithm}_key{suffix}"
                key.write_text("fixture key")
                key.chmod(0o666)
        receipt = self.directory / "launch.json"
        starter = self.directory / "start fixture.sh"
        starter.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} - <<'PY'\nimport json\nfrom pathlib import Path\n"
            f"directory = Path({str(restored)!r})\n"
            "modes = {key.name: key.stat().st_mode & 0o777 for key in directory.iterdir()}\n"
            "assert all(mode == (0o644 if name.endswith('.pub') else 0o600) "
            "for name, mode in modes.items()), modes\n"
            f"Path({str(receipt)!r}).write_text(json.dumps(modes))\nPY\n")
        starter.chmod(0o700)
        script = deployment.startup_command()[2]
        script = script.replace("/workspace/.epub-tts-runtime/ssh-host-keys", shlex.quote(str(saved)))
        script = script.replace("/etc/ssh/", shlex.quote(str(restored)) + "/")
        script = script.replace("/start.sh", shlex.quote(str(starter)))
        # The fixture must restore its existing keys; never run the machine's
        # ssh-keygen even if a regression selects the generation branch.
        script = script.replace("ssh-keygen -A", "false")
        subprocess.run(["bash", "-c", script], check=True, capture_output=True, timeout=10)
        modes = json.loads(receipt.read_text())
        self.assertEqual(len(modes), 4)
        for name in modes:
            self.assertEqual((restored / name).read_text(), "fixture key")
            self.assertEqual((saved / name).stat().st_mode & 0o777, 0o666)

    def test_changed_port_updates_stopped_pod_without_replacing_volume(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        pod = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8081/http"],
               "dockerStartCmd": deployment.startup_command()}
        updated = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8082/http"],
                   "dockerStartCmd": deployment.startup_command()}
        self.api.get_pod.return_value = updated
        self.assertEqual(deployment.configure_runtime(pod), updated)
        self.api.update_pod.assert_called_once_with("pod_fixture", {"ports": ["22/tcp", "8082/http"]})
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()
        self.api.start_pod.assert_not_called()
        self.api.stop_pod.assert_not_called()

    def test_unchanged_runtime_does_not_mutate_running_worker(self):
        deployment = self.deployment()
        pod = {"desiredStatus": "RUNNING", "ports": ["8082/http", "22/tcp"],
               "dockerStartCmd": deployment.startup_command()}
        self.assertIs(deployment.configure_runtime(pod), pod)
        self.api.update_pod.assert_not_called()
        self.api.stop_pod.assert_not_called()

    def test_changed_startup_stops_before_patching_only_command(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        pod = {"desiredStatus": "RUNNING", "ports": deployment.service_ports(),
               "dockerStartCmd": ["bash", "-lc", "old startup"]}
        updated = {"desiredStatus": "EXITED", "ports": deployment.service_ports(),
                   "dockerStartCmd": deployment.startup_command()}
        calls = Mock()
        deployment.stop = calls.stop
        calls.attach_mock(self.api.update_pod, "update")
        self.api.get_pod.return_value = updated
        self.assertEqual(deployment.configure_runtime(pod), updated)
        self.assertEqual([call[0] for call in calls.mock_calls], ["stop", "update"])
        self.api.update_pod.assert_called_once_with("pod_fixture", {"cmd": deployment.startup_command()})
        self.api.create_volume.assert_not_called()
        self.api.create_pod.assert_not_called()
        self.api.start_pod.assert_not_called()

    def test_uncertain_port_update_is_read_back_without_retrying(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        self.api.update_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        old = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8081/http"],
               "dockerStartCmd": deployment.startup_command()}
        self.api.get_pod.return_value = old
        with self.assertRaisesRegex(RuntimeError, "runtime update was not confirmed"):
            deployment.configure_runtime(old)
        self.assertEqual(self.api.update_pod.call_count, 1)
        self.api.start_pod.assert_not_called()

    def test_runtime_update_requires_readback_of_command_and_ports(self):
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        old = {"desiredStatus": "EXITED", "ports": ["22/tcp", "8081/http"],
               "dockerStartCmd": ["bash", "-lc", "old startup"]}
        self.api.get_pod.return_value = {**old, "ports": deployment.service_ports()}
        with self.assertRaisesRegex(RuntimeError, "runtime update was not confirmed"):
            deployment.configure_runtime(old)
        self.api.update_pod.assert_called_once_with("pod_fixture", {
            "ports": deployment.service_ports(), "cmd": deployment.startup_command()})
        self.api.start_pod.assert_not_called()

    def test_dead_supervisor_fails_launch_without_waiting_full_timeout(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        deployment.ssh = Mock(side_effect=[b"", b"", b"SSH warning.\nEPUB_TTS_WORKER_STATUS=EXITED\n", b"Bind failed.\n"])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
            client.return_value.request.side_effect = ChapterClientError("unavailable")
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                deployment.launch()
        self.assertEqual(client.return_value.request.call_count, 1)
        self.assertIn("tail -n 60", deployment.ssh.call_args.args[0])
        self.sleep.assert_not_called()

    def test_command_output_hides_only_exact_probe_lines_and_keeps_capture_and_redaction(self):
        deployment = self.deployment()
        output = (b"EPUB_TTS_WORKER_STATUS=RUNNING\n"
                  b"EPUB_TTS_WORKER_STATUS=EXITED\r\n"
                  b"Warning mentions EPUB_TTS_WORKER_STATUS=EXITED; retain this diagnostic.\n"
                  b"Failure detail with fixture-provider-secret\n")
        process = Mock(stdout=BytesIO(output), returncode=0)
        process.poll.return_value = 0
        with patch("epub_tts_narrator.renderer.runpod.subprocess.Popen", return_value=process):
            self.assertEqual(deployment._run(["fixture-command"]), output)
        displayed = [call.args[0] for call in self.notify.call_args_list]
        self.assertEqual(len(displayed), 2)
        self.assertIn("Warning mentions EPUB_TTS_WORKER_STATUS=EXITED", displayed[0])
        self.assertIn("Failure detail", displayed[1])
        self.assertNotIn("fixture-provider-secret", displayed[1])
        self.assertTrue(process.stdout.closed)

    def test_launch_waiting_updates_are_throttled_and_ready_reports_elapsed(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError

        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        clock = [0.0]
        requests = [0]

        def request(*args):
            clock[0] += 10
            requests[0] += 1
            if requests[0] <= 5:
                raise ChapterClientError("loading")
            return {"maintenance": False}

        def ssh(command, **kwargs):
            if command.startswith("python3 -c "):
                clock[0] += 2
                return b"EPUB_TTS_WORKER_STATUS=RUNNING\n"
            return b""

        deployment.ssh = Mock(side_effect=ssh)
        self.sleep.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        with (patch("epub_tts_narrator.renderer.runpod.time.monotonic", side_effect=lambda: clock[0]),
              patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client):
            client.return_value.request.side_effect = request
            deployment.launch()
        messages = [call.args[0] for call in self.notify.call_args_list]
        self.assertEqual([message for message in messages if message.startswith("Still waiting")], [
            "Still waiting for Drama 3 and the chapter service (46s elapsed)…",
            "Still waiting for Drama 3 and the chapter service (80s elapsed)…",
        ])
        self.assertEqual(messages[-1], "Chapter service ready (95s startup).")
        self.assertEqual(requests[0], 6)

    def test_launch_timeout_still_prints_diagnostic_tail(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError

        deployment = self.deployment()
        deployment.cfg["startup_timeout_seconds"] = 1
        deployment.ssh = Mock(return_value=b"EPUB_TTS_WORKER_STATUS=RUNNING\n")
        with (patch("epub_tts_narrator.renderer.runpod.time.monotonic", side_effect=[0, 0, 2, 2]),
              patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client):
            client.return_value.request.side_effect = ChapterClientError("loading")
            with self.assertRaisesRegex(RuntimeError, "failed to start before"):
                deployment.launch()
        self.assertIn("tail -n 60", deployment.ssh.call_args.args[0])
        self.assertFalse(any(call.args[0].startswith("Chapter service ready") for call in self.notify.call_args_list))

    def test_prepared_launch_uses_maintenance_and_encode_launch_consumes_jobs(self):
        for prepare in (True, False):
            with self.subTest(prepare=prepare):
                deployment = self.deployment()
                deployment.state["pod_id"] = "pod_fixture"
                deployment.ssh = Mock(return_value=b"")
                with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
                    client.return_value.request.return_value = {"maintenance": prepare}
                    deployment.launch(prepare=prepare)
                launched = next(call.args[0] for call in deployment.ssh.call_args_list if call.args[0].startswith("nohup "))
                action = "serve-maintenance" if prepare else "serve"
                self.assertIn(deployment.worker_command(action) + " > ", launched)
                commands = [call.args[0] for call in deployment.ssh.call_args_list]
                self.assertEqual(deployment.worker_command("prepare") in commands, prepare)

    def test_launch_rejects_wrong_mode_instead_of_smoking_or_encoding(self):
        for prepare, reported in ((True, {}), (True, {"maintenance": False}),
                                  (False, {"maintenance": True})):
            with self.subTest(prepare=prepare, reported=reported):
                deployment = self.deployment()
                deployment.state["pod_id"] = "pod_fixture"
                deployment.ssh = Mock(return_value=b"")
                with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
                    client.return_value.request.return_value = reported
                    with self.assertRaisesRegex(RuntimeError, "wrong maintenance mode"):
                        deployment.launch(prepare=prepare)

    def test_healthy_supervisor_ignores_ssh_warning_output(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError
        deployment = self.deployment()
        deployment.state["pod_id"] = "pod_fixture"
        warning = (b"WARNING: connection is not using a post-quantum key exchange algorithm.\n"
                   b"Banner mentions EPUB_TTS_WORKER_STATUS=EXITED without reporting it.\n"
                   b"EPUB_TTS_WORKER_STATUS=RUNNING\n")
        deployment.ssh = Mock(side_effect=[b"", b"", warning])
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
            client.return_value.request.side_effect = [ChapterClientError("loading"), {}]
            deployment.launch()
        self.assertEqual(client.return_value.request.call_count, 2)
        self.assertEqual(deployment.ssh.call_count, 3)
        self.assertFalse(any("tail -n 60" in call.args[0] for call in deployment.ssh.call_args_list))
        self.sleep.assert_called_once_with(5)

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

    H100 = "NVIDIA H100 80GB HBM3"

    def isolated_state(self, label):
        self.settings["deployment"]["state_dir"] = str(self.directory / f"state-{label}")

    def h100_row(self, **overrides):
        row = {"id": self.H100, "name": "H100 SXM", "availability": "HIGH",
               "price": {"secure": 2.59},
               "dataCenters": [{"id": self.settings["deployment"]["data_center_id"], "availability": "HIGH"}]}
        row.update(overrides)
        return row

    def fallback_row(self):
        return deepcopy(self.rtx_row)

    def test_available_affordable_h100_is_preferred_for_new_creations(self):
        deployment = self.deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row(), self.h100_row()]}
        deployment.provision()
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["gpu"]["id"], self.H100)
        self.assertEqual(deployment.state["gpu_type_id"], self.H100)
        self.assertEqual(spec["dataCenterIds"], ["US-FIXTURE-1"])
        self.assertEqual(spec["mounts"]["network"][0]["volumeId"], "volume_fixture")

    def test_chosen_gpu_is_persisted_before_the_creation_post(self):
        deployment = self.deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.h100_row()]}
        saved_at_post = []
        def create(spec):
            saved_at_post.append(json.loads(deployment.state_path.read_text()).get("gpu_type_id"))
            return {"id": "pod_fixture", "cost": 2.59, "status": "PROVISIONING"}
        self.api.create_pod.side_effect = create
        deployment.provision()
        self.assertEqual(saved_at_post, [self.H100])

    def test_pending_h100_creation_checks_placement_and_rate_before_adoption(self):
        for index, change in enumerate(({"gpu": {"id": "wrong-gpu", "count": 1}}, {"cost": 2.60})):
            with self.subTest(change=change):
                self.isolated_state(f"reconcile-{index}")
                deployment = self.deployment()
                self.api.catalog_gpus.return_value = {"gpus": [self.h100_row()]}
                self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
                self.api.list_pods.return_value = []
                with self.assertRaises(RunpodAmbiguousOutcome):
                    deployment.provision()
                spec = self.api.create_pod.call_args.args[0]
                candidate = {**deepcopy(spec), "id": "recovered_pod", "dataCenterId": "US-FIXTURE-1",
                             "cost": 2.59, **change}
                self.api.get_pod_details.return_value = candidate
                self.api.get_pod_details.side_effect = None
                self.api.list_pods.return_value = [{"id": "recovered_pod", "name": spec["name"]}]
                restored = self.deployment()
                with self.assertRaises(RuntimeError):
                    restored.provision()
                self.assertEqual(restored.pod_id, "recovered_pod")
                self.assertTrue(restored.state["pod_creation_pending"])
                self.api.create_pod.reset_mock()

    def test_ambiguous_h100_create_keeps_the_same_gpu_on_restart(self):
        deployment = self.deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.h100_row(), self.fallback_row()]}
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("uncertain")
        name = deployment.cfg["name"] + "-" + deployment.state["deployment_id"]
        self.api.list_pods.return_value = [{"name": name, "id": "recovered_pod", "costPerHr": "2.59"}]
        deployment.provision()
        self.assertEqual(deployment.state["gpu_type_id"], self.H100)
        self.api.catalog_gpus.return_value = {"gpus": []}
        restored = self.deployment()
        self.assertEqual(restored.gpu_type_id, self.H100)
        self.api.get_pod.return_value = {"desiredStatus": "RUNNING", "costPerHr": 2.59,
                                         "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}
        restored.configure_runtime = Mock(side_effect=lambda pod: pod)
        restored.ssh = Mock(return_value=b"")
        restored.start()
        self.api.create_pod.assert_called_once()
        self.assertEqual(self.api.catalog_gpus.call_count, 1)

    def test_over_limit_unstocked_misplaced_or_malformed_h100_falls_back(self):
        for index, row in enumerate((self.h100_row(price=2.60), self.h100_row(price=2.61), self.h100_row(availability="NONE"),
                                     self.h100_row(dataCenters=[{"id": "US-OTHER-1", "availability": "HIGH"}]),
                                     self.h100_row(price=True), self.h100_row(price="2.59"))):
            with self.subTest(h100=row):
                self.isolated_state(f"fallback-{index}")
                self.api.catalog_gpus.return_value = {"gpus": [row, self.fallback_row()]}
                deployment = self.deployment()
                deployment.provision()
                spec = self.api.create_pod.call_args.args[0]
                self.assertEqual(spec["gpu"]["id"], self.fallback_row()["id"])
                self.assertEqual(deployment.state["gpu_type_id"], self.fallback_row()["id"])
                self.api.create_pod.reset_mock()

    def test_prefer_h100_false_pins_the_configured_gpu(self):
        self.settings["deployment"]["prefer_h100"] = False
        deployment = self.deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.h100_row(), self.fallback_row()]}
        deployment.provision()
        self.assertEqual(self.api.create_pod.call_args.args[0]["gpu"]["id"], self.fallback_row()["id"])

    def test_explicit_pinned_h100_can_use_a_higher_configured_cap(self):
        self.settings["deployment"].update(prefer_h100=False, gpu_type_id=self.H100, max_hourly_price=4.10)
        self.api.catalog_gpus.return_value = {"gpus": [self.h100_row(price={"secure": 3.0})]}
        self.api.create_pod.return_value = {"id": "pinned_h100", "cost": 3.0}
        deployment = self.deployment()
        deployment.provision()
        self.assertEqual(deployment.pod_id, "pinned_h100")

    def test_legacy_source_gpu_is_read_before_a_cross_gpu_replacement(self):
        deployment, pods, _ = self.capacity_deployment()
        pods["old_pod"]["gpu"]["id"] = self.H100
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row()]}
        deployment.start()
        self.assertEqual(deployment.state["gpu_type_id"], self.fallback_row()["id"])
        self.assertEqual(pods["old_pod"]["status"], "EXITED")

    def test_actual_h100_rate_must_be_strictly_under_the_preference_limit(self):
        for index, (cost, created) in enumerate(((2.59, True), (2.60, False), (2.65, False))):
            with self.subTest(cost=cost):
                self.isolated_state(f"actual-{index}")
                deployment = self.deployment()
                self.api.catalog_gpus.return_value = {"gpus": [self.h100_row()]}
                self.api.create_pod.return_value = {"id": "pod_fixture", "cost": cost, "status": "PROVISIONING"}
                if created:
                    deployment.provision()
                    self.assertEqual(deployment.pod_id, "pod_fixture")
                else:
                    with self.assertRaisesRegex(RuntimeError, "rate exceeds"):
                        deployment.provision()
                    # The known ID is retained so shutdown can stop the worker.
                    self.assertEqual(deployment.pod_id, "pod_fixture")
        self.assertNotIn("pod_creation_pending", deployment.state)

    def test_capacity_replacement_prefers_h100_target_and_preserves_volume_and_old_pod(self):
        deployment, pods, _ = self.capacity_deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row(), self.h100_row()]}
        deployment.start()
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["gpu"]["id"], self.H100)
        self.assertEqual(spec["mounts"], pods["old_pod"]["mounts"])
        self.assertEqual(spec["dataCenterIds"], ["US-FIXTURE-1"])
        self.assertEqual(pods["old_pod"]["status"], "EXITED")
        self.assertEqual(deployment.state["gpu_type_id"], self.H100)
        self.api.create_pod.assert_called_once()
        self.api.create_volume.assert_not_called()

    def test_capacity_replacement_validates_the_old_pod_against_its_source_gpu(self):
        deployment, pods, _ = self.capacity_deployment()
        deployment.state["gpu_type_id"] = self.H100
        deployment.save()
        pods["old_pod"]["gpu"] = {"id": self.H100, "count": 1}
        # No affordable H100 in today's catalog: the target falls back to RTX.
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row()]}
        deployment.start()
        spec = self.api.create_pod.call_args.args[0]
        self.assertEqual(spec["gpu"]["id"], deployment.cfg["gpu_type_id"])
        self.assertEqual(pods["old_pod"]["status"], "EXITED")
        self.assertEqual(deployment.state["gpu_type_id"], deployment.cfg["gpu_type_id"])
        self.api.create_pod.assert_called_once()

    def test_pending_replacement_uses_recorded_target_without_todays_catalog(self):
        deployment, pods, create = self.capacity_deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row(), self.h100_row()]}
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        pending = deepcopy(deployment.state["pod_replacement"])
        self.assertEqual(pending["spec"]["gpu"]["id"], self.H100)
        self.assertEqual(pending["source_gpu_id"], deployment.cfg["gpu_type_id"])
        self.assertEqual(pending["fallback_gpu_id"], deployment.cfg["gpu_type_id"])
        self.assertTrue(pending["prefer_h100"])
        create(pending["spec"])
        self.api.catalog_gpus.return_value = {"gpus": []}
        restored = self.deployment()
        restored.configure_runtime = Mock(side_effect=lambda pod: pod)
        restored.ssh = Mock(return_value=b"")
        restored.start()
        self.assertEqual(restored.pod_id, "new_pod")
        self.assertEqual(restored.state["gpu_type_id"], self.H100)
        self.assertEqual(restored.gpu_type_id, self.H100)
        self.assertNotIn("pod_replacement", restored.state)
        self.api.create_pod.assert_called_once()

    def test_pending_replacement_rejects_changed_gpu_configuration_without_another_post(self):
        for field, value, default in (("gpu_type_id", "NVIDIA H100 NVL", self.fallback_row()["id"]),
                                      ("prefer_h100", False, True)):
            with self.subTest(field=field):
                deployment, _, _ = self.capacity_deployment()
                self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
                with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
                    deployment.start()
                self.settings["deployment"][field] = value
                restored = self.deployment()
                with self.assertRaisesRegex(RuntimeError, "differs from the configured deployment"):
                    restored.start()
                self.assertEqual(restored.pod_id, "old_pod")
                self.settings["deployment"][field] = default
        self.api.create_pod.assert_called_once()

    def test_legacy_pending_replacement_without_gpu_fields_keeps_old_same_gpu_semantics(self):
        deployment, pods, create = self.capacity_deployment()
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        intent = deployment.state["pod_replacement"]
        for key in ("source_gpu_id", "fallback_gpu_id", "prefer_h100"):
            intent.pop(key, None)
        deployment.save()
        create(intent["spec"])
        restored = self.deployment()
        restored.configure_runtime = Mock(side_effect=lambda pod: pod)
        restored.ssh = Mock(return_value=b"")
        restored.start()
        self.assertEqual(restored.pod_id, "new_pod")

    def test_pending_replacement_adoption_enforces_actual_h100_rate(self):
        deployment, pods, create = self.capacity_deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row(), self.h100_row()]}
        self.api.create_pod.side_effect = RunpodAmbiguousOutcome("lost response")
        with self.assertRaisesRegex(RuntimeError, "UNCONFIRMED"):
            deployment.start()
        create(deployment.state["pod_replacement"]["spec"])
        pods["new_pod"]["cost"] = 2.65
        restored = self.deployment()
        with self.assertRaisesRegex(RuntimeError, "rate is unavailable"):
            restored.start()
        self.assertEqual(restored.pod_id, "old_pod")
        self.assertEqual(restored.state["pod_replacement"]["candidate_id"], "new_pod")
        self.api.create_pod.assert_called_once()

    def test_existing_healthy_pod_is_not_replaced_for_h100_preference(self):
        deployment, _, _ = self.capacity_deployment()
        self.api.catalog_gpus.return_value = {"gpus": [self.fallback_row(), self.h100_row()]}
        self.api.start_pod.side_effect = None
        deployment.start()
        self.api.create_pod.assert_not_called()
        self.api.create_volume.assert_not_called()
        self.assertEqual(deployment.pod_id, "old_pod")


if __name__ == "__main__":
    unittest.main()
