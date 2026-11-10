"""Lock-free service reuse, the live-session stop gate, and ensure_service.

All service contact is a fake ChapterClient; no network, no Runpod API, and
no deployment lock is taken by the read-only paths.
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

from epub_tts_narrator.renderer.chapter_client import ChapterClientError
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.runpod import (Deployment, SERVICE_MAINTENANCE, _live_work_refusal,
                                               add_parser, configuration_stamp, ensure_service,
                                               probe_service, read_state, run_command)
from epub_tts_narrator.renderer.runpod_config import DEPLOYMENT_DEFAULTS


class RunpodReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = self.directory / "epub-tts-narrator.toml"
        self.overrides = self.directory / "runpod.local.toml"
        self.config.write_text("[runpod]\n")
        self.overrides.write_text("[render]\n")
        renderer = deepcopy(DEFAULTS)
        renderer["chapter_server"]["runpod_api_key"] = "fixture-provider-secret"
        self.settings = {
            "deployment": dict(deepcopy(DEPLOYMENT_DEFAULTS), state_dir=str(self.directory / "state"),
                               image="fixture/image@sha256:" + "a" * 64),
            "runpod_api_key": "fixture-provider-secret",
            "renderer_config": renderer,
            "override_raw": {},
            "config_path": str(self.config),
            "overrides_path": str(self.overrides),
        }
        self.api = Mock()
        # A running worker that confirms its stop on the read-back.
        self.api.get_pod.side_effect = [{"desiredStatus": "RUNNING"}, {"desiredStatus": "EXITED"}]
        self.notify = Mock()
        self.requests = []

    def state_dir(self):
        return Path(self.settings["deployment"]["state_dir"])

    def write_state(self, **overrides):
        state = {"pod_id": "pod_fixture", "api_key": "fixture-worker-secret", "bootstrapped": True,
                 "configuration_stamp": configuration_stamp(self.settings)}
        drop = overrides.pop("drop", ())
        state.update(overrides)
        for key in drop:
            state.pop(key, None)
        self.state_dir().mkdir(parents=True, exist_ok=True)
        (self.state_dir() / "state.json").write_text(json.dumps(state))

    def state_bytes(self):
        return (self.state_dir() / "state.json").read_bytes()

    def state_identity(self):
        """Write-sensitive file identity: any rewrite changes mtime and inode."""
        info = (self.state_dir() / "state.json").stat()
        return (info.st_mtime_ns, info.st_ino, info.st_size)

    def serve(self, responses=None, error=None):
        """Patch ChapterClient with a fake that records (method, path) requests."""
        requests = self.requests
        responses = responses or {}

        class Client:
            def __init__(self, settings):
                self.settings = settings

            def request(self, method, path, payload=None, *, timeout=None):
                requests.append((method, path))
                if error is not None:
                    raise error
                return responses.get(path.split("?")[0], {})

        return patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", Client)

    def args(self, *extra):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest="command", required=True))
        return parser.parse_args(["runpod", *extra])

    def run_stop(self, *extra):
        with patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings), \
                patch("epub_tts_narrator.renderer.runpod.RunpodAPI", return_value=self.api), \
                patch("epub_tts_narrator.renderer.runpod.time.sleep"), \
                redirect_stdout(StringIO()):
            return run_command(self.args("stop", *extra))

    def fake_deployment(self, *, reuse=False):
        fake = Mock()
        fake.pod_id = "pod_fixture"
        fake.url = "https://pod_fixture-8082.proxy.runpod.net"
        fake.cfg = self.settings["deployment"]
        fake.state = {"api_key": "fixture-worker-secret", "deployment_id": "d" * 12}
        fake.secrets = {"fixture-worker-secret"}
        fake.reuse_service.return_value = reuse
        return fake

    def ensure_service_patches(self, fake, build):
        return dict(
            load_settings=patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=self.settings),
            deployment=patch("epub_tts_narrator.renderer.runpod.Deployment", return_value=fake),
            build=patch("epub_tts_narrator.renderer.runpod_config.build_client_config", side_effect=build),
            getsignal=patch("epub_tts_narrator.renderer.runpod.signal.getsignal", side_effect=lambda sig: "previous"),
            signal=patch("epub_tts_narrator.renderer.runpod.signal.signal"))

    def test_stop_parses_the_force_flag(self):
        self.assertFalse(self.args("stop").force)
        self.assertTrue(self.args("stop", "--force").force)

    def test_stop_refuses_while_a_connected_client_owns_work(self):
        self.write_state()
        with self.serve({"v1/jobs": {"jobs": [
                {"id": "a" * 8, "state": "complete", "owner_connected": True, "paused": False}]}}):
            with self.assertRaisesRegex(ValueError, "--force") as caught:
                self.run_stop()
        self.assertNotIn("fixture-worker-secret", str(caught.exception))
        self.api.stop_pod.assert_not_called()
        self.assertIn(("GET", "v1/jobs?limit=256"), self.requests)

    def test_stop_refuses_while_jobs_are_queued_generating_or_assembling(self):
        for state in ("queued", "generating", "assembling"):
            with self.subTest(state=state):
                self.requests.clear()
                self.write_state()
                with self.serve({"v1/jobs": {"jobs": [
                        {"id": "a" * 8, "state": state, "owner_connected": False, "paused": True}]}}):
                    with self.assertRaisesRegex(ValueError, "--force"):
                        self.run_stop()
                self.api.stop_pod.assert_not_called()

    def test_stop_proceeds_when_every_job_is_terminal(self):
        self.write_state()
        with self.serve({"v1/jobs": {"jobs": [
                {"id": "a" * 8, "state": "complete", "owner_connected": False, "paused": False},
                {"id": "b" * 8, "state": "failed", "owner_connected": False, "paused": False},
                {"id": "c" * 8, "state": "cancelled", "owner_connected": False, "paused": False}]}}):
            self.assertEqual(self.run_stop(), 0)
        self.api.stop_pod.assert_called_once_with("pod_fixture")

    def test_stop_proceeds_when_the_service_is_unreachable(self):
        self.write_state()
        with self.serve(error=ChapterClientError("Chapter service connection failed")):
            self.assertEqual(self.run_stop(), 0)
        self.api.stop_pod.assert_called_once_with("pod_fixture")

    def test_stop_force_overrides_live_work(self):
        self.write_state()
        with self.serve({"v1/jobs": {"jobs": [
                {"id": "a" * 8, "state": "generating", "owner_connected": True, "paused": False}]}}):
            self.assertEqual(self.run_stop("--force"), 0)
        self.api.stop_pod.assert_called_once_with("pod_fixture")

    def test_stop_without_a_worker_proceeds_without_probing(self):
        with self.serve():
            self.assertEqual(self.run_stop(), 0)
        self.assertEqual(self.requests, [])
        self.api.stop_pod.assert_not_called()

    def test_live_work_refusal_counts_live_rows_and_names_the_flag(self):
        constructed, requests = [], []

        class Client:
            def __init__(self, settings):
                constructed.append(settings)

            def request(self, method, path, payload=None, *, timeout=None):
                requests.append((method, path))
                return {"jobs": [{"state": "complete", "owner_connected": False},
                                 {"state": "queued", "owner_connected": False},
                                 {"state": "complete", "owner_connected": True}]}

        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", Client):
            refusal = _live_work_refusal("https://pod_fixture-8082.proxy.runpod.net", "fixture-worker-secret")
        self.assertEqual(constructed[0]["timeout_seconds"], 5)
        self.assertEqual(requests, [("GET", "v1/jobs?limit=256")])
        self.assertIn("2 chapter job(s)", refusal)
        self.assertIn("--force", refusal)
        self.assertNotIn("fixture-worker-secret", refusal)

    def test_live_work_refusal_treats_malformed_answers_as_no_live_work(self):
        for response in (None, "text", {}, {"jobs": "nope"}, {"jobs": [None, "row"]}):
            with self.subTest(response=response):
                class Client:
                    def __init__(self, settings):
                        pass

                    def request(self, method, path, payload=None, *, timeout=None):
                        return response

                with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", Client):
                    self.assertIsNone(_live_work_refusal("https://pod_fixture-8082.proxy.runpod.net", "key"))

    def test_deployment_live_work_refusal_needs_a_worker_and_key(self):
        deployment = Deployment(self.settings, api=self.api, notify=self.notify)
        with patch("epub_tts_narrator.renderer.chapter_client.ChapterClient") as client:
            self.assertIsNone(deployment.live_work_refusal())
        client.assert_not_called()

    def test_read_state_parses_state_without_creating_anything(self):
        self.assertIsNone(read_state(self.state_dir()))
        self.assertFalse(self.state_dir().exists())
        self.write_state()
        self.assertEqual(read_state(self.state_dir())["pod_id"], "pod_fixture")
        (self.state_dir() / "state.json").write_text("not json")
        self.assertIsNone(read_state(self.state_dir()))

    def test_probe_returns_a_client_config_for_a_healthy_running_service(self):
        self.write_state()
        before, identity = self.state_bytes(), self.state_identity()
        with self.serve({"v1/capabilities": {"maintenance": False, "ephemeral_runs": 1, "warm_activation": 1}}):
            result = probe_service(self.settings)
        self.assertEqual(result, {"url": "https://pod_fixture-8082.proxy.runpod.net",
                                  "api_key": "fixture-worker-secret",
                                  "timeout_seconds": DEFAULTS["remote_chapters"]["timeout_seconds"]})
        self.assertEqual(self.requests, [("GET", "v1/capabilities")])
        self.assertEqual(self.state_bytes(), before)
        self.assertEqual(self.state_identity(), identity)
        self.assertFalse((self.state_dir() / "operation.lock").exists())

    def test_probe_reports_maintenance_without_activating(self):
        self.write_state()
        with self.serve({"v1/capabilities": {"maintenance": True, "ephemeral_runs": 1, "warm_activation": 1}}):
            self.assertEqual(probe_service(self.settings), SERVICE_MAINTENANCE)
        self.assertEqual(self.requests, [("GET", "v1/capabilities")])

    def test_probe_returns_none_when_the_state_gates_fail_without_probing(self):
        self.state_dir().mkdir(parents=True, exist_ok=True)
        for state in ({"bootstrapped": False}, {"drop": ("pod_id",)}, {"drop": ("api_key",)},
                      {"drop": ("configuration_stamp",)}):
            with self.subTest(state=state):
                self.write_state(**state)
                self.requests.clear()
                with self.serve():
                    self.assertIsNone(probe_service(self.settings))
                self.assertEqual(self.requests, [])

    def test_probe_returns_none_without_any_state_and_creates_nothing(self):
        with self.serve():
            self.assertIsNone(probe_service(self.settings))
        self.assertEqual(self.requests, [])
        self.assertFalse(self.state_dir().exists())

    def test_probe_returns_none_when_the_stamp_service_or_capabilities_do_not_match(self):
        self.write_state()
        self.config.write_text("[runpod]\nidle_timeout_seconds = 300\n")
        with self.serve():
            self.assertIsNone(probe_service(self.settings))
        self.assertEqual(self.requests, [])

        self.write_state()
        with self.serve(error=ChapterClientError("Chapter service connection failed")):
            self.assertIsNone(probe_service(self.settings))
        self.assertEqual(self.requests, [("GET", "v1/capabilities")])

        self.requests.clear()
        with self.serve({"v1/capabilities": {"maintenance": False, "ephemeral_runs": 0, "warm_activation": 1}}):
            self.assertIsNone(probe_service(self.settings))
        self.assertEqual(self.requests, [("GET", "v1/capabilities")])

    def test_ensure_service_joins_a_healthy_service_without_lock_or_deployment(self):
        self.write_state()
        before, identity = self.state_bytes(), self.state_identity()
        built = {}

        def build(renderer_config, url, api_key, override_raw=None):
            built.update(url=url, api_key=api_key, override_raw=override_raw)
            return {"remote_chapters": {"url": url, "api_key": api_key}}

        with self.serve({"v1/capabilities": {"maintenance": False, "ephemeral_runs": 1, "warm_activation": 1}}), \
                patch("epub_tts_narrator.renderer.runpod_config.build_client_config", side_effect=build), \
                patch("epub_tts_narrator.renderer.runpod.Deployment") as deployment, \
                patch("epub_tts_narrator.renderer.runpod.signal.signal") as signal_signal:
            result = ensure_service(self.settings, notify=self.notify)
        self.assertEqual(result, {"remote_chapters": {"url": "https://pod_fixture-8082.proxy.runpod.net",
                                                      "api_key": "fixture-worker-secret"}})
        self.assertEqual(built, {"url": "https://pod_fixture-8082.proxy.runpod.net",
                                 "api_key": "fixture-worker-secret",
                                 "override_raw": self.settings["override_raw"]})
        deployment.assert_not_called()
        signal_signal.assert_not_called()
        self.assertEqual(self.state_bytes(), before)
        self.assertEqual(self.state_identity(), identity)
        self.assertFalse((self.state_dir() / "operation.lock").exists())
        self.notify.assert_called_once_with("Reusing the running chapter service and its loaded inference model.")

    def test_ensure_service_drives_the_locked_ready_flow_when_no_service_runs(self):
        fake = self.fake_deployment()
        sentinel = {"remote_chapters": {}}
        patches = self.ensure_service_patches(fake, Mock(return_value=sentinel))
        with patches["load_settings"], patches["deployment"], patches["build"], \
                patches["getsignal"], patches["signal"], redirect_stdout(StringIO()):
            result = ensure_service(self.settings, notify=self.notify)
        self.assertIs(result, sentinel)
        fake.reuse_service.assert_called_once_with()
        fake.provision.assert_called_once_with()
        fake.start.assert_called_once_with()
        fake.ensure_ready.assert_called_once_with(source="local", update=True)
        self.assertEqual([name for name, _, _ in fake.mock_calls],
                         ["reuse_service", "provision", "start", "ensure_ready"])
        self.assertTrue((self.state_dir() / "operation.lock").exists())
        self.notify.assert_called_once_with("Worker left running; compute stops after 300 idle seconds. "
                                            "The next book can reuse the loaded service.")

    def test_ensure_service_activates_a_maintenance_service_under_the_lock(self):
        self.write_state()
        fake = self.fake_deployment(reuse=True)
        sentinel = {"remote_chapters": {}}
        patches = self.ensure_service_patches(fake, Mock(return_value=sentinel))
        with self.serve({"v1/capabilities": {"maintenance": True, "ephemeral_runs": 1, "warm_activation": 1}}), \
                patches["load_settings"], patches["deployment"], patches["build"], \
                patches["getsignal"], patches["signal"], redirect_stdout(StringIO()):
            self.assertIs(ensure_service(self.settings, notify=self.notify), sentinel)
        # The lock-free probe saw maintenance and never activated; the locked
        # reuse path performed the activation behind reuse_service.
        self.assertEqual(self.requests, [("GET", "v1/capabilities")])
        fake.provision.assert_not_called()
        fake.start.assert_not_called()
        fake.ensure_ready.assert_not_called()

    def test_ensure_service_stops_a_worker_that_never_became_ready(self):
        fake = self.fake_deployment()
        fake.ensure_ready.side_effect = RuntimeError("fixture setup failure")
        patches = self.ensure_service_patches(fake, Mock(return_value={"remote_chapters": {}}))
        with patches["load_settings"], patches["deployment"], patches["build"], \
                patches["getsignal"], patches["signal"], redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "fixture setup failure"):
                ensure_service(self.settings, notify=self.notify)
        fake.stop.assert_called_once_with()

    def test_ensure_service_reports_billing_when_even_the_cleanup_stop_fails(self):
        fake = self.fake_deployment()
        fake.ensure_ready.side_effect = RuntimeError("fixture setup failure")
        fake.stop.side_effect = RuntimeError("stop failed fixture-worker-secret")
        patches = self.ensure_service_patches(fake, Mock(return_value={"remote_chapters": {}}))
        with patches["load_settings"], patches["deployment"], patches["build"], \
                patches["getsignal"], patches["signal"], redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "billing may continue") as caught:
                ensure_service(self.settings, notify=self.notify)
        message = str(caught.exception)
        self.assertIn("fixture setup failure", message)
        self.assertNotIn("fixture-worker-secret", message)

    def test_ensure_service_returns_a_validated_client_config_for_a_running_service(self):
        self.write_state()
        with self.serve({"v1/capabilities": {"maintenance": False, "ephemeral_runs": 1, "warm_activation": 1}}), \
                redirect_stdout(StringIO()):
            result = ensure_service(self.settings, notify=self.notify)
        self.assertEqual(result["remote_chapters"]["url"], "https://pod_fixture-8082.proxy.runpod.net")
        self.assertEqual(result["remote_chapters"]["api_key"], "fixture-worker-secret")


if __name__ == "__main__":
    unittest.main()
