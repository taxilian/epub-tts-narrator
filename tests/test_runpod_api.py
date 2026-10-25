"""Runpod management safety without credentials or paid API calls."""

from io import BytesIO
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from epub_tts_narrator.renderer.runpod_api import RunpodAPI, RunpodAPIError, RunpodAmbiguousOutcome


def response(value=None, *, status=200, raw=None):
    result = Mock(status=status)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.read.return_value = json.dumps(value).encode() if raw is None else raw
    return result


class RunpodAPITests(unittest.TestCase):
    def setUp(self):
        self.opener = Mock()
        self.build = patch("epub_tts_narrator.renderer.runpod_api.build_opener", return_value=self.opener).start()
        self.sleep = patch("epub_tts_narrator.renderer.runpod_api.time.sleep").start()
        self.addCleanup(patch.stopall)
        self.api = RunpodAPI("fixture-explicit-key", timeout=17)

    def test_lists_preserve_fields_and_explicit_auth_ignores_environment(self):
        payload = [{"id": "pod_1", "publicIp": "203.0.113.7", "portMappings": {"22": 10341},
                    "futureProviderField": {"a": "b"}}]
        self.opener.open.return_value = response(payload)
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "wrong-ambient-key", "HTTPS_PROXY": "http://wrong.invalid"}):
            self.assertEqual(payload, self.api.list_pods(name="our-worker", includeMachine=True))
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://rest.runpod.io/v1/pods?name=our-worker&includeMachine=true")
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.headers["Authorization"], "Bearer fixture-explicit-key")
        self.assertEqual(request.headers["User-agent"], "epub-tts-narrator/1")
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 17})
        self.assertEqual(self.build.call_args.args[0].proxies, {})

    def test_get_pod_keeps_runtime_mapping(self):
        payload = {"id": "pod_1", "publicIp": "203.0.113.7", "portMappings": {"22": 10341}}
        self.opener.open.return_value = response(payload)
        self.assertEqual(self.api.get_pod("pod_1"), payload)
        self.assertEqual(self.opener.open.call_args.args[0].full_url, "https://rest.runpod.io/v1/pods/pod_1")

    def test_create_pod_uses_v2_and_preserves_create_price(self):
        spec = {"name": "fixture", "image": "fixture/image@sha256:" + "a" * 64,
                "gpu": {"id": "fixture-gpu", "count": 1, "minCudaVersion": "12.9"},
                "dataCenterIds": ["US-CO-1"],
                "mounts": {"network": [{"volumeId": "vol_1", "path": "/workspace"}]},
                "ports": ["22/tcp", "8081/http"], "cmd": ["/start.sh"],
                "env": {"PUBLIC_KEY": "fixture-public-key"}}
        payload = {"id": "new_pod", "cost": 2.49, "status": "PROVISIONING", "dataCenterId": "US-CO-1"}
        self.opener.open.return_value = response(payload, status=201)
        self.assertEqual(self.api.create_pod(spec), payload)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(json.loads(request.data), spec)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_detailed_pod_read_uses_v2_and_confirms_identity(self):
        payload = {"id": "pod_1", "gpu": {"id": "fixture-gpu", "count": 1},
                   "mounts": {"network": [{"volumeId": "vol_1", "path": "/workspace"}]}}
        self.opener.open.return_value = response(payload)
        self.assertEqual(self.api.get_pod_details("pod_1"), payload)
        self.assertEqual(self.opener.open.call_args.args[0].full_url, "https://api.runpod.io/v2/pods/pod_1")
        for wrong in ({}, {"id": "another_pod"}):
            self.opener.open.return_value = response(wrong)
            with self.assertRaises(RunpodAPIError) as caught:
                self.api.get_pod_details("pod_1")
            self.assertFalse(caught.exception.ambiguous)

    def test_start_stop_use_v2_actions_and_return_updated_pod(self):
        for action, status in (("start", "STARTING"), ("stop", "EXITED")):
            with self.subTest(action=action):
                self.opener.open.reset_mock()
                payload = {"id": "pod_1", "status": status, "futureField": "preserved"}
                self.opener.open.return_value = response(payload)
                self.assertEqual(getattr(self.api, action + "_pod")("pod_1"), payload)
                request = self.opener.open.call_args.args[0]
                self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods/pod_1/action")
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.headers["Content-type"], "application/json")
                self.assertEqual(json.loads(request.data), {"action": action})
                self.assertEqual(self.opener.open.call_count, 1)

    def test_actions_require_confirmation_for_requested_pod_without_retry(self):
        for action in (self.api.start_pod, self.api.stop_pod):
            for raw in (b"", b"{}", b"[]", b'{"id":"another_pod"}', b'{"id":"../../escape"}'):
                with self.subTest(action=action.__name__, raw=raw):
                    self.opener.open.reset_mock()
                    self.opener.open.return_value = response(raw=raw)
                    with self.assertRaises(RunpodAmbiguousOutcome):
                        action("pod_1")
                    self.assertEqual(self.opener.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_action_server_error_does_not_retry_or_fall_back_to_v1(self):
        for action in (self.api.start_pod, self.api.stop_pod):
            with self.subTest(action=action.__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = HTTPError(
                    "https://private.invalid", 500, "secret-detail", {}, BytesIO())
                with self.assertRaises(RunpodAmbiguousOutcome):
                    action("pod_1")
                self.assertEqual(self.opener.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_update_pod_only_sends_requested_fields_without_starting(self):
        spec = {"ports": ["22/tcp", "8082/http"]}
        payload = {"id": "pod_1", "status": "EXITED", "ports": spec["ports"],
                   "mounts": {"network": [{"volumeId": "vol_1", "path": "/workspace"}]}}
        self.opener.open.return_value = response(payload)
        self.assertEqual(self.api.update_pod("pod_1", spec), payload)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods/pod_1")
        self.assertEqual(request.method, "PATCH")
        self.assertEqual(json.loads(request.data), spec)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_update_pod_uncertain_result_never_retries(self):
        for result in (TimeoutError("secret-value"), response(raw=b"{}")):
            with self.subTest(result=type(result).__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = [result]
                with self.assertRaises(RunpodAmbiguousOutcome):
                    self.api.update_pod("pod_1", {"ports": ["22/tcp", "8082/http"]})
                self.assertEqual(self.opener.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_volume_operations_use_v1(self):
        self.opener.open.return_value = response([])
        self.assertEqual(self.api.list_volumes(), [])
        self.assertEqual(self.opener.open.call_args.args[0].full_url, "https://rest.runpod.io/v1/networkvolumes")
        spec = {"name": "models", "size": 150, "dataCenterId": "US-KS-2"}
        self.opener.open.return_value = response(dict(spec, id="vol_1"), status=201)
        self.assertEqual(self.api.create_volume(spec)["id"], "vol_1")
        self.assertEqual(json.loads(self.opener.open.call_args.args[0].data), spec)
        self.opener.open.return_value = response(dict(spec, id="vol_1"))
        self.assertEqual(self.api.get_volume("vol_1")["size"], 150)
        self.assertTrue(self.opener.open.call_args.args[0].full_url.endswith("/networkvolumes/vol_1"))

    def test_catalog_uses_v2_and_comma_separated_arrays(self):
        self.opener.open.return_value = response({"gpus": []})
        self.assertEqual(self.api.catalog_gpus(include=["AVAILABILITY"], product=["POD"],
                                               countryCodes=["US", "CA"], count=1), {"gpus": []})
        self.assertEqual(self.opener.open.call_args.args[0].full_url,
                         "https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&countryCodes=US%2CCA&count=1")
        self.opener.open.return_value = response({"dataCenters": []})
        self.assertEqual(self.api.datacenters(include="GPU_AVAILABILITY"), {"dataCenters": []})
        self.assertEqual(self.opener.open.call_args.args[0].full_url,
                         "https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY")

    def test_transient_get_retries_are_bounded(self):
        self.opener.open.side_effect = [URLError("credential-url-must-not-escape"),
                                        HTTPError("https://private.invalid", 503, "private-error", {}, BytesIO()),
                                        response([])]
        self.assertEqual(self.api.list_pods(), [])
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [0.5, 1.0])

    def test_get_failure_is_sanitized_after_retries(self):
        self.opener.open.side_effect = URLError("fixture-explicit-key private-server-output")
        with self.assertRaises(RunpodAPIError) as caught:
            self.api.list_pods()
        self.assertNotIn("fixture-explicit-key", str(caught.exception))
        self.assertNotIn("private-server-output", str(caught.exception))
        self.assertFalse(caught.exception.ambiguous)
        self.assertTrue(caught.exception.retryable)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(self.opener.open.call_count, 3)

    def test_mutation_transport_failure_never_retries(self):
        for method, argument in ((self.api.create_pod, {}), (self.api.create_volume, {}),
                                 (self.api.start_pod, "pod_1"), (self.api.stop_pod, "pod_1")):
            with self.subTest(method=method.__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = TimeoutError("secret-value")
                with self.assertRaises(RunpodAmbiguousOutcome) as caught:
                    method(argument)
                self.assertTrue(caught.exception.ambiguous)
                self.assertFalse(caught.exception.retryable)
                self.assertIn("rerun this command to reconcile", str(caught.exception))
                self.assertNotIn("secret-value", str(caught.exception))
                self.assertEqual(self.opener.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_server_error_on_create_is_ambiguous_and_not_retried(self):
        self.opener.open.side_effect = HTTPError("https://private.invalid", 502, "fixture-secret", {}, BytesIO(b"secret-body"))
        with self.assertRaises(RunpodAmbiguousOutcome) as caught:
            self.api.create_volume({"name": "fixture"})
        self.assertEqual(caught.exception.status, 502)
        self.assertIn("POST /networkvolumes", str(caught.exception))
        self.assertIn("rerun this command to reconcile", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(self.opener.open.call_count, 1)

    def test_authorization_or_validation_rejection_is_definite(self):
        for status in (400, 401, 403, 404, 409, 422, 429):
            with self.subTest(status=status):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = HTTPError("https://private.invalid", status, "secret-detail", {}, BytesIO())
                with self.assertRaises(RunpodAPIError) as caught:
                    self.api.create_pod({})
                self.assertEqual(caught.exception.status, status)
                self.assertFalse(caught.exception.ambiguous)
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(self.opener.open.call_count, 1)

    def test_get_authorization_rejection_is_not_retried(self):
        self.opener.open.side_effect = HTTPError("https://private.invalid", 401, "secret-detail", {}, BytesIO())
        with self.assertRaises(RunpodAPIError):
            self.api.list_pods()
        self.assertEqual(self.opener.open.call_count, 1)

    def test_known_host_capacity_error_is_sanitized_and_not_retried(self):
        body = json.dumps({
            "detail": "There are not enough free GPUs on the host machine to start this pod.",
            "extra": "fixture-explicit-key private-provider-text",
        }).encode()
        self.opener.open.side_effect = HTTPError(
            "https://private.invalid", 400, "private-provider-text", {}, BytesIO(body))
        with self.assertRaisesRegex(RunpodAPIError, "assigned host has no free GPU capacity") as caught:
            self.api.start_pod("pod_1")
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.reason, "host_capacity")
        self.assertFalse(caught.exception.ambiguous)
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("fixture-explicit-key", str(caught.exception))
        self.assertNotIn("private-provider-text", str(caught.exception))
        self.assertEqual(self.opener.open.call_count, 1)
        self.sleep.assert_not_called()

    def test_unknown_or_malformed_error_body_keeps_generic_rejection(self):
        for body in (b"not JSON fixture-explicit-key", b"[]", b"null", b'{}', b'\xff', b'[' * 2000,
                     b'{"detail":"fixture-explicit-key"}',
                     b'{"detail":"There are not enough free GPUs on the host machine to start this pod. secret"}'):
            with self.subTest(body=body):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = HTTPError(
                    "https://private.invalid", 400, "secret-detail", {}, BytesIO(body))
                with self.assertRaises(RunpodAPIError) as caught:
                    self.api.start_pod("pod_1")
                self.assertEqual(str(caught.exception), "Runpod rejected the request settings (HTTP 400)")
                self.assertIsNone(caught.exception.reason)
                self.assertFalse(caught.exception.ambiguous)
                self.assertEqual(self.opener.open.call_count, 1)

    def test_error_body_read_is_bounded_and_read_failures_preserve_status(self):
        for body in (b"x" * (16 * 1024 + 1), OSError("fixture-explicit-key")):
            with self.subTest(body=type(body).__name__):
                error = HTTPError("https://private.invalid", 400, "secret-detail", {}, BytesIO())
                error.read = Mock(side_effect=body) if isinstance(body, Exception) else Mock(return_value=body)
                self.opener.open.side_effect = error
                with self.assertRaises(RunpodAPIError) as caught:
                    self.api.start_pod("pod_1")
                error.read.assert_called_once_with(16 * 1024 + 1)
                self.assertTrue(error.closed)
                self.assertEqual(str(caught.exception), "Runpod rejected the request settings (HTTP 400)")
                self.assertFalse(caught.exception.ambiguous)

    def test_capacity_detail_does_not_override_ambiguous_server_failure(self):
        body = json.dumps({
            "detail": "There are not enough free GPUs on the host machine to start this pod."}).encode()
        self.opener.open.side_effect = HTTPError(
            "https://private.invalid", 500, "secret-detail", {}, BytesIO(body))
        with self.assertRaises(RunpodAmbiguousOutcome) as caught:
            self.api.start_pod("pod_1")
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_invalid_or_incomplete_create_success_is_ambiguous(self):
        for raw in (b"", b"html fixture-secret", b"[]", b"null", b'{}', b'{"id":"../../escape"}'):
            with self.subTest(raw=raw):
                self.opener.open.reset_mock()
                self.opener.open.return_value = response(status=201, raw=raw)
                with self.assertRaises(RunpodAmbiguousOutcome):
                    self.api.create_pod({})
                self.assertEqual(self.opener.open.call_count, 1)

    def test_oversized_response_is_rejected_with_bounded_read(self):
        self.opener.open.return_value = response(raw=b"x" * (8 * 1024 * 1024 + 1))
        with self.assertRaises(RunpodAPIError):
            self.api.list_pods()
        self.opener.open.return_value.read.assert_called_once_with(8 * 1024 * 1024 + 1)

    def test_redirect_handler_does_not_forward_bearer(self):
        handler = self.build.call_args.args[1]
        with self.assertRaises(HTTPError) as caught:
            handler.redirect_request(Mock(full_url="https://rest.runpod.io/v1/pods"), BytesIO(),
                                     307, "moved", {}, "https://attacker.invalid/")
        self.assertEqual(caught.exception.code, 307)
        self.opener.open.assert_not_called()

    def test_ids_and_queries_cannot_escape_fixed_host(self):
        for value in ("../pods", "https://other.invalid", "pod?token=secret", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.api.get_pod(value)
        with self.assertRaises(ValueError):
            self.api.list_pods(**{"bad&key": "x"})
        with self.assertRaises(ValueError):
            self.api.list_pods(name={"not": "a string"})
        self.opener.open.assert_not_called()

    def test_bad_credentials_timeout_and_creation_settings_fail_locally(self):
        for key in (None, "", "key\nheader", "key space", "\u2603"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                RunpodAPI(key)
        for timeout in (None, True, 0, -1, 301, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                RunpodAPI("fixture", timeout=timeout)
        for spec in (None, [], {"not-json": object()}, {"bad": float("nan")}):
            with self.subTest(spec=type(spec)), self.assertRaises(ValueError):
                self.api.create_pod(spec)
        self.opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
