"""Provider tests use a mocked transport and never open a network socket."""

import io
import errno
import json
import socket
import ssl
from email.message import Message
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from threading import get_ident
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from epub_tts_narrator.provider import (
    MAX_RESPONSE_BYTES,
    MAX_ERROR_BYTES,
    OpenAICompatibleClient,
    OutputTruncatedError,
    ProviderError,
    _NoRedirect,
)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.read_limits = []

    def read(self, size=-1):
        self.read_limits.append(size)
        return super().read(size)


def completion(content='{"characters": []}', finish_reason="stop", **message_fields):
    message = {"role": "assistant", "content": content, **message_fields}
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.opener = Mock()
        self.opener_patch = patch("epub_tts_narrator.provider.build_opener", return_value=self.opener)
        self.build_opener = self.opener_patch.start()
        self.addCleanup(self.opener_patch.stop)

    def respond(self, value, status=200):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
        response = FakeResponse(raw, status=status)
        self.opener.open.return_value = response
        return response

    def test_serializes_messages_and_configurable_model(self):
        self.respond(completion())
        client = OpenAICompatibleClient(
            model="user-selected-model", api_key="private-test-key", max_output_tokens=8000
        )
        self.assertEqual(client.complete("Analyze the book.", {"text": "Café"}), {"characters": []})
        args, kwargs = self.opener.open.call_args
        request = args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
        self.assertEqual(kwargs, {"timeout": 120.0})
        body = json.loads(request.data)
        self.assertEqual(body["model"], "user-selected-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_completion_tokens"], 8000)
        self.assertNotIn("temperature", body)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("JSON object", body["messages"][0]["content"])
        self.assertEqual(json.loads(body["messages"][1]["content"]), {"text": "Café"})

    def test_local_provider_without_key_or_json_mode(self):
        self.respond(completion())
        client = OpenAICompatibleClient(
            "local-model", "http://localhost:8080/v1/", json_mode=False,
            max_output_tokens=1024, token_limit_field="max_tokens", timeout=12,
        )
        client.complete("Analyze", {})
        request = self.opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.full_url, "http://localhost:8080/v1/chat/completions")
        self.assertNotIn("response_format", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertIn("JSON object", body["messages"][0]["content"])

    def test_unset_token_limit_is_omitted(self):
        self.respond(completion())
        OpenAICompatibleClient("model").complete("Analyze", {})
        body = json.loads(self.opener.open.call_args.args[0].data)
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("reasoning_effort", body)

    def test_optional_reasoning_effort_is_sent_and_affects_cache_identity(self):
        self.respond(completion())
        client = OpenAICompatibleClient("model", reasoning_effort="high")
        client.complete("Analyze", {})
        body = json.loads(self.opener.open.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(client.identity["reasoning_effort"], "high")
        for invalid in ("bad", [], False, 1):
            with self.subTest(invalid=invalid), self.assertRaises(ProviderError):
                OpenAICompatibleClient("model", reasoning_effort=invalid)

    def test_identity_contains_configuration_without_credentials(self):
        client = OpenAICompatibleClient("model", api_key="secret-key")
        expected = {
            "model": "model", "base_url": "https://api.openai.com/v1",
            "json_mode": True, "max_output_tokens": None,
            "token_limit_field": "max_completion_tokens",
        }
        self.assertEqual(client.identity, expected)
        client.identity["model"] = "changed"
        self.assertEqual(client.identity, expected)
        self.assertNotIn("secret-key", repr(client.identity))

    def test_accepts_https_and_literal_loopback_urls(self):
        for url in (
            "https://example.com/v1", "http://localhost:8000/v1",
            "http://127.0.0.1:1234/v1", "http://127.12.34.56/v1", "http://[::1]:8000/v1",
        ):
            with self.subTest(url=url):
                self.assertEqual(OpenAICompatibleClient("model", url).base_url, url)

    def test_rejects_invalid_or_insecure_urls_without_echoing_them(self):
        for url in (
            "http://example.com/v1", "http://192.168.1.3/v1", "http://localhost.example/v1",
            "file:///tmp/api", "example.com/v1", "https:///v1", "https://example.com:99999",
            "https://[invalid", "https://user:secret@example.com/v1",
            "https://example.com/v1?key=secret", "https://example.com/v1#secret",
            "https://example.com/v1?", "https://example.com/v1#",
            "\nhttps://example.com/v1", "https://example.com/has space", "https://example.com\\@evil.test",
            "http://2130706433/v1", "http://%31%32%37.0.0.1/v1", "https://example.com:0/v1", "",
        ):
            with self.subTest(url=url), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model", url)
            self.assertNotIn("secret", str(caught.exception))
        self.opener.open.assert_not_called()

    def test_rejects_invalid_options_before_transport(self):
        for options in (
            {"model": ""}, {"model": None}, {"timeout": 0}, {"timeout": -1},
            {"timeout": float("nan")}, {"timeout": float("inf")}, {"timeout": True}, {"timeout": 10 ** 1000},
            {"max_output_tokens": 0}, {"max_output_tokens": True}, {"max_output_tokens": 1.5},
            {"json_mode": "yes"}, {"token_limit_field": "temperature"}, {"token_limit_field": []},
            {"api_key": "secret\nInjected: header"}, {"api_key": 10},
            {"max_retries": -1}, {"max_retries": 11}, {"max_retries": True}, {"max_retries": 1.5},
            {"retry_initial_delay": 0}, {"retry_max_delay": 61}, {"retry_initial_delay": True},
            {"retry_initial_delay": float("nan")}, {"retry_max_delay": float("inf")},
            {"retry_initial_delay": 30, "retry_max_delay": 5},
        ):
            with self.subTest(options=options), self.assertRaises(ProviderError):
                OpenAICompatibleClient(**{"model": "model", **options})
        self.opener.open.assert_not_called()

    def test_redirect_handler_is_installed(self):
        OpenAICompatibleClient("model")
        self.assertIsInstance(self.build_opener.call_args.args[0], _NoRedirect)

    def test_redirect_handler_refuses_forwarding_authorization(self):
        request = Request("https://api.example/v1", headers={"Authorization": "Bearer secret"})
        response = FakeResponse(b"private server body")
        with self.assertRaisesRegex(ProviderError, "redirect refused"):
            _NoRedirect().redirect_request(request, response, 302, "secret", {}, "https://evil.test")
        self.assertTrue(response.closed)
        self.assertEqual(response.read_limits, [])
        self.opener.open.assert_not_called()

    def test_http_errors_are_redacted_and_not_retried(self):
        for status in (301, 302, 307, 308, 400, 401, 403, 429, 500):
            body = Mock()
            self.opener.open.reset_mock()
            self.opener.open.side_effect = HTTPError(
                "https://example.com/secret", status, "server-secret", {}, body,
            )
            with self.subTest(status=status), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model", api_key="api-secret").complete("system-secret", {})
            self.assertNotIn("secret", str(caught.exception))
            self.opener.open.assert_called_once()
            if status in {429, 500}:
                body.read.assert_called_once_with(MAX_ERROR_BYTES + 1)
            else:
                body.read.assert_not_called()
            body.close.assert_called_once()

    def test_unexpected_non_success_status_is_rejected(self):
        for status in (302, 401, 500):
            response = self.respond(completion(), status)
            with self.subTest(status=status), self.assertRaises(ProviderError):
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertEqual(response.read_limits, [MAX_ERROR_BYTES + 1] if status == 500 else [])

    def http_error(self, status=429, details=None, retry_after=None):
        headers = Message()
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        body = FakeResponse(json.dumps({"error": details or {"code": "concurrent_budget_exceeded", "retryable": True}}).encode())
        return HTTPError("https://example.com/private-key", status, "private-secret", headers, body)

    def test_transient_retries_preserve_requests_cache_identity_and_concurrency(self):
        failures = [self.http_error(), self.http_error(details={"code": "tpm_uncached_exceeded"})]
        self.opener.open.side_effect = failures + [FakeResponse(json.dumps(completion()).encode())]
        client = OpenAICompatibleClient("model", api_key="private-key", max_retries=2)
        self.assertEqual(client.identity, OpenAICompatibleClient("model").identity)
        with patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), self.assertLogs("epub_tts_narrator.provider") as logs:
            self.assertEqual(client.complete("private-source", {"text": "private-book"}), {"characters": []})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0])
        requests = [call.args[0] for call in self.opener.open.call_args_list]
        self.assertEqual(len(requests), 3)
        self.assertEqual(len({request.data for request in requests}), 1)
        self.assertTrue(all(error.fp.closed for error in failures))
        self.assertIn("concurrent_budget_exceeded", " ".join(logs.output))
        self.assertNotIn("private", " ".join(logs.output))

    def test_retry_exhaustion_is_bounded_and_exposes_only_safe_metadata(self):
        self.opener.open.side_effect = [self.http_error(details={"code": "tpm_uncached_exceeded", "message": "private-key private-book"}) for _ in range(4)]
        with patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), patch("epub_tts_narrator.provider._LOGGER"), self.assertRaises(ProviderError) as caught:
            OpenAICompatibleClient("model", max_retries=3, retry_initial_delay=30, retry_max_delay=40).complete("Analyze", {})
        self.assertEqual(self.opener.open.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 40.0, 40.0])
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.code, "tpm_uncached_exceeded")
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private", str(caught.exception))

    def test_quota_and_explicit_nonretryable_failures_are_not_retried(self):
        for status, details in (
            (429, {"code": "insufficient_quota", "retryable": True}),
            (429, {"code": "private-secret", "message": "Account credits exhausted"}),
            (429, {"code": "tpm_uncached_exceeded", "retryable": False}),
            (503, {"code": "context_length_exceeded"}),
            (401, {"retryable": True}), (403, {"retryable": True}), (400, {"retryable": True}),
        ):
            self.opener.open.reset_mock()
            self.opener.open.side_effect = self.http_error(status, details)
            with self.subTest(status=status, details=details), patch("epub_tts_narrator.provider.time.sleep") as sleep, self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
            self.opener.open.assert_called_once()
            sleep.assert_not_called()
            self.assertFalse(caught.exception.retryable)
            self.assertNotIn("private", str(caught.exception))

    def test_retry_after_numeric_and_http_date_are_respected(self):
        for header in ("25", "Thu, 01 Jan 1970 00:17:05 GMT"):
            self.opener.open.reset_mock()
            self.opener.open.side_effect = [self.http_error(retry_after=header), FakeResponse(json.dumps(completion()).encode())]
            with self.subTest(header=header), patch("epub_tts_narrator.provider.time.time", return_value=1000), patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), patch("epub_tts_narrator.provider._LOGGER"):
                OpenAICompatibleClient("model", max_retries=1).complete("Analyze", {})
            sleep.assert_called_once_with(25.0)

    def test_retry_after_beyond_bound_stops_without_retrying_too_early(self):
        self.opener.open.side_effect = self.http_error(retry_after="120")
        with patch("epub_tts_narrator.provider.time.sleep") as sleep, self.assertRaises(ProviderError) as caught:
            OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
        self.opener.open.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(caught.exception.retry_after_seconds, 120.0)

    def test_invalid_retry_after_and_unknown_error_codes_do_not_leak(self):
        for header in ("private-secret", "-1", "nan", "inf"):
            self.opener.open.reset_mock()
            self.opener.open.side_effect = [self.http_error(details={"code": "private-secret"}, retry_after=header), self.http_error(details={"code": "private-secret"})]
            with self.subTest(header=header), patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), self.assertLogs("epub_tts_narrator.provider") as logs, self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model", max_retries=1).complete("Analyze", {})
            sleep.assert_called_once_with(5.0)
            self.assertIsNone(caught.exception.code)
            self.assertNotIn("private", str(caught.exception) + " ".join(logs.output))

    def test_enabling_retries_does_not_repeat_unknown_transport_or_model_failures(self):
        self.opener.open.side_effect = URLError("private-secret")
        with patch("epub_tts_narrator.provider.time.sleep") as sleep, self.assertRaises(ProviderError):
            OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
        self.opener.open.assert_called_once()
        sleep.assert_not_called()
        self.opener.open.reset_mock(side_effect=True)
        self.respond(completion("private-invalid-json"))
        with self.assertRaises(ProviderError):
            OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
        self.opener.open.assert_called_once()

    def test_typed_transient_transport_failures_retry_same_request_in_same_worker(self):
        failures = [TimeoutError("private-secret"), RemoteDisconnected("private-secret"),
                    IncompleteRead(b"private-book", 100), ConnectionResetError("private-secret"),
                    ConnectionRefusedError("private-secret"), ConnectionAbortedError("private-secret"),
                    BrokenPipeError("private-secret"), socket.gaierror(socket.EAI_AGAIN, "private-host"),
                    OSError(errno.ETIMEDOUT, "private-secret"),
                    URLError(URLError(OSError(errno.ECONNRESET, "private-url")))]
        for failure in failures:
            for wrapped in (False, True):
                with self.subTest(failure=type(failure).__name__, wrapped=wrapped):
                    self.opener.open.reset_mock()
                    attempts, threads = [], []
                    def open_request(request, **kwargs):
                        attempts.append(request)
                        threads.append(get_ident())
                        if len(attempts) == 1:
                            raise URLError(failure) if wrapped else failure
                        return FakeResponse(json.dumps(completion()).encode())
                    self.opener.open.side_effect = open_request
                    client = OpenAICompatibleClient("model", api_key="private-key", max_retries=1)
                    self.assertEqual(client.identity, OpenAICompatibleClient("model").identity)
                    with patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), self.assertLogs("epub_tts_narrator.provider") as logs:
                        self.assertEqual(client.complete("private-source", {"text": "private-book"}), {"characters": []})
                    self.assertEqual(len(attempts), 2)
                    self.assertEqual(attempts[0].data, attempts[1].data)
                    self.assertEqual(attempts[0].headers, attempts[1].headers)
                    self.assertEqual(threads, [get_ident(), get_ident()])
                    sleep.assert_called_once_with(5.0)
                    self.assertIn("transient transport failure", " ".join(logs.output))
                    self.assertNotIn("private", " ".join(logs.output))
                    self.assertNotIn("HTTP", " ".join(logs.output))

    def test_transport_read_failure_closes_partial_response_before_retry(self):
        class PartialResponse(FakeResponse):
            def read(self, size=-1):
                raise IncompleteRead(b"private-source", 100)
        partial = PartialResponse(b"")
        self.opener.open.side_effect = [partial, FakeResponse(json.dumps(completion()).encode())]
        with patch("epub_tts_narrator.provider.time.sleep"), patch("epub_tts_narrator.provider._LOGGER"):
            self.assertEqual(OpenAICompatibleClient("model", max_retries=1).complete("Analyze", {}), {"characters": []})
        self.assertTrue(partial.closed)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_transport_and_http_failures_share_one_bounded_retry_budget(self):
        self.opener.open.side_effect = [TimeoutError("private-key"), self.http_error(status=503),
                                      URLError(ConnectionResetError("private-url"))]
        with patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider.random.uniform", return_value=0), self.assertLogs("epub_tts_narrator.provider") as logs, self.assertRaises(ProviderError) as caught:
            OpenAICompatibleClient("model", max_retries=2, retry_initial_delay=30, retry_max_delay=40).complete("private-book", {})
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [30.0, 40.0])
        self.assertTrue(caught.exception.retryable)
        self.assertIsNone(caught.exception.status_code)
        self.assertIsNone(caught.exception.__cause__)
        output = " ".join(logs.output)
        self.assertIn("transient transport failure", output)
        self.assertIn("HTTP 503", output)
        self.assertNotIn("private", output + str(caught.exception))

    def test_certificate_invalid_url_and_permanent_transport_errors_never_retry(self):
        cyclic = URLError("private-url")
        cyclic.reason = cyclic
        for error in (ssl.SSLCertVerificationError(1, "private-certificate"),
                      URLError(ssl.SSLCertVerificationError(1, "private-certificate")),
                      ssl.SSLError(errno.ECONNRESET, "private-certificate"),
                      ValueError("private-invalid-url"), URLError(ValueError("private-url")),
                      OSError(errno.EACCES, "private-permission"),
                      socket.gaierror(socket.EAI_NONAME, "private-host"),
                      HTTPException("private-invalid-response"),
                      URLError("timed out: private-url"), cyclic):
            with self.subTest(error=type(error).__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = error
                with patch("epub_tts_narrator.provider.time.sleep") as sleep, patch("epub_tts_narrator.provider._LOGGER") as logger, self.assertRaises(ProviderError) as caught:
                    OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
                self.opener.open.assert_called_once()
                sleep.assert_not_called()
                logger.warning.assert_not_called()
                self.assertFalse(caught.exception.retryable)
                self.assertNotIn("private", str(caught.exception))
        self.opener.open.reset_mock()
        with self.assertRaises(ProviderError):
            OpenAICompatibleClient("model", base_url="https://private-user:private-key@example.com", max_retries=5)
        self.opener.open.assert_not_called()

    def test_safe_numeric_rate_limit_headers_are_reported_in_errors_and_retry_logs(self):
        error = self.http_error(details={"code": "tpm_uncached_exceeded", "message": "private-secret"})
        for key, value in {
            "X-TPM-Limit-Tokens": "210000", "X-TPM-Limit-Max": "200000",
            "X-Concurrent-Limit-Active": "6", "X-Concurrent-Limit-Max": "5",
            "X-TPM-Limit-Dimension": "private-secret", "Authorization": "private-secret",
        }.items():
            error.headers[key] = value
        self.opener.open.side_effect = [error, self.http_error()]
        with patch("epub_tts_narrator.provider.time.sleep"), patch("epub_tts_narrator.provider.random.uniform", return_value=0), self.assertLogs("epub_tts_narrator.provider") as logs, self.assertRaises(ProviderError):
            OpenAICompatibleClient("model", max_retries=1).complete("Analyze", {})
        output = " ".join(logs.output)
        self.assertIn("tpm_tokens=210000, tpm_max=200000", output)
        self.assertIn("concurrent_active=6, concurrent_max=5", output)
        self.assertNotIn("private", output)
        self.opener.open.side_effect = self.http_error()
        self.opener.open.side_effect.headers["X-TPM-Limit-Max"] = "200000"
        with self.assertRaises(ProviderError) as caught:
            OpenAICompatibleClient("model").complete("Analyze", {})
        self.assertEqual(caught.exception.rate_limits, {"tpm_max": 200000})
        self.assertIn("tpm_max=200000", str(caught.exception))

    def test_invalid_rate_limit_header_values_are_omitted(self):
        for value in ("private-secret", "nan", "inf", "-1", "9" * 129):
            error = self.http_error()
            error.headers["X-TPM-Limit-Tokens"] = value
            self.opener.open.side_effect = error
            with self.subTest(value=value), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertEqual(caught.exception.rate_limits, {})
            self.assertNotIn("Rate limits:", str(caught.exception))
            self.assertNotIn("private", str(caught.exception))
    def test_transport_errors_are_redacted_and_not_retried_by_default(self):
        for error in (URLError("secret"), TimeoutError("secret"), OSError("secret")):
            self.opener.open.reset_mock()
            self.opener.open.side_effect = error
            with self.subTest(error=type(error)), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertNotIn("secret", str(caught.exception))
            self.opener.open.assert_called_once()

    def test_response_read_has_fixed_bound(self):
        response = self.respond(b"x" * (MAX_RESPONSE_BYTES + 2))
        with self.assertRaisesRegex(ProviderError, "16 MiB"):
            OpenAICompatibleClient("model").complete("Analyze", {})
        self.assertEqual(response.read_limits, [MAX_RESPONSE_BYTES + 1])
        self.assertTrue(response.closed)

    def test_rejects_bad_transport_json(self):
        for value in (b"\xff", b"private server text", b"[]", b"null", b'{"x": NaN}'):
            self.respond(value)
            with self.subTest(value=value), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertNotIn("private server text", str(caught.exception))

    def test_rejects_missing_or_malformed_choices(self):
        for value in ({}, {"choices": []}, {"choices": {}}, {"choices": [None]}, {"choices": ["secret"]}):
            self.respond(value)
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "choice"):
                OpenAICompatibleClient("model").complete("Analyze", {})

    def test_rejects_provider_error_envelope_without_echoing_it(self):
        self.respond({"error": {"message": "secret server message"}})
        with self.assertRaisesRegex(ProviderError, "error response") as caught:
            OpenAICompatibleClient("model").complete("Analyze", {})
        self.assertNotIn("secret", str(caught.exception))

    def test_requires_successful_finish_reason(self):
        for reason in (None, "length", "content_filter", "tool_calls", "function_call", "secret"):
            self.respond(completion(finish_reason=reason))
            with self.subTest(reason=reason), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertNotIn("secret", str(caught.exception))

    def test_length_reason_rejects_even_syntactically_complete_json(self):
        self.respond(completion('{"characters": []}', finish_reason="length"))
        with self.assertRaisesRegex(OutputTruncatedError, "truncated"):
            OpenAICompatibleClient("model").complete("Analyze", {})

    def test_refusal_is_not_accepted_even_with_valid_json(self):
        self.respond(completion(refusal="secret refusal detail"))
        with self.assertRaisesRegex(ProviderError, "refused") as caught:
            OpenAICompatibleClient("model").complete("Analyze", {})
        self.assertNotIn("secret", str(caught.exception))

    def test_rejects_malformed_messages(self):
        for message in (None, [], "secret", {"role": "user", "content": "{}"}):
            self.respond({"choices": [{"finish_reason": "stop", "message": message}]})
            with self.subTest(message=message), self.assertRaises(ProviderError):
                OpenAICompatibleClient("model").complete("Analyze", {})

    def test_rejects_missing_or_nontext_content(self):
        for content in (None, "", "   ", [], {}, 12):
            self.respond(completion(content=content))
            with self.subTest(content=content), self.assertRaises(ProviderError):
                OpenAICompatibleClient("model").complete("Analyze", {})

    def test_requires_model_output_to_be_one_json_object(self):
        for content in ("private response", "[]", "null", "```json\n{}\n```", '{} trailing', '{"x": Infinity}'):
            self.respond(completion(content=content))
            with self.subTest(content=content), self.assertRaises(ProviderError) as caught:
                OpenAICompatibleClient("model").complete("Analyze", {})
            self.assertNotIn("private response", str(caught.exception))

    def test_rejects_invalid_input_before_sending(self):
        for system, payload in ((None, {}), ("Analyze", []), ("Analyze", {"x": float("nan")}), ("Analyze", {"x": object()})):
            with self.subTest(system=system, payload=payload), self.assertRaises(ProviderError):
                OpenAICompatibleClient("model").complete(system, payload)
        self.opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
