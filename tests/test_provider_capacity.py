"""Provider capacity signals stay typed, bounded, redacted, and nonretryable."""

import io
import json
from unittest import TestCase
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from epub_tts_narrator.adaptive_requests import capacity_error
from epub_tts_narrator.provider import OpenAICompatibleClient, ProviderError, ResponseCapacityError


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status


def sse(*events):
    return b"".join(
        b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n"
        for event in events
    )


class ProviderCapacityTests(TestCase):
    def setUp(self):
        self.opener = Mock()
        patcher = patch("epub_tts_narrator.provider.build_opener", return_value=self.opener)
        patcher.start()
        self.addCleanup(patcher.stop)

    def http_error(self, status, details):
        body = FakeResponse(json.dumps({"error": details}).encode())
        return HTTPError("https://example.invalid/private", status, "private reason", {}, body)

    def test_http_400_and_413_expose_only_the_safe_context_capacity_code(self):
        for status in (400, 413):
            with self.subTest(status=status):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = self.http_error(status, {
                    "code": "context_length_exceeded", "message": "private book and key", "retryable": True,
                })
                with patch("epub_tts_narrator.provider.time.sleep") as sleep, self.assertRaises(ProviderError) as caught:
                    OpenAICompatibleClient("model", max_retries=5).complete("private system", {})
                self.assertEqual(caught.exception.code, "context_length_exceeded")
                self.assertEqual(caught.exception.status_code, status)
                self.assertFalse(caught.exception.retryable)
                self.assertTrue(capacity_error(caught.exception))
                self.assertNotIn("private", str(caught.exception))
                self.opener.open.assert_called_once()
                sleep.assert_not_called()

    def test_generic_400_and_413_errors_remain_redacted_and_nonretryable(self):
        for status, code in ((400, "server_error"), (413, "rate_limit_exceeded"), (400, "private-code")):
            with self.subTest(status=status, code=code):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = self.http_error(status, {
                    "code": code, "message": "private book and key", "retryable": True,
                })
                with patch("epub_tts_narrator.provider.time.sleep") as sleep, self.assertRaises(ProviderError) as caught:
                    OpenAICompatibleClient("model", max_retries=5).complete("private system", {})
                self.assertIsNone(caught.exception.code)
                self.assertFalse(caught.exception.retryable)
                self.assertFalse(capacity_error(caught.exception))
                self.assertNotIn("private", str(caught.exception))
                self.opener.open.assert_called_once()
                sleep.assert_not_called()

    def test_nonstream_response_size_limit_raises_typed_capacity_without_retry(self):
        self.opener.open.return_value = FakeResponse(b"x" * 100)
        with patch("epub_tts_narrator.provider.MAX_RESPONSE_BYTES", 32), \
                patch("epub_tts_narrator.provider.time.sleep") as sleep, \
                self.assertRaises(ResponseCapacityError) as caught:
            OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
        self.assertTrue(capacity_error(caught.exception))
        self.assertFalse(caught.exception.retryable)
        self.opener.open.assert_called_once()
        sleep.assert_not_called()

    def test_stream_response_size_limit_raises_typed_capacity_without_retry(self):
        body = sse({"choices": [{"delta": {"content": "x" * 100}, "finish_reason": None}]}, "[DONE]")
        self.opener.open.return_value = FakeResponse(body)
        with patch("epub_tts_narrator.provider.MAX_RESPONSE_BYTES", 32), \
                patch("epub_tts_narrator.provider.time.sleep") as sleep, \
                self.assertRaises(ResponseCapacityError) as caught:
            OpenAICompatibleClient("model", max_retries=5).complete("Analyze", {})
        self.assertTrue(capacity_error(caught.exception))
        self.assertFalse(caught.exception.retryable)
        self.opener.open.assert_called_once()
        sleep.assert_not_called()
