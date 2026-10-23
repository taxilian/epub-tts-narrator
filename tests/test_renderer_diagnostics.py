"""Fatal CLI diagnostics keep type, safe code location, and provider codes without leaks."""

import unittest
from urllib.error import HTTPError, URLError

from epub_tts_narrator.renderer.diagnostics import exception_detail, format_fatal


class FormatFatalTests(unittest.TestCase):
    def failing_helper(self):
        raise ValueError("fixture failure mentioning secret-value")

    def test_includes_type_redacted_message_and_structural_frames_only(self):
        try:
            self.failing_helper()
        except ValueError as exc:
            report = format_fatal(exc, secrets={"secret-value"})
        self.assertIn("ValueError: fixture failure mentioning [REDACTED]", report)
        self.assertNotIn("secret-value", report)
        self.assertIn("failing_helper", report)
        self.assertRegex(report, r"test_renderer_diagnostics\.py:\d+")
        # Frames carry no source line text and no exception chain.
        self.assertNotIn("raise ValueError", report)
        self.assertNotIn("Traceback", report)

    def test_exception_without_traceback_still_identifies_type(self):
        self.assertEqual(format_fatal(RuntimeError("plain failure")), "RuntimeError: plain failure")

    def test_terminal_controls_are_removed_from_message(self):
        report = format_fatal(RuntimeError("\x1b[31mfailed\x1b[0m"))
        self.assertEqual(report, "RuntimeError: failed")


class ExceptionDetailTests(unittest.TestCase):
    def test_http_error_keeps_type_and_status_only(self):
        exc = HTTPError("https://example.invalid/secret-path", 503, "unavailable", {}, None)
        detail = exception_detail(exc)
        self.assertEqual(detail, "HTTPError, HTTP status 503")
        self.assertNotIn("secret-path", detail)

    def test_url_error_keeps_type_without_raw_reason(self):
        self.assertEqual(exception_detail(URLError("raw connection detail")), "URLError")

    def test_sdk_client_error_keeps_provider_code_and_http_status(self):
        class ClientError(Exception):
            def __init__(self):
                super().__init__("raw secret detail")
                self.response = {"Error": {"Code": "AccessDenied", "Message": "raw detail"},
                                 "ResponseMetadata": {"HTTPStatusCode": 403}}

        detail = exception_detail(ClientError())
        self.assertEqual(detail, "ClientError, provider code AccessDenied, HTTP status 403")
        self.assertNotIn("raw", detail)

    def test_unusual_provider_code_is_not_echoed(self):
        class ClientError(Exception):
            def __init__(self):
                super().__init__("raw")
                self.response = {"Error": {"Code": "injected text with spaces"}}

        self.assertEqual(exception_detail(ClientError()), "ClientError")

    def test_os_error_keeps_errno_code(self):
        self.assertEqual(exception_detail(OSError(2, "no such file")), "FileNotFoundError, errno 2")


if __name__ == "__main__":
    unittest.main()
