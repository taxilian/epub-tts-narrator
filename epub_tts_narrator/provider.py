"""Small client with opt-in bounded retries for transient request failures."""

from __future__ import annotations

import ipaddress
import errno
import json
import logging
import math
import random
import socket
import ssl
import time
from email.utils import parsedate_to_datetime
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
_TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_CONNECTION_ERRNOS = {
    errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
    errno.ECONNABORTED, errno.EPIPE,
}
_NONRETRYABLE_CODES = {
    "insufficient_quota", "quota_exceeded", "billing_hard_limit_reached",
    "credits_exhausted", "invalid_api_key", "authentication_error",
    "permission_denied", "model_not_found", "context_length_exceeded",
}
_SAFE_ERROR_CODES = _NONRETRYABLE_CODES | {
    "concurrent_budget_exceeded", "tpm_uncached_exceeded", "rate_limit_exceeded",
    "rate_limit_error", "concurrency_limit_exceeded", "tpm_limit_exceeded",
    "tokens_per_minute_exceeded", "uncached_tpm_limit_exceeded", "server_error",
    "internal_server_error", "service_unavailable", "overloaded",
}
_RATE_LIMIT_HEADERS = {
    "X-TPM-Limit-Tokens": "tpm_tokens",
    "X-TPM-Limit-Max": "tpm_max",
    "X-Concurrent-Limit-Active": "concurrent_active",
    "X-Concurrent-Limit-Max": "concurrent_max",
}
_LOGGER = logging.getLogger(__name__)
_JSON_INSTRUCTION = (
    "Return a single JSON object only, without Markdown code fences or "
    "surrounding commentary."
)


class ProviderError(Exception):
    """An invalid configuration, transport failure, or unusable model response."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 code: str | None = None, retryable: bool = False,
                 retry_after_seconds: float | None = None,
                 rate_limits: dict[str, int | float] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code if isinstance(code, str) and code in _SAFE_ERROR_CODES else None
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.rate_limits = dict(rate_limits or {})


class OutputTruncatedError(ProviderError):
    """The provider exhausted its output budget, not a transport outage."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credentials (or book text) to a redirect destination.
        if fp is not None:
            fp.close()
        raise ProviderError("API redirect refused. Configure the final API base URL.")


def _base_url(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderError("The API base URL must be a nonempty HTTP or HTTPS URL.")
    if any(ord(char) < 33 or ord(char) == 127 for char in value) or "\\" in value:
        raise ProviderError("The API base URL contains invalid characters.")
    if "?" in value or "#" in value:
        raise ProviderError("The API base URL must not contain a query or fragment.")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        port = parts.port  # Validate malformed and out-of-range ports, too.
    except ValueError:
        raise ProviderError("The API base URL is invalid.") from None
    if parts.scheme not in {"http", "https"} or not hostname:
        raise ProviderError("The API base URL must be an absolute HTTP or HTTPS URL.")
    if parts.username is not None or parts.password is not None:
        raise ProviderError("The API base URL must not contain embedded credentials.")
    if port == 0 or "%" in hostname:
        raise ProviderError("The API base URL has an invalid host or port.")
    if parts.scheme == "http":
        local = hostname.lower() == "localhost"
        if not local:
            try:
                local = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if not local:
            raise ProviderError("HTTP is allowed only for localhost or loopback addresses.")
    return value.rstrip("/")


def _reject_nonfinite(value: str):
    raise ValueError("Non-finite JSON number")


def _json_object(value: str, description: str) -> dict:
    try:
        parsed = json.loads(value, parse_constant=_reject_nonfinite)
    except (ValueError, RecursionError):
        raise ProviderError(f"{description} is not valid JSON.") from None
    if not isinstance(parsed, dict):
        raise ProviderError(f"{description} must be a JSON object.")
    return parsed


def _transient_transport(error: BaseException) -> bool:
    """Classify typed failures only; exception text may contain private data."""
    # urllib wraps socket failures in URLError.reason. Never interpret strings
    # as error codes, or recursively follow an unbounded/cyclic wrapper chain.
    for _ in range(5):
        if not isinstance(error, URLError):
            break
        error = error.reason
    if isinstance(error, (ValueError, ssl.SSLError)):
        return False
    if isinstance(error, (TimeoutError, RemoteDisconnected, IncompleteRead,
                          ConnectionResetError, ConnectionRefusedError,
                          ConnectionAbortedError, BrokenPipeError)):
        return True
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    return isinstance(error, OSError) and error.errno in _TRANSIENT_CONNECTION_ERRNOS


class OpenAICompatibleClient:
    """Send completions with bounded request retries; callers cache and resume work.

    Credentials are deliberately excluded from ``identity``. An omitted key is
    supported for local inference servers. This module does not read environment
    variables or choose a default model.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        json_mode: bool = True,
        max_output_tokens: int | None = None,
        token_limit_field: str = "max_completion_tokens",
        reasoning_effort: str | None = None,
        max_retries: int = 0,
        retry_initial_delay: float = 5.0,
        retry_max_delay: float = 60.0,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ProviderError("A nonempty model name is required.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ProviderError("The API timeout must be a positive, finite number.")
        try:
            valid_timeout = math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ProviderError("The API timeout must be a positive, finite number.")
        if not isinstance(json_mode, bool):
            raise ProviderError("JSON mode must be true or false.")
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ProviderError("The output token limit must be a positive integer.")
        if not isinstance(token_limit_field, str) or token_limit_field not in {"max_completion_tokens", "max_tokens"}:
            raise ProviderError("The token limit field must be max_completion_tokens or max_tokens.")
        if reasoning_effort is not None and (not isinstance(reasoning_effort, str) or reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}):
            raise ProviderError("The reasoning effort is not a supported level.")
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ProviderError("The API retry count must be an integer from 0 to 10.")
        for delay in (retry_initial_delay, retry_max_delay):
            if type(delay) not in {int, float} or not 1 <= delay <= 60:
                raise ProviderError("API retry delays must be numbers from 1 to 60 seconds.")
        if retry_initial_delay > retry_max_delay:
            raise ProviderError("The initial API retry delay must not exceed the maximum delay.")
        if api_key is not None:
            if not isinstance(api_key, str) or any(
                ord(char) < 32 or ord(char) > 126 for char in api_key
            ):
                raise ProviderError("The API key contains invalid header characters.")

        self.model = model.strip()
        self.base_url = _base_url(base_url)
        self.timeout = float(timeout)
        self.json_mode = json_mode
        self.max_output_tokens = max_output_tokens
        self.token_limit_field = token_limit_field
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_initial_delay = float(retry_initial_delay)
        self.retry_max_delay = float(retry_max_delay)
        self._api_key = api_key
        self._opener = build_opener(_NoRedirect())

    @property
    def identity(self) -> dict:
        """Return cache-relevant provider configuration without credentials."""
        identity = {
            "model": self.model,
            "base_url": self.base_url,
            "json_mode": self.json_mode,
            "max_output_tokens": self.max_output_tokens,
            "token_limit_field": self.token_limit_field,
        }
        if self.reasoning_effort is not None:
            identity["reasoning_effort"] = self.reasoning_effort
        return identity

    def complete(self, system: str, payload: dict) -> dict:
        if not isinstance(system, str) or not isinstance(payload, dict):
            raise ProviderError("Completion inputs must be a system string and a payload object.")
        try:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": f"{system}\n\n{_JSON_INSTRUCTION}"},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    },
                ],
            }
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            if self.max_output_tokens is not None:
                body[self.token_limit_field] = self.max_output_tokens
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise ProviderError("The completion payload cannot be encoded as JSON.") from None

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        raw = self._request(encoded, headers)

        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError("The API response exceeds the 16 MiB limit.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ProviderError("The API response is not valid UTF-8 JSON.") from None
        result = _json_object(text, "The API response")
        if result.get("error") is not None:
            raise ProviderError("The API returned an error response.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("The API response has no valid completion choice.")
        choice = choices[0]
        reason = choice.get("finish_reason")
        if reason == "length":
            raise OutputTruncatedError("The model output was truncated. Increase the output token limit or reduce the input.")
        if reason == "content_filter":
            raise ProviderError("The model output was blocked by the provider's content filter.")
        if reason != "stop":
            raise ProviderError("The API did not return a completed text response.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("The API response has no valid completion message.")
        if message.get("role", "assistant") != "assistant":
            raise ProviderError("The API response has an invalid completion role.")
        if message.get("refusal") not in (None, ""):
            raise ProviderError("The model refused the request.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("The API response has no text content.")
        return _json_object(content, "The model output")

    def _request(self, encoded: bytes, headers: dict) -> bytes:
        # Retry synchronously inside the caller's worker. Never create additional
        # workers or change the serialized request/cache identity between attempts.
        for attempt in range(self.max_retries + 1):
            failure = None
            try:
                request = Request(
                    f"{self.base_url}/chat/completions", data=encoded, headers=headers, method="POST"
                )
                with self._opener.open(request, timeout=self.timeout) as response:
                    if not 200 <= response.status < 300:
                        failure = self._http_failure(response.status, getattr(response, "headers", {}), response)
                    else:
                        return response.read(MAX_RESPONSE_BYTES + 1)
            except HTTPError as error:
                try:
                    failure = self._http_failure(error.code, error.headers, error)
                finally:
                    error.close()
            except (TimeoutError, URLError, OSError, HTTPException, ValueError) as error:
                # Only typed transient failures may retry. Never surface raw
                # reasons, URLs, partial response bodies, or TLS diagnostics.
                failure = ProviderError(
                    "The API request failed or timed out. Check the connection and API configuration.",
                    retryable=_transient_transport(error),
                )
            if not failure.retryable or attempt == self.max_retries:
                raise failure from None
            delay = min(self.retry_max_delay, self.retry_initial_delay * 2 ** attempt)
            delay = min(self.retry_max_delay, delay + random.uniform(0, delay / 4))
            if failure.retry_after_seconds is not None:
                # Never ignore a longer server cooldown by clipping it to our
                # bound. Stop and leave cached work available for a later resume.
                if failure.retry_after_seconds > self.retry_max_delay:
                    raise failure from None
                delay = max(delay, failure.retry_after_seconds)
            if failure.status_code is None:
                _LOGGER.warning("API transient transport failure; retry %d/%d in %.1f seconds.",
                                attempt + 1, self.max_retries, delay)
            else:
                limit_summary = ", ".join(f"{key}={value}" for key, value in failure.rate_limits.items())
                _LOGGER.warning("API HTTP %s%s%s; retry %d/%d in %.1f seconds.",
                                failure.status_code, f" ({failure.code})" if failure.code else "",
                                f" [{limit_summary}]" if limit_summary else "",
                                attempt + 1, self.max_retries, delay)
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    @classmethod
    def _http_failure(cls, status: int, headers, response) -> ProviderError:
        retryable = status in _TRANSIENT_HTTP_STATUSES
        code = None
        if retryable:
            # Read only a small bounded error envelope, and expose only known
            # codes. Providers sometimes echo credentials or input in messages.
            try:
                raw = response.read(MAX_ERROR_BYTES + 1)
                value = json.loads(raw) if isinstance(raw, bytes) and len(raw) <= MAX_ERROR_BYTES else None
            except (ValueError, TypeError, RecursionError, OSError, HTTPException):
                value = None
            details = value.get("error") if isinstance(value, dict) else None
            if not isinstance(details, dict):
                details = value if isinstance(value, dict) else {}
            raw_code = details.get("code")
            code = raw_code if isinstance(raw_code, str) and raw_code in _SAFE_ERROR_CODES else None
            # A provider's explicit nonretryable verdict always wins. Do not
            # turn billing/quota exhaustion into repeated paid model requests.
            if details.get("retryable") is False or (isinstance(value, dict) and value.get("retryable") is False):
                retryable = False
            diagnostic = " ".join(str(details.get(key, "")) for key in ("code", "type", "message")).lower()
            if code in _NONRETRYABLE_CODES or any(term in diagnostic for term in (
                "quota", "billing", "credit", "payment", "daily limit", "monthly limit",
                "authentication", "invalid api key", "permission denied",
            )):
                retryable = False
        retry_after = cls._retry_after(headers)
        rate_limits = cls._rate_limits(headers)
        message = cls._http_error_message(status)
        if code is not None:
            message += f" Provider code: {code}."
        if rate_limits:
            message += " Rate limits: " + ", ".join(f"{key}={value}" for key, value in rate_limits.items()) + "."
        return ProviderError(message, status_code=status, code=code, retryable=retryable,
                             retry_after_seconds=retry_after, rate_limits=rate_limits)

    @staticmethod
    def _rate_limits(headers) -> dict[str, int | float]:
        # Only documented numeric headers are diagnostic output. Never include
        # arbitrary header names/values, dimensions, or server message text.
        result = {}
        for header, key in _RATE_LIMIT_HEADERS.items():
            raw = headers.get(header) if headers is not None else None
            if not isinstance(raw, str) or len(raw) > 128:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0:
                result[key] = int(value) if value.is_integer() else value
        return result

    @staticmethod
    def _retry_after(headers) -> float | None:
        value = headers.get("Retry-After") if headers is not None else None
        if not isinstance(value, str):
            return None
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    return None
                delay = max(0.0, date.timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                return None
        return delay if math.isfinite(delay) and delay >= 0 else None

    @staticmethod
    def _http_error_message(status: int) -> str:
        if 300 <= status < 400:
            return "API redirect refused. Configure the final API base URL."
        if status in {401, 403}:
            return f"API request failed (HTTP {status}). Check the API key and permissions."
        if status == 429:
            return "API request failed (HTTP 429). Check quota and rate limits before rerunning."
        return f"API request failed (HTTP {status})."
