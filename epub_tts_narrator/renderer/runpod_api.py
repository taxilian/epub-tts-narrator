"""Small, file-credential-only Runpod management client.

Pod creation, configuration, lifecycle actions, and the GPU/data-center catalog
use v2. Pod reads and volume management retain v1. Hosts are fixed, redirects
and ambient HTTP proxies are disabled, and provider error bodies are never
exposed. A failed mutation is never retried here: callers must reconcile
uncertain outcomes before creating another paid resource.
"""

from __future__ import annotations

from http.client import HTTPException
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


_V1 = "https://rest.runpod.io/v1"
_V2 = "https://api.runpod.io/v2"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_ERROR_BYTES = 16 * 1024
_HOST_CAPACITY_DETAIL = "There are not enough free GPUs on the host machine to start this pod."
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class RunpodAPIError(RuntimeError):
    """Sanitized failure with machine-readable status and outcome certainty."""

    def __init__(self, message, *, status=None, ambiguous=False, retryable=False):
        super().__init__(message)
        self.status = status
        self.ambiguous = ambiguous
        self.retryable = retryable


class RunpodAmbiguousOutcome(RunpodAPIError):
    """A mutation may have succeeded; do not issue a replacement blindly."""

    def __init__(self, message, *, status=None):
        super().__init__(message, status=status, ambiguous=True, retryable=False)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # HTTPError preserves the status for classification. Neither Location
        # nor the provider body becomes part of the user-facing exception.
        raise HTTPError(request.full_url, code, "Redirect refused", headers, fp)


def _resource_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Runpod resource ID must contain only letters, digits, underscores, and hyphens")
    return value


def _query_values(values, *, comma_lists):
    output = {}
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key):
            raise ValueError("Invalid Runpod query field")
        if value is None:
            continue
        parts = value if isinstance(value, (tuple, list)) else [value]
        if any(not isinstance(part, (str, int, float, bool)) for part in parts):
            raise ValueError("Runpod query values must be strings, numbers, or booleans")
        parts = [str(part).lower() if isinstance(part, bool) else str(part) for part in parts]
        output[key] = ",".join(parts) if comma_lists else parts
    return urlencode(output, doseq=not comma_lists)


class RunpodAPI:
    """Use an explicit key loaded by the caller from its configuration file.

    Successful payloads are returned intact, including Pod publicIp and
    portMappings. GETs retry transient failures at most twice. Mutations never
    retry, including start and stop; inspect the Pod afterward if the result
    is ambiguous. Acknowledging start/stop does not mean the transition has
    finished, so callers still need to poll desiredStatus/runtime readiness.
    """

    def __init__(self, api_key, timeout=30):
        if (not isinstance(api_key, str) or not api_key or len(api_key) > 8192
                or any(ord(char) < 33 or ord(char) > 126 for char in api_key)):
            raise ValueError("An explicit Runpod API key is required in the configuration file")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= 300):
            raise ValueError("Runpod request timeout must be between 0 and 300 seconds")
        self._api_key = api_key
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def list_pods(self, **filters):
        return self._request("GET", "/pods", query=filters, expected=list)

    def get_pod(self, pod_id, **expansions):
        return self._request("GET", f"/pods/{_resource_id(pod_id)}", query=expansions, expected=dict)

    def create_pod(self, spec):
        """Create from the v2 nested request schema; return its cost/status fields."""
        return self._request("POST", "/pods", spec=spec, v2=True, expected=dict, require_id=True)

    def update_pod(self, pod_id, spec):
        """Patch only supplied configuration fields; inspect status before starting.

        Omitted fields, including mounts, are preserved by Runpod. This does
        not request a lifecycle action, and an uncertain result is never retried.
        """
        return self._request("PATCH", f"/pods/{_resource_id(pod_id)}", spec=spec,
                             v2=True, expected=dict, require_id=True)

    def start_pod(self, pod_id):
        return self._pod_action(pod_id, "start")

    def stop_pod(self, pod_id):
        return self._pod_action(pod_id, "stop")

    def _pod_action(self, pod_id, action):
        pod_id = _resource_id(pod_id)
        result = self._request("POST", f"/pods/{pod_id}/action", spec={"action": action},
                               v2=True, expected=dict, require_id=True)
        # v2 start/stop return the updated Pod. An empty response or another
        # resource's response cannot confirm this action; observe before retry.
        if result["id"] != pod_id:
            raise self._invalid_response(True)
        return result

    def list_volumes(self):
        return self._request("GET", "/networkvolumes", expected=list)

    def get_volume(self, volume_id):
        return self._request("GET", f"/networkvolumes/{_resource_id(volume_id)}", expected=dict)

    def create_volume(self, spec):
        return self._request("POST", "/networkvolumes", spec=spec, expected=dict, require_id=True)

    def catalog_gpus(self, **query):
        """Read v2 GPU pricing; include='AVAILABILITY', product='POD' adds stock."""
        return self._request("GET", "/catalog/gpus", query=query, v2=True, expected=dict)

    def datacenters(self, **query):
        """Read v2 data centers, optionally include='GPU_AVAILABILITY'."""
        return self._request("GET", "/catalog/datacenters", query=query, v2=True, expected=dict)

    def _request(self, method, path, *, spec=None, query=None, v2=False,
                 expected=None, allow_empty=False, require_id=False):
        mutation = method != "GET"
        body = None
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json",
                   "User-Agent": "epub-tts-narrator/1"}
        if require_id and not isinstance(spec, dict):
            raise ValueError("Runpod creation settings must be a JSON object")
        if spec is not None:
            if not isinstance(spec, dict):
                raise ValueError("Runpod creation settings must be a JSON object")
            try:
                body = json.dumps(spec, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                raise ValueError("Runpod creation settings must contain valid JSON values") from None
            headers["Content-Type"] = "application/json"
        elif mutation:
            body = b""
        url = (_V2 if v2 else _V1) + path
        if query:
            encoded = _query_values(query, comma_lists=v2)
            if encoded:
                url += "?" + encoded
        request = Request(url, data=body, headers=headers, method=method)
        attempts = 1 if mutation else 3
        for attempt in range(attempts):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    status = response.status
                    if not 200 <= status < 300:
                        raise self._http_error(status, mutation)
                    content = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(content) > _MAX_RESPONSE_BYTES:
                    raise self._invalid_response(mutation)
                if not content.strip() and allow_empty:
                    return None
                try:
                    result = json.loads(content)
                except (ValueError, UnicodeError):
                    raise self._invalid_response(mutation) from None
                if (not isinstance(result, (dict, list))
                        or (expected is not None and not isinstance(result, expected))
                        or (isinstance(result, list) and any(not isinstance(row, dict) for row in result))):
                    raise self._invalid_response(mutation)
                if require_id:
                    try:
                        _resource_id(result.get("id"))
                    except ValueError:
                        raise self._invalid_response(mutation) from None
                return result
            except HTTPError as exc:
                status = exc.code
                try:
                    error_body = exc.read(_MAX_ERROR_BYTES + 1)
                except (URLError, OSError, HTTPException):
                    error_body = b""
                finally:
                    exc.close()
                failure = self._http_error(status, mutation, error_body=error_body)
            except (URLError, OSError, HTTPException):
                if mutation:
                    failure = RunpodAmbiguousOutcome(
                        "Runpod did not confirm the request. It may have succeeded; check saved resources before retrying.")
                else:
                    failure = RunpodAPIError("Unable to reach the Runpod API after a bounded request", retryable=True)
            except RunpodAPIError as exc:
                failure = exc
            if failure.retryable and attempt + 1 < attempts:
                time.sleep(0.5 * (2 ** attempt))
                continue
            # Do not attach a urllib exception containing the URL, response
            # details, or credentials as the displayed exception cause.
            raise failure from None

    @staticmethod
    def _invalid_response(mutation):
        if mutation:
            return RunpodAmbiguousOutcome(
                "Runpod returned an unreadable or incomplete success response. The request may have succeeded; "
                "check saved resources before retrying.")
        return RunpodAPIError("Runpod returned an unreadable or unexpected response")

    @staticmethod
    def _http_error(status, mutation, *, error_body=b""):
        if mutation and (status >= 500 or status == 408 or 300 <= status < 400):
            return RunpodAmbiguousOutcome(
                f"Runpod did not confirm the request (HTTP {status}). It may have succeeded; "
                "check saved resources before retrying.", status=status)
        if status == 400 and mutation and len(error_body) <= _MAX_ERROR_BYTES:
            try:
                problem = json.loads(error_body)
            except (ValueError, UnicodeError, RecursionError):
                problem = None
            # Translate only this exact observed provider diagnosis. No
            # arbitrary provider text or echoed request values are exposed.
            if isinstance(problem, dict) and problem.get("detail") == _HOST_CAPACITY_DETAIL:
                return RunpodAPIError(
                    "Runpod cannot restart this Pod because its assigned host has no free GPU capacity "
                    "(HTTP 400). The Pod remains stopped; retry later or replace the Pod while keeping "
                    "its existing network volume.", status=status)
        detail = {
            400: "rejected the request settings", 401: "rejected the API key",
            403: "denied permission for this operation", 404: "could not find the requested resource",
            409: "reported a resource conflict", 422: "rejected the request settings",
            429: "rate limited the request",
        }.get(status, "rejected the request")
        return RunpodAPIError(f"Runpod {detail} (HTTP {status})", status=status,
                              retryable=not mutation and status in _TRANSIENT_STATUSES)
