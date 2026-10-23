"""Complete diagnostic messages with credentials and terminal controls removed."""

from __future__ import annotations

import json
import re
import traceback
import unicodedata
from urllib.parse import quote, quote_plus


_TERMINAL_SEQUENCES = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[PX^_][\s\S]*?\x1b\\|[@-Z\\-_])"
)


def configured_secrets(config):
    """Collect configured credentials without treating arbitrary settings as secret."""
    secrets = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key).lower().replace("-", "_")
                credential = name in {
                    "api_key", "apikey", "password", "passwd", "token", "secret",
                    "authorization", "access_token", "refresh_token", "auth_token", "client_secret",
                } or name.endswith(("_api_key", "_password", "_token", "_secret"))
                if credential and isinstance(item, str) and item:
                    secrets.add(item)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(config)
    return secrets


def safe_error_text(value, secrets=()):
    """Retain the full error; mask exact credentials, never whole provider messages."""
    text = str(value)
    variants = set()
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        variants.update((secret, quote(secret, safe=""), quote(secret), quote_plus(secret),
                         json.dumps(secret, ensure_ascii=True)[1:-1],
                         json.dumps(secret, ensure_ascii=False)[1:-1]))
    # Longer overlapping values must be removed before any shorter credential.
    for secret in sorted(variants, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = _TERMINAL_SEQUENCES.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(character for character in text if character in "\n\t"
                   or not unicodedata.category(character).startswith("C"))


_SAFE_PROVIDER_CODE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")


def exception_detail(exc):
    """Identify a failure by type and status codes only, never by its raw text.

    SDK and HTTP errors can embed credentials, endpoints, or request payloads
    in their message; the structured status and provider code stay actionable
    without echoing that text.
    """
    parts = [type(exc).__name__]
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        code = error.get("Code") if isinstance(error, dict) else None
        if isinstance(code, str) and _SAFE_PROVIDER_CODE.fullmatch(code):
            parts.append(f"provider code {code}")
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict) and status is None:
            status = metadata.get("HTTPStatusCode")
    if isinstance(status, int) and not isinstance(status, bool):
        parts.append(f"HTTP status {status}")
    errno = getattr(exc, "errno", None)
    if isinstance(errno, int) and not isinstance(errno, bool):
        parts.append(f"errno {errno}")
    return ", ".join(parts)


def format_fatal(exc, secrets=()):
    """Fatal CLI report: exception type, redacted message, and code location.

    Frames report file, line, and function only — no source lines, no locals,
    no exception chains — so book text never reaches the output.
    """
    lines = [f"{type(exc).__name__}: {exc}"]
    if exc.__traceback__ is not None:
        for frame in traceback.extract_tb(exc.__traceback__):
            lines.append(f"  at {frame.name} ({frame.filename}:{frame.lineno})")
    return safe_error_text("\n".join(lines), secrets)
