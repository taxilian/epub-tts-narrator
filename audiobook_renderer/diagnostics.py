"""Complete diagnostic messages with credentials and terminal controls removed."""

from __future__ import annotations

import json
import re
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
