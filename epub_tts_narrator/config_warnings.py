"""Keep stale configuration fields advisory without leaking their values."""

import re
import sys


def known_settings(settings, allowed, *, section):
    """Return recognized keys, warning about each ignored field on stderr.

    Only field names are reported, never their values or the configuration file.
    Unusual or credential-shaped names are also omitted from diagnostics.
    """
    result = {}
    for key, value in settings.items():
        if key in allowed:
            result[key] = value
            continue
        safe_key = (key if isinstance(key, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key)
                    and not key.startswith(("sk_", "apikey_", "hf_", "AKIA", "ASIA"))
                    else "<redacted-key>")
        print(f"Warning: ignoring unknown configuration key {section}.{safe_key}.", file=sys.stderr)
    return result
