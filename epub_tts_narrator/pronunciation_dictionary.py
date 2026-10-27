"""Offline, revision-pinned CMUdict readings; never pronunciation approvals."""

from functools import lru_cache
import gzip
import hashlib
from importlib.resources import files
import re


DICTIONARY_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"


@lru_cache(maxsize=1)
def _dictionary() -> dict[str, tuple[str, ...]]:
    data = gzip.decompress(files("epub_tts_narrator").joinpath("data/cmudict.dict.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != DICTIONARY_SHA256:
        raise ValueError("Bundled pronunciation dictionary checksum mismatch")
    readings: dict[str, list[str]] = {}
    for line in data.decode("utf-8").splitlines():
        fields = line.partition("#")[0].split()
        if fields:
            word = re.sub(r"\(\d+\)$", "", fields[0])
            readings.setdefault(word, []).append(" ".join(fields[1:]))
    return {word: tuple(variants) for word, variants in readings.items()}


def dictionary_readings(word: str) -> tuple[str, ...]:
    """Return all readings; case folding is lookup-only, never identity folding."""
    return _dictionary().get(word.casefold(), ())


def review_candidates(document, entries, *, automatic=False, name_words=None):
    """Hide only automatic dictionary-covered pending rows, without deleting them.

    Cast membership comes from exact observed names, not model-assigned kinds.
    Drafts and explicit decisions remain visible regardless of dictionary hits.
    Fresh model output cannot claim manual-review intent through its reason.
    """
    from .pronunciation_discovery import character_name_candidates
    from .pronunciation_groups import entry_words

    names = ({entry["word"] for entry in character_name_candidates(document)}
             if name_words is None else name_words)
    return [entry for entry in entries
            if entry["status"] != "pending"
            or not automatic and (entry.get("draft") or entry.get("review_requested") or entry.get("case_conflict")
                                  or entry.get("reason") == "Manually selected for pronunciation review.")
            or any(word in names for word in entry_words(entry)) or not dictionary_readings(entry["word"])]
