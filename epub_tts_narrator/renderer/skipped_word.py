"""Accept a take that leaves out one source word every take leaves out, when it is a source error.

Source text sometimes contains a typo such as a stray word ("when it he
eventually got to work"). A good reading skips it, and ordinary verification
rightly never excuses a missing word, so every take fails. This overlay runs
after ``validate_audio`` on a failed take (like ``heard_variants``), so
validation keys and caches are unchanged. It accepts only when all hold:

* removing exactly one source word (never a number or negation) makes the
  transcript a direct match: exact, word spacing, or complete Double Metaphone;
* at least ``POLICY["min_takes"]`` distinct takes of this request left out that
  same word, and no take of this request read it;
* the text judge says the word is an error in the source text.

The accepted report keeps the original failure and records the audit; it never
claims a verbatim match.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unicodedata

POLICY = {"name": "skipped_source_word", "version": 1, "min_takes": 2, "max_candidates": 12}
_CONTEXT_WORDS = 4
# Takes generated from other words than the request's own text (clip repairs,
# spelling rewrites) neither count toward nor against a skipped source word.
_REWRITTEN = ("speech_resolution", "clip_repair")


def _rewritten(metadata):
    return isinstance(metadata, dict) and any(key in metadata for key in _REWRITTEN)


def _source(text):
    return unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")


def _direct(expected, transcript):
    from .phonetic_match import phonetic_equivalence
    from .quality import compare_words, word_spacing_equivalence
    if compare_words(expected, transcript)["wer"] == 0:
        return "exact"
    if word_spacing_equivalence(expected, transcript) is not None:
        return "word_spacing"
    if phonetic_equivalence(expected, transcript) is not None:
        return "double_metaphone"
    return None


def _without(source, start, end):
    before, after = source[:start].rstrip(), source[end:].lstrip()
    spaced = source[:start] != before or source[end:] != after
    return before + (" " if before and after and spaced else "") + after


def find(expected_text, transcript):
    """The one source word whose removal makes ``transcript`` a direct match, or None."""
    from .phonetic_match import _ALPHABETIC_WORD, _PROTECTED
    from .quality import _lexeme_matches, compare_words, normalize_words
    if not isinstance(expected_text, str) or not isinstance(transcript, str):
        return None
    source = _source(expected_text)
    tokens, index = [], 0
    for match in _lexeme_matches(source):
        words = normalize_words(match.group())
        tokens.append((match, index, words))
        index += len(words)
    try:
        alignment = compare_words(expected_text, transcript)["alignment"]
    except ValueError:
        return None
    if not any(edit["operation"] == "delete" for edit in alignment):
        return None
    changed = {position for edit in alignment if edit["operation"] != "equal"
               for position in range(edit["expected_start"], edit["expected_end"])}
    candidates = [(match, position, words[0]) for match, position, words in tokens
                  if position in changed and len(words) == 1
                  and _ALPHABETIC_WORD.fullmatch(words[0]) and "n't" not in words[0]
                  and words[0].replace("'", "") not in _PROTECTED]
    if not candidates or len(candidates) > POLICY["max_candidates"]:
        return None
    found = {}
    for match, position, word in candidates:
        trimmed = _without(source, match.start(), match.end())
        method = _direct(trimmed, transcript)
        if method is None:
            continue
        key = hashlib.sha256(json.dumps(normalize_words(trimmed)).encode("utf-8")).hexdigest()
        found.setdefault(key, (match, position, word, trimmed, method))
    if len(found) != 1:
        return None
    key, (match, position, word, trimmed, method) = next(iter(found.items()))
    at = [row[0] for row in tokens].index(match)
    before = [row[0].group() for row in tokens[max(0, at - _CONTEXT_WORDS):at]]
    after = [row[0].group() for row in tokens[at + 1:at + 1 + _CONTEXT_WORDS]]
    return {"word": match.group(), "word_index": position, "key": key, "method": method,
            "context": " ".join([*before, f"[{match.group()}]", *after])}


def _read(expected_text, transcript, position):
    """Whether a transcript plainly read the source word at ``position``."""
    from .quality import compare_words
    try:
        alignment = compare_words(expected_text, transcript)["alignment"]
    except ValueError:
        return False
    return any(edit["operation"] == "equal" and edit["expected_start"] <= position < edit["expected_end"]
               for edit in alignment)


def _takes(directory, request_hash, expected_text, found, audio_sha256):
    """Distinct takes of this request leaving out the same word; None if one read it."""
    from .common import read_json
    takes = {audio_sha256}
    for path in sorted(Path(directory).glob("attempt-*.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError):
            continue
        report = record.get("report") if isinstance(record, dict) else None
        audio = record.get("audio_sha256") if isinstance(record, dict) else None
        if (record.get("request_hash") != request_hash or not isinstance(report, dict)
                or not isinstance(audio, str) or audio in takes or _rewritten(record.get("engine_metadata"))):
            continue
        heard = [report.get("transcript")]
        corroboration = report.get("corroboration")
        if isinstance(corroboration, dict):
            heard.append(corroboration.get("transcript"))
        heard = [text for text in heard if isinstance(text, str) and text.strip()]
        if any(_read(expected_text, text, found["word_index"]) for text in heard):
            return None
        if any((other := find(expected_text, text)) is not None and other["key"] == found["key"]
               for text in heard):
            takes.add(audio)
    return takes


def eligible(report):
    from .quality import text_confirmation_unavailable
    return (isinstance(report, dict) and report.get("passed") is not True
            and report.get("purpose", "speech") == "speech" and report.get("reasons") == ["text_mismatch"]
            and "history_echo" not in report and not text_confirmation_unavailable(report)
            and report.get("resolution_pending") is not True
            and report.get("text_judge_skipped") == "incomplete_spoken_coverage"
            and isinstance(report.get("expected_text"), str)
            and isinstance(report.get("transcript"), str) and bool(report["transcript"].strip()))


def apply(report, *, directory, verifier, request_hash, audio_sha256, engine_metadata=None):
    """Return ``(report, outcome)``; outcome is None, ``{"pending": …}``, ``{"rejected": …}`` or ``{"accepted": …}``."""
    if (not eligible(report) or not isinstance(audio_sha256, str) or directory is None
            or _rewritten(engine_metadata)):
        return report, None
    judge = getattr(verifier, "judge_skipped_word", None)
    identity = getattr(verifier, "reference_judge_identity", None)
    if not callable(judge) or not isinstance(identity, dict):
        return report, None
    expected = report["expected_text"]
    heard = [report["transcript"]]
    corroboration = report.get("corroboration")
    if isinstance(corroboration, dict) and isinstance(corroboration.get("transcript"), str):
        heard.append(corroboration["transcript"])
    for transcript in heard:
        found = find(expected, transcript)
        if found is not None:
            break
    else:
        return report, None
    takes = _takes(directory, request_hash, expected, found, audio_sha256)
    if takes is None:
        return report, None
    audit = {"policy": deepcopy(POLICY), "word": found["word"], "context": found["context"],
             "method": found["method"], "transcript": transcript, "takes": sorted(takes)}
    if len(takes) < POLICY["min_takes"]:
        return dict(report, skipped_source_word=audit), {"pending": audit}
    try:
        judgment = judge(expected, transcript, found["word"], found["context"])
    except Exception:
        judgment = None
    if (not isinstance(judgment, dict) or judgment.get("identity") != identity
            or judgment.get("decision") not in {"source_error", "belongs", "uncertain", "unavailable"}):
        judgment = {"decision": "unavailable", "error_code": "invalid_judgment"}
    audit["judgment"] = deepcopy(judgment)
    if judgment["decision"] != "source_error":
        return dict(report, skipped_source_word=audit), {"rejected": audit}
    accepted = {**deepcopy(report), "passed": True, "reasons": [], "previous_reasons": list(report["reasons"]),
                "text_confirmation_method": "skipped_source_word", "verified_transcript": transcript,
                "skipped_source_word": audit}
    return accepted, {"accepted": audit}
