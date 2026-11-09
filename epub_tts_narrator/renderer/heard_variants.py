"""Heard variants: how this book's recognizer consistently hears particular words.

The speech recognizer often turns an invented name, or another word it does not
know, into ordinary words ("Kvothe" -> "quote"). Each time a take fails only
because of such a word, the misheard words are recorded as a *sighting* of a
variant of that word, bound to the recognizer, the take's audio and its
transcript. A variant becomes *trusted* when either

* three independent takes (three different audio files) were heard the same
  way and the variant passes a sound-plausibility check ("repeated"), or
* three separate sound reviews approved exactly that pair in an accepted take
  ("review"), or
* a person accepted a ``pronounce-name`` test sentence whose audio the same
  recognizer heard that way ("listened").

A later take whose only differences are covered by existing deterministic
rules or by trusted variants is accepted with
``text_confirmation_method = "heard_variant"``; it never claims a verbatim
match (the original word metrics and failure reason stay in the report).

Fail-closed rules: a variant never excuses a missing or extra whole word, a
changed number or negation, a function word, a reordered word, a history echo,
or any failed acoustic check. A take with any difference outside eligible
blocks records nothing and is not accepted.

Storage is book-local ``heard-variants.json`` beside the analysis, inherited
read-only by books continued from it (nearest book first); new sightings are
written only to the rendering book. No model, network or identity change.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
import threading
import time
import unicodedata

FILENAME = "heard-variants.json"
FORMAT = "epub-tts-heard-variants"
VERSION = 1
POLICY = {
    "name": "heard_variants", "version": 1,
    "repeat_takes": 3, "max_blocks": 4, "max_fraction": 0.2, "max_block_words": 3,
    "plausibility": "double_metaphone_similarity_v1",
}
SOURCES = frozenset({"render", "review", "listened"})
TRUSTING_SOURCES = frozenset({"review", "listened"})
MAX_SIGHTINGS = 16
MAX_VARIANTS_PER_WORD = 16
MAX_FILE_BYTES = 32 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*\Z", re.UNICODE)
_LOCKS_GUARD = threading.Lock()
_LOCKS = {}


# Identity -------------------------------------------------------------------

def recognizer_key(verifier):
    """Digest of the primary recognizer's path-free identity, or None.

    Wrapper policies (JEV, sound review, recovery) do not change what the
    recognizer hears, so a variant survives their changes; a different model,
    package or decoding option starts a fresh set.
    """
    from .recognition_workers import comparable_identity, identity_digest, recognizers
    try:
        primary, _ = recognizers(verifier)
    except ValueError:
        return None
    identity = getattr(primary, "identity", None)
    if not isinstance(identity, dict):
        return None
    try:
        return identity_digest(comparable_identity(identity))
    except (TypeError, ValueError):
        return None


# Alignment ------------------------------------------------------------------

def _bare(word):
    return word.replace("'", "")


def difference_blocks(expected_text, transcript):
    """Maximal runs of non-equal word edits between exact anchors, or None.

    Each block carries normalized words and the original text spans so the
    existing span-level rules can be applied to exactly that difference.
    """
    from .phonetic_review import _original_words
    from .quality import compare_words, normalize_words
    if not isinstance(expected_text, str) or not isinstance(transcript, str):
        return None
    try:
        comparison = compare_words(expected_text, transcript)
    except ValueError:
        return None
    expected, heard = normalize_words(expected_text), normalize_words(transcript)
    expected_original = _original_words(expected_text, expected)
    heard_original = _original_words(transcript, heard)
    if expected_original is None or heard_original is None:
        return None
    blocks, current = [], None
    for edit in comparison["alignment"]:
        if edit["operation"] == "equal":
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if current is None:
            current = dict(expected_start=edit["expected_start"], actual_start=edit["actual_start"])
        current.update(expected_end=edit["expected_end"], actual_end=edit["actual_end"])
    if current is not None:
        blocks.append(current)

    def span(text, originals, start, end):
        if start == end:
            return ""
        return text[originals[start][1]:originals[end - 1][2]]

    for block in blocks:
        block["expected"] = expected[block["expected_start"]:block["expected_end"]]
        block["heard"] = heard[block["actual_start"]:block["actual_end"]]
        block["expected_span"] = span(expected_text, expected_original, block["expected_start"], block["expected_end"])
        block["heard_span"] = span(transcript, heard_original, block["actual_start"], block["actual_end"])
    return {"blocks": blocks, "expected_word_count": len(expected)}


def _covered(block):
    """Already acceptable by the existing deterministic span rules."""
    from .phonetic_match import phonetic_equivalence
    from .quality import word_spacing_equivalence
    if not block["expected"] or not block["heard"]:
        return False
    try:
        return (word_spacing_equivalence(block["expected_span"], block["heard_span"]) is not None
                or phonetic_equivalence(block["expected_span"], block["heard_span"]) is not None)
    except ValueError:
        return False


def _structural_reason(block):
    """None when a block may be a heard variant; otherwise why it may not."""
    from .phonetic_match import _FUNCTION_WORDS, _PROTECTED
    expected, heard = block["expected"], block["heard"]
    limit = POLICY["max_block_words"]
    if not expected:
        return "extra_words"
    if not heard:
        return "missing_words"
    if len(expected) > limit or len(heard) > limit:
        return "too_many_words"
    words = expected + heard
    if any(not _WORD.fullmatch(word) for word in words):
        return "not_a_plain_word"
    if any(_bare(word) in _PROTECTED or "n't" in word for word in words):
        return "number_or_negation"
    if any(_bare(word) in _FUNCTION_WORDS for word in expected):
        return "function_word"
    if all(_bare(word) in _FUNCTION_WORDS for word in heard):
        return "function_word"
    if set(expected) & set(heard):
        return "added_or_omitted_word"
    return None


def _ascii(word):
    return unicodedata.normalize("NFKD", _bare(word)).encode("ascii", "ignore").decode().lower()


def _distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def plausibility(expected, heard):
    """Bounded sound resemblance of two word groups, or None when implausible.

    Letter lengths must stay within a factor of two, and the best pair of
    concatenated Double Metaphone keys must be at least half similar, or share
    their first sound and be at least 0.3 similar.
    """
    from itertools import product
    from .phonetic_match import _load_double_metaphone
    left_letters, right_letters = "".join(map(_ascii, expected)), "".join(map(_ascii, heard))
    if not left_letters or not right_letters:
        return None
    ratio = len(left_letters) / len(right_letters)
    if not 0.5 <= ratio <= 2.0:
        return None
    double_metaphone = _load_double_metaphone()

    def keys(words):
        options = []
        for word in words:
            found = tuple(dict.fromkeys(key for key in double_metaphone(_ascii(word)) if key))
            if not found:
                return ()
            options.append(found)
        return tuple({"".join(parts) for parts in product(*options)})

    best = None
    for left in keys(expected):
        for right in keys(heard):
            similarity = 1 - _distance(left, right) / max(len(left), len(right))
            if similarity >= 0.5 or (left[0] == right[0] and similarity >= 0.3):
                if best is None or similarity > best["similarity"]:
                    best = {"expected_key": left, "heard_key": right, "similarity": round(similarity, 4)}
    return best


def eligible_blocks(expected_text, transcript, *, require_plausible=True):
    """Blocks a heard variant must cover, or None when any difference is ineligible."""
    found = difference_blocks(expected_text, transcript)
    if found is None:
        return None
    uncovered = [block for block in found["blocks"] if not _covered(block)]
    allowed = max(1, int(POLICY["max_fraction"] * found["expected_word_count"]))
    if not uncovered or len(uncovered) > min(allowed, POLICY["max_blocks"]):
        return None
    result = []
    for block in uncovered:
        if _structural_reason(block) is not None:
            return None
        sound = plausibility(block["expected"], block["heard"])
        if require_plausible and sound is None:
            return None
        result.append({"expected": list(block["expected"]), "heard": list(block["heard"]), "plausibility": sound})
    return result


# Storage --------------------------------------------------------------------

def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Heard variants must be stored in a regular file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Heard variants exceed their size limit")

    def reject(_):
        raise ValueError("Heard variants contain a non-finite value")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _book_id(directory):
    source = _read(Path(directory) / "source.json")
    if not isinstance(source, dict) or not isinstance(source.get("book"), dict) or not isinstance(source["book"].get("id"), str):
        raise ValueError("Heard variants require this book's source")
    return source["book"]["id"]


def _owners(directory):
    directory = Path(directory).resolve()
    manifest = directory / "manifest.json"
    if manifest.is_file() and isinstance(_read(manifest).get("continuation"), dict):
        from ..analysis_directory import continuation_directories
        return [directory, *continuation_directories(directory)]
    return [directory]


def _valid_sighting(value):
    return (isinstance(value, dict) and set(value) == {"source", "audio_sha256", "request_hash", "transcript_sha256"}
            and value["source"] in SOURCES and isinstance(value["audio_sha256"], str) and _HEX.match(value["audio_sha256"])
            and (value["request_hash"] is None or isinstance(value["request_hash"], str) and _HEX.match(value["request_hash"]))
            and isinstance(value["transcript_sha256"], str) and _HEX.match(value["transcript_sha256"]))


def _validate(record, book_id):
    if (not isinstance(record, dict) or set(record) != {"format", "version", "book_id", "recognizers"}
            or record["format"] != FORMAT or type(record["version"]) is not int or record["version"] != VERSION
            or record["book_id"] != book_id or not isinstance(record["recognizers"], dict)):
        raise ValueError("Heard variants are invalid or belong to a different book")
    for recognizer, words in record["recognizers"].items():
        if not _HEX.match(recognizer) or not isinstance(words, dict):
            raise ValueError("Heard variants have an invalid recognizer entry")
        for key, variants in words.items():
            if not isinstance(variants, dict) or len(variants) > MAX_VARIANTS_PER_WORD:
                raise ValueError("Heard variants have an invalid word entry")
            for heard_key, variant in variants.items():
                if (not isinstance(variant, dict) or set(variant) != {"expected", "heard", "sightings"}
                        or not isinstance(variant["expected"], list) or " ".join(variant["expected"]) != key
                        or not isinstance(variant["heard"], list) or " ".join(variant["heard"]) != heard_key
                        or not all(isinstance(word, str) and _WORD.fullmatch(word)
                                   for word in variant["expected"] + variant["heard"])
                        or not isinstance(variant["sightings"], list)
                        or not 1 <= len(variant["sightings"]) <= MAX_SIGHTINGS
                        or not all(map(_valid_sighting, variant["sightings"]))
                        or len({row["audio_sha256"] for row in variant["sightings"]}) != len(variant["sightings"])):
                    raise ValueError("Heard variant record is invalid")
    return record


def load_local(directory):
    """This book's own record (empty when absent); never an ancestor's."""
    directory = Path(directory)
    book_id = _book_id(directory)
    path = directory / FILENAME
    if not path.exists() and not path.is_symlink():
        return {"format": FORMAT, "version": VERSION, "book_id": book_id, "recognizers": {}}
    return _validate(_read(path), book_id)


def effective_variants(directory, recognizer):
    """{(expected key, heard key): variant} visible here, sightings merged nearest first."""
    result = {}
    for owner in _owners(directory):
        if not (owner / FILENAME).exists() and not (owner / FILENAME).is_symlink():
            continue
        words = load_local(owner)["recognizers"].get(recognizer, {})
        for key, variants in words.items():
            for heard_key, variant in variants.items():
                merged = result.setdefault((key, heard_key), {
                    "expected": list(variant["expected"]), "heard": list(variant["heard"]), "sightings": [],
                    "origins": []})
                seen = {row["audio_sha256"] for row in merged["sightings"]}
                merged["sightings"].extend(deepcopy(row) for row in variant["sightings"] if row["audio_sha256"] not in seen)
                merged["origins"].append(str(owner))
    return result


def trust_basis(variant):
    """'listened', 'review' or 'repeated' when trusted; None otherwise."""
    sources = {row["source"] for row in variant["sightings"]}
    for basis in ("listened", "review"):
        if basis in sources:
            return basis
    takes = {row["audio_sha256"] for row in variant["sightings"]}
    if len(takes) >= POLICY["repeat_takes"] and plausibility(variant["expected"], variant["heard"]) is not None:
        return "repeated"
    return None


@contextmanager
def _locked(directory):
    from .run_locks import acquire_directory_lock
    directory = Path(directory).resolve()
    if directory.is_symlink():
        raise ValueError("Heard variant directory must not be a symlink")
    with _LOCKS_GUARD:
        local = _LOCKS.setdefault(str(directory), threading.Lock())
    with local:
        lock = None
        for delay in (0, 0.05, 0.1, 0.2, 0.4, 0.8):
            time.sleep(delay)
            try:
                lock = acquire_directory_lock(directory / ".heard-variants.lock")
                break
            except FileExistsError as exc:  # Held by another process (e.g. pronounce-name).
                failure = exc
        if lock is None:
            raise failure
        try:
            yield
        finally:
            lock.rmdir()


def record_sightings(directory, recognizer, blocks, *, source, audio_sha256, request_hash, transcript):
    """Durably add one take's sightings; return the effective variants for those blocks."""
    from .common import digest, write_json
    if source not in SOURCES or not _HEX.match(recognizer or ""):
        raise ValueError("Heard variant sighting has an invalid source or recognizer")
    sighting = {"source": source, "audio_sha256": audio_sha256, "request_hash": request_hash,
                "transcript_sha256": digest(transcript)}
    if not _valid_sighting(sighting):
        raise ValueError("Heard variant sighting is not bound to audio and transcript hashes")
    directory = Path(directory)
    with _locked(directory):
        record = load_local(directory)
        words = record["recognizers"].setdefault(recognizer, {})
        changed = False
        for block in blocks:
            key, heard_key = " ".join(block["expected"]), " ".join(block["heard"])
            variants = words.setdefault(key, {})
            if heard_key not in variants and len(variants) >= MAX_VARIANTS_PER_WORD:
                continue  # Bounded: a word heard too many different ways learns no more.
            variant = variants.setdefault(heard_key, {"expected": list(block["expected"]),
                                                      "heard": list(block["heard"]), "sightings": []})
            if (any(row["audio_sha256"] == audio_sha256 for row in variant["sightings"])
                    or len(variant["sightings"]) >= MAX_SIGHTINGS):
                continue
            variant["sightings"].append(dict(sighting))
            changed = True
        if not words:
            record["recognizers"].pop(recognizer)
        if changed:
            _validate(record, record["book_id"])
            write_json(directory / FILENAME, record)
    current = effective_variants(directory, recognizer)
    return [current.get((" ".join(block["expected"]), " ".join(block["heard"]))) for block in blocks]


# Verification overlay -------------------------------------------------------

def _transcript_ok(report):
    from .quality import text_confirmation_unavailable
    return (report.get("purpose", "speech") == "speech" and report.get("reasons") == ["text_mismatch"]
            and "history_echo" not in report and not text_confirmation_unavailable(report)
            and report.get("resolution_pending") is not True
            and isinstance(report.get("transcript"), str) and bool(report["transcript"].strip())
            and isinstance(report.get("expected_text"), str))


def apply(report, *, directory, verifier, request_hash, audio_sha256):
    """Learn from one verified take; accept a failed take whose words are all trusted variants.

    Returns ``(report, outcome)``: outcome is None (nothing applied),
    ``{"accepted": ...}`` or ``{"observed": ...}``. A passed report is never
    changed; a failed report changes only into an audited ``heard_variant``
    acceptance.
    """
    if directory is None or not isinstance(report, dict) or not isinstance(audio_sha256, str):
        return report, None
    recognizer = recognizer_key(verifier)
    if recognizer is None:
        return report, None
    if report.get("passed") is True:
        method = report.get("text_confirmation_method")
        source = ("review" if method in {"phonetic_primary", "phonetic_secondary"} else
                  "render" if method in {"jev_primary", "jev_secondary"} else None)
        transcript = report.get("verified_transcript")
        if source is None or not isinstance(transcript, str) or "history_echo" in report:
            return report, None
        blocks = eligible_blocks(report.get("expected_text"), transcript, require_plausible=source == "render")
        if blocks is None:
            return report, None
        record_sightings(directory, recognizer, blocks, source=source, audio_sha256=audio_sha256,
                         request_hash=request_hash, transcript=transcript)
        return report, {"observed": len(blocks), "source": source}
    if not _transcript_ok(report):
        return report, None
    transcript = report["transcript"]
    blocks = eligible_blocks(report["expected_text"], transcript, require_plausible=False)
    if blocks is None:
        return report, None
    # Plausibility gates new learning; an already trusted listened/reviewed
    # variant was approved by a person or reviewers for exactly these words.
    known = effective_variants(directory, recognizer)

    def approved(block):
        variant = known.get((" ".join(block["expected"]), " ".join(block["heard"])))
        return variant is not None and trust_basis(variant) in TRUSTING_SOURCES

    if not all(block["plausibility"] is not None or approved(block) for block in blocks):
        return report, None
    variants = record_sightings(directory, recognizer, blocks, source="render", audio_sha256=audio_sha256,
                                request_hash=request_hash, transcript=transcript)
    rows = []
    for block, variant in zip(blocks, variants):
        basis = trust_basis(variant) if variant is not None else None
        rows.append({"expected": block["expected"], "heard": block["heard"], "basis": basis,
                     "takes": len({row["audio_sha256"] for row in variant["sightings"]}) if variant else 0,
                     "sightings": deepcopy(variant["sightings"]) if variant else [],
                     "plausibility": block["plausibility"]})
    if any(row["basis"] is None for row in rows):
        observed = dict(report)
        observed["heard_variant_observations"] = [
            {key: row[key] for key in ("expected", "heard", "takes")} for row in rows]
        return observed, {"observed": len(rows), "rows": rows}
    accepted = {**deepcopy(report), "passed": True, "reasons": [], "previous_reasons": list(report["reasons"]),
                "text_confirmation_method": "heard_variant", "verified_transcript": transcript,
                "heard_variants": {"policy": deepcopy(POLICY), "recognizer": recognizer, "variants": rows}}
    return accepted, {"accepted": rows}


def learn_from_listening(directory, verifier, sentence, transcript, name_words, *, audio_sha256):
    """Record how the recognizer heard a person-approved test sentence.

    Every difference must lie on a use of the name (``name_words``: its
    normalized forms); otherwise the preview is not clean and nothing is kept.
    Returns the learned (expected, heard) pairs.
    """
    recognizer = recognizer_key(verifier)
    if recognizer is None:
        return []
    found = difference_blocks(sentence, transcript)
    if found is None:
        return []
    uncovered = [block for block in found["blocks"] if not _covered(block)]
    if any(_structural_reason(block) is not None or not set(block["expected"]) <= set(name_words)
           for block in uncovered):
        return []
    blocks = [{"expected": block["expected"], "heard": block["heard"]} for block in uncovered]
    if blocks:
        record_sightings(directory, recognizer, blocks, source="listened", audio_sha256=audio_sha256,
                         request_hash=None, transcript=transcript)
    return [(" ".join(block["expected"]), " ".join(block["heard"])) for block in blocks]
