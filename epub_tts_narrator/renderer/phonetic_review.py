"""Bounded pronunciation review; complete word coverage is enforced locally."""

from __future__ import annotations

from copy import deepcopy
import math

from .common import digest
from .speech_coverage import SPEECH_COVERAGE_VERSION, speech_coverage
from .verification_prompts import PHONETIC_PROMPT

_LIMITS = {"max_pairs": 4, "max_fraction": 0.2,
           "minimum_allowed_substitutions": 1, "max_words": 2000,
           "context_words": 3, "guard_version": 3}
_PROVEN_COVERAGE_OPERATIONS = frozenset({"contraction", "colloquial", "spelling_boundary", "letter_names"})


def _original_words(text, words):
    """Map normalized words back to original spellings without guessing offsets."""
    from .quality import _lexeme_matches, normalize_words

    result, normalized = [], []
    # Apostrophe replacement retains character offsets in the original string.
    for match in _lexeme_matches(text.replace("’", "'").replace("‘", "'")):
        source = text[match.start():match.end()]
        expanded = normalize_words(source)
        normalized.extend(expanded)
        result.extend([(source, match.start(), match.end())] * len(expanded))
    return result if normalized == words else None


def substitution_pairs(expected_text, transcript):
    """Return bound sound candidates eligible for review, never an approval.

    The complete transcript must pass the shared coverage gate first. Its
    ordered alignment identifies every candidate, including a single name
    transcribed as two/three words. A model cannot overlook a mismatch outside
    its local contexts: coverage failure prevents all review calls.
    """
    from .quality import normalize_words, word_spacing_equivalence

    if (not isinstance(expected_text, str) or not isinstance(transcript, str)
            or len(expected_text) > 100000 or len(transcript) > 100000):
        return None
    expected, heard = normalize_words(expected_text), normalize_words(transcript)
    if (not expected or not heard
            or max(len(expected), len(heard)) > _LIMITS["max_words"]):
        return None
    coverage = speech_coverage(expected_text, transcript)
    if not coverage["passed"]:
        return None
    changed = [change for change in coverage["alignment"]
               if change["operation"] not in _PROVEN_COVERAGE_OPERATIONS]
    if not changed:
        return None
    expected_original = _original_words(expected_text, expected)
    heard_original = _original_words(transcript, heard)
    if expected_original is None or heard_original is None:
        return None

    def context(text, originals, start, end):
        left = max(0, start - _LIMITS["context_words"])
        right = min(len(originals), end + _LIMITS["context_words"])
        return text[originals[left][1]:originals[right - 1][2]]

    pairs = []
    for change in changed:
        start, end = change["expected_start"], change["expected_end"]
        heard_start, heard_end = change["actual_start"], change["actual_end"]
        source_start, source_end = expected_original[start][1], expected_original[end - 1][2]
        transcript_start, transcript_end = heard_original[heard_start][1], heard_original[heard_end - 1][2]
        requested_span = expected_text[source_start:source_end]
        recognized_span = transcript[transcript_start:transcript_end]
        # Complete phonetic coverage also labels literal spacing variants as
        # split/join operations. Those need no model vote, including when a
        # separate uncertain name occurs elsewhere in the same passage.
        if word_spacing_equivalence(requested_span, recognized_span) is not None:
            continue
        pairs.append({
            "id": f"pair_{len(pairs) + 1}", "word_index": start,
            "expected_start": start, "expected_end": end,
            "actual_start": heard_start, "actual_end": heard_end,
            "expected_source_start": source_start, "expected_source_end": source_end,
            "recognized_source_start": transcript_start, "recognized_source_end": transcript_end,
            "expected_word": requested_span,
            "recognized_word": recognized_span,
            "coverage_operation": change["operation"], "coverage_rule": change.get("rule"),
            "requested_context": context(expected_text, expected_original, start, end),
            "recognized_context": context(transcript, heard_original, heard_start, heard_end),
        })
    allowed = max(_LIMITS["minimum_allowed_substitutions"], math.floor(_LIMITS["max_fraction"] * len(expected)))
    return pairs if pairs and len(pairs) <= min(allowed, _LIMITS["max_pairs"]) else None


def _valid_answers(value, pairs):
    if not isinstance(value, dict) or set(value) != {"pairs"}:
        return False
    answers = value["pairs"]
    if not isinstance(answers, list) or len(answers) != len(pairs):
        return False
    return all(
        isinstance(answer, dict) and set(answer) == {"id", "plausible", "reason"}
        and answer["id"] == pair["id"] and type(answer["plausible"]) is bool
        and isinstance(answer["reason"], str) and bool(answer["reason"].strip())
        and len(answer["reason"]) <= 400
        for answer, pair in zip(answers, pairs)
    )


def _decision(votes, pairs):
    return "accepted" if all(
        sum(vote["pairs"][index]["plausible"] for vote in votes) >= 2
        for index in range(len(pairs))
    ) else "rejected"


def _identity(provider):
    identity = {"name": "bounded_pronunciation_review", "version": 2,
                "provider": deepcopy(provider), "prompt_sha256": digest(PHONETIC_PROMPT),
                "coverage_version": SPEECH_COVERAGE_VERSION,
                "reviewers": 3, "required_votes": 2, "limits": deepcopy(_LIMITS)}
    identity["policy_sha256"] = digest(identity)
    return identity


def _valid_identity(identity):
    try:
        return (isinstance(identity, dict) and isinstance(identity.get("provider"), dict)
                and identity == _identity(identity["provider"]))
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False


class PronunciationReviewer:
    """Three separate sound reviews of spans from a complete coverage alignment."""

    def __init__(self, client):
        self._client = client
        self._identity = _identity(client.identity)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def review(self, expected_text, transcript):
        pairs = substitution_pairs(expected_text, transcript)
        record = {"identity": self.identity, "prompt": PHONETIC_PROMPT,
                  "expected_text": expected_text, "transcript": transcript,
                  "pairs": pairs, "votes": [], "decision": "ineligible"}
        if pairs is None:
            return record
        for reviewer in range(1, 4):
            try:
                if self._client.identity != self._identity["provider"]:
                    return dict(record, decision="unavailable", error_code="provider_changed")
                # No passage-level scores or previous answers are sent. Each
                # completion sees only the same small, ordered word-pair payload.
                response = self._client.complete(PHONETIC_PROMPT, {"pairs": deepcopy(pairs)})
            except Exception:
                return dict(record, decision="unavailable", error_code="request_failed")
            if not _valid_answers(response, pairs):
                return dict(record, decision="unavailable", error_code="invalid_response")
            record["votes"].append({"reviewer": reviewer, "pairs": deepcopy(response["pairs"])})
        record["decision"] = _decision(record["votes"], pairs)
        return record


def validate_review(record, expected_text, transcript, identity):
    """Accept only a complete, correctly bound audit with a majority for every pair."""
    if (not _valid_identity(identity) or not isinstance(record, dict) or set(record) != {
            "identity", "prompt", "expected_text", "transcript", "pairs", "votes", "decision"}
            or record["identity"] != identity or record["prompt"] != PHONETIC_PROMPT
            or record["expected_text"] != expected_text or record["transcript"] != transcript
            or record["decision"] != "accepted"):
        return False
    pairs = substitution_pairs(expected_text, transcript)
    if pairs is None or record["pairs"] != pairs:
        return False
    votes = record["votes"]
    if not isinstance(votes, list) or len(votes) != 3:
        return False
    for index, vote in enumerate(votes, 1):
        if (not isinstance(vote, dict) or set(vote) != {"reviewer", "pairs"}
                or type(vote["reviewer"]) is not int or vote["reviewer"] != index
                or not _valid_answers({"pairs": vote["pairs"]}, pairs)):
            return False
    return _decision(votes, pairs) == "accepted"
