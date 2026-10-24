"""Bounded model proposals for recurring speech errors, with separate sound votes.

This module never generates speech or changes book text.  Its input is the
immutable original passage and failed-take evidence; the caller owns retry
budgets and persistence.  A model proposes an action, local checks constrain it,
and three independent sound judgments must support every accepted substitution
or proposed pronunciation edit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re

from .common import digest
from .address_speech import address_spans, address_spelling_equivalent
from .letter_names import letter_name_rewrite_equivalent, make_letter_name_matcher
from .phonetic_match import _FUNCTION_WORDS, _PROTECTED
from .phonetic_review import _original_words
from .quality import _lexeme_matches, compare_words, normalize_words
from .speech_coverage import (
    SPEECH_COVERAGE_VERSION, _CONTRACTIONS, speech_coverage, stretched_word_equivalent,
)
from ..provider import OutputTruncatedError
from .verification_prompts import (_ADDRESS_PROMPT_SHA256, _ADDRESS_SOUND_SHA256, _FEEDBACK_PROMPT_SHA256,
                                   _FEEDBACK_SOUND_SHA256, _LEGACY_LIMITS, _LEGACY_PROMPT_SHA256,
                                   _LEGACY_SOUND_SHA256, _STUTTER_PROMPT_SHA256, _STUTTER_SOUND_SHA256,
                                   _TEXT_GATE_PROMPT_SHA256, _TEXT_GATE_SOUND_SHA256, PROMPT, SOUND_PROMPT)

_ADDRESS_LIMITS = {"max_failures": 30, "max_text_chars": 20000, "max_payload_chars": 200000,
                   "max_pairs": 4, "max_span_words": 3, "max_span_chars": 120,
                   "max_address_chars": 1024, "max_address_spelling_chars": 4096,
                   "reviewers": 3, "required_votes": 2, "guard_version": 1}
_LIMITS = {**_ADDRESS_LIMITS, "max_prior_decisions": 5, "max_feedback_chars": 30000}
_ACTIONS = frozenset({"accept", "rewrite", "plain_delivery", "retry", "split", "stop"})
# Verification reasons that describe measurable audio rather than recognized
# wording. They can motivate a pronunciation rewrite but never acceptance.
_ACOUSTIC_REASONS = frozenset({"silent_audio", "audio_too_quiet", "excessive_silence",
                               "internal_silence_stall", "excessive_clipping",
                               "audio_too_short", "audio_too_long"})
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SPELLING = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:[ \-‐‑][A-Za-z]+(?:['’][A-Za-z]+)*)*\Z")


def _identity(provider):
    identity = {"name": "recurring_speech_resolver", "version": 2,
                "provider": deepcopy(provider), "planner_prompt_sha256": digest(PROMPT),
                "sound_prompt_sha256": digest(SOUND_PROMPT),
                "coverage_version": SPEECH_COVERAGE_VERSION, "limits": deepcopy(_LIMITS)}
    identity["policy_sha256"] = digest(identity)
    return identity


def _valid_identity(identity):
    try:
        return (isinstance(identity, dict) and isinstance(identity.get("provider"), dict)
                and identity == _identity(identity["provider"]))
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def _legacy_rewrite_identity(identity):
    """Recognize only the previous committed policy for read-only reuse.

    Coverage 5 added address matching to coverage 4. Neither can affect the
    old rewrite format, which permits only plain one-to-three-word spellings.
    New decisions never use this policy or its payload shape.
    """
    if (not isinstance(identity, dict) or not isinstance(identity.get("provider"), dict)
            or type(identity.get("coverage_version")) is not int
            or identity["coverage_version"] not in {4, 5}):
        return False
    expected = {"name": "recurring_speech_resolver", "version": 1,
                "provider": deepcopy(identity["provider"]),
                "planner_prompt_sha256": _LEGACY_PROMPT_SHA256,
                "sound_prompt_sha256": _LEGACY_SOUND_SHA256,
                "coverage_version": identity["coverage_version"], "limits": deepcopy(_LEGACY_LIMITS)}
    expected["policy_sha256"] = digest(expected)
    return identity == expected


def _address_rewrite_identity(identity):
    """Read accepted audits from the fixed pre-feedback whole-address policy."""
    if not isinstance(identity, dict) or not isinstance(identity.get("provider"), dict):
        return False
    expected = {"name": "recurring_speech_resolver", "version": 1,
                "provider": deepcopy(identity["provider"]),
                "planner_prompt_sha256": _ADDRESS_PROMPT_SHA256,
                "sound_prompt_sha256": _ADDRESS_SOUND_SHA256,
                "coverage_version": 5, "limits": deepcopy(_ADDRESS_LIMITS)}
    expected["policy_sha256"] = digest(expected)
    return identity == expected


def _feedback_rewrite_identity(identity):
    """Reuse reviewed rewrites from the fixed pre-stutter feedback policy."""
    if (not isinstance(identity, dict) or not isinstance(identity.get("provider"), dict)
            or type(identity.get("coverage_version")) is not int
            or identity["coverage_version"] not in {5, 6, 7}):
        return False
    expected = {"name": "recurring_speech_resolver", "version": 1,
                "provider": deepcopy(identity["provider"]),
                "planner_prompt_sha256": _FEEDBACK_PROMPT_SHA256,
                "sound_prompt_sha256": _FEEDBACK_SOUND_SHA256,
                "coverage_version": identity["coverage_version"], "limits": deepcopy(_LIMITS)}
    expected["policy_sha256"] = digest(expected)
    return identity == expected


def _stutter_rewrite_identity(identity):
    """Reuse reviewed rewrites from the fixed pre-delivery-control policy."""
    if not isinstance(identity, dict) or not isinstance(identity.get("provider"), dict):
        return False
    expected = {"name": "recurring_speech_resolver", "version": 1,
                "provider": deepcopy(identity["provider"]),
                "planner_prompt_sha256": _STUTTER_PROMPT_SHA256,
                "sound_prompt_sha256": _STUTTER_SOUND_SHA256,
                "coverage_version": 7, "limits": deepcopy(_LIMITS)}
    expected["policy_sha256"] = digest(expected)
    return identity == expected


def _clean_text_rewrite_identity(identity):
    """Reuse accepted audits from the fixed pre-acoustic-rewrite policy.

    That policy allowed a rewrite only after a clean transcription mismatch;
    its payloads therefore never offered rewrite for acoustic failures, and
    they validate under that older rule, never the current one.
    """
    if (not isinstance(identity, dict) or not isinstance(identity.get("provider"), dict)
            or type(identity.get("coverage_version")) is not int):
        return False
    expected = {"name": "recurring_speech_resolver", "version": 1,
                "provider": deepcopy(identity["provider"]),
                "planner_prompt_sha256": _TEXT_GATE_PROMPT_SHA256,
                "sound_prompt_sha256": _TEXT_GATE_SOUND_SHA256,
                "coverage_version": identity["coverage_version"], "limits": deepcopy(_LIMITS)}
    expected["policy_sha256"] = digest(expected)
    return identity == expected


def _prior_decisions(value):
    if value is None:
        value = []
    if (not isinstance(value, list) or len(value) > _LIMITS["max_prior_decisions"]
            or len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > _LIMITS["max_feedback_chars"]):
        raise ValueError("Prior recovery decisions exceed their feedback limit")
    for row in value:
        if (not isinstance(row, dict)
                or set(row) != {"attempt", "proposal", "decision", "error_code", "validation_error", "sound_reviews"}
                or type(row["attempt"]) is not int or row["attempt"] < 1
                or row["decision"] not in {"resolved", "blocked", "unavailable"}
                or any(row[key] is not None and (not isinstance(row[key], str) or len(row[key]) > limit)
                       for key, limit in (("error_code", 100), ("validation_error", 800)))
                or not isinstance(row["sound_reviews"], list) or len(row["sound_reviews"]) > 3
                or any(not isinstance(review, dict) or set(review) != {"reviewer", "response"}
                       or type(review["reviewer"]) is not int or not 1 <= review["reviewer"] <= 3
                       for review in row["sound_reviews"])):
            raise ValueError("Invalid prior recovery decision feedback")
    return deepcopy(value)


def _payload(expected_text, failures, can_rewrite, can_split, *, legacy_rewrite=False,
             address_rewrite=False, previous_decisions=None, can_plain_delivery=False,
             acoustic_rewrite=False):
    if (not isinstance(expected_text, str) or not expected_text.strip()
            or len(expected_text) > _LIMITS["max_text_chars"]
            or type(can_rewrite) is not bool or type(can_split) is not bool
            or type(can_plain_delivery) is not bool
            or not isinstance(failures, list) or not 1 <= len(failures) <= _LIMITS["max_failures"]):
        raise ValueError("Invalid speech resolution input")
    # JSON validation also rejects non-finite numbers, unexpected objects, and
    # pathological evidence sizes. Retain the complete supplied evidence in the
    # audit rather than silently dropping metrics or source recognizer details.
    serialized = json.dumps(failures, ensure_ascii=False, allow_nan=False)
    if len(serialized) + len(expected_text) > _LIMITS["max_payload_chars"]:
        raise ValueError("Speech resolution evidence exceeds its size limit")
    attempts = set()
    for failure in failures:
        if (not isinstance(failure, dict)
                or not {"attempt", "audio_sha256", "transcript", "reasons"} <= set(failure)
                or set(failure) - {"attempt", "audio_sha256", "transcript", "reasons",
                                   "secondary_transcript", "metrics", "tts_text", "generation"}
                or type(failure["attempt"]) is not int or failure["attempt"] < 1
                or failure["attempt"] in attempts
                or not isinstance(failure["audio_sha256"], str)
                or not _HEX.fullmatch(failure["audio_sha256"])
                or not isinstance(failure["transcript"], str)
                or len(failure["transcript"]) > _LIMITS["max_text_chars"]
                or not isinstance(failure["reasons"], list) or not failure["reasons"]
                or len(failure["reasons"]) > 16
                or any(not isinstance(reason, str) or not reason or len(reason) > 1000
                       for reason in failure["reasons"])
                or ("metrics" in failure and not isinstance(failure["metrics"], dict))
                or ("generation" in failure and (not isinstance(failure["generation"], dict)
                    or set(failure["generation"]) - {"synthesis_text", "direction", "applied_control"}
                    or any(not isinstance(value, str) or len(value) > _LIMITS["max_text_chars"]
                           for value in failure["generation"].values())))
                or ("tts_text" in failure and (not isinstance(failure["tts_text"], str)
                    or not failure["tts_text"].strip()
                    or len(failure["tts_text"]) > _LIMITS["max_text_chars"]))
                or (failure.get("secondary_transcript") is not None
                    and (not isinstance(failure["secondary_transcript"], str)
                         or len(failure["secondary_transcript"]) > _LIMITS["max_text_chars"]))):
            raise ValueError("Invalid failed-take evidence")
        attempts.add(failure["attempt"])
    permitted = ["retry", "stop"]
    if can_split:
        permitted.append("split")
    if can_plain_delivery:
        permitted.append("plain_delivery")
    # Acceptance requires a genuine transcription mismatch of the latest take,
    # never an unavailable recognizer/provider and never failed acoustic
    # checks. A pronunciation rewrite may also respond to a persistent spelling
    # problem when the latest take additionally failed an acoustic check, but
    # that rewrite still needs its sound jury and a fully passing new take.
    latest = failures[-1]["reasons"]
    if latest == ["text_mismatch"]:
        permitted.append("accept")
    if can_rewrite and (latest == ["text_mismatch"]
                        or (acoustic_rewrite
                            and any(reason in _ACOUSTIC_REASONS for reason in latest))):
        permitted.append("rewrite")
    source_words = [{"start": match.start(), "end": match.end(),
                     "source": expected_text[match.start():match.end()]}
                    for match in _lexeme_matches(expected_text.replace("’", "'").replace("‘", "'"))]
    if len(source_words) > 2000:
        raise ValueError("Speech resolution passage has too many source words")
    result = {"requested_text": expected_text, "source_words": source_words,
              "failures": deepcopy(failures), "permitted_actions": permitted}
    if not legacy_rewrite:
        result["source_addresses"] = address_spans(expected_text)
    if not legacy_rewrite and not address_rewrite:
        result["previous_decisions"] = _prior_decisions(previous_decisions)
    elif previous_decisions:
        raise ValueError("Historical recovery policies cannot include new feedback")
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > _LIMITS["max_payload_chars"]:
        raise ValueError("Speech resolution payload exceeds its size limit")
    return result


def _reason(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 800


def _safe_span(value):
    if (not isinstance(value, str) or len(value) > _LIMITS["max_span_chars"]
            or not _SPELLING.fullmatch(value)):
        return False
    lexemes = list(_lexeme_matches(value.replace("’", "'")))
    if not 1 <= len(lexemes) <= _LIMITS["max_span_words"]:
        return False
    for token in lexemes:
        raw, word = token.group(), token.group().casefold()
        if (len(word.replace("'", "")) < 2 or (len(raw) >= 2 and raw.isupper())
                or word in _CONTRACTIONS or "n't" in word
                or word in _FUNCTION_WORDS or word.replace("'", "") in _PROTECTED
                or any(stretched_word_equivalent(word, protected)
                       for protected in _FUNCTION_WORDS | _PROTECTED)):
            return False
    return True


def _stretched_fragment(source, replacement):
    """Rewrite-only shortening of a consonant run; never direct audio acceptance."""
    return (isinstance(source, str) and isinstance(replacement, str)
            and 3 <= len(source) <= _LIMITS["max_span_chars"] and source.isascii()
            and not source.isupper() and re.fullmatch(r"[bcdfghjklmnpqrstvwxz]", replacement) is not None
            and source.casefold() == replacement * len(source))


def _stretched_span(source, replacement):
    """Allow spelling-only restoration of function words without weakening them.

    Every source token must survive in order. A different function word,
    negation, number, added word or missing word cannot use this exception.
    """
    if (not isinstance(source, str) or not isinstance(replacement, str)
            or max(len(source), len(replacement)) > _LIMITS["max_span_chars"]
            or not _SPELLING.fullmatch(source) or not _SPELLING.fullmatch(replacement)):
        return False
    before, after = normalize_words(source), normalize_words(replacement)
    return (1 <= len(before) == len(after) <= _LIMITS["max_span_words"]
            and before != after and all(left == right or stretched_word_equivalent(left, right)
                                       for left, right in zip(before, after)))


def _stutter_span(source, replacement):
    """Match only incomplete repeated prefixes while retaining the whole word.

    Hyphens must explicitly mark one or two identical short prefixes of the
    final word. A complete repeated word, initialism, compound, missing final
    word, or new word cannot use this exception. The independent sound jury
    still decides whether omitting/respelling that onset is plausible.
    """
    if (not isinstance(source, str) or not isinstance(replacement, str)
            or max(len(source), len(replacement)) > _LIMITS["max_span_chars"]
            or not _SPELLING.fullmatch(source) or not _SPELLING.fullmatch(replacement)):
        return False
    parts = re.split(r"[-‐‑]", source)
    if not 2 <= len(parts) <= _LIMITS["max_span_words"]:
        return False
    prefix, word = parts[0].casefold(), parts[-1].replace("’", "'").casefold()
    if (not re.fullmatch(r"[a-z]{1,3}", prefix)
            or (len(prefix) > 1 and any(char in "aeiouy" for char in prefix))
            or not re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", parts[-1])
            or len(prefix) >= len(word)
            or not word.startswith(prefix) or parts[-1].isupper()
            or any(part.casefold() != prefix for part in parts[:-1])):
        return False
    after = normalize_words(replacement)
    allowed_prefixes = {prefix}
    if not any(char in "aeiou" for char in prefix):
        allowed_prefixes.add(prefix + "uh")
    return (1 <= len(after) <= len(parts) and after[-1] == word
            and all(fragment in allowed_prefixes for fragment in after[:-1]))


def _stutter_accept_pairs(expected_text, transcript, left, right, expected, actual):
    """Cover the whole passage with exact words and narrowly marked stutters.

    Do not combine this exception with fuzzy matching elsewhere: every other
    word must survive exactly, and no added/deleted complete word is allowed.
    """
    if not any(char in expected_text for char in "-‐‑"):
        return None
    pending = [{} for _ in range(len(left) + 1)]
    pending[0][0] = []
    for start in range(len(left)):
        for heard, pairs in pending[start].items():
            if heard >= len(right):
                continue
            if expected[start] == actual[heard]:
                pending[start + 1].setdefault(heard + 1, pairs)
            if len(pairs) >= _LIMITS["max_pairs"]:
                continue
            for count in range(2, min(_LIMITS["max_span_words"], len(left) - start) + 1):
                source = expected_text[left[start][1]:left[start + count - 1][2]]
                for heard_count in range(1, min(count, len(right) - heard) + 1):
                    target = transcript[right[heard][1]:right[heard + heard_count - 1][2]]
                    if _stutter_span(source, target):
                        pair = _pair(expected_text, transcript, left, right, start, start + count,
                                     heard, heard + heard_count, len(pairs) + 1)
                        pending[start + count].setdefault(heard + heard_count, pairs + [pair])
    return pending[-1].get(len(right))


def _pair(expected_text, transcript, left, right, start, end, actual_start, actual_end, number):
    source_start, source_end = left[start][1], left[end - 1][2]
    heard_start, heard_end = right[actual_start][1], right[actual_end - 1][2]
    def context(text, words, a, b):
        return text[words[max(0, a - 3)][1]:words[min(len(words), b + 3) - 1][2]]
    return {"id": f"pair_{number}", "expected_word": expected_text[source_start:source_end],
            "recognized_word": transcript[heard_start:heard_end],
            "expected_start": source_start, "expected_end": source_end,
            "recognized_start": heard_start, "recognized_end": heard_end,
            "requested_context": context(expected_text, left, start, end),
            "recognized_context": context(transcript, right, actual_start, actual_end)}


def _accept_pairs(expected_text, transcript, *, validation_errors=None):
    def reject(reason):
        if validation_errors is not None:
            validation_errors.append(reason)
        return None

    expected, actual = normalize_words(expected_text), normalize_words(transcript)
    if not expected or not actual or max(len(expected), len(actual)) > 2000:
        return reject("Acceptance requires nonempty transcripts within the 2,000-word comparison limit.")
    left, right = _original_words(expected_text, expected), _original_words(transcript, actual)
    if left is None or right is None:
        return reject("Acceptance could not map the normalized transcript words back to their source text.")
    stutter_pairs = _stutter_accept_pairs(expected_text, transcript, left, right, expected, actual)
    if stutter_pairs:
        return stutter_pairs
    letter_matches = make_letter_name_matcher(
        expected_text, expected, [(start, end, word.casefold()) for word, start, end in left],
        transcript, actual, [(start, end, word.casefold()) for word, start, end in right],
        protected=_PROTECTED)
    coverage = speech_coverage(expected_text, transcript)
    if coverage["passed"]:
        changes = coverage["alignment"]
    else:
        if len(expected) != len(actual):
            return reject("Acceptance failed complete word coverage: the transcript has a different word count, "
                          "and no supported spelling or pronunciation alignment accounts for every word.")
        alignment = compare_words(expected_text, transcript)["alignment"]
        # Equal token counts alone can conceal one omitted word plus an added
        # one. A complete minimal alignment must consist only of substitutions.
        if any(row["operation"] not in {"equal", "substitute"} for row in alignment):
            return reject("Acceptance failed complete word coverage: the alignment contains missing, extra, or moved words.")
        changes = [row for row in alignment if row["operation"] == "substitute"]
        if (not changes or {word for row in changes for word in row["expected"]}
                & {word for row in changes for word in row["actual"]}):
            return reject("Acceptance found reordered words rather than an isolated pronunciation variation.")
    if len(changes) > _LIMITS["max_pairs"]:
        return reject("Acceptance needs more than four separate pronunciation comparisons.")
    pairs = []
    for change in changes:
        start, end = change["expected_start"], change["expected_end"]
        heard_start, heard_end = change["actual_start"], change["actual_end"]
        if not (1 <= end - start <= _LIMITS["max_span_words"]
                and 1 <= heard_end - heard_start <= _LIMITS["max_span_words"]):
            return reject("Acceptance contains a pronunciation span longer than three words.")
        if not coverage["passed"] and letter_matches.blocks_fuzzy(
                start, heard_start, end - start, heard_end - heard_start):
            return reject("Acceptance would change a protected spoken letter or initialism.")
        pair = _pair(expected_text, transcript, left, right, start, end,
                     heard_start, heard_end, len(pairs) + 1)
        if not coverage["passed"] and not all(_safe_span(pair[key]) for key in (
                "expected_word", "recognized_word")):
            return reject("Acceptance would change a protected number, negation, function word, "
                          "or a span outside the permitted pronunciation format.")
        pairs.append(pair)
    return pairs


def _proposal(proposal, payload, *, validation_errors=None, legacy_rewrite=False):
    def reject(reason):
        if validation_errors is not None:
            validation_errors.append(reason)
        return None

    if (not isinstance(proposal, dict) or set(proposal) != {"action", "edits", "reason"}
            or not isinstance(proposal["action"], str) or proposal["action"] not in _ACTIONS
            or not isinstance(proposal["edits"], list) or not _reason(proposal["reason"])):
        return reject("The model response must contain only a supported action, an edits array, and a brief reason.")
    if proposal["action"] not in payload["permitted_actions"]:
        return reject("The proposed action is not permitted for this take's verification failures or recovery capabilities.")
    action, expected = proposal["action"], payload["requested_text"]
    result = deepcopy(proposal)
    if action != "rewrite":
        if proposal["edits"]:
            return reject("Only a rewrite action may include text edits.")
        if action == "accept":
            transcript = payload["failures"][-1]["transcript"]
            pairs = _accept_pairs(expected, transcript, validation_errors=validation_errors)
            if pairs is None:
                return None
            result["acceptance_transcript"] = transcript
        else:
            pairs = []
        return dict(result, validated_pairs=pairs)
    edits = proposal["edits"]
    if not 1 <= len(edits) <= _LIMITS["max_pairs"]:
        return reject("A pronunciation rewrite requires between one and four edits.")
    source_lexemes = list(_lexeme_matches(expected.replace("’", "'").replace("‘", "'")))
    boundaries_start = {match.start() for match in source_lexemes}
    boundaries_end = {match.end() for match in source_lexemes}
    addresses = [] if legacy_rewrite else payload["source_addresses"]
    cursor, pieces, pairs = 0, [], []
    for edit in edits:
        if (not isinstance(edit, dict) or set(edit) != {"start", "end", "source", "replacement"}
                or type(edit["start"]) is not int or type(edit["end"]) is not int
                or not cursor <= edit["start"] < edit["end"] <= len(expected)
                or edit["source"] != expected[edit["start"]:edit["end"]]):
            return reject("A rewrite edit has invalid fields or offsets: edits must be ordered, nonoverlapping, "
                          "and copy complete words exactly from the original text.")
        address = next((row for row in addresses
                        if row["start"] == edit["start"] and row["end"] == edit["end"]), None)
        stutter = _stutter_span(edit["source"], edit["replacement"])
        fragment = _stretched_fragment(edit["source"], edit["replacement"])
        letter_name = (isinstance(edit["replacement"], str)
                       and len(edit["replacement"]) <= _LIMITS["max_span_chars"]
                       and letter_name_rewrite_equivalent(edit["source"], edit["replacement"]))
        if address is not None:
            if (len(edit["source"]) > _LIMITS["max_address_chars"]
                    or not isinstance(edit["replacement"], str)
                    or not 1 <= len(edit["replacement"]) <= _LIMITS["max_address_spelling_chars"]
                    or any(ord(char) < 32 or ord(char) == 127 for char in edit["replacement"])
                    or not address_spelling_equivalent(edit["source"], edit["replacement"])):
                return reject("An address rewrite must spell the complete original URL or email in order, "
                              "preserving every component and code character without extra words or markup.")
        elif edit["start"] not in boundaries_start or edit["end"] not in boundaries_end:
            return reject("A rewrite edit has invalid offsets: copy complete words or one complete source_addresses entry.")
        elif (any(edit["start"] < row["end"] and row["start"] < edit["end"] for row in addresses)
                or (not stutter and not fragment and not letter_name
                    and not _stretched_span(edit["source"], edit["replacement"])
                    and (not _safe_span(edit["source"]) or not _safe_span(edit["replacement"])))):
            return reject("A rewrite must use one to three plain pronunciation words or one complete source_addresses "
                          "entry; it cannot change numbers, negation, function words, initialisms, or include instructions or markup.")
        if normalize_words(edit["source"]) == normalize_words(edit["replacement"]):
            return reject("The proposed rewrite does not change the normalized TTS wording.")
        # Do not let a multiword spelling consume a retained neighboring word;
        # that would disguise a deletion or insertion as a name respelling.
        before, after = normalize_words(edit["source"]), normalize_words(edit["replacement"])
        if (address is None and set(before) & set(after)
                and not stutter and not _stretched_span(edit["source"], edit["replacement"])):
            return reject("A rewrite overlaps retained words and could conceal an added or missing word.")
        span_coverage = speech_coverage(edit["source"], edit["replacement"]) if address is None else None
        if address is None and not stutter and not fragment and not letter_name and ((len(before) != len(after) and not span_coverage["passed"])
                or (not span_coverage["passed"]
                    and _accept_pairs(edit["source"], edit["replacement"]) is None)):
            return reject("The pronunciation rewrite failed complete word coverage for the edited span.")
        selected = [index for index, match in enumerate(source_lexemes)
                    if edit["start"] <= match.start() and match.end() <= edit["end"]]
        context = expected[source_lexemes[max(0, selected[0] - 3)].start():
                           source_lexemes[min(len(source_lexemes) - 1, selected[-1] + 3)].end()]
        pairs.append({"id": f"pair_{len(pairs) + 1}", "expected_word": edit["source"],
                      "recognized_word": edit["replacement"], "expected_start": edit["start"],
                      "expected_end": edit["end"], "requested_context": context})
        pieces.extend((expected[cursor:edit["start"]], edit["replacement"]))
        cursor = edit["end"]
    pieces.append(expected[cursor:])
    tts_text = "".join(pieces)
    if any(failure.get("tts_text") == tts_text for failure in payload["failures"]):
        return reject("The proposed TTS spelling was already used in a failed take.")
    return dict(result, tts_text=tts_text, validated_pairs=pairs)


def _reconsiderable_candidate(payload):
    previous = payload.get("previous_decisions", [])
    for row in reversed(previous):
        raw = row.get("proposal")
        if (row.get("decision") != "blocked" or row.get("error_code") != "invalid_proposal"
                or row.get("sound_reviews") or not isinstance(raw, dict) or raw.get("action") != "rewrite"):
            continue
        if any(isinstance(other.get("proposal"), dict)
               and other["proposal"].get("edits") == raw.get("edits")
               and (other.get("decision") == "resolved"
                    or other.get("error_code") == "sound_review_rejected" and other.get("sound_reviews"))
               for other in previous):
            continue
        proposal = _proposal(raw, payload)
        if proposal is not None:
            return row, proposal
    return None


def _valid_votes(response, pairs):
    if not isinstance(response, dict) or set(response) != {"pairs"}:
        return False
    answers = response["pairs"]
    return (isinstance(answers, list) and len(answers) == len(pairs)
            and all(isinstance(answer, dict) and set(answer) == {"id", "plausible", "reason"}
                    and answer["id"] == pair["id"] and type(answer["plausible"]) is bool
                    and isinstance(answer["reason"], str) and bool(answer["reason"].strip())
                    and len(answer["reason"]) <= 400 for answer, pair in zip(answers, pairs)))


def _approved(reviews, pairs):
    return all(sum(review["response"]["pairs"][index]["plausible"] for review in reviews)
               >= _LIMITS["required_votes"] for index in range(len(pairs)))


class SpeechResolver:
    """One bounded proposal and, when needed, three separate sound reviews."""

    def __init__(self, client, *, secrets=()):
        self._client = client
        self._identity = _identity(client.identity)
        self._secrets = tuple(secrets)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def resolve(self, expected_text, failures, *, can_rewrite=True, can_split=True,
                previous_decisions=None, can_plain_delivery=False):
        audit = {"identity": self.identity, "prompt": PROMPT, "payload": None,
                 "proposal": None, "reviews": [], "decision": "blocked"}
        try:
            audit["payload"] = _payload(expected_text, failures, can_rewrite, can_split,
                                        previous_decisions=previous_decisions,
                                        can_plain_delivery=can_plain_delivery,
                                        acoustic_rewrite=True)
        except (ValueError, TypeError, OverflowError, RecursionError):
            return dict(audit, error_code="invalid_request")
        try:
            if self._client.identity != self._identity["provider"]:
                return dict(audit, decision="unavailable", error_code="provider_changed")
            audit["proposal"] = self._client.complete(PROMPT, deepcopy(audit["payload"]))
        except Exception as exc:
            from .diagnostics import safe_error_text
            return dict(audit, decision="unavailable", error_code=("output_truncated"
                        if isinstance(exc, OutputTruncatedError) else "request_failed"),
                        error=safe_error_text(exc, secrets=self._secrets))
        validation_errors = []
        try:
            proposal = _proposal(audit["proposal"], audit["payload"], validation_errors=validation_errors)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
            proposal = None
        if proposal is None:
            return dict(audit, error_code="invalid_proposal", validation_error=(validation_errors[0]
                        if validation_errors else "The model proposal contains invalid or unsupported data."))
        return self._review_proposal(audit, proposal)

    def reconsider(self, expected_text, failures, *, can_rewrite=True, can_split=True,
                   previous_decisions=None, can_plain_delivery=False):
        """Review a prior locally rejected candidate without a new planner call."""
        try:
            payload = _payload(expected_text, failures, can_rewrite, can_split,
                               previous_decisions=previous_decisions,
                               can_plain_delivery=can_plain_delivery, acoustic_rewrite=True)
        except (ValueError, TypeError, OverflowError, RecursionError):
            return None
        candidate = _reconsiderable_candidate(payload)
        if candidate is None:
            return None
        row, proposal = candidate
        audit = {"identity": self.identity, "prompt": PROMPT, "payload": payload,
                 "proposal": deepcopy(row["proposal"]), "reviews": [], "decision": "blocked",
                 "reconsidered": {"attempt": row["attempt"], "error_code": "invalid_proposal"}}
        return self._review_proposal(audit, proposal)

    def _review_proposal(self, audit, proposal):
        pairs = proposal["validated_pairs"]
        if proposal["action"] in {"accept", "rewrite"}:
            for reviewer in range(1, _LIMITS["reviewers"] + 1):
                payload = {"pairs": deepcopy(pairs)}
                try:
                    if self._client.identity != self._identity["provider"]:
                        return dict(audit, decision="unavailable", error_code="provider_changed")
                    response = self._client.complete(SOUND_PROMPT, deepcopy(payload))
                except Exception as exc:
                    from .diagnostics import safe_error_text
                    return dict(audit, decision="unavailable", error_code=("output_truncated"
                                if isinstance(exc, OutputTruncatedError) else "review_failed"),
                                error=safe_error_text(exc, secrets=self._secrets))
                audit["reviews"].append({"reviewer": reviewer, "prompt": SOUND_PROMPT,
                                         "payload": payload, "response": deepcopy(response)})
                if not _valid_votes(response, pairs):
                    return dict(audit, decision="unavailable" if "reconsidered" in audit else "blocked",
                                error_code="invalid_review")
            if not _approved(audit["reviews"], pairs):
                return dict(audit, error_code="sound_review_rejected")
        audit["decision"] = "resolved"
        return audit


def validate_resolution(audit, expected_text, failures, identity, *, can_rewrite=True, can_split=True,
                        previous_decisions=None, can_plain_delivery=None):
    """Recompute all trusted checks before a persisted or fresh plan is used."""
    try:
        legacy = _legacy_rewrite_identity(identity)
        address = _address_rewrite_identity(identity)
        feedback = _feedback_rewrite_identity(identity)
        stutter = _stutter_rewrite_identity(identity)
        clean_text = _clean_text_rewrite_identity(identity)
        historical = legacy or address or feedback or stutter or clean_text
        # Historical rewrite markers do not know the caller's current engine
        # capabilities. Fresh decisions must pass an explicit boolean; omission
        # only replays the capability bound into an existing audit's payload.
        if can_plain_delivery is None:
            payload = audit.get("payload") if isinstance(audit, dict) else None
            can_plain_delivery = (isinstance(payload, dict)
                                  and isinstance(payload.get("permitted_actions"), list)
                                  and "plain_delivery" in payload["permitted_actions"])
        if previous_decisions is None and not legacy and not address and isinstance(audit, dict):
            payload = audit.get("payload")
            previous_decisions = payload.get("previous_decisions") if isinstance(payload, dict) else None
        planner_hash = (_LEGACY_PROMPT_SHA256 if legacy else _ADDRESS_PROMPT_SHA256 if address
                        else _FEEDBACK_PROMPT_SHA256 if feedback else _STUTTER_PROMPT_SHA256 if stutter
                        else _TEXT_GATE_PROMPT_SHA256 if clean_text else digest(PROMPT))
        sound_hash = (_LEGACY_SOUND_SHA256 if legacy else _ADDRESS_SOUND_SHA256 if address
                      else _FEEDBACK_SOUND_SHA256 if feedback else _STUTTER_SOUND_SHA256 if stutter
                      else _TEXT_GATE_SOUND_SHA256 if clean_text else digest(SOUND_PROMPT))
        if (not (_valid_identity(identity) or historical) or not isinstance(audit, dict)
                or set(audit) - {"reconsidered"} != {"identity", "prompt", "payload", "proposal", "reviews", "decision"}
                or audit["identity"] != identity
                or digest(audit["prompt"]) != planner_hash
                or audit["decision"] != "resolved"
                or (historical and not clean_text and (can_plain_delivery or not isinstance(audit["proposal"], dict)
                                or audit["proposal"].get("action") != "rewrite"))
                or audit["payload"] != _payload(expected_text, failures, can_rewrite, can_split,
                                                legacy_rewrite=legacy, address_rewrite=address,
                                                previous_decisions=previous_decisions,
                                                can_plain_delivery=can_plain_delivery,
                                                acoustic_rewrite=not historical)):
            return None
        if "reconsidered" in audit:
            candidate = _reconsiderable_candidate(audit["payload"]) if not historical else None
            if (candidate is None or not isinstance(audit["reconsidered"], dict)
                    or type(audit["reconsidered"].get("attempt")) is not int
                    or audit["reconsidered"] != {
                    "attempt": candidate[0]["attempt"], "error_code": "invalid_proposal"}
                    or audit["proposal"] != candidate[0]["proposal"]):
                return None
        proposal = _proposal(audit["proposal"], audit["payload"], legacy_rewrite=legacy)
        if proposal is None or not isinstance(audit["reviews"], list):
            return None
        pairs, reviews = proposal["validated_pairs"], audit["reviews"]
        if proposal["action"] not in {"accept", "rewrite"}:
            return proposal if not reviews else None
        if len(reviews) != _LIMITS["reviewers"]:
            return None
        for number, review in enumerate(reviews, 1):
            if (not isinstance(review, dict) or set(review) != {"reviewer", "prompt", "payload", "response"}
                    or type(review["reviewer"]) is not int or review["reviewer"] != number
                    or digest(review["prompt"]) != sound_hash
                    or review["payload"] != {"pairs": pairs}
                    or not _valid_votes(review["response"], pairs)):
                return None
        return proposal if _approved(reviews, pairs) else None
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return None
