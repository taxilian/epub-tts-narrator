"""Choose an audition script from a character's already-attributed book text.

Selection never writes the analyzer bundle. Joined statements retain separate
source ranges; creative casting text is explicitly marked as non-source material.
Offsets use Python/Unicode code points, matching the analyzer and render plan.
"""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

from epub_tts_narrator.spoken_text import build_spoken, effective_delivery, spoken_source_ranges


SELECTION_VERSION = 4
_MAX_CHARS = 650
_MIN_WORDS = 25
_TARGET_MIN_WORDS = 35
_TARGET_WORDS = 45
_MAX_WORDS = 60
_MIN_EXCERPT_WORDS = 4
_WORDS = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)
_SENTENCE_END = re.compile(r"""[.!?]+[”’"']*(?=\s|$)""")
_EXTREME = re.compile(
    r"\b(?:shout\w*|scream\w*|yell\w*|shriek\w*|sob\w*|weep\w*|"
    r"wail\w*|hyster\w*|furious\w*|fury|rage|enraged|panting|"
    r"gasp\w*|roar\w*|screech\w*|nonverbal|non-verbal|"
    r"cut off|interrupted|mid-word)\b", re.IGNORECASE,
)
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e",
}


class MissingSampleScript(ValueError):
    def __init__(self, voice_id):
        self.voice_id = voice_id
        super().__init__(f"Voice {voice_id} needs a character-specific audition script: no suitable source "
                         f"selection reaches {_MIN_WORDS} words. Prepare sample scripts for this book before casting.")


def sample_word_count(text):
    return len(_WORDS.findall(text))


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _speech_wrapper_end(text, start, end):
    """Find a matched speech delimiter without treating contractions as quotes."""
    if end - start < 2:
        return None
    opening = text[start]
    closing = {'"': '"', "“": "”", "'": "'", "‘": "’"}.get(opening)
    if closing is None:
        return None
    depth = 1
    for offset in range(start + 1, end):
        char = text[offset]
        # Word-internal apostrophes are lexical content, not quote delimiters.
        if (char in {"'", "’"} and offset + 1 < end
                and text[offset - 1].isalnum() and text[offset + 1].isalnum()):
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return offset
        elif char == opening:
            depth += 1
    return None


def _speech_bounds(text):
    """Remove one balanced speech wrapper, retaining an exact source slice.

    The caller has already established that this is speech, rather than quoted
    wording in narration. Only a pair surrounding the entire utterance counts;
    separate quotations and apostrophes must not become stray quote marks.
    """
    start, end = _trim(text, 0, len(text))
    if _speech_wrapper_end(text, start, end) == end - 1:
        return _trim(text, start + 1, end - 1)
    return start, end


def strip_speech_wrapper(text):
    """Strip a balanced outer speech pair, preserving embedded quoted wording.

    This is for audition speech only. Source-backed selection uses the same
    bounds directly so its original block and spoken offsets remain accurate.
    """
    start, end = _speech_bounds(text)
    return text[start:end]


def has_repeated_speech_wrappers(text):
    """Whether all prose consists of separately quoted utterances.

    Ordinary prose containing quoted wording is deliberately left alone. This
    check lets creative generation repair formatting without deleting meaningful
    punctuation or rewriting a saved source transcript.
    """
    cursor, end = _trim(text, 0, len(text))
    count = 0
    while cursor < end:
        closing = _speech_wrapper_end(text, cursor, end)
        if closing is None or not text[cursor + 1:closing].strip():
            return False
        count += 1
        cursor, _ = _trim(text, closing + 1, end)
    return count > 1


def _complete(text):
    return bool(re.search(r"""[.!?][”’"']*$""", text))


def _spans(text):
    """Whole short utterances or consecutive complete sentences, never word cuts."""
    end_points = []
    for match in _SENTENCE_END.finditer(text):
        before = text[:match.start()]
        token = re.search(r"([\w.]+)$", before)
        # Do not manufacture sentence starts after common titles or initials.
        if text[match.start()] == "." and token:
            word = token.group(1)
            if word.casefold() in _ABBREVIATIONS or len(word) == 1:
                continue
        end_points.append(match.end())
    if len(text) <= _MAX_CHARS:
        start, end = _trim(text, 0, len(text))
        if start < end:
            yield start, end
    # A long utterance can be excerpted only at genuine sentence boundaries.
    starts = [0, *end_points[:-1]]
    seen = set()
    for start in starts:
        for end in end_points:
            if end <= start:
                continue
            span = _trim(text, start, end)
            if span[1] - span[0] > _MAX_CHARS:
                break
            if span[0] < span[1] and span not in seen:
                seen.add(span)
                yield span


def _role_priority(row, role):
    kind = row.get("kind")
    if role.get("kind") == "dialogue":
        if kind == "dialogue":
            return 0
        if kind in {"thought", "letter"}:
            return 1
    if role.get("kind") == "narration" and kind in {
        "narration", "letter", "other",
    }:
        return 2
    return None


def _eligible_performers(row):
    if row.get("simultaneous_speakers"):
        return row["simultaneous_speakers"]
    return [{"voice_id": row.get("voice_id"), "role_id": row.get("role_id")}]


def _safe_text(text):
    if any(char in text for char in "[]<>"):
        return False
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
           and char not in "\n\r\t" for char in text):
        return False
    if "://" in text or re.search(r"\S+@\S+", text):
        return False
    letters = [char for char in text if char.isalpha()]
    if letters and sum(char.isupper() for char in letters) / len(letters) > .55:
        return False
    if re.search(r"([A-Za-z])\1{2,}", text):
        return False
    return not text.rstrip("”’\"'").endswith(("—", "–", "-"))


def _candidate_data(bundle, voice_id, canonical_id):
    book = bundle["source"]["book"]
    voice_ids = {voice["id"] for voice in bundle["cast"]["voices"]}
    if voice_id not in voice_ids:
        raise ValueError("Sample voice is not in the bundle cast")
    canonical = {item: canonical_id(item) for item in voice_ids}
    target = canonical[voice_id]
    source_base = {
        "selection_version": SELECTION_VERSION,
        "canonical_voice_id": target,
        "source_fingerprint": bundle["manifest"]["source_fingerprint"],
        "book_id": book["id"], "book_title": book.get("title", ""),
    }
    roles = {role["id"]: role for role in bundle["cast"]["roles"]}
    rejected, candidates = Counter(), []
    for chapter_number, chapter in enumerate(bundle["chapters"], 1):
        for row_number, row in enumerate(chapter["rows"]):
            # An explicit unison passage is also something each member actually
            # says. Use it as a solo audition without changing its attribution.
            performers = [performer for performer in _eligible_performers(row)
                          if canonical.get(performer.get("voice_id")) == target]
            if not performers:
                continue
            performer = min(performers, key=lambda item: (item["voice_id"] != target, item["role_id"]))
            original_voice, role_id = performer["voice_id"], performer["role_id"]
            role = roles.get(role_id, {})
            priority = _role_priority(row, role)
            if priority is None or role.get("voice_id") != original_voice:
                rejected["non_speech_role"] += 1
                continue
            attribution = row.get("attribution", {})
            alternatives = attribution.get("candidates", [])
            if (attribution.get("review_required") is not False
                    or attribution.get("basis") not in {"explicit", "context", "user"}
                    or any(item != role_id for item in alternatives)):
                rejected["uncertain_attribution"] += 1
                continue
            try:
                direction = effective_delivery(row)
                spoken, _ = build_spoken(row)
            except (ValueError, KeyError):
                rejected["unsupported_source_mapping"] += 1
                continue
            if _EXTREME.search(direction):
                rejected["extreme_delivery"] += 1
                continue
            found = False
            speech_start, speech_end = (
                _speech_bounds(spoken) if row["kind"] in {"dialogue", "thought", "letter"}
                else (0, len(spoken))
            )
            for start, end in _spans(spoken[speech_start:speech_end]):
                start, end = start + speech_start, end + speech_start
                text = spoken[start:end]
                words = _WORDS.findall(text)
                if not _MIN_EXCERPT_WORDS <= len(words) <= _MAX_WORDS or not _safe_text(text):
                    continue
                score = (priority, original_voice != target, not _complete(text),
                         attribution["basis"] == "context", abs(len(words) - _TARGET_WORDS),
                         chapter_number, row.get("sequence", row_number), start, end)
                source = {
                    **source_base, "kind": "book_excerpt",
                    "chapter_id": row["chapter_id"],
                    "chapter_title": chapter["entry"].get("title", ""),
                    "chapter_number": chapter_number,
                    "script_path": chapter["entry"].get("path"),
                    "segment_id": row["id"], "segment_ids": [row["id"]],
                    "voice_id": original_voice, "role_id": role_id,
                    "segment_kind": row["kind"], "block_id": row["block_id"],
                    "source_ranges": spoken_source_ranges(row, start, end),
                    "spoken_start": start, "spoken_end": end,
                    "word_count": len(words), "short_sample": len(words) < _TARGET_MIN_WORDS,
                    "attribution_basis": attribution["basis"],
                }
                if row.get("simultaneous_speakers"):
                    source["simultaneous_speakers"] = [dict(item) for item in row["simultaneous_speakers"]]
                candidates.append({"score": score, "reference_text": text, "source": source,
                                   "order": (chapter_number, row.get("sequence", row_number), start)})
                found = True
            if not found:
                rejected["insufficient_suitable_text"] += 1
    return source_base, rejected, candidates, canonical


def _joined_sample(candidates, source_base):
    ordered = sorted(candidates, key=lambda candidate: candidate["order"])
    if len(ordered) == 1:
        return {key: ordered[0][key] for key in ("reference_text", "source")}
    excerpts, texts, cursor = [], [], 0
    for candidate in ordered:
        text = candidate["reference_text"]
        excerpts.append(dict(candidate["source"], text=text, sample_start=cursor, sample_end=cursor + len(text)))
        texts.append(text)
        cursor += len(text) + 1
    text = " ".join(texts)
    words = len(_WORDS.findall(text))
    return {"reference_text": text, "source": {
        **source_base, "kind": "book_excerpts", "separator": " ",
        "excerpts": excerpts,
        "segment_ids": [item["segment_id"] for item in excerpts],
        "word_count": words, "short_sample": words < _TARGET_MIN_WORDS,
        "detail": "Separate source statements joined for an audition; not one contiguous book passage.",
    }}


def _choose(candidates, source_base):
    # Preserve one best span per segment and avoid repeating the same statement
    # from different scenes merely to make the audition longer.
    pool, seen_segments, seen_words = [], set(), set()
    for candidate in sorted(candidates, key=lambda item: item["score"]):
        segment_id = candidate["source"]["segment_id"]
        words = tuple(word.casefold() for word in _WORDS.findall(candidate["reference_text"]))
        if segment_id in seen_segments or words in seen_words:
            continue
        seen_segments.add(segment_id)
        seen_words.add(words)
        pool.append(candidate)
    selections = []
    for seed in pool[:32]:
        chosen = [seed]
        words = seed["source"]["word_count"]
        chars = len(seed["reference_text"])
        while words < _TARGET_MIN_WORDS:
            available = [candidate for candidate in pool if candidate not in chosen
                         and words + candidate["source"]["word_count"] <= _MAX_WORDS
                         and chars + 1 + len(candidate["reference_text"]) <= _MAX_CHARS]
            if not available:
                break
            addition = min(available, key=lambda item: (
                item["score"][0], abs(words + item["source"]["word_count"] - _TARGET_WORDS),
                item["score"][1:]))
            chosen.append(addition)
            words += addition["source"]["word_count"]
            chars += 1 + len(addition["reference_text"])
        if words < _MIN_WORDS:
            continue
        quality = (words < _TARGET_MIN_WORDS, max(item["score"][0] for item in chosen),
                   sum(item["score"][1] for item in chosen), abs(words - _TARGET_WORDS),
                   len(chosen), tuple(item["order"] for item in chosen))
        selections.append((quality, chosen))
    if not selections:
        return None
    return _joined_sample(min(selections, key=lambda item: item[0])[1], source_base)


def select_book_sample(bundle, voice_id, *, canonical_id):
    """Return a substantial source audition, or None if creative text is needed."""
    source_base, _, candidates, _ = _candidate_data(bundle, voice_id, canonical_id)
    return _choose(candidates, source_base)


def validate_casting_script(script):
    """Use the same audition safety limits during model validation and selection."""
    if (not isinstance(script, dict) or script.get("basis") != "casting"
            or not isinstance(script.get("text"), str)
            or not isinstance(script.get("reason"), str) or not script["reason"].strip()
            or not _MIN_WORDS <= sample_word_count(script["text"]) <= _MAX_WORDS
            or len(script["text"]) > _MAX_CHARS or not _safe_text(script["text"])):
        raise ValueError("Invalid prepared casting sample: use 25–60 clean spoken words within 650 characters")
    return script


def select_sample(bundle, voice_id, *, canonical_id, fallback_text=None):
    """Choose 25–60 words of clean book speech or a prepared creative audition.

    Shared voice aliases choose the same text. Every joined statement has its own
    source mapping. ``fallback_text`` remains an accepted legacy argument but is
    never substituted for a missing character-specific casting script.
    """
    source_base, rejected, candidates, canonical = _candidate_data(bundle, voice_id, canonical_id)
    selected = _choose(candidates, source_base)
    if selected is not None:
        return selected
    target = source_base["canonical_voice_id"]
    profiles = sorted((voice for voice in bundle["cast"]["voices"] if canonical[voice["id"]] == target),
                      key=lambda voice: (voice["id"] != target, voice["id"]))
    overridden = set(bundle.get("audition_script_overrides", []))
    provenance_scripts = bundle["manifest"].get("provenance", {}).get("voice_samples", {}).get("scripts", {})
    script_sets = [bundle.get("audition_scripts", {}),
                   {key: script for key, script in provenance_scripts.items() if key not in overridden},
                   {profile["id"]: profile.get("sample_script") for profile in profiles
                    if profile["id"] not in overridden}]
    keys = list(dict.fromkeys([target, *[profile["id"] for profile in profiles]]))
    prepared = [(key, scripts[key]) for scripts in script_sets for key in keys if key in scripts]
    for script_voice_id, script in prepared:
        if script is None:
            continue
        validate_casting_script(script)
        text = script["text"]
        return {"reference_text": text, "source": {
            **source_base, "kind": "casting_script", "basis": "casting", "voice_id": script_voice_id,
            "reason": script["reason"], "word_count": len(_WORDS.findall(text)),
            "detail": "Original character-specific audition text; not a quotation from the book.",
            "excluded_segments": dict(sorted(rejected.items())),
        }}
    raise MissingSampleScript(voice_id)
