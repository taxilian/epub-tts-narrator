"""Choose an audition script from a character's already-attributed book text.

Selection never writes the analyzer bundle or combines separated utterances.
Offsets use Python/Unicode code points, matching the analyzer and render plan.
"""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .bundle import ranges, spoken_mapping


SELECTION_VERSION = 2
_MAX_CHARS = 400
_MIN_WORDS = 12
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


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


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


def select_sample(bundle, voice_id, *, canonical_id, fallback_text):
    """Return ``reference_text`` and an auditable ``source`` description.

    ``bundle`` is a validated ``load_bundle`` result. ``canonical_id`` is the
    voice store's resolver; names are deliberately never used for matching.
    All aliases of a saved voice select the same passage. A caller can persist
    this result in a series draft to keep it unchanged in later books.
    """
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
        "book_id": book["id"],
        "book_title": book.get("title", ""),
    }
    roles = {role["id"]: role for role in bundle["cast"]["roles"]}
    rejected = Counter()
    candidates = []
    for chapter_number, chapter in enumerate(bundle["chapters"], 1):
        for row_number, row in enumerate(chapter["rows"]):
            original_voice = row.get("voice_id")
            if canonical.get(original_voice) != target:
                continue
            role = roles.get(row.get("role_id"), {})
            priority = _role_priority(row, role)
            if priority is None or role.get("voice_id") != original_voice:
                rejected["non_speech_role"] += 1
                continue
            attribution = row.get("attribution", {})
            alternatives = attribution.get("candidates", [])
            if (
                attribution.get("review_required") is not False
                or attribution.get("basis") not in {"explicit", "context", "user"}
                or any(item != row["role_id"] for item in alternatives)
            ):
                rejected["uncertain_attribution"] += 1
                continue
            if _EXTREME.search(row.get("delivery", {}).get("direction", "")):
                rejected["extreme_delivery"] += 1
                continue
            try:
                spoken, mapping = spoken_mapping(row)
            except (ValueError, KeyError):
                rejected["unsupported_source_mapping"] += 1
                continue
            found = False
            for start, end in _spans(spoken):
                text = spoken[start:end]
                words = _WORDS.findall(text)
                if len(words) < _MIN_WORDS:
                    continue
                # The engine interprets square/angle brackets as control syntax.
                # Only analyzer-provided bracket removal is accepted, above.
                if any(char in text for char in "[]<>"):
                    continue
                # Hidden format/control characters cannot safely be passed to
                # speech engines as literal text. Reject this span instead of
                # deleting characters and losing its exact source mapping.
                if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                       and char not in "\n\r\t" for char in text):
                    continue
                if "://" in text or re.search(r"\S+@\S+", text):
                    continue
                letters = [char for char in text if char.isalpha()]
                if letters and sum(char.isupper() for char in letters) / len(letters) > .55:
                    continue
                if re.search(r"([A-Za-z])\1{2,}", text):
                    continue
                if text.rstrip("”’\"'").endswith(("—", "–", "-")):
                    continue
                # Prefer finished sentences; short source rows can still be an
                # exact complete spoken clause interrupted by a narration tag.
                fit = 0 if 20 <= len(words) <= 60 else 1
                score = (
                    priority, original_voice != target, fit,
                    not _complete(text),
                    attribution["basis"] == "context",
                    abs(len(words) - 35),
                    chapter_number, row.get("sequence", row_number), start, end,
                )
                source = {
                    **source_base,
                    "kind": "book_excerpt",
                    "chapter_id": row["chapter_id"],
                    "chapter_title": chapter["entry"].get("title", ""),
                    "chapter_number": chapter_number,
                    "script_path": chapter["entry"].get("path"),
                    "segment_id": row["id"],
                    "segment_ids": [row["id"]],
                    "voice_id": original_voice,
                    "role_id": row["role_id"],
                    "segment_kind": row["kind"],
                    "block_id": row["block_id"],
                    "source_ranges": ranges(mapping[start:end]),
                    "spoken_start": start,
                    "spoken_end": end,
                    "word_count": len(words),
                    "short_sample": len(words) < 20,
                    "attribution_basis": attribution["basis"],
                }
                candidates.append((score, {"reference_text": text, "source": source}))
                found = True
            if not found:
                rejected["insufficient_suitable_text"] += 1
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return {
        "reference_text": fallback_text,
        "source": {
            **source_base,
            "kind": "fallback",
            "reason": "insufficient_character_dialogue",
            "detail": (
                "No clearly attributed, suitably delivered contiguous passage "
                f"of at least {_MIN_WORDS} spoken words fits within {_MAX_CHARS} characters."
            ),
            "excluded_segments": dict(sorted(rejected.items())),
        },
    }
