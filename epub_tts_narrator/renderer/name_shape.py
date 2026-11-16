"""Reject a take whose chosen pronunciation came out with its syllables rearranged.

A confirmed pronunciation reaches the engine as phonemes, but Drama 3 does not
always follow them: a name read ``V AH2 L EH1 D AH0`` (vah-LED-uh) can come
out as "VAL-deh-uh". The recognizer hears that difference ("Valeda" against
"Valdea"), but every word-level check accepts both: Double Metaphone ignores
vowels, so both spellings reduce to the same consonants.

This check compares the *shape* of the chosen reading with the shape of what
was heard for that word: how many consonants come before, between and after
its vowels. A recognizer spelling a name differently keeps the shape
(``Valeda``, ``Voleda`` and ``Valida`` all have one consonant, a vowel, one
consonant, a vowel, one consonant and a vowel), while the misreading moves a
consonant across a vowel (``Valdea``: one, two, none). The take is rejected
only when the heard word has the same number of vowel groups and the same
number of consonants as the reading but they sit differently. Anything else
(a heard word with other consonants, a dropped syllable, a dictionary word in
its place) is left to the ordinary checks, which already decided the take.

Only person-chosen CMU readings are checked (``pronunciation`` controls with
``cmu_arpabet``), one word at a time; respellings and multi-word readings are
not. It runs on chapter speech that otherwise passed verification.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

NAME_SHAPE_VERSION = 2  # 2: a possessive heard as a name ending in s (``Valedas``)
REASON = "pronunciation_mismatch"
_VOWELS = frozenset("aeiou")
_DIGRAPHS = ("th", "sh", "ch", "ph", "wh", "gh", "ck")
_POSSESSIVE = re.compile(r"['’]s$")
SWEEP_FILENAME = "name-shape-sweep.json"
# 2: best-effort choices are swept too (every acceptance is read again once).
# 3: a best-effort choice is also retired when a chosen name is missing from
#    it or it inserted other speech (``misread(require_heard=True)``,
#    ``exhaustion_selection.inserted_words``).
SWEEP_POLICY = 3
_STRETCH = 40  # words: a longer differing stretch is not a misheard name
_MTIME_SLACK = 120  # seconds: an acceptance written while a sweep ran is read by the next one


def _clusters(units):
    """Consonant counts before, between and after each run of vowels: ``[1, 1, 2, 0]``."""
    clusters, run, after_vowel = [], 0, False
    for unit in units:
        if unit is None:  # a vowel
            if not after_vowel:
                clusters.append(run)
                run = 0
            after_vowel = True
        else:
            run += unit
            after_vowel = False
    clusters.append(run)
    return clusters


def reading_shape(phonemes):
    """The shape of a CMU reading: vowels carry a stress digit, everything else is one consonant."""
    from ..performance_plan import phoneme_symbols
    phones = phoneme_symbols(phonemes)  # boundary marks are not sounds
    return _clusters([None if phone[-1:].isdigit() else 1 for phone in phones])


def spelling_shape(word):
    """The shape of a heard spelling, reading English letter groups as one sound each."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    units, index = [], 0
    while index < len(letters):
        letter = letters[index]
        following = letters[index + 1] if index + 1 < len(letters) else ""
        if letter in _VOWELS or (letter == "y" and following not in _VOWELS):
            units.append(None)
        elif letters[index:index + 2] in _DIGRAPHS:
            units.append(1)
            index += 1
        elif letter == "h" and units and units[-1] is not None:
            pass  # A breathy "h" after a consonant (dh, nh, kh) adds no consonant of its own.
        elif index and letters[index - 1] == letter:
            pass  # A doubled letter is one sound.
        else:
            units.append(2 if letter == "x" else 1)
        index += 1
    return _clusters(units)


def rearranged(phonemes, heard):
    """Whether ``heard`` has the reading's vowels and consonants but in other places."""
    expected, actual = reading_shape(phonemes), spelling_shape(heard)
    return len(expected) == len(actual) and sum(expected) == sum(actual) and expected != actual


def _readings(performance):
    """Person-chosen one-word CMU readings of a passage: ``[(control, phonemes)]``."""
    from .quality import normalize_words
    found = []
    for control in (performance or {}).get("controls", []):
        if (control.get("kind") != "pronunciation" or control.get("alphabet") != "cmu_arpabet"
                or len(normalize_words(control.get("text") or "")) != 1):
            continue
        phonemes = control["phonemes"].strip()
        if _POSSESSIVE.search(control["text"]):
            # The reading of a possessive ends in its "s" sound; the name itself is the rest.
            phonemes = re.sub(r"\s+[SZ]$", "", phonemes)
        found.append((control, phonemes))
    return found


def reading_key(phonemes):
    """One chosen reading, whatever its group and boundary marks and spacing: ``S EH2 N EH1 D R AH0``."""
    from ..performance_plan import phoneme_symbols
    return " ".join(phoneme_symbols(phonemes))


def chosen_readings(performance):
    """Keys of the passage's person-chosen one-word CMU readings (possessives share the name's key)."""
    return {reading_key(phonemes) for _, phonemes in _readings(performance)}


def _heard(expected_text, transcript, performance):
    """``[(control, phonemes, heard)]`` per chosen reading: ``heard`` is the word heard in its place.

    ``heard`` is ``True`` when the recognizer wrote the word as written,
    ``None`` when nothing in the transcript can be placed against it.
    """
    from difflib import SequenceMatcher
    from .close_match import _alignment
    from .quality import normalize_words
    readings = _readings(performance)
    if not readings or not isinstance(transcript, str) or not transcript.strip():
        return []
    expected, heard = normalize_words(expected_text), normalize_words(transcript)
    if not heard:
        return []
    # Matching runs come from difflib; only the differing stretch around a
    # name is aligned word by word, so long paragraphs stay cheap.
    opcodes = SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes()
    found = []
    for control, phonemes in readings:
        index = len(normalize_words(expected_text[:control["start"]]))
        opcode = next((item for item in opcodes if item[1] <= index < item[2]), None)
        if opcode is not None and opcode[0] == "equal":
            found.append((control, phonemes, True))
            continue
        if opcode is None or opcode[0] != "replace" or opcode[2] - opcode[1] > _STRETCH or opcode[4] - opcode[3] > _STRETCH:
            found.append((control, phonemes, None))
            continue
        _, first, last, heard_first, heard_last = opcode
        change = next((item for item in _alignment(expected[first:last], heard[heard_first:heard_last])
                       if item["expected_start"] <= index - first < item["expected_end"]), None)
        if (change is None or change["operation"] in {"equal", "deleted", "added"}
                or change["expected_end"] - change["expected_start"] != 1 or not change["actual"]):
            found.append((control, phonemes, None))
            continue
        found.append((control, phonemes, "".join(change["actual"])))
    return found


def _compared(control, phonemes, heard):
    """``(phonemes, word)`` to compare for one heard word.

    A possessive heard with its own ``'s`` compares the name itself. One heard
    otherwise (``Valedas``: the recognizer wrote the
    possessive as a name ending in its "s" sound) compares the whole reading,
    possessive sound included, with the whole heard word.
    """
    if _POSSESSIVE.search(control["text"]) and not _POSSESSIVE.search(heard):
        return control["phonemes"].strip(), heard
    return phonemes, _POSSESSIVE.sub("", heard)


def check(expected_text, transcript, performance):
    """``[{"text", "heard", "expected_shape", "heard_shape"}]`` for every chosen reading heard rearranged."""
    found = []
    for control, phonemes, heard in _heard(expected_text, transcript, performance):
        if not isinstance(heard, str):
            continue
        phonemes, word = _compared(control, phonemes, heard)
        if rearranged(phonemes, word):
            found.append({"text": control["text"], "heard": heard,
                          "expected_shape": reading_shape(phonemes), "heard_shape": spelling_shape(word)})
    return found


def heard_right(expected_text, transcript, performance):
    """Keys of the chosen readings this take was heard saying in their own shape.

    A word written exactly as in the text, or spelled another way with the
    reading's shape (``Voleda`` for ``Va'Leda``), counts; a rearranged,
    missing or unplaceable word does not. Chapter history uses these takes as
    examples of the reading (``continuity.ChapterContinuity.takes``).
    """
    right = set()
    for control, phonemes, heard in _heard(expected_text, transcript, performance):
        if heard is True:
            right.add(reading_key(phonemes))
        elif isinstance(heard, str):
            compared, word = _compared(control, phonemes, heard)
            if spelling_shape(word) == reading_shape(compared):
                right.add(reading_key(phonemes))
    return right


def apply(report, expected_text, performance, *, purpose="speech"):
    """Fail a passed chapter-speech report whose chosen readings were heard rearranged; the report."""
    if purpose != "speech" or report.get("passed") is not True or performance is None:
        return report
    transcript = report.get("verified_transcript") or report.get("transcript")
    found = check(expected_text, transcript, performance)
    if not found:
        return report
    return {**report, "passed": False, "reasons": [REASON],
            "name_shape": {"version": NAME_SHAPE_VERSION, "words": found,
                           "passed_method": report.get("text_confirmation_method")},
            "text_confirmation_method": None}


def describe(report):
    """``Va'Leda heard as Valdea`` for a rejected report, or ``""``."""
    words = (report.get("name_shape") or {}).get("words") or []
    return "; ".join(f"{item['text']} heard as {item['heard']}" for item in words)


def misread(record, *, require_heard=False):
    """``Va'Leda heard as Valdea`` when a take misread a chosen reading, else ``""``.

    Judged from its saved transcript as well as its report: a take verified
    by a path that once skipped this check (clip repair before 2026-11-15)
    carries no rejection reason, yet must never be published either. With
    ``require_heard`` (best-effort choices, which already failed ordinary
    verification) a chosen reading that is missing or cannot be placed in the
    transcript counts too (``Va'Leda not heard``): a take that replaced the
    name with other words says it no better than one that rearranged it.
    """
    report = record.get("report") if isinstance(record, dict) else None
    if not isinstance(report, dict):
        return ""
    if REASON in (report.get("reasons") or []):
        return describe(report) or "a chosen pronunciation heard rearranged"
    request = record.get("request")
    if not isinstance(request, dict):
        return ""
    try:
        from .performance_runtime import verification_options
        performance = verification_options(request).get("performance")
    except (ValueError, KeyError, TypeError):
        return ""
    expected, transcript = report.get("expected_text") or "", report.get("verified_transcript") or report.get("transcript")
    found = [f"{item['text']} heard as {item['heard']}" for item in check(expected, transcript, performance) or []]
    if require_heard and not found:
        right = heard_right(expected, transcript, performance)
        found = [f"{control['text']} not heard" for control, phonemes in _readings(performance)
                 if reading_key(phonemes) not in right]
    return "; ".join(found)


def retire_misread(work):
    """Retire every take in use whose chosen reading was heard rearranged; their take folders.

    Takes accepted before this check existed are found from their saved
    transcripts (no recognition runs), so the passages render again under the
    check. So are best-effort choices: a take that misreads a name a person
    chose the pronunciation of is never published. A take a person chose or
    marked good is kept: a person already decided it.
    """
    from .common import read_json, write_json
    from .performance_runtime import verification_options
    from .take_feedback import load
    from .take_review import retire_acceptance
    work = Path(work)
    clips = work / "clips"
    if not clips.is_dir() or clips.is_symlink():
        return []
    good = {entry.get("audio_sha256") for entry in load(work)["takes"].values() if entry.get("verdict") == "good"}
    # Takes accepted under this check passed it already; only acceptances
    # written since the last sweep (by any renderer) are read again.
    marker = work / SWEEP_FILENAME
    saved = read_json(marker) if marker.is_file() and not marker.is_symlink() else {}
    since = (saved.get("checked_through", 0) if saved.get("version") == NAME_SHAPE_VERSION
             and saved.get("policy") == SWEEP_POLICY else 0)
    started = time.time()
    retired = []
    for path in sorted([*clips.glob("*/accepted.json"), *clips.glob("*/parts/*/accepted.json")]):
        if path.is_symlink() or not path.is_file() or path.stat().st_mtime < since - _MTIME_SLACK:
            continue
        data = path.read_bytes()
        if b'"cmu_arpabet"' not in data and b'"exhaustion_selection"' not in data:
            continue
        record = read_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("report"), dict):
            continue
        report, metadata = record["report"], record.get("engine_metadata") or {}
        if (report.get("passed") is not True or "manual_selection" in metadata
                or record.get("audio_sha256") in good or not isinstance(record.get("request"), dict)
                or record["request"].get("kind") not in {"speech", "paragraph_speech", "speech_recovery_part"}):
            continue
        try:
            performance = verification_options(record["request"]).get("performance")
        except (ValueError, KeyError, TypeError):
            continue
        found = check(report.get("expected_text") or "", report.get("verified_transcript") or report.get("transcript"),
                      performance)
        evidence = {"reason": REASON, "version": NAME_SHAPE_VERSION, "words": found}
        if not found and "exhaustion_selection" in metadata:
            # A best-effort choice already failed ordinary verification: it
            # must still say every chosen name, and nothing else at length.
            from .exhaustion_selection import INSERTED_STRETCH, inserted_words
            missing = misread(record, require_heard=True)
            extra = inserted_words(record)
            if missing or extra >= INSERTED_STRETCH:
                evidence = {"reason": REASON if missing else "inserted_speech", "version": NAME_SHAPE_VERSION,
                            "missing": missing, "inserted_words": extra}
                found = True
        if found:
            retire_acceptance(path.parent, record, "name-shape", evidence)
            retired.append(path.parent)
    write_json(marker, {"version": NAME_SHAPE_VERSION, "policy": SWEEP_POLICY, "checked_through": started})
    return retired
