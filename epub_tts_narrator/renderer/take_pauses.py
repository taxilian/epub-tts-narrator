"""Even out sentence and clause pauses inside accepted takes during assembly.

Between clips, assembly already sets each sentence/clause gap to a total
target. Inside one take the TTS model chooses its own pauses (often far longer
than the target, and sometimes a stall of several seconds in the middle of a
phrase), and a sentence-split recovery joins its parts with almost none. This
module measures the quiet stretch actually present at each interior
punctuation boundary and returns frame edits that make that whole stretch
equal the same target. Accepted WAVs are never modified; the edits are applied
while the chapter is written and recorded in its timing.

Quiet is judged against the take itself (``quiet_windows``): the model often
fills a pause with a faint noise floor that flickers around -60 dBFS, so a
fixed threshold would see one long pause as many short sounds. A window is
quiet when it is far below the take's speech; a faint sound that rises clearly
above the take's own floor for 100ms or more (a breath, a soft consonant, a
whispered word) stays audible, so the pause is trimmed beside it, never
through it.

Some pauses are meant (``deliberate_spans``): a line directed as hesitant,
halting, stammering, slow, breathless and the like, a ``slow`` delivery
control, an ellipsis, or a stammered restart across a dash ("I—I"). There the
model's own pause is kept up to ``DELIBERATE_PAUSE_MS`` instead of being set to
the target; only a longer one is shortened, to that limit.

Boundaries are located only where the verifier's word timings align exactly
with the expected words on both sides; anything uncertain is only ever
shortened. Whisper stretches word timestamps across gaps, so timings only
choose which quiet stretch belongs to a boundary; its length is always
measured in audio.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from .pacing import audible_threshold, window_peaks


TAKE_PAUSE_VERSION = 2
# A meant pause (see the module docstring) keeps the model's length up to this.
DELIBERATE_PAUSE_MS = 1200
# A quiet stretch shorter than this is a stop closure or a join, not a pause
# that can be located safely. A boundary with none is left untouched.
_MIN_QUIET_MS = 30
# Inside the neighbouring words, only this much of each edge is searched.
_WORD_EDGE_SECONDS = 0.1
# Quiet: this far below the take's speech peaks, and never louder than
# ``_QUIET_CEILING_DBFS``; a clip's own -60 dBFS quiet always counts.
_BELOW_SPEECH_DB = 40
_QUIET_CEILING_DBFS = -45
# A sound this far above the take's quiet floor, lasting this long, is kept.
_SOUND_ABOVE_FLOOR_DB = 10
_SOUND_MS = 100
# A click this short between two quiet stretches at least this long is part of the pause;
# a word's own release burst sits beside speech, never between two quiet stretches.
_BLIP_MS = 20
_BLIP_NEIGHBOUR_MS = 50
_BOUNDARY = re.compile(
    r"(?P<sentence>(?:[.!?…]+))[\"'”’»)\]}]*(?=\s)"
    r"|(?P<clause>[,;:])[\"'”’»)\]}]*(?=\s)"
    r"|(?P<dash>(?<=\D)\s*[—–]\s*(?=\D)|\s*—\s*)")
_CLOSING = "\"'”’»)]} \t\n"
# Directions whose speaker is meant to pause: hesitation, slowness, broken
# breath or attention. General delivery vocabulary, never book words.
_HESITANT = re.compile(
    r"\b(?:hesita\w*|halting\w*|falter\w*|stammer\w*|stutter\w*|stumbl\w*|tentative\w*"
    r"|trail\w*|paus\w*|slow\w*|deliberat\w*|measured|drawl\w*|dazed|groggy|drowsy|sleepy"
    r"|dreamy|dreamily|absent\w*|distracted\w*|vague\w*|breathless\w*|gasp\w*|pant\w*|sob\w*"
    r"|weep\w*|tearful\w*|exhausted|weak\w*|feebl\w*|dying|drunk\w*|slurr\w*"
    r"|thoughtful\w*|musing\w*|ponder\w*|reflective\w*|uncertain\w*|unsure|reluctant\w*"
    r"|grudging\w*|awkward\w*|groping|searching)\b", re.I)


def _period_can_end(text, index):
    from .speech_recovery import _period_can_end as can_end
    return can_end(text, index)


def text_boundaries(text):
    """Return ``[(offset, kind)]`` for interior sentence and clause boundaries.

    ``offset`` splits the text just after the punctuation and any closing
    quotation marks. Abbreviations and initials do not end sentences, matching
    sentence-split recovery.
    """
    found = []
    for match in _BOUNDARY.finditer(text):
        if match["sentence"]:
            start = match.start("sentence")
            if match["sentence"] == "." and not _period_can_end(text, start):
                continue
            kind = "sentence"
        else:
            kind = "clause"
        found.append((match.end(), kind))
    return found


def hesitant_direction(direction):
    """Whether a delivery direction asks the speaker to pause."""
    return isinstance(direction, str) and _HESITANT.search(direction) is not None


def _meant_punctuation(text, offset):
    """An ellipsis, or a dash between a word and its stammered restart ("I—I")."""
    before = text[:offset].rstrip(_CLOSING)
    if before.endswith(("…", "...")):
        return True
    if not before.endswith(("—", "–", "-")):
        return False
    left = re.search(r"(\w+)\W*$", before)
    right = re.match(r"\W*(\w+)", text[offset:])
    if not left or not right:
        return False
    first, second = left[1].lower(), right[1].lower()
    return second.startswith(first) or first.startswith(second)


def deliberate_spans(clip, performance=None):
    """``[(start, end)]`` of ``clip["spoken_text"]`` whose pauses are meant.

    A paragraph group's members keep their own directions; ``slow`` delivery
    controls mark only their span.
    """
    text, spans = clip["spoken_text"], []

    def direction(row):
        delivery = row.get("delivery") if isinstance(row, dict) else None
        return delivery.get("direction") if isinstance(delivery, dict) else delivery

    turns = clip.get("paragraph_turns")
    if isinstance(turns, list) and turns:
        if "".join(turn.get("spoken_text", "") for turn in turns) == text:
            cursor = 0
            for turn in turns:
                if hesitant_direction(direction(turn.get("segment"))):
                    spans.append((cursor, cursor + len(turn["spoken_text"])))
                cursor += len(turn["spoken_text"])
    elif hesitant_direction(direction(clip.get("segment"))):
        spans.append((0, len(text)))
    for control in (performance or {}).get("controls", ()):
        if control.get("kind") == "delivery" and control.get("tag") == "slow":
            spans.append((control["start"], control["end"]))
    return sorted(spans)


def _timed_tokens(words):
    from .quality import normalize_words
    tokens = []
    for index, word in enumerate(words):
        if (not isinstance(word, dict) or not isinstance(word.get("word"), str)
                or type(word.get("start")) not in (int, float) or type(word.get("end")) not in (int, float)
                or not 0 <= word["start"] <= word["end"]):
            return None
        tokens.extend((token, index) for token in normalize_words(word["word"]))
    return tokens


def _stretches(flags, value):
    """Maximal ``(start, end)`` runs of ``value`` in ``flags``."""
    runs, start = [], None
    for index, flag in enumerate(list(flags) + [not value]):
        if flag == value and start is None:
            start = index
        elif flag != value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def quiet_windows(wav_path):
    """Return ``(rate, frames, window, audible)``, quiet judged against the take.

    See the module docstring: far below the take's speech is quiet, a faint
    sound clearly above the take's own quiet floor for ``_SOUND_MS`` stays
    audible, and a click of ``_BLIP_MS`` inside a long pause is part of it.
    """
    rate, frames, window, peaks = window_peaks(wav_path)
    if not peaks:
        return rate, frames, window, []
    strict = audible_threshold(peaks)
    ordered = sorted(peaks)
    speech = ordered[int(.9 * (len(ordered) - 1))]
    loose = max(strict, min(32768 * 10 ** (_QUIET_CEILING_DBFS / 20),
                            speech * 10 ** (-_BELOW_SPEECH_DB / 20)))
    below = sorted(peak for peak in peaks if peak <= loose)
    floor = below[len(below) // 4] if below else 0
    sound = max(strict, floor * 10 ** (_SOUND_ABOVE_FLOOR_DB / 20))
    audible = [peak > loose for peak in peaks]
    longest = max(1, round(_SOUND_MS / 10))
    faint = [not loud and peak > sound for loud, peak in zip(audible, peaks)]
    for start, end in _stretches(faint, True):
        if end - start >= longest:
            audible[start:end] = [True] * (end - start)
    blip, neighbour = max(1, round(_BLIP_MS / 10)), round(_BLIP_NEIGHBOUR_MS / 10)
    quiet = _stretches(audible, False)
    if quiet:
        # Merged left to right, so a stall broken by several clicks becomes one pause.
        current = quiet[0]
        for following in quiet[1:]:
            if (following[0] - current[1] <= blip and current[1] - current[0] >= neighbour
                    and following[1] - following[0] >= neighbour):
                audible[current[1]:following[0]] = [False] * (following[0] - current[1])
                current = (current[0], following[1])
            else:
                current = following
    return rate, frames, window, audible


def _quiet_runs(audible):
    """Maximal interior quiet runs as ``(first_window, end_window)``."""
    # A run reaching either end of the take is an outer edge, not a pause.
    return [run for run in _stretches(audible, False) if run[0] > 0 and run[1] < len(audible)]


def boundary_edits(wav_path, text, words, *, sentence_frames, clause_frames,
                   deliberate=(), deliberate_frames=0):
    """Return ``(edits, summary)`` that set interior boundary pauses to targets.

    Each edit is ``{"source_frame": F, "remove_frames": N}`` or
    ``{"source_frame": F, "insert_frames": N}`` in the accepted WAV's frames,
    sorted and nonoverlapping. A target of zero leaves that boundary kind as
    generated. The measured quiet stretch includes everything quiet between
    the two words, so the result is a total, never the model's pause plus an
    added one.

    Any other interior quiet stretch is only ever shortened, never
    lengthened: with usable word timings a stall in the middle of a phrase is
    capped at the clause target (the sentence target when clause pauses are
    disabled); without timings, every stretch is capped at the longer target.

    ``deliberate`` lists ``(start, end)`` spans of ``text`` whose pauses are
    meant (``deliberate_spans``). A boundary inside one, or at an ellipsis or a
    stammered dash, keeps its pause between its target and
    ``deliberate_frames``; a stall inside one is capped at
    ``deliberate_frames``.
    """
    from .quality import normalize_words
    summary = {"version": TAKE_PAUSE_VERSION, "boundaries": 0, "adjusted": 0,
               "unchanged": 0, "unlocated": 0, "capped": 0, "deliberate": 0}
    targets = {"sentence": sentence_frames, "clause": clause_frames}
    cap = max(sentence_frames, clause_frames)
    if cap <= 0:
        return [], summary
    rate, frames, window, audible = quiet_windows(wav_path)
    runs = _quiet_runs(audible)
    used, edits = set(), []
    allowance = max(cap, deliberate_frames) if deliberate else cap

    def set_length(run, target, *, ceiling=None):
        start, end = run[0] * window, min(frames, run[1] * window)
        length = end - start
        if ceiling is not None and target <= length <= ceiling:
            return False
        goal = ceiling if ceiling is not None and length > ceiling else target
        if abs(length - goal) < window:
            return False
        if length > goal:
            edits.append({"source_frame": start + goal // 2, "remove_frames": length - goal})
        else:
            edits.append({"source_frame": start + length // 2, "insert_frames": goal - length})
        return True

    def meant(offset):
        return any(start <= offset <= end for start, end in deliberate)

    expected = normalize_words(text)
    # Expected-token index -> whether a pause before that word is meant.
    meant_tokens = set()
    for start, end in deliberate:
        meant_tokens.update(range(len(normalize_words(text[:start])) + 1,
                                  len(normalize_words(text[:end])) + 1))
    whole = bool(expected) and set(range(1, len(expected))) <= meant_tokens
    boundaries = [(offset, kind) for offset, kind in text_boundaries(text) if targets[kind] > 0]
    timed = _timed_tokens(words) if isinstance(words, list) and words else None
    timed_word, token_of_word = {}, {}
    if timed and expected:
        matcher = SequenceMatcher(None, expected, [token for token, _ in timed], autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                timed_word[block.a + offset] = timed[block.b + offset][1]
                token_of_word.setdefault(timed[block.b + offset][1], block.a + offset)
    minimum = max(1, -(-round(rate * _MIN_QUIET_MS / 1000) // window))
    for offset, kind in boundaries:
        before = normalize_words(text[:offset])
        count = len(before)
        if not 0 < count < len(expected) or before + normalize_words(text[offset:]) != expected:
            continue  # Punctuation at the very start/end of the take is a clip edge.
        summary["boundaries"] += 1
        left, right = timed_word.get(count - 1), timed_word.get(count)
        if left is None or right is None or left >= right:
            summary["unlocated"] += 1
            continue
        first, second = words[left], words[right]
        low = first["start"] + min(_WORD_EDGE_SECONDS, (first["end"] - first["start"]) / 2)
        high = second["end"] - min(_WORD_EDGE_SECONDS, (second["end"] - second["start"]) / 2)
        low_window, high_window = int(low * rate // window), int(-(-high * rate // window))
        candidates = [run for run in runs if run not in used
                      and run[0] < high_window and run[1] > low_window
                      and run[1] - run[0] >= minimum]
        if not candidates:
            summary["unlocated"] += 1
            continue
        run = max(candidates, key=lambda item: (item[1] - item[0], -item[0]))
        used.add(run)
        if (meant(offset) or _meant_punctuation(text, offset)) and deliberate_frames > targets[kind]:
            summary["deliberate"] += 1
            changed = set_length(run, targets[kind], ceiling=deliberate_frames)
        else:
            changed = set_length(run, targets[kind])
        summary["adjusted" if changed else "unchanged"] += 1
    if timed_word and clause_frames > 0:
        cap = clause_frames
    starts = sorted((word["start"], index) for index, word in enumerate(words or ())
                    if index in token_of_word)
    for run in runs:
        if run in used:
            continue
        limit = cap
        if deliberate:
            if whole:
                limit = max(cap, allowance)
            elif starts:
                # The first aligned word starting after the pause begins says where it is.
                at = run[0] * window / rate
                following = next((index for start, index in starts if start >= at), None)
                if following is not None and token_of_word[following] in meant_tokens:
                    limit = max(cap, allowance)
        if (run[1] - run[0]) * window > limit + window:
            set_length(run, limit)
            summary["capped"] += 1
    edits.sort(key=lambda edit: edit["source_frame"])
    return edits, summary


def edited_length(edits):
    """Net frame change of a validated edit list."""
    return sum(edit.get("insert_frames", 0) - edit.get("remove_frames", 0) for edit in edits)


def validate_edits(edits, start, end):
    """Edits must be sorted, inside ``[start, end)`` and nonoverlapping."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("Take pause edits must be a nonempty list")
    cursor = start
    for edit in edits:
        if not isinstance(edit, dict) or "source_frame" not in edit or len(edit) != 2:
            raise ValueError("Take pause edit is invalid")
        frame = edit["source_frame"]
        (key, count), = ((key, value) for key, value in edit.items() if key != "source_frame")
        if (key not in ("remove_frames", "insert_frames") or type(frame) is not int
                or type(count) is not int or count <= 0 or not cursor <= frame < end
                or (key == "remove_frames" and frame + count > end)):
            raise ValueError("Take pause edit is out of range")
        cursor = frame + count if key == "remove_frames" else frame


_WORDS_FORMAT = "epub-tts-take-pause-words"


def _recognized_words(path, verifier, cache_dir):
    """Audio-only word timings for one saved take, cached by audio and model.

    Only the primary local MLX Whisper verifier is used, serialized on its own
    lane. No known text is forced onto the audio: ``boundary_edits`` uses only
    words that match the expected text exactly, so recognition errors elsewhere
    in the take cannot move a pause.
    """
    from .censorship_alignment import _model_identity, _primary_verifier, _serialized_verifier
    from .common import digest, read_json, regular_file, write_json
    from .stored_audio import audio_hash, regular_audio
    from pathlib import Path
    path = regular_audio(path)
    primary = _primary_verifier(verifier)
    binding = {"audio_sha256": audio_hash(path), "model": _model_identity(primary),
               "version": TAKE_PAUSE_VERSION}
    directory = Path(cache_dir)
    if directory.is_symlink():
        raise ValueError("Pause word timing cache must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (digest(binding) + ".json")
    if target.exists() or target.is_symlink():
        saved = read_json(regular_file(target))
        if (isinstance(saved, dict) and saved.get("format") == _WORDS_FORMAT
                and saved.get("binding") == binding and isinstance(saved.get("words"), list)
                and saved.get("words_sha256") == digest(saved["words"])):
            return saved["words"]
    facade = _serialized_verifier(verifier)

    def recognize():
        if facade is not None:
            facade._runtime.initialize_mlx()
        from .stored_audio import materialize
        with materialize(path) as wav:
            return primary.transcribe(wav)

    raw = facade._call("take_pause_words", recognize, (), {}) if facade is not None else recognize()
    try:
        # Recognizers return NumPy scalars; persist plain JSON numbers.
        words = [{"word": str(word["word"]), "start": float(word["start"]), "end": float(word["end"])}
                 for segment in (raw.get("segments") or [] if isinstance(raw, dict) else [])
                 if isinstance(segment, dict) for word in segment.get("words") or []
                 if isinstance(word, dict)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Recognized word timings are malformed") from exc
    if _timed_tokens(words) is None:
        raise ValueError("Recognized word timings are malformed")
    if audio_hash(path) != binding["audio_sha256"]:
        raise ValueError("Saved take changed during word timing")
    write_json(target, {"format": _WORDS_FORMAT, "binding": binding, "words": words,
                        "words_sha256": digest(words)})
    return words


def align_missing_timings(clips, acceptance, verifier, cache_dir, *, notify=print):
    """Attach audio-only word timings to takes whose verification kept none.

    A take accepted through the secondary recognizer has no primary word
    timings. The primary Whisper verifier recognizes the unchanged accepted WAV
    once (cached); it never re-verifies or regenerates speech. A take that
    cannot be recognized keeps its pauses capped without located boundaries.
    """
    from .censorship_alignment import CensorshipAlignmentUnavailable
    reports = {item["chunk_id"]: item.get("report") for item in acceptance}
    result, timed, failed = [], 0, 0
    for clip in clips:
        report = reports.get(clip.get("chunk_id"))
        if (clip.get("wav_path") is None or "speaker_lanes" in clip or "pause_alignment" in clip
                or not isinstance(report, dict) or report.get("words")
                or not isinstance(report.get("expected_text"), str)
                or not text_boundaries(report["expected_text"])):
            result.append(clip)
            continue
        try:
            words = _recognized_words(clip["wav_path"], verifier, cache_dir)
        except (CensorshipAlignmentUnavailable, ValueError, OSError, KeyError, TypeError):
            # Timing is a pacing aid, never an acceptance step: speech stays as accepted.
            failed += 1
            result.append(clip)
            continue
        timed += 1
        result.append(dict(clip, pause_alignment={"expected_text": report["expected_text"],
                                                  "words": words}))
    if timed or failed:
        notify(f"  Word timings for pause normalization: {timed} saved take(s) recognized"
               + (f", {failed} left with capped pauses only" if failed else "") + ".")
    return result
