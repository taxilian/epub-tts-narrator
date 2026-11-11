"""Even out sentence and clause pauses inside accepted takes during assembly.

Between clips, assembly already sets each sentence/clause gap to a total
target. Inside one take the TTS model chooses its own pauses (often far longer
than the target), and a sentence-split recovery joins its parts with almost
none. This module measures the quiet stretch actually present at each interior
punctuation boundary, using the same 10ms audible/quiet definition as clip
edges (``pacing.activity``), and returns frame edits that make that whole
stretch equal the same target. Accepted WAVs are never modified; the edits are
applied while the chapter is written and recorded in its timing.

Boundaries are located only where the verifier's word timings align exactly
with the expected words on both sides; anything uncertain stays as generated.
Whisper stretches word timestamps across gaps, so timings only choose which
quiet stretch belongs to a boundary; its length is always measured in audio.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from .pacing import activity


TAKE_PAUSE_VERSION = 1
# A quiet stretch shorter than this is a stop closure or a join, not a pause
# that can be located safely. A boundary with none is left untouched.
_MIN_QUIET_MS = 30
# Inside the neighbouring words, only this much of each edge is searched.
_WORD_EDGE_SECONDS = 0.1
_BOUNDARY = re.compile(
    r"(?P<sentence>(?:[.!?…]+))[\"'”’»)\]}]*(?=\s)"
    r"|(?P<clause>[,;:])[\"'”’»)\]}]*(?=\s)"
    r"|(?P<dash>(?<=\D)\s*[—–]\s*(?=\D)|\s*—\s*)")


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


def _quiet_runs(audible):
    """Maximal interior quiet runs as ``(first_window, end_window)``."""
    runs, start = [], None
    for index, loud in enumerate(audible):
        if not loud and start is None:
            start = index
        elif loud and start is not None:
            if start > 0:
                runs.append((start, index))
            start = None
    return runs  # A run reaching the end of the take is an outer edge, not a pause.


def boundary_edits(wav_path, text, words, *, sentence_frames, clause_frames):
    """Return ``(edits, summary)`` that set interior boundary pauses to targets.

    Each edit is ``{"source_frame": F, "remove_frames": N}`` or
    ``{"source_frame": F, "insert_frames": N}`` in the accepted WAV's frames,
    sorted and nonoverlapping. A target of zero leaves that boundary kind as
    generated. The measured quiet stretch includes everything below the quiet
    threshold between the two words, so the result is a total, never the
    model's pause plus an added one.

    Any other interior quiet stretch is only ever shortened, never
    lengthened: with usable word timings a stall in the middle of a phrase is
    capped at the clause target (the sentence target when clause pauses are
    disabled); without timings, every stretch is capped at the longer target.
    """
    from .quality import normalize_words
    summary = {"version": TAKE_PAUSE_VERSION, "boundaries": 0, "adjusted": 0,
               "unchanged": 0, "unlocated": 0, "capped": 0}
    targets = {"sentence": sentence_frames, "clause": clause_frames}
    cap = max(sentence_frames, clause_frames)
    if cap <= 0:
        return [], summary
    rate, frames, window, audible = activity(wav_path)
    runs = _quiet_runs(audible)
    used, edits = set(), []

    def set_length(run, target):
        start, end = run[0] * window, min(frames, run[1] * window)
        length = end - start
        if abs(length - target) < window:
            return False
        if length > target:
            edits.append({"source_frame": start + target // 2, "remove_frames": length - target})
        else:
            edits.append({"source_frame": start + length // 2, "insert_frames": target - length})
        return True

    boundaries = [(offset, kind) for offset, kind in text_boundaries(text) if targets[kind] > 0]
    expected = normalize_words(text) if boundaries else []
    timed = _timed_tokens(words) if isinstance(words, list) and words else None
    timed_word = {}
    if timed and expected:
        matcher = SequenceMatcher(None, expected, [token for token, _ in timed], autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                timed_word[block.a + offset] = timed[block.b + offset][1]
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
        summary["adjusted" if set_length(run, targets[kind]) else "unchanged"] += 1
    if timed_word and clause_frames > 0:
        cap = clause_frames
    for run in runs:
        if run not in used and (run[1] - run[0]) * window > cap + window:
            set_length(run, cap)
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
    from .common import digest, file_hash, read_json, regular_file, write_json
    from pathlib import Path
    path = regular_file(path)
    primary = _primary_verifier(verifier)
    binding = {"audio_sha256": file_hash(path), "model": _model_identity(primary),
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
        return primary.transcribe(path)

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
    if file_hash(path) != binding["audio_sha256"]:
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
