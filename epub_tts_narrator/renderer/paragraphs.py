"""Group contiguous source passages without changing their saved voice assignments."""

from __future__ import annotations

from copy import deepcopy

from .common import digest


_GROUP_VERSION = 1
_MAX_SPEAKERS = 16


def paragraph_supported(capabilities, chunk):
    """Check whether a worker can retain this paragraph's saved voice boundaries."""
    turns = chunk.get("paragraph_turns")
    if turns is None or capabilities.get("multi_speaker") is True:
        return True
    if capabilities.get("single_speaker_paragraph") is not True:
        return False
    voices = {turn["segment"].get("voice_id") for turn in turns}
    return len(voices) == 1 and None not in voices


def _source_piece(chunk):
    """Return an exact contiguous source slice, or decline unsafe grouping."""
    row = chunk["segment"]
    ranges = chunk.get("source_ranges", [])
    if not ranges:
        return None
    pieces, cursor = [], ranges[0].get("start")
    first = cursor
    if type(cursor) is not int:
        return None
    for span in ranges:
        start, end = span.get("start"), span.get("end")
        if (type(start) is not int or type(end) is not int or start != cursor
                or not row["start"] <= start < end <= row["end"]):
            return None
        pieces.append(row["text"][start - row["start"]:end - row["start"]])
        cursor = end
    text = "".join(pieces)
    if len(text) != cursor - first or text != chunk["spoken_text"]:
        return None
    return first, cursor, text


def _group(chunks):
    if len(chunks) < 2:
        return chunks
    pieces = [_source_piece(chunk) for chunk in chunks]
    if any(piece is None for piece in pieces) or any(
            previous[1] != following[0] for previous, following in zip(pieces, pieces[1:])):
        return chunks
    identity = {"paragraph_group_version": _GROUP_VERSION,
                "member_chunk_ids": [chunk["chunk_id"] for chunk in chunks]}
    identifier = digest(identity)[:32]
    first, last = chunks[0], chunks[-1]
    text = "".join(piece[2] for piece in pieces)
    row = deepcopy(first["segment"])
    row.update(id=f"paragraph:{identifier}", start=pieces[0][0], end=pieces[-1][1],
               text=text, spoken_text=text, role_id=None, voice_id=None,
               delivery={"direction": "", "evidence": []})
    # A chunk cut in the middle of a source row must not inherit that row's
    # initial blank cue or final paragraph cue on every resulting group.
    if first["spoken_start"] != 0:
        row.pop("break_before", None)
    last_spoken = last["segment"].get("spoken_text", last["segment"]["text"])
    row["break_after"] = (last["segment"].get("break_after", "none")
                          if last["spoken_end"] == len(last_spoken) else "none")
    return [{"segment": row, "spoken_text": text,
             "spoken_start": 0, "spoken_end": len(text),
             "source_ranges": [deepcopy(span) for chunk in chunks for span in chunk["source_ranges"]],
             "silent_control": False, "chunk_id": identifier,
             "paragraph_turns": deepcopy(chunks)}]


def _new_groups(chunks, *, max_chars, preserved, multi_speaker):
    # Original accepted subclips win over new grouping for the whole paragraph.
    # A saved paragraph-group ID does not appear among its original member IDs,
    # so the same deterministic group can be reconstructed and reused.
    if (any(chunk["chunk_id"] in preserved for chunk in chunks)
            or any(chunk["segment"].get("performance", {}).get("kind") == "bracketed"
                   or chunk["segment"].get("simultaneous_speakers")
                   or chunk.get("paragraph_turns") for chunk in chunks)):
        return chunks
    result, pending, voices, performers, characters = [], [], set(), set(), 0

    def flush():
        nonlocal pending, voices, performers, characters
        result.extend(_group(pending))
        pending, voices, performers, characters = [], set(), set(), 0

    for chunk in chunks:
        voice = chunk["segment"].get("voice_id")
        if (chunk.get("silent_control") or not voice
                or not chunk["segment"].get("role_id") or _source_piece(chunk) is None):
            flush()
            result.append(chunk)
            continue
        proposed_voices = voices | {voice}
        performer = (chunk["segment"]["role_id"], voice)
        if pending and (characters + len(chunk["spoken_text"]) > max_chars
                        or len(proposed_voices) > _MAX_SPEAKERS
                        or (not multi_speaker and performer not in performers)
                        or _source_piece(pending[-1])[1] != _source_piece(chunk)[0]):
            flush()
        pending.append(chunk)
        voices.add(voice)
        performers.add(performer)
        characters += len(chunk["spoken_text"])
    flush()
    return result


def _paragraph(chunks, *, max_chars, preserved, multi_speaker, saved_groups):
    if any(chunk["segment"].get("performance", {}).get("kind") == "bracketed"
           or chunk["segment"].get("simultaneous_speakers") for chunk in chunks):
        return chunks
    if not saved_groups:
        return _new_groups(chunks, max_chars=max_chars, preserved=preserved,
                           multi_speaker=multi_speaker)
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    result, gap_start, cursor = [], 0, 0
    while cursor < len(chunks):
        anchored = None
        # Candidates were sorted by descending length; scanning source positions
        # first makes overlapping saved groups deterministic, independent of
        # their order in a directory listing or acceptance report.
        for members in saved_groups.get(identifiers[cursor], ()):
            end = cursor + len(members)
            if identifiers[cursor:end] != list(members):
                continue
            current = chunks[cursor:end]
            voice_count = len({chunk["segment"].get("voice_id") for chunk in current})
            performer_count = len({(chunk["segment"].get("role_id"), chunk["segment"].get("voice_id"))
                                   for chunk in current})
            if voice_count > _MAX_SPEAKERS or (not multi_speaker and performer_count > 1):
                continue
            if any(chunk.get("silent_control") or chunk.get("paragraph_turns")
                   or not chunk["segment"].get("voice_id")
                   or not chunk["segment"].get("role_id")
                   or chunk["segment"].get("performance", {}).get("kind") == "bracketed"
                   or chunk["segment"].get("simultaneous_speakers") for chunk in current):
                continue
            candidate = _group(current)
            if len(candidate) == 1 and "paragraph_turns" in candidate[0]:
                anchored = candidate[0]
                break
        if anchored is None:
            cursor += 1
            continue
        result.extend(_new_groups(chunks[gap_start:cursor], max_chars=max_chars,
                                  preserved=preserved, multi_speaker=multi_speaker))
        result.append(anchored)
        cursor += len(anchored["paragraph_turns"])
        gap_start = cursor
    result.extend(_new_groups(chunks[gap_start:], max_chars=max_chars,
                              preserved=preserved, multi_speaker=multi_speaker))
    return result


def group_paragraphs(plan, *, max_chars=1000, preserved_chunk_ids=frozenset(),
                     preserved_groups=(), multi_speaker=False):
    """Return a detached plan with safe groups of two or more source chunks.

    Source blocks, role/voice changes, and explicit blank-paragraph boundaries limit
    each performance by default. A model's speaker tags cannot guarantee an
    audible switch, and transcript verification cannot detect the wrong voice.
    ``multi_speaker=True`` remains available for interpreting legacy plans.
    Oversized paragraphs split only between existing chunks. Original chunk
    dictionaries remain in ``paragraph_turns`` for speaker/delivery/sync use.
    Existing single chunks, unsupported passages, and accepted original clips
    keep their original identities and source ranges.

    ``preserved_groups`` contains saved groups' original member chunk IDs. Exact
    consecutive matches anchor their prior membership before individual-clip
    preservation is applied to the intervening gaps. This retains a saved group
    next to a saved leftover singleton, even after changing ``max_chars``.
    Saved groups also obey the current role/voice boundary policy: an old mixed
    performance cannot bypass separate synthesis after an upgrade.
    Overlapping candidates prefer the earliest source position and then the
    longest group at that position. Current source and metadata always win.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("Paragraph max_chars must be a positive integer")
    if type(multi_speaker) is not bool:
        raise ValueError("multi_speaker must be boolean")
    preserved = set(preserved_chunk_ids)
    saved_groups = {}
    for members in preserved_groups:
        if (not isinstance(members, (list, tuple)) or len(members) < 2
                or any(not isinstance(identifier, str) or not identifier for identifier in members)
                or len(set(members)) != len(members)):
            raise ValueError("preserved_groups must contain distinct member chunk ID lists")
        saved_groups.setdefault(members[0], set()).add(tuple(members))
    saved_groups = {first: sorted(groups, key=lambda members: (-len(members), members))
                    for first, groups in saved_groups.items()}
    result = deepcopy(plan)
    for chapter in result["chapters"]:
        grouped, paragraph = [], []

        def flush():
            nonlocal paragraph
            grouped.extend(_paragraph(paragraph, max_chars=max_chars, preserved=preserved,
                                      multi_speaker=multi_speaker, saved_groups=saved_groups))
            paragraph = []

        for chunk in chapter["chunks"]:
            row = chunk["segment"]
            if paragraph and (row["block_id"] != paragraph[-1]["segment"]["block_id"]
                              or (chunk["spoken_start"] == 0
                                  and row.get("break_before") == "blank_paragraph")):
                flush()
            paragraph.append(chunk)
        flush()
        chapter["chunks"] = grouped
    return result
