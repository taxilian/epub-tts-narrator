"""Terminal speaker choices after automatic assignment is exhausted."""

from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import os
import re
import shlex
import sys
import textwrap

from .contracts import has_assigned_speakers, readiness, validate_document
from .pipeline import fingerprint
from .speaker_completion import ATTRIBUTION_ISSUE_KINDS, unassigned_segments
from .speaker_suggestions import rank_speakers
from .narration_identity import author_narrator_name


def _display(value):
    # Book text and model explanations are display data, never terminal controls.
    text = re.sub(r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)", "", str(value))
    text = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", "", text)
    return "".join(char for char in text if char in "\n\t" or char.isprintable())


def _context_runs(block, segments, labels, start, end, *, tail=False, limit=1000):
    """Clip context in source coordinates, then label each contiguous voice run."""
    text = block["text"]
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    truncated = end - start > limit
    if truncated:
        if tail:
            start = end - limit
        else:
            end = start + limit
    runs = []
    for segment in segments:
        left, right = max(start, segment["start"]), min(end, segment["end"])
        if right <= left:
            continue
        role_ids = (tuple(speaker["role_id"] for speaker in segment["simultaneous_speakers"])
                    if segment.get("simultaneous_speakers") else
                    (segment["role_id"],) if has_assigned_speakers(segment) else ())
        if runs and runs[-1][0] == role_ids:
            runs[-1][1] += text[left:right]
        else:
            runs.append([role_ids, text[left:right]])
    visible = []
    for role_ids, excerpt in runs:
        if excerpt.strip():
            name = "; ".join(labels[role_id] for role_id in role_ids) if role_ids else "Unassigned"
            if len(role_ids) > 1:
                name += " together"
            visible.append([name, excerpt.strip()])
    if truncated and visible:
        if tail:
            visible[0][1] = "[context truncated] … " + visible[0][1]
        else:
            visible[-1][1] += " … [context truncated]"
    return visible


def _role_label(role, characters, book):
    character = characters.get(role.get("character_id"))
    author = author_narrator_name(role, book)
    if author is not None:
        return f"{author} (author)"
    name = character["name"] if character else "Narrator" if role["kind"] == "narration" else role["description"]
    return f"{name} (narrator)" if character and role["kind"] == "narration" else name


def _role_labels(roles, characters, book):
    labels = {role["id"]: _role_label(role, characters, book) for role in roles}
    for describe in (lambda role: role.get("description", ""),
                     lambda role: characters.get(role.get("character_id"), {}).get("description", "")):
        groups = {}
        for role in roles:
            groups.setdefault(_display(labels[role["id"]]).casefold(), []).append(role)
        for group in groups.values():
            if len(group) > 1:
                for role in group:
                    description = describe(role).strip()
                    if description:
                        labels[role["id"]] += " — " + description
    groups = {}
    for role in roles:
        groups.setdefault(_display(labels[role["id"]]).casefold(), []).append(role)
    for group in groups.values():
        if len(group) > 1:
            # Distinct cast records can have completely identical human metadata.
            for role in group:
                labels[role["id"]] += f" [{role['id']}]"
    return labels


def _terminal_width(output_stream):
    try:
        columns = os.get_terminal_size(output_stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        columns = 88
    return max(32, min(columns, 100))


def _resume_command(book_path):
    if book_path is None:
        return "epub-tts-narrator resolve-speakers BOOK_DIRECTORY"
    path = Path(book_path)
    command = "review" if path.suffix.lower() == ".json" else "resolve-speakers"
    return shlex.join(["epub-tts-narrator", command, str(path)])


def prompt_speaker_decisions(document, *, book_path=None, input_stream=None, output_stream=None,
                            jev_client=None):
    """Ask for every remaining speaker, preserving progress if input ends.

    This function is deliberately called only by the CLI after automatic review.
    Redirected standard input does not prevent use of the controlling terminal.
    Explicit input streams never fall back to a different input source.
    """
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid analysis before speaker decisions: " + "; ".join(errors[:12]))
    result = deepcopy(document)
    pending = unassigned_segments(result)
    if not pending:
        return result
    if input_stream is None and not getattr(sys.stdin, "isatty", lambda: False)():
        # Shell loops and pipelines can redirect stdin even when a person is
        # using a terminal. Ask on that terminal instead of consuming the pipe
        # or silently skipping the required speaker choices.
        with ExitStack() as stack:
            try:
                terminal_input = stack.enter_context(open("/dev/tty", "r", encoding="utf-8"))
                terminal_output = (output_stream if output_stream is not None else
                                   stack.enter_context(open("/dev/tty", "w", encoding="utf-8")))
            except OSError:
                pass
            else:
                return prompt_speaker_decisions(
                    document, book_path=book_path, input_stream=terminal_input,
                    output_stream=terminal_output, jev_client=jev_client)
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    color = (getattr(output_stream, "isatty", lambda: False)()
             and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ)
    styles = {"context": "\x1b[36m", "target": "\x1b[1;33m"}
    def say(message="", *, style=None):
        rendered = _display(message)
        # Add only our own terminal controls, after sanitizing untrusted content.
        if color and style and rendered:
            rendered = styles[style] + rendered + "\x1b[0m"
        print(rendered, file=output_stream, flush=True)
    width = _terminal_width(output_stream)
    def prose(message, indent="", *, style=None):
        # Display wrapping only: saved source text and character offsets remain exact.
        for paragraph in _display(message).splitlines() or [""]:
            wrapped = textwrap.fill(paragraph, width=width, initial_indent=indent,
                                    subsequent_indent=indent, replace_whitespace=False)
            for line in wrapped.splitlines() or [""]:
                say(line, style=style)
    if not getattr(input_stream, "isatty", lambda: False)():
        say(f"Speaker selection incomplete: {len(pending)} spoken passage(s) need a voice, but no interactive terminal is available. Progress will be saved.")
        say("Finish speaker selection from a terminal with: " + _resume_command(book_path))
        return result

    characters = {row["id"]: row for row in result["cast"]["characters"]}
    roles = result["cast"]["roles"]
    roles_by_id = {role["id"]: role for role in roles}
    labels = _role_labels(roles, characters, result["source"]["book"])
    blocks = result["source"]["blocks"]
    positions = {block["id"]: index for index, block in enumerate(blocks)}
    segments_by_block = {}
    for segment in result["segments"]:
        segments_by_block.setdefault(segment["block_id"], []).append(segment)
    def context(entries, *, tail=False):
        for neighbor, start, end in entries:
            for name, excerpt in _context_runs(neighbor, segments_by_block[neighbor["id"]],
                                              labels, start, end, tail=tail):
                prose(name, indent="  ", style="context")
                prose(excerpt, indent="  ", style="context")
                say()
    decisions = []
    prose(f"Automatic speaker assignment is finished. {len(pending)} passage(s) still need a voice.")
    prose("Enter accepts the best guess. O shows all voices. Q saves progress and stops.")
    prose("Comma-separated numbers mean those characters speak these SAME words together.")
    for number, item in enumerate(pending, 1):
        position = positions[item["block_id"]]
        block = blocks[position]
        say()
        say("─" * width)
        prose(f"Passage {number}/{len(pending)} — {result['source']['book']['title']}")
        chapter = next(row for row in result["source"]["chapters"] if row["id"] == block["chapter_id"])
        prose(chapter["title"])
        before = [(neighbor, 0, len(neighbor["text"])) for neighbor in blocks[max(0, position - 2):position]
                  if neighbor["chapter_id"] == block["chapter_id"]]
        after = [(neighbor, 0, len(neighbor["text"])) for neighbor in blocks[position + 1:position + 3]
                 if neighbor["chapter_id"] == block["chapter_id"]]
        if block["text"][:item["start"]]:
            before.append((block, 0, item["start"]))
        if block["text"][item["end"]:]:
            after.insert(0, (block, item["end"], len(block["text"])))
        say()
        context(before, tail=True)
        prose(f"PASSAGE TO ASSIGN — {item['end'] - item['start']} characters", style="target")
        prose(item["text"], indent="> ", style="target")
        say()
        context(after)
        prose("Your choice changes only the marked passage; surrounding assignments stay as shown.")
        if jev_client is not None:
            say("\nRanking likely speakers with JEV…")
        ranked = rank_speakers(result, item, jev_client=jev_client)
        suggestions = []
        for suggestion in ranked:
            if suggestion["role_id"] in roles_by_id and suggestion["role_id"] not in {
                    existing["role_id"] for existing in suggestions}:
                suggestions.append(suggestion)
            if len(suggestions) == 3:
                break
        shortlist = [roles_by_id[suggestion["role_id"]] for suggestion in suggestions]
        expanded = False
        def show_menu():
            say()
            if expanded:
                say("All voices")
                for choice, role in enumerate(roles, 1):
                    prose(f"{choice}. {labels[role['id']]}", indent="  ")
                say("  B. Back to likely speakers")
            elif shortlist:
                say("Most likely speakers")
                for choice, (role, suggestion) in enumerate(zip(shortlist, suggestions), 1):
                    best = " — best guess (Enter)" if choice == 1 else ""
                    prose(f"{choice}. {labels[role['id']]}{best}", indent="  ")
                    if suggestion.get("reason"):
                        prose(suggestion["reason"], indent="     ")
            else:
                say("No likely speaker found. Choose O to see all voices.")
            if not expanded:
                say(f"  O. Other — show all {len(roles)} voices")
            say("  Q. Save and stop")
        show_menu()
        stop = False
        while True:
            controls = "B = back, Q = save and stop" if expanded else (
                "Enter = 1, O = Other, Q = save and stop" if shortlist else "O = Other, Q = save and stop")
            print(f"Speaker number(s), {controls}: ", end="", file=output_stream, flush=True)
            try:
                answer = input_stream.readline()
            except (EOFError, KeyboardInterrupt, OSError):
                answer = ""
            if answer == "":
                say("\nInput ended before speaker selection finished. Progress will be saved; this run is incomplete.")
                stop = True
                break
            answer = answer.strip().casefold()
            if answer == "q":
                stop = True
                break
            if answer in {"o", "other"}:
                expanded = True
                show_menu()
                continue
            if answer == "b" and expanded:
                expanded = False
                show_menu()
                continue
            visible_roles = roles if expanded else shortlist
            if answer == "" and not expanded and shortlist:
                answer = "1"
            parts = [part.strip() for part in answer.split(",")]
            if (not all(part.isdecimal() and len(part) <= 8 for part in parts)
                    or any(not 1 <= int(part) <= len(visible_roles) for part in parts)):
                prose("Choose a displayed number or " + ("B to return to likely speakers." if expanded else "O for other voices.")
                      + " Q saves and stops.")
                prose("Enter selects a voice only from the likely-speaker list.")
                continue
            choices = [visible_roles[int(part) - 1] for part in parts]
            if len({role["id"] for role in choices}) != len(choices):
                say("Choose each speaker only once.")
                continue
            if len(choices) > 1 and (any(role["kind"] != "dialogue" for role in choices)
                                    or len({role["voice_id"] for role in choices}) != len(choices)):
                say("Simultaneous speech requires distinct dialogue roles with distinct voices; narration cannot be a group member.")
                continue
            before = deepcopy(item)
            if len(choices) == 1:
                role = choices[0]
                item.update(role_id=role["id"], voice_id=role["voice_id"])
                item.pop("simultaneous_speakers", None)
                if role["kind"] == "narration" and item["kind"] in {"dialogue", "thought", "letter"}:
                    item["kind"] = "narration"
                elif role["kind"] == "dialogue" and item["kind"] not in {"dialogue", "thought", "letter"}:
                    item["kind"] = "dialogue"
            else:
                item.update(role_id=None, voice_id=None,
                            simultaneous_speakers=[{"role_id": role["id"], "voice_id": role["voice_id"]} for role in choices])
                if item["kind"] not in {"dialogue", "thought", "letter"}:
                    item["kind"] = "dialogue"
            selected = "; ".join(labels[role["id"]] for role in choices)
            item["attribution"] = {"basis": "user", "reason": "User selected " + selected + ".",
                                   "evidence": deepcopy(before["attribution"]["evidence"]),
                                   "candidates": [], "review_required": False}
            decisions.append({"segment_id": item["id"], "block_id": item["block_id"],
                              "start": item["start"], "end": item["end"], "text_fingerprint": fingerprint(item["text"]),
                              "selected_roles": [role["id"] for role in choices],
                              "selected_voices": [role["voice_id"] for role in choices],
                              "previous_attribution": before["attribution"], "previous_kind": before["kind"],
                              "kind": item["kind"], "simultaneous": len(choices) > 1})
            say("Assigned to " + selected + (" together." if len(choices) > 1 else "."))
            break
        if stop:
            break

    remaining = unassigned_segments(result)
    if decisions:
        covered = {row["segment_id"] for row in decisions}
        covered.update(row["block_id"] for row in decisions
                       if not any(item["block_id"] == row["block_id"] for item in remaining))
        for issue in result["issues"]:
            if not issue["resolved"] and issue["kind"] in ATTRIBUTION_ISSUE_KINDS and issue["target"] in covered:
                issue.update(resolved=True, resolution={"method": "speaker_decision", "basis": "user"})
        provenance = result["provenance"]
        if "speaker_decisions" in provenance:
            provenance.setdefault("speaker_decision_history", []).append(deepcopy(provenance["speaker_decisions"]))
        provenance["speaker_decisions"] = {"version": 1, "source_fingerprint": fingerprint(result["source"]),
                                          "cast_fingerprint": fingerprint(result["cast"]),
                                          "input_script_fingerprint": fingerprint(document["segments"]),
                                          "output_script_fingerprint": fingerprint(result["segments"]),
                                          "decisions": decisions, "remaining_unassigned_segments": len(remaining)}
    result["readiness"] = readiness(result)
    if result["readiness"]["errors"]:
        raise ValueError("Invalid speaker decisions: " + "; ".join(result["readiness"]["errors"][:12]))
    say(f"Speaker decisions: {len(decisions)} assigned; {len(remaining)} unassigned spoken passage(s) remain.")
    if remaining:
        say("Speaker selection incomplete. Finish from a terminal with: " + _resume_command(book_path))
    return result
