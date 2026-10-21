"""Optional terminal decisions after automatic speaker resolution is exhausted."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shlex
import sys

from .contracts import readiness, validate_document
from .pipeline import fingerprint
from .speaker_completion import ATTRIBUTION_ISSUE_KINDS, unassigned_segments


def _display(value):
    # Book text and model explanations are display data, never terminal controls.
    return "".join(char for char in str(value) if char in "\n\t" or char.isprintable())


def _role_label(role, characters):
    character = characters.get(role.get("character_id"))
    name = character["name"] if character else "Narrator" if role["kind"] == "narration" else role["description"]
    return f"{name} ({role['kind']}; {role['id']})"


def _findings(document, segment, labels):
    audit = document["provenance"].get("speaker_completion", {})
    case = next((row for row in audit.get("cases", []) if row.get("block_id") == segment["block_id"]), None)
    if case is None or not case.get("rounds"):
        return []
    rows = []
    for vote in case["rounds"][-1].get("votes", []):
        names = []
        for item in vote.get("proposed_segments", []):
            if item["start"] < segment["end"] and item["end"] > segment["start"]:
                role_ids = ([row["role_id"] for row in item["simultaneous_speakers"]]
                            if item.get("simultaneous_speakers") else [item.get("role_id")])
                for role_id in role_ids:
                    label = labels.get(role_id, "unresolved")
                    if label not in names:
                        names.append(label)
        response = vote.get("raw_response")
        response = response if isinstance(response, dict) else {}
        reason = response.get("reason", "No valid source-bound decision.")
        rows.append(f"{vote['reviewer_id']}: {', '.join(names) or 'no valid decision'} — {str(reason)[:280]}")
    return rows


def _resume_command(book_path):
    if book_path is None:
        return "epub-audiobook resolve-speakers BOOK_DIRECTORY"
    path = Path(book_path)
    command = "review" if path.suffix.lower() == ".json" else "resolve-speakers"
    return shlex.join(["epub-audiobook", command, str(path)])


def prompt_speaker_decisions(document, *, book_path=None, input_stream=None, output_stream=None):
    """Ask once per remaining passage in a terminal; preserve an orderly draft otherwise.

    This function is deliberately called only by the CLI after automatic review.
    Neither the analysis library nor a noninteractive caller reads standard input.
    """
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid analysis before speaker decisions: " + "; ".join(errors[:12]))
    result = deepcopy(document)
    pending = unassigned_segments(result)
    if not pending:
        return result
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stderr if output_stream is None else output_stream
    def say(message=""):
        print(_display(message), file=output_stream, flush=True)
    if not getattr(input_stream, "isatty", lambda: False)():
        say(f"Automatic review left {len(pending)} unassigned spoken passage(s). Saving a draft; no terminal input is available.")
        say("Review them from a terminal with: " + _resume_command(book_path))
        return result

    characters = {row["id"]: row for row in result["cast"]["characters"]}
    roles = result["cast"]["roles"]
    labels = {role["id"]: _role_label(role, characters) for role in roles}
    blocks = result["source"]["blocks"]
    positions = {block["id"]: index for index, block in enumerate(blocks)}
    decisions = []
    say(f"Automatic review is finished. {len(pending)} passage(s) still need a speaker decision.")
    say("Choose a role number. Comma-separated numbers mean those characters speak these SAME words together.")
    say("Enter or p leaves this passage pending; q stops and keeps decisions already made.")
    for number, item in enumerate(pending, 1):
        position = positions[item["block_id"]]
        block = blocks[position]
        say(f"\nPassage {number}/{len(pending)} — {result['source']['book']['title']} — {item['block_id']}:{item['start']}:{item['end']}")
        for offset in range(max(0, position - 1), min(len(blocks), position + 2)):
            neighbor = blocks[offset]
            if neighbor["chapter_id"] != block["chapter_id"]:
                continue
            label = "Source block" if offset == position else "Context"
            text = neighbor["text"] if offset == position else neighbor["text"][:1000]
            say(f"{label} [{neighbor['id']}]: {text}")
        say(f"Passage to assign: {item['text']}")
        candidates = [labels[role_id] for role_id in item["attribution"]["candidates"] if role_id in labels]
        if candidates:
            say("Earlier candidates: " + "; ".join(candidates))
        for finding in _findings(result, item, labels):
            say(finding)
        for choice, role in enumerate(roles, 1):
            say(f"  {choice}. {labels[role['id']]}")
        stop = False
        while True:
            print("Speaker number(s), Enter/p to leave pending, q to stop: ", end="", file=output_stream, flush=True)
            try:
                answer = input_stream.readline()
            except (EOFError, KeyboardInterrupt, OSError):
                answer = ""
            if answer == "":
                say("\nInput ended. Keeping decisions made so far; remaining passages stay pending.")
                stop = True
                break
            answer = answer.strip().casefold()
            if answer == "q":
                stop = True
                break
            if answer in {"", "p"}:
                break
            parts = [part.strip() for part in answer.split(",")]
            if (not all(part.isdecimal() and len(part) <= 8 for part in parts)
                    or any(not 1 <= int(part) <= len(roles) for part in parts)):
                say("Enter a listed role number, comma-separated numbers, p, or q.")
                continue
            choices = [roles[int(part) - 1] for part in parts]
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
        say("The remaining passages stay pending. Continue from a terminal with: " + _resume_command(book_path))
    return result
