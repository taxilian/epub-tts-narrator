"""Line-oriented per-finding review with staged, validated manual edits."""

from __future__ import annotations

from copy import deepcopy
import sys

from .attribution_edits import apply_edits, edit_direction, edit_review, edit_speaker
from .attribution_verification import restrict_findings
from .speaker_decisions import _role_labels


def print_finding(finding, index):
    """Show complete decision context wherever a finding needs an answer."""
    print(f"{index}. chapter {finding['chapter_id']} segment {finding['segment_id']} "
          f"(block {finding['block_id']})")
    if finding["review_only"]:
        print("   needs review")
    else:
        print(f"   from {finding['from_label']} to {finding['to_label']}")
    print(f"   {finding['text']}")
    for item in finding["evidence"]:
        print(f"   evidence {item['block_id']}: {item['quote']}")


def _readline(prompt):
    print(prompt, end="", flush=True)
    try:
        answer = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        answer = ""
    if answer == "":
        raise EOFError from None
    return answer.rstrip("\r\n")


def _speaker_label(segment, labels):
    if segment.get("simultaneous_speakers"):
        role_ids = [speaker["role_id"] for speaker in segment["simultaneous_speakers"]]
    else:
        role_ids = [segment["role_id"]] if segment.get("role_id") else []
    return ", ".join(labels[role_id] for role_id in role_ids) if role_ids else "unassigned"


def _manual_edit(document, proposal):
    """Return a confirmed edit, or None on explicit cancellation; EOF aborts staging."""
    roles = list(document["cast"]["roles"])
    roles_by_id = {role["id"]: role for role in roles}
    labels = _role_labels(roles, {row["id"]: row for row in document["cast"]["characters"]},
                          document["source"]["book"])
    staged = deepcopy(proposal)

    while True:
        print(f"Proposed speaker: {_speaker_label(staged, labels)}")
        print("Existing cast:")
        for index, role in enumerate(roles, 1):
            print(f"   {index}. {labels[role['id']]} ({role['id']}, {role['kind']})")
        answer = _readline("Speaker number, U for unresolved, Enter to keep, or /cancel: ")
        command = answer.strip().casefold()
        if command == "/cancel":
            return None
        if not command:
            break
        try:
            if command in {"u", "unresolved"}:
                staged = edit_speaker(staged, [], roles_by_id, labels)
            else:
                choice = int(command)
                if not 1 <= choice <= len(roles):
                    raise ValueError(f"Speaker number must be from 1 to {len(roles)}")
                staged = edit_speaker(staged, [roles[choice - 1]["id"]], roles_by_id, labels)
        except ValueError as error:
            print(error)
            continue
        break

    while True:
        current = staged["delivery"]["direction"] or "(none)"
        print(f"Proposed delivery direction: {current}")
        answer = _readline("New direction (240 characters max), Enter to keep, '-' to clear, or /cancel: ")
        command = answer.strip().casefold()
        if command == "/cancel":
            return None
        if not answer:
            break
        try:
            staged = edit_direction(staged, "" if command == "-" else answer)
        except ValueError as error:
            print(error)
            continue
        break

    while True:
        current = "yes" if staged["attribution"]["review_required"] else "no"
        answer = _readline(f"Review required is {current}; [y] yes, [n] no, Enter to keep, or /cancel: ")
        command = answer.strip().casefold()
        if command == "/cancel":
            return None
        if not command:
            break
        if command not in {"y", "n"}:
            print("Please enter y, n, Enter, or /cancel.")
            continue
        try:
            staged = edit_review(staged, command == "y")
        except ValueError as error:
            print(error)
            continue
        break

    while True:
        print(f"Staged speaker: {_speaker_label(staged, labels)}")
        print(f"Staged delivery direction: {staged['delivery']['direction'] or '(none)'}")
        print(f"Staged review required: {'yes' if staged['attribution']['review_required'] else 'no'}")
        answer = _readline("[y] confirm this manual edit, or /cancel to discard it: ")
        command = answer.strip().casefold()
        if command == "y":
            return staged
        if command == "/cancel":
            return None
        print("Please enter y to confirm or /cancel to discard the staged edit.")


def edit_findings(document, audited, found):
    """Review findings and return revised document, proposal count, and manual-edit count."""
    originals = {segment["id"]: segment for segment in document["segments"]}
    proposals = {segment["id"]: segment for segment in audited["segments"]}
    accepted, manual, finished = [], {}, False
    for index, finding in enumerate(found, 1):
        segment_id = finding["segment_id"]
        while True:
            print_finding(finding, index)
            try:
                answer = _readline(f"Decision {index}/{len(found)}: [y] accept proposal / [e] manual edit / "
                                   "[n] leave unchanged / [Q] finish: ")
            except EOFError:
                print("\nInput ended; only findings confirmed so far will be applied.")
                answer = "q"
            command = answer.strip().casefold()
            if command == "q":
                finished = True
                break
            if command == "n":
                break
            if command == "y":
                accepted.append(segment_id)
                break
            if command != "e":
                print("Please enter y, e, n, or Q.")
                continue
            try:
                edited = _manual_edit(document, proposals[segment_id])
            except EOFError:
                print("\nInput ended; discarding the unconfirmed manual edit. "
                      "Only findings confirmed so far will be applied.")
                finished = True
                break
            if edited is None:
                print("Manual edit discarded.")
                continue
            if edited == originals[segment_id]:
                print("Manual edit matches the original; this finding is left unchanged.")
            elif edited == proposals[segment_id]:
                accepted.append(segment_id)
                print("Proposal accepted.")
            else:
                manual[segment_id] = edited
                print("Manual edit confirmed.")
            break
        if finished:
            break
    revised, applied = restrict_findings(document, audited, accepted)
    if manual:
        revised = apply_edits(revised, manual)
    return revised, len(applied), len(manual)
