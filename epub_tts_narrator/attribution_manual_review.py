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


class FindingReview:
    """Decisions for each finding, for the full-screen review (no terminal I/O).

    A finding starts undecided. Accepting takes the verifier's proposal;
    leaving keeps the saved segment; speaker, direction and review-flag edits
    start from the proposal, exactly like the line review's manual edit. A
    new character (``new_character``) is staged as an unassigned passage and
    assigned through ``speaker_decisions`` when the review is finished, so its
    cast addition is a receipt-bound speaker decision (``finish``).
    """

    def __init__(self, document, audited, found):
        self.document, self.audited, self.found = document, audited, list(found)
        self.originals = {segment["id"]: segment for segment in document["segments"]}
        self.proposals = {segment["id"]: segment for segment in audited["segments"]}
        self.roles = list(document["cast"]["roles"])
        self.roles_by_id = {role["id"]: role for role in self.roles}
        self.characters = {row["id"]: row for row in document["cast"]["characters"]}
        self.labels = _role_labels(self.roles, self.characters, document["source"]["book"])
        from .speaker_decisions import _role_label
        self.names = {role["id"]: _role_label(role, self.characters, document["source"]["book"]) for role in self.roles}
        self.decisions = {}

    def segment_id(self, index):
        return self.found[index]["segment_id"]

    def status(self, index):
        """"undecided", "accept", "leave", "edit" or "new"."""
        return self.decisions.get(self.segment_id(index), {}).get("status", "undecided")

    def staged(self, index):
        """The segment as it would be published for this finding's current decision."""
        segment_id = self.segment_id(index)
        decision = self.decisions.get(segment_id)
        if decision is None or decision["status"] == "accept":
            return self.proposals[segment_id]
        if decision["status"] == "leave":
            return self.originals[segment_id]
        return decision["segment"]

    def _editable(self, index):
        decision = self.decisions.get(self.segment_id(index))
        return deepcopy(decision["segment"] if decision and decision["status"] in {"edit", "new"}
                        else self.proposals[self.segment_id(index)])

    def _stage(self, index, segment, **extra):
        segment_id = self.segment_id(index)
        if extra.get("status") == "new":
            self.decisions[segment_id] = {"segment": segment, **extra}
        elif segment == self.originals[segment_id]:
            self.decisions[segment_id] = {"status": "leave"}
        elif segment == self.proposals[segment_id]:
            self.decisions[segment_id] = {"status": "accept"}
        else:
            self.decisions[segment_id] = {"status": "edit", "segment": segment}

    def accept(self, index):
        self.decisions[self.segment_id(index)] = {"status": "accept"}

    def leave(self, index):
        self.decisions[self.segment_id(index)] = {"status": "leave"}

    def undo(self, index):
        self.decisions.pop(self.segment_id(index), None)

    def accept_undecided(self):
        for index in range(len(self.found)):
            if self.status(index) == "undecided":
                self.accept(index)

    def set_speaker(self, index, role_ids):
        """An existing-cast speaker (several: together; none: unresolved); ValueError if refused."""
        self._stage(index, edit_speaker(self._editable(index), list(role_ids), self.roles_by_id, self.labels))

    def set_direction(self, index, text):
        decision = self.decisions.get(self.segment_id(index), {})
        segment = edit_direction(self._editable(index), text)
        if decision.get("status") == "new":
            self._stage(index, segment, status="new", name=decision["name"], description=decision["description"])
        else:
            self._stage(index, segment)

    def set_review(self, index, required):
        segment = edit_review(self._editable(index), bool(required))
        self._stage(index, segment)

    def new_names(self):
        """Names of new characters staged so far, in first-use order."""
        return list(dict.fromkeys(decision["name"] for decision in self.decisions.values()
                                  if decision["status"] == "new"))

    def new_character(self, index, name, description):
        """Give this finding a character the cast lacks; another finding may reuse the same new name."""
        name, description = name.strip(), description.strip()
        if not name or not description:
            raise ValueError("A new character needs a name and a voice description")
        if any(row["name"].casefold() == name.casefold() for row in self.characters.values()):
            raise ValueError(f"{name} is already in the cast; choose them from the list instead")
        earlier = next((decision for decision in self.decisions.values()
                        if decision["status"] == "new" and decision["name"].casefold() == name.casefold()), None)
        if earlier is not None:
            name, description = earlier["name"], earlier["description"]
        segment = edit_speaker(self._editable(index), [], self.roles_by_id, self.labels)
        self._stage(index, segment, status="new", name=name, description=description)

    def counts(self):
        statuses = [self.status(index) for index in range(len(self.found))]
        return {status: statuses.count(status) for status in ("undecided", "accept", "edit", "new", "leave")}

    def finish(self):
        """``(candidate, speaker_input, proposals, manual)`` for publication.

        ``speaker_input`` is None without new characters (an ordinary
        attribution revision); otherwise it is the trusted pre-decision
        document and ``candidate`` adds the new characters through
        ``speaker_decisions.prompt_speaker_decisions``.
        """
        accepted = [segment_id for segment_id, decision in self.decisions.items() if decision["status"] == "accept"]
        edited = {segment_id: decision["segment"] for segment_id, decision in self.decisions.items()
                  if decision["status"] in {"edit", "new"}}
        revised, applied = restrict_findings(self.document, self.audited, accepted)
        if edited:
            revised = apply_edits(revised, edited)
        new = {segment_id: decision for segment_id, decision in self.decisions.items() if decision["status"] == "new"}
        if not new:
            return revised, None, len(applied), len(edited)
        from .speaker_decisions import prompt_speaker_decisions

        def chooser(view):
            decision = new.get(view["segment_id"])
            if decision is None:
                return {"action": "skip"}
            known = next((role["role_id"] for role in view["roles"]
                          if view["names"].get(role["role_id"], "").casefold() == decision["name"].casefold()), None)
            if known is not None:
                return {"action": "assign", "roles": [known]}
            return {"action": "new", "name": decision["name"], "description": decision["description"]}
        candidate = prompt_speaker_decisions(revised, chooser=chooser)
        return candidate, revised, len(applied), len(edited)
