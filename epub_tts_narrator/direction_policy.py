"""Which rules a book's line directions were made with, and whether they are current.

``VERSION`` names how lines are directed today: the direction prompt, reading
the narration's cues for each line (``speech_cues``) and compiling them. Bump
it whenever that changes, so rerunning ``analyze BOOK`` refreshes books made
with the old rules (``direction_refresh_command.refresh_outdated_directions``).

A book's directions are current when a whole-book direction refresh recorded
this version (``provenance.direction_refreshes[].policy``) or its narrator
read did (``provenance.narrator_read.direction_policy``). They are outdated
when they come from these rules at an older version: a narrator read or a
direction refresh without this version's record. A consensus analysis that
was never refreshed keeps its own strategy's directions (``consensus_only``);
``refresh-directions`` gives it these rules on request.

A direction a person typed (``delivery.basis == "user"``, set by
``attribution_edits.edit_direction``) is theirs: refreshes keep it.
"""

VERSION = 1


def current(document):
    """Whether ``document``'s line directions were made with the current rules."""
    provenance = document.get("provenance") or {}
    for record in provenance.get("direction_refreshes") or []:
        if isinstance(record, dict) and record.get("policy") == VERSION and record.get("chapters") is None:
            return True
    narrator = provenance.get("narrator_read")
    return isinstance(narrator, dict) and narrator.get("direction_policy") == VERSION


def outdated(document):
    """Whether these rules made ``document``'s directions at an older version."""
    provenance = document.get("provenance") or {}
    return not current(document) and bool(provenance.get("narrator_read") or provenance.get("direction_refreshes"))


def consensus_only(document):
    """Whether the directions are a never-refreshed consensus analysis's own."""
    provenance = document.get("provenance") or {}
    return not (provenance.get("narrator_read") or provenance.get("direction_refreshes"))


def typed(segment):
    """Whether a person typed this segment's direction."""
    delivery = segment.get("delivery")
    return isinstance(delivery, dict) and delivery.get("basis") == "user"
