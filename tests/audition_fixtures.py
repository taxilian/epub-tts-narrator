"""Prepared original audition text for offline renderer fixtures only."""

from copy import deepcopy


SCRIPT = (
    "The last boat reached the harbor before sunrise. I watched the lanterns move across the water "
    "and counted every passenger as they stepped ashore. We had enough blankets for everyone, "
    "and the kitchen fire was already burning."
)


def script_response():
    return {"text": SCRIPT, "reason": "An original calm audition for this offline character fixture."}


def prepare_bundle(bundle):
    """Model the per-character script preparation completed before real casting."""
    prepared = deepcopy(bundle)
    prepared["audition_scripts"] = {
        voice["id"]: {**script_response(), "basis": "casting"}
        for voice in prepared["cast"]["voices"]
    }
    return prepared


def prepare_document(document):
    """Persist fixture scripts through normal bundle export and integrity checks."""
    prepared = deepcopy(document)
    prepared["provenance"]["voice_samples"] = {
        "scripts": {voice["id"]: {**script_response(), "basis": "casting"}
                    for voice in prepared["cast"]["voices"]},
    }
    return prepared


def give_lines(bundle, voice_id, *, role_id=None):
    """Give ``voice_id`` lines in a loaded bundle, as a real book would; casting offers only voices with lines.

    With ``role_id`` the role's existing lines move to the voice (what
    reassigning that role's voice means in a real book); otherwise one copy
    of the book's first line is spoken by it.
    """
    rows = [row for chapter in bundle["chapters"] for row in chapter.get("rows", [])]
    if role_id is not None:
        for row in rows:
            if row.get("role_id") == role_id:
                row["voice_id"] = voice_id
        return bundle
    line = deepcopy(rows[0])
    line.update(id=line["id"] + "-" + voice_id, voice_id=voice_id)
    bundle["chapters"][0]["rows"].append(line)
    return bundle
