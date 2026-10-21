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
