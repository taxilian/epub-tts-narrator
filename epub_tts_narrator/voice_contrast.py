"""Deterministic textual contrast audit for declared voice profiles.

This compares only recognized audible dimensions in each profile's structured
traits. It is advisory text analysis of profile records: it never measures how
synthesized voices actually sound, cannot prove two voices are confusable, and
never blocks rendering or demands recasting where source evidence genuinely
supports similar voices. Missing dimensions are unknown, not differences, and
no traits are invented to resolve a finding.
"""

VERSION = 1

# Recognized audible dimensions, matched after conservative normalization:
# casefold, whitespace collapse, a small synonym map, and singular fallback.
_DIMENSIONS = {"pitch", "register", "timbre", "texture", "resonance", "weight",
               "pace", "rhythm", "accent", "articulation", "intonation", "volume"}
_SYNONYMS = {"tempo": "pace", "speed": "pace", "rate": "pace", "speaking rate": "pace",
             "tone": "timbre", "tone color": "timbre", "tone colour": "timbre",
             "vocal weight": "weight", "loudness": "volume"}


def _canonical_name(name):
    """Return the recognized dimension name, or None when unrecognized."""
    if not isinstance(name, str):
        return None
    key = " ".join(name.split()).casefold().rstrip(":.,;")
    key = _SYNONYMS.get(key, key)
    if key in _DIMENSIONS:
        return key
    if key.endswith("es") and key[:-2] in _DIMENSIONS:
        return key[:-2]
    if key.endswith("s") and key[:-1] in _DIMENSIONS:
        return key[:-1]
    return None


def _canonical_value(value):
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split()).casefold().strip(" .;:!?,")
    return collapsed or None


def known_dimensions(profile):
    """Return recognized dimension -> normalized declared value; absent names stay unknown."""
    result = {}
    for trait in profile.get("traits") or []:
        if not isinstance(trait, dict):
            continue
        name = _canonical_name(trait.get("name"))
        value = _canonical_value(trait.get("value"))
        if name and value:
            result.setdefault(name, value)
    return result


def contrast_findings(profiles, canonical_id=None):
    """Return advisory potential-confusion findings for distinct canonical voices.

    Pairs compare distinct canonical voices only; aliases of one canonical voice
    and shared-voice duplicates never pair. Identical known profiles, or fewer
    than two demonstrably different known dimensions, produce one finding per
    pair carrying voice IDs and dimension names only. Unrecognized or missing
    dimensions are unknown on both sides and never count as a difference.
    """
    groups = {}
    for profile in profiles:
        voice_id = profile.get("id")
        canonical = canonical_id(voice_id) if canonical_id is not None else voice_id
        groups.setdefault(canonical, []).append(voice_id)
    dimensions = {profile.get("id"): known_dimensions(profile) for profile in profiles}
    findings = []
    canonical_ids = sorted(groups, key=str)
    for index, left in enumerate(canonical_ids):
        for right in canonical_ids[index + 1:]:
            left_id, right_id = groups[left][0], groups[right][0]
            dims_left, dims_right = dimensions[left_id], dimensions[right_id]
            shared = sorted(set(dims_left) & set(dims_right))
            differing = [name for name in shared if dims_left[name] != dims_right[name]]
            matching = [name for name in shared if dims_left[name] == dims_right[name]]
            if shared and not differing:
                kind = "identical_known_profile"
            elif len(differing) < 2:
                kind = "insufficient_known_contrast"
            else:
                continue
            findings.append({"kind": kind, "voice_ids": [left_id, right_id],
                             "matching_dimensions": matching,
                             "differing_dimensions": differing})
    return findings
