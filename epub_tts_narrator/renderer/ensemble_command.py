"""A person sets up, changes or declines a group's ensemble (``COLLECTIVE_VOICES.md`` §5.5).

The cast screen (``cast_tui``) drives this; nothing here runs on its own and
nothing chooses for the person. A plan is computed from the book as saved, so
publication can refuse a book that changed meanwhile (manifest receipts):

* ``plan_ensemble``: the group's own voice stays the lead; ``voices`` is the
  total count a person chose (2–8, the proposal is only the default), filled
  by named characters' voices the person picked, then by pool voices. Pool
  voices already in the ensemble are kept in order; new ones get stable IDs
  and a description that is the group's with one varied trait each, so each
  is designed separately (never cloned) and a person can edit it in casting.
* ``plan_decline``: records ``mode: single`` so the proposal is never asked
  again; pool voices nobody uses any more are removed.

Both publish through ``analysis_revision`` scope ``ensembles``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

from ..ensembles import MAX_VOICES, MODES, RECIPES, ensemble, proposed_voices

# One varied trait per pool voice, so members are told apart when designed.
VARIATIONS = (
    "a little older, with a lower pitch",
    "a little younger and brighter",
    "rougher, with a breathier edge",
    "smoother, with a slower pace",
    "a little higher, with a quicker pace",
    "deeper and more resonant",
    "lighter, with a slightly nasal tone",
)


@dataclass
class EnsemblePlan:
    path: Path
    hashes: dict
    candidate: dict
    name: str
    summary: str
    added: tuple = ()  # new pool voices, described properly once saved (``describe_new_voices``)


def _load(bundle):
    from ..analysis_revision import _completed_directory
    path = Path(bundle["directory"]).resolve()
    document, _, _ = _completed_directory(path)
    # Only this book's own state is bound: later books continuing it are only re-linked to its new
    # history, worked out from their files under their locks, so their own saves (an analysis of
    # book 4 running meanwhile) never make this book's change stale.
    receipt = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()}
    return path, document, receipt


def _group(document, voice_id):
    character = next((item for item in document["cast"]["characters"] if item.get("voice_id") == voice_id), None)
    if character is None:
        raise ValueError("Only a character's own voice can lead an ensemble")
    narration = {role["voice_id"] for role in document["cast"]["roles"] if role.get("kind") == "narration"}
    if voice_id in narration:
        raise ValueError("The narration voice cannot be an ensemble")
    return character


def view(bundle, voice_id):
    """What the ensemble screen shows: the proposal, the current ensemble and choosable characters."""
    cast = bundle["cast"]
    voices = {voice["id"]: voice for voice in cast["voices"]}
    character = next((item for item in cast["characters"] if item.get("voice_id") == voice_id), None)
    if character is None:
        raise ValueError("Only a character's own voice can lead an ensemble")
    group = character.get("group") if isinstance(character.get("group"), dict) else None
    chosen = ensemble(voices.get(voice_id))
    named = {item["voice_id"]: item["name"] for item in cast["characters"]
             if item["voice_id"] != voice_id and not voices.get(item["voice_id"], {}).get("ensemble")}
    members = list((chosen or {}).get("members", []))
    default = (1 + len(members) if chosen and members else
               (group or {}).get("proposed_voices") or proposed_voices((group or {}).get("size"), (group or {}).get("scale")) or 3)
    return {"name": character["name"], "group": deepcopy(group), "ensemble": deepcopy(chosen),
            "voices": max(2, min(MAX_VOICES, default)),
            "mode": (chosen or {}).get("mode") or (group or {}).get("proposed_mode") or "unison",
            "mix": (chosen or {}).get("mix"),
            "named_members": [member for member in members if member in named],
            "choosable": [(voice, name) for voice, name in named.items()],
            "evidence": [item["quote"] for item in (group or {}).get("evidence", [])][:3]}


def _pool_id(lead, number):
    return f"{lead}-pool-{number}"


def _with_ensemble(document, voice_id, members, mode, mix):
    """``document`` with ``voice_id`` leading ``members``; stale pool voices removed, new ones added."""
    candidate = deepcopy(document)
    voices = candidate["cast"]["voices"]
    lead = next(voice for voice in voices if voice["id"] == voice_id)
    character = _group(candidate, voice_id)
    record = {"members": [member for member, _ in members], "mode": mode, "basis": "user"}
    if mix is not None:
        record["mix"] = mix
    lead["ensemble"] = record
    keep = {member for member, _ in members}
    candidate["cast"]["voices"] = [voice for voice in voices
                                   if voice.get("member_of") != voice_id or voice["id"] in keep]
    existing = {voice["id"] for voice in candidate["cast"]["voices"]}
    for member, description in members:
        if member not in existing:
            candidate["cast"]["voices"].append({
                "id": member, "description": description, "traits": [], "member_of": voice_id})
    del character
    return candidate


def plan_ensemble(bundle, voice_id, *, voices, mode, mix=None, named=()):
    """Plan an ensemble of ``voices`` voices (lead included) for the group whose voice is ``voice_id``."""
    if mode not in MODES or mode == "single":
        raise ValueError("Choose unison, alternate, pool, staggered or layered (decline keeps the group's own voice)")
    if mix is not None and mix not in RECIPES:
        raise ValueError(f"Unknown mix {mix!r}; use one of {', '.join(RECIPES)}")
    if type(voices) is not int or not 2 <= voices <= MAX_VOICES:
        raise ValueError(f"An ensemble has 2 to {MAX_VOICES} voices, its lead included")
    named = list(dict.fromkeys(named))
    if len(named) + 1 > voices:
        raise ValueError("More named characters were chosen than the ensemble has voices")
    path, document, receipt = _load(bundle)
    character = _group(document, voice_id)
    by_id = {voice["id"]: voice for voice in document["cast"]["voices"]}
    for member in named:
        owner = next((item for item in document["cast"]["characters"] if item["voice_id"] == member), None)
        if owner is None or member == voice_id or "ensemble" in by_id.get(member, {}):
            raise ValueError(f"{member} is not another individual character's voice")
    lead = by_id[voice_id]
    current = [member for member in (ensemble(lead) or {}).get("members", [])
               if by_id.get(member, {}).get("member_of") == voice_id]
    pool = []
    number = 2
    while len(named) + len(pool) + 1 < voices:
        if current:
            member = current.pop(0)
            pool.append((member, by_id[member]["description"]))
            continue
        member = _pool_id(voice_id, number)
        number += 1
        if member in by_id:
            continue
        variation = VARIATIONS[(len(pool)) % len(VARIATIONS)]
        description = (f"{lead['description'].rstrip('. ')}. Another of {character['name']}: {variation}.")
        pool.append((member, description))
    members = [(member, None) for member in named] + pool
    candidate = _with_ensemble(document, voice_id, members, mode, mix)
    summary = (f"{character['name']}: {voices} voices ({len(named)} named, {len(pool)} pool), {mode}"
               + (f", mixed {mix}" if mix else ""))
    added = tuple(member for member, _ in pool if member not in by_id)
    return EnsemblePlan(path, receipt, candidate, character["name"], summary, added)


def plan_decline(bundle, voice_id):
    """Keep the group in its own voice and stop proposing it (``mode: single``)."""
    path, document, receipt = _load(bundle)
    character = _group(document, voice_id)
    candidate = deepcopy(document)
    voices = candidate["cast"]["voices"]
    next(voice for voice in voices if voice["id"] == voice_id)["ensemble"] = {
        "members": [], "mode": "single", "basis": "user"}
    candidate["cast"]["voices"] = [voice for voice in voices if voice.get("member_of") != voice_id]
    return EnsemblePlan(path, receipt, candidate, character["name"],
                        f"{character['name']}: one voice (the group's own); not proposed again")


def describe_new_voices(plan, session, *, config_path=None, progress=print):
    """Have the analysis model describe a saved plan's new pool voices so they sound apart.

    ``session`` is a casting session on the saved book. Each new voice gets its
    group's context and the other voices' descriptions (``voice_prompt_refresh``
    ``group_voice``) and a fresh audition script. Without a usable analysis
    model the simple variations stay; ``r`` rewrites a voice later. True when
    they were described.
    """
    from ..provider import ProviderError
    if not plan.added:
        return False
    try:
        session.refresh_prompts(voice_ids=list(plan.added), config_path=config_path)
    except (ProviderError, ValueError, OSError) as error:
        progress(f"Kept simple descriptions for {plan.name}'s new voices ({error}); r rewrites one with the model")
        return False
    progress(f"Described {len(plan.added)} new voice(s) for {plan.name} so they sound apart; e edits one")
    return True


def apply(plan, *, progress=print):
    """Publish a planned ensemble change (backups, descendants rebound)."""
    from ..analysis_revision import revise_directories
    from ..voice_profile_guard import require_valid_profiles
    require_valid_profiles(plan.path)
    return revise_directories({plan.path: plan.candidate}, expected_manifest_hashes=plan.hashes,
                              scope="ensembles", progress=progress)
