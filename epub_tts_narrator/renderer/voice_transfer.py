"""`import-voices BOOK --from OTHER`: reuse another book's chosen voices for the same characters.

A re-analysis (a cleaner EPUB, a new strategy) gets new voice IDs, so voices
are matched by character, never by ID: the narration voice to the narration
voice, and a character's voice to the other book's voice whose character has
the same name (else exactly one voice sharing a name or alias). Anything
ambiguous or unmatched is reported and left for ``cast``. Derived voices
(aging, conditions) are skipped: they are cloned from their main voice.

Each match goes through ``workflow.import_voice`` with the other book's
selected reference recording and its verified transcript, so it is normalized
and transcribed again, and a voice this book already chose itself is never
replaced. The voice's book-local speech traits (``voice-traits.json``) come
too, unless this book set its own.
"""

from copy import deepcopy

from .bundle import load_bundle


def _names(bundle):
    """voice id -> (primary names, all names) of the characters using it; narration as "narrator"."""
    found = {}
    for character in bundle["cast"]["characters"]:
        names = found.setdefault(character["voice_id"], (set(), set()))
        names[0].add(character["name"].casefold())
        names[1].update(name.casefold() for name in [character["name"], *character.get("aliases", [])])
    return found


def _narrator_voice(bundle):
    return next((role["voice_id"] for role in bundle["cast"]["roles"] if role["id"] == "narrator_main"), None)


def match_voices(target, source):
    """([(target voice, source voice, label)], [(target voice, label, reason)]) by character."""
    derived = {voice["id"] for voice in target["cast"]["voices"] if voice.get("derived_from")}
    source_derived = {voice["id"] for voice in source["cast"]["voices"] if voice.get("derived_from")}
    theirs = {voice: names for voice, names in _names(source).items() if voice not in source_derived}
    ours = _names(target)
    shown = {}
    for character in target["cast"]["characters"]:
        shown.setdefault(character["voice_id"], []).append(character["name"])
    narrator, their_narrator = _narrator_voice(target), _narrator_voice(source)
    matches, unmatched = [], []
    for voice in target["cast"]["voices"]:
        voice_id = voice["id"]
        if voice_id in derived:
            continue
        primary, every = ours.get(voice_id, (set(), set()))
        label = ", ".join(sorted(shown.get(voice_id, []))) or ("narrator" if voice_id == narrator else voice_id)
        if voice_id == narrator and their_narrator is not None and not primary:
            matches.append((voice_id, their_narrator, "narrator"))
            continue
        exact = [other for other, names in theirs.items() if primary & names[0]]
        loose = [other for other, names in theirs.items() if every & names[1]]
        if len(exact) == 1:
            matches.append((voice_id, exact[0], label))
        elif not exact and len(loose) == 1:
            matches.append((voice_id, loose[0], label))
        elif voice_id == narrator and their_narrator is not None and not exact and not loose:
            # A first-person narrator's voice is the narration voice in both books.
            matches.append((voice_id, their_narrator, label))
        else:
            unmatched.append((voice_id, label, "several voices share its names" if exact or loose
                              else "no character of that name"))
    return matches, unmatched


def import_voices(bundle, config, global_config, source_directory, engine, verifier, *, dry_run=False,
                  output=print):
    """Import matched voices; returns {"imported", "kept", "missing", "unmatched"} counts."""
    from .book_context import resolve_book_config, voice_store
    from .speech_traits import save_speech_traits, speech_trait_overrides
    from .workflow import import_voice

    source = load_bundle(source_directory)
    if source["directory"] == bundle["directory"]:
        raise ValueError("--from must name another book")
    source_config = resolve_book_config(source, deepcopy(global_config))
    theirs, ours = voice_store(source, source_config), voice_store(bundle, config)
    their_traits = speech_trait_overrides(source, deepcopy(global_config))
    our_traits = speech_trait_overrides(bundle, deepcopy(global_config))
    matches, unmatched = match_voices(bundle, source)
    counts = {"imported": 0, "kept": 0, "missing": 0, "unmatched": len(unmatched)}
    for voice_id, their_id, label in matches:
        existing = ours.load(voice_id)
        if existing is not None and not existing.get("inherited"):
            output(f"  kept     {label}: this book already chose its voice")
            counts["kept"] += 1
            continue
        saved = theirs.load(their_id)
        if saved is None:
            output(f"  missing  {label}: no voice chosen for it in {source['directory']}")
            counts["missing"] += 1
            continue
        if existing is not None and existing.get("reference_sha256") and existing["reference_sha256"] == saved.get("reference_sha256"):
            output(f"  kept     {label}: already uses that voice (inherited)")
            counts["kept"] += 1
            continue
        if dry_run:
            output(f"  would import {label}: {saved['reference_audio']}")
            counts["imported"] += 1
            continue
        result = import_voice(bundle, voice_id, saved["reference_audio"], saved["reference_text"],
                              config, engine, verifier, verification=saved.get("verification"))
        if their_id in their_traits and voice_id not in our_traits:
            save_speech_traits(bundle, config, voice_id, their_traits[their_id])
        output(f"  imported {label}: {result['reference_audio']}")
        counts["imported"] += 1
    for voice_id, label, reason in unmatched:
        output(f"  skipped  {label}: {reason}; choose it with cast")
    return counts
