"""Revising voice-design descriptions from a person's instructions (the cast screen's ``i``).

A person types what should change ("remove every mention of accent; the cue
adds it") and the analysis text model rewrites the chosen voices'
descriptions accordingly. Each voice is sent with its current description
and the accent cue the application adds by itself when the voice is
designed (``speech_traits.with_accent``), so the model can tell which
wording would conflict with it. Nothing is saved here: ``revise`` returns
the proposed descriptions for the person to review, and saving one is an
ordinary casting edit (``CastingSession.update_prompt``) that changes the
draft only, never a recording.

Requests go through the book's request cache, so the same instructions on
the same descriptions ask nothing new.
"""

from __future__ import annotations

from .description_revision_prompts import REVISE

BATCH = 20  # voices per request


def _plain(description, limit):
    from .casting import _validate_text
    text = " ".join(description.split()) if isinstance(description, str) else ""
    _validate_text(text, "Voice description", limit)
    if any(character in text for character in "[]<>`{}"):
        raise ValueError("a description must be plain direction without brackets or markup")
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        raise ValueError("a description must not be wrapped in quotation marks")
    return text


def revise(requests, voices, instructions, *, progress=lambda text: None):
    """Proposed descriptions: ``{voice_id: new description}`` for the voices the model changed.

    ``voices`` are ``[{"voice_id", "name", "description", "accent"}]``;
    unchanged voices are left out of the result.
    """
    from .voice_prompts import MAX_PROMPT_CHARACTERS
    instructions = " ".join(str(instructions or "").split())
    if not instructions:
        raise ValueError("Type what should change in the descriptions")
    voices = [voice for voice in voices if voice.get("description")]
    if not voices:
        raise ValueError("None of the chosen voices has a description to revise")
    proposals = {}
    for start in range(0, len(voices), BATCH):
        batch = voices[start:start + BATCH]
        ids = {f"v{index + 1}": voice for index, voice in enumerate(batch)}
        payload = {"mode": "voice_description_revision", "v": 1, "instructions": instructions,
                   "voices": [[short, voice["name"], voice["description"], voice.get("accent") or ""]
                              for short, voice in ids.items()]}

        def validate(response, ids=ids):
            rows = response.get("d") if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"d":[[id,description]]}')
            found, problems = {}, []
            for row in rows:
                if not (isinstance(row, list) and len(row) == 2 and row[0] in ids):
                    problems.append('each changed voice must be [id, description] with an id from "voices"')
                    continue
                try:
                    found[ids[row[0]]["voice_id"]] = _plain(row[1], MAX_PROMPT_CHARACTERS)
                except ValueError as error:
                    problems.append(f"voice {row[0]}: {error}")
            if problems:
                raise ValueError("; ".join(dict.fromkeys(problems)))
            return found
        if len(voices) > BATCH:
            progress(f"Revising descriptions {start + 1}–{start + len(batch)} of {len(voices)}")
        proposals.update(requests.run(REVISE, payload, validate))
    current = {voice["voice_id"]: " ".join(voice["description"].split()) for voice in voices}
    return {voice_id: text for voice_id, text in proposals.items() if text != current[voice_id]}


def book_requests(bundle, config_path, *, progress):
    """The analysis model through the book's request cache, or None without ``provider.model``."""
    from epub_tts_narrator.config import load_config
    from epub_tts_narrator.pipeline import CachedRequests
    from epub_tts_narrator.provider import OpenAICompatibleClient
    from .accent_assignment import _book_cache
    analysis = load_config(config_path)
    if not analysis["provider"]["model"]:
        return None
    client = OpenAICompatibleClient(**analysis["provider"])
    progress(f"Revising voice descriptions with {client.model} at {client.base_url}; character names and voice "
             "descriptions are sent to this endpoint.")
    settings = analysis["analysis"]
    return CachedRequests(client, _book_cache(bundle, config_path), repair_attempts=settings["repair_attempts"],
                          max_request_chars=settings["max_request_chars"], progress=progress)
