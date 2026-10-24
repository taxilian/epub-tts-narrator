from copy import deepcopy

from ..spoken_text import MAX_DELIVERY_CHARACTERS, effective_delivery
from .bundle import chunk_speakers


def performance_direction(chunk, voice_id):
    scene_direction = effective_delivery(chunk["segment"])
    traits_by_voice = chunk.get("speech_traits")
    if traits_by_voice is None:
        return scene_direction
    if not isinstance(traits_by_voice, dict):
        raise ValueError("Speech traits bound to a passage must be a voice mapping")
    traits = traits_by_voice.get(voice_id, "")
    if traits:
        from ..speech_traits import validate_speech_traits

        traits = validate_speech_traits(traits)
    direction = "; ".join(value for value in (traits, scene_direction) if value)
    if len(direction) > MAX_DELIVERY_CHARACTERS:
        raise ValueError(
            f"Combined speech traits and delivery exceeds {MAX_DELIVERY_CHARACTERS} characters; "
            "shorten the stable traits or passage direction"
        )
    return direction


def bind_plan_speech_traits(plan, cast):
    from ..speech_traits import speech_traits

    result = deepcopy(plan)
    profiles = {profile["id"]: profile for profile in cast["voices"]}

    def bind(chunk):
        chunk.pop("speech_traits", None)
        turns = chunk.get("paragraph_turns")
        if turns is not None:
            for turn in turns:
                bind(turn)
            return
        bound = {}
        for speaker in chunk_speakers(chunk):
            voice_id = speaker["voice_id"]
            profile = profiles.get(voice_id)
            if profile is None:
                raise ValueError(f"Voice ID is not in the bundle cast: {voice_id}")
            traits = speech_traits(profile)
            if traits:
                bound[voice_id] = traits
        if bound:
            chunk["speech_traits"] = bound

    for chapter in result["chapters"]:
        for chunk in chapter["chunks"]:
            bind(chunk)
    return result
