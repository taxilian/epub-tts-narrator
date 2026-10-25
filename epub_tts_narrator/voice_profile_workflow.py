"""Publish profile-only updates while retaining the existing book and voice assets."""

import hashlib
import json
from pathlib import Path
import shlex

from .audition_scripts import FILENAME as AUDITION_SIDECAR
from .pipeline import CachedRequests, fingerprint
from .voice_profile_refresh import (apply_profile_updates, audition_sidecar_rebind,
                                    profiles_changed, refresh_voice_profiles)
from .voice_profile_guard import refresh_locks
from .voice_profile_state import ProfileCheckpoint, clear_state, read_state


def refresh_book_voice_profiles(directory, config_path=None, *, workers=1, progress=print):
    with refresh_locks(directory):
        return _refresh_locked(directory, config_path, workers=workers, progress=progress)


def _refresh_locked(directory, config_path=None, *, workers=1, progress=print):
    from .analysis_directory import _completed_directory, continuation_directories
    from .analysis_revision import related_directories, revise_directories
    from .config import load_config
    from .provider import OpenAICompatibleClient
    from .series import series_context
    from .voice_profile_refresh import VERSION, SYSTEM
    from .voice_prompt_refresh import _provider_identity

    directory = Path(directory).resolve()
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Voice profile refresh workers must be between one and five")
    config = load_config(config_path)
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
    original, registry, _ = _completed_directory(directory)
    directories = related_directories([directory])
    ancestors = continuation_directories(directory)
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
              for path in directories}
    previous = read_state(directory)
    if previous:
        source_hash, cast_hash = fingerprint(original["source"]), fingerprint(original["cast"])
        if source_hash != previous["binding"]["source_fingerprint"]:
            raise ValueError("Source changed during pending voice profile refresh; refusing stale proposals")
        if (previous.get("phase") == "publishing" and cast_hash == previous.get("expected_after")
                and original["provenance"].get("voice_profile_refresh") == previous.get("expected_audit")):
            for name, expected in previous.get("expected_sidecars", {}).items():
                if name != AUDITION_SIDECAR or (directory / name).is_symlink():
                    raise ValueError("Invalid refresh publication sidecar")
                if fingerprint(json.loads((directory / name).read_text(encoding="utf-8"))) != expected:
                    raise ValueError("Refresh publication sidecar changed; refusing to clear pending state")
            for path in directories:
                continuation_directories(path)
            clear_state(directory)
            return {"changed": cast_hash != previous["binding"]["cast_fingerprint"],
                    "directories": [str(path) for path in directories], "resumed": True}
        if cast_hash != previous["binding"]["cast_fingerprint"]:
            raise ValueError("Cast changed during pending voice profile refresh; refusing stale proposals")
    cache = Path(settings["cache_dir"] or directory.parent / ".epub-audiobook-cache").resolve()
    if any(cache == book or book in cache.parents for book in [directory, *ancestors, *directories]):
        raise ValueError("The cache directory must be outside the existing book and its series directories")
    client = OpenAICompatibleClient(**config["provider"])
    progress(f"Refreshing voice profiles with {client.model} at {client.base_url}; character profiles, "
             "casting context, and source evidence are sent to this endpoint.")
    progress(f"Request cache: {cache}")
    requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    bundle = {"source": original["source"], "cast": original["cast"],
              "manifest": {"source_fingerprint": fingerprint(original["source"]),
                           "provenance": original["provenance"]},
               "series_context": series_context(dict(registry, books=registry["books"][:-1]), original["source"])}
    binding = {"source_fingerprint": fingerprint(original["source"]),
               "cast_fingerprint": fingerprint(original["cast"]),
               "guidance_fingerprint": fingerprint(SYSTEM), "provider": _provider_identity(requests),
               "profile_version": VERSION, "voice_ids": [voice["id"] for voice in original["cast"]["voices"]],
               "series_fingerprint": fingerprint(bundle["series_context"])}
    resume = ["epub-tts-narrator", "refresh-voice-profiles", str(directory), "--workers", str(workers)]
    if config_path is not None:
        resume.extend(["--config", str(Path(config_path).resolve())])
    checkpoint = ProfileCheckpoint(directory, requests, binding, resume_command=shlex.join(resume))
    result = refresh_voice_profiles(bundle, checkpoint,
                                    workers=min(workers, 5, settings["attribution_workers"]),
                                    progress=progress)
    if not profiles_changed(original, result["profiles"]):
        checkpoint.publishing(original)
        clear_state(directory)
        progress("All voice profiles already match the refreshed guidance; no changes to publish.")
        return {"changed": False, "directories": [str(path) for path in directories]}
    revised = apply_profile_updates(original, result["profiles"], provider=result["provider"],
                                    guidance_fingerprint=result["guidance_fingerprint"])
    sidecar = directory / AUDITION_SIDECAR
    sidecars = {}
    if sidecar.exists() or sidecar.is_symlink():
        sidecars = {directory: {AUDITION_SIDECAR: audition_sidecar_rebind(
            sidecar, source_fingerprint=result["source_fingerprint"],
            cast_fingerprint_before=result["cast_fingerprint_before"],
            cast_fingerprint_after=fingerprint(revised["cast"]),
             voice_ids=[voice["id"] for voice in revised["cast"]["voices"]])}}
    checkpoint.publishing(revised, sidecars.get(directory))
    published = revise_directories({directory: revised}, directories=directories,
                                   expected_manifest_hashes=hashes, scope="voice-profiles",
                                    sidecars=sidecars, progress=progress)
    clear_state(directory)
    progress(f"Updated voice profiles in {directory}; original files and revision journal: "
             f"{published['backup_directory']}")
    return {"changed": True, **published}
