"""Several clips of one designed voice, joined into that voice's saved reference.

Drama 3 clones a voice more closely from more reference speech (its technical
report: speaker similarity 0.605 from one clip, 0.633 from three), and Fish
recommends two or three clips of 15–20 seconds. Drama 3 was trained both on
several clips joined into one reference and on separate reference blocks
(Appendix A, Figure 8). A joined reference needs nothing new from any engine,
worker or transport: it is one longer recording with its exact transcript.

Casting and automatic voice creation ask the voice designer for the first clip
as before (the voice's audition script), then for up to two more clips of the
same voice speaking other book excerpts by the same character, or, when the
book has too few, scripts the text model wrote in that character's style while
the auditions were prepared (``audition_scripts``). Every clip is
verified like any reference, and the clips are joined in order with a short
pause. The joined recording passes the same hard acoustic gates as any take;
its words are the verified clips' words, so it is not transcribed again (a
long recording mostly measures the recognizer: invented closing words, other
spellings of names). Its report records ``basis: joined_verified_clips`` and
each clip's audio checksum and take request.

Only the Gemini designer keeps one voice across several scripts (one hosted
voice per creation request); other designers would design a new voice for each
clip, so their voices keep one clip. A character without enough clean book
speech and no prepared invented scripts keeps fewer clips. An extra clip that fails verification is left out,
and a joined recording that fails an acoustic gate leaves the first clip alone
as the reference. Once the hosted voice is deleted (an interrupted creation),
missing clips cannot come from the same voice, so creation starts again with a
fresh voice (``FreshVoiceRequired``).
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import wave

from .audio_naming import descriptive_audio_filename
from .common import digest, file_hash, read_json
from .workflow import RenderBlocked, SpeechVerificationFailed, verified_take


PAUSE_SECONDS = 0.5


class FreshVoiceRequired(RenderBlocked):
    """The hosted voice that spoke the first clips is gone; another clip cannot match them."""


def keeps_one_voice(designer_identity):
    """Whether this designer keeps one voice across several scripts in one creation."""
    return isinstance(designer_identity, dict) and designer_identity.get("engine") == "gemini_voice_design"


def plan_scripts(config, designer_identity, bundle, voice_id, canonical_id, first_source):
    """The extra clips' scripts for one creation request, as ``[{"text", "source"}]``.

    Empty when the setting asks for one clip, the designer cannot keep one voice,
    there is no book to choose from, or the character has neither more suitable
    speech nor prepared invented scripts.
    """
    wanted = (config.get("casting") or {}).get("reference_clips", 3) - 1
    if (wanted < 1 or not isinstance(bundle, dict) or not {"cast", "chapters"} <= set(bundle)
            or not keeps_one_voice(designer_identity)):
        return []
    from .samples import prepared_extra_samples, select_extra_samples
    present = next((voice["id"] for voice in bundle["cast"]["voices"]
                    if canonical_id(voice["id"]) == canonical_id(voice_id)), None)
    if present is None:
        return []
    exclude = (first_source or {}).get("segment_ids", []) if isinstance(first_source, dict) else []
    samples = select_extra_samples(bundle, present, canonical_id=canonical_id,
                                   exclude_segments=exclude, count=wanted)
    # Book speech first; invented scripts prepared with the auditions fill the rest.
    samples += prepared_extra_samples(bundle, present, canonical_id=canonical_id, count=wanted - len(samples))
    return [{"text": sample["reference_text"], "source": deepcopy(sample["source"])} for sample in samples]


def _hosted_voice_removed(directory):
    return any((Path(directory) / name).exists() or (Path(directory) / name).is_symlink()
               for name in ("gemini-voice-cleanup.json", "gemini-voice-cleanup-request.json"))


def _generator(designer, prompt, text, directory, label):
    """Speak ``text`` in the creation folder (where the hosted voice lives), then hand the take over.

    The Gemini designer keys its hosted voice by the folder a take is written
    to, so every clip is first written beside the first clip's takes.
    """
    from .voice_bindings import design_voice

    def generate(path, seed):
        staged = Path(directory) / descriptive_audio_filename("reference-clip-take", label, f"seed-{seed}",
                                                              suffix=".wav")
        staged.unlink(missing_ok=True)  # This creation's own leftover from an interrupted attempt.
        metadata = design_voice(designer, prompt, text, staged, seed=seed)
        os.replace(staged, path)
        return metadata
    return generate


def join_wavs(paths, output, *, pause_seconds=PAUSE_SECONDS):
    """Concatenate mono PCM WAVs of one format with silence between them (stdlib only)."""
    params = None
    frames = []
    for path in paths:
        with wave.open(str(path), "rb") as source:
            current = source.getparams()
            if current.nchannels != 1:
                raise ValueError("Reference clips must be mono to be joined")
            if params is not None and (current.sampwidth, current.framerate) != (params.sampwidth, params.framerate):
                raise ValueError("Reference clips must share one sample format and rate to be joined")
            params = current
            frames.append(source.readframes(source.getnframes()))
    silence = b"\0" * (round(params.framerate * pause_seconds) * params.sampwidth)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name("." + output.name + ".partial")
    with wave.open(str(staged), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(params.sampwidth)
        target.setframerate(params.framerate)
        target.writeframes(silence.join(frames))
    os.replace(staged, output)
    return output


def add_clips(first_wav, first_accepted, scripts, *, designer, prompt, directory, request, verifier, config,
              label, notify=print):
    """The joined reference for this creation, or the first clip alone.

    Returns ``(wav, accepted, clips)``: the recording and acceptance record to
    publish, and the clip provenance for the voice's creation metadata (empty
    when only the first clip is used).
    """
    directory = Path(directory)
    clips = [{"index": 1, "wav": Path(first_wav), "accepted": first_accepted,
              "text": first_accepted["report"].get("reference_text", request["text"]),
              "source": request.get("sample_source")}]
    removed = _hosted_voice_removed(directory)
    for index, script in enumerate(scripts, 2):
        clip_directory = directory / f"clip-{index}"
        if removed and not (clip_directory / "accepted.json").is_file():
            raise FreshVoiceRequired("The temporary Gemini voice that spoke this voice's first reference clip "
                                     "was removed before its other clips were made")
        clip_request = dict(request, clip=index, text=script["text"], sample_source=script["source"])
        notify(f"  Reference clip {index} of {len(scripts) + 1} for {request['voice_id']}")
        try:
            wav, accepted = verified_take(
                script["text"], _generator(designer, prompt, script["text"], directory, f"{label}-clip-{index}"),
                clip_directory, clip_request, verifier, config, notify=notify, purpose="reference")
        except SpeechVerificationFailed:
            notify(f"  Reference clip {index} failed verification; the voice is saved without it.")
            continue
        if accepted.get("engine_metadata", {}).get("description_applied") is not True:
            raise RenderBlocked("Saved reference clip has no evidence that the voice description was applied")
        clips.append({"index": index, "wav": Path(wav), "accepted": accepted,
                      "text": accepted["report"].get("reference_text", script["text"]), "source": script["source"]})
    if len(clips) == 1:
        return first_wav, first_accepted, []
    joined_directory = directory / "joined"
    joined = joined_directory / descriptive_audio_filename(
        "joined-reference-source", label, f"{len(clips)}-clips", suffix=".wav")
    hashes = [file_hash(clip["wav"]) for clip in clips]
    marker = joined.with_suffix(".json")
    if not (joined.is_file() and marker.is_file() and read_json(marker).get("clips_sha256") == hashes):
        join_wavs([clip["wav"] for clip in clips], joined)
        from .common import write_json
        write_json(marker, {"clips_sha256": hashes, "pause_seconds": PAUSE_SECONDS})
    text = " ".join(clip["text"].strip() for clip in clips)
    # Every word was verified clip by clip, and joining only adds silence
    # between unchanged PCM; transcribing the long recording again mostly
    # measures the recognizer (invented closing words, other name spellings).
    # The joined recording still passes every hard acoustic gate.
    from .quality import acoustic_check
    try:
        metrics, reasons = acoustic_check(joined, config.get("quality"))
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        metrics, reasons = {}, [f"invalid_audio: {exc}"]
    if reasons:
        notify("  The joined reference clips failed an audio check (" + ", ".join(reasons)
               + "); the voice is saved with its first clip only.")
        return first_wav, first_accepted, []
    provenance = [{"index": clip["index"], "reference_text": clip["text"], "audio_sha256": hashes[number],
                   "request_hash": clip["accepted"].get("request_hash"),
                   **({"sample_source": deepcopy(clip["source"])} if clip["source"] is not None else {})}
                  for number, clip in enumerate(clips)]
    joined_request = dict(request, text=text, joined_clips=hashes, joined_pause_seconds=PAUSE_SECONDS)
    joined_request.pop("sample_source", None)
    report = {"passed": True, "purpose": "reference", "basis": "joined_verified_clips",
              "expected_text": text, "reference_text": text, "reasons": [], "metrics": metrics,
              "clips": [{"index": clip["index"], "audio_sha256": clip["audio_sha256"],
                         "request_hash": clip["request_hash"]} for clip in provenance],
              "pause_seconds": PAUSE_SECONDS}
    accepted = {"report": report, "seed": clips[0]["accepted"].get("seed"),
                "request_hash": digest(joined_request), "audio_sha256": file_hash(joined),
                "engine_metadata": {**clips[0]["accepted"].get("engine_metadata", {}),
                                    "joined_reference_clips": len(clips)}}
    return joined, accepted, provenance


def creation_metadata(clips):
    """Fields recording a joined reference's clips in a voice's creation metadata."""
    if not clips:
        return {}
    return {"reference_clips": clips, "reference_clip_pause_seconds": PAUSE_SECONDS,
            "reference_clips_sha256": digest([clip["audio_sha256"] for clip in clips])}
