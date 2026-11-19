"""Crowd murmur: a group's voices talking quietly under a passage (``COLLECTIVE_VOICES.md`` §5.7).

A segment's ``murmur`` (``FORMAT.md``) names a group and the segment its
span ends with. When a person has cast that group's ensemble, the render:

1. makes **murmur takes**: each of the group's voices (at most 8) says a few
   short conversational sentences from a fixed, project-owned set (never book
   text), or, with ``from_line``, the book line many people say. Every take
   passes ordinary text verification (``verified_take``, purpose
   ``reference``: no chapter history);
2. builds one **bed loop** per group and text choice: each voice's takes in a
   row with seeded gaps, voices offset from one another, summed, low-passed
   and levelled. A loop of generic sentences must *fail* recognition: if the
   primary recognizer hears a run of ``MAX_HEARD_RUN`` or more of its words,
   it is filtered harder once and, if still heard, refused (the passage renders
   without it). A ``from_line`` loop says the book's own words, so it is not
   checked;
3. mixes the loop under the span while the chapter is composed: about
   ``LEVEL_DB`` under the span's speech, ducked under every spoken frame, with
   the span's shape (steady, swell, fade). Not a frame of the chapter moves, so
   takes, verification, censorship, history and ``sync.json`` are untouched.

Beds are cached under ``render-work/murmurs/KEY/`` and their keys and hashes
join the chapter identity; a remote chapter service renders without them.
"""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import re

from .common import digest, read_json, write_json

RECIPE_VERSION = 1
MAX_VOICES = 8
LOOP_SECONDS = 20.0
MAX_HEARD_RUN = 3
LEVEL_DB = {"low": -24.0, "medium": -18.0}
DUCK_DB = 5.0
SHAPES = ("steady", "swell", "fade", "swell_fade")
CUTOFFS_HZ = (2800.0, 1800.0)  # the first try, then the harder filter

# Ordinary conversational sentences, written for this project (never book text).
SENTENCES = (
    "Did you see who came in just now?",
    "I told him it would never work.",
    "Keep your voice down, they can hear us.",
    "Over there, by the far door.",
    "Have you heard anything more about it?",
    "She said they would be here by now.",
    "I don't like the look of this at all.",
    "Move a little, I can't see a thing.",
    "Who was that, do you know him?",
    "It was the same last time, remember?",
    "Wait, listen, something is happening.",
    "I thought it would take much longer.",
    "Not now, we can talk about it later.",
    "Well, that is a surprise, isn't it?",
    "Somebody should tell them, really.",
    "Look at that, did you ever see such a thing?",
)
_WORDS = re.compile(r"[\w']+")


def lane_texts(lane, count=4):
    """The sentences lane ``lane`` says (distinct per lane, deterministic)."""
    return [SENTENCES[(lane * count + index) % len(SENTENCES)] for index in range(count)]


def bed_key(voices, texts, sample_rate):
    """The cache key of one bed: its recipe, voices (by recording), texts and rate."""
    return digest({"kind": "murmur_bed", "version": RECIPE_VERSION, "voices": voices, "texts": texts,
                   "sample_rate": sample_rate, "loop_seconds": LOOP_SECONDS})


def plan_spans(rows, cast):
    """``(spans, skipped)`` for one chapter's rows: each murmur with its group's cast voices.

    A span whose group has no ensemble a person cast is skipped (listed, never
    blocking): its voices are not designed yet.
    """
    from ..ensembles import ensemble, lanes_of
    characters = {character["id"]: character for character in cast["characters"]}
    voices = {voice["id"]: voice for voice in cast["voices"]}
    by_id = {row["id"]: row for row in rows}
    spans, skipped = [], []
    for row in rows:
        value = row.get("murmur")
        if not isinstance(value, dict):
            continue
        group = characters.get(value.get("group"))
        lead = voices.get((group or {}).get("voice_id"))
        if lead is None or not (ensemble(lead) or {}).get("members"):
            skipped.append({"segment_id": row["id"], "group": (group or {}).get("name", value.get("group"))})
            continue
        line = by_id.get(value.get("from_line")) if value.get("from_line") else None
        spans.append({"start": row["id"], "until": value["until"], "group": group["id"], "name": group["name"],
                      "shape": value.get("shape", "steady"), "level": value.get("level", "low"),
                      "voices": lanes_of(lead)[:MAX_VOICES],
                      "from_line": line["id"] if line else None,
                      "line_text": _spoken(line) if line else None})
    return spans, skipped


def _spoken(row):
    from ..spoken_text import build_spoken
    return build_spoken(row)[0].strip().strip("\"“”‘’'")


# Bed building -------------------------------------------------------------------

def _samples(path):
    from .lane_mix import from_bytes
    from .stored_audio import open_wave
    with open_wave(path) as stream:
        return from_bytes(stream.readframes(stream.getnframes()))


def _fraction(*parts):
    return int(digest(list(parts))[:12], 16) / float(16 ** 12)


def _lowpass(values, hz, sample_rate):
    from .lane_mix import _lowpass as one_pole
    # Two passes: a gentler slope than one, still deterministic and length preserving.
    return one_pole(one_pole(values, hz, sample_rate), hz, sample_rate)


def build_loop(lanes, key, sample_rate, cutoff):
    """Mix each lane's takes (lists of PCM16 arrays) into one levelled loop of ``LOOP_SECONDS``."""
    frames = int(LOOP_SECONDS * sample_rate)
    total = array("d", bytes(8 * frames))
    for number, takes in enumerate(lanes):
        cursor = int(_fraction(key, "start", number) * 2.0 * sample_rate)
        index = 0
        while cursor < frames and takes:
            take = takes[index % len(takes)]
            for offset, value in enumerate(take):
                if cursor + offset >= frames:
                    break
                total[cursor + offset] += value
            cursor += len(take) + int((0.2 + 0.6 * _fraction(key, "gap", number, index)) * sample_rate)
            index += 1
    filtered = _lowpass(total, cutoff, sample_rate)
    from .lane_mix import gated_rms
    level = gated_rms(filtered, sample_rate) or 1.0
    scale = 3000.0 / level
    return array("h", (max(-32768, min(32767, round(value * scale))) for value in filtered))


def heard_run(transcript_words, texts):
    """The longest run of consecutive words the recognizer heard from the murmur sentences."""
    sentences = [[word.casefold() for word in _WORDS.findall(text)] for text in texts]
    heard = [word.casefold() for word in transcript_words]
    best = 0
    for sentence in sentences:
        for start in range(len(heard)):
            for begin in range(len(sentence)):
                length = 0
                while (start + length < len(heard) and begin + length < len(sentence)
                       and heard[start + length] == sentence[begin + length]):
                    length += 1
                best = max(best, length)
    return best


def _transcript_words(verifier, path):
    result = verifier.transcribe(path)
    text = result.get("text") if isinstance(result, dict) else str(result or "")
    if not text and isinstance(result, dict):
        text = " ".join(segment.get("text", "") for segment in result.get("segments", []))
    return _WORDS.findall(text or "")


def ensure_bed(span, voices, engine, verifier, config, work, *, notify=print):
    """The bed loop for ``span`` (``{"key", "audio_sha256"}``), made once and cached; None if refused.

    ``voices`` maps voice ID to the render's loaded voice assets. With no
    engine (``assemble``), only an already cached bed is used.
    """
    sample_rate = config["render"]["sample_rate"]
    texts = ([[span["line_text"]] for _ in span["voices"]] if span.get("line_text")
             else [lane_texts(number) for number, _ in enumerate(span["voices"])])
    identity = [voices[voice_id]["asset_hash"] for voice_id in span["voices"]]
    key = bed_key(identity, texts, sample_rate)
    folder = Path(work) / "murmurs" / key
    record_path = folder / "bed.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("status") == "refused":
            return None
        from .stored_audio import audio_hash
        wav = folder / record["audio_file"]
        if audio_hash(wav) == record["audio_sha256"]:
            return {"key": key, "audio_sha256": record["audio_sha256"]}
    if engine is None:
        return None
    from .workflow import verified_take
    notify(f"Recording the murmur of {span['name']} ({len(span['voices'])} voices)")
    lanes, takes_record = [], []
    for number, (voice_id, lane) in enumerate(zip(span["voices"], texts)):
        voice = voices[voice_id]
        lane_takes = []
        for text in lane:
            request = {"kind": "murmur_take", "version": RECIPE_VERSION, "engine": engine.identity,
                       "voice_asset_hash": voice["asset_hash"], "text": text, "base_seed": config["render"]["seed"]}
            directory = Path(work) / "murmur-takes" / digest(request)

            def generate(path, seed, v=voice, t=text):
                return engine.synthesize(t, v, path, seed=seed, direction="talking quietly to a neighbour")
            wav, accepted = verified_take(text, generate, directory, request, verifier, config,
                                          notify=notify, purpose="reference")
            lane_takes.append(_samples(wav))
            takes_record.append({"voice_id": voice_id, "text_sha256": digest(text),
                                 "audio_sha256": accepted["audio_sha256"]})
        lanes.append(lane_takes)
    folder.mkdir(parents=True, exist_ok=True)
    flat = [sentence for lane in texts for sentence in lane]
    for attempt, cutoff in enumerate(CUTOFFS_HZ):
        loop = build_loop(lanes, key, sample_rate, cutoff)
        name = f"murmur-bed-{span['group']}-{key[:12]}-try{attempt + 1}.wav"
        _write(folder / name, loop, sample_rate)
        if span.get("line_text"):
            run = 0
        else:
            run = heard_run(_transcript_words(verifier, folder / name), flat)
        if run < MAX_HEARD_RUN:
            from .stored_audio import audio_hash
            record = {"format": "audiobook-renderer-murmur", "version": RECIPE_VERSION, "key": key,
                      "status": "ok", "audio_file": name, "audio_sha256": audio_hash(folder / name),
                      "cutoff_hz": cutoff, "heard_run": run, "takes": takes_record}
            write_json(record_path, record)
            return {"key": key, "audio_sha256": record["audio_sha256"]}
        notify(f"The murmur of {span['name']} was still intelligible ({run} words heard); filtering it harder")
    write_json(record_path, {"format": "audiobook-renderer-murmur", "version": RECIPE_VERSION, "key": key,
                             "status": "refused", "heard_run": run, "takes": takes_record})
    notify(f"The murmur of {span['name']} stays intelligible; the passage renders without it")
    return None


def _write(path, samples, sample_rate):
    import wave
    from .lane_mix import to_bytes
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        stream.writeframes(to_bytes(samples))


def prepare_chapter_murmurs(chapter, voices, engine, verifier, config, work, *, notify=print):
    """Attach ``chapter["murmurs"]`` (path-free bed keys) for the chapter's planned spans."""
    attached = []
    for span in chapter.get("murmur_spans", []):
        if any(voice_id not in voices for voice_id in span["voices"]):
            continue
        bed = ensure_bed(span, voices, engine, verifier, config, work, notify=notify)
        if bed is not None:
            attached.append({key: span[key] for key in ("start", "until", "group", "shape", "level")}
                            | {"bed": bed, "recipe_version": RECIPE_VERSION})
    if attached:
        chapter["murmurs"] = attached
    else:
        chapter.pop("murmurs", None)
    return attached


# Mixing under a composed chapter ----------------------------------------------------

def _envelope(position, shape):
    if shape == "swell":
        return 0.3 + 0.7 * min(1.0, position / 0.4)
    if shape == "fade":
        return 1.0 if position < 0.5 else 1.0 - 0.8 * (position - 0.5) / 0.5
    if shape == "swell_fade":
        return min(_envelope(position, "swell"), _envelope(position, "fade"))
    return 1.0


def apply_beds(wav_path, segments, murmurs, work, sample_rate):
    """Mix every attached bed loop under its span of the composed chapter, in place; frames unchanged."""
    import wave
    from .lane_mix import from_bytes, gated_rms, to_bytes
    from .stored_audio import audio_hash
    frames_by_segment = {}
    for segment in segments:
        start, end = segment["chapter_start_frame"], segment["chapter_end_frame"]
        known = frames_by_segment.setdefault(segment["segment_id"], [start, end])
        known[0], known[1] = min(known[0], start), max(known[1], end)
    with wave.open(str(wav_path), "rb") as stream:
        chapter = from_bytes(stream.readframes(stream.getnframes()))
    applied = []
    block = max(1, int(0.05 * sample_rate))
    for murmur in murmurs:
        if murmur["start"] not in frames_by_segment or murmur["until"] not in frames_by_segment:
            continue
        start = frames_by_segment[murmur["start"]][0]
        end = frames_by_segment[murmur["until"]][1]
        if end <= start:
            continue
        folder = Path(work) / "murmurs" / murmur["bed"]["key"]
        record = read_json(folder / "bed.json")
        wav = folder / record["audio_file"]
        if audio_hash(wav) != murmur["bed"]["audio_sha256"]:
            raise ValueError("A murmur bed changed after it was made; render again to rebuild it")
        loop = _samples(wav)
        speech = gated_rms(chapter[start:end], sample_rate) or 3000.0
        gain = speech * 10 ** (LEVEL_DB.get(murmur["level"], LEVEL_DB["low"]) / 20) / 3000.0
        duck = 10 ** (-DUCK_DB / 20)
        edge = int(0.3 * sample_rate)
        length = end - start
        for block_start in range(start, end, block):
            block_end = min(end, block_start + block)
            piece = chapter[block_start:block_end]
            loud = math.sqrt(sum(value * value for value in piece) / len(piece)) if piece else 0.0
            factor = duck if loud >= 328 else 1.0
            for frame in range(block_start, block_end):
                position = (frame - start) / length
                ramp = min(1.0, (frame - start) / edge, (end - frame) / edge) if edge else 1.0
                value = loop[(frame - start) % len(loop)] * gain * factor * _envelope(position, murmur["shape"]) * ramp
                chapter[frame] = max(-32768, min(32767, chapter[frame] + round(value)))
        applied.append({"start": murmur["start"], "until": murmur["until"], "group": murmur["group"],
                        "chapter_start_frame": start, "chapter_end_frame": end})
    with wave.open(str(wav_path), "wb") as stream:
        stream.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        stream.writeframes(to_bytes(chapter))
    return applied
