"""Book pronunciation review and isolated listening previews with saved voices.

Auditions are never accepted audiobook clips or newly created voice references.
Only an explicit confirmation changes the book's pronunciation registry.
"""

from __future__ import annotations

from array import array
from copy import deepcopy
import math
from pathlib import Path
import re
import sys
import wave

from ..pronunciation_groups import entry_words, family_words, find_pronunciation, natural_spelling
from .audition import AudioPlayer, display
from .common import digest, file_hash, pcm_sha256, read_json, regular_file, write_json
from .run_locks import acquire_directory_lock


_FORMAT = "epub-tts-pronunciation-preview"
_DRAFT_FORMAT = "epub-tts-pronunciation-drafts"
_ESTIMATE_FORMAT = "epub-tts-pronunciation-cmu-estimates"
_SELECTED = object()
_MAX_SAMPLE = 360


def _pronunciation(say_as=None, phonemes=None):
    from ..pronunciations import validate_pronunciation
    if say_as is not None and phonemes is not None:
        raise ValueError("Choose --say-as or --phonemes, not both")
    if say_as is None and phonemes is None:
        return None
    return validate_pronunciation({"alphabet": "respelling" if say_as is not None else "cmu_arpabet",
                                   "value": say_as if say_as is not None else phonemes})


def _description(pronunciation):
    if not pronunciation:
        return "default pronunciation"
    prefix = "say as " if pronunciation["alphabet"] == "respelling" else "CMU phonemes "
    return prefix + pronunciation["value"]


def _preferred(row):
    """A pending guess must never displace an explicit current decision."""
    return (row.get("draft") or row.get("pronunciation") or
            (row.get("proposal") if row.get("status") == "pending" else None))


def _context(entry):
    evidence = entry.get("evidence", [])
    if evidence and isinstance(evidence[0], dict):
        return str(evidence[0].get("quote", "")).replace("\n", " ")[:180]
    return ""


def _show(rows, output, formatter=None):
    output(display(f"\nPronunciations — {sum(row['status'] == 'pending' for row in rows)} pending / {len(rows)} total"))
    for index, row in enumerate(rows, 1):
        extra = " — inherited" if row.get("inherited") else ""
        if row.get("case_conflict"):
            extra += " — conflicting capitalization choices"
        draft = row.get("draft")
        description = formatter(row) if formatter is not None else _description(_preferred(row))
        if draft:
            description += " (draft; not applied to the book)"
        elif row["status"] == "pending" and row.get("proposal"):
            description += " (guessed; not verified or applied)"
            if (row.get("proposal_origin_directory") and
                    row["proposal_origin_directory"] != row.get("origin_directory")):
                description += " (inherited proposal)"
        output(display(f"{index:3}. {row['word']} — {row['status']}{extra}; {description}"))
        if row.get("preparation_error"):
            output(display("     Audio unavailable: " + row["preparation_error"]))
        if row.get("reason"):
            output(display("     " + str(row["reason"]).replace("\n", " ")))
        if _context(row):
            output(display("     “" + _context(row) + "”"))


def _spoken_span(mapping, row, match):
    """Spoken-text span of an exact source match, or None when a replacement covers it."""
    absolute_start, absolute_end = row["start"] + match.start(), row["start"] + match.end()
    for span in mapping:
        if span["source_start"] <= absolute_start and absolute_end <= span["source_end"]:
            if span["kind"] == "source":
                return (span["spoken_start"] + absolute_start - span["source_start"],
                        span["spoken_start"] + absolute_end - span["source_start"])
            return span["spoken_start"], span["spoken_end"]
    return None


def _preview_row(row, word):
    """Copy of a row omitting only normalization edits overlapping an exact occurrence.

    Canonical rows are never mutated: the copy drops solely the analyzer edits
    that cover the target word, so the preview can map unambiguously while all
    other approved edits stay intact. Production rendering keeps the strict
    mapping guards on the canonical row.
    """
    from ..pronunciations import WORD_PATTERN
    record = row.get("speech_normalization") or {}
    edits = record.get("edits") or []
    if not edits:
        return row
    spans = [(match.start(), match.end()) for match in WORD_PATTERN.finditer(row["text"])
             if match.group() == word]
    kept = [edit for edit in edits
            if not any(edit["start"] < end and start < edit["end"] for start, end in spans)]
    if len(kept) == len(edits):
        return row
    copy = deepcopy(row)
    normalization = dict(copy.get("speech_normalization") or {})
    normalization["edits"] = kept
    copy["speech_normalization"] = normalization
    copy.pop("spoken_text", None)  # stale derived form; build_spoken recomputes it exactly
    return copy


def _sample_bounds(text, first, last):
    """Bound a real source excerpt at words, preferring sentence boundaries."""
    if len(text) <= _MAX_SAMPLE:
        return 0, len(text)
    boundaries = [0, *[match.end() for match in re.finditer(r"[.!?][\"'”’]?\s+", text)], len(text)]
    before = max(value for value in boundaries if value <= first)
    after = min(value for value in boundaries if value >= last)
    if after - before <= _MAX_SAMPLE:
        return before, after
    before = max(0, first - 110)
    if before:
        space = re.search(r"\s+", text[before:first])
        before = before + space.end() if space else first
    after = min(len(text), before + _MAX_SAMPLE)
    if after < last:
        before, after = first, min(len(text), max(last, first + _MAX_SAMPLE))
    if after < len(text):
        boundary = text.rfind(" ", last, after)
        after = boundary if boundary >= last else last
    return before, after


def _preview_audio(path):
    """Reject physically empty/silent/corrupt audio, without judging wording."""
    with wave.open(str(regular_file(path)), "rb") as source:
        rate, frames = source.getframerate(), source.getnframes()
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("Pronunciation preview must be mono PCM16 audio")
        if not rate or not 0 < frames <= rate * 120:
            raise ValueError("Pronunciation preview duration is invalid")
        pcm = source.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("Pronunciation preview audio is truncated")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(value * value for value in samples) / frames) / 32768
    if rms < 10 ** (-60 / 20):
        raise ValueError("Pronunciation preview is silent or too quiet to audition")
    return {"frames": frames, "sample_rate": rate, "duration_seconds": frames / rate}


def _remove_redundant_wavs(folder, state):
    for name, field, hasher in (("raw.wav", "raw_sha256", file_hash),
                                ("normalized.wav", "pcm_sha256", pcm_sha256)):
        path = folder / name
        if path.exists() or path.is_symlink():
            if hasher(regular_file(path)) != state.get(field):
                raise ValueError("Redundant pronunciation preview audio changed; it has been retained")
            path.unlink()


def _retry_rejected_take(folder, state, request):
    """Discard only our hash-bound failed take after another explicit play."""
    rejections = state.get("rejections", [])
    if not rejections or not isinstance(rejections[-1], dict):
        raise ValueError("Invalid pronunciation preview rejection audit")
    rejected = rejections[-1]
    audit = {key: value for key, value in rejected.items() if key != "sha256"}
    if (rejected.get("sha256") != digest(audit) or rejected.get("request_sha256") != digest(request)
            or rejected.get("take") != state.get("take")
            or rejected.get("seed") != state.get("take_seed")
            or rejected.get("raw_sha256") != state.get("raw_sha256")):
        raise ValueError("Pronunciation preview rejection audit changed; audio has been retained")
    removable = []
    for name, field in (("raw.wav", "raw_sha256"), ("normalized.wav", "normalized_sha256")):
        path = folder / name
        if path.exists() or path.is_symlink():
            if file_hash(regular_file(path)) != rejected.get(field):
                raise ValueError("Rejected pronunciation preview changed; audio has been retained")
            removable.append(path)
    for path in removable:
        path.unlink()
    return {"format": _FORMAT, "request": request, "status": "pending",
            "purpose": "unverified_pronunciation_listening_preview",
            "take": state["take"] + 1, "take_seed": (state["take_seed"] + 1) % (2**32),
            "rejections": rejections}


def _reject_preview(folder, state, reason):
    """Checkpoint a physical rejection without authorizing a retry loop."""
    rejection = {"request_sha256": digest(state["request"]), "take": state["take"],
                 "seed": state["take_seed"], "raw_sha256": state["raw_sha256"],
                 "normalized_sha256": file_hash(regular_file(folder / "normalized.wav")),
                 "reason": str(reason), "kind": "physically_invalid_audio"}
    rejection["sha256"] = digest(rejection)
    state.setdefault("rejections", []).append(rejection)
    state["status"] = "rejected"
    write_json(folder / "preview.json", state)
    raise ValueError(f"{reason}. Play again to generate a new take; the rejection audit is saved in {folder / 'preview.json'}") from reason


class PronunciationReviewSession:
    def __init__(self, bundle, config, *, engine_factory=None, store=None, notify=print, proposal_fn=None):
        self.bundle, self.config = bundle, deepcopy(config)
        self.notify = notify
        self._engine_factory, self._engine, self._store = engine_factory, None, store
        self._resolved = None
        self.proposal_fn = proposal_fn
        self.preparation_errors = {}

    @property
    def settings(self):
        if self._resolved is None:
            from .book_context import resolve_book_config
            self._resolved = resolve_book_config(self.bundle, self.config)
        return self._resolved

    @property
    def directory(self):
        path = Path(self.settings["render"]["work_dir"]) / "pronunciations"
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Pronunciation preview directories must not be symlinks")
        return path

    def _drafts(self):
        path = self.directory / "drafts.json"
        if not path.exists() and not path.is_symlink():
            return {}
        value = read_json(regular_file(path))
        if (not isinstance(value, dict) or value.get("format") != _DRAFT_FORMAT or value.get("version") != 1
                or value.get("book_id") != self.bundle["source"]["book"]["id"]
                or not isinstance(value.get("entries"), dict) or value.get("sha256") != digest(value["entries"])):
            raise ValueError("Invalid pronunciation audition drafts")
        from ..pronunciations import validate_pronunciation, validate_word
        for word, pronunciation in value["entries"].items():
            validate_word(word)
            validate_pronunciation(pronunciation)
        return value["entries"]

    def _write_draft(self, word, pronunciation, *, variants=()):
        from ..pronunciations import load_pronunciations, validate_pronunciation, validate_word
        validate_word(word)
        if pronunciation is not None:
            pronunciation = validate_pronunciation(pronunciation)
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = acquire_directory_lock(self.directory / ".drafts.lock")
        try:
            entries = self._drafts()
            rows = load_pronunciations(self.bundle["directory"])
            members = family_words(word, rows, {*entries, word, *variants})
            for member in {*members, *variants}:
                entries.pop(member, None)
            if pronunciation is not None:
                entries[word] = pronunciation
            write_json(self.directory / "drafts.json", {
                "format": _DRAFT_FORMAT, "version": 1, "book_id": self.bundle["source"]["book"]["id"],
                "entries": entries, "sha256": digest(entries)})
        finally:
            lock.rmdir()

    def clear_drafts(self, row):
        self._write_draft(row["word"], None, variants=entry_words(row))

    def rows(self, word=None):
        from ..pronunciations import load_pronunciations, word_id
        rows = load_pronunciations(self.bundle["directory"])
        drafts = self._drafts()
        result = []
        for row in rows:
            # Retain the exact spelling used by a saved draft so its existing
            # preview/played receipt remains valid. Conflicting legacy drafts
            # stay visible until the listener explicitly chooses one.
            choices = {}
            for variant in entry_words(row):
                if variant in drafts:
                    choice = drafts[variant]
                    choices.setdefault((choice["alphabet"], choice["value"]), []).append(variant)
            if len(choices) <= 1:
                partitions = [(list(entry_words(row)), next(iter(choices.values()), []))]
            else:
                partitions = [(members, members) for members in choices.values()]
                remaining = [variant for variant in entry_words(row) if variant not in drafts]
                if remaining:
                    partitions.append((remaining, []))
            for variants, drafted in partitions:
                preferred = drafted or variants
                spelling = row["word"] if row["word"] in preferred else min(preferred, key=natural_spelling)
                selected = dict(row, word=spelling, id=word_id(spelling), variants=variants)
                if spelling != row["word"]:
                    selected.setdefault("origin_word", row["word"])
                if drafted:
                    selected["draft"] = drafts[spelling]
                if len(choices) > 1:
                    selected["case_conflict"] = True
                if spelling in self.preparation_errors:
                    selected["preparation_error"] = self.preparation_errors[spelling]
                result.append(selected)
        result.sort(key=lambda row: (row["word"].lower(), row["word"]))
        if word is None:
            return result
        selected = find_pronunciation(result, word)
        return [selected] if selected is not None else []

    def _estimates(self):
        """Display-only CMU estimates never become pronunciation choices."""
        from ..pronunciations import validate_proposal, validate_word
        path = self.directory / "cmu-estimates.json"
        if not path.exists() and not path.is_symlink():
            return {}
        value = read_json(regular_file(path))
        if (not isinstance(value, dict) or value.get("format") != _ESTIMATE_FORMAT
                or value.get("version") != 1 or value.get("book_id") != self.bundle["source"]["book"]["id"]
                or value.get("source_fingerprint") != self.bundle["manifest"]["source_fingerprint"]
                or not isinstance(value.get("entries"), dict)
                or value.get("sha256") != digest(value["entries"])):
            raise ValueError("Invalid pronunciation CMU estimates or estimates belong to different book/source")
        for key, entry in value["entries"].items():
            if not isinstance(entry, dict) or set(entry) != {"request", "pronunciation"}:
                raise ValueError("Invalid pronunciation CMU estimate entry")
            request = entry["request"]
            if not isinstance(request, dict) or set(request) != {"word", "respelling"} or key != digest(request):
                raise ValueError("Invalid pronunciation CMU estimate request")
            validate_word(request["word"])
            if request["respelling"] is not None:
                validate_word(request["respelling"])
            validate_proposal(entry["pronunciation"])
        return value["entries"]

    def _guess(self, word, respelling=None):
        from ..pronunciations import validate_proposal
        if self.proposal_fn is None:
            raise ValueError("CMU pronunciation suggestions require the configured analysis model; "
                             "use P or --phonemes to supply phones directly")
        return validate_proposal(self.proposal_fn(word, respelling))

    def _estimate(self, word, respelling=None, *, prepare=False):
        from ..pronunciations import validate_word
        validate_word(word)
        if respelling is not None:
            validate_word(respelling)
        request = {"word": word, "respelling": respelling}
        key = digest(request)
        existing = self._estimates().get(key)
        if existing is not None or not prepare:
            return existing["pronunciation"] if existing is not None else None
        pronunciation = self._guess(word, respelling)
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = acquire_directory_lock(self.directory / ".cmu-estimates.lock")
        try:
            entries = self._estimates()
            entries.setdefault(key, {"request": request, "pronunciation": pronunciation})
            write_json(self.directory / "cmu-estimates.json", {
                "format": _ESTIMATE_FORMAT, "version": 1,
                "book_id": self.bundle["source"]["book"]["id"],
                "source_fingerprint": self.bundle["manifest"]["source_fingerprint"],
                "entries": entries, "sha256": digest(entries)})
            return entries[key]["pronunciation"]
        finally:
            lock.rmdir()

    def description(self, row, choice=_SELECTED, *, prepare=False):
        """Describe selected phones or a clearly labeled text-based estimate.

        ``None`` explicitly means the engine's default, even when the row has
        a correction. Estimates describe a plausible reading, not measured
        engine output, and never change draft/approval or preview identity.
        """
        choice = _preferred(row) if choice is _SELECTED else choice
        if choice is not None and choice["alphabet"] == "cmu_arpabet":
            return _description(choice)
        respelling = choice["value"] if choice is not None else None
        estimate = self._estimate(row["word"], respelling, prepare=prepare)
        phones = estimate["value"] if estimate is not None else "unavailable"
        if choice is None:
            return f"default pronunciation: {phones} (CMU estimate)"
        return f"{_description(choice)}; CMU estimate: {phones}"

    def prepare_review(self, word=None):
        """Prepare pending CMU and both listening paths before opening the menu.

        Each proposal, display estimate, and preview checkpoints independently;
        rerunning reuses completed work. Preparation never plays or approves a
        pronunciation, changes an explicit default, or creates a voice.
        Unavailable individual previews remain editable in the menu; systemic
        model/transport failures stop preparation with completed work intact.
        """
        from ..pronunciations import save_proposal
        from .diagnostics import configured_secrets, safe_error_text
        rows = self.rows(word)
        total = len(rows)
        if not total:
            return
        self.notify(f"Preparing pronunciation CMU 0/{total}; completed suggestions and audio are reused on restart.")
        for index, row in enumerate(rows, 1):
            self.notify(display(f"Preparing pronunciation CMU {index}/{total}: {row['word']}"))
            try:
                if row["status"] == "pending" and _preferred(row) is None:
                    save_proposal(self.bundle["directory"], self.bundle, row["word"], self._guess(row["word"]))
                    current = self.rows(row["word"])
                    if not current:
                        raise ValueError("Pronunciation entry changed during preparation; rerun to resume")
                    row = current[0]
                self.description(row, prepare=True)
            except Exception:
                self.notify(display(f"CMU preparation stopped at {index}/{total}: {row['word']}. "
                                    "Completed work is retained; rerun the same command to resume."))
                raise
        rows = self.rows(word)
        total = len(rows)
        self.notify(f"Preparing pronunciation audio 0/{total}")
        ready, failed = 0, 0
        for index, row in enumerate(rows, 1):
            choice = _preferred(row)
            self.notify(display(f"Preparing pronunciation audio {index}/{total}: {row['word']}"))
            try:
                self.audition_isolated(row["word"], choice)
                self.audition(row["word"], choice)
                self.preparation_errors.pop(row["word"], None)
                ready += 1
            except ValueError as exc:
                reason = safe_error_text(exc, secrets=configured_secrets(self.config))
                self.preparation_errors[row["word"]] = reason
                failed += 1
                self.notify(display(f"Audio unavailable for {row['word']}: {reason}"))
            except Exception:
                self.notify(display(f"Audio preparation stopped at {index}/{total}: {row['word']}. "
                                    "Completed work is retained; rerun the same command to resume."))
                raise
        self.notify(f"Pronunciation preparation complete: {ready}/{total} audio ready; {failed} unavailable. "
                    "No pronunciations were accepted during preparation.")

    def _clear_preparation_error(self, word, pronunciation):
        if word in self.preparation_errors:
            current = self.rows(word)
            if current and _preferred(current[0]) == pronunciation:
                self.preparation_errors.pop(word, None)

    def edit(self, word, pronunciation):
        if not self.rows(word):
            raise ValueError("Choose a word from the pronunciation menu")
        self._write_draft(word, pronunciation)

    def propose(self, word, respelling=None):
        """Keep generated phones pending until their exact audition is approved."""
        from ..pronunciations import set_pronunciation, validate_pronunciation, validate_word
        validate_word(word)
        if respelling is not None:
            _pronunciation(say_as=respelling)
        if self.proposal_fn is None:
            raise ValueError("CMU pronunciation suggestions require the configured analysis model; "
                             "use P or --phonemes to supply phones directly")
        pronunciation = validate_pronunciation(self.proposal_fn(word, respelling))
        if pronunciation["alphabet"] != "cmu_arpabet":
            raise ValueError("Pronunciation suggestion must provide CMU Arpabet phones")
        if not self.rows(word):
            set_pronunciation(self.bundle["directory"], self.bundle, word, status="pending")
        self._write_draft(word, pronunciation)
        self.notify(display(f"Guessed {_description(pronunciation)}. Audition these exact phones before confirming."))
        return pronunciation

    def selection(self, row):
        proposed = _preferred(row)
        if (proposed is None and row.get("status") == "pending"
                and self.proposal_fn is not None):
            from ..pronunciations import save_proposal
            save_proposal(self.bundle["directory"], self.bundle, row["word"], self._guess(row["word"]))
            current = self.rows(row["word"])
            return _preferred(current[0]) if current else None
        return proposed

    def confirm(self, row):
        from ..pronunciations import set_pronunciation
        current = self.rows(row["word"])
        pronunciation = _preferred(current[0]) if current else None
        if pronunciation is None:
            raise ValueError("Enter a correction first, or use I to keep the default pronunciation")
        preview = self.audition(row["word"], pronunciation, cached_only=True)
        if preview.get("played") != {"request_sha256": digest(preview["request"]),
                                     "audio_sha256": preview["audio_sha256"]}:
            raise ValueError("Play this exact pronunciation with the current voice before confirming it")
        set_pronunciation(self.bundle["directory"], self.bundle, row["word"], pronunciation, status="confirmed")
        self.clear_drafts(current[0])

    def ignore(self, word):
        from ..pronunciations import set_pronunciation
        current = self.rows(word)
        set_pronunciation(self.bundle["directory"], self.bundle, word, status="ignored")
        self.clear_drafts(current[0] if current else {"word": word})

    def reset(self, word):
        from ..pronunciations import reset_pronunciation
        current = self.rows(word)
        reset_pronunciation(self.bundle["directory"], word)
        self.clear_drafts(current[0] if current else {"word": word})

    def _occurrences(self, word):
        """Saved-voice candidates where the exact whole word occurs, best first.

        Selection only needs the word's original source occurrence and a saved
        voice; it never maps pronunciations, so one ambiguously-normalized
        occurrence cannot poison the candidates for the whole menu.
        """
        from ..pronunciations import WORD_PATTERN
        from ..spoken_text import build_spoken
        from .book_context import voice_store
        from .speech_text import trim_speech_text
        if self._store is None:
            self._store = voice_store(self.bundle, self.settings)
        candidates, selected_voices = [], {}
        for chapter in self.bundle["chapters"]:
            for row in chapter["rows"]:
                matches = [match for match in WORD_PATTERN.finditer(row["text"]) if match.group() == word]
                if not matches:
                    continue
                try:
                    spoken, mapping = build_spoken(row)
                except (KeyError, ValueError):
                    continue  # A malformed row must not block other occurrences.
                if not any(character.isalnum() for character in trim_speech_text(spoken)):
                    continue
                spans = [_spoken_span(mapping, row, match) for match in matches]
                speakers = row.get("simultaneous_speakers") or [{"voice_id": row.get("voice_id"), "role_id": row.get("role_id")}]
                for speaker in speakers:
                    voice_id = speaker.get("voice_id")
                    if not voice_id:
                        continue
                    if voice_id not in selected_voices:
                        selected_voices[voice_id] = self._store.load(voice_id)
                    voice = selected_voices[voice_id]
                    if voice is None:
                        continue
                    for span in spans:
                        if span is None:
                            # The word is consumed by an approved replacement;
                            # usable for an isolated preview, never an excerpt.
                            candidates.append((10 ** 6, len(candidates), {
                                "row": row, "text": spoken, "start": 0, "end": 0, "spanless": True,
                                "voice": voice, "voice_id": voice_id, "role_id": speaker.get("role_id"),
                                "chapter_title": chapter["entry"]["title"]}))
                            continue
                        start, end = _sample_bounds(spoken, span[0], span[1])
                        candidates.append((abs(min(end - start, 240) - 160), len(candidates), {
                            "row": row, "text": spoken, "start": start, "end": end, "spanless": False,
                            "voice": voice, "voice_id": voice_id, "role_id": speaker.get("role_id"),
                            "chapter_title": chapter["entry"]["title"]}))
        return [item[2] for item in sorted(candidates, key=lambda item: item[:2])]

    def _sample(self, word):
        candidates = self._occurrences(word)
        if not candidates:
            raise ValueError("No spoken excerpt with an existing saved voice contains this word; cast its character first")
        return candidates[0]

    def audition(self, word, pronunciation=None, *, cached_only=False):
        """Return a resumable excerpt preview; never create or modify a voice.

        Correction previews omit, on a copy of the row, only the analyzer
        normalization edits overlapping an exact occurrence of the word; every
        other approved edit is preserved. When no occurrence maps unambiguously
        (the word is covered by multiword edits everywhere), the preview falls
        back to the isolated word with the selected saved voice. The renderer's
        production mapping guards are unchanged.
        """
        from ..pronunciations import validate_pronunciation, validate_word
        from ..speech_traits import speech_traits
        from .engines import create_engine
        from .speech_identity import voice_conditioning_identity
        validate_word(word)
        if pronunciation is not None:
            pronunciation = validate_pronunciation(pronunciation)
        candidates = self._occurrences(word)
        if not candidates:
            return self.audition_isolated(word, pronunciation, cached_only=cached_only)
        if self._engine is None:
            self._engine = (self._engine_factory or create_engine)(self.settings["engine"])
        engine = self._engine
        if pronunciation is not None and engine.capabilities.get("structured_performance") is not True:
            raise ValueError("This synthesis engine does not support structured pronunciation previews")
        entries = ({word: {"word": word, "status": "confirmed", "pronunciation": pronunciation}}
                   if pronunciation is not None else None)
        for sample in candidates:
            voice = sample["voice"]
            binding = getattr(engine, "voice_binding_identity", engine.identity)
            if any(voice.get("created_with", {}).get(key) != binding.get(key) for key in ("engine", "model")):
                raise ValueError("Selected character voice is not bound to this synthesis engine")
            record, spoken_start, spoken_end = None, sample["start"], sample["end"]
            if entries is not None or sample.get("spanless"):
                plan = self._excerpt_plan(sample, word, entries)
                if plan is None:
                    continue  # This occurrence is ambiguously normalized; try the next.
                text, record, spoken_start, spoken_end = plan
            else:
                text = sample["text"][sample["start"]:sample["end"]]
            profile = next(item for item in self.bundle["cast"]["voices"] if item["id"] == sample["voice_id"])
            from .speech_traits import speech_trait_overrides
            overrides = speech_trait_overrides(self.bundle, self.settings)
            traits = overrides[sample["voice_id"]] if sample["voice_id"] in overrides else speech_traits(profile)
            if record is None:
                from ..spoken_text import effective_delivery
                direction = "; ".join(part for part in (traits, effective_delivery(sample["row"])) if part)
                options = {"direction": direction}
            else:
                from ..spoken_text import effective_delivery
                scene = deepcopy(sample["row"])
                scene.pop("speech_normalization", None)
                options = {"direction": effective_delivery(scene),
                           "performance": record, "traits": traits}
            request = {"version": 1, "engine_sha256": digest(engine.identity), "voice": voice_conditioning_identity(voice),
                       "text": text, "word": word, "pronunciation": pronunciation, "options": options,
                       "seed": self.settings["render"]["seed"], "sample_rate": self.settings["render"]["sample_rate"]}
            context = {"segment_id": sample["row"]["id"], "voice_id": sample["voice_id"],
                       "chapter_title": sample["chapter_title"], "spoken_start": spoken_start,
                       "spoken_end": spoken_end}
            return self._preview_cache(word, pronunciation, text, options, request, engine, voice,
                                       purpose="unverified_pronunciation_listening_preview", context=context,
                                       cached_only=cached_only)
        return self.audition_isolated(word, pronunciation, cached_only=cached_only)

    def _excerpt_plan(self, sample, word, entries):
        """Excerpt text, merged record, and bounds from the target-edits-omitted copy.

        Returns None when this occurrence cannot map unambiguously even with
        overlapping analyzer edits omitted on the copy.
        """
        from ..performance_plan import slice_performance
        from ..pronunciations import WORD_PATTERN
        from ..spoken_text import build_spoken
        from .pronunciation_runtime import merge_row_pronunciations
        preview_row = _preview_row(sample["row"], word)
        try:
            spoken, mapping = build_spoken(preview_row)
            match = next((item for item in WORD_PATTERN.finditer(preview_row["text"]) if item.group() == word), None)
            if match is None:
                return None
            span = _spoken_span(mapping, preview_row, match)
            if span is None:
                return None
            merged = merge_row_pronunciations(preview_row, None, entries)
            if merged is None:
                return None
            start, end = _sample_bounds(spoken, span[0], span[1])
            return spoken[start:end], slice_performance(merged, spoken, start, end), start, end
        except ValueError:
            return None

    def audition_isolated(self, word, pronunciation=None, *, cached_only=False):
        """Return a resumable FLAC preview of the isolated word; distinct cache identity.

        The listening queue speaks the word alone, twice, instead of a bounded
        source excerpt. The word-only pronunciation control is constructed
        directly, without source normalization mapping, and the voice is the
        saved voice of any row containing the exact whole word. The request
        carries ``isolated: True`` and the bare word text so its cache folder
        and receipt never collide with excerpt previews of the same choice.
        """
        from ..performance_plan import PERFORMANCE_VERSION, text_sha256, validate_performance
        from ..pronunciations import validate_pronunciation, validate_word
        from .engines import create_engine
        from .speech_identity import voice_conditioning_identity
        validate_word(word)
        if pronunciation is not None:
            pronunciation = validate_pronunciation(pronunciation)
        sample = self._sample(word)
        if self._engine is None:
            self._engine = (self._engine_factory or create_engine)(self.settings["engine"])
        engine, voice = self._engine, sample["voice"]
        binding = getattr(engine, "voice_binding_identity", engine.identity)
        if any(voice.get("created_with", {}).get(key) != binding.get(key) for key in ("engine", "model")):
            raise ValueError("Selected character voice is not bound to this synthesis engine")
        if pronunciation is not None and engine.capabilities.get("structured_performance") is not True:
            raise ValueError("This synthesis engine does not support structured pronunciation previews")
        controls = []
        if pronunciation is not None:
            controls.append({"kind": "pronunciation", "start": 0, "end": len(word), "text": word,
                             "alphabet": pronunciation["alphabet"],
                             "replacement" if pronunciation["alphabet"] == "respelling" else "phonemes":
                                 pronunciation["value"]})
        record = validate_performance({"version": PERFORMANCE_VERSION, "text_sha256": text_sha256(word),
                                       "controls": controls}, word)
        options = {"performance": record} if controls else {"direction": ""}
        from ..speech_traits import speech_traits
        from .speech_traits import speech_trait_overrides
        overrides = speech_trait_overrides(self.bundle, self.settings)
        profile = next(item for item in self.bundle["cast"]["voices"] if item["id"] == sample["voice_id"])
        traits = overrides[sample["voice_id"]] if sample["voice_id"] in overrides else speech_traits(profile)
        if traits:
            options["traits" if controls else "direction"] = traits
        request = {"version": 1, "engine_sha256": digest(engine.identity), "voice": voice_conditioning_identity(voice),
                   "text": word, "word": word, "pronunciation": pronunciation, "options": options,
                   "isolated": True,
                   "seed": self.settings["render"]["seed"], "sample_rate": self.settings["render"]["sample_rate"]}
        context = {"segment_id": sample["row"]["id"], "voice_id": sample["voice_id"],
                   "chapter_title": sample["chapter_title"], "isolated": True}
        return self._preview_cache(word, pronunciation, word, options, request, engine, voice,
                                   purpose="unverified_pronunciation_isolated_preview", context=context,
                                   cached_only=cached_only)

    def _preview_cache(self, word, pronunciation, text, options, request, engine, voice, *, purpose, context, cached_only):
        key = digest(request)
        folder = self.directory / key
        if folder.is_symlink():
            raise ValueError("Pronunciation preview cache must not be a symlink")
        if cached_only and not folder.is_dir():
            raise ValueError("Play this exact pronunciation with the current voice before confirming it")
        self.directory.mkdir(parents=True, exist_ok=True)
        folder.mkdir(exist_ok=True)
        lock = acquire_directory_lock(folder / ".preview.lock")
        try:
            state_path, audio = folder / "preview.json", folder / "sample.flac"
            state = read_json(regular_file(state_path)) if state_path.exists() or state_path.is_symlink() else None
            if state is not None:
                if not isinstance(state, dict) or state.get("format") != _FORMAT or state.get("request") != request:
                    raise ValueError("Pronunciation preview request changed")
                if state.get("status") == "ready":
                    if file_hash(regular_file(audio)) != state.get("audio_sha256"):
                        raise ValueError("Pronunciation preview changed; remove its cache entry to regenerate")
                    _remove_redundant_wavs(folder, state)
                    self._clear_preparation_error(word, pronunciation)
                    return dict(state, path=str(audio))
                if cached_only:
                    raise ValueError("Play this exact pronunciation with the current voice before confirming it")
                if state.get("status") == "rejected":
                    state = _retry_rejected_take(folder, state, request)
            if cached_only:
                raise ValueError("Play this exact pronunciation with the current voice before confirming it")
            state = state or {"format": _FORMAT, "request": request, "status": "pending",
                               "purpose": purpose}
            state.setdefault("take", 1)
            state.setdefault("take_seed", request["seed"])
            write_json(state_path, state)
            raw, normalized = folder / "raw.wav", folder / "normalized.wav"
            if state.get("raw_sha256") is None:
                if raw.exists() or raw.is_symlink():
                    regular_file(raw).unlink()
                self.notify(display(f"Generating listening preview: {word} — {_description(pronunciation)}"))
                engine.synthesize(text, voice, raw, seed=state["take_seed"], **options)
                state.update(raw_sha256=file_hash(regular_file(raw)), status="generated")
                write_json(state_path, state)
            elif file_hash(regular_file(raw)) != state["raw_sha256"]:
                raise ValueError("Generated pronunciation preview changed before compression")
            from .assembly import normalize_clip
            from .chapter_executor import _flac_master
            if normalized.exists() or normalized.is_symlink():
                regular_file(normalized).unlink()
            normalize_clip(raw, normalized, sample_rate=request["sample_rate"], ffmpeg=self.settings["render"]["ffmpeg"])
            try:
                metrics = _preview_audio(normalized)
            except (ValueError, wave.Error, EOFError) as exc:
                _reject_preview(folder, state, exc)
            if audio.exists() or audio.is_symlink():
                regular_file(audio).unlink()
            pcm_hash = _flac_master(normalized, audio, self.settings["render"]["ffmpeg"])
            state.update(status="ready", audio_sha256=file_hash(audio), pcm_sha256=pcm_hash, metrics=metrics,
                          context=context)
            write_json(state_path, state)
            _remove_redundant_wavs(folder, state)
            self._clear_preparation_error(word, pronunciation)
            return dict(state, path=str(audio))
        finally:
            lock.rmdir()

    def mark_played(self, preview):
        """Bind successful playback (including any-key stop) to this exact take.

        Returns the hash-bound played receipt written for this exact audio;
        callers that accept on completed playback must verify against it.
        """
        folder = self.directory / digest(preview["request"])
        if folder.is_symlink():
            raise ValueError("Pronunciation preview cache must not be a symlink")
        lock = acquire_directory_lock(folder / ".preview.lock")
        try:
            state_path = folder / "preview.json"
            state = read_json(regular_file(state_path))
            if (state.get("status") != "ready" or state.get("request") != preview["request"]
                    or state.get("audio_sha256") != preview.get("audio_sha256")
                    or file_hash(regular_file(folder / "sample.flac")) != preview["audio_sha256"]):
                raise ValueError("Pronunciation preview changed during playback; play it again before confirming")
            state["played"] = {"request_sha256": digest(state["request"]), "audio_sha256": state["audio_sha256"]}
            write_json(state_path, state)
            return dict(state["played"])
        finally:
            lock.rmdir()

    def close(self):
        if self._engine is not None:
            close = getattr(self._engine, "close", None)
            if callable(close):
                close()


def run_pronunciation_review(bundle, config, args, *, input_fn=None, output=print,
                             engine_factory=None, store=None, player=None, playback_fn=None, proposal_fn=None):
    """Dispatch explicit edits or a keyboard-driven proposal/listening menu."""
    from ..pronunciations import load_pronunciations, set_pronunciation, validate_word
    from ..provider import ProviderError
    action = getattr(args, "action", "review")
    word = getattr(args, "word", None)
    correction = _pronunciation(getattr(args, "say_as", None), getattr(args, "phonemes", None))
    if action in {"set", "ignore", "reset", "play"} and not word:
        raise ValueError(f"--word is required for --action {action}")
    if word is not None:
        validate_word(word)
    if correction is not None and action not in {"set", "play"}:
        raise ValueError("--say-as and --phonemes are used with --action set or play")
    if action == "list":
        rows = load_pronunciations(bundle["directory"])
        selected = find_pronunciation(rows, word) if word is not None else None
        _show(rows if word is None else ([selected] if selected is not None else []), output)
        return 0
    if action == "set":
        if correction is None:
            raise ValueError("Provide --say-as or --phonemes; use --action ignore to keep the default")
        if correction["alphabet"] == "cmu_arpabet":
            set_pronunciation(bundle["directory"], bundle, word, correction, status="confirmed")
            PronunciationReviewSession(bundle, config)._write_draft(word, None)
            output(display(f"Confirmed {word}: {_description(correction)}. Existing audio is unchanged."))
            return 0
    if action == "ignore":
        PronunciationReviewSession(bundle, config).ignore(word)
        output(display(f"Keeping the default pronunciation for {word}."))
        return 0
    if action == "reset":
        PronunciationReviewSession(bundle, config).reset(word)
        output(display(f"Removed the local pronunciation decision for {word}; inherited decisions apply."))
        return 0
    if action not in {"review", "play", "set"}:
        raise ValueError("Unsupported pronunciation review action")
    if input_fn is None:
        from .terminal import enable_line_editing
        enable_line_editing()
        input_fn = input
    if playback_fn is None:
        from .terminal import play_sample
        playback_fn = play_sample
    session = PronunciationReviewSession(bundle, config, engine_factory=engine_factory, store=store, notify=output,
                                         proposal_fn=proposal_fn)
    player = player or AudioPlayer(config.get("casting", {}).get("player", ["/usr/bin/afplay"]))

    def say(value):
        output(display(value))

    def play(selected_word, proposed):
        preview = session.audition(selected_word, proposed)
        row = {"word": selected_word}
        say(f"Auditioning {selected_word}: {session.description(row, proposed, prepare=action == 'review')}.")
        say("Listening preview — wording and pronunciation are for you to judge; this is not accepted book audio.")
        say(f"{preview['context']['chapter_title']}: {preview['request']['text']}")
        say("Press any key to stop.")
        playback_fn(player, preview["path"])
        session.mark_played(preview)

    last = None
    try:
        if action in {"play", "set"}:
            current = session.rows(word)
            if correction is not None and correction["alphabet"] == "respelling":
                proposed = session.propose(word, correction["value"])
            else:
                proposed = correction if correction is not None else (session.selection(current[0]) if current else None)
            play(word, proposed)
            if correction is not None and correction["alphabet"] == "respelling":
                say("CMU draft auditioned; open pronunciation review and use C to confirm it. Existing decisions are unchanged.")
            return 0
        session.prepare_review(word)
        if word is None:
            from .pronunciation_review_queue import run_pronunciation_queue
            return run_pronunciation_queue(
                session, player, playback_fn, input_fn=input_fn, output=output,
                interactive=sys.stdin.isatty() and sys.stdout.isatty())
        while True:
            rows = session.rows(word)
            _show(rows, output, session.description)
            if not rows:
                say("No words are awaiting review. Use pronunciations --action discover to find candidates.")
                return 0
            say("Number: play proposal/default   B [number]: baseline   E [number]: edit respelling   P [number]: edit CMU phonemes   C [number]: confirm   I [number]: keep default   R [number]: reset   Q: quit")
            command = input_fn("> ").strip()
            if command.casefold() == "q":
                return 0
            try:
                parts = command.split(maxsplit=2)
                if not parts:
                    continue
                verb = "play" if parts[0].isdecimal() else parts[0].casefold()
                selection = parts[0] if verb == "play" else (parts[1] if len(parts) > 1 else None)
                if selection is None:
                    row = find_pronunciation(rows, last)
                    if row is None:
                        raise ValueError("Enter a number from the pronunciation menu")
                elif not selection.isdecimal() or not 1 <= int(selection) <= len(rows):
                    raise ValueError("Enter a number from the pronunciation menu")
                else:
                    row = rows[int(selection) - 1]
                last = row["word"]
                if verb == "play":
                    play(last, session.selection(row))
                elif verb == "b":
                    play(last, None)
                elif verb in {"e", "p"}:
                    say(f"Current pronunciation for {last}: {session.description(row)}")
                    prompt = ("Say it as (plain spelling; empty cancels): " if verb == "e" else
                              "CMU phonemes (for example M AA1 R AH0; empty cancels): ")
                    value = parts[2] if len(parts) > 2 else input_fn(prompt).strip()
                    if not value:
                        continue
                    if verb == "e":
                        proposed = session.propose(last, value)
                    else:
                        proposed = _pronunciation(phonemes=value)
                        session.edit(last, proposed)
                    say("Draft saved for audition; C confirms it for future rendering.")
                    play(last, proposed)
                elif verb == "c":
                    session.confirm(row)
                    say("Pronunciation confirmed. Existing audio is unchanged.")
                elif verb == "i":
                    session.ignore(last)
                    say("Default pronunciation selected.")
                    try:
                        say(session.description(row, None, prepare=True))
                    except (ValueError, RuntimeError, OSError, ProviderError):
                        say("CMU estimate unavailable; B can retry it. The default choice is saved.")
                elif verb == "r":
                    session.reset(last)
                    say("Local decision and audition draft removed; inherited decisions apply.")
                else:
                    raise ValueError("Use a number, B, E, P, C, I, R, or Q")
            except (ValueError, RuntimeError, OSError, ProviderError) as exc:
                from .diagnostics import configured_secrets, safe_error_text
                say("Action stopped: " + safe_error_text(exc, secrets=configured_secrets(config)))
    except EOFError:
        say("Pronunciation review closed; decisions and audition drafts are saved.")
        return 0
    finally:
        player.stop()
        session.close()
