"""Casting screen state and actions, independent of any terminal library.

``CastController`` wraps a ``CastingSession``: it classifies every voice,
filters the cast, describes the highlighted voice, and turns each keypress
into either an immediate read (playback) or a background job on one serial
``JobQueue`` (generation, acceptance, edits, sharing, clip counts). Running
every write on that one worker keeps the session's locks uncontended and the
screen responsive. The controller never accepts a take, creates a performer,
or regenerates a voice unless the person asked for exactly that.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

from epub_tts_narrator.tui_jobs import FAILED, RUNNING, JobQueue

from .audition import _label, display


DESIGNERS = {"local": "locally", "gemini": "with Gemini", "elevenlabs": "with ElevenLabs"}
SAVE_LANE, GENERATE_LANE, ASK_LANE = "save", "generate", "ask"
GENERATIONS = 3  # Gemini voices generated at once on the screen (at most casting.gemini_concurrency)


def generation_limit(config):
    """How many Gemini generations the cast screen runs at once."""
    casting = config.get("casting") if isinstance(config, dict) else None
    value = (casting or {}).get("gemini_concurrency", 1) if isinstance(casting, dict) else 1
    return max(1, min(GENERATIONS, value)) if isinstance(value, int) else 1

# Display order when sorting by status: what needs attention first.
STATUSES = ("failed", "generating", "queued", "needs regeneration", "missing", "pending",
            "accepted", "shared", "derived", "inherited")
STATUS_FILTERS = (None, "missing", "pending", "failed", "accepted", "inherited", "derived", "shared")

FIELDS = {
    "prompt": "Voice description",
    "reference_text": "Audition script",
    "speech_traits": "Speech traits",
    "accent": "Accent cue",
}


def label(row):
    return display(_label(row))


def no_accent(row):
    """How a voice without an accent cue is described: a voice with no lines in this book needs none."""
    if row.get("speaks") is False:
        return "(none: no lines in this book, so none is needed until a book where this character speaks)"
    return "(none)"


def canonical(row):
    return row.get("canonical_voice_id", row["voice_id"])


def voice_status(row, active=None):
    """``(status, note)`` for one casting row; ``active`` maps canonical IDs to job states."""
    state = (active or {}).get(canonical(row))
    if state == RUNNING:
        return "generating", ""
    if state is not None:
        return "queued", ""
    regeneration = row.get("regeneration")
    if regeneration:
        note = " (inherited)" if regeneration.get("inherited") else ""
        if regeneration.get("state") == "failed":
            return "failed", (regeneration.get("error") or "") + note
        return "needs regeneration", note.strip()
    selected = row.get("selected")
    if selected is None:
        return ("pending", f"{len(row['candidates'])} take(s), none accepted") if row.get("candidates") else ("missing", "")
    if row.get("derived_from"):
        base = row["derived_from"].get("voice_id")
        stale = "; its main voice changed since cloning" if row.get("clone_stale") else ""
        return "derived", f"cloned from {base}{stale}"
    if canonical(row) != row["voice_id"]:
        return "shared", f"uses {canonical(row)}"
    if selected.get("inherited"):
        return "inherited", "from an earlier book"
    return "accepted", ""


def next_status_filter(current):
    index = STATUS_FILTERS.index(current) if current in STATUS_FILTERS else 0
    return STATUS_FILTERS[(index + 1) % len(STATUS_FILTERS)]


@dataclass
class CastFilter:
    text: str = ""
    status: str | None = None
    by_status: bool = False

    def matches(self, row, status):
        if self.status is not None and status != self.status:
            return False
        words = self.text.casefold().split()
        if not words:
            return True
        haystack = " ".join([label(row), row["voice_id"], *row.get("characters", []),
                             row.get("description", "")]).casefold()
        return all(word in haystack for word in words)


def _short(text, limit=60):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def candidate_rows(row):
    """The voice's takes for the comparison table, newest last, with the accepted one marked."""
    selected = row.get("selected") or {}
    result = []
    for candidate in row.get("candidates", []):
        creation = candidate.get("creation", {})
        designer = creation.get("designer") or candidate.get("created_with") or {}
        clips = creation.get("reference_clips")
        result.append({
            "candidate_id": candidate["candidate_id"],
            "accepted": bool(candidate.get("selected")),
            "created": str(creation.get("created_at") or "")[:16].replace("T", " "),
            "designer": str(designer.get("engine") or "").replace("_voice_design", ""),
            "clips": len(clips) if isinstance(clips, list) and clips else 1,
            "verified": "yes" if (candidate.get("verification") or {}).get("passed") is True else "recheck",
            "audio": candidate.get("reference_audio"),
        })
    if selected.get("inherited"):
        creation = selected.get("creation", {})
        result.insert(0, {
            "candidate_id": selected.get("candidate_id", "inherited"),
            "accepted": True, "created": str(creation.get("created_at") or "")[:16].replace("T", " "),
            "designer": "inherited", "clips": len(creation.get("reference_clips") or []) or 1,
            "verified": "yes", "audio": selected.get("reference_audio"),
        })
    return result


def detail_sections(row, rows, status, note):
    """``[(title, text, key)]`` describing one voice; ``key`` names the edit key or ''."""
    by_id = {other["voice_id"]: other for other in rows}
    derived = bool(row.get("derived_from"))
    users = [label(other) for other in rows
             if other["voice_id"] != row["voice_id"] and canonical(other) == canonical(row)
             and canonical(row) == row["voice_id"]]
    if canonical(row) != row["voice_id"]:
        target = by_id.get(canonical(row))
        sharing = "Uses the voice of " + (label(target) if target else canonical(row))
    elif users:
        sharing = "Used by " + "; ".join(users)
    else:
        sharing = "Not shared"
    source = row.get("sample_source") or {}
    if source.get("kind") == "book_excerpt":
        script_source = "From " + str(source.get("chapter_title") or "the book")
    elif source.get("kind") == "user":
        script_source = "Your own script"
    elif source.get("kind") == "fallback":
        script_source = "Fallback script: no suitable book passage"
    elif source.get("kind") == "unprepared":
        script_source = "Not prepared yet: " + str(source.get("reason") or "")
    else:
        script_source = str(source.get("kind") or "")
    clips = row.get("reference_clips") or {}
    clip_text = f"{clips.get('count', 1)} ({'this voice' if clips.get('source') == 'voice' else 'default'})"
    if not clips.get("designer_records_several", False):
        clip_text += "; the current designer records one clip"
    sections = [
        ("Status", status + (f" — {note}" if note else ""), ""),
        ("Voice ID", row["voice_id"], ""),
        ("Characters", ", ".join(row.get("characters", [])) or "—", ""),
        ("Clone direction" if derived else FIELDS["prompt"], row.get("prompt", ""), "e"),
        ("Analysis description", row.get("description", ""), ""),
        (FIELDS["reference_text"], row.get("reference_text", ""), "s"),
        ("Script source", script_source, ""),
        (FIELDS["speech_traits"], row.get("speech_traits") or "(none)", "t"),
        (FIELDS["accent"] + " (said before every line)", (row.get("accent") or no_accent(row))
         + {"assigned": " (the book's accent for this character's origin)",
                  "description": " (from the voice description)"}.get(row.get("accent_source"), "")
         + (f"; suggested: {row['accent_suggestion']}" if row.get("accent_suggestion")
            and row.get("accent_suggestion") != row.get("accent") else ""), "A"),
        ("Sharing", sharing, "d"),
        ("Reference clips", clip_text, "c"),
    ]
    if derived:
        sections.insert(3, ("Derived from", str(row["derived_from"].get("voice_id")), ""))
    return sections


class CastController:
    """Casting state for a screen: rows, statuses, filtering, and job-backed actions."""

    def __init__(self, session, player, *, config_path=None, jobs=None, on_change=None,
                 clock=time.monotonic, describe_error=None):
        self.session = session
        self.player = player
        self.config_path = config_path
        self._on_change = on_change or (lambda: None)
        self._clock = clock
        self.messages = deque(maxlen=500)
        if describe_error is None:
            from .diagnostics import configured_secrets, safe_error_text
            secrets = configured_secrets(getattr(session, "config", {}) or {})
            describe_error = lambda error: display(safe_error_text(error, secrets=secrets))  # noqa: E731
        self.jobs = jobs or JobQueue(on_change=self._job_changed, describe_error=describe_error, clock=clock)
        # Saving a field never waits for another voice's generation.
        self.jobs.add_independent_lane(SAVE_LANE)
        # Asking the text model to revise descriptions changes nothing, so it waits for nothing either.
        self.jobs.add_independent_lane(ASK_LANE)
        self.requests_factory = None  # tests supply the model; otherwise the book's analysis provider
        # Gemini generations run up to GENERATIONS at once, sharing one store lease and verifier.
        self._pool = None
        self.generation_limit = generation_limit(getattr(session, "config", {}) or {})
        if self.generation_limit > 1:
            self.jobs.add_lane(GENERATE_LANE, self.generation_limit, opening=self._open_pool,
                               closing=self._close_pool)
        self.filter = CastFilter()
        self._rows = []
        self.playing = None
        self.compare = None  # (voice_id, candidate_id) marked for A/B
        session.notify = lambda message: self.log(str(message))

    # Reading -----------------------------------------------------------------

    def refresh(self):
        self._rows = self.session.rows()
        return self._rows

    @property
    def rows(self):
        return self._rows

    def row(self, voice_id):
        return next((row for row in self._rows if row["voice_id"] == voice_id), None)

    def status(self, row):
        return voice_status(row, self.jobs.active_scopes("voice"))

    def visible(self):
        active = self.jobs.active_scopes("voice")
        result = [(row, *voice_status(row, active)) for row in self._rows]
        result = [item for item in result if self.filter.matches(item[0], item[1])]
        if self.filter.by_status:
            result.sort(key=lambda item: (STATUSES.index(item[1]), label(item[0]).casefold()))
        return result

    def counts(self):
        active = self.jobs.active_scopes("voice")
        counts = {}
        for row in self._rows:
            status = voice_status(row, active)[0]
            counts[status] = counts.get(status, 0) + 1
        return counts

    def ready(self):
        return sum(row.get("selected") is not None for row in self._rows)

    def details(self, row):
        return detail_sections(row, self._rows, *self.status(row))

    def log(self, text, severity="info"):
        self.messages.append((self._clock(), severity, display(text)))
        self._on_change()

    # Playback ----------------------------------------------------------------

    def sample(self, row, candidate_id=None):
        """The audio path to play: one take, or the accepted take."""
        if candidate_id is not None:
            item = next((item for item in candidate_rows(row) if item["candidate_id"] == candidate_id), None)
            if item is None or not item["audio"]:
                raise ValueError("That take has no saved audio")
            return item["audio"]
        if row.get("selected") is None:
            raise ValueError("No accepted take yet; generate one (g) or pick a take to play")
        return row["selected"]["reference_audio"]

    def play(self, row, candidate_id=None):
        """Start playback (blocking while a playback copy is prepared); stop with ``stop``."""
        path = self.sample(row, candidate_id)
        self.player.start(path)
        self.playing = (row["voice_id"], candidate_id)
        self.log(f"Playing {label(row)}" + (f" take {candidate_id}" if candidate_id else ""))
        return path

    def poll_playback(self):
        if self.playing is None:
            return False
        try:
            running = self.player.poll()
        except (OSError, ValueError) as error:
            self.playing = None
            self.log(f"Playback stopped: {error}", "error")
            return False
        if not running:
            self.playing = None
        return running

    def stop(self):
        if self.playing is not None:
            self.player.stop()
            self.playing = None

    def mark_compare(self, row, candidate_id):
        self.compare = (row["voice_id"], candidate_id)
        self.log(f"Marked take {candidate_id} of {label(row)} for comparison; press x on another take")

    def compare_partner(self, row, candidate_id):
        """The other take to play when comparing, or None when nothing is marked for this voice."""
        if self.compare is None or self.compare[0] != row["voice_id"]:
            return None
        return self.compare[1] if candidate_id != self.compare[1] else None

    # Confirmation text for actions with wide effects --------------------------

    def regenerate_warning(self, row):
        others = [label(other) for other in self._rows
                  if other["voice_id"] != row["voice_id"] and canonical(other) == canonical(row)]
        warnings = []
        if others:
            warnings.append("This voice is shared; the new voice also replaces it for " + "; ".join(others) + ".")
        if (row.get("selected") or {}).get("inherited"):
            warnings.append("It is inherited from an earlier book; this book gets its own new voice.")
        if row.get("selected") is not None:
            warnings.append("The accepted take is replaced by the new one (old takes stay available).")
        return " ".join(warnings)

    # Actions (background jobs) -----------------------------------------------

    def _submit(self, title, run, row=None, voices=None):
        scope = {"voice": voices if voices is not None else ([canonical(row)] if row else [])}
        if row is not None:
            scope["voice_id"] = row["voice_id"]
        return self.jobs.submit(title, run, scope=scope)

    def _submit_save(self, title, run, row):
        """A quick save: at once, unless this voice has work queued or running, which it must follow."""
        if canonical(row) in self.jobs.active_scopes("voice"):
            return self._submit(title, run, row)
        return self.jobs.submit(title, run, scope={"voice": [canonical(row)], "voice_id": row["voice_id"]},
                                lane=SAVE_LANE)

    def regenerate(self, row, designer="local"):
        if designer not in DESIGNERS:
            raise ValueError("Unknown voice designer")
        voice_id = row["voice_id"]
        title = f"Generate {label(row)} {DESIGNERS[designer]}"
        if designer == "gemini" and self.generation_limit > 1 and not row.get("derived_from"):
            # A changed voice is cloned from its main voice by the local model, so it stays serial.
            return self.jobs.submit(title, lambda: self._pool.regenerate(voice_id),
                                    scope={"voice": [canonical(row)], "voice_id": voice_id}, lane=GENERATE_LANE)
        return self._submit(title, lambda: self.session.regenerate(voice_id, voice_designer=designer), row)

    def _open_pool(self):
        from .casting_batch import GenerationPool
        pool = GenerationPool(self.session, self.generation_limit, designer="gemini")
        pool.open()
        self._pool = pool

    def _close_pool(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def accept(self, row, candidate_id):
        voice_id = row["voice_id"]
        return self._submit(f"Accept take {candidate_id} for {label(row)}",
                            lambda: self.session.accept(voice_id, candidate_id), row)

    def validate_field(self, row, field, value):
        """Raise ``ValueError`` with the session's own message before anything is queued."""
        from ..speech_traits import validate_speech_traits
        from .casting import MAX_PROMPT_CHARACTERS, MAX_REFERENCE_CHARACTERS, _validate_text
        if field == "prompt":
            _validate_text(value, "Voice prompt", MAX_PROMPT_CHARACTERS)
        elif field == "reference_text":
            _validate_text(value, "Sample text", MAX_REFERENCE_CHARACTERS)
        elif field == "speech_traits":
            if value != "":
                validate_speech_traits(value)
        elif field == "accent":
            if value != "":
                from .speech_traits import validate_accent
                validate_accent(value)
        else:
            raise ValueError("Unknown voice field")

    def field_value(self, row, field):
        if field not in FIELDS:
            raise ValueError("Unknown voice field")
        if field == "accent" and not row.get("accent"):
            # Offer the cue the voice description suggests; saving it is still the person's choice.
            return row.get("accent_suggestion") or ""
        return row.get(field) or ""

    def save_field(self, row, field, value):
        self.validate_field(row, field, value)
        voice_id = row["voice_id"]
        return self._submit_save(f"Save {FIELDS[field].lower()} for {label(row)}",
                                 lambda: self.session.update_prompt(voice_id, **{field: value}), row)

    def share(self, row, target):
        if row["voice_id"] == target["voice_id"]:
            raise ValueError("Choose a different voice to share")
        voice_id, target_id = row["voice_id"], target["voice_id"]
        return self._submit(f"Share {label(target)}'s voice with {label(row)}",
                            lambda: self.session.share_voice(voice_id, target_id), row,
                            voices=[canonical(row), canonical(target)])

    def stop_sharing(self, row):
        voice_id = row["voice_id"]
        return self._submit(f"Stop sharing for {label(row)}",
                            lambda: self.session.stop_sharing(voice_id), row)

    def set_clips(self, row, count):
        voice_id = row["voice_id"]
        text = "the default" if count is None else str(count)
        return self._submit_save(f"Use {text} reference clip(s) for {label(row)}",
                                 lambda: self.session.set_reference_clips(voice_id, count), row)

    def refresh_prompt(self, row):
        if row.get("derived_from"):
            raise ValueError("A cloned voice keeps its clone direction; edit it with e")
        voice_id = row["voice_id"]
        return self._submit(f"Refresh description and script for {label(row)}",
                            lambda: self.session.refresh_prompts(voice_ids=[voice_id],
                                                                 config_path=self.config_path), row)

    def revisable(self, rows):
        """The rows whose descriptions ``i`` can revise: designed voices (a clone keeps its direction)."""
        return [row for row in rows if not row.get("derived_from")]

    def revise_descriptions(self, rows, instructions, on_ready):
        """Ask the text model to revise ``rows``' descriptions as ``instructions`` say; nothing is saved.

        ``on_ready({voice_id: (current, proposed)})`` is called from the job's
        thread with the voices the model changed, for the person to review.
        """
        from .description_revision import book_requests, revise
        rows = self.revisable(rows)
        if not rows:
            raise ValueError("A cloned voice keeps its clone direction; choose a designed voice")
        if not " ".join(str(instructions or "").split()):
            raise ValueError("Type what should change in the descriptions")
        voices = [{"voice_id": row["voice_id"], "name": label(row),
                   "description": row.get("prompt") or row.get("description") or "",
                   "accent": row.get("accent") or ""} for row in rows]

        def run():
            factory = self.requests_factory or (lambda: book_requests(
                self.session.bundle, self.config_path, progress=self.log))
            requests = factory()
            if requests is None:
                raise ValueError("Set provider.model in the TOML configuration to revise descriptions with a model")
            proposals = revise(requests, voices, instructions, progress=self.log)
            current = {voice["voice_id"]: voice["description"] for voice in voices}
            changes = {voice_id: (current[voice_id], text) for voice_id, text in proposals.items()}
            self.log(f"The model proposes {len(changes)} changed description(s) of {len(voices)}.")
            on_ready(changes)
            return len(changes)
        return self.jobs.submit(f"Revise {len(voices)} description(s): {display(instructions)[:60]}", run,
                                scope={"revise": [canonical(row) for row in rows]}, lane=ASK_LANE)

    def save_descriptions(self, changes):
        """Save reviewed descriptions as each voice's draft (``{voice_id: description}``); recordings are unchanged."""
        jobs = []
        for voice_id, text in changes.items():
            row = self.row(voice_id)
            if row is not None:
                jobs.append(self.save_field(row, "prompt", text))
        return jobs

    def prepare_pending(self):
        """Prepare every voice without an accepted take, reusing saved takes first."""
        pending = [row for row in self._rows if row.get("selected") is None and not row.get("regeneration")]
        if not pending:
            return None
        voices = [canonical(row) for row in pending]
        ids = [row["voice_id"] for row in pending]

        def run():
            results = self.session.ensure_ready_many(ids)
            by_id = {row["voice_id"]: row for row in pending}
            failed = [(result["voice_id"], result["error"]) for result in results if result["error"] is not None]
            for voice_id, error in failed:
                self.log(f"Could not prepare {label(by_id[voice_id])}: {error}", "error")
            if failed:
                raise ValueError(f"{len(failed)} of {len(results)} voice(s) could not be prepared; "
                                 "select one and press g to retry it")
            return results
        return self._submit(f"Prepare {len(pending)} voice(s) without an accepted take", run, voices=voices)

    def prepare_playback(self):
        """Warm normalized playback copies of accepted takes, as the line menu does."""
        from .audition import _prepare_playback_samples
        return self.jobs.submit("Prepare playback copies",
                                lambda: _prepare_playback_samples(self.player, self.session.rows(), self.log))

    def startup(self):
        """The line menu's opening work, as visible jobs: drafts, pending voices, playback."""
        prepare = getattr(self.session, "prepare_samples", None)
        if callable(prepare):
            self.jobs.submit("Save default voice drafts", prepare)
        self.refresh()
        self.prepare_pending()
        self.prepare_playback()

    def retry(self, job_id):
        return self.jobs.retry(job_id)

    def failures(self):
        return self.jobs.failures()

    def close(self):
        self.jobs.close()
        self.stop()

    def _job_changed(self, job):
        if job.state == RUNNING:
            self.log("Started: " + job.label)
        elif job.state == FAILED:
            self.log(f"Failed: {job.label}: {job.error}", "error")
        elif job.state == "done":
            self.log("Done: " + job.label)
        elif job.state == "cancelled":
            self.log("Cancelled: " + job.label, "warning")
        else:
            self._on_change()
