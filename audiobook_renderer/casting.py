"""Voice casting with automatic selection and immutable audition history.

Startup adopts an existing verified take or generates a missing voice. Bulk
regeneration clears its active choices first and keeps failures visible until
explicitly repaired. Browsing saved voices stays model-free.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from .common import digest, file_hash, read_json, write_json
from .diagnostics import configured_secrets, safe_error_text
from .quality import validate_audio, validation_key
from .voice_bindings import design_voice, designer_identity, require_designer
from .voice_prompts import (MAX_PROMPT_CHARACTERS, audiobook_voice_prompt,
                           default_voice_prompt as _default_voice_prompt, voice_prompt_description)
from .workflow import RenderBlocked, VerificationUnavailable, _seed, verified_take


MAX_REFERENCE_CHARACTERS = 1000
MAX_PENDING_RECORD_BYTES = 4 * 1024 * 1024
MAX_PENDING_AUDIO_BYTES = 128 * 1024 * 1024


def _validate_text(value, label, limit):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    if len(value) > limit:
        raise ValueError(f"{label} must be at most {limit} characters")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
        raise ValueError(f"{label} contains control characters")
    return value


def _require_available_confirmation(report):
    judge = report.get("text_judge") if isinstance(report, dict) else None
    if isinstance(judge, dict) and judge.get("decision") == "unavailable":
        raise VerificationUnavailable(
            "JEV text confirmation is unavailable. The saved voice was retained; "
            "rerun the same command to retry verification without regenerating it."
        )


class CastingSession:
    """Series-scoped casting operations shared by CLI and terminal UI."""

    def __init__(self, bundle, config, *, engine_factory=None, verifier_factory=None,
                 designer_factory=None, notify=print):
        from .book_context import resolve_book_config, voice_store
        config = resolve_book_config(bundle, config)
        self.bundle = deepcopy(bundle)
        self.config = deepcopy(config)
        from .config import resolve_voice_design
        self.config["voice_design"] = resolve_voice_design(config)
        self.notify = notify
        self.store = voice_store(bundle, config)
        self._engine_factory = engine_factory
        self._verifier_factory = verifier_factory
        self._designer_factory = designer_factory
        self._engine = None
        self._verifier = None
        self._designer = None
        self._samples = {}
        self._closed = False
        self._profiles = {voice["id"]: deepcopy(voice) for voice in bundle["cast"]["voices"]}
        if len(self._profiles) != len(bundle["cast"]["voices"]):
            raise ValueError("Duplicate voice ID in cast")
        self._validate_paths()

    @property
    def notice(self):
        if self.config.get("voice_design", {}).get("type") == "gemini":
            return ("Gemini creates new voices remotely from the description and sample script; "
                    "the audiobook engine reuses their verified saved recordings locally.")
        if (self.config.get("voice_design", {}).get("type", "engine") == "engine"
                and self.config.get("engine", {}).get("type") == "drama3_mlx"):
            return "Local Fish Audio Drama 3 designs this voice from the prompt, then reuses the saved recording."
        if self.config.get("voice_design", {}).get("type", "engine") != "engine":
            return "The voice designer applies this prompt; the audiobook engine reuses the saved reference."
        return "Creating a voice requires an engine with voice-design support or a configured voice designer."

    def _validate_paths(self):
        if self._closed:
            raise ValueError("Casting session is closed")
        expected = self.bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
        if expected and expected != self.config["voices"]["series_id"]:
            raise ValueError("Configured voices.series_id must match the analyzer's series ID")
        from .book_context import validate_book_paths
        validate_book_paths(self.bundle, self.config)
        if self.store.directory.is_symlink():
            raise ValueError("The series voice directory must not be a symlink")
        drafts = self.store.directory / "casting-drafts"
        if drafts.is_symlink():
            raise ValueError("The casting draft directory must not be a symlink")
        audits = self.store.directory / "casting-acceptance"
        if audits.is_symlink():
            raise ValueError("The casting acceptance directory must not be a symlink")

    def _profile(self, voice_id):
        if not isinstance(voice_id, str) or voice_id not in self._profiles:
            raise ValueError("Voice ID is not in the bundle cast")
        canonical = self.store.canonical_id(voice_id)
        profile = deepcopy(self._profiles.get(canonical, self._profiles[voice_id]))
        profile["id"] = canonical
        return profile

    @contextmanager
    def _locked(self):
        self._validate_paths()
        lock = self.store.lock()
        try:
            self.store.persist_aliases()
            yield
        finally:
            lock.rmdir()

    def _draft_path(self, voice_id):
        voice_id = self.store.canonical_id(voice_id)
        return self.store.directory / "casting-drafts" / (digest(voice_id)[:24] + ".json")

    def _sample_default(self, voice_id):
        from .samples import select_sample
        canonical = self.store.canonical_id(voice_id)
        if canonical not in self._samples:
            # A later book may expose only an alias while the series store
            # retains a canonical voice ID from an earlier cast.
            present_id = next(item for item in self._profiles
                              if self.store.canonical_id(item) == canonical)
            self._samples[canonical] = select_sample(
                self.bundle, present_id, canonical_id=self.store.canonical_id,
                fallback_text=self.config["voices"]["reference_text"])
        return deepcopy(self._samples[canonical])

    def _is_generic_script(self, text):
        from .config import DEFAULTS
        return text.strip() in {
            DEFAULTS["voices"]["reference_text"].strip(),
            self.config["voices"]["reference_text"].strip(),
        }

    def _sample_draft(self, prompt, text, source, voice_id):
        from .samples import sample_word_count
        # A user-entered script or an established book excerpt survives later
        # books. Legacy defaults and short automatic scripts acquire a fuller
        # audition. This never edits a saved audio transcript or cloning audio.
        if source is not None and (source.get("kind") == "user" or
                (source.get("kind") in {"book_excerpt", "book_excerpts", "casting_script"}
                 and sample_word_count(text) >= 25)):
            return {"prompt": prompt, "reference_text": text, "sample_source": deepcopy(source)}
        if source is None and not self._is_generic_script(text):
            return {"prompt": prompt, "reference_text": text, "sample_source": {"kind": "user"}}
        sample = self._sample_default(voice_id)
        source = sample["source"]
        if sample["reference_text"] != text:
            source["replaces_reference_text"] = text
        return {"prompt": prompt, "reference_text": sample["reference_text"], "sample_source": source}

    def _draft(self, voice_id, selected=None):
        profile = self._profile(voice_id)
        voice_id = self.store.canonical_id(voice_id)
        path = self._draft_path(voice_id)
        if not path.exists() and not path.is_symlink():
            creation = selected.get("creation", {}) if selected else {}
            text = (creation.get("requested_reference_text", selected["reference_text"])
                    if selected else self.config["voices"]["reference_text"])
            prompt = creation.get("requested_description")
            if prompt is None:
                prompt = _default_voice_prompt(profile)
            return self._sample_draft(
                prompt, text,
                creation.get("sample_source"), voice_id)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Casting draft must be a regular file")
        record = read_json(path)
        keys = {"format", "version", "series_id", "voice_id", "prompt", "reference_text", "sha256"}
        if isinstance(record, dict) and record.get("version") == 2:
            keys.add("sample_source")
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError("Invalid casting draft")
        payload = {key: value for key, value in record.items() if key != "sha256"}
        if (record["format"] != "audiobook-casting-draft" or type(record["version"]) is not int
                or record["version"] not in {1, 2} or record["series_id"] != self.store.series_id
                or record["voice_id"] != voice_id or record["sha256"] != digest(payload)):
            raise ValueError("Casting draft identity or checksum changed")
        _validate_text(record["prompt"], "Voice prompt", MAX_PROMPT_CHARACTERS)
        _validate_text(record["reference_text"], "Sample text", MAX_REFERENCE_CHARACTERS)
        source = record.get("sample_source")
        if record["version"] == 2 and (
                not isinstance(source, dict) or source.get("kind") not in {
                    "user", "book_excerpt", "book_excerpts", "casting_script", "fallback"}):
            raise ValueError("Invalid casting sample source")
        return self._sample_draft(record["prompt"], record["reference_text"], source, voice_id)

    def _save_draft(self, voice_id, draft):
        voice_id = self.store.canonical_id(voice_id)
        payload = {
            "format": "audiobook-casting-draft", "version": 2,
            "series_id": self.store.series_id, "voice_id": voice_id, **draft,
        }
        record = {**payload, "sha256": digest(payload)}
        path = self._draft_path(voice_id)
        if path.exists() and read_json(path) == record:
            return False
        write_json(path, record)
        return True

    def prepare_samples(self):
        """Save excerpt defaults without loading a model or replacing a voice."""
        prepared, seen = [], set()
        with self._locked():
            for voice_id in self._profiles:
                canonical = self.store.canonical_id(voice_id)
                if canonical in seen:
                    continue
                seen.add(canonical)
                draft = self._draft(voice_id, self.store.load(voice_id))
                changed = self._save_draft(voice_id, draft)
                prepared.append({"voice_id": canonical, "updated": changed, **deepcopy(draft)})
        return prepared

    def rows(self):
        from .samples import MissingSampleScript
        self._validate_paths()
        cast = self.bundle["cast"]
        character_names = {character["id"]: character["name"] for character in cast["characters"]}
        rows = []
        for voice_id, profile in self._profiles.items():
            selected = self.store.load(voice_id)
            try:
                draft = self._draft(voice_id, selected)
            except MissingSampleScript as exc:
                # Listing, playback, and loudness normalization need no model
                # call or new script. Generation still uses _draft directly
                # and cannot synthesize an unprepared placeholder.
                creation = selected.get("creation", {}) if selected else {}
                draft = {
                    "prompt": creation.get("requested_description", _default_voice_prompt(self._profile(voice_id))),
                    "reference_text": selected["reference_text"] if selected else "",
                    "sample_source": {"kind": "unprepared", "reason": str(exc)},
                }
            characters = [character["name"] for character in cast["characters"]
                          if character.get("voice_id") == voice_id]
            roles = [role for role in cast["roles"] if role.get("voice_id") == voice_id]
            if characters:
                label = ", ".join(characters)
            else:
                narrators = list(dict.fromkeys(
                    character_names[role["character_id"]] for role in roles
                    if role.get("kind") == "narration" and role.get("character_id") in character_names))
                prefix, colon, _ = profile["description"].partition(":")
                prefix = prefix.strip()
                if narrators:
                    label = ", ".join(narrators) + " (narration)"
                elif colon and 0 < len(prefix) <= 60:
                    label = prefix.removesuffix(" voice")
                else:
                    label = ", ".join(dict.fromkeys(role["kind"].capitalize() for role in roles)) or voice_id
            rows.append({
                "voice_id": voice_id,
                "canonical_voice_id": self.store.canonical_id(voice_id),
                "label": label,
                "characters": characters,
                "roles": [role["id"] for role in roles],
                "description": profile["description"],
                **draft,
                "selected": selected,
                "regeneration": self.store.regeneration_status(voice_id),
                "candidates": self.store.list_candidates(voice_id),
                "notice": self.notice,
            })
        return rows

    def update_prompt(self, voice_id, prompt=None, reference_text=None):
        self._profile(voice_id)
        if prompt is not None:
            _validate_text(prompt, "Voice prompt", MAX_PROMPT_CHARACTERS)
        if reference_text is not None:
            _validate_text(reference_text, "Sample text", MAX_REFERENCE_CHARACTERS)
        with self._locked():
            draft = self._draft(voice_id, self.store.load(voice_id))
            voice_id = self.store.canonical_id(voice_id)
            if prompt is not None:
                draft["prompt"] = prompt
            if reference_text is not None:
                draft["reference_text"] = reference_text
                draft["sample_source"] = {"kind": "user"}
            _validate_text(draft["prompt"], "Voice prompt", MAX_PROMPT_CHARACTERS)
            _validate_text(draft["reference_text"], "Sample text", MAX_REFERENCE_CHARACTERS)
            self._save_draft(voice_id, draft)
        return deepcopy(draft)

    def _get_engine(self):
        if self._engine is None:
            factory = self._engine_factory
            if factory is None:
                from .engines import create_engine
                factory = create_engine
            self._engine = factory(deepcopy(self.config["engine"]))
        return self._engine

    def _get_verifier(self):
        if self._verifier is None:
            if self._verifier_factory is None:
                from .verification import create_verifier
                self._verifier = create_verifier(self.config["verification"], self.config.get("jev"))
            else:
                self._verifier = self._verifier_factory(
                    self.config["verification"]["model"], language=self.config["verification"]["language"])
        return self._verifier

    def _get_designer(self):
        engine = self._get_engine()
        settings = self.config.get("voice_design", {"type": "engine"})
        if settings["type"] == "engine":
            try:
                return require_designer(engine)
            except ValueError as exc:
                raise RenderBlocked(str(exc)) from exc
        if self._designer is None:
            factory = self._designer_factory
            if factory is None:
                from .voice_design import create_designer
                factory = create_designer
            self._designer = factory(deepcopy(settings))
        try:
            return require_designer(engine, self._designer)
        except ValueError as exc:
            raise RenderBlocked(str(exc)) from exc

    def generate(self, voice_id):
        """Save a verified candidate without changing the active selection."""
        self._profile(voice_id)
        with self._locked():
            return self._generate_locked(voice_id)

    def _prepare_gemini_request(self, request, directory, *, fresh=False):
        """Reconnect interrupted work, or deliberately start a fresh voice.

        Startup recovery retains an ambiguous creation's marker. Explicit
        regeneration requests a new voice and preserves earlier attempt files
        without reusing or requiring manual repair of their remote identities.
        """
        from .gemini_voice_design import (
            GeminiVoiceDesigner, _read_state, _write_state, _VOICE_ID,
        )

        root = directory.parent
        if root.is_symlink():
            raise RenderBlocked("The casting candidate directory must not be a symlink")
        recoveries = []
        recovery_prompt = None
        for pending_path in (() if fresh else sorted(root.glob("*/gemini-voice-pending.json"))):
            previous_directory = pending_path.parent
            request_path = previous_directory / "casting-request.json"
            try:
                if (previous_directory.is_symlink() or pending_path.is_symlink()
                        or not pending_path.is_file()):
                    raise ValueError
                # Older attempts saved the request after the first failure.
                # New candidates persist it before sending any remote request.
                if not request_path.exists():
                    request_path = next(iter(sorted(previous_directory.glob("attempt-*.json"))))
                    if (request_path.is_symlink() or not request_path.is_file()
                            or request_path.stat().st_size > 4 * 1024 * 1024):
                        raise ValueError
                    previous = read_json(request_path)["request"]
                else:
                    if (request_path.is_symlink() or not request_path.is_file()
                            or request_path.stat().st_size > 4 * 1024 * 1024):
                        raise ValueError
                    previous = read_json(request_path)
                if (not isinstance(previous, dict) or digest(previous) != previous_directory.name
                        or previous.get("kind") != "casting_candidate"
                        or not isinstance(previous.get("voice_id"), str)
                        or not isinstance(previous.get("series_id"), str)):
                    raise ValueError
            except (OSError, ValueError, TypeError, KeyError, StopIteration):
                raise RenderBlocked(
                    f"Cannot identify the pending Gemini creation at {pending_path}. "
                    "Inspect its casting request before generating another remote voice."
                ) from None
            if (previous["series_id"] != request["series_id"]
                    or self.store.canonical_id(previous["voice_id"]) != request["voice_id"]):
                continue
            fingerprint = digest({"designer": previous.get("designer"),
                                  "description": previous.get("prompt")})
            pending = _read_state(pending_path, fingerprint)
            state_path = previous_directory / "gemini-voice.json"
            # A crash after saving the returned ID can leave both files. The
            # completed state is authoritative and requires no remote request.
            state = (_read_state(state_path, fingerprint)
                     if state_path.exists() or state_path.is_symlink() else pending)
            remote_id = state.get("voice_id")
            if not isinstance(remote_id, str) or not _VOICE_ID.fullmatch(remote_id):
                display_name = "Audiobook voice " + digest(str(previous_directory.resolve()))[:16]
                raise RenderBlocked(GeminiVoiceDesigner._pending_message(pending_path, display_name))
            if (previous.get("designer") != request["designer"]
                    or not isinstance(previous.get("prompt"), str)
                    or voice_prompt_description(previous["prompt"]) != voice_prompt_description(request["prompt"])):
                raise RenderBlocked(
                    f"The repaired Gemini creation at {pending_path} belongs to a different "
                    "voice prompt or Gemini configuration. Restore those settings to finish "
                    "this recovery before requesting a new voice."
                )
            if recovery_prompt is not None and recovery_prompt != previous["prompt"]:
                raise RenderBlocked("Multiple repaired Gemini voices have different original prompts; inspect their pending files")
            recovery_prompt = previous["prompt"]
            recoveries.append((pending_path, remote_id, fingerprint, previous_directory.name))
        if len({item[1] for item in recoveries}) > 1:
            raise RenderBlocked(
                "Multiple different repaired Gemini voices exist for this character. "
                "Inspect the pending files before choosing which remote voice to recover."
            )
        if recoveries and recovery_prompt != request["prompt"]:
            # Reconnecting an existing remote voice must retain the exact
            # description that created it, not claim newly added context.
            request = dict(request, prompt=recovery_prompt)
            directory = directory.parent / digest(request)
        if directory.exists() or directory.is_symlink():
            raise RenderBlocked("Casting candidate directory already exists")
        directory.mkdir(parents=True)
        write_json(directory / "casting-request.json", request)
        if not recoveries:
            return request, directory
        pending_path, remote_id, fingerprint, previous_hash = recoveries[0]
        recovered = {
            "version": 1, "request_fingerprint": fingerprint, "voice_id": remote_id,
            "recovered_from_pending": True, "recovered_from_request_hash": previous_hash,
        }
        # The new request and its remote ID are both durable before resolving
        # any original pending marker. Keep the source ID as an audit record.
        _write_state(directory / "gemini-voice.json", recovered)
        for pending_path, remote_id, fingerprint, previous_hash in recoveries:
            state_path = pending_path.parent / "gemini-voice.json"
            if not state_path.exists():
                _write_state(state_path, {
                    "version": 1, "request_fingerprint": fingerprint, "voice_id": remote_id,
                    "recovered_from_pending": True, "recovered_to_request_hash": directory.name,
                })
            pending_path.unlink()
        self.notify("Recovered the existing Gemini voice; no new remote voice will be created.")
        return request, directory

    def _pending_confirmation(self, request, root):
        """Find one unpublished, intact take awaiting only external verification."""
        root = Path(root).absolute()
        if any(path.is_symlink() for path in (root, *root.parents)):
            raise RenderBlocked("Pending casting verification cannot follow symlink directories")
        if not root.exists():
            return None
        if not root.is_dir():
            raise RenderBlocked("The casting candidate directory must be a directory")

        def read_record(path):
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size > MAX_PENDING_RECORD_BYTES):
                raise ValueError("unsafe record")
            record = read_json(path)
            if not isinstance(record, dict):
                raise ValueError("invalid record")
            return record

        saved = self.store.list_candidates(request["voice_id"])
        published_ids = {item["candidate_id"] for item in saved}
        published_hashes = {item.get("creation", {}).get("request_hash") for item in saved}
        current = {key: value for key, value in request.items() if key != "nonce"}
        matches = []
        for directory in sorted(root.iterdir()):
            if directory.is_symlink():
                raise RenderBlocked("Pending casting verification cannot follow symlink directories")
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                continue
            accepted = directory / "accepted.json"
            if accepted.exists() or accepted.is_symlink() or directory.name in published_hashes:
                continue
            paths = sorted(directory.glob("attempt-*.json"))
            if not paths:
                continue
            path = paths[-1]
            try:
                record = read_record(path)
                report = record.get("report", {})
                judge = report.get("text_judge") if isinstance(report, dict) else None
                if (not isinstance(judge, dict) or judge.get("decision") != "unavailable"
                        or report.get("purpose") != "reference" or report.get("passed") is not False):
                    continue
                previous = record.get("request")
                if (not isinstance(previous, dict) or record.get("request_hash") != digest(previous)
                        or digest(previous) != directory.name
                        or not isinstance(previous.get("nonce"), str)
                        or not re.fullmatch(r"[0-9a-f]{32}", previous["nonce"])):
                    raise ValueError("request checksum")
                if previous["nonce"] in published_ids:
                    continue
                if {key: value for key, value in previous.items() if key != "nonce"} != current:
                    continue
                attempt = record.get("attempt")
                if (type(attempt) is not int or not 1 <= attempt <= 10
                        or path.name != f"attempt-{attempt:03d}.json"
                        or type(record.get("seed")) is not int
                        or record.get("seed") != _seed(previous["base_seed"], previous, attempt)
                        or report.get("expected_text") != previous["text"]):
                    raise ValueError("attempt identity")
                name = f"attempt-{attempt:03d}.wav"
                wav = directory / name
                if (record.get("audio_file") != name or wav.is_symlink() or not wav.is_file()
                        or not 0 < wav.stat().st_size <= MAX_PENDING_AUDIO_BYTES
                        or not isinstance(record.get("audio_sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", record["audio_sha256"])
                        or file_hash(wav) != record["audio_sha256"]):
                    raise ValueError("audio checksum")
                generated = record.get("engine_metadata")
                if (not isinstance(generated, dict) or generated.get("description_applied") is not True
                        or generated.get("requested_description", previous["prompt"]) != previous["prompt"]
                        or generated.get("reference_text", previous["text"]) != previous["text"]):
                    raise ValueError("voice description")
                durable_request = directory / "casting-request.json"
                if (durable_request.exists() or durable_request.is_symlink()) and read_record(durable_request) != previous:
                    raise ValueError("durable request checksum")
                matches.append((directory, previous, attempt))
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                raise RenderBlocked(
                    f"Cannot safely resume pending voice verification at {path}; "
                    "inspect its request and audio before generating another voice."
                ) from None
        if len(matches) > 1:
            raise RenderBlocked("Multiple matching voice samples await verification; inspect the casting candidates before choosing one")
        return matches[0] if matches else None

    def _generate_locked(self, voice_id, *, fresh=False):
        profile = self._profile(voice_id)
        draft = self._draft(voice_id, self.store.load(voice_id))
        prompt = audiobook_voice_prompt(_validate_text(draft["prompt"], "Voice prompt", MAX_PROMPT_CHARACTERS))
        text = _validate_text(draft["reference_text"], "Sample text", MAX_REFERENCE_CHARACTERS)
        engine, designer = self._get_engine(), self._get_designer()
        verifier = self._get_verifier()
        self._save_draft(voice_id, draft)
        voice_id = self.store.canonical_id(voice_id)
        nonce = uuid.uuid4().hex
        request = {
            "kind": "casting_candidate", "nonce": nonce, "engine": engine.identity,
            "designer": designer_identity(designer),
            "series_id": self.store.series_id, "voice_id": voice_id, "profile": profile,
            "prompt": prompt, "text": text, "base_seed": self.config["render"]["seed"],
            "sample_source": deepcopy(draft["sample_source"]),
            "source_fingerprint": self.bundle["manifest"]["source_fingerprint"],
        }
        directory = Path(self.config["render"]["work_dir"]) / "casting-candidates" / digest(request)
        pending = None if fresh else self._pending_confirmation(request, directory.parent)
        if not fresh and pending is None and voice_prompt_description(prompt) != prompt:
            # A pre-context sample awaiting verification keeps its actual
            # request and recording; this is recovery, not a new design.
            legacy_request = dict(request, prompt=voice_prompt_description(prompt))
            pending = self._pending_confirmation(legacy_request, directory.parent)
        if pending is not None:
            directory, request, resume_attempt = pending
            nonce = request["nonce"]
            self.notify(f"Retrying verification of the saved voice sample for {voice_id}; no new voice is being generated.")
        else:
            resume_attempt = None
        if pending is None and self.config.get("voice_design", {}).get("type") == "gemini":
            request, directory = self._prepare_gemini_request(request, directory, fresh=fresh)
        prompt = request["prompt"]
        if pending is None:
            self.notify(f"Generating voice sample for {voice_id} using {designer.identity.get('model', 'configured designer')}")
        wav, accepted = verified_take(
            text, lambda path, seed: design_voice(designer, prompt, text, path, seed=seed),
            directory,
            request, verifier, self.config, notify=self.notify, purpose="reference", resume_attempt=resume_attempt,
        )
        if accepted.get("engine_metadata", {}).get("description_applied") is not True:
            raise RenderBlocked("Saved generation has no evidence that the voice description was applied")
        reference_text = accepted["report"].get("reference_text", text)
        metadata = {
            **accepted.get("engine_metadata", {}),
            "method": "casting_candidate", "candidate_id": nonce,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_description": prompt,
            "requested_reference_text": text,
            "sample_source": deepcopy(draft["sample_source"]),
            "reference_text": reference_text,
            "description_applied": True,
            "voice_design_supported": True,
            "designer": designer_identity(designer),
            "note": "The voice designer received the character description; listening determines casting suitability.",
            "seed": accepted["seed"], "request_hash": accepted["request_hash"],
            "verifier": deepcopy(verifier.identity), "quality": deepcopy(self.config["quality"]),
            "validation_key": validation_key(reference_text, verifier, self.config["quality"],
                                             purpose="reference"),
        }
        return self.store.publish_candidate(
            voice_id, deepcopy(profile), wav, reference_text, engine.identity,
            accepted["report"], metadata, candidate_id=nonce)

    def ensure_ready(self, voice_id):
        """Reuse an active voice, adopt the latest saved take, or create one.

        Existing active voices win even when a newer unselected candidate exists.
        Selecting a previously saved candidate applies the same identity and
        verification checks as an explicit selection. A generation interrupted
        after publishing its candidate can therefore be resumed without another
        synthesis request.
        """
        self._profile(voice_id)
        with self._locked():
            selected = self.store.load(voice_id)
            if selected is not None:
                return selected
            regeneration = self.store.regeneration_status(voice_id)
            if regeneration is not None:
                raise RenderBlocked(
                    f"Voice {voice_id} regeneration is {regeneration['state']}; no active voice. "
                    "Use G or GG in the casting menu to retry."
                )
            candidates = self.store.list_candidates(voice_id)
            designed = [candidate for candidate in candidates if candidate.get("description_applied") is True]
            candidate = designed[-1] if designed else self._generate_locked(voice_id)
            return self._accept_locked(voice_id, candidate["candidate_id"])

    @contextmanager
    def _using_voice_designer(self, choice):
        """Select a designer for one action, preserving the session default."""
        if choice is None:
            yield
            return
        from .config import select_voice_designer
        settings = select_voice_designer(self.config, choice)["voice_design"]
        original_settings = self.config["voice_design"]
        original_designer = self._designer
        self.config["voice_design"] = settings
        self._designer = None
        try:
            yield
        finally:
            temporary_designer = self._designer
            self.config["voice_design"] = original_settings
            self._designer = original_designer
            if temporary_designer is not original_designer and temporary_designer is not self._engine:
                close = getattr(temporary_designer, "close", None)
                if callable(close):
                    close()

    def begin_regeneration(self, voice_ids):
        """Clear all bulk targets before synthesis, preserving their casting edits."""
        with self._locked():
            drafts = {}
            for voice_id in voice_ids:
                self._profile(voice_id)
                canonical = self.store.canonical_id(voice_id)
                if canonical not in drafts:
                    draft = self._draft(voice_id, self.store.load(voice_id))
                    _validate_text(draft["prompt"], "Voice prompt", MAX_PROMPT_CHARACTERS)
                    _validate_text(draft["reference_text"], "Sample text", MAX_REFERENCE_CHARACTERS)
                    drafts[canonical] = draft
            # Preflight every target before removing any active selection. A
            # selected reference may be the only source of custom casting edits.
            for canonical, draft in drafts.items():
                self._save_draft(canonical, draft)
            for canonical in drafts:
                self.store.invalidate(canonical)

    def regenerate(self, voice_id, *, voice_designer=None):
        """Create a fresh voice and select its verified reference under one lock."""
        self._profile(voice_id)
        with self._locked():
            regeneration = self.store.regeneration_status(voice_id)
            if regeneration and regeneration["inherited"]:
                # A retry in a descendant owns its new state, never the parent's.
                self._save_draft(voice_id, self._draft(voice_id))
                self.store.invalidate(voice_id)
            try:
                with self._using_voice_designer(voice_designer):
                    candidate = self._generate_locked(voice_id, fresh=True)
                    selected = self._accept_locked(voice_id, candidate["candidate_id"])
            except (Exception, KeyboardInterrupt) as exc:
                if regeneration and self.store.regeneration_status(voice_id):
                    error = ("Regeneration interrupted." if isinstance(exc, KeyboardInterrupt)
                             else safe_error_text(exc, secrets=configured_secrets(self.config)))
                    self.store.mark_regeneration_failed(voice_id, error.strip() or type(exc).__name__)
                raise
            remote_id = selected.get("creation", {}).get("remote_voice_id")
            if remote_id:
                self.notify(f"Selected Gemini voice ID: {remote_id}")
            return selected

    def recover_attempt(self, voice_id, attempt_path, *, select=True):
        """Recheck and preserve a saved casting take without synthesizing again.

        Recovery binds the original request and normalized audio to their saved
        hashes. It never edits the rejected attempt or replaces an existing
        candidate. Selecting a recovered take also requires the current draft to
        match, so recovering an older experiment cannot silently undo later edits.
        """
        profile = self._profile(voice_id)
        with self._locked():
            voice_id = self.store.canonical_id(voice_id)
            path = Path(attempt_path).absolute()
            if (path.is_symlink() or not path.is_file()
                    or any(parent.is_symlink() for parent in path.parents)):
                raise ValueError("Casting attempt must be a regular, nonsymlink file")
            original_hash = file_hash(path)
            record = read_json(path)
            if not isinstance(record, dict) or not isinstance(record.get("request"), dict):
                raise ValueError("Invalid casting attempt record")
            request = record["request"]
            if record.get("request_hash") != digest(request):
                raise ValueError("Casting attempt request checksum changed")
            if (request.get("kind") != "casting_candidate"
                    or request.get("series_id") != self.store.series_id
                    or request.get("voice_id") != voice_id):
                raise ValueError("Casting attempt belongs to a different series or voice")
            if request.get("profile") != profile:
                raise ValueError("Casting attempt character profile does not match this cast")
            prompt = _validate_text(request.get("prompt"), "Voice prompt", MAX_PROMPT_CHARACTERS)
            text = _validate_text(request.get("text"), "Sample text", MAX_REFERENCE_CHARACTERS)
            if (type(record.get("attempt")) is not int or record["attempt"] < 1
                    or type(record.get("seed")) is not int or not 0 <= record["seed"] < 2**32):
                raise ValueError("Invalid casting attempt number or seed")
            name = record.get("audio_file")
            if (not isinstance(name, str) or not name or name != Path(name).name
                    or "\\" in name or not name.endswith(".wav")):
                raise ValueError("Invalid casting attempt audio path")
            wav = path.parent / name
            if wav.is_symlink() or not wav.is_file():
                raise ValueError("Casting attempt audio must be a regular, nonsymlink file")
            audio_hash = record.get("audio_sha256")
            if (not isinstance(audio_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", audio_hash)
                    or file_hash(wav) != audio_hash):
                raise ValueError("Casting attempt audio checksum changed")
            generated = record.get("engine_metadata", {})
            if (not isinstance(generated, dict) or generated.get("description_applied") is not True
                    or generated.get("requested_description", prompt) != prompt
                    or generated.get("reference_text", text) != text):
                raise RenderBlocked("Casting attempt has no matching applied voice description")
            if select:
                draft = self._draft(voice_id, self.store.load(voice_id))
                script_matches = (draft["reference_text"] == text or
                                  draft["sample_source"].get("replaces_reference_text") == text)
                if voice_prompt_description(draft["prompt"]) != voice_prompt_description(prompt) or not script_matches:
                    raise RenderBlocked("Casting attempt does not match the current prompt and sample text")
            engine, verifier = self._get_engine(), self._get_verifier()
            self._check_engine(request.get("engine", {}), engine.identity)
            self.notify(f"Rechecking saved voice sample for {voice_id}")
            report = validate_audio(wav, text, verifier, self.config["quality"], purpose="reference")
            if not report.get("passed"):
                _require_available_confirmation(report)
                raise RenderBlocked("Saved voice sample failed current verification: "
                                    + ", ".join(report.get("reasons", [])))
            reference_text = report.get("reference_text", text)
            # Recheck after transcription before copying or selecting anything.
            if file_hash(path) != original_hash or file_hash(wav) != audio_hash:
                raise RenderBlocked("Casting attempt changed while it was being verified")
            candidate_id = digest({"recovered_casting_attempt": record,
                                   "reference_text": reference_text})[:32]
            candidate = next((saved for saved in self.store.list_candidates(voice_id)
                              if saved["candidate_id"] == candidate_id), None)
            provenance = {
                "attempt_file": str(path), "attempt_sha256": original_hash,
                "request_hash": record["request_hash"], "attempt": record["attempt"],
                "audio_file": name, "audio_sha256": audio_hash,
            }
            if candidate is None:
                metadata = {
                    **deepcopy(generated),
                    "method": "recovered_casting_candidate", "candidate_id": candidate_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "requested_description": prompt, "description_applied": True,
                    "requested_reference_text": text, "reference_text": reference_text,
                    **({"sample_source": deepcopy(request["sample_source"])}
                       if "sample_source" in request else {}),
                    "voice_design_supported": True, "seed": record["seed"],
                    "request_hash": record["request_hash"], "recovered_from": provenance,
                    "designer": deepcopy(request.get("designer", {})),
                    "verifier": deepcopy(verifier.identity), "quality": deepcopy(self.config["quality"]),
                    "validation_key": validation_key(
                        reference_text, verifier, self.config["quality"], purpose="reference"),
                    "note": "Recovered without resynthesis after rechecking the original saved take.",
                }
                candidate = self.store.publish_candidate(
                    voice_id, deepcopy(request["profile"]), wav, reference_text, request["engine"],
                    report, metadata, candidate_id=candidate_id)
            elif (candidate["reference_sha256"] != audio_hash
                  or candidate["reference_text"] != reference_text
                  or candidate.get("creation", {}).get("recovered_from", {}).get("attempt_sha256")
                  != original_hash):
                raise RenderBlocked("Recovered candidate provenance does not match the saved attempt")
            if not select:
                return candidate
            return self._accept_locked(voice_id, candidate_id)

    def generate_missing(self):
        """Make each unselected voice ready, reusing saved candidates first."""
        selected = []
        seen = set()
        for row in self.rows():
            canonical = row["canonical_voice_id"]
            if row["selected"] is None and canonical not in seen:
                seen.add(canonical)
                selected.append(self.ensure_ready(row["voice_id"]))
        return selected

    def accept(self, voice_id, candidate_id):
        """Select a saved candidate after checking engine and verification policy."""
        self._profile(voice_id)
        with self._locked():
            return self._accept_locked(voice_id, candidate_id)

    def _accept_locked(self, voice_id, candidate_id):
        voice_id = self.store.canonical_id(voice_id)
        selected = self.store.load(voice_id)
        candidate = self.store.load_candidate(voice_id, candidate_id)
        if candidate is None:
            raise ValueError("Voice candidate does not exist")
        engine, verifier = self._get_engine(), self._get_verifier()
        created = candidate.get("created_with", {})
        identity = engine.identity
        self._check_engine(created, identity)
        creation = candidate.get("creation", {})
        key = validation_key(candidate["reference_text"], verifier, self.config["quality"],
                             purpose="reference")
        if (creation.get("validation_key") == key and creation.get("quality") == self.config["quality"]
                and creation.get("verifier") == verifier.identity
                and candidate["verification"].get("passed") is True):
            report = candidate["verification"]
        else:
            self.notify(f"Rechecking {voice_id} against the current verification settings")
            report = validate_audio(
                candidate["reference_audio"], candidate["reference_text"], verifier,
                self.config["quality"], purpose="reference")
            if not report.get("passed"):
                _require_available_confirmation(report)
                raise RenderBlocked("Voice candidate failed current verification: " + ", ".join(report["reasons"]))
            if report.get("reference_text", candidate["reference_text"]) != candidate["reference_text"]:
                raise RenderBlocked(
                    "The saved voice needs a corrected reference transcript. Recover its original "
                    "attempt to preserve a new candidate with matching audio and text.")
        # Store the decision's evidence independently of the immutable take.
        audit = {
            "format": "audiobook-casting-acceptance", "version": 1,
            "series_id": self.store.series_id, "voice_id": voice_id,
            "candidate_id": candidate_id, "asset_hash": candidate["asset_hash"],
            "reference_sha256": file_hash(candidate["reference_audio"]),
            "previous_asset_hash": selected["asset_hash"] if selected else None,
            "engine": identity, "verifier": verifier.identity, "quality": self.config["quality"],
            "validation_key": key, "report": report,
        }
        write_json(self.store.directory / "casting-acceptance" / (uuid.uuid4().hex + ".json"), audit)
        return self.store.select_candidate(
            voice_id, candidate_id, expected_asset_hash=selected["asset_hash"] if selected else None)

    @staticmethod
    def _check_engine(created, identity):
        if not isinstance(created, dict):
            raise RenderBlocked("Candidate has no playback engine identity")
        if any(created.get(key) != identity.get(key) for key in ("engine", "model")):
            raise RenderBlocked("Candidate was created for a different engine or model")
        if ("checkpoint_files_sha256" in identity
                and created.get("checkpoint_files_sha256") != identity["checkpoint_files_sha256"]):
            raise RenderBlocked("Candidate was created with different model weights")

    def close(self):
        if self._closed:
            return
        self._closed = True
        for resource in (self._designer, self._engine, self._verifier):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
