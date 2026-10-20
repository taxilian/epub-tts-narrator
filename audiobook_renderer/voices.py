"""Immutable local voice assets with read-only inheritance from earlier books."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import tempfile

from .common import digest, file_hash, read_json, write_json


class VoiceStore:
    """Immutable voice references and explicit, reversible active selections.

    Callers must hold ``lock()`` while publishing or selecting voices. Revisions
    never overwrite a reference; only the small selection pointer is replaced.
    Original v1 assets remain in their original folders with unchanged hashes.

    ``parents`` are earlier books' stores, nearest first. Active lookup is live:
    a reference created in an earlier book after this store was opened is still
    available. All writes, candidate operations, and locks belong to this store.
    Inherited references are never copied just to reuse them. Returned bindings
    include ``inherited`` and ``voice_directory`` as unhashed lookup metadata.

    ``alias_loader`` reads this book's editable explicit links on every lookup.
    In that mode saved alias metadata is an integrity-checked cache only; the
    current cast and local policy remain authoritative. Parent policy is never
    copied into the child's cache.
    """

    LEGACY_CANDIDATE = "legacy"

    def __init__(self, directory, series_id, *, aliases=None, cast=None, explicit_aliases=None,
                 parents=(), alias_loader=None):
        from .voice_bindings import canonical_aliases, cast_voice_owners, validate_aliases
        self.series_id = series_id
        self.directory = Path(directory) / digest(series_id)[:24]
        self.parents = self._parent_stores(parents)
        if alias_loader is not None and not callable(alias_loader):
            raise ValueError("A book voice alias loader must be callable")
        if alias_loader is not None and explicit_aliases:
            raise ValueError("Book voice aliases must come from the book alias file")
        self._alias_loader = alias_loader
        self._explicit_aliases = validate_aliases(explicit_aliases or {})
        self._aliases = validate_aliases({
            **(aliases or {}),
            **(canonical_aliases(cast, self._explicit_aliases) if cast is not None
               else self._explicit_aliases),
        })
        self._owners = cast_voice_owners(cast) if cast is not None else {}
        if cast is not None or self.parents or alias_loader is not None:
            self._alias_state()

    @property
    def alias_loader(self):
        """The live local policy reader, when this store belongs to a book."""
        return self._alias_loader

    def _parent_stores(self, parents):
        """Flatten recursive ancestors in lookup order without changing ownership."""
        result, visited = [], {self.directory.resolve()}
        for parent in parents:
            if not isinstance(parent, VoiceStore) or parent.series_id != self.series_id:
                raise ValueError("Inherited voice stores must belong to the same series")
            for ancestor in (parent, *parent.parents):
                path = ancestor.directory.resolve()
                if path == self.directory.resolve():
                    raise ValueError("A voice store cannot inherit from itself")
                if path not in visited:
                    visited.add(path)
                    result.append(ancestor)
        return tuple(result)

    def _local_alias_state(self):
        from .voice_bindings import validate_aliases, validate_alias_ownership
        path = self.directory / "voice-aliases.json"
        saved, explicit, owners = {}, {}, {}
        if path.exists() or path.is_symlink():
            self._check_directory(self.directory, required=True)
            self._regular_file(path)
            record = read_json(path)
            fields = {"format", "version", "series_id", "aliases", "sha256"}
            if isinstance(record, dict) and record.get("version") == 2:
                fields |= {"explicit_aliases", "voice_owners"}
            if (not isinstance(record, dict) or set(record) != fields
                    or record["format"] != "audiobook-voice-aliases"
                    or type(record["version"]) is not int or record["version"] not in (1, 2)
                    or record["series_id"] != self.series_id
                    or record["sha256"] != digest({k: v for k, v in record.items() if k != "sha256"})):
                raise ValueError("Invalid series voice aliases or checksum")
            saved = validate_aliases(record["aliases"])
            if record["version"] == 2:
                explicit = validate_aliases(record["explicit_aliases"])
                owners = record["voice_owners"]
                if (not isinstance(owners, dict) or any(
                    not isinstance(key, str) or not key.strip()
                    or not isinstance(value, str) or not value.strip()
                    for key, value in owners.items()
                )):
                    raise ValueError("Invalid persisted voice character ownership")
                if any(saved.get(key) != value for key, value in explicit.items()):
                    raise ValueError("Explicit alias provenance does not match saved aliases")
        if self._alias_loader is not None:
            # This file is a checked cache, not editable book policy. Older
            # versions copied inherited policy into it, so its edges must not
            # freeze a parent's later edits or resurrect a deleted local edge.
            explicit = validate_aliases(self._alias_loader())
            aliases = validate_aliases({**self._aliases, **explicit})
        else:
            for key, value in self._aliases.items():
                if key in saved and saved[key] != value:
                    raise ValueError("A persisted voice alias has a different canonical identity")
            aliases = validate_aliases({**saved, **self._aliases})
            explicit = validate_aliases({**explicit, **self._explicit_aliases})
        if any(aliases.get(key) != value for key, value in explicit.items()):
            raise ValueError("Explicit alias provenance does not match effective aliases")
        for voice_id, character_id in self._owners.items():
            if (voice_id in owners and owners[voice_id] != character_id
                    and voice_id not in explicit):
                raise ValueError("A persisted voice ID now belongs to a different character")
        owners = {**owners, **self._owners}
        validate_alias_ownership(aliases, explicit, owners)
        return aliases, explicit, owners

    def _alias_state(self):
        """Combine identity evidence, never infer links from character names.

        The nearest explicit policy wins for its source voice, including over
        automatic cast relationships in later books. Without an explicit
        decision, conflicting automatic aliases fail closed. Ownership remains
        exact; names never establish identity.
        """
        from .voice_bindings import validate_aliases, validate_alias_ownership
        states = [store._local_alias_state() for store in (*reversed(self.parents), self)]
        aliases, explicit, owners = {}, {}, {}
        for _, local_explicit, _ in states:
            explicit.update(local_explicit)
        for local_aliases, local_explicit, local_owners in states:
            for key, value in local_aliases.items():
                if key not in explicit and key in aliases and aliases[key] != value:
                    raise ValueError("An inherited voice alias has a different canonical identity")
            for voice_id, character_id in local_owners.items():
                if (voice_id in owners and owners[voice_id] != character_id
                        and voice_id not in local_explicit):
                    raise ValueError("An inherited voice ID now belongs to a different character")
            aliases.update(local_aliases)
            owners.update(local_owners)
        aliases.update(explicit)
        aliases = validate_aliases(aliases)
        explicit = validate_aliases(explicit)
        validate_alias_ownership(aliases, explicit, owners)
        return aliases, explicit, owners

    def _alias_mapping(self):
        return self._alias_state()[0]

    def canonical_id(self, voice_id):
        aliases = self._alias_mapping()
        while voice_id in aliases:
            voice_id = aliases[voice_id]
        return voice_id

    def persist_aliases(self):
        """Save checked local metadata while the caller holds this store's lock."""
        # Validate the full chain, but never turn inherited policy into local
        # policy. Managed books keep editable decisions in their own root file.
        aliases, explicit, owners = self._alias_state()
        if self._alias_loader is not None:
            aliases, explicit, _ = self._local_alias_state()
            owners = dict(self._owners)
        path = self.directory / "voice-aliases.json"
        if not aliases and not owners and not (self._alias_loader is not None and path.exists()):
            return
        payload = {"format": "audiobook-voice-aliases", "version": 2,
                   "series_id": self.series_id, "aliases": aliases,
                   "explicit_aliases": explicit, "voice_owners": owners}
        self._prepare_directory(self.directory)
        write_json(path, {**payload, "sha256": digest(payload)})

    def folder(self, voice_id):
        return self.directory / digest(voice_id)[:24]

    @staticmethod
    def _validate_candidate_id(candidate_id, *, legacy=True):
        if (legacy and candidate_id == VoiceStore.LEGACY_CANDIDATE):
            return candidate_id
        if not isinstance(candidate_id, str) or not re.fullmatch(r"[0-9a-f]{32}", candidate_id):
            raise ValueError("Candidate ID must be a lowercase UUID hex string")
        return candidate_id

    def _candidate_folder(self, voice_id, candidate_id):
        self._validate_candidate_id(candidate_id)
        if candidate_id == self.LEGACY_CANDIDATE:
            return self.folder(voice_id)
        return self.directory / "candidates" / digest(voice_id)[:24] / candidate_id

    def _selection_path(self, voice_id):
        return self.directory / "selections" / (digest(voice_id)[:24] + ".json")

    def _check_directory(self, path, *, required=False):
        """Reject symlinks in every renderer-owned component, even dangling ones."""
        path = Path(path)
        relative = path.relative_to(self.directory)
        current = self.directory.parent
        for part in (None, self.directory.name, *relative.parts):
            if part is not None:
                current = current / part
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise ValueError("Voice storage directories must not be symlinks or files")
        if required and not path.is_dir():
            raise ValueError("Voice candidate does not exist")

    def _prepare_directory(self, path):
        self._check_directory(path)
        path.mkdir(parents=True, exist_ok=True)
        self._check_directory(path, required=True)

    @staticmethod
    def _regular_file(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Voice asset must be a regular, nonsymlink file")
        return path

    def _read_binding(self, voice_id, candidate_id):
        folder = self._candidate_folder(voice_id, candidate_id)
        self._check_directory(folder, required=True)
        self._regular_file(folder / "voice.json")
        self._regular_file(folder / "voice.sha256")
        record = read_json(folder / "voice.json")
        if (folder / "voice.sha256").read_text().strip() != file_hash(folder / "voice.json"):
            raise ValueError("Persisted voice metadata changed")
        if (not isinstance(record, dict) or
                record.get("format") != "audiobook-voice" or
                type(record.get("version")) is not int or record.get("version") != 1 or
                record.get("series_id") != self.series_id or record.get("voice_id") != voice_id):
            raise ValueError("Invalid voice registry binding")
        expected_id = None if candidate_id == self.LEGACY_CANDIDATE else candidate_id
        if record.get("candidate_id") != expected_id:
            raise ValueError("Invalid voice candidate binding")
        # A discriminated binding allows future engines with preset voice IDs.
        if record.get("kind") != "reference":
            raise ValueError("This engine requires a reference voice; preset bindings need a compatible backend")
        if (not isinstance(record.get("reference_text"), str) or not record["reference_text"].strip()
                or not isinstance(record.get("verification"), dict)
                or record.get("verification", {}).get("passed") is not True):
            raise ValueError("Persisted reference voice has no verified transcript")
        path = self._regular_file(folder / "reference.wav")
        if file_hash(path) != record.get("reference_sha256"):
            raise ValueError("Persisted voice reference changed; refusing to regenerate or replace it")
        return dict(record, candidate_id=candidate_id, reference_audio=str(path.resolve()),
                    asset_hash=digest(record), inherited=False,
                    voice_directory=str(self.directory.resolve()))

    def _read_selection(self, voice_id):
        path = self._selection_path(voice_id)
        self._check_directory(path.parent)
        if not path.exists() and not path.is_symlink():
            return None
        self._regular_file(path)
        pointer = read_json(path)
        fields = {"format", "version", "series_id", "voice_id", "candidate_id",
                  "asset_hash", "selection_sha256"}
        if (not isinstance(pointer, dict) or set(pointer) != fields or
                pointer.get("format") != "audiobook-voice-selection" or
                type(pointer.get("version")) is not int or pointer.get("version") != 1 or
                pointer.get("series_id") != self.series_id or pointer.get("voice_id") != voice_id):
            raise ValueError("Invalid active voice selection")
        self._validate_candidate_id(pointer["candidate_id"])
        if (not isinstance(pointer["asset_hash"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", pointer["asset_hash"]) or
                pointer["selection_sha256"] != digest({
                    key: value for key, value in pointer.items() if key != "selection_sha256"
                })):
            raise ValueError("Active voice selection changed")
        return pointer

    def load(self, voice_id):
        """Resolve the nearest active reference without writing or copying it."""
        voice_id = self.canonical_id(voice_id)
        local = self._load_local(voice_id)
        if local is not None:
            return local
        for parent in self.parents:
            binding = parent._load_local(voice_id)
            if binding is not None:
                return dict(binding, inherited=True)
        return None

    def _load_local(self, voice_id):
        """Read a canonical ID locally; invalid overrides never fall through."""
        pointer = self._read_selection(voice_id)
        if pointer is not None:
            binding = self._read_binding(voice_id, pointer["candidate_id"])
            if binding["asset_hash"] != pointer["asset_hash"]:
                raise ValueError("Active voice candidate changed")
            return binding
        folder = self.folder(voice_id)
        self._check_directory(folder)
        if not folder.exists():
            return None
        return self._read_binding(voice_id, self.LEGACY_CANDIDATE)

    def load_candidate(self, voice_id, candidate_id):
        """Read a local revision; ancestor candidates cannot be selected here."""
        voice_id = self.canonical_id(voice_id)
        return self._read_binding(voice_id, candidate_id)

    def list_candidates(self, voice_id):
        """List local revisions only, with their effective selection state.

        The effective inherited reference is available through ``load()``. It
        is not a local candidate and must never be passed to local selection.
        """
        voice_id = self.canonical_id(voice_id)
        active = self.load(voice_id)  # Invalid or dangling pointers must fail closed.
        candidates = []
        legacy = self.folder(voice_id)
        self._check_directory(legacy)
        if legacy.exists():
            candidates.append(self.load_candidate(voice_id, self.LEGACY_CANDIDATE))
        revisions = self.directory / "candidates" / digest(voice_id)[:24]
        self._check_directory(revisions)
        if revisions.exists():
            alternatives = []
            for path in sorted(revisions.iterdir()):
                self._validate_candidate_id(path.name, legacy=False)
                candidate = self.load_candidate(voice_id, path.name)
                created_ns = (path / "voice.json").stat().st_mtime_ns
                created_at = candidate.get("creation", {}).get("created_at")
                if isinstance(created_at, str):
                    try:
                        timestamp = datetime.fromisoformat(created_at)
                        if timestamp.tzinfo is not None:
                            created_ns = int(timestamp.timestamp() * 1_000_000_000)
                    except (ValueError, OverflowError):
                        pass  # Older integrations may not supply an ISO timestamp.
                alternatives.append((created_ns, path.name, candidate))
            candidates.extend(candidate for _, _, candidate in sorted(alternatives))
        return [
            dict(candidate, selected=active is not None and not active["inherited"] and
                 candidate["candidate_id"] == active["candidate_id"] and
                 candidate["asset_hash"] == active["asset_hash"])
            for candidate in candidates
        ]

    def select_candidate(self, voice_id, candidate_id, expected_asset_hash=None):
        """Activate a revision under the caller's series lock.

        ``expected_asset_hash``, when supplied, must match the CURRENT active
        binding, so an outdated audition cannot silently replace a newer choice.
        """
        voice_id = self.canonical_id(voice_id)
        active = self.load(voice_id)
        if expected_asset_hash is not None and (
                active is None or active["asset_hash"] != expected_asset_hash):
            raise ValueError("Active voice changed; reload before selecting a candidate")
        candidate = self.load_candidate(voice_id, candidate_id)
        pointer = {
            "format": "audiobook-voice-selection", "version": 1,
            "series_id": self.series_id, "voice_id": voice_id,
            "candidate_id": candidate_id, "asset_hash": candidate["asset_hash"],
        }
        pointer["selection_sha256"] = digest(pointer)
        path = self._selection_path(voice_id)
        self._prepare_directory(path.parent)
        if path.is_symlink():
            raise ValueError("Active voice selection must not be a symlink")
        write_json(path, pointer)
        return self.load(voice_id)

    def _publish(self, voice_id, profile, wav_path, reference_text, engine_identity,
                 report, metadata, *, candidate_id):
        if (not isinstance(report, dict) or report.get("passed") is not True or
                not isinstance(reference_text, str) or not reference_text.strip()):
            raise ValueError("Only a verified reference with an exact transcript can be saved")
        self._regular_file(Path(wav_path))
        destination = self._candidate_folder(voice_id, candidate_id)
        self._check_directory(destination)
        if destination.exists():
            raise ValueError("Voice candidate already exists; references are never replaced")
        self._prepare_directory(destination.parent)
        with tempfile.TemporaryDirectory(prefix=".voice-", dir=self.directory) as temp:
            stage = Path(temp)
            shutil.copyfile(wav_path, stage / "reference.wav")
            record = {
                "format": "audiobook-voice", "version": 1, "kind": "reference",
                "series_id": self.series_id, "voice_id": voice_id, "profile": profile,
                "reference_text": reference_text,
                "reference_sha256": file_hash(stage / "reference.wav"),
                "created_with": engine_identity, "creation": metadata, "verification": report,
                "description_applied": metadata.get("description_applied", False),
            }
            if candidate_id != self.LEGACY_CANDIDATE:
                record["candidate_id"] = candidate_id
            write_json(stage / "voice.json", record)
            (stage / "voice.sha256").write_text(file_hash(stage / "voice.json") + "\n")
            # Reserve first, then publish manifest last, without overwriting.
            destination.mkdir()
            try:
                os.link(stage / "reference.wav", destination / "reference.wav")
                os.link(stage / "voice.sha256", destination / "voice.sha256")
                os.link(stage / "voice.json", destination / "voice.json")
            except BaseException:
                # All files in this just-reserved directory belong to this operation.
                shutil.rmtree(destination)
                raise
        return self.load_candidate(voice_id, candidate_id)

    def publish(self, voice_id, profile, wav_path, reference_text, engine_identity, report, metadata):
        voice_id = self.canonical_id(voice_id)
        destination = self.folder(voice_id)
        self._check_directory(destination)
        if destination.exists() or self._load_local(voice_id) is not None:
            raise ValueError("Voice already exists; reference voices are never implicitly replaced")
        self._publish(voice_id, profile, wav_path, reference_text, engine_identity,
                      report, metadata, candidate_id=self.LEGACY_CANDIDATE)
        return self.load(voice_id)

    def publish_candidate(self, voice_id, profile, wav_path, reference_text, engine_identity,
                          report, metadata, *, candidate_id):
        """Save a verified alternative without selecting it or changing old assets."""
        voice_id = self.canonical_id(voice_id)
        self._validate_candidate_id(candidate_id, legacy=False)
        self.load(voice_id)  # Do not add revisions to an invalid active binding.
        return self._publish(voice_id, profile, wav_path, reference_text, engine_identity,
                             report, metadata, candidate_id=candidate_id)

    def lock(self):
        self._prepare_directory(self.directory)
        path = self.directory / ".lock"
        path.mkdir()
        return path
