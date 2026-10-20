"""Validate the portable, source-preserving handoff to an audio renderer.

Offsets are Python/Unicode character offsets into extracted block text, with an
inclusive ``start`` and exclusive ``end``.  Validation deliberately accepts extra
metadata, while checking all fields that affect source fidelity or voice choice.
"""

from __future__ import annotations

from typing import Any
import re


FORMAT = "epub-audiobook-analysis"
VERSION = 1
SEGMENT_KINDS = {"narration", "dialogue", "thought", "letter", "heading", "other"}
ATTRIBUTION_BASES = {"explicit", "context", "unresolved", "user"}


class _Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.blocks: dict[str, dict[str, Any]] = {}

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def object(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.error(path, "must be an object")
            return {}
        return value

    def array(self, value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
        if not isinstance(value, list):
            self.error(path, "must be an array")
            return []
        if nonempty and not value:
            self.error(path, "must not be empty")
        return value

    def string(self, value: Any, path: str, *, nonempty: bool = True) -> bool:
        if not isinstance(value, str):
            self.error(path, "must be a string")
            return False
        if nonempty and not value.strip():
            self.error(path, "must not be empty")
            return False
        return True

    def nullable_string(self, value: Any, path: str) -> bool:
        return value is None or self.string(value, path)

    def choice(self, value: Any, options: set[str], path: str) -> bool:
        if not isinstance(value, str) or value not in options:
            self.error(path, f"must be one of {', '.join(sorted(options))}")
            return False
        return True

    def boolean(self, value: Any, path: str) -> bool:
        if not isinstance(value, bool):
            self.error(path, "must be a boolean")
            return False
        return True

    def strings(self, value: Any, path: str) -> list[str]:
        items = self.array(value, path)
        return [item for i, item in enumerate(items) if self.string(item, f"{path}[{i}]")]

    def entries(
        self, value: Any, path: str, *, nonempty: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        items = self.array(value, path, nonempty=nonempty)
        records = []
        indexed = {}
        for i, item in enumerate(items):
            record = self.object(item, f"{path}[{i}]")
            records.append(record)
            identifier = record.get("id")
            if self.string(identifier, f"{path}[{i}].id"):
                if identifier in indexed:
                    self.error(f"{path}[{i}].id", f"duplicate ID {identifier!r}")
                else:
                    indexed[identifier] = record
        return records, indexed

    def reference(self, value: Any, index: dict[str, Any], path: str) -> bool:
        if not self.string(value, path):
            return False
        if value not in index:
            self.error(path, f"unknown reference {value!r}")
            return False
        return True

    def evidence(self, value: Any, path: str, *, required: bool = False) -> None:
        items = self.array(value, path, nonempty=required)
        for i, item in enumerate(items):
            entry_path = f"{path}[{i}]"
            entry = self.object(item, entry_path)
            valid_block = self.reference(entry.get("block_id"), self.blocks, f"{entry_path}.block_id")
            valid_quote = self.string(entry.get("quote"), f"{entry_path}.quote")
            if valid_block and valid_quote:
                text = self.blocks[entry["block_id"]].get("text")
                if isinstance(text, str) and entry["quote"] not in text:
                    message = "must be an exact substring of the referenced block"
                    candidates = list(re.finditer(re.escape(entry["quote"]), text, re.IGNORECASE))
                    if len(candidates) == 1:
                        message += "; exact source spelling starts with " + repr(candidates[0].group(0)[:240])
                    self.error(f"{entry_path}.quote", message)

    def source(self, value: Any) -> None:
        source = self.object(value, "source")
        book = self.object(source.get("book"), "source.book")
        self.string(book.get("id"), "source.book.id")
        self.string(book.get("title"), "source.book.title")
        self.strings(book.get("authors"), "source.book.authors")
        if "language" not in book:
            self.error("source.book.language", "is required (string or null)")
        self.nullable_string(book.get("language"), "source.book.language")
        chapters, chapter_index = self.entries(source.get("chapters"), "source.chapters", nonempty=True)
        blocks, self.blocks = self.entries(source.get("blocks"), "source.blocks", nonempty=True)
        self.block_records = blocks
        self.array(source.get("exclusions"), "source.exclusions")
        self.strings(source.get("warnings"), "source.warnings")
        self.string(source.get("extraction_policy"), "source.extraction_policy")

        chapter_order: list[str] = []
        membership: dict[str, str] = {}
        for i, chapter in enumerate(chapters):
            path = f"source.chapters[{i}]"
            self.string(chapter.get("title"), f"{path}.title")
            self.string(chapter.get("source_href"), f"{path}.source_href")
            ids = self.strings(chapter.get("block_ids"), f"{path}.block_ids")
            chapter_id = chapter.get("id")
            for j, block_id in enumerate(ids):
                self.reference(block_id, self.blocks, f"{path}.block_ids[{j}]")
                if block_id in membership:
                    self.error(f"{path}.block_ids[{j}]", "block belongs to more than one chapter position")
                membership[block_id] = chapter_id
                chapter_order.append(block_id)

        block_order = []
        for i, block in enumerate(blocks):
            path = f"source.blocks[{i}]"
            block_id = block.get("id")
            if isinstance(block_id, str):
                block_order.append(block_id)
            self.reference(block.get("chapter_id"), chapter_index, f"{path}.chapter_id")
            self.string(block.get("text"), f"{path}.text")
            location = self.object(block.get("source"), f"{path}.source")
            self.string(location.get("href"), f"{path}.source.href")
            self.string(location.get("tag"), f"{path}.source.tag")
            if "element_id" not in location:
                self.error(f"{path}.source.element_id", "is required (string or null)")
            self.nullable_string(location.get("element_id"), f"{path}.source.element_id")
            if isinstance(block_id, str):
                if block_id not in membership:
                    self.error(f"{path}.id", "block is missing from chapter block_ids")
                elif membership[block_id] != block.get("chapter_id"):
                    self.error(f"{path}.chapter_id", "does not match the chapter containing this block")
        if chapter_order != block_order:
            self.error("source.chapters", "chapter block_ids must reproduce source.blocks in exactly the same order")

    def cast(self, value: Any) -> dict[str, dict[str, Any]]:
        cast = self.object(value, "cast")
        characters, character_index = self.entries(cast.get("characters"), "cast.characters")
        roles, role_index = self.entries(cast.get("roles"), "cast.roles")
        voices, voice_index = self.entries(cast.get("voices"), "cast.voices")
        character_voices: dict[str, str] = {}
        for i, character in enumerate(characters):
            path = f"cast.characters[{i}]"
            self.string(character.get("name"), f"{path}.name")
            self.strings(character.get("aliases"), f"{path}.aliases")
            self.string(character.get("description"), f"{path}.description")
            self.evidence(character.get("evidence"), f"{path}.evidence")
            voice_id = character.get("voice_id")
            if self.reference(voice_id, voice_index, f"{path}.voice_id"):
                if voice_id in character_voices:
                    self.error(f"{path}.voice_id", "distinct characters must have distinct voice IDs")
                character_voices[voice_id] = character.get("id")

        for i, role in enumerate(roles):
            path = f"cast.roles[{i}]"
            self.choice(role.get("kind"), {"narration", "dialogue"}, f"{path}.kind")
            self.string(role.get("description"), f"{path}.description")
            self.reference(role.get("voice_id"), voice_index, f"{path}.voice_id")
            if "character_id" not in role:
                self.error(f"{path}.character_id", "is required (string or null)")
            character_id = role.get("character_id")
            if character_id is not None and self.reference(character_id, character_index, f"{path}.character_id"):
                if role.get("kind") == "dialogue" and role.get("voice_id") != character_index[character_id].get("voice_id"):
                    self.error(f"{path}.voice_id", "dialogue role voice must match its character voice")

        for i, voice in enumerate(voices):
            path = f"cast.voices[{i}]"
            self.string(voice.get("description"), f"{path}.description")
            traits = self.array(voice.get("traits"), f"{path}.traits")
            for j, value in enumerate(traits):
                trait_path = f"{path}.traits[{j}]"
                trait = self.object(value, trait_path)
                self.string(trait.get("name"), f"{trait_path}.name")
                self.string(trait.get("value"), f"{trait_path}.value")
                self.choice(trait.get("basis"), {"text", "casting"}, f"{trait_path}.basis")
                self.evidence(trait.get("evidence"), f"{trait_path}.evidence", required=trait.get("basis") == "text")
        self.voice_index = voice_index
        return role_index

    def segments(self, value: Any, roles: dict[str, dict[str, Any]]) -> None:
        segments, _ = self.entries(value, "segments", nonempty=True)
        block_positions = {block_id: i for i, block_id in enumerate(self.blocks)}
        coverage = {block_id: 0 for block_id in self.blocks}
        last_position = -1
        for i, segment in enumerate(segments):
            path = f"segments[{i}]"
            block_id = segment.get("block_id")
            valid_block = self.reference(block_id, self.blocks, f"{path}.block_id")
            valid_text = self.string(segment.get("text"), f"{path}.text", nonempty=False)
            self.choice(segment.get("kind"), SEGMENT_KINDS, f"{path}.kind")
            start, end = segment.get("start"), segment.get("end")
            valid_offsets = True
            for label, offset in (("start", start), ("end", end)):
                if type(offset) is not int or offset < 0:
                    self.error(f"{path}.{label}", "must be a nonnegative integer (not boolean)")
                    valid_offsets = False
            if valid_offsets and end <= start:
                self.error(path, "end must be greater than start")
                valid_offsets = False
            if valid_block:
                position = block_positions[block_id]
                if position < last_position:
                    self.error(f"{path}.block_id", "segments must follow source block order")
                last_position = max(last_position, position)
                source_text = self.blocks[block_id].get("text")
                if valid_offsets:
                    if start != coverage[block_id]:
                        self.error(f"{path}.start", f"must be {coverage[block_id]} for contiguous block coverage")
                    coverage[block_id] = end
                    if isinstance(source_text, str):
                        if end > len(source_text):
                            self.error(f"{path}.end", "exceeds source block length")
                        if valid_text and segment["text"] != source_text[start:end]:
                            self.error(f"{path}.text", "must exactly match source block text at [start:end]")

            delivery = self.object(segment.get("delivery"), f"{path}.delivery")
            self.string(delivery.get("direction"), f"{path}.delivery.direction", nonempty=False)
            self.evidence(delivery.get("evidence"), f"{path}.delivery.evidence")
            bracket_spans = self.bracket_spans.get(block_id, []) if isinstance(block_id, str) else []
            if self.require_bracket_delivery and valid_offsets:
                overlaps_brackets = any(span["start"] < end and span["end"] > start for span in bracket_spans)
                if overlaps_brackets and "performance" not in segment:
                    self.error(f"{path}.performance", "required for source bracket text under bracket delivery policy")
            if "performance" in segment or "spoken_text" in segment:
                if not self.require_bracket_delivery:
                    self.error(f"{path}.performance", "requires provenance.bracket_delivery")
                performance = self.object(segment.get("performance"), f"{path}.performance")
                self.choice(performance.get("kind"), {"bracketed"}, f"{path}.performance.kind")
                self.choice(performance.get("delimiters"), {"omit"}, f"{path}.performance.delimiters")
                self.string(performance.get("direction"), f"{path}.performance.direction")
                if performance.get("direction") != self.bracket_direction:
                    self.error(f"{path}.performance.direction", "must match provenance bracket delivery direction")
                self.string(performance.get("group_id"), f"{path}.performance.group_id")
                for flag in ("enter", "exit"):
                    self.boolean(performance.get(flag), f"{path}.performance.{flag}")
                self.string(segment.get("spoken_text"), f"{path}.spoken_text", nonempty=False)
                if valid_text and segment.get("spoken_text") != segment["text"].translate(str.maketrans("", "", "[]")):
                    self.error(f"{path}.spoken_text", "may only omit square bracket delimiters from exact text")
                enclosing = next((span for span in bracket_spans
                                  if valid_offsets and span["start"] <= start < end <= span["end"]), None)
                if enclosing is None:
                    self.error(f"{path}.performance", "must be within a balanced source bracket group")
                else:
                    expected = {"group_id": enclosing["group_id"],
                                "enter": enclosing["first"] and start == enclosing["start"],
                                "exit": enclosing["last"] and end == enclosing["end"]}
                    for key, value in expected.items():
                        if performance.get(key) != value:
                            self.error(f"{path}.performance.{key}", "must match source bracket boundaries")
            attribution = self.object(segment.get("attribution"), f"{path}.attribution")
            self.choice(attribution.get("basis"), ATTRIBUTION_BASES, f"{path}.attribution.basis")
            self.string(attribution.get("reason"), f"{path}.attribution.reason")
            self.evidence(attribution.get("evidence"), f"{path}.attribution.evidence")
            self.boolean(attribution.get("review_required"), f"{path}.attribution.review_required")
            candidates = self.strings(attribution.get("candidates"), f"{path}.attribution.candidates")
            for j, candidate in enumerate(candidates):
                self.reference(candidate, roles, f"{path}.attribution.candidates[{j}]")
            if attribution.get("basis") == "unresolved" and attribution.get("review_required") is not True:
                self.error(f"{path}.attribution.review_required", "must be true for unresolved attribution")
            if "role_id" not in segment:
                self.error(f"{path}.role_id", "is required (string or null)")
            if "voice_id" not in segment:
                self.error(f"{path}.voice_id", "is required (string or null)")
            role_id, voice_id = segment.get("role_id"), segment.get("voice_id")
            if role_id is None:
                if voice_id is not None:
                    self.error(f"{path}.voice_id", "must be null when role_id is null")
                if attribution.get("basis") != "unresolved":
                    self.error(f"{path}.attribution.basis", "must be unresolved when role_id is null")
                if attribution.get("review_required") is not True:
                    self.error(f"{path}.attribution.review_required", "must be true when role_id is null")
            else:
                valid_role = self.reference(role_id, roles, f"{path}.role_id")
                self.reference(voice_id, self.voice_index, f"{path}.voice_id")
                if valid_role and voice_id != roles[role_id].get("voice_id"):
                    self.error(f"{path}.voice_id", "must match the selected role voice")

        for block_id, block in self.blocks.items():
            text = block.get("text")
            if isinstance(text, str) and coverage[block_id] != len(text):
                self.error("segments", f"block {block_id!r} is not fully covered ({coverage[block_id]}/{len(text)} characters)")

    def validate(self, document: Any) -> list[str]:
        doc = self.object(document, "document")
        if doc.get("format") != FORMAT:
            self.error("format", f"must be {FORMAT!r}")
        if type(doc.get("version")) is not int or doc.get("version") != VERSION:
            self.error("version", f"must be integer {VERSION}")
        self.source(doc.get("source"))
        provenance = doc.get("provenance")
        self.require_bracket_delivery = isinstance(provenance, dict) and "bracket_delivery" in provenance
        self.bracket_spans = {}
        self.bracket_direction = None
        bracket_warnings = None
        if not self.errors:
            from .performance import bracket_ranges
            self.bracket_spans, bracket_warnings = bracket_ranges(doc["source"])
        if self.require_bracket_delivery:
            marker = self.object(provenance["bracket_delivery"], "provenance.bracket_delivery")
            if type(marker.get("version")) is not int or marker.get("version") != 1:
                self.error("provenance.bracket_delivery.version", "must be integer 1")
            self.choice(marker.get("basis"), {"source_delimiters"}, "provenance.bracket_delivery.basis")
            self.string(marker.get("direction"), "provenance.bracket_delivery.direction")
            self.bracket_direction = marker.get("direction")
            group_count = len({span["group_id"] for spans in self.bracket_spans.values() for span in spans})
            if type(marker.get("groups")) is not int or marker.get("groups") != group_count:
                self.error("provenance.bracket_delivery.groups", "must match balanced source bracket group count")
            self.array(marker.get("warnings"), "provenance.bracket_delivery.warnings")
            if bracket_warnings is not None and marker.get("warnings") != bracket_warnings:
                self.error("provenance.bracket_delivery.warnings", "must match unmatched source bracket warnings")
        roles = self.cast(doc.get("cast"))
        self.segments(doc.get("segments"), roles)
        issues, _ = self.entries(doc.get("issues"), "issues")
        for i, issue in enumerate(issues):
            path = f"issues[{i}]"
            for field in ("kind", "target", "reason"):
                self.string(issue.get(field), f"{path}.{field}")
            self.boolean(issue.get("resolved"), f"{path}.resolved")
        self.object(doc.get("provenance"), "provenance")
        return self.errors


def validate_document(document: Any) -> list[str]:
    """Return structural/fidelity errors without trusting supplied readiness flags.

    A valid document may still need human review; use :func:`readiness` before
    generating audio.  Inputs with malformed JSON types return errors, not crashes.
    """
    return _Validator().validate(document)


def review_items(document: Any) -> list[dict[str, Any]]:
    """Return unresolved issues and segments that still require a voice decision."""
    if not isinstance(document, dict):
        return []
    result: list[dict[str, Any]] = []
    issues = document.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict) and issue.get("resolved") is not True:
                result.append({"type": "issue", **{key: issue.get(key) for key in ("id", "kind", "target", "reason")}})
    segments = document.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            attribution = segment.get("attribution")
            if not isinstance(attribution, dict):
                attribution = {}
            if (
                attribution.get("review_required") is not False
                or attribution.get("basis") == "unresolved"
                or segment.get("role_id") is None
                or segment.get("voice_id") is None
            ):
                result.append({
                    "type": "segment",
                    "id": segment.get("id"),
                    "block_id": segment.get("block_id"),
                    "text": segment.get("text"),
                    "reason": attribution.get("reason") or "Speaker attribution requires review.",
                    "candidates": attribution.get("candidates") if isinstance(attribution.get("candidates"), list) else [],
                })
    return result


def readiness(document: Any) -> dict[str, Any]:
    """Compute readiness from validated content and outstanding review decisions."""
    errors = validate_document(document)
    review_count = len(review_items(document))
    return {"ready": not errors and review_count == 0, "errors": errors, "review_count": review_count}
