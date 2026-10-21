"""Bounded, dependency-free EPUB text extraction in package spine order.

This is a text extractor, not an EPUB renderer or a DRM remover. Each emitted
block refers to the XHTML resource and nearest containing block element.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile
import zlib


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000
MAX_XML_DEPTH = 256

_XHTML = "application/xhtml+xml"
_DC = "http://purl.org/dc/elements/1.1/"
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "body", "caption", "center",
    "dd", "details", "dialog", "div", "dl", "dt", "fieldset", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hgroup", "hr", "legend", "li", "main", "nav", "ol", "p", "pre", "section",
    "summary", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
})
_SKIP_TAGS = frozenset({"script", "style", "template", "svg", "math", "canvas"})
_NON_TEXT_TAGS = frozenset({"img", "image", "audio", "video", "object", "embed", "iframe"})
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_FONT_CUES = {"font-style": ("emphasis", "italic"), "font-weight": ("strong", "bold")}


def _css_syntax(text: str) -> tuple[str, bool]:
    """Hide strings/comments from structural matching without interpreting CSS.

    Keep string tokens visibly opaque: a quoted "italic" is not an italic
    keyword, and removing an extra string must not make an invalid value valid.
    Escapes outside strings remain visible for the conservative caller to reject.
    """
    masked = list(text)
    index = 0
    while index < len(text):
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                return "".join(masked), False
            end += 2
            masked[index:end] = " " * (end - index)
            index = end
        elif text[index] in {"'", '"'}:
            start, quote = index, text[index]
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                    # A backslash CRLF is one escaped newline.
                    if index <= len(text) and text[index - 1:index + 1] == "\r\n":
                        index += 1
                elif text[index] == quote:
                    index += 1
                    masked[start:index] = "?" * (index - start)
                    break
                elif text[index] in "\r\n\f":
                    return "".join(masked), False
                else:
                    index += 1
            else:
                return "".join(masked), False
        else:
            index += 1
    return "".join(masked), True


def _font_declarations(text: str, warn) -> dict[str, bool | None]:
    """Read explicit source cues, conservatively declining CSS expressions."""
    text, valid = _css_syntax(text)
    if not valid or "\\" in text:
        warn("Complex CSS declarations were not interpreted as emphasis.")
        return {"emphasis": None, "strong": None}
    if re.search(r"[{}()]", text):
        if re.search(r"\b(?:font(?:-style|-weight)?|all)\s*:", text, re.IGNORECASE):
            warn("Complex CSS declarations were not interpreted as emphasis.")
            return {"emphasis": None, "strong": None}
        return {}
    values: dict[str, set[bool | None]] = {}
    for declaration in text.split(";"):
        prop, separator, value = declaration.partition(":")
        if not separator:
            continue
        prop = prop.strip().lower()
        value = re.sub(r"\s*!\s*important\s*$", "", value.strip().lower())
        if prop in {"font", "all"}:
            values.setdefault("emphasis", set()).add(None)
            values.setdefault("strong", set()).add(None)
        elif prop in _FONT_CUES:
            kind, positive = _FONT_CUES[prop]
            values.setdefault(kind, set()).add(True if value == positive else False if value == "normal" else None)
    result = {}
    for kind, choices in values.items():
        result[kind] = next(iter(choices)) if len(choices) == 1 else None
        if result[kind] is None:
            warn("Conflicting or unsupported CSS emphasis declarations were left unresolved.")
    return result


def _class_emphasis(root: ET.Element, href: str, read, warn) -> dict[str, dict[str, bool | None]]:
    """Recognize flat .class rules only; do not render CSS or resolve a cascade."""
    texts = []
    blocked: set[str] = set()
    for node in root.iter():
        if _tag(node) == "style":
            if node.get("media", "all").strip().lower() in {"", "all"}:
                texts.append("".join(node.itertext()))
            else:
                warn(f"{href}: conditional stylesheet emphasis unavailable; class cues left unresolved.")
                blocked.update({"emphasis", "strong"})
        elif _tag(node) == "link" and "stylesheet" in node.get("rel", "").lower().split():
            try:
                name = _resolve_href(href, node.get("href", ""), remote_ok=True)
                if name is None or node.get("media", "all").strip().lower() not in {"", "all"}:
                    raise ValueError("remote or conditional stylesheet")
                texts.append(read(name).decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError) as exc:
                warn(f"{href}: stylesheet emphasis unavailable ({exc}); class cues left unresolved.")
                blocked.update({"emphasis", "strong"})
    choices: dict[str, dict[str, set[bool | None]]] = {}
    for text in texts:
        text, valid = _css_syntax(text)
        # Quoted editorial metadata/font-family names are not stylesheet syntax.
        # Real conditional rules, imports, nesting, and unquoted escapes still
        # require a CSS parser/cascade; never match their inner rules as flat CSS.
        rules = list(re.finditer(r"([^{}]+)\{([^{}]*)\}", text))
        residue = re.sub(r"([^{}]+)\{([^{}]*)\}", "", text).strip()
        if not valid or residue or re.search(r"[@\\]", text):
            relevant = {kind for prop, (kind, _) in _FONT_CUES.items()
                        if re.search(r"\b" + prop + r"\s*:", text, re.IGNORECASE)}
            if (not valid or "\\" in text
                    or re.search(r"\b(?:font|all)\s*:|@import\b", text, re.IGNORECASE)):
                relevant.update({"emphasis", "strong"})
            if relevant:
                warn(f"{href}: complex stylesheet rules were not resolved; class emphasis may be unavailable.")
                blocked.update(relevant)
            continue
        for rule in rules:
            declarations = _font_declarations(rule[2], lambda message: warn(f"{href}: {message}"))
            if not declarations:
                continue
            selector = rule[1].strip()
            if not re.fullmatch(r"\.[A-Za-z_][A-Za-z0-9_-]*", selector):
                blocked.update(declarations)
                warn(f"{href}: unsupported CSS selector for emphasis; class cues left unresolved.")
                continue
            entry = choices.setdefault(selector[1:], {})
            for kind, value in declarations.items():
                entry.setdefault(kind, set()).add(value)
    result = {}
    for class_name, entry in choices.items():
        result[class_name] = {}
        for kind, values in entry.items():
            value = next(iter(values)) if len(values) == 1 and kind not in blocked else None
            result[class_name][kind] = value
            if value is None:
                warn(f"{href}: conflicting or unsupported CSS emphasis for .{class_name} was left unresolved.")
    return result


def _normalized_spans(chunks: list[tuple[str, frozenset]]) -> tuple[str, list[dict]]:
    """Map raw source cue ranges into the existing whitespace-normalized text."""
    raw = "".join(text for text, _ in chunks)
    words = list(re.finditer(r"\S+", raw))
    text = " ".join(word[0] for word in words)
    if not words:
        return text, []
    starts, ends, offsets = [], [], []
    offset = 0
    for word in words:
        starts.append(word.start())
        ends.append(word.end())
        offsets.append(offset)
        offset += len(word[0]) + 1
    ranges: dict[tuple[str, str], list[list[int]]] = {}
    position = 0
    for chunk, cues in chunks:
        for cue in cues:
            previous = ranges.setdefault(cue, [])
            if previous and previous[-1][1] == position:
                previous[-1][1] += len(chunk)
            else:
                previous.append([position, position + len(chunk)])
        position += len(chunk)
    spans = []
    for (kind, origin), intervals in ranges.items():
        for start, end in intervals:
            first, last = bisect_right(ends, start), bisect_left(starts, end) - 1
            if first <= last:
                spans.append({"start": offsets[first] + max(0, start - starts[first]),
                              "end": offsets[last] + min(end, ends[last]) - starts[last],
                              "kind": kind, "origin": origin})
    return text, sorted(spans, key=lambda span: (span["start"], span["end"], span["kind"], span["origin"]))


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _member_name(name: str) -> str:
    """Validate an archive member name without decoding literal ZIP filenames."""
    parts = name.rstrip("/").split("/")
    if (not name or name.startswith("/") or "\\" in name or ":" in parts[0]
            or any(ord(char) < 32 for char in name)
            or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return name


def _resolve_href(base: str, href: str, *, remote_ok: bool = False) -> str | None:
    """Resolve a resource URL relative to an archive file, without leaving ZIP."""
    if not href or any(ord(char) < 32 for char in href) or "\\" in href:
        raise ValueError(f"Unsafe or empty EPUB resource URL: {href!r}")
    try:
        parsed = urlsplit(href)
    except ValueError as exc:
        raise ValueError(f"Invalid EPUB resource URL: {href!r}") from exc
    if parsed.scheme or parsed.netloc:
        if remote_ok and parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            return None
        raise ValueError(f"External EPUB resource is not supported: {href!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"EPUB resource URL must not contain query or fragment: {href!r}")
    if re.search(r"%(?![0-9a-fA-F]{2})", parsed.path):
        raise ValueError(f"Invalid URL escape in EPUB resource: {href!r}")
    try:
        decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 URL escape in EPUB resource: {href!r}") from exc
    if not decoded or decoded.startswith("/") or "\\" in decoded or any(ord(c) < 32 for c in decoded):
        raise ValueError(f"Unsafe EPUB resource path: {href!r}")
    parts = base.split("/")[:-1] if base else []
    for part in decoded.split("/"):
        if part == "..":
            if not parts:
                raise ValueError(f"EPUB resource escapes the archive root: {href!r}")
            parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    result = "/".join(parts)
    _member_name(result)
    return result


def _parse_xml(data: bytes, name: str) -> ET.Element:
    # NUL removal also catches declarations encoded as UTF-16 or UTF-32.
    # Ordinary external XHTML DOCTYPE declarations are safe: ElementTree never
    # loads their external DTD. Internal subsets and entity declarations are not.
    probe = data.replace(b"\x00", b"")
    if re.search(br"<!ENTITY\b|<!DOCTYPE\b[^>]*\[", probe, re.IGNORECASE):
        raise ValueError(f"XML entity declarations/internal DTDs are not supported: {name}")
    try:
        root = ET.fromstring(data)
    except (ET.ParseError, LookupError, ValueError) as exc:
        raise ValueError(f"Malformed XML in {name}: {exc}") from exc
    stack = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        if depth > MAX_XML_DEPTH:
            raise ValueError(f"XML nesting exceeds safety limit in {name}")
        stack.extend((child, depth + 1) for child in element)
    if any("{http://www.w3.org/XML/1998/namespace}base" in node.attrib for node in root.iter()):
        raise ValueError(f"xml:base is unsupported; resource resolution would be ambiguous: {name}")
    return root


class _Archive:
    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        entries = archive.infolist()
        if len(entries) > MAX_MEMBERS:
            raise ValueError("EPUB archive has too many members")
        self.entries: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in entries:
            if info.orig_filename != info.filename:
                raise ValueError(f"Unsafe ZIP member path: {info.orig_filename!r}")
            _member_name(info.filename)
            if info.filename in self.entries:
                raise ValueError(f"Duplicate ZIP member: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symbolic link ZIP member is unsupported: {info.filename}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"ZIP member exceeds safety size limit: {info.filename}")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("EPUB expanded size exceeds safety limit")
            if (info.file_size > 1024 * 1024
                    and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO):
                raise ValueError(f"Suspicious ZIP compression ratio: {info.filename}")
            self.entries[info.filename] = info

    def read(self, name: str) -> bytes:
        info = self.entries.get(name)
        if info is None or info.is_dir():
            raise ValueError(f"Missing EPUB resource: {name}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP content is unsupported: {name}")
        try:
            with self.archive.open(info) as stream:
                result = stream.read(MAX_MEMBER_BYTES + 1)
        except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error) as exc:
            raise ValueError(f"Unable to read EPUB resource {name}: {exc}") from exc
        if len(result) > MAX_MEMBER_BYTES or len(result) != info.file_size:
            raise ValueError(f"Invalid or oversized ZIP member: {name}")
        return result


def _body_blocks(root: ET.Element, href: str, warn, class_styles: dict) -> list[dict]:
    bodies = [node for node in root if _tag(node) == "body"]
    if _tag(root) != "html" or len(bodies) != 1:
        raise ValueError(f"XHTML resource must have one html/body element: {href}")
    body = bodies[0]
    rows: list[dict] = []
    chunks: list[tuple[str, frozenset]] = []
    pending_blank = False

    def flush(owner: ET.Element) -> None:
        nonlocal pending_blank
        text, spans = _normalized_spans(chunks)
        chunks.clear()
        if text:
            row = {"text": text, "source": {
                "href": href,
                "element_id": owner.get("id") or owner.get("{http://www.w3.org/XML/1998/namespace}id"),
                "tag": _tag(owner),
            }}
            if spans:
                row["inline_spans"] = spans
            if pending_blank:
                row["break_before"] = "blank_paragraph"
                pending_blank = False
            rows.append(row)

    def node_cues(node: ET.Element, inherited: frozenset) -> frozenset:
        cues = set(inherited)
        tag = _tag(node)
        if tag in {"em", "i"}:
            cues.add(("emphasis", "tag"))
        elif tag in {"strong", "b"}:
            cues.add(("strong", "tag"))
        declarations: dict[str, set[bool | None]] = {}
        for class_name in node.get("class", "").split():
            for kind, value in class_styles.get(class_name, {}).items():
                declarations.setdefault(kind, set()).add(value)
        for kind, value in _font_declarations(node.get("style", ""), lambda message: warn(f"{href}: {message}")).items():
            declarations.setdefault(kind, set()).add(value)
        for kind, values in declarations.items():
            cues.discard((kind, "style"))
            if values == {True}:
                cues.add((kind, "style"))
            elif len(values) > 1 or None in values:
                warn(f"{href}: conflicting or unsupported CSS emphasis on an element was left unresolved.")
        return frozenset(cues)

    def walk(node: ET.Element, owner: ET.Element, inherited: frozenset) -> None:
        nonlocal pending_blank
        cues = node_cues(node, inherited)
        if node.get("hidden") is not None or node.get("aria-hidden") == "true":
            warn(f"{href}: hidden/aria-hidden text is retained; visibility is not rendered.")
        if node.text:
            chunks.append((node.text, cues))
        for child in node:
            tag = _tag(child)
            if tag in _SKIP_TAGS:
                warn(f"{href}: skipped <{tag}> content; no rendering or speech conversion was performed.")
            elif tag in _NON_TEXT_TAGS:
                warn(f"{href}: <{tag}> media and alternative text are not narrated; review the source.")
                # A media element can contain a textual HTML fallback. Retain
                # that fallback in DOM order, but do not invent image narration.
                if tag not in {"img", "image", "embed"}:
                    walk(child, owner, cues)
            elif tag == "br":
                chunks.append(("\n", cues))
            elif tag in _BLOCK_TAGS:
                flush(owner)
                walk(child, child, cues)
                flush(child)
            else:
                walk(child, owner, cues)
            if child.tail:
                chunks.append((child.tail, cues))
        if (_tag(node) == "p" and not _normalize("".join(node.itertext()))
                and not any(_tag(child) in _SKIP_TAGS | _NON_TEXT_TAGS for child in node.iter())):
            pending_blank = True

    walk(body, body, frozenset())
    flush(body)
    return rows


def extract_epub(path: str | Path, *, include_auxiliary: bool = False) -> dict:
    """Extract XHTML text in spine order, raising ValueError for unsafe/lost input.

    By default navigation documents and spine entries marked ``linear="no"``
    are excluded and recorded. ``include_auxiliary`` includes those *in the
    spine*. Non-spine documents are always excluded. Repeated spine resources
    remain repeated, with distinct chapter/block IDs and a warning.
    """
    path = Path(path)
    try:
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("EPUB archive exceeds safety size limit")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Unable to open EPUB: {exc}") from exc

    warnings: list[str] = []
    exclusions: list[dict] = []

    def warn(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def exclude(href: str, reason: str) -> None:
        entry = {"href": href, "reason": reason}
        if entry not in exclusions:
            exclusions.append(entry)

    with archive:
        resources = _Archive(archive)
        if "mimetype" in resources.entries:
            if resources.read("mimetype").strip() != b"application/epub+zip":
                raise ValueError("Archive mimetype is not application/epub+zip")
        else:
            warn("EPUB has no mimetype member; continuing with its container/package documents.")
        encrypted: set[str] = set()
        if "META-INF/encryption.xml" in resources.entries:
            encryption = _parse_xml(resources.read("META-INF/encryption.xml"), "META-INF/encryption.xml")
            for node in encryption.iter():
                if _tag(node) == "cipherreference":
                    encrypted.add(_resolve_href("", node.get("URI", "")))

        def xml(name: str) -> ET.Element:
            if name in encrypted:
                raise ValueError(f"Encrypted EPUB content is unsupported: {name}")
            return _parse_xml(resources.read(name), name)

        container = xml("META-INF/container.xml")
        rootfiles = [node for node in container.iter() if _tag(node) == "rootfile"
                     and node.get("media-type") == "application/oebps-package+xml"]
        if not rootfiles:
            raise ValueError("EPUB container has no OPF package rootfile")
        if len(rootfiles) > 1:
            warn("EPUB has multiple package renditions; only the first OPF rootfile is processed.")
        package_path = _resolve_href("", rootfiles[0].get("full-path", ""))
        package = xml(package_path)
        if _tag(package) != "package":
            raise ValueError("EPUB rootfile is not an OPF package")

        def section(tag: str) -> ET.Element:
            found = [node for node in package if _tag(node) == tag]
            if len(found) != 1:
                raise ValueError(f"OPF package must contain exactly one {tag} section")
            return found[0]

        metadata, manifest, spine = section("metadata"), section("manifest"), section("spine")

        def values(tag: str) -> list[str]:
            return [text for node in metadata if node.tag == f"{{{_DC}}}{tag}"
                    if (text := _normalize("".join(node.itertext())))]

        titles, authors, languages = values("title"), values("creator"), values("language")
        if not titles:
            warn("OPF title is missing; using the EPUB filename.")
        if not languages:
            warn("OPF language is missing; recorded as 'und'.")
        book = {"id": digest, "title": titles[0] if titles else path.stem,
                "authors": authors, "language": languages[0] if languages else "und"}
        items: dict[str, dict] = {}
        for node in manifest:
            if _tag(node) != "item":
                continue
            item_id = node.get("id")
            if not item_id or item_id in items:
                raise ValueError("OPF manifest contains a missing or duplicate item id")
            href = node.get("href", "")
            items[item_id] = {
                "href": href, "path": _resolve_href(package_path, href, remote_ok=True),
                "type": node.get("media-type", ""), "properties": node.get("properties", "").split(),
                "fallback": node.get("fallback"),
            }

        def supported(item_id: str) -> dict:
            visited: set[str] = set()
            while True:
                if item_id in visited:
                    raise ValueError("OPF manifest fallback chain contains a cycle")
                visited.add(item_id)
                if item_id not in items:
                    raise ValueError(f"OPF references an unknown manifest item: {item_id}")
                item = items[item_id]
                if item["type"] == _XHTML and item["path"] is not None:
                    return item
                if not item["fallback"]:
                    raise ValueError(f"Unsupported spine resource without a local XHTML fallback: {item['href']} ({item['type']})")
                warn(f"Using XHTML fallback for unsupported spine resource: {item['href']}")
                item_id = item["fallback"]

        chapters: list[dict] = []
        blocks: list[dict] = []
        used_paths: set[str] = set()
        referenced_ids: set[str] = set()
        spine_entries = [node for node in spine if _tag(node) == "itemref"]
        if not spine_entries:
            raise ValueError("EPUB package has an empty spine")
        for entry in spine_entries:
            item_id = entry.get("idref", "")
            if item_id not in items:
                raise ValueError(f"Spine references an unknown manifest item: {item_id}")
            referenced_ids.add(item_id)
            original = items[item_id]
            href = original["path"] or original["href"]
            if entry.get("linear", "yes") not in {"yes", "no"}:
                raise ValueError(f"Invalid spine linear value for {href}")
            if not include_auxiliary:
                if "nav" in original["properties"]:
                    exclude(href, "navigation document")
                    continue
                if entry.get("linear") == "no":
                    exclude(href, "auxiliary spine entry (linear=no)")
                    continue
            item = supported(item_id)
            href = item["path"]
            if not include_auxiliary and "nav" in item["properties"]:
                exclude(href, "navigation document used as fallback")
                continue
            if href in used_paths:
                warn(f"Repeated spine resource is preserved as another chapter: {href}")
            used_paths.add(href)
            root = xml(href)
            def read_stylesheet(name: str) -> bytes:
                if name in encrypted:
                    raise ValueError(f"Encrypted EPUB stylesheet is unsupported: {name}")
                return resources.read(name)

            class_styles = _class_emphasis(root, href, read_stylesheet, warn)
            rows = _body_blocks(root, href, warn, class_styles)
            chapter_id = f"ch{len(chapters) + 1:04d}"
            heading = next((row["text"] for row in rows if row["source"]["tag"] in _HEADINGS), None)
            document_title = next((_normalize("".join(node.itertext())) for node in root.iter()
                                   if _tag(node) == "title"), "")
            chapter = {"id": chapter_id, "title": heading or document_title or PurePosixPath(href).stem,
                       "source_href": href, "block_ids": []}
            for row in rows:
                block_id = f"b{len(blocks) + 1:06d}"
                blocks.append({"id": block_id, "chapter_id": chapter_id, **row})
                chapter["block_ids"].append(block_id)
            chapters.append(chapter)
            if not rows:
                warn(f"{href}: spine document contains no extractable body text.")

        for item_id, item in items.items():
            if item_id not in referenced_ids and item["path"] not in used_paths and item["type"] == _XHTML:
                exclude(item["path"] or item["href"], "navigation document outside spine" if "nav" in item["properties"] else "document outside reading-order spine")
        if not blocks:
            raise ValueError("EPUB has no extractable text in the selected spine documents")
        warn("Text follows XHTML DOM order with normalized whitespace; CSS, scripts, and visual layout are not rendered.")
        return {"book": book, "chapters": chapters, "blocks": blocks,
                "exclusions": exclusions, "warnings": warnings,
                "extraction_policy": "spine-xhtml-dom-v1; whitespace=collapse; footnotes=retain; "
                + ("auxiliary-spine=include" if include_auxiliary else "auxiliary-spine=exclude")
                + "; inline-cues=source-v1; blank-paragraphs=mark"}
