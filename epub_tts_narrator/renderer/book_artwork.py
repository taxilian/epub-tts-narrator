"""Validated EPUB publication metadata and square, uncropped audiobook artwork."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

from ..epub import MAX_ARCHIVE_BYTES, _Archive, _parse_xml, _resolve_href, _tag
from .interrupts import run_subprocess


_DC = "http://purl.org/dc/elements/1.1/"
_OPF = "http://www.idpf.org/2007/opf"
_SIZE = 1600
_MAX_PIXELS = 40_000_000
_MAX_DIMENSION = 16_384
_STYLE = "blurred-letterbox-v1"


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "template"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "template"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text(node, *, html=False) -> str:
    text = " ".join(node.itertext())
    if html:
        parser = _PlainText()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    # Embedded metadata is data, never markup or terminal control sequences.
    return " ".join(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).split())


def read_publication_metadata(epub: str | Path, expected_book_id: str) -> dict:
    """Read book-bound EPUB metadata independently of its cover availability."""
    epub = Path(epub)
    try:
        if epub.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("EPUB archive exceeds safety size limit")
        if _hash(epub) != expected_book_id:
            raise ValueError("Publication EPUB does not match the analyzed book ID")
        with zipfile.ZipFile(epub) as archive:
            resources = _Archive(archive)
            if ("mimetype" in resources.entries
                    and resources.read("mimetype").strip() != b"application/epub+zip"):
                raise ValueError("Archive mimetype is not application/epub+zip")
            encrypted = set()
            if "META-INF/encryption.xml" in resources.entries:
                encryption = _parse_xml(resources.read("META-INF/encryption.xml"), "encryption.xml")
                encrypted = {_resolve_href("", node.get("URI", "")) for node in encryption.iter()
                             if _tag(node) == "cipherreference"}

            def xml(name):
                if name in encrypted:
                    raise ValueError("Encrypted EPUB publication assets are unsupported")
                return _parse_xml(resources.read(name), name)

            container = xml("META-INF/container.xml")
            rootfiles = [node for node in container.iter() if _tag(node) == "rootfile"
                         and node.get("media-type") == "application/oebps-package+xml"]
            if not rootfiles:
                raise ValueError("EPUB container has no OPF package")
            package = xml(_resolve_href("", rootfiles[0].get("full-path", "")))
            if _tag(package) != "package":
                raise ValueError("EPUB rootfile is not an OPF package")
            sections = [node for node in package if _tag(node) == "metadata"]
            if len(sections) != 1:
                raise ValueError("EPUB must have exactly one metadata section")
            metadata = sections[0]
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Unable to read the EPUB publication assets; check the EPUB path and archive") from exc

    def nodes(name):
        return [node for node in metadata if node.tag == f"{{{_DC}}}{name}"]

    def values(name, *, html=False):
        return list(dict.fromkeys(value for node in nodes(name) if (value := _text(node, html=html))))

    def refinement(node, name):
        identifier = node.get("id")
        if not identifier:
            return ""
        return next((_text(meta) for meta in metadata
                     if _tag(meta) == "meta" and meta.get("refines") == "#" + identifier
                     and meta.get("property") == name), "")

    def role(node):
        return (refinement(node, "role") or node.get(f"{{{_OPF}}}role", "")).lower()

    titles = nodes("title")
    main_title = next((node for node in titles if refinement(node, "title-type") == "main"),
                      titles[0] if titles else None)
    authors = list(dict.fromkeys(_text(node) for node in nodes("creator")
                                if role(node) in {"", "aut", "author"} and _text(node)))
    book = {"id": expected_book_id, "title": (_text(main_title) if main_title is not None else "") or epub.stem,
            "authors": authors, "language": next(iter(values("language")), "und")}
    for name in ("publisher", "description", "rights"):
        found = values(name, html=name == "description")
        if found:
            book[name] = "\n\n".join(found)
    dates = nodes("date")
    publication_date = next((node for node in dates if node.get(f"{{{_OPF}}}event") == "publication"),
                            next((node for node in dates if not node.get(f"{{{_OPF}}}event")), None))
    if publication_date is not None and _text(publication_date):
        book["date"] = _text(publication_date)
    for name, key in (("identifier", "identifiers"), ("subject", "subjects")):
        found = values(name)
        if found:
            book[key] = found
    contributors = list(dict.fromkeys(_text(node) for node in nodes("contributor") + nodes("creator")
                                      if _text(node) and _text(node) not in authors))
    if contributors:
        book["contributors"] = contributors
    collections = [node for node in metadata if _tag(node) == "meta"
                   and node.get("property") == "belongs-to-collection"
                   and refinement(node, "collection-type") == "series"]
    if collections:
        book["series"] = _text(collections[0])
        position = refinement(collections[0], "group-position")
        if position:
            book["series_index"] = position
    else:
        for name, key in (("calibre:series", "series"), ("calibre:series_index", "series_index")):
            found = next((node.get("content", "").strip() for node in metadata
                          if _tag(node) == "meta" and node.get("name") == name), "")
            if found:
                book[key] = found
    return book


def _command(command: list[str], message: str) -> bytes:
    try:
        result = run_subprocess(command, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=90)
        return result.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(message) from exc


def _image_info(path: Path, ffprobe: str) -> dict:
    raw = _command([ffprobe, "-v", "error", "-max_alloc", "268435456", "-max_pixels", str(_MAX_PIXELS),
                    "-threads", "1", "-probesize", "33554432", "-analyzeduration", "1000000",
                    "-protocol_whitelist", "file,pipe", "-f", "image2", "-pattern_type", "none",
                    "-i", str(path), "-show_entries", "stream=codec_name,width,height", "-of", "json"],
                   "Unable to decode the EPUB cover; supply a valid JPEG or PNG cover and verify FFprobe is installed")
    try:
        streams = json.loads(raw)["streams"]
        if len(streams) != 1:
            raise ValueError
        info = streams[0]
        width, height = info["width"], info["height"]
        if (info["codec_name"] not in {"mjpeg", "png"}
                or not isinstance(width, int) or not isinstance(height, int)
                or not 0 < width <= _MAX_DIMENSION or not 0 < height <= _MAX_DIMENSION
                or width * height > _MAX_PIXELS):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EPUB cover is invalid or too large; use a JPEG/PNG up to 40 megapixels and 16384 pixels per side") from exc
    return {"width": width, "height": height, "codec": info["codec_name"]}


def prepare_book_artwork(epub: str | Path, output_dir: str | Path, expected_book_id: str, *,
                         ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    """Retain the original cover and make a 1600px square cover for publication.

    The foreground uses every source pixel at its original aspect ratio. Only
    the separately blurred, dimmed background is cropped. Provenance paths are
    relative to output_dir so publication staging can move the whole directory.
    """
    # Lazy import avoids a cycle when assembly consumes these publication assets.
    from .assembly import extract_cover

    epub, destination = Path(epub), Path(output_dir)
    book = read_publication_metadata(epub, expected_book_id)
    if destination.is_symlink():
        raise ValueError("Publication artwork directory must not be a symbolic link")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".artwork-", dir=destination) as temporary:
        staging = Path(temporary)
        original = extract_cover(epub, staging, expected_book_id)
        if original is None:
            raise ValueError("The EPUB has no supported declared cover. Add a JPEG or PNG cover to the EPUB; audiobook publication requires cover artwork")
        info = _image_info(original, ffprobe)
        original = original.rename(staging / ("original-cover" + original.suffix))
        cover = staging / "cover.jpg"
        # Crop before enlarging the backdrop to bound memory even for a very
        # narrow source. Foreground scaling never crops or stretches the cover.
        graph = (f"[0:v]setsar=1,split=2[back][front];"
                 f"[back]crop=min(iw\\,ih):min(iw\\,ih),scale={_SIZE}:{_SIZE}:flags=lanczos,"
                 "gblur=sigma=40:steps=3,eq=brightness=-0.12:saturation=0.75[blur];"
                 f"[front]scale={_SIZE}:{_SIZE}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[fit];"
                 "[blur][fit]overlay=(W-w)/2:(H-h)/2:format=auto,format=yuvj444p[out]")
        _command([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                  "-max_alloc", "268435456", "-max_pixels", str(_MAX_PIXELS), "-threads", "1",
                  "-probesize", "33554432", "-analyzeduration", "1000000", "-protocol_whitelist", "file,pipe",
                  "-f", "image2", "-pattern_type", "none", "-i", str(original),
                  "-filter_complex_threads", "1", "-filter_complex", graph, "-map", "[out]",
                  "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "2", "-threads", "1",
                  "-update", "1", str(cover)],
                 "Unable to create square cover artwork; check the original cover and FFmpeg installation")
        square = _image_info(cover, ffprobe)
        if square != {"width": _SIZE, "height": _SIZE, "codec": "mjpeg"}:
            raise ValueError("Generated cover did not verify as a 1600 by 1600 JPEG; publication stopped")
        artwork = {"version": 1, "style": _STYLE, "epub_sha256": expected_book_id,
                   "source_image": {"path": original.name, "sha256": _hash(original), **info},
                   "cover": {"path": cover.name, "sha256": _hash(cover), **square}}
        for source in (original, cover):
            target = destination / source.name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Publication artwork output must be a regular file")
            if target.exists() and _hash(target) != _hash(source):
                raise ValueError("Artwork output already contains a different cover; choose an empty publication staging directory")
        for source in (original, cover):
            target = destination / source.name
            if not target.exists():
                # Never overwrite a concurrently created file or follow a symlink.
                target.hardlink_to(source)
    return {"book": book, "cover_path": destination / "cover.jpg", "artwork": artwork}
