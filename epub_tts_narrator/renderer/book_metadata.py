"""EPUB publication assets and portable audiobook container metadata."""

from copy import deepcopy
from pathlib import Path
import tempfile


PUBLICATION_ASSET_VERSION = 2


def accepted_source_epubs(bundle):
    """SHA-256 values of EPUBs that belong to this book.

    The book ID is the original EPUB's SHA-256. A serial updated with appended
    chapters keeps that ID and records its latest EPUB in
    ``provenance.book_updates``; either file supplies cover and metadata.
    """
    from ..book_update import accepted_epub_digests
    return accepted_epub_digests(bundle.get("manifest", {})) | {bundle["source"]["book"]["id"]}


def source_epub_digest(bundle, epub):
    """The digest to check ``epub`` against: its own when it belongs to this book.

    The original EPUB (the book ID) and the latest update's EPUB pass by bytes.
    Otherwise the EPUB is accepted by content when the updater's own comparison
    proves it extracts to the same book as the saved analysis, so a repacked or
    typo-corrected revision at the remembered location can still supply the
    cover and metadata. The saved source, book ID, and every render identity
    stay unchanged; anything else keeps the book ID and publication fails closed.
    """
    from .common import file_hash
    actual = file_hash(epub)
    if actual in accepted_source_epubs(bundle):
        return actual
    from ..book_update import same_book_epub
    if same_book_epub(bundle["source"], epub):
        return actual
    return bundle["source"]["book"]["id"]


def check_publication_source(bundle, config):
    """Fail before synthesis if required artwork cannot be obtained."""
    from .assembly import extract_cover
    epub = config["render"].get("epub")
    required = config["render"].get("require_cover", True)
    if not epub:
        if required:
            raise ValueError("The original EPUB is needed for the audiobook cover and metadata. "
                             "Pass --epub PATH; new analyses remember their EPUB automatically.")
        return
    if not Path(epub).is_file():
        raise ValueError("The original EPUB is no longer available; pass --epub PATH to its current location.")
    expected = source_epub_digest(bundle, epub)
    with tempfile.TemporaryDirectory(prefix="epub-cover-check-") as temporary:
        cover = extract_cover(epub, temporary, expected)
        if cover is None and required:
            raise ValueError("The EPUB has no supported declared JPEG/PNG cover. "
                             "Add a supported cover to the source before analysis, or explicitly set "
                             "render.require_cover = false to publish without artwork.")
        if cover is not None:
            from .book_artwork import prepare_book_artwork
            prepare_book_artwork(epub, Path(temporary) / "checked", expected,
                                 ffmpeg=config["render"]["ffmpeg"], ffprobe=config["render"]["ffprobe"])


def publication_assets(bundle, config, output_dir, *, notify=lambda message: None):
    """Enrich publication data without changing analyzed text or fingerprints."""
    from .book_artwork import prepare_book_artwork, read_publication_metadata
    book = deepcopy(bundle["source"]["book"])
    epub = config["render"].get("epub")
    if not epub:
        check_publication_source(bundle, config)
        notify("Cover requirement explicitly disabled; publishing without artwork.")
        return book, None, None
    # An explicit opt-out is only for genuinely absent/unsupported artwork;
    # corrupt archives, wrong books, or conversion failures still stop publication.
    from .common import file_hash
    expected = source_epub_digest(bundle, epub)
    if expected == file_hash(epub) and expected not in accepted_source_epubs(bundle):
        notify("The EPUB is a newer revision of the analyzed book; its cover and metadata are "
               "used and the saved analysis is unchanged. Run `epub-tts-narrator analyze EPUB "
               "-o BOOK_DIRECTORY` to absorb its text changes into the audiobook.")
    if not config["render"].get("require_cover", True):
        from .assembly import extract_cover
        with tempfile.TemporaryDirectory(prefix="epub-cover-check-") as temporary:
            if extract_cover(epub, temporary, expected) is None:
                book.update(read_publication_metadata(epub, expected))
                book["id"] = bundle["source"]["book"]["id"]
                notify("Cover requirement explicitly disabled; EPUB has no supported cover.")
                return book, None, None
    assets = prepare_book_artwork(epub, output_dir, expected,
                                 ffmpeg=config["render"]["ffmpeg"],
                                 ffprobe=config["render"]["ffprobe"])
    book.update(assets["book"])
    book["id"] = bundle["source"]["book"]["id"]
    # Artwork is under this fixed subdirectory in the published package.
    for image in ("source_image", "cover"):
        assets["artwork"][image]["path"] = "artwork/" + assets["artwork"][image]["path"]
    notify("Prepared square EPUB cover and book metadata.")
    return book, assets["cover_path"], assets["artwork"]


def book_tags(book):
    """Use standard iTunes/M4B tags, retaining richer EPUB data in the manifest."""
    title = str(book.get("title", "Audiobook"))
    author = "; ".join(book.get("authors", []))
    tags = {"title": title, "album": title, "artist": author, "album_artist": author,
            "genre": "Audiobook", "media_type": "2"}
    for source, target in (("date", "date"), ("publisher", "publisher"),
                           ("rights", "copyright"), ("description", "description")):
        if book.get(source):
            tags[target] = str(book[source])
    if book.get("description"):
        tags["synopsis"] = str(book["description"])
    if book.get("series"):
        tags["grouping"] = str(book["series"])
    return tags


def metadata_arguments(book):
    result = []
    for key, value in book_tags(book).items():
        if "\x00" in value:
            raise ValueError("NUL is not allowed in audio metadata")
        result.extend(("-metadata", f"{key}={value}"))
    # MP4's packed language field requires ISO 639-2, not an EPUB BCP-47 tag.
    language = str(book.get("language", "und")).lower().split("-")[0]
    language = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita",
                "pt": "por", "nl": "nld", "ja": "jpn", "zh": "zho", "ko": "kor",
                "ru": "rus", "ar": "ara", "hi": "hin", "sv": "swe", "da": "dan",
                "no": "nor", "fi": "fin", "pl": "pol", "tr": "tur", "cs": "ces",
                "el": "ell", "he": "heb", "uk": "ukr"}.get(language, language)
    if len(language) == 3 and language.isascii() and language.isalpha():
        result.extend(("-metadata:s:a:0", f"language={language}"))
    return result


def check_book_tags(inspection, book):
    tags = inspection.get("format", {}).get("tags", {})
    expected = book_tags(book)
    for key in ("title", "album", "artist", "album_artist", "genre", "media_type",
                "date", "copyright", "description", "synopsis", "grouping"):
        if expected.get(key) and tags.get(key) != expected[key]:
            raise ValueError(f"Encoded audiobook is missing or changed its {key} metadata")
