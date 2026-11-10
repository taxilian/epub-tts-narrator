"""Shared test helpers used by multiple test modules (not a test module itself)."""

from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import wave
import zipfile

from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def service_session(service):
    """Open an ephemeral connected-client session for a ChapterJobs under test."""
    session = os.urandom(32).hex()
    service.sessions.open(session)
    return session


class Terminal(io.StringIO):
    def isatty(self):
        return True


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def rebind_audiobook_filename(directory, filename, *, record_path=True):
    """Simulate an intact preexisting publication with its recorded output name."""
    directory = Path(directory)
    record = read_json(directory / "render.json")
    audio = record["audio"]
    sync = read_json(directory / "sync.json")
    current = audio.get("audiobook", {}).get("path") or sync.get("audio_file") or "book.m4b"
    if current != filename:
        (directory / current).rename(directory / filename)
    audiobook = audio.setdefault("audiobook", {})
    if record_path:
        audiobook["path"] = filename
    else:
        audiobook.pop("path", None)
    sync["audio_file"] = filename
    write_json(directory / "sync.json", sync)
    if isinstance(audio.get("sync"), dict):
        audio["sync"]["sha256"] = file_hash(directory / "sync.json")
    write_json(directory / "audio-manifest.json", audio)
    files = record["files_sha256"]
    files.pop(current, None)
    files[filename] = file_hash(directory / filename)
    files["sync.json"] = file_hash(directory / "sync.json")
    files["audio-manifest.json"] = file_hash(directory / "audio-manifest.json")
    write_json(directory / "render.json", record)
    return record


def pcm_hash(path):
    with wave.open(str(path), "rb") as audio:
        return hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()


def make_epub(path, title, identifier, body):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        archive.writestr("content.opf", f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:identifier id="id">{identifier}</dc:identifier><dc:language>en</dc:language></metadata><manifest><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/></spine></package>')
        archive.writestr("one.xhtml", body)
    return path


def repack_epub(epub, path):
    """Rewrite an EPUB with different archive bytes and identical documents.

    A rebuilt or re-exported edition of the same book changes container bytes
    (and with them the EPUB's SHA-256) without changing its extracted content.
    """
    with zipfile.ZipFile(epub) as archive:
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return path


def make_serial_epub(path, title, identifier, chapters):
    """An EPUB with one spine document per (heading, paragraphs) chapter.

    Serial publications grow by appending chapters; writing the same leading
    chapters again reproduces their extracted blocks exactly.
    """
    manifest, spine, documents = [], [], {}
    for number, (heading, paragraphs) in enumerate(chapters, 1):
        name = f"chapter-{number:03d}.xhtml"
        manifest.append(f'<item id="c{number}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{number}"/>')
        body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
        documents[name] = (f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{heading}</title></head>'
                           f'<body><h1>{heading}</h1>{body}</body></html>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        archive.writestr("content.opf", f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:identifier id="id">{identifier}</dc:identifier><dc:language>en</dc:language></metadata><manifest>{"".join(manifest)}</manifest><spine>{"".join(spine)}</spine></package>')
        for name, document in documents.items():
            archive.writestr(name, document)
    return path


def timed_transcript(text):
    words = text.split()
    return {"text": text, "segments": [{
        "text": text, "start": 0.0, "end": 1.0,
        "avg_logprob": -0.1, "no_speech_prob": 0.001,
        "words": [
            {"word": word, "start": index / len(words),
             "end": (index + 1) / len(words), "probability": 0.99}
            for index, word in enumerate(words)
        ],
    }]}


class FakeEntryPoints(list):
    def select(self, *, name):
        return FakeEntryPoints(entry for entry in self if entry.name == name)


PARENTHESIS_TRIM_CASES = (
    ("(Lie, ", "Lie,"),
    ("I thought fiercely). ", "I thought fiercely."),
    ("(Lie.)", "Lie."),
    ("He said (quietly) that he agreed.", "He said (quietly) that he agreed."),
    ("He said (quietly).", "He said (quietly)."),
)


def renderer_config(root=None, *, series_id=None, checkpoint="unused-tts", asr_model="unused-asr",
                    max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE, **render):
    config = deepcopy(DEFAULTS)
    # Synthetic renderer fixtures have no publication EPUB/artwork.
    render.setdefault("require_cover", False)
    if root is not None:
        config["engine"]["checkpoint"] = str(root / checkpoint)
        config["verification"]["model"] = str(root / asr_model)
        config["voices"].update(directory=str(root / "voices"), series_id=series_id)
        render.setdefault("work_dir", str(root / "work"))
        render.setdefault("output_dir", str(root / "output"))
    config["render"].update(max_attempts=max_attempts, ffmpeg=ffmpeg, ffprobe=ffprobe, **render)
    config["quality"] = validate_policy({})
    return config


def fixture_codes(text, voices, *, frames=4):
    """Generated-codes metadata that a chapter-history fake engine returns.

    Every renderer engine must return the codec tokens it generated so the
    next take can continue the chapter. Fakes return small valid codes for
    the speaker-tagged text they were asked to speak.
    """
    voices = list(voices)
    distinct = []
    for voice in voices:
        if voice.get("asset_hash") not in [item.get("asset_hash") for item in distinct]:
            distinct.append(voice)
    if not text.lstrip().startswith("<|speaker:"):
        text = f"<|speaker:0|> {text}"
    seed = sum(map(ord, text)) % 1000
    return {"generated_codes": {"codes": [[(seed + row) % 1024] * frames for row in range(10)],
                                "codebook_size": 1024, "text": text,
                                "speakers": [voice.get("asset_hash") for voice in distinct]}}


HISTORY_CAPABILITY = {"chapter_continuity": True}


def route_mlx_takes_to_fake_generate(testcase):
    """Serve the MLX engine's watched decode loop from a fake model's ``generate``.

    Fake Drama 3 models implement only the vendored ``Drama3.generate``
    interface. The engine now runs its own copy of that loop
    (``mlx_drama3_decode.generate_segment``, checked against the vendored
    runtime by ``scripts/check_drama3_silence_stop.py --mlx``), so this routes
    each take to ``model.generate(..., single_chunk=True, attempts=1)`` with the
    engine's arguments and records the take's silence watch in
    ``testcase.mlx_watches``. A fake that yields nothing is a take without frames.
    """
    from unittest.mock import patch
    from epub_tts_narrator.renderer.silence_stop import SilenceWatch

    testcase.mlx_watches = []

    def generate_segment(model, text, *, watch=None, **options):
        testcase.mlx_watches.append(watch)
        return next(iter(model.generate(text, single_chunk=True, attempts=1, **options)), None)

    def silence_watch(model, text):
        return SilenceWatch(text, lambda frames: None, sample_rate=44100)

    for name, fake in (("generate_segment", generate_segment), ("silence_watch", silence_watch)):
        patcher = patch(f"epub_tts_narrator.renderer.mlx_drama3_decode.{name}", fake)
        patcher.start()
        testcase.addCleanup(patcher.stop)
