"""Shared test helpers used by multiple test modules (not a test module itself)."""

from copy import deepcopy
import hashlib
import io
import os
import shutil
import wave
import zipfile

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
