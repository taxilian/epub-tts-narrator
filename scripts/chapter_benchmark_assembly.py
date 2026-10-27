"""Final benchmark book assembly over validated chapter downloads.

Each downloaded benchmark chapter is revalidated with the production client
checks, its lossless master is decoded and PCM-verified, and the four groups
are packaged into one M4B through the production packaging path. This is the
same final assembly contract as a normal audiobook, never a re-encode of
lossy chapter files.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.chapter_client import _run_audio, _validate_metadata
from epub_tts_narrator.renderer.common import file_hash, pcm_sha256, regular_file, write_json


def _decoded_chapter(folder, record, scratch, render):
    """Production metadata validation plus a PCM-verified lossless decode."""
    metadata = _validate_metadata(folder, record)
    composed = deepcopy(metadata.get("chapter", metadata))
    chapter = record["selection"]["chapter"]
    wav = scratch / f"{chapter['id']}.wav"
    if composed["frames"]:
        _run_audio([render["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                    "-i", str(folder / "chapter.flac"), "-map", "0:a:0", "-c:a", "pcm_s16le", str(wav)])
        if pcm_sha256(wav) != metadata.get("pcm_data_sha256"):
            raise ValueError("Decoded chapter master differs from the server's lossless PCM")
    else:
        with wave.open(str(wav), "wb") as stream:
            stream.setparams((1, 2, render["sample_rate"], 0, "NONE", "not compressed"))
    composed["wav_path"] = str(wav)
    composed.pop("pcm_sha256", None)
    composed.update(number=chapter["number"], title=chapter["title"])
    return composed


def assemble_book(corpus, downloaded, config, destination, *, notify=lambda message: None):
    """Package four validated groups into one published M4B; returns timing/path.

    ``downloaded`` maps each group to its (folder, record/descriptor) exactly as
    the production client tracks validated downloads.
    """
    from epub_tts_narrator.renderer.assembly import package_audio
    from epub_tts_narrator.renderer.workflow import _publish

    began = time.perf_counter()
    render = config["render"]
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Benchmark book output already exists; choose a new --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-book-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        chapters = []
        for group in range(4):
            if group not in downloaded:
                raise ValueError(f"Benchmark group {group + 1} has no validated download; "
                                 "complete every group before assembly")
            folder, record = downloaded[group]
            chapters.append(_decoded_chapter(folder, record, stage, render))
        audio = package_audio(chapters, stage, book=deepcopy(corpus["book"]),
                              chapter_format="m4a", aac_bitrate=render["aac_bitrate"],
                              mp3_bitrate=render["mp3_bitrate"], ffmpeg=render["ffmpeg"],
                              ffprobe=render["ffprobe"],
                              source_fingerprint=corpus["source_fingerprint"], notify=notify)
        result = {"format": "audiobook-render", "version": 1, "status": "complete",
                  "source_fingerprint": corpus["source_fingerprint"], "partial_book": False,
                  "series_id": None, "audio": audio,
                  "benchmark": {"corpus_sha256": corpus["corpus_sha256"], "input_hash": corpus["input_hash"],
                                "mode": "chapter_pipeline_benchmark"},
                  "files_sha256": {path.relative_to(stage).as_posix(): file_hash(path)
                                   for path in sorted(stage.rglob("*")) if path.is_file()}}
        write_json(stage / "render.json", result)
        _publish(stage, destination)
    seconds = time.perf_counter() - began
    notify(f"Benchmark book assembled: {destination / 'book.m4b'}")
    return {"seconds": seconds, "path": str(destination / "book.m4b"), "result": result}
