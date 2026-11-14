"""New renders keep FLAC; books rendered with WAVs keep working and convert-flac converts them exactly."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer import flac_migration, stored_audio
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import file_hash
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.run_locks import acquire_directory_lock
from epub_tts_narrator.renderer.workflow import render
from audition_fixtures import prepare_bundle, prepare_document
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_renderer_stored_audio import float_wav, tone
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier, pcm_key


# Every writer that keeps audio as FLAC; patching them out renders as older versions did.
_WRITERS = ("workflow", "chapter_cache", "loudness", "clip_repair_execute")


@contextlib.contextmanager
def wav_renderer():
    """Render exactly as before FLAC storage: every kept file stays a WAV."""
    with contextlib.ExitStack() as stack:
        for module in _WRITERS:
            stack.enter_context(patch(f"epub_tts_narrator.renderer.{module}.store_quietly",
                                      side_effect=lambda *args, **kwargs: None))
        yield


class StoringVerifier(FixtureVerifier):
    """The fixture verifier, reading a take in whichever form it is kept."""

    def transcribe(self, path):
        with stored_audio.materialize(path) as wav:
            return super().transcribe(wav)


def files(root, suffix):
    return sorted(path for path in Path(root).rglob("*" + suffix)
                  if path.is_file() and not any(part.startswith(".") for part in path.parts))


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for render integration")
class FlacRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = renderer_config(self.root, series_id="fixture-series",
                                      checkpoint="unused-tts-model", asr_model="unused-asr-model")
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = StoringVerifier(self.fixtures)
        self.bundle = prepare_bundle(load_bundle(EXAMPLE / "bundle"))
        self.work = Path(self.config["render"]["work_dir"])

    def run_render(self, output=None):
        config = self.config
        if output is not None:
            config = {**config, "render": {**config["render"], "output_dir": str(output)}}
        return render(self.bundle, config, self.engine, self.verifier, notify=lambda _: None)

    def records(self):
        return {path: path.read_bytes() for path in self.work.rglob("*.json")}

    def test_new_render_keeps_takes_and_chapters_only_as_flac_that_restore_their_recorded_checksums(self):
        self.assertEqual(self.run_render()["status"], "complete")
        self.assertEqual(files(self.work / "clips", ".wav"), [])
        self.assertEqual(files(self.work / "chapters", ".wav"), [])
        takes = files(self.work / "clips", ".flac")
        chapters = files(self.work / "chapters", ".flac")
        self.assertTrue(takes and chapters)
        from epub_tts_narrator.renderer.common import read_json
        for accepted in (self.work / "clips").glob("*/accepted.json"):
            record = read_json(accepted)
            self.assertTrue(record["audio_file"].endswith(".wav"), "records keep naming the WAV")
            self.assertEqual(stored_audio.audio_hash(accepted.parent / record["audio_file"]), record["audio_sha256"])
        for chapter in (self.work / "chapters").glob("*/chapter.json"):
            record = read_json(chapter)
            self.assertEqual(stored_audio.audio_hash(chapter.parent / record["wav_file"]), record["pcm_sha256"])

        # A rerun reuses everything; nothing is generated or recognized again.
        speech, recognized = len(self.engine.speech_calls), len(self.verifier.calls)
        self.assertEqual(self.run_render(self.root / "second-output")["status"], "complete")
        self.assertEqual((len(self.engine.speech_calls), len(self.verifier.calls)), (speech, recognized))
        self.assertTrue(files(self.root / "second-output", ".m4b"))

    def test_a_wav_book_renders_and_assembles_partly_and_fully_converted_without_new_speech(self):
        with wav_renderer():
            self.assertEqual(self.run_render()["status"], "complete")
        takes, chapters = files(self.work / "clips", ".wav"), files(self.work / "chapters", ".wav")
        self.assertTrue(takes and chapters)
        self.assertEqual(files(self.work, ".flac"), [])
        speech, recognized = len(self.engine.speech_calls), len(self.verifier.calls)
        frames = {path: pcm_key(path) for path in chapters}

        # The newer renderer reuses the WAV book as it is.
        self.assertEqual(self.run_render(self.root / "unchanged")["status"], "complete")
        self.assertEqual(files(self.work / "chapters", ".flac"), [])
        self.assertEqual(files(self.work / "clips", ".flac"), [])

        # Only the takes converted: the chapter is still the WAV it was.
        for take in takes:
            stored_audio.store(take)
        self.assertEqual(self.run_render(self.root / "partly")["status"], "complete")

        # Every take and chapter converted, records untouched.
        before = self.records()
        bundle = self.bundle
        report = flac_migration.convert_book(bundle, self.config, confirm=lambda report: True)
        self.assertTrue(report["converted"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(files(self.work / "clips", ".wav"), [])
        self.assertEqual(files(self.work / "chapters", ".wav"), [])
        self.assertEqual(self.records(), before)
        for path, frame_hash in frames.items():
            with stored_audio.open_wave(path) as audio:
                import hashlib
                self.assertEqual(hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest(), frame_hash)
        self.assertEqual(self.run_render(self.root / "converted")["status"], "complete")
        self.assertEqual((len(self.engine.speech_calls), len(self.verifier.calls)), (speech, recognized))
        books = [files(self.root / name, ".m4b")[0] for name in ("unchanged", "partly", "converted")]
        from epub_tts_narrator.renderer.assembly import probe_audio
        durations = {probe_audio(book, ffprobe=FFPROBE)["format"]["duration"] for book in books}
        self.assertEqual(len(durations), 1, "every form assembles the same book length")

        # Converting again finds nothing.
        again = flac_migration.convert_book(bundle, self.config, dry_run=True)
        self.assertEqual(again["groups"]["clips"]["files"], 0)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for packaging")
class MixedChapterPackagingTests(unittest.TestCase):
    def test_a_book_with_wav_and_flac_chapters_packages_the_same_m4b_as_all_wav(self):
        from epub_tts_narrator.renderer import assembly, chapter_cache
        from test_renderer_assembly import clip
        from test_renderer_loudness import signal
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        book = {"id": "book-id", "title": "A Book", "authors": ["An Author"]}
        ready = []
        with wav_renderer():
            for number, amplitude in ((1, 1800), (2, 2600)):
                source = signal(root / f"speech-{number}.wav", amplitude)
                chapter = {"id": f"chapter{number}", "title": f"Chapter {number}", "number": number,
                           "script_path": f"chapters/{number:04d}.jsonl", "_ordinal": number, "_total_chapters": 2}
                ready.append(chapter_cache.prepare_chapter(
                    chapter, [clip(source)], root / "work" / "chapters", book=book,
                    render_settings={"sample_rate": 44100, "normalize_loudness": True},
                    context={"plan": number}, notify=lambda _: None))
        assembly.package_audio(ready, root / "wav-book", book=book)
        stored_audio.store(ready[1]["wav_path"])
        for item in ready:
            reloaded = chapter_cache.load_chapter(item["cache_directory"])
            self.assertEqual(reloaded["pcm_sha256"], item["pcm_sha256"])
        concat = []
        run = assembly._run

        def record(arguments, **kwargs):
            if "concat" in arguments:
                concat.append(Path(arguments[arguments.index("-i") + 1]).read_text())
            return run(arguments, **kwargs)
        with patch.object(assembly, "_run", side_effect=record):
            assembly.package_audio(ready, root / "mixed-book", book=book)
        self.assertNotIn(".flac", concat[0], "mixed forms join exact WAV copies")
        stored_audio.store(ready[0]["wav_path"])
        concat.clear()
        with patch.object(assembly, "_run", side_effect=record):
            assembly.package_audio(ready, root / "flac-book", book=book)
        self.assertEqual(concat[0].count(".flac'"), 2)
        probes = [assembly.probe_audio(next((root / name).glob("*.m4b")), ffprobe=FFPROBE)
                  for name in ("wav-book", "mixed-book", "flac-book")]
        self.assertEqual(len({probe["format"]["duration"] for probe in probes}), 1)
        self.assertEqual(len({str(probe["chapters"]) for probe in probes}), 1)
        self.assertEqual(sorted(path.name for path in (root / "work").rglob("*.wav")
                                if not path.name.startswith(".")), [])


class ConvertCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config_path = self.root / "renderer.toml"
        self.config_path.write_text('[engine]\ncheckpoint="./unused-tts-model"\n'
                                    '[verification]\nmodel="./unused-asr-model"\n')
        self.book = self.root / "book"
        export_bundle(prepare_document(valid_document()), self.book)
        self.bundle = load_bundle(self.book)
        self.config = resolve_book_config(self.bundle, load_config(self.config_path))
        self.work = Path(self.config["render"]["work_dir"])
        clips = self.work / "clips" / ("a" * 64)
        chapters = self.work / "chapters" / ("b" * 64)
        mixes = self.work / "mixes" / ("c" * 64)
        previews = self.work / "voice-previews" / ("d" * 64)
        for folder in (clips, chapters, mixes, previews, self.work / "casting-candidates" / "e"):
            folder.mkdir(parents=True)
        self.take = tone(clips / "attempt-001-speech-take.wav")
        self.raw = float_wav(clips / "attempt-001-speech-raw.wav")
        self.chapter = tone(chapters / "chapter-0001.wav", seconds=1)
        self.mix = tone(mixes / "simultaneous-mix-key-c-take.wav")
        self.preview = tone(previews / "hero-loudness-sample.wav")
        self.casting = tone(self.work / "casting-candidates" / "e" / "attempt-001-casting-take.wav")
        self.hidden = tone(clips / ".attempt-001-staging.wav")
        (self.book / "audiobook").mkdir()
        self.published = tone(self.book / "audiobook" / "untouched.wav")
        self.checksums = {path: file_hash(path) for path in
                          (self.take, self.raw, self.chapter, self.mix, self.preview, self.casting)}

    def run_command(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                patch("sys.stdin", io.StringIO("")):
            status = main([str(arg) for arg in ("convert-flac", self.book, "--config", self.config_path, *args)])
        return status, out.getvalue(), err.getvalue()

    def test_dry_run_and_no_terminal_convert_nothing(self):
        status, out, _ = self.run_command("--dry-run")
        self.assertEqual(status, 0)
        self.assertIn("Dry run: nothing was converted.", out)
        self.assertIn("Total 5 WAV file(s)", out)
        status, out, _ = self.run_command()
        self.assertEqual(status, 0)
        self.assertIn("Rerun with --yes", out)
        for path, checksum in self.checksums.items():
            self.assertEqual(file_hash(path), checksum)
        self.assertEqual(stored_audio.audio_files(self.take), [self.take])

    def test_yes_converts_kept_render_wavs_exactly_and_leaves_everything_else(self):
        status, out, _ = self.run_command("--yes")
        self.assertEqual(status, 0, out)
        self.assertIn("Total 4 file(s)", out)
        self.assertIn("Kept 1 WAV file(s) FLAC cannot hold exactly", out)
        for path in (self.take, self.chapter, self.mix, self.preview):
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix(".flac").is_file())
            self.assertEqual(stored_audio.audio_hash(path), self.checksums[path])
        for path in (self.raw, self.casting, self.hidden, self.published):
            self.assertTrue(path.is_file(), path)
        self.assertEqual(file_hash(self.raw), self.checksums[self.raw])
        status, out, _ = self.run_command("--yes")
        self.assertEqual(status, 0)
        self.assertIn("Total 0 file(s)", out)
        self.assertIn("Kept 1 WAV file(s)", out)

    def test_only_selects_groups_and_rejects_unknown_ones(self):
        status, _, _ = self.run_command("--yes", "--only", "chapters")
        self.assertEqual(status, 0)
        self.assertTrue(stored_audio.is_stored(self.chapter))
        self.assertFalse(stored_audio.is_stored(self.take))
        status, _, err = self.run_command("--only", "voices")
        self.assertNotEqual(status, 0)
        self.assertIn("--only takes one or more of clips, chapters, mixes, previews", err)

    def test_a_book_in_use_is_refused_without_converting(self):
        lock = acquire_directory_lock(self.work / ".render.lock")
        self.addCleanup(lock.rmdir)
        status, _, err = self.run_command("--yes")
        self.assertEqual(status, 2)
        self.assertIn("is using this book", err)
        self.assertTrue(self.take.is_file())

    def test_an_interrupted_conversion_resumes_and_a_failure_keeps_the_wav(self):
        real = stored_audio.store
        calls = []

        def interrupted(path, **kwargs):
            calls.append(path)
            if path == self.chapter:
                raise stored_audio.StoredAudioError("simulated decoder mismatch")
            return real(path, **kwargs)
        with patch.object(stored_audio, "store", side_effect=interrupted):
            status, out, _ = self.run_command("--yes")
        self.assertEqual(status, 2)
        self.assertIn("Failed, WAV kept", out)
        self.assertTrue(self.chapter.is_file())
        self.assertEqual(file_hash(self.chapter), self.checksums[self.chapter])
        status, out, _ = self.run_command("--yes")
        self.assertEqual(status, 0, out)
        self.assertTrue(stored_audio.is_stored(self.chapter))


if __name__ == "__main__":
    unittest.main()
