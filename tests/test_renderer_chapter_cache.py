"""Durable chapter audio, cache integrity, exact timing, and lossless packaging."""

from pathlib import Path
from copy import deepcopy
import shutil
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from audiobook_renderer import assembly, chapter_cache
from audiobook_renderer.chapter_cache import load_chapter, prepare_chapter
from audiobook_renderer.common import digest, file_hash, read_json, write_json
from test_renderer_assembly import clip, make_wav
from test_renderer_loudness import signal


class ChapterSettingsTests(unittest.TestCase):
    def test_pacing_policy_is_versioned_and_rejects_invalid_values(self):
        settings = chapter_cache.chapter_settings({})
        self.assertEqual(settings["version"], 2)
        self.assertIs(settings["normalize_pauses"], True)
        self.assertEqual((settings["sentence_pause_ms"], settings["clause_pause_ms"]), (220, 100))
        for key, values in {
            "normalize_pauses": (1, "true", None),
            "sentence_pause_ms": (-1, 5001, True, 220.0, float("nan")),
            "clause_pause_ms": (-1, 5001, False, "100", float("inf")),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    chapter_cache.chapter_settings({key: value})


class ChapterTimingValidationTests(unittest.TestCase):
    def setUp(self):
        self.segment = {"chapter_start_frame": 0, "chapter_end_frame": 1000,
                        "source_audio_start_frame": 100, "source_audio_end_frame": 1100}

    def validate(self, segment):
        chapter_cache._timings({"frames": 1000, "segments": [segment]})

    def test_trim_offsets_are_optional_but_must_be_paired_and_match_playback_duration(self):
        self.validate(self.segment)
        self.validate({"chapter_start_frame": 0, "chapter_end_frame": 1000})
        for changes in ({"source_audio_start_frame": None}, {"source_audio_end_frame": None},
                        {"source_audio_start_frame": -1}, {"source_audio_end_frame": 50},
                        {"source_audio_start_frame": True}, {"source_audio_end_frame": 1100.0},
                        {"source_audio_end_frame": 1200}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "source audio"):
                self.validate({**self.segment, **changes})
        segment = dict(self.segment)
        del segment["source_audio_start_frame"]
        with self.assertRaisesRegex(ValueError, "source audio"):
            self.validate(segment)

    def test_paragraph_source_segments_have_valid_shape_and_spoken_offsets(self):
        source = {"chunk_id": "c1", "segment_id": "s1", "block_id": "b1",
                  "voice_id": "v1", "role_id": "r1", "spoken_text": "Hi",
                  "spoken_start": 20, "spoken_end": 22,
                  "source_ranges": [{"start": 40, "end": 42}]}
        segment = dict(self.segment, source_segments=[source, deepcopy(source)])
        self.validate(segment)
        for bad in (None, [], [source], [source, None]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "paragraph source"):
                self.validate(dict(segment, source_segments=bad))
        for changes in ({"voice_id": None}, {"spoken_end": 23}, {"spoken_start": True},
                        {"spoken_text": ""}, {"source_ranges": []},
                        {"source_ranges": [{"start": 42, "end": 40}]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "paragraph source"):
                self.validate(dict(segment, source_segments=[source, {**source, **changes}]))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class ChapterCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = signal(self.root / "speech.wav", 1800)
        self.clips = [clip(self.source)]
        self.chapter = {"id": "chapter7", "title": "A Later Chapter", "number": 7,
                        "script_path": "chapters/0007.jsonl", "_ordinal": 1, "_total_chapters": 1}
        self.book = {"id": "book-id", "title": "A Book", "authors": ["An Author"]}
        self.settings = {"sample_rate": 44100, "normalize_loudness": True}
        self.context = {"plan_hash": "plan", "voices": {"voice": "asset"},
                        "verifier": "checked-v1", "quality": {"max_wer": 0}}

    def prepare(self, **kwargs):
        return prepare_chapter(
            self.chapter, self.clips, self.root / "work" / "chapters", book=self.book,
            render_settings=self.settings, context=self.context, notify=lambda _: None, **kwargs)

    def test_ready_chapter_preserves_frames_source_and_reuses_verified_encodings(self):
        original = file_hash(self.source)
        first = self.prepare()
        self.assertEqual(first["frames"], 44100 * 4 + 11025)
        self.assertEqual(first["segments"][0]["chapter_end_frame"], 44100 * 4)
        self.assertEqual(first["number"], 7)
        self.assertEqual(first["normalization"]["status"], "normalized")
        self.assertAlmostEqual(first["normalization"]["output_lufs"], -20, delta=1)
        # Independently measure the published PCM instead of trusting the
        # normalization pass's report about its own output.
        measured = chapter_cache._loudness_pass(
            Path(first["wav_path"]), "loudnorm=I=-20:TP=-1.5:LRA=50:print_format=json",
            None, first["settings"])
        self.assertAlmostEqual(float(measured["input_i"]), -20, delta=1)
        self.assertLessEqual(float(measured["input_tp"]), -1.3)
        self.assertEqual(file_hash(self.source), original)
        with patch.object(chapter_cache, "compose_chapter", side_effect=AssertionError("recomposed")), \
             patch.object(chapter_cache, "_normalize", side_effect=AssertionError("renormalized")), \
             patch.object(chapter_cache, "encode_chapter", side_effect=AssertionError("reencoded")):
            self.assertEqual(self.prepare(), first)
        self.assertEqual(load_chapter(first["cache_directory"]), first)

    def test_changed_pause_voice_policy_or_clip_invalidates_checkpoint(self):
        first = self.prepare()
        self.settings["paragraph_pause_ms"] = 400
        pause = self.prepare()
        self.assertNotEqual(first["cache_key"], pause["cache_key"])
        self.assertEqual(pause["frames"] - first["frames"], 6615)
        self.context["voices"]["voice"] = "new-asset"
        voice = self.prepare()
        self.assertNotEqual(pause["cache_key"], voice["cache_key"])
        self.context["verifier"] = "checked-v2"
        policy = self.prepare()
        self.assertNotEqual(voice["cache_key"], policy["cache_key"])
        signal(self.source, 9000)
        changed_audio = self.prepare()
        self.assertNotEqual(policy["cache_key"], changed_audio["cache_key"])
        self.assertTrue(Path(first["encoded_path"]).is_file())

    def test_pacing_changes_recompose_saved_clips_with_updated_settings(self):
        self.settings["normalize_loudness"] = False
        self.clips = [clip(self.source, break_after="none"),
                      clip(self.source, identifier="b2:0:6", break_after="none")]
        original = file_hash(self.source)
        first = self.prepare()
        previous = first
        for key, value in (("sentence_pause_ms", 440), ("clause_pause_ms", 180),
                           ("normalize_pauses", False)):
            self.settings[key] = value
            with patch.object(chapter_cache, "compose_chapter", wraps=assembly.compose_chapter) as compose:
                changed = self.prepare()
            self.assertNotEqual(changed["cache_key"], previous["cache_key"])
            self.assertEqual(compose.call_args.kwargs[key], value)
            self.assertEqual(changed["settings"][key], value)
            self.assertEqual(file_hash(self.source), original)
            self.assertTrue(Path(previous["encoded_path"]).is_file())
            previous = changed
        # Recomposition remains independently cacheable, including when the
        # user returns to the default pacing settings later.
        for key in ("sentence_pause_ms", "clause_pause_ms", "normalize_pauses"):
            self.settings.pop(key)
        with patch.object(chapter_cache, "compose_chapter", side_effect=AssertionError("recomposed")):
            self.assertEqual(self.prepare(), first)

    def test_audio_and_metadata_corruption_are_rejected(self):
        for filename in ("chapter.wav", "chapter.m4a", "chapter.json"):
            with self.subTest(filename=filename):
                self.context["test"] = filename
                ready = self.prepare()
                path = Path(ready["cache_directory"]) / filename
                if filename.endswith("json"):
                    record = read_json(path)
                    record["chapter"]["frames"] += 1
                    write_json(path, record)
                else:
                    data = bytearray(path.read_bytes())
                    data[-1] ^= 1
                    path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "integrity"):
                    self.prepare()

    def test_nonobject_metadata_is_a_controlled_validation_error(self):
        ready = self.prepare()
        metadata = Path(ready["cache_directory"]) / "chapter.json"
        for value in (None, "unexpected text", ["unexpected", "array"]):
            with self.subTest(value=value):
                write_json(metadata, value)
                with self.assertRaisesRegex(ValueError, "metadata must be an object"):
                    load_chapter(metadata.parent)

    def test_impossible_cached_timing_is_rejected_even_with_updated_record_checksum(self):
        ready = self.prepare()
        path = Path(ready["cache_directory"]) / "chapter.json"
        record = read_json(path)
        record.pop("record_sha256")
        record["chapter"]["segments"][0]["chapter_end_frame"] = record["chapter"]["frames"] + 1
        write_json(path, dict(record, record_sha256=digest(record)))
        with self.assertRaisesRegex(ValueError, "synchronization"):
            load_chapter(path.parent)

    def test_interruption_does_not_publish_partial_chapter(self):
        original = file_hash(self.source)
        with patch.object(chapter_cache, "encode_chapter", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare()
        self.assertEqual(list((self.root / "work" / "chapters").iterdir()), [])
        self.assertEqual(file_hash(self.source), original)
        self.assertTrue(Path(self.prepare()["encoded_path"]).is_file())

    def test_loudness_timeout_uses_long_chapter_budget_and_preserves_source(self):
        original = file_hash(self.source)
        with patch.object(chapter_cache.subprocess, "run",
                          side_effect=subprocess.TimeoutExpired("ffmpeg", 3600)) as run:
            with self.assertRaisesRegex(ValueError, "Unable to normalize chapter"):
                self.prepare()
        self.assertEqual(run.call_args.kwargs["timeout"], 3600)
        self.assertEqual(file_hash(self.source), original)
        self.assertEqual(list((self.root / "work" / "chapters").iterdir()), [])

    def test_short_and_empty_chapters_record_explicit_loudness_skips(self):
        make_wav(self.source, frames=2205)
        ready = self.prepare()
        self.assertEqual(ready["normalization"]["status"], "skipped")
        self.assertEqual(ready["normalization"]["reason"], "too_short_for_integrated_loudness")
        self.assertTrue(Path(ready["encoded_path"]).is_file())
        self.clips = []
        empty = self.prepare()
        self.assertEqual(empty["frames"], 0)
        self.assertIsNone(empty["encoded_path"])
        self.assertEqual(empty["normalization"]["reason"], "empty_chapter")

    def test_long_silence_is_never_silently_accepted(self):
        signal(self.source, 0)
        with self.assertRaisesRegex(ValueError, "measurable loudness"):
            self.prepare()
        self.assertEqual(list((self.root / "work" / "chapters").iterdir()), [])

    def test_disabled_normalization_keeps_exact_composed_pcm(self):
        self.settings["normalize_loudness"] = False
        ready = self.prepare()
        original = assembly.compose_chapter(self.clips, self.root / "original.wav")
        self.assertEqual(file_hash(ready["wav_path"]), file_hash(original["wav_path"]))
        self.assertEqual(ready["normalization"]["status"], "disabled")

    def test_normalization_cannot_change_timing(self):
        original = chapter_cache._loudness_pass

        def alter_frames(source, audio_filter, destination, settings):
            result = original(source, audio_filter, destination, settings)
            if destination is not None:
                make_wav(destination, frames=4410)
            return result

        with patch.object(chapter_cache, "_loudness_pass", side_effect=alter_frames):
            with self.assertRaisesRegex(ValueError, "PCM timeline"):
                self.prepare()
        self.assertEqual(list((self.root / "work" / "chapters").iterdir()), [])

    def test_package_reuses_chapter_bytes_and_encodes_book_from_pcm(self):
        ready = self.prepare()
        calls = []
        run = assembly._run

        def record_command(arguments, **kwargs):
            if "-f" in arguments and "concat" in arguments:
                concat_path = Path(arguments[arguments.index("-i") + 1])
                calls.append(concat_path.read_text())
            return run(arguments, **kwargs)

        output = self.root / "book"
        with patch.object(assembly, "encode_chapter", side_effect=AssertionError("chapter reencoded")), \
             patch.object(assembly, "_run", side_effect=record_command):
            manifest = assembly.package_audio([ready], output, book=self.book)
        self.assertEqual(len(calls), 1)
        self.assertIn("chapter.wav", calls[0])
        self.assertNotIn("chapter.m4a", calls[0])
        self.assertEqual(file_hash(output / "chapters/chapter-0007.m4a"), ready["encoded_sha256"])
        self.assertEqual(manifest["chapters"][0]["ordinal"], 1)
        self.assertEqual(manifest["chapters"][0]["number"], 7)
        self.assertEqual(manifest["chapters"][0]["normalization"], ready["normalization"])

    def test_package_rejects_changed_cached_encoding(self):
        ready = self.prepare()
        ready["encoded_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "integrity"):
            assembly.package_audio([ready], self.root / "book", book=self.book)

    def test_subset_reencoding_preserves_original_track_number_and_total(self):
        self.chapter.update(_ordinal=7, _total_chapters=10)
        ready = self.prepare()
        self.assertEqual(assembly.probe_audio(ready["encoded_path"])["format"]["tags"]["track"], "7/10")
        ready["encoded_path"] = None
        ready["encoded_sha256"] = None
        output = self.root / "mp3-book"
        manifest = assembly.package_audio([ready], output, book=self.book, chapter_format="mp3")
        tags = assembly.probe_audio(output / "chapters/chapter-0007.mp3")["format"]["tags"]
        self.assertEqual(tags["track"], "7/10")
        self.assertEqual(manifest["chapters"][0]["ordinal"], 1)
        self.assertEqual(manifest["chapters"][0]["number"], 7)

    def test_cache_root_symlink_is_rejected(self):
        (self.root / "work").mkdir()
        external = self.root / "elsewhere"
        external.mkdir()
        (self.root / "work" / "chapters").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare()
        self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
