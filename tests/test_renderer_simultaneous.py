"""Explicit unison is verified per performer, then mixed without changing time."""

from copy import deepcopy
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from audition_fixtures import prepare_bundle
from unittest.mock import patch
import wave

from audiobook_renderer.assembly import mix_speakers
from audiobook_renderer.bundle import chunk_speakers, load_bundle, make_plan
from audiobook_renderer.chapter_cache import prepare_mixture
from audiobook_renderer.common import digest, file_hash, read_json, write_json
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.workflow import RenderBlocked, assemble, render
from epub_audiobook.cli import export_bundle
from epub_audiobook.pipeline import segment
from test_renderer_workflow import EXAMPLE, FixtureEngine, FixtureVerifier, pcm_key, timed_transcript


def pcm(path, samples):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return path


def samples(path):
    with wave.open(str(path), "rb") as audio:
        raw = audio.readframes(audio.getnframes())
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


class SimultaneousMixerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def lanes(self):
        paths = [pcm(self.root / "a.wav", [30000, 30000, 10000, 10000]),
                 pcm(self.root / "b.wav", [30000, 10000])]
        return [{"role_id": f"role{index}", "voice_id": f"voice{index}", "wav_path": str(path),
                 "audio_sha256": file_hash(path), "request_hash": digest({"request": index}),
                 "validation_key": digest({"validation": index}), "report": {"passed": True}}
                for index, path in enumerate(paths)]

    def test_fixed_average_aligns_starts_pads_ends_and_prevents_clipping(self):
        lanes = self.lanes()
        before = [file_hash(lane["wav_path"]) for lane in lanes]
        output = self.root / "mixed.wav"
        result = mix_speakers(lanes, output)
        self.assertEqual(samples(output), [30000, 20000, 5000, 5000])
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["weight"], .5)
        self.assertEqual(result["peak_amplitude"], 30000)
        self.assertEqual(before, [file_hash(lane["wav_path"]) for lane in lanes])

    def test_only_verified_lanes_can_enter_mix_cache_and_cached_bytes_are_checked(self):
        lanes = self.lanes()
        bad = deepcopy(lanes)
        bad[1]["report"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "Every simultaneous speaker"):
            prepare_mixture(bad, self.root / "mixes", notify=lambda _: None)
        self.assertFalse((self.root / "mixes").exists())
        mixed = prepare_mixture(lanes, self.root / "mixes", notify=lambda _: None)
        self.assertEqual([lane["end_frame"] for lane in mixed["speaker_lanes"]], [4, 2])
        with patch("audiobook_renderer.chapter_cache.mix_speakers", side_effect=AssertionError("Remixed")):
            self.assertEqual(prepare_mixture(lanes, self.root / "mixes", notify=lambda _: None), mixed)
        path = Path(mixed["wav_path"])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "integrity"):
            prepare_mixture(lanes, self.root / "mixes", notify=lambda _: None)

    def test_truncated_lane_and_cancelling_audio_never_publish(self):
        lanes = self.lanes()
        short = Path(lanes[1]["wav_path"])
        short.write_bytes(short.read_bytes()[:-2])
        lanes[1]["audio_sha256"] = file_hash(short)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            mix_speakers(lanes, self.root / "bad.wav")
        self.assertFalse((self.root / "bad.wav").exists())
        pcm(Path(lanes[0]["wav_path"]), [10000] * 4)
        pcm(short, [-10000] * 4)
        for lane in lanes:
            lane["audio_sha256"] = file_hash(lane["wav_path"])
        with self.assertRaisesRegex(ValueError, "silent"):
            mix_speakers(lanes, self.root / "silent.wav")
        self.assertFalse((self.root / "silent.wav").exists())

    def test_audio_changed_between_verification_and_mixing_never_publishes(self):
        lanes = self.lanes()

        def changed_before_mix(accepted, target, **kwargs):
            pcm(Path(accepted[0]["wav_path"]), [1000] * 4)
            return mix_speakers(accepted, target, **kwargs)

        with patch("audiobook_renderer.chapter_cache.mix_speakers", side_effect=changed_before_mix):
            with self.assertRaisesRegex(ValueError, "changed before mixing"):
                prepare_mixture(lanes, self.root / "mixes", notify=lambda _: None)
        self.assertEqual(list((self.root / "mixes").iterdir()), [])

    def test_mixture_interruption_keeps_inputs_and_publishes_nothing(self):
        lanes = self.lanes()
        original = [file_hash(lane["wav_path"]) for lane in lanes]
        with patch("audiobook_renderer.chapter_cache.mix_speakers", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                prepare_mixture(lanes, self.root / "mixes", notify=lambda _: None)
        self.assertEqual(list((self.root / "mixes").iterdir()), [])
        self.assertEqual(original, [file_hash(lane["wav_path"]) for lane in lanes])


class LaneEngine(FixtureEngine):
    def __init__(self):
        super().__init__({})
        self.performers = {}

    def synthesize(self, text, voice, path, *, seed, direction=""):
        result = super().synthesize(text, voice, path, seed=seed, direction=direction)
        longer = voice["voice_id"] == "voice_lena"
        frames = 66150 if longer else 44100
        frequency = 330 if longer else 220
        pcm(path, [round(5000 * math.sin(2 * math.pi * frequency * index / 44100))
                   for index in range(frames)])
        key = pcm_key(path)
        self.fixture_transcripts[key] = text
        self.performers[key] = voice["voice_id"]
        return result


class LaneVerifier(FixtureVerifier):
    def __init__(self, engine):
        super().__init__(engine.fixture_transcripts)
        self.engine = engine
        self.failed_voice = None

    def transcribe(self, path):
        self.calls.append(str(path))
        key = pcm_key(path)
        text = self.fixture_transcripts[key]
        if self.engine.performers.get(key) == self.failed_voice and self.failed_voice is not None:
            text = "A completely wrong phrase was spoken."
        transcript = timed_transcript(text)
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        for segment in transcript["segments"]:
            segment["end"] *= duration
            for word in segment["words"]:
                word["start"] *= duration
                word["end"] *= duration
        return transcript


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class SimultaneousWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        document = read_json(EXAMPLE / "reviewed.json")
        self.group_text = "“We are ready. Let us begin,”"
        tag = " Mara and Lena said together."
        block_id = "b000007"
        document["source"]["chapters"].append({"id": "ch0002", "title": "Unison Pilot",
            "source_href": "unison.xhtml", "block_ids": [block_id]})
        document["source"]["blocks"].append({"id": block_id, "chapter_id": "ch0002",
            "text": self.group_text + tag, "source": {"href": "unison.xhtml", "element_id": None, "tag": "p"}})
        first = segment(document["source"], block_id, 0, self.group_text, None, kind="dialogue",
                        basis="explicit", reason="Mara and Lena explicitly speak together.",
                        evidence=[{"block_id": block_id, "quote": "Mara and Lena said together"}],
                        candidates=[], review_required=False)
        first["simultaneous_speakers"] = [
            {"role_id": "mara_dialogue", "voice_id": "voice_mara"},
            {"role_id": "lena_dialogue", "voice_id": "voice_lena"},
        ]
        narrator = next(role for role in document["cast"]["roles"] if role["id"] == "narrator_main")
        document["segments"].extend([first, segment(document["source"], block_id,
            len(self.group_text), tag, narrator, reason="Narrative speech tag.")])
        export_bundle(document, self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="unison-fixture")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                         max_attempts=2, chapter_numbers=[2])
        self.config["quality"] = validate_policy({})
        self.engine = LaneEngine()
        self.verifier = LaneVerifier(self.engine)

    def run_render(self, *, chapters_only=False):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      chapters_only=chapters_only, notify=lambda _: None)

    def test_planning_collects_each_voice_and_exact_same_source_text(self):
        plan = make_plan(self.bundle, self.config)
        self.assertEqual(plan["errors"], [])
        self.assertEqual(set(plan["voice_ids"]), {"voice_mara", "voice_lena"})
        group = plan["chapters"][0]["chunks"][0]
        self.assertEqual([speaker["voice_id"] for speaker in chunk_speakers(group)], ["voice_mara", "voice_lena"])
        self.assertEqual(group["spoken_text"], self.group_text)

    def test_explicit_voice_alias_keeps_both_performers_and_reuses_reference(self):
        write_json(self.root / "bundle" / "voice-aliases.json", {
            "format": "audiobook-book-voice-aliases", "version": 1,
            "series_id": "unison-fixture", "book_id": self.bundle["source"]["book"]["id"],
            "aliases": {"voice_lena": "voice_mara"},
        })
        self.run_render(chapters_only=True)
        first, second = self.engine.speech_calls[:2]
        self.assertEqual(first["voice_hash"], second["voice_hash"])
        record = read_json(self.root / "work" / "completed-chapters" / "chapter-0002.json")
        lanes = record["acceptance"][0]["speaker_lanes"]
        self.assertEqual([lane["voice_id"] for lane in lanes], ["voice_mara", "voice_lena"])
        self.assertNotEqual(lanes[0]["request_hash"], lanes[1]["request_hash"])
        assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertTrue((self.root / "output" / "book.m4b").is_file())

    def test_analysis_update_lock_blocks_render_and_assemble_before_audio(self):
        lock = self.root / "bundle" / ".analysis-update.lock"
        lock.mkdir()
        with self.assertRaisesRegex(RenderBlocked, "analysis is being updated"):
            self.run_render(chapters_only=True)
        with self.assertRaisesRegex(RenderBlocked, "analysis is being updated"):
            assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertEqual(self.engine.speech_calls, [])
        self.assertFalse((self.root / "work").exists())
        self.assertTrue(lock.exists())

    def test_book_lock_covers_custom_work_directory_and_racing_revision(self):
        from audiobook_renderer.workflow import _lock_book_audio
        book_lock = self.root / "bundle" / ".audio-render.lock"
        revision_lock = self.root / "bundle" / ".analysis-update.lock"

        def begin_revision_after_book_lock(bundle):
            self.assertTrue(book_lock.is_dir())
            revision_lock.mkdir()
            raise RenderBlocked("analysis is being updated")

        with patch("audiobook_renderer.workflow._check_analysis_update", side_effect=begin_revision_after_book_lock):
            with self.assertRaisesRegex(RenderBlocked, "analysis is being updated"):
                _lock_book_audio(self.bundle)
        self.assertFalse(book_lock.exists())
        revision_lock.rmdir()
        book_lock.mkdir()
        with self.assertRaisesRegex(RenderBlocked, "rendering or assembling this book"):
            self.run_render(chapters_only=True)
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.assertTrue(book_lock.exists())
        self.assertEqual(self.engine.speech_calls, [])
        book_lock.rmdir()
        self.run_render(chapters_only=True)
        self.assertFalse(book_lock.exists())

    def test_completed_revision_cannot_leave_a_stale_loaded_render_plan(self):
        manifest = self.root / "bundle" / "manifest.json"
        current = read_json(manifest)
        current["revision_note"] = "Revised after the renderer loaded its source"
        write_json(manifest, current)
        with self.assertRaisesRegex(RenderBlocked, "analysis changed after it was loaded"):
            self.run_render(chapters_only=True)
        self.assertEqual(self.engine.speech_calls, [])
        self.assertFalse((self.root / "bundle" / ".audio-render.lock").exists())
        self.assertFalse((self.root / "work" / ".render.lock").exists())

    def test_each_performer_is_verified_and_source_sync_is_shared_once(self):
        self.run_render()
        first, second = self.engine.speech_calls[:2]
        self.assertEqual(first["text"], second["text"])
        self.assertNotEqual(first["voice_hash"], second["voice_hash"])
        output = self.root / "output"
        acceptance = read_json(output / "verification.json")[0]
        self.assertEqual(acceptance["report"]["method"], "independently_verified_simultaneous_lanes")
        self.assertEqual([lane["end_frame"] for lane in acceptance["speaker_lanes"]], [44100, 66150])
        self.assertTrue(all(lane["report"]["passed"] for lane in acceptance["speaker_lanes"]))
        sync = read_json(output / "sync.json")
        group = sync["segments"][0]
        self.assertEqual(group["chapter_end_frame"] - group["chapter_start_frame"], 66150)
        self.assertEqual(group["source_text"], first["text"])
        self.assertEqual([lane["book_end_frame"] for lane in group["speaker_lanes"]], [44100, 66150])
        self.assertEqual(group["book_end_frame"], 66150)
        self.assertEqual([lane["voice_id"] for lane in group["speaker_lanes"]], ["voice_mara", "voice_lena"])
        self.assertEqual(sum(item["chunk_id"] == group["chunk_id"] for item in sync["segments"]), 1)
        self.assertFalse(any("mixes" in path for path in self.verifier.calls))
        self.assertTrue((output / "book.m4b").is_file())

    def test_failed_second_lane_cannot_publish_a_mix_chapter_or_book(self):
        self.verifier.failed_voice = "voice_lena"
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.run_render()
        work = self.root / "work"
        self.assertFalse((work / "mixes").exists())
        self.assertFalse((work / "completed-chapters").exists())
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(len(list((work / "clips").glob("*/accepted.json"))), 1)

    def test_interrupted_second_lane_reuses_first_verified_lane(self):
        self.engine.interrupt_speech_number = 2
        with self.assertRaises(KeyboardInterrupt):
            self.run_render(chapters_only=True)
        first_request = list((self.root / "work" / "clips").glob("*/accepted.json"))[0]
        first_bytes = first_request.read_bytes()
        self.engine.interrupt_speech_number = None
        self.run_render(chapters_only=True)
        self.assertEqual(first_request.read_bytes(), first_bytes)
        first_text = self.group_text
        first_voice = self.engine.speech_calls[0]["voice_hash"]
        self.assertEqual(sum(row["text"] == first_text and row["voice_hash"] == first_voice
                             for row in self.engine.speech_calls), 1)
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("ASR or TTS called")):
            self.run_render(chapters_only=True)
            assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertTrue((self.root / "output" / "book.m4b").is_file())

    def test_changed_lane_verification_cannot_pass_restoration(self):
        self.run_render(chapters_only=True)
        path = self.root / "work" / "completed-chapters" / "chapter-0002.json"
        record = read_json(path)
        record.pop("record_sha256")
        record["acceptance"][0]["speaker_lanes"][1]["report"]["passed"] = False
        write_json(path, dict(record, record_sha256=digest(record)))
        with self.assertRaisesRegex(RenderBlocked, "simultaneous speaker lacks matching verification"):
            assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
