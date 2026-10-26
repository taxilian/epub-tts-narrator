"""Edition masks preserve verified takes, exact frames, and independent lanes."""

from copy import deepcopy
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.assembly import compose_chapter
from epub_tts_narrator.renderer.chapter_cache import (
    _timings, chapter_settings, load_chapter, prepare_chapter, prepare_mixture,
)
from epub_tts_narrator.renderer.censorship import DEFAULT_POLICY, plan_masks, policy_identity
from epub_tts_narrator.renderer.common import file_hash
from test_renderer_assembly import clip


RATE = 8000


def pcm(path, values):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<h", value) for value in values))
    return path


def samples(path):
    with wave.open(str(path), "rb") as stream:
        data = stream.readframes(stream.getnframes())
    return list(struct.unpack("<" + "h" * (len(data) // 2), data))


class CensorshipAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = dict(DEFAULT_POLICY, enabled=True, words=["fuck"])
        self.wav = pcm(self.root / "original.wav", [2000] * RATE)
        self.item = clip(self.wav, text="Oh fuck now", break_after="none")

    def report(self, start=.2, end=.7):
        return {"passed": True, "transcript": "Oh fuck now", "word_timing_status": "available", "words": [
            {"word": "Oh", "start": 0, "end": .2, "probability": .99},
            {"word": "fuck", "start": start, "end": end, "probability": .99},
            {"word": "now", "start": .7, "end": 1., "probability": .99},
        ]}

    def audit(self, item=None, *, report=None):
        item = item or self.item
        with wave.open(str(item["wav_path"]), "rb") as source:
            frames = source.getnframes()
        return plan_masks(item["spoken_text"], report or self.report(), self.policy,
                          sample_rate=RATE, frames=frames, audio_sha256=file_hash(item["wav_path"]))

    def compose(self, name, items=None, *, policy=None, normalize=False):
        return compose_chapter(items or [self.item], self.root / name, sample_rate=RATE,
                               normalize_pauses=normalize, paragraph_pause_ms=0,
                               censorship_policy=policy)

    def test_masked_edition_retains_word_onset_source_frames_and_untouched_original(self):
        original = self.wav.read_bytes()
        self.item["censorship"] = self.audit()
        before = self.compose("plain.wav")
        after = self.compose("muted.wav", policy=self.policy)
        self.assertEqual(before["frames"], after["frames"])
        self.assertEqual(self.wav.read_bytes(), original)
        mask = self.item["censorship"]["masks"][0]
        data = samples(after["wav_path"])
        self.assertEqual(data[:mask["mute_start_frame"]], [2000] * mask["mute_start_frame"])
        self.assertTrue(all(value == 0 for value in data[
            mask["silence_start_frame"]:mask["silence_end_frame"]]))
        self.assertEqual(data[mask["mute_end_frame"]:], [2000] * (RATE - mask["mute_end_frame"]))
        self.assertEqual(after["segments"][0]["source_ranges"], before["segments"][0]["source_ranges"])
        _timings(after)
        self.assertEqual(after["segments"][0]["censorship"]["lanes"][0]["audit"], self.item["censorship"])

    def test_missing_or_stale_mask_audit_cannot_publish_enabled_edition(self):
        with self.assertRaises(ValueError):
            self.compose("missing.wav", policy=self.policy)
        self.assertFalse((self.root / "missing.wav").exists())
        self.item["censorship"] = self.audit()
        pcm(self.wav, [2100] * RATE)
        with self.assertRaises(ValueError):
            self.compose("changed.wav", policy=self.policy)
        self.assertFalse((self.root / "changed.wav").exists())

    def test_disabled_policy_is_byte_identical_and_ignores_old_audit(self):
        plain = self.compose("plain.wav")
        self.item["censorship"] = {"invalid": "unused"}
        disabled = self.compose("disabled.wav", policy=dict(DEFAULT_POLICY, enabled=False))
        self.assertEqual(Path(plain["wav_path"]).read_bytes(), Path(disabled["wav_path"]).read_bytes())
        self.assertEqual(plain["segments"], disabled["segments"])
        self.assertEqual(chapter_settings({}), chapter_settings({}, censorship_policy=DEFAULT_POLICY))

    def test_edge_trimming_uses_original_audio_before_first_and_final_word_mask(self):
        for text, words in (("fuck now", [(.1, .6, "fuck"), (.6, .9, "now")]),
                            ("Oh fuck", [(.1, .4, "Oh"), (.4, .9, "fuck")])):
            with self.subTest(text=text):
                self.wav = pcm(self.root / "edges.wav", [0] * 800 + [2200] * 6400 + [0] * 800)
                item = clip(self.wav, text=text, break_after="none")
                report = {"passed": True, "transcript": text, "words": [
                    {"start": start, "end": end, "word": word, "probability": .99}
                    for start, end, word in words]}
                item["censorship"] = self.audit(item, report=report)
                plain = self.compose(text + "plain.wav", [item], normalize=True)
                muted = self.compose(text + "muted.wav", [item], policy=self.policy, normalize=True)
                self.assertEqual(plain["frames"], muted["frames"])
                for key in ("source_audio_start_frame", "source_audio_end_frame",
                            "chapter_start_frame", "chapter_end_frame"):
                    self.assertEqual(plain["segments"][0][key], muted["segments"][0][key])
                _timings(muted)

    def test_simultaneous_lanes_mute_independently_before_average(self):
        other = pcm(self.root / "other.wav", [4000] * RATE)
        lanes = [{"role_id": "first", "voice_id": "voice_1", "wav_path": self.wav,
                  "audio_sha256": file_hash(self.wav), "request_hash": "first", "validation_key": "first",
                  "report": self.report()},
                 {"role_id": "second", "voice_id": "voice_2", "wav_path": other,
                  "audio_sha256": file_hash(other), "request_hash": "second", "validation_key": "second",
                  "report": self.report(.4, .7)}]
        mixed = prepare_mixture(lanes, self.root / "mixes", sample_rate=RATE,
                                preserve_lanes=True, notify=lambda _: None)
        self.item["segment"]["simultaneous_speakers"] = [
            {"role_id": lane["role_id"], "voice_id": lane["voice_id"]} for lane in lanes]
        self.item.update(mixed)
        # Assembly metadata remains the original verified lane identity.
        from epub_tts_narrator.renderer.chapter_cache import LANE_FIELDS
        self.item["speaker_lanes"] = [{key: lane[key] for key in LANE_FIELDS}
                                      for lane in mixed["speaker_lanes"]]
        for lane in self.item["censorship_lanes"]:
            lane["censorship"] = self.audit(dict(self.item, wav_path=lane["wav_path"]), report=lane["report"])
        plain_hash = file_hash(mixed["wav_path"])
        result = self.compose("separate.wav", policy=self.policy)
        data = samples(result["wav_path"])
        self.assertEqual(data[0], 3000)
        # At 350 ms only lane one has reached its muted tail. Lane two's
        # unaffected voice remains audible instead of a union-of-masks mute.
        self.assertEqual(data[2800], 2000)
        self.assertEqual(file_hash(mixed["wav_path"]), plain_hash)
        self.assertEqual(result["frames"], RATE)
        _timings(result)

    def test_sync_rejects_fabricated_chapter_mask_offsets(self):
        self.item["censorship"] = self.audit()
        result = self.compose("muted.wav", policy=self.policy)
        broken = deepcopy(result)
        broken["segments"][0]["censorship"]["lanes"][0]["chapter_masks"][0]["chapter_end_frame"] += 1
        with self.assertRaisesRegex(ValueError, "frame mapping"):
            _timings(broken)

    def test_download_requires_requested_masks_bound_to_original_verified_audio(self):
        from epub_tts_narrator.renderer.chapter_client import _validate_censorship_sources, ChapterClientError
        self.item["silent_control"] = False
        self.item["censorship"] = self.audit()
        result = self.compose("muted.wav", policy=self.policy)
        spec = {"censorship_policy": self.policy, "chapter": {"chunks": [self.item]}}
        accepted = [{"chunk_id": self.item["chunk_id"], "audio_sha256": file_hash(self.wav)}]
        _validate_censorship_sources(result, spec, accepted)
        with self.assertRaisesRegex(ChapterClientError, "unrequested"):
            _validate_censorship_sources(result, {"chapter": spec["chapter"]}, accepted)
        broken = deepcopy(result)
        del broken["segments"][0]["censorship"]
        with self.assertRaisesRegex(ChapterClientError, "missing"):
            _validate_censorship_sources(broken, spec, accepted)
        with self.assertRaisesRegex(ChapterClientError, "verified audio"):
            _validate_censorship_sources(result, spec, [dict(accepted[0], audio_sha256="0" * 64)])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
    def test_policy_rebuilds_chapter_without_changing_original_or_legacy_cache(self):
        self.item["censorship"] = self.audit()
        chapter = {"id": "chapter-0001", "number": 1, "title": "Chapter 1"}
        settings = {"sample_rate": RATE, "normalize_pauses": False, "normalize_loudness": False}
        def prepare(policy=None):
            return prepare_chapter(chapter, [self.item], self.root / "chapters", book={"title": "Book"},
                                   render_settings=settings, context={}, censorship_policy=policy,
                                   notify=lambda _: None)
        before = prepare()
        original = file_hash(self.wav)
        after = prepare(self.policy)
        self.assertNotEqual(before["cache_key"], after["cache_key"])
        self.assertEqual(before["frames"], after["frames"])
        self.assertEqual(after["settings"]["censorship"], policy_identity(self.policy))
        self.assertEqual(load_chapter(after["cache_directory"]), after)
        with patch("epub_tts_narrator.renderer.chapter_cache.compose_chapter", side_effect=AssertionError("recomposed")):
            self.assertEqual(prepare(self.policy), after)
            self.assertEqual(prepare(DEFAULT_POLICY), before)
        self.assertEqual(file_hash(self.wav), original)
        from epub_tts_narrator.renderer.assembly import package_audio
        from epub_tts_narrator.renderer.common import read_json
        package_audio([before, dict(after, number=2, id="chapter-0002")], self.root / "book",
                      book={"title": "Book"})
        sync = read_json(self.root / "book" / "sync.json")
        word = sync["segments"][1]["censorship"]["lanes"][0]["chapter_masks"][0]
        self.assertEqual(word["book_start_frame"], before["frames"] + word["chapter_start_frame"])
        self.assertEqual(word["book_end_frame"], before["frames"] + word["chapter_end_frame"])


class CensorshipProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"version": 1, "book": {"title": "Book"}, "source_fingerprint": "a" * 64,
                     "renderer_fingerprint": "b" * 64, "chapter": {"id": "chapter-0001", "number": 1,
                     "title": "Chapter 1", "chunks": []}, "total_chapters": 1, "voice_bindings": {},
                     "render": {}}

    def test_remote_policy_is_optional_but_validated_when_present(self):
        from epub_tts_narrator.renderer.censorship import validate_policy
        from epub_tts_narrator.renderer.chapter_protocol import validate_spec
        self.assertEqual(validate_spec(self.spec), self.spec)
        policy = validate_policy({"enabled": True, "words": ["fuck"]})
        spec = dict(self.spec, censorship_policy=policy)
        self.assertEqual(validate_spec(spec), spec)
        for invalid in (False, {"enabled": True}, dict(policy, enabled=False), dict(policy, arbitrary="value")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_spec(dict(self.spec, censorship_policy=invalid))

    def test_absent_policy_never_inherits_worker_enabled_censorship(self):
        from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
        from epub_tts_narrator.renderer.config import DEFAULTS
        from epub_tts_narrator.renderer.censorship import validate_policy
        config = deepcopy(DEFAULTS)
        config["censorship"] = validate_policy({"enabled": True, "words": ["fuck"]})
        executor = ChapterExecutor(config)
        self.assertFalse(executor._settings(self.spec)["censorship"]["enabled"])
        spec = dict(self.spec, censorship_policy=config["censorship"])
        self.assertEqual(executor._settings(spec)["censorship"], config["censorship"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
    def test_remote_edition_reuses_verified_speech_and_publishes_checked_masks(self):
        from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
        from epub_tts_narrator.renderer.chapter_client import _validate_metadata
        from epub_tts_narrator.renderer.common import digest, read_json
        from epub_tts_narrator.renderer.config import DEFAULTS
        from epub_tts_narrator.renderer.censorship import validate_policy
        from test_renderer_chapter_pipeline import chapter_spec
        from test_renderer_workflow import FixtureEngine, FixtureVerifier, write_tone
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(DEFAULTS)
            config["render"].update(normalize_loudness=False, max_attempts=1)
            engine = FixtureEngine({})
            verifier = FixtureVerifier(engine.fixture_transcripts)
            executor = ChapterExecutor(config, engine=engine, verifier=verifier)
            self.addCleanup(executor.close)
            spec = chapter_spec(executor, 1, "Oh fuck now.", "c" * 64)
            voice = root / "reference.wav"
            write_tone(voice)
            voices = {"voice": {"asset_hash": "c" * 64, "reference_audio": str(voice),
                                "reference_text": "Reference voice sample."}}
            prepared = executor.synthesize(spec, voices, root / "work", notify=lambda _: None)
            plain = executor.assemble(spec, prepared, root / "work", notify=lambda _: None)
            updated = dict(spec, censorship_policy=validate_policy({"enabled": True, "words": ["fuck"]}))
            with patch.object(engine, "synthesize", side_effect=AssertionError("speech regenerated")), \
                 patch.object(verifier, "transcribe", side_effect=AssertionError("speech reverified")):
                censored = executor.synthesize(updated, voices, root / "work", notify=lambda _: None)
            with patch.object(executor, "_components", side_effect=AssertionError("model access during assembly")):
                artifacts = executor.assemble(updated, censored, root / "work", notify=lambda _: None)
            plain_meta, muted_meta = read_json(plain["chapter.json"]), read_json(artifacts["chapter.json"])
            self.assertEqual(plain_meta["frames"], muted_meta["frames"])
            self.assertNotEqual(plain_meta["pcm_sha256"], muted_meta["pcm_sha256"])
            self.assertEqual(prepared["acceptance"], censored["acceptance"])
            self.assertTrue(muted_meta["segments"][0]["censorship"]["lanes"][0]["audit"]["masks"])
            self.assertEqual(_validate_metadata(artifacts["chapter.json"].parent,
                                               {"spec": updated, "selection": updated}), muted_meta)
