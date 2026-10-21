"""Workflow acceptance tests with actual PCM/FFmpeg and test-only model doubles."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.assembly import probe_audio
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import RenderBlocked, render, verified_take
from epub_tts_narrator.cli import export_bundle
from audition_fixtures import prepare_bundle


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def pcm_key(path):
    with wave.open(str(path), "rb") as handle:
        return hashlib.sha256(handle.readframes(handle.getnframes())).hexdigest()


def write_tone(path, frequency=220):
    """A waveform fixture, not synthetic speech or a production TTS fallback."""
    count, rate = 44100, 44100
    samples = [round(5000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(count)]
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(struct.pack("<" + "h" * count, *samples))


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


class FixtureEngine:
    """Registers PCM fixtures with the test verifier; no real model is loaded."""

    identity = {"engine": "fixture", "model": "fixture-model-v1", "settings": {"version": 1}}
    capabilities = {"reference_voice": True, "voice_design": True}

    def __init__(self, fixture_transcripts):
        self.fixture_transcripts = fixture_transcripts
        self.voice_calls = []
        self.speech_calls = []
        self.interrupt_speech_number = None

    def _write(self, text, path, seed):
        frequency = 150 + int(digest({"text": text, "seed": seed})[:10], 16) % 4000
        write_tone(path, frequency)
        self.fixture_transcripts[pcm_key(path)] = text
        return {"fixture": True, "seed": seed}

    def create_voice(self, description, text, path, *, seed):
        self.voice_calls.append({"description": description, "text": text, "seed": seed})
        return dict(self._write(text, path, seed), description_applied=True)

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.speech_calls.append({"text": text, "voice_hash": voice["asset_hash"],
                                  "seed": seed, "direction": direction})
        if self.interrupt_speech_number == len(self.speech_calls):
            raise KeyboardInterrupt("test interruption")
        return self._write(text, path, seed)


class FixtureVerifier:
    """Looks up a waveform fixture; the production verifier has no such table."""

    def __init__(self, fixture_transcripts, version=1):
        self.fixture_transcripts = fixture_transcripts
        self.identity = {"backend": "fixture-verifier", "version": version}
        self.calls = []
        self.wrong_texts = set()

    def transcribe(self, path):
        self.calls.append(str(path))
        text = self.fixture_transcripts[pcm_key(path)]
        if text in self.wrong_texts:
            text = "Unrelated invented words were inserted."
        return timed_transcript(text)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for workflow integration")
class RendererWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts-model")
        self.config["verification"]["model"] = str(self.root / "unused-asr-model")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="fixture-series")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
        )
        self.config["quality"] = validate_policy({})
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)
        self.bundle = prepare_bundle(load_bundle(EXAMPLE / "bundle"))

    def take(self, *, verifier=None, directory=None, request=None, text="Hello there."):
        request = request or {"kind": "fixture", "text": text, "engine": self.engine.identity}
        return verified_take(
            text, lambda path, seed: self.engine._write(text, path, seed),
            directory or self.root / "takes", request, verifier or self.verifier,
            self.config, notify=lambda _: None,
        )

    def run_render(self, bundle=None, config=None):
        return render(bundle or self.bundle, config or self.config, self.engine,
                      self.verifier, notify=lambda _: None)

    def test_accepted_take_resumes_without_generation_or_reverification(self):
        wav, first = self.take()
        calls = len(self.verifier.calls)
        fixture_count = len(self.fixtures)
        resumed_wav, resumed = self.take()
        self.assertEqual(wav, resumed_wav)
        self.assertEqual(first, resumed)
        self.assertEqual(len(self.verifier.calls), calls)
        self.assertEqual(len(self.fixtures), fixture_count)
        self.assertEqual(file_hash(wav), first["audio_sha256"])

    def test_changed_verifier_reuses_accepted_audio_without_reverification(self):
        wav, first = self.take()
        verifier = FixtureVerifier(self.fixtures, version=2)
        verifier.wrong_texts.add("Hello there.")
        resumed_wav, resumed = self.take(verifier=verifier)
        self.assertEqual(wav, resumed_wav)
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(first, resumed)
        self.assertEqual(first["audio_sha256"], resumed["audio_sha256"])
        self.assertEqual(len(list((self.root / "takes").glob("attempt-*-raw.wav"))), 1)

    def test_changed_quality_keeps_original_successful_acceptance(self):
        wav, first = self.take()
        self.config["quality"]["min_duration_seconds"] = 2.0
        resumed_wav, resumed = self.take()
        self.assertEqual(resumed_wav, wav)
        self.assertEqual(resumed, first)
        self.assertEqual(len(self.verifier.calls), 1)

    def test_changed_verification_key_does_not_invalidate_accepted_take(self):
        wav, first = self.take()
        with patch("epub_tts_narrator.renderer.workflow.make_validation_key", side_effect=AssertionError("rechecked policy")):
            self.assertEqual((wav, first), self.take())

    def test_missing_accepted_audio_regenerates_and_verifies_only_that_take(self):
        wav, first = self.take()
        for _ in range(2):
            wav.unlink()
            with patch.object(self.engine, "_write", wraps=self.engine._write) as generate:
                replacement, accepted = self.take()
            self.assertEqual(replacement, wav)
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.verifier.calls), 3)
        self.assertEqual(len(list((self.root / "takes").glob("attempt-001-regenerated-*.wav"))), 2)

    def test_changed_policy_rechecks_unsuccessful_saved_audio_without_new_synthesis(self):
        self.verifier.wrong_texts.add("Hello there.")
        with self.assertRaises(RenderBlocked):
            self.take()
        verifier = FixtureVerifier(self.fixtures, version=2)
        with patch.object(self.engine, "_write", side_effect=AssertionError("regenerated saved attempt")):
            wav, accepted = self.take(verifier=verifier)
        self.assertEqual(wav.name, "attempt-001.wav")
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(verifier.calls), 1)

    def test_corrupt_accepted_audio_is_blocked_before_asr(self):
        wav, _ = self.take()
        before = len(self.verifier.calls)
        data = bytearray(wav.read_bytes())
        data[-1] ^= 1
        wav.write_bytes(data)
        with self.assertRaisesRegex(RenderBlocked, "Cached audio changed"):
            self.take()
        self.assertEqual(len(self.verifier.calls), before)

    def test_request_mismatch_cannot_reuse_accepted_audio(self):
        self.take()
        with self.assertRaisesRegex(RenderBlocked, "Cached audio changed"):
            self.take(request={"kind": "different"})

    def test_failed_speech_stops_after_bounded_attempts_and_never_publishes(self):
        first_text = self.bundle["chapters"][0]["rows"][0]["text"]
        self.verifier.wrong_texts.add(first_text)
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.run_render()
        self.assertEqual(len(self.engine.speech_calls), 2)
        self.assertFalse(Path(self.config["render"]["output_dir"]).exists())
        work = Path(self.config["render"]["work_dir"])
        self.assertEqual(read_json(work / "status.json")["status"], "blocked")
        self.assertFalse((work / ".render.lock").exists())
        attempts = list((work / "clips").glob("*/attempt-*.json"))
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(read_json(path)["report"]["passed"] is False for path in attempts))
        self.assertFalse(list((work / "clips").glob("*/accepted.json")))
        # Each command gets a bounded fresh retry budget; failed takes stay saved.
        original_attempts = {path: path.read_bytes() for path in attempts}
        with self.assertRaises(RenderBlocked):
            self.run_render()
        self.assertEqual(len(self.engine.speech_calls), 4)
        self.assertEqual(len(list((work / "clips").glob("*/attempt-*.json"))), 4)
        self.assertEqual(len({call["seed"] for call in self.engine.speech_calls}), 4)
        for path, contents in original_attempts.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_interrupted_render_resumes_accepted_chunks_and_voice_assets(self):
        self.engine.interrupt_speech_number = 3
        with self.assertRaises(KeyboardInterrupt):
            self.run_render()
        self.assertFalse(Path(self.config["render"]["output_dir"]).exists())
        voice_calls = len(self.engine.voice_calls)
        work = Path(self.config["render"]["work_dir"])
        self.assertEqual(len(list((work / "clips").glob("*/accepted.json"))), 2)
        self.engine.interrupt_speech_number = None
        result = self.run_render()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.engine.voice_calls), voice_calls)
        # Two accepted chunks were reused; only the interrupted call is extra.
        expected_speech_chunks = sum(
            any(c.isalnum() for c in row.get("spoken_text", row["text"]))
            for chapter in self.bundle["chapters"] for row in chapter["rows"]
        )
        self.assertEqual(len(self.engine.speech_calls), expected_speech_chunks + 1)

    def test_two_books_reuse_voices_and_publish_source_bound_sync_and_audio(self):
        first = self.run_render()
        output = Path(self.config["render"]["output_dir"])
        voice_calls = len(self.engine.voice_calls)
        self.assertEqual(voice_calls, 2)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(read_json(output / "source.json"), self.bundle["source"])
        self.assertEqual(read_json(output / "cast.json"), self.bundle["cast"])
        for name, checksum in first["files_sha256"].items():
            self.assertEqual(file_hash(output / name), checksum)
        self.assertFalse((output / "pcm").exists())
        self.assertTrue((output / "chapters" / "chapter-0001.m4a").is_file())
        inspection = probe_audio(output / "book.m4b", ffprobe=FFPROBE)
        self.assertEqual(len(inspection["chapters"]), 1)
        self.assertEqual(inspection["chapters"][0]["tags"]["title"],
                         self.bundle["chapters"][0]["entry"]["title"])
        sync = read_json(output / "sync.json")
        self.assertEqual(sync["source_fingerprint"], self.bundle["manifest"]["source_fingerprint"])
        self.assertEqual(sync["book_id"], self.bundle["source"]["book"]["id"])
        self.assertEqual(sync["audio_sha256"], file_hash(output / "book.m4b"))
        rows = {row["id"]: row for chapter in self.bundle["chapters"] for row in chapter["rows"]}
        self.assertEqual(len(sync["segments"]), len(rows))
        previous_end = 0
        for segment in sync["segments"]:
            source = rows[segment["segment_id"]]
            self.assertEqual(segment["source_text"], source["text"])
            self.assertEqual(segment["source_start"], source["start"])
            self.assertEqual(segment["source_end"], source["end"])
            self.assertGreaterEqual(segment["book_start_frame"], previous_end)
            self.assertGreater(segment["book_end_frame"], segment["book_start_frame"])
            previous_end = segment["book_end_frame"]
        verification = read_json(output / "verification.json")
        self.assertTrue(all(item["report"]["passed"] for item in verification))
        # Export another valid book with the same cast/series voice IDs.
        second_document = read_json(EXAMPLE / "reviewed.json")
        second_document["source"]["book"]["id"] = hashlib.sha256(b"second fixture book").hexdigest()
        second_document["source"]["book"]["title"] = "The Next Lantern"
        second_bundle_dir = self.root / "second-bundle"
        export_bundle(second_document, second_bundle_dir)
        second_bundle = load_bundle(second_bundle_dir)
        second_config = deepcopy(self.config)
        second_config["render"]["output_dir"] = str(self.root / "second-output")
        second_config["render"]["work_dir"] = str(self.root / "second-work")
        second = self.run_render(second_bundle, second_config)
        self.assertEqual(len(self.engine.voice_calls), voice_calls)
        self.assertEqual(first["voice_assets"], second["voice_assets"])
        self.assertNotEqual(first["source_fingerprint"], second["source_fingerprint"])

    def test_persisted_voice_audio_cannot_change_or_be_implicitly_replaced(self):
        self.run_render()
        store = VoiceStore(self.config["voices"]["directory"], self.config["voices"]["series_id"])
        voice_id = self.bundle["cast"]["voices"][0]["id"]
        binding = store.load(voice_id)
        self.assertEqual(binding["asset_hash"], store.load(voice_id)["asset_hash"])
        path = Path(binding["reference_audio"])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "reference changed"):
            store.load(voice_id)
        with self.assertRaisesRegex(ValueError, "already exists|reference changed"):
            store.publish(voice_id, binding["profile"], path, binding["reference_text"],
                          self.engine.identity, {"passed": True}, {})

    def test_series_namespace_separates_equal_voice_ids(self):
        original = VoiceStore(self.config["voices"]["directory"], "series-one")
        other = VoiceStore(self.config["voices"]["directory"], "series-two")
        self.assertNotEqual(original.folder("same-voice"), other.folder("same-voice"))
        wav = self.root / "voice.wav"
        write_tone(wav)
        original.publish("same-voice", {"description": "fixture"}, wav, "Hello.",
                         self.engine.identity, {"passed": True}, {"description_applied": True})
        self.assertIsNotNone(original.load("same-voice"))
        self.assertIsNone(other.load("same-voice"))

    def test_voice_transcript_metadata_is_immutable_and_failed_voice_cannot_publish(self):
        store = VoiceStore(self.config["voices"]["directory"], "fixture-series")
        wav = self.root / "voice.wav"
        write_tone(wav)
        with self.assertRaisesRegex(ValueError, "Only a verified"):
            store.publish("voice", {"description": "fixture"}, wav, "Hello.",
                          self.engine.identity, {"passed": False}, {})
        self.assertIsNone(store.load("voice"))
        store.publish("voice", {"description": "fixture"}, wav, "Hello.",
                      self.engine.identity, {"passed": True}, {})
        metadata_path = store.folder("voice") / "voice.json"
        record = read_json(metadata_path)
        record["reference_text"] = "Unexpected transcript change."
        metadata_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            store.load("voice")

    def test_model_and_bundle_descendants_are_not_writable_run_locations(self):
        for protected in (
            self.bundle["directory"], self.config["engine"]["checkpoint"],
            self.config["verification"]["model"],
        ):
            for section, key in (("render", "work_dir"), ("render", "output_dir"), ("voices", "directory")):
                with self.subTest(protected=protected, field=key):
                    config = deepcopy(self.config)
                    config[section][key] = str(Path(protected) / "forbidden-render-output")
                    with self.assertRaisesRegex(RenderBlocked, "must not overlap"):
                        self.run_render(config=config)
        self.assertEqual(self.engine.voice_calls, [])
        self.assertEqual(self.engine.speech_calls, [])


if __name__ == "__main__":
    unittest.main()
