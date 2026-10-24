"""Offline diagnostics explain cache misses and never change saved audio."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cache_report import cache_report
from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.workflow import make_render_plan, _speech_request


class CacheReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        example = Path(__file__).resolve().parents[1] / "examples/last-lantern/bundle"
        shutil.copytree(example, self.root / "book")
        self.bundle = load_bundle(self.root / "book")
        self.bundle["chapters"] = self.bundle["chapters"][:1]
        self.bundle["chapters"][0]["rows"] = self.bundle["chapters"][0]["rows"][:1]
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))
        self.engine = {"engine": "drama3_mlx", "model": "fishaudio/drama-3",
                       "checkpoint_files_sha256": {"weights": "original"},
                       "installed_version": "0.1.1", "runtime": "original"}
        self.store = voice_store(self.bundle, self.config)
        self.plan = make_render_plan(self.bundle, self.config)
        self.chunk = self.plan["chapters"][0]["chunks"][0]
        self.voice_id = self.chunk["segment"]["voice_id"]
        sample = self.root / "sample.wav"
        self.wave(sample)
        self.voice = self.store.publish(self.voice_id, {}, sample, "Reference words.",
                                        self.engine, {"passed": True}, {})
        self.request = _speech_request(self.chunk, self.voice, self.engine, self.config)

    @staticmethod
    def wave(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            output.writeframes(b"\x01\x00" * 100)

    def save(self, request=None, *, indexed=False, metadata=None, passed=True):
        request = deepcopy(request or self.request)
        key = digest(request)
        audio = Path(self.config["render"]["work_dir"]) / "clips" / key / "attempt-001.wav"
        self.wave(audio)
        record = {"request": request, "request_hash": key,
                  "audio_file": audio.name, "audio_sha256": file_hash(audio),
                  "validation_key": digest("successful verification"),
                  "engine_metadata": metadata or {}, "report": {"passed": passed}}
        write_json(audio.parent / "accepted.json", record)
        if indexed:
            cache = SpeechCache(self.config["render"]["work_dir"], {self.voice_id: self.voice})
            cache.remember(cache.identity(request), record)
        return audio, record

    def report(self, engine=None):
        return cache_report(self.bundle, self.config, engine or self.engine)

    def snapshot(self):
        return {str(path.relative_to(self.root)): (file_hash(path), path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def test_runtime_and_position_changes_reuse_saved_audio_without_writes(self):
        saved = deepcopy(self.request)
        saved["chunk"]["segment"]["sequence"] += 100
        saved["chunk"]["chunk_id"] = "previous-position"
        self.save(saved)
        before = self.snapshot()

        report = self.report(dict(self.engine, runtime="vendored", installed_version="0.1.1+epub.1"))

        self.assertEqual(report["summary"]["hits"], 1)
        self.assertEqual(report["summary"]["misses"], 0)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((Path(self.config["render"]["work_dir"]) / "speech-cache").exists())

    def test_current_speech_traits_match_verified_audio_in_read_only_report(self):
        from epub_tts_narrator.renderer.engines import _directed_text
        from epub_tts_narrator.renderer.speech_text import trim_speech_text

        traits = "Neutral General American accent; clipped phrasing"
        profile = next(voice for voice in self.bundle["cast"]["voices"]
                       if voice["id"] == self.voice_id)
        profile["speechTraits"] = traits
        request = deepcopy(self.request)
        request["chunk"]["speech_traits"] = {self.voice_id: traits}
        text, delivery = _directed_text(trim_speech_text(request["chunk"]["spoken_text"]), traits)
        self.save(request, metadata={"synthesis_text": text, **delivery})
        before = self.snapshot()

        report = self.report()

        self.assertEqual(report["summary"]["hits"], 1)
        self.assertEqual(report["summary"]["misses"], 0)
        self.assertEqual(self.snapshot(), before)

    def test_changed_weights_explain_miss_without_exposing_raw_engine_metadata(self):
        _, record = self.save()
        current = dict(self.engine, checkpoint_files_sha256={"weights": "changed"},
                       checkpoint="/private/irrelevant", api_key="never-export-this")
        report = self.report(current)

        self.assertEqual(report["summary"]["misses"], 1)
        self.assertIn("engine.checkpoint_files_sha256.weights", report["reasons"])
        self.assertEqual(report["examples"][0]["saved_request_hash"], record["request_hash"])
        self.assertNotIn("never-export-this", str(report))
        self.assertNotIn("/private/irrelevant", str(report))

    def test_alias_mismatch_does_not_invent_sampling_changes(self):
        saved = deepcopy(self.request)
        saved["engine"].update(engine="drama3_mlx_audio",
            model="mlx-community/fish-audio-drama-3-4bit", temperature=1.0,
            top_p=.9, top_k=30, max_new_tokens=2048)
        self.save(saved)
        self.config["engine"]["cache_model_alias"] = "fishaudio/drama-3"

        report = self.report()

        self.assertEqual(report["reasons"], {"engine.cache_model_alias": 1})

    def test_alias_reports_only_sampling_values_that_actually_changed(self):
        saved = deepcopy(self.request)
        saved["engine"].update(temperature=.75, top_p=.9, top_k=30, max_new_tokens=2048)
        self.save(saved)
        self.config["engine"]["cache_model_alias"] = "fishaudio/drama-3"

        report = self.report()

        self.assertEqual(report["reasons"], {"engine.sampling.temperature": 1})

    def test_changed_reference_is_explained_by_original_index(self):
        self.save(indexed=True)
        # A legitimate new selection changes only the conditioning transcript.
        sample = self.root / "new-sample.wav"
        self.wave(sample)
        replacement = self.store.publish_candidate(self.voice_id, {}, sample,
            "A different reference transcript.", self.engine, {"passed": True}, {}, candidate_id="a" * 32)
        self.store.select_candidate(self.voice_id, replacement["candidate_id"])
        report = self.report()

        self.assertEqual(report["summary"]["hits"], 0)
        self.assertIn("voice.reference_text", report["reasons"])
        self.assertNotIn("A different reference transcript", str(report))

    def test_missing_audio_and_unresolvable_legacy_voice_are_explicit(self):
        audio, _ = self.save()
        audio.unlink()
        self.assertIn("missing_audio", self.report()["reasons"])
        audio.parent.joinpath("accepted.json").unlink()
        old = deepcopy(self.request)
        old["voice_asset_hash"] = digest("unavailable historic voice")
        self.save(old)
        report = self.report()

        self.assertIn("unresolved_saved_voice", report["reasons"])
        self.assertEqual(report["summary"]["unresolved_saved_voice"], 1)

    def test_recorded_input_changed_is_not_silently_reported_as_no_cache(self):
        self.save(metadata={"synthesis_text": "Previously the wrong input was sent."})
        report = self.report()

        self.assertIn("recorded_synthesis_text_changed", report["reasons"])
        self.assertEqual(report["summary"]["hits"], 0)

    def test_grouping_change_is_distinguished_from_missing_audio(self):
        grouped = deepcopy(self.request)
        grouped["kind"] = "paragraph_speech"
        grouped.pop("voice_asset_hash")
        grouped["voice_assets"] = {self.voice_id: self.voice["asset_hash"]}
        turn = deepcopy(grouped["chunk"])
        grouped["chunk"]["paragraph_turns"] = [turn]
        self.save(grouped)
        report = self.report()

        self.assertIn("request_kind_or_grouping", report["reasons"])
        self.assertNotIn("missing_audio", report["reasons"])

    def test_failed_record_cannot_be_counted_as_available_audio(self):
        self.save(passed=False)
        report = self.report()

        self.assertEqual(report["summary"]["accepted_records"], 0)
        self.assertEqual(report["summary"]["hits"], 0)
        self.assertEqual(report["summary"]["rejected_records"], 1)

    def test_diagnostic_prefers_same_speech_over_neighboring_text_with_fewer_differences(self):
        correct = deepcopy(self.request)
        correct["engine"].update(model="previous/model", checkpoint_files_sha256={"weights": "previous"})
        _, expected = self.save(correct)
        nearby = deepcopy(self.request)
        nearby["chunk"]["spoken_text"] = "A neighboring sentence, not this passage."
        nearby["chunk"]["source_ranges"] = [{"start": 1000, "end": 1040}]
        self.save(nearby)

        report = self.report()

        self.assertEqual(report["examples"][0]["saved_request_hash"], expected["request_hash"])
        self.assertIn("engine.model", report["reasons"])
        self.assertNotIn("text_or_delivery", report["reasons"])

    def test_missing_current_voice_and_limited_examples_do_not_create_directories(self):
        shutil.rmtree(self.store.folder(self.voice_id))
        before = self.snapshot()
        report = cache_report(self.bundle, self.config, self.engine, max_examples=0)

        self.assertEqual(report["summary"]["missing_voice_requests"], 1)
        self.assertEqual(report["reasons"]["missing_current_voice"], 1)
        self.assertEqual(report["examples"], [])
        self.assertEqual(report["examples_omitted"], 1)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(Path(self.config["render"]["work_dir"]).exists())

    def test_bad_audio_checksum_blocks_diagnostic_instead_of_claiming_hit(self):
        audio, _ = self.save(indexed=True)
        audio.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "audio changed"):
            self.report()


if __name__ == "__main__":
    unittest.main()
