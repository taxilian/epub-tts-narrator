"""Offline audio-bound forced alignment, separate from acceptance and TTS."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from threading import RLock, get_ident
import unittest
from unittest.mock import Mock, patch
import wave

from epub_tts_narrator.renderer.censorship_alignment import (
    CensorshipAlignmentFailed, CensorshipAlignmentUnavailable, _mlx_align, align_saved_audio,
)


class CensorshipAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.audio = self.directory / "accepted.wav"
        self.cache = self.directory / "alignment"
        self.write_audio()
        self.verifier = SimpleNamespace(
            model_path=self.directory / "model", language="en",
            identity={"backend": "mlx-whisper", "model_sha256": {"weights.npz": "a" * 64},
                      "package_version": "0.test", "packages": {}, "language": "en"})
        self.words = [{"word": "Oh", "start": 0.1, "end": 0.3},
                      {"word": " shit.", "start": 0.3, "end": 0.7}]
        self.runner = Mock(side_effect=lambda *args: deepcopy(self.words))

    def write_audio(self, duration=1, sample=1):
        with wave.open(str(self.audio), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(sample.to_bytes(2, "little", signed=True) * round(16000 * duration))

    def align(self, text="Oh shit.", verifier="default", **kwargs):
        kwargs.setdefault("report", {"transcript": text})
        return align_saved_audio(self.audio, text, self.verifier if verifier == "default" else verifier,
                                 self.cache, runner=self.runner, **kwargs)

    def test_cached_only_assembly_reuses_audio_bound_alignment_without_models(self):
        first = self.align()
        second = self.align(verifier=None)
        self.assertEqual(first, second)
        self.assertEqual(self.runner.call_count, 1)
        self.assertEqual(first["word_timing_status"], "available")
        self.assertEqual(first["alignment_method"], "mlx_whisper_attention_dtw")
        self.assertNotIn("passed", first)
        self.assertTrue(list(self.cache.glob("*.lock.lease")))
        self.assertFalse(list(self.cache.glob("*.lock")))

    def test_changed_audio_does_not_reuse_old_timings(self):
        first = self.align()
        self.write_audio(sample=2)
        second = self.align()
        self.assertEqual(self.runner.call_count, 2)
        self.assertNotEqual(first["alignment_audio_sha256"], second["alignment_audio_sha256"])

    def test_changed_text_does_not_reuse_old_timings(self):
        self.align()
        self.words[1]["word"] = " blast."
        self.align("Oh blast.")
        self.assertEqual(self.runner.call_count, 2)

    def test_configured_model_change_refreshes_alignment(self):
        self.align()
        self.verifier.identity["model_sha256"] = {"weights.npz": "b" * 64}
        self.align()
        self.assertEqual(self.runner.call_count, 2)

    def test_pending_is_saved_before_inference_and_failed_work_can_resume(self):
        def fail(*args):
            state = json.loads(next(self.cache.glob("*.json")).read_text())
            self.assertEqual(state["status"], "pending")
            raise RuntimeError("inference stopped")
        self.runner.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "inference stopped"):
            self.align()
        self.assertEqual(json.loads(next(self.cache.glob("*.json")).read_text())["status"], "failed")
        self.runner.side_effect = lambda *args: deepcopy(self.words)
        self.assertEqual(self.align()["words"], self.words)

    def test_missing_cached_alignment_requests_asr_and_never_tts(self):
        with self.assertRaisesRegex(CensorshipAlignmentUnavailable, "without regenerating speech"):
            self.align(verifier=None)
        self.runner.assert_not_called()

    def test_reference_and_judge_wrappers_reuse_the_primary_model(self):
        wrapped = SimpleNamespace(verifier=SimpleNamespace(primary=self.verifier, reference=object()))
        self.align(verifier=wrapped)
        self.assertIs(self.runner.call_args.args[2], self.verifier)

    def test_unsupported_verifier_is_explicit_and_not_substituted(self):
        other = SimpleNamespace(identity={"backend": "faster-whisper"}, model_path="model")
        with self.assertRaisesRegex(CensorshipAlignmentUnavailable, "MLX Whisper"):
            self.align(verifier=other)
        self.runner.assert_not_called()

    def test_edited_cached_words_are_not_trusted(self):
        self.align()
        target = next(self.cache.glob("*.json"))
        saved = json.loads(target.read_text())
        saved["result"]["words"][1]["start"] = 0.01
        target.write_text(json.dumps(saved))
        with self.assertRaisesRegex(CensorshipAlignmentFailed, "invalid"):
            self.align(verifier=None)

    def test_invalid_boundaries_or_lost_words_never_become_ready_alignment(self):
        cases = [
            [{"word": "Oh shit.", "start": 0, "end": float("nan")}],
            [{"word": "Oh shit.", "start": 0, "end": 0}],
            [{"word": "Oh shit.", "start": 0, "end": 1.1}],
            [{"word": "Oh", "start": 0.2, "end": 0.6}, {"word": " shit.", "start": 0.5, "end": 0.8}],
            [{"word": "Oh", "start": 0.2, "end": 0.6}],
            [{"word": "Oh shit.", "start": 0, "end": 0.5, "probability": -0.1}],
        ]
        for words in cases:
            with self.subTest(words=words):
                self.words = words
                with self.assertRaises(CensorshipAlignmentFailed):
                    self.align()
                self.assertEqual(json.loads(next(self.cache.glob("*.json")).read_text())["status"], "failed")

    def test_long_audio_needs_real_segment_boundaries_and_never_even_spacing(self):
        self.write_audio(duration=31)
        with self.assertRaisesRegex(CensorshipAlignmentUnavailable, "audio-only recognition"):
            self.align()
        self.runner.assert_not_called()

    def test_long_audio_uses_only_exact_ordered_existing_segments(self):
        self.write_audio(duration=31)
        segments = [{"text": "Oh", "start": 0.0, "end": 2.0},
                    {"text": " shit.", "start": 28.0, "end": 31.0}]
        self.words[1].update(start=28.1, end=28.7)
        self.align(report={"transcript": "Oh shit.", "segments": segments})
        self.assertEqual([(item["start"], item["end"]) for item in self.runner.call_args.args[3]], [(0, 2), (28, 31)])
        self.assertEqual("".join(item["text"] for item in self.runner.call_args.args[3]), "Oh shit.")

    def test_long_audio_decodes_only_for_real_segment_windows_and_resume_reuses_it(self):
        self.write_audio(duration=31)
        self.verifier.transcribe = Mock(return_value={"text": "OH SHIT!", "segments": [
            {"text": "OH", "start": 0.0, "end": 2.0}, {"text": " SHIT!", "start": 28.0, "end": 31.0}]})
        self.words[1].update(start=28.1, end=28.7)
        first = self.align()
        self.assertEqual(first["alignment_segment_source"], "fresh_audio_only_asr")
        self.verifier.transcribe.assert_called_once_with(self.audio)
        self.assertEqual(self.align(), first)
        self.assertEqual(self.align(verifier=None), first)
        self.assertEqual(self.verifier.transcribe.call_count, 1)

    def test_completed_segmentation_is_retained_when_alignment_is_interrupted(self):
        self.write_audio(duration=31)
        self.verifier.transcribe = Mock(return_value={"segments": [
            {"text": "Oh", "start": 0.0, "end": 2.0}, {"text": "shit", "start": 28.0, "end": 31.0}], "text": "Oh shit."})
        self.runner.side_effect = RuntimeError("interrupted alignment")
        with self.assertRaises(RuntimeError):
            self.align()
        self.words[1].update(start=28.1, end=28.7)
        self.runner.side_effect = lambda *args: deepcopy(self.words)
        self.align()
        self.assertEqual(self.verifier.transcribe.call_count, 1)

    def test_fresh_segmentation_cannot_relabel_missing_or_different_words(self):
        self.write_audio(duration=31)
        self.verifier.transcribe = Mock(return_value={"segments": [
            {"text": "Oh", "start": 0.0, "end": 2.0}, {"text": "blast", "start": 28.0, "end": 31.0}], "text": "Oh blast."})
        with self.assertRaisesRegex(CensorshipAlignmentUnavailable, "words do not exactly cover"):
            self.align()
        self.runner.assert_not_called()

    def test_long_audio_rejects_approximate_segment_text_or_overlaps(self):
        self.write_audio(duration=31)
        cases = [
            [{"text": "Oh", "start": 0.0, "end": 2.0}, {"text": " blast.", "start": 28.0, "end": 31.0}],
            [{"text": "Oh", "start": 0.0, "end": 3.0}, {"text": " shit.", "start": 2.0, "end": 31.0}],
            [{"text": "Oh shit.", "start": 0.0, "end": 31.0}],
        ]
        for segments in cases:
            with self.subTest(segments=segments):
                with self.assertRaises(CensorshipAlignmentUnavailable):
                    self.align(report={"transcript": "Oh shit.", "segments": segments})
        self.runner.assert_not_called()

    def test_missing_swearword_is_never_forced_onto_neighboring_audio_even_after_acceptance(self):
        self.verifier.transcribe = Mock(return_value={"text": "Oh.", "segments": []})
        accepted = {"passed": True, "transcript": "Oh.", "expected_text": "Oh shit.",
                    "acceptance_policy": "llm_exhaustion_selection"}
        for _ in range(2):
            with self.assertRaisesRegex(CensorshipAlignmentFailed, "missing word"):
                self.align(report=accepted)
        self.runner.assert_not_called()
        self.verifier.transcribe.assert_called_once_with(self.audio)

    def test_missing_recognition_triggers_one_audio_only_decode_and_then_resumes(self):
        self.verifier.transcribe = Mock(return_value={"text": "OH SHIT!", "segments": []})
        first = self.align(report={"passed": True, "word_timing_status": "unavailable"})
        self.assertEqual(first["alignment_coverage_source"], "fresh_audio_only_asr")
        self.assertEqual(self.align(verifier=None, report=None), first)
        self.verifier.transcribe.assert_called_once_with(self.audio)

    def test_independent_exact_words_with_unusable_timestamps_do_not_need_new_decode(self):
        self.verifier.transcribe = Mock(side_effect=AssertionError("needless ASR"))
        self.align(report={"passed": True, "transcript": "OH, shit!", "word_timing_status": "unavailable"})
        self.runner.assert_called_once()
        self.verifier.transcribe.assert_not_called()

    def test_audio_change_during_alignment_cannot_publish_timing(self):
        def mutate(*args):
            self.write_audio(sample=9)
            return self.words
        self.runner.side_effect = mutate
        with self.assertRaisesRegex(CensorshipAlignmentFailed, "changed during"):
            self.align()

    def test_pool_facade_aligns_on_original_inference_lane_and_cached_read_stays_idle(self):
        from epub_tts_narrator.renderer.local_runtime import LocalRuntime
        from epub_tts_narrator.renderer.pool import _SerializedVerifier
        runtime, reviews = LocalRuntime(), LocalRuntime(name="alignment-test-reviews")
        self.addCleanup(runtime.close)
        self.addCleanup(reviews.close)
        facade = _SerializedVerifier(self.verifier, RLock(), runtime, reviews)
        observed = []
        def align(*args):
            observed.append(get_ident())
            return deepcopy(self.words)
        self.runner.side_effect = align
        self.align(verifier=facade)
        self.assertEqual(observed, [runtime._thread_id])
        self.assertNotEqual(observed[0], get_ident())
        with patch.object(facade, "_call", side_effect=AssertionError("cache must not enter inference")):
            self.align(verifier=facade)

    def test_symlink_input_and_cache_are_rejected(self):
        link = self.directory / "link.wav"
        link.symlink_to(self.audio)
        with self.assertRaises(ValueError):
            align_saved_audio(link, "Oh shit.", self.verifier, self.cache, runner=self.runner)
        self.cache.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(CensorshipAlignmentUnavailable, "symlink"):
            self.align()

    def test_adapter_runs_teacher_forced_alignment_with_shared_model_no_decode(self):
        modules = {name: ModuleType(name) for name in ("mlx", "mlx.core", "mlx_whisper", "mlx_whisper.audio",
                   "mlx_whisper.timing", "mlx_whisper.tokenizer", "mlx_whisper.transcribe")}
        modules["mlx"].core = modules["mlx.core"]
        modules["mlx.core"].float16 = "float16"
        model = SimpleNamespace(is_multilingual=True, num_languages=99, dims=SimpleNamespace(n_mels=128, n_text_ctx=448))
        holder = SimpleNamespace(get_model=Mock(return_value=model))
        modules["mlx_whisper.transcribe"].ModelHolder = holder
        tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2]), sot_sequence=[9, 10])
        modules["mlx_whisper.tokenizer"].get_tokenizer = Mock(return_value=tokenizer)
        audio_module = modules["mlx_whisper.audio"]
        audio_module.SAMPLE_RATE, audio_module.HOP_LENGTH = 16000, 160
        audio_module.N_FRAMES, audio_module.N_SAMPLES = 3000, 480000
        audio_module.load_audio = Mock(return_value=[0] * 16000)
        mel = SimpleNamespace(astype=Mock(return_value="mel"))
        audio_module.log_mel_spectrogram = Mock(return_value="unpadded")
        audio_module.pad_or_trim = Mock(return_value=mel)
        timing = modules["mlx_whisper.timing"]
        timing.find_alignment = Mock(return_value=[SimpleNamespace(**word, probability=0.8) for word in self.words])
        timing.merge_punctuations = Mock()
        windows = [{"start": 0, "end": 1, "text": "Oh shit."}]
        with patch.dict("sys.modules", modules):
            words = _mlx_align(self.audio, "Oh shit.", self.verifier, windows)
        holder.get_model.assert_called_once_with(str(self.verifier.model_path), "float16")
        timing.find_alignment.assert_called_once_with(model, tokenizer, [1, 2], "mel", 100)
        self.assertEqual(words[1]["start"], 0.3)
        self.assertEqual(words[1]["probability"], 0.8)


if __name__ == "__main__":
    unittest.main()
