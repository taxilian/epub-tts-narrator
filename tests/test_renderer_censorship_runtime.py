"""Censor source words without altering accepted speech or loading a TTS model."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.spoken_text import build_spoken
from epub_tts_narrator.renderer.censorship import CensorshipAlignmentRequired, validate_policy
from epub_tts_narrator.renderer.censorship_runtime import censorship_targets, prepare_chapter_censorship
from epub_tts_narrator.renderer.common import file_hash
from test_renderer_assembly import clip, make_wav


class CensorshipRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = make_wav(self.root / "accepted.wav", frames=44100)
        self.policy = validate_policy({"enabled": True, "words": ["fuck", "shit"]})
        self.item = clip(self.audio, text="Oh fuck now", break_after="none")
        self.report = {"passed": True, "transcript": "Oh fuck now", "words": [
            {"word": "Oh", "start": 0, "end": .2, "probability": .99},
            {"word": "fuck", "start": .2, "end": .7, "probability": .99},
            {"word": "now", "start": .7, "end": 1, "probability": .99}]}
        self.accepted = [{"chunk_id": self.item["chunk_id"], "audio_sha256": file_hash(self.audio),
                          "report": deepcopy(self.report)}]

    def normalized(self, original, replacement):
        row = self.item["segment"]
        start = row["text"].index(original)
        row["speech_normalization"] = {"version": 1, "edits": [{
            "start": start, "end": start + len(original), "original": original,
            "replacement": replacement, "direction": "Natural speech", "reason": "Pronunciation"}]}
        spoken, _ = build_spoken(row)
        self.item.update(spoken_text=spoken, spoken_end=len(spoken))

    def prepare(self, **kwargs):
        return prepare_chapter_censorship([self.item], self.accepted, self.policy,
                                         notify=lambda _: None, **kwargs)[0]

    def test_source_targets_preserve_case_boundaries_and_ignore_substrings(self):
        self.item = clip(self.audio, text="Shit! The shittake and fuchsia are fine.")
        self.assertEqual(censorship_targets(self.item, self.policy), [
            {"text": "Shit", "normalized": "shit", "start": 0, "end": 4}])

    def test_source_profanity_survives_approved_phonetic_spelling_and_nonzero_source_offset(self):
        self.item["segment"]["start"] = 50
        self.item["segment"]["end"] += 50
        self.normalized("fuck", "fuhk")
        self.assertEqual(censorship_targets(self.item, self.policy), [
            {"text": "fuhk", "normalized": "fuck", "start": 3, "end": 7}])

    def test_multiple_spoken_tokens_can_map_to_one_original_word(self):
        self.normalized("fuck", "fuh kuh")
        self.assertEqual(censorship_targets(self.item, self.policy), [
            {"text": "fuh kuh", "normalized": "fuck", "start": 3, "end": 10}])

    def test_multiword_replacement_is_not_guessed_or_silently_skipped(self):
        self.normalized("fuck now", "fuhk nau")
        with self.assertRaisesRegex(CensorshipAlignmentRequired, "multiword_replacement"):
            censorship_targets(self.item, self.policy)

    def test_target_outside_chunk_is_ignored_but_partial_word_is_not(self):
        self.item.update(spoken_start=8, spoken_end=11, spoken_text="now")
        self.assertEqual(censorship_targets(self.item, self.policy), [])
        self.item.update(spoken_start=4, spoken_text="uck now")
        with self.assertRaisesRegex(CensorshipAlignmentRequired, "split_between_audio_clips"):
            censorship_targets(self.item, self.policy)

    def test_paragraph_targets_keep_offsets_across_turns(self):
        first = clip(self.audio, text="Wait. ")
        paragraph = dict(self.item, spoken_text="Wait. Oh fuck now", paragraph_turns=[first, self.item])
        self.assertEqual(censorship_targets(paragraph, self.policy), [
            {"text": "fuck", "normalized": "fuck", "start": 9, "end": 13}])
        paragraph["spoken_text"] += "invented"
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            censorship_targets(paragraph, self.policy)

    def test_disabled_policy_never_requests_word_timings(self):
        self.policy["enabled"] = False
        self.accepted = []
        self.assertIs(prepare_chapter_censorship([self.item], [], self.policy)[0], self.item)

    def test_verified_timestamps_need_no_alignment_and_originals_are_immutable(self):
        before = deepcopy(self.accepted), deepcopy(self.item), self.audio.read_bytes()
        factory = Mock(side_effect=AssertionError("No model"))
        item = self.prepare(cache_dir=self.root / "alignment", verifier_factory=factory)
        self.assertEqual(len(item["censorship"]["masks"]), 1)
        self.assertEqual(before, (self.accepted, self.item, self.audio.read_bytes()))
        factory.assert_not_called()

    def test_no_matching_word_needs_no_timing_even_when_asr_omitted_words(self):
        self.item["segment"]["text"] = self.item["spoken_text"] = "Oh tree now"
        self.accepted[0]["report"].pop("words")
        self.assertEqual(self.prepare()["censorship"]["masks"], [])

    def test_missing_timestamps_use_cached_alignment_without_loading_model(self):
        self.accepted[0]["report"].pop("words")
        factory = Mock(side_effect=AssertionError("No model"))
        with patch("epub_tts_narrator.renderer.censorship_alignment.align_saved_audio", return_value=self.report) as align:
            result = self.prepare(cache_dir=self.root / "alignment", verifier_factory=factory)
        self.assertEqual(len(result["censorship"]["masks"]), 1)
        self.assertIsNone(align.call_args.args[2])
        factory.assert_not_called()
        self.assertNotIn("words", self.accepted[0]["report"])

    def test_missing_alignment_cache_lazily_uses_asr_without_touching_acceptance(self):
        from epub_tts_narrator.renderer.censorship_alignment import CensorshipAlignmentUnavailable
        self.accepted[0]["report"].pop("words")
        verifier, factory = object(), Mock()
        factory.return_value = verifier
        with patch("epub_tts_narrator.renderer.censorship_alignment.align_saved_audio",
                   side_effect=[CensorshipAlignmentUnavailable("missing"), self.report]) as align:
            self.prepare(cache_dir=self.root / "alignment", verifier_factory=factory)
        factory.assert_called_once_with()
        self.assertIs(align.call_args.args[2], verifier)
        self.assertNotIn("words", self.accepted[0]["report"])

    def test_bad_alignment_preserves_accepted_audio_and_does_not_request_another_model(self):
        from epub_tts_narrator.renderer.censorship_alignment import CensorshipAlignmentFailed
        self.accepted[0]["report"].pop("words")
        before = self.audio.read_bytes(), deepcopy(self.accepted)
        factory = Mock(side_effect=AssertionError("No replacement model"))
        with patch("epub_tts_narrator.renderer.censorship_alignment.align_saved_audio",
                   side_effect=CensorshipAlignmentFailed("unsafe")):
            with self.assertRaises(CensorshipAlignmentFailed):
                self.prepare(cache_dir=self.root / "alignment", verifier_factory=factory)
        self.assertEqual(before, (self.audio.read_bytes(), self.accepted))
        factory.assert_not_called()

    def test_changed_or_unaccepted_audio_cannot_be_masked(self):
        self.accepted[0]["audio_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "unchanged independently accepted"):
            self.prepare()
        self.accepted[0]["audio_sha256"] = file_hash(self.audio)
        self.accepted[0]["report"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "unchanged independently accepted"):
            self.prepare()
