"""Deterministic truncation escalation through the real split-recovery workflow."""

from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.fish_http import FishHTTPError
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, SpeechVerificationFailed, _render_chunk,
)
from legacy_recovery_fixtures import LegacySpeechResolver as SpeechResolver
from shared_fixtures import renderer_config
from test_renderer_recurring_resolution import PlanClient, ResolutionEngine
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryVerifier
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


TEXT = "Arrgh the pain has returned. Everyone waited outside."
FIRST = "Arrgh the pain has returned."
SECOND = "Everyone waited outside."


class DramaFailureEngine(ResolutionEngine):
    def __init__(self, registry):
        super().__init__(registry)
        self.whole_error = None

    def synthesize(self, text, voice, path, *, seed, direction=""):
        if text == TEXT and self.whole_error is not None:
            self.calls.append({"text": text, "voice": deepcopy(voice),
                               "seed": seed, "direction": direction})
            raise self.whole_error
        return super().synthesize(text, voice, path, seed=seed, direction=direction)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for truncation recovery")
class TruncationSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = renderer_config(max_attempts=3, sample_rate=44100)
        self.registry = {}
        self.engine = DramaFailureEngine(self.registry)
        self.base = RecoveryVerifier(self.registry)
        self.client = PlanClient([{"action": "retry", "edits": [],
                                   "reason": "Try another complete take."}])
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.voice = reference("truncation-character")
        voice_path = self.work / "truncation-character-reference.wav"
        write_tone(voice_path)
        self.voice.update(reference_audio=str(voice_path), reference_sha256=file_hash(voice_path))
        self.chunk = speech_request(self.voice, TEXT)["chunk"]
        self.messages = []

    def render_chunk(self, budget=None):
        return _render_chunk(
            self.chunk, 1, 1, {self.voice["voice_id"]: self.voice},
            self.engine, self.engine.identity, self.verifier, self.config,
            self.work, {}, Lock(), notify=self.messages.append,
            retry_budget=budget if budget is not None else SpeechRetryBudget(),
        )

    def call_texts(self):
        return [call["text"] for call in self.engine.calls]

    def markers(self):
        return list((self.work / "clips").glob("*/speech-recovery.json"))

    def test_repeated_drama3_truncation_splits_after_two_whole_attempts_without_resolver(self):
        self.engine.whole_error = EngineError("Drama 3 did not finish the text: 'silence'")

        result, accepted = self.render_chunk()

        self.assertEqual(self.call_texts(), [TEXT, TEXT, FIRST, SECOND])
        self.assertEqual(self.client.plans, [])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(record["request"]["recovery"]["strategy"], "sentence_split")
        self.assertEqual(len(self.markers()), 1)

    def test_persistent_text_mismatch_keeps_the_existing_whole_take_ladder(self):
        self.engine.heard = lambda given, number: (
            given.replace("Arrgh", "Oh") if given == TEXT else given
        )

        _, accepted = self.render_chunk()

        self.assertEqual(self.call_texts(), [TEXT] * 6 + [FIRST, SECOND])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual([row["attempt"] for row in self.client.plans[0]["failures"]], [1, 2, 3])
        self.assertEqual(len(self.markers()), 1)

    def test_fish_transport_failure_does_not_trigger_early_split(self):
        self.engine.whole_error = FishHTTPError(
            "Fish HTTP request failed (fixture): No route to host"
        )

        with self.assertRaises(SpeechVerificationFailed) as failure:
            self.render_chunk()

        self.assertIn("No route to host", str(failure.exception))
        self.assertEqual(self.call_texts(), [TEXT] * 3)
        self.assertEqual(self.client.plans, [])
        self.assertEqual(self.markers(), [])

    def test_truncation_after_rejected_join_does_not_repeat_disabled_split(self):
        self.config["verification"]["max_resolution_attempts"] = 6
        self.engine.whole_error = EngineError("Drama 3 did not finish the text: 'length'")
        self.base.fail_assembly = True

        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()

        self.assertEqual(self.call_texts(), [TEXT, TEXT, FIRST, SECOND, TEXT])
        self.assertEqual(self.client.plans, [])
        markers = self.markers()
        self.assertEqual(len(markers), 1)
        marker_path = markers[0]
        self.assertFalse(read_json(marker_path.parent / "speech-resolution.json")["binding"]["can_split"])
        joined = self.work / "clips" / read_json(marker_path)["recovery_request_hash"]
        self.assertEqual(len(list(joined.glob("attempt-*.json"))), 1)
        self.assertFalse((joined / "accepted.json").exists())




class SilentStopEngine(ResolutionEngine):
    """Whole takes come back cut after the silence watch stopped them, saying ``heard``."""

    def __init__(self, registry, heard):
        super().__init__(registry)
        self.heard = lambda text, number: heard if text == TEXT else text

    def synthesize(self, text, voice, path, *, seed, direction=""):
        metadata = super().synthesize(text, voice, path, seed=seed, direction=direction)
        return {**metadata, "finish_reason": "silence"} if text == TEXT else metadata


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for truncation recovery")
class SilenceStoppedTakeTests(unittest.TestCase):
    """A take the silence watch stopped is kept and verified, not thrown away."""

    setUp = TruncationSplitTests.setUp
    render_chunk = TruncationSplitTests.render_chunk
    call_texts = TruncationSplitTests.call_texts
    markers = TruncationSplitTests.markers

    def test_a_stopped_take_that_said_every_word_is_used(self):
        self.engine = SilentStopEngine(self.registry, TEXT)
        result, accepted = self.render_chunk()
        self.assertEqual(self.call_texts(), [TEXT])
        self.assertTrue(accepted["report"]["passed"])
        record = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(record["engine_metadata"]["finish_reason"], "silence")
        self.assertEqual(self.markers(), [])

    def test_a_stopped_take_missing_words_is_a_truncation_that_splits(self):
        self.engine = SilentStopEngine(self.registry, FIRST)
        result, accepted = self.render_chunk()
        self.assertEqual(self.call_texts(), [TEXT, TEXT, FIRST, SECOND])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(len(self.markers()), 1)
        stopped = [read_json(path)["report"] for path in (self.work / "clips").glob("*/attempt-*.json")
                   if read_json(path)["request"]["kind"] == "speech" and "recovery" not in read_json(path)["request"]
                   and read_json(path)["request"]["chunk"]["spoken_text"] == TEXT]
        self.assertEqual([report.get("generation_stopped") for report in stopped], ["silence", "silence"])
        self.assertTrue(all(report["passed"] is False for report in stopped))

    def test_a_stopped_take_is_never_a_best_effort_choice(self):
        from epub_tts_narrator.renderer.exhaustion_selection import select_retained
        self.engine = SilentStopEngine(self.registry, FIRST)
        # Every take stops short, parts included, so recovery runs out.
        self.engine.heard = lambda text, number: "Arrgh the pain"
        with self.assertRaises(SpeechVerificationFailed):
            self.render_chunk()
        directory = next(path.parent for path in (self.work / "clips").glob("*/attempt-001.json")
                         if read_json(path)["request"]["kind"] == "speech")
        request = read_json(directory / "attempt-001.json")["request"]
        with self.assertRaisesRegex(ValueError, "went silent before finishing"):
            select_retained(directory, request, self.client)


if __name__ == "__main__":
    unittest.main()
