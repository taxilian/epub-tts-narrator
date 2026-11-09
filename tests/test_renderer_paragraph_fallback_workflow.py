"""Failed paragraph performances recover through verified original source turns."""

from copy import deepcopy
from threading import Lock
from time import sleep
import unittest
from unittest.mock import patch

from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import EngineError, format_dialogue, prepare_synthesis
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, VerificationUnavailable, make_render_plan,
)
from epub_tts_narrator.spoken_text import build_spoken
import epub_tts_narrator.renderer.workflow as workflow
from legacy_recovery_fixtures import LegacySpeechResolver as SpeechResolver
from shared_fixtures import fixture_codes
import test_renderer_paragraph_workflow as paragraph_fixtures
from test_renderer_recurring_resolution import PlanClient
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureVerifier


class TransportFailure(RuntimeError):
    transport_error = True
    retryable = True


class StructuredParagraphFixtureEngine(paragraph_fixtures.ParagraphFixtureEngine):
    capabilities = {**paragraph_fixtures.ParagraphFixtureEngine.capabilities,
                    "structured_performance": True}
    identity = {"engine": "drama3_mlx", "model": "fixture-model-v1",
                "checkpoint_files_sha256": {"weights.safetensors": "fixture-weights"},
                "temperature": 1.0, "top_p": .9, "top_k": 30}
    voice_binding_identity = paragraph_fixtures.ParagraphFixtureEngine.identity

    def __init__(self):
        super().__init__()
        self.performance_calls = []

    def synthesize_dialogue(self, turns, path, *, seed):
        self.dialogue_calls.append({
            "turns": [{"text": turn["text"], "voice_hash": turn["voice"]["asset_hash"],
                       "voice_id": turn["voice"]["voice_id"], "direction": turn["direction"]}
                      for turn in turns],
            "seed": seed,
        })
        text = "".join(turn["text"] for turn in turns)
        formatted, ordered_voices, formatting = format_dialogue(turns)
        return {**self._write(text, path, seed), "synthesis_text": formatted, **formatting,
                **fixture_codes(formatted, ordered_voices)}

    def synthesize(self, text, voice, path, *, seed, direction="", performance=None, traits=""):
        prepared, compiled, delivery = prepare_synthesis(
            text, direction, performance=performance, traits=traits)
        self.performance_calls.append({"text": text, "performance": deepcopy(performance)})
        return {**super().synthesize(prepared, voice, path, seed=seed, direction=direction),
                **delivery, "synthesis_text": compiled}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for paragraph recovery")
class ParagraphFallbackWorkflowTests(unittest.TestCase):
    # Reuse only fixture setup/helpers, not the existing test methods.
    def setUp(self):
        paragraph_fixtures.ParagraphWorkflowTests.setUp(self)
        # Group recovery still applies to contiguous passages by one saved
        # voice. Mixed voices now stay separate before synthesis is attempted.
        for row in self.bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000004":
                row["voice_id"] = "voice_lena"
                row["role_id"] = "lena_dialogue"

    chapters = paragraph_fixtures.ParagraphWorkflowTests.chapters
    package = paragraph_fixtures.ParagraphWorkflowTests.package
    accepted_clips = paragraph_fixtures.ParagraphWorkflowTests.accepted_clips
    grouped_clips = paragraph_fixtures.ParagraphWorkflowTests.grouped_clips

    def target(self):
        return next(chunk for chunk in make_render_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                    if chunk["segment"]["block_id"] == "b000004")

    def target_calls(self):
        text = "".join(row["text"] for row in self.bundle["chapters"][0]["rows"]
                       if row["block_id"] == "b000004")
        return [call for call in self.engine.dialogue_calls
                if "".join(turn["text"] for turn in call["turns"]) == text]

    def assert_individual_plan(self, original):
        chunks = [chunk for chunk in make_render_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                  if chunk["segment"]["block_id"] == "b000004"]
        self.assertEqual(chunks, original["paragraph_turns"])
        self.assertTrue(all("paragraph_turns" not in chunk for chunk in chunks))

    def snapshot_accepted(self):
        return {path: (record, file_hash(path.parent / record["audio_file"]))
                for path, record in self.accepted_clips().items()}

    def assert_snapshot_unchanged(self, before):
        for path, (record, checksum) in before.items():
            self.assertEqual(read_json(path), record)
            self.assertEqual(file_hash(path.parent / record["audio_file"]), checksum)

    def test_failed_group_recovers_in_same_run_with_original_voices_and_source_sync(self):
        original = deepcopy(self.target())
        self.verifier.wrong_texts.add(original["spoken_text"])
        original_dialogue = self.engine.synthesize_dialogue
        accepted_before_failure = {}

        def dialogue(turns, path, *, seed):
            if "".join(turn["text"] for turn in turns) == original["spoken_text"]:
                accepted_before_failure.update(self.snapshot_accepted())
            return original_dialogue(turns, path, seed=seed)

        with patch.object(self.engine, "synthesize_dialogue", side_effect=dialogue):
            completed = self.chapters()
        self.assertEqual(completed["status"], "chapters_complete")
        self.assertEqual(len(self.target_calls()), 2)
        self.assertEqual(len(self.engine.dialogue_calls), 3)  # earlier good group + two failures
        self.assertEqual(len(self.engine.speech_calls), 6)  # four original singles + two source turns
        self.assert_individual_plan(original)
        self.assertTrue(accepted_before_failure)
        self.assert_snapshot_unchanged(accepted_before_failure)
        self.assertEqual(len(self.grouped_clips()), 1)
        accepted_target = [record for record in self.accepted_clips().values()
                           if record["request"]["chunk"]["segment"]["block_id"] == "b000004"]
        self.assertEqual(len(accepted_target), 2)
        self.assertTrue(all(record["report"]["passed"] for record in accepted_target))
        self.assertTrue(all(record["request"]["kind"] == "speech" for record in accepted_target))
        self.assertEqual({record["request"]["chunk"]["chunk_id"] for record in accepted_target},
                         {turn["chunk_id"] for turn in original["paragraph_turns"]})
        failed_attempts = [read_json(path) for path in (self.root / "work" / "clips").glob("*/attempt-*.json")
                           if read_json(path)["request"]["chunk"]["chunk_id"] == original["chunk_id"]]
        self.assertEqual(len(failed_attempts), 2)
        self.assertTrue(all(not record["report"]["passed"] for record in failed_attempts))

        snapshot = self.snapshot_accepted()
        counts = (len(self.engine.speech_calls), len(self.engine.dialogue_calls), len(self.verifier.calls))
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("regenerated clip")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")):
            self.assertEqual(self.chapters()["chapters"], completed["chapters"])
            self.assertEqual(self.package()["status"], "complete")
        self.assert_snapshot_unchanged(snapshot)
        self.assertEqual(counts, (len(self.engine.speech_calls), len(self.engine.dialogue_calls), len(self.verifier.calls)))
        rows = [row for row in read_json(self.root / "output" / "sync.json")["segments"]
                if row["block_id"] == "b000004"]
        self.assertEqual([row["segment_id"] for row in rows],
                         [turn["segment"]["id"] for turn in original["paragraph_turns"]])
        self.assertEqual([row["voice_id"] for row in rows],
                         [turn["segment"]["voice_id"] for turn in original["paragraph_turns"]])
        self.assertEqual([row["source_ranges"] for row in rows],
                         [turn["source_ranges"] for turn in original["paragraph_turns"]])
        self.assertEqual("".join(row["source_text"] for row in rows), original["spoken_text"])
        self.assertEqual("".join(row["spoken_text"] for row in rows), original["spoken_text"])
        self.assertTrue(all(row.get("precision") != "paragraph" for row in rows))
        self.assertTrue(all(row["book_end_frame"] > row["book_start_frame"] for row in rows))
        self.assertLessEqual(rows[0]["book_end_frame"], rows[1]["book_start_frame"])

    def test_interruption_before_first_fallback_child_persists_split_and_reuses_prior_audio(self):
        original = deepcopy(self.target())
        self.verifier.wrong_texts.add(original["spoken_text"])
        from epub_tts_narrator.renderer.speech_text import trim_speech_text
        first_child = trim_speech_text(original["paragraph_turns"][0]["spoken_text"])
        synthesize = self.engine.synthesize

        def interrupt(text, *args, **kwargs):
            if text == first_child:
                raise KeyboardInterrupt("first fallback child interrupted")
            return synthesize(text, *args, **kwargs)

        with patch.object(self.engine, "synthesize", side_effect=interrupt):
            with self.assertRaisesRegex(KeyboardInterrupt, "first fallback child"):
                self.chapters()
        self.assert_individual_plan(original)
        self.assertEqual(len(self.target_calls()), 2)
        before = self.snapshot_accepted()
        self.assertEqual(len(self.grouped_clips()), 1)
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.assertFalse((self.root / "bundle" / ".audio-render.lock").exists())
        self.assertEqual(self.chapters()["status"], "chapters_complete")
        self.assertEqual(len(self.target_calls()), 2)
        self.assert_snapshot_unchanged(before)
        self.assert_individual_plan(original)

    def test_repeated_drama3_truncation_splits_expressive_paragraph_and_resumes_fallback(self):
        row = next(row for row in self.bundle["chapters"][0]["rows"]
                   if row["block_id"] == "b000004")
        spoken = build_spoken(row)[0]
        row["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(spoken),
            "controls": [{"kind": "pronunciation", "start": 0, "end": 4, "text": "Lena",
                          "alphabet": "respelling", "replacement": "LEE-nuh"}],
        }
        self.engine = StructuredParagraphFixtureEngine()
        client = PlanClient([{"action": "retry", "edits": [],
                              "reason": "Try another complete take."}])
        self.verifier = RecurringSpeechVerifier(
            FixtureVerifier(self.engine.fixture_transcripts), SpeechResolver(client))
        original = deepcopy(self.target())
        self.assertEqual(original["paragraph_turns"][0]["expressive_performance"],
                         row["performance_controls"])
        dialogue = self.engine.synthesize_dialogue
        failed_turns, accepted_before_failure = [], {}

        def unfinished(turns, path, *, seed):
            if "".join(turn["text"] for turn in turns) == original["spoken_text"]:
                if not failed_turns:
                    accepted_before_failure.update(self.snapshot_accepted())
                failed_turns.append(deepcopy(turns))
                raise EngineError("Drama 3 did not finish the text: 'silence'")
            return dialogue(turns, path, seed=seed)

        synthesize = self.engine.synthesize
        first_child = original["paragraph_turns"][0]["spoken_text"]
        interrupted = []

        def interrupt_first_child(text, *args, **kwargs):
            if text == first_child and not interrupted:
                interrupted.append(text)
                raise KeyboardInterrupt("first truncation fallback child interrupted")
            return synthesize(text, *args, **kwargs)

        with patch.object(self.engine, "synthesize_dialogue", side_effect=unfinished), \
             patch.object(self.engine, "synthesize", side_effect=interrupt_first_child):
            with self.assertRaisesRegex(KeyboardInterrupt, "first truncation fallback child"):
                self.chapters()

        self.assertEqual(len(failed_turns), 2)
        self.assertEqual(client.plans, [])
        self.assertEqual(interrupted, [first_child])
        self.assert_individual_plan(original)
        self.assertTrue(accepted_before_failure)
        self.assert_snapshot_unchanged(accepted_before_failure)
        self.assertEqual(len(list((self.root / "work" / "paragraph-fallbacks").glob("*.json"))), 1)
        failed_attempts = [read_json(path) for path in (self.root / "work" / "clips").glob("*/attempt-*.json")
                           if read_json(path)["request"]["chunk"]["chunk_id"] == original["chunk_id"]]
        self.assertEqual(len(failed_attempts), 2)
        self.assertTrue(all(record["report"]["reasons"] == ["generation_or_verification_error"]
                            and record["report"]["error_type"] == "EngineError"
                            and record["audio_file"] is None for record in failed_attempts))

        before = self.snapshot_accepted()
        completed = self.chapters()

        self.assertEqual(completed["status"], "chapters_complete")
        self.assertEqual(len(failed_turns), 2)
        self.assert_individual_plan(original)
        self.assert_snapshot_unchanged(before)
        accepted_target = sorted(
            (record for record in self.accepted_clips().values()
             if record["request"]["chunk"]["segment"]["block_id"] == "b000004"),
            key=lambda record: record["request"]["chunk"]["segment"]["start"])
        self.assertEqual([record["request"]["chunk"] for record in accepted_target],
                         original["paragraph_turns"])
        self.assertTrue(all(record["request"]["kind"] == "speech"
                            and record["report"]["passed"] for record in accepted_target))
        controlled = next(call for call in self.engine.performance_calls
                          if call["text"] == first_child)
        self.assertEqual(controlled["performance"],
                         original["paragraph_turns"][0]["expressive_performance"])

        snapshot = self.snapshot_accepted()
        counts = (len(self.engine.speech_calls), len(self.engine.dialogue_calls), len(self.verifier.calls))
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("regenerated clip")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")):
            self.assertEqual(self.chapters()["chapters"], completed["chapters"])
        self.assert_snapshot_unchanged(snapshot)
        self.assertEqual(counts, (len(self.engine.speech_calls), len(self.engine.dialogue_calls),
                                  len(self.verifier.calls)))

    def test_failed_individual_child_gets_fresh_attempts_only_in_next_render_invocation(self):
        original = deepcopy(self.target())
        from epub_tts_narrator.renderer.speech_text import trim_speech_text
        child_text = trim_speech_text(original["paragraph_turns"][0]["spoken_text"])
        self.verifier.wrong_texts.update({original["spoken_text"], child_text})
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.chapters()
        self.assertEqual(len(self.target_calls()), 2)
        self.assertEqual(sum(call["text"] == child_text for call in self.engine.speech_calls), 2)
        self.assert_individual_plan(original)
        self.assertFalse(list((self.root / "work" / "completed-chapters").glob("*.json")))
        before = self.snapshot_accepted()
        counts = (len(self.engine.dialogue_calls), len(self.engine.speech_calls), len(self.verifier.calls))
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.chapters()
        self.assertEqual(counts[0], len(self.engine.dialogue_calls))
        self.assertEqual(counts[1] + 2, len(self.engine.speech_calls))
        self.assertEqual(counts[2] + 2, len(self.verifier.calls))
        child_calls = [call for call in self.engine.speech_calls if call["text"] == child_text]
        self.assertEqual(len(child_calls), 4)
        self.assertEqual(len({call["seed"] for call in child_calls}), 4)
        self.assert_snapshot_unchanged(before)
        self.verifier.wrong_texts.remove(child_text)
        self.assertEqual(self.chapters()["status"], "chapters_complete")
        self.assertEqual(len(self.target_calls()), 2)  # never retry the failed full paragraph
        self.assertEqual(sum(call["text"] == child_text for call in self.engine.speech_calls), 5)
        self.assert_snapshot_unchanged(before)

    def test_recast_failed_paragraph_overrides_old_group_boundary_without_touching_old_audio(self):
        original = deepcopy(self.target())
        self.chapters()
        before = self.snapshot_accepted()
        self.assertEqual(len(self.grouped_clips()), 2)
        store = VoiceStore(self.config["voices"]["directory"], self.config["voices"]["series_id"])
        old_voice = store.load("voice_lena")
        replacement = self.root / "replacement.wav"
        self.engine._write(old_voice["reference_text"], replacement, 9182)
        store.publish_candidate("voice_lena", old_voice["profile"], replacement,
                                old_voice["reference_text"], self.engine.identity,
                                {"passed": True}, {"fixture": True}, candidate_id="b" * 32)
        store.select_candidate("voice_lena", "b" * 32)
        new_voice = store.load("voice_lena")
        self.assertNotEqual(new_voice["asset_hash"], old_voice["asset_hash"])
        self.verifier.wrong_texts.add(original["spoken_text"])

        self.assertEqual(self.chapters()["status"], "chapters_complete")
        self.assert_individual_plan(original)
        self.assertEqual(len(self.engine.dialogue_calls), 4)  # original pair + two recast rejections
        # Four original singles and the two recast source turns; history is
        # same-voice, so later narrator takes never heard the recast voice.
        self.assertEqual(len(self.engine.speech_calls), 6)
        self.assert_snapshot_unchanged(before)
        child = next(record for path, record in self.accepted_clips().items()
                     if path not in before and record["request"]["chunk"]["segment"]["voice_id"] == "voice_lena")
        self.assertEqual(child["request"]["voice_asset_hash"], new_voice["asset_hash"])
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("regenerated clip")):
            self.assertEqual(self.chapters()["status"], "chapters_complete")
            self.assertEqual(self.package()["status"], "complete")

    def test_transport_failure_does_not_split_a_paragraph(self):
        original = deepcopy(self.target())
        dialogue = self.engine.synthesize_dialogue
        failures = []

        def unavailable(turns, path, *, seed):
            if "".join(turn["text"] for turn in turns) == original["spoken_text"]:
                failures.append(seed)
                raise TransportFailure("fixture renderer disconnected")
            return dialogue(turns, path, seed=seed)

        with patch.object(self.engine, "synthesize_dialogue", side_effect=unavailable):
            with self.assertRaisesRegex(RenderBlocked, "renderer disconnected"):
                self.chapters()
        self.assertEqual(len(failures), 2)
        self.assertEqual(self.target(), original)
        self.assertEqual(len(self.engine.speech_calls), 2)
        self.assertEqual(self.chapters()["status"], "chapters_complete")
        self.assertEqual(len(self.grouped_clips()), 2)

    def test_jev_outage_retains_paragraph_for_reverification_without_fallback(self):
        original = deepcopy(self.target())
        validate = workflow.validate_audio

        def unavailable(path, text, *args, **kwargs):
            report = validate(path, text, *args, **kwargs)
            if text == original["spoken_text"]:
                report.update(passed=False, reasons=["text_mismatch"],
                              text_judge={"decision": "unavailable", "error_code": "fixture_outage"})
            return report

        with patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=unavailable):
            with self.assertRaises(VerificationUnavailable):
                self.chapters()
        self.assertEqual(self.target(), original)
        self.assertEqual(len(self.target_calls()), 1)
        self.assertEqual(len(self.engine.speech_calls), 2)
        self.assertEqual(self.chapters()["status"], "chapters_complete")
        self.assertEqual(len(self.target_calls()), 1)  # original saved waveform was reverified
        self.assertEqual(len(self.grouped_clips()), 2)

    def test_parallel_capacity_still_runs_chapter_fallback_sequentially_in_order(self):
        original = deepcopy(self.target())
        self.verifier.wrong_texts.add(original["spoken_text"])
        # Engine capacity never overlaps takes within one chapter: each take
        # is conditioned on the accepted takes before it.
        self.engine.max_parallel = self.engine.max_inflight = 2
        guard, active, peak = Lock(), [0], [0]

        def exclusive(method):
            def call(*args, **kwargs):
                with guard:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                try:
                    sleep(0.01)
                    return method(*args, **kwargs)
                finally:
                    with guard:
                        active[0] -= 1
            return call

        with patch.object(self.engine, "synthesize", side_effect=exclusive(self.engine.synthesize)), \
             patch.object(self.engine, "synthesize_dialogue", side_effect=exclusive(self.engine.synthesize_dialogue)):
            completed = self.chapters()
        self.assertEqual(peak[0], 1)
        self.assertEqual(completed["status"], "chapters_complete")
        self.assertEqual(len(self.target_calls()), 2)
        self.assertEqual(len(self.engine.dialogue_calls), 3)
        self.assertEqual(len(self.engine.speech_calls), 6)
        self.assert_individual_plan(original)
        expected = [chunk["chunk_id"] for chunk in make_render_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                    if not chunk["silent_control"]]
        checkpoint = read_json(next((self.root / "work" / "completed-chapters").glob("*.json")))
        self.assertEqual([row["chunk_id"] for row in checkpoint["acceptance"]], expected)
        self.assertEqual(len(self.accepted_clips()), len(expected))
        before = self.snapshot_accepted()
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("regenerated clip")):
            self.assertEqual(self.chapters()["chapters"], completed["chapters"])
        self.assert_snapshot_unchanged(before)


if __name__ == "__main__":
    unittest.main()
