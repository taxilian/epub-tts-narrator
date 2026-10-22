"""Graceful cancellation preserves verified work without starting new takes."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.interrupts import RenderInterrupted, RenderInterrupts
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import make_render_plan, render
from audition_fixtures import prepare_bundle
from test_renderer_pool import pool_config, worker
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class GracefulRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="interrupt-fixture")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE, paragraph_synthesis=False,
        )
        self.config["quality"] = validate_policy({})
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)
        self.messages = []

    def publish_voices(self, identity=None):
        store = VoiceStore(self.config["voices"]["directory"], "interrupt-fixture")
        for profile in self.bundle["cast"]["voices"]:
            reference = self.root / (profile["id"] + ".wav")
            self.engine._write("Reference words.", reference, 17)
            store.publish(profile["id"], profile, reference, "Reference words.",
                          identity or self.engine.identity, {"passed": True}, {})

    def run_render(self, engine=None):
        return render(self.bundle, self.config, engine or self.engine, self.verifier,
                      notify=self.messages.append, chapters_only=True)

    def accepted(self):
        return {path: read_json(path) for path in (self.root / "work" / "clips").glob("*/accepted.json")}

    def assert_saved(self, count):
        records = self.accepted()
        self.assertEqual(len(records), count)
        indices = list((self.root / "work" / "speech-cache").glob("*.json"))
        self.assertEqual(len(indices), count)
        for path, record in records.items():
            self.assertIs(record["report"]["passed"], True)
            self.assertEqual(file_hash(path.parent / record["audio_file"]), record["audio_sha256"])
            self.assertTrue(any(read_json(index)["accepted_sha256"] == file_hash(path) for index in indices))
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        self.assertFalse((Path(self.bundle["directory"]) / ".audio-render.lock").exists())
        return records

    def test_stop_during_generation_finishes_verification_and_cache_then_resumes(self):
        self.publish_voices()
        original = self.engine.synthesize
        with RenderInterrupts() as control:
            def generate(*args, **kwargs):
                result = original(*args, **kwargs)
                control.requested = True
                return result

            with patch.object(self.engine, "synthesize", side_effect=generate):
                with self.assertRaises(RenderInterrupted):
                    self.run_render()
        self.assertEqual(len(self.engine.speech_calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        saved = {path: path.read_bytes() for path in self.assert_saved(1)}
        self.assertFalse(list((self.root / "work" / "completed-chapters").glob("*.json")))

        expected = sum(not chunk["silent_control"] for chapter in make_render_plan(
            self.bundle, self.config, multi_speaker=False)["chapters"] for chunk in chapter["chunks"])
        result = self.run_render()
        self.assertEqual(result["status"], "chapters_complete")
        self.assertEqual(len(self.engine.speech_calls), expected)
        self.assertEqual(len(self.verifier.calls), expected)
        self.assertTrue(all(path.read_bytes() == before for path, before in saved.items()))
        self.assertFalse(any("rejected:" in message for message in self.messages))

    def test_stop_during_verification_saves_current_audio_and_does_not_start_next_chapter(self):
        # Put the first performance in its own chapter so the stop also covers
        # the chapter boundary, not just the next chunk in the same chapter.
        original_chapter = self.bundle["chapters"][0]
        first, following = deepcopy(original_chapter), deepcopy(original_chapter)
        first["rows"] = first["rows"][:1]
        following["rows"] = following["rows"][1:]
        following["entry"]["title"] = "Following chapter"
        following["entry"]["path"] = "scripts/chapter-0002.json"
        self.bundle["chapters"] = [first, following]
        self.publish_voices()
        original = self.verifier.transcribe
        with RenderInterrupts() as control:
            def transcribe(path):
                result = original(path)
                control.requested = True
                return result

            with patch.object(self.verifier, "transcribe", side_effect=transcribe):
                with self.assertRaises(RenderInterrupted):
                    self.run_render()
        self.assertEqual(len(self.engine.speech_calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assert_saved(1)
        self.assertFalse(any("Processing chapter 2" in message for message in self.messages))

    def test_failed_active_take_retains_rejection_without_acceptance_or_retry(self):
        self.publish_voices()
        first_text = make_render_plan(self.bundle, self.config, multi_speaker=False)["chapters"][0]["chunks"][0]["spoken_text"]
        self.verifier.wrong_texts.add(first_text)
        original = self.engine.synthesize
        with RenderInterrupts() as control:
            def generate(*args, **kwargs):
                result = original(*args, **kwargs)
                control.requested = True
                return result

            with patch.object(self.engine, "synthesize", side_effect=generate):
                with self.assertRaises(RenderInterrupted):
                    self.run_render()
        self.assertEqual(len(self.engine.speech_calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assert_saved(0)
        attempts = list((self.root / "work" / "clips").glob("*/attempt-*.json"))
        self.assertEqual(len(attempts), 1)
        report = read_json(attempts[0])["report"]
        self.assertIs(report["passed"], False)
        self.assertIn("text_mismatch", report["reasons"])
        self.assertFalse(any("Recovering speech" in message for message in self.messages))

    def test_two_active_workers_both_finish_and_publish_before_stop(self):
        self.assert_pool_drains(2)

    def test_waiting_pool_work_does_not_start_a_third_take_after_stop(self):
        self.assert_pool_drains(3)

    def assert_pool_drains(self, inflight):
        self.config["render_pool"] = pool_config(worker("first"), worker("second"))["render_pool"]
        pool = RenderPool(self.config)
        pool.max_inflight = inflight
        self.addCleanup(pool.close)
        self.publish_voices(pool.voice_binding_identity)
        engines = [FixtureEngine(self.fixtures), FixtureEngine(self.fixtures)]
        both_started = Barrier(2)
        with RenderInterrupts() as control:
            def generation(engine):
                original = engine.synthesize
                def generate(*args, **kwargs):
                    both_started.wait(5)
                    control.requested = True
                    return original(*args, **kwargs)
                return generate

            for entry, engine in zip(pool._workers, engines):
                entry._engine = engine
                engine.synthesize = generation(engine)
            with self.assertRaises(RenderInterrupted):
                self.run_render(pool)
        self.assertEqual([len(engine.speech_calls) for engine in engines], [1, 1])
        self.assertEqual(len(self.verifier.calls), 2)
        saved = self.assert_saved(2)
        self.assertEqual({record["engine_metadata"]["render_worker"] for record in saved.values()},
                         {"first", "second"})
        self.assertFalse(any("rejected:" in message for message in self.messages))

    def test_cli_owns_interrupt_context_and_returns_130_after_cache_publication(self):
        self.publish_voices()
        control = RenderInterrupts()
        original = self.engine.synthesize
        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            control.requested = True
            return result

        output, error = StringIO(), StringIO()
        with (patch("epub_tts_narrator.renderer.cli.load_config", return_value=deepcopy(self.config)),
              patch("epub_tts_narrator.renderer.interrupts.RenderInterrupts", return_value=control),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier),
              patch.object(self.engine, "synthesize", side_effect=generate),
              redirect_stdout(output), redirect_stderr(error)):
            status = main(["render", self.bundle["directory"], "--chapters-only"])
        self.assertEqual(status, 130, output.getvalue() + error.getvalue())
        self.assertIn("retained", output.getvalue() + error.getvalue())
        self.assertEqual(len(self.engine.speech_calls), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assert_saved(1)
        self.assertNotIn("rejected:", output.getvalue())

    def test_cli_permanent_generation_failure_after_stop_preserves_error_and_returns_130(self):
        self.assert_failed_cli_stop("generation")

    def test_cli_verification_outage_after_stop_preserves_pending_audio_and_returns_130(self):
        self.assert_failed_cli_stop("verification")

    def assert_failed_cli_stop(self, failure):
        import epub_tts_narrator.renderer.workflow as workflow

        class PermanentFailure(RuntimeError):
            retryable = False

        self.publish_voices()
        control = RenderInterrupts()
        original_generation = self.engine.synthesize
        original_validation = workflow.validate_audio

        def generate(*args, **kwargs):
            result = original_generation(*args, **kwargs)
            control.requested = True
            if failure == "generation":
                raise PermanentFailure("Fixture permanent generation error")
            return result

        def validate(*args, **kwargs):
            report = original_validation(*args, **kwargs)
            return {**report, "passed": False, "reasons": ["text_mismatch"],
                    "text_judge": {"decision": "unavailable", "error_code": "fixture_timeout"}}

        output, error = StringIO(), StringIO()
        with (patch("epub_tts_narrator.renderer.cli.load_config", return_value=deepcopy(self.config)),
              patch("epub_tts_narrator.renderer.interrupts.RenderInterrupts", return_value=control),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier),
              patch.object(self.engine, "synthesize", side_effect=generate),
              patch.object(workflow, "validate_audio", side_effect=validate),
              redirect_stdout(output), redirect_stderr(error)):
            status = main(["render", self.bundle["directory"], "--chapters-only"])
        self.assertEqual(status, 130, output.getvalue() + error.getvalue())
        self.assertEqual(len(self.engine.speech_calls), 1)
        self.assert_saved(0)
        self.assertEqual(read_json(self.root / "work" / "status.json")["status"], "interrupted")
        attempts = list((self.root / "work" / "clips").glob("*/attempt-*.json"))
        self.assertEqual(len(attempts), 1)
        record = read_json(attempts[0])
        if failure == "generation":
            self.assertEqual(len(self.verifier.calls), 0)
            self.assertEqual(record["report"]["error"], "Fixture permanent generation error")
            self.assertIs(record["report"]["retryable"], False)
        else:
            self.assertEqual(len(self.verifier.calls), 1)
            self.assertEqual(record["report"]["text_judge"]["decision"], "unavailable")
            self.assertEqual(file_hash(attempts[0].parent / record["audio_file"]), record["audio_sha256"])

    def test_stop_during_resumed_asr_does_not_start_another_recovery_model_request(self):
        from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
        from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
        from epub_tts_narrator.renderer.workflow import VerificationUnavailable, verified_take
        from test_renderer_recurring_resolution import PlanClient, ResolutionEngine, TEXT
        from test_renderer_speech_recovery_integration import RecoveryVerifier

        registry = {}
        engine = ResolutionEngine(registry)
        recognizer = RecoveryVerifier(registry)
        client = PlanClient()
        client.unavailable = True
        verifier = RecurringSpeechVerifier(recognizer, SpeechResolver(client))
        request = {"kind": "speech", "text": TEXT, "engine": engine.identity}
        voice = {"asset_hash": "fixture-reference-asset", "kind": "reference",
                 "reference_text": "Fixture reference voice.", "reference_sha256": "1" * 64}
        directory = self.root / "resume-takes"

        def take():
            return verified_take(TEXT,
                lambda path, seed: engine.synthesize(TEXT, voice, path, seed=seed),
                directory, request, verifier, self.config, notify=self.messages.append)

        with self.assertRaises(VerificationUnavailable):
            take()
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(len(client.plans), 1)
        original = recognizer.transcribe
        with RenderInterrupts() as control:
            def transcribe(path):
                result = original(path)
                control.requested = True
                return result

            with patch.object(recognizer, "transcribe", side_effect=transcribe):
                with self.assertRaises(RenderInterrupted):
                    take()
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(len(client.plans), 1)
        self.assertFalse((directory / "accepted.json").exists())
        pending = read_json(directory / "attempt-003.json")
        self.assertTrue(pending["report"]["resolution_pending"])
        self.assertEqual(file_hash(directory / pending["audio_file"]), pending["audio_sha256"])


if __name__ == "__main__":
    unittest.main()
