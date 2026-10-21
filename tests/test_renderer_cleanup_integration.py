"""Exercise cleanup through accepted speech, configuration, locks, and the CLI."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer import workflow
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.quality import validate_policy
from audition_fixtures import prepare_document
from test_contracts import valid_document
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for audio integration")
class AcceptedSpeechCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "take"
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.config["quality"] = validate_policy({})
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)
        self.messages = []
        self.text = "The last lantern still shines."

    def take(self, *, kind="speech", purpose="speech", generate=None):
        return workflow.verified_take(
            self.text, generate or (lambda path, seed: self.engine._write(self.text, path, seed)),
            self.directory, {"kind": kind, "text": self.text, "engine": self.engine.identity},
            self.verifier, self.config, purpose=purpose, notify=self.messages.append,
        )

    def test_accepted_speech_removes_raw_but_resumes_without_models(self):
        for kind in ("speech", "paragraph_speech"):
            with self.subTest(kind=kind):
                self.directory = self.root / kind
                audio, accepted = self.take(kind=kind)
                self.assertTrue(audio.is_file())
                self.assertEqual(file_hash(audio), accepted["audio_sha256"])
                self.assertTrue((self.directory / "attempt-001.json").is_file())
                self.assertFalse(list(self.directory.glob("*-raw.wav")))
                self.assertEqual(read_json(self.directory / "accepted.json"), accepted)
                with (patch.object(self.engine, "_write", side_effect=AssertionError("regenerated speech")),
                      patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified speech"))):
                    self.assertEqual((audio, accepted), self.take(kind=kind))

    def test_existing_acceptance_is_cleaned_when_resumed_without_invalidation(self):
        self.config["render"]["keep_intermediate_audio"] = True
        audio, accepted = self.take()
        raw = self.directory / "attempt-001-raw.wav"
        self.assertTrue(raw.is_file())
        self.config["render"]["keep_intermediate_audio"] = False
        with (patch.object(self.engine, "_write", side_effect=AssertionError("regenerated speech")),
              patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified speech"))):
            self.assertEqual((audio, accepted), self.take())
        self.assertFalse(raw.exists())
        self.assertEqual(file_hash(audio), accepted["audio_sha256"])

    def test_reference_purpose_keeps_original_audio_even_for_recognized_request(self):
        # Purpose alone must protect a sample even if a caller supplies a speech
        # request; protecting only the request kind would not be sufficient.
        audio, accepted = self.take(purpose="reference")
        raw = self.directory / "attempt-001-raw.wav"
        self.assertTrue(raw.is_file())
        checksum = file_hash(raw)
        self.assertEqual((audio, accepted), self.take(purpose="reference"))
        self.assertEqual(file_hash(raw), checksum)

    def test_explicit_keep_setting_preserves_generated_and_cached_intermediates(self):
        self.config["render"]["keep_intermediate_audio"] = True
        first = self.take()
        raw = self.directory / "attempt-001-raw.wav"
        self.assertTrue(raw.is_file())
        checksum = file_hash(raw)
        self.assertEqual(first, self.take())
        self.assertEqual(file_hash(raw), checksum)

    def test_failed_speech_retains_normalized_audio_and_attempt_records(self):
        self.verifier.wrong_texts.add(self.text)
        with self.assertRaises(workflow.RenderBlocked):
            self.take()
        self.assertFalse((self.directory / "accepted.json").exists())
        for number in (1, 2):
            prefix = f"attempt-{number:03d}"
            self.assertTrue((self.directory / f"{prefix}.wav").is_file())
            self.assertTrue((self.directory / f"{prefix}-raw.wav").is_file())
            self.assertFalse(read_json(self.directory / f"{prefix}.json")["report"]["passed"])

    def test_accepted_later_attempt_keeps_earlier_failed_normalized_audio(self):
        wrong = {"passed": False, "reasons": ["text_mismatch"]}
        good = {"passed": True, "reasons": []}
        with patch.object(workflow, "validate_audio", side_effect=[wrong, good]):
            audio, accepted = self.take()
        self.assertEqual(audio.name, "attempt-002.wav")
        self.assertEqual(accepted["attempt"], 2)
        self.assertTrue((self.directory / "attempt-001.wav").is_file())
        self.assertFalse(read_json(self.directory / "attempt-001.json")["report"]["passed"])
        self.assertTrue((self.directory / "attempt-002.json").is_file())
        self.assertFalse(list(self.directory.glob("*-raw.wav")))

    def test_pending_confirmation_retains_audio_for_verification_resume(self):
        pending = {"passed": False, "reasons": ["text_mismatch"],
                   "text_judge": {"decision": "unavailable"}}
        with patch.object(workflow, "validate_audio", return_value=pending):
            with self.assertRaises(workflow.VerificationUnavailable):
                self.take()
        self.assertFalse((self.directory / "accepted.json").exists())
        self.assertTrue((self.directory / "attempt-001.wav").is_file())
        self.assertTrue((self.directory / "attempt-001-raw.wav").is_file())
        self.assertEqual(read_json(self.directory / "attempt-001.json")["report"], pending)

    def test_revalidated_legacy_acceptance_cleans_intermediates_after_success(self):
        self.config["render"]["keep_intermediate_audio"] = True
        audio, accepted = self.take()
        accepted["report"] = {"passed": False, "reasons": ["text_mismatch"]}
        write_json(self.directory / "accepted.json", accepted)
        self.config["render"]["keep_intermediate_audio"] = False
        with patch.object(self.engine, "_write", side_effect=AssertionError("regenerated speech")):
            resumed, revalidated = self.take()
        self.assertEqual(resumed, audio)
        self.assertTrue(revalidated["report"]["passed"])
        self.assertEqual(len(self.verifier.calls), 2)
        self.assertFalse((self.directory / "attempt-001-raw.wav").exists())

    def test_cleanup_error_warns_but_preserves_successful_acceptance(self):
        with patch("epub_tts_narrator.renderer.cleanup.clean_take_intermediates",
                   side_effect=OSError("test disk permission failure")):
            audio, accepted = self.take()
            with patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified speech")):
                self.assertEqual((audio, accepted), self.take())
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(read_json(self.directory / "accepted.json"), accepted)
        self.assertEqual(file_hash(audio), accepted["audio_sha256"])
        self.assertTrue((self.directory / "attempt-001-raw.wav").is_file())
        self.assertTrue(any("test disk permission failure" in message for message in self.messages))


class CleanupCommandIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config_path = self.root / "renderer.toml"
        self.config_path.write_text('[engine]\ncheckpoint="./unused-tts-model"\n'
                                    '[verification]\nmodel="./unused-asr-model"\n')
        self.book = self.root / "book"
        export_bundle(prepare_document(valid_document()), self.book)
        self.bundle = load_bundle(self.book)
        self.config = load_config(self.config_path)
        self.work = self.book / "render-work"

    def run_cli(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main([str(arg) for arg in args])
        return status, output.getvalue(), error.getvalue()

    def test_cleanup_command_wires_dry_run_and_discard_without_models_or_plan(self):
        work = self.root / "external-work"
        work.mkdir()
        with (patch.object(workflow, "cleanup_render_work", wraps=workflow.cleanup_render_work) as cleanup,
              patch("epub_tts_narrator.renderer.cli.make_plan", side_effect=AssertionError("planned speech")),
              patch("epub_tts_narrator.renderer.cli.voice_store", side_effect=AssertionError("loaded voices")),
              patch("epub_tts_narrator.renderer.engines.create_engine", side_effect=AssertionError("loaded TTS")),
              patch("epub_tts_narrator.renderer.verification.create_verifier", side_effect=AssertionError("loaded ASR"))):
            status, output, error = self.run_cli(
                "cleanup", self.book, "--config", self.config_path, "--work-dir", work,
                "--dry-run", "--discard-render-cache")
        self.assertEqual(status, 0, error)
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.args[1]["render"]["work_dir"], str(work))
        self.assertEqual(cleanup.call_args.kwargs, {"dry_run": True, "discard_render_cache": True})
        self.assertIn("voice", output.lower())
        self.assertFalse((self.book / "voices").exists())
        self.assertFalse((self.book / ".audio-render.lock").exists())
        self.assertFalse((work / ".render.lock").exists())

    def test_active_render_or_book_lock_refuses_cleanup_and_preserves_originals(self):
        for lock_name in (".render.lock", ".audio-render.lock"):
            with self.subTest(lock=lock_name):
                self.work.mkdir(exist_ok=True)
                lock = (self.work if lock_name == ".render.lock" else self.book) / lock_name
                lock.mkdir()
                self.addCleanup(lambda path=lock: path.rmdir() if path.exists() else None)
                candidate = self.work / "clips" / "example" / "attempt-001-raw.wav"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(b"original audio fixture")
                before = candidate.read_bytes()
                with patch("epub_tts_narrator.renderer.cleanup.cleanup_work",
                           side_effect=AssertionError("cleanup entered despite existing lock")):
                    with self.assertRaises(workflow.RenderBlocked):
                        workflow.cleanup_render_work(self.bundle, self.config, discard_render_cache=True)
                self.assertEqual(candidate.read_bytes(), before)
                self.assertTrue(lock.is_dir())
                other = self.book / ".audio-render.lock" if lock_name == ".render.lock" else self.work / ".render.lock"
                self.assertFalse(other.exists())
                lock.rmdir()

    def test_analysis_update_lock_also_refuses_cleanup_and_releases_owned_locks(self):
        self.work.mkdir()
        lock = self.book / ".analysis-update.lock"
        lock.mkdir()
        with patch("epub_tts_narrator.renderer.cleanup.cleanup_work",
                   side_effect=AssertionError("cleanup entered during analysis update")):
            with self.assertRaises(workflow.RenderBlocked):
                workflow.cleanup_render_work(self.bundle, self.config)
        self.assertTrue(lock.is_dir())
        self.assertFalse((self.book / ".audio-render.lock").exists())
        self.assertFalse((self.work / ".render.lock").exists())

    def test_cleanup_failure_releases_owned_locks_without_editing_status(self):
        self.work.mkdir()
        status = self.work / "status.json"
        status.write_text('{"status": "blocked", "error": "old failure"}\n')
        original = status.read_bytes()
        with patch("epub_tts_narrator.renderer.cleanup.cleanup_work", side_effect=OSError("fixture failure")):
            with self.assertRaisesRegex(OSError, "fixture failure"):
                workflow.cleanup_render_work(self.bundle, self.config)
        self.assertFalse((self.book / ".audio-render.lock").exists())
        self.assertFalse((self.work / ".render.lock").exists())
        self.assertEqual(status.read_bytes(), original)

    def test_keep_intermediate_audio_defaults_false_and_requires_a_boolean(self):
        self.assertIs(self.config["render"]["keep_intermediate_audio"], False)
        original = self.config_path.read_text()
        for setting, expected in (("true", True), ("false", False)):
            self.config_path.write_text(original + f"[render]\nkeep_intermediate_audio={setting}\n")
            self.assertIs(load_config(self.config_path)["render"]["keep_intermediate_audio"], expected)
            # The same global file must also remain readable by analysis.
            from epub_tts_narrator.config import load_config as load_analysis_config
            load_analysis_config(self.config_path)
        for invalid in ('"true"', "1", "0", "[]"):
            with self.subTest(invalid=invalid):
                self.config_path.write_text(original + f"[render]\nkeep_intermediate_audio={invalid}\n")
                with self.assertRaisesRegex(ValueError, "keep_intermediate_audio"):
                    load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
