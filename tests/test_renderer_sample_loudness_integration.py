"""Audition normalization is automatic, model-free, and separate from references."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.audition import AudioPlayer, _prepare_playback_samples, run_casting
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.cli import export_bundle
from test_contracts import valid_document
from test_renderer_audition import PlayerDouble, SessionDouble, SharedSessionDouble


class SamplePlaybackIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sample = self.root / "original.wav"
        self.sample.write_bytes(b"original reference")
        self.preview = self.root / "preview.wav"
        self.preview.write_bytes(b"normalized preview")

    def test_player_plays_normalized_copy_and_leaves_reference_unchanged(self):
        normalizer = Mock()
        normalizer.prepare.return_value = self.preview
        process = Mock()
        process.poll.return_value = 0
        process.returncode = 0
        player = AudioPlayer(["afplay"], normalizer=normalizer)
        with patch("epub_tts_narrator.renderer.audition.shutil.which", return_value="/usr/bin/afplay"), \
                patch("epub_tts_narrator.renderer.audition.subprocess.Popen", return_value=process) as popen:
            player.start(self.sample)
            player.stop()
        normalizer.prepare.assert_called_once_with(self.sample)
        self.assertEqual(popen.call_args.args[0], ["/usr/bin/afplay", str(self.preview)])
        self.assertEqual(self.sample.read_bytes(), b"original reference")

    def test_player_normalization_failure_never_falls_back_to_unmatched_volume(self):
        normalizer = Mock()
        normalizer.prepare.side_effect = ValueError("silent sample")
        player = AudioPlayer(["afplay"], normalizer=normalizer)
        with patch("epub_tts_narrator.renderer.audition.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "silent sample"):
                player.start(self.sample)
        popen.assert_not_called()
        self.assertIsNone(player.process)
        self.assertIsNone(player.errors)

    def test_prewarm_includes_inherited_and_candidate_takes_but_deduplicates_aliases(self):
        session = SharedSessionDouble()
        session.row("v1")["selected"]["inherited"] = True
        session.generate("v1")  # A previous/unselected take still needs batch preparation.
        player = PlayerDouble()
        player.prepare = Mock()
        output = []
        ready, failed = _prepare_playback_samples(
            player, session.rows(), output.append, include_candidates=True)
        self.assertEqual((ready, failed), (3, 0))
        paths = [call.args[0] for call in player.prepare.call_args_list]
        self.assertEqual(len(set(paths)), 3)
        self.assertIn("/fixtures/v1-1.wav", paths)
        self.assertIn("/fixtures/v1-2.wav", paths)
        self.assertIn("/fixtures/v2-1.wav", paths)
        self.assertFalse(any(event[0] == "regenerate" for event in session.events))

    def test_interactive_prepares_after_missing_voices_before_first_menu_and_continues_on_error(self):
        session = SessionDouble()
        player = PlayerDouble()
        events, output = [], []

        def prepare(path):
            self.assertTrue(all(row["selected"] is not None for row in session.rows()))
            events.append(path)
            if path.endswith("v1-1.wav"):
                raise ValueError("preview failed")

        player.prepare = prepare

        def answer(_prompt):
            self.assertEqual(events, ["/fixtures/v1-1.wav", "/fixtures/vn-1.wav"])
            return "q"

        run_casting(session, player, input_fn=answer, output=output.append)
        self.assertTrue(any("Could not prepare playback for Mara" in line for line in output))
        self.assertTrue(any("Voice casting — 2/2 ready" in line for line in output))
        self.assertTrue(all(row["selected"] is not None for row in session.rows()))


class SampleNormalizationCommandTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.config_path = self.root / "settings.toml"
        self.base_config = (
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            '[voices]\nseries_id="test-series"\n'
        )
        self.config_path.write_text(self.base_config)
        self.bundle = self.root / "bundle"
        export_bundle(valid_document(), self.bundle)

    def run_cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["cast", str(self.bundle), "--config", str(self.config_path), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def saved_candidates(self):
        source = self.root / "reference.wav"
        source.write_bytes(b"original reference fixture")
        store = VoiceStore(self.root / "voices", "test-series")
        args = ("v1", valid_document()["cast"]["voices"][1], source, "Hello there.",
                {"engine": "fixture"}, {"passed": True}, {"description_applied": True})
        first = store.publish(*args)
        candidate = store.publish_candidate(*args, candidate_id="a" * 32)
        store.select_candidate("v1", candidate["candidate_id"])
        return store, [first, candidate]

    @staticmethod
    def snapshot(path):
        return {str(file.relative_to(path)): file.read_bytes()
                for file in path.rglob("*") if file.is_file()}

    def test_batch_prepares_existing_candidates_without_models_playback_or_asset_changes(self):
        store, candidates = self.saved_candidates()
        before = self.snapshot(store.directory)
        with patch("epub_tts_narrator.renderer.loudness.SampleNormalizer") as normalizer, \
                patch("epub_tts_narrator.renderer.engines.create_engine") as engine, \
                patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier, \
                patch("epub_tts_narrator.renderer.audition.subprocess.Popen") as popen:
            code, output, error = self.run_cli("--action", "normalize-samples")
        self.assertEqual(code, 0, error)
        normalizer.assert_called_once_with(
            self.root / "render-work" / "voice-previews",
            target_lufs=-20.0, true_peak_db=-1.5, ffmpeg="ffmpeg")
        self.assertEqual({call.args[0] for call in normalizer.return_value.prepare.call_args_list},
                         {Path(candidate["reference_audio"]) for candidate in candidates})
        self.assertEqual(normalizer.return_value.prepare.call_count, 2)
        self.assertEqual(self.snapshot(store.directory), before)
        self.assertIn("2 sample(s)", output)
        engine.assert_not_called()
        verifier.assert_not_called()
        popen.assert_not_called()

    def test_batch_reports_failure_after_trying_remaining_candidates(self):
        store, _ = self.saved_candidates()
        before = self.snapshot(store.directory)
        with patch("epub_tts_narrator.renderer.loudness.SampleNormalizer") as normalizer:
            normalizer.return_value.prepare.side_effect = [ValueError("cannot measure loudness"), Path("ready.wav")]
            code, output, error = self.run_cli("--action", "normalize-samples")
        self.assertEqual(code, 2)
        self.assertEqual(normalizer.return_value.prepare.call_count, 2)
        self.assertIn("Could not prepare playback", output)
        self.assertIn("1 playback copies are ready", error)
        self.assertEqual(self.snapshot(store.directory), before)

    def test_play_constructs_normalizer_with_global_settings_and_book_work_override(self):
        self.saved_candidates()
        self.config_path.write_text(self.base_config + (
            '[casting]\nsample_loudness_lufs=-24.0\nsample_true_peak_db=-3.0\n'
            '[render]\nffmpeg="custom-ffmpeg"\n'))
        work = self.root / "custom-work"
        with patch("epub_tts_narrator.renderer.loudness.SampleNormalizer") as normalizer, \
                patch("epub_tts_narrator.renderer.audition.AudioPlayer") as player:
            code, _, error = self.run_cli("--action", "play", "--voice-id", "v1", "--work-dir", str(work))
        self.assertEqual(code, 0, error)
        normalizer.assert_called_once_with(work / "voice-previews", target_lufs=-24.0,
                                           true_peak_db=-3.0, ffmpeg="custom-ffmpeg")
        self.assertIs(player.call_args.kwargs["normalizer"], normalizer.return_value)

    def test_disabled_normalization_uses_raw_playback_and_rejects_batch(self):
        self.saved_candidates()
        self.config_path.write_text(self.base_config + '[casting]\nnormalize_samples=false\n')
        with patch("epub_tts_narrator.renderer.loudness.SampleNormalizer") as normalizer, \
                patch("epub_tts_narrator.renderer.audition.AudioPlayer") as player:
            code, _, error = self.run_cli("--action", "play", "--voice-id", "v1")
        self.assertEqual(code, 0, error)
        normalizer.assert_not_called()
        self.assertIsNone(player.call_args.kwargs["normalizer"])
        code, _, error = self.run_cli("--action", "normalize-samples")
        self.assertEqual(code, 2)
        self.assertIn("casting.normalize_samples = true", error)

    def test_batch_rejects_per_voice_and_generation_flags_before_session_creation(self):
        for args in (("--voice-id", "v1"), ("--candidate-id", "a" * 32),
                     ("--prompt-file", "missing.txt"), ("--text-file", "missing.txt")):
            with self.subTest(args=args), patch("epub_tts_narrator.renderer.casting.CastingSession") as session:
                code, _, error = self.run_cli("--action", "normalize-samples", *args)
                self.assertEqual(code, 2)
                self.assertIn(args[0], error)
                session.assert_not_called()

    def test_loudness_settings_defaults_and_allowed_bounds(self):
        config = load_config(self.config_path)
        self.assertIs(config["casting"]["normalize_samples"], True)
        self.assertEqual(config["casting"]["sample_loudness_lufs"], -20.0)
        self.assertEqual(config["casting"]["sample_true_peak_db"], -1.5)
        for loudness, peak in ((-40, -9), (-10, 0), (-20.5, -2.5)):
            self.config_path.write_text(self.base_config + (
                f'[casting]\nsample_loudness_lufs={loudness}\nsample_true_peak_db={peak}\n'))
            self.assertEqual(load_config(self.config_path)["casting"]["sample_loudness_lufs"], loudness)

    def test_loudness_settings_reject_invalid_types_nonfinite_and_bounds(self):
        invalid = {
            "normalize_samples": ("1", '"true"'),
            "sample_loudness_lufs": ("true", '"-20"', "nan", "inf", "-41", "-9"),
            "sample_true_peak_db": ("false", '"-1.5"', "nan", "-inf", "-10", "0.1"),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.config_path.write_text(self.base_config + f"[casting]\n{key}={value}\n")
                    with self.assertRaisesRegex(ValueError, "casting." + key):
                        load_config(self.config_path)
