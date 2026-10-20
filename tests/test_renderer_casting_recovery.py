"""Saved rejected takes are rechecked without resynthesis or lost provenance."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from audiobook_renderer.bundle import load_bundle
from audiobook_renderer.audition import casting_command
from audiobook_renderer.casting import CastingSession
from audiobook_renderer.cli import parser
from audiobook_renderer.common import digest, file_hash, read_json, write_json
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy, validation_key
from audiobook_renderer.workflow import RenderBlocked


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class ReferenceVerifier:
    identity = {"engine": "fixture-asr", "version": 1}

    def __init__(self, text):
        self.text = text
        self.calls = []

    def transcribe(self, path):
        self.calls.append(str(path))
        # These overlapping timestamps reproduce the casting rejection. The
        # independent lexical result is complete; references do not need sync.
        return {"text": self.text, "segments": [{
            "text": self.text, "start": 0.0, "end": 1.0,
            "avg_logprob": -0.1, "no_speech_prob": 0.001,
            "words": [{"word": word, "start": 0.0, "end": 1.0, "probability": 0.99}
                      for word in self.text.split()],
        }]}


class CorroboratingReferenceVerifier(ReferenceVerifier):
    identity = {"engine": "fixture-asr", "version": 2,
                "corroboration": {"engine": "independent-reference-asr", "version": 1}}
    corroboration_identity = identity["corroboration"]

    def __init__(self, text):
        super().__init__(text)
        self.corroboration_calls = []

    def corroborate_reference(self, path):
        self.corroboration_calls.append(str(path))
        return {"text": self.text, "segments": [{"text": self.text}]}


class CastingRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = load_bundle(EXAMPLE)
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "tts-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="recovery-series")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"))
        self.config["quality"] = validate_policy({})
        self.voice_id = self.bundle["cast"]["voices"][0]["id"]
        self.profile = deepcopy(self.bundle["cast"]["voices"][0])
        self.text = self.config["voices"]["reference_text"]
        self.verifier = ReferenceVerifier(self.text)
        self.engine = SimpleNamespace(
            identity={"engine": "fixture-tts", "model": "fixture-v1",
                      "checkpoint_files_sha256": {"weights": "unchanged"}},
            create_voice=Mock(side_effect=AssertionError("Recovery must not synthesize")),
        )
        self.designer_factory = Mock(side_effect=AssertionError("Recovery must not load a designer"))
        self.session = CastingSession(
            self.bundle, self.config, engine_factory=lambda _: self.engine,
            verifier_factory=lambda *args, **kwargs: self.verifier,
            designer_factory=self.designer_factory, notify=lambda _: None)
        self.addCleanup(self.session.close)
        # Real earlier casting runs persisted their original prompt before
        # generating an attempt. Keep that legacy description-only draft so a
        # changed default never rewrites the experiment being recovered.
        self.session.update_prompt(self.voice_id, self.profile["description"], self.text)

    def attempt(self, number=1):
        request = {
            "kind": "casting_candidate", "nonce": "a" * 32,
            "engine": deepcopy(self.engine.identity),
            "designer": {"engine": "fixture-designer", "model": "fixture-design-v1"},
            "series_id": self.session.store.series_id, "voice_id": self.voice_id,
            "profile": deepcopy(self.profile), "prompt": self.profile["description"],
            "text": self.text, "base_seed": 1729,
        }
        directory = self.root / "work" / "casting-candidates" / digest(request)
        directory.mkdir(parents=True, exist_ok=True)
        wav = directory / f"attempt-{number:03}.wav"
        rate = 44100
        samples = [round(4000 * math.sin(2 * math.pi * (210 + number) * i / rate))
                   for i in range(rate)]
        with wave.open(str(wav), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack("<" + "h" * rate, *samples))
        record = {
            "request_hash": digest(request), "request": request, "attempt": number,
            "seed": 1800 + number, "validation_key": "old-speech-key",
            "report": {"passed": False, "reasons": ["invalid_word_timestamps"]},
            "engine_metadata": {"description_applied": True,
                                "requested_description": request["prompt"],
                                "reference_text": self.text},
            "audio_file": wav.name, "audio_sha256": file_hash(wav),
        }
        path = directory / f"attempt-{number:03}.json"
        write_json(path, record)
        return path

    def rewrite(self, path, change, *, update_request_hash=False):
        record = read_json(path)
        change(record)
        if update_request_hash:
            record["request_hash"] = digest(record["request"])
        write_json(path, record)

    def test_recovery_revalidates_selects_and_preserves_original_files(self):
        path = self.attempt()
        original = read_json(path)
        original_bytes = path.read_bytes()
        recovered = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(recovered["reference_sha256"], original["audio_sha256"])
        self.assertEqual(recovered["reference_text"], self.text)
        self.assertEqual(recovered["verification"]["purpose"], "reference")
        self.assertEqual(recovered["creation"]["validation_key"], validation_key(
            self.text, self.verifier, self.config["quality"], purpose="reference"))
        self.assertEqual(recovered["creation"]["recovered_from"]["attempt_sha256"], file_hash(path))
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], recovered["asset_hash"])
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(file_hash(path.parent / original["audio_file"]), original["audio_sha256"])
        self.assertEqual(len(self.verifier.calls), 1)
        self.engine.create_voice.assert_not_called()
        self.designer_factory.assert_not_called()

    def test_repeated_recovery_keeps_one_immutable_candidate(self):
        path = self.attempt()
        first = self.session.recover_attempt(self.voice_id, path)
        second = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(first["asset_hash"], second["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 1)
        self.assertEqual(len(self.verifier.calls), 2)

    def test_corroborated_recovery_saves_actual_transcript_without_rewriting_original_request(self):
        path = self.attempt()
        original_bytes = path.read_bytes()
        original = read_json(path)
        audio = path.parent / original["audio_file"]
        actual_text = self.text.replace("wind in the trees", "wind and the trees")
        self.assertNotEqual(actual_text, self.text)
        self.verifier = CorroboratingReferenceVerifier(actual_text)

        first = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(first["reference_text"], actual_text)
        self.assertEqual(first["creation"]["reference_text"], actual_text)
        self.assertEqual(first["creation"]["requested_reference_text"], self.text)
        self.assertEqual(first["creation"]["validation_key"], validation_key(
            actual_text, self.verifier, self.config["quality"], purpose="reference"))
        self.assertTrue(first["verification"]["reference_text_adjusted"])
        self.assertEqual(first["verification"]["corroboration"]["decision"], "agreed_reference_transcript")
        self.assertEqual(self.session.rows()[0]["reference_text"], self.text)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(read_json(path)["request"], original["request"])
        self.assertEqual(read_json(path)["report"], original["report"])
        self.assertEqual(file_hash(audio), original["audio_sha256"])
        self.assertEqual(first["reference_sha256"], original["audio_sha256"])

        second = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(second["asset_hash"], first["asset_hash"])
        self.assertEqual(second["candidate_id"], first["candidate_id"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 1)
        self.assertEqual(len(self.verifier.corroboration_calls), 2)
        self.engine.create_voice.assert_not_called()
        self.designer_factory.assert_not_called()

    def test_new_generation_saves_corroborated_transcript_and_retains_requested_script(self):
        path = self.attempt()
        source = path.parent / read_json(path)["audio_file"]
        actual_text = self.text.replace("wind in the trees", "wind and the trees")
        self.verifier = CorroboratingReferenceVerifier(actual_text)
        self.engine.capabilities = {"reference_voice": True, "voice_design": True}

        def create_voice(prompt, text, destination, *, seed):
            shutil.copyfile(source, destination)
            return {"description_applied": True, "requested_description": prompt,
                    "reference_text": text, "seed": seed}

        self.engine.create_voice = Mock(side_effect=create_voice)
        with patch("audiobook_renderer.workflow.normalize_clip",
                   side_effect=lambda source, destination, **kwargs: shutil.copyfile(source, destination)):
            saved = self.session.regenerate(self.voice_id)
        self.assertEqual(saved["reference_text"], actual_text)
        self.assertEqual(saved["creation"]["requested_reference_text"], self.text)
        self.assertEqual(saved["creation"]["validation_key"], validation_key(
            actual_text, self.verifier, self.config["quality"], purpose="reference"))
        self.assertEqual(saved["verification"]["corroboration"]["decision"], "agreed_reference_transcript")
        self.assertEqual(self.session.rows()[0]["reference_text"], self.text)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], saved["asset_hash"])
        self.engine.create_voice.assert_called_once()
        self.assertEqual(len(self.verifier.corroboration_calls), 1)
        generated_records = list((self.root / "work" / "casting-candidates").glob("*/accepted.json"))
        self.assertEqual(len(generated_records), 1)
        accepted = read_json(generated_records[0])
        self.assertEqual(accepted["request"]["text"], self.text)
        self.assertEqual(accepted["report"]["reference_text"], actual_text)

    def test_accept_cannot_relabel_an_immutable_binding_after_transcript_correction(self):
        original = self.session.recover_attempt(self.voice_id, self.attempt())
        audio_before = file_hash(original["reference_audio"])
        audit_directory = self.session.store.directory / "casting-acceptance"
        audits_before = sorted(path.name for path in audit_directory.glob("*.json"))
        actual_text = self.text.replace("wind in the trees", "wind and the trees")
        self.verifier = CorroboratingReferenceVerifier(actual_text)
        # Model a reopened casting session using the updated verifier.
        self.session._verifier = self.verifier

        with self.assertRaisesRegex(RenderBlocked, "corrected reference transcript"):
            self.session.accept(self.voice_id, original["candidate_id"])
        preserved = self.session.store.load_candidate(self.voice_id, original["candidate_id"])
        self.assertEqual(preserved["asset_hash"], original["asset_hash"])
        self.assertEqual(preserved["reference_text"], self.text)
        self.assertEqual(file_hash(preserved["reference_audio"]), audio_before)
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], original["asset_hash"])
        self.assertEqual(sorted(path.name for path in audit_directory.glob("*.json")), audits_before)
        self.assertEqual(len(self.verifier.corroboration_calls), 1)

    def test_new_transcript_decision_creates_a_separate_recovery_candidate(self):
        path = self.attempt()
        original_bytes = path.read_bytes()
        actual_text = self.text.replace("wind in the trees", "wind and the trees")
        self.verifier = CorroboratingReferenceVerifier(actual_text)
        first = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(first["reference_text"], actual_text)
        self.assertEqual(len(self.verifier.corroboration_calls), 1)

        self.verifier.text = self.text
        second = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(second["reference_text"], self.text)
        self.assertNotEqual(second["candidate_id"], first["candidate_id"])
        self.assertEqual(first["reference_sha256"], second["reference_sha256"])
        self.assertEqual(self.session.store.load_candidate(
            self.voice_id, first["candidate_id"])["asset_hash"], first["asset_hash"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], second["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 2)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(len(self.verifier.corroboration_calls), 1)
        repeated = self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(repeated["asset_hash"], second["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 2)
        self.engine.create_voice.assert_not_called()

    def test_other_successful_attempt_is_saved_without_changing_selection(self):
        first = self.session.recover_attempt(self.voice_id, self.attempt(1))
        second = self.session.recover_attempt(self.voice_id, self.attempt(2), select=False)
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 2)

    def test_failed_recheck_preserves_existing_selection_and_does_not_publish(self):
        first = self.session.recover_attempt(self.voice_id, self.attempt(1))
        self.verifier.text = "Invented unrelated words."
        with self.assertRaisesRegex(RenderBlocked, "text_mismatch"):
            self.session.recover_attempt(self.voice_id, self.attempt(2))
        self.assertEqual(self.session.store.load(self.voice_id)["asset_hash"], first["asset_hash"])
        self.assertEqual(len(self.session.store.list_candidates(self.voice_id)), 1)

    def test_newer_draft_prevents_selection_but_allows_retaining_old_experiment(self):
        path = self.attempt()
        self.session.update_prompt(self.voice_id, prompt="A newer voice direction.")
        with self.assertRaisesRegex(RenderBlocked, "current prompt"):
            self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(self.verifier.calls, [])
        candidate = self.session.recover_attempt(self.voice_id, path, select=False)
        self.assertEqual(candidate["creation"]["requested_description"], self.profile["description"])
        self.assertIsNone(self.session.store.load(self.voice_id))
        self.assertEqual(self.session.rows()[0]["prompt"], "A newer voice direction.")

    def test_request_and_audio_checksums_reject_changes_before_asr(self):
        for field in ("request", "audio"):
            with self.subTest(field=field):
                path = self.attempt()
                if field == "request":
                    self.rewrite(path, lambda d: d["request"].update(text="Changed sample."))
                else:
                    (path.parent / read_json(path)["audio_file"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    self.session.recover_attempt(self.voice_id, path)
                self.assertEqual(self.verifier.calls, [])
                self.assertIsNone(self.session.store.load(self.voice_id))

    def test_foreign_voice_series_profile_and_engine_weights_are_rejected(self):
        cases = [
            ("series", lambda r: r.update(series_id="different-series"), "different series or voice"),
            ("voice", lambda r: r.update(voice_id="different-voice"), "different series or voice"),
            ("profile", lambda r: r["profile"].update(description="Different character"), "profile"),
            ("engine", lambda r: r["engine"].update(model="other-model"), "different engine or model"),
            ("weights", lambda r: r["engine"].update(checkpoint_files_sha256={}), "different model weights"),
        ]
        for label, change, error in cases:
            with self.subTest(case=label):
                path = self.attempt()
                self.rewrite(path, lambda record: change(record["request"]), update_request_hash=True)
                with self.assertRaisesRegex(ValueError, error):
                    self.session.recover_attempt(self.voice_id, path)
                self.assertEqual(self.verifier.calls, [])

    def test_uncontrolled_voice_cannot_be_recovered_as_designed(self):
        path = self.attempt()
        self.rewrite(path, lambda d: d["engine_metadata"].update(description_applied=False))
        with self.assertRaisesRegex(RenderBlocked, "applied voice description"):
            self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(self.verifier.calls, [])

    def test_audio_traversal_and_symlink_are_rejected(self):
        path = self.attempt()
        self.rewrite(path, lambda d: d.update(audio_file="../outside.wav"))
        with self.assertRaisesRegex(ValueError, "audio path"):
            self.session.recover_attempt(self.voice_id, path)
        path = self.attempt()
        wav = path.parent / read_json(path)["audio_file"]
        target = self.root / "outside.wav"
        wav.rename(target)
        wav.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            self.session.recover_attempt(self.voice_id, path)
        self.assertEqual(self.verifier.calls, [])


class CastingRecoveryCommandTests(unittest.TestCase):
    def arguments(self, *options):
        return parser().parse_args([
            "cast", str(EXAMPLE), "--config", "renderer.toml", *options])

    def test_recover_action_selects_saved_take_by_default(self):
        args = self.arguments("--action", "recover", "--voice-id", "voice-1",
                              "--attempt-file", "/tmp/attempt-001.json")
        session = Mock()
        session.rows.return_value = [{"voice_id": "voice-1"}]
        session.recover_attempt.return_value = {
            "candidate_id": "a" * 32, "reference_audio": "/tmp/saved-reference.wav"}
        output = StringIO()
        with (patch("audiobook_renderer.casting.CastingSession", return_value=session),
              patch("audiobook_renderer.audition.AudioPlayer") as player,
              redirect_stdout(output)):
            result = casting_command(args, {}, {"casting": {"player": ["afplay"]}})
        self.assertEqual(result, 0)
        session.recover_attempt.assert_called_once_with(
            "voice-1", Path("/tmp/attempt-001.json"), select=True)
        self.assertIn("Recovered voice saved and selected", output.getvalue())
        self.assertIn("a" * 32, output.getvalue())
        session.regenerate.assert_not_called()
        player.return_value.start.assert_not_called()
        session.close.assert_called_once()

    def test_keep_unselected_saves_alternative_without_selection(self):
        args = self.arguments("--action", "recover", "--voice-id", "voice-1",
                              "--attempt-file", "/tmp/attempt-001.json", "--keep-unselected")
        session = Mock()
        session.rows.return_value = [{"voice_id": "voice-1"}]
        session.recover_attempt.return_value = {
            "candidate_id": "a" * 32, "reference_audio": "/tmp/saved-reference.wav"}
        output = StringIO()
        with (patch("audiobook_renderer.casting.CastingSession", return_value=session),
              patch("audiobook_renderer.audition.AudioPlayer"),
              redirect_stdout(output)):
            self.assertEqual(casting_command(args, {}, {"casting": {"player": ["afplay"]}}), 0)
        session.recover_attempt.assert_called_once_with(
            "voice-1", Path("/tmp/attempt-001.json"), select=False)
        self.assertIn("without changing the active voice", output.getvalue())

    def test_invalid_recovery_options_fail_before_opening_a_session(self):
        cases = [
            (("--action", "recover", "--attempt-file", "/tmp/attempt.json"), "--voice-id"),
            (("--action", "recover", "--voice-id", "voice-1"), "--attempt-file is required"),
            (("--action", "list", "--attempt-file", "/tmp/attempt.json"), "--attempt-file is used"),
            (("--action", "list", "--keep-unselected"), "--keep-unselected is used"),
            (("--action", "recover", "--voice-id", "voice-1", "--attempt-file", "/tmp/a.json",
              "--candidate-id", "a" * 32), "--candidate-id"),
            (("--action", "recover", "--voice-id", "voice-1", "--attempt-file", "/tmp/a.json",
              "--text-file", "/tmp/text.txt"), "--prompt-file and --text-file"),
        ]
        for options, error in cases:
            with self.subTest(options=options):
                with patch("audiobook_renderer.casting.CastingSession") as factory:
                    with self.assertRaisesRegex(ValueError, error):
                        casting_command(self.arguments(*options), {}, {})
                    factory.assert_not_called()

    def test_recovery_failure_is_not_reported_as_success_and_resources_close(self):
        args = self.arguments("--action", "recover", "--voice-id", "voice-1",
                              "--attempt-file", "/tmp/attempt-001.json")
        session = Mock()
        session.rows.return_value = [{"voice_id": "voice-1"}]
        session.recover_attempt.side_effect = RenderBlocked("text_mismatch")
        output = StringIO()
        with (patch("audiobook_renderer.casting.CastingSession", return_value=session),
              patch("audiobook_renderer.audition.AudioPlayer") as player,
              redirect_stdout(output)):
            with self.assertRaisesRegex(RenderBlocked, "text_mismatch"):
                casting_command(args, {}, {"casting": {"player": ["afplay"]}})
        self.assertNotIn("Recovered voice", output.getvalue())
        session.close.assert_called_once()
        player.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
