"""A verification outage must not turn a restart into a new character voice."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from audiobook_renderer.bundle import load_bundle
from audiobook_renderer.casting import CastingSession
from audiobook_renderer.common import digest, read_json, write_json
from audiobook_renderer.config import DEFAULTS, select_voice_designer
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.workflow import RenderBlocked, VerificationUnavailable, _seed
from audition_fixtures import prepare_bundle


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class Designer:
    capabilities = {"reference_voice": True, "voice_design": True}

    def __init__(self, remote=False):
        self.identity = {"engine": "gemini_fixture" if remote else "local_fixture", "model": "fixture-v1"}
        self.remote, self.calls = remote, []

    def create_voice(self, prompt, text, path, *, seed):
        self.calls.append((prompt, text, str(path), seed))
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            output.writeframes((seed % 1000 + 1).to_bytes(2, "little") * 4410)
        metadata = {"description_applied": True}
        if self.remote:
            metadata["external_voice_id"] = "voice_existing_remote"
            write_json(Path(path).parent / "gemini-voice.json", {
                "version": 1, "voice_id": metadata["external_voice_id"],
                "request_fingerprint": digest({"designer": self.identity, "description": prompt})})
        return metadata


class Fixture:
    def __init__(self, test, *, gemini=False):
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = prepare_bundle(load_bundle(EXAMPLE))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "tts-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="outage-series")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"), max_attempts=2)
        self.config["quality"] = validate_policy({})
        self.engine, self.designer = Designer(), Designer(remote=True)
        if gemini:
            self.config["gemini"]["api_key"] = "fixture-only-key"
            self.config = select_voice_designer(self.config, "gemini")
        self.verifier = SimpleNamespace(identity={"backend": "fixture-asr", "version": 1})
        self.voice = self.bundle["cast"]["voices"][0]["id"]
        self.verdict, self.checked, self.notices = "unavailable", [], []
        self.test = test
        self.session = self.new_session()

    def new_session(self, *, config=None, bundle=None):
        session = CastingSession(bundle or self.bundle, config or self.config,
            engine_factory=lambda _: self.engine, designer_factory=lambda _: self.designer,
            verifier_factory=lambda *args, **kwargs: self.verifier, notify=self.notices.append)
        self.test.addCleanup(session.close)
        return session

    def report(self, path, text, *args, **kwargs):
        self.checked.append(Path(path))
        decision = self.verdict.pop(0) if isinstance(self.verdict, list) else self.verdict
        passed = decision == "passed"
        report = {"passed": passed, "purpose": kwargs["purpose"], "expected_text": text,
                  "reasons": [] if passed else ["text_mismatch"]}
        if decision in {"unavailable", "different", "uncertain"}:
            report["text_judge"] = {"decision": decision}
        return report

    def run(self, session=None, action="ensure_ready"):
        with patch("audiobook_renderer.workflow.validate_audio", side_effect=self.report), \
                patch("audiobook_renderer.casting.validate_audio", side_effect=self.report):
            return getattr(session or self.session, action)(self.voice)

    def pending(self, *, action="ensure_ready"):
        with self.test.assertRaises(VerificationUnavailable):
            self.run(action=action)
        paths = sorted((self.root / "work" / "casting-candidates").glob("*/attempt-*.json"))
        return paths[-1], read_json(paths[-1])


class JevCastingRecoveryTests(unittest.TestCase):
    def setUp(self):
        normalize = patch("audiobook_renderer.workflow.normalize_clip",
                          side_effect=lambda raw, wav, **kwargs: shutil.copyfile(raw, wav))
        normalize.start()
        self.addCleanup(normalize.stop)

    def test_restart_ensure_ready_and_generate_reuse_pending_bytes_and_nonce(self):
        for action in ("ensure_ready", "generate"):
            with self.subTest(action=action):
                fixture = Fixture(self)
                path, record = fixture.pending(action=action)
                wav = path.parent / record["audio_file"]
                original = wav.read_bytes()
                fixture.session.close()
                resumed = fixture.new_session()
                fixture.verdict = "passed"
                selected = fixture.run(resumed, action)
                self.assertEqual(len(fixture.engine.calls), 1)
                self.assertEqual(selected["candidate_id"], record["request"]["nonce"])
                self.assertEqual(selected["creation"]["request_hash"], record["request_hash"])
                self.assertEqual(wav.read_bytes(), original)
                self.assertEqual(Path(selected["reference_audio"]).read_bytes(), original)
                self.assertEqual(fixture.checked[-1], wav)
                self.assertIn("Retrying verification", "\n".join(fixture.notices))
                replacement = fixture.run(resumed, "regenerate")
                self.assertEqual(len(fixture.engine.calls), 2)
                self.assertNotEqual(replacement["candidate_id"], selected["candidate_id"])

    def test_explicit_regeneration_leaves_pending_verification_take_and_creates_new_candidate(self):
        fixture = Fixture(self)
        path, record = fixture.pending()
        wav = path.parent / record["audio_file"]
        original = wav.read_bytes()
        fixture.verdict = "passed"
        replacement = fixture.run(fixture.new_session(), "regenerate")
        self.assertEqual(len(fixture.engine.calls), 2)
        self.assertNotEqual(replacement["candidate_id"], record["request"]["nonce"])
        self.assertEqual(read_json(path), record)
        self.assertEqual(wav.read_bytes(), original)
        self.assertEqual(fixture.session.store.load(fixture.voice)["candidate_id"],
                         replacement["candidate_id"])

    def test_repeated_outage_preserves_pending_take_without_generation(self):
        fixture = Fixture(self)
        path, record = fixture.pending()
        for _ in range(2):
            with self.assertRaises(VerificationUnavailable):
                fixture.run(fixture.new_session())
        self.assertEqual(len(fixture.engine.calls), 1)
        self.assertEqual(read_json(path)["request"], record["request"])
        self.assertIsNone(fixture.session.store.load(fixture.voice))

    def test_gemini_restart_keeps_existing_remote_identity_and_skips_preparation(self):
        fixture = Fixture(self, gemini=True)
        path, record = fixture.pending()
        remote = path.parent / "gemini-voice.json"
        remote_bytes = remote.read_bytes()
        fixture.verdict = "passed"
        resumed = fixture.new_session()
        with patch.object(resumed, "_prepare_gemini_request", side_effect=AssertionError("Must retain the existing remote voice")):
            selected = fixture.run(resumed)
        self.assertEqual(len(fixture.designer.calls), 1)
        self.assertEqual(fixture.engine.calls, [])
        self.assertEqual(remote.read_bytes(), remote_bytes)
        self.assertEqual(selected["creation"]["external_voice_id"], "voice_existing_remote")
        self.assertEqual(selected["candidate_id"], record["request"]["nonce"])

    def test_changed_request_inputs_do_not_reuse_pending_candidate(self):
        for change in ("prompt", "script", "profile", "source", "seed", "engine", "designer"):
            with self.subTest(change=change):
                fixture = Fixture(self)
                path, record = fixture.pending()
                config, bundle = deepcopy(fixture.config), deepcopy(fixture.bundle)
                if change == "prompt":
                    fixture.session.update_prompt(fixture.voice, prompt="A changed character voice.")
                elif change == "script":
                    fixture.session.update_prompt(fixture.voice, reference_text="A different audition script.")
                elif change == "profile":
                    bundle["cast"]["voices"][0]["description"] = "Changed profile."
                elif change == "source":
                    bundle["manifest"]["source_fingerprint"] = "changed-source-fingerprint"
                elif change == "seed":
                    config["render"]["seed"] += 1
                elif change == "engine":
                    fixture.engine.identity = {**fixture.engine.identity, "model": "fixture-v2"}
                else:
                    config["gemini"]["api_key"] = "fixture-only-key"
                    config = select_voice_designer(config, "gemini")
                fixture.verdict = "passed"
                selected = fixture.run(fixture.new_session(config=config, bundle=bundle))
                self.assertNotEqual(selected["candidate_id"], record["request"]["nonce"])
                self.assertEqual(len(fixture.engine.calls) + len(fixture.designer.calls), 2)
                self.assertEqual(read_json(path), record)

    def test_pending_second_attempt_skips_rejected_first_after_policy_change_and_lower_limit(self):
        fixture = Fixture(self)
        fixture.verdict = ["different", "unavailable"]
        path, record = fixture.pending()
        self.assertEqual(record["attempt"], 2)
        rejected = (path.parent / "attempt-001.json").read_bytes()
        fixture.verifier.identity = {"backend": "fixture-asr", "version": 2}
        fixture.config["render"]["max_attempts"] = 1
        fixture.checked.clear()
        fixture.verdict = "passed"
        selected = fixture.run(fixture.new_session())
        self.assertEqual(fixture.checked, [path.parent / "attempt-002.wav"])
        self.assertEqual(len(fixture.engine.calls), 2)
        self.assertEqual(selected["creation"]["seed"], record["seed"])
        self.assertEqual((path.parent / "attempt-001.json").read_bytes(), rejected)

    def test_nonpending_rejected_and_accepted_takes_are_not_resumed_by_regenerate(self):
        for verdict in ("different", "uncertain", "passed"):
            with self.subTest(verdict=verdict):
                fixture = Fixture(self)
                fixture.config["render"]["max_attempts"] = 1
                fixture.session.config["render"]["max_attempts"] = 1
                fixture.verdict = verdict
                if verdict == "passed":
                    first = fixture.run(action="generate")
                    original_nonce = first["candidate_id"]
                else:
                    with self.assertRaises(RenderBlocked):
                        fixture.run(action="generate")
                    path = next((fixture.root / "work" / "casting-candidates").glob("*/attempt-001.json"))
                    original_nonce = read_json(path)["request"]["nonce"]
                fixture.verdict = "passed"
                selected = fixture.run(fixture.new_session(), "regenerate")
                self.assertNotEqual(selected["candidate_id"], original_nonce)
                self.assertEqual(len(fixture.engine.calls), 2)

    def test_tampered_pending_files_block_before_generation_or_verification(self):
        for change in ("request", "hash", "audio", "missing_audio", "audio_symlink", "directory_symlink", "record_symlink", "oversize", "oversize_audio"):
            with self.subTest(change=change):
                fixture = Fixture(self)
                path, record = fixture.pending()
                wav = path.parent / record["audio_file"]
                if change == "request":
                    record["request"]["prompt"] += " changed"
                    write_json(path, record)
                elif change == "hash":
                    record["audio_sha256"] = "0" * 64
                    write_json(path, record)
                elif change == "audio":
                    wav.write_bytes(b"changed")
                elif change == "missing_audio":
                    wav.unlink()
                elif change == "oversize_audio":
                    with wav.open("r+b") as stream:
                        stream.truncate(128 * 1024 * 1024 + 1)
                elif change in {"audio_symlink", "record_symlink"}:
                    target = wav if change == "audio_symlink" else path
                    moved = fixture.root / target.name
                    target.rename(moved)
                    target.symlink_to(moved)
                elif change == "directory_symlink":
                    moved = fixture.root / "moved-candidate"
                    path.parent.rename(moved)
                    path.parent.symlink_to(moved, target_is_directory=True)
                else:
                    path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
                fixture.checked.clear()
                with self.assertRaises(RenderBlocked):
                    fixture.run(fixture.new_session())
                self.assertEqual(len(fixture.engine.calls), 1)
                self.assertEqual(fixture.checked, [])

    def test_two_matching_pending_candidates_block_instead_of_picking_one(self):
        fixture = Fixture(self)
        path, record = fixture.pending()
        duplicate = deepcopy(record)
        duplicate["request"]["nonce"] = "f" * 32
        duplicate["request_hash"] = digest(duplicate["request"])
        duplicate["seed"] = _seed(duplicate["request"]["base_seed"], duplicate["request"], duplicate["attempt"])
        second = path.parent.parent / duplicate["request_hash"]
        second.mkdir()
        shutil.copyfile(path.parent / record["audio_file"], second / record["audio_file"])
        write_json(second / path.name, duplicate)
        fixture.checked.clear()
        with self.assertRaisesRegex(RenderBlocked, "Multiple matching"):
            fixture.run(fixture.new_session())
        self.assertEqual(len(fixture.engine.calls), 1)
        self.assertEqual(fixture.checked, [])

    def test_saved_unselected_voice_outage_pauses_without_generation(self):
        fixture = Fixture(self)
        fixture.verdict = "passed"
        candidate = fixture.run(action="generate")
        original = Path(candidate["reference_audio"]).read_bytes()
        fixture.verifier.identity = {"backend": "fixture-asr", "version": 2}
        fixture.verdict = "unavailable"
        with self.assertRaises(VerificationUnavailable):
            fixture.run(fixture.new_session())
        self.assertEqual(len(fixture.engine.calls), 1)
        self.assertEqual(Path(candidate["reference_audio"]).read_bytes(), original)
        self.assertIsNone(fixture.session.store.load(fixture.voice))

    def test_published_candidate_is_not_resumed_even_if_work_report_looks_pending(self):
        fixture = Fixture(self)
        fixture.verdict = "passed"
        candidate = fixture.run(action="generate")
        directory = fixture.root / "work" / "casting-candidates" / candidate["creation"]["request_hash"]
        (directory / "accepted.json").unlink()
        path = directory / "attempt-001.json"
        record = read_json(path)
        record["report"].update(passed=False, text_judge={"decision": "unavailable"})
        write_json(path, record)
        selected = fixture.run(fixture.new_session(), "regenerate")
        self.assertNotEqual(selected["candidate_id"], candidate["candidate_id"])
        self.assertEqual(len(fixture.engine.calls), 2)

    def test_explicit_recovery_outage_preserves_audio_and_pauses(self):
        fixture = Fixture(self)
        path, record = fixture.pending()
        wav = path.parent / record["audio_file"]
        original = wav.read_bytes()
        with patch("audiobook_renderer.casting.validate_audio", side_effect=fixture.report):
            with self.assertRaises(VerificationUnavailable):
                fixture.new_session().recover_attempt(fixture.voice, path)
        self.assertEqual(wav.read_bytes(), original)
        self.assertEqual(len(fixture.engine.calls), 1)
        self.assertEqual(read_json(path), record)


if __name__ == "__main__":
    unittest.main()
