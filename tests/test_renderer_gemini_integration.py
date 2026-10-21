"""Gemini casting selection leaves local rendering and saved voices intact."""

import base64
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.book_context import resolve_book_config
from audiobook_renderer.bundle import load_bundle
from audition_fixtures import prepare_bundle, prepare_document
from audiobook_renderer.casting import CastingSession
from audiobook_renderer.cli import main, parser
from audiobook_renderer.common import read_json, write_json
from audiobook_renderer.config import DEFAULTS, load_config, select_voice_designer, resolve_voice_design
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.voice_prompts import AUDIOBOOK_VOICE_CONTEXT, audiobook_voice_prompt
from audiobook_renderer.gemini_voice_design import GeminiVoiceDesigner, _RequestFailure
from epub_audiobook.cli import export_bundle
from epub_audiobook.config import load_config as load_analyzer_config
from test_contracts import valid_document
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferenceDesigner, ReferencePlaybackEngine, TranscriptVerifier,
)


SECRET = "fixture-gemini-key-do-not-echo"


class GeminiSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config_path = self.root / "epub-audiobook.toml"
        self.base_config = ('[engine]\ncheckpoint="./tts-model"\n'
                            '[verification]\nmodel="./asr-model"\n')
        self.config_path.write_text(self.base_config)
        self.bundle = self.root / "bundle"
        export_bundle(prepare_document(valid_document()), self.bundle)

    def add_gemini(self, fields=None):
        self.config_path.write_text(self.base_config + "[gemini]\n" + (
            fields if fields is not None else f'api_key="{SECRET}"\n'))

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_local_default_requires_no_gemini_credentials(self):
        config = load_config(self.config_path)
        self.assertEqual(config["voice_design"]["type"], "auto")
        self.assertEqual(resolve_voice_design(config)["type"], "engine")
        self.assertIsNone(config["gemini"]["api_key"])
        selected = select_voice_designer(config, None)
        self.assertEqual(selected, config)
        self.assertIsNot(selected, config)
        selected["engine"]["type"] = "fixture"
        self.assertNotEqual(selected["engine"], config["engine"])
        self.assertEqual(select_voice_designer(config, "local")["voice_design"], {"type": "engine"})

    def test_gemini_is_selected_per_run_from_global_file(self):
        self.add_gemini('api_key="' + SECRET + '"\nmodel="gemini-3.8-flash-tts"\n'
                        'language_code="en-GB"\ntimeout_seconds=150\n')
        original = self.config_path.read_bytes()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "wrong-environment-key"}):
            config = load_config(self.config_path)
            selected = select_voice_designer(config, "gemini")
        self.assertEqual(selected["voice_design"], {"type": "gemini", **config["gemini"]})
        self.assertEqual(selected["voice_design"]["api_key"], SECRET)
        self.assertEqual(selected["voice_design"]["language_code"], "en-GB")
        self.assertEqual(selected["voice_design"]["timeout_seconds"], 150)
        self.assertEqual(config["voice_design"]["type"], "auto")
        self.assertEqual(self.config_path.read_bytes(), original)

    def test_explicit_gemini_requires_file_credentials_even_when_environment_has_key(self):
        config = load_config(self.config_path)
        with patch.dict("os.environ", {"GEMINI_API_KEY": SECRET, "GOOGLE_API_KEY": SECRET}):
            with self.assertRaisesRegex(ValueError, r"gemini\.api_key") as raised:
                select_voice_designer(config, "gemini")
        self.assertNotIn(SECRET, str(raised.exception))
        for value in (None, "", "   "):
            config["gemini"]["api_key"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"gemini\.api_key"):
                select_voice_designer(config, "gemini")

    def test_omitted_choice_preserves_existing_local_designer(self):
        config = deepcopy(DEFAULTS)
        config["voice_design"].update(type="qwen3_mlx", checkpoint="/fixture/local-model")
        selected = select_voice_designer(config, None)
        self.assertEqual(selected["voice_design"], config["voice_design"])
        local = select_voice_designer(config, "local")
        self.assertEqual(local["voice_design"], {"type": "engine"})
        self.assertEqual(config["voice_design"]["type"], "qwen3_mlx")

    def test_combined_global_config_is_accepted_by_analyzer(self):
        self.add_gemini()
        with self.config_path.open("a") as stream:
            stream.write('[provider]\nmodel="fixture-analysis-model"\napi_key="fixture-analysis-key"\n')
        renderer = load_config(self.config_path)
        analyzer = load_analyzer_config(self.config_path)
        self.assertEqual(renderer["gemini"]["api_key"], SECRET)
        self.assertEqual(analyzer["provider"]["model"], "fixture-analysis-model")
        self.assertNotIn("gemini", analyzer)

    def test_invalid_gemini_settings_do_not_echo_key(self):
        for field in ('timeout_seconds="' + SECRET + '"', 'model=42', 'language_code=[]'):
            with self.subTest(field=field):
                self.add_gemini(f'api_key="{SECRET}"\n{field}\n')
                with self.assertRaises(ValueError) as raised:
                    load_config(self.config_path)
                self.assertNotIn(SECRET, str(raised.exception))

    def test_parser_accepts_designer_option_only_for_cast(self):
        args = parser().parse_args(["cast", str(self.bundle)])
        self.assertIsNone(args.voice_designer)
        for choice in ("local", "gemini"):
            args = parser().parse_args(["cast", str(self.bundle), "--voice-designer", choice])
            self.assertEqual(args.voice_designer, choice)
        for command in ("render", "create-voices", "voices", "plan"):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parser().parse_args([command, str(self.bundle), "--voice-designer", "gemini"])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_passes_explicit_gemini_to_casting_without_editing_config(self):
        self.add_gemini()
        original = self.config_path.read_bytes()
        with patch("audiobook_renderer.audition.casting_command", return_value=0) as casting:
            status, stdout, stderr = self.run_cli(
                "cast", self.bundle, "--config", self.config_path,
                "--voice-designer", "gemini", "--action", "generate-missing")
        self.assertEqual(status, 0, stderr)
        args, _, config = casting.call_args.args
        self.assertEqual(args.action, "generate-missing")
        self.assertEqual(config["voice_design"]["type"], "gemini")
        self.assertEqual(config["voice_design"]["api_key"], SECRET)
        self.assertEqual(config["engine"]["type"], DEFAULTS["engine"]["type"])
        self.assertNotIn(SECRET, stdout + stderr)
        self.assertEqual(self.config_path.read_bytes(), original)

    def test_cli_default_prefers_gemini_with_configured_credentials(self):
        self.add_gemini()
        with patch("audiobook_renderer.audition.casting_command", return_value=0) as casting:
            status, stdout, stderr = self.run_cli(
                "cast", self.bundle, "--config", self.config_path, "--action", "generate-missing")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(resolve_voice_design(casting.call_args.args[2])["type"], "gemini")
        self.assertNotIn(SECRET, stdout + stderr)

    def test_cli_missing_gemini_key_stops_before_casting(self):
        with patch("audiobook_renderer.audition.casting_command") as casting:
            status, stdout, stderr = self.run_cli(
                "cast", self.bundle, "--config", self.config_path,
                "--voice-designer", "gemini", "--action", "generate-missing")
        self.assertEqual(status, 2)
        self.assertIn("gemini.api_key", stderr)
        self.assertNotIn(SECRET, stdout + stderr)
        casting.assert_not_called()


class GeminiFixtureDesigner(ReferenceDesigner):
    identity = {"engine": "gemini_voice_design", "model": "gemini-3.8-flash-tts"}
    capabilities = {"voice_design": True}

    def create_voice(self, prompt, text, path, *, seed):
        result = super().create_voice(prompt, text, path, seed=seed)
        return {**result, "external_voice_id": "voice_fixture_gemini"}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class GeminiCastingIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voices"]["series_id"] = "gemini-casting-fixture"
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.config["gemini"]["api_key"] = SECRET
        self.config = select_voice_designer(self.config, "gemini")
        self.transcripts = {}
        self.engine = ReferencePlaybackEngine(self.transcripts)
        self.designer = GeminiFixtureDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.factory_calls = []
        self.notices = []
        self.bundle = self.make_bundle("book-one")

    def make_bundle(self, name):
        destination = self.root / name
        export_bundle(read_json(EXAMPLE / "reviewed.json"), destination)
        return prepare_bundle(load_bundle(destination))

    def session(self, bundle=None, config=None, designer_factory=None):
        def factory(settings):
            self.factory_calls.append(deepcopy(settings))
            return self.designer
        session = CastingSession(
            prepare_bundle(bundle or self.bundle), config or self.config,
            engine_factory=lambda _: self.engine,
            verifier_factory=lambda *_args, **_kwargs: self.verifier,
            designer_factory=designer_factory or factory, notify=self.notices.append,
        )
        self.addCleanup(session.close)
        return session

    def test_gemini_reference_is_verified_and_bound_to_local_clone_engine(self):
        session = self.session()
        prompt = "A warm adult female voice, clear and gently paced."
        text = "The lantern will guide our journey home."
        session.update_prompt("voice_mara", prompt, text)
        selected = session.regenerate("voice_mara")
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(self.designer.calls[0]["prompt"], audiobook_voice_prompt(prompt))
        self.assertEqual(self.designer.calls[0]["text"], text)
        self.assertEqual(self.factory_calls, [self.config["voice_design"]])
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(selected["created_with"], self.engine.identity)
        self.assertEqual(selected["creation"]["designer"], self.designer.identity)
        self.assertEqual(selected["creation"]["external_voice_id"], "voice_fixture_gemini")
        self.assertTrue(selected["verification"]["passed"])
        self.assertGreater(len(self.verifier.calls), 0)
        self.assertEqual(selected["reference_text"], text)
        self.assertNotIn(SECRET, json.dumps(selected))
        self.assertNotIn(SECRET, "\n".join(self.notices))
        for path in self.root.rglob("*.json"):
            with self.subTest(path=path.relative_to(self.root)):
                self.assertNotIn(SECRET, path.read_text())

    def test_existing_gemini_reference_is_reused_in_later_book_without_remote_designer(self):
        first = self.session()
        legacy_prompt = "An established voice from the earlier book."
        first.update_prompt("voice_mara", legacy_prompt)
        with patch("audiobook_renderer.casting.audiobook_voice_prompt", side_effect=lambda prompt: prompt):
            selected = first.ensure_ready("voice_mara")
        self.assertEqual(selected["creation"]["requested_description"], legacy_prompt)
        before = Path(selected["reference_audio"]).read_bytes()
        later_bundle = self.make_bundle("book-two")
        local = select_voice_designer(self.config, "local")
        local["gemini"]["api_key"] = None
        later_config = resolve_book_config(later_bundle, local)
        later_config["_voice_parents"] = [self.bundle["directory"]]
        later = self.session(
            bundle=later_bundle, config=later_config,
            designer_factory=lambda _: self.fail("An inherited voice must not be recreated"))
        reused = later.ensure_ready("voice_mara")
        self.assertEqual(reused["asset_hash"], selected["asset_hash"])
        self.assertEqual(reused["reference_audio"], selected["reference_audio"])
        self.assertTrue(reused["inherited"])
        self.assertEqual(reused["creation"], selected["creation"])
        self.assertEqual(Path(reused["reference_audio"]).read_bytes(), before)
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(self.engine.calls, [])

    def remote_designer(self, responses, calls):
        designer = GeminiVoiceDesigner(SECRET)

        def post(endpoint, payload):
            calls.append((endpoint, deepcopy(payload)))
            if endpoint == "/voices":
                # The association is present even if the process stops before
                # the normal attempt record can be written.
                requests = list(self.root.rglob("casting-request.json"))
                self.assertTrue(requests)
                for path in requests:
                    self.assertNotIn(SECRET, path.read_text())
                response = responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                return response
            self.assertEqual(endpoint, "/interactions")
            text = payload["input"][0]["content"][0]["text"]
            path = self.root / f"remote-fixture-{len(calls)}.wav"
            self.designer.write_audio(text, path, payload["generation_config"]["seed"])
            return {"status": "completed", "steps": [{"type": "model_output", "content": [{
                "type": "audio", "mime_type": "audio/wav",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }]}]}

        designer._post = post
        return designer

    def pending_remote_voice(self, *, interrupted=False):
        calls = []
        error = KeyboardInterrupt() if interrupted else _RequestFailure("Gemini request timed out")
        designer = self.remote_designer([error], calls)
        session = self.session(designer_factory=lambda _: designer)
        with self.assertRaises(KeyboardInterrupt if interrupted else (ValueError, RuntimeError)):
            session.ensure_ready("voice_mara")
        paths = list(self.root.rglob("gemini-voice-pending.json"))
        self.assertEqual(len(paths), 1)
        return paths[0], calls

    def test_context_is_sent_in_voice_creation_but_not_in_spoken_sample(self):
        calls = []
        designer = self.remote_designer([{"id": "voice_context_fixture"}], calls)
        session = self.session(designer_factory=lambda _: designer)
        prompt = "A thin, high rasp with a rapid, urgent pace."
        script = "I will meet you at the gate."
        session.update_prompt("voice_mara", prompt, script)
        selected = session.regenerate("voice_mara")
        self.assertEqual([endpoint for endpoint, _ in calls], ["/voices", "/interactions"])
        self.assertEqual(calls[0][1]["voice"]["prompted"]["input"],
                         AUDIOBOOK_VOICE_CONTEXT + " " + prompt)
        self.assertEqual(calls[1][1]["input"][0]["content"][0]["text"], script)
        self.assertEqual(selected["creation"]["requested_description"], audiobook_voice_prompt(prompt))
        self.assertEqual(selected["reference_text"], script)
        self.assertEqual(session.rows()[0]["prompt"], prompt)

    def test_restart_missing_voice_preparation_does_not_duplicate_ambiguous_remote_creation(self):
        pending, calls = self.pending_remote_voice()
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        with self.assertRaisesRegex(ValueError, "may already have succeeded") as raised:
            restarted.ensure_ready("voice_mara")
        self.assertIn(str(pending), str(raised.exception))
        self.assertEqual([endpoint for endpoint, _ in calls], ["/voices"])
        self.assertEqual(list(self.root.rglob("gemini-voice-pending.json")), [pending])

    def test_interrupted_create_is_associated_before_any_attempt_record_exists(self):
        pending, calls = self.pending_remote_voice(interrupted=True)
        self.assertTrue((pending.parent / "casting-request.json").is_file())
        self.assertEqual(list(pending.parent.glob("attempt-*.json")), [])
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        with self.assertRaisesRegex(ValueError, "may already have succeeded"):
            restarted.ensure_ready("voice_mara")
        self.assertEqual(len(calls), 1)

    def test_missing_voice_preparation_recovers_repaired_remote_id_without_new_creation(self):
        pending, calls = self.pending_remote_voice()
        value = read_json(pending)
        value["voice_id"] = "voice_repaired"
        write_json(pending, value)
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        selected = restarted.ensure_ready("voice_mara")
        self.assertEqual([endpoint for endpoint, _ in calls], ["/voices", "/interactions"])
        self.assertEqual(calls[-1][1]["generation_config"]["speech_config"], [{"voice": "voice_repaired"}])
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_repaired")
        self.assertTrue(selected["verification"]["passed"])
        self.assertFalse(pending.exists())
        self.assertEqual(read_json(pending.parent / "gemini-voice.json")["voice_id"], "voice_repaired")
        for path in self.root.rglob("*.json"):
            self.assertNotIn(SECRET, path.read_text())

    def test_repaired_remote_id_cannot_be_reused_for_changed_description(self):
        pending, calls = self.pending_remote_voice()
        value = read_json(pending)
        value["voice_id"] = "voice_repaired"
        write_json(pending, value)
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        restarted.update_prompt("voice_mara", "A completely different voice, low and gravelly.")
        with self.assertRaisesRegex(ValueError, "different voice prompt"):
            restarted.ensure_ready("voice_mara")
        self.assertTrue(pending.exists())
        self.assertEqual(len(calls), 1)

    def test_pending_other_character_does_not_prevent_generation(self):
        pending, calls = self.pending_remote_voice()
        designer = self.remote_designer([{"id": "voice_lena_remote"}], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        selected = restarted.ensure_ready("voice_lena")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_lena_remote")
        self.assertTrue(pending.exists())
        self.assertEqual([endpoint for endpoint, _ in calls], ["/voices", "/voices", "/interactions"])

    def test_corrupt_pending_association_blocks_automatic_missing_voice_preparation(self):
        pending, calls = self.pending_remote_voice()
        path = pending.parent / "casting-request.json"
        request = read_json(path)
        request["voice_id"] = "voice_another_character"
        write_json(path, request)
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        with self.assertRaisesRegex(ValueError, "Cannot identify the pending Gemini creation"):
            restarted.ensure_ready("voice_mara")
        self.assertEqual(len(calls), 1)
        self.assertTrue(pending.exists())

    def test_recovery_handoff_failure_keeps_original_pending_marker(self):
        pending, calls = self.pending_remote_voice()
        value = read_json(pending)
        value["voice_id"] = "voice_repaired"
        write_json(pending, value)
        designer = self.remote_designer([], calls)
        restarted = self.session(designer_factory=lambda _: designer)
        with patch("audiobook_renderer.gemini_voice_design._write_state", side_effect=OSError("disk fixture")):
            with self.assertRaisesRegex(OSError, "disk fixture"):
                restarted.ensure_ready("voice_mara")
        self.assertTrue(pending.exists())
        self.assertEqual(read_json(pending)["voice_id"], "voice_repaired")
        self.assertEqual(len(calls), 1)
        selected = self.session(designer_factory=lambda _: designer).ensure_ready("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_repaired")
        self.assertEqual([endpoint for endpoint, _ in calls], ["/voices", "/interactions"])

    def test_explicit_regeneration_after_verification_failure_creates_a_fresh_remote_voice(self):
        calls = []
        designer = self.remote_designer([{"id": "voice_bad_take"}, {"id": "voice_new_take"}], calls)
        session = self.session(designer_factory=lambda _: designer)
        self.verifier.wrong = True
        with self.assertRaises((ValueError, RuntimeError)):
            session.ensure_ready("voice_mara")
        self.assertEqual(list(self.root.rglob("gemini-voice-pending.json")), [])
        self.verifier.wrong = False
        selected = session.regenerate("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_new_take")
        self.assertEqual([endpoint for endpoint, _ in calls],
                         ["/voices", "/interactions", "/voices", "/interactions"])

    def test_failed_gemini_replacement_preserves_selected_voice(self):
        session = self.session()
        selected = session.ensure_ready("voice_mara")
        before = Path(selected["reference_audio"]).read_bytes()
        self.designer.error = RuntimeError("Gemini request failed")
        with self.assertRaises((ValueError, RuntimeError)):
            session.regenerate("voice_mara")
        current = session.store.load("voice_mara")
        self.assertEqual(current["asset_hash"], selected["asset_hash"])
        self.assertEqual(Path(current["reference_audio"]).read_bytes(), before)
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
