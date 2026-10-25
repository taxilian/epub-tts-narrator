"""Download decisions precede expensive work and changes to saved voices."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.audition import run_casting
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.model_setup import ModelSetup, ModelSetupError, WHISPER
from audition_fixtures import prepare_document
from test_contracts import valid_document
from test_renderer_model_setup import Terminal, complete


class ModelSetupIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle_path = self.root / "bundle"
        export_bundle(prepare_document(valid_document()), self.bundle_path)
        self.bundle = load_bundle(self.bundle_path)
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(
            '[engine]\ncheckpoint="./models/drama-3"\n'
            '[verification]\nmodel="./models/whisper-large-v3-turbo"\n'
            '[voices]\nseries_id="fixture"\n'
            '[render]\nrequire_cover=false\n')
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "models" / "drama-3")
        self.config["verification"]["model"] = str(self.root / "models" / "whisper-large-v3-turbo")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="fixture")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "audio"), require_cover=False)

    def run_cli(self, command, *args):
        output, error = StringIO(), StringIO()
        command_args = [command, str(self.bundle_path)] if command != "doctor" else [command]
        with redirect_stdout(output), redirect_stderr(error):
            status = main([*command_args, "--config", str(self.config_path), *args])
        return status, output.getvalue(), error.getvalue()

    def test_render_prepares_downloads_before_scripts_models_and_synthesis(self):
        events = []
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure",
                    side_effect=lambda name: events.append(name)),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts",
                    side_effect=lambda bundle, *a, **k: events.append("scripts") or bundle),
              patch("epub_tts_narrator.renderer.engines.create_engine",
                    side_effect=lambda *a: events.append("load_engine") or Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier",
                    side_effect=lambda *a: events.append("load_verifier") or Mock()),
              patch("epub_tts_narrator.renderer.workflow.render",
                    side_effect=lambda *a, **k: events.append("render") or {})):
            status, _, error = self.run_cli("render")
        self.assertEqual(status, 0, error)
        self.assertEqual(events, ["engine", "verification", "voice_design", "scripts",
                                  "load_engine", "load_verifier", "render"])

    def test_declining_render_setup_stops_before_network_script_work_or_models(self):
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure",
                    side_effect=ModelSetupError("Download declined; rerun when ready.")),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts,
              patch("epub_tts_narrator.renderer.engines.create_engine") as engine,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier):
            status, _, error = self.run_cli("render")
        self.assertEqual(status, 2)
        self.assertIn("Download declined", error)
        scripts.assert_not_called()
        engine.assert_not_called()
        verifier.assert_not_called()

    def test_render_downloads_missing_whisper_cache_snapshot_without_changing_config(self):
        cache = (self.root / "huggingface" / "hub").resolve()
        snapshot = cache / "models--mlx-community--whisper-large-v3-turbo" / "snapshots" / WHISPER.revision
        self.config_path.write_text(
            '[engine]\ncheckpoint="./models/drama-3"\n'
            f'[verification]\nmodel={json.dumps(str(snapshot))}\n'
            '[voices]\nseries_id="fixture"\n'
            '[huggingface]\napi_key="hf_fixture"\n'
            '[render]\nrequire_cover=false\n')
        before = self.config_path.read_bytes()
        complete(self.root / "models" / "drama-3")
        prompts = StringIO()
        download = Mock(side_effect=lambda **kwargs: complete(snapshot, "whisper"))
        def setup(config):
            return ModelSetup(config, input_stream=Terminal("yes\n"), output_stream=prompts,
                              downloader=download)
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup", side_effect=setup),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()) as verifier,
              patch("epub_tts_narrator.renderer.workflow.render", return_value={}) as render):
            status, _, error = self.run_cli("render")
        self.assertEqual(status, 0, error)
        download.assert_called_once_with(repo_id=WHISPER.repo, revision=WHISPER.revision,
                                         cache_dir=str(cache), token="hf_fixture", max_workers=5)
        self.assertIn("Download this model now?", prompts.getvalue())
        self.assertNotIn("hf_fixture", prompts.getvalue())
        self.assertEqual(verifier.call_args.args[0]["model"], str(snapshot))
        render.assert_called_once()
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_read_only_and_assembly_commands_never_prepare_models(self):
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.workflow.assemble")):
            for command in ("plan", "voices", "doctor", "assemble"):
                with self.subTest(command=command):
                    self.run_cli(command)
            self.run_cli("cast", "--action", "list")
        setup.assert_not_called()

    def session(self, setup_factory):
        session = CastingSession(self.bundle, self.config, model_setup_factory=setup_factory)
        self.addCleanup(session.close)
        return session

    def test_bulk_regeneration_decline_preserves_every_saved_selection(self):
        setup = Mock()
        setup.ensure.side_effect = [None, ModelSetupError("Whisper download declined")]
        session = self.session(lambda config: setup)
        voice_ids = [voice["id"] for voice in self.bundle["cast"]["voices"]]
        with (patch.object(session.store, "invalidate") as invalidate,
              patch.object(session, "_save_draft") as save):
            with self.assertRaisesRegex(ModelSetupError, "Whisper download declined"):
                session.begin_regeneration(voice_ids)
        invalidate.assert_not_called()
        save.assert_not_called()
        self.assertFalse(session.store.directory.exists())

    def test_single_regeneration_decline_preserves_inherited_failure_and_designer_choice(self):
        self.config["gemini"]["api_key"] = "fixture-gemini-key"
        selections = []
        def factory(config):
            def ensure(component):
                selections.append((component, config["voice_design"]["type"]))
                if component == "verification":
                    raise ModelSetupError("Whisper download declined")
            return Mock(ensure=ensure)
        session = self.session(factory)
        with (patch.object(session.store, "invalidate") as invalidate,
              patch.object(session.store, "mark_regeneration_failed") as fail,
              patch.object(session, "_generate_locked") as generate):
            with self.assertRaises(ModelSetupError):
                session.regenerate(self.bundle["cast"]["voices"][0]["id"], voice_designer="local")
        self.assertEqual(selections, [("engine", "engine"), ("verification", "engine")])
        self.assertEqual(session.config["voice_design"]["type"], "gemini")
        invalidate.assert_not_called()
        fail.assert_not_called()
        generate.assert_not_called()

    def test_casting_startup_pauses_after_one_model_setup_failure_and_keeps_menu(self):
        session, player = Mock(), Mock()
        row = {"voice_id": "v1", "selected": None, "candidates": [], "label": "Mara"}
        session.rows.return_value = [row, {**row, "voice_id": "v2", "label": "Lena"}]
        session.ensure_ready.side_effect = ModelSetupError("Download declined")
        lines = []
        run_casting(session, player, input_fn=lambda _: "q", output=lines.append)
        session.ensure_ready.assert_called_once_with("v1")
        self.assertTrue(any("Voice preparation paused: Download declined" in line for line in lines))
        self.assertTrue(any("Voice casting — 0/2 ready" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
