"""The rendering coordinator and synthesis server prepare different models."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.model_setup import ModelSetupError
from audition_fixtures import prepare_document
from test_contracts import valid_document
from test_renderer_distributed_config import worker_table


class DistributedCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = self.root / "global.toml"
        self.bundle = self.root / "book"
        export_bundle(prepare_document(valid_document()), self.bundle)

    def configure(self, workers="", *, local=True, verifier=True):
        self.config.write_text(
            ('[engine]\ncheckpoint="./drama-3"\n' if local else "")
            + ('[verification]\nmodel="./whisper"\n' if verifier else "")
            + workers)

    def run_cli(self, command, *args):
        argv = [command]
        if command not in {"serve", "doctor"}:
            argv.append(str(self.bundle))
        argv.extend(["--config", str(self.config), *map(str, args)])
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = main(argv)
        return status, output.getvalue(), error.getvalue()

    def test_serve_needs_only_engine_and_never_loads_a_book_or_verifier(self):
        self.configure(verifier=False)
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.server.serve") as serve,
              patch("epub_tts_narrator.renderer.cli.load_bundle") as bundle,
              patch("epub_tts_narrator.renderer.cli.voice_store") as store,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier):
            status, _, error = self.run_cli("serve")
        self.assertEqual(status, 0, error)
        setup.assert_called_once_with("engine")
        serve.assert_called_once()
        self.assertIsNone(serve.call_args.args[0]["verification"]["model"])
        self.assertEqual(serve.call_args.args[0]["server"]["host"], "127.0.0.1")
        self.assertEqual(serve.call_args.args[0]["server"]["port"], 8080)
        bundle.assert_not_called()
        store.assert_not_called()
        verifier.assert_not_called()

    def test_serve_flags_override_only_explicit_bind_settings_without_editing_config(self):
        self.configure(
            '[server]\nhost="127.0.0.2"\nport=8091\napi_key="fixture-server-key"\n',
            verifier=False,
        )
        before = self.config.read_bytes()
        for args, expected_host, expected_port in (
            ((), "127.0.0.2", 8091),
            (("--host", "0.0.0.0"), "0.0.0.0", 8091),
            (("--port", "8092"), "127.0.0.2", 8092),
            (("--host", "0.0.0.0", "--port", "8092"), "0.0.0.0", 8092),
        ):
            with self.subTest(args=args):
                with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
                      patch("epub_tts_narrator.renderer.server.serve") as serve):
                    status, output, error = self.run_cli("serve", *args)
                self.assertEqual(status, 0, error)
                setup.assert_called_once_with("engine")
                serve.assert_called_once()
                settings = serve.call_args.args[0]["server"]
                self.assertEqual(settings["host"], expected_host)
                self.assertEqual(settings["port"], expected_port)
                self.assertEqual(settings["api_key"], "fixture-server-key")
                self.assertNotIn("fixture-server-key", output + error)
                self.assertEqual(self.config.read_bytes(), before)

    def test_serve_creates_missing_key_and_reuses_it_on_subsequent_starts(self):
        for server_text in (
            "",
            '[server]\nhost="127.0.0.2"\nport=8091\n',
            '[server]\napi_key="" # created automatically\n',
        ):
            with self.subTest(server_text=server_text):
                self.configure(server_text, verifier=False)
                before = tomllib.loads(self.config.read_text())
                with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
                      patch("epub_tts_narrator.renderer.server.serve") as serve):
                    status, output, error = self.run_cli("serve")
                self.assertEqual(status, 0, error)
                setup.assert_called_once_with("engine")
                serve.assert_called_once()
                saved = tomllib.loads(self.config.read_text())
                key = saved["server"]["api_key"]
                self.assertIsInstance(key, str)
                self.assertGreaterEqual(len(key), 32)
                self.assertEqual(serve.call_args.args[0]["server"]["api_key"], key)
                self.assertNotIn(key, output + error)
                self.assertIn("server.api_key", output)
                self.assertIn(str(self.config), output)
                expected = {**before, "server": {**before.get("server", {}), "api_key": key}}
                self.assertEqual(saved, expected)

                after_first_start = self.config.read_bytes()
                with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
                      patch("epub_tts_narrator.renderer.server.serve") as serve):
                    status, output, error = self.run_cli("serve")
                self.assertEqual(status, 0, error)
                self.assertEqual(serve.call_args.args[0]["server"]["api_key"], key)
                self.assertNotIn(key, output + error)
                self.assertEqual(self.config.read_bytes(), after_first_start)

    def test_serve_key_creation_does_not_persist_bind_overrides(self):
        self.configure('[server]\nhost="127.0.0.2"\nport=8091\n', verifier=False)
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.server.serve") as serve):
            status, _, error = self.run_cli("serve", "--host", "0.0.0.0", "--port", "8092")
        self.assertEqual(status, 0, error)
        saved = tomllib.loads(self.config.read_text())
        self.assertEqual(saved["server"]["host"], "127.0.0.2")
        self.assertEqual(saved["server"]["port"], 8091)
        settings = serve.call_args.args[0]["server"]
        self.assertEqual(settings["host"], "0.0.0.0")
        self.assertEqual(settings["port"], 8092)
        self.assertEqual(settings["api_key"], saved["server"]["api_key"])

    def test_serve_saves_key_to_the_selected_explicit_default_or_legacy_config(self):
        canonical = self.root / "epub-tts-narrator.toml"
        legacy = self.root / "epub-audiobook.toml"
        config_text = '[engine]\ncheckpoint="./drama-3"\n'
        for selection in ("explicit", "canonical", "legacy"):
            with self.subTest(selection=selection):
                self.config.write_text(config_text)
                legacy.write_text(config_text)
                canonical.unlink(missing_ok=True)
                if selection != "legacy":
                    canonical.write_text(config_text)
                target = {"explicit": self.config, "canonical": canonical, "legacy": legacy}[selection]
                argv = ["serve"]
                if selection == "explicit":
                    argv.extend(["--config", str(self.config)])
                output, error = StringIO(), StringIO()
                with (patch("epub_tts_narrator.config.Path.cwd", return_value=self.root),
                      patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
                      patch("epub_tts_narrator.renderer.server.serve") as serve,
                      redirect_stdout(output), redirect_stderr(error)):
                    status = main(argv)
                self.assertEqual(status, 0, error.getvalue())
                saved = tomllib.loads(target.read_text())
                self.assertEqual(serve.call_args.args[0]["server"]["api_key"], saved["server"]["api_key"])
                self.assertIn(str(target), output.getvalue())
                self.assertNotIn(saved["server"]["api_key"], output.getvalue() + error.getvalue())
                for candidate in (self.config, canonical, legacy):
                    if candidate != target and candidate.exists():
                        self.assertEqual(candidate.read_text(), config_text)
                if selection == "legacy":
                    self.assertFalse(canonical.exists())

    def test_serve_cannot_start_with_an_unsaved_generated_key(self):
        self.configure(verifier=False)
        before = self.config.read_bytes()
        with (patch("epub_tts_narrator.renderer.server_config.ensure_server_api_key",
                    side_effect=OSError("Cannot save server API key: permission denied")),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.server.serve") as serve):
            status, _, error = self.run_cli("serve")
        self.assertEqual(status, 2)
        self.assertIn("permission denied", error)
        setup.assert_not_called()
        serve.assert_not_called()
        self.assertEqual(self.config.read_bytes(), before)

    def test_serve_invalid_bind_flags_fail_before_model_setup_or_server_start(self):
        self.configure(verifier=False)
        before = self.config.read_bytes()
        for args, diagnostic in (
            (("--host", ""), "server.host"),
            (("--host", "bad host"), "server.host"),
            (("--host", "localhost\n"), "server.host"),
            (("--port", "0"), "server.port"),
            (("--port", "65536"), "server.port"),
            (("--port", "-1"), "server.port"),
        ):
            with self.subTest(args=args):
                with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
                      patch("epub_tts_narrator.renderer.server.serve") as serve):
                    status, _, error = self.run_cli("serve", *args)
                self.assertEqual(status, 2)
                self.assertIn(diagnostic, error)
                setup.assert_not_called()
                serve.assert_not_called()
                self.assertEqual(self.config.read_bytes(), before)

    def test_serve_noninteger_port_is_a_usage_error_before_model_setup(self):
        self.configure(verifier=False)
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.server.serve") as serve,
              self.assertRaises(SystemExit) as error):
            self.run_cli("serve", "--port", "eight-thousand")
        self.assertEqual(error.exception.code, 2)
        setup.assert_not_called()
        serve.assert_not_called()

    def test_declining_server_model_download_does_not_start_server(self):
        self.configure(verifier=False)
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure",
                    side_effect=ModelSetupError("Download declined")),
              patch("epub_tts_narrator.renderer.server.serve") as serve):
            status, _, error = self.run_cli("serve")
        self.assertEqual(status, 2)
        self.assertIn("Download declined", error)
        serve.assert_not_called()

    def test_remote_only_renders_existing_voices_without_local_download(self):
        self.configure(worker_table("other-mac"), local=False)
        engine, verifier, store = Mock(), Mock(), Mock()
        engine.wrap_verifier.return_value = verifier
        store.load.return_value = {"kind": "reference"}
        with (patch("epub_tts_narrator.renderer.cli.voice_store", return_value=store),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.engines.create_engine") as local,
              patch("epub_tts_narrator.renderer.pool.create_render_engine", return_value=engine) as factory,
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts,
              patch("epub_tts_narrator.renderer.workflow.render", return_value={}) as render):
            status, output, error = self.run_cli("render", "--chapters-only")
        self.assertEqual(status, 0, error)
        setup.assert_called_once_with("verification")
        local.assert_not_called()
        scripts.assert_not_called()
        factory.assert_called_once()
        self.assertIsNone(factory.call_args.args[0]["engine"]["checkpoint"])
        self.assertIs(render.call_args.args[2], engine)
        self.assertIs(render.call_args.args[3], verifier)
        engine.wrap_verifier.assert_called_once_with(verifier)
        self.assertEqual(render.call_args.kwargs, {"chapters_only": True})
        self.assertIn("other-mac (1 slot(s))", output)
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_mixed_renderers_prepare_engine_and_verifier_and_close_both_on_failure(self):
        self.configure(worker_table("this-mac", "local") + worker_table("other-mac"))
        engine, verifier, store = Mock(), Mock(), Mock()
        engine.wrap_verifier.return_value = verifier
        store.load.return_value = {"kind": "reference"}
        verifier.close.side_effect = RuntimeError("verifier cleanup fixture")
        with (patch("epub_tts_narrator.renderer.cli.voice_store", return_value=store),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.pool.create_render_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render",
                    side_effect=RuntimeError("render fixture"))):
            status, _, error = self.run_cli("render")
        self.assertEqual(status, 2)
        self.assertEqual([call.args for call in setup.call_args_list], [("engine",), ("verification",)])
        self.assertIn("render fixture", error)
        self.assertIn("verifier cleanup fixture", error)
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_remote_only_missing_voices_explains_casting_requirement_before_inference(self):
        self.configure(worker_table(), local=False)
        store = Mock()
        store.load.return_value = None
        with (patch("epub_tts_narrator.renderer.cli.voice_store", return_value=store),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.pool.create_render_engine") as engine,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier,
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts):
            status, _, error = self.run_cli("render")
        self.assertEqual(status, 2)
        self.assertIn("Missing voices", error)
        self.assertIn("cast", error)
        engine.assert_not_called()
        verifier.assert_not_called()
        scripts.assert_not_called()

    def test_remote_only_can_prepare_missing_voices_with_configured_gemini(self):
        self.configure('[gemini]\napi_key="fixture-gemini-key"\n' + worker_table(), local=False)
        engine, verifier, store = Mock(), Mock(), Mock()
        engine.wrap_verifier.return_value = verifier
        store.load.return_value = None
        with (patch("epub_tts_narrator.renderer.cli.voice_store", return_value=store),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.engines.create_engine") as local,
              patch("epub_tts_narrator.renderer.pool.create_render_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts",
                    side_effect=lambda bundle, *a, **k: bundle) as scripts,
              patch("epub_tts_narrator.renderer.workflow.render", return_value={}) as render):
            status, output, error = self.run_cli("render")
        self.assertEqual(status, 0, error)
        self.assertEqual([call.args for call in setup.call_args_list],
                         [("verification",), ("voice_design",)])
        local.assert_not_called()
        scripts.assert_called_once()
        self.assertTrue(scripts.call_args.kwargs["voice_ids"])
        render.assert_called_once()
        self.assertNotIn("fixture-gemini-key", output + error)
        engine.close.assert_called_once()
        verifier.close.assert_called_once()

    def test_serve_rejects_remote_only_configuration_without_a_local_model(self):
        self.configure(worker_table(), local=False, verifier=False)
        before = self.config.read_bytes()
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.server.serve") as serve):
            status, _, error = self.run_cli("serve")
        self.assertEqual(status, 2)
        self.assertIn("local engine.checkpoint", error)
        setup.assert_not_called()
        serve.assert_not_called()
        self.assertEqual(self.config.read_bytes(), before)

    def test_plan_reports_only_worker_names_types_and_slots_without_loading_models(self):
        self.configure(worker_table("mac", "local")
                       + worker_table("online", api_key="private-api-key", concurrency=3))
        before = self.config.read_bytes()
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.engines.create_engine") as local,
              patch("epub_tts_narrator.renderer.pool.create_render_engine") as pool,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier):
            status, output, error = self.run_cli("plan")
        self.assertEqual(status, 0, error)
        self.assertEqual(json.loads(output)["renderers"], [
            {"name": "mac", "type": "local", "concurrency": 1},
            {"name": "online", "type": "fish_http", "concurrency": 3},
        ])
        self.assertNotIn("private-api-key", output + error)
        self.assertNotIn("https://render.example", output + error)
        self.assertEqual(self.config.read_bytes(), before)
        for factory in (setup, local, pool, verifier):
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
