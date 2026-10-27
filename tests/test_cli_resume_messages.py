"""CLI recovery explains validated cache reuse without exposing private inputs."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_progress import inspect_pending, is_pending_directory
from epub_tts_narrator.cli import main
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.provider import ProviderError
from test_cli import NarrativeClient
from test_contracts import valid_document


class RecordingClient(NarrativeClient):
    def __init__(self, *args, fail=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail = fail
        self.modes = []

    def complete(self, system, payload):
        self.modes.append(payload.get("mode", "attribution"))
        if self.fail and payload.get("mode") == "reconcile":
            raise ProviderError("Fixture API timeout", retryable=True)
        return super().complete(system, payload)


class CLIResumeMessageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.enterContext(contextlib.chdir(self.root))
        self.config = self.root / "epub-audiobook.toml"
        self.config.write_text(
            '[provider]\nmodel = "fixture-model"\n'
            'base_url = "http://127.0.0.1:8000/v1"\n'
            'api_key = "fixture-secret-never-print"\n', encoding="utf-8")
        self.document = valid_document()
        self.source = self.root / "source.json"
        write_json(self.source, {"format": "epub-audiobook-source", "version": 1,
                                 "source": self.document["source"]})
        self.analysis = self.root / "analysis.json"
        write_json(self.analysis, self.document)

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(arg) for arg in args])
        self.assertNotIn("fixture-secret-never-print", stdout.getvalue() + stderr.getvalue())
        return status, stdout.getvalue(), stderr.getvalue()

    def test_timeout_reports_recoverable_cache_and_same_command_reuses_discovery(self):
        output = self.root / "book-one"
        clients = []

        def client(*args, **kwargs):
            instance = RecordingClient(*args, fail=not clients, **kwargs)
            clients.append(instance)
            return instance

        command = ("analyze", self.source, "-o", output)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=client):
            status, stdout, stderr = self.run_cli(*command)
            self.assertEqual(status, 1)
            self.assertIn("Fixture API timeout", stderr)
            cache = self.root / ".epub-audiobook-cache"
            self.assertIn(f"Request cache: {cache}", stderr)
            self.assertIn(f"Resume cache: {cache}", stderr)
            self.assertIn("same input, output, and configuration", stderr)
            self.assertIn("incomplete requests will be retried", stderr)
            self.assertTrue(list(cache.glob("*.json")))
            # The failed run checkpointed its pending state instead of leaving
            # no output, but the directory is never a completed bundle.
            self.assertTrue(is_pending_directory(output))
            self.assertEqual(inspect_pending(output)["status"], "failed")
            self.assertNotIn("Wrote analysis", stdout)
            status, _, stderr = self.run_cli(*command)
            self.assertEqual(status, 0, stderr)
            self.assertIn("reused automatically", stderr)
            self.assertIn("revalidated locally", stderr)
            self.assertNotIn("Resume cache:", stderr)
        self.assertIn("discover", clients[0].modes)
        self.assertNotIn("discover", clients[1].modes)
        self.assertTrue((output / "manifest.json").exists())

    def test_review_failure_reports_config_relative_cache_and_preserves_output(self):
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write('[analysis]\ncache_dir = "private-cache"\n')
        output = self.root / "reviewed.json"
        output.write_text("old review", encoding="utf-8")
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient), \
                patch("epub_tts_narrator.cli.audit_attribution", side_effect=ProviderError("Fixture timeout")):
            status, _, stderr = self.run_cli("review-attribution", self.analysis, "-o", output, "--force")
        self.assertEqual(status, 1)
        self.assertIn(f"Request cache: {self.root / 'private-cache'}", stderr)
        self.assertIn(f"Resume cache: {self.root / 'private-cache'}", stderr)
        self.assertIn("same input, output, and configuration", stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "old review")
        # Merely announcing the selected cache must not create it or imply a
        # particular request was saved before the provider returned anything.
        self.assertFalse((self.root / "private-cache").exists())

    def test_interrupt_preserves_existing_message_and_adds_known_cache_path(self):
        output = self.root / "interrupted-book"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient), \
                patch("epub_tts_narrator.cli.analyze", side_effect=KeyboardInterrupt):
            status, stdout, stderr = self.run_cli("analyze", self.source, "-o", output)
        self.assertEqual(status, 130)
        self.assertIn("Interrupted. Completed model requests remain cached; checkpointed analysis "
                      "progress is kept in the output directory; rerun the same command to resume.", stderr)
        self.assertIn(f"Resume cache: {self.root / '.epub-audiobook-cache'}", stderr)
        self.assertTrue(is_pending_directory(output))
        record = inspect_pending(output)
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(record["completed_stages"], [])
        self.assertNotIn("Wrote analysis", stdout)

    def test_validation_failure_during_model_work_has_recovery_guidance(self):
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient), \
                patch("epub_tts_narrator.cli.analyze", side_effect=ValueError("Fixture validation failure")):
            status, _, stderr = self.run_cli("analyze", self.source, "-o", self.root / "invalid-book")
        self.assertEqual(status, 1)
        self.assertIn("Fixture validation failure", stderr)
        self.assertIn("Resume cache:", stderr)

    def test_preflight_errors_and_plan_do_not_announce_model_work(self):
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli("analyze", self.root / "absent.epub", "-o", self.root / "book")
            self.assertEqual(status, 1)
            self.assertNotIn("Request cache:", stderr)
            self.assertNotIn("Resume cache:", stderr)
            status, _, stderr = self.run_cli("plan", self.source, "-o", self.root / "planned-book")
            self.assertEqual(status, 0, stderr)
            self.assertNotIn("Request cache:", stderr)
            self.assertNotIn("Resume cache:", stderr)
            provider.assert_not_called()

    def test_early_interrupt_does_not_invent_a_cache_path(self):
        with patch("epub_tts_narrator.cli.load_config", side_effect=KeyboardInterrupt):
            status, _, stderr = self.run_cli("analyze", self.source, "-o", self.root / "book")
        self.assertEqual(status, 130)
        self.assertIn("Interrupted.", stderr)
        self.assertNotIn("Request cache:", stderr)
        self.assertNotIn("Resume cache:", stderr)


if __name__ == "__main__":
    unittest.main()
