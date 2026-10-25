import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.runpod import add_parser, deployment_lock, run_command


class RunpodProgressTests(unittest.TestCase):
    def test_progress_observes_running_and_paused_jobs_even_during_an_operation(self):
        calls = []

        class Client:
            def __init__(self, settings):
                pass

            def request(self, method, path):
                calls.append((method, path))
                if "/events?" in path:
                    return {"events": [{"message": "Chapter 5 / 23 | Blocks 153 / 154 complete | elapsed 00:23 | private extra"}]}
                return {"jobs": [
                    {"id": "a" * 64, "state": "generating", "paused": False, "owner_connected": True,
                     "chapter_number": 5, "total_chapters": 23, "last_event": 1000,
                     "spec": "do not print book text", "session": "private-token"},
                    {"id": "b" * 64, "state": "queued", "paused": True, "owner_connected": False,
                     "chapter_number": 1, "total_chapters": 59, "last_event": 1001}]}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "state.json").write_text(json.dumps({"pod_id": "pod", "api_key": "private-token"}))
            settings = {"deployment": {"state_dir": folder, "service_port": 8082}}
            parser = argparse.ArgumentParser()
            add_parser(parser.add_subparsers(dest="command"))
            args = parser.parse_args(["runpod", "progress"])
            output = StringIO()
            with deployment_lock(root), redirect_stdout(output), \
                    patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=settings), \
                    patch("epub_tts_narrator.renderer.runpod_progress.ChapterClient", Client), \
                    patch("epub_tts_narrator.renderer.runpod.Deployment") as deployment:
                self.assertEqual(run_command(args), 0)
            self.assertIn("Chapter 5 / 23: generating; blocks 153 / 154", output.getvalue())
            self.assertIn("paused: no connected rendering client", output.getvalue())
            for private in ("private-token", "book text", "private extra"):
                self.assertNotIn(private, output.getvalue())
            self.assertTrue(all(method == "GET" for method, _ in calls))
            deployment.assert_not_called()

    def test_no_state_progress_does_not_contact_any_service(self):
        from epub_tts_narrator.renderer.runpod_progress import run_progress
        with tempfile.TemporaryDirectory() as folder, \
                patch("epub_tts_narrator.renderer.runpod_progress.ChapterClient") as client, \
                redirect_stdout(StringIO()) as output:
            args = argparse.Namespace(limit=100, poll_seconds=2.0, job=None, watch=False, json=False)
            self.assertEqual(run_progress({"deployment": {"state_dir": folder}}, args), 0)
            client.assert_not_called()
            self.assertIn("No deployment state", output.getvalue())

    def test_acknowledged_chapter_disappearing_during_observation_is_not_an_error(self):
        from epub_tts_narrator.renderer.chapter_client import ChapterClientError
        from epub_tts_narrator.renderer.runpod_progress import run_progress
        with tempfile.TemporaryDirectory() as folder, \
                patch("epub_tts_narrator.renderer.runpod_progress.ChapterClient") as client, \
                redirect_stdout(StringIO()) as output:
            (Path(folder) / "state.json").write_text(json.dumps({"pod_id": "pod", "api_key": "private"}))
            client.return_value.request.side_effect = [
                {"jobs": [{"id": "a" * 64, "state": "generating", "last_event": 4}]},
                ChapterClientError("Run acknowledged and removed", status=404)]
            args = argparse.Namespace(limit=100, poll_seconds=2.0, job=None, watch=False, json=True)
            self.assertEqual(run_progress({"deployment": {"state_dir": folder, "service_port": 8082}}, args), 0)
            self.assertEqual(json.loads(output.getvalue()), {"jobs": []})
