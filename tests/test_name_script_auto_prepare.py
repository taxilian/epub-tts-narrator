"""Analysis never auto-prepares pronunciation data; explicit name-script helpers remain resumable."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.name_script import binding
from epub_tts_narrator.name_script_state import load_name_script, read_state, save_state
from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.pronunciations import preparation_pending, save_candidates, set_pronunciation
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.scoped_pronunciations import scope_for
from name_script_fixtures import PHONES, book, requests
from shared_fixtures import snapshot
from test_cli import NarrativeClient


class AutoNameClient(NarrativeClient):
    def complete(self, system, payload):
        if payload.get("mode") == "name_script":
            target = payload["window"]
            decisions = []
            for item in target["candidates"]:
                before = target["text"][:item["start"]]
                after = target["text"][item["end"]:]
                if before.endswith("her ") or after.startswith(" with"):
                    decisions.append("ordinary")
                else:
                    decisions.append(item["names"][0])
            return {"decisions": decisions}
        return super().complete(system, payload)


class AnalysisWithoutPronunciationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(contextlib.chdir(self.root))
        self.config = self.root / "config.toml"
        self.config.write_text('[provider]\nmodel = "test-model"\nbase_url = "http://127.0.0.1:8000/v1"\n')
        self.book = self.root / "book"
        self.bundle = book(self.book, two=True)
        self.document = self.bundle

    def run_cli(self, *args, client=None):
        output = io.StringIO()
        provider = {"return_value": client} if client is not None else {
            "side_effect": AssertionError("Completed analysis called the pronunciation provider"),
        }
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", **provider), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", **provider), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main([str(item) for item in (*args, "--config", self.config)])
        return code, output.getvalue()

    def prepare(self, client=None):
        client = client or AutoNameClient("test-model", "http://127.0.0.1:8000/v1")
        return prepare_name_script(self.book, {}, requests=requests(self.root, client))

    def assert_no_sidecars(self, directory):
        for name in ("pronunciations.json", "name-script.json", "scoped-pronunciations.json"):
            self.assertFalse((directory / name).exists(), name)

    def test_completed_analysis_does_not_prepare_missing_name_script_or_call_model(self):
        self.assertFalse((self.book / "name-script.json").exists())
        before = snapshot(self.book)
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(snapshot(self.book), before)
        self.assert_no_sidecars(self.book)

    def test_completed_analysis_preserves_explicit_name_script_and_approved_saved_data(self):
        self.prepare()
        set_pronunciation(self.book, self.document, "Ally", PHONES)
        set_pronunciation(self.book, self.document, "Ally", PHONES, scope=scope_for(self.document, "N1"))
        before = snapshot(self.book)
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(snapshot(self.book), before)
        self.assertEqual(load_name_script(self.book, self.document)["unresolved"], [])

    def test_fresh_analysis_skips_name_identification_pronunciation_and_spelling_work(self):
        from shared_fixtures import make_epub
        from test_pipeline import FakeClient

        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n'
                               '[analysis]\nidentity_consensus=false\n')
        epub = make_epub(self.root / "fresh.epub", "Fresh", "fresh",
                         '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Mara spoke softly.</p></body></html>')
        client = FakeClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        directory = self.root / "fresh"
        code, output = self.run_cli("analyze", epub, "-o", directory, client=client)

        self.assertEqual(code, 0, output)
        self.assert_no_sidecars(directory)
        retired = {"speech_spelling_scan", "speech_spelling_review", "pronunciation_discovery",
                   "pronunciation_proposal", "name_script"}
        self.assertTrue(all(payload.get("mode") not in retired for _, payload in client.calls))
        analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
        self.assertFalse(any("speech_normalization" in row for row in analysis["segments"]))
        self.assertFalse(any("spoken_text" in row for row in analysis["segments"]))

    def test_fresh_analysis_without_name_candidates_creates_no_pronunciation_sidecars(self):
        from shared_fixtures import make_epub

        epub = make_epub(self.root / "empty.epub", "Empty", "empty",
                         '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>The rain stopped.</p></body></html>')
        directory = self.root / "empty"
        client = NarrativeClient("test-model", "http://127.0.0.1:8000/v1")
        code, output = self.run_cli("analyze", epub, "-o", directory, client=client)
        self.assertEqual(code, 0, output)
        self.assert_no_sidecars(directory)

    def test_ordinary_usage_of_known_alias_still_requires_explicit_rule_scope(self):
        from epub_tts_narrator.renderer.cli import main as renderer_main

        class OrdinaryClient(AutoNameClient):
            def complete(self, system, payload):
                if payload.get("mode") == "name_script":
                    return {"decisions": ["ordinary"] * len(payload["window"]["candidates"])}
                return super().complete(system, payload)

        self.prepare(OrdinaryClient("test-model", "http://127.0.0.1:8000/v1"))
        save_candidates(self.book, self.document, [])
        registry = self.book / "pronunciations.json"
        before = registry.read_bytes()
        output = io.StringIO()

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = renderer_main(["pronunciations", str(self.book), "--config", str(self.config),
                                  "--action", "set", "--word", "Ally", "--phonemes", "AE1 L IY0"])

        self.assertNotEqual(code, 0)
        self.assertIn("--class", output.getvalue())
        self.assertEqual(registry.read_bytes(), before)

    def test_pending_name_script_is_ignored_by_analysis_and_resumes_only_explicitly(self):
        class FailingClient(AutoNameClient):
            def complete(self, system, payload):
                if payload.get("mode") == "name_script":
                    raise ProviderError("offline name identification outage")
                return super().complete(system, payload)

        with self.assertRaises(ProviderError):
            self.prepare(FailingClient("test-model", "http://127.0.0.1:8000/v1"))
        self.assertTrue(preparation_pending(self.book, self.document))
        state = read_state(self.book)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["windows"], [])
        analysis_bytes = (self.book / "analysis.json").read_bytes()
        pending_bytes = (self.book / "name-script.json").read_bytes()
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(read_state(self.book)["status"], "pending")
        self.assertEqual((self.book / "name-script.json").read_bytes(), pending_bytes)
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_bytes)
        self.prepare()
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_bytes)

    def test_stale_binding_is_ignored_by_analysis_and_repaired_only_explicitly(self):
        self.prepare()
        state = read_state(self.book)
        stale = deepcopy(state)
        stale["binding"] = {"source": "0" * 64, "identities": "0" * 64, "version": 1}
        save_state(self.book, stale)
        stale_bytes = (self.book / "name-script.json").read_bytes()
        self.assertTrue(preparation_pending(self.book, self.document))
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual((self.book / "name-script.json").read_bytes(), stale_bytes)
        self.prepare()
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual(read_state(self.book)["binding"], binding(self.document))

    def test_absent_or_corrupt_state_is_pending(self):
        self.prepare()
        save_candidates(self.book, self.document, [])
        self.assertFalse(preparation_pending(self.book, self.document))
        (self.book / "name-script.json").unlink()
        self.assertTrue(preparation_pending(self.book, self.document))
        (self.book / "name-script.json").write_text('{"bad": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid name-script checkpoint"):
            preparation_pending(self.book, self.document)

    def test_ready_unresolved_is_complete_search_and_manual_resolution_is_retained(self):
        class DisagreeClient(AutoNameClient):
            def complete(self, system, payload):
                if payload.get("mode") == "name_script" and payload.get("reviewer") == 2:
                    target = payload["window"]
                    return {"decisions": ["unresolved"] * len(target["candidates"])}
                return super().complete(system, payload)

        self.prepare(DisagreeClient("test-model", "http://127.0.0.1:8000/v1"))
        save_candidates(self.book, self.document, [])
        state = read_state(self.book)
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["windows"][0]["unresolved"])
        self.assertFalse(preparation_pending(self.book, self.document))
        name_script_bytes = (self.book / "name-script.json").read_bytes()
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual((self.book / "name-script.json").read_bytes(), name_script_bytes)
        from epub_tts_narrator.name_script_workflow import resolve_mention
        unresolved = load_name_script(self.book, self.document)["unresolved"]
        selected = unresolved[0]
        resolve_mention(self.book, selected["mention_id"], selected["names"][0])
        remaining = load_name_script(self.book, self.document)["unresolved"]
        self.assertNotIn(selected["mention_id"], [item["mention_id"] for item in remaining])
        resolved_bytes = (self.book / "name-script.json").read_bytes()
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual((self.book / "name-script.json").read_bytes(), resolved_bytes)

    def test_resume_reuses_completed_windows_and_first_vote_of_interrupted_window(self):
        class InterruptedClient(AutoNameClient):
            events = []
            fail = True

            def complete(self, system, payload):
                if payload.get("mode") == "name_script":
                    event = (payload["window"]["block_id"], payload["reviewer"])
                    type(self).events.append(event)
                    if type(self).fail and event == ("b2", 2):
                        type(self).fail = False
                        raise ProviderError("Interrupted name classification")
                return super().complete(system, payload)

        client = InterruptedClient("test-model", "http://127.0.0.1:8000/v1")
        with self.assertRaises(ProviderError):
            self.prepare(client)
        state = read_state(self.book)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(len(state["windows"]), 1)
        self.assertEqual(InterruptedClient.events, [("b1", 1), ("b1", 2), ("b2", 1), ("b2", 2)])
        analysis_bytes = (self.book / "analysis.json").read_bytes()

        self.prepare(client)

        self.assertEqual(InterruptedClient.events,
                         [("b1", 1), ("b1", 2), ("b2", 1), ("b2", 2), ("b2", 2)])
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_bytes)
