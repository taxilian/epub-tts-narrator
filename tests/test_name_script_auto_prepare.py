"""Automatic missing/pending/stale name identification through real CLI flows."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.name_script import binding
from epub_tts_narrator.name_script_state import load_name_script, read_state, save_state
from epub_tts_narrator.pronunciations import preparation_pending
from epub_tts_narrator.provider import ProviderError
from name_script_fixtures import book
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


class NameScriptAutoPrepareTests(unittest.TestCase):
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
        self.enterContext(patch("epub_tts_narrator.cli.OpenAICompatibleClient", AutoNameClient))
        self.enterContext(patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", AutoNameClient))

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main([str(item) for item in (*args, "--config", self.config)])
        return code, output.getvalue()

    def test_missing_name_script_is_prepared_automatically_and_completed_resume_needs_no_model(self):
        self.assertFalse((self.book / "name-script.json").exists())
        code, output = self.run_cli("discover-pronunciations", self.book)
        self.assertEqual(code, 0, output)
        state = read_state(self.book)
        self.assertEqual(state["status"], "ready")
        self.assertFalse(preparation_pending(self.book, self.document))
        self.assertEqual(len(state["windows"]), 2)
        self.assertEqual(state["windows"][0]["text"],
                         "(N1:Ally) greeted her ally softly. Ally with your friends. (N1:ALLY) waved.")
        name_script_bytes = (self.book / "name-script.json").read_bytes()
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=AssertionError("Completed work called provider")), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", side_effect=AssertionError("Completed work called provider")):
            code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual((self.book / "name-script.json").read_bytes(), name_script_bytes)

    def test_existing_directory_analyze_prepares_missing_name_script(self):
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual(load_name_script(self.book, self.document)["unresolved"], [])

    def test_fresh_analysis_populates_identified_names_without_confirming_pronunciations(self):
        from shared_fixtures import make_epub
        from test_pipeline import FakeClient
        from epub_tts_narrator.pronunciations import load_pronunciations

        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n'
                               '[analysis]\nidentity_consensus=false\n')
        epub = make_epub(self.root / "fresh.epub", "Fresh", "fresh",
                         '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Mara spoke softly.</p></body></html>')
        client = FakeClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        directory = self.root / "fresh"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", return_value=client):
            code, output = self.run_cli("analyze", epub, "-o", directory)

        self.assertEqual(code, 0, output)
        state = read_state(directory)
        self.assertEqual(state["status"], "ready")
        self.assertTrue(any("Mara" in name["aliases"] for name in state["names"].values()))
        self.assertTrue(any(payload.get("mode") == "name_script" for _, payload in client.calls))
        self.assertFalse(any(entry["status"] == "confirmed" for entry in load_pronunciations(directory)))
        self.assertFalse((directory / "scoped-pronunciations.json").exists())

    def test_no_name_candidates_complete_locally_without_provider_configuration(self):
        from shared_fixtures import make_epub
        from epub_tts_narrator.pronunciation_workflow import prepare_book_pronunciations

        epub = make_epub(self.root / "empty.epub", "Empty", "empty",
                         '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>The rain stopped.</p></body></html>')
        directory = self.root / "empty"
        code, output = self.run_cli("analyze", epub, "-o", directory)
        self.assertEqual(code, 0, output)
        self.assertEqual(read_state(directory)["names"], {})
        (directory / "name-script.json").unlink()

        with patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient",
                   side_effect=AssertionError("Empty identification called provider")):
            prepare_book_pronunciations(directory, {"analysis": {"attribution_workers": 1}},
                                        progress=lambda message: None)

        self.assertEqual(read_state(directory)["status"], "ready")
        self.assertEqual(read_state(directory)["names"], {})

    def test_ordinary_usage_of_known_alias_still_requires_explicit_rule_scope(self):
        from epub_tts_narrator.renderer.cli import main as renderer_main

        class OrdinaryClient(AutoNameClient):
            def complete(self, system, payload):
                if payload.get("mode") == "name_script":
                    return {"decisions": ["ordinary"] * len(payload["window"]["candidates"])}
                return super().complete(system, payload)

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", OrdinaryClient), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", OrdinaryClient):
            code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        registry = self.book / "pronunciations.json"
        before = registry.read_bytes()
        output = io.StringIO()

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = renderer_main(["pronunciations", str(self.book), "--config", str(self.config),
                                  "--action", "set", "--word", "Ally", "--phonemes", "AE1 L IY0"])

        self.assertNotEqual(code, 0)
        self.assertIn("--class", output.getvalue())
        self.assertEqual(registry.read_bytes(), before)

    def test_pending_name_script_resumes_after_provider_failure_without_new_book_analysis(self):
        class FailingClient(AutoNameClient):
            def complete(self, system, payload):
                if payload.get("mode") == "name_script":
                    raise ProviderError("offline name identification outage")
                return super().complete(system, payload)

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", FailingClient), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", FailingClient):
            code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 1, output)
        self.assertTrue(preparation_pending(self.book, self.document))
        state = read_state(self.book)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["windows"], [])
        analysis_bytes = (self.book / "analysis.json").read_bytes()
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_bytes)

    def test_stale_binding_reprepares_and_restores_binding(self):
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertFalse(preparation_pending(self.book, self.document))
        state = read_state(self.book)
        stale = deepcopy(state)
        stale["binding"] = {"source": "0" * 64, "identities": "0" * 64, "version": 1}
        save_state(self.book, stale)
        self.assertTrue(preparation_pending(self.book, self.document))
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual(read_state(self.book)["binding"], binding(self.document))

    def test_absent_or_corrupt_state_is_pending(self):
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
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

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", DisagreeClient), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", DisagreeClient):
            code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
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
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=AssertionError("Manual resolution called provider")), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", side_effect=AssertionError("Manual resolution called provider")):
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

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", InterruptedClient), \
                patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", InterruptedClient):
            code, output = self.run_cli("analyze", self.book)
            self.assertEqual(code, 1, output)
            state = read_state(self.book)
            self.assertEqual(state["status"], "pending")
            self.assertEqual(len(state["windows"]), 1)
            self.assertEqual(InterruptedClient.events, [("b1", 1), ("b1", 2), ("b2", 1), ("b2", 2)])
            analysis_bytes = (self.book / "analysis.json").read_bytes()

            code, output = self.run_cli("analyze", self.book)

        self.assertEqual(code, 0, output)
        self.assertEqual(InterruptedClient.events,
                         [("b1", 1), ("b1", 2), ("b2", 1), ("b2", 2), ("b2", 2)])
        self.assertEqual(read_state(self.book)["status"], "ready")
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_bytes)
