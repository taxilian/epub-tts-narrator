"""Real CLI analysis/discovery handoff, with only the remote model replaced."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.pronunciations import discovery_pending, preparation_pending, load_pronunciations, set_pronunciation
from epub_tts_narrator.provider import ProviderError
from shared_fixtures import make_epub, snapshot
from test_cli import NarrativeClient


class CandidateClient(NarrativeClient):
    modes = []
    fail_scan = False
    fail_proposal = False

    def complete(self, system, payload):
        type(self).modes.append(payload.get("mode", "attribution"))
        if payload.get("mode") == "pronunciation_proposal":
            if self.fail_proposal:
                raise ProviderError("offline phoneme proposal outage")
            return {"phonemes": "K W AO1 R M"}
        if payload.get("mode") == "pronunciation_discovery":
            if self.fail_scan:
                raise ProviderError("offline discovery outage")
            block = next(row for row in payload["target_blocks"] if "Qarm" in row["text"])
            return {"candidates": [{"word": "Qarm", "kind": "name", "reason": "Unusual spelling.",
                                    "evidence": [{"block_id": block["block_id"], "quote": "Qarm"}]}]}
        return super().complete(system, payload)


class PronunciationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(contextlib.chdir(self.root))
        self.config = self.root / "global.toml"
        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid/v1"\n')
        self.book = self.root / "book"
        self.epub = make_epub(self.root / "story.epub", "Test", "test",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Qarm opened the door.</p></body></html>')
        CandidateClient.modes = []
        CandidateClient.fail_scan = False
        CandidateClient.fail_proposal = False
        self.enterContext(patch("epub_tts_narrator.cli.OpenAICompatibleClient", CandidateClient))
        self.enterContext(patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", CandidateClient))

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main([str(item) for item in (*args, "--config", self.config)])
        return code, output.getvalue()

    def test_fresh_analysis_collects_unconfirmed_candidates_and_completed_resume_needs_no_model(self):
        code, output = self.run_cli("analyze", self.epub, "-o", self.book)
        self.assertEqual(code, 0, output)
        entries = load_pronunciations(self.book)
        self.assertEqual([(item["word"], item["status"]) for item in entries], [("Qarm", "pending")])
        self.assertEqual(entries[0]["proposal"], {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertNotIn("pronunciation", entries[0])
        before = snapshot(self.book)
        calls = list(CandidateClient.modes)
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, calls)
        self.assertEqual(snapshot(self.book), before)

    def test_discovery_outage_retains_book_and_resume_does_not_repeat_character_or_passage_analysis(self):
        CandidateClient.fail_scan = True
        code, output = self.run_cli("analyze", self.epub, "-o", self.book)
        self.assertEqual(code, 1, output)
        analysis = (self.book / "analysis.json").read_bytes()
        self.assertTrue(discovery_pending(self.book, json.loads(analysis)))
        CandidateClient.fail_scan = False
        CandidateClient.modes = []
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, ["pronunciation_discovery", "pronunciation_proposal"])
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis)

    def test_proposal_outage_resumes_without_repeating_discovery_or_book_analysis(self):
        CandidateClient.fail_proposal = True
        code, output = self.run_cli("analyze", self.epub, "-o", self.book)
        self.assertEqual(code, 1, output)
        analysis = (self.book / "analysis.json").read_bytes()
        document = json.loads(analysis)
        self.assertFalse(discovery_pending(self.book, document))
        self.assertTrue(preparation_pending(self.book, document))
        self.assertEqual(load_pronunciations(self.book)[0]["status"], "pending")
        CandidateClient.fail_proposal = False
        CandidateClient.modes = []
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, ["pronunciation_proposal"])
        self.assertFalse(preparation_pending(self.book, document))
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis)

    def test_readable_correction_proposer_is_lazy_cached_and_never_confirms_a_guess(self):
        from epub_tts_narrator.pronunciation_workflow import create_pronunciation_proposer
        code, output = self.run_cli("analyze", self.epub, "-o", self.book)
        self.assertEqual(code, 0, output)
        before = snapshot(self.book)
        CandidateClient.modes = []
        propose = create_pronunciation_proposer(self.book, self.config, progress=lambda message: None)
        self.assertEqual(CandidateClient.modes, [])
        self.assertEqual(propose("Qarm", "Quarm"), {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertEqual(propose("Qarm", "Quarm"), {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertEqual(CandidateClient.modes, ["pronunciation_proposal"])
        self.assertEqual(snapshot(self.book), before)

    def test_backfill_preserves_user_cmu_choice_and_does_not_change_analysis_or_scripts(self):
        CandidateClient.fail_scan = True
        self.run_cli("analyze", self.epub, "-o", self.book)
        document = json.loads((self.book / "analysis.json").read_text())
        value = {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"}
        set_pronunciation(self.book, document, "Qarm", value)
        files = {path: path.read_bytes() for path in self.book.rglob("*") if path.is_file()
                 and path.name not in {"pronunciations.json"}}
        CandidateClient.fail_scan = False
        CandidateClient.modes = []
        code, output = self.run_cli("discover-pronunciations", self.book)
        self.assertEqual(code, 0, output)
        entry = load_pronunciations(self.book)[0]
        self.assertEqual(entry["status"], "confirmed")
        self.assertEqual(entry["pronunciation"], value)
        self.assertEqual(CandidateClient.modes, ["pronunciation_discovery"])
        for path, data in files.items():
            self.assertEqual(path.read_bytes(), data)

    def test_public_renderer_cli_saves_cmu_correction_without_engine_or_asr_configuration(self):
        from epub_tts_narrator.renderer.cli import main as render_main
        code, output = self.run_cli("analyze", self.epub, "-o", self.book)
        self.assertEqual(code, 0, output)
        calls = list(CandidateClient.modes)
        output = io.StringIO()
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(output),
              patch("epub_tts_narrator.renderer.engines.create_engine", side_effect=AssertionError("No model needed"))):
            status = render_main(["pronunciations", str(self.book), "--config", str(self.config),
                                  "--action", "set", "--word", "Qarm", "--phonemes", "K W AO1 R M"])
        self.assertEqual(status, 0, output.getvalue())
        self.assertEqual(load_pronunciations(self.book)[0]["pronunciation"],
                         {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertEqual(CandidateClient.modes, calls)
