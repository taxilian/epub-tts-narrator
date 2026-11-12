import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_progress import CACHE_DIRNAME
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

    def test_fresh_analysis_creates_no_pronunciation_candidates_and_resume_needs_no_model(self):
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        self.assertEqual(code, 0, output)
        entries = load_pronunciations(self.book)
        self.assertEqual(entries, [])
        self.assertFalse((self.book / "pronunciations.json").exists())
        self.assertFalse((self.book / "name-script.json").exists())
        before = snapshot(self.book)
        calls = list(CandidateClient.modes)
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, calls)
        self.assertEqual(snapshot(self.book), before)

    def test_discovery_outage_cannot_fail_analysis_or_trigger_backfill_on_resume(self):
        CandidateClient.fail_scan = True
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        self.assertEqual(code, 0, output)
        analysis = (self.book / "analysis.json").read_bytes()
        self.assertTrue(discovery_pending(self.book, json.loads(analysis)))
        CandidateClient.fail_scan = False
        CandidateClient.modes = []
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, [])
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis)

    def test_proposal_outage_cannot_fail_analysis_or_trigger_suggestions_on_resume(self):
        CandidateClient.fail_proposal = True
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        self.assertEqual(code, 0, output)
        analysis = (self.book / "analysis.json").read_bytes()
        document = json.loads(analysis)
        self.assertTrue(discovery_pending(self.book, document))
        self.assertTrue(preparation_pending(self.book, document))
        self.assertEqual(load_pronunciations(self.book), [])
        CandidateClient.fail_proposal = False
        CandidateClient.modes = []
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, [])
        self.assertTrue(preparation_pending(self.book, document))
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis)

    def test_readable_correction_proposer_is_lazy_cached_and_never_confirms_a_guess(self):
        from epub_tts_narrator.pronunciation_workflow import create_pronunciation_proposer
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        self.assertEqual(code, 0, output)
        before = snapshot(self.book)
        CandidateClient.modes = []
        propose = create_pronunciation_proposer(self.book, self.config, progress=lambda message: None)
        self.assertEqual(CandidateClient.modes, [])
        self.assertEqual(propose("Qarm", "Quarm"), {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertEqual(propose("Qarm", "Quarm"), {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"})
        self.assertEqual(CandidateClient.modes, ["pronunciation_proposal"])
        # Only the book's own request cache may change; published files stay byte-identical.
        published = lambda files: {name: data for name, data in files.items()
                                   if not name.startswith(CACHE_DIRNAME + "/")}
        self.assertEqual(published(snapshot(self.book)), published(before))

    def test_resume_preserves_user_cmu_choice_without_pronunciation_backfill(self):
        CandidateClient.fail_scan = True
        self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        document = json.loads((self.book / "analysis.json").read_text())
        value = {"alphabet": "cmu_arpabet", "value": "K W AO1 R M"}
        set_pronunciation(self.book, document, "Qarm", value)
        files = {path: path.read_bytes() for path in self.book.rglob("*") if path.is_file()
                 and path.name not in {"pronunciations.json"}}
        CandidateClient.fail_scan = False
        CandidateClient.modes = []
        code, output = self.run_cli("analyze", self.book)
        self.assertEqual(code, 0, output)
        entry = load_pronunciations(self.book)[0]
        self.assertEqual(entry["status"], "confirmed")
        self.assertEqual(entry["pronunciation"], value)
        self.assertEqual(CandidateClient.modes, [])
        for path, data in files.items():
            self.assertEqual(path.read_bytes(), data)

    def test_public_renderer_cli_saves_cmu_correction_without_engine_or_asr_configuration(self):
        from epub_tts_narrator.renderer.cli import main as render_main
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
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

    def test_resume_preserves_legacy_pending_entries_and_discovery_marker_without_model_work(self):
        from epub_tts_narrator.pipeline import write_json
        from epub_tts_narrator.pronunciations import DISCOVERY_VERSION, save_candidates, word_id
        code, output = self.run_cli("analyze", self.epub, "-o", self.book, "--strategy", "consensus")
        self.assertEqual(code, 0, output)
        document = json.loads((self.book / "analysis.json").read_text())
        block = document["source"]["blocks"][0]
        legacy = {"id": word_id("door"), "word": "door", "kind": "name", "status": "pending",
                  "reason": "Legacy model candidate.", "evidence": [{"block_id": block["id"], "quote": "door"}]}
        save_candidates(self.book, document, [legacy])
        registry = json.loads((self.book / "pronunciations.json").read_text())
        registry["discovery"]["version"] = DISCOVERY_VERSION - 1
        write_json(self.book / "pronunciations.json", registry, overwrite=True)
        before = (self.book / "pronunciations.json").read_bytes()
        CandidateClient.modes = []

        code, output = self.run_cli("analyze", self.book)

        self.assertEqual(code, 0, output)
        self.assertEqual(CandidateClient.modes, [])
        self.assertEqual((self.book / "pronunciations.json").read_bytes(), before)
        stored = {row["word"]: row for row in load_pronunciations(self.book)}
        self.assertNotIn("proposal", stored["door"])
        self.assertTrue(preparation_pending(self.book, document))
