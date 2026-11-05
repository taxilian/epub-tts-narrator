import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.config import load_config
from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.pronunciation_workflow import create_pronunciation_proposer
from epub_tts_narrator.scoped_pronunciations import scope_for
from name_script_fixtures import NameClient, PHONES, book, requests


class ScopedPronunciationProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = book(self.root / "book", two=True)

    def test_proposal_context_identifies_only_selected_entity_and_its_mentions(self):
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        client = NameClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        config = self.root / "config.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n')
        with patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", return_value=client):
            proposer = create_pronunciation_proposer(self.bundle["directory"], config,
                progress=lambda message: None, scope=scope_for(self.bundle, "N2"))
            self.assertEqual(client.calls, [])
            self.assertEqual(proposer("Ally"), PHONES)
            self.assertEqual(proposer("Ally"), PHONES)
        self.assertEqual(len(client.calls), 1)
        request = client.calls[0]
        self.assertEqual(request["scope"]["character_id"], self.bundle["cast"]["characters"][1]["id"])
        self.assertEqual([item["block_id"] for item in request["context"]], ["b2"])
        self.assertEqual([item["character_id"] for item in request["cast_context"]], [request["scope"]["character_id"]])

    def test_explicit_legacy_preparation_uses_configured_client_and_resume_needs_no_calls(self):
        client = NameClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        config = self.root / "config.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n')
        with patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", return_value=client), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prepare_name_script(self.bundle["directory"], load_config(config))
            calls = len(client.calls)
            prepare_name_script(self.bundle["directory"], load_config(config))
        self.assertEqual(calls, 4)
        self.assertEqual(len(client.calls), calls)

    def test_special_proposal_uses_ordinary_occurrences_not_same_spelling_character(self):
        from name_script_fixtures import TEXT
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        client = NameClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        config = self.root / "config.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n')
        with patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", return_value=client):
            proposer = create_pronunciation_proposer(self.bundle["directory"], config,
                progress=lambda message: None, scope={"class": "special"})
            proposer("Ally")
        self.assertEqual(client.calls[0]["cast_context"], [])
        self.assertEqual([item["evidence"]["start"] for item in client.calls[0]["context"]], [TEXT.index("Ally with")])

    def test_renderer_manual_set_honors_saved_scope_without_model_requests(self):
        from epub_tts_narrator.renderer.cli import main as renderer_main
        from epub_tts_narrator.scoped_pronunciations import load
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        client = NameClient()
        client.model, client.base_url = "fixture", "https://offline.invalid"
        config = self.root / "config.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="https://offline.invalid"\n')
        with patch("epub_tts_narrator.pronunciation_workflow.OpenAICompatibleClient", return_value=client), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = renderer_main(["pronunciations", str(self.bundle["directory"]), "--class", "name", "--name-id", "N2",
                                  "--action", "set", "--word", "Ally", "--phonemes", PHONES["value"],
                                  "--config", str(config)])
        self.assertEqual(code, 0)
        rows = load(self.bundle["directory"], scope_for(self.bundle, "N2"))
        self.assertEqual(rows[0]["pronunciation"], PHONES)
        self.assertEqual(rows[0]["status"], "confirmed")
        self.assertEqual(load(self.bundle["directory"], scope_for(self.bundle, "N1")), [])
        self.assertEqual(client.calls, [])
