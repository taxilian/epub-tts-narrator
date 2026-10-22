"""Optional casting auditions cannot discard completed book analysis."""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.audition_scripts import audition_preparation_pending
from epub_tts_narrator.cli import main
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.provider import ProviderError
from test_contracts import valid_document
from test_pipeline import FakeClient
from audition_fixtures import script_response


class AuditionClient(FakeClient):
    model, base_url = "fixture-model", "http://offline.invalid/v1"

    def __init__(self, *, fail=True, provider_error=False, auditions_only=False):
        super().__init__()
        self.fail = fail
        self.provider_error = provider_error
        self.auditions_only = auditions_only
        self.audition_calls = []

    def complete(self, system, payload):
        if payload.get("mode") == "audition_script":
            self.audition_calls.append(deepcopy(payload))
            if self.fail and payload["characters"]:
                if self.provider_error:
                    raise ProviderError("Fixture audition service unavailable")
                return {"text": "Yesss. " + script_response()["text"],
                        "reason": "A character-specific audition written in stylized spelling."}
            return script_response()
        if self.auditions_only:
            raise AssertionError("A completed book must retry only its missing audition")
        return super().complete(system, payload)


class AuditionAnalysisCompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "book"
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps({"format": "epub-audiobook-source", "version": 1,
                                           "source": valid_document()["source"]}))
        self.config = self.root / "global.toml"
        self.config.write_text(
            '[provider]\nmodel="fixture-model"\nbase_url="http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus=false\nattribution_review=false\n'
            'attribution_workers=1\nrepair_attempts=1\n')

    def command(self, client, *, resume=False):
        output, error = StringIO(), StringIO()
        arguments = ["analyze", str(self.output)] if resume else ["analyze", str(self.source), "-o", str(self.output)]
        arguments.extend(["--config", str(self.config)])
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(error),
              patch("sys.stdin", StringIO()),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client) as provider):
            status = main(arguments)
        return status, output.getvalue(), error.getvalue(), provider

    def test_validation_exhaustion_saves_book_and_good_auditions_then_retries_only_missing(self):
        first = AuditionClient()
        status, output, error, _ = self.command(first)
        self.assertEqual(status, 0, error)
        self.assertIn("Bundle created", output)
        saved = _completed_directory(self.output)[0]
        self.assertEqual(validate_document(saved), [])
        self.assertEqual(saved["source"], valid_document()["source"])
        self.assertTrue(audition_preparation_pending(saved))
        prepared = saved["provenance"]["voice_samples"]
        self.assertEqual(len(prepared["scripts"]), 1)
        self.assertEqual(len(prepared["pending"]), 1)
        failed_voice = next(iter(prepared["pending"]))
        self.assertNotIn(failed_voice, prepared["scripts"])
        self.assertNotIn(failed_voice, prepared["samples"])
        self.assertIn("Mara", prepared["pending"][failed_voice]["name"])
        self.assertIn("Yesss", prepared["pending"][failed_voice]["error"])
        self.assertEqual(len(first.audition_calls), 3)  # one success + two rejected responses
        cached = self.root / ".epub-audiobook-cache"
        self.assertTrue(any("Yesss" in path.read_text() for path in (cached / "failed").glob("*.json")))
        self.assertFalse(any(json.loads(path.read_text()).get("text", "").startswith("Yesss")
                             for path in cached.glob("*.json")))
        (self.output / "voices").mkdir()
        voice_reference = self.output / "voices" / "reference.wav"
        voice_reference.write_bytes(b"Keep established cloning reference")
        original_source, original_cast = deepcopy(saved["source"]), deepcopy(saved["cast"])
        original_script = deepcopy(prepared["scripts"])

        resumed = AuditionClient(fail=False, auditions_only=True)
        status, _, error, _ = self.command(resumed, resume=True)
        self.assertEqual(status, 0, error)
        complete = _completed_directory(self.output)[0]
        self.assertFalse(audition_preparation_pending(complete))
        self.assertEqual(complete["source"], original_source)
        self.assertEqual(complete["cast"], original_cast)
        self.assertEqual([call["voice"]["id"] for call in resumed.audition_calls], [failed_voice])
        for voice, script in original_script.items():
            self.assertEqual(complete["provenance"]["voice_samples"]["scripts"][voice], script)
        self.assertEqual(voice_reference.read_bytes(), b"Keep established cloning reference")
        before = {path.relative_to(self.output): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        status, _, error, provider = self.command(AuditionClient(auditions_only=True), resume=True)
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        self.assertEqual({path.relative_to(self.output): path.read_bytes()
                          for path in self.output.rglob("*") if path.is_file()}, before)

    def test_optional_audition_provider_failure_still_publishes_complete_analysis(self):
        status, _, error, _ = self.command(AuditionClient(provider_error=True))
        self.assertEqual(status, 0, error)
        saved = _completed_directory(self.output)[0]
        self.assertEqual(validate_document(saved), [])
        self.assertTrue(audition_preparation_pending(saved))
        pending = saved["provenance"]["voice_samples"]["pending"]
        self.assertIn("Fixture audition service unavailable", next(iter(pending.values()))["error"])
        self.assertEqual(len(saved["provenance"]["voice_samples"]["scripts"]), 1)


if __name__ == "__main__":
    unittest.main()
