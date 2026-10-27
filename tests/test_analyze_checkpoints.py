"""Interrupted directory analysis checkpoints each stage and resumes without redoing work."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.analysis_progress import (
    PENDING_DIRNAME, STAGES, inspect_pending, is_pending_directory,
)
from epub_tts_narrator.audition_scripts import prepare_document_samples
from epub_tts_narrator.cli import main
from epub_tts_narrator.pipeline import analyze
from epub_tts_narrator.speaker_completion import unassigned_segments
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from epub_tts_narrator.speech_normalization import normalize_document_speech
from shared_fixtures import Terminal, make_epub
from test_pipeline import FakeClient

EPUB_BODY = ('<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head>'
             '<body><h1>One</h1><p>Mara, the lighthouse keeper, said softly, “Hello.”</p>'
             '<p>They left.</p></body></html>')

# Each checkpointed stage is owned by one CLI-visible function; the local imports
# inside the analyze flow resolve these modules on every run.
STAGE_FUNCTIONS = {
    "analyzed": "epub_tts_narrator.cli.analyze",
    "speech-normalized-1": "epub_tts_narrator.speech_normalization.normalize_document_speech",
    "samples-prepared-1": "epub_tts_narrator.audition_scripts.prepare_document_samples",
    "speakers-decided": "epub_tts_narrator.cli.prompt_speaker_decisions",
    "speech-normalized-2": "epub_tts_narrator.speech_normalization.normalize_document_speech",
    "samples-prepared-2": "epub_tts_narrator.audition_scripts.prepare_document_samples",
}
REAL_FUNCTIONS = {
    "epub_tts_narrator.cli.analyze": analyze,
    "epub_tts_narrator.speech_normalization.normalize_document_speech": normalize_document_speech,
    "epub_tts_narrator.audition_scripts.prepare_document_samples": prepare_document_samples,
    "epub_tts_narrator.cli.prompt_speaker_decisions": prompt_speaker_decisions,
}
# Model work that only the `analyzed` stage performs; a resume past it must
# never repeat these requests.
ANALYSIS_MODES = {None, "discover", "reconcile", "voice_ensemble", "identity_scan"}


class PipelineClient(FakeClient):
    """FakeClient shaped like OpenAICompatibleClient for the CLI boundary."""

    def __init__(self, model, base_url, *, unresolved=False, **kwargs):
        super().__init__(unresolved=unresolved)
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}


class AnalyzeCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(contextlib.chdir(self.root))
        self.config = self.root / "epub-audiobook.toml"
        self.config.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus = false\nattribution_review = false\n'
            'repair_attempts = 0\nattribution_workers = 5\n', encoding="utf-8")
        self.epub = make_epub(self.root / "story.epub", "Test Story", "test-story", EPUB_BODY)

    def clients(self, **flags):
        made = []

        def factory(*args, **kwargs):
            client = PipelineClient(*args, **kwargs, **flags)
            made.append(client)
            return client

        return factory, made

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def spy(self, target):
        calls = []
        real = REAL_FUNCTIONS[target]

        def wrapper(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        return wrapper, calls

    def interrupter(self, target, call_number):
        calls = []
        real = REAL_FUNCTIONS[target]

        def wrapper(*args, **kwargs):
            calls.append(1)
            if len(calls) == call_number:
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        return wrapper

    def analyze_command(self, output):
        return ("analyze", self.epub, "-o", output, "--config", self.config)

    def test_pending_directory_is_created_before_the_first_model_request(self):
        output = self.root / "book"
        factory, _ = self.clients()
        observed = []
        real_analyze = analyze

        def spy(source, client, cache, **kwargs):
            observed.append((output / PENDING_DIRNAME / "state.json").is_file())
            return real_analyze(source, client, cache, **kwargs)

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("epub_tts_narrator.cli.analyze", side_effect=spy):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        self.assertEqual(observed, [True])
        self.assertFalse((output / PENDING_DIRNAME).exists(),
                         "A successful publish must remove the pending state")
        _completed_directory(output)

    def test_interrupt_during_each_stage_resumes_there_without_redoing_completed_stages(self):
        for index, stage in enumerate(STAGES):
            with self.subTest(stage=stage):
                output = self.root / f"book-{index}-{stage}"
                target = STAGE_FUNCTIONS[stage]
                call_number = 1 + sum(1 for earlier in STAGES[:index]
                                      if STAGE_FUNCTIONS[earlier] == target)
                factory, _ = self.clients()
                with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                        patch(target, side_effect=self.interrupter(target, call_number)):
                    status, _, stderr = self.run_cli(*self.analyze_command(output))
                self.assertEqual(status, 130, stderr)
                self.assertIn("checkpointed analysis progress", stderr)
                record = inspect_pending(output)
                self.assertEqual(record["completed_stages"], list(STAGES[:index]))
                self.assertEqual(record["status"], "interrupted")
                self.assertTrue(is_pending_directory(output))
                # The rerun resumes at the interrupted stage; earlier stages are
                # neither re-executed locally nor re-requested from the model.
                factory, made = self.clients()
                spies = {}
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory))
                    for target_name in set(STAGE_FUNCTIONS.values()):
                        wrapper, calls = self.spy(target_name)
                        stack.enter_context(patch(target_name, side_effect=wrapper))
                        spies[target_name] = calls
                    status, _, stderr = self.run_cli(*self.analyze_command(output))
                self.assertEqual(status, 0, stderr)
                expected = {
                    "epub_tts_narrator.cli.analyze": 1 if index == 0 else 0,
                    "epub_tts_narrator.speech_normalization.normalize_document_speech":
                        (1 if index <= 1 else 0) + (1 if index <= 4 else 0),
                    "epub_tts_narrator.audition_scripts.prepare_document_samples":
                        (1 if index <= 2 else 0) + 1,
                    "epub_tts_narrator.cli.prompt_speaker_decisions": 1 if index <= 3 else 0,
                }
                for target_name, count in expected.items():
                    self.assertEqual(len(spies[target_name]), count,
                                     f"{target_name} ran {len(spies[target_name])}x, expected {count}")
                if index > 0:
                    modes = {payload.get("mode") for client in made for _, payload in client.calls}
                    self.assertFalse(modes & ANALYSIS_MODES,
                                     f"completed analysis stages issued new model requests: {modes}")
                _completed_directory(output)
                self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_failure_records_failed_status_and_a_rerun_completes(self):
        output = self.root / "book"
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("epub_tts_narrator.speech_normalization.normalize_document_speech",
                      side_effect=ValueError("Fixture normalization failure")):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 1, stderr)
        self.assertIn("Fixture normalization failure", stderr)
        record = inspect_pending(output)
        self.assertEqual(record["status"], "failed")
        self.assertIn("Fixture normalization failure", record["error"])
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        _completed_directory(output)
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_resume_into_speakers_decided_replays_the_speaker_decision_journal(self):
        output = self.root / "book"
        cache = self.root / ".epub-audiobook-cache"
        pre_prompt = []

        def capture_and_interrupt(document, **kwargs):
            pre_prompt.append(deepcopy(document))
            result = prompt_speaker_decisions(document, **kwargs)
            raise KeyboardInterrupt

        factory, _ = self.clients(unresolved=True)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("sys.stdin", Terminal("1\n")), \
                patch("epub_tts_narrator.cli.prompt_speaker_decisions",
                      side_effect=capture_and_interrupt):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 130, stderr)
        journals = list((cache / "speaker-decisions").glob("*.json"))
        self.assertEqual(len(journals), 1)
        record = json.loads(journals[0].read_text(encoding="utf-8"))
        self.assertEqual(record["scope"], "speaker-decisions")
        self.assertEqual(record["status"], "complete")
        self.assertEqual(len(record["decisions"]), 1)
        # The rerun restores the byte-identical pre-prompt document from the
        # snapshot, so the journal key matches and replays without a terminal.
        replayed = []

        def capture_replay(document, **kwargs):
            replayed.append(document)
            return prompt_speaker_decisions(document, **kwargs)

        factory, _ = self.clients(unresolved=True)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("sys.stdin", Terminal("")), \
                patch("epub_tts_narrator.cli.prompt_speaker_decisions",
                      side_effect=capture_replay):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        self.assertIn("Replaying 1 saved speaker decision", stderr)
        self.assertEqual(replayed[0], pre_prompt[0])
        document, _, _ = _completed_directory(output)
        self.assertFalse(unassigned_segments(document))
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_existing_output_dispatches_completed_pending_and_foreign_directories(self):
        completed = self.root / "completed-book"
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(completed))
        self.assertEqual(status, 0, stderr)
        # A completed directory still goes to resume_book, never to reanalysis.
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("epub_tts_narrator.cli.analyze",
                      side_effect=AssertionError("resume_book must not reanalyze")):
            status, stdout, stderr = self.run_cli(*self.analyze_command(completed))
        self.assertEqual(status, 0, stderr)
        self.assertIn("Saved analysis", stdout)
        self.assertFalse((completed / PENDING_DIRNAME).exists())
        # A pending directory resumes the checkpointed run instead of resume_book.
        pending = self.root / "pending-book"
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("epub_tts_narrator.cli.analyze", side_effect=KeyboardInterrupt):
            status, _, _ = self.run_cli(*self.analyze_command(pending))
        self.assertEqual(status, 130)
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, stdout, stderr = self.run_cli(*self.analyze_command(pending))
        self.assertEqual(status, 0, stderr)
        self.assertNotIn("Resuming saved analysis", stdout)
        _completed_directory(pending)
        # A foreign directory is refused without any model work, as before.
        foreign = self.root / "foreign"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("unrelated", encoding="utf-8")
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli(*self.analyze_command(foreign))
        self.assertEqual(status, 1)
        provider.assert_not_called()
        self.assertIn("manifest.json", stderr)
        self.assertEqual([path.name for path in foreign.iterdir()], ["notes.txt"])


if __name__ == "__main__":
    unittest.main()
