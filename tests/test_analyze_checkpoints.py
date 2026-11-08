"""Interrupted directory analysis checkpoints each stage and resumes without redoing work."""

from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.analysis_progress import (
    CACHE_DIRNAME, PENDING_DIRNAME, PROGRESS_VERSION, STAGES, PendingAnalysis, create_pending,
    inspect_pending, is_pending_directory,
)
from epub_tts_narrator.audition_scripts import prepare_document_samples
from epub_tts_narrator.cli import main
from epub_tts_narrator.pipeline import analyze
from epub_tts_narrator.speaker_completion import unassigned_segments
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from shared_fixtures import Terminal, make_epub, snapshot
from test_pipeline import FakeClient

EPUB_BODY = ('<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head>'
             '<body><h1>One</h1><p>Mara, the lighthouse keeper, said softly, “Hello.”</p>'
             '<p>They left.</p></body></html>')

STAGE_FUNCTIONS = {
    "analyzed": "epub_tts_narrator.cli.analyze",
    "samples-prepared-1": "epub_tts_narrator.audition_scripts.prepare_document_samples",
    "speakers-decided": "epub_tts_narrator.cli.prompt_speaker_decisions",
    "samples-prepared-2": "epub_tts_narrator.audition_scripts.prepare_document_samples",
}
REAL_FUNCTIONS = {
    "epub_tts_narrator.cli.analyze": analyze,
    "epub_tts_narrator.audition_scripts.prepare_document_samples": prepare_document_samples,
    "epub_tts_narrator.cli.prompt_speaker_decisions": prompt_speaker_decisions,
}
# Model work that only the `analyzed` stage performs; a resume past it must
# never repeat these requests.
ANALYSIS_MODES = {None, "discover", "reconcile", "voice_ensemble", "identity_scan"}
RETIRED_MODES = {"speech_spelling_scan", "speech_spelling_review", "pronunciation_discovery",
                 "pronunciation_proposal", "name_script"}
RETIRED_TARGETS = (
    "epub_tts_narrator.speech_normalization.normalize_document_speech",
    "epub_tts_narrator.pronunciation_workflow.prepare_book_pronunciations",
    "epub_tts_narrator.pronunciation_discovery.discover_pronunciations",
    "epub_tts_narrator.pronunciation_proposals.propose_name_pronunciations",
    "epub_tts_narrator.name_script_workflow.prepare_name_script",
)
PRONUNCIATION_SIDECARS = ("pronunciations.json", "name-script.json", "scoped-pronunciations.json")


def interrupt_stage(stage, call_number=1, error=None):
    if stage in STAGE_FUNCTIONS:
        target = STAGE_FUNCTIONS[stage]
        calls = []
        real = REAL_FUNCTIONS[target]

        def wrapper(*args, **kwargs):
            calls.append(1)
            if len(calls) == call_number:
                raise error or KeyboardInterrupt
            return real(*args, **kwargs)

        return patch(target, side_effect=wrapper)
    real_complete = PendingAnalysis.complete_stage

    def checkpoint(pending, current, document):
        if current == stage:
            raise error or KeyboardInterrupt
        return real_complete(pending, current, document)

    return patch("epub_tts_narrator.analysis_progress.PendingAnalysis.complete_stage", autospec=True,
                 side_effect=checkpoint)


class PipelineClient(FakeClient):
    """FakeClient shaped like OpenAICompatibleClient for the CLI boundary."""

    def __init__(self, model, base_url, *, unresolved=False, **kwargs):
        super().__init__(unresolved=unresolved)
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}

    def complete(self, system, payload):
        if payload.get("mode") in RETIRED_MODES:
            raise AssertionError(f"Retired pronunciation request: {payload['mode']}")
        return super().complete(system, payload)


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
        with contextlib.ExitStack() as stack:
            for target in RETIRED_TARGETS:
                stack.enter_context(patch(target, side_effect=AssertionError("Automatic pronunciation work is retired")))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            status = main([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def spy(self, target):
        calls = []
        real = REAL_FUNCTIONS[target]

        def wrapper(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        return wrapper, calls

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

    def test_deleting_output_removes_the_cache_and_forces_fresh_model_work(self):
        output = self.root / "book"
        factory, first = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        cache = output / CACHE_DIRNAME
        self.assertTrue(cache.is_dir())
        self.assertFalse((self.root / ".epub-audiobook-cache").exists())
        first_calls = len(first[0].calls)
        self.assertGreater(first_calls, 0)
        shutil.rmtree(output)
        self.assertFalse(cache.exists())
        factory, second = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        self.assertEqual(len(second[0].calls), first_calls)
        self.assertFalse((self.root / ".epub-audiobook-cache").exists())

    def test_interrupt_during_each_stage_resumes_there_without_redoing_completed_stages(self):
        for index, stage in enumerate(STAGES):
            with self.subTest(stage=stage):
                output = self.root / f"book-{index}-{stage}"
                target = STAGE_FUNCTIONS.get(stage)
                call_number = (1 + sum(1 for earlier in STAGES[:index]
                                       if STAGE_FUNCTIONS.get(earlier) == target)) if target else 1
                factory, _ = self.clients()
                with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                        interrupt_stage(stage, call_number):
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
                document, _, _ = _completed_directory(output)
                self.assertFalse(any("speech_normalization" in row for row in document["segments"]))
                self.assertFalse(any((output / name).exists() for name in PRONUNCIATION_SIDECARS))
                self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_failure_at_an_inert_checkpoint_records_failed_status_and_a_rerun_completes(self):
        output = self.root / "book"
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                interrupt_stage("speech-normalized-1", error=ValueError("Fixture checkpoint failure")):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 1, stderr)
        self.assertIn("Fixture checkpoint failure", stderr)
        record = inspect_pending(output)
        self.assertEqual(record["status"], "failed")
        self.assertIn("Fixture checkpoint failure", record["error"])
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        _completed_directory(output)
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_resume_into_speakers_decided_replays_the_speaker_decision_journal(self):
        output = self.root / "book"
        cache = output / CACHE_DIRNAME
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


class AnalyzeCheckpointEdgeCaseTests(unittest.TestCase):
    """CLI-level refusal and isolation edges around checkpointed resume."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(contextlib.chdir(self.root))
        self.config = self.root / "epub-audiobook.toml"
        self.write_config()
        self.epub = make_epub(self.root / "story.epub", "Test Story", "test-story", EPUB_BODY)

    def write_config(self, *, chunk_chars=2400, model="fixture-model", reasoning_effort=None):
        provider = f'[provider]\nmodel = "{model}"\nbase_url = "http://offline.invalid/v1"\n'
        if reasoning_effort is not None:
            provider += f'reasoning_effort = "{reasoning_effort}"\n'
        self.config.write_text(
            provider +
            f'[analysis]\nchunk_chars = {chunk_chars}\nidentity_consensus = false\n'
            'attribution_review = false\nrepair_attempts = 0\nattribution_workers = 5\n',
            encoding="utf-8")

    def clients(self, **flags):
        made = []

        def factory(*args, **kwargs):
            client = PipelineClient(*args, **kwargs, **flags)
            made.append(client)
            return client

        return factory, made

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            for target in RETIRED_TARGETS:
                stack.enter_context(patch(target, side_effect=AssertionError("Automatic pronunciation work is retired")))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            status = main([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def spy(self, target):
        calls = []
        real = REAL_FUNCTIONS[target]

        def wrapper(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        return wrapper, calls

    def analyze_command(self, output):
        return ("analyze", self.epub, "-o", output, "--config", self.config)

    @staticmethod
    def model_calls(made):
        return [payload for client in made for _, payload in client.calls]

    def interrupt_after_analyzed(self, output):
        """Checkpoint exactly the analyzed stage, then interrupt the run."""
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                interrupt_stage("speech-normalized-1"):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 130, stderr)
        self.assertEqual(inspect_pending(output)["completed_stages"], ["analyzed"])

    def state_bytes(self, output):
        return (output / PENDING_DIRNAME / "state.json").read_bytes()

    def test_rerun_with_a_changed_epub_refuses_and_preserves_pending_progress(self):
        output = self.root / "book"
        self.interrupt_after_analyzed(output)
        before = self.state_bytes(output)
        other = make_epub(self.root / "other.epub", "Other Story", "other-story",
                          EPUB_BODY.replace("Mara", "Jules"))
        factory, made = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli("analyze", other, "-o", output, "--config", self.config)
        self.assertEqual(status, 1)
        self.assertIn("different run configuration", stderr)
        self.assertIn("identity.source_fingerprint", stderr)
        self.assertIn("state.json", stderr)
        self.assertEqual(self.model_calls(made), [], "A refused rerun must not issue model requests")
        # The refusal never discards the checkpointed progress.
        self.assertEqual(self.state_bytes(output), before)
        self.assertEqual(inspect_pending(output)["completed_stages"], ["analyzed"])
        # The original command still resumes the preserved progress.
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        _completed_directory(output)
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_rerun_with_changed_options_or_prompt_version_is_refused_by_name(self):
        for field in ("identity.options_fingerprint", "identity.prompt_version"):
            with self.subTest(field=field):
                output = self.root / f"book-{field.split('.')[-1]}"
                self.write_config()
                self.interrupt_after_analyzed(output)
                before = self.state_bytes(output)
                mutation = contextlib.nullcontext()
                if field == "identity.options_fingerprint":
                    self.write_config(chunk_chars=1200)
                else:
                    mutation = patch("epub_tts_narrator.analysis_progress.PROMPT_VERSION", "0-foreign")
                factory, made = self.clients()
                with mutation, patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
                    status, _, stderr = self.run_cli(*self.analyze_command(output))
                self.assertEqual(status, 1)
                self.assertIn("different run configuration", stderr)
                self.assertIn(field, stderr)
                self.assertIn("state.json", stderr)
                self.assertEqual(self.model_calls(made), [], "A refused rerun must not issue model requests")
                self.assertEqual(self.state_bytes(output), before)
                self.assertEqual(inspect_pending(output)["completed_stages"], ["analyzed"])

    def test_model_or_reasoning_change_resumes_completed_stages(self):
        output = self.root / "book-provider-change"
        self.write_config()
        self.interrupt_after_analyzed(output)
        self.write_config(model="other-model", reasoning_effort="low")
        factory, _ = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        _completed_directory(output)
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_rerun_with_a_tampered_state_record_fails_closed_naming_state_json(self):
        for mutate in ("version", "record_sha256"):
            with self.subTest(mutate=mutate):
                output = self.root / f"book-{mutate}"
                self.interrupt_after_analyzed(output)
                path = output / PENDING_DIRNAME / "state.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                if mutate == "version":
                    record["version"] = PROGRESS_VERSION + 1
                else:
                    record["error"] = "tampered outside the signed record"
                path.write_text(json.dumps(record), encoding="utf-8")
                factory, made = self.clients()
                with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
                    status, _, stderr = self.run_cli(*self.analyze_command(output))
                self.assertEqual(status, 1)
                self.assertIn("Invalid pending analysis record", stderr)
                self.assertIn("state.json", stderr)
                self.assertIn(mutate, stderr)
                self.assertEqual(self.model_calls(made), [], "A refused rerun must not issue model requests")
                # No silent restart: the tampered record stays, and nothing is published.
                self.assertTrue(is_pending_directory(output))
                self.assertFalse((output / "manifest.json").exists())
                tampered = json.loads(path.read_text(encoding="utf-8"))
                if mutate == "version":
                    self.assertEqual(tampered["version"], PROGRESS_VERSION + 1)
                else:
                    self.assertEqual(tampered["error"], "tampered outside the signed record")

    def test_pending_directory_with_foreign_content_is_refused_without_touching_anything(self):
        output = self.root / "book"
        self.interrupt_after_analyzed(output)
        stray = output / "stray-notes.txt"
        stray.write_text("not analysis state", encoding="utf-8")
        before = self.state_bytes(output)
        # A stray sibling makes the directory unrecognizable as pending, so the
        # run refuses it as an incomplete foreign directory instead of resuming.
        self.assertFalse(is_pending_directory(output))
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 1)
        self.assertIn("manifest.json", stderr)
        provider.assert_not_called()
        self.assertEqual(stray.read_text(encoding="utf-8"), "not analysis state")
        self.assertEqual(self.state_bytes(output), before)
        record = json.loads((output / PENDING_DIRNAME / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(record["completed_stages"], ["analyzed"])

    @unittest.skipUnless(os.name == "posix", "POSIX fcntl required")
    def test_concurrent_analyze_is_refused_until_the_pending_lease_is_released(self):
        output = self.root / "book"
        self.interrupt_after_analyzed(output)
        holder = create_pending(output, inspect_pending(output)["identity"])
        try:
            factory, made = self.clients()
            with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
                status, _, stderr = self.run_cli(*self.analyze_command(output))
            self.assertEqual(status, 1)
            self.assertIn("Another analysis is running", stderr)
            self.assertEqual(self.model_calls(made), [], "A refused rerun must not issue model requests")
            self.assertEqual(inspect_pending(output)["completed_stages"], ["analyzed"])
        finally:
            holder.close()
        # Once the lease is released, the same command attaches and resumes
        # without repeating the checkpointed analyzed stage.
        factory, _ = self.clients()
        wrapper, calls = self.spy("epub_tts_narrator.cli.analyze")
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                patch("epub_tts_narrator.cli.analyze", side_effect=wrapper):
            status, _, stderr = self.run_cli(*self.analyze_command(output))
        self.assertEqual(status, 0, stderr)
        self.assertEqual(calls, [])
        _completed_directory(output)
        self.assertFalse((output / PENDING_DIRNAME).exists())

    def test_different_output_gets_fresh_checkpoints_without_pronunciation_work(self):
        first = self.root / "book-01"
        factory, made = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(first))
        self.assertEqual(status, 0, stderr)
        _completed_directory(first)
        first_modes = [payload.get("mode") for payload in self.model_calls(made)]
        self.assertTrue(RETIRED_MODES.isdisjoint(first_modes))
        self.assertFalse(any((first / name).exists() for name in PRONUNCIATION_SIDECARS))
        published = snapshot(first)
        # The second output starts its own pending checkpoints from the first stage.
        second = self.root / "book-02"
        factory, second_run = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory), \
                interrupt_stage("speech-normalized-1"):
            status, _, stderr = self.run_cli(*self.analyze_command(second))
        self.assertEqual(status, 130, stderr)
        self.assertEqual(inspect_pending(second)["completed_stages"], ["analyzed"])
        factory, third_run = self.clients()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=factory):
            status, _, stderr = self.run_cli(*self.analyze_command(second))
        self.assertEqual(status, 0, stderr)
        _completed_directory(second)
        self.assertFalse((second / PENDING_DIRNAME).exists())
        self.assertFalse(any((second / name).exists() for name in PRONUNCIATION_SIDECARS))
        self.assertEqual(snapshot(first), published, "The first book's published output stays untouched")
        rerun_modes = [payload.get("mode") for payload in self.model_calls(second_run + third_run)]
        self.assertTrue(RETIRED_MODES.isdisjoint(rerun_modes))


if __name__ == "__main__":
    unittest.main()
