"""Rerunning analysis completes saved books without rediscovering their cast."""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_audiobook.analysis_directory import (
    _completed_directory, continuation_directories, continuation_registry,
    fresh_registry, publish_directory,
)
from epub_audiobook.audition_scripts import prepare_document_samples
from epub_audiobook.analysis_revision import related_directories
from epub_audiobook.cli import export_bundle, main, source_from
from epub_audiobook.pipeline import CachedRequests, analyze, fingerprint, write_json
from epub_audiobook.provider import ProviderError
from epub_audiobook.series import link_analysis
from epub_audiobook.speaker_completion import complete_speakers, unassigned_segments
import test_directory_cli as directory_fixtures
from test_pipeline import FakeClient


class Terminal(StringIO):
    def isatty(self):
        return True


class CompletionClient:
    model = "fixture"
    base_url = "http://offline.invalid/v1"
    identity = {"provider": "resume-test", "model": "fixture"}

    def __init__(self, *, fail_after=None, uncertain=False):
        self.calls = []
        self.fail_after = fail_after
        self.uncertain = uncertain

    def complete(self, system, payload):
        if payload.get("mode") != "speaker_completion":
            raise AssertionError("A resumed analysis must only request missing speaker decisions")
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ProviderError("Fixture timeout")
        self.calls.append(deepcopy(payload))
        unit = payload["target_units"][0]
        evidence = [{"block_id": unit["block_id"], "quote": unit["text"]}]
        roles = {role["kind"]: role["id"] for role in payload["final_cast"]["roles"]}
        pieces = []
        for original in payload["original_segments"]:
            piece = {key: deepcopy(original[key]) for key in ("text", "kind", "delivery")}
            missing = self.uncertain and original["role_id"] is None
            piece.update(role_id=None if missing else roles["dialogue" if piece["kind"] == "dialogue" else "narration"],
                         basis="unresolved" if missing else "explicit", reason="The source names the speaker.",
                         evidence=evidence, candidates=[], review_required=missing)
            pieces.append(piece)
        return {"block_id": unit["block_id"], "reason": "Source-bound decision.",
                "evidence": evidence, "segments": pieces}


class TruncatedCompletionClient(CompletionClient):
    """Two complete reviewers cannot outvote a reviewer with no usable answer."""

    def __init__(self, *, failed_block=None):
        super().__init__()
        self.failed_block = failed_block

    def complete(self, system, payload):
        if (payload["reviewer_id"] == "speaker_completion_3"
                and (self.failed_block is None or payload["target_units"][0]["block_id"] == self.failed_block)):
            self.calls.append(deepcopy(payload))
            raise ProviderError("The model output was truncated. Increase the output token limit or reduce the input.")
        return super().complete(system, payload)


class AnalyzeResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = self.root / "global.toml"
        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="http://offline.invalid/v1"\n'
                               '[analysis]\nattribution_workers=1\nrepair_attempts=0\n')

    def seed(self, number=1, *, missing=True, missing_all=False, parent=None, prepared=False,
             advisory=False, completed_jury=False):
        epub = directory_fixtures.DirectoryCLITests.make_epub(self, number)
        source = source_from(epub)
        document = analyze(source, FakeClient(), self.root / "seed-cache", identity_consensus=False,
                           attribution_review=False, attribution_workers=1)
        path = self.root / f"book-{number:02d}"
        registry = continuation_registry(parent, source) if parent else fresh_registry(path)
        document = link_analysis(document, registry)
        if missing:
            rows = (document["segments"] if missing_all else
                    [next(item for item in document["segments"] if item["kind"] == "dialogue")])
            for row in rows:
                row.update(role_id=None, voice_id=None)
                row["attribution"].update(basis="unresolved", review_required=True, candidates=[], reason="No consensus.")
        if advisory:
            document["issues"].append({"id": "optional-note", "kind": "casting", "target": document["cast"]["voices"][0]["id"],
                                       "reason": "Optional performance nuance.", "resolved": False})
        if prepared:
            document = prepare_document_samples(document, CachedRequests(FakeClient(), self.root / "audition-cache"))
        if completed_jury:
            document = complete_speakers(document, CachedRequests(
                CompletionClient(uncertain=True), self.root / "completed-jury-cache", repair_attempts=0), workers=1)
        publish_directory(document, registry, path, export_bundle, continue_from=parent)
        (path / "voices").mkdir()
        (path / "voices" / "reference.wav").write_bytes(b"Existing voice remains unchanged")
        return epub, path, document

    def run_cli(self, *args, answers="", client=None, command="analyze"):
        output, error = StringIO(), StringIO()
        with (patch("sys.stdin", Terminal(answers)), contextlib.redirect_stdout(output),
              contextlib.redirect_stderr(error),
              patch("epub_audiobook.cli.analyze", side_effect=AssertionError("Must reuse saved analysis")),
              patch("epub_audiobook.cli.OpenAICompatibleClient", return_value=client or CompletionClient()) as provider):
            status = main([command, *map(str, args), "--config", str(self.config)])
        return status, output.getvalue(), error.getvalue(), provider

    def snapshot(self, path):
        return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}

    def test_same_epub_and_output_resume_only_missing_speakers_with_original_cast(self):
        epub, path, original = self.seed()
        client = CompletionClient()
        status, output, error, _ = self.run_cli(epub, "-o", path, client=client)
        self.assertEqual(status, 0, error)
        self.assertEqual(len(client.calls), 3)
        updated = _completed_directory(path)[0]
        self.assertEqual(unassigned_segments(updated), [])
        self.assertEqual(updated["cast"], original["cast"])
        self.assertEqual(updated["source"], original["source"])
        self.assertIn("Resuming saved analysis", output)
        self.assertTrue(list((path / ".analysis-revisions").glob("*/revision.json")))
        self.assertEqual((path / "voices/reference.wav").read_bytes(), b"Existing voice remains unchanged")

    def test_directory_input_accepts_omitted_or_matching_output_and_reprompts_after_jury(self):
        for explicit in (False, True):
            with self.subTest(explicit_output=explicit):
                _, path, _ = self.seed(2 if explicit else 1)
                args = [path, "-o", path] if explicit else [path]
                client = CompletionClient(uncertain=True)
                status, _, error, _ = self.run_cli(*args, answers="2\n", client=client)
                self.assertEqual(status, 0, error)
                result = _completed_directory(path)[0]
                self.assertFalse(unassigned_segments(result))
                self.assertTrue(any(row["attribution"]["basis"] == "user" for row in result["segments"]))
                self.assertIn("Speaker number(s)", error)

    def test_advisory_notes_alone_need_no_model_or_file_rewrite(self):
        epub, path, _ = self.seed(missing=False, advisory=True)
        self.assertTrue(json.loads((path / "review.json").read_text()))
        before = self.snapshot(path)
        for args in ([path], [epub, "-o", path]):
            status, output, error, provider = self.run_cli(*args)
            self.assertEqual(status, 0, error)
            provider.assert_not_called()
            self.assertIn("no model calls or file changes", output)
            self.assertEqual(self.snapshot(path), before)

    def test_wrong_epub_cannot_overwrite_existing_book_or_call_model(self):
        _, path, _ = self.seed()
        other = directory_fixtures.DirectoryCLITests.make_epub(self, 2)
        before = self.snapshot(path)
        status, _, error, provider = self.run_cli(other, "-o", path)
        self.assertEqual(status, 1)
        self.assertIn("Input source does not match", error)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(path), before)

    def test_invalid_existing_bundle_is_rejected_before_model_calls(self):
        _, path, _ = self.seed()
        (path / "cast.json").write_text("{}")
        before = self.snapshot(path)
        status, _, error, provider = self.run_cli(path)
        self.assertEqual(status, 1)
        self.assertIn("checksum mismatch", error)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(path), before)

    def test_continue_from_must_match_saved_parent_and_parent_is_unchanged(self):
        _, parent, _ = self.seed(missing=False)
        epub, current, _ = self.seed(2, parent=parent)
        original_parent = self.snapshot(parent)
        status, _, error, provider = self.run_cli(epub, "-o", current, "--continue-from", self.root / "wrong")
        self.assertEqual(status, 1)
        self.assertIn("saved continuation parent", error)
        provider.assert_not_called()
        status, _, error, _ = self.run_cli(epub, "-o", current, "--continue-from", parent)
        self.assertEqual(status, 0, error)
        self.assertEqual(continuation_directories(current), [parent])
        self.assertEqual(self.snapshot(parent), original_parent)

    def test_parent_revision_refreshes_descendant_links_and_keeps_descendant_voices(self):
        _, parent, _ = self.seed()
        _, child, original = self.seed(2, parent=parent, missing=False)
        old_child_manifest = (child / "manifest.json").read_bytes()
        status, _, error, _ = self.run_cli(parent)
        self.assertEqual(status, 0, error)
        self.assertEqual(continuation_directories(child), [parent])
        self.assertNotEqual((child / "manifest.json").read_bytes(), old_child_manifest)
        self.assertEqual(_completed_directory(child)[0]["cast"], original["cast"])
        self.assertEqual((child / "voices/reference.wav").read_bytes(), b"Existing voice remains unchanged")

    def test_existing_prepared_auditions_refresh_locally_after_human_selection(self):
        _, path, _ = self.seed(prepared=True)
        with patch("epub_audiobook.audition_scripts.prepare_document_samples", wraps=prepare_document_samples) as prepare:
            status, _, error, _ = self.run_cli(path, answers="2\n", client=CompletionClient(uncertain=True))
        self.assertEqual(status, 0, error)
        prepare.assert_called_once()
        self.assertIsNone(prepare.call_args.args[1])

    def test_timeout_preserves_bundle_and_reuses_completed_jury_requests_on_next_run(self):
        _, path, _ = self.seed()
        before = self.snapshot(path)
        first = CompletionClient(fail_after=1)
        status, _, error, _ = self.run_cli(path, client=first)
        self.assertEqual(status, 1)
        self.assertIn("Resume cache", error)
        self.assertEqual(self.snapshot(path), before)
        retry = CompletionClient()
        status, _, error, _ = self.run_cli(path, client=retry)
        self.assertEqual(status, 0, error)
        self.assertEqual(len(first.calls) + len(retry.calls), 3)

    def test_quitting_unresolved_speakers_saves_progress_and_returns_incomplete(self):
        _, path, _ = self.seed()
        status, _, error, _ = self.run_cli(path, answers="q\n", client=CompletionClient(uncertain=True))
        self.assertEqual(status, 2, error)
        self.assertIn("Speaker selection incomplete", error)
        self.assertEqual(len(unassigned_segments(_completed_directory(path)[0])), 1)

    def test_completed_unresolved_jury_prompts_immediately_without_provider_or_configured_model(self):
        _, path, original = self.seed(completed_jury=True)
        self.config.write_text('[analysis]\nattribution_workers=1\n')
        status, _, error, provider = self.run_cli(path, answers="2\n")
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        self.assertIn("Speaker number(s)", error)
        updated = _completed_directory(path)[0]
        self.assertFalse(unassigned_segments(updated))
        self.assertEqual(updated["source"], original["source"])
        self.assertEqual(updated["cast"], original["cast"])

    def test_resolve_speakers_reuses_completed_jury_without_provider_or_configured_model(self):
        _, path, _ = self.seed(completed_jury=True)
        self.config.write_text('[analysis]\nattribution_workers=1\n')
        status, _, error, provider = self.run_cli(path, answers="2\n", command="resolve-speakers")
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        self.assertIn("Speaker number(s)", error)
        self.assertFalse(unassigned_segments(_completed_directory(path)[0]))

    def test_partial_manual_answers_do_not_restart_completed_juries_for_remaining_blocks(self):
        _, path, original = self.seed(missing_all=True, completed_jury=True)
        self.assertGreater(len(unassigned_segments(original)), 2)
        status, _, error, provider = self.run_cli(path, answers="1\n1\nq\n")
        self.assertEqual(status, 2, error)
        provider.assert_not_called()
        intermediate = _completed_directory(path)[0]
        choices = {row["id"]: row for row in intermediate["segments"] if row["attribution"]["basis"] == "user"}
        self.assertEqual(len(unassigned_segments(intermediate)), len(unassigned_segments(original)) - 2)
        self.assertTrue({row["block_id"] for row in choices.values()} &
                        {row["block_id"] for row in unassigned_segments(intermediate)},
                        "A saved choice and a remaining passage must share one source block")
        answers = "1\n" * len(unassigned_segments(intermediate))
        status, _, error, provider = self.run_cli(path, answers=answers)
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        updated = _completed_directory(path)[0]
        self.assertFalse(unassigned_segments(updated))
        self.assertEqual({row["id"]: row for row in updated["segments"] if row["id"] in choices}, choices)
        self.assertEqual(updated["source"], original["source"])
        self.assertEqual(updated["cast"], original["cast"])

    def test_exhausted_truncation_reaches_manual_prompt_without_accepting_two_votes(self):
        _, path, original = self.seed()
        client = TruncatedCompletionClient()
        status, _, error, _ = self.run_cli(path, answers="1\n", client=client)
        self.assertEqual(status, 0, error)
        self.assertIn("truncated", error)
        self.assertIn("Speaker number(s)", error)
        self.assertEqual(len(client.calls), 4, "One truncated vote gets just one identical retry")
        self.assertEqual({payload["round"] for payload in client.calls}, {1})
        updated = _completed_directory(path)[0]
        self.assertFalse(unassigned_segments(updated))
        chosen = next(row for row in updated["segments"] if row["attribution"]["basis"] == "user")
        narrator = next(role for role in original["cast"]["roles"] if role["kind"] == "narration")
        self.assertEqual(chosen["role_id"], narrator["id"], "Only the human, not two reviewers, chose this performer")
        previous_assigned = {row["id"]: row for row in original["segments"] if row["role_id"] is not None}
        self.assertEqual({row["id"]: row for row in updated["segments"] if row["id"] in previous_assigned},
                         previous_assigned)

    def test_quitting_after_truncation_saves_manual_fallback_and_resume_makes_no_api_calls(self):
        _, path, _ = self.seed()
        client = TruncatedCompletionClient()
        status, _, error, _ = self.run_cli(path, answers="q\n", client=client)
        self.assertEqual(status, 2, error)
        self.assertIn("Speaker number(s)", error)
        self.assertEqual(len(client.calls), 4)
        saved = _completed_directory(path)[0]
        self.assertEqual(len(unassigned_segments(saved)), 1)
        status, _, error, provider = self.run_cli(path, answers="2\n")
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        self.assertIn("Speaker number(s)", error)
        self.assertFalse(unassigned_segments(_completed_directory(path)[0]))

    def test_truncated_block_keeps_other_automatic_assignments_when_manual_selection_quits(self):
        _, path, original = self.seed(missing_all=True)
        last_block = unassigned_segments(original)[-1]["block_id"]
        client = TruncatedCompletionClient(failed_block=last_block)
        status, _, error, _ = self.run_cli(path, answers="q\n", client=client)
        self.assertEqual(status, 2, error)
        self.assertIn("Speaker number(s)", error)
        saved = _completed_directory(path)[0]
        self.assertEqual({row["block_id"] for row in unassigned_segments(saved)}, {last_block})
        self.assertTrue(all(row["role_id"] is not None for row in saved["segments"] if row["block_id"] != last_block))
        self.assertEqual(saved["source"], original["source"])
        self.assertEqual(saved["cast"], original["cast"])

    def test_source_bound_attribution_edits_apply_without_model_calls(self):
        _, path, document = self.seed()
        row = unassigned_segments(document)[0]
        role = next(item for item in document["cast"]["roles"] if item["kind"] == "dialogue")
        edits = self.root / "edits.json"
        write_json(edits, {"source_fingerprint": fingerprint(document["source"]),
                           "cast_fingerprint": fingerprint(document["cast"]),
                           "assignments": [{"segment_id": row["id"], "role_id": role["id"]}]})
        status, _, error, provider = self.run_cli(path, "--edits", edits)
        self.assertEqual(status, 0, error)
        provider.assert_not_called()
        updated = _completed_directory(path)[0]
        self.assertFalse(unassigned_segments(updated))
        self.assertEqual(updated["cast"], document["cast"])

    def test_cast_edits_are_explicitly_rejected_instead_of_ignored(self):
        _, path, document = self.seed()
        cast = deepcopy(document["cast"])
        cast["voices"][0]["description"] = "A different voice proposal"
        edits = self.root / "cast-edits.json"
        write_json(edits, {"source_fingerprint": fingerprint(document["source"]),
                           "cast_fingerprint": fingerprint(document["cast"]), "cast": cast})
        before = self.snapshot(path)
        status, _, error, provider = self.run_cli(path, "--edits", edits)
        self.assertEqual(status, 1)
        self.assertIn("attribution edits only", error)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(path), before)

    def test_directory_input_rejects_different_output_and_force(self):
        _, path, _ = self.seed()
        for args, expected in ((["-o", self.root / "other"], "same directory"), (["--force"], "legacy .json")):
            with self.subTest(args=args):
                status, _, error, provider = self.run_cli(path, *args)
                self.assertEqual(status, 1)
                self.assertIn(expected, error)
                provider.assert_not_called()

    def test_concurrent_revision_between_load_and_hash_inventory_is_not_overwritten(self):
        _, path, _ = self.seed()
        def concurrent_change(paths):
            result = related_directories(paths)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["concurrent_revision"] = "preserve this newer manifest"
            write_json(manifest_path, manifest, overwrite=True)
            return result
        with patch("epub_audiobook.analysis_revision.related_directories", side_effect=concurrent_change):
            status, _, error, provider = self.run_cli(path)
        self.assertEqual(status, 1)
        self.assertIn("Analysis changed while loading", error)
        provider.assert_not_called()
        self.assertEqual(json.loads((path / "manifest.json").read_text())["concurrent_revision"],
                         "preserve this newer manifest")
        self.assertFalse((path / ".analysis-revisions").exists())


if __name__ == "__main__":
    unittest.main()
