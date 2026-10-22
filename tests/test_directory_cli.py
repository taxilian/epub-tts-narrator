"""The normal CLI needs one global config and immutable per-book handoffs."""

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.cli import main, read_json
from epub_tts_narrator.pipeline import analyze
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.series import validate_registry
from test_pipeline import FakeClient


class ContinuityClient(FakeClient):
    """Give the real pipeline evidence for the recurring lighthouse keeper."""

    def __init__(self, model, base_url, **kwargs):
        super().__init__()
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}
        self.fail_reconciliation = False

    def complete(self, system, payload):
        if payload.get("mode") == "speech_spelling_scan":
            self.calls.append((system, deepcopy(payload)))
            return {"segments": [{"segment_id": row["segment_id"], "edits": []}
                                 for row in payload["target_segments"]]}
        return super().complete(system, payload)

    def discover(self, payload):
        result = super().discover(payload)
        prior = payload.get("series_cast", {}).get("characters", [])
        if result["characters"] and prior:
            previous = next(character for character in prior if character["name"] == "Mara")
            unit = next(unit for unit in payload["target_units"] if "lighthouse keeper" in unit["text"])
            result["characters"][0]["series_identity"] = {
                "character_id": previous["id"], "basis": "context", "review_required": False,
                "reason": "The returning lighthouse keeper establishes continuity beyond the shared name.",
                "evidence": [{"block_id": unit["block_id"], "quote": unit["text"]}],
            }
        return result

    def reconcile(self, payload):
        if self.fail_reconciliation:
            raise ProviderError("Fixture connection interrupted after discovery")
        return super().reconcile(payload)


class DirectoryCLITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "epub-audiobook.toml"
        self.config.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus = false\nattribution_review = false\n'
            'repair_attempts = 0\nattribution_workers = 5\n', encoding="utf-8")
        self.clients = []

    @contextlib.contextmanager
    def cwd(self):
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            yield
        finally:
            os.chdir(previous)

    def client(self, *args, **kwargs):
        client = ContinuityClient(*args, **kwargs)
        self.clients.append(client)
        return client

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with self.cwd(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main([str(arg) for arg in args])
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def make_epub(self, number=1):
        path = self.root / f"Book {number}.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
            archive.writestr("content.opf", f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Book {number}</dc:title><dc:identifier id="id">book-{number}</dc:identifier><dc:language>en</dc:language></metadata><manifest><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/></spine></package>')
            archive.writestr("one.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head><body><h1>One</h1><p>Mara, the lighthouse keeper, said softly, “Hello.”</p><p>They left.</p></body></html>')
        return path

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def seed_book(self):
        source, output = self.make_epub(), self.root / "book-one"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", source, "-o", output)
        self.assertEqual(status, 0, stderr)
        return source, output

    def seed_second_book(self, prior):
        output = self.root / "book-two"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", self.make_epub(2), "-o", output,
                                           "--continue-from", prior)
        self.assertEqual(status, 0, stderr)
        return output

    def assert_bundle(self, output, book_count):
        bundle = load_bundle(output)
        document = read_json(output / "analysis.json")
        registry = read_json(output / "series-cast.json")
        validate_registry(registry)
        self.assertEqual(len(registry["books"]), book_count)
        self.assertEqual(bundle["cast"], document["cast"])
        self.assertEqual(bundle["source"], document["source"])
        self.assertEqual(bundle["manifest"]["provenance"]["series"]["series_id"], registry["series_id"])
        self.assertTrue((output / "scripts" / "chapter-0001.jsonl").is_file())
        self.assertFalse((output / "script.jsonl").exists())
        return document, registry

    def test_one_config_runs_three_books_with_cumulative_immutable_history(self):
        original_config = self.config.read_bytes()
        outputs, prior_snapshots, documents, registries = [], [], [], []
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client), \
                patch("epub_tts_narrator.cli.analyze", wraps=analyze) as pipeline:
            for number in range(1, 4):
                output = self.root / f"book-{number}"
                args = ["analyze", self.make_epub(number), "-o", output]
                if outputs:
                    args += ["--continue-from", outputs[-1]]
                status, _, stderr = self.run_cli(*args)
                self.assertEqual(status, 0, stderr)
                for previous, before in zip(outputs, prior_snapshots):
                    self.assertEqual(self.snapshot(previous), before)
                document, registry = self.assert_bundle(output, number)
                documents.append(document)
                registries.append(registry)
                outputs.append(output)
                prior_snapshots.append(self.snapshot(output))
            self.assertEqual(pipeline.call_count, 3)
            self.assertTrue(all(call.kwargs["attribution_workers"] == 5 for call in pipeline.call_args_list))
        self.assertEqual(self.config.read_bytes(), original_config)
        self.assertEqual(len({registry["series_id"] for registry in registries}), 1)
        self.assertEqual(len({document["cast"]["characters"][0]["id"] for document in documents}), 1)
        self.assertEqual(len({document["cast"]["characters"][0]["voice_id"] for document in documents}), 1)
        for index, client in enumerate(self.clients):
            discovery = next(payload for _, payload in client.calls if payload.get("mode") == "discover")
            self.assertEqual(len(discovery["series_cast"]["prior_books"]), index)
        self.assertEqual(registries[-1]["books"][:2], registries[1]["books"])

    def test_default_output_is_a_ready_to_load_book_directory(self):
        epub = self.make_epub()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", epub)
        self.assertEqual(status, 0, stderr)
        self.assert_bundle(self.root / "Book 1.audiobook", 1)

    def test_unassigned_passages_save_bundle_without_terminal_and_return_incomplete_status(self):
        epub, output = self.make_epub(), self.root / "incomplete-book"

        def unresolved_client(*args, **kwargs):
            client = self.client(*args, **kwargs)
            client.unresolved = True
            return client

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=unresolved_client), \
                patch("sys.stdin", io.StringIO()), \
                patch("epub_tts_narrator.speaker_decisions.open", side_effect=OSError("No controlling terminal")):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 2, stderr)
        document, _ = self.assert_bundle(output, 1)
        self.assertFalse(document["readiness"]["ready"])
        self.assertFalse(load_bundle(output)["manifest"]["readiness"]["ready"])
        self.assertTrue(read_json(output / "review.json"))
        self.assertIn("Bundle created", stdout)
        self.assertIn("no interactive terminal is available", stderr)
        self.assertIn("Speaker selection incomplete", stderr)
        self.assertNotIn("draft", stdout)

    def test_assigned_advisory_items_create_usable_bundle_without_review_step(self):
        epub, output = self.make_epub(), self.root / "advisory-book"

        def advisory_analysis(*args, **kwargs):
            document = analyze(*args, **kwargs)
            document["segments"][0]["attribution"]["review_required"] = True
            return document

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client), \
                patch("epub_tts_narrator.cli.analyze", side_effect=advisory_analysis):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 0, stderr)
        document, _ = self.assert_bundle(output, 1)
        self.assertTrue(document["readiness"]["ready"])
        self.assertTrue(load_bundle(output)["manifest"]["readiness"]["ready"])
        self.assertTrue(read_json(output / "review.json"))
        self.assertIn("advisory notes. Bundle created", stdout)
        self.assertIn("No review step is required", stdout)
        self.assertNotIn("draft", stdout)

    def test_plan_for_a_continuation_does_not_call_model_or_write_any_files(self):
        _, prior = self.seed_book()
        epub, output = self.make_epub(2), self.root / "planned-book"
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, stdout, stderr = self.run_cli("plan", epub, "-o", output, "--continue-from", prior)
        self.assertEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(json.loads(stdout)["series"]["books"], 1)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse(output.exists())

    def test_default_plan_does_not_create_the_output_or_cache(self):
        epub = self.make_epub()
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, stdout, stderr = self.run_cli("plan", epub)
        self.assertEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(json.loads(stdout)["series"]["books"], 0)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse((self.root / "Book 1.audiobook").exists())

    def test_failed_run_resumes_completed_requests_using_the_same_command(self):
        epub, output = self.make_epub(), self.root / "resumable"

        def fail_once(*args, **kwargs):
            client = self.client(*args, **kwargs)
            client.fail_reconciliation = len(self.clients) == 1
            return client

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=fail_once):
            status, _, stderr = self.run_cli("analyze", epub, "-o", output)
            self.assertEqual(status, 1, stderr)
            self.assertIn("Fixture connection interrupted", stderr)
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "series-cast.json").exists())
            status, _, stderr = self.run_cli("analyze", epub, "-o", output)
            self.assertEqual(status, 0, stderr)
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(any(payload.get("mode") == "discover" for _, payload in self.clients[0].calls))
        self.assertFalse(any(payload.get("mode") == "discover" for _, payload in self.clients[1].calls))
        self.assert_bundle(output, 1)

    def test_legacy_json_output_remains_supported(self):
        epub, output = self.make_epub(), self.root / "legacy.json"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
            status, _, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(read_json(output)["format"], "epub-audiobook-analysis")
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertFalse((self.root / "series-cast.json").exists())

    def test_legacy_json_output_saves_incomplete_analysis_and_returns_status_two(self):
        epub, output = self.make_epub(), self.root / "incomplete-legacy.json"

        def unresolved_client(*args, **kwargs):
            client = self.client(*args, **kwargs)
            client.unresolved = True
            return client

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=unresolved_client), \
                patch("sys.stdin", io.StringIO()), \
                patch("epub_tts_narrator.speaker_decisions.open", side_effect=OSError("No controlling terminal")):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 2, stderr)
        document = read_json(output)
        self.assertEqual(document["format"], "epub-audiobook-analysis")
        self.assertFalse(document["readiness"]["ready"])
        self.assertIn("Speaker selection incomplete", stderr)
        self.assertIn("Exiting with status 2", stderr)
        self.assertNotIn("draft", stdout)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_continuation_rejects_conflicting_legacy_options_before_model_calls(self):
        _, prior = self.seed_book()
        epub = self.make_epub(2)
        cases = [
            ["-o", self.root / "legacy.json"],
            ["-o", self.root / "new-book", "--series", self.root / "other-series.json"],
        ]
        for extra in cases:
            with self.subTest(extra=extra), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                before = self.snapshot(self.root)
                status, _, stderr = self.run_cli("analyze", epub, "--continue-from", prior, *extra)
                self.assertNotEqual(status, 0, stderr)
                provider.assert_not_called()
                self.assertEqual(self.snapshot(self.root), before)
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write('series_registry = "other-series.json"\n')
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            before = self.snapshot(self.root)
            status, _, stderr = self.run_cli("analyze", epub, "--continue-from", prior,
                                           "-o", self.root / "new-book")
        self.assertNotEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(self.root), before)

    def test_missing_or_invalid_continuation_fails_before_model_calls(self):
        epub = self.make_epub()
        invalid = self.root / "invalid-prior"
        invalid.mkdir()
        (invalid / "series-cast.json").write_text('{"format":"wrong"}', encoding="utf-8")
        empty = self.root / "empty-prior"
        empty.mkdir()
        for previous in (self.root / "missing-prior", invalid, empty):
            for command in ("analyze", "plan"):
                with self.subTest(previous=previous, command=command), \
                        patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                    before = self.snapshot(self.root)
                    status, _, stderr = self.run_cli(command, epub, "-o", self.root / "new-book",
                                                   "--continue-from", previous)
                    self.assertNotEqual(status, 0, stderr)
                    provider.assert_not_called()
                    self.assertEqual(self.snapshot(self.root), before)

    def test_same_book_cannot_be_appended_as_its_own_sequel(self):
        epub, prior = self.seed_book()
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli("analyze", epub, "-o", self.root / "duplicate",
                                           "--continue-from", prior)
        self.assertNotEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(self.root), before)

    def test_output_cannot_replace_or_nest_inside_continuation_directory(self):
        _, prior = self.seed_book()
        epub = self.make_epub(2)
        for output in (prior, prior / "nested-output", prior.parent):
            with self.subTest(output=output), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                before = self.snapshot(self.root)
                status, _, stderr = self.run_cli("analyze", epub, "-o", output, "--continue-from", prior)
                self.assertNotEqual(status, 0, stderr)
                provider.assert_not_called()
                self.assertEqual(self.snapshot(self.root), before)

    def test_cache_cannot_write_into_prior_book_even_with_an_external_output(self):
        _, prior = self.seed_book()
        epub = self.make_epub(2)
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write(f'cache_dir = "{prior / "new-cache"}"\n')
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli("analyze", epub, "-o", self.root / "new-book",
                                           "--continue-from", prior)
        self.assertNotEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(self.root), before)

    def test_third_book_output_cannot_be_inside_a_grandparent_analysis(self):
        _, first = self.seed_book()
        second = self.seed_second_book(first)
        epub = self.make_epub(3)
        output = first / "nested-third-book"
        before = self.snapshot(self.root)
        for command in ("analyze", "plan"):
            with self.subTest(command=command), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                status, _, stderr = self.run_cli(command, epub, "-o", output,
                                               "--continue-from", second)
                self.assertNotEqual(status, 0, stderr)
                self.assertIn("every ancestor book", stderr)
                provider.assert_not_called()
                self.assertFalse(output.exists())
                self.assertEqual(self.snapshot(self.root), before)

    def test_third_book_cache_cannot_be_inside_a_grandparent_analysis(self):
        _, first = self.seed_book()
        second = self.seed_second_book(first)
        epub, output, cache = self.make_epub(3), self.root / "book-three", first / "nested-cache"
        with self.config.open("a", encoding="utf-8") as stream:
            stream.write(f'cache_dir = "{cache}"\n')
        before = self.snapshot(self.root)
        for command in ("analyze", "plan"):
            with self.subTest(command=command), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                status, _, stderr = self.run_cli(command, epub, "-o", output,
                                               "--continue-from", second)
                self.assertNotEqual(status, 0, stderr)
                self.assertIn("every ancestor book", stderr)
                provider.assert_not_called()
                self.assertFalse(cache.exists())
                self.assertFalse(output.exists())
                self.assertEqual(self.snapshot(self.root), before)

    def test_existing_completed_output_is_preserved_without_model_calls_on_rerun(self):
        epub, output = self.seed_book()
        before = self.snapshot(output)
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, stderr = self.run_cli("analyze", epub, "-o", output)
        self.assertEqual(status, 0, stderr)
        provider.assert_not_called()
        self.assertEqual(self.snapshot(output), before)


if __name__ == "__main__":
    unittest.main()
