"""CLI tests run the analysis pipeline with scripted clients and never open a network socket."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle, main, read_json
from epub_tts_narrator.pipeline import fingerprint, write_json
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.series import link_analysis, load_registry, save_registry
from shared_fixtures import make_epub
from test_contracts import valid_document
from audition_fixtures import script_response


class NarrativeClient:
    calls = 0

    def __init__(self, model, base_url, **kwargs):
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}

    def complete(self, system, payload):
        type(self).calls += 1
        if payload.get("mode") == "speech_spelling_scan":
            return {"segments": [{"segment_id": row["segment_id"], "edits": []}
                                 for row in payload["target_segments"]]}
        if payload.get("mode") == "audition_script":
            return script_response()
        if payload.get("mode") == "attribution_review_scan":
            return {"corrections": []}
        if payload.get("mode") == "identity_scan":
            return {"cases": []}
        if "mode" in payload:
            return {"characters": [], "roles": [], "voices": [], "issues": [], "merges": [], "summary": "Narrative text."}
        return {"units": [{"unit_id": unit["unit_id"], "segments": [{
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narrative prose.", "evidence": [], "candidates": [],
            "review_required": False, "delivery": {"direction": "", "evidence": []},
        }]} for unit in payload["target_units"]]}


class StyledNarrativeClient(NarrativeClient):
    """Propose one source spelling edit and independently approve it."""

    def complete(self, system, payload):
        if payload.get("mode") == "speech_spelling_scan":
            segments = []
            for row in payload["target_segments"]:
                edits = []
                if "Nooooooo" in row["text"]:
                    start = row["text"].index("Nooooooo")
                    edits.append({"start": start, "end": start + len("Nooooooo"),
                                  "original": "Nooooooo", "replacement": "No",
                                  "direction": "Draw out the vowel in No.",
                                  "reason": "Repeated vowel spelling conveys a prolonged exclamation."})
                segments.append({"segment_id": row["segment_id"], "edits": edits})
            return {"segments": segments}
        if payload.get("mode") == "speech_spelling_review":
            return {"votes": [{"edit_id": proposal["edit_id"], "verdict": "approve",
                               "reason": "The same word is retained with its prolonged delivery."}
                              for proposal in payload["proposals"]]}
        return super().complete(system, payload)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # A real user's global config in the checkout must not affect fixtures
        # which intentionally exercise missing/default configuration.
        self.enterContext(contextlib.chdir(self.root))
        self.analysis = self.root / "analysis.json"
        self.config = self.root / "config.toml"
        self.config.write_text('[provider]\nmodel = "test-model"\nbase_url = "http://127.0.0.1:8000/v1"\n')
        write_json(self.analysis, valid_document())

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(a) for a in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def make_epub(self, paragraph="The rain stopped."):
        return make_epub(self.root / "story.epub", "Test Story", "test",
                         f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head><body><h1>One</h1><p>{paragraph}</p></body></html>')

    def test_fresh_analyze_exports_reviewed_spoken_spelling_without_changing_source(self):
        output = self.root / "styled-book"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", StyledNarrativeClient):
            status, _, stderr = self.run_cli("analyze", self.make_epub("Nooooooo!"),
                                           "-o", output, "--config", self.config)
        self.assertEqual(status, 0, stderr)
        analysis = read_json(output / "analysis.json")
        source = next(row for row in analysis["source"]["blocks"] if row["text"] == "Nooooooo!")
        segment = next(row for row in analysis["segments"] if row["block_id"] == source["id"])
        self.assertEqual(segment["text"], "Nooooooo!")
        self.assertEqual(segment["spoken_text"], "No!")
        self.assertEqual((segment["start"], segment["end"]), (0, len(source["text"])))
        decisions = analysis["provenance"]["speech_normalization"]["decisions"]
        self.assertEqual(len(decisions[0]["votes"]), 3)
        manifest = read_json(output / "manifest.json")
        script = [json.loads(line) for entry in manifest["files"]["scripts"]
                  for line in (output / entry["path"]).read_text().splitlines()]
        rendered = next(row for row in script if row["block_id"] == source["id"])
        self.assertEqual(rendered["spoken_text"], "No!")

    def test_cli_extract_analyze_cached_export_round_trip(self):
        epub = self.make_epub()
        source = self.root / "source.json"
        output = self.root / "new-analysis.json"
        self.assertEqual(self.run_cli("extract", epub, "-o", source)[0], 0)
        status, stdout, stderr = self.run_cli("plan", source)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(json.loads(stdout)["baseline_model_requests"], 5)
        self.assertEqual(json.loads(stdout)["speech_spelling_scan_requests_minimum"], 1)
        self.assertEqual(json.loads(stdout)["identity_scan_requests"], 1)
        NarrativeClient.calls = 0
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient):
            status, _, stderr = self.run_cli("analyze", source, "-o", output, "--config", self.config)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(NarrativeClient.calls, 6)
            status, _, stderr = self.run_cli("analyze", source, "-o", self.root / "second.json", "--config", self.config)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(NarrativeClient.calls, 6)
        self.assertEqual(self.run_cli("validate", output, "--ready")[0], 0)
        bundle = self.root / "bundle"
        self.assertEqual(self.run_cli("export", output, "-o", bundle)[0], 0)
        manifest = read_json(bundle / "manifest.json")
        self.assertTrue(manifest["readiness"]["ready"])
        for name, checksum in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), checksum)
        self.assertEqual(manifest["version"], 2)
        self.assertNotIn("script", manifest["files"])
        self.assertFalse((bundle / "script.jsonl").exists())
        script = [json.loads(line) for chapter in manifest["files"]["scripts"]
                  for line in (bundle / chapter["path"]).read_text().splitlines()]
        self.assertEqual([s["text"] for s in script], ["One", "The rain stopped."])
        self.assertEqual([s["sequence"] for s in script], [0, 1])
        self.assertTrue(all(s["break_after"] == "paragraph" for s in script))

    def test_plan_includes_scan_overhead_but_not_unknown_case_count(self):
        source_path = self.root / "long-source.json"
        source = valid_document()["source"]
        for block in source["blocks"]:
            block["text"] = "A paragraph of narrative prose. " * 20
        write_json(source_path, {"format": "epub-audiobook-source", "version": 1, "source": source})
        for whole_book in (False, True):
            for consensus in (False, True):
                with self.subTest(whole_book=whole_book, consensus=consensus):
                    self.config.write_text('[analysis]\nchunk_chars = 256\n'
                                           f'whole_book_discovery = {str(whole_book).lower()}\n'
                                           f'identity_consensus = {str(consensus).lower()}\n')
                    with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                        status, output, stderr = self.run_cli("plan", source_path, "--config", self.config)
                    self.assertEqual(status, 0, stderr)
                    provider.assert_not_called()
                    result = json.loads(output)
                    batches = result["analysis_batches"]
                    self.assertGreater(batches, 1)
                    scans = (1 if whole_book else batches) if consensus else 0
                    base = batches + 2 if whole_book else 2 * batches + 1
                    self.assertEqual(result["identity_consensus"], consensus)
                    self.assertEqual(result["identity_scan_requests"], scans)
                    self.assertEqual(result["baseline_model_requests"], base + scans + 1)
                    if consensus:
                        self.assertIn("unknown before analysis", result["note"])
                        self.assertIn("series matches", result["note"])

    def test_file_can_explicitly_disable_identity_consensus(self):
        self.config.write_text('[provider]\nmodel = "test-model"\n'
                               'base_url = "http://127.0.0.1:8000/v1"\n'
                               '[analysis]\nidentity_consensus = false\n')
        NarrativeClient.calls = 0
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient):
            status, _, stderr = self.run_cli("analyze", self.make_epub(), "--config", self.config,
                                           "-o", self.root / "without-consensus.json")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(NarrativeClient.calls, 5)

    def test_plan_counts_configured_cumulative_reconciliation_rounds(self):
        source_path = self.root / "long-source.json"
        source = valid_document()["source"]
        for block in source["blocks"]:
            block["text"] = "A paragraph of narrative prose. " * 20
        write_json(source_path, {"format": "epub-audiobook-source", "version": 1, "source": source})
        for whole_book in (False, True):
            for consensus in (False, True):
                for rounds in (2, 3):
                    with self.subTest(whole_book=whole_book, consensus=consensus, rounds=rounds):
                        self.config.write_text('[analysis]\nchunk_chars = 256\n'
                                               f'whole_book_discovery = {str(whole_book).lower()}\n'
                                               f'identity_consensus = {str(consensus).lower()}\n'
                                               f'cast_reconciliation_rounds = {rounds}\n')
                        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                            status, stdout, stderr = self.run_cli("plan", source_path, "--config", self.config)
                        self.assertEqual(status, 0, stderr)
                        provider.assert_not_called()
                        result = json.loads(stdout)
                        batches = result["analysis_batches"]
                        self.assertGreater(batches, 1)
                        base = batches + 1 if whole_book else 2 * batches
                        scans = (1 if whole_book else batches) if consensus else 0
                        self.assertEqual(result["cast_reconciliation_rounds"], rounds)
                        self.assertEqual(result["baseline_model_requests"], base + rounds + scans + 1)

    def test_analyze_passes_file_configured_reconciliation_rounds(self):
        self.config.write_text('[provider]\nmodel = "test-model"\n'
                               'base_url = "http://127.0.0.1:8000/v1"\n'
                               '[analysis]\ncast_reconciliation_rounds = 2\n'
                               'cast_reconciliation_notes = ["", "Check source context."]\n')
        NarrativeClient.calls = 0
        output = self.root / "cumulative.json"
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient):
            status, _, stderr = self.run_cli("analyze", self.make_epub(), "--config", self.config, "-o", output)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(NarrativeClient.calls, 7)
        self.assertEqual(read_json(output)["provenance"]["cast_reconciliation_rounds"], 2)
        self.assertEqual(read_json(output)["provenance"]["cast_reconciliation_notes"], ["", "Check source context."])

    def test_plan_budgets_series_scan_only_for_prior_books_and_identity_audit(self):
        registry_path = self.root / "series.json"
        empty = load_registry(registry_path)
        linked = link_analysis(valid_document(), empty)
        save_registry(registry_path, empty, linked)
        registry_before = registry_path.read_bytes()
        new_registry_path = self.root / "empty-series.json"
        source = self.make_epub()
        for whole_book in (False, True):
            for consensus in (False, True):
                for prior_books in (False, True):
                    with self.subTest(whole_book=whole_book, consensus=consensus, prior_books=prior_books):
                        self.config.write_text('[analysis]\n'
                                               f'whole_book_discovery = {str(whole_book).lower()}\n'
                                               f'identity_consensus = {str(consensus).lower()}\n')
                        selected = registry_path if prior_books else new_registry_path
                        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                            status, stdout, stderr = self.run_cli("plan", source, "--config", self.config,
                                                                 "--series", selected)
                        self.assertEqual(status, 0, stderr)
                        provider.assert_not_called()
                        result = json.loads(stdout)
                        expected = int(prior_books and consensus)
                        self.assertEqual(result["series_identity_scan_requests_minimum"], expected)
                        self.assertEqual(result["series"]["books"], int(prior_books))
                        base = result["analysis_batches"] + 2 if whole_book else 2 * result["analysis_batches"] + 1
                        self.assertEqual(result["baseline_model_requests"], base + result["identity_scan_requests"] + expected + 1)
                        if expected:
                            self.assertIn("nonempty discovered cast", result["note"])
                            self.assertIn("empty cast needs no scan", result["note"])
                            self.assertIn("one additional continuity round", result["note"])
                            self.assertIn("two rounds maximum", result["note"])
                            self.assertIn("three fresh jury requests for every match", result["note"])
                            self.assertIn("previously approved matches", result["note"])
                            if whole_book:
                                self.assertIn("one complete-source continuity scan", result["note"])
                            else:
                                self.assertIn("additional character/context scan batches", result["note"])
                        else:
                            self.assertNotIn("additional continuity round", result["note"])
        self.assertEqual(registry_path.read_bytes(), registry_before)
        self.assertFalse(new_registry_path.exists())

    def test_attribution_review_configuration_reaches_analyze_and_plan(self):
        self.config.write_text('[provider]\nmodel = "test-model"\n'
                               'base_url = "http://127.0.0.1:8000/v1"\n'
                               '[analysis]\nattribution_review = true\n'
                               'attribution_review_chunk_chars = 512\nattribution_workers = 8\n'
                               'cast_review_notes = "Fallback clue"\n'
                               'attribution_review_notes = "Specific clue"\n')
        source = self.make_epub()
        status, stdout, stderr = self.run_cli("plan", source, "--config", self.config)
        self.assertEqual(status, 0, stderr)
        plan = json.loads(stdout)
        self.assertTrue(plan["attribution_review"])
        self.assertEqual(plan["attribution_review_scan_requests"], 1)
        self.assertEqual(plan["baseline_model_requests"], 6)
        self.assertIn("three jury requests per review batch", plan["note"])
        observed = {}

        def review(document, requests, **kwargs):
            observed.update(kwargs)
            observed["requests"] = requests
            self.assertIn("bracket_delivery", document["provenance"])
            return document

        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient), \
                patch("epub_tts_narrator.attribution_review.audit_attribution", side_effect=review):
            status, _, stderr = self.run_cli("analyze", source, "--config", self.config,
                                           "-o", self.root / "review-enabled.json")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(observed["workers"], 5)
        self.assertEqual(observed["review_chunk_chars"], 512)
        self.assertEqual(observed["review_notes"], "Specific clue")
        self.assertEqual(observed["requests"].repair_attempts, 1)

    def test_standalone_attribution_review_is_cached_and_does_not_write_series(self):
        registry = self.root / "series.json"
        registry.write_text("untouched registry")
        self.config.write_text('[provider]\nmodel = "file-model"\n'
                               'base_url = "http://127.0.0.1:9000/v1"\napi_key = "file-only-secret"\n'
                               '[analysis]\nseries_registry = "series.json"\n'
                               'cache_dir = "review-cache"\nrepair_attempts = 2\n'
                               'attribution_review = false\ncast_review_notes = "Fallback clue"\n')
        original = self.analysis.read_bytes()
        output = self.root / "reviewed-attribution.json"
        NarrativeClient.calls = 0
        with patch.dict("os.environ", {"EPUB_ANALYZER_API_KEY": "wrong-secret"}), \
                patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=NarrativeClient) as provider, \
                patch("epub_tts_narrator.cli.load_registry") as load_series, \
                patch("epub_tts_narrator.cli.save_registry") as save_series:
            status, stdout, stderr = self.run_cli("review-attribution", self.analysis, "--config", self.config, "-o", output)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(provider.call_args.kwargs["api_key"], "file-only-secret")
            calls = NarrativeClient.calls
            self.assertGreater(calls, 0)
            status, _, stderr = self.run_cli("review-attribution", self.analysis, "--config", self.config,
                                           "-o", self.root / "second-review.json")
            self.assertEqual(status, 0, stderr)
            self.assertEqual(NarrativeClient.calls, calls)
            status, _, stderr = self.run_cli("review-attribution", self.analysis, "--config", self.config,
                                           "-o", output, "--force")
            self.assertEqual(status, 0, stderr)
            self.assertEqual(NarrativeClient.calls, calls)
            load_series.assert_not_called()
            save_series.assert_not_called()
        self.assertTrue((self.root / "review-cache").is_dir())
        self.assertEqual(registry.read_text(), "untouched registry")
        self.assertEqual(self.analysis.read_bytes(), original)
        result = read_json(output)
        self.assertEqual(result["source"], read_json(self.analysis)["source"])
        self.assertEqual(result["cast"], read_json(self.analysis)["cast"])
        self.assertEqual(result["provenance"]["attribution_review"]["review_notes_fingerprint"], fingerprint("Fallback clue"))
        for content in (stdout, stderr, output.read_text()):
            self.assertNotIn("file-only-secret", content)
            self.assertNotIn("wrong-secret", content)

    def test_standalone_attribution_review_guards_outputs_before_model_calls(self):
        registry = self.root / "series.json"
        registry.write_text("untouched")
        with self.config.open("a") as stream:
            stream.write('[analysis]\nseries_registry = "series.json"\n')
        bad = self.root / "bad.json"
        bad.write_text("{}")
        cases = [(self.analysis, self.analysis, []), (self.analysis, self.config, ["--force"]),
                 (self.analysis, registry, ["--force"]), (bad, self.root / "new.json", [])]
        for source, output, extras in cases:
            with self.subTest(source=source, output=output), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
                status, _, stderr = self.run_cli("review-attribution", source, "--config", self.config,
                                               "-o", output, *extras)
                self.assertEqual(status, 1, stderr)
                provider.assert_not_called()
        self.assertEqual(registry.read_text(), "untouched")

    def test_standalone_attribution_review_failure_does_not_replace_output(self):
        output = self.root / "reviewed.json"
        output.write_text("preserved old output")
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient", NarrativeClient), \
                patch("epub_tts_narrator.cli.audit_attribution", side_effect=ProviderError("HTTP 429")):
            status, _, stderr = self.run_cli("review-attribution", self.analysis, "--config", self.config,
                                           "-o", output, "--force")
        self.assertEqual(status, 1)
        self.assertIn("HTTP 429", stderr)
        self.assertEqual(output.read_text(), "preserved old output")

    def test_export_keeps_advisory_notes_without_requiring_an_approval_flag(self):
        doc = valid_document()
        doc["segments"][0]["attribution"]["review_required"] = True
        for allow_review in (False, True):
            with self.subTest(allow_review=allow_review):
                output = self.root / f"bundle-{allow_review}"
                export_bundle(doc, output, allow_review=allow_review)
                self.assertTrue(read_json(output / "manifest.json")["readiness"]["ready"])
                self.assertEqual(len(read_json(output / "review.json")), 1)

    def test_cli_export_returns_success_for_advisory_notes_and_reports_no_review_gate(self):
        doc = valid_document()
        doc["segments"][0]["attribution"]["review_required"] = True
        write_json(self.analysis, doc, overwrite=True)
        status, output, diagnostics = self.run_cli("export", self.analysis, "-o", self.root / "advisory-bundle")
        self.assertEqual(status, 0, diagnostics)
        self.assertIn("1 advisory notes", output)
        self.assertIn("No review step is required", output)
        self.assertNotIn("draft", output)

    def test_cli_export_preserves_unassigned_passages_and_returns_incomplete_status(self):
        doc = valid_document()
        doc["segments"][1].update(role_id=None, voice_id=None)
        doc["segments"][1]["attribution"].update(basis="unresolved", review_required=True)
        write_json(self.analysis, doc, overwrite=True)
        bundle = self.root / "incomplete-bundle"
        status, output, diagnostics = self.run_cli("export", self.analysis, "-o", bundle)
        self.assertEqual(status, 2, diagnostics)
        self.assertTrue((bundle / "manifest.json").exists())
        self.assertFalse(read_json(bundle / "manifest.json")["readiness"]["ready"])
        self.assertIn("1 spoken passage(s) still need a voice", diagnostics)
        self.assertIn("Exiting with status 2", diagnostics)
        self.assertNotIn("No review step is required", output)

    def test_export_chapters_preserves_global_order_and_empty_chapters(self):
        doc = valid_document()
        doc["source"]["chapters"] = [
            {"id": "empty-first", "title": "Front Matter", "source_href": "front.xhtml", "block_ids": []},
            {"id": "../unsafe/chapter", "title": "../../outside.jsonl", "source_href": "one.xhtml", "block_ids": ["b1"]},
            {"id": "empty-middle", "title": "Interlude", "source_href": "middle.xhtml", "block_ids": []},
            {"id": "/absolute/chapter", "title": "A quoted “chapter” 🛰️", "source_href": "two.xhtml", "block_ids": ["b2"]},
            {"id": "empty-last", "title": "End", "source_href": "end.xhtml", "block_ids": []},
        ]
        doc["source"]["blocks"][0]["chapter_id"] = "../unsafe/chapter"
        doc["source"]["blocks"][1]["chapter_id"] = "/absolute/chapter"
        doc["source"]["blocks"][1]["break_before"] = "blank_paragraph"
        text = '“Ça va?” 🛰️ e\u0301 \\ "yes".'
        doc["source"]["blocks"][1]["text"] = text
        doc["segments"][2].update(text=text, end=len(text))
        original = deepcopy(doc)
        target = self.root / "chapters"
        export_bundle(doc, target)
        self.assertEqual(doc, original)
        self.assertFalse((target / "script.jsonl").exists())
        manifest = read_json(target / "manifest.json")
        entries = manifest["files"]["scripts"]
        self.assertEqual([entry["chapter_id"] for entry in entries],
                         [chapter["id"] for chapter in doc["source"]["chapters"]])
        self.assertEqual([entry["path"] for entry in entries],
                         [f"scripts/chapter-{index:04d}.jsonl" for index in range(1, 6)])
        self.assertEqual([(entry["segment_count"], entry["sequence_start"], entry["sequence_end"])
                          for entry in entries], [(0, 0, 0), (2, 0, 2), (0, 2, 2), (1, 2, 3), (0, 3, 3)])
        all_rows = []
        for entry in entries:
            payload = (target / entry["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["sha256"][entry["path"]])
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            self.assertEqual(len(rows), entry["segment_count"])
            self.assertEqual([row["sequence"] for row in rows],
                             list(range(entry["sequence_start"], entry["sequence_end"])))
            self.assertTrue(all(row["chapter_id"] == entry["chapter_id"] for row in rows))
            all_rows.extend(rows)
        self.assertEqual([row["id"] for row in all_rows], [row["id"] for row in doc["segments"]])
        self.assertEqual([row["sequence"] for row in all_rows], [0, 1, 2])
        self.assertEqual([row["break_after"] for row in all_rows], ["none", "paragraph", "paragraph"])
        self.assertNotIn("break_before", all_rows[0])
        self.assertEqual(all_rows[2]["break_before"], "blank_paragraph")
        for before, row in zip(doc["segments"], all_rows):
            self.assertEqual(before, {key: row[key] for key in before})
        self.assertEqual(read_json(target / "source.json"), doc["source"])
        self.assertEqual(read_json(target / "cast.json"), doc["cast"])
        self.assertEqual(manifest["provenance"], doc["provenance"])
        self.assertEqual(set(manifest["sha256"]),
                         {"source.json", "cast.json", "review.json", *(entry["path"] for entry in entries)})
        self.assertEqual({str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()},
                         {"manifest.json", *manifest["sha256"]})

    def test_export_link_failure_never_publishes_manifest(self):
        target = self.root / "incomplete"
        import os
        original_link = os.link

        def fail_chapter_link(source, destination):
            if Path(source).parent.name == "scripts":
                raise OSError("simulated chapter publication failure")
            return original_link(source, destination)

        with patch("epub_tts_narrator.cli.os.link", side_effect=fail_chapter_link):
            with self.assertRaisesRegex(OSError, "simulated"):
                export_bundle(valid_document(), target)
        self.assertTrue(target.is_dir())
        self.assertFalse((target / "manifest.json").exists())
        self.assertEqual(list(self.root.glob(".audiobook-export-*")), [])

    def test_export_refuses_existing_directory_without_mutation(self):
        target = self.root / "existing"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            export_bundle(valid_document(), target)
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(len(list(target.iterdir())), 1)

    def test_export_does_not_replace_concurrently_created_directory(self):
        target = self.root / "concurrent"
        original_writer = write_json
        inode = []

        def writer(path, value, **kwargs):
            original_writer(path, value, **kwargs)
            if Path(path).name == "manifest.json":
                target.mkdir()
                inode.append(target.stat().st_ino)

        with patch("epub_tts_narrator.cli.write_json", writer):
            with self.assertRaises(FileExistsError):
                export_bundle(valid_document(), target)
        self.assertEqual(target.stat().st_ino, inode[0])
        self.assertEqual(list(target.iterdir()), [])

    def test_validation_and_review_status(self):
        doc = valid_document()
        doc["segments"][0]["attribution"]["review_required"] = True
        write_json(self.analysis, doc, overwrite=True)
        self.assertEqual(self.run_cli("validate", self.analysis)[0], 0)
        self.assertEqual(self.run_cli("validate", self.analysis, "--ready")[0], 0)
        status, output, _ = self.run_cli("review", self.analysis)
        review = json.loads(output)
        self.assertEqual(status, 0)
        self.assertEqual(review["cast_fingerprint"], fingerprint(doc["cast"]))
        self.assertEqual(review["source_fingerprint"], fingerprint(doc["source"]))
        self.assertEqual(len(review["items"]), 1)

    def test_validate_ready_fails_for_missing_speaker_even_when_review_is_advisory(self):
        doc = valid_document()
        doc["segments"][1].update(role_id=None, voice_id=None)
        doc["segments"][1]["attribution"].update(basis="unresolved", review_required=True)
        write_json(self.analysis, doc, overwrite=True)
        self.assertEqual(self.run_cli("validate", self.analysis)[0], 0)
        self.assertEqual(self.run_cli("validate", self.analysis, "--ready")[0], 2)

    def test_existing_output_prevents_any_model_calls(self):
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider:
            status, _, _ = self.run_cli("analyze", self.root / "missing.epub", "-o", self.analysis, "--config", self.config)
        self.assertEqual(status, 1)
        provider.assert_not_called()

    def test_provider_settings_come_from_file_and_keys_do_not_enter_outputs(self):
        self.config.write_text('[provider]\nmodel = "file-model"\nbase_url = "http://127.0.0.1:9000/v1"\napi_key = "file-only-secret"\njson_mode = false\n')
        output = self.root / "configured.json"
        with patch.dict("os.environ", {"EPUB_ANALYZER_MODEL": "wrong-model", "EPUB_ANALYZER_API_KEY": "wrong-secret"}):
            with patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=NarrativeClient) as factory:
                status, stdout, stderr = self.run_cli("analyze", self.make_epub(), "--config", self.config, "-o", output)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(factory.call_args.kwargs["model"], "file-model")
        self.assertEqual(factory.call_args.kwargs["api_key"], "file-only-secret")
        self.assertFalse(factory.call_args.kwargs["json_mode"])
        for content in (stdout, stderr, output.read_text()):
            self.assertNotIn("file-only-secret", content)
            self.assertNotIn("wrong-secret", content)

    def test_apply_human_correction_keeps_original_unchanged(self):
        original = self.analysis.read_bytes()
        doc = read_json(self.analysis)
        edits = self.root / "edits.json"
        write_json(edits, {"source_fingerprint": fingerprint(doc["source"]),
                           "cast_fingerprint": fingerprint(doc["cast"]),
                           "assignments": [{"segment_id": "s2", "role_id": "mara"}]})
        reviewed = self.root / "reviewed.json"
        status, _, stderr = self.run_cli("apply", self.analysis, "--edits", edits, "-o", reviewed)
        self.assertEqual(status, 0, stderr)
        self.assertEqual(read_json(reviewed)["segments"][1]["attribution"]["basis"], "user")
        self.assertEqual(self.analysis.read_bytes(), original)

    def test_malformed_files_fail_without_tracebacks(self):
        bad = self.root / "bad.json"
        bad.write_text('{"format": 1}')
        status, _, stderr = self.run_cli("export", bad, "-o", self.root / "bad-bundle")
        self.assertEqual(status, 1)
        self.assertIn("Error:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_null_correction_members_fail_without_tracebacks(self):
        doc = read_json(self.analysis)
        for correction in ({"assignments": [None]}, {"blocks": [None]},
                           {"blocks": [{"block_id": "b1", "segments": [None]}]}):
            edits = self.root / "bad-edits.json"
            write_json(edits, dict(correction, source_fingerprint=fingerprint(doc["source"]),
                                   cast_fingerprint=fingerprint(doc["cast"])), overwrite=True)
            status, _, stderr = self.run_cli("apply", self.analysis, "--edits", edits, "-o", self.root / "new.json")
            self.assertEqual(status, 1)
            self.assertIn("Error:", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_atomic_json_no_clobber(self):
        before = self.analysis.read_bytes()
        with self.assertRaises(FileExistsError):
            write_json(self.analysis, {"changed": True})
        self.assertEqual(before, self.analysis.read_bytes())
        self.assertFalse(list(self.root.glob(".writing-*")))


if __name__ == "__main__":
    unittest.main()
