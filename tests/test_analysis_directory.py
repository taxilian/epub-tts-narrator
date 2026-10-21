from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (
    continuation_registry, default_output, fresh_registry, publish_directory, validate_output,
)
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.series import link_analysis, load_registry
from test_series import fixture, propose


class AnalysisDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def first(self, name="first"):
        output = self.root / name
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        publish_directory(document, registry, output, export_bundle)
        return output, document

    def read(self, path):
        return json.loads(path.read_text())

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def rehash_metadata(self, directory, name):
        manifest = self.read(directory / "manifest.json")
        manifest["analysis_directory"]["sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        write_json(directory / "manifest.json", manifest, overwrite=True)

    def test_default_output_uses_working_directory_and_input_stem(self):
        with patch("epub_tts_narrator.analysis_directory.Path.cwd", return_value=self.root):
            self.assertEqual(default_output("/books/My Book.epub"), self.root / "My Book.audiobook")

    def test_validate_output_is_read_only_and_rejects_files_directories_and_dangling_links(self):
        missing = self.root / "nested" / "new"
        self.assertEqual(validate_output(missing), missing)
        self.assertFalse(missing.parent.exists())
        existing_file = self.root / "file"
        existing_file.write_text("preserve")
        existing_directory = self.root / "directory"
        existing_directory.mkdir()
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        for path in (existing_file, existing_directory, dangling):
            with self.subTest(path=path), self.assertRaises(FileExistsError):
                validate_output(path)
        with self.assertRaisesRegex(ValueError, "parent is not a directory"):
            validate_output(existing_file / "child")
        self.assertEqual(existing_file.read_text(), "preserve")

    def test_fresh_registry_identity_is_stable_and_does_not_write(self):
        output = self.root / "first"
        first = fresh_registry(output)
        self.assertEqual(first, fresh_registry(output))
        self.assertEqual(first, load_registry(output / "series-cast.json"))
        self.assertFalse(output.exists())

    def test_three_book_chain_preserves_voice_ids_and_all_previous_outputs(self):
        previous, first = self.first()
        original_character = first["cast"]["characters"][0]
        prior_outputs = [previous]
        snapshots = [self.snapshot(previous)]
        for number in (2, 3):
            current = fixture(sequel=True)
            current["source"]["book"]["id"] = f"book{number}"
            current["source"]["book"]["title"] = f"Return {number}"
            registry = continuation_registry(previous, current["source"])
            self.assertEqual(len(registry["books"]), number - 1)
            propose(current, original_character["id"])
            linked = link_analysis(current, registry)
            self.assertEqual(linked["cast"]["characters"][0]["voice_id"], original_character["voice_id"])
            output = self.root / f"book-{number}"
            publish_directory(linked, registry, output, export_bundle, continue_from=previous)
            self.assertEqual(len(self.read(output / "series-cast.json")["books"]), number)
            for prior, snapshot in zip(prior_outputs, snapshots):
                self.assertEqual(self.snapshot(prior), snapshot)
            prior_outputs.append(output)
            snapshots.append(self.snapshot(output))
            previous = output

    def test_publication_is_renderer_compatible_and_preserves_input_objects(self):
        from epub_tts_narrator.renderer.bundle import load_bundle

        output = self.root / "result"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        before = deepcopy((document, registry))
        state = publish_directory(document, registry, output, export_bundle)
        self.assertEqual((document, registry), before)
        loaded = load_bundle(output)
        self.assertEqual(loaded["cast"], document["cast"])
        self.assertEqual(loaded["manifest"]["readiness"], state)
        self.assertEqual(self.read(output / "analysis.json"), document)
        self.assertFalse(list(self.root.glob(".audiobook-analysis-*")))

    def test_assigned_analysis_is_ready_with_advisory_notes_preserved(self):
        output = self.root / "draft"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        document["segments"][0]["attribution"]["review_required"] = True
        state = publish_directory(document, registry, output, export_bundle)
        self.assertTrue(state["ready"])
        self.assertTrue(self.read(output / "manifest.json")["readiness"]["ready"])
        self.assertTrue(self.read(output / "review.json"))
        self.assertEqual(len(continuation_registry(output, fixture(sequel=True)["source"])["books"]), 1)

    def test_legacy_advisory_readiness_continues_without_rewriting_prior_book(self):
        output = self.root / "legacy"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        document["segments"][0]["attribution"]["review_required"] = True
        publish_directory(document, registry, output, export_bundle)
        legacy = {"ready": False, "errors": [], "review_count": 1}
        for name in ("analysis.json", "series-cast.json", "manifest.json"):
            record = self.read(output / name)
            if name == "series-cast.json":
                record["books"][-1]["readiness"] = legacy
            else:
                record["readiness"] = legacy
            write_json(output / name, record, overwrite=True)
            if name != "manifest.json":
                self.rehash_metadata(output, name)
        before = self.snapshot(output)
        continued = continuation_registry(output, fixture(sequel=True)["source"])
        self.assertEqual(continued["books"][-1]["readiness"], legacy)
        self.assertEqual(self.snapshot(output), before)

    def test_readiness_compatibility_does_not_accept_wrong_review_counts(self):
        output, _ = self.first()
        manifest = self.read(output / "manifest.json")
        manifest["readiness"] = {"ready": False, "errors": [], "review_count": 99}
        write_json(output / "manifest.json", manifest, overwrite=True)
        with self.assertRaisesRegex(ValueError, "bundle and analysis disagree"):
            continuation_registry(output, fixture(sequel=True)["source"])

    def test_continuation_requires_complete_metadata_without_creating_missing_registry(self):
        output, _ = self.first()
        for filename in ("manifest.json", "analysis.json", "series-cast.json", "scripts/chapter-0001.jsonl"):
            path = output / filename
            contents = path.read_bytes()
            path.unlink()
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "incomplete"):
                continuation_registry(output, fixture(sequel=True)["source"])
            self.assertFalse(path.exists())
            path.write_bytes(contents)

    def test_continuation_rejects_same_book_and_returns_independent_registry(self):
        output, document = self.first()
        with self.assertRaisesRegex(ValueError, "already in the continuation history"):
            continuation_registry(output, document["source"])
        registry = continuation_registry(output, fixture(sequel=True)["source"])
        registry["books"].clear()
        self.assertEqual(len(self.read(output / "series-cast.json")["books"]), 1)

    def test_continuation_rejects_stale_cast_registry_and_wrong_series(self):
        output, _ = self.first()
        registry_path = output / "series-cast.json"
        original = self.read(registry_path)
        mutations = [
            lambda registry: registry["books"][-1]["cast"]["voices"][0].update(description="Different voice"),
            lambda registry: registry.update(series_id="0" * 32),
            lambda registry: registry["books"][-1].update(source_fingerprint="different"),
        ]
        for mutation in mutations:
            changed = deepcopy(original)
            mutation(changed)
            write_json(registry_path, changed, overwrite=True)
            self.rehash_metadata(output, "series-cast.json")
            with self.assertRaises(ValueError):
                continuation_registry(output, fixture(sequel=True)["source"])
        write_json(registry_path, original, overwrite=True)

    def test_entire_prior_history_is_bound_by_manifest_checksum(self):
        first_output, first_document = self.first()
        second = fixture(sequel=True)
        registry = continuation_registry(first_output, second["source"])
        propose(second, first_document["cast"]["characters"][0]["id"])
        second = link_analysis(second, registry)
        output = self.root / "second"
        publish_directory(second, registry, output, export_bundle)
        prior = self.read(output / "series-cast.json")
        latest = deepcopy(prior["books"][-1])
        prior["books"][0]["cast"]["voices"][0]["description"] = "Untracked change to old history"
        self.assertEqual(prior["books"][-1], latest)
        write_json(output / "series-cast.json", prior, overwrite=True)
        third_source = deepcopy(second["source"])
        third_source["book"]["id"] = "book3"
        with self.assertRaisesRegex(ValueError, "checksum mismatch: series-cast.json"):
            continuation_registry(output, third_source)

    def test_malformed_registry_metadata_reports_value_error(self):
        output, _ = self.first()
        path = output / "series-cast.json"
        registry = self.read(path)
        registry["books"][0]["cast"]["characters"][0]["voice_id"] = []
        write_json(path, registry, overwrite=True)
        self.rehash_metadata(output, "series-cast.json")
        with self.assertRaisesRegex(ValueError, "Invalid analysis directory metadata"):
            continuation_registry(output, fixture(sequel=True)["source"])

    def test_continuation_requires_metadata_checksums_even_when_bundle_is_valid(self):
        output, _ = self.first()
        path = output / "manifest.json"
        manifest = self.read(path)
        for invalid in (None, {}, {"version": 1, "sha256": []}):
            manifest["analysis_directory"] = invalid
            write_json(path, manifest, overwrite=True)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "continuation metadata"):
                continuation_registry(output, fixture(sequel=True)["source"])

    def test_continuation_rejects_scripts_that_disagree_even_with_updated_checksum(self):
        output, _ = self.first()
        chapter_path = output / "scripts/chapter-0001.jsonl"
        rows = [json.loads(line) for line in chapter_path.read_text().splitlines()]
        rows[0]["text"] = "Changed words"
        chapter_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = self.read(output / "manifest.json")
        manifest["sha256"]["scripts/chapter-0001.jsonl"] = hashlib.sha256(chapter_path.read_bytes()).hexdigest()
        write_json(output / "manifest.json", manifest, overwrite=True)
        with self.assertRaisesRegex(ValueError, "scripts and analysis disagree"):
            continuation_registry(output, fixture(sequel=True)["source"])

    def test_continuation_rejects_symlinked_metadata_and_chapter_directories(self):
        output, _ = self.first()
        metadata = output / "series-cast.json"
        saved = self.root / "saved-registry.json"
        metadata.rename(saved)
        metadata.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            continuation_registry(output, fixture(sequel=True)["source"])
        metadata.unlink()
        saved.rename(metadata)
        scripts = output / "scripts"
        saved_scripts = self.root / "saved-scripts"
        scripts.rename(saved_scripts)
        scripts.symlink_to(saved_scripts, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            continuation_registry(output, fixture(sequel=True)["source"])

    def test_manifest_is_published_after_all_artifacts(self):
        output = self.root / "ordered"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        original_link = os.link
        published = []

        def observe(source, destination):
            destination = Path(destination)
            if destination.is_relative_to(output):
                published.append(str(destination.relative_to(output)))
                if destination.name == "manifest.json":
                    self.assertTrue((output / "analysis.json").is_file())
                    self.assertTrue((output / "series-cast.json").is_file())
                    self.assertTrue((output / "scripts/chapter-0001.jsonl").is_file())
            return original_link(source, destination)

        with patch("epub_tts_narrator.analysis_directory.os.link", side_effect=observe):
            publish_directory(document, registry, output, export_bundle)
        self.assertEqual(published[-1], "manifest.json")

    def test_publication_error_and_interrupt_roll_back_owned_output_then_retry(self):
        for failure in (OSError("disk failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                output = self.root / type(failure).__name__
                registry = fresh_registry(output)
                document = link_analysis(fixture(), registry)
                original_link = os.link

                def fail(source, destination):
                    if Path(destination) == output / "scripts/chapter-0001.jsonl":
                        raise failure
                    return original_link(source, destination)

                with patch("epub_tts_narrator.analysis_directory.os.link", side_effect=fail):
                    with self.assertRaises(type(failure)):
                        publish_directory(document, registry, output, export_bundle)
                self.assertFalse(output.exists())
                self.assertFalse(list(self.root.glob(".audiobook-analysis-*")))
                self.assertEqual(fresh_registry(output), registry)
                publish_directory(document, registry, output, export_bundle)
                self.assertTrue((output / "manifest.json").is_file())

    def test_concurrent_existing_output_is_never_replaced_or_cleaned(self):
        output = self.root / "race"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)

        def race_exporter(*args, **kwargs):
            state = export_bundle(*args, **kwargs)
            output.mkdir()
            (output / "keep").write_text("another process")
            return state

        with self.assertRaises(FileExistsError):
            publish_directory(document, registry, output, race_exporter)
        self.assertEqual({path.name for path in output.iterdir()}, {"keep"})
        self.assertEqual((output / "keep").read_text(), "another process")

    def test_rollback_preserves_concurrently_replaced_file(self):
        output = self.root / "replacement"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        original_link = os.link

        def fail_after_replacement(source, destination):
            if Path(destination) == output / "cast.json":
                target = output / "analysis.json"
                target.unlink()
                target.write_text("user replacement")
                raise OSError("interrupted publication")
            return original_link(source, destination)

        with patch("epub_tts_narrator.analysis_directory.os.link", side_effect=fail_after_replacement):
            with self.assertRaises(OSError):
                publish_directory(document, registry, output, export_bundle)
        self.assertEqual({path.name for path in output.iterdir()}, {"analysis.json"})
        self.assertEqual((output / "analysis.json").read_text(), "user replacement")

    def test_failed_staging_never_reserves_final_output(self):
        output = self.root / "failed"
        registry = fresh_registry(output)
        document = link_analysis(fixture(), registry)
        with patch("epub_tts_narrator.analysis_directory.save_registry", side_effect=ValueError("bad snapshot")):
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                publish_directory(document, registry, output, export_bundle)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".audiobook-analysis-*")))


if __name__ == "__main__":
    unittest.main()
