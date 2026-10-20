"""Run-artifact checks: exact source handoff, dynamic ZIP safety and rollback."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from epub_audiobook.cli import export_bundle
from epub_audiobook.contracts import readiness, review_items
from epub_audiobook.pipeline import fingerprint, write_json
from test_contracts import valid_document

HELPERS = Path(__file__).resolve().parents[1] / "scripts" / "legacy_series"
sys.path.insert(0, str(HELPERS))
import bundle_validation as validation
import migrate_chapter_bundles as migration
import package_results as packaging


def legacy_bundle(document, folder):
    folder.mkdir()
    for name, value in (("source", document["source"]), ("cast", document["cast"]),
                        ("review", review_items(document))):
        write_json(folder / (name + ".json"), value)
    script = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                     for row in validation.rendered_rows(document))
    (folder / "script.jsonl").write_text(script, encoding="utf-8")
    names = (*validation.FIXED_FILES, "script.jsonl")
    write_json(folder / "manifest.json", {
        "format": "epub-audiobook-bundle", "version": 1,
        "book": document["source"]["book"], "chapters": document["source"]["chapters"],
        "source_fingerprint": fingerprint(document["source"]), "readiness": readiness(document),
        "offsets": "Zero-based Unicode code points within source block text; end exclusive.",
        "files": {name.split(".")[0]: name for name in names},
        "sha256": {name: validation.sha((folder / name).read_bytes()) for name in names},
        "provenance": document["provenance"],
    })


def several_chapters():
    document = valid_document()
    first = document["source"]["chapters"][0]
    first["block_ids"] = ["b1"]
    document["source"]["chapters"].extend([
        {"id": "empty", "title": "Empty", "source_href": "empty.xhtml", "block_ids": []},
        {"id": "ch2", "title": "Two", "source_href": "text/ch2.xhtml", "block_ids": ["b2"]},
    ])
    document["source"]["blocks"][1]["chapter_id"] = "ch2"
    document["source"]["blocks"][1]["source"]["href"] = "text/ch2.xhtml"
    document["source"]["blocks"][1]["break_before"] = "scene"
    return document


class SeriesBundleToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.document = several_chapters()

    def exported(self):
        folder = self.root / "bundle"
        export_bundle(self.document, folder)
        return folder, validation.load_bundle(folder)

    def fixture_run(self):
        for number in range(1, 8):
            folder = self.root / f"book-{number:02d}"
            folder.mkdir()
            document = deepcopy(self.document)
            document["source"]["book"]["id"] = f"book{number}"
            write_json(folder / "analysis.json", document)
            write_json(folder / "source.json", {"source": document["source"]})
            write_json(folder / "registry-before.json", {})
            write_json(folder / "final-source-audit.json", {"analysis_fingerprint": fingerprint(document)})
            legacy_bundle(document, folder / "bundle")
        write_json(self.root / "series-cast.json", {})
        write_json(self.root / "inventory.json", {})
        (self.root / "soulship-tts-handoff.zip").write_bytes(b"historical archive")

    def test_exact_v1_and_v2_validation_including_empty_chapter(self):
        old = self.root / "old"
        legacy_bundle(self.document, old)
        self.assertEqual(validation.validate_native_bundle(validation.load_bundle(old), "", self.document)["version"], 1)
        _, entries = self.exported()
        checked = validation.validate_native_bundle(entries, "", self.document)
        self.assertEqual((checked["version"], checked["script_files"], checked["segments"]), (2, 3, 3))
        manifest = json.loads(entries["manifest.json"])
        empty = manifest["files"]["scripts"][1]
        self.assertEqual((empty["sequence_start"], empty["sequence_end"], empty["segment_count"]), (2, 2, 0))

    def test_recomputed_hash_cannot_hide_row_chapter_performance_or_sequence_tampering(self):
        _, original = self.exported()
        filename = "scripts/chapter-0001.jsonl"
        for field, value in (("id", "different"), ("sequence", 9), ("chapter_id", "ch2"),
                             ("break_after", "paragraph"), ("voice_id", "vn"),
                             ("delivery", {"direction": "shout", "evidence": []})):
            with self.subTest(field=field):
                entries = dict(original)
                rows = [json.loads(line) for line in entries[filename].decode().splitlines()]
                row_index = 1 if field == "voice_id" else 0
                rows[row_index][field] = value
                entries[filename] = "".join(json.dumps(row) + "\n" for row in rows).encode()
                manifest = json.loads(entries["manifest.json"])
                manifest["sha256"][filename] = validation.sha(entries[filename])
                entries["manifest.json"] = json.dumps(manifest).encode()
                with self.assertRaises(ValueError):
                    validation.validate_native_bundle(entries, "", self.document)

    def test_manifest_inventory_order_bounds_paths_and_native_metadata_are_checked(self):
        _, entries = self.exported()
        original = json.loads(entries["manifest.json"])
        mutations = [
            lambda m: m["files"]["scripts"].reverse(),
            lambda m: m["files"]["scripts"][0].update(path="../settings.local.toml"),
            lambda m: m["files"]["scripts"][0].update(path="/tmp/script.jsonl"),
            lambda m: m["files"]["scripts"][0].update(sequence_start=True),
            lambda m: m["files"]["scripts"][1].update(sequence_end=3),
            lambda m: m["book"].update(title="Changed"),
            lambda m: m["provenance"].update(other="fabricated"),
            lambda m: m["sha256"].update({"settings.local.toml": "0" * 64}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                altered = deepcopy(original)
                mutate(altered)
                with self.assertRaises(ValueError):
                    validation.validate_native_bundle(dict(entries, **{"manifest.json": json.dumps(altered).encode()}),
                                                      "", self.document)

    def test_logical_chapters_share_native_id_without_losing_or_reassigning_rows(self):
        document = valid_document()
        source_chapter = document["source"]["chapters"][0]
        blocks, segments = [], []
        for number, text in enumerate(("CHAPTER FIVE", "The first scene opens.",
                                       "CHAPTER SIX", "The second scene opens."), 1):
            identity = f"b{number}"
            heading = number in (1, 3)
            blocks.append({"id": identity, "chapter_id": "ch1", "text": text,
                           "kind": "heading" if heading else "paragraph",
                           "source": {"href": "text/ch1.xhtml", "element_id": None,
                                      "tag": "h1" if heading else "p"}})
            row = deepcopy(document["segments"][0])
            row.update(id=f"s{number}", block_id=identity, start=0, end=len(text), text=text)
            segments.append(row)
        document["source"]["blocks"] = blocks
        source_chapter["block_ids"] = [block["id"] for block in blocks]
        document["segments"] = segments
        document["cast"]["characters"][0]["evidence"] = []
        document["cast"]["voices"][1]["traits"] = []
        folder = self.root / "logical"
        export_bundle(document, folder)
        entries = validation.load_bundle(folder)
        result = validation.validate_native_bundle(entries, "", document)
        self.assertEqual(result["script_files"], 2)
        scripts = json.loads(entries["manifest.json"])["files"]["scripts"]
        self.assertEqual([row["chapter_id"] for row in scripts], ["ch1", "ch1"])
        self.assertEqual([row["heading_block_id"] for row in scripts], ["b1", "b3"])
        self.assertEqual([(row["sequence_start"], row["sequence_end"]) for row in scripts], [(0, 2), (2, 4)])
        self.assertEqual(migration.script_rows(entries), validation.rendered_rows(document))

    def test_directory_rejects_stale_monolithic_script_and_symlink(self):
        folder, entries = self.exported()
        extra = folder / "script.jsonl"
        extra.write_text("stale")
        with self.assertRaises(ValueError):
            validation.load_bundle(folder)
        extra.unlink()
        chapter = folder / "scripts" / "chapter-0001.jsonl"
        outside = self.root / "external"
        outside.write_bytes(chapter.read_bytes())
        chapter.unlink()
        chapter.symlink_to(outside)
        with self.assertRaises(ValueError):
            validation.load_bundle(folder)

    def archive_entries(self):
        _, native = self.exported()
        entries = {f"book-{number:02d}/bundle/{name}": data
                   for number in range(1, 8) for name, data in native.items()}
        entries.update({name: b"fixture" for name in (*packaging.SHARED_FILES, "FORMAT.md", "README.txt")})
        return entries

    def test_dynamic_archive_inventory_round_trip_preserves_old_zip_and_exclusive_destination(self):
        entries = self.archive_entries()
        old = self.root / "soulship-tts-handoff.zip"
        old.write_bytes(b"old")
        result = packaging.write_archive(entries, {}, self.root)
        with zipfile.ZipFile(self.root / result["filename"]) as archive:
            self.assertEqual(archive.namelist(), list(packaging.archive_inventory(entries)))
            self.assertTrue(all(archive.read(name) == data for name, data in entries.items()))
        self.assertEqual(old.read_bytes(), b"old")
        with self.assertRaises(ValueError):
            packaging.write_archive(entries, {}, self.root)
        self.assertEqual(list(self.root.glob(".package-results-*")), [])

    def test_archive_rejects_extra_files_changed_input_or_recomputed_layout(self):
        entries = self.archive_entries()
        bound = self.root / "bound"
        bound.write_bytes(b"before")
        bindings = {bound: validation.sha(bound.read_bytes())}
        bound.write_bytes(b"after")
        with self.assertRaises(ValueError):
            packaging.write_archive(entries, bindings, self.root)
        with self.assertRaises(ValueError):
            packaging.write_archive(dict(entries, **{"settings.local.toml": b"secret"}), {}, self.root)
        self.assertEqual(list(self.root.glob("*.zip")), [])

    def test_migration_preserves_authoritative_inputs_and_original_bundles_and_zip(self):
        self.fixture_run()
        before = migration.snapshot(self.root)
        old = {n: validation.load_bundle(self.root / f"book-{n:02d}/bundle") for n in range(1, 8)}
        result = migration.migrate(self.root)
        self.assertEqual(migration.snapshot(self.root), before)
        self.assertEqual(result["provider_calls"], 0)
        for number in range(1, 8):
            folder = self.root / f"book-{number:02d}"
            self.assertEqual(validation.load_bundle(folder / migration.BACKUP), old[number])
            document = json.loads((folder / "analysis.json").read_text())
            self.assertEqual(validation.validate_native_bundle(validation.load_bundle(folder / "bundle"),
                                                              "", document)["version"], 2)
        with self.assertRaises(ValueError):
            migration.migrate(self.root)
        self.assertFalse((self.root / ".chapter-split-migration.lock").exists())

    def test_migration_rolls_back_mid_install_failure_without_losing_originals(self):
        self.fixture_run()
        old = {n: validation.load_bundle(self.root / f"book-{n:02d}/bundle") for n in range(1, 8)}
        original_rename = migration.os.rename

        def fail_book_four(source, destination):
            if Path(source).name == "book-04" and ".chapter-split-stage-" in str(source):
                raise OSError("injected installation failure")
            return original_rename(source, destination)

        with patch.object(migration.os, "rename", side_effect=fail_book_four):
            with self.assertRaises(OSError):
                migration.migrate(self.root)
        for number in range(1, 8):
            folder = self.root / f"book-{number:02d}"
            self.assertEqual(validation.load_bundle(folder / "bundle"), old[number])
            self.assertFalse((folder / migration.BACKUP).exists())
        self.assertFalse((self.root / migration.LEDGER).exists())
        self.assertFalse((self.root / ".chapter-split-migration.lock").exists())
        self.assertEqual(list(self.root.glob(".chapter-split-stage-*")), [])

    def test_migration_input_change_during_staging_aborts_before_any_move(self):
        self.fixture_run()
        exporter = migration.export_bundle
        calls = 0

        def mutate_after_export(document, path, **kwargs):
            nonlocal calls
            result = exporter(document, path, **kwargs)
            calls += 1
            if calls == 7:
                (self.root / "series-cast.json").write_text('{"changed": true}')
            return result

        with patch.object(migration, "export_bundle", side_effect=mutate_after_export):
            with self.assertRaises(ValueError):
                migration.migrate(self.root)
        self.assertFalse((self.root / migration.LEDGER).exists())
        for number in range(1, 8):
            folder = self.root / f"book-{number:02d}"
            self.assertFalse((folder / migration.BACKUP).exists())
            self.assertEqual(json.loads((folder / "bundle/manifest.json").read_text())["version"], 1)

    def test_migration_ledger_publication_failure_restores_all_original_bundles(self):
        self.fixture_run()
        original_link = migration.os.link

        def fail_ledger(source, destination):
            if Path(destination).name == migration.LEDGER:
                raise OSError("injected ledger publication failure")
            return original_link(source, destination)

        with patch.object(migration.os, "link", side_effect=fail_ledger):
            with self.assertRaises(OSError):
                migration.migrate(self.root)
        for number in range(1, 8):
            folder = self.root / f"book-{number:02d}"
            self.assertFalse((folder / migration.BACKUP).exists())
            self.assertEqual(json.loads((folder / "bundle/manifest.json").read_text())["version"], 1)
        self.assertFalse((self.root / migration.LEDGER).exists())


if __name__ == "__main__":
    unittest.main()
