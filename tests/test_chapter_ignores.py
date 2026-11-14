"""Book-local chapter ignores: sidecar API, locator resolution, plan emission."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator import chapter_ignores
from epub_tts_narrator.book_update import _rebind_sidecars
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.run_locks import acquire_directory_lock
from epub_tts_narrator.renderer.config import DEFAULTS
from test_contracts import valid_document


def multi_chapter_document(count=3, *, mara_in_every_chapter=False):
    """A valid analyzed document with one plain partition per EPUB chapter."""
    document = valid_document()
    template_block = document["source"]["blocks"][0]
    template_segment = document["segments"][0]
    blocks, segments, chapters = [], [], []
    for index in range(1, count + 1):
        chapter_id = f"ch{index}"
        items = [(f"The lantern moved along the old harbor wall in part {index}.", "narration", "narrator", "vn")]
        if index == 2 or mara_in_every_chapter:
            items.append(("Mara said softly, “The tide is turning tonight.”", "dialogue", "mara", "v1"))
        identifiers = []
        for block_index, (text, kind, role_id, voice_id) in enumerate(items, 1):
            identifier = f"b{index}_{block_index}"
            blocks.append(dict(deepcopy(template_block), id=identifier, chapter_id=chapter_id, text=text,
                               source={"href": f"text/{chapter_id}.xhtml", "element_id": None, "tag": "p"}))
            segments.append(dict(deepcopy(template_segment), id=f"s{index}_{block_index}", block_id=identifier,
                                 text=text, start=0, end=len(text), kind=kind, role_id=role_id, voice_id=voice_id))
            identifiers.append(identifier)
        chapters.append({"id": chapter_id, "title": f"Chapter {index}",
                         "source_href": f"text/{chapter_id}.xhtml", "block_ids": identifiers})
    document["source"]["chapters"] = chapters
    document["source"]["blocks"] = blocks
    document["segments"] = segments
    document["cast"]["characters"][0]["evidence"] = []
    for voice in document["cast"]["voices"]:
        voice["traits"] = []
    return document


def exported_book(document, root, name="book"):
    target = Path(root) / name
    export_bundle(document, target)
    return target


class ChapterIgnoreSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.document = multi_chapter_document()
        self.directory = exported_book(self.document, self.temp)
        self.bundle = load_bundle(self.directory)
        self.view = chapter_ignores.bundle_document(self.bundle)
        self.hash = self.bundle["bundle_hash"]

    def set_ignored(self, index, ignored, *, expected_manifest_hash=None, document=None):
        return chapter_ignores.set_ignored(self.directory, document or self.view, index, ignored,
                                           expected_manifest_hash=self.hash if expected_manifest_hash is None
                                           else expected_manifest_hash)

    def test_toggle_saves_exact_locator_and_round_trips_indices(self):
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, self.view), frozenset())
        self.assertEqual(self.set_ignored(1, True), frozenset({1}))
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, self.view), frozenset({1}))
        record = read_json(self.directory / chapter_ignores.FILENAME)
        self.assertEqual(record["format"], "epub-audiobook-chapter-ignores")
        self.assertEqual(record["book_id"], self.document["source"]["book"]["id"])
        partition = self.view["source"]["chapters"][1]
        self.assertEqual(record["ignored"], [{"chapter_id": "ch2",
                                              "block_ids": list(partition["block_ids"])}])
        self.assertEqual(self.set_ignored(1, True), frozenset({1}))
        self.assertEqual(self.set_ignored(1, False), frozenset())
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, self.view), frozenset())
        self.assertEqual(read_json(self.directory / chapter_ignores.FILENAME)["ignored"], [])

    def test_chapter_index_outside_the_partitions_is_refused_with_the_range(self):
        for index in (-1, 3, "1", 1.0):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "between 0 and 2"):
                    self.set_ignored(index, True)
        self.assertFalse((self.directory / chapter_ignores.FILENAME).exists())

    def test_manifest_hash_mismatch_writes_nothing(self):
        with self.assertRaisesRegex(ValueError, "changed after it was loaded"):
            self.set_ignored(1, True, expected_manifest_hash="0" * 64)
        self.assertFalse((self.directory / chapter_ignores.FILENAME).exists())
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, self.view), frozenset())
        self.assertEqual(self.set_ignored(1, True), frozenset({1}))

    def test_toggle_holds_the_analysis_update_lock_and_rereads_the_latest_choice(self):
        held = acquire_directory_lock(self.directory / chapter_ignores.UPDATE_LOCK)
        try:
            with self.assertRaisesRegex(ValueError, "analysis revision of this book is already active"):
                self.set_ignored(1, True)
        finally:
            held.rmdir()
        self.assertFalse((self.directory / chapter_ignores.FILENAME).exists())
        self.assertEqual(self.set_ignored(1, True), frozenset({1}))
        # A second toggle rereads the latest sidecar under the lock, so an
        # earlier concurrent choice is never lost.
        self.assertEqual(self.set_ignored(0, True), frozenset({0, 1}))

    def test_stale_locator_fails_closed_with_a_review_hint(self):
        self.set_ignored(1, True)
        record = read_json(self.directory / chapter_ignores.FILENAME)
        record["ignored"][0]["block_ids"] = ["missing-block"]
        write_json(self.directory / chapter_ignores.FILENAME, record)
        with self.assertRaisesRegex(ValueError, "no longer matches any chapter partition"):
            chapter_ignores.ignored_indices(self.directory, self.view)

    def test_foreign_book_sidecar_fails_closed(self):
        record = {"format": "epub-audiobook-chapter-ignores", "version": 1,
                  "book_id": "another-book", "ignored": []}
        write_json(self.directory / chapter_ignores.FILENAME, record)
        with self.assertRaisesRegex(ValueError, "book-local and never inherited"):
            chapter_ignores.ignored_indices(self.directory, self.view)

    def test_malformed_sidecars_fail_closed(self):
        base = {"format": "epub-audiobook-chapter-ignores", "version": 1,
                "book_id": self.document["source"]["book"]["id"], "ignored": []}
        cases = {
            "missing ignored": {key: value for key, value in base.items() if key != "ignored"},
            "null ignored": dict(base, ignored=None),
            "string ignored": dict(base, ignored="ch2"),
            "dict ignored": dict(base, ignored={"chapter_id": "ch2"}),
            "extra field": dict(base, unexpected=True),
            "wrong format": dict(base, format="something-else"),
            "wrong version": dict(base, version=2),
            "boolean version": dict(base, version=True),
            "float version": dict(base, version=1.0),
            "entry missing block_ids": dict(base, ignored=[{"chapter_id": "ch2"}]),
            "entry null block_ids": dict(base, ignored=[{"chapter_id": "ch2", "block_ids": None}]),
            "entry duplicate blocks": dict(base, ignored=[{"chapter_id": "ch2", "block_ids": ["b2_1", "b2_1"]}]),
        }
        for name, value in cases.items():
            with self.subTest(case=name):
                write_json(self.directory / chapter_ignores.FILENAME, value)
                with self.assertRaises(ValueError):
                    chapter_ignores.ignored_indices(self.directory, self.view)

    def test_toggle_with_a_stale_existing_locator_writes_nothing(self):
        stale = {"format": "epub-audiobook-chapter-ignores", "version": 1,
                 "book_id": self.document["source"]["book"]["id"],
                 "ignored": [{"chapter_id": "ch2", "block_ids": ["missing-block"]}]}
        sidecar = self.directory / chapter_ignores.FILENAME
        write_json(sidecar, stale)
        before = sidecar.read_bytes()
        with self.assertRaisesRegex(ValueError, "no longer matches any chapter partition"):
            self.set_ignored(0, True)
        self.assertEqual(sidecar.read_bytes(), before)
        with self.assertRaises(ValueError):
            chapter_ignores.ignored_indices(self.directory, self.view)

    def test_empty_chapter_partition_round_trips(self):
        document = multi_chapter_document(1)
        document["source"]["chapters"].append({"id": "ch2", "title": "Chapter Two",
                                               "source_href": "text/ch2.xhtml", "block_ids": []})
        empty_root = exported_book(document, self.temp, name="empty-book")
        view = chapter_ignores.bundle_document(load_bundle(empty_root))
        saved = chapter_ignores.set_ignored(empty_root, view, 1, True,
                                            expected_manifest_hash=digest(read_json(empty_root / "manifest.json")))
        self.assertEqual(saved, frozenset({1}))
        self.assertEqual(read_json(empty_root / chapter_ignores.FILENAME)["ignored"],
                         [{"chapter_id": "ch2", "block_ids": []}])
        self.assertEqual(chapter_ignores.ignored_indices(empty_root, view), frozenset({1}))
        self.assertEqual(chapter_ignores.set_ignored(empty_root, view, 1, False,
                                                     expected_manifest_hash=digest(read_json(empty_root / "manifest.json"))),
                          frozenset())

    def test_current_manifest_receipt_does_not_authorize_a_stale_document(self):
        stale = deepcopy(self.view)
        stale["source"]["chapters"][1]["block_ids"].reverse()

        with self.assertRaisesRegex(ValueError, "source changed after it was loaded"):
            self.set_ignored(1, True, document=stale)

        self.assertFalse((self.directory / chapter_ignores.FILENAME).exists())

    def test_current_manifest_receipt_does_not_authorize_stale_partition_boundaries(self):
        manifest = read_json(self.directory / "manifest.json")
        manifest["files"]["scripts"] = manifest["files"]["scripts"][:1]
        write_json(self.directory / "manifest.json", manifest)

        with self.assertRaisesRegex(ValueError, "chapter partitions changed after it was loaded"):
            self.set_ignored(1, True, expected_manifest_hash=digest(manifest))

        self.assertFalse((self.directory / chapter_ignores.FILENAME).exists())

    def test_choice_follows_its_text_across_title_and_ordinal_changes(self):
        self.set_ignored(1, True)
        reordered = deepcopy(self.view)
        reordered["source"]["chapters"] = [reordered["source"]["chapters"][2],
                                           reordered["source"]["chapters"][1],
                                           reordered["source"]["chapters"][0]]
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, reordered), frozenset({1}))
        retitled = deepcopy(self.view)
        for chapter in retitled["source"]["chapters"]:
            chapter["title"] = f"Retitled {chapter['title']}"
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, retitled), frozenset({1}))

    def test_freeze_snapshots_the_live_choice_into_the_config(self):
        config = {}
        self.assertEqual(chapter_ignores.freeze(self.bundle, config), frozenset())
        self.assertEqual(config[chapter_ignores.SNAPSHOT_KEY], frozenset())
        self.set_ignored(1, True)
        self.assertEqual(chapter_ignores.freeze(self.bundle, config), frozenset({1}))

    def test_included_source_keeps_only_rendered_blocks(self):
        self.assertEqual(chapter_ignores.included_source(self.bundle, frozenset()), self.bundle["source"])
        included = chapter_ignores.included_source(self.bundle, frozenset({1}))
        kept_ids = {block["id"] for block in included["blocks"]}
        self.assertEqual(kept_ids, {f"b1_1", f"b3_1"})
        self.assertEqual([chapter["id"] for chapter in included["chapters"]], ["ch1", "ch3"])
        self.assertEqual(included["book"], self.bundle["source"]["book"])


class ChapterIgnorePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.document = multi_chapter_document()
        self.directory = exported_book(self.document, self.temp)
        self.bundle = load_bundle(self.directory)
        self.config = deepcopy(DEFAULTS)

    def plan(self, config=None):
        return make_plan(self.bundle, config or self.config)

    def test_pristine_book_plans_byte_identically_without_and_with_an_empty_sidecar(self):
        pristine = self.plan()
        view = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, view, 1, True, expected_manifest_hash=self.bundle["bundle_hash"])
        chapter_ignores.set_ignored(self.directory, view, 1, False, expected_manifest_hash=self.bundle["bundle_hash"])
        self.assertEqual(self.plan(), pristine)
        self.assertTrue(pristine["chapters"])
        self.assertTrue(all("ignored" not in chapter for chapter in pristine["chapters"]))

    def test_ignored_chapter_keeps_identity_with_zero_chunks_and_no_readiness(self):
        directory = exported_book(multi_chapter_document(mara_in_every_chapter=True), self.temp, name="every")
        bundle = load_bundle(directory)
        view = chapter_ignores.bundle_document(bundle)
        chapter_ignores.set_ignored(directory, view, 1, True, expected_manifest_hash=bundle["bundle_hash"])
        plan = make_plan(bundle, deepcopy(DEFAULTS))
        self.assertFalse(plan["partial_book"])
        self.assertEqual([chapter["number"] for chapter in plan["chapters"]], [1, 2, 3])
        ignored = plan["chapters"][1]
        self.assertEqual(ignored, {"number": 2, "id": "chapter-0002", "title": "Chapter 2",
                                   "script_path": "scripts/chapter-0002.jsonl", "chunks": [], "ignored": True})
        self.assertEqual(plan["voice_ids"], ["v1", "vn"])
        self.assertEqual(plan["errors"], [])

    def test_ignored_chapter_excludes_its_unique_voices(self):
        directory = exported_book(multi_chapter_document(), self.temp, name="voices-book")
        bundle = load_bundle(directory)
        view = chapter_ignores.bundle_document(bundle)
        chapter_ignores.set_ignored(directory, view, 1, True, expected_manifest_hash=bundle["bundle_hash"])
        plan = make_plan(bundle, deepcopy(DEFAULTS))
        self.assertEqual(plan["voice_ids"], ["vn"])

    def test_ignore_skips_missing_speakers_only_for_ignored_chapters(self):
        for chapter in self.bundle["chapters"]:
            for row in chapter["rows"]:
                if row["block_id"].startswith("b2_"):
                    row["voice_id"] = None
        errors = self.plan()["errors"]
        self.assertTrue(any("s2_1" in error for error in errors), errors)
        view = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, view, 1, True, expected_manifest_hash=self.bundle["bundle_hash"])
        self.assertEqual(self.plan()["errors"], [])
        for chapter in self.bundle["chapters"]:
            for row in chapter["rows"]:
                if row["block_id"].startswith("b1_"):
                    row["voice_id"] = None
        errors = self.plan()["errors"]
        self.assertTrue(any("s1_1" in error for error in errors), errors)
        self.assertTrue(all("s2_" not in error for error in errors), errors)
        chapter_ignores.set_ignored(self.directory, view, 1, False, expected_manifest_hash=self.bundle["bundle_hash"])
        errors = self.plan()["errors"]
        self.assertTrue(any("s2_1" in error for error in errors), errors)

    def test_remaining_chapters_keep_their_chunk_ids_and_explicit_selection_stays_partial(self):
        pristine = self.plan()
        view = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, view, 1, True, expected_manifest_hash=self.bundle["bundle_hash"])
        ignored = self.plan()
        self.assertEqual([chunk["chunk_id"] for chunk in pristine["chapters"][0]["chunks"]],
                         [chunk["chunk_id"] for chunk in ignored["chapters"][0]["chunks"]])
        self.assertEqual([chunk["chunk_id"] for chunk in pristine["chapters"][2]["chunks"]],
                         [chunk["chunk_id"] for chunk in ignored["chapters"][2]["chunks"]])
        partial_config = deepcopy(self.config)
        partial_config["render"]["chapter_numbers"] = [1]
        self.assertTrue(self.plan(partial_config)["partial_book"])
        self.assertTrue(ignored["partial_book"] is False)

    def test_frozen_snapshot_survives_live_sidecar_changes_and_recovery_replans(self):
        view = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, view, 1, True, expected_manifest_hash=self.bundle["bundle_hash"])
        frozen = deepcopy(self.config)
        frozen[chapter_ignores.SNAPSHOT_KEY] = frozenset()
        self.assertEqual([chapter["number"] for chapter in self.plan(frozen)["chapters"] if chapter.get("ignored")], [])
        frozen[chapter_ignores.SNAPSHOT_KEY] = frozenset({0})
        chapter_ignores.set_ignored(self.directory, view, 1, False, expected_manifest_hash=self.bundle["bundle_hash"])
        self.assertEqual([chapter["number"] for chapter in self.plan(frozen)["chapters"] if chapter.get("ignored")], [1])

    def test_ignored_chapters_make_no_performance_plan_demand(self):
        expressive = deepcopy(self.config)
        expressive["render"]["performance_mode"] = "expressive"
        view = chapter_ignores.bundle_document(self.bundle)
        for index in range(3):
            chapter_ignores.set_ignored(self.directory, view, index, True,
                                        expected_manifest_hash=self.bundle["bundle_hash"])
        plan = self.plan(expressive)
        self.assertEqual([chapter["number"] for chapter in plan["chapters"] if chapter.get("ignored")], [1, 2, 3])
        chapter_ignores.set_ignored(self.directory, view, 1, False, expected_manifest_hash=self.bundle["bundle_hash"])
        with self.assertRaisesRegex(ValueError, "prepared performance plan"):
            self.plan(expressive)

    def test_labelling_model_runs_on_the_full_document_only_when_included_uses_are_pending(self):
        from epub_tts_narrator.name_pronunciations import save_definition
        from epub_tts_narrator.renderer.name_pronunciation_tags import prepare_name_pronunciations
        save_definition(self.directory, "tide", "T AY1 D", status="confirmed", source="manual",
                        parts_of_speech=["common_noun"])
        observed = {}

        def record_call(directory, config, **kwargs):
            observed["config"] = config
            observed["document"] = kwargs.get("document")
            return {}

        with patch("epub_tts_narrator.name_pronunciation_model.tag_defined_names", side_effect=record_call), \
                patch("epub_tts_narrator.config.load_config", return_value={"sentinel": True}):
            prepare_name_pronunciations(self.bundle, "unused.toml", ignored=frozenset({1}))
            self.assertEqual(observed, {})
            prepare_name_pronunciations(self.bundle, "unused.toml")
        self.assertEqual(observed["config"], {"sentinel": True})
        self.assertIs(observed["document"], self.bundle)


class ChapterIgnoreRebindTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.document = multi_chapter_document()
        self.directory = exported_book(self.document, self.temp)
        self.bundle = load_bundle(self.directory)
        self.view = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, self.view, 1, True,
                                    expected_manifest_hash=self.bundle["bundle_hash"])

    def test_title_only_update_preserves_the_exact_locator(self):
        retitled = deepcopy(self.document)
        for chapter in retitled["source"]["chapters"]:
            chapter["title"] = f"Retitled {chapter['title']}"
        records = _rebind_sidecars(self.directory, self.document, retitled)
        self.assertEqual(records[chapter_ignores.FILENAME],
                         read_json(self.directory / chapter_ignores.FILENAME))

    def test_changed_partition_layout_refuses_before_publishing(self):
        grown = deepcopy(self.document)
        template_block = grown["source"]["blocks"][0]
        identifier = "b2_extra"
        grown["source"]["blocks"].append(dict(deepcopy(template_block), id=identifier, chapter_id="ch2",
                                              text="A later paragraph joined this chapter.",
                                              source={"href": "text/ch2.xhtml", "element_id": None, "tag": "p"}))
        grown["source"]["chapters"][1]["block_ids"].append(identifier)
        from epub_tts_narrator.book_update import compare_sources
        aligned = compare_sources(self.document["source"], grown["source"])
        # The update pipeline merges first, so the published document carries
        # the aligned block IDs; the ignored partition's list still changed.
        updated = {"source": aligned["source"], "cast": self.document["cast"],
                   "segments": self.document["segments"]}
        self.assertNotEqual(updated["source"]["chapters"][1]["block_ids"],
                            self.document["source"]["chapters"][1]["block_ids"])
        with self.assertRaisesRegex(ValueError, "review the chapter ignores before updating"):
            _rebind_sidecars(self.directory, self.document, updated)
        # The sidecar itself is untouched by the refusal.
        self.assertEqual(chapter_ignores.ignored_indices(self.directory, self.view), frozenset({1}))

    def test_rebind_without_a_sidecar_is_a_no_op(self):
        empty_directory = exported_book(self.document, self.temp, name="plain")
        self.assertIsNone(chapter_ignores.rebind_validated(empty_directory, self.view, self.view))


class IgnoredLegacyNameMentionTests(unittest.TestCase):
    """A confirmed scoped rule raises on unresolved mentions only where they render."""

    def setUp(self):
        from epub_tts_narrator.name_script_state import load_name_script
        from epub_tts_narrator.name_script_workflow import prepare_name_script
        from epub_tts_narrator.scoped_pronunciations import set_choice
        from name_script_fixtures import NameClient, requests
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp).resolve()
        self.directory = self.root / "book"
        self.bundle = ignored_name_book(self.directory)
        # A Special rule confirmed before the name script exists is the one real
        # flow that leaves unresolved mentions reachable alongside a rule: the
        # confirmation gates only apply once name-script.json exists.
        set_choice(self.directory, self.bundle, "Ally", {"alphabet": "cmu_arpabet", "value": "AE1 L IY0"},
                   status="confirmed", scope={"class": "special"})
        result = prepare_name_script(self.directory, {}, requests=requests(self.root, NameClient(disagree=True)))
        self.assertEqual(sorted({item["block_id"] for item in result["unresolved"]}), ["b1", "b2"])

    def resolve_first_chapter(self):
        from epub_tts_narrator.name_script_state import load_name_script
        from epub_tts_narrator.name_script_workflow import resolve_mention
        for item in load_name_script(self.directory, self.bundle)["unresolved"]:
            if item["block_id"] == "b1":
                resolve_mention(self.directory, item["mention_id"], "N1")

    def ignore_chapter_two(self):
        return chapter_ignores.set_ignored(self.directory, chapter_ignores.bundle_document(self.bundle), 1, True,
                                           expected_manifest_hash=self.bundle["bundle_hash"])

    def plan(self):
        return make_plan(self.bundle, deepcopy(DEFAULTS))

    def test_only_included_unresolved_mentions_block(self):
        with self.assertRaisesRegex(ValueError, "Unresolved saved name mention"):
            self.plan()
        self.ignore_chapter_two()
        with self.assertRaisesRegex(ValueError, "Unresolved saved name mention"):
            self.plan()
        self.resolve_first_chapter()
        plan = self.plan()
        self.assertEqual([chapter["number"] for chapter in plan["chapters"] if chapter.get("ignored")], [2])
        chapter_ignores.set_ignored(self.directory, chapter_ignores.bundle_document(self.bundle), 1, False,
                                    expected_manifest_hash=self.bundle["bundle_hash"])
        with self.assertRaisesRegex(ValueError, "Unresolved saved name mention"):
            self.plan()

    def test_preflight_applies_the_same_filter(self):
        from epub_tts_narrator.renderer.workflow import _preflight
        self.ignore_chapter_two()
        with self.assertRaisesRegex(ValueError, "Unresolved saved name mention"):
            _preflight(self.bundle, deepcopy(DEFAULTS), chapters_only=True)
        self.resolve_first_chapter()
        config, plan, store = _preflight(self.bundle, deepcopy(DEFAULTS), chapters_only=True)
        self.assertEqual([chapter["number"] for chapter in plan["chapters"] if chapter.get("ignored")], [2])


def ignored_name_book(directory):
    """The name-script fixture layout with one block per source chapter."""
    document = valid_document()
    document["cast"]["roles"][0]["id"] = "narrator_main"
    for segment in document["segments"]:
        if segment["role_id"] == "narrator":
            segment["role_id"] = "narrator_main"
    document["cast"]["characters"][0].update(name="Ally", aliases=[], evidence=[{"block_id": "b1", "quote": "Ally"}])
    document["cast"]["voices"][1]["traits"] = []
    template = deepcopy(document["segments"][0])
    texts = [("b1", "ch1", "Ally greeted her ally softly."), ("b2", "ch2", "Ally waited.")]
    for block, (_, chapter_id, text) in zip(document["source"]["blocks"], texts):
        block.update(chapter_id=chapter_id, text=text, source={"href": f"text/{chapter_id}.xhtml",
                                                               "element_id": None, "tag": "p"})
    document["source"]["chapters"] = [
        {"id": chapter_id, "title": f"Chapter {index}", "source_href": f"text/{chapter_id}.xhtml",
         "block_ids": [block_id]} for index, (block_id, chapter_id, _) in enumerate(texts, 1)]
    document["segments"] = [dict(deepcopy(template), id=block_id, block_id=block_id, start=0,
                                 end=len(text), text=text, kind="narration", role_id="narrator_main",
                                 voice_id="vn") for block_id, _, text in texts]
    from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
    from epub_tts_narrator.series import link_analysis
    registry = fresh_registry(directory)
    publish_directory(link_analysis(document, registry), registry, directory, export_bundle)
    return load_bundle(directory)


if __name__ == "__main__":
    unittest.main()
