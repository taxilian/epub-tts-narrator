"""Per-book configuration must never redirect writes into earlier books."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.series import link_analysis
from test_series import fixture, propose


class BookContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first = self.make_book("first")
        self.second = self.make_book("second", self.first)
        self.bundle = load_bundle(self.second)
        self.config = deepcopy(DEFAULTS)

    def make_book(self, name, previous=None):
        output = self.root / name
        document = fixture(sequel=previous is not None)
        document["source"]["book"]["id"] = name
        registry = (continuation_registry(previous, document["source"]) if previous
                    else fresh_registry(output))
        if previous:
            propose(document, registry["books"][0]["cast"]["characters"][0]["id"])
        publish_directory(link_analysis(document, registry), registry, output, export_bundle,
                          continue_from=previous)
        return output

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_binding_is_read_only_and_leaves_global_config_reusable(self):
        original = deepcopy(self.config)
        before = self.snapshot()
        first = resolve_book_config(load_bundle(self.first), self.config)
        second = resolve_book_config(self.bundle, self.config)
        self.assertEqual(self.config, original)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(second["voices"]["directory"], str(self.second / "voices"))
        self.assertEqual(second["render"]["work_dir"], str(self.second / "render-work"))
        self.assertEqual(second["render"]["output_dir"], str(self.second / "audiobook"))
        self.assertEqual(first["voices"]["series_id"], second["voices"]["series_id"])
        self.assertEqual(second["_voice_parents"], [str(self.first)])

    def test_output_and_work_overrides_cannot_target_ancestor_or_source(self):
        targets = [self.first, self.first / "new-output", self.first / "voices",
                   self.root, self.second, self.second / "scripts",
                   self.second / "manifest.json", self.second / "voices"]
        before = self.snapshot()
        for argument in ("output", "work_dir"):
            for target in targets:
                with self.subTest(argument=argument, target=target), self.assertRaises(ValueError):
                    resolve_book_config(self.bundle, self.config, **{argument: target})
        self.assertEqual(self.snapshot(), before)

    def test_explicit_locations_outside_books_remain_supported(self):
        config = resolve_book_config(self.bundle, self.config,
                                     output=self.root / "finished-audio",
                                     work_dir=self.root / "temporary-work")
        self.assertEqual(config["render"]["output_dir"], str(self.root / "finished-audio"))
        self.assertEqual(config["render"]["work_dir"], str(self.root / "temporary-work"))
        self.assertFalse((self.root / "temporary-work").exists())

    def test_voice_symlink_cannot_redirect_local_writes(self):
        for destination in (self.first / "voices", self.root / "elsewhere"):
            link = self.second / "voices"
            link.symlink_to(destination, target_is_directory=True)
            try:
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    resolve_book_config(self.bundle, self.config)
            finally:
                link.unlink()
            self.assertFalse(destination.exists())

    def test_output_symlink_cannot_redirect_writes_into_ancestor(self):
        redirect = self.root / "redirect"
        redirect.symlink_to(self.first, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestor"):
            resolve_book_config(self.bundle, self.config, output=redirect / "audiobook")

    def test_ancestor_voice_symlink_is_rejected_on_lookup(self):
        destination = self.root / "outside-voices"
        destination.mkdir()
        (self.first / "voices").symlink_to(destination, target_is_directory=True)
        config = resolve_book_config(self.bundle, self.config)
        store = voice_store(self.bundle, config)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            store.load(self.bundle["cast"]["voices"][0]["id"])
        self.assertEqual(list(destination.iterdir()), [])
        self.assertFalse((self.second / "voices").exists())

    def test_bound_config_cannot_be_reused_for_different_book(self):
        config = resolve_book_config(self.bundle, self.config)
        with self.assertRaisesRegex(ValueError, "another book"):
            resolve_book_config(load_bundle(self.first), config)

    def test_global_book_specific_settings_fail_before_writing(self):
        before = self.snapshot()
        for section, key in (("voices", "directory"), ("voices", "series_id"),
                             ("render", "work_dir"), ("render", "output_dir"),
                             ("render", "epub")):
            config = deepcopy(self.config)
            config[section][key] = str(self.root / "legacy-value")
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, "remove"):
                resolve_book_config(self.bundle, config)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
