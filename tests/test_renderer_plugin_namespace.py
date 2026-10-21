"""The package rename preserves external TTS adapters and one lookup policy."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from epub_tts_narrator.renderer.cli import doctor
from epub_tts_narrator.renderer.engines import (
    EngineError, create_engine, engine_entry_points,
)


CURRENT = "epub_tts_narrator.tts"
LEGACY = "audiobook_renderer.tts"


class EntryPoints(list):
    def select(self, *, name):
        return EntryPoints(entry for entry in self if entry.name == name)


class PluginNamespaceTests(unittest.TestCase):
    def entry(self, name="sample_tts"):
        plugin = SimpleNamespace(
            identity={"engine": name, "model": "test"},
            capabilities={"voice_design": False, "reference_voice": True,
                          "preset_voice": False, "delivery_control": False},
            create_voice=Mock(), synthesize=Mock(),
        )
        return SimpleNamespace(name=name, load=Mock(return_value=Mock(return_value=plugin)))

    def registry(self, current=(), legacy=()):
        groups = {CURRENT: EntryPoints(current), LEGACY: EntryPoints(legacy)}
        return patch("epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
                     side_effect=lambda *, group: groups[group])

    def test_current_name_wins_without_loading_or_querying_legacy(self):
        current, legacy = self.entry(), self.entry()
        with self.registry([current], [legacy]) as lookup:
            result = create_engine({"type": "sample_tts", "options": {}})
        self.assertIs(result, current.load.return_value.return_value)
        lookup.assert_called_once_with(group=CURRENT)
        legacy.load.assert_not_called()

    def test_legacy_name_is_used_when_current_group_has_only_other_names(self):
        unrelated, legacy = self.entry("unrelated"), self.entry()
        with self.registry([unrelated], [legacy]) as lookup:
            result = create_engine({"type": "sample_tts", "options": {}})
        self.assertIs(result, legacy.load.return_value.return_value)
        self.assertEqual(lookup.call_args_list, [call(group=CURRENT), call(group=LEGACY)])
        unrelated.load.assert_not_called()

    def test_duplicate_current_or_legacy_names_are_rejected_without_loading(self):
        for use_current in (True, False):
            duplicates = [self.entry(), self.entry()]
            fallback = self.entry()
            current = duplicates if use_current else []
            legacy = [fallback] if use_current else duplicates
            with self.subTest(current=use_current), self.registry(current, legacy) as lookup:
                with self.assertRaisesRegex(EngineError, "ambiguous"):
                    create_engine({"type": "sample_tts"})
                self.assertEqual(lookup.call_count, 1 if use_current else 2)
            for entry in duplicates + [fallback]:
                entry.load.assert_not_called()

    def test_missing_name_checks_both_groups(self):
        with self.registry([self.entry("other")], [self.entry("another")]) as lookup:
            self.assertEqual(engine_entry_points("sample_tts"), [])
            self.assertEqual(lookup.call_args_list, [call(group=CURRENT), call(group=LEGACY)])

    def test_invalid_current_plugin_does_not_fall_back_to_legacy(self):
        current, legacy = self.entry(), self.entry()
        current.load.side_effect = ImportError("fixture")
        with self.registry([current], [legacy]) as lookup:
            with self.assertRaisesRegex(EngineError, "could not initialize"):
                create_engine({"type": "sample_tts"})
        lookup.assert_called_once_with(group=CURRENT)
        legacy.load.assert_not_called()

    def test_doctor_uses_same_precedence_and_ambiguity_without_initializing_plugins(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            (model / "weights.npz").write_bytes(b"fixture")
            config = {
                "engine": {"type": "sample_tts", "checkpoint": None},
                "verification": {"model": str(model)},
                "render": {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"},
            }
            current, legacy, duplicate = self.entry(), self.entry(), self.entry()
            scenarios = [
                ([current], [legacy], True, [call(group=CURRENT)]),
                ([], [legacy], True, [call(group=CURRENT), call(group=LEGACY)]),
                ([current, duplicate], [legacy], False, [call(group=CURRENT)]),
                ([], [legacy, duplicate], False, [call(group=CURRENT), call(group=LEGACY)]),
                ([], [], False, [call(group=CURRENT), call(group=LEGACY)]),
            ]
            for primary, old, expected, calls in scenarios:
                with (self.subTest(primary=len(primary), legacy=len(old)),
                      self.registry(primary, old) as lookup,
                      patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="test"),
                      patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture")):
                    report = doctor(config)
                self.assertIs(report["checks"]["tts_plugin_installed"], expected)
                self.assertIs(report["ready"], expected)
                self.assertEqual(lookup.call_args_list, calls)
            for entry in (current, legacy, duplicate):
                entry.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
