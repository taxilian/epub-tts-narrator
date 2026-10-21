"""Book-local voices follow the analyzed lineage, including late-created assets.

The audio and ASR are deterministic fixtures. These tests exercise publication,
casting, imports, and both CLIs without a provider or speech-model request.
"""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.cli import main as renderer_main
from epub_tts_narrator.renderer.config import DEFAULTS, load_config as renderer_config
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import import_voice
from epub_tts_narrator.analysis_directory import (
    continuation_registry, fresh_registry, publish_directory,
)
from epub_tts_narrator.cli import export_bundle, main as analyzer_main
from epub_tts_narrator.series import link_analysis
from test_pipeline import FakeClient
from test_renderer_voice_design_integration import (
    FFMPEG, FFPROBE, ReferenceDesigner, TranscriptVerifier,
)
from test_series import fixture, propose
from audition_fixtures import prepare_bundle


class BookFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "tts-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["render"].update(ffmpeg=FFMPEG or "ffmpeg", ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = ReferenceDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)

    @staticmethod
    def snapshot(directory):
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def publish(self, number, parent=None):
        directory = self.root / f"book-{number:02}"
        document = fixture(sequel=parent is not None)
        document["source"]["book"].update(id=f"book-{number}", title=f"Journey {number}")
        if parent is None:
            registry = fresh_registry(directory)
        else:
            registry = continuation_registry(parent["directory"], document["source"])
            propose(document, parent["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle,
                          continue_from=parent["directory"] if parent else None)
        return load_bundle(directory)

    def lineage(self):
        first = self.publish(1)
        second = self.publish(2, first)
        third = self.publish(3, second)
        sibling = self.publish(4, first)
        return first, second, third, sibling

    def session(self, bundle, *, models=True):
        def forbidden(*args, **kwargs):
            self.fail("Inherited voices must be reused without initializing a model")
        return CastingSession(
            prepare_bundle(bundle), self.config,
            engine_factory=(lambda _: self.engine) if models else forbidden,
            verifier_factory=(lambda *args, **kwargs: self.verifier) if models else forbidden,
            designer_factory=forbidden, notify=lambda _: None,
        )

    def store(self, bundle):
        return voice_store(bundle, resolve_book_config(bundle, self.config))


@unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
class BookVoiceWorkflowTests(BookFixture):
    def test_late_casting_and_nearest_override_preserve_earlier_and_sibling_books(self):
        first, second, third, sibling = self.lineage()
        books = (first, second, third, sibling)
        self.assertTrue(all(not (Path(book["directory"]) / "voices").exists() for book in books))
        voice_id = first["cast"]["characters"][0]["voice_id"]
        narrator_id = next(role["voice_id"] for role in first["cast"]["roles"]
                           if role["kind"] == "narration")
        self.assertNotEqual(voice_id, narrator_id)
        self.assertTrue(all(book["cast"]["characters"][0]["voice_id"] == voice_id for book in books))

        # Open the descendant before any clone exists. Parent references are
        # looked up live rather than frozen into analysis or session creation.
        with contextlib.closing(self.session(third, models=False)) as third_session:
            self.assertIsNone(third_session.store.load(voice_id))
            with contextlib.closing(self.session(first)) as first_session:
                self.assertEqual(len(first_session.generate_missing()), 2)
                original = first_session.store.load(voice_id)
                narration = first_session.store.load(narrator_id)
            first_before = self.snapshot(Path(first["directory"]))
            sibling_before = self.snapshot(Path(sibling["directory"]))
            self.assertEqual(third_session.generate_missing(), [])
            inherited = third_session.store.load(voice_id)
            self.assertEqual(inherited["asset_hash"], original["asset_hash"])
            self.assertEqual(inherited["reference_audio"], original["reference_audio"])
            self.assertTrue(inherited["inherited"])
            self.assertEqual(len(self.engine.calls), 2)

            with contextlib.closing(self.session(second)) as second_session:
                second_session.update_prompt(
                    voice_id, "A mature, warm alto with crisp consonants.",
                    "We have reached the harbor safely and will begin the next journey at dawn.")
                replacement = second_session.regenerate(voice_id)
                self.assertNotEqual(replacement["asset_hash"], original["asset_hash"])
                self.assertTrue(Path(replacement["reference_audio"]).is_relative_to(
                    Path(second["directory"]) / "voices"))
                self.assertEqual(second_session.generate_missing(), [])

            self.assertEqual(third_session.generate_missing(), [])
            self.assertEqual(third_session.store.load(voice_id)["asset_hash"], replacement["asset_hash"])
            # Book 2 has a store, but no local narrator: lookup must keep going
            # to book 1 for that specific voice rather than stop at the store.
            self.assertEqual(third_session.store.load(narrator_id)["reference_audio"],
                             narration["reference_audio"])
            self.assertEqual(len(self.engine.calls), 3)

        self.assertEqual(self.store(first).load(voice_id)["asset_hash"], original["asset_hash"])
        self.assertEqual(self.store(sibling).load(voice_id)["asset_hash"], original["asset_hash"])
        self.assertEqual(self.store(second).load(voice_id)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.snapshot(Path(first["directory"])), first_before)
        self.assertEqual(self.snapshot(Path(sibling["directory"])), sibling_before)
        self.assertFalse((Path(third["directory"]) / "voices").exists())

    def test_import_overrides_only_inherited_voice_and_refuses_local_replacement(self):
        first, second, third, sibling = self.lineage()
        voice_id = first["cast"]["characters"][0]["voice_id"]
        with contextlib.closing(self.session(first)) as session:
            session.generate_missing()
            original = session.store.load(voice_id)
        first_before = self.snapshot(Path(first["directory"]))
        sibling_before = self.snapshot(Path(sibling["directory"]))
        text = "I have brought a different lantern for the journey through the mountain pass."
        audio = self.root / "chosen-reference.wav"
        self.engine.write_audio(text, audio, seed=613)
        imported = import_voice(second, voice_id, audio, text, self.config,
                                self.engine, self.verifier)
        self.assertNotEqual(imported["asset_hash"], original["asset_hash"])
        self.assertTrue(Path(imported["reference_audio"]).is_relative_to(
            Path(second["directory"]) / "voices"))
        self.assertEqual(self.store(third).load(voice_id)["asset_hash"], imported["asset_hash"])
        self.assertEqual(self.store(sibling).load(voice_id)["asset_hash"], original["asset_hash"])
        second_before = self.snapshot(Path(second["directory"]))
        with self.assertRaisesRegex(ValueError, "already exists"):
            import_voice(second, voice_id, audio, text, self.config, self.engine, self.verifier)
        self.assertEqual(self.snapshot(Path(second["directory"])), second_before)
        self.assertEqual(self.snapshot(Path(first["directory"])), first_before)
        self.assertEqual(self.snapshot(Path(sibling["directory"])), sibling_before)


class SharedGlobalConfigurationWorkflowTests(BookFixture):
    @staticmethod
    def run_cli(function, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = function([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_both_clis_use_one_global_file_and_book_locations_without_series_settings(self):
        config_path = self.root / "epub-audiobook.toml"
        config_path.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus = false\nattribution_review = false\n'
            'attribution_workers = 5\nrepair_attempts = 0\n'
            '[engine]\ntype = "drama3_mlx"\ncheckpoint = "./tts-model"\n'
            '[voice_design]\ntype = "engine"\n'
            '[verification]\ntype = "mlx_whisper"\nmodel = "./asr-model"\n'
            '[render]\nmax_attempts = 2\n'
            '[quality]\nmax_wer = 0.0\n', encoding="utf-8")
        original_config = config_path.read_bytes()
        source = self.root / "source.json"
        source.write_text(json.dumps({"format": "epub-audiobook-source", "version": 1,
                                      "source": fixture()["source"]}), encoding="utf-8")
        output = self.root / "book-output"
        client = FakeClient()
        client.model = client.identity["model"]
        client.base_url = client.identity["base_url"]
        with contextlib.chdir(self.root):
            with patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client):
                status, _, error = self.run_cli(analyzer_main, "analyze", source, "-o", output)
            self.assertEqual(status, 0, error)
            self.assertGreater(len(client.calls), 0)
            status, report, error = self.run_cli(renderer_main, "plan", output)
            self.assertEqual(status, 0, error)
            self.assertEqual(json.loads(report)["book"], fixture()["source"]["book"]["title"])
            bundle = load_bundle(output)
            loaded = renderer_config()
        self.assertIsNone(loaded["voices"]["series_id"])
        self.assertIsNone(loaded["voices"]["directory"])
        resolved = resolve_book_config(bundle, loaded)
        self.assertEqual(resolved["voices"]["series_id"],
                         bundle["manifest"]["provenance"]["series"]["series_id"])
        self.assertEqual(Path(resolved["voices"]["directory"]), output / "voices")
        self.assertEqual(Path(resolved["render"]["work_dir"]), output / "render-work")
        self.assertEqual(Path(resolved["render"]["output_dir"]), output / "audiobook")
        self.assertEqual(config_path.read_bytes(), original_config)
        for name in ("voices", "render-work", "audiobook"):
            self.assertFalse((output / name).exists())


if __name__ == "__main__":
    unittest.main()
