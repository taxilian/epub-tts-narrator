"""Book sample provenance reaches newly created voices without replacing saved ones."""

from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main, parser
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import ensure_voice
from test_renderer_workflow import FFMPEG, FixtureEngine, FixtureVerifier


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


@unittest.skipUnless(FFMPEG, "FFmpeg is required for reference recording integration")
class SampleWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle = load_bundle(EXAMPLE)
        self.config = deepcopy(DEFAULTS)
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="sample-series")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                     max_attempts=1, ffmpeg=FFMPEG)
        self.config["quality"] = validate_policy({})
        transcripts = {}
        self.engine = FixtureEngine(transcripts)
        self.verifier = FixtureVerifier(transcripts)
        self.profile = self.bundle["cast"]["voices"][0]
        self.voice_id = self.profile["id"]
        self.store = VoiceStore(
            self.config["voices"]["directory"], self.config["voices"]["series_id"],
            cast=self.bundle["cast"])
        self.sample = {
            "reference_text": "A book passage chosen for this character.",
            "source": {
                "kind": "book_excerpt", "book_id": self.bundle["source"]["book"]["id"],
                "segment_ids": ["example-segment"], "chapter_id": "example-chapter",
            },
        }

    def create_voice(self, **kwargs):
        lock = self.store.lock()
        try:
            return ensure_voice(
                self.store, self.voice_id, self.profile, self.engine, self.verifier,
                self.config, self.config["render"]["work_dir"], notify=lambda _: None, **kwargs)
        finally:
            lock.rmdir()

    def test_book_sample_and_provenance_reach_generation_and_saved_reference(self):
        with patch("epub_tts_narrator.renderer.samples.select_sample", return_value=self.sample) as select:
            saved = self.create_voice(bundle=self.bundle)
        select.assert_called_once_with(
            self.bundle, self.voice_id, canonical_id=self.store.canonical_id,
            fallback_text=self.config["voices"]["reference_text"])
        self.assertEqual(self.engine.voice_calls[0]["text"], self.sample["reference_text"])
        self.assertEqual(saved["reference_text"], self.sample["reference_text"])
        self.assertEqual(saved["creation"]["requested_reference_text"], self.sample["reference_text"])
        self.assertEqual(saved["creation"]["sample_source"], self.sample["source"])
        accepted_paths = list((self.root / "work" / "voice-candidates").glob("*/accepted.json"))
        self.assertEqual(len(accepted_paths), 1)
        accepted = read_json(accepted_paths[0])
        self.assertEqual(accepted["request"]["sample_source"], self.sample["source"])
        self.assertEqual(accepted["request"]["text"], self.sample["reference_text"])

    def test_existing_reference_is_reused_before_sample_selection_without_relabeling(self):
        saved = self.create_voice()
        before = {p: p.read_bytes() for p in self.store.folder(self.voice_id).iterdir() if p.is_file()}
        generation_count, verification_count = len(self.engine.voice_calls), len(self.verifier.calls)
        with patch("epub_tts_narrator.renderer.samples.select_sample",
                   side_effect=AssertionError("An existing voice must not select a new sample")):
            reused = self.create_voice(bundle=self.bundle)
        self.assertEqual(reused, saved)
        self.assertEqual(reused["reference_text"], self.config["voices"]["reference_text"])
        self.assertEqual(len(self.engine.voice_calls), generation_count)
        self.assertEqual(len(self.verifier.calls), verification_count)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_legacy_call_without_bundle_keeps_configured_script(self):
        with patch("epub_tts_narrator.renderer.samples.select_sample") as select:
            saved = self.create_voice()
        select.assert_not_called()
        self.assertEqual(self.engine.voice_calls[0]["text"], self.config["voices"]["reference_text"])
        self.assertNotIn("sample_source", saved["creation"])

    def test_later_book_alias_selects_with_present_role_id_and_saves_canonical_voice(self):
        canonical_voice = "earlier-book-primary-voice"
        self.store = VoiceStore(
            self.config["voices"]["directory"], self.config["voices"]["series_id"],
            cast=self.bundle["cast"], explicit_aliases={self.voice_id: canonical_voice})
        self.assertNotIn(canonical_voice, {voice["id"] for voice in self.bundle["cast"]["voices"]})
        saved = self.create_voice(bundle=self.bundle)
        self.assertEqual(saved["voice_id"], canonical_voice)
        self.assertEqual(saved["creation"]["sample_source"]["canonical_voice_id"], canonical_voice)
        self.assertEqual(self.store.load(self.voice_id), saved)
        self.assertEqual(len(self.engine.voice_calls), 1)

    def test_create_voices_cli_supplies_bundle_for_each_new_voice(self):
        with (patch("epub_tts_narrator.renderer.cli.load_config", return_value=self.config),
              patch("epub_tts_narrator.renderer.cli.load_bundle", return_value=self.bundle),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts", return_value=self.bundle),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier),
              patch("epub_tts_narrator.renderer.workflow.ensure_voice") as ensure,
              redirect_stdout(io.StringIO())):
            result = main(["create-voices", str(EXAMPLE), "--config", "unused.toml"])
        self.assertEqual(result, 0)
        self.assertGreater(len(ensure.call_args_list), 0)
        self.assertTrue(all(call.kwargs["bundle"] is self.bundle for call in ensure.call_args_list))


class SampleCommandTests(unittest.TestCase):
    def test_prepare_samples_is_an_explicit_cast_action(self):
        args = parser().parse_args(["cast", str(EXAMPLE), "--config", "renderer.toml",
                                    "--action", "prepare-samples"])
        self.assertEqual(args.action, "prepare-samples")


if __name__ == "__main__":
    unittest.main()
