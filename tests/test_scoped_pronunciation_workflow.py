from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.name_script_workflow import prepare_name_script
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.common import file_hash
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.renderer.workflow import render
from epub_tts_narrator.scoped_pronunciations import scope_for, set_choice
from name_script_fixtures import NameClient, PHONES, TEXT, book, requests
from shared_fixtures import HISTORY_CAPABILITY, fixture_codes, renderer_config
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier, write_tone


class ScopedSurfaceEngine(FixtureEngine):
    identity = {"engine": "drama3_mlx", "model": "fishaudio/drama-3", "fixture": "scoped-surface-v1"}
    capabilities = {"reference_voice": True, "structured_performance": True, **HISTORY_CAPABILITY}

    def synthesize(self, text, voice, path, *, seed, direction="", performance=None, traits=""):
        _, compiled, metadata = prepare_synthesis(text, direction, performance=performance, traits=traits)
        self.speech_calls.append({"text": text, "compiled": compiled,
                                  "performance": deepcopy(performance), "voice_hash": voice["asset_hash"]})
        return {**self._write(text, path, seed), **metadata, **fixture_codes(text, [voice])}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class ScopedPronunciationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = book(self.root / "book", two=True)
        self.config = renderer_config(self.root, paragraph_synthesis=False)
        for section, key in (("voices", "directory"), ("voices", "series_id"),
                             ("render", "work_dir"), ("render", "output_dir"), ("render", "epub")):
            self.config[section][key] = None
        self.engine = ScopedSurfaceEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.config = resolve_book_config(self.bundle, self.config)
        self.store = voice_store(self.bundle, self.config)
        reference = self.root / "reference.wav"
        write_tone(reference)
        for profile in self.bundle["cast"]["voices"]:
            self.store.publish(profile["id"], profile, reference, "Unchanged reference.",
                               self.engine.identity, {"passed": True}, {})

    def run_render(self):
        return render(self.bundle, self.config, self.engine, self.verifier, chapters_only=True,
                      notify=lambda message: None, voice_store_override=self.store)

    def test_render_and_resume_regenerate_only_the_entity_with_changed_pronunciation(self):
        canonical = {path: file_hash(path) for path in Path(self.bundle["directory"]).rglob("*") if path.is_file()}
        references = {path: file_hash(path) for path in Path(self.config["voices"]["directory"]).rglob("*.wav")}
        self.run_render()
        baseline = len(self.engine.speech_calls)
        prepare_name_script(self.bundle["directory"], {}, requests=requests(self.root, NameClient()))
        self.run_render()
        self.assertEqual(len(self.engine.speech_calls), baseline)
        set_choice(self.bundle["directory"], self.bundle, "Ally", PHONES, scope=scope_for(self.bundle, "N1"))

        self.run_render()

        # The changed take is regenerated; the following take in the chapter is
        # regenerated only because it continues that take's history, and keeps
        # its unchanged pronunciation.
        self.assertEqual(len(self.engine.speech_calls), baseline + 2)
        changed, following = self.engine.speech_calls[-2:]
        original_following = next(call for call in self.engine.speech_calls[:baseline]
                                   if call["text"] == following["text"])
        self.assertNotEqual(following["text"], TEXT)
        self.assertEqual(following["compiled"], original_following["compiled"])
        self.assertEqual(following["voice_hash"], original_following["voice_hash"])
        self.assertEqual(changed["text"], TEXT)
        self.assertEqual([item["start"] for item in changed["performance"]["controls"]], [0, TEXT.index("ALLY")])
        self.assertNotIn("(N1:", changed["compiled"])
        self.assertIn("her ally", changed["compiled"])
        self.assertIn("Ally with", changed["compiled"])
        self.run_render()
        self.assertEqual(len(self.engine.speech_calls), baseline + 2)
        self.assertEqual(canonical, {path: file_hash(path) for path in canonical})
        self.assertTrue(references)
        self.assertEqual(references, {path: file_hash(path) for path in references})
