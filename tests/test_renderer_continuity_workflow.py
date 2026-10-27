from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
import tempfile
import unittest

from audition_fixtures import prepare_bundle
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json, file_hash
from epub_tts_narrator.renderer.engines import Drama3MLXEngine
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.workflow import render, assemble, make_render_plan, _render_chapter_chunks
from shared_fixtures import renderer_config
import test_renderer_continuity as fixtures
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier, write_tone
from test_renderer_paragraphs import passages


class LocalAdapter(Drama3MLXEngine):
    pass


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ContinuityWorkflowTests(unittest.TestCase):
    def setUp(self):
        import soundfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = renderer_config(self.root, series_id="continuity-workflow")
        self.local = LocalAdapter(self.root)
        self.local._identity = {"engine": "drama3_mlx", "model": "fishaudio/drama-3", "fixture": True}
        self.model = self.local._model = fixtures.Model()
        self.local._soundfile = soundfile
        self.verifier = fixtures.Verifier(self.model)

    def test_normal_render_checkpoint_resume_and_assemble_keep_references(self):
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        creator = FixtureEngine({})
        creator.identity = deepcopy(self.local.identity)
        render(bundle, self.config, creator, FixtureVerifier(creator.fixture_transcripts),
               chapters_only=True, notify=lambda message: None)
        reference_files = {path: file_hash(path) for path in Path(self.config["voices"]["directory"]).rglob("*.wav")}
        self.config["render"]["continuity"] = "chapter"
        self.assertEqual(make_render_plan(bundle, self.config)["continuity"], "chapter")
        first = render(bundle, self.config, self.local, self.verifier, chapters_only=True, notify=lambda message: None)
        self.assertTrue(any(options["previous_takes"] for text, options in self.model.calls))
        generated = len(self.model.calls)
        second = render(bundle, self.config, self.local, self.verifier, chapters_only=True, notify=lambda message: None)
        self.assertEqual(first["chapters"], second["chapters"])
        self.assertEqual(len(self.model.calls), generated)
        result = assemble(bundle, self.config, chapters_only=True, notify=lambda message: None)
        self.assertEqual(result["status"], "chapters_complete")
        self.assertEqual(reference_files, {path: file_hash(path) for path in reference_files})

    def test_pool_chapters_overlap_without_running_successors_before_acceptance(self):
        self.config["render"]["continuity"] = "chapter"
        self.config["render_pool"].update(workers=[{"name": "local", "type": "local", "concurrency": 1}],
            scheduler="pipeline", chapter_concurrency=2, max_inflight=8)
        pool = RenderPool(self.config)
        self.addCleanup(pool.close)
        self.local._identity.update(model_revision=pool.identity["model_revision"], **pool.identity["sampling"])
        pool._workers[0]._engine = self.local
        reference = self.root / "reference.wav"
        write_tone(reference)
        voices = {"v1": {"kind": "reference", "voice_id": "v1", "asset_hash": "a" * 64,
            "reference_audio": str(reference), "reference_sha256": file_hash(reference),
            "reference_text": "Fixed reference."}}
        first_started, second_started = Event(), Event()
        transcribe = self.verifier.transcribe
        def overlapping_asr(path):
            result = transcribe(path)
            if result["text"] == "First chapter.":
                first_started.set()
                if not second_started.wait(10):
                    raise AssertionError("Independent chapter did not overlap acceptance")
            elif result["text"] == "Other chapter.":
                second_started.set()
            return result
        self.verifier.transcribe = overlapping_asr
        chapters = [passages(["First chapter.", "First successor."], block="a"),
                    passages(["Other chapter.", "Other successor."], block="b")]
        for chapter in chapters:
            for chunk in chapter:
                chunk["segment"]["delivery"]["direction"] = ""
        def run(index):
            return _render_chapter_chunks({"number": index + 1, "chunks": chapters[index]}, voices,
                pool, pool.identity, self.verifier, self.config, self.root / "work", notify=lambda message: None,
                scheduler_owner=("run", index))
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(run, 0)
            self.assertTrue(first_started.wait(10))
            second = workers.submit(run, 1)
            first.result(timeout=20)
            second.result(timeout=20)
        histories = {text: [old for old, codes in options["previous_takes"]]
                     for text, options in self.model.calls}
        self.assertEqual(histories["First successor."], ["First chapter."])
        self.assertEqual(histories["Other successor."], ["Other chapter."])
        self.assertEqual(histories["Other chapter."], [])

    def test_config_parses_off_default_and_rejects_unknown_modes(self):
        from epub_tts_narrator.renderer.config import load_config
        path = self.root / "config.toml"
        base = '[engine]\ncheckpoint="./model"\n[verification]\nmodel="./asr"\n'
        path.write_text(base)
        self.assertEqual(load_config(path)["render"]["continuity"], "off")
        path.write_text(base + '[render]\ncontinuity="chapter"\n')
        self.assertEqual(load_config(path)["render"]["continuity"], "chapter")
        path.write_text(base + '[render]\ncontinuity="all"\n')
        with self.assertRaisesRegex(ValueError, "continuity"):
            load_config(path)
