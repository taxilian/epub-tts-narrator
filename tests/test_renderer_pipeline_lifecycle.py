"""Pipeline lifecycle: mode identity, warm cache reuse, and shared take dedup.

Locks end-to-end lifecycle behavior the lane/scheduler unit tests do not
cover: switching scheduler modes over one book consumes exactly the same
accepted caches with zero new generation, and two chapters presenting the
identical performance request share one take lock, one retry series, and
one accepted take instead of racing over the same attempt files.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.engines import DRAMA3_MODEL_REVISION
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import render
from audition_fixtures import prepare_bundle
from shared_fixtures import FFMPEG, FFPROBE, renderer_config, timed_transcript
from test_renderer_workflow import EXAMPLE, FixtureEngine, FixtureVerifier, pcm_key

WATCHDOG = 30


def deadline_loop(seconds, condition):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class FirstAttemptFailsVerifier(FixtureVerifier):
    """Rejects the marker text exactly once, then verifies normally."""

    def __init__(self, fixture_transcripts, fail_text):
        super().__init__(fixture_transcripts)
        self.fail_text = fail_text
        self.remaining_failures = 1

    def transcribe(self, path):
        self.calls.append(str(path))
        if self.remaining_failures and self.fixture_transcripts[pcm_key(path)] == self.fail_text:
            self.remaining_failures -= 1
            return timed_transcript("Deliberately wrong recognized words for the first attempt.")
        return FixtureVerifier.transcribe(self, path)


class ThreadRecordingEngine(FixtureEngine):
    """Records the thread identity of every synthesis call."""

    def __init__(self, fixture_transcripts):
        super().__init__(fixture_transcripts)
        self.lock = threading.Lock()
        self.model_threads = []

    def synthesize(self, text, voice, path, *, seed, direction=""):
        with self.lock:
            self.model_threads.append(threading.get_ident())
        return FixtureEngine.synthesize(self, text, voice, path, seed=seed, direction=direction)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class PipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = renderer_config(self.root, series_id="pipeline-lifecycle",
                                      checkpoint="unused-tts", asr_model="unused-asr",
                                      max_attempts=3, paragraph_synthesis=False)
        self.config["render_pool"] = {
            "workers": [{"name": "local", "type": "local", "concurrency": 1}],
            "model_revision": DRAMA3_MODEL_REVISION, "scheduler": "legacy",
            "chapter_concurrency": 2,
        }
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.verifier = FixtureVerifier(self.fixtures)

    def pool_for(self, scheduler, engine=None):
        config = deepcopy(self.config)
        config["render_pool"]["scheduler"] = scheduler
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        pool._workers[0]._engine = engine or self.engine
        return pool

    def publish_voices(self, pool):
        store = VoiceStore(self.config["voices"]["directory"], "pipeline-lifecycle")
        for profile in self.bundle["cast"]["voices"]:
            reference = self.root / (profile["id"] + ".wav")
            self.engine._write("Reference words.", reference, 17)
            store.publish(profile["id"], profile, reference, "Reference words.",
                          pool.voice_binding_identity, {"passed": True}, {})

    def render_chapters(self, pool, verifier=None):
        return render(self.bundle, self.config, pool, verifier or self.verifier,
                      notify=lambda _: None, chapters_only=True)

    def accepted_records(self):
        return sorted((self.root / "work" / "clips").glob("*/accepted.json"))

    def clear_chapter_checkpoints(self):
        for path in (self.root / "work" / "completed-chapters").glob("*.json"):
            path.unlink()

    def test_scheduler_mode_switch_reuses_accepted_caches_without_new_generation(self):
        legacy = self.pool_for("legacy")
        pipeline = self.pool_for("pipeline")
        self.assertEqual(pipeline.identity, legacy.identity)
        self.publish_voices(legacy)
        self.render_chapters(legacy)
        first_calls = len(self.engine.speech_calls)
        self.assertGreater(first_calls, 0)

        def accepted_snapshot():
            return {path.parent.name: read_json(path) for path in self.accepted_records()}

        def audio_snapshot(records):
            return {name: file_hash(self.root / "work" / "clips" / name / record["audio_file"])
                    for name, record in records.items()}

        def cache_snapshot():
            return {path.name: read_json(path)
                    for path in (self.root / "work" / "speech-cache").glob("*.json")}

        before = accepted_snapshot()
        audio = audio_snapshot(before)
        indices = cache_snapshot()

        # Clip-level reuse, not the chapter checkpoint: drop completed-chapter
        # state so every chunk must resolve through the accepted speech cache.
        self.clear_chapter_checkpoints()
        self.render_chapters(pipeline)
        self.assertEqual(len(self.engine.speech_calls), first_calls)
        self.clear_chapter_checkpoints()
        self.render_chapters(legacy)
        self.assertEqual(len(self.engine.speech_calls), first_calls)

        after = accepted_snapshot()
        self.assertEqual(after, before)
        self.assertEqual(audio_snapshot(after), audio)
        self.assertEqual(cache_snapshot(), indices)

    def test_identical_chapter_text_shares_one_take_retry_series_and_acceptance(self):
        marker = self.bundle["chapters"][0]["rows"][0]
        first = {"entry": deepcopy(self.bundle["chapters"][0]["entry"]), "rows": [marker]}
        first["entry"].update(title="Marker chapter one",
                              path="scripts/chapter-0001.jsonl")
        second = {"entry": deepcopy(self.bundle["chapters"][0]["entry"]),
                  "rows": [deepcopy(marker)]}
        second["entry"].update(title="Marker chapter two",
                               path="scripts/chapter-0002.jsonl")
        self.bundle["chapters"] = [first, second]
        pool = self.pool_for("pipeline")
        self.publish_voices(pool)
        verifier = FirstAttemptFailsVerifier(self.fixtures, marker["text"])
        self.render_chapters(pool, verifier)

        marker_calls = [call for call in self.engine.speech_calls
                        if call["text"] == marker["text"]]
        self.assertEqual(len(marker_calls), 2)
        self.assertEqual(len({call["seed"] for call in marker_calls}), 2)
        # The second chapter waited on the shared take lock and reused the
        # accepted take: no extra generation, and the failed first attempt is
        # never re-run. The accepted audio may be re-transcribed once per
        # cache-hit validation, so only distinct files and counts bind.
        self.assertEqual(verifier.remaining_failures, 0)
        transcribed = [Path(path).name for path in verifier.calls]
        self.assertEqual(set(transcribed), {"attempt-001.wav", "attempt-002.wav"})
        self.assertEqual(transcribed.count("attempt-001.wav"), 1)
        self.assertGreaterEqual(transcribed.count("attempt-002.wav"), 1)
        accepted = self.accepted_records()
        self.assertEqual(len(accepted), 1)
        attempts = sorted(accepted[0].parent.glob("attempt-*.json"))
        self.assertEqual(len(attempts), 2)
        self.assertIs(read_json(attempts[0])["report"]["passed"], False)
        self.assertIs(read_json(accepted[0])["report"]["passed"], True)

    def test_pipeline_local_workers_share_one_persistent_model_thread(self):
        self.config["render_pool"]["workers"] = [
            {"name": "local", "type": "local", "concurrency": 2}]
        self.config["render_pool"]["scheduler"] = "pipeline"
        pool = RenderPool(deepcopy(self.config))
        self.addCleanup(pool.close)
        engine = ThreadRecordingEngine(self.fixtures)
        pool._workers[0]._engine = engine
        caller = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(caller.shutdown)
        caller_threads = []

        def submit(index):
            caller_threads.append(threading.get_ident())
            return pool.synthesize(f"shared words {index}", {"asset_hash": "fixture"},
                                   str(self.root / f"t{index}.wav"), seed=index)

        futures = [caller.submit(submit, index) for index in range(4)]
        for future in futures:
            future.result(WATCHDOG)
        self.assertEqual(len(engine.model_threads), 4)
        # Every lane worker delegated to the one persistent LocalRuntime
        # thread; no caller thread ran a model call itself.
        self.assertEqual(len(set(engine.model_threads)), 1)
        self.assertGreaterEqual(len(set(caller_threads)), 2)
        self.assertNotIn(engine.model_threads[0], set(caller_threads))


if __name__ == "__main__":
    unittest.main()
