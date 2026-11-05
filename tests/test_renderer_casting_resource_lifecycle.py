"""Casting verification affinity and bounded Gemini worker ownership."""

import importlib
from types import SimpleNamespace
from threading import get_ident
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.casting_batch import run_gemini_batch
import test_renderer_casting_concurrency as concurrency
import test_renderer_gemini_fresh_regeneration as fixtures
import test_renderer_voice_design_integration as voice_fixtures


MLX_TRANSCRIBE = importlib.import_module("mlx_whisper.transcribe")
ModelHolder = MLX_TRANSCRIBE.ModelHolder


class GlobalCacheVerifier(voice_fixtures.TranscriptVerifier):
    def __init__(self, transcripts, accesses):
        super().__init__(transcripts)
        self.accesses = accesses

    def transcribe(self, path):
        model = ModelHolder.get_model("offline-fixture-model", dtype=None)
        self.accesses.append((get_ident(), model.loader_thread, id(model)))
        return super().transcribe(path)


@unittest.skipUnless(fixtures.FFMPEG, "FFmpeg is required for casting integration")
class CastingModelAffinityTests(unittest.TestCase):
    setUp = fixtures.GeminiFreshRegenerationTests.setUp
    session = concurrency.CastingConcurrencyTests.session

    def model_session(self):
        previous = ModelHolder.model, ModelHolder.model_path
        ModelHolder.model = ModelHolder.model_path = None
        self.addCleanup(setattr, ModelHolder, "model", previous[0])
        self.addCleanup(setattr, ModelHolder, "model_path", previous[1])
        loads, accesses, verifiers = [], [], []

        def load_model(path, *, dtype):
            model = SimpleNamespace(loader_thread=get_ident())
            loads.append(model)
            return model

        patcher = patch.object(MLX_TRANSCRIBE, "load_model", side_effect=load_model)
        patcher.start()
        self.addCleanup(patcher.stop)

        def verifier_factory(*_args, **_kwargs):
            verifier = GlobalCacheVerifier(self.transcripts, accesses)
            verifiers.append(verifier)
            return verifier

        session = self.session(concurrency.GeminiOverlapTransport(self, blocked=()), verifier_factory=verifier_factory)
        return session, loads, accesses, verifiers

    def assert_model_stays_on_loading_thread(self, loads, accesses, verifiers):
        self.assertEqual([(caller, loader) for caller, loader, _ in accesses if caller != loader], [])
        self.assertEqual(len(loads), 1)
        self.assertEqual(len({model_id for _, _, model_id in accesses}), 1)
        self.assertEqual(len(verifiers), 1)

    def test_batch_then_single_regeneration_reuses_model_on_its_loading_thread(self):
        session, loads, accesses, verifiers = self.model_session()

        results = session.ensure_ready_many(["voice_mara", "voice_lena"])
        session.regenerate("voice_mara")

        self.assertTrue(all(row["selected"] is not None for row in results))
        self.assert_model_stays_on_loading_thread(loads, accesses, verifiers)

    def test_single_then_batch_reuses_model_on_its_loading_thread(self):
        session, loads, accesses, verifiers = self.model_session()

        session.regenerate("voice_mara")
        results = session.ensure_ready_many(["voice_mara", "voice_lena"])

        self.assertTrue(all(row["selected"] is not None for row in results))
        self.assert_model_stays_on_loading_thread(loads, accesses, verifiers)


class AllocationSession:
    def __init__(self, voice_ids):
        self.config = {
            "casting": {"gemini_concurrency": 4},
            "engine": {},
            "verification": {"model": "offline-fixture", "language": "en"},
        }
        self._engine_factory = lambda _settings: SimpleNamespace(
            identity={"engine": "fixture", "model": "fixture"},
            capabilities={},
        )
        self._verifier_factory = lambda *_args, **_kwargs: SimpleNamespace(identity={"backend": "fixture"})
        self._cleanup_attempted = set()
        self.voice_ids = voice_ids
        self.invalidated = None
        self.verifier = SimpleNamespace(identity={"backend": "fixture"})

    def prepare_models(self, *, generation):
        pass

    def _retry_gemini_cleanup(self):
        pass

    def _invalidate_regeneration_targets(self, voice_ids):
        self.invalidated = tuple(voice_ids)

    def _batch_verifier(self, fatal):
        return self.verifier


class AllocationWorker:
    def __init__(self, session, index):
        self.session = session
        self.index = index
        self.active = 0
        self.maximum_active = 0
        self.closed = False

    def _get_designer(self):
        pass

    def regenerate(self, voice_id):
        if self.session.invalidated != tuple(self.session.voice_ids):
            raise AssertionError("generation started before every target was invalidated")
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            return {"voice_id": voice_id, "worker_index": self.index}
        finally:
            self.active -= 1

    def close(self):
        self.closed = True


class CastingWorkerAllocationTests(unittest.TestCase):
    def test_large_batch_constructs_only_configured_workers_and_completes_every_job(self):
        voice_ids = [f"voice_{index:03d}" for index in range(100)]
        representatives = [{"voice_id": voice_id, "canonical_voice_id": voice_id} for voice_id in voice_ids]
        session = AllocationSession(voice_ids)
        workers = []

        def worker_factory(_lease, _resources):
            worker = AllocationWorker(session, len(workers))
            workers.append(worker)
            return worker

        results = run_gemini_batch(session, representatives, regenerate=True, worker_factory=worker_factory)

        self.assertEqual(session.invalidated, tuple(voice_ids))
        self.assertEqual(len(workers), 4)
        self.assertEqual([
            row["selected"]["voice_id"] if row["selected"] is not None else None
            for row in results
        ], voice_ids)
        self.assertTrue(all(worker.closed for worker in workers))
        self.assertEqual(max(worker.maximum_active for worker in workers), 1)


if __name__ == "__main__":
    unittest.main()
