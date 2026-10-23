"""Bound complete chunk work across chapters without changing speech identity."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.config import DEFAULTS, validate_distributed_settings
from epub_tts_narrator.renderer.interrupts import RenderInterrupted
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import _render_chapter_chunks, verified_take
from test_renderer_speech_identity import reference, speech_request
from epub_tts_narrator.renderer.speech_identity import generation_identity


def configuration(capacity=2, limit=4):
    config = deepcopy(DEFAULTS)
    config["render_pool"].update(max_inflight=limit, workers=[{
        "name": "fixture", "type": "fish_http", "url": "http://fixture.invalid/v1/tts",
        "concurrency": capacity}])
    return config


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def pool(self, capacity=2, limit=4):
        config = configuration(capacity, limit)
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        return config, pool

    def chapter(self, pool, config, number=1, count=4, verifier=None):
        chunks = [{"chunk_id": f"{number}-{i}", "spoken_text": f"Words {i}."} for i in range(count)]
        return _render_chapter_chunks({"number": number, "chunks": chunks}, {}, pool, pool.identity,
                                      verifier, config, self.directory, notify=lambda _: None,
                                      speech_cache=object())

    def test_config_validation_and_automatic_window_preserve_speech_identity(self):
        for value in (True, -1, 1.5, "4", 257, 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_inflight"):
                    validate_distributed_settings(configuration(limit=value))
        for value in (0, 2, 256):
            validate_distributed_settings(configuration(limit=value))
        _, automatic = self.pool(limit=0)
        _, explicit = self.pool(limit=32)
        self.assertEqual(automatic.max_inflight, 3)
        self.assertIsNone(automatic._admission)
        self.assertEqual(explicit.max_inflight, 32)
        self.assertEqual(automatic.identity, explicit.identity)
        voice = reference()
        request = speech_request(voice, engine=automatic.identity)
        other = dict(request, engine=explicit.identity)
        voices = {voice["asset_hash"]: voice}
        self.assertEqual(generation_identity(request, voices), generation_identity(other, voices))

    def test_two_chapters_share_one_bound_before_entering_chunk_work(self):
        config, pool = self.pool()
        guard, release, all_waiting, all_active = Lock(), Event(), Event(), Event()
        active = maximum = attempts = 0
        original_admit = pool.admit_chunk

        @contextmanager
        def admission():
            nonlocal attempts
            with guard:
                attempts += 1
                if attempts == 8:
                    all_waiting.set()
            with original_admit():
                yield

        def render(chunk, *args, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
                if active == 4:
                    all_active.set()
            try:
                if not release.wait(5):
                    raise AssertionError("admitted chunks were never released")
                return chunk, None
            finally:
                with guard:
                    active -= 1

        with patch.object(pool, "admit_chunk", side_effect=admission), \
                patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=render), \
                ThreadPoolExecutor(max_workers=2) as callers:
            futures = [callers.submit(self.chapter, pool, config, number) for number in (1, 2)]
            try:
                self.assertTrue(all_waiting.wait(5))
                self.assertTrue(all_active.wait(5))
                self.assertEqual(active, 4)
                self.assertEqual(maximum, 4)
            finally:
                release.set()
            for number, future in enumerate(futures, 1):
                clips, reports = future.result(5)
                self.assertEqual([clip["chunk_id"] for clip in clips], [f"{number}-{i}" for i in range(4)])
                self.assertEqual(reports, [])
        self.assertEqual(maximum, 4)

    def test_waiting_admission_cancels_without_leaking_or_consuming_a_tts_slot(self):
        _, pool = self.pool(capacity=1, limit=1)
        waiting, cancelled = Event(), Event()
        acquire = pool._admission.acquire

        def wait(*args, **kwargs):
            waiting.set()
            return acquire(*args, **kwargs)

        def check():
            if cancelled.is_set():
                raise RenderInterrupted()

        def blocked():
            with pool.admit_chunk():
                raise AssertionError("cancelled chunk entered")

        with pool.admit_chunk(), patch.object(pool._admission, "acquire", side_effect=wait), \
                patch("epub_tts_narrator.renderer.interrupts.stop_if_requested", side_effect=check), \
                ThreadPoolExecutor(max_workers=1) as callers:
            future = callers.submit(blocked)
            self.assertTrue(waiting.wait(5))
            cancelled.set()
            with self.assertRaises(RenderInterrupted):
                future.result(5)
            self.assertEqual(pool._slots.qsize(), 1)
        # Both cancellation and a failed admitted transaction return the permit.
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with pool.admit_chunk():
                raise RuntimeError("fixture")
        with pool.admit_chunk():
            self.assertEqual(pool._slots.qsize(), 1)

    def test_retrying_take_uses_parent_admission_and_reuses_accepted_bytes(self):
        config, pool = self.pool(capacity=1, limit=1)
        config["render"].update(max_attempts=2, keep_intermediate_audio=True)
        seeds = []

        def synthesize(text, voice, path, *, seed, direction=""):
            seeds.append(seed)
            path.write_bytes(b"fixture waveform")
            return {}

        pool._workers[0]._engine = SimpleNamespace(identity={"engine": "fixture"}, synthesize=synthesize)
        verifier = SimpleNamespace(identity={"backend": "fixture"})
        request = {"kind": "speech", "engine": pool.identity, "text": "Words."}

        def render(chunk, *args, **kwargs):
            wav, accepted = verified_take("Words.", lambda path, seed: pool.synthesize("Words.", {}, path, seed=seed),
                self.directory / "take", request, verifier, config, notify=lambda _: None)
            return dict(chunk, wav_path=str(wav)), accepted

        def normalize(raw, wav, **kwargs):
            wav.write_bytes(raw.read_bytes())

        with patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=render), \
                patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize), \
                patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=[
                    {"passed": False, "reasons": ["text_mismatch"]}, {"passed": True, "reasons": []}]) as verify, \
                ThreadPoolExecutor(max_workers=1) as callers:
            clips, reports = callers.submit(self.chapter, pool, config, count=1).result(5)
            self.assertEqual(len(seeds), 2)
            self.assertNotEqual(*seeds)
            self.assertTrue(reports[0]["report"]["passed"])
            self.assertEqual(callers.submit(self.chapter, pool, config, count=1).result(5), (clips, reports))
            self.assertEqual(len(seeds), 2)
            self.assertEqual(verify.call_count, 2)

    def test_explicit_backlog_keeps_all_tts_slots_busy_during_text_review(self):
        config, pool = self.pool(capacity=2, limit=4)
        guard, review_started, release_review, next_batch = Lock(), Event(), Event(), Event()
        batch_barrier = Barrier(2)
        generated = active = maximum = 0

        def synthesize(*args, **kwargs):
            nonlocal generated, active, maximum
            with guard:
                generated += 1
                current = generated
                active += 1
                maximum = max(maximum, active)
            try:
                if current > 2 and not review_started.wait(5):
                    raise AssertionError("review did not start")
                batch_barrier.wait(5)
                if current == 4:
                    next_batch.set()
                return {}
            finally:
                with guard:
                    active -= 1

        def judge(*args):
            review_started.set()
            if not release_review.wait(5):
                raise AssertionError("review was never released")
            return {"decision": "equivalent"}

        pool._workers[0]._engine = SimpleNamespace(identity={"engine": "fixture"}, synthesize=synthesize)
        reviewer = JevTranscriptJudge({})
        verifier = pool.wrap_verifier(ReferenceTextJudgingVerifier(
            SimpleNamespace(identity={"backend": "fixture"}, transcribe=lambda path: {"text": "Words."}), reviewer))
        self.addCleanup(verifier.close)

        def render(chunk, *args, **kwargs):
            pool.synthesize(chunk["spoken_text"], {}, self.directory / "unused.wav", seed=1)
            verifier.transcribe("unused.wav")
            verifier.judge_reference("Words.", "words")
            return chunk, None

        with patch.object(reviewer, "judge", side_effect=judge), \
                patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=render), \
                ThreadPoolExecutor(max_workers=1) as callers:
            future = callers.submit(self.chapter, pool, config)
            try:
                self.assertTrue(next_batch.wait(5), "TTS stopped behind queued reviews")
                self.assertEqual(maximum, 2)
                self.assertFalse(future.done())
            finally:
                release_review.set()
            self.assertEqual(len(future.result(5)[0]), 4)


if __name__ == "__main__":
    unittest.main()
