"""Renderer leaf operations route through bounded lanes only under a scheduling scope."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from threading import Event, current_thread, get_ident
import math
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import epub_tts_narrator
from epub_tts_narrator import jev as jev_module
from epub_tts_narrator.jev import JevClient
from epub_tts_narrator.provider import OpenAICompatibleClient
from epub_tts_narrator.renderer import assembly as assembly_module
from epub_tts_narrator.renderer import quality as quality_module
from epub_tts_narrator.renderer.assembly import normalize_clip
from epub_tts_narrator.renderer.chapter_verification import FasterWhisperVerifier
from epub_tts_narrator.renderer.clip_repair_state import ClipRepairState
from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition
from epub_tts_narrator.renderer.quality import _audio_metrics, validate_audio, validate_policy
from epub_tts_narrator.renderer.recovery_client import RecoveryModelClient
from epub_tts_narrator.renderer.render_resource_context import resource_scope
from epub_tts_narrator.renderer.render_resources import RenderResources
import test_renderer_workflow as workflow_fixture

# Import provenance: renderer tests must exercise this worktree's package.
assert Path(epub_tts_narrator.__file__).resolve().is_relative_to(
    Path(__file__).resolve().parents[1])

FFMPEG = workflow_fixture.FFMPEG
SECRET = "fixture-routing-secret-never-echoed"


def pipeline_resources(**overrides):
    values = {"tts_slots": 2, "synthesis_backlog": 4, "review_concurrency": 2,
              "max_inflight": 8}
    values.update(overrides)
    return RenderResources(**values)


def tone_wav(path, seconds=0.3, rate=44100, amplitude=8000):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        frames = bytearray()
        for index in range(int(rate * seconds)):
            frames += struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * index / rate)))
        audio.writeframes(bytes(frames))


def spy_on(module, name):
    """Record the execution thread of the module's direct leaf body."""
    original = getattr(module, name)
    threads = []

    def recording(*args, **kwargs):
        threads.append(get_ident())
        return original(*args, **kwargs)

    return patch.object(module, name, side_effect=recording), threads


def transcript_result(text, end):
    return {"text": text, "segments": [{
        "text": " " + text, "start": 0.0, "end": end, "avg_logprob": -0.1,
        "no_speech_prob": 0.01, "compression_ratio": 1.0,
        "words": [{"word": " " + word, "start": start, "end": stop, "probability": 0.9}
                  for word, start, stop in zip(text.split(), (0.0, 0.15), (0.15, end))]}]}


class AudioMetricsRoutingTests(unittest.TestCase):
    def test_audio_metrics_route_to_the_preparation_lane_under_a_scope(self):
        runtime = pipeline_resources()
        self.addCleanup(runtime.close)
        recording, threads = spy_on(quality_module, "_measure_audio_metrics")
        with recording:
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "tone.wav"
                tone_wav(source)
                with resource_scope(runtime, ("run", 1)):
                    report = _audio_metrics(source, validate_policy(None))
        self.assertIn("duration_seconds", report)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], get_ident())

    def test_audio_metrics_without_a_scope_stay_direct(self):
        recording, threads = spy_on(quality_module, "_measure_audio_metrics")
        with recording:
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "tone.wav"
                tone_wav(source)
                report = _audio_metrics(source, validate_policy(None))
        self.assertIn("duration_seconds", report)
        self.assertEqual(threads, [get_ident()])

    def test_validate_audio_keeps_its_parent_off_the_preparation_lane(self):
        runtime = pipeline_resources()
        self.addCleanup(runtime.close)
        threads = {"metrics": [], "transcribe": []}
        original = getattr(quality_module, "_measure_audio_metrics")

        def measured(path, policy):
            threads["metrics"].append(get_ident())
            return original(path, policy)

        class Recognizer:
            identity = {"backend": "fixture-asr"}

            def transcribe(self, path):
                threads["transcribe"].append(get_ident())
                return transcript_result("spoken words", 0.3)

        with patch.object(quality_module, "_measure_audio_metrics", side_effect=measured):
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "tone.wav"
                tone_wav(source)
                with resource_scope(runtime, ("run", 1)):
                    report = validate_audio(source, "spoken words", Recognizer())
        self.assertTrue(report["passed"], report["reasons"])
        self.assertNotEqual(threads["metrics"][0], get_ident())
        self.assertEqual(threads["transcribe"], [get_ident()])


class NormalizeClipRoutingTests(unittest.TestCase):
    def prepare(self):
        runtime = pipeline_resources()
        self.addCleanup(runtime.close)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        source, target = Path(folder.name) / "raw.wav", Path(folder.name) / "norm.wav"
        tone_wav(source)
        return runtime, source, target

    @unittest.skipUnless(FFMPEG, "FFmpeg required for real normalization leaves")
    def test_normalize_clip_routes_to_the_preparation_lane_under_a_scope(self):
        runtime, source, target = self.prepare()
        recording, threads = spy_on(assembly_module, "_normalize_clip")
        with recording:
            with resource_scope(runtime, ("run", 1)):
                info = normalize_clip(source, target, ffmpeg="ffmpeg")
        self.assertEqual(info["sample_rate"], 44100)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], get_ident())
        self.assertTrue(target.exists())

    @unittest.skipUnless(FFMPEG, "FFmpeg required for real normalization leaves")
    def test_normalize_clip_without_a_scope_stays_direct(self):
        _, source, target = self.prepare()
        recording, threads = spy_on(assembly_module, "_normalize_clip")
        with recording:
            normalize_clip(source, target, ffmpeg="ffmpeg")
        self.assertEqual(threads, [get_ident()])
        self.assertTrue(target.exists())


class FakeJevResponse:
    status = 200

    def read(self, size):
        payload = {"model": "jev-2026-10-21", "answers": {
            "roughly_same": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 1, "output_tokens": 1}}
        return json.dumps(payload).encode()[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeJevOpener:
    def __init__(self, record, gate=None):
        self.record = record
        self.gate = gate

    def open(self, request, timeout):
        self.record["openers"].add(id(self))
        self.record["threads"].append(get_ident())
        if "first_appended" in self.record:
            self.record["first_appended"].set()
        if self.gate is not None and not self.gate.wait(5):
            raise AssertionError("jev leaf was never released")
        return FakeJevResponse()


class JevLeafRoutingTests(unittest.TestCase):
    def client(self):
        return JevClient(SECRET)

    def questions(self):
        from epub_tts_narrator.jev_prompts import QUESTIONS
        return deepcopy(QUESTIONS)

    def test_jev_leaf_request_routes_to_the_review_lane_with_a_fresh_opener(self):
        runtime = pipeline_resources()
        self.addCleanup(runtime.close)
        record = {"openers": set(), "threads": []}
        client = self.client()
        with patch.object(jev_module, "build_opener",
                          side_effect=lambda handler: FakeJevOpener(record)):
            with resource_scope(runtime, ("run", 1)):
                result = client.evaluate({"passage": "a"}, self.questions())
        self.assertEqual(result["answers"]["roughly_same"]["noul"], 0.99)
        self.assertEqual(len(record["openers"]), 1, "one fresh opener per request")
        self.assertNotEqual(record["threads"][0], get_ident())

    def test_second_leaf_waits_for_the_declared_single_review_worker(self):
        runtime = pipeline_resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        record = {"openers": set(), "threads": [], "first_appended": Event()}
        gate, first_started = Event(), Event()
        client = self.client()

        def opener_factory(handler):
            blocked = len(record["threads"]) == 0
            opener = FakeJevOpener(record, gate if blocked else None)
            original_open = opener.open

            def opening(request, timeout):
                if blocked:
                    first_started.set()
                return original_open(request, timeout)
            opener.open = opening
            return opener

        def evaluate_scoped(passage):
            with resource_scope(runtime, ("run", 1)):
                return client.evaluate({"passage": passage}, questions())

        questions = self.questions
        with patch.object(jev_module, "build_opener", side_effect=opener_factory):
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(evaluate_scoped, "a")
                self.assertTrue(first_started.wait(5))
                second = callers.submit(evaluate_scoped, "b")
                # The single declared review worker is provably blocked inside
                # the first leaf, so the second leaf cannot have started.
                self.assertTrue(record["first_appended"].wait(5))
                self.assertEqual(len(record["threads"]), 1)
                gate.set()
                first.result(5)
                second.result(5)
        self.assertEqual(len(record["threads"]), 2)
        self.assertEqual(len(set(record["threads"])), 1, "one serialized review worker")

    def test_jev_without_a_renderer_scope_keeps_the_reusable_shared_opener(self):
        record = {"openers": set(), "threads": []}
        client = self.client()
        client._opener = FakeJevOpener(record)
        with patch.object(jev_module, "build_opener",
                          side_effect=AssertionError("pipeline transport built without a scope")):
            result = client.evaluate({"passage": "a"}, self.questions())
        self.assertEqual(result["answers"]["roughly_same"]["noul"], 0.99)
        self.assertEqual(record["threads"], [get_ident()])
        self.assertEqual(len(record["openers"]), 1)


def completion_response(value, reason="stop"):
    return json.dumps({"choices": [{"finish_reason": reason, "message": {
        "role": "assistant", "content": json.dumps(value)}}]}).encode()


class RecoveryClientRoutingTests(unittest.TestCase):
    def settings(self):
        return {"model": "fixture-routing-model", "base_url": "https://fixture.invalid/v1",
                "max_output_tokens": 65536}

    def test_each_output_budget_attempt_routes_to_the_review_lane(self):
        runtime = pipeline_resources(review_concurrency=2)
        self.addCleanup(runtime.close)
        client = RecoveryModelClient(self.settings())
        threads = []

        def request(_client, encoded, _headers):
            threads.append(get_ident())
            return completion_response({"decision": "ok"})

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            with resource_scope(runtime, ("run", 1)):
                self.assertEqual(client.complete("Choose.", {"decision": "x"}), {"decision": "ok"})
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], get_ident())

    def test_truncation_retries_each_take_a_lane_job_and_keep_budget_order(self):
        runtime = pipeline_resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        client = RecoveryModelClient(self.settings())
        threads = []

        def request(_client, encoded, _headers):
            threads.append(get_ident())
            return completion_response({"decision": "ok"},
                                       reason="length" if len(threads) < 3 else "stop")

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            with resource_scope(runtime, ("run", 1)):
                self.assertEqual(client.complete("Choose.", {"decision": "x"}), {"decision": "ok"})
        self.assertEqual(len(threads), 3)
        self.assertNotIn(get_ident(), threads, "attempts must not run on the caller")

    def test_recovery_client_without_a_scope_stays_direct(self):
        client = RecoveryModelClient(self.settings())
        threads = []

        def request(_client, encoded, _headers):
            threads.append(get_ident())
            return completion_response({"decision": "ok"})

        with patch.object(OpenAICompatibleClient, "_request", autospec=True, side_effect=request):
            self.assertEqual(client.complete("Choose.", {}), {"decision": "ok"})
        self.assertEqual(threads, [get_ident()])


class ChapterAsrRoutingTests(unittest.TestCase):
    def verifier(self, folder):
        model = Path(folder) / "model"
        model.mkdir()
        for name in ("config.json", "model.bin", "tokenizer.json"):
            (model / name).write_bytes(b"fixture")
        return FasterWhisperVerifier(model)

    def test_faster_whisper_transcription_routes_to_the_recognition_lane(self):
        runtime = pipeline_resources()
        self.addCleanup(runtime.close)
        with tempfile.TemporaryDirectory() as folder:
            verifier = self.verifier(folder)
            threads = []
            with patch.object(verifier, "_transcribe",
                              side_effect=lambda path, timed: (
                                  threads.append(get_ident()), {"text": "words"})[1]):
                with resource_scope(runtime, ("run", 1)):
                    self.assertEqual(verifier.transcribe(Path("x")), {"text": "words"})
                    self.assertEqual(verifier.transcribe_reference(Path("x")), {"text": "words"})
            self.assertEqual(len(threads), 2)
            self.assertTrue(all(thread != get_ident() for thread in threads))

    def test_faster_whisper_without_a_scope_stays_direct(self):
        with tempfile.TemporaryDirectory() as folder:
            verifier = self.verifier(folder)
            threads = []
            with patch.object(verifier, "_transcribe",
                              side_effect=lambda path, timed: (
                                  threads.append(get_ident()), {"text": "words"})[1]):
                self.assertEqual(verifier.transcribe(Path("x")), {"text": "words"})
            self.assertEqual(threads, [get_ident()])


PLAN = {"parts": [{"text": "Hum.", "delivery": "thoughtful", "speech_traits": ""}],
        "reason": "Try alternate phonetics."}


class RoutingPlanClient:
    identity = {"fixture": "routing"}

    def __init__(self):
        self.reviews = []

    def complete(self, prompt, payload):
        if payload["mode"] == "clip_repair":
            return deepcopy(PLAN)
        self.reviews.append({"thread": get_ident(), "name": current_thread().name,
                             "reviewer": payload["reviewer"]})
        return {"approve": True, "reason": "Reasonable clip-local experiment."}


class ClipRepairVoteRoutingTests(unittest.TestCase):
    def request(self):
        return {"kind": "speech", "chunk": {"spoken_text": "Hum."}, "voice": {"id": "v"}}

    def prepare(self, client):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return ClipRepairState(folder.name, self.request(), client)

    def test_two_blind_votes_share_the_shared_review_coordination_capacity(self):
        runtime = pipeline_resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        client = RoutingPlanClient()
        state = self.prepare(client)
        with resource_scope(runtime, ("run", 1)):
            plan = state.prepare([{"reason": "mismatch"}], 3)
        self.assertEqual([part["text"] for part in plan["parts"]], ["Hum."])
        self.assertEqual(sorted(review["reviewer"] for review in client.reviews), [1, 2])
        self.assertTrue(all(review["thread"] != get_ident() for review in client.reviews))
        self.assertTrue(all(review["name"].startswith("render-review-coordination")
                            for review in client.reviews))

    def test_vote_leaves_route_to_the_review_lane_not_the_caller_or_coordinator(self):
        runtime = pipeline_resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        leaf_threads = []
        client = RoutingPlanClient()
        original = RoutingPlanClient.complete

        def complete(self, prompt, payload):
            if payload["mode"] == "clip_repair_review":
                leaf_threads.append(current_thread().name)
            return original(self, prompt, payload)

        state = self.prepare(client)
        with patch.object(RoutingPlanClient, "complete", complete):
            with resource_scope(runtime, ("run", 1)):
                state.prepare([{"reason": "mismatch"}], 3)
        self.assertTrue(leaf_threads)
        self.assertTrue(all(name.startswith("render-review-") for name in leaf_threads),
                        leaf_threads)

    def test_queued_votes_do_not_run_after_the_owner_predicate_expired(self):
        runtime = pipeline_resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        client = RoutingPlanClient()
        state = self.prepare(client)
        with scoped_stop_condition(lambda: True, RenderPaused("owner lost")):
            with resource_scope(runtime, ("run", 1)):
                with self.assertRaises(RenderPaused):
                    state.prepare([{"reason": "mismatch"}], 3)
        self.assertEqual(client.reviews, [],
                         "queued votes must not run after the owner predicate expired")

    def test_votes_without_a_scope_keep_the_per_clip_executor(self):
        client = RoutingPlanClient()
        state = self.prepare(client)
        with nullcontext():
            plan = state.prepare([{"reason": "mismatch"}], 3)
        self.assertEqual(plan["parts"][0]["text"], "Hum.")
        self.assertTrue(all(review["name"].startswith("clip-repair-review")
                            for review in client.reviews))
        self.assertEqual(sorted(review["reviewer"] for review in client.reviews), [1, 2])


if __name__ == "__main__":
    unittest.main()
