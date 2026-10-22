"""Pool scheduling, local runtime exclusion, and byte-bound resume integration."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import DRAMA3_MODEL_REVISION, EngineError
from epub_tts_narrator.renderer.pool import RenderPool, create_render_engine
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, _render_chapter_chunks, assemble, ensure_voice,
    make_render_plan, render, verified_take,
)
from epub_tts_narrator.cli import export_bundle
from audition_fixtures import prepare_bundle
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier
from test_renderer_paragraph_workflow import ParagraphFixtureEngine


def worker(name, *, local=False, concurrency=1, multi_speaker=True):
    if local:
        return {"name": name, "type": "local", "concurrency": concurrency}
    return {"name": name, "type": "fish_http", "url": f"http://{name}.local:8080/v1/tts",
            "api_key": "private-token", "model": "drama-3", "timeout_seconds": 300,
            "concurrency": concurrency, "multi_speaker": multi_speaker}


def pool_config(*workers):
    config = deepcopy(DEFAULTS)
    config["engine"]["checkpoint"] = "/does/not/exist/drama-3"
    config["render_pool"] = {"workers": list(workers), "model_revision": DRAMA3_MODEL_REVISION}
    return config


class LocalFixtureEngine(FixtureEngine):
    identity = {"engine": "drama3_mlx", "model": "fishaudio/drama-3", "fixture": True}
    capabilities = {**FixtureEngine.capabilities, "multi_speaker": False}


class ContractLocalFixtureEngine(ParagraphFixtureEngine):
    """PCM-only engine carrying an actual local adapter's compatibility inputs."""

    voice_binding_identity = LocalFixtureEngine.identity

    def __init__(self, config, transcripts):
        super().__init__()
        self.fixture_transcripts = transcripts
        contract = RenderPool(config).identity
        self.identity = {
            "engine": "drama3_mlx", "model": contract["model"], "adapter_version": 1,
            "runtime": "Sogni-AI/drama3-mlx", "installed_version": "fixture-version",
            "installed_revision": "fixture-runtime-revision",
            "model_revision": contract["model_revision"],
            "checkpoint": config["engine"]["checkpoint"],
            "checkpoint_files_sha256": {"model.safetensors": "a" * 64},
            "single_chunk": True, "output_encoding": "wav-float32-mono",
            **contract["sampling"],
        }


class QuantizedLocalFixtureEngine(ParagraphFixtureEngine):
    voice_binding_identity = {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    def __init__(self, config, transcripts):
        super().__init__()
        self.fixture_transcripts = transcripts
        self.identity = {
            "engine": "drama3_mlx_audio", "adapter_version": 1,
            "model": "mlx-community/fish-audio-drama-3-4bit", "runtime": "mlx-audio",
            "model_revision": "quantized-fixture-revision", "installed_version": "0.6.1",
            "checkpoint_files_sha256": {"model-00001.safetensors": "c" * 64},
            "stream": False, "chunking": "mlx-audio-fish_drama3",
            "reference_decoder": "mlx_audio.audio_io.read-float32-mono",
            "output_encoding": "wav-float32-mono",
            **{key: config["engine"][key] for key in
               ("temperature", "top_p", "top_k", "max_new_tokens")},
        }


class PoolUnitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_empty_pool_preserves_existing_factory(self):
        config = pool_config()
        sentinel = object()
        with patch("epub_tts_narrator.renderer.pool.create_engine", return_value=sentinel) as create:
            self.assertIs(create_render_engine(config), sentinel)
        create.assert_called_once_with(config["engine"])

    def test_remote_only_pool_never_loads_local_model(self):
        config = pool_config(worker("remote"))
        with patch("epub_tts_narrator.renderer.pool.create_engine", side_effect=AssertionError("local model")), \
             patch("epub_tts_narrator.renderer.fish_http.FishHTTPEngine") as remote:
            remote.return_value.synthesize.return_value = {"seed": 2}
            remote.return_value.identity = {"engine": "remote-fixture", "model": "drama-3"}
            engine = create_render_engine(config)
            remote.assert_not_called()
            result = engine.synthesize("Hello", {}, self.root / "unused.wav", seed=2)
            self.assertEqual(result["render_worker"], "remote")
            remote.assert_called_once()
            with self.assertRaisesRegex(EngineError, "Remote workers reuse saved voices"):
                engine.create_voice("description", "script", self.root / "unused.wav", seed=2)

    def test_identity_excludes_worker_membership_order_urls_keys_and_local_paths(self):
        first = pool_config(worker("local", local=True), worker("remote"))
        second = deepcopy(first)
        second["engine"]["checkpoint"] = "/another/mac/model"
        second["render_pool"]["workers"] = [worker("new", concurrency=4)]
        second["render_pool"]["workers"][0]["api_key"] = "changed-token"
        self.assertEqual(RenderPool(first).identity, RenderPool(second).identity)
        second["engine"]["temperature"] = .5
        self.assertNotEqual(RenderPool(first).identity, RenderPool(second).identity)
        second = deepcopy(first)
        second["render_pool"]["model_revision"] = "another-revision"
        self.assertNotEqual(RenderPool(first).identity, RenderPool(second).identity)
        self.assertNotIn("private-token", repr(RenderPool(first).identity))

    def test_paragraphs_require_support_from_every_worker(self):
        pool = RenderPool(pool_config(worker("local", local=True), worker("remote", multi_speaker=False)))
        self.assertFalse(pool.capabilities["multi_speaker"])
        with self.assertRaisesRegex(EngineError, "Every render worker"):
            pool.synthesize_dialogue([], self.root / "unused.wav", seed=0)

    def test_free_fast_worker_takes_more_work_without_waiting_for_slow_worker(self):
        pool = RenderPool(pool_config(worker("slow"), worker("fast")))
        slow_started, release_slow = Event(), Event()
        calls = []

        class Slow:
            identity = {"engine": "fixture-slow"}
            def synthesize(self, text, *args, **kwargs):
                calls.append(("slow", text))
                slow_started.set()
                if not release_slow.wait(5):
                    raise AssertionError("slow worker was never released")
                return {}

        class Fast:
            identity = {"engine": "fixture-fast"}
            def synthesize(self, text, *args, **kwargs):
                calls.append(("fast", text))
                return {}

        pool._workers[0]._engine, pool._workers[1]._engine = Slow(), Fast()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(pool.synthesize, "first", {}, self.root / "first", seed=1)
            self.assertTrue(slow_started.wait(5))
            try:
                second = executor.submit(pool.synthesize, "second", {}, self.root / "second", seed=1)
                self.assertEqual(second.result(5)["render_worker"], "fast")
                third = executor.submit(pool.synthesize, "third", {}, self.root / "third", seed=1)
                self.assertEqual(third.result(5)["render_worker"], "fast")
            finally:
                release_slow.set()
            self.assertEqual(first.result(5)["render_worker"], "slow")
        self.assertEqual(calls, [("slow", "first"), ("fast", "second"), ("fast", "third")])

    def test_local_synthesis_and_all_optional_verifier_methods_share_one_lock(self):
        pool = RenderPool(pool_config(worker("local", local=True)))
        local_started, release = Event(), Event()
        events = []

        class Local:
            identity = {"engine": "fixture"}
            def synthesize(self, *args, **kwargs):
                local_started.set()
                if not release.wait(5):
                    raise AssertionError("local worker was never released")
                events.append("synthesis")
                return {}

        class Verifier:
            identity = {"backend": "fixture"}
            def transcribe(self, path):
                events.append("asr")
                return {"text": "words"}
            def judge_reference(self, *args):
                events.append("judge")
                return {}

        pool._workers[0]._engine = Local()
        wrapped = pool.wrap_verifier(Verifier())
        self.assertEqual(wrapped.identity, {"backend": "fixture"})
        self.assertFalse(hasattr(wrapped, "corroborate_reference"))
        with ThreadPoolExecutor(max_workers=3) as executor:
            generation = executor.submit(pool.synthesize, "text", {}, self.root / "unused", seed=1)
            self.assertTrue(local_started.wait(5))
            asr = executor.submit(wrapped.transcribe, self.root / "unused")
            judge = executor.submit(wrapped.judge_reference, "words", "words")
            try:
                self.assertEqual(events, [])
            finally:
                release.set()
            generation.result(5)
            asr.result(5)
            judge.result(5)
        self.assertEqual(events[0], "synthesis")
        self.assertEqual(set(events[1:]), {"asr", "judge"})

    def test_scheduler_restores_source_order_with_bounded_inflight_jobs(self):
        engine = SimpleNamespace(max_parallel=2)
        second_done = Event()
        active, maximum = 0, 0
        guard = Lock()
        completion = []

        def fake_chunk(chunk, number, *args, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            if number == 1:
                if not second_done.wait(5):
                    raise AssertionError("second job did not run concurrently")
            else:
                second_done.set()
            with guard:
                active -= 1
                completion.append(number)
            return {"chunk_id": number}, {"chunk_id": number}

        with patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=fake_chunk):
            clips, reports = _render_chapter_chunks({"chunks": [{}] * 5}, {}, engine, {}, None,
                                                    {}, self.root, notify=lambda _: None)
        self.assertEqual([clip["chunk_id"] for clip in clips], [1, 2, 3, 4, 5])
        self.assertEqual([report["chunk_id"] for report in reports], [1, 2, 3, 4, 5])
        self.assertEqual(completion[0], 2)
        self.assertEqual(maximum, 2)

    def test_failure_drains_active_jobs_and_stops_new_dispatch(self):
        engine = SimpleNamespace(max_parallel=2)
        both_started = Barrier(2)
        calls, drained = [], Event()

        def fake_chunk(chunk, number, *args, **kwargs):
            calls.append(number)
            both_started.wait(5)
            if number == 1:
                raise KeyboardInterrupt("fixture cancellation")
            time.sleep(.02)
            drained.set()
            return {}, None

        with patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=fake_chunk):
            with self.assertRaises(KeyboardInterrupt):
                _render_chapter_chunks({"chunks": [{}] * 5}, {}, engine, {}, None,
                                       {}, self.root, notify=lambda _: None)
        self.assertTrue(drained.is_set())
        self.assertEqual(sorted(calls), [1, 2])


class TransportFailure(RuntimeError):
    transport_error = True
    def __init__(self, *, retryable=True):
        super().__init__("fixture transport failure")
        self.retryable = retryable


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class PoolWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = pool_config(worker("first", concurrency=1), worker("second", concurrency=1))
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="pool-fixture")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                     max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.config["quality"] = validate_policy({})
        self.pool = RenderPool(self.config)
        self.fixtures = {}
        self.engines = [ParagraphFixtureEngine(), ParagraphFixtureEngine()]
        for worker_entry, engine in zip(self.pool._workers, self.engines):
            engine.fixture_transcripts = self.fixtures
            worker_entry._engine = engine
        self.verifier = FixtureVerifier(self.fixtures)
        # Existing reference assets remain compatible; no remote voice creation.
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        for profile in self.bundle["cast"]["voices"]:
            reference = self.root / (profile["id"] + ".wav")
            self.engines[0]._write("Reference words.", reference, 17)
            store.publish(profile["id"], profile, reference, "Reference words.",
                          self.pool.voice_binding_identity, {"passed": True}, {})

    def chapters(self, engine=None):
        return render(self.bundle, self.config, engine or self.pool, self.verifier,
                      chapters_only=True, notify=lambda _: None)

    def accepted(self):
        return {path: read_json(path) for path in (self.root / "work" / "clips").glob("*/accepted.json")}

    def test_parallel_paragraphs_verify_serially_preserve_order_and_resume(self):
        # Keep one same-role paragraph to exercise grouped parallel work; the
        # different-role/different-voice paragraph remains separate.
        for row in self.bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        counter, maximum = 0, 0
        guard = Lock()
        original = self.verifier.transcribe

        def transcribe(path):
            nonlocal counter, maximum
            with guard:
                counter += 1
                maximum = max(maximum, counter)
            time.sleep(.005)
            result = original(path)
            with guard:
                counter -= 1
            return result

        with patch.object(self.verifier, "transcribe", side_effect=transcribe):
            first = self.chapters()
        self.assertEqual(maximum, 1)
        self.assertEqual(sum(len(e.voice_calls) for e in self.engines), 0)
        self.assertEqual(sum(len(e.dialogue_calls) for e in self.engines), 1)
        self.assertEqual(sum(len(e.speech_calls) for e in self.engines), 6)
        reports = self.accepted()
        self.assertEqual(len(reports), 7)
        self.assertTrue(all(row["report"]["passed"] for row in reports.values()))
        self.assertTrue(all(row["engine_metadata"]["render_worker"] in {"first", "second"}
                            for row in reports.values()))
        expected = [c["chunk_id"] for c in make_render_plan(self.bundle, self.config)["chapters"][0]["chunks"]
                   if not c["silent_control"]]
        checkpoint = read_json(next((self.root / "work" / "completed-chapters").glob("*.json")))
        self.assertEqual([row["chunk_id"] for row in checkpoint["acceptance"]], expected)
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("reverification")):
            self.assertEqual(self.chapters()["chapters"], first["chapters"])
            final = assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertEqual(final["status"], "complete")

    def test_one_extra_inflight_job_keeps_remote_busy_while_verifier_waits(self):
        self.config["render"]["paragraph_synthesis"] = False
        self.config["render_pool"]["workers"] = [worker("single")]
        single = RenderPool(self.config)
        single._workers[0]._engine = self.engines[0]
        self.assertEqual(single.max_parallel, 1)
        self.assertEqual(single.max_inflight, 2)
        plan = make_render_plan(self.bundle, self.config)
        chunks = [chunk for chunk in plan["chapters"][0]["chunks"] if not chunk["silent_control"]][:2]
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        generated_second = Event()
        original_generation, original_asr = self.engines[0].synthesize, self.verifier.transcribe
        generation_calls, recognition_calls = [], []

        def generation(*args, **kwargs):
            generation_calls.append(args[0])
            result = original_generation(*args, **kwargs)
            if len(generation_calls) == 2:
                generated_second.set()
            return result

        def recognize(path):
            recognition_calls.append(path)
            if len(recognition_calls) == 1 and not generated_second.wait(5):
                raise AssertionError("remote synthesis stalled behind verification")
            return original_asr(path)

        with patch.object(self.engines[0], "synthesize", side_effect=generation), \
             patch.object(self.verifier, "transcribe", side_effect=recognize):
            clips, reports = _render_chapter_chunks({"chunks": chunks}, voices, single,
                single.identity, single.wrap_verifier(self.verifier), self.config,
                self.root / "work", notify=lambda _: None)
        self.assertEqual(len(clips), 2)
        self.assertTrue(all(report["report"]["passed"] for report in reports))
        self.assertEqual(len(generation_calls), 2)

    def test_endpoint_membership_and_model_location_changes_keep_completed_chapter(self):
        first = self.chapters()
        self.config["render_pool"]["workers"] = [worker("replacement", concurrency=3)]
        self.config["render_pool"]["workers"][0]["api_key"] = "new-token"
        self.config["engine"]["checkpoint"] = "/another/mac/checkpoint"
        replacement = RenderPool(self.config)
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("resynthesized")), \
             patch("epub_tts_narrator.renderer.pool._Worker.engine", side_effect=AssertionError("loaded worker")):
            self.assertEqual(self.chapters(replacement)["chapters"], first["chapters"])
            self.assertEqual(assemble(self.bundle, self.config, notify=lambda _: None)["status"], "complete")

    def test_changed_pool_sampling_or_revision_requires_rerender_for_assembly(self):
        self.chapters()
        for section, key, value in (("engine", "temperature", .3),
                                    ("render_pool", "model_revision", "new-revision")):
            old = self.config[section][key]
            self.config[section][key] = value
            with self.assertRaisesRegex(RenderBlocked, "is stale"):
                assemble(self.bundle, self.config, notify=lambda _: None)
            self.config[section][key] = old

    def test_local_pool_local_switch_preserves_completed_book_without_loading_workers(self):
        local = ContractLocalFixtureEngine(self.config, self.fixtures)
        remote_workers = deepcopy(self.config["render_pool"]["workers"])
        self.config["render_pool"]["workers"] = []
        first = render(self.bundle, self.config, local, self.verifier, notify=lambda _: None)
        published = {path: file_hash(path) for path in (self.root / "output").rglob("*")
                     if path.is_file()}
        calls = len(local.speech_calls), len(local.dialogue_calls), len(self.verifier.calls)

        self.config["render_pool"]["workers"] = remote_workers
        remote_only = RenderPool(self.config)
        with patch("epub_tts_narrator.renderer.pool._Worker.engine", side_effect=AssertionError("loaded worker")), \
             patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("reverified speech")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=AssertionError("repackaged audiobook")):
            pooled = render(self.bundle, self.config, remote_only, self.verifier, notify=lambda _: None)
            self.config["render_pool"]["workers"] = []
            returned = render(self.bundle, self.config, local, self.verifier, notify=lambda _: None)

        self.assertEqual(first["audio"]["chapters"], pooled["audio"]["chapters"])
        self.assertEqual(first["audio"]["chapters"], returned["audio"]["chapters"])
        self.assertEqual(calls, (len(local.speech_calls), len(local.dialogue_calls), len(self.verifier.calls)))
        self.assertEqual(published, {path: file_hash(path) for path in (self.root / "output").rglob("*")
                                     if path.is_file()})

    def test_local_and_pool_reuse_indexed_and_legacy_successes_in_both_directions(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        local = ContractLocalFixtureEngine(self.config, self.fixtures)

        for origin in ("local", "pool"):
            for indexed in (False, True):
                with self.subTest(origin=origin, indexed=indexed):
                    work = self.root / f"switch-{origin}-{indexed}"
                    start, destination = (local, self.pool) if origin == "local" else (self.pool, local)
                    first = _render_chapter_chunks({"chunks": [chunk]}, voices, start,
                        start.identity, self.verifier, self.config, work, notify=lambda _: None)
                    if not indexed:
                        for path in (work / "speech-cache").glob("*.json"):
                            path.unlink()
                    count = len(self.verifier.calls)

                    with patch.object(destination, "synthesize", side_effect=AssertionError("regenerated compatible speech")):
                        resumed = _render_chapter_chunks({"chunks": [chunk]}, voices, destination,
                            destination.identity, self.verifier, self.config, work, notify=lambda _: None)

                    self.assertEqual(resumed, first)
                    self.assertEqual(len(self.verifier.calls), count)

    def test_pool_model_revision_and_sampling_changes_do_not_reuse_local_speech(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        local = ContractLocalFixtureEngine(self.config, self.fixtures)

        for section, key, value in (("engine", "temperature", .3),
                                   ("render_pool", "model_revision", "new-model-revision")):
            with self.subTest(change=key):
                work = self.root / f"changed-{key}"
                first = _render_chapter_chunks({"chunks": [chunk]}, voices, local,
                    local.identity, self.verifier, self.config, work, notify=lambda _: None)
                config = deepcopy(self.config)
                config[section][key] = value
                changed = RenderPool(config)
                for entry, engine in zip(changed._workers, self.engines):
                    entry._engine = engine
                count = sum(len(engine.speech_calls) for engine in self.engines)

                second = _render_chapter_chunks({"chunks": [chunk]}, voices, changed,
                    changed.identity, self.verifier, config, work, notify=lambda _: None)

                self.assertEqual(sum(len(engine.speech_calls) for engine in self.engines), count + 1)
                self.assertNotEqual(first[0][0]["wav_path"], second[0][0]["wav_path"])
                self.assertTrue(second[1][0]["report"]["passed"])

    def test_pool_local_worker_reuse_ignores_runtime_but_keeps_weight_invalidation(self):
        self.config["render"]["paragraph_synthesis"] = False
        for change, value in (("checkpoint_files_sha256", {"model.safetensors": "b" * 64}),
                              ("installed_revision", "different-runtime-revision")):
            with self.subTest(change=change):
                local = ContractLocalFixtureEngine(self.config, self.fixtures)
                for engine in self.engines:
                    engine.identity = deepcopy(local.identity)
                self.config["render"]["work_dir"] = str(self.root / change)
                self.chapters()
                local.identity[change] = value
                self.config["render_pool"]["workers"] = []

                self.chapters(local)

                if change == "installed_revision":
                    self.assertEqual(len(local.speech_calls), 0)
                else:
                    self.assertGreater(len(local.speech_calls), 0)
                self.config["render_pool"]["workers"] = [worker("first"), worker("second")]

    def test_explicit_model_alias_preserves_completed_book_across_quantized_runtime(self):
        local = ContractLocalFixtureEngine(self.config, self.fixtures)
        self.config["render_pool"]["workers"] = []
        first = render(self.bundle, self.config, local, self.verifier, notify=lambda _: None)
        published = {path: file_hash(path) for path in (self.root / "output").rglob("*")
                     if path.is_file()}
        config = deepcopy(self.config)
        config["engine"].update(model="mlx-community/fish-audio-drama-3-4bit",
                                cache_model_alias="fishaudio/drama-3")
        quantized = QuantizedLocalFixtureEngine(config, self.fixtures)

        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("reverified speech")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=AssertionError("repackaged audiobook")):
            assembled = assemble(self.bundle, config, notify=lambda _: None)
            resumed = render(self.bundle, config, quantized, self.verifier, notify=lambda _: None)

        self.assertEqual(first["audio"]["chapters"], assembled["audio"]["chapters"])
        self.assertEqual(first["audio"]["chapters"], resumed["audio"]["chapters"])
        self.assertEqual(quantized.speech_calls, [])
        self.assertEqual(quantized.dialogue_calls, [])
        self.assertEqual(published, {path: file_hash(path) for path in (self.root / "output").rglob("*")
                                     if path.is_file()})
        config["engine"]["temperature"] = .3
        with self.assertRaisesRegex(RenderBlocked, "is stale"):
            assemble(self.bundle, config, notify=lambda _: None)

    def test_only_alias_matching_original_model_reuses_its_verified_speech(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        local = ContractLocalFixtureEngine(self.config, self.fixtures)

        for number, alias in enumerate((None, "some-other-model", "fishaudio/drama-3")):
            with self.subTest(alias=alias):
                work = self.root / f"alias-{number}"
                first = _render_chapter_chunks({"chunks": [chunk]}, voices, local,
                    local.identity, self.verifier, self.config, work, notify=lambda _: None)
                config = deepcopy(self.config)
                config["engine"]["model"] = "mlx-community/fish-audio-drama-3-4bit"
                if alias is not None:
                    config["engine"]["cache_model_alias"] = alias
                quantized = QuantizedLocalFixtureEngine(config, self.fixtures)

                second = _render_chapter_chunks({"chunks": [chunk]}, voices, quantized,
                    quantized.identity, self.verifier, config, work, notify=lambda _: None)

                if alias == "fishaudio/drama-3":
                    self.assertEqual(quantized.speech_calls, [])
                    self.assertEqual(first, second)
                else:
                    self.assertEqual(len(quantized.speech_calls), 1)
                    self.assertNotEqual(first[0][0]["wav_path"], second[0][0]["wav_path"])

    def test_first_assembly_with_model_alias_preserves_actual_saved_generation_provenance(self):
        local = ContractLocalFixtureEngine(self.config, self.fixtures)
        self.config["render_pool"]["workers"] = []
        self.chapters(local)
        self.assertFalse((self.root / "output" / "render.json").exists())
        config = deepcopy(self.config)
        config["engine"].update(model="mlx-community/fish-audio-drama-3-4bit",
                                cache_model_alias="fishaudio/drama-3")
        count = len(self.verifier.calls)

        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("reverified speech")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("reencoded chapter")):
            result = assemble(self.bundle, config, notify=lambda _: None)

        published = read_json(self.root / "output" / "render.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(published["engine"], local.identity)
        self.assertEqual(len(self.verifier.calls), count)
        self.assertTrue((self.root / "output" / "book.m4b").is_file())

    def test_model_alias_does_not_reuse_changed_sampling_text_or_reference(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        local = ContractLocalFixtureEngine(self.config, self.fixtures)

        for change in ("sampling", "text", "reference"):
            with self.subTest(change=change):
                work = self.root / f"alias-change-{change}"
                first = _render_chapter_chunks({"chunks": [chunk]}, voices, local,
                    local.identity, self.verifier, self.config, work, notify=lambda _: None)
                config = deepcopy(self.config)
                config["engine"].update(model="mlx-community/fish-audio-drama-3-4bit",
                                        cache_model_alias="fishaudio/drama-3")
                changed_chunk, changed_voices = deepcopy(chunk), dict(voices)
                if change == "sampling":
                    config["engine"]["temperature"] = .3
                elif change == "text":
                    changed_chunk["spoken_text"] += " More words."
                    changed_chunk["segment"]["text"] += " More words."
                    changed_chunk["segment"]["end"] += len(" More words.")
                    changed_chunk["source_ranges"][-1]["end"] += len(" More words.")
                else:
                    voice_id = chunk["segment"]["voice_id"]
                    replacement_store = VoiceStore(self.root / "replacement-voices", "pool-fixture")
                    reference = self.root / "replacement-reference.wav"
                    local._write("A new voice reference.", reference, 237)
                    replacement_store.publish(voice_id, voices[voice_id]["profile"], reference,
                        "A new voice reference.", local.voice_binding_identity, {"passed": True}, {})
                    changed_voices[voice_id] = replacement_store.load(voice_id)
                quantized = QuantizedLocalFixtureEngine(config, self.fixtures)

                second = _render_chapter_chunks({"chunks": [changed_chunk]}, changed_voices,
                    quantized, quantized.identity, self.verifier, config, work, notify=lambda _: None)

                self.assertEqual(len(quantized.speech_calls), 1)
                self.assertNotEqual(first[0][0]["wav_path"], second[0][0]["wav_path"])
                self.assertTrue(second[1][0]["report"]["passed"])

    def test_transient_generation_errors_resume_after_repair(self):
        request = {"kind": "fixture", "engine": self.pool.identity}
        folder = self.root / "transient"
        fail = SimpleNamespace(calls=0)

        def generate(path, seed):
            fail.calls += 1
            raise TransportFailure()

        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            verified_take("Hello there.", generate, folder, request, self.verifier,
                          self.config, notify=lambda _: None)
        self.assertEqual(fail.calls, 2)
        self.assertTrue(read_json(folder / "attempt-001.json")["report"]["retry_on_resume"])
        wav, accepted = verified_take("Hello there.",
            lambda path, seed: self.engines[0]._write("Hello there.", path, seed),
            folder, request, self.verifier, self.config, notify=lambda _: None)
        self.assertTrue(wav.is_file())
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["attempt"], 1)

    def test_permanent_http_error_stops_immediately_but_credentials_can_be_repaired(self):
        request, folder = {"kind": "fixture"}, self.root / "credentials"
        with patch.object(self.engines[0], "_write", side_effect=TransportFailure(retryable=False)) as generate:
            with self.assertRaisesRegex(RenderBlocked, "fixture transport failure"):
                verified_take("Hello.", generate, folder, request, self.verifier,
                              self.config, notify=lambda _: None)
            self.assertEqual(generate.call_count, 1)
        wav, accepted = verified_take("Hello.",
            lambda path, seed: self.engines[0]._write("Hello.", path, seed),
            folder, request, self.verifier, self.config, notify=lambda _: None)
        self.assertTrue(accepted["report"]["passed"])

    def test_duplicate_request_is_generated_only_once(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        clips, reports = _render_chapter_chunks({"chunks": [chunk, chunk]}, voices, self.pool,
            self.pool.identity, self.pool.wrap_verifier(self.verifier), self.config,
            self.root / "work", notify=lambda _: None)
        self.assertEqual(sum(len(e.speech_calls) for e in self.engines), 1)
        self.assertEqual(clips[0]["wav_path"], clips[1]["wav_path"])
        self.assertEqual(reports[0], reports[1])

    def test_concurrent_identical_speech_at_different_positions_keeps_both_current_mappings(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        original = deepcopy(next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"]))
        moved = deepcopy(original)
        moved["chunk_id"] = "moved-chunk"
        moved["segment"].update(id="moved-segment", block_id="moved-block", sequence=90)
        moved["segment"]["start"] += 100
        moved["segment"]["end"] += 100
        moved["source_ranges"] = [{"start": item["start"] + 100, "end": item["end"] + 100}
                                   for item in moved["source_ranges"]]
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}

        clips, reports = _render_chapter_chunks({"chunks": [original, moved]}, voices, self.pool,
            self.pool.identity, self.pool.wrap_verifier(self.verifier), self.config,
            self.root / "work", notify=lambda _: None)

        self.assertGreater(self.pool.max_parallel, 1)
        self.assertEqual(sum(len(engine.speech_calls) for engine in self.engines), 1)
        self.assertEqual(len(self.verifier.calls), 1)
        self.assertEqual(clips[0]["wav_path"], clips[1]["wav_path"])
        self.assertEqual([report["chunk_id"] for report in reports], [original["chunk_id"], "moved-chunk"])
        self.assertEqual(reports[0]["request_hash"], reports[1]["request_hash"])
        for expected, clip in zip((original, moved), clips):
            self.assertEqual(clip["segment"], expected["segment"])
            self.assertEqual(clip["source_ranges"], expected["source_ranges"])
        self.assertEqual(len(self.accepted()), 1)

    def test_simultaneous_lanes_with_identical_reference_content_share_verified_audio_and_resume(self):
        self.config["render"]["paragraph_synthesis"] = False
        plan = make_render_plan(self.bundle, self.config)
        chunk = deepcopy(next(c for c in plan["chapters"][0]["chunks"] if not c["silent_control"]))
        chunk["segment"].update(role_id=None, voice_id=None, simultaneous_speakers=[
            {"role_id": "mara_dialogue", "voice_id": "voice_mara"},
            {"role_id": "lena_dialogue", "voice_id": "voice_lena"},
        ])
        store = VoiceStore(self.config["voices"]["directory"], "pool-fixture")
        voices = {key: store.load(key) for key in plan["voice_ids"]}
        args = ({"chunks": [chunk]}, voices, self.pool, self.pool.identity,
                self.pool.wrap_verifier(self.verifier), self.config, self.root / "work")
        clips, reports = _render_chapter_chunks(*args, notify=lambda _: None)
        self.assertEqual(sum(len(e.speech_calls) for e in self.engines), 1)
        lanes = reports[0]["speaker_lanes"]
        self.assertEqual([lane["voice_id"] for lane in lanes], ["voice_mara", "voice_lena"])
        self.assertEqual([lane["role_id"] for lane in lanes], ["mara_dialogue", "lena_dialogue"])
        self.assertTrue(all(lane["report"]["passed"] for lane in lanes))
        self.assertEqual(lanes[0]["request_hash"], lanes[1]["request_hash"])
        self.assertEqual(lanes[0]["audio_sha256"], lanes[1]["audio_sha256"])
        self.assertEqual(reports[0]["report"]["method"], "independently_verified_simultaneous_lanes")
        with patch.object(self.pool, "synthesize", side_effect=AssertionError("regenerated lanes")):
            resumed = _render_chapter_chunks(*args, notify=lambda _: None)
        self.assertEqual(resumed, (clips, reports))

    def test_missing_reference_created_once_locally_keeps_ordinary_voice_binding(self):
        config = deepcopy(self.config)
        config["voice_design"]["type"] = "engine"
        config["render_pool"]["workers"] = [worker("local", local=True)]
        pool = RenderPool(config)
        local = LocalFixtureEngine(self.fixtures)
        pool._workers[0]._engine = local
        store = VoiceStore(self.root / "empty-voices", "new-pool-fixture")
        profile = self.bundle["cast"]["voices"][0]
        lock = store.lock()
        try:
            first = ensure_voice(store, profile["id"], profile, pool, pool.wrap_verifier(self.verifier),
                                 config, self.root / "new-voices-work", notify=lambda _: None)
            self.assertEqual(first["created_with"], pool.voice_binding_identity)
            self.assertEqual(len(local.voice_calls), 1)
            second = ensure_voice(store, profile["id"], profile, pool, pool.wrap_verifier(self.verifier),
                                  config, self.root / "new-voices-work", notify=lambda _: None)
            self.assertEqual(first, second)
            self.assertEqual(len(local.voice_calls), 1)
        finally:
            lock.rmdir()

    def test_failure_keeps_book_locks_until_other_work_finishes_and_resumes(self):
        self.config["render"]["paragraph_synthesis"] = False
        self.pool.max_inflight = 2  # Keep this failure race to exactly two active jobs.
        both_started = Barrier(2)
        checked = []
        original = self.pool.synthesize
        first_text = make_render_plan(self.bundle, self.config)["chapters"][0]["chunks"][0]["spoken_text"]
        calls = []

        def generation(text, *args, **kwargs):
            calls.append(text)
            both_started.wait(5)
            if text == first_text:
                raise KeyboardInterrupt("fixture stop")
            time.sleep(.03)
            checked.append(((self.root / "bundle" / ".audio-render.lock").is_dir(),
                            (self.root / "work" / ".render.lock").is_dir()))
            return original(text, *args, **kwargs)

        with patch.object(self.pool, "synthesize", side_effect=generation):
            with self.assertRaises(KeyboardInterrupt):
                self.chapters()
        self.assertEqual(len(calls), 2)
        self.assertEqual(checked, [(True, True)])
        self.assertFalse((self.root / "bundle" / ".audio-render.lock").exists())
        self.assertFalse((self.root / "work" / ".render.lock").exists())
        before = self.accepted()
        self.assertEqual(len(before), 1)
        self.chapters()
        after = self.accepted()
        for path, record in before.items():
            self.assertEqual(after[path], record)


if __name__ == "__main__":
    unittest.main()
