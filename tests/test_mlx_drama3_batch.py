"""Batched MLX Drama 3 decoding against the vendored loop, on a tiny random model on the CPU."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import mlx.core as mx

from scripts.check_drama3_mlx_batching import (
    check_forced, check_independence, check_sampler, python_noise, random_prompt, tiny_model)
from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError
from epub_tts_narrator.renderer.mlx_drama3_batch import BatchedMLXBackend, sample_rows, sampling_noise
from epub_tts_narrator.renderer.silence_stop import SilenceWatch, stop_seconds


def u64(values):
    return mx.array(np.array(values, dtype=np.uint64))


class TinyModelTests(unittest.TestCase):
    """No GPU and no model files: the vendored classes with random weights on MLX's CPU device."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        cls.model = tiny_model()
        rng = np.random.default_rng(3)
        cls.prompts = [random_prompt(cls.model, rows, rng) for rows in (40, 23, 61)]

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def vendored_frames(self, max_new=16):
        from drama3_mlx.generate import generate_frames
        frames = []
        for index, prompt in enumerate(self.prompts):
            mx.random.seed(index)
            sampled, _ = generate_frames(self.model, prompt, max_new_tokens=max_new + 3 * index)
            frames.append(sampled if len(sampled) else prompt[-1:])
        return frames

    def test_rows_joining_and_leaving_give_the_vendored_logits(self):
        result = check_forced(self.model, self.prompts, self.vendored_frames(), 1e-4)
        self.assertTrue(result["passed"], result)

    def test_a_request_in_a_batch_gets_exactly_its_own_frames(self):
        result = check_independence(self.model, self.prompts, 12)
        self.assertTrue(result["passed"], result)

    def test_each_row_draws_its_main_token_from_its_own_seed(self):
        # Row independence alone cannot see rows sharing one noise stream, so
        # check one step's main draw against that row's own (seed, step 0) noise.
        backend = BatchedMLXBackend(self.model, rows=3)
        seeds = (11, 2**63 + 3, 4242)
        for slot, seed in enumerate(seeds):
            backend.prefill(slot, SimpleNamespace(prompt=self.prompts[0], max_new_tokens=4, temperature=1.0,
                                                  top_k=30, top_p=0.95, seed=seed))
        state = backend._state
        scores = self.model.backbone.lm_head(state["hidden"])[:, self.model.tower._main_candidates]
        width = scores.shape[-1]
        params = (mx.full((3,), 1.0), mx.full((3,), 30, dtype=mx.int32), mx.full((3,), 0.95))
        first = np.array(self.model.tower._main_candidates[sample_rows(
            scores, *params, sampling_noise(u64(list(seeds)), u64([0, 0, 0]), 0, width))])
        retry = np.array(self.model.tower._main_candidates[sample_rows(
            scores, *params, sampling_noise(u64(list(seeds)), u64([0, 0, 0]), 1, width))])
        history = np.array(state["history"])
        results = backend.step([0, 1, 2])
        for row, (frame, finite) in enumerate(results):
            expected = retry[row] if first[row] in history[row] and first[row] >= self.model.tower.semantic_begin else first[row]
            self.assertEqual(frame[0], int(expected), f"row {row}")
            self.assertTrue(finite)

    def test_a_row_that_does_not_fit_is_refused_without_joining_the_batch(self):
        backend = BatchedMLXBackend(self.model, rows=2)
        request = SimpleNamespace(prompt=self.prompts[0], max_new_tokens=5000, temperature=0.7,
                                  top_k=30, top_p=0.7, seed=1)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            backend.prefill(0, request)
        self.assertEqual(backend.order, [])
        with self.assertRaisesRegex(ValueError, "rows must be"):
            BatchedMLXBackend(self.model, rows=17)

    def test_stepping_a_different_set_of_rows_is_refused(self):
        backend = BatchedMLXBackend(self.model, rows=2)
        backend.prefill(1, SimpleNamespace(prompt=self.prompts[1], max_new_tokens=4, temperature=0.7,
                                           top_k=30, top_p=0.7, seed=1))
        with self.assertRaisesRegex(RuntimeError, "different set of rows"):
            backend.step([0, 1])
        self.assertEqual(len(backend.step([1])), 1)
        backend.release(1)
        self.assertEqual(backend.order, [])


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, self.device)

    def test_noise_is_the_cuda_counter_hash_stream_and_never_zero(self):
        noise = np.array(sampling_noise(u64([7, 2**64 - 1]), u64([0, 9]), 4, 6))
        want = np.array([python_noise(7, 0, 4, 6), python_noise(2**64 - 1, 9, 4, 6)])
        np.testing.assert_allclose(noise, want, rtol=1e-6)
        self.assertTrue((noise > 0).all())
        # Different draws, steps or seeds give different noise.
        self.assertFalse(np.allclose(noise[0], np.array(sampling_noise(u64([7]), u64([0]), 5, 6))[0]))

    def test_distribution_matches_the_vendored_sampler(self):
        result = check_sampler(20000)
        self.assertTrue(result["passed"], result)

    def test_top_k_one_is_the_argmax_and_dropped_tokens_are_never_drawn(self):
        logits = mx.array([[0.1, 3.0, 0.2, 2.9], [5.0, 0.0, 0.0, 0.0]])
        noise = sampling_noise(u64([1, 2]), u64([0, 0]), 0, 4)
        picks = sample_rows(logits, mx.array([0.7, 0.7]), mx.array([1, 30], dtype=mx.int32),
                            mx.array([0.99, 0.5]), noise)
        # Row 0: top-k 1. Row 1: the top token alone crosses top-p and is always kept.
        self.assertEqual(np.array(picks).tolist(), [1, 0])
        for seed in range(50):
            rows = mx.broadcast_to(mx.array([[2.0, 1.9, -9.0, -9.0]]), (1, 4))
            pick = sample_rows(rows, mx.array([1.0]), mx.array([2], dtype=mx.int32), mx.array([1.0]),
                               sampling_noise(u64([seed]), u64([0]), 0, 4))
            self.assertIn(int(np.array(pick)[0]), (0, 1))


class FakeScheduler:
    def __init__(self, frames, finish="eos"):
        self.frames, self.finish, self.calls = frames, finish, []

    def submit(self, prompt, max_new_tokens, **options):
        self.calls.append((prompt, max_new_tokens, options))
        future = Mock()
        future.result.return_value = (self.frames, self.finish)
        return future

    def close(self):
        pass


class BatchedEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.engine = Drama3MLXEngine(Path(temporary.name))

    def test_batch_rows_are_validated_and_change_the_identity(self):
        with patch("epub_tts_narrator.renderer.engines._checkpoint_fingerprints", return_value=({}, None)):
            single = self.engine.identity
            for value in (0, 17, 2.0, True):
                with self.subTest(value=value), self.assertRaisesRegex(EngineError, "batch_rows"):
                    self.engine.set_batch_rows(value)
            self.engine.set_batch_rows(4)
            batched = self.engine.identity
        self.assertEqual(self.engine.concurrent_requests, 4)
        self.assertEqual({key: batched[key] for key in ("decoder", "decoder_version", "batch_rows", "sampler")},
                         {"decoder": "batched", "decoder_version": 1, "batch_rows": 4, "sampler": "counter-hash-v2"})
        self.assertEqual({k: v for k, v in batched.items() if k not in ("decoder", "decoder_version", "batch_rows", "sampler")},
                         single)
        self.engine._model = object()
        with self.assertRaisesRegex(EngineError, "before the model loads"):
            self.engine.set_batch_rows(2)

    def test_a_batched_take_keeps_its_own_seed_sampling_budget_and_silence_watch(self):
        from epub_tts_narrator.renderer.request_sampling import request_sampling
        frames = [[205, 5, 1, 2, 3], [206, 6, 1, 2, 3]]
        self.engine._scheduler = scheduler = FakeScheduler(frames)
        self.engine.batch_rows = 2
        self.engine._mlx = lambda function, *args, **kwargs: function(*args, **kwargs)
        model = SimpleNamespace(sample_rate=44100, codec=SimpleNamespace(decode=lambda codes: np.zeros(4096, np.float32)))
        text = "<|speaker:0|> Wait. [pause for 1 seconds] Now."
        with request_sampling({"temperature": 0.5, "top_k": 20, "top_p": 0.9}):
            segment = self.engine._batched_segment(model, text, np.zeros((30, 5)), 77, 120)
        [(prompt, budget, options)] = scheduler.calls
        self.assertEqual((budget, options["seed"], options["temperature"], options["top_k"], options["top_p"]),
                         (120, 77, 0.5, 20, 0.9))
        self.assertIsInstance(options["watch"], SilenceWatch)
        self.assertEqual(options["watch"].limit, stop_seconds(text))
        np.testing.assert_array_equal(segment.codes, np.array(frames)[:, 1:].T)
        self.assertEqual((segment.finish_reason, segment.text), ("eos", text))
        scheduler.frames = []
        self.assertIsNone(self.engine._batched_segment(model, text, np.zeros((30, 5)), 77, 120))


class BatchedDirectingLayoutTests(unittest.TestCase):
    def test_the_history_layout_marks_its_shared_start_for_the_prefix_cache(self):
        from scripts.check_drama3_mlx_batching import TINY_CONFIG
        from drama3_mlx.config import Drama3Config
        from epub_tts_narrator.renderer.mlx_drama3_batch import prefix_ends

        class CharTokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=[ord(c) % 90 for c in text])

        with tempfile.TemporaryDirectory() as folder:
            engine = Drama3MLXEngine(Path(folder))
        config = Drama3Config.from_dict(TINY_CONFIG)
        engine._scheduler = scheduler = FakeScheduler([[205, 5, 1, 2, 3]])
        engine.batch_rows = 2
        engine._mlx = lambda function, *args, **kwargs: function(*args, **kwargs)
        model = SimpleNamespace(tokenizer=CharTokenizer(), model=SimpleNamespace(config=config), sample_rate=44100,
                                codec=SimpleNamespace(decode=lambda codes: np.zeros(4096, np.float32)))
        reference = SimpleNamespace(text="<|speaker:0|> Saved.", codes=np.ones((4, 6), dtype=np.int64))
        previous = [("<|speaker:0|> First.", np.full((4, 5), 2)), ("<|speaker:0|> Latest.", np.full((4, 3), 3))]
        engine._directing_segment(model, "<|speaker:0|> Now.", [reference], previous, 5)
        [(_prompt, _budget, options)] = scheduler.calls
        self.assertEqual(list(options["prefix_hints"]),
                         prefix_ends(model.tokenizer, config, "<|speaker:0|> Now.", [reference.text], [reference.codes], previous))
        self.assertEqual(len(options["prefix_hints"]), 2)


class LocalBatchingConfigTests(unittest.TestCase):
    def test_a_local_worker_with_concurrency_batches_only_drama3_mlx(self):
        from epub_tts_narrator.renderer.config import DEFAULTS, validate_distributed_settings
        from copy import deepcopy
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local", "concurrency": 4}]
        validate_distributed_settings(config)
        config["engine"]["type"] = "mlx_audio_drama3"
        with self.assertRaisesRegex(ValueError, "needs engine.type = 'drama3_mlx'"):
            validate_distributed_settings(config)

    def test_the_pool_sets_the_local_engines_batch_rows_from_its_concurrency(self):
        from epub_tts_narrator.renderer.pool import _Worker
        engine = Mock(concurrent_requests=8)
        with patch("epub_tts_narrator.renderer.pool.create_engine", return_value=engine):
            worker = _Worker({"name": "local", "type": "local", "concurrency": 8}, {}, {})
            self.assertIs(worker.engine(), engine)
        engine.set_batch_rows.assert_called_once_with(8)
        self.assertTrue(worker.batched)
        # A local engine that does not decode batches stays on the shared model thread.
        unbatched = _Worker({"name": "local", "type": "local", "concurrency": 2}, {}, {})
        unbatched._engine = SimpleNamespace()
        self.assertFalse(unbatched.batched)
        plain = _Worker({"name": "local", "type": "local", "concurrency": 1}, {}, {})
        self.assertFalse(plain.batched)
        with patch("epub_tts_narrator.renderer.pool.create_engine", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(EngineError, "decodes batches"):
                _Worker({"name": "local", "type": "local", "concurrency": 2}, {}, {}).engine()


if __name__ == "__main__":
    unittest.main()


class BatchedLocalPoolTests(unittest.TestCase):
    """A local worker with concurrency N runs N takes at once on a batching engine."""

    def test_a_batched_local_worker_runs_its_slots_concurrently_off_the_model_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        from copy import deepcopy
        from threading import Condition, Event, get_ident
        from epub_tts_narrator.renderer.config import DEFAULTS
        from epub_tts_narrator.renderer.pool import RenderPool

        state = {"condition": Condition(), "active": 0, "peak": 0, "threads": set(), "gate": Event(), "rows": None}

        class BatchingEngine:
            identity = {"engine": "fixture-batching"}
            capabilities = {"multi_speaker": True, "chapter_continuity": True}

            def set_batch_rows(self, rows):
                state["rows"] = rows

            @property
            def concurrent_requests(self):
                return state["rows"]

            def synthesize(self, text, voice, output_path, *, seed, direction=""):
                with state["condition"]:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                    state["threads"].add(get_ident())
                    state["condition"].notify_all()
                if not state["gate"].wait(10):
                    raise AssertionError("engine was never released")
                with state["condition"]:
                    state["active"] -= 1
                return {}

        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local", "concurrency": 3}]
        config["render_pool"]["scheduler"] = "pipeline"
        with patch("epub_tts_narrator.renderer.pool.create_engine", return_value=BatchingEngine()):
            pool = RenderPool(config)
            self.addCleanup(pool.close)
            with ThreadPoolExecutor(max_workers=5) as callers:
                futures = [callers.submit(pool.synthesize, f"words {index}", {}, f"unused-{index}", seed=index)
                           for index in range(5)]
                with state["condition"]:
                    self.assertTrue(state["condition"].wait_for(lambda: state["active"] == 3, 10))
                state["gate"].set()
                for future in futures:
                    future.result(10)
        self.assertEqual((state["rows"], state["peak"]), (3, 3))
        self.assertGreaterEqual(len(state["threads"]), 3)


class PrefixCacheTests(unittest.TestCase):
    """A cached prompt start must give exactly the logits of a full prefill (on the CPU, bit for bit)."""

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        cls.model = tiny_model()
        rng = np.random.default_rng(5)
        shared = random_prompt(cls.model, 300, rng)
        cls.first = np.concatenate([shared, random_prompt(cls.model, 40, rng)])
        cls.second = np.concatenate([shared, random_prompt(cls.model, 55, rng)])
        from drama3_mlx.generate import generate_frames
        mx.random.seed(2)
        cls.frames, _ = generate_frames(cls.model, cls.second, max_new_tokens=12)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def request(self, prompt, hints=()):
        return SimpleNamespace(prompt=prompt, max_new_tokens=12, temperature=0.7, top_k=30, top_p=0.7,
                               seed=1, prefix_hints=hints)

    def forced_logits(self, backend):
        out = []
        for frame in self.frames:
            main, heads = backend.forced_step([0], {0: frame.tolist()})[0]
            out.append((np.array(main), np.array(heads)))
        backend.release(0)
        return out

    def test_a_cached_prompt_start_gives_bit_identical_logits_and_stays_unchanged(self):
        fresh = BatchedMLXBackend(self.model, rows=1)
        fresh.prefill(0, self.request(self.second))
        want = self.forced_logits(fresh)
        cached = BatchedMLXBackend(self.model, rows=1)
        cached.prefill(0, self.request(self.first, hints=(300,)))
        cached.release(0)
        self.assertEqual(cached.prefixes.stats()["stored"], 1)
        for attempt in range(2):  # the second hit proves decoding never altered the snapshot
            with self.subTest(attempt=attempt):
                cached.prefill(0, self.request(self.second, hints=(300,)))
                got = self.forced_logits(cached)
                for (want_main, want_heads), (got_main, got_heads) in zip(want, got):
                    np.testing.assert_array_equal(want_main, got_main)
                    np.testing.assert_array_equal(want_heads, got_heads)
        stats = cached.prefixes.stats()
        self.assertEqual((stats["hits"], stats["rows_reused"], stats["stored"]), (2, 512, 1))

    def test_a_start_cached_at_the_first_chunk_boundary_works_and_is_exact(self):
        # A short reference ends before row 256, so the snapshot is taken at
        # row 128, before the codebook tower has cached anything.
        rng = np.random.default_rng(9)
        shared = random_prompt(self.model, 150, rng)
        first = np.concatenate([shared, random_prompt(self.model, 30, rng)])
        second = np.concatenate([shared, random_prompt(self.model, 45, rng)])
        fresh = BatchedMLXBackend(self.model, rows=1)
        fresh.prefill(0, self.request(second))
        want = [fresh.forced_step([0], {0: frame.tolist()})[0] for frame in self.frames[:6]]
        cached = BatchedMLXBackend(self.model, rows=1)
        cached.prefill(0, self.request(first, hints=(150,)))
        cached.release(0)
        cached.prefill(0, self.request(second, hints=(150,)))
        got = [cached.forced_step([0], {0: frame.tolist()})[0] for frame in self.frames[:6]]
        self.assertEqual((cached.prefixes.stats()["hits"], cached.prefixes.stats()["rows_reused"]), (1, 128))
        for (want_main, want_heads), (got_main, got_heads) in zip(want, got):
            np.testing.assert_array_equal(np.array(want_main), np.array(got_main))
            np.testing.assert_array_equal(np.array(want_heads), np.array(got_heads))

    def test_prompts_with_a_different_start_never_hit(self):
        backend = BatchedMLXBackend(self.model, rows=1)
        backend.prefill(0, self.request(self.first, hints=(300,)))
        backend.release(0)
        other = self.second.copy()
        other[5, 0] = (other[5, 0] + 1) % self.model.config.eos_token_id
        backend.prefill(0, self.request(other, hints=(300,)))
        self.assertEqual(backend.prefixes.stats()["hits"], 0)

    def test_the_cache_evicts_the_least_recently_used_entries(self):
        from epub_tts_narrator.renderer.mlx_drama3_batch import _PrefixCache
        cache = _PrefixCache(entries=2, max_bytes=1 << 40)
        hidden = mx.zeros((2, 2))
        for name in ("a", "b", "c"):
            cache.store(name, {"caches": [], "tower_caches": [], "hidden": hidden})
        self.assertNotIn("a", cache)
        self.assertIn("b", cache)
        self.assertEqual(cache.stats()["entries"], 2)
        small = _PrefixCache(entries=10, max_bytes=hidden.nbytes)
        small.store("a", {"caches": [], "tower_caches": [], "hidden": hidden})
        small.store("b", {"caches": [], "tower_caches": [], "hidden": hidden})
        self.assertEqual((small.stats()["entries"], "a" in small), (1, False))

    def test_prefix_ends_are_where_prompts_with_the_same_references_part(self):
        from drama3_mlx.prompt import build_prompt
        from epub_tts_narrator.renderer.mlx_drama3_batch import prefix_ends

        class CharTokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=[ord(c) % 90 for c in text])

        tokenizer, config = CharTokenizer(), self.model.config
        reference = ("<|speaker:0|> A saved line.", np.ones((config.num_codebooks, 7), dtype=np.int64))
        pinned = ("<|speaker:0|> First.", np.full((config.num_codebooks, 5), 2, dtype=np.int64))
        latest = ("<|speaker:0|> Latest.", np.full((config.num_codebooks, 4), 3, dtype=np.int64))
        text = "<|speaker:0|> xyz"
        ends = prefix_ends(tokenizer, config, text, [reference[0]], [reference[1]], [pinned, latest])
        full = build_prompt(text, tokenizer, config, [reference[0]], [reference[1]], [pinned, latest])
        alone = build_prompt(text, tokenizer, config, [reference[0]], [reference[1]], [])
        with_pinned = build_prompt(text, tokenizer, config, [reference[0]], [reference[1]], [pinned])
        self.assertEqual(len(ends), 2)
        self.assertEqual(prefix_ends(tokenizer, config, text, [reference[0]], [reference[1]], []), ends[:1])
        # Up to each end the prompts agree, whatever follows.
        np.testing.assert_array_equal(full[: ends[0]], alone[: ends[0]])
        np.testing.assert_array_equal(full[: ends[1]], with_pinned[: ends[1]])
        # And each end is exactly where the following part (a turn, or the new line) starts.
        self.assertEqual(len(alone) - ends[0], len(with_pinned) - ends[1])
        self.assertEqual(ends[1] - ends[0], len(with_pinned) - len(alone))


class BatchedServeTests(unittest.TestCase):
    """A drama3_mlx serve worker with batch_rows decodes concurrent controller requests together."""

    def test_two_requests_reach_the_batch_together_with_their_own_seed_sampling_and_prefix_hints(self):
        import copy
        import threading
        import soundfile
        from drama3_mlx.config import Drama3Config
        from drama3_mlx.model import Reference
        from scripts.check_drama3_mlx_batching import TINY_CONFIG
        from test_fish_http import wav_bytes
        from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
        from epub_tts_narrator.renderer.server import make_server

        raw = copy.deepcopy(TINY_CONFIG)
        raw["audio_decoder_config"]["num_codebooks"] = 10  # the wire carries Drama 3's 10 codebooks
        config = Drama3Config.from_dict(raw)

        class CharTokenizer:
            def encode(self, text):
                return SimpleNamespace(ids=[ord(c) % 90 for c in text])

        class TogetherScheduler:
            """Holds each submission until both are inside the batch at once."""

            def __init__(self):
                self.calls, self.together = [], threading.Barrier(2, timeout=10)

            def submit(self, prompt, max_new_tokens, **options):
                self.calls.append((len(prompt), options))
                self.together.wait()
                future = Mock()
                future.result.return_value = ([[205, *range(1, 11)], [206, *range(2, 12)]], "eos")
                future.batch_stats = {"rows": 2, "joined_active": 1, "waited_seconds": 0.0, "steps": 2,
                                      "mean_rows": 2.0, "peak_rows": 2, "prompt_rows": len(prompt),
                                      "cached_rows": 0, "prefill_seconds": 0.01}
                return future

            def close(self):
                pass

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        engine = Drama3MLXEngine(root)
        server_config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "", "max_request_bytes": 104857600,
                                    "max_pending_requests": 4, "batch_rows": 2}, "engine": {"type": "drama3_mlx"}}
        try:
            messages = []
            instance = make_server(server_config, engine, notify=messages.append)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.assertEqual(engine.concurrent_requests, 2)
        engine._scheduler = scheduler = TogetherScheduler()
        engine._mlx = lambda function, *args, **kwargs: function(*args, **kwargs)
        engine._soundfile = soundfile
        engine._model = SimpleNamespace(
            tokenizer=CharTokenizer(), model=SimpleNamespace(config=config), sample_rate=16000,
            codec=SimpleNamespace(decode=lambda codes: np.full(4000, 0.1, np.float32)),
            reference=lambda audio, rate, text: Reference(text=text, codes=np.ones((10, 8), dtype=np.int64)))
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(lambda: (instance.shutdown(), thread.join(), instance.server_close()))
        url = f"http://127.0.0.1:{instance.server_port}/v1/tts"
        reference = root / "reference.wav"
        reference.write_bytes(wav_bytes())
        voice = {"kind": "reference", "reference_audio": str(reference), "reference_text": "The reference words.",
                 "voice_id": "character-1"}
        clients = [FishHTTPEngine({"url": url, "name": f"client-{n}", "api_key": ""},
                                  {"temperature": temperature, "top_p": 0.7, "top_k": 30})
                   for n, temperature in enumerate((0.5, 0.9))]
        results, errors = {}, []

        def speak(client, seed):
            try:
                results[seed] = client.synthesize("A passage.", voice, root / f"take-{seed}.wav", seed=seed)
            except Exception as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=speak, args=(client, seed)) for client, seed in zip(clients, (11, 22))]
        for item in threads:
            item.start()
        for item in threads:
            item.join(20)
        self.assertEqual(errors, [])
        by_seed = {options["seed"]: options for _rows, options in scheduler.calls}
        self.assertEqual((by_seed[11]["temperature"], by_seed[22]["temperature"]), (0.5, 0.9))
        # The prompt's reference part is marked for the worker's prefix cache.
        self.assertTrue(all(options["prefix_hints"] for options in by_seed.values()))
        self.assertTrue(all((root / f"take-{seed}.wav").exists() for seed in (11, 22)))
        # Each completed request's log line says how it shared the batch.
        completed = [line for line in messages if "completed in" in line]
        self.assertEqual(len(completed), 2)
        self.assertTrue(all("; batch: joined 1 running of 2 rows" in line and "2.0 rows on average" in line
                            for line in completed), completed)
