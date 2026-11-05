"""Chapter history crosses every transport: serve over HTTP, pools and the CUDA engine."""

from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.continuity import ChapterContinuity, compose
from epub_tts_narrator.renderer.engines import EngineError
from test_fish_http import FakeEngine, wav_bytes


def voice(root, name, asset):
    path = Path(root) / f"{name}.wav"
    path.write_bytes(wav_bytes())
    return {"kind": "reference", "voice_id": name, "asset_hash": asset * 64,
            "reference_audio": str(path), "reference_sha256": hashlib.sha256(wav_bytes()).hexdigest(),
            "reference_text": f"{name} reference."}


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class HTTPHistoryTests(unittest.TestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
        from epub_tts_narrator.renderer.server import make_server
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = FakeEngine()
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                             "max_request_bytes": 104857600, "max_pending_requests": 4},
                  "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(config, self.engine, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        def stop():
            self.server.shutdown()
            thread.join()
            self.server.server_close()
        self.addCleanup(stop)
        url = f"http://127.0.0.1:{self.server.server_port}/v1/tts"
        self.client = FishHTTPEngine({"url": url, "name": "other-mac", "api_key": "private-test-key"})

    def test_history_turns_reach_the_worker_and_codes_come_back(self):
        narrator, hero = voice(self.root, "narrator", "a"), voice(self.root, "hero", "b")
        references = [dict(hero, speaker=0), dict(narrator, speaker=1)]
        history = [("<|speaker:1|> Pinned narration.", [[5] * 4] * 10),
                   ("<|speaker:0|> Latest line.", [[6] * 4] * 10)]
        metadata, codes, size = self.client._generate_wire(
            "<|speaker:0|> New line.", references, history, self.root / "take.wav", seed=7)
        call = self.engine.calls[0]
        self.assertEqual(call["text"], "<|speaker:0|> New line.")
        self.assertEqual(call["history"], history)
        self.assertEqual([(text, speaker) for _, text, speaker in call["references"]],
                         [("hero reference.", 0), ("narrator reference.", 1)])
        self.assertEqual((len(codes), size, metadata["history_turns"]), (10, 1024, 2))
        self.assertTrue(metadata["seed_applied"])

    def test_health_check_requires_the_history_protocol(self):
        self.client.check_continuity()
        with patch.object(type(self.client), "_get_json", return_value={"protocol": "fish-inline-reference-wav-v1"}):
            self.client._history_checked = False
            with self.assertRaisesRegex(EngineError, "does not support chapter history"):
                self.client.check_continuity()

    def test_stock_fish_and_sglang_protocols_are_rejected(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
        for protocol in ("fish_native_drama3", "fish_sglang_drama3"):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(EngineError, "chapter history"):
                FishHTTPEngine({"url": "http://127.0.0.1:1/v1/tts", "protocol": protocol})

    def test_server_rejects_history_with_undeclared_speakers(self):
        hero = voice(self.root, "hero", "b")
        with self.assertRaisesRegex(EngineError, "400"):
            self.client._generate_wire("<|speaker:0|> New line.", [dict(hero, speaker=0)],
                                       [("<|speaker:3|> Unknown.", [[1] * 2] * 10)],
                                       self.root / "take.wav", seed=7)
        self.assertEqual(self.engine.calls, [])


class PoolAffinityTests(unittest.TestCase):
    def test_chapter_stays_on_the_worker_of_its_first_take(self):
        from epub_tts_narrator.renderer.pool import _Slots
        workers = [types.SimpleNamespace(settings={"name": "local", "concurrency": 1}),
                   types.SimpleNamespace(settings={"name": "second-mac", "concurrency": 2})]
        slots = _Slots(workers)
        chapter = ChapterContinuity({"engine": "fixture"}, {})
        with chapter.transaction({"kind": "speech", "voice_asset_hash": "a" * 64}):
            first = slots.acquire()
            slots.release(first)
            self.assertEqual(chapter.worker, "second-mac")
            other = ChapterContinuity({"engine": "fixture"}, {})
            other.worker = "local"
            busy = []
            with other.transaction({"kind": "speech", "voice_asset_hash": "a" * 64}):
                busy.append(slots.acquire())
            self.assertEqual(busy[0].settings["name"], "local")
            # The pinned chapter waits for its own worker, never the free one.
            self.assertIs(slots.acquire(), workers[1])
        acquired = []
        waiter = threading.Thread(target=lambda: acquired.append(
            _pinned_acquire(slots, "local")))
        waiter.start()
        waiter.join(0.2)
        self.assertEqual(acquired, [])
        slots.release(busy[0])
        waiter.join(2)
        self.assertEqual([item.settings["name"] for item in acquired], ["local"])


def _pinned_acquire(slots, name):
    chapter = ChapterContinuity({"engine": "fixture"}, {})
    chapter.worker = name
    with chapter.transaction({"kind": "speech", "voice_asset_hash": "a" * 64}):
        return slots.acquire()


class ComposeTests(unittest.TestCase):
    def test_outside_a_chapter_there_is_no_history(self):
        references, history = compose("<|speaker:0|> Preview.", [{"asset_hash": "a" * 64}])
        self.assertEqual((references, history), ([{"asset_hash": "a" * 64, "speaker": 0}], []))

    def test_request_tags_must_name_request_voices(self):
        with self.assertRaisesRegex(ValueError, "speaker tag"):
            compose("<|speaker:1|> Nobody.", [{"asset_hash": "a" * 64}])


class _FakeTensor(list):
    def __init__(self, rows):
        super().__init__(rows)

    def cpu(self):
        return self


@contextmanager
def fake_fish_speech(limit_rows=None):
    """Stand-ins for the pinned Fish library functions the CUDA engine calls."""
    calls = {"prompts": [], "seeds": []}

    class PromptTooLong(ValueError):
        pass

    def build(text, tokenizer, config, texts, tokens, history=()):
        rows = 10 * (1 + len(history))
        if limit_rows is not None and rows > limit_rows:
            raise PromptTooLong("too long")
        calls["prompts"].append({"text": text, "references": list(texts), "history": list(history)})
        return [0] * rows

    class Frames:
        def __len__(self):
            return 3

        def __getitem__(self, key):
            return self

        @property
        def T(self):
            return self

        def contiguous(self):
            return self

        def to(self, device):
            return self

        def __getitem_none__(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[1, 2, 3]] * 10

    def generate(model, prompt, **options):
        calls["sampling"] = options
        return Frames(), "eos"

    numpy = __import__("numpy")
    codec = types.SimpleNamespace(
        sample_rate=44100, device="cuda:0",
        from_indices=lambda codes: [types.SimpleNamespace(squeeze=lambda: types.SimpleNamespace(
            float=lambda: types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
                numpy=lambda: numpy.full(44100, 0.1, dtype=numpy.float32)))))])
    model = types.SimpleNamespace(tokenizer=object(), config=types.SimpleNamespace(
        audio_decoder=types.SimpleNamespace(vocab_size=4096)))
    generate_module = types.SimpleNamespace(
        Drama3DecodeGraphs=types.SimpleNamespace(graph_capacity=lambda model: 10240),
        Drama3PromptTooLongError=PromptTooLong, build_drama3_prompt=build, generate_drama3=generate)
    torch = types.SimpleNamespace(tensor=lambda rows, dtype=None: _FakeTensor(rows), long="long",
                                  inference_mode=lambda: _null())
    modules = {"fish_speech": types.ModuleType("fish_speech"),
               "fish_speech.models": types.ModuleType("fish_speech.models"),
               "fish_speech.models.text2semantic": types.ModuleType("fish_speech.models.text2semantic"),
               "fish_speech.models.text2semantic.fish_omni": types.ModuleType("fish_speech.models.text2semantic.fish_omni"),
               "fish_speech.models.text2semantic.fish_omni.generate": generate_module,
               "fish_speech.utils": types.SimpleNamespace(set_seed=lambda seed: calls["seeds"].append(seed)),
               "torch": torch}
    with patch.dict(sys.modules, modules):
        yield calls, model, codec


@contextmanager
def _null():
    yield


@unittest.skipUnless(importlib.util.find_spec("soundfile") and importlib.util.find_spec("numpy"),
                     "requires renderer audio dependencies")
class FishSpeechCudaEngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        from epub_tts_narrator.renderer.fish_speech_engine import FishSpeechCudaEngine
        self.engine = FishSpeechCudaEngine(self.root)
        self.engine._reference_codes = lambda voice, codec: (voice["reference_text"], f"codes-{voice['voice_id']}")

    def run_wire(self, history, limit_rows=None):
        with fake_fish_speech(limit_rows) as (calls, model, codec):
            self.engine._model, self.engine._codec = model, codec
            narrator = dict(voice(self.root, "narrator", "a"), speaker=1)
            hero = dict(voice(self.root, "hero", "b"), speaker=0)
            result = self.engine._generate_wire("<|speaker:0|> New line.", [hero, narrator], history,
                                                self.root / f"take-{len(calls['seeds'])}-{limit_rows}.wav",
                                                seed=(1 << 32) - 1)
        return calls, result

    def test_history_and_tagged_references_reach_fish_prompt_builder(self):
        history = [("<|speaker:1|> Pinned.", [[1] * 2] * 10), ("<|speaker:0|> Latest.", [[2] * 2] * 10)]
        calls, (metadata, codes, size) = self.run_wire(history)
        prompt = calls["prompts"][0]
        self.assertEqual(prompt["references"], ["<|speaker:0|> hero reference.", "<|speaker:1|> narrator reference."])
        self.assertEqual([text for text, codes in prompt["history"]], ["<|speaker:1|> Pinned.", "<|speaker:0|> Latest."])
        self.assertEqual(calls["seeds"], [((1 << 32) - 1) % (1 << 31)])
        self.assertEqual((calls["sampling"]["temperature"], calls["sampling"]["top_p"], calls["sampling"]["top_k"]),
                         (0.7, 0.7, 30))
        self.assertEqual((metadata["history_turns"], size, len(codes)), (2, 4096, 10))

    def test_prompt_overflow_drops_latest_then_pinned(self):
        history = [("<|speaker:1|> Pinned.", [[1] * 2] * 10), ("<|speaker:0|> Latest.", [[2] * 2] * 10)]
        calls, (metadata, codes, size) = self.run_wire(history, limit_rows=20)
        self.assertEqual([text for text, codes in calls["prompts"][0]["history"]], ["<|speaker:1|> Pinned."])
        self.assertEqual(metadata["history_turns"], 1)
        # The pinned narrator is no longer referenced once its turn is dropped.
        calls, (metadata, codes, size) = self.run_wire(history, limit_rows=10)
        self.assertEqual(calls["prompts"][0]["references"], ["<|speaker:0|> hero reference."])
        self.assertEqual(metadata["history_turns"], 0)

    def test_create_engine_builds_the_cuda_engine_from_config(self):
        from epub_tts_narrator.renderer.engines import create_engine
        from epub_tts_narrator.renderer.fish_speech_engine import FishSpeechCudaEngine
        engine = create_engine({"type": "fish_speech_cuda", "checkpoint": str(self.root), "temperature": 0.7,
                                "top_p": 0.7, "top_k": 30, "max_new_tokens": 2048, "model": "fishaudio/drama-3",
                                "experts": "q8", "options": {}, "cache_model_alias": None})
        self.assertIsInstance(engine, FishSpeechCudaEngine)
        self.assertTrue(engine.capabilities["chapter_continuity"])
        with self.assertRaisesRegex(EngineError, "CUDA device"):
            FishSpeechCudaEngine(self.root, device="cpu")


class DefaultLocalSchedulingTests(unittest.TestCase):
    def test_unconfigured_local_rendering_overlaps_chapters_on_one_model_slot(self):
        from epub_tts_narrator.renderer.pool import RenderPool, create_render_engine
        from epub_tts_narrator.renderer.render_pipeline_config import effective_scheduling
        from shared_fixtures import renderer_config
        with tempfile.TemporaryDirectory() as directory:
            config = renderer_config(Path(directory))
            config["render_pool"]["workers"] = []
            engine = create_render_engine(config)
            self.addCleanup(engine.close)
            self.assertIsInstance(engine, RenderPool)
            self.assertEqual(engine.max_parallel, 1)
            self.assertIsNotNone(engine.logical)
            effective = effective_scheduling({**config, "render_pool": {
                **config["render_pool"], "workers": [{"name": "local", "type": "local", "concurrency": 1}]}})
            self.assertEqual((effective["scheduler"], effective["chapter_concurrency"]), ("pipeline", 4))
            self.assertTrue(engine.capabilities["chapter_continuity"])
