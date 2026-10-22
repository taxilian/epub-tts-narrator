"""Model selection across factories, saved voices, pools and the HTTP boundary."""

import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.drama3_models import DEFAULT_DRAMA3_MODEL, DRAMA3_MODELS
from epub_tts_narrator.renderer.engines import Drama3MLXEngine, create_engine
from epub_tts_narrator.renderer.local_runtime import requires_mlx
from epub_tts_narrator.renderer.mlx_audio_engine import MLXAudioDrama3Engine
from epub_tts_narrator.renderer.pool import create_render_engine
from epub_tts_narrator.renderer.server import make_server
from epub_tts_narrator.renderer.voice_bindings import require_designer
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import (
    RenderBlocked, _chapter_context, ensure_voice, make_render_plan,
)
from audition_fixtures import prepare_bundle
from test_fish_http import FakeEngine, wav_bytes
from test_renderer_workflow import EXAMPLE


QUANTIZED = [model.repo for model in DRAMA3_MODELS.values() if model.runtime == "mlx_audio"]
FOUR_BIT = "mlx-community/fish-audio-drama-3-4bit"


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "quantized-model"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps({"model_type": "fish_drama3"}))
        self.original_checkpoint = self.root / "original-model"
        self.original_checkpoint.mkdir()
        (self.original_checkpoint / "config.json").write_text(json.dumps({"model_type": "fish_omni"}))

    def engine_config(self, model=FOUR_BIT):
        settings = deepcopy(DEFAULTS["engine"])
        settings["model"] = model
        settings["checkpoint"] = str(self.original_checkpoint if model == DEFAULT_DRAMA3_MODEL
                                     else self.checkpoint)
        return settings

    def test_omitted_or_explicit_original_model_keeps_sogni_voice_design(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                settings = self.engine_config(DEFAULT_DRAMA3_MODEL)
                if not explicit:
                    settings.pop("model")

                engine = create_engine(settings)
                self.addCleanup(engine.close)

                self.assertIsInstance(engine, Drama3MLXEngine)
                self.assertEqual(engine.identity["model"], DEFAULT_DRAMA3_MODEL)
                self.assertIs(require_designer(engine), engine)

    def test_each_quantized_selection_routes_to_reference_conditioned_mlx_audio(self):
        for model in QUANTIZED:
            with self.subTest(model=model):
                # The legacy experts setting must not override a converted checkpoint.
                settings = self.engine_config(model)
                settings["experts"] = "q8"

                engine = create_engine(settings)
                self.addCleanup(engine.close)

                self.assertIsInstance(engine, MLXAudioDrama3Engine)
                self.assertEqual(engine.identity["model"], model)
                self.assertTrue(engine.capabilities["reference_voice"])
                with self.assertRaisesRegex(ValueError, "voice designer|reference recording"):
                    require_designer(engine)

    def test_both_real_adapters_require_mlx_but_identity_shaped_fakes_do_not(self):
        for model in (DEFAULT_DRAMA3_MODEL, FOUR_BIT):
            with self.subTest(model=model):
                engine = create_engine(self.engine_config(model))
                self.addCleanup(engine.close)

                self.assertTrue(requires_mlx(engine))
                self.assertFalse(requires_mlx(SimpleNamespace(identity=engine.identity)))

    def test_switching_to_quantized_model_reuses_the_original_immutable_reference(self):
        store = VoiceStore(self.root / "voices", "same-series")
        source = self.root / "reference.wav"
        source.write_bytes(wav_bytes())
        original = create_engine(self.engine_config(DEFAULT_DRAMA3_MODEL))
        self.addCleanup(original.close)
        profile = {"description": "Calm storyteller."}
        lock = store.lock()
        self.addCleanup(lock.rmdir)
        saved = store.publish("narrator", profile, source, "Reference words.", original.identity,
                              {"passed": True}, {"description_applied": True})
        before = {path.relative_to(store.directory): path.read_bytes()
                  for path in store.directory.rglob("*") if path.is_file()}

        for model in QUANTIZED:
            with self.subTest(model=model):
                engine = create_engine(self.engine_config(model))
                self.addCleanup(engine.close)
                # No verifier or designer is needed to reuse an accepted reference.
                reused = ensure_voice(store, "narrator", profile, engine, None, {},
                                      self.root / "work", notify=lambda _: None)

                self.assertEqual(reused["asset_hash"], saved["asset_hash"])
                self.assertEqual(reused["reference_audio"], saved["reference_audio"])
                self.assertEqual(reused["reference_text"], "Reference words.")
                self.assertEqual(Path(reused["reference_audio"]).read_bytes(), wav_bytes())
        after = {path.relative_to(store.directory): path.read_bytes()
                 for path in store.directory.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_quantized_model_without_a_saved_voice_requires_a_designer(self):
        store = VoiceStore(self.root / "voices", "same-series")
        engine = create_engine(self.engine_config())
        self.addCleanup(engine.close)
        config = deepcopy(DEFAULTS)
        config["voice_design"]["type"] = "engine"
        lock = store.lock()
        self.addCleanup(lock.rmdir)

        with self.assertRaisesRegex(RenderBlocked, "voice designer|reference recording"):
            ensure_voice(store, "missing", {"description": "Calm storyteller."}, engine,
                         None, config, self.root / "work", notify=lambda _: None)

        self.assertIsNone(store.load("missing"))
        self.assertFalse((self.root / "work").exists())

    def test_pool_model_selection_changes_take_identity_but_preserves_voice_binding(self):
        identities = []
        bindings = []
        for model in (DEFAULT_DRAMA3_MODEL, *QUANTIZED):
            with self.subTest(model=model):
                config = deepcopy(DEFAULTS)
                config["engine"] = self.engine_config(model)
                # The same operator revision cannot make distinct models interchangeable.
                config["render_pool"] = {
                    "model_revision": "same-operator-revision",
                    "workers": [{"type": "local", "name": "local", "concurrency": 1}],
                }
                pool = create_render_engine(config)
                self.addCleanup(pool.close)

                identities.append(pool.identity)
                bindings.append(pool.voice_binding_identity)
                if model == DEFAULT_DRAMA3_MODEL:
                    self.assertIs(require_designer(pool), pool)
                else:
                    with self.assertRaisesRegex(ValueError, "voice designer|reference recording"):
                        require_designer(pool)

        self.assertEqual(len({json.dumps(identity, sort_keys=True) for identity in identities}),
                         len(identities))
        self.assertTrue(all(binding == bindings[0] for binding in bindings))


class ModelChapterResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["render"]["work_dir"] = str(self.root / "work")
        plan = make_render_plan(self.bundle, self.config)
        self.assertFalse(plan["errors"])
        self.chapter = plan["chapters"][0]
        self.engine_identity = {"engine": "drama3_mlx", "model": DEFAULT_DRAMA3_MODEL}
        self.verifier_identity = {"backend": "fixture-verifier"}
        store = VoiceStore(self.root / "voices", "chapter-cache-fixture")
        lock = store.lock()
        self.addCleanup(lock.rmdir)
        reference = self.root / "reference.wav"
        reference.write_bytes(wav_bytes())
        self.voices = {
            voice_id: store.publish(voice_id, {"description": "A fixture narrator."}, reference,
                                    "Reference words.", self.engine_identity,
                                    {"passed": True}, {"description_applied": True})
            for voice_id in plan["voice_ids"]
        }

    def context(self, config, identity=None):
        return _chapter_context(self.bundle, self.chapter, config, self.voices,
                                identity or self.engine_identity, self.verifier_identity)

    def test_explicit_default_keeps_completed_sogni_chapters_compatible(self):
        legacy = deepcopy(self.config)
        legacy["engine"].pop("model")

        self.assertEqual(self.context(legacy), self.context(self.config))

    def test_new_builtin_default_does_not_invalidate_existing_plugin_chapters(self):
        current = deepcopy(self.config)
        current["engine"]["type"] = "custom_plugin"
        legacy = deepcopy(current)
        legacy["engine"].pop("model")
        identity = {"engine": "custom_plugin", "model": "custom-model"}

        self.assertEqual(self.context(legacy, identity), self.context(current, identity))

    def test_selected_quantized_model_invalidates_a_completed_chapter_even_during_assembly(self):
        contexts = []
        for model in (DEFAULT_DRAMA3_MODEL, *QUANTIZED):
            config = deepcopy(self.config)
            config["engine"]["model"] = model
            # Assembly uses saved engine identity without loading a model. The
            # configured selection must still distinguish incompatible audio.
            contexts.append(self.context(config))

        self.assertEqual(len({json.dumps(context, sort_keys=True) for context in contexts}),
                         len(contexts))


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class SelectedModelHTTPTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.config = {
            "server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                       "max_request_bytes": 104857600, "max_pending_requests": 4},
            "engine": {"type": "drama3_mlx", "model": FOUR_BIT},
        }
        try:
            self.server = make_server(self.config, self.engine)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def post(self, model=None):
        body = {"text": "Hello.", "format": "wav", "normalize": False,
                "references": [{"audio": base64.b64encode(wav_bytes()).decode(), "text": "Reference."}]}
        headers = {"Content-Type": "application/json", "Authorization": "Bearer private-test-key"}
        if model is not None:
            headers["model"] = model
        request = Request(self.url + "/v1/tts", data=json.dumps(body).encode(), headers=headers)
        return urlopen(request, timeout=5)

    def test_configured_model_and_omitted_header_both_reach_inference(self):
        for model in (FOUR_BIT, None):
            with self.subTest(model=model), self.post(model) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), wav_bytes())

        self.assertEqual([call["text"] for call in self.engine.calls], ["Hello.", "Hello."])

    def test_different_quantized_model_and_original_aliases_are_rejected_before_inference(self):
        for model in (*[candidate for candidate in QUANTIZED if candidate != FOUR_BIT],
                      DEFAULT_DRAMA3_MODEL, "drama-3"):
            with self.subTest(model=model), self.assertRaises(HTTPError) as raised:
                self.post(model)
            with raised.exception as response:
                self.assertEqual(response.code, 400)
                self.assertIn(FOUR_BIT, json.loads(response.read())["error"]["message"])

        self.assertEqual(self.engine.calls, [])

    def test_health_identifies_the_configured_checkpoint(self):
        request = Request(self.url + "/health", headers={"Authorization": "Bearer private-test-key"})

        with urlopen(request, timeout=5) as response:
            health = json.loads(response.read())

        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["model"], FOUR_BIT)
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
