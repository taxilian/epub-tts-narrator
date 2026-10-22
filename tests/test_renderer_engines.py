"""Adapter failure-contract tests without importing a 42 GB model."""

from array import array
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError, create_engine
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.voices import VoiceStore


class FakeEntryPoints(list):
    def select(self, *, name):
        return [entry for entry in self if entry.name == name]


@dataclass
class FakeReference:
    text: str
    codes: object


class RendererEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = Drama3MLXEngine(self.root)
        self.model = Mock(sample_rate=100)
        self.engine._model = self.model
        self.sf = self.engine._soundfile = Mock()
        self.sf.write.side_effect = lambda path, *args, **kwargs: Path(path).write_bytes(b"wave")
        self.output = self.root / "output.wav"

    def segment(self, text="Hello.", finish="eos", audio=None):
        return SimpleNamespace(
            text=text, finish_reason=finish,
            audio=array("f", [0.1] * 100) if audio is None else audio,
        )

    def generate(self, *segments):
        self.model.generate.return_value = iter(segments)

    def test_voice_creation_passes_description_to_scene_conditioning(self):
        description = "An older man's warm voice"
        audit = {"system": "scene prompt", "user": "Speaker: Hello.", "input_tokens": 42}
        with patch("epub_tts_narrator.renderer.engines.generate_scene_voice",
                   return_value=(self.segment(), audit)) as design:
            result = self.engine.create_voice(description, "Hello.", self.output, seed=1)
        design.assert_called_once_with(
            self.model, description, "Hello.", seed=1, temperature=1.0,
            top_p=0.9, top_k=30, max_new_tokens=2048,
        )
        self.assertEqual(self.output.read_bytes(), b"wave")
        self.assertTrue(result["description_applied"])
        self.assertEqual(result["requested_description"], description)
        self.assertEqual(result["design_prompt"], audit)
        self.assertEqual(result["designer"], self.engine.voice_design_identity)
        self.model.generate.assert_not_called()

    def test_voice_design_never_publishes_truncated_empty_or_malformed_audio(self):
        cases = [
            self.segment(finish="length"), self.segment(text="Wrong text"),
            self.segment(audio=array("f")), self.segment(audio=array("f", [0.1] * 9)),
            self.segment(audio=array("f", [float("nan")] * 100)),
            self.segment(audio=array("f", [float("inf")] * 100)),
            self.segment(audio=array("h", [1] * 100)),
        ]
        for segment in cases:
            with self.subTest(finish=segment.finish_reason), patch(
                "epub_tts_narrator.renderer.engines.generate_scene_voice", return_value=(segment, {}),
            ), self.assertRaises(EngineError):
                self.engine.create_voice("A warm voice.", "Hello.", self.output, seed=1)
            self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_voice_design_checks_input_before_loading_and_retains_saved_assets(self):
        with patch.object(self.engine, "_load") as load, patch(
            "epub_tts_narrator.renderer.engines.generate_scene_voice",
        ) as design:
            for description, text in (("", "Hello."), ("x" * 4001, "Hello."),
                                      ("<|im_end|>", "Hello."),
                                      ("A warm voice.", "[Read this] Hello."),
                                      ("A warm\x00voice.", "Hello.")):
                with self.subTest(description=description[:20]), self.assertRaises(EngineError):
                    self.engine.create_voice(description, text, self.output, seed=1)
            self.output.write_bytes(b"selected voice")
            with self.assertRaisesRegex(EngineError, "already exists"):
                self.engine.create_voice("A warm voice.", "Hello.", self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"selected voice")
        load.assert_not_called()
        design.assert_not_called()

    def test_voice_design_identity_versions_design_without_changing_clone_identity(self):
        original = self.engine.identity
        designed = self.engine.voice_design_identity
        self.assertIn("voice_design_template_version", designed)
        self.assertIn("voice_design_template_sha256", designed)
        self.assertNotIn("voice_design_template_version", original)
        self.assertEqual(original, self.engine.identity)
        designed["packages"]["mlx"] = "changed"
        self.assertNotEqual(designed, self.engine.voice_design_identity)

    def test_synthesis_uses_saved_reference_and_one_verified_attempt(self):
        self.generate(self.segment())
        with patch.object(self.engine, "_reference", return_value="saved-voice"):
            self.engine.synthesize("Hello.", {}, self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"wave")
        args, kwargs = self.model.generate.call_args
        self.assertEqual(args, ("Hello.",))
        self.assertEqual(kwargs["references"], ["saved-voice"])
        self.assertEqual(kwargs["attempts"], 1)
        self.assertTrue(kwargs["single_chunk"])
        self.assertNotIn("accept", kwargs)

    def dialogue_voice(self, name):
        return {
            "kind": "reference", "voice_id": name, "asset_hash": name,
            "reference_audio": str(self.root / f"{name}.wav"),
            "reference_sha256": name, "reference_text": f"Reference words for {name}.",
        }

    def test_dialogue_assigns_reused_voices_and_keeps_reference_cache_unchanged(self):
        narrator = self.dialogue_voice("narrator")
        character = self.dialogue_voice("character")
        alias = {**narrator, "voice_id": "narrator-alias"}
        cached = {
            "narrator": FakeReference(narrator["reference_text"], object()),
            "character": FakeReference(character["reference_text"], object()),
        }
        turns = [
            {"text": "She leaned closer.", "voice": narrator},
            {"text": "Listen to me.", "voice": character, "direction": "whispering"},
            {"text": "Then she smiled.", "voice": alias, "direction": "warmly"},
        ]
        before = deepcopy(turns)
        engine_text = (
            "<|speaker:0|> She leaned closer.\n"
            "<|speaker:1|> <whisper>Listen to me.</whisper>\n"
            "<|speaker:0|> Then she smiled."
        )
        self.generate(self.segment(text=engine_text))
        with patch.object(self.engine, "_reference", side_effect=lambda v: cached[v["asset_hash"]]) as reference:
            result = self.engine.synthesize_dialogue(turns, self.output, seed=31)
        self.assertEqual(reference.call_count, 2)
        args, kwargs = self.model.generate.call_args
        self.assertEqual(args, (engine_text,))
        self.assertEqual(kwargs["seed"], 31)
        self.assertEqual(kwargs["attempts"], 1)
        self.assertTrue(kwargs["single_chunk"])
        for i, name in enumerate(("narrator", "character")):
            tagged = kwargs["references"][i]
            self.assertIsNot(tagged, cached[name])
            self.assertIs(tagged.codes, cached[name].codes)
            self.assertEqual(tagged.text, f"<|speaker:{i}|> {cached[name].text}")
            self.assertEqual(cached[name].text, f"Reference words for {name}.")
        self.assertEqual(turns, before)
        self.assertEqual((result["speaker_count"], result["turn_count"]), (2, 3))
        self.assertEqual(result["speaker_mapping"], [
            {"speaker_id": 0, "voice_id": "narrator", "asset_hash": "narrator"},
            {"speaker_id": 1, "voice_id": "character", "asset_hash": "character"},
        ])
        self.assertTrue(result["turn_directions"][1]["direction_applied"])
        self.assertFalse(result["turn_directions"][2]["direction_applied"])
        self.assertTrue(self.engine.capabilities["multi_speaker"])
        self.assertEqual(self.output.read_bytes(), b"wave")

    def test_dialogue_validates_all_turns_and_speaker_limit_before_loading(self):
        valid = {"text": "Hello.", "voice": self.dialogue_voice("narrator")}
        cases = [[], None, [valid, {}], [valid, {**valid, "text": "<|speaker:3|> Bad."}],
                 [valid, {**valid, "text": "A [Fireball]."}],
                 [valid, {**valid, "direction": "<whisper>"}],
                 [valid, {**valid, "voice": {"kind": "preset"}}],
                 [valid, {**valid, "voice": {**valid["voice"], "reference_text": "[fake]"}}],
                 [{**valid, "text": "<|speaker:"}, {**valid, "text": "3|> Bad."}],
                 [{"text": "Hello.", "voice": self.dialogue_voice(str(i))} for i in range(17)]]
        with patch.object(self.engine, "_reference") as reference, patch.object(self.engine, "_load") as load:
            for turns in cases:
                with self.subTest(turns=turns), self.assertRaises(EngineError):
                    self.engine.synthesize_dialogue(turns, self.output, seed=1)
            with self.assertRaisesRegex(EngineError, "seed"):
                self.engine.synthesize_dialogue([valid], self.output, seed=-1)
        reference.assert_not_called()
        load.assert_not_called()
        self.model.generate.assert_not_called()

    def test_dialogue_combines_adjacent_same_voice_and_applied_direction_without_text_changes(self):
        narrator = self.dialogue_voice("narrator")
        character = self.dialogue_voice("character")
        turns = [
            {"text": "He looked ", "voice": narrator},
            {"text": "back. ", "voice": narrator, "direction": "warmly"},
            {"text": "Hello.", "voice": character},
            {"text": "Stay ", "voice": character, "direction": "whisper"},
            {"text": "here.", "voice": character, "direction": "whispering"},
        ]
        expected = (
            "<|speaker:0|> He looked back. \n"
            "<|speaker:1|> Hello.\n"
            "<|speaker:1|> <whisper>Stay here.</whisper>"
        )
        self.generate(self.segment(text=expected))
        with patch.object(self.engine, "_reference", side_effect=lambda voice: FakeReference(
            voice["reference_text"], object(),
        )):
            result = self.engine.synthesize_dialogue(turns, self.output, seed=31)
        self.assertEqual(self.model.generate.call_args.args, (expected,))
        self.assertEqual(result["turn_count"], 5)
        self.assertEqual(result["rendered_turn_count"], 3)
        self.assertEqual(len(result["turn_directions"]), 5)
        self.assertEqual(result["turn_directions"][1]["direction"], "warmly")
        self.assertFalse(result["turn_directions"][1]["direction_applied"])
        self.assertEqual(result["turn_directions"][4]["direction"], "whispering")
        self.assertTrue(result["turn_directions"][4]["direction_applied"])

    def test_dialogue_retains_failure_and_exclusive_publication_contract(self):
        voice = self.dialogue_voice("narrator")
        turns = [{"text": "Hello.", "voice": voice}]
        cached = FakeReference(voice["reference_text"], object())
        for segments in ([], [self.segment(text="<|speaker:0|> Hello.", finish="length")],
                         [self.segment(text="Wrong text")],
                         [self.segment(text="<|speaker:0|> Hello.")] * 2):
            self.generate(*segments)
            with patch.object(self.engine, "_reference", return_value=cached), self.assertRaises(EngineError):
                self.engine.synthesize_dialogue(turns, self.output, seed=1)
            self.assertFalse(self.output.exists())
        self.model.generate.side_effect = ValueError("prompt rows plus output exceed model context")
        with patch.object(self.engine, "_reference", return_value=cached), self.assertRaisesRegex(EngineError, "model context"):
            self.engine.synthesize_dialogue(turns, self.output, seed=1)
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"accepted recording")
        with patch.object(self.engine, "_reference") as reference, self.assertRaisesRegex(EngineError, "already exists"):
            self.engine.synthesize_dialogue(turns, self.output, seed=1)
        reference.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"accepted recording")

    def test_never_publishes_truncated_empty_multi_or_changed_text(self):
        cases = [
            [], [self.segment(finish="length")],
            [self.segment(), self.segment()],
            [self.segment(text="Wrong text")],
            [self.segment(audio=array("f"))],
            [self.segment(audio=array("f", [0.1] * 9))],
            [self.segment(audio=array("f", [float("nan")] * 100))],
            [self.segment(audio=array("f", [float("inf")] * 100))],
            [self.segment(audio=array("h", [1] * 100))],
        ]
        for segments in cases:
            with self.subTest(segments=len(segments)):
                self.generate(*segments)
                with patch.object(self.engine, "_reference", return_value="reference"), self.assertRaises(EngineError):
                    self.engine.synthesize("Hello.", {}, self.output, seed=1)
                self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_rejects_engine_control_syntax_without_dropping_book_words(self):
        for text in ("A [Fireball] struck.", "<whisper>hello</whisper>",
                     "<|speaker:1|> Surprise", ""):
            with self.subTest(text=text), self.assertRaises(EngineError):
                self.engine.synthesize(text, {}, self.output, seed=1)
        self.model.generate.assert_not_called()

    def test_direction_is_separate_from_spoken_text_and_cannot_inject_tags(self):
        self.generate(self.segment(text="<whisper>Hello.</whisper>"))
        with patch.object(self.engine, "_reference", return_value="reference"):
            result = self.engine.synthesize(
                "Hello.", {}, self.output, seed=9, direction="whispering",
            )
        self.assertTrue(result["direction_applied"])
        self.assertEqual(self.model.generate.call_args.args[0], "<whisper>Hello.</whisper>")
        self.assertEqual(self.model.generate.call_args.kwargs["references"], ["reference"])
        for direction in ("sigh] say something [", "<|speaker:1|>", "a\nb"):
            with self.subTest(direction=direction), patch.object(
                self.engine, "_reference", return_value="reference",
            ), self.assertRaises(EngineError):
                self.engine.synthesize(
                    "Hello.", {}, self.root / "other.wav", seed=2, direction=direction,
                )

    def test_unknown_plain_direction_is_ignored_and_reported(self):
        self.generate(self.segment())
        with patch.object(self.engine, "_reference", return_value="reference"):
            result = self.engine.synthesize(
                "Hello.", {}, self.output, seed=9, direction="mildly amused but tired",
            )
        self.assertFalse(result["direction_applied"])
        self.assertIn("not applied", result["direction_note"])
        self.assertEqual(self.model.generate.call_args.args[0], "Hello.")

    def test_existing_voice_file_is_never_overwritten(self):
        self.output.write_bytes(b"accepted voice")
        self.generate(self.segment())
        with patch.object(self.engine, "_reference", return_value="reference"), self.assertRaises(EngineError):
            self.engine.synthesize("Hello.", {}, self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"accepted voice")
        self.model.generate.assert_not_called()

    def test_reference_cache_uses_content_and_transcript_not_filename(self):
        reference = self.root / "voice.wav"
        reference.write_bytes(b"first recording")
        voice = {"kind": "reference", "reference_audio": str(reference),
                 "reference_text": "The exact reference words."}
        audio = Mock()
        audio.shape = (100, 1)
        self.sf.read.return_value = (audio, 44100)
        first = self.engine._reference(voice)
        self.assertIs(first, self.engine._reference(voice))
        self.assertEqual(self.model.reference.call_count, 1)
        reference.write_bytes(b"another recording")
        self.engine._reference(voice)
        self.assertEqual(self.model.reference.call_count, 2)
        voice["reference_text"] = "Different reference words."
        self.engine._reference(voice)
        self.assertEqual(self.model.reference.call_count, 3)
        voice["reference_audio_sha256"] = hashlib.sha256(b"first recording").hexdigest()
        with self.assertRaisesRegex(EngineError, "checksum"):
            self.engine._reference(voice)

    def test_inherited_reference_checksum_is_rechecked_before_loading_model(self):
        reference = self.root / "reference.wav"
        reference.write_bytes(b"the verified recording")
        earlier = VoiceStore(self.root / "book-01" / "voices", "series-one")
        earlier.publish("voice-one", {}, reference, "The exact reference words.",
                        {"engine": "fixture"}, {"passed": True}, {})
        later = VoiceStore(self.root / "book-02" / "voices", "series-one", parents=(earlier,))
        binding = later.load("voice-one")
        self.assertTrue(binding["inherited"])
        # The persisted field takes priority over the old adapter-only spelling.
        binding["reference_audio_sha256"] = "0" * 64
        audio = Mock()
        audio.shape = (100, 1)
        self.sf.read.return_value = (audio, 44100)
        self.engine._reference(binding)
        self.assertEqual(self.model.reference.call_count, 1)

        # A reference changed after lookup must not reach model loading or reuse
        # the cached encoded reference, even if the legacy checksum is updated.
        changed = b"a changed recording after the saved binding was loaded"
        Path(binding["reference_audio"]).write_bytes(changed)
        binding["reference_audio_sha256"] = hashlib.sha256(changed).hexdigest()
        with patch.object(self.engine, "_load") as load:
            with self.assertRaisesRegex(EngineError, "checksum"):
                self.engine._reference(binding)
        load.assert_not_called()
        self.assertEqual(self.model.reference.call_count, 1)
        self.assertEqual(self.sf.read.call_count, 1)

    def test_runtime_identity_distinguishes_required_from_installed(self):
        with patch("epub_tts_narrator.renderer.engines._runtime_details",
                   return_value={"installed_version": None, "installed_revision": None}):
            identity = self.engine.identity
        self.assertEqual(identity["required_runtime_version"], "0.1.1+epub.1")
        self.assertIsNone(identity["installed_version"])
        self.assertTrue(self.engine.capabilities["voice_design"])
        self.assertTrue(self.engine.capabilities["reference_voice"])

    def test_identity_covers_weight_content_and_invalidates_changed_file_cache(self):
        weights = self.root / "model-00001-of-00001.safetensors"
        weights.write_bytes(b"first weights")
        first = self.engine.identity
        self.assertEqual(
            first["checkpoint_files_sha256"][weights.name],
            hashlib.sha256(b"first weights").hexdigest(),
        )
        self.assertTrue((self.root / ".renderer-checkpoint-hashes-v1.json").exists())
        self.assertEqual(Drama3MLXEngine(self.root).identity, first)
        weights.write_bytes(b"other weights")
        second = Drama3MLXEngine(self.root).identity
        self.assertNotEqual(first["checkpoint_files_sha256"], second["checkpoint_files_sha256"])
        self.assertEqual(second["checkpoint_files_sha256"][weights.name],
                         hashlib.sha256(b"other weights").hexdigest())
        self.assertIsNone(second["model_revision"])

    def test_no_backend_fallback_or_bad_numeric_config(self):
        with self.assertRaises(EngineError):
            create_engine({"name": "does-not-exist", "checkpoint": str(self.root)})
        for key, value in (("top_p", 2), ("temperature", float("nan")),
                           ("top_k", True), ("max_new_tokens", 0),
                           ("min_audio_seconds", 0)):
            with self.subTest(key=key), self.assertRaises(EngineError):
                Drama3MLXEngine(self.root, **{key: value})

    def test_complete_toml_engine_table_reaches_builtin_factory(self):
        checkpoint = self.root / "model"
        checkpoint.mkdir()
        config_path = self.root / "renderer.toml"
        config_path.write_text(
            '[engine]\n'
            'type = "drama3_mlx"\ncheckpoint = "./model"\n'
            'experts = "q4"\ntemperature = 0.95\ntop_p = 0.85\n'
            'top_k = 25\nmax_new_tokens = 1500\n'
            '[verification]\nmodel = "./asr-model"\n'
            '[voices]\nseries_id = "test-series"\n',
            encoding="utf-8",
        )
        config = load_config(config_path)
        with patch.object(Drama3MLXEngine, "_load") as load, patch(
            "epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
        ) as entry_points:
            engine = create_engine(config["engine"])
        self.assertIsInstance(engine, Drama3MLXEngine)
        self.assertEqual(engine.checkpoint, checkpoint.resolve())
        self.assertEqual(engine.experts, "q4")
        self.assertEqual((engine.temperature, engine.top_p, engine.top_k), (0.95, 0.85, 25))
        self.assertEqual(engine.max_new_tokens, 1500)
        self.assertIsNone(engine._model)
        load.assert_not_called()
        entry_points.assert_not_called()

    def plugin(self, **overrides):
        values = {
            "identity": {"engine": "example_tts", "model": "example-v1"},
            "capabilities": {
                "voice_design": False, "reference_voice": True,
                "preset_voice": False, "delivery_control": False,
            },
            "create_voice": Mock(),
            "synthesize": Mock(),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def entry(self, result, *, name="example_tts", factory=None):
        factory = factory if factory is not None else Mock(return_value=result)
        return SimpleNamespace(name=name, load=Mock(return_value=factory))

    def test_registered_plugin_receives_only_explicit_plugin_options(self):
        plugin = self.plugin()
        entry = self.entry(plugin)
        options = {"speaker_mode": "reference", "local_model": "/models/example"}
        config = {
            "type": "example_tts", "checkpoint": None, "options": options,
            "experts": "q8", "temperature": 1.0, "top_p": 0.9,
            "top_k": 30, "max_new_tokens": 2048,
        }
        before = deepcopy(config)
        with patch("epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
                   return_value=FakeEntryPoints([entry])) as points, patch(
                       "epub_tts_narrator.renderer.engines.Drama3MLXEngine",
                   ) as builtin:
            engine = create_engine(config)
        self.assertIs(engine, plugin)
        points.assert_called_once_with(group="epub_tts_narrator.tts")
        entry.load.assert_called_once_with()
        entry.load.return_value.assert_called_once_with(options)
        self.assertIsNot(entry.load.return_value.call_args.args[0], options)
        self.assertEqual(config, before)
        builtin.assert_not_called()

    def test_unknown_and_ambiguous_plugins_fail_without_builtin_fallback(self):
        for entries in ([], [self.entry(self.plugin()), self.entry(self.plugin())]):
            with self.subTest(count=len(entries)), patch(
                "epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
                return_value=FakeEntryPoints(entries),
            ), patch("epub_tts_narrator.renderer.engines.Drama3MLXEngine") as builtin:
                with self.assertRaises(EngineError):
                    create_engine({"type": "example_tts", "options": {}})
                builtin.assert_not_called()
                for entry in entries:
                    entry.load.assert_not_called()

    def test_malformed_plugin_contract_is_rejected_before_synthesis(self):
        malformed = [
            object(),
            self.plugin(identity=None),
            self.plugin(capabilities={"reference_voice": "yes"}),
            self.plugin(create_voice=None),
            self.plugin(synthesize="not a method"),
            self.plugin(capabilities={**self.plugin().capabilities, "multi_speaker": "yes"}),
            self.plugin(capabilities={**self.plugin().capabilities, "multi_speaker": True}),
        ]
        for plugin in malformed:
            entry = self.entry(plugin)
            with self.subTest(plugin=plugin), patch(
                "epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
                return_value=FakeEntryPoints([entry]),
            ), patch("epub_tts_narrator.renderer.engines.Drama3MLXEngine") as builtin:
                with self.assertRaises(EngineError):
                    create_engine({"type": "example_tts", "options": {}})
                builtin.assert_not_called()

    def test_plugin_load_or_factory_failure_has_no_implicit_fallback(self):
        load_error = self.entry(None)
        load_error.load.side_effect = ImportError("Optional engine dependency is missing")
        factory_error = self.entry(None, factory=Mock(side_effect=RuntimeError("Model cannot load")))
        not_a_factory = self.entry(None)
        not_a_factory.load.return_value = None
        for entry in (load_error, factory_error, not_a_factory):
            with self.subTest(entry=entry), patch(
                "epub_tts_narrator.renderer.engines.importlib.metadata.entry_points",
                return_value=FakeEntryPoints([entry]),
            ), patch("epub_tts_narrator.renderer.engines.Drama3MLXEngine") as builtin:
                with self.assertRaises(EngineError):
                    create_engine({"type": "example_tts", "options": {}})
                builtin.assert_not_called()


if __name__ == "__main__":
    unittest.main()
