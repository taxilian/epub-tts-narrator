"""MLX-Audio adapter contracts with only third-party/GPU boundaries replaced."""

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError
from epub_tts_narrator.renderer.mlx_audio_engine import MLXAudioDrama3Engine


MODEL = "mlx-community/fish-audio-drama-3-4bit"


class FakeArray(list):
    @property
    def ndim(self):
        return 2 if self and isinstance(self[0], list) else 1

    @property
    def shape(self):
        return (len(self), len(self[0])) if self.ndim == 2 else (len(self),)

    def copy(self):
        return FakeArray(deepcopy(self))


def all_finite(values):
    return all(all_finite(value) if isinstance(value, list) else math.isfinite(value)
               for value in values)


class MLXAudioEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps({"model_type": "fish_drama3"}))
        (self.checkpoint / "model.safetensors").write_bytes(b"quantized model")
        (self.checkpoint / "codec.safetensors").write_bytes(b"codec weights")
        (self.checkpoint / "chat_template.jinja").write_text("speech template")
        self.engine = MLXAudioDrama3Engine(self.checkpoint, model=MODEL, max_new_tokens=128)
        self.output = self.root / "output.wav"
        self.voice = self.reference("narrator")
        self.model = SimpleNamespace(
            config=SimpleNamespace(model_type="fish_drama3"), sample_rate=100,
            generate=Mock(),
        )
        self.loader = Mock(return_value=self.model)
        self.mx = ModuleType("mlx.core")
        self.mx.random = SimpleNamespace(seed=Mock())
        self.mx.synchronize = Mock()
        self.mx.clear_cache = Mock()
        mlx = ModuleType("mlx")
        mlx.core = self.mx
        np = ModuleType("numpy")
        np.float32 = "float32"
        np.asarray = lambda values, dtype: FakeArray(values)
        np.isfinite = lambda values: SimpleNamespace(all=lambda: all_finite(values))
        np.concatenate = lambda arrays: FakeArray(value for array in arrays for value in array)
        self.sf = ModuleType("soundfile")
        self.sf.read = Mock(return_value=(FakeArray([[0.1]] * 20), 24000))
        self.written = []
        self.sf.write = Mock(side_effect=self.write_audio)
        utils = ModuleType("mlx_audio.tts.utils")
        utils.load_model = self.loader
        audio_io = ModuleType("mlx_audio.audio_io")
        self.decoded = []

        def decode(stream, *, dtype, sample_rate, nchannels):
            self.decoded.append((stream.read(), sample_rate, nchannels))
            return FakeArray([0.1] * 20), sample_rate

        self.decode = audio_io.read = Mock(side_effect=decode)
        chunking = ModuleType("mlx_audio.tts.models.fish_drama3.chunking")
        self.split = chunking.split_text = Mock(side_effect=lambda text: [text])
        modules = {
            "mlx": mlx, "mlx.core": self.mx, "numpy": np, "soundfile": self.sf,
            "mlx_audio": ModuleType("mlx_audio"),
            "mlx_audio.tts": ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils, "mlx_audio.audio_io": audio_io,
            "mlx_audio.tts.models": ModuleType("mlx_audio.tts.models"),
            "mlx_audio.tts.models.fish_drama3": ModuleType("mlx_audio.tts.models.fish_drama3"),
            "mlx_audio.tts.models.fish_drama3.chunking": chunking,
        }
        fake_modules = patch.dict("sys.modules", modules)
        fake_modules.start()
        self.addCleanup(fake_modules.stop)
        versions = patch(
            "epub_tts_narrator.renderer.mlx_audio_engine.importlib.metadata.version",
            return_value="0.6.1",
        )
        self.version = versions.start()
        self.addCleanup(versions.stop)

    def reference(self, name):
        path = self.root / f"{name}.wav"
        blob = f"recording of {name}".encode()
        path.write_bytes(blob)
        return {
            "kind": "reference", "voice_id": name, "asset_hash": name,
            "reference_audio": str(path), "reference_text": f"The {name} reference.",
            "reference_sha256": hashlib.sha256(blob).hexdigest(),
        }

    def write_audio(self, path, audio, rate, *, subtype):
        self.written.append({"samples": list(audio), "rate": rate, "subtype": subtype})
        Path(path).write_bytes(b"complete wave")

    def segment(self, *, audio=None, index=0, final=True, tokens=25, **overrides):
        audio = [0.1] * 10 if audio is None else audio
        values = dict(audio=audio, sample_rate=100, samples=len(audio),
                      segment_idx=index, token_count=tokens,
                      is_streaming_chunk=False, is_final_chunk=final)
        values.update(overrides)
        return SimpleNamespace(**values)

    def generate(self, *segments, chunks=1):
        self.split.side_effect = None
        self.split.return_value = [f"chunk {index}" for index in range(chunks)]
        self.model.generate.return_value = iter(segments)

    def test_reference_clone_retains_directions_seed_sampling_and_verified_bytes(self):
        self.generate(self.segment())
        before = deepcopy(self.voice)
        result = self.engine.synthesize("Hello.", self.voice, self.output, seed=42, direction="whispering")

        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["text"], "<whisper>Hello.</whisper>")
        self.assertEqual(request["ref_text"], [self.voice["reference_text"]])
        self.assertEqual(self.decoded, [(Path(self.voice["reference_audio"]).read_bytes(), 100, 1)])
        self.assertEqual((request["temperature"], request["top_p"], request["top_k"]), (1.0, 0.9, 30))
        self.assertEqual(request["max_tokens"], 128)
        self.assertFalse(request["stream"])
        self.mx.random.seed.assert_called_once_with(42)
        self.assertEqual(self.voice, before)
        self.assertEqual(self.output.read_bytes(), b"complete wave")
        self.assertEqual(result["finish_reason"], "inferred_eos")
        self.assertTrue(result["direction_applied"])
        self.assertEqual((result["sample_rate"], result["sample_count"]), (100, 10))
        self.assertEqual(self.written[0]["subtype"], "FLOAT")
        self.assertEqual(list(self.root.glob(".take-*")), [])

    def test_freeform_style_is_delivered_with_the_same_reference_voice(self):
        self.generate(self.segment())
        before = deepcopy(self.voice)
        result = self.engine.synthesize("No!", self.voice, self.output, seed=42,
                                        direction="shout with a sustained vowel")
        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["text"], "[shout with a sustained vowel] No!")
        self.assertEqual(request["ref_text"], [self.voice["reference_text"]])
        self.assertEqual(self.voice, before)
        self.assertTrue(result["direction_applied"])

    def test_wrapping_quotes_cannot_form_a_second_runtime_generation_chunk(self):
        try:
            runtime = importlib.metadata.distribution("mlx-audio")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("requires installed MLX-Audio chunker; no model or GPU needed")
        path = runtime.locate_file("mlx_audio/tts/models/fish_drama3/chunking.py")
        if not path.is_file():
            self.skipTest("installed MLX-Audio has no Drama 3 chunker")
        spec = importlib.util.spec_from_file_location("runtime_chunking_for_test", path)
        chunker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunker)
        # Reproduce the real 299-byte first chunk plus orphan three-byte quote
        # with synthetic text instead of embedding a passage from the book.
        source = '“' + ("The travelers planned their journey. " * 9)[:292] + '….” '
        self.assertEqual(len(source.encode("utf-8")), 303)
        self.split.side_effect = chunker.split_text

        def generate(**request):
            chunks = chunker.split_text(request["text"])
            for index, chunk in enumerate(chunks):
                self.assertTrue(any(char.isalnum() for char in chunk), repr(chunk))
                yield self.segment(index=index, final=index == len(chunks) - 1)

        self.model.generate.side_effect = generate
        result = self.engine.synthesize(source, self.voice, self.output, seed=1)
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(self.model.generate.call_args.kwargs["text"], source[1:-2])
        self.assertEqual(result["synthesis_text"], source[1:-2])

    def test_synthesis_strips_only_target_edges_and_retains_reference_text(self):
        self.voice["reference_text"] = ' “Saved reference.” '
        before = deepcopy(self.voice)
        self.generate(self.segment())
        self.engine.synthesize(' « Don\'t say "never". » ', self.voice, self.output,
                               seed=1, direction="whispering")
        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["text"], '<whisper>Don\'t say "never".</whisper>')
        self.assertEqual(request["ref_text"], [' “Saved reference.” '])
        self.assertEqual(self.voice, before)

    def test_parenthesis_fragments_are_cleaned_only_in_target_before_generation(self):
        self.voice["reference_text"] = "(A saved reference)."
        original = deepcopy(self.voice)
        reference_bytes = Path(self.voice["reference_audio"]).read_bytes()
        cases = (
            ("(Lie, ", "Lie,"),
            ("I thought fiercely). ", "I thought fiercely."),
            ("(Lie.)", "Lie."),
            ("He said (quietly) that he agreed.", "He said (quietly) that he agreed."),
            ("He said (quietly).", "He said (quietly)."),
        )
        for index, (source, target) in enumerate(cases):
            with self.subTest(source=source):
                self.generate(self.segment())
                result = self.engine.synthesize(
                    source, self.voice, self.root / f"parenthesis-{index}.wav",
                    seed=1, direction="whisper",
                )
                expected = f"<whisper>{target}</whisper>"
                request = self.model.generate.call_args.kwargs
                self.assertEqual(request["text"], expected)
                self.assertEqual(result["synthesis_text"], expected)
                self.split.assert_called_with(expected)
                self.assertEqual(request["ref_text"], ["(A saved reference)."])
                self.assertEqual(self.decoded[-1], (reference_bytes, 100, 1))
                self.assertEqual(self.voice, original)

    def test_all_text_chunks_are_concatenated_in_order_before_publication(self):
        self.generate(self.segment(audio=[0.1] * 8, final=False),
                      self.segment(audio=[0.2] * 6, index=1), chunks=2)
        result = self.engine.synthesize("A long first part. A second part.", self.voice, self.output, seed=1)

        self.assertEqual(self.written[0]["samples"], [0.1] * 8 + [0.2] * 6)
        self.assertEqual(result["sample_count"], 14)
        self.assertEqual(result["duration_seconds"], 0.14)
        self.assertEqual(result["token_counts"], [25, 25])
        self.assertEqual(result["chunk_count"], 2)

    def test_dialogue_maps_repeated_voices_without_mutating_saved_references(self):
        other = self.reference("character")
        turns = [
            {"text": "He looked ", "voice": self.voice},
            {"text": "back. ", "voice": self.voice, "direction": "warmly"},
            {"text": "Stay here.", "voice": other, "direction": "whisper"},
            {"text": "Then he left.", "voice": self.voice},
        ]
        original = deepcopy(turns)
        self.generate(self.segment())
        result = self.engine.synthesize_dialogue(turns, self.output, seed=9)

        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["text"], "<|speaker:0|> He looked\n"
                         "<|speaker:0|> [warmly] back.\n"
                         "<|speaker:1|> <whisper>Stay here.</whisper>\n"
                         "<|speaker:0|> Then he left.")
        self.assertEqual(request["ref_text"], ["<|speaker:0|> The narrator reference.",
                                               "<|speaker:1|> The character reference."])
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(result["rendered_turn_count"], 4)
        self.assertEqual(turns, original)

    def test_http_inline_reference_speaker_ids_and_controls_are_preserved(self):
        voice = {**self.voice, "speaker": 3}
        self.generate(self.segment())
        text = "<|speaker:3|> [nervous] Hello."
        self.engine.synthesize_fish(text, [voice], self.output, seed=1)
        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["text"], text)
        self.assertEqual(request["ref_text"], ["<|speaker:3|> The narrator reference."])
        self.assertEqual(voice["reference_text"], "The narrator reference.")

    def test_incomplete_or_unordered_results_never_publish_a_partial_take(self):
        cases = [
            ([], 1), ([self.segment(final=False)], 2),
            ([self.segment()], 2),
            ([self.segment(index=1)], 1),
            ([self.segment(final=False), self.segment(index=2)], 2),
            ([self.segment(final=False), self.segment(index=1, final=False)], 2),
            ([self.segment(), self.segment(index=1)], 1),
            ([self.segment(is_final_chunk=1)], 1),
            ([self.segment(is_streaming_chunk=True)], 1),
        ]
        for segments, chunks in cases:
            with self.subTest(segments=segments, chunks=chunks):
                self.generate(*segments, chunks=chunks)
                with self.assertRaises(EngineError):
                    self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
                self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_cap_in_any_chunk_discards_the_entire_take(self):
        self.generate(self.segment(final=False), self.segment(index=1, tokens=128), chunks=2)
        with self.assertRaisesRegex(EngineError, "token limit"):
            self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_invalid_samples_or_metadata_never_publish_audio(self):
        cases = [
            self.segment(audio=[]), self.segment(audio=[0.1]),
            self.segment(audio=[[0.1]] * 10),
            self.segment(audio=[float("nan")] * 10),
            self.segment(audio=[float("inf")] * 10),
            self.segment(sample_rate=101), self.segment(sample_rate=True),
            self.segment(samples=9), self.segment(samples=True),
            self.segment(tokens=0), self.segment(tokens=True), self.segment(tokens=1.2),
        ]
        for segment in cases:
            with self.subTest(segment=segment):
                self.generate(segment)
                with self.assertRaises(EngineError):
                    self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
                self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_generation_exception_after_one_chunk_leaves_no_output(self):
        def failing_generator():
            yield self.segment(final=False)
            raise ValueError("context overflow")

        self.generate(chunks=2)
        self.model.generate.return_value = failing_generator()
        with self.assertRaisesRegex(EngineError, "context overflow"):
            self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_changed_reference_checksum_is_rechecked_on_every_request(self):
        self.generate(self.segment())
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        Path(self.voice["reference_audio"]).write_bytes(b"a substituted recording")
        self.voice["reference_audio_sha256"] = hashlib.sha256(b"a substituted recording").hexdigest()
        with self.assertRaisesRegex(EngineError, "checksum"):
            self.engine.synthesize("Hello.", self.voice, self.root / "next.wav", seed=1)
        self.assertEqual(self.model.generate.call_count, 1)
        self.assertEqual(len(self.decoded), 1)

    def test_nonfinite_reference_is_rejected_before_model_loading(self):
        self.sf.read.return_value = (FakeArray([[float("nan")]] * 20), 24000)
        with self.assertRaisesRegex(EngineError, "reference audio"):
            self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_runtime_decoder_receives_the_bytes_that_passed_checksum_validation(self):
        original = Path(self.voice["reference_audio"]).read_bytes()

        def load_after_file_changes(_path):
            Path(self.voice["reference_audio"]).write_bytes(b"concurrent replacement")
            return self.model

        self.loader.side_effect = load_after_file_changes
        self.generate(self.segment())
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertEqual(self.decoded[0][0], original)

    def test_invalid_input_and_descriptive_design_do_not_load_the_model(self):
        calls = [
            lambda: self.engine.synthesize("[instruction] Hello.", self.voice, self.output, seed=1),
            lambda: self.engine.synthesize("Hello.", self.voice, self.output, seed=True),
            lambda: self.engine.synthesize("Hello.", self.voice, self.output, seed=1, direction="<inject>"),
            lambda: self.engine.synthesize("Hello.", {**self.voice, "reference_text": "[inject]"}, self.output, seed=1),
            lambda: self.engine.synthesize_fish("<|speaker:1|> Hello.", [{**self.voice, "speaker": 0}], self.output, seed=1),
            lambda: self.engine.synthesize_fish("Hello.", [], self.output, seed=1),
            lambda: self.engine.synthesize_dialogue([], self.output, seed=1),
        ]
        for call in calls:
            with self.assertRaises(EngineError):
                call()
        with self.assertRaisesRegex(EngineError, r"separate \[voice_design\]"):
            self.engine.create_voice("A warm voice", "Hello.", self.output, seed=1)
        self.loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_audio_is_not_overwritten_or_used_for_a_new_generation(self):
        self.output.write_bytes(b"accepted recording")
        with self.assertRaisesRegex(EngineError, "already exists"):
            self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.loader.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"accepted recording")

    def test_publication_race_preserves_the_winner_and_cleans_staging(self):
        def competing_publication(_source, destination):
            Path(destination).write_bytes(b"concurrent accepted recording")
            raise FileExistsError("output already exists")

        self.generate(self.segment())
        with patch("epub_tts_narrator.renderer.mlx_audio_engine.os.link", side_effect=competing_publication):
            with self.assertRaises(EngineError):
                self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"concurrent accepted recording")
        self.assertEqual(list(self.root.glob(".take-*")), [])

    def test_stable_supported_patch_releases_can_load_and_generate_audio(self):
        for version in ("0.6.1", "0.6.2", "0.6.1.post1"):
            with self.subTest(version=version):
                self.version.return_value = version
                engine = MLXAudioDrama3Engine(self.checkpoint, model=MODEL)
                output = self.root / f"{version}.wav"
                self.generate(self.segment())

                engine.synthesize("Hello.", self.voice, output, seed=1)

                self.assertEqual(output.read_bytes(), b"complete wave")
                self.assertEqual(engine.identity["installed_version"], version)

    def test_missing_older_prerelease_and_future_minor_runtimes_are_rejected_before_loading(self):
        for version in (None, "", "invalid", "0.5.7", "0.6.0", "0.6.0.post1",
                        "0.6.1rc1", "0.6.2a1", "0.6.2.dev1", "0.7.0", "1.0.0"):
            self.version.side_effect = importlib.metadata.PackageNotFoundError if version is None else None
            self.version.return_value = version
            with self.subTest(version=version), self.assertRaisesRegex(EngineError, "mlx-audio>=0.6.1,<0.7") as caught:
                self.engine.warmup()
            self.assertIn("uv pip install --python .venv-renderer/bin/python", str(caught.exception))
            self.assertIn("-r renderer.requirements.txt", str(caught.exception))
        self.loader.assert_not_called()

    def test_wrong_checkpoint_family_is_rejected_before_loading(self):
        (self.checkpoint / "config.json").write_text('{"model_type":"qwen3_tts"}')
        with self.assertRaisesRegex(EngineError, "fish_drama3"):
            MLXAudioDrama3Engine(self.checkpoint, model=MODEL)
        with self.assertRaisesRegex(EngineError, "fish_drama3"):
            self.engine.warmup()
        self.loader.assert_not_called()

    def test_unsupported_model_and_bad_sampling_settings_fail_without_loading(self):
        for options in (
            {"model": "unsupported/drama-3"}, {"model": "fishaudio/drama-3"},
            {"temperature": float("nan")}, {"top_p": 2}, {"top_k": True},
            {"max_new_tokens": 0}, {"max_new_tokens": 2049}, {"min_audio_seconds": 0},
        ):
            with self.subTest(options=options), self.assertRaises(EngineError):
                MLXAudioDrama3Engine(self.checkpoint, **{"model": MODEL, **options})
        self.loader.assert_not_called()

    def test_close_releases_model_caches_and_next_request_loads_again(self):
        self.generate(self.segment())
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.engine.close()
        self.mx.clear_cache.assert_called_once_with()
        self.generate(self.segment())
        self.engine.synthesize("Again.", self.voice, self.root / "again.wav", seed=2)
        self.assertEqual(self.loader.call_count, 2)

    def test_model_runtime_and_content_changes_isolate_generated_audio_cache_identity(self):
        first = self.engine.identity
        same = MLXAudioDrama3Engine(self.checkpoint, model=MODEL, max_new_tokens=128).identity
        self.assertEqual(first, same)
        self.assertNotEqual(first, Drama3MLXEngine(self.checkpoint).identity)
        self.assertNotEqual(first, MLXAudioDrama3Engine(self.checkpoint, model=MODEL, top_p=0.8).identity)
        self.assertNotEqual(first, MLXAudioDrama3Engine(
            self.checkpoint, model="mlx-community/fish-audio-drama-3-8bit", max_new_tokens=128,
        ).identity)
        for name in ("codec.safetensors", "chat_template.jinja", "model.safetensors"):
            with self.subTest(file=name):
                old = MLXAudioDrama3Engine(self.checkpoint, model=MODEL).identity
                (self.checkpoint / name).write_bytes(b"replacement content")
                new = MLXAudioDrama3Engine(self.checkpoint, model=MODEL).identity
                self.assertNotEqual(old, new)
                self.assertEqual(new["checkpoint_files_sha256"][name],
                                 hashlib.sha256(b"replacement content").hexdigest())
        self.version.return_value = "0.6.2"
        self.assertNotEqual(first, MLXAudioDrama3Engine(self.checkpoint, model=MODEL, max_new_tokens=128).identity)
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
