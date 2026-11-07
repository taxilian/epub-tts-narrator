"""MLX-Audio adapter contracts with only third-party/GPU boundaries replaced."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError
from epub_tts_narrator.renderer.mlx_audio_engine import MLXAudioDrama3Engine
from shared_fixtures import PARENTHESIS_TRIM_CASES


MODEL = "mlx-community/fish-audio-drama-3-4bit"


class FakeModel:
    """The internals MLX-Audio's own generate uses: voice cache, frames, sampler, codec."""

    sample_rate = 100
    num_codebooks = 10

    def __init__(self):
        self.config = SimpleNamespace(model_type="fish_drama3")
        self.tower = SimpleNamespace(vocab_size=1024)
        self.voices, self.prompts, self.sampling = [], [], []
        self.frame_count = 5
        self.rows_per_turn = 20
        self.audio = None
        self.codec = SimpleNamespace(reset_streaming_state=Mock(), streaming_step=self.decode)

    def _voice(self, audios, texts):
        self.voices.append(list(texts))
        return SimpleNamespace(references=[(text, None) for text in texts])

    def build_frames(self, references, history, text):
        self.prompts.append({"references": [item[0] for item in references],
                             "history": [item[0] for item in history], "text": text})
        return np.zeros((self.rows_per_turn * (1 + len(history)), 11)), 3

    def _frames(self, frames, system_rows, voice, max_frames, temperature, top_k, top_p):
        self.sampling.append((max_frames, temperature, top_k, top_p))
        for index in range(min(self.frame_count, max_frames)):
            yield np.full(10, index, dtype=np.int32)

    def decode(self, frames):
        if self.audio is not None:
            return np.array([[self.audio]], dtype=np.float32)
        return np.full((1, 1, frames.shape[2] * 4), 0.1, dtype=np.float32)


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
        self.model = FakeModel()
        self.loader = Mock(return_value=self.model)
        self.mx = ModuleType("mlx.core")
        self.mx.random = SimpleNamespace(seed=Mock())
        self.mx.synchronize = Mock()
        self.mx.clear_cache = Mock()
        self.mx.eval = Mock()
        self.mx.array = lambda values: np.asarray(values)
        self.mx.stack = lambda values, axis: np.stack(values, axis=axis)
        mlx = ModuleType("mlx")
        mlx.core = self.mx
        self.sf = ModuleType("soundfile")
        self.sf.read = Mock(return_value=(np.full((20, 1), 0.1, dtype=np.float32), 24000))
        self.written = []
        self.sf.write = Mock(side_effect=self.write_audio)
        utils = ModuleType("mlx_audio.tts.utils")
        utils.load_model = self.loader
        audio_io = ModuleType("mlx_audio.audio_io")
        self.decoded = []

        def decode(stream, *, dtype, sample_rate, nchannels):
            self.decoded.append((stream.read(), sample_rate, nchannels))
            return np.full(20, 0.1, dtype=np.float32), sample_rate

        self.decode = audio_io.read = Mock(side_effect=decode)
        modules = {
            "mlx": mlx, "mlx.core": self.mx, "soundfile": self.sf,
            "mlx_audio": ModuleType("mlx_audio"),
            "mlx_audio.tts": ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils, "mlx_audio.audio_io": audio_io,
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

    def prompt(self):
        return self.model.prompts[-1]

    def test_reference_clone_retains_directions_seed_sampling_and_verified_bytes(self):
        before = deepcopy(self.voice)
        result = self.engine.synthesize("Hello.", self.voice, self.output, seed=42, direction="whispering")
        self.assertEqual(self.prompt()["text"], "<|speaker:0|> <whisper>Hello.</whisper>")
        self.assertEqual(self.model.voices[-1], ["<|speaker:0|> The narrator reference."])
        self.assertEqual(self.decoded, [(Path(self.voice["reference_audio"]).read_bytes(), 100, 1)])
        self.assertEqual(self.model.sampling[-1], (128, 0.7, 30, 0.7))
        self.mx.random.seed.assert_called_once_with(42)
        self.assertEqual(self.voice, before)
        self.assertEqual(self.output.read_bytes(), b"complete wave")
        self.assertEqual(result["finish_reason"], "inferred_eos")
        self.assertTrue(result["direction_applied"])
        self.assertEqual((result["sample_rate"], result["sample_count"], result["token_counts"]), (100, 20, [5]))
        self.assertEqual(self.written[0]["subtype"], "FLOAT")
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])
        self.assertNotIn("generated_codes", result)

    def test_freeform_style_is_delivered_with_the_same_reference_voice(self):
        result = self.engine.synthesize("No!", self.voice, self.output, seed=42,
                                        direction="shout with a sustained vowel")
        self.assertEqual(self.prompt()["text"], "<|speaker:0|> [shout with a sustained vowel] No!")
        self.assertEqual(self.model.voices[-1], ["<|speaker:0|> The narrator reference."])
        self.assertTrue(result["direction_applied"])

    def test_synthesis_strips_only_target_edges_and_retains_reference_text(self):
        self.voice["reference_text"] = ' “Saved reference.” '
        before = deepcopy(self.voice)
        self.engine.synthesize(' « Don\'t say "never". » ', self.voice, self.output,
                               seed=1, direction="whispering")
        self.assertEqual(self.prompt()["text"], '<|speaker:0|> <whisper>Don\'t say "never".</whisper>')
        self.assertEqual(self.model.voices[-1], ['<|speaker:0|>  “Saved reference.” '])
        self.assertEqual(self.voice, before)

    def test_parenthesis_fragments_are_cleaned_only_in_target_before_generation(self):
        self.voice["reference_text"] = "(A saved reference)."
        original = deepcopy(self.voice)
        reference_bytes = Path(self.voice["reference_audio"]).read_bytes()
        for index, (source, target) in enumerate(PARENTHESIS_TRIM_CASES):
            with self.subTest(source=source):
                result = self.engine.synthesize(
                    source, self.voice, self.root / f"parenthesis-{index}.wav",
                    seed=1, direction="whisper",
                )
                expected = f"<whisper>{target}</whisper>"
                self.assertEqual(self.prompt()["text"], f"<|speaker:0|> {expected}")
                self.assertEqual(result["synthesis_text"], expected)
                self.assertEqual(self.model.voices[-1], ["<|speaker:0|> (A saved reference)."])
                self.assertEqual(self.decoded[-1], (reference_bytes, 100, 1))
                self.assertEqual(self.voice, original)

    def test_dialogue_maps_repeated_voices_without_mutating_saved_references(self):
        other = self.reference("character")
        turns = [
            {"text": "He looked ", "voice": self.voice},
            {"text": "back. ", "voice": self.voice, "direction": "warmly"},
            {"text": "Stay here.", "voice": other, "direction": "whisper"},
            {"text": "Then he left.", "voice": self.voice},
        ]
        original = deepcopy(turns)
        result = self.engine.synthesize_dialogue(turns, self.output, seed=9)
        self.assertEqual(self.prompt()["text"], "<|speaker:0|> He looked\n"
                         "<|speaker:0|> [warmly] back.\n"
                         "<|speaker:1|> <whisper>Stay here.</whisper>\n"
                         "<|speaker:0|> Then he left.")
        self.assertEqual(self.model.voices[-1], ["<|speaker:0|> The narrator reference.",
                                                 "<|speaker:1|> The character reference."])
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(result["rendered_turn_count"], 4)
        self.assertEqual(turns, original)

    def test_http_inline_reference_speaker_ids_and_controls_are_preserved(self):
        voice = {**self.voice, "speaker": 3}
        text = "<|speaker:3|> [nervous] Hello."
        result = self.engine.synthesize_fish(text, [voice], self.output, seed=1)
        self.assertEqual(self.prompt()["text"], text)
        self.assertEqual(self.model.voices[-1], ["<|speaker:3|> The narrator reference."])
        self.assertEqual(voice["reference_text"], "The narrator reference.")
        self.assertEqual((len(result["codes"]), result["codebook_size"]), (10, 1024))

    def test_history_turns_reach_the_frame_builder_with_fish_overflow_fallback(self):
        other = self.reference("character")
        history = [("<|speaker:1|> Pinned narration.", [[1] * 2] * 10),
                   ("<|speaker:0|> Latest line.", [[2] * 2] * 10)]
        references = [dict(other, speaker=0), dict(self.voice, speaker=1)]
        metadata, codes, size = self.engine._generate_wire(
            "<|speaker:0|> New.", references, history, self.output, seed=3)
        self.assertEqual(self.prompt()["history"], ["<|speaker:1|> Pinned narration.", "<|speaker:0|> Latest line."])
        self.assertEqual((metadata["history_turns"], metadata["history_available"]), (2, 2))
        self.assertEqual((len(codes), len(codes[0]), size), (10, 5, 1024))
        self.model.rows_per_turn = 3000
        metadata, _, _ = self.engine._generate_wire(
            "<|speaker:0|> Again.", references, history, self.root / "again.wav", seed=3)
        self.assertEqual(self.prompt()["history"], ["<|speaker:1|> Pinned narration."])
        self.assertEqual(metadata["history_turns"], 1)
        self.model.rows_per_turn = 9000
        with self.assertRaisesRegex(EngineError, "too long even without chapter history"):
            self.engine._generate_wire("<|speaker:0|> Last.", references, history, self.root / "last.wav", seed=3)

    def test_token_cap_empty_frames_and_bad_audio_never_publish(self):
        cases = [("frame_count", 128, "token limit"), ("frame_count", 0, "no audio frames"),
                 ("audio", [float("nan")] * 20, "malformed"), ("audio", [0.1] * 3, "short")]
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field, value=str(value)[:12]):
                self.model = FakeModel()
                setattr(self.model, field, value)
                self.loader.return_value = self.model
                self.engine.close()
                output = self.root / f"bad-{index}.wav"
                with self.assertRaisesRegex(EngineError, message):
                    self.engine.synthesize("Hello.", self.voice, output, seed=1)
                self.assertFalse(output.exists())
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])

    def test_missing_runtime_internals_fail_clearly(self):
        del FakeModel._frames
        try:
            with self.assertRaisesRegex(EngineError, "internals do not support chapter history"):
                self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        finally:
            FakeModel._frames = FakeModelFrames
        self.assertFalse(self.output.exists())

    def test_changed_reference_checksum_is_rechecked_on_every_request(self):
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        Path(self.voice["reference_audio"]).write_bytes(b"a substituted recording")
        self.voice["reference_audio_sha256"] = hashlib.sha256(b"a substituted recording").hexdigest()
        with self.assertRaisesRegex(EngineError, "checksum"):
            self.engine.synthesize("Hello.", self.voice, self.root / "next.wav", seed=1)
        self.assertEqual(len(self.model.prompts), 1)
        self.assertEqual(len(self.decoded), 1)

    def test_nonfinite_reference_is_rejected_before_model_loading(self):
        self.sf.read.return_value = (np.full((20, 1), np.nan, dtype=np.float32), 24000)
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
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertEqual(self.decoded[0][0], original)

    def test_invalid_input_and_descriptive_design_do_not_load_the_model(self):
        calls = [
            lambda: self.engine.synthesize("[instruction] Hello.", self.voice, self.output, seed=1),
            lambda: self.engine.synthesize("Hello.", self.voice, self.output, seed=True),
            lambda: self.engine.synthesize("Hello.", self.voice, self.output, seed=1, direction="<inject>"),
            lambda: self.engine.synthesize("Hello.", {**self.voice, "reference_text": "[inject]"}, self.output, seed=1),
            lambda: self.engine.synthesize_fish("<|speaker:1|> Hello.", [{**self.voice, "speaker": 0}], self.output, seed=1),
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

        with patch("epub_tts_narrator.renderer.mlx_audio_engine.os.link", side_effect=competing_publication):
            with self.assertRaises(EngineError):
                self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"concurrent accepted recording")
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])

    def test_stable_supported_patch_releases_can_load_and_generate_audio(self):
        for version in ("0.6.1", "0.6.2", "0.6.1.post1"):
            with self.subTest(version=version):
                self.version.return_value = version
                engine = MLXAudioDrama3Engine(self.checkpoint, model=MODEL)
                output = self.root / f"{version}.wav"

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
        self.engine.synthesize("Hello.", self.voice, self.output, seed=1)
        self.engine.close()
        self.mx.clear_cache.assert_called_once_with()
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


FakeModelFrames = FakeModel._frames


if __name__ == "__main__":
    unittest.main()
