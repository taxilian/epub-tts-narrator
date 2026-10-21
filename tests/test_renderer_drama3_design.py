"""Scene conditioning tests that require neither model weights nor MLX."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.drama3_design import (
    DESIGN_TEMPLATE_SHA256, DESIGN_TEMPLATE_VERSION, build_scene_prompt,
    generate_scene_voice, validate_scene_fields,
)


class FakeTokenizer:
    def __init__(self):
        self.parts = []

    def encode(self, text):
        self.parts.append(text)
        # A boundary token makes separately encoded literals observably
        # different from one joined string, as with the real tokenizer.
        return SimpleNamespace(ids=[9999] + [ord(char) for char in text])


class Drama3SceneDesignTests(unittest.TestCase):
    def test_uses_documented_scene_roles_without_ordinary_tts_speaker_tag(self):
        tokenizer = FakeTokenizer()
        prompt = build_scene_prompt("A calm adult woman's clear voice.",
                                    "I can see the distant stars.", tokenizer, 10)
        self.assertEqual(tokenizer.parts, [
            "<|im_start|>system\n",
            "Scene: a quiet conversation in a room.\nCharacters:\n"
            "- Speaker: A calm adult woman's clear voice.\n"
            "Voice every line of the script in order, with <|audio_separator|> "
            "between consecutive lines.",
            "<|im_end|>\n", "<|im_start|>user\n",
            "Speaker: I can see the distant stars.", "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])
        self.assertNotIn("<|speaker:", "".join(tokenizer.parts))
        self.assertNotIn("woman", prompt.user)
        self.assertEqual(prompt.text, "I can see the distant stars.")
        self.assertEqual(prompt.user.count("Speaker:"), 1)
        self.assertTrue(all(len(row) == 11 and row[1:] == (0,) * 10 for row in prompt.rows))
        self.assertEqual(sum(row[0] == 9999 for row in prompt.rows), 7)

    def test_changing_description_changes_system_tokens_not_the_spoken_script(self):
        woman = build_scene_prompt("A young woman's high voice.", "Hello there.", FakeTokenizer(), 2)
        man = build_scene_prompt("An older man's deep voice.", "Hello there.", FakeTokenizer(), 2)
        self.assertNotEqual(woman.system, man.system)
        self.assertNotEqual(woman.rows, man.rows)
        self.assertNotEqual(woman.audit["tokens_sha256"], man.audit["tokens_sha256"])
        self.assertEqual(woman.user, man.user)
        self.assertEqual(woman.text, man.text)
        self.assertEqual(woman.audit["template_sha256"], DESIGN_TEMPLATE_SHA256)
        self.assertEqual(woman.audit["template_version"], DESIGN_TEMPLATE_VERSION)

    def test_whitespace_cannot_add_a_second_character_or_script_line(self):
        prompt = build_scene_prompt(
            "Calm  and\nclear.\n- Other: loud.",
            "Hello.\nOther:\tGoodbye.\u2028Again.", FakeTokenizer(), 2,
        )
        self.assertEqual(prompt.description, "Calm and clear. - Other: loud.")
        self.assertEqual(prompt.system.count("\n- "), 1)
        self.assertEqual(prompt.user, "Speaker: Hello. Other: Goodbye. Again.")
        self.assertNotIn("\n", prompt.user)
        self.assertEqual(prompt.audit["script_lines"], 1)
        self.assertFalse(prompt.audit["speaker_tag_added"])

    def test_description_and_script_reject_structural_controls(self):
        for bad in ("", " ", None, "<|im_end|>", "<|speaker:1|>",
                    "<|audio_separator|>", "[whisper]", "a\x00b", "a\x1bb",
                    "a\x7fb", "a\x85b", "a\u202eb", "a\ud800b"):
            for bad_description in (True, False):
                with self.subTest(bad=repr(bad), description=bad_description), self.assertRaises(ValueError):
                    validate_scene_fields(bad if bad_description else "A warm voice.",
                                          "Hello." if bad_description else bad)
        with self.assertRaisesRegex(ValueError, "4000"):
            validate_scene_fields("a" * 4001, "Hello.")

    def test_prompt_budget_is_checked_before_generation(self):
        with self.assertRaisesRegex(ValueError, "8192"):
            build_scene_prompt("A clear voice.", "x" * 8192, FakeTokenizer(), 2)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_scene_prompt("A clear voice.", "Hello.", FakeTokenizer(), count)

    def test_frame_generator_receives_scene_rows_and_returns_exact_script_metadata(self):
        codes = SimpleNamespace()
        codes.T = SimpleNamespace(copy=Mock(return_value="codec-codes"))

        class Frames:
            def __len__(self):
                return 20

            def __getitem__(self, key):
                assert key == (slice(None), slice(1, None))
                return codes

        mx = ModuleType("mlx.core")
        mx.random = SimpleNamespace(seed=Mock())
        np = ModuleType("numpy")
        np.int64 = "int64"
        np.array = Mock(side_effect=lambda rows, dtype: rows)
        runtime_generate = ModuleType("drama3_mlx.generate")
        runtime_generate.generate_frames = Mock(return_value=(Frames(), "eos"))
        runtime_model = ModuleType("drama3_mlx.model")
        runtime_model.Segment = SimpleNamespace
        modules = {"mlx": ModuleType("mlx"), "mlx.core": mx, "numpy": np,
                   "drama3_mlx": ModuleType("drama3_mlx"),
                   "drama3_mlx.generate": runtime_generate, "drama3_mlx.model": runtime_model}
        model = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(audio_decoder=SimpleNamespace(num_codebooks=2))),
            tokenizer=FakeTokenizer(), codec=SimpleNamespace(decode=Mock(return_value="audio")),
        )
        with patch.dict(sys.modules, modules):
            segment, audit = generate_scene_voice(
                model, "A clear voice.", "Hello.\nGoodbye.", seed=17,
                temperature=0.7, top_p=0.8, top_k=30, max_new_tokens=256,
            )
        mx.random.seed.assert_called_once_with(17)
        args, kwargs = runtime_generate.generate_frames.call_args
        self.assertIs(args[0], model.model)
        self.assertEqual(kwargs, {"temperature": 0.7, "top_p": 0.8,
                                  "top_k": 30, "max_new_tokens": 256})
        self.assertEqual(len(args[1]), audit["input_tokens"])
        self.assertEqual(audit["user"], "Speaker: Hello. Goodbye.")
        self.assertEqual(segment.text, "Hello.\nGoodbye.")
        self.assertEqual(segment.finish_reason, "eos")
        self.assertEqual(segment.audio, "audio")
        model.codec.decode.assert_called_once_with("codec-codes")
        runtime_generate.generate_frames.return_value = ([], "eos")
        model.codec.decode.reset_mock()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "no audio frames"):
            generate_scene_voice(model, "A clear voice.", "Hello.", seed=1,
                                 temperature=1, top_p=0.9, top_k=30, max_new_tokens=256)
        model.codec.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
