import unittest

from epub_tts_narrator.renderer.performance_compiler import (
    PERFORMANCE_CAPABILITIES, compile_performance, validate_compiled_text,
)
from test_performance_plan import delivery, pronunciation, record


class PerformanceCompilerTests(unittest.TestCase):
    def test_habitual_traits_do_not_turn_whispering_into_generic_style(self):
        text = "Keep this secret."
        result = compile_performance(text, record(text), traits="Precise consonants", direction="whispering")
        self.assertEqual(result["text"], "[Precise consonants] <whisper>Keep this secret.</whisper>")
        self.assertEqual(result["expected_text"], text)

    def test_paired_direction_changes_only_the_requested_words(self):
        text = "It may explode."
        result = compile_performance(text, record(text, delivery(text, 7, 14, tag="emphasis")))
        self.assertEqual(result["text"], "It may <emphasis>explode</emphasis>.")
        self.assertEqual(result["warnings"], [])

    def test_all_documented_paired_controls_compile_without_losing_lexical_text(self):
        text = "Come here."
        for tag in PERFORMANCE_CAPABILITIES["drama3"]["scoped_delivery"]:
            with self.subTest(tag=tag):
                result = compile_performance(text, record(text, delivery(text, tag=tag)))
                self.assertEqual(result["text"], f"<{tag}>{text}</{tag}>")
                self.assertEqual(result["expected_text"], text)

    def test_scoped_style_reset_is_explicitly_reported_as_approximate(self):
        text = "Yes. Now listen."
        result = compile_performance(text, record(text, delivery(text, 0, 4, tag="style", direction="dry sarcasm")), traits="Slow cadence")
        self.assertIn("[dry sarcasm] Yes. [Slow cadence] ", result["text"])
        self.assertIn("scoped_style_reset_is_approximate", result["warnings"])
        self.assertEqual(result["expected_text"], text)

    def test_phonemes_replace_only_bound_word_inside_paired_delivery(self):
        text = "Jas, come here."
        result = compile_performance(text, record(text, delivery(text), pronunciation(text, 0, 3)))
        self.assertEqual(result["text"], "<whisper><|phoneme_start|>JH AE1 S<|phoneme_end|>, come here.</whisper>")
        self.assertEqual(result["expected_text"], text)
        self.assertEqual(result["verification"]["lexical_text"], text)

    def test_pause_and_vocalization_do_not_become_expected_spoken_words(self):
        text = "All right."
        result = compile_performance(text, record(text,
            {"kind": "pause", "offset": 0, "duration_ms": 500},
            {"kind": "vocalization", "offset": 4, "sound": "sigh"},
            {"kind": "pause", "offset": len(text), "duration_ms": 1000}))
        self.assertEqual(result["expected_text"], text)
        self.assertIn("[pause for 0.5 seconds]", result["text"])
        self.assertIn("[sigh]", result["text"])
        self.assertEqual(result["verification"]["requested_pause_ms"], 1500)
        self.assertTrue(result["verification"]["preserve_leading_silence"])
        self.assertTrue(result["verification"]["preserve_trailing_silence"])
        self.assertEqual(result["warnings"], ["pause_duration_is_approximate"])

    def test_readable_respelling_changes_model_input_but_keeps_canonical_text_and_controls(self):
        text = "Qarm, come here."
        control = {"kind": "pronunciation", "start": 0, "end": 4, "text": "Qarm",
                   "alphabet": "respelling", "replacement": "Quarm"}
        result = compile_performance(text, record(text, delivery(text), control))
        self.assertEqual(result["text"], "<whisper>Quarm, come here.</whisper>")
        self.assertEqual(result["expected_text"], text)
        self.assertEqual(result["verification"]["lexical_text"], text)
        self.assertEqual(result["applied_controls"][-1], control)

    def test_adjacent_scopes_and_boundary_events_stay_balanced(self):
        text = "go stop"
        result = compile_performance(text, record(text, delivery(text, 0, 3), delivery(text, 3, 7, tag="forceful"),
            {"kind": "vocalization", "offset": 3, "sound": "breath"}))
        self.assertEqual(result["text"], "<whisper>go </whisper> [breath] <forceful>stop</forceful>")

    def test_plain_backend_reports_every_unsupported_control(self):
        text = "Go."
        result = compile_performance(text, record(text, delivery(text)), direction="calm", backend="plain")
        self.assertEqual(result["text"], text)
        self.assertEqual(result["applied_controls"], [])
        self.assertEqual(result["warnings"], ["plain_backend_unsupported_direction", "plain_backend_unsupported_delivery"])

    def test_book_markup_must_not_pass_the_literal_text_boundary(self):
        for text in ("[laugh] hello", "<whisper>hello</whisper>", "<|speaker:1|>hello", "<|phoneme_start|>JH AE1 S<|phoneme_end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                compile_performance(text, record(text))

    def test_wire_validator_rejects_unknown_tokens_unbalanced_scopes_and_bad_phones(self):
        for text in ("<|speaker:1|>hello", "<unknown>hello</unknown>", "<whisper>hello", "[hi[hello]",
                     "<whisper><soft>hello</whisper></soft>", "<|phoneme_end|>", "<|phoneme_start|>bad<|phoneme_end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_compiled_text(text)

    def test_no_standalone_event_support_is_claimed_or_enabled(self):
        self.assertFalse(PERFORMANCE_CAPABILITIES["drama3"]["standalone_vocal_events"])
        with self.assertRaises(ValueError):
            compile_performance("...", record("...", {"kind": "vocalization", "offset": 0, "sound": "sigh"}))
