"""The adapter/cache boundary agrees on performance without loading a model."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from epub_tts_narrator.renderer.engines import Drama3MLXEngine, EngineError, format_dialogue, prepare_synthesis
from epub_tts_narrator.renderer.fish_http import FishHTTPEngine, trim_wire_text, validate_wire_text
from epub_tts_narrator.renderer.mlx_audio_engine import MLXAudioDrama3Engine
from epub_tts_narrator.renderer.speech_identity import generation_identity, paragraph_rewrite_turn
from test_performance_plan import delivery, pronunciation, record
from test_renderer_speech_identity import paragraph_request, reference, speech_request


class PerformanceAdapterTests(unittest.TestCase):
    def test_noncontiguous_formatting_trim_projects_control_offsets_exactly(self):
        text = ' "(I thought fiercely)." '
        prepared, directed, metadata = prepare_synthesis(text, performance=record(text, delivery(text)))
        self.assertEqual(prepared, "I thought fiercely.")
        self.assertEqual(directed, "<whisper>I thought fiercely.</whisper>")
        self.assertEqual(metadata["expressive_performance"]["plan"]["controls"][0]["text"], prepared)

    def test_new_adapters_use_one_compiler_and_preserve_its_audit(self):
        text = ' "Jas, stay." '
        plan = record(text, delivery(text), pronunciation(text, 2, 5))
        voice = reference()
        expected = prepare_synthesis(text, performance=plan, traits="Measured consonants")[1]
        sogni = object.__new__(Drama3MLXEngine)
        sogni._reference = Mock(return_value="encoded-reference")
        sogni._generate_audio = Mock(return_value={"synthesis_text": expected})
        mlx = object.__new__(MLXAudioDrama3Engine)
        mlx._generate_audio = Mock(return_value={"synthesis_text": expected})
        fish = object.__new__(FishHTTPEngine)
        fish._request = Mock(return_value={})
        for engine in (sogni, mlx, fish):
            with self.subTest(engine=type(engine).__name__), patch(
                    "epub_tts_narrator.renderer.fish_http.read_reference", return_value=(b"reference", "Reference text.")):
                result = engine.synthesize(text, voice, Path("/unused/output.wav"), seed=42,
                                           performance=plan, traits="Measured consonants")
                self.assertEqual(result["synthesis_text"], expected)
                self.assertEqual(result["expressive_performance"]["expected_text"], "Jas, stay.")
                call = engine._request if engine is fish else engine._generate_audio
                self.assertEqual(call.call_args.args[0], expected)

    def test_stale_plan_is_rejected_before_reference_encoding(self):
        engine = object.__new__(Drama3MLXEngine)
        engine._reference = Mock()
        with self.assertRaises(EngineError):
            engine.synthesize("Changed text.", reference(), Path("/unused"), seed=1, performance=record("Original text."))
        engine._reference.assert_not_called()

    def test_legacy_cache_identity_remains_the_existing_shape(self):
        voice = reference()
        request = speech_request(voice, "Hello.")
        result = generation_identity(request, {voice["asset_hash"]: voice})
        self.assertEqual(set(result), {"version", "kind", "engine", "base_seed", "sample_rate", "text", "voice"})
        self.assertEqual(result["text"], "Hello.")
        self.assertEqual(prepare_synthesis(' "Hello." ', "whisper")[1], "<whisper>Hello.</whisper>")

    def test_new_identity_requires_exact_compiled_audit_but_ignores_evidence(self):
        voice = reference()
        text = "Jas spoke."
        request = speech_request(voice, text)
        request["chunk"].update(expressive_performance=record(text, delivery(text)),
                                expressive_traits={voice["voice_id"]: "Precise consonants"},
                                expressive_direction="")
        voices = {voice["asset_hash"]: voice}
        result = generation_identity(request, voices)
        self.assertEqual(result["text"], "[Precise consonants] <whisper>Jas spoke.</whisper>")
        _, directed, metadata = prepare_synthesis(text, performance=request["chunk"]["expressive_performance"], traits="Precise consonants")
        metadata["synthesis_text"] = directed
        self.assertEqual(result, generation_identity(request, voices, engine_metadata=metadata))
        reviewed = deepcopy(request)
        reviewed["chunk"]["expressive_performance"]["controls"][0]["evidence"] = [{"block_id": "b2", "quote": "whispered"}]
        self.assertEqual(result, generation_identity(reviewed, voices))
        metadata["expressive_performance"]["traits"] = "Different traits"
        with self.assertRaisesRegex(ValueError, "expressive performance"):
            generation_identity(request, voices, engine_metadata=metadata)

    def test_paragraph_cache_and_formatter_bind_each_scoped_turn(self):
        voice = reference()
        request = paragraph_request([("Quiet. ", voice, ""), ("Loud.", voice, "")])
        for chunk, tag in zip(request["chunk"]["paragraph_turns"], ("whisper", "forceful")):
            text = chunk["spoken_text"]
            chunk.update(expressive_performance=record(text, delivery(text, tag=tag)), expressive_direction="")
        turns = [{"text": c["spoken_text"], "voice": voice, "performance": c["expressive_performance"]}
                 for c in request["chunk"]["paragraph_turns"]]
        directed, _, metadata = format_dialogue(turns)
        metadata["synthesis_text"] = directed
        result = generation_identity(request, {voice["asset_hash"]: voice}, engine_metadata=metadata)
        self.assertEqual(result["text"], directed)
        self.assertIn("<whisper>Quiet.</whisper>", directed)
        self.assertIn("<forceful>Loud.</forceful>", directed)
        self.assertEqual(len(metadata["turn_directions"]), 2)
        self.assertIsNone(paragraph_rewrite_turn(request, {voice["asset_hash"]: voice}))

    def test_unchanged_compiled_input_reuses_legacy_generation_identity(self):
        voice = reference()
        old = speech_request(voice, "Hello.")
        new = deepcopy(old)
        new["chunk"].update(expressive_performance=record("Hello."), expressive_direction="")
        voices = {voice["asset_hash"]: voice}
        self.assertEqual(generation_identity(new, voices), generation_identity(old, voices))
        old["chunk"]["segment"]["delivery"]["direction"] = "whisper"
        new["chunk"]["expressive_performance"] = record("Hello.", delivery("Hello."))
        self.assertEqual(generation_identity(new, voices), generation_identity(old, voices))

    def test_wire_preserves_paired_controls_and_phonemes_without_permitting_speaker_injection(self):
        valid = "<|speaker:0|> <emphasis><|phoneme_start|>JH AE1 S<|phoneme_end|></emphasis>, stay."
        validate_wire_text(valid, {0})
        self.assertEqual(trim_wire_text(valid, {0}), valid)
        for invalid in (valid.replace("speaker:0", "speaker:1"),
                        "<|speaker:0|> <whisper>Stay.\n<|speaker:1|> Go.</whisper>",
                        "<|speaker:0|> <|phoneme_start|>AE3<|phoneme_end|>"):
            with self.subTest(invalid=invalid), self.assertRaises(EngineError):
                validate_wire_text(invalid, {0})

    def test_wire_preserves_inline_scope_whitespace_and_punctuation_exactly(self):
        for tag in ("whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"):
            for text, start, end in (("Meet Jas now.", 5, 9), ("Meet Jas now.", 4, 8),
                                     ('He said "stay" quietly.', 8, 14)):
                with self.subTest(tag=tag, start=start, end=end):
                    plan = record(text, delivery(text, start, end, tag=tag))
                    compiled = prepare_synthesis(text, performance=plan)[1]
                    self.assertEqual(trim_wire_text(compiled, {None}), compiled)
                    dialogue = "<|speaker:0|> " + compiled
                    self.assertEqual(trim_wire_text(dialogue, {0}), dialogue)
        with self.assertRaisesRegex(EngineError, "nested too deeply"):
            trim_wire_text("<whisper>" * 40 + "Speech." + "</whisper>" * 40, {None})
