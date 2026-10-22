"""Uniform paragraphs may use audited spellings without changing source timing."""

from copy import deepcopy
from pathlib import Path
from threading import Lock
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import format_dialogue
from epub_tts_narrator.renderer.paragraphs import _group
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_identity import generation_identity, paragraph_rewrite_turn
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import _render_chunk
from test_renderer_recurring_resolution import PlanClient, ResolutionEngine, TEXT
from test_renderer_speech_identity import paragraph_request, reference
from test_renderer_speech_recovery_integration import RecoveryVerifier
from test_renderer_workflow import FFMPEG, FFPROBE


class ParagraphPronunciationGuardTests(unittest.TestCase):
    def setUp(self):
        self.voice = reference()
        self.voices = {self.voice["asset_hash"]: self.voice}

    def test_only_exactly_equivalent_uniform_paragraphs_are_eligible(self):
        request = paragraph_request([("Hello, ", self.voice, "whisper"),
                                     ("stay here.", self.voice, "whisper")])
        before = deepcopy(request)
        turn = paragraph_rewrite_turn(request, self.voices)
        self.assertEqual(turn, {"text": "Hello, stay here.", "voice": self.voice, "direction": "whisper"})
        self.assertEqual(request, before)
        other = reference("other")
        variants = [
            paragraph_request([("Hello, ", self.voice, ""), ("stay here.", other, "")]),
            paragraph_request([("Hello, ", self.voice, ""), ("stay here.", self.voice, "whisper")]),
            # Unsupported direction wording is still kept distinct conservatively.
            paragraph_request([("Hello, ", self.voice, "calm"), ("stay here.", self.voice, "kindly")]),
            # Trimming each turn removes interior quotes; collapsing must not.
            paragraph_request([("Hello, ", self.voice, ""), ('"stay here."', self.voice, "")]),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertIsNone(paragraph_rewrite_turn(variant, {**self.voices, other["asset_hash"]: other}))


class ParagraphResolutionEngine(ResolutionEngine):
    identity = {"engine": "fish_http", "model": "offline-paragraph-pronunciation"}

    def synthesize_dialogue(self, turns, path, *, seed):
        self.dialogue_calls.append(deepcopy(turns))
        metadata = self._write("".join(turn["text"] for turn in turns), path, seed)
        formatted, _, formatting = format_dialogue(turns)
        return {**metadata, "synthesis_text": formatted, **formatting}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class ParagraphPronunciationIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.config["quality"] = validate_policy({})
        self.config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        registry = {}
        self.engine = ParagraphResolutionEngine(registry)
        self.base = RecoveryVerifier(registry)
        self.client = PlanClient([{"action": "rewrite", "reason": "Try a pronunciation spelling.",
                                  "edits": [{"start": 0, "end": 6, "source": "Insang", "replacement": "Insaan"}]}])
        self.verifier = RecurringSpeechVerifier(self.base, SpeechResolver(self.client))
        self.voice = reference("voice")
        boundary = TEXT.index(" while")
        turns = []
        for index, (start, end) in enumerate(((0, boundary), (boundary, len(TEXT)))):
            text = TEXT[start:end]
            row = {"id": f"segment-{index}", "chapter_id": "chapter-0001", "block_id": "b000001",
                   "start": start, "end": end, "text": text, "spoken_text": text,
                   "role_id": "character", "voice_id": "voice", "delivery": {"direction": "", "evidence": []}}
            turns.append({"chunk_id": digest(index)[:32], "spoken_text": text, "spoken_start": 0,
                          "spoken_end": len(text), "silent_control": False, "segment": row,
                          "source_ranges": [{"start": start, "end": end}]})
        self.chunk = _group(turns)[0]
        self.messages = []

    def render(self):
        return _render_chunk(self.chunk, 1, 1, {"voice": self.voice}, self.engine,
                             self.engine.identity, self.verifier, self.config, self.work,
                             {}, Lock(), notify=self.messages.append)

    def test_audited_uniform_paragraph_rewrite_preserves_source_and_reuses_cache(self):
        original = deepcopy(self.chunk)
        result, accepted = self.render()
        self.assertEqual(self.chunk, original)
        self.assertEqual(result, {**original, "wav_path": result["wav_path"]})
        self.assertEqual([len(turns) for turns in self.engine.dialogue_calls], [2, 2, 2, 1])
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["expected_text"], TEXT)
        saved = read_json(Path(result["wav_path"]).parent / "accepted.json")
        self.assertEqual(saved["request"]["chunk"], original)
        metadata = saved["engine_metadata"]
        expected, _, formatting = format_dialogue(self.engine.dialogue_calls[-1])
        self.assertEqual(metadata["synthesis_text"], expected)
        self.assertTrue(all(metadata[key] == value for key, value in formatting.items()))
        voices = {self.voice["asset_hash"]: self.voice}
        self.assertEqual(generation_identity(saved["request"], voices, engine_metadata=metadata),
                         generation_identity(saved["request"], voices))
        counts = len(self.engine.dialogue_calls), len(self.base.calls), len(self.client.plans)
        self.assertEqual(self.render(), (result, accepted))
        self.assertEqual((len(self.engine.dialogue_calls), len(self.base.calls), len(self.client.plans)), counts)

        # Neither a valid audit nor an accepted report permits different actual
        # synthesis text, voice controls, or grouping metadata to enter the cache.
        for field, value in (("synthesis_text", expected + " Extra invented words."),
                             ("turn_count", 2), ("turn_directions", [])):
            altered = deepcopy(metadata)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                generation_identity(saved["request"], voices, engine_metadata=altered)
        altered = deepcopy(saved["request"])
        altered["chunk"]["paragraph_turns"][1]["segment"]["delivery"]["direction"] = "whisper"
        with self.assertRaises(ValueError):
            generation_identity(altered, voices, engine_metadata=metadata)
        other = reference("other")
        altered = deepcopy(saved["request"])
        altered["voice_assets"]["other"] = other["asset_hash"]
        altered["chunk"]["paragraph_turns"][1]["segment"]["voice_id"] = "other"
        with self.assertRaises(ValueError):
            generation_identity(altered, {**voices, other["asset_hash"]: other}, engine_metadata=metadata)
        altered = deepcopy(metadata)
        altered["speech_resolution"]["tts_text"] += " Invented dialogue."
        with self.assertRaises(ValueError):
            generation_identity(saved["request"], voices, engine_metadata=altered)


if __name__ == "__main__":
    unittest.main()
