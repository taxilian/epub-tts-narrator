"""Character speech may use its own Drama 3 sampling; narration keeps [engine]'s."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.request_sampling import current_sampling
from epub_tts_narrator.renderer.role_sampling import (
    chunk_overrides, dialogue_overrides, effective, is_character_speech, with_overrides)
from epub_tts_narrator.renderer.workflow import (
    SpeechRetryBudget, _chapter_context, _paragraph_request, _render_chunk, _speech_request)
from test_renderer_speech_identity import paragraph_request, reference, speech_request
from test_renderer_speech_recovery_integration import RecoveryVerifier, pcm
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


POOL = {"engine": "fish_tts_pool", "model": "fishaudio/drama-3", "contract_version": 1,
        "model_revision": "r1", "sampling": {"temperature": 0.7, "top_p": 0.7, "top_k": 30,
                                             "max_new_tokens": 2048, "experts": "q8"}}


def config(**dialogue):
    result = deepcopy(DEFAULTS)
    result["dialogue_sampling"].update(dialogue)
    return result


def chunk(kind, text="Hello."):
    row = speech_request(reference("one"), text)["chunk"]
    row["segment"]["kind"] = kind
    return row


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "audiobook-render.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path, require_verifier=False, require_engine=False)

    def test_unset_by_default_and_validated(self):
        self.assertEqual(self.load("")["dialogue_sampling"], {"temperature": None, "top_p": None})
        loaded = self.load("[dialogue_sampling]\ntemperature = 0.95\n")
        self.assertEqual(loaded["dialogue_sampling"], {"temperature": 0.95, "top_p": None})
        for text, message in (("temperature = 0", "temperature must be positive"),
                              ("temperature = 2.5", "at most 2"), ("top_p = 1.5", "at most 1"),
                              ("temperature = true", "temperature must be positive"),
                              ('temperature = "hot"', "temperature must be positive")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.load("[dialogue_sampling]\n" + text + "\n")
        with self.assertRaisesRegex(ValueError, "requires engine.type = drama3_mlx"):
            self.load('[engine]\ntype = "some_plugin"\n[dialogue_sampling]\ntemperature = 0.9\n')

    def test_only_values_that_differ_from_engine_count(self):
        self.assertEqual(dialogue_overrides(config()), {})
        self.assertEqual(dialogue_overrides(config(temperature=0.7, top_p=0.7)), {})
        self.assertEqual(dialogue_overrides(config(temperature=0.95)), {"temperature": 0.95})
        self.assertEqual(dialogue_overrides(config(temperature=0.95, top_p=0.8)),
                         {"temperature": 0.95, "top_p": 0.8})


class SelectionTests(unittest.TestCase):
    def test_character_speech_kinds(self):
        for kind in ("dialogue", "thought", "letter"):
            self.assertTrue(is_character_speech(chunk(kind)), kind)
        for kind in ("narration", "heading", "other"):
            self.assertFalse(is_character_speech(chunk(kind)), kind)

    def test_paragraph_uses_dialogue_sampling_only_when_every_turn_is_character_speech(self):
        one, two = reference("one"), reference("two")
        paragraph = paragraph_request([("Hi.", one, ""), ("Bye.", two, "")])["chunk"]
        settings = config(temperature=0.95)
        self.assertEqual(chunk_overrides(paragraph, settings), {"temperature": 0.95})
        paragraph["paragraph_turns"][1]["segment"]["kind"] = "narration"
        self.assertEqual(chunk_overrides(paragraph, settings), {})

    def test_identity_records_the_requests_sampling(self):
        self.assertEqual(with_overrides(POOL, {})["sampling"], POOL["sampling"])
        changed = with_overrides(POOL, {"temperature": 0.95})
        self.assertEqual(changed["sampling"]["temperature"], 0.95)
        self.assertEqual(POOL["sampling"]["temperature"], 0.7)
        local = {"engine": "drama3_mlx", "temperature": 0.7, "top_p": 0.7}
        self.assertEqual(with_overrides(local, {"top_p": 0.8}), {"engine": "drama3_mlx", "temperature": 0.7,
                                                                 "top_p": 0.8})


class RequestIdentityTests(unittest.TestCase):
    def test_narration_requests_are_unchanged_and_dialogue_requests_differ(self):
        voice = reference("one")
        plain, hot = config(), config(temperature=0.95)
        narration = chunk("narration")
        self.assertEqual(_speech_request(narration, voice, POOL, plain),
                         _speech_request(narration, voice, POOL, hot))
        dialogue = chunk("dialogue")
        request = _speech_request(dialogue, voice, POOL, hot)
        self.assertEqual(request["engine"]["sampling"]["temperature"], 0.95)
        self.assertNotEqual(digest(request), digest(_speech_request(dialogue, voice, POOL, plain)))
        paragraph = paragraph_request([("Hi.", voice, "")])["chunk"]
        self.assertEqual(_paragraph_request(paragraph, {"one": voice}, POOL, hot)["engine"]["sampling"]["temperature"],
                         0.95)

    def test_chapter_context_binds_the_setting_only_when_it_changes_audio(self):
        voice = reference("one")
        bundle = {"bundle_hash": "b", "manifest": {"source_fingerprint": "s"}, "chapters": [{}]}
        chapter = {"id": "c1", "number": 1, "chunks": [chunk("dialogue")]}
        plain = _chapter_context(bundle, chapter, config(), {"one": voice}, POOL, {"v": 1})
        same = _chapter_context(bundle, chapter, config(temperature=0.7), {"one": voice}, POOL, {"v": 1})
        hot = _chapter_context(bundle, chapter, config(temperature=0.95), {"one": voice}, POOL, {"v": 1})
        self.assertEqual(plain["engine_config_hash"], same["engine_config_hash"])
        self.assertNotEqual(plain["engine_config_hash"], hot["engine_config_hash"])


class EffectiveSamplingTests(unittest.TestCase):
    def test_request_values_override_engine_attributes(self):
        from epub_tts_narrator.renderer.request_sampling import request_sampling

        class Engine:
            temperature, top_p, top_k = 0.7, 0.7, 30

        self.assertEqual(effective(Engine()), {"temperature": 0.7, "top_p": 0.7, "top_k": 30})
        with request_sampling({"temperature": 0.95}):
            self.assertEqual(effective(Engine()), {"temperature": 0.95, "top_p": 0.7, "top_k": 30})


class SamplingEngine:
    identity = {"engine": "fish_http", "model": "offline-drama-3", "temperature": 0.7, "top_p": 0.7, "top_k": 30}

    def __init__(self, registry):
        self.registry, self.seen = registry, []

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self.seen.append(current_sampling())
        write_tone(path, frequency=220 + len(self.seen) * 31)
        self.registry[hashlib.sha256(pcm(path)).hexdigest()] = text
        return {"fixture": True, "synthesis_text": text}


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for take integration")
class TakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.registry = {}
        self.engine = SamplingEngine(self.registry)
        self.verifier = RecoveryVerifier(self.registry)
        self.voice = reference("character")
        path = self.work / "character-reference.wav"
        write_tone(path)
        self.voice.update(reference_audio=str(path), reference_sha256=file_hash(path))

    def render(self, row, settings):
        settings["quality"] = validate_policy({})
        settings["render"].update(max_attempts=1, sample_rate=44100, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        result, accepted = _render_chunk(row, 1, 1, {self.voice["voice_id"]: self.voice}, self.engine,
                                         self.engine.identity, self.verifier, settings, self.work, {}, Lock(),
                                         notify=lambda message: None, retry_budget=SpeechRetryBudget())
        return read_json(Path(result["wav_path"]).parent / "accepted.json")

    def test_dialogue_takes_generate_with_their_sampling_and_narration_with_the_engines(self):
        dialogue = speech_request(self.voice, "Run, now.")["chunk"]
        record = self.render(dialogue, config(temperature=0.95))
        self.assertEqual(self.engine.seen, [{"temperature": 0.95}])
        self.assertEqual(record["request"]["engine"]["temperature"], 0.95)
        self.assertIsNone(current_sampling())

        narration = speech_request(self.voice, "The road was long.")["chunk"]
        narration["segment"]["kind"] = "narration"
        record = self.render(narration, config(temperature=0.95))
        self.assertEqual(self.engine.seen[-1], None)
        self.assertEqual(record["request"]["engine"]["temperature"], 0.7)

        # Turning the setting off regenerates the dialogue take at engine sampling.
        self.render(dialogue, config())
        self.assertEqual(len(self.engine.seen), 3)
        self.assertIsNone(self.engine.seen[-1])


if __name__ == "__main__":
    unittest.main()
