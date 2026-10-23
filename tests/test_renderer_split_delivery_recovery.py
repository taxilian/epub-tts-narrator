"""Sentence recovery retains a verified plain-delivery decision through assembly."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.speech_identity import generation_identity, single_speech_recovery_part
from epub_tts_narrator.renderer.speech_recovery import _legacy_accepted_part, recover_speech, split_recovery_text
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, SpeechVerificationFailed
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery import AudioVerifier
from test_renderer_speech_resolver import Client
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine


TEXT = "Arrgh! The pain is fading. We should leave now."


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class SplitDeliveryRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.voice = reference()
        self.voices = {self.voice["asset_hash"]: self.voice}
        self.request = speech_request(self.voice, TEXT)
        self.direction = "With a pained, breaking cry"
        self.request["chunk"]["segment"]["delivery"]["direction"] = self.direction
        self.request["synthesis_identity"] = generation_identity(self.request, self.voices)
        self.parent = self.work / "clips" / digest(self.request)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE,
                                     keep_intermediate_audio=True)
        self.config["quality"]["max_wer"] = 0.0
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.asr = AudioVerifier(self.fixtures)
        self.client = Client(action="plain_delivery")
        self.verifier = RecurringSpeechVerifier(self.asr, SpeechResolver(self.client))
        self.calls, self.messages = [], []
        self.fail_directed = True
        self.interrupt_last = False

    def generate(self, text, path, seed):
        self.calls.append(("directed", text))
        if self.interrupt_last and text.startswith("We should"):
            raise KeyboardInterrupt()
        heard = text.replace("Arrgh! ", "") if self.fail_directed else text
        metadata = self.engine._write(heard, path, seed)
        directed, delivery = _directed_text(text.strip(), self.direction)
        return {**metadata, "synthesis_text": directed, **delivery}

    def plain(self, text, path, seed):
        self.calls.append(("plain", text))
        metadata = self.engine._write(text, path, seed)
        directed, delivery = _directed_text(text.strip(), "")
        return {**metadata, "synthesis_text": directed, **delivery}

    def recover(self, *, plain=True):
        return recover_speech(TEXT, self.generate, self.parent, self.request, self.verifier, self.config,
                              notify=self.messages.append, retry_budget=SpeechRetryBudget(),
                              voices_by_asset=self.voices, synthesize_plain=self.plain if plain else None)

    def test_exact_leaf_binding_drops_stale_identity_but_preserves_voice_and_source(self):
        part = split_recovery_text(TEXT)[0]
        leaf = single_speech_recovery_part(self.request, part)
        self.assertNotIn("synthesis_identity", leaf)
        self.assertEqual(leaf["chunk"]["spoken_text"], part["text"])
        self.assertEqual(leaf["voice_asset_hash"], self.voice["asset_hash"])
        self.assertEqual(leaf["chunk"]["segment"], self.request["chunk"]["segment"])
        with self.assertRaises(ValueError):
            single_speech_recovery_part(self.request, {**part, "text": "Different words."})

    def test_model_selected_plain_child_is_verified_bound_and_reused(self):
        wav, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        self.assertEqual(sum(kind == "plain" for kind, _ in self.calls), 1)
        self.assertEqual(sum(kind == "directed" for kind, _ in self.calls), 4)
        audit = accepted["engine_metadata"]["recovery"]
        self.assertEqual(audit["child_acceptance_version"], 1)
        first = audit["parts"][0]["acceptance"]
        self.assertIn("delivery_recovery", first["engine_metadata"])
        self.assertEqual(first["request"]["synthesis_request"]["voice_asset_hash"], self.voice["asset_hash"])
        self.assertEqual(generation_identity(accepted["request"], self.voices,
                                            engine_metadata=accepted["engine_metadata"]),
                         generation_identity(self.request, self.voices))
        calls = list(self.calls)
        self.assertEqual(self.recover(), (wav, accepted))
        self.assertEqual(self.calls, calls)

    def test_existing_accepted_legacy_child_keeps_its_request_and_waveform(self):
        self.fail_directed, self.interrupt_last = False, True
        with self.assertRaises(KeyboardInterrupt):
            self.recover(plain=False)
        saved = next((self.work / "clips").glob("*/parts/*/accepted.json"))
        previous = saved.read_bytes()
        legacy = read_json(saved)
        self.assertNotIn("synthesis_request", legacy["request"])
        self.interrupt_last = False
        _, accepted = self.recover()
        self.assertEqual(saved.read_bytes(), previous)
        first = accepted["engine_metadata"]["recovery"]["parts"][0]["acceptance"]
        self.assertEqual(first, legacy)
        self.assertEqual(sum(text.startswith("Arrgh") for _, text in self.calls), 1)
        self.assertEqual(generation_identity(accepted["request"], self.voices,
                                            engine_metadata=accepted["engine_metadata"]),
                         generation_identity(self.request, self.voices))

    def test_joined_audio_still_requires_complete_transcription(self):
        self.asr.final_transcript = "Arrgh! The pain is fading. We leave now."
        with self.assertRaisesRegex(SpeechVerificationFailed, "final full-text verification"):
            self.recover()
        self.assertEqual(sum(kind == "plain" for kind, _ in self.calls), 1)
        marker = read_json(self.parent / "speech-recovery.json")
        final = self.parent.parent / marker["recovery_request_hash"]
        self.assertFalse((final / "accepted.json").exists())

    def test_legacy_passed_flag_cannot_hide_changed_source_request_or_audio(self):
        self.fail_directed, self.interrupt_last = False, True
        with self.assertRaises(KeyboardInterrupt):
            self.recover(plain=False)
        saved = next((self.work / "clips").glob("*/parts/*/accepted.json"))
        original = read_json(saved)
        self.assertTrue(_legacy_accepted_part(saved.parent, original["request"]))
        for change in ("request", "source", "audio"):
            altered = deepcopy(original)
            if change == "request":
                altered["request"]["part"]["text"] = "Different words."
            elif change == "source":
                altered["report"]["expected_text"] = "Different words."
            else:
                altered["audio_sha256"] = digest("different bytes")
            write_json(saved, altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _legacy_accepted_part(saved.parent, original["request"])
        write_json(saved, original)

    def test_rehashed_child_cannot_change_voice_text_or_neutral_decision(self):
        _, accepted = self.recover()
        for change in ("voice", "text", "marker", "waveform"):
            metadata = deepcopy(accepted["engine_metadata"])
            evidence = metadata["recovery"]["parts"][0]
            child = evidence["acceptance"]
            if change == "voice":
                child["request"]["synthesis_request"]["voice_asset_hash"] = digest("another voice")
            elif change == "text":
                child["request"]["synthesis_request"]["chunk"]["spoken_text"] = "The pain is fading."
            elif change == "marker":
                del child["engine_metadata"]["delivery_recovery"]
            else:
                child["audio_sha256"] = digest("different audio")
            evidence["accepted_sha256"] = hashlib.sha256(
                (json.dumps(child, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_identity(accepted["request"], self.voices, engine_metadata=metadata)
