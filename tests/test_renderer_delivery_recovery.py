"""Plain-delivery fallback changes acting instructions, never speech coverage."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.delivery_recovery import (
    delivery_marker, plain_delivery_request, validate_delivery_marker,
)
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_resolver import Client, failure


class DeliveryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.voice = reference()
        self.voices = {self.voice["asset_hash"]: self.voice}
        self.request = speech_request(self.voice, "Arrgh! The pain is fading.")
        self.request["chunk"]["segment"]["delivery"]["direction"] = "With a pained, breaking cry"

    def metadata(self, request=None):
        request = request or self.request
        return {"synthesis_text": request["chunk"]["spoken_text"], "direction": "",
                "direction_applied": False, "applied_control": None,
                "delivery_recovery": delivery_marker(request, self.resolution(request))}

    def resolution(self, request=None):
        request = request or self.request
        failures = [failure("The pain is fading.", attempt=number) for number in range(1, 4)]
        audit = SpeechResolver(Client(action="plain_delivery")).resolve(
            request["chunk"]["spoken_text"], failures, can_plain_delivery=True)
        self.assertEqual(audit["decision"], "resolved")
        return {"action": "plain_delivery", "audit": audit, "failures": failures}

    def test_fallback_changes_only_the_copied_delivery_direction(self):
        original = deepcopy(self.request)
        plain = plain_delivery_request(self.request)
        self.assertEqual(self.request, original)
        self.assertEqual(plain["chunk"]["segment"]["delivery"]["direction"], "")
        plain["chunk"]["segment"]["delivery"]["direction"] = original["chunk"]["segment"]["delivery"]["direction"]
        self.assertEqual(plain, original)
        plain["chunk"]["segment"]["delivery"]["evidence"].append({"example": "copied"})
        self.assertEqual(self.request, original)

    def test_fallback_is_not_a_new_text_or_voice_cache_identity(self):
        self.request["synthesis_identity"] = generation_identity(self.request, self.voices)
        expected = generation_identity(self.request, self.voices)
        metadata = self.metadata()
        actual = generation_identity(self.request, self.voices, engine_metadata=metadata)
        self.assertEqual(actual, expected)
        plain = plain_delivery_request(self.request)
        self.assertNotEqual(generation_identity(plain, self.voices), expected)
        self.assertEqual(validate_delivery_marker(self.request, metadata["delivery_recovery"]), plain)

    def test_unaudited_plain_take_cannot_satisfy_a_directed_request(self):
        metadata = self.metadata()
        metadata.pop("delivery_recovery")
        with self.assertRaisesRegex(ValueError, "delivery style"):
            generation_identity(self.request, self.voices, engine_metadata=metadata)

    def test_actual_omission_added_words_or_remaining_style_cannot_use_fallback(self):
        changes = [
            {"synthesis_text": "The pain is fading."},
            {"synthesis_text": "Arrgh! The pain is fading. Thank you."},
            {"synthesis_text": "[With a pained, breaking cry] Arrgh! The pain is fading."},
            {"direction": "With a pained, breaking cry"},
            {"direction_applied": True}, {"applied_control": "style_prefix"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_identity(self.request, self.voices,
                                    engine_metadata={**self.metadata(), **change})

    def test_original_text_voice_direction_and_settings_remain_bound(self):
        original_metadata = self.metadata()
        requests = []
        changed = deepcopy(self.request)
        changed["chunk"]["spoken_text"] = "The pain is fading."
        requests.append(changed)
        changed = deepcopy(self.request)
        changed["voice_asset_hash"] = digest("different voice")
        requests.append(changed)
        changed = deepcopy(self.request)
        changed["chunk"]["segment"]["delivery"]["direction"] = "Whisper"
        requests.append(changed)
        changed = deepcopy(self.request)
        changed["engine"]["temperature"] = 0.7
        requests.append(changed)
        for changed in requests:
            with self.subTest(request=changed), self.assertRaises(ValueError):
                generation_identity(changed, self.voices, engine_metadata=original_metadata)

    def test_marker_cannot_change_policy_or_bypass_the_model_decision(self):
        for change in ({"audit": {}}, {"failures": []},
                       {"policy_sha256": digest("other policy")}, {"replacement_direction": "softly"},
                       {"original_text": "Different words."}, {"extra": "ignored"}):
            metadata = self.metadata()
            metadata["delivery_recovery"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_identity(self.request, self.voices, engine_metadata=metadata)

    def test_plain_delivery_requires_the_explicit_model_action_and_capability(self):
        for change in ({"action": "accept"}, {"audit": {}}, {"failures": []}):
            resolution = {**self.resolution(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                delivery_marker(self.request, resolution)
        failures = [failure("The pain is fading.", attempt=number) for number in range(1, 4)]
        blocked = SpeechResolver(Client(action="plain_delivery")).resolve(
            self.request["chunk"]["spoken_text"], failures, can_plain_delivery=False)
        with self.assertRaises(ValueError):
            delivery_marker(self.request, {"action": "plain_delivery", "audit": blocked,
                                           "failures": failures})

    def test_saved_decision_retains_the_exact_failed_take_evidence(self):
        metadata = self.metadata()
        original = deepcopy(metadata)
        marker = metadata["delivery_recovery"]
        marker["failures"][0]["transcript"] = "A different recognized passage."
        with self.assertRaises(ValueError):
            generation_identity(self.request, self.voices, engine_metadata=metadata)
        self.assertEqual(generation_identity(self.request, self.voices, engine_metadata=original),
                         generation_identity(self.request, self.voices))

    def test_unknown_engines_paragraphs_assembly_and_normalized_styles_are_excluded(self):
        requests = []
        plain = deepcopy(self.request)
        plain["chunk"]["segment"]["delivery"]["direction"] = ""
        requests.append(plain)
        requests.append({**self.request, "engine": {"engine": "unknown_plugin"}})
        requests.append({**self.request, "kind": "paragraph_speech"})
        requests.append({**self.request, "recovery": {"strategy": "sentence_split"}})
        altered = deepcopy(self.request)
        altered["chunk"]["segment"]["speech_normalization"] = {"version": 1, "edits": [
            {"direction": "Sustain the vowel"}]}
        requests.append(altered)
        altered = deepcopy(self.request)
        altered["chunk"]["spoken_text"] = "[screams] Arrgh!"
        requests.append(altered)
        for request in requests:
            with self.subTest(request=request):
                self.assertIsNone(plain_delivery_request(request))
                with self.assertRaises(ValueError):
                    delivery_marker(request, self.resolution())

    def test_recovery_child_preserves_its_source_and_synthesis_binding(self):
        text = self.request["chunk"]["spoken_text"]
        child = {"kind": "speech_recovery_part", "version": 1,
                 "parent_request_hash": digest("parent"), "recovery_request_hash": digest("recovery"),
                 "part": {"start": 12, "end": 12 + len(text), "text": text},
                 "synthesis_request": deepcopy(self.request)}
        resolution = self.resolution()
        marker = delivery_marker(child, resolution)
        plain = validate_delivery_marker(child, marker)
        self.assertEqual(plain["part"], child["part"])
        self.assertEqual(plain["synthesis_request"], plain_delivery_request(self.request))
        self.assertEqual(marker, delivery_marker(self.request, resolution))
        # Parent assembly rechecks the child's actual synthesis request.
        metadata = {**self.metadata(), "delivery_recovery": marker}
        self.assertEqual(generation_identity(child["synthesis_request"], self.voices, engine_metadata=metadata),
                         generation_identity(self.request, self.voices))
        changed = deepcopy(child)
        changed["part"]["text"] = "Other text."
        self.assertIsNone(plain_delivery_request(changed))
        del changed["synthesis_request"]
        self.assertIsNone(plain_delivery_request(changed))

    def test_delivery_fallback_composes_with_independently_reviewed_pronunciation(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/speech-rewrite-policy-v1.json").read_text())
        request = deepcopy(self.request)
        request["chunk"]["spoken_text"] = fixture["expected_text"]
        rewrite = {"version": 1, "original_text": fixture["expected_text"],
                   "tts_text": "Zekra waited outside.", "audit": fixture["audit"],
                   "failures": fixture["failures"]}
        metadata = {**self.metadata(request), "synthesis_text": rewrite["tts_text"],
                    "speech_resolution": rewrite}
        self.assertEqual(generation_identity(request, self.voices, engine_metadata=metadata),
                         generation_identity(request, self.voices))
        corrupted = deepcopy(metadata)
        corrupted["speech_resolution"]["tts_text"] = "Zekra waited."
        corrupted["synthesis_text"] = "Zekra waited."
        with self.assertRaises(ValueError):
            generation_identity(request, self.voices, engine_metadata=corrupted)
        without_rewrite = deepcopy(metadata)
        del without_rewrite["speech_resolution"]
        with self.assertRaises(ValueError):
            generation_identity(request, self.voices, engine_metadata=without_rewrite)


if __name__ == "__main__":
    unittest.main()
