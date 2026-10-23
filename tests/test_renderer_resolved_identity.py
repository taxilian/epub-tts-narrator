"""A pronunciation rewrite never becomes a generic cache-text override."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.speech_identity import generation_identity
from test_renderer_speech_identity import paragraph_request, reference, speech_request


class ResolvedSpeechIdentityTests(unittest.TestCase):
    def setUp(self):
        self.voice = reference()
        self.voices = {self.voice["asset_hash"]: self.voice}
        self.original = "Vandire is waiting here."
        self.rewritten = "Van dyer is waiting here."
        self.request = speech_request(self.voice, self.original)
        self.proposal = {"action": "rewrite", "tts_text": self.rewritten}
        self.metadata = {
            "synthesis_text": self.rewritten,
            "speech_resolution": {
                "version": 1, "original_text": self.original,
                "tts_text": self.rewritten,
                "audit": {"identity": {"name": "fixture"},
                          "payload": {"permitted_actions": ["rewrite", "split"]}},
                "failures": [{"transcript": "Van deer is waiting here."}],
            },
        }

    def identity(self, *, request=None, metadata=None, voices=None):
        return generation_identity(
            self.request if request is None else request,
            self.voices if voices is None else voices,
            engine_metadata=self.metadata if metadata is None else metadata)

    def validator(self, **kwargs):
        return patch("epub_tts_narrator.renderer.speech_resolver.validate_resolution",
                     return_value=self.proposal, **kwargs)

    def test_validated_rewrite_retains_original_logical_identity(self):
        expected = generation_identity(self.request, self.voices)
        with self.validator() as validate:
            self.assertEqual(self.identity(), expected)
        marker = self.metadata["speech_resolution"]
        validate.assert_called_once_with(
            marker["audit"], self.original, marker["failures"], marker["audit"]["identity"],
            can_rewrite=True, can_split=True)
        actual = deepcopy(self.request)
        actual["chunk"]["spoken_text"] = self.rewritten
        self.assertNotEqual(expected, generation_identity(actual, self.voices))

    def test_unmarked_changed_synthesis_text_is_still_rejected(self):
        with self.assertRaises(ValueError):
            self.identity(metadata={"synthesis_text": self.rewritten})

    def test_invalid_marker_cannot_enable_a_rewrite(self):
        variants = []
        for field, value in (("version", True), ("version", 2),
                             ("original_text", self.original + " Again."),
                             ("tts_text", ""), ("tts_text", self.original),
                             ("audit", None)):
            metadata = deepcopy(self.metadata)
            metadata["speech_resolution"][field] = value
            variants.append(metadata)
        metadata = deepcopy(self.metadata)
        metadata["speech_resolution"]["unexpected"] = True
        variants.append(metadata)
        variants.append(dict(self.metadata, speech_resolution=None))
        metadata = deepcopy(self.metadata)
        metadata.pop("synthesis_text")
        variants.extend([metadata, dict(self.metadata, synthesis_text="")])
        for payload in (None, {}, {"permitted_actions": "rewrite"}):
            metadata = deepcopy(self.metadata)
            metadata["speech_resolution"]["audit"]["payload"] = payload
            variants.append(metadata)
        for metadata in variants:
            with self.subTest(metadata=metadata), self.validator() as validate:
                with self.assertRaises(ValueError):
                    self.identity(metadata=metadata)
                validate.assert_not_called()

    def test_invalid_or_mismatched_audit_is_rejected(self):
        for proposal in (None, {}, {"action": "accept", "tts_text": self.rewritten},
                         {"action": "rewrite", "tts_text": "Different speech."}):
            with self.subTest(proposal=proposal), patch(
                    "epub_tts_narrator.renderer.speech_resolver.validate_resolution",
                    return_value=proposal), self.assertRaises(ValueError):
                self.identity()

    def test_actual_synthesis_input_and_delivery_must_match_the_rewrite(self):
        request = deepcopy(self.request)
        request["chunk"]["segment"]["delivery"]["direction"] = "whisper"
        expected = generation_identity(request, self.voices)
        for text in (self.rewritten, "<whisper>" + self.rewritten + "</whisper>"):
            with self.subTest(text=text), self.validator():
                metadata = dict(self.metadata, synthesis_text=text, applied_control="whisper")
                self.assertEqual(self.identity(request=request, metadata=metadata), expected)
        for metadata in (
                dict(self.metadata, synthesis_text=self.original),
                dict(self.metadata, synthesis_text="Van dyer has gone away."),
                dict(self.metadata, synthesis_text=self.rewritten, applied_control=None)):
            with self.subTest(metadata=metadata), self.validator(), self.assertRaises(ValueError):
                self.identity(request=request, metadata=metadata)

    def test_paragraph_rewrite_still_requires_actual_dialogue_formatted_input(self):
        from epub_tts_narrator.renderer.engines import format_dialogue
        request = paragraph_request([("Vandire ", self.voice, ""), ("is waiting here.", self.voice, "")])
        with self.validator() as validate, self.assertRaises(ValueError):
            self.identity(request=request)
        validate.assert_called_once()
        text, _, controls = format_dialogue([{"text": self.rewritten, "voice": self.voice, "direction": ""}])
        with self.validator():
            actual = self.identity(request=request, metadata={**self.metadata, **controls, "synthesis_text": text})
        self.assertEqual(actual, generation_identity(request, self.voices))

    def test_original_source_text_is_bound_exactly(self):
        request = deepcopy(self.request)
        request["chunk"]["spoken_text"] = " " + self.original
        with self.validator() as validate, self.assertRaises(ValueError):
            self.identity(request=request)
        validate.assert_not_called()

    def test_voice_and_sampling_still_determine_reuse(self):
        with self.validator():
            original = self.identity()
            changed_voice = dict(self.voice, reference_sha256=digest("new reference"))
            changed = self.identity(voices={self.voice["asset_hash"]: changed_voice})
            self.assertNotEqual(original, changed)
            request = deepcopy(self.request)
            request["engine"]["temperature"] = .8
            self.assertNotEqual(original, self.identity(request=request))
            with self.assertRaises(ValueError):
                self.identity(voices={self.voice["asset_hash"]: reference("wrong")})

    def test_identity_does_not_modify_request_or_audit(self):
        self.request["synthesis_identity"] = {"stale": "source identity"}
        before = deepcopy((self.request, self.voices, self.metadata))
        with self.validator():
            self.identity()
        self.assertEqual((self.request, self.voices, self.metadata), before)

    def audited_fixture(self, *, can_split=True):
        from epub_tts_narrator.renderer.speech_resolver import PROMPT, SpeechResolver

        class Client:
            identity = {"provider": "local-test-fixture"}

            def complete(self, prompt, payload):
                if prompt == PROMPT:
                    return {"action": "rewrite", "edits": [{"start": 0, "end": 5,
                        "source": "Qarmy", "replacement": "Karmy"}],
                        "reason": "Try a phonetic respelling of the invented name."}
                return {"pairs": [{"id": pair["id"], "plausible": True,
                    "reason": "The initial spellings can represent the same sound."}
                    for pair in payload["pairs"]]}

        original, rewrite = "Qarmy waited.", "Karmy waited."
        failures = [{"attempt": 1, "audio_sha256": "a" * 64,
                     "transcript": rewrite, "reasons": ["text_mismatch"]}]
        audit = SpeechResolver(Client()).resolve(original, failures, can_split=can_split)
        self.assertEqual(audit["decision"], "resolved")
        request = speech_request(self.voice, original)
        metadata = {"synthesis_text": rewrite, "speech_resolution": {
            "version": 1, "original_text": original, "tts_text": rewrite,
            "audit": audit, "failures": failures}}
        return request, metadata

    def test_complete_real_resolver_audit_binds_the_alternate_input(self):
        request, metadata = self.audited_fixture()
        self.assertEqual(self.identity(request=request, metadata=metadata),
                         generation_identity(request, self.voices))

    def test_unsplittable_passage_can_reuse_its_validated_rewrite(self):
        request, metadata = self.audited_fixture(can_split=False)
        self.assertNotIn("split", metadata["speech_resolution"]["audit"]["payload"]["permitted_actions"])
        self.assertEqual(self.identity(request=request, metadata=metadata),
                         generation_identity(request, self.voices))

    def test_tampered_real_audit_or_failed_take_evidence_cannot_enable_reuse(self):
        request, metadata = self.audited_fixture()
        variants = []
        changed = deepcopy(metadata)
        changed["speech_resolution"]["audit"]["proposal"]["edits"][0]["source"] = "Other"
        variants.append(changed)
        changed = deepcopy(metadata)
        changed["speech_resolution"]["failures"][0]["transcript"] = "Extra speech."
        variants.append(changed)
        changed = deepcopy(metadata)
        changed["speech_resolution"]["audit"]["reviews"].pop()
        variants.append(changed)
        changed = deepcopy(metadata)
        changed["speech_resolution"]["audit"]["identity"]["version"] = 99
        variants.append(changed)
        for changed in variants:
            with self.subTest(metadata=changed), self.assertRaises(ValueError):
                self.identity(request=request, metadata=changed)


if __name__ == "__main__":
    unittest.main()
