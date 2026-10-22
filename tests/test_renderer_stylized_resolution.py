"""Repeated stylized spelling failures can recover without losing spoken words."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.recurring_resolution import (
    RecurringResolution, accepted_report, confirm_rewritten_report,
)
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
from test_renderer_workflow import write_tone


SOURCE = "Wheerrre issss sheeee"
SPOKEN = "Where is she"


class ResolutionClient:
    """Replace only the external planner and sound judgments, not local guards."""

    identity = {"model": "offline-stylized-resolution", "base_url": "https://fixture.invalid/v1"}

    def __init__(self, action="accept", *, source=SOURCE, replacement=SPOKEN, approve=True):
        self.proposal = {"action": action, "edits": [],
                         "reason": "Printed repeated letters convey prolonged speech, with all words retained."}
        if action == "rewrite":
            self.proposal["edits"] = [{"start": 0, "end": len(source), "source": source,
                                       "replacement": replacement}]
        self.plans, self.reviews = [], []
        self.approve = approve

    def complete(self, system, payload):
        if "pairs" in payload:
            self.reviews.append(deepcopy(payload))
            return {"pairs": [{"id": pair["id"], "plausible": self.approve,
                               "reason": "Every intended word survives; repeated letters mark delivery."}
                              for pair in payload["pairs"]]}
        self.plans.append(deepcopy(payload))
        return deepcopy(self.proposal)


def failures(transcript=SPOKEN):
    return [{"attempt": number, "audio_sha256": str(number) * 64,
             "transcript": transcript, "secondary_transcript": "", "reasons": ["text_mismatch"]}
            for number in (1, 2, 3)]


def report(transcript=SPOKEN, source=SOURCE):
    return {"passed": False, "reasons": ["text_mismatch"],
            "expected_text": source, "transcript": transcript}


class StylizedSpeechResolutionTests(unittest.TestCase):
    def resolve(self, *, transcript=SPOKEN, source=SOURCE, action="accept", approve=True):
        client = ResolutionClient(action, source=source, replacement=transcript, approve=approve)
        resolver = SpeechResolver(client)
        evidence = failures(transcript)
        audit = resolver.resolve(source, evidence)
        proposal = validate_resolution(audit, source, evidence, resolver.identity)
        return client, resolver, evidence, audit, proposal

    def test_complete_style_spelling_is_eligible_for_independent_acceptance_jury(self):
        client, _, _, audit, proposal = self.resolve()
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["action"], "accept")
        self.assertEqual(proposal["acceptance_transcript"], SPOKEN)
        self.assertEqual(len(client.reviews), 3)
        self.assertEqual(client.plans[0]["requested_text"], SOURCE)

    def test_style_normalization_can_retry_synthesis_while_retaining_source_text(self):
        client, resolver, evidence, audit, proposal = self.resolve(action="rewrite")
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["tts_text"], SPOKEN)
        self.assertEqual(len(client.reviews), 3)
        marker = {"version": 1, "original_text": SOURCE, "tts_text": SPOKEN,
                  "audit": audit, "failures": evidence}
        original = report()
        accepted = confirm_rewritten_report(original, marker)
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["expected_text"], SOURCE)
        self.assertEqual(accepted["transcript"], SPOKEN)
        self.assertFalse(original["passed"])
        self.assertEqual(validate_resolution(audit, SOURCE, evidence, resolver.identity), proposal)

    def test_style_proposal_still_needs_independent_sound_approval(self):
        client, _, _, audit, proposal = self.resolve(approve=False)
        self.assertIsNone(proposal)
        self.assertEqual(audit.get("error_code"), "sound_review_rejected")
        self.assertEqual(len(client.reviews), 3)

    def test_spelling_rewrite_can_retain_unchanged_words_inside_stretched_span(self):
        client, _, _, audit, proposal = self.resolve(
            source="Wheerrre is sheeee", transcript="Where is she", action="rewrite")
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        self.assertEqual(proposal["tts_text"], "Where is she")
        self.assertEqual(len(client.reviews), 3)
        for replacement in ("Where she", "Where is he", "Where she is"):
            with self.subTest(replacement=replacement):
                _, _, _, _, proposal = self.resolve(
                    source="Wheerrre is sheeee", transcript=replacement, action="rewrite")
                self.assertIsNone(proposal)

    def test_repeated_omissions_additions_or_changed_pronouns_cannot_be_accepted(self):
        for transcript in ("Where she", "Where is", "Where is she really", "Where is he",
                           "Where was she"):
            with self.subTest(transcript=transcript):
                _, _, _, _, proposal = self.resolve(transcript=transcript)
                self.assertIsNone(proposal)

    def test_spelling_rewrite_cannot_remove_words_or_change_pronouns(self):
        for transcript in ("Where she", "Where is", "Where is she really", "Where is he",
                           "Where was she"):
            with self.subTest(transcript=transcript):
                _, _, _, _, proposal = self.resolve(transcript=transcript, action="rewrite")
                self.assertIsNone(proposal)

    def test_style_spelling_does_not_excuse_dropped_negation(self):
        source = "Sheeee issss not here"
        for action in ("accept", "rewrite"):
            with self.subTest(action=action):
                _, _, _, _, proposal = self.resolve(source=source, transcript="She is here", action=action)
                self.assertIsNone(proposal)

    def test_same_negation_can_lose_letter_stretch_but_cannot_change_to_affirmation(self):
        for action in ("accept", "rewrite"):
            with self.subTest(action=action):
                _, _, _, audit, proposal = self.resolve(source="Noooo", transcript="No", action=action)
                self.assertIsNotNone(proposal, audit.get("validation_error"))
                _, _, _, _, proposal = self.resolve(source="Noooo", transcript="Yes", action=action)
                self.assertIsNone(proposal)

    def test_ordinary_double_letter_function_word_cannot_be_rewritten_as_stylization(self):
        _, _, _, _, proposal = self.resolve(source="off", transcript="of", action="rewrite")
        self.assertIsNone(proposal)

    def test_rewritten_take_still_fails_if_actual_transcript_omits_one_word(self):
        _, _, evidence, audit, proposal = self.resolve(action="rewrite")
        self.assertIsNotNone(proposal, audit.get("validation_error"))
        marker = {"version": 1, "original_text": SOURCE, "tts_text": SPOKEN,
                  "audit": audit, "failures": evidence}
        for transcript in ("Where she", "Where is", "Where is she now", "Where is he"):
            with self.subTest(transcript=transcript):
                original = report(transcript)
                self.assertEqual(confirm_rewritten_report(original, marker), original)

    def test_three_repeated_style_failures_produce_persisted_recovery_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            client = ResolutionClient()
            resolver = SpeechResolver(client)
            request = {"kind": "speech", "text": SOURCE}
            session = RecurringResolution(directory, request, SOURCE, resolver, {},
                                          can_rewrite=True, can_split=True)
            messages, records = [], []
            for number in (1, 2, 3):
                wav = directory / f"attempt-{number:03d}.wav"
                write_tone(wav, frequency=200 + number * 20)
                record = {"attempt": number, "request_hash": digest(request),
                          "audio_file": wav.name, "audio_sha256": file_hash(wav), "report": report()}
                write_json(directory / f"attempt-{number:03d}.json", record)
                records.append(record)
                resolution = session.consider(record, notify=messages.append)
                if number < 3:
                    self.assertIsNone(resolution)
            self.assertIsNotNone(resolution, "\n".join(messages))
            self.assertEqual(resolution["action"], "accept")
            self.assertEqual(len(client.plans), 1)
            self.assertEqual(len(client.reviews), 3)
            accepted = accepted_report(records[-1]["report"], resolution)
            self.assertTrue(accepted["passed"])
            self.assertEqual(accepted["expected_text"], SOURCE)
            self.assertTrue((directory / "speech-resolution.json").is_file())
            resumed = RecurringResolution(directory, request, SOURCE, resolver, {},
                                          can_rewrite=True, can_split=True)
            replay = resumed.consider(records[-1], notify=messages.append)
            self.assertEqual(replay, resolution)
            self.assertEqual(len(client.plans), 1, "Resuming must reuse the reviewed decision")


if __name__ == "__main__":
    unittest.main()
