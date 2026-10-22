"""Later recovery decisions receive the reasons earlier proposals could not work."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from test_renderer_workflow import write_tone


TEXT = "Insang waited beside the old gate."


class FeedbackClient:
    identity = {"model": "offline-feedback", "base_url": "https://fixture.invalid/v1"}

    def __init__(self, *, reject_sound=False, always_accept=False, oversized=False):
        self.plans, self.reviews = [], []
        self.reject_sound, self.always_accept, self.oversized = reject_sound, always_accept, oversized

    def complete(self, prompt, payload):
        if "pairs" in payload:
            self.reviews.append(deepcopy(payload))
            return {"pairs": [{"id": pair["id"], "plausible": len(self.plans) > 1,
                               "reason": "The original proposal did not sound plausible."}
                              for pair in payload["pairs"]]}
        self.plans.append(deepcopy(payload))
        previous = payload.get("previous_decisions", [])
        if previous and not self.always_accept:
            if self.reject_sound:
                return {"action": "rewrite", "edits": [{"start": 0, "end": 6,
                        "source": "Insang", "replacement": "Insaan"}],
                        "reason": "Try different TTS spelling after the earlier sound review rejected acceptance."}
            return {"action": "retry", "edits": [],
                    "reason": "The earlier acceptance failed word coverage; try a complete new take."}
        return {"action": "accept", "edits": [],
                "reason": "x" * 50000 if self.oversized else "The repeated spelling may reflect recognition."}


class RecoveryFeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.request = {"kind": "speech", "text": TEXT}
        self.messages = []

    def session(self, client):
        resolver = SpeechResolver(client)
        return RecurringResolution(self.directory, self.request, TEXT, resolver,
                                   {"recurrence_threshold": 1, "max_resolution_rounds": 10},
                                   can_rewrite=True, can_split=True)

    def consider(self, session, number, *, missing=True):
        wav = self.directory / f"attempt-{number:03d}.wav"
        write_tone(wav, frequency=220 + number * 20)
        heard = TEXT.replace("Insang", "Insane")
        if missing:
            heard = heard.replace("beside ", "")
        record = {"attempt": number, "request_hash": digest(self.request),
                  "audio_file": wav.name, "audio_sha256": file_hash(wav),
                  "report": {"passed": False, "reasons": ["text_mismatch"],
                             "expected_text": TEXT, "transcript": heard}}
        write_json(self.directory / f"attempt-{number:03d}.json", record)
        return session.consider(record, notify=self.messages.append)

    def test_second_decision_receives_local_rejection_and_changes_its_action(self):
        client = FeedbackClient()
        session = self.session(client)
        self.assertIsNone(self.consider(session, 1))
        resolved = self.consider(session, 2)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["action"], "retry")
        feedback = client.plans[1]["previous_decisions"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["proposal"]["action"], "accept")
        self.assertEqual(feedback[0]["decision"], "blocked")
        self.assertEqual(feedback[0]["error_code"], "invalid_proposal")
        self.assertIn("coverage", feedback[0]["validation_error"])
        self.assertEqual(feedback[0]["sound_reviews"], [])

    def test_resume_preserves_sound_review_outcomes_and_binds_later_rewrite(self):
        client = FeedbackClient(reject_sound=True)
        self.assertIsNone(self.consider(self.session(client), 1, missing=False))
        resumed = self.session(client)
        resolved = self.consider(resumed, 2, missing=False)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["action"], "rewrite")
        feedback = client.plans[1]["previous_decisions"]
        self.assertEqual(feedback[0]["error_code"], "sound_review_rejected")
        self.assertEqual(len(feedback[0]["sound_reviews"]), 3)
        self.assertTrue(all(review["response"]["pairs"][0]["plausible"] is False
                            for review in feedback[0]["sound_reviews"]))
        marker = self.session(client).active_rewrite()
        self.assertEqual(marker["tts_text"], TEXT.replace("Insang", "Insaan"))
        self.assertEqual(marker["audit"]["payload"]["previous_decisions"], feedback)
        self.assertEqual(len(client.plans), 2)

    def test_legacy_history_without_separate_feedback_field_can_resume(self):
        client = FeedbackClient()
        self.assertIsNone(self.consider(self.session(client), 1))
        path = self.directory / "speech-resolution.json"
        state = read_json(path)
        state["history"][0].pop("previous_decisions", None)
        state.pop("record_sha256")
        write_json(path, {**state, "record_sha256": digest(state)})
        resolution = self.consider(self.session(client), 2)
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["action"], "retry")
        self.assertEqual(client.plans[1]["previous_decisions"][0]["attempt"], 1)

    def test_changed_historical_feedback_invalidates_saved_rewrite(self):
        client = FeedbackClient(reject_sound=True)
        session = self.session(client)
        self.consider(session, 1, missing=False)
        self.assertIsNotNone(self.consider(session, 2, missing=False))
        path = self.directory / "speech-resolution.json"
        state = read_json(path)
        state["history"][-1]["audit"]["payload"]["previous_decisions"] = []
        state.pop("record_sha256")
        write_json(path, {**state, "record_sha256": digest(state)})
        with self.assertRaisesRegex(ValueError, "no longer validates|feedback"):
            self.session(client).active_rewrite()

    def test_feedback_uses_bounded_recent_decisions(self):
        client = FeedbackClient(always_accept=True)
        session = self.session(client)
        for number in range(1, 8):
            self.assertIsNone(self.consider(session, number))
        feedback = client.plans[-1]["previous_decisions"]
        self.assertEqual([row["attempt"] for row in feedback], [2, 3, 4, 5, 6])
        self.assertLessEqual(len(json.dumps(feedback, ensure_ascii=False)), 30000)

    def test_oversized_rejected_proposal_is_summarized_without_blocking_followup(self):
        client = FeedbackClient(oversized=True)
        session = self.session(client)
        self.assertIsNone(self.consider(session, 1))
        resolved = self.consider(session, 2)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["action"], "retry")
        feedback = client.plans[-1]["previous_decisions"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["error_code"], "invalid_proposal")
        self.assertLessEqual(len(json.dumps(feedback, ensure_ascii=False)), 30000)


if __name__ == "__main__":
    unittest.main()
