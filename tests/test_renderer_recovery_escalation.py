"""Mixed recurring failures escalate together, and applied repairs get feedback.

Regression coverage for bounded feedback-driven recovery: the recurrence
trigger counts distinct failed takes once each, an applied pronunciation or
delivery repair that fails its new take reaches the planner immediately, an
acoustic failure can motivate a pronunciation rewrite but can never be
accepted, and saved decisions replay without new model rounds.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.recurring_resolution import (
    RecurringResolution, accepted_report, failure_signatures,
)
from epub_tts_narrator.renderer.speech_resolver import (
    PROMPT, SOUND_PROMPT, SpeechResolver, _payload, _proposal, validate_resolution,
)


TEXT = "Insang waited beside the old gate."
REQUEST = {"kind": "speech", "text": TEXT}


def rewrite(reason="Try a pronunciation spelling.", replacement="Insaan"):
    return {"action": "rewrite", "edits": [{"start": 0, "end": 6, "source": "Insang",
                                            "replacement": replacement}],
            "reason": reason}


class EscalationClient:
    identity = {"model": "offline-escalation", "base_url": "https://fixture.invalid/v1"}

    def __init__(self, actions=None):
        self.actions = list(actions or [{"action": "accept", "edits": [],
                                         "reason": "The spelling may be recognition variation."}])
        self.plans, self.reviews = [], []

    def complete(self, prompt, payload):
        if "pairs" in payload:
            self.reviews.append(deepcopy(payload))
            return {"pairs": [{"id": pair["id"], "plausible": True,
                               "reason": "Plausible pronunciation spelling."}
                              for pair in payload["pairs"]]}
        self.plans.append(deepcopy(payload))
        position = min(len(self.plans) - 1, len(self.actions) - 1)
        return deepcopy(self.actions[position])


class EscalationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.messages = []

    def session(self, client, *, threshold=3, rounds=5):
        resolver = SpeechResolver(client)
        return RecurringResolution(self.directory, REQUEST, TEXT, resolver,
                                   {"recurrence_threshold": threshold,
                                    "max_resolution_rounds": rounds},
                                   can_rewrite=True, can_split=False)

    def record(self, attempt, reasons, transcript):
        path = self.directory / f"attempt-{attempt:03d}.wav"
        path.write_bytes(f"byte-bound fixture take {attempt}".encode())
        value = {"request_hash": digest(REQUEST), "request": REQUEST,
                 "attempt": attempt, "audio_file": path.name, "audio_sha256": file_hash(path),
                 "engine_metadata": {},
                 "report": {"passed": False, "reasons": list(reasons), "transcript": transcript}}
        write_json(self.directory / f"attempt-{attempt:03d}.json", value)
        return value

    def consider(self, session, record):
        return session.consider(record, notify=self.messages.append)

    def test_text_and_acoustic_signatures_stay_separate_but_too_long_counts(self):
        text = failure_signatures(TEXT, {"passed": False, "reasons": ["text_mismatch"],
                                         "transcript": "The Insane waited."})
        too_long = failure_signatures(TEXT, {"passed": False, "reasons": ["audio_too_long"],
                                             "transcript": None})
        silent = failure_signatures(TEXT, {"passed": False, "reasons": ["silent_audio"],
                                           "transcript": None})
        self.assertTrue(text)
        self.assertTrue(too_long.isdisjoint(text))
        self.assertTrue(too_long.isdisjoint(silent))
        self.assertEqual(failure_signatures(TEXT, {"passed": False, "reasons": ["audio_too_long"],
                                                   "transcript": "ignored"}), too_long)

    def test_mixed_text_and_acoustic_failures_escalate_after_three_takes(self):
        client = EscalationClient([rewrite()])
        session = self.session(client)
        self.assertIsNone(self.consider(session, self.record(
            1, ["text_mismatch"], "The Insane waited beside the old gate.")))
        self.assertIsNone(self.consider(session, self.record(2, ["audio_too_short"], None)))
        third = self.record(3, ["text_mismatch", "audio_too_quiet"], "The Insane waited.")
        result = self.consider(session, third)
        self.assertIsNotNone(result, "Three distinct failed takes must reach the planner")
        self.assertEqual(result["action"], "rewrite",
                         "A mixed failure may motivate a pronunciation rewrite")
        self.assertEqual([failure["attempt"] for failure in client.plans[0]["failures"]], [1, 2, 3])
        self.assertEqual(client.plans[0]["failures"][-1]["reasons"],
                         ["text_mismatch", "audio_too_quiet"])
        self.assertIn("rewrite", client.plans[0]["permitted_actions"])
        self.assertNotIn("accept", client.plans[0]["permitted_actions"],
                         "A mixed acoustic failure can never be accepted")
        self.assertEqual(len(client.reviews), 3)

    def test_one_take_with_several_acoustic_reasons_counts_once(self):
        client = EscalationClient([rewrite()])
        session = self.session(client)
        self.assertIsNone(self.consider(session, self.record(
            1, ["silent_audio", "audio_too_quiet"], None)))
        second = self.record(2, ["silent_audio", "audio_too_quiet"], None)
        self.assertIsNone(self.consider(session, second),
                          "Two takes with two reasons each are two takes, not four")
        third = self.record(3, ["silent_audio", "audio_too_quiet"], None)
        self.assertEqual(self.consider(session, third)["action"], "rewrite")
        self.assertEqual(len(client.plans), 1)

    def test_too_long_takes_reach_the_planner_without_acceptance(self):
        client = EscalationClient([rewrite()])
        session = self.session(client)
        self.assertIsNone(self.consider(session, self.record(1, ["audio_too_long"], None)))
        self.assertIsNone(self.consider(session, self.record(2, ["audio_too_long"], None)))
        third = self.record(3, ["audio_too_long"], None)
        result = self.consider(session, third)
        self.assertIsNotNone(result, "Repeated generation-length failures must escalate")
        self.assertEqual(result["action"], "rewrite")
        self.assertEqual([failure["attempt"] for failure in client.plans[0]["failures"]], [1, 2, 3])
        self.assertTrue(all(failure["transcript"] == "" for failure in client.plans[0]["failures"]))
        self.assertIn("rewrite", client.plans[0]["permitted_actions"])
        self.assertNotIn("accept", client.plans[0]["permitted_actions"],
                         "Audio that was never transcribed cannot be accepted")
        self.assertEqual(len(client.reviews), 3, "The rewrite still needs its sound jury")
        self.assertEqual(session.state["handled"], {"takes": 3})

    def test_failed_applied_repair_reaches_the_planner_on_its_next_take(self):
        client = EscalationClient([rewrite(), {"action": "retry", "edits": [],
                                               "reason": "Try a fresh complete performance."}])
        session = self.session(client, rounds=2)
        self.assertIsNone(self.consider(session, self.record(
            1, ["text_mismatch"], TEXT.replace("Insang", "Insane").replace("beside ", ""))))
        self.assertIsNone(self.consider(session, self.record(
            2, ["text_mismatch"], TEXT.replace("Insang", "Insane").replace("beside ", ""))))
        third = self.record(3, ["text_mismatch"],
                            TEXT.replace("Insang", "Insane").replace("beside ", ""))
        self.assertEqual(self.consider(session, third)["action"], "rewrite")
        fourth = self.record(4, ["text_mismatch"],
                             TEXT.replace("Insang", "Insaan").replace("beside ", ""))
        resumed = self.session(client, rounds=2)
        result = self.consider(resumed, fourth)
        self.assertIsNotNone(result, "The disproved rewrite must get immediate feedback")
        self.assertEqual(result["action"], "retry")
        self.assertEqual(len(client.plans), 2)
        self.assertEqual(resumed.state["handled"], {"takes": 4})
        fifth = self.record(5, ["text_mismatch"], "The Insane waited.")
        self.assertIsNone(self.consider(resumed, fifth), "Rounds stay bounded")
        self.assertEqual(len(client.plans), 2)

    def test_immediate_feedback_message_names_the_disproved_repair(self):
        client = EscalationClient([rewrite(), {"action": "retry", "edits": [],
                                               "reason": "Try a fresh complete performance."}])
        session = self.session(client, rounds=2)
        self.assertIsNone(self.consider(session, self.record(
            1, ["text_mismatch"], TEXT.replace("Insang", "Insane"))))
        self.assertIsNone(self.consider(session, self.record(
            2, ["text_mismatch"], TEXT.replace("Insang", "Insane"))))
        self.consider(session, self.record(3, ["text_mismatch"],
                                           TEXT.replace("Insang", "Insane")))
        fourth = self.record(4, ["text_mismatch"], TEXT.replace("Insang", "Insaan"))
        self.consider(session, fourth)
        self.assertTrue(any("rewrite recovery failed its new take" in message
                            for message in self.messages))
        self.assertFalse(any("Insang" in message or "Insaan" in message
                             for message in self.messages),
                         "Notifications summarize counts, never passage text")

    def test_invalid_proposal_leaves_evidence_pending_for_the_next_take(self):
        client = EscalationClient()
        session = self.session(client, rounds=3)
        self.assertIsNone(self.consider(session, self.record(
            1, ["text_mismatch"], TEXT.replace("Insang", "Insane").replace("beside ", ""))))
        self.assertIsNone(self.consider(session, self.record(
            2, ["text_mismatch"], TEXT.replace("Insang", "Insane").replace("beside ", ""))))
        first = self.consider(session, self.record(
            3, ["text_mismatch"], TEXT.replace("Insang", "Insane").replace("beside ", "")))
        self.assertIsNone(first, "A missing word cannot be accepted")
        self.assertEqual(len(client.plans), 1)
        self.assertEqual(session.state["handled"], {},
                         "A blocked proposal must not consume the evidence")
        fourth = self.record(4, ["text_mismatch"], "The Insane waited.")
        self.consider(session, fourth)
        self.assertEqual(len(client.plans), 2,
                         "Unconsumed evidence stays pending without three new takes")

    def test_saved_accept_decision_replays_without_a_new_model_round(self):
        client = EscalationClient()
        session = self.session(client, threshold=1)
        first = self.record(1, ["text_mismatch"], TEXT.replace("Insang", "Insane"))
        resolved = self.consider(session, first)
        self.assertEqual(resolved["action"], "accept")
        state = (self.directory / "speech-resolution.json").read_bytes()
        resumed = self.session(client, threshold=1)
        replay = self.consider(resumed, first)
        self.assertEqual(replay["action"], "accept")
        self.assertEqual(replay["audit"], resolved["audit"])
        self.assertEqual(len(client.plans), 1)
        self.assertEqual((self.directory / "speech-resolution.json").read_bytes(), state)

    def test_acoustic_latest_gate_permits_rewrite_but_never_acceptance(self):
        failures = [{"attempt": 1, "audio_sha256": "1" * 64, "transcript": "",
                     "reasons": ["audio_too_long"]}]
        legacy = _payload(TEXT, failures, True, False, acoustic_rewrite=False)
        fresh = _payload(TEXT, failures, True, False, acoustic_rewrite=True)
        self.assertNotIn("accept", legacy["permitted_actions"])
        self.assertNotIn("accept", fresh["permitted_actions"])
        self.assertNotIn("rewrite", legacy["permitted_actions"],
                         "The earlier policy offered no rewrite for acoustic failures")
        self.assertIn("rewrite", fresh["permitted_actions"])
        self.assertEqual({key: value for key, value in legacy.items() if key != "permitted_actions"},
                         {key: value for key, value in fresh.items() if key != "permitted_actions"})

    def test_pre_acoustic_policy_rewrite_audits_still_validate(self):
        client = EscalationClient()
        identity = SpeechResolver(client).identity
        legacy = {key: value for key, value in identity.items() if key != "policy_sha256"}
        legacy["version"] = 1
        legacy["policy_sha256"] = digest(legacy)
        failures = [{"attempt": 1, "audio_sha256": "1" * 64, "transcript": "The Insane waited.",
                     "reasons": ["text_mismatch"]}]
        payload = _payload(TEXT, failures, True, False, previous_decisions=[],
                           acoustic_rewrite=False)
        proposal = _proposal(rewrite(), payload)
        reviews = [{"reviewer": number, "prompt": SOUND_PROMPT,
                    "payload": {"pairs": proposal["validated_pairs"]},
                    "response": {"pairs": [{"id": pair["id"], "plausible": True,
                                            "reason": "Plausible pronunciation spelling."}
                                           for pair in proposal["validated_pairs"]]}}
                   for number in (1, 2, 3)]
        audit = {"identity": legacy, "prompt": PROMPT, "payload": payload,
                 "proposal": rewrite(), "reviews": reviews, "decision": "resolved"}
        validated = validate_resolution(audit, TEXT, failures, legacy,
                                        can_rewrite=True, can_split=False)
        self.assertIsNotNone(validated, "Saved decisions from the earlier policy must replay")
        self.assertEqual(validated["action"], "rewrite")

    def test_fresh_acoustic_rewrite_decision_validates_and_replays(self):
        client = EscalationClient([rewrite()])
        session = self.session(client, threshold=1)
        first = self.record(1, ["audio_too_long"], None)
        resolved = self.consider(session, first)
        self.assertEqual(resolved["action"], "rewrite")
        marker = session.active_rewrite()
        self.assertEqual(marker["tts_text"], TEXT.replace("Insang", "Insaan"))
        audit = resolved["audit"]
        self.assertIsNotNone(
            validate_resolution(audit, TEXT, resolved["failures"], audit["identity"],
                                can_rewrite=True, can_split=False,
                                previous_decisions=[]),
            "A fresh acoustic-motivated rewrite must stay replayable")

    def test_upgrade_retains_reviewed_repairs_and_acceptance_without_new_model_calls(self):
        for action in (rewrite(), {"action": "plain_delivery", "edits": [],
                                   "reason": "Remove the acting direction."},
                       {"action": "accept", "edits": [], "reason": "Plausible spelling."}):
            with self.subTest(action=action["action"]), tempfile.TemporaryDirectory() as temporary:
                self.directory = Path(temporary)
                client = EscalationClient([action])
                resolver = SpeechResolver(client)
                settings = {"recurrence_threshold": 1, "max_resolution_rounds": 2}
                session = RecurringResolution(self.directory, REQUEST, TEXT, resolver, settings,
                                             can_rewrite=True, can_split=False,
                                             can_plain_delivery=True)
                first = self.record(1, ["text_mismatch"], TEXT.replace("Insang", "Insane"))
                self.consider(session, first)
                saved = read_json(session.path)
                body = {key: value for key, value in saved.items() if key != "record_sha256"}
                previous_identity = {key: value for key, value in resolver.identity.items()
                                     if key != "policy_sha256"}
                previous_identity["version"] = 1
                previous_identity["policy_sha256"] = digest(previous_identity)
                body["binding"].update(version=2, resolver=previous_identity)
                for row in body["history"]:
                    row["audit"]["identity"] = previous_identity
                    row.pop("applied", None)
                write_json(session.path, {**body, "record_sha256": digest(body)})

                resumed = RecurringResolution(self.directory, REQUEST, TEXT, resolver, settings,
                                             can_rewrite=True, can_split=False,
                                             can_plain_delivery=True)

                if action["action"] == "rewrite":
                    self.assertEqual(resumed.active_rewrite()["tts_text"],
                                     TEXT.replace("Insang", "Insaan"))
                elif action["action"] == "plain_delivery":
                    self.assertEqual(resumed.active_delivery()["action"], "plain_delivery")
                else:
                    self.assertEqual(self.consider(resumed, first)["action"], "accept")
                self.assertEqual(len(client.plans), 1)
                self.assertEqual(len(resumed.state["history"]), 1)

    def test_acoustic_failure_is_never_accepted_even_after_a_repair_round(self):
        client = EscalationClient([rewrite()])
        session = self.session(client, threshold=1)
        first = self.record(1, ["audio_too_long"], None)
        result = self.consider(session, first)
        report = read_json(self.directory / "attempt-001.json")["report"]
        with self.assertRaises(ValueError):
            accepted_report(report, result)


if __name__ == "__main__":
    unittest.main()
