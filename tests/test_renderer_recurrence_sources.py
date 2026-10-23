"""Changing ASR guesses must not disguise one recurring source-word failure."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.recurring_resolution import RecurringResolution, failure_signatures
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from test_renderer_recurring_resolution import PlanClient


TEXT = "Arrgh! The pain has returned."


def report(transcript, secondary=None):
    result = {"passed": False, "reasons": ["text_mismatch"], "transcript": transcript}
    if secondary is not None:
        result["secondary_transcript"] = secondary
        result["corroboration"] = {"transcript": secondary}
    return result


class SourceFailureSignatureTests(unittest.TestCase):
    def test_deletion_and_different_substitutions_share_the_source_span(self):
        variants = ["The pain has returned.", "Oh! The pain has returned.",
                    "Are! The pain has returned.", "How! The pain has returned."]
        signatures = [failure_signatures(TEXT, report(text)) for text in variants]
        self.assertTrue(signatures[0])
        self.assertEqual(len(signatures[0]), 1)
        for signature in signatures[1:]:
            self.assertEqual(signature, signatures[0])

    def test_primary_and_secondary_count_one_source_problem_per_take(self):
        expected = failure_signatures(TEXT, report("The pain has returned."))
        combined = failure_signatures(TEXT, report("The pain has returned.", "Oh! The pain has returned."))
        self.assertEqual(combined, expected)
        self.assertEqual(len(combined), 1)

    def test_added_word_anchor_is_not_a_source_word_substitution(self):
        substitution = failure_signatures(TEXT, report("Oh! The pain has returned."))
        added = failure_signatures(TEXT, report("Oh! Arrgh! The pain has returned."))
        another_addition = failure_signatures(TEXT, report("Well! Arrgh! The pain has returned."))
        self.assertTrue(added)
        self.assertTrue(added.isdisjoint(substitution))
        self.assertEqual(added, another_addition,
                         "The repeated insertion location counts even when the extra word changes")

    def test_distinct_source_positions_do_not_merge(self):
        missing_first = failure_signatures(TEXT, report("The pain has returned."))
        missing_later = failure_signatures(TEXT, report("Arrgh! The has returned."))
        self.assertTrue(missing_later)
        self.assertTrue(missing_first.isdisjoint(missing_later))

    def test_accepted_transcript_and_acoustic_failure_do_not_gain_word_signatures(self):
        accepted = dict(report("The pain has returned."), passed=True, reasons=[])
        self.assertEqual(failure_signatures(TEXT, accepted), set())
        acoustic = dict(report("The pain has returned."), reasons=["internal_silence_stall"])
        self.assertTrue(failure_signatures(TEXT, acoustic).isdisjoint(
            failure_signatures(TEXT, report("The pain has returned."))))


class SourceRecurrenceStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.request = {"kind": "speech", "text": TEXT}
        self.client = PlanClient([{"action": "retry", "edits": [], "reason": "Try a fresh performance."}])
        self.resolver = SpeechResolver(self.client)
        self.settings = {"recurrence_threshold": 3, "max_resolution_rounds": 5}
        self.messages = []

    def session(self):
        return RecurringResolution(self.directory, self.request, TEXT, self.resolver,
                                   self.settings, can_rewrite=True, can_split=False)

    def record(self, attempt, transcript, secondary=None):
        path = self.directory / f"attempt-{attempt:03d}.wav"
        path.write_bytes(f"byte-bound fixture take {attempt}".encode())
        value = {"request_hash": digest(self.request), "request": self.request,
                 "attempt": attempt, "audio_file": path.name, "audio_sha256": file_hash(path),
                 "engine_metadata": {}, "report": report(transcript, secondary)}
        write_json(self.directory / f"attempt-{attempt:03d}.json", value)
        return value

    def consider(self, session, record):
        return session.consider(record, notify=self.messages.append)

    def test_planner_receives_actual_controls_and_bounded_acoustic_measurements(self):
        session = self.session()
        record = self.record(1, "The pain has returned.")
        record['engine_metadata'] = {
            'synthesis_text': '[pained, breaking into a cry] ' + TEXT,
            'direction': 'pained, breaking into a cry', 'applied_control': 'style_prefix'}
        record['report']['metrics'] = {
            'silence_fraction': 0.91, 'duration_seconds': 9.2, 'wer': 0.2,
            'alignment': [{'expected': 'arrgh', 'actual': ''}] * 10000,
            'asr_segments': [{'text': 'The pain has returned.'}] * 10000}
        evidence = session._failure(record)
        self.assertEqual(evidence['generation'], record['engine_metadata'])
        self.assertEqual(evidence['metrics'], {
            'silence_fraction': 0.91, 'duration_seconds': 9.2, 'wer': 0.2})
        self.assertEqual(len(record['report']['metrics']['alignment']), 10000)

    def test_varied_arrgh_failures_trigger_review_on_third_take(self):
        session = self.session()
        variants = ["Oh! The pain has returned.", "The pain has returned.",
                    "Are! The pain has returned."]
        for attempt, transcript in enumerate(variants, 1):
            result = self.consider(session, self.record(attempt, transcript))
            if attempt < 3:
                self.assertIsNone(result)
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(result["action"], "retry")
        self.assertEqual([item["attempt"] for item in self.client.plans[0]["failures"]], [1, 2, 3])
        self.assertEqual([item["transcript"] for item in self.client.plans[0]["failures"]], variants)
        self.assertEqual(list(session.state["history"][0]["trigger_counts"].values()), [3])
        self.assertTrue(any("(3 takes)" in message for message in self.messages))

    def test_rechecking_the_same_take_does_not_increase_recurrence(self):
        session = self.session()
        first = self.record(1, "The pain has returned.", "Oh! The pain has returned.")
        for _ in range(8):
            self.assertIsNone(self.consider(session, deepcopy(first)))
        self.assertEqual(len(session.records), 1)
        self.assertEqual(self.client.plans, [])
        second = self.record(2, "Are! The pain has returned.", "The pain has returned.")
        self.assertIsNone(self.consider(session, second))
        third = self.record(3, "How! The pain has returned.", "Oh! The pain has returned.")
        self.assertEqual(self.consider(session, third)["action"], "retry")
        for _ in range(8):
            self.assertIsNone(self.consider(session, deepcopy(third)))
        self.assertEqual(len(self.client.plans), 1)
        self.assertEqual(list(session.state["handled"].values()), [3])
        self.assertEqual(len(session.records), 3)

    def test_version_one_state_preserves_decisions_but_recounts_source_failures(self):
        session = self.session()
        latest = None
        for attempt in range(1, 4):
            latest = self.record(attempt, "Oh! The pain has returned.")
            self.consider(session, latest)
        saved = read_json(session.path)
        legacy = {key: value for key, value in saved.items() if key != "record_sha256"}
        legacy["binding"]["version"] = 1
        legacy["handled"] = {digest({"legacy-operation-sensitive-signature": True}): 3}
        history = deepcopy(legacy["history"])
        write_json(session.path, {**legacy, "record_sha256": digest(legacy)})
        resumed = self.session()
        self.assertEqual(resumed.binding["version"], 2)
        self.assertEqual(resumed.state["binding"], resumed.binding)
        self.assertEqual(resumed.state["history"], history)
        self.assertEqual(resumed.state["handled"], {})
        self.assertEqual(sorted(resumed.records), [1, 2, 3])
        self.assertEqual(self.consider(resumed, latest)["action"], "retry")
        self.assertEqual(len(self.client.plans), 2)
        self.assertEqual(resumed.state["history"][:1], history)
        persisted = read_json(resumed.path)
        self.assertEqual(persisted["binding"]["version"], 2)
        self.assertEqual(persisted["history"][:1], history)
        self.assertEqual(list(persisted["handled"].values()), [3])


if __name__ == "__main__":
    unittest.main()
