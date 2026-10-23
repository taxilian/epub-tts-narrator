"""Completed or truncated speaker juries reach durable manual fallback."""

from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.speaker_completion import (
    OUTPUT_TRUNCATED_MESSAGE, REVIEWERS, automatic_completion_targets,
    complete_speakers, unassigned_segments,
)
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from shared_fixtures import Terminal
from test_speaker_completion import Requests, PairedRequests, missing_document, paired_document


def uncertain(result, payload):
    for piece in result["segments"]:
        piece.update(role_id=None, basis="unresolved", review_required=True)
        piece.pop("simultaneous_role_ids", None)
    return result


class AutomaticCompletionTargetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocked = complete_speakers(missing_document(3), Requests(uncertain))

    def test_completed_unchanged_cases_need_no_more_automatic_calls(self):
        self.assertEqual(automatic_completion_targets(self.blocked), [])
        self.assertEqual(automatic_completion_targets(missing_document(3)), ["b1", "b2", "b3"])
        self.assertEqual(automatic_completion_targets(complete_speakers(missing_document(), Requests())), [])

    def test_changed_block_only_is_retried_in_source_order(self):
        document = deepcopy(self.blocked)
        for item in document["segments"]:
            if item["block_id"] in {"b1", "b3"}:
                item["attribution"]["reason"] = "A later attribution edit."
        self.assertEqual(automatic_completion_targets(document), ["b1", "b3"])

    def test_changed_source_or_cast_invalidates_saved_exhaustion(self):
        for kind in ("source", "cast"):
            document = deepcopy(self.blocked)
            if kind == "source":
                document["source"]["book"]["title"] += " changed"
            else:
                document["cast"]["voices"][0]["description"] += " changed"
            with self.subTest(kind=kind):
                self.assertEqual(automatic_completion_targets(document), ["b1", "b2", "b3"])

    def test_history_is_usable_when_latest_audit_only_covers_other_blocks(self):
        document = deepcopy(self.blocked)
        audit = document["provenance"].pop("speaker_completion")
        document["provenance"]["speaker_completion_history"] = [None, audit]
        document["provenance"]["speaker_completion"] = {"status": "interrupted"}
        self.assertEqual(automatic_completion_targets(document), [])

    def test_malformed_and_incomplete_audits_are_not_reused(self):
        mutations = [
            lambda audit: audit.update(status="interrupted"),
            lambda audit: audit.update(version="future"),
            lambda audit: audit.update(context_chars=[1, 2, 3]),
            lambda audit: audit.update(cases=None),
            lambda audit: audit["cases"][0].update(status=[]),
            lambda audit: audit["cases"][0].update(status="needs_user"),
            lambda audit: audit["cases"][0].update(rounds=audit["cases"][0]["rounds"][:2]),
            lambda audit: audit["cases"][0]["rounds"][0].update(round=True),
            lambda audit: audit["cases"][0]["rounds"][0]["votes"].pop(),
            lambda audit: audit["cases"][0]["rounds"][0]["votes"][0].update(status="interrupted"),
            lambda audit: audit["cases"][0]["rounds"][0]["votes"][0].update(status=[]),
            lambda audit: audit["cases"][0]["rounds"][0]["votes"][0].update(reviewer_id=[]),
            lambda audit: audit["cases"][0]["rounds"][0]["votes"][0].update(reviewer_id=REVIEWERS[1][0]),
            lambda audit: audit["cases"][0].update(applied_segments=None),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = deepcopy(self.blocked)
                mutate(document["provenance"]["speaker_completion"])
                self.assertIn("b1", automatic_completion_targets(document))

    def test_partial_manual_decisions_elsewhere_do_not_restart_remaining_cases(self):
        document = prompt_speaker_decisions(self.blocked, input_stream=Terminal("o\n1\nq\n"), output_stream=StringIO())
        self.assertEqual(len(unassigned_segments(document)), 2)
        self.assertEqual(automatic_completion_targets(document), [])

    def test_partial_manual_decisions_in_same_block_use_audited_fingerprint_chain(self):
        blocked = complete_speakers(paired_document(), PairedRequests(uncertain))
        first = prompt_speaker_decisions(blocked, input_stream=Terminal("o\n2\nq\n"), output_stream=StringIO())
        self.assertEqual(len(unassigned_segments(first)), 2)
        self.assertEqual(automatic_completion_targets(first), [])
        second = prompt_speaker_decisions(first, input_stream=Terminal("o\n1\nq\n"), output_stream=StringIO())
        self.assertEqual(len(unassigned_segments(second)), 1)
        self.assertEqual(automatic_completion_targets(second), [])
        broken = deepcopy(second)
        broken["provenance"]["speaker_decision_history"] = []
        self.assertEqual(automatic_completion_targets(broken), ["b1"])
        altered = deepcopy(second)
        unassigned_segments(altered)[0]["attribution"]["reason"] += " Edited."
        self.assertEqual(automatic_completion_targets(altered), ["b1"])


class TruncatedSpeakerCompletionTests(unittest.TestCase):
    def test_exhausted_truncation_is_nonpositive_and_does_not_abort_other_cases(self):
        class Client(Requests):
            identity = {"provider": "test", "model": "test"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if payload["target_units"][0]["block_id"] == "b1" and payload["reviewer_id"] == REVIEWERS[0][0]:
                    raise ProviderError(OUTPUT_TRUNCATED_MESSAGE)
                return self.response(payload)
        client, messages = Client(), []
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0, progress=messages.append)
            result = complete_speakers(missing_document(2), requests, workers=1, progress=messages.append)
        self.assertEqual(validate_document(result), [])
        self.assertEqual({item["block_id"] for item in unassigned_segments(result)}, {"b1"})
        failed, resolved = result["provenance"]["speaker_completion"]["cases"]
        self.assertEqual(failed["status"], "needs_user")
        self.assertEqual(len(failed["rounds"]), 1)
        self.assertEqual(failed["rounds"][0]["agreed_spans"], [])
        vote = failed["rounds"][0]["votes"][0]
        self.assertEqual(vote["status"], "error")
        self.assertIs(vote["resolved"], False)
        self.assertIs(vote["output_truncated"], True)
        self.assertEqual(vote["reason"], OUTPUT_TRUNCATED_MESSAGE)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(len(client.calls), 7)  # Three per block plus the bounded identical retry.
        self.assertTrue(all(payload["round"] == 1 for payload in client.calls))
        self.assertTrue(any("remaining passages will be offered" in message for message in messages))
        self.assertEqual(automatic_completion_targets(result), [])

    def test_later_truncation_keeps_earlier_unanimously_resolved_spans(self):
        class TruncatedRequests(Requests):
            def run(self, system, payload, validate, *, cancelled=None):
                if payload["round"] == 2 and payload["reviewer_id"] == REVIEWERS[0][0]:
                    self.calls.append(deepcopy(payload))
                    raise ProviderError(OUTPUT_TRUNCATED_MESSAGE)
                return super().run(system, payload, validate, cancelled=cancelled)
        def dissent(result, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                result["segments"][1].update(role_id=None, basis="unresolved", review_required=True)
            return result
        requests = TruncatedRequests(dissent)
        result = complete_speakers(missing_document(1), requests)
        self.assertEqual(result["segments"][0]["role_id"], "narrator_main")
        self.assertIsNone(result["segments"][1]["role_id"])
        case = result["provenance"]["speaker_completion"]["cases"][0]
        self.assertEqual(case["status"], "needs_user")
        self.assertEqual(len(case["rounds"]), 2)
        self.assertEqual(case["applied_segments"], result["segments"])
        self.assertEqual(automatic_completion_targets(result), [])

    def test_other_provider_failures_still_propagate(self):
        for message in ("Timed out", "Authentication failed", OUTPUT_TRUNCATED_MESSAGE + " Extra"):
            class FailedRequests(Requests):
                def run(self, system, payload, validate, *, cancelled=None):
                    raise ProviderError(message)
            with self.subTest(message=message), self.assertRaisesRegex(ProviderError, "^"):
                complete_speakers(missing_document(1), FailedRequests(), workers=1)


if __name__ == "__main__":
    unittest.main()
