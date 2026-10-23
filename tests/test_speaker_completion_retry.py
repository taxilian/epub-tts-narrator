"""Explicit completion retries reconsider only missing source-bound assignments."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import CachedRequests, fingerprint
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.speaker_completion import automatic_completion_targets, complete_speakers, unassigned_segments
from epub_tts_narrator.speaker_completion_prompts import REVIEWERS, SYSTEM
from test_speaker_completion import Requests, missing_document, paired_document, response
from test_speaker_completion_fallback import uncertain


class Client:
    identity = {"model": "fixture", "provider": "offline-retry-test"}

    def __init__(self, *, unknown=False, fail_after=None):
        self.calls = []
        self.unknown = unknown
        self.fail_after = fail_after

    def complete(self, system, payload):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise ProviderError("Fixture timeout")
        self.calls.append(deepcopy(payload))
        result = response(payload)
        return uncertain(result, payload) if self.unknown else result


def partly_assigned_document(*, basis="context"):
    document = missing_document(1)
    original = document["segments"][0]
    cut = original["text"].index("“")
    known, missing = deepcopy(original), deepcopy(original)
    known.update(id="saved-assigned-record", end=cut, text=original["text"][:cut],
                 role_id="narrator_main", voice_id="vn", kind="narration")
    known["attribution"].update(basis=basis, review_required=False, candidates=[],
                                reason="Keep this established source assignment.")
    known["delivery"] = {"direction": "Quietly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
    missing.update(start=cut, text=original["text"][cut:], id="still-missing")
    document["segments"] = [known, missing]
    return document


class SpeakerCompletionRetryTests(unittest.TestCase):
    def test_retry_hides_unassigned_guesses_but_keeps_exact_source_delivery_and_private_originals(self):
        document = partly_assigned_document()
        missing = document["segments"][1]
        missing["kind"] = "thought"
        missing["attribution"].update(reason="A speculative earlier explanation.", candidates=["mara"],
                                       evidence=[{"block_id": "b1", "quote": "Hello"}])
        before = deepcopy(document)
        requests = Requests()
        result = complete_speakers(document, requests, retry_unresolved=True)
        unknown = requests.calls[0]["original_segments"][1]
        self.assertEqual(set(unknown), {"id", "block_id", "start", "end", "text", "delivery", "role_id", "state"})
        self.assertEqual(unknown["state"], "unassigned")
        self.assertIsNone(unknown["role_id"])
        for key in ("id", "block_id", "start", "end", "text", "delivery"):
            self.assertEqual(unknown[key], missing[key])
        self.assertEqual(requests.calls[0]["input_block_fingerprint"], fingerprint(before["segments"]))
        self.assertEqual(result["provenance"]["speaker_completion"]["cases"][0]["original_segments"], before["segments"])
        self.assertEqual(document, before)
        self.assertEqual(unassigned_segments(result), [])

    def test_retry_keeps_assigned_user_and_unison_attribution_in_request(self):
        document = paired_document()
        first, narrator, _ = document["segments"]
        first.update(role_id=None, voice_id=None, simultaneous_speakers=[
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"}])
        first["attribution"].update(basis="user", reason="The user selected these joint speakers.",
                                     candidates=[], review_required=False)
        narrator.update(role_id="narrator_main", voice_id="vn")
        narrator["attribution"].update(basis="explicit", reason="Established narrative tag.",
                                        candidates=[], review_required=False)
        before = deepcopy(document)

        class NeutralRequests(Requests):
            def response(self, payload):
                evidence = [{"block_id": "b1", "quote": payload["target_units"][0]["text"]}]
                segments = []
                for original in payload["original_segments"]:
                    piece = {key: deepcopy(original[key]) for key in ("text", "delivery")}
                    piece.update(role_id=original["role_id"] if "state" not in original else "mara",
                                 kind=original.get("kind", "dialogue"), basis="explicit", reason="Source-bound selection.",
                                 evidence=evidence, candidates=[], review_required=False)
                    if "simultaneous_role_ids" in original:
                        piece["simultaneous_role_ids"] = original["simultaneous_role_ids"]
                    segments.append(piece)
                return {"block_id": "b1", "reason": "Complete source assignment.", "evidence": evidence,
                        "segments": segments}

        requests = NeutralRequests()
        result = complete_speakers(document, requests, retry_unresolved=True)
        protected, narrative, missing = requests.calls[0]["original_segments"]
        for original, compact in ((first, protected), (narrator, narrative)):
            self.assertNotIn("state", compact)
            for key in ("id", "block_id", "start", "end", "text", "kind", "role_id", "delivery"):
                self.assertEqual(compact[key], original[key])
            for key, value in original["attribution"].items():
                self.assertEqual(compact[key], value)
        self.assertEqual(protected["simultaneous_role_ids"], ["mara", "lena"])
        self.assertEqual(missing["state"], "unassigned")
        self.assertEqual(result["segments"][:2], before["segments"][:2])
        self.assertEqual(document, before)
        self.assertEqual(unassigned_segments(result), [])

    def test_explicit_retry_includes_exhausted_cases_and_advances_fresh_request_namespace(self):
        client = Client(unknown=True)
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = complete_speakers(missing_document(1), requests, workers=1)
            self.assertEqual(automatic_completion_targets(first), [])
            self.assertEqual(automatic_completion_targets(first, retry_exhausted=True), ["b1"])
            second = complete_speakers(first, requests, retry_unresolved=True, workers=1)
            third = complete_speakers(second, requests, retry_unresolved=True, workers=1)
        self.assertEqual(len(client.calls), 27)
        tokens = [client.calls[index]["completion_retry_token"] for index in (0, 9, 18)]
        self.assertIsNone(tokens[0])
        self.assertTrue(tokens[1])
        self.assertNotEqual(tokens[1], tokens[2])
        self.assertEqual(first["segments"], third["segments"])
        self.assertEqual(third["provenance"]["speaker_completion"]["retry_unresolved"], True)

    def test_legacy_finished_case_produces_a_retry_token(self):
        document = complete_speakers(missing_document(1), Requests(uncertain))
        document["provenance"]["speaker_completion"]["version"] = "1"
        requests = Requests()
        result = complete_speakers(document, requests, retry_unresolved=True)
        self.assertEqual(unassigned_segments(result), [])
        self.assertTrue(requests.calls[0]["completion_retry_token"])
        self.assertEqual(len(requests.calls), 3)

    def test_interrupted_retry_reuses_successful_votes_from_same_saved_document(self):
        document = complete_speakers(missing_document(1), Requests(uncertain))
        before = deepcopy(document)
        first, resumed = Client(fail_after=1), Client()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProviderError, "timeout"):
                complete_speakers(document, CachedRequests(first, directory, repair_attempts=0),
                                  workers=1, retry_unresolved=True)
            requests = CachedRequests(resumed, directory, repair_attempts=0)
            result = complete_speakers(document, requests, workers=1, retry_unresolved=True)
        self.assertEqual(document, before)
        self.assertEqual(requests.cache_hits, 1)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(resumed.calls), 2)
        self.assertEqual(first.calls[0]["completion_retry_token"], resumed.calls[0]["completion_retry_token"])
        self.assertEqual(unassigned_segments(result), [])

    def test_unrelated_manual_assignment_does_not_invalidate_target_vote_cache(self):
        document = complete_speakers(missing_document(2), Requests(uncertain))
        first, resumed = Client(fail_after=1), Client()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ProviderError):
                complete_speakers(document, CachedRequests(first, directory, repair_attempts=0),
                                  workers=1, target_block_ids=["b2"], retry_unresolved=True)
            edited = deepcopy(document)
            known = edited["segments"][0]
            known.update(role_id="narrator_main", voice_id="vn")
            known["attribution"].update(basis="user", review_required=False, candidates=[], reason="User choice.")
            requests = CachedRequests(resumed, directory, repair_attempts=0)
            result = complete_speakers(edited, requests, workers=1, target_block_ids=["b2"], retry_unresolved=True)
        self.assertEqual(requests.cache_hits, 1)
        self.assertEqual(len(resumed.calls), 2)
        self.assertEqual(first.calls[0]["completion_retry_token"], resumed.calls[0]["completion_retry_token"])
        self.assertEqual(first.calls[0]["input_block_fingerprint"], resumed.calls[0]["input_block_fingerprint"])
        self.assertNotIn("input_script_fingerprint", first.calls[0])
        self.assertEqual(result["provenance"]["speaker_completion"]["input_script_fingerprint"],
                         fingerprint(edited["segments"]))
        self.assertEqual(result["segments"][0], known)

    def test_retry_preserves_all_assigned_metadata_and_rejects_reassignment(self):
        for basis in ("context", "explicit", "user"):
            with self.subTest(basis=basis):
                document = partly_assigned_document(basis=basis)
                result = complete_speakers(document, Requests(), retry_unresolved=True)
                self.assertEqual(validate_document(result), [])
                self.assertEqual(result["segments"][0], document["segments"][0])
                self.assertEqual(unassigned_segments(result), [])

                def change_known(row, payload):
                    row["segments"][0].update(role_id="mara", kind="dialogue")
                    return row
                rejected = complete_speakers(document, Requests(change_known), retry_unresolved=True)
                self.assertEqual(rejected["segments"], document["segments"])
                self.assertTrue(all(vote["status"] == "error"
                                    for turn in rejected["provenance"]["speaker_completion"]["cases"][0]["rounds"]
                                    for vote in turn["votes"]))

    def test_retry_keeps_spans_resolved_in_earlier_rounds(self):
        def vary(row, payload):
            if payload["round"] == 1 and payload["reviewer_id"] == REVIEWERS[0][0]:
                row["segments"][1].update(role_id=None, basis="unresolved", review_required=True)
            elif payload["round"] == 2:
                row["segments"][0].update(role_id="mara", kind="dialogue")
            return row
        result = complete_speakers(missing_document(1), Requests(vary), retry_unresolved=True)
        self.assertEqual([item["role_id"] for item in result["segments"]], ["narrator_main", "mara"])
        self.assertEqual(unassigned_segments(result), [])

    def test_bracket_decoration_retains_assigned_ids_delivery_and_source_metadata_exactly(self):
        document = missing_document(1, bracket=True)
        known = next(item for item in document["segments"] if "performance" in item)
        known.update(id="saved-bracket-identity", role_id="mara", voice_id="v1", kind="dialogue")
        known["attribution"].update(basis="context", reason="Established speaker.", review_required=False, candidates=[])
        known["delivery"] = {"direction": "Softly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
        expected = deepcopy(known)
        result = complete_speakers(document, Requests(), retry_unresolved=True)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(next(item for item in result["segments"] if "performance" in item), expected)
        self.assertEqual(unassigned_segments(result), [])

    def test_one_unassigned_segment_splits_into_sequential_speakers_and_narration(self):
        document = paired_document()
        parts = [("“Yes,”", "mara", "dialogue"), (" Mara said. ", "narrator_main", "narration"),
                 ("“No,”", "lena", "dialogue"), (" Lena replied.", "narrator_main", "narration")]
        text = "".join(part[0] for part in parts)
        document["source"]["blocks"][0]["text"] = text
        initial = document["segments"][0]
        initial.update(id="unsplit-source", start=0, end=len(text), text=text)
        document["segments"] = [initial]

        class SplitRequests(Requests):
            def response(self, payload):
                evidence = [{"block_id": "b1", "quote": text}]
                return {"block_id": "b1", "reason": "Two sequential named speakers and their narrative tags.",
                        "evidence": evidence, "segments": [
                            {"text": piece, "role_id": role, "kind": kind, "basis": "explicit",
                             "reason": "The source names each successive speaker.", "evidence": evidence,
                             "candidates": [], "review_required": False, "delivery": {"direction": "", "evidence": []}}
                            for piece, role, kind in parts]}

        requests = SplitRequests()
        result = complete_speakers(document, requests, retry_unresolved=True)
        self.assertEqual(validate_document(result), [])
        self.assertEqual([(item["text"], item["role_id"], item["kind"]) for item in result["segments"]], parts)
        self.assertEqual("".join(item["text"] for item in result["segments"]), text)
        self.assertFalse(any(item.get("simultaneous_speakers") for item in result["segments"]))
        self.assertEqual(len(requests.calls), 3)

    def test_unassigned_dialogue_can_be_reclassified_as_narration_only_with_unanimous_kind(self):
        document = missing_document(1)
        document["segments"][0]["kind"] = "dialogue"

        def narrate(row, payload):
            for piece in row["segments"]:
                piece.update(role_id="narrator_main", kind="narration")
            return row

        result = complete_speakers(document, Requests(narrate), retry_unresolved=True)
        self.assertEqual(unassigned_segments(result), [])
        self.assertTrue(all(item["kind"] == "narration" for item in result["segments"]))

        def disagree_kind(row, payload):
            for piece in row["segments"]:
                piece.update(role_id="mara", kind="thought" if payload["reviewer_id"] == REVIEWERS[0][0] else "dialogue")
            return row

        unchanged = complete_speakers(document, Requests(disagree_kind), retry_unresolved=True)
        self.assertEqual(unchanged["segments"], document["segments"])
        self.assertIn("typography is evidence", SYSTEM)
        self.assertIn("old unassigned", SYSTEM)
        self.assertIn("Adjacent calls may be sequential", SYSTEM)


if __name__ == "__main__":
    unittest.main()
