"""Every unassigned spoken block receives a blind, source-bound three-way jury."""

from concurrent.futures import CancelledError
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
import tempfile
import time
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.performance import apply_bracket_delivery
from epub_tts_narrator.pipeline import CachedRequests, analyze, fingerprint
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.speaker_completion import (
    REVIEWERS, SYSTEM, _final_context, _unanimous, complete_speakers, unassigned_segments,
)
from test_attribution_review import fixture, correction
from test_contracts import valid_document
from test_pipeline import FakeClient


def missing_document(count=2, **kwargs):
    document = fixture(count=count, **kwargs)
    for item in document["segments"]:
        item.update(role_id=None, voice_id=None)
        item["attribution"].update(basis="unresolved", review_required=True, reason="Unknown speaker.")
    return document


def response(payload):
    result = correction(payload["target_units"][0])
    return result


def paired_document():
    document = missing_document(1)
    document["cast"]["characters"].append({"id": "c2", "name": "Lena", "aliases": [], "description": "Another speaker.", "evidence": [], "voice_id": "v2"})
    document["cast"]["voices"].append({"id": "v2", "description": "Second voice.", "traits": []})
    document["cast"]["roles"].append({"id": "lena", "kind": "dialogue", "voice_id": "v2", "character_id": "c2", "description": "Lena speaking."})
    parts = [("“No!”", "dialogue"), (" Mara and Lena said together. ", "narration"), ("“Actually, yes.”", "dialogue")]
    text = "".join(part for part, _ in parts)
    document["source"]["blocks"][0]["text"] = text
    document["cast"]["characters"][0]["evidence"] = []
    original = document["segments"].pop()
    start = 0
    for text, kind in parts:
        item = deepcopy(original)
        item.update(text=text, kind=kind, start=start, end=start + len(text), id=f"b1:{start}:{start + len(text)}")
        document["segments"].append(item)
        start += len(text)
    return document


class Requests:
    def __init__(self, modify=None):
        self.modify = modify
        self.calls = []
        self.systems = []
        self.lock = Lock()
        self.active = self.max_active = 0

    def response(self, payload):
        result = response(payload)
        return self.modify(result, payload) if self.modify else result

    def run(self, system, payload, validate, *, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Cancelled")
        with self.lock:
            self.calls.append(deepcopy(payload))
            self.systems.append(system)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(.004)
            return validate(self.response(payload))
        finally:
            with self.lock:
                self.active -= 1


class PairedRequests(Requests):
    def response(self, payload):
        evidence = [{"block_id": "b1", "quote": "Mara and Lena said together"}]
        rows = []
        for index, original in enumerate(payload["original_segments"]):
            item = {key: deepcopy(original[key]) for key in ("text", "kind", "delivery")}
            item.update(role_id="narrator_main" if index == 1 else None, basis="explicit",
                        reason="The source names both speakers.", evidence=evidence, candidates=[], review_required=False)
            if index != 1:
                item["simultaneous_role_ids"] = ["mara", "lena"]
            rows.append(item)
        result = {"block_id": "b1", "reason": "The source names both speakers.", "evidence": evidence, "segments": rows}
        return self.modify(result, payload) if self.modify else result


class SpeakerCompletionTests(unittest.TestCase):
    def test_pipeline_always_runs_completion_even_if_review_is_disabled_or_scanner_omits_block(self):
        class ResolvingClient(FakeClient):
            def complete(self, system, payload):
                if payload.get("mode") not in {"speaker_completion", "attribution_review_scan"}:
                    return super().complete(system, payload)
                self.calls.append((system, deepcopy(payload)))
                if payload["mode"] == "attribution_review_scan":
                    return {"corrections": []}
                unit = payload["target_units"][0]
                evidence = [{"block_id": unit["block_id"], "quote": "Mara said softly"}]
                role = next(row["id"] for row in payload["final_cast"]["roles"] if row["kind"] == "dialogue")
                keys = ("text", "kind", "role_id", "basis", "reason", "evidence", "candidates", "review_required", "delivery")
                pieces = [{key: deepcopy(item[key]) for key in keys} for item in payload["original_segments"]]
                for piece in pieces:
                    piece.update(role_id=role if piece["kind"] == "dialogue" else "narrator_main",
                                 basis="explicit", reason="The source names Mara.", evidence=evidence,
                                 candidates=[], review_required=False)
                return {"block_id": unit["block_id"], "reason": "Mara speaks after a narration tag.",
                        "evidence": evidence, "segments": pieces}
        for semantic_review in (False, True):
            with self.subTest(semantic_review=semantic_review), tempfile.TemporaryDirectory() as directory:
                client = ResolvingClient(unresolved=True)
                result = analyze(valid_document()["source"], client, Path(directory), identity_consensus=False,
                                 attribution_review=semantic_review, attribution_workers=3, repair_attempts=0)
                completions = [payload for _, payload in client.calls if payload.get("mode") == "speaker_completion"]
                self.assertEqual(len(completions), 3)
                self.assertEqual(unassigned_segments(result), [])
                self.assertTrue(result["readiness"]["ready"])
                self.assertEqual(result["provenance"]["speaker_completion"]["resolved_blocks"], 1)

    def test_every_unassigned_block_is_juried_without_a_scanner(self):
        document, requests = missing_document(4), Requests()
        before = deepcopy(document)
        result = complete_speakers(document, requests)
        self.assertEqual(document, before)
        self.assertEqual(result["source"], document["source"])
        self.assertEqual(result["cast"], document["cast"])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(unassigned_segments(result), [])
        self.assertEqual(len(requests.calls), 12)
        self.assertTrue(all(call["mode"] == "speaker_completion" for call in requests.calls))
        self.assertEqual({call["target_units"][0]["block_id"] for call in requests.calls}, {"b1", "b2", "b3", "b4"})
        audit = result["provenance"]["speaker_completion"]
        self.assertEqual(audit["status"], "complete")
        self.assertEqual(audit["resolved_blocks"], 4)
        self.assertEqual(audit["input_script_fingerprint"], fingerprint(document["segments"]))
        for case in audit["cases"]:
            votes = case["rounds"][0]["votes"]
            self.assertEqual(len(votes), 3)
            self.assertEqual({vote["reviewer_id"] for vote in votes}, {row[0] for row in REVIEWERS})
            self.assertTrue(all(vote["raw_response"] for vote in votes))
            self.assertEqual(case["status"], "resolved")
        self.assertTrue(result["readiness"]["ready"])

    def test_second_round_expands_context_without_peer_votes(self):
        def modify(result, payload):
            if payload["round"] == 1 and payload["reviewer_id"] == REVIEWERS[1][0]:
                result["segments"][1].update(role_id="narrator_main", kind="narration")
            return result
        document, requests = missing_document(50, long=True), Requests(modify)
        result = complete_speakers(document, requests, target_block_ids=["b25"])
        self.assertEqual(len(requests.calls), 6)
        first, second = result["provenance"]["speaker_completion"]["cases"][0]["rounds"]
        self.assertEqual(first["status"], "partial")
        self.assertEqual(second["status"], "unanimous")
        self.assertGreater(second["context_chars"], first["context_chars"])
        self.assertTrue(all("votes" not in call and "proposals" not in call and "peer_answers" not in call for call in requests.calls))
        self.assertTrue(all(call["original_segments"] == requests.calls[0]["original_segments"] for call in requests.calls))
        self.assertTrue(all(system == SYSTEM for system in requests.systems))
        self.assertEqual(result["provenance"]["speaker_completion"]["remaining_unassigned_segments"], 49)
        self.assertEqual(result["segments"][0], document["segments"][0])

    def test_no_consensus_retains_nulls_and_reports_blocked(self):
        def modify(result, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                result["segments"][1].update(role_id=None, basis="unresolved", review_required=True)
            return result
        document, requests = missing_document(), Requests(modify)
        result = complete_speakers(document, requests)
        self.assertEqual([item["text"] for item in unassigned_segments(result)], ["“Hello.”"] * 2)
        self.assertEqual([item["role_id"] for item in result["segments"]], ["narrator_main", None] * 2)
        self.assertEqual(len(requests.calls), 18)
        self.assertEqual(result["provenance"]["speaker_completion"]["status"], "blocked")
        self.assertEqual(result["issues"], document["issues"])
        self.assertFalse(result["readiness"]["ready"])

    def test_three_matching_unknown_answers_are_not_resolved(self):
        def unresolved(row, payload):
            for piece in row["segments"]:
                piece.update(role_id=None, basis="unresolved", review_required=True, candidates=["mara"])
            return row
        document, requests = missing_document(1), Requests(unresolved)
        result = complete_speakers(document, requests)
        case = result["provenance"]["speaker_completion"]["cases"][0]
        self.assertEqual(len(requests.calls), 9)
        self.assertEqual(case["status"], "unresolved")
        self.assertTrue(all(vote["status"] == "valid" and not vote["resolved"]
                            for turn in case["rounds"] for vote in turn["votes"]))
        self.assertEqual(result["segments"], document["segments"])

    def test_wrong_boundary_is_real_disagreement_even_with_same_roles(self):
        def shift(row, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                row["segments"][0]["text"] += row["segments"][1]["text"][:2]
                row["segments"][1]["text"] = row["segments"][1]["text"][2:]
            return row
        document = missing_document(1)
        result = complete_speakers(document, Requests(shift))
        self.assertEqual([item["text"] for item in unassigned_segments(result)], ["“H"])
        self.assertEqual("".join(item["text"] for item in result["segments"]), document["source"]["blocks"][0]["text"])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["provenance"]["speaker_completion"]["unresolved_blocks"], 1)

    def test_partial_completion_discards_only_stale_normalization_on_split_spans(self):
        from epub_tts_narrator.spoken_text import build_spoken

        document = missing_document(2)
        for block, row in zip(document["source"]["blocks"], document["segments"]):
            text = block["text"].replace("Hello", "Hellooo")
            block["text"] = text
            row.update(text=text, end=len(text), id=f"{block['id']}:0:{len(text)}")
            start = text.index("Hellooo")
            row["speech_normalization"] = {"version": 1, "edits": [{
                "start": start, "end": start + len("Hellooo"), "original": "Hellooo",
                "replacement": "Hello", "direction": "Prolong the final vowel.",
                "reason": "Repeated vowel conveys an elongated greeting.",
            }]}
            row["spoken_text"] = build_spoken(row)[0]
        self.assertEqual(validate_document(document), [])
        original = deepcopy(document)

        def shift(response, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                response["segments"][0]["text"] += response["segments"][1]["text"][:2]
                response["segments"][1]["text"] = response["segments"][1]["text"][2:]
            return response

        result = complete_speakers(document, Requests(shift), target_block_ids=["b1"],
                                   retry_unresolved=True)

        self.assertEqual(document, original)
        self.assertEqual(validate_document(result), [])
        first = [row for row in result["segments"] if row["block_id"] == "b1"]
        self.assertEqual("".join(row["text"] for row in first), original["source"]["blocks"][0]["text"])
        self.assertEqual([row["text"] for row in first if row["role_id"] is None], ["“H"])
        self.assertTrue(all("speech_normalization" not in row for row in first))
        self.assertEqual([row for row in result["segments"] if row["block_id"] == "b2"],
                         [original["segments"][1]])

    def test_source_edits_unknown_roles_unseen_evidence_and_duplicate_outputs_fail_closed(self):
        mutations = [
            lambda row: row["segments"][1].update(text="Changed source"),
            lambda row: row["segments"][1].update(role_id="invented"),
            lambda row: row["segments"][1]["evidence"][0].update(block_id="hidden"),
            lambda row: row["segments"][1]["evidence"][0].update(quote="invented words"),
            lambda row: row.update(block_id="b99"),
            lambda row: row.update(cast={}),
            lambda row: row["segments"].append(deepcopy(row["segments"][0])),
            lambda row: row["segments"][1].update(evidence=[]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                def modify(row, payload):
                    mutation(row)
                    return row
                document = missing_document(1)
                result = complete_speakers(document, Requests(modify))
                self.assertEqual(result["segments"], document["segments"])
                self.assertEqual(result["provenance"]["speaker_completion"]["status"], "blocked")
                self.assertTrue(all(vote["status"] == "error" and vote["raw_response"] is not None
                                    for turn in result["provenance"]["speaker_completion"]["cases"][0]["rounds"] for vote in turn["votes"]))

    def test_source_offsets_and_deterministic_bracket_cues_survive(self):
        document = missing_document(1, bracket=True)
        result = complete_speakers(document, Requests())
        self.assertEqual(validate_document(result), [])
        self.assertEqual("".join(item["text"] for item in result["segments"]), document["source"]["blocks"][0]["text"])
        for item in result["segments"]:
            self.assertEqual(item["text"], document["source"]["blocks"][0]["text"][item["start"]:item["end"]])
        bracket = [item for item in result["segments"] if "performance" in item]
        self.assertEqual(len(bracket), 1)
        self.assertEqual(bracket[0]["spoken_text"], "Friend")
        self.assertEqual(bracket[0]["role_id"], "mara")

    def test_user_assignment_is_preserved_while_unassigned_remainder_completes(self):
        document = missing_document(1)
        initial = document["segments"][0]
        cut = initial["text"].index("“")
        protected = deepcopy(initial)
        protected.update(id="user-kept-id", text=initial["text"][:cut], end=cut, role_id="narrator_main", voice_id="vn")
        protected["attribution"].update(basis="user", review_required=False, reason="User correction.")
        remainder = deepcopy(initial)
        remainder.update(id=f"b1:{cut}:{len(initial['text'])}", text=initial["text"][cut:], start=cut)
        document["segments"] = [protected, remainder]
        result = complete_speakers(document, Requests())
        self.assertEqual(result["segments"][0], protected)
        self.assertEqual(result["segments"][1]["role_id"], "mara")
        def bad(row, payload):
            row["segments"][0].update(role_id="mara", kind="dialogue")
            return row
        blocked = complete_speakers(document, Requests(bad))
        self.assertEqual(blocked["segments"], document["segments"])

    def test_existing_delivery_cannot_be_dropped_or_rewritten(self):
        document = missing_document(1)
        document["segments"][0]["delivery"] = {"direction": "Quietly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
        def vary(row, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                for item in row["segments"]:
                    item["delivery"] = {"direction": "Softly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
            return row
        result = complete_speakers(document, Requests(vary))
        self.assertEqual(unassigned_segments(result), [])
        self.assertTrue(all(item["delivery"] == document["segments"][0]["delivery"] for item in result["segments"]))
        votes = result["provenance"]["speaker_completion"]["cases"][0]["rounds"][0]["votes"]
        self.assertEqual(len({vote["assignment_fingerprint"] for vote in votes}), 1)
        self.assertEqual([vote["raw_response"]["segments"][0]["delivery"]["direction"] for vote in votes], ["", "Softly", ""])

    def test_invalid_delivery_evidence_is_still_rejected_before_normalization(self):
        def invalid(row, payload):
            row["segments"][0]["delivery"] = {"direction": "Quietly", "evidence": [{"block_id": "b1", "quote": "fabricated words"}]}
            return row
        document = missing_document(1)
        result = complete_speakers(document, Requests(invalid))
        self.assertEqual(result["segments"], document["segments"])
        self.assertEqual(result["provenance"]["speaker_completion"]["status"], "blocked")

    def test_only_covered_attribution_issues_are_resolved(self):
        document = missing_document(1)
        document["issues"] = [
            {"id": "a", "kind": "attribution_consensus", "target": "b1", "reason": "Uncertain speaker.", "resolved": False},
            {"id": "b", "kind": "speaker_attribution", "target": document["segments"][0]["id"], "reason": "Uncertain speaker.", "resolved": False},
            {"id": "c", "kind": "casting", "target": "b1", "reason": "Casting review.", "resolved": False},
            {"id": "d", "kind": "identity", "target": "c1", "reason": "Identity review.", "resolved": False},
        ]
        result = complete_speakers(document, Requests())
        self.assertEqual([item["resolved"] for item in result["issues"]], [True, True, False, False])
        self.assertEqual(result["issues"][2:], document["issues"][2:])
        self.assertTrue(result["readiness"]["ready"])

    def test_cosmetic_segmentation_and_reasons_do_not_break_consensus(self):
        def modify(row, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                item = row["segments"].pop(0)
                first, second = deepcopy(item), deepcopy(item)
                first["text"], second["text"] = item["text"][:5], item["text"][5:]
                first["reason"] = "Different evidence explanation."
                row["segments"][0:0] = [first, second]
            return row
        result = complete_speakers(missing_document(1), Requests(modify))
        self.assertEqual(result["provenance"]["speaker_completion"]["resolved_blocks"], 1)

    def test_unanimous_quote_applies_even_when_another_quote_in_same_block_is_disputed(self):
        def dissent(row, payload):
            if payload["reviewer_id"] == REVIEWERS[1][0]:
                row["segments"][2].pop("simultaneous_role_ids")
                row["segments"][2]["role_id"] = "mara"
            return row
        document = paired_document()
        document["issues"] = [{"id": str(index), "kind": "speaker_attribution", "target": target,
                               "reason": "Uncertain speaker.", "resolved": False}
                              for index, target in enumerate(["b1", document["segments"][0]["id"], document["segments"][2]["id"]])]
        result = complete_speakers(document, PairedRequests(dissent))
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["source"], document["source"])
        self.assertEqual(result["cast"], document["cast"])
        self.assertEqual(result["segments"][0]["simultaneous_speakers"], [
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"}])
        self.assertEqual(result["segments"][2], document["segments"][2])
        self.assertEqual([item["text"] for item in unassigned_segments(result)], ["“Actually, yes.”"])
        self.assertEqual([issue["resolved"] for issue in result["issues"]], [False, True, False])
        self.assertFalse(result["readiness"]["ready"])
        audit = result["provenance"]["speaker_completion"]
        self.assertEqual((audit["resolved_blocks"], audit["updated_blocks"], audit["remaining_unassigned_segments"]), (0, 1, 1))
        self.assertEqual(audit["cases"][0]["status"], "partial")
        self.assertEqual(len(audit["cases"][0]["rounds"]), 3)

    def test_common_boundaries_accept_equivalent_ensemble_segmentation_without_splitting_audio_text(self):
        def split(row, payload):
            if payload["reviewer_id"] in {REVIEWERS[0][0], REVIEWERS[1][0]}:
                quote = row["segments"].pop(0)
                pieces = [deepcopy(quote), deepcopy(quote)]
                at = 2 if payload["reviewer_id"] == REVIEWERS[0][0] else 3
                pieces[0]["text"], pieces[1]["text"] = quote["text"][:at], quote["text"][at:]
                pieces[1]["reason"] = "A cosmetic split with a different explanation."
                pieces[1]["basis"] = "context"
                row["segments"][0:0] = pieces
            return row
        document = paired_document()
        result = complete_speakers(document, PairedRequests(split))
        self.assertEqual(validate_document(result), [])
        self.assertEqual(unassigned_segments(result), [])
        self.assertEqual([item["text"] for item in result["segments"]], [item["text"] for item in document["segments"]])

    def test_unrelated_review_is_retained_as_stable_issue_when_assignment_becomes_simultaneous(self):
        document = paired_document()
        document["segments"][0].update(role_id="mara", voice_id="v1")
        document["segments"][0]["attribution"].update(basis="context", reason="Check performance nuance.", review_required=True)
        first = complete_speakers(document, PairedRequests())
        second = complete_speakers(document, PairedRequests())
        self.assertEqual(validate_document(first), [])
        self.assertEqual(unassigned_segments(first), [])
        self.assertFalse(first["segments"][0]["attribution"]["review_required"])
        self.assertEqual(first["issues"], second["issues"])
        self.assertEqual(len(first["issues"]), 1)
        issue = first["issues"][0]
        self.assertEqual(issue["kind"], "preserved_segment_review")
        self.assertEqual(issue["reason"], "Check performance nuance.")
        self.assertEqual(issue["source_segment_id"], document["segments"][0]["id"])
        self.assertFalse(issue["resolved"])
        self.assertTrue(first["readiness"]["ready"])

    def test_final_round_retrieves_candidate_identity_evidence_across_chapters_without_peer_answers(self):
        document = missing_document(2)
        evidence_block = document["source"]["blocks"][1]
        evidence_block["chapter_id"] = "ch2"
        evidence_block["text"] = "Mara is the inner voice."
        document["source"]["chapters"][0]["block_ids"] = ["b1"]
        chapter = deepcopy(document["source"]["chapters"][0])
        chapter.update(id="ch2", title="Later", block_ids=["b2"])
        document["source"]["chapters"].append(chapter)
        document["segments"][1].update(text=evidence_block["text"], end=len(evidence_block["text"]), role_id="narrator_main", voice_id="vn")
        document["segments"][1]["attribution"].update(basis="context", review_required=False)
        document["segments"][0]["attribution"]["candidates"] = ["mara"]
        document["cast"]["characters"][0]["evidence"] = [{"block_id": "b2", "quote": evidence_block["text"]}]
        def resolve_with_evidence(row, payload):
            if payload["round"] < 3:
                row["segments"][1].update(role_id=None, basis="unresolved", review_required=True, candidates=["mara"])
            else:
                row["segments"][1]["evidence"] = [{"block_id": "b2", "quote": "Mara is the inner voice."}]
            return row
        requests = Requests(resolve_with_evidence)
        result = complete_speakers(document, requests)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(unassigned_segments(result), [])
        self.assertEqual(len(requests.calls), 9)
        self.assertTrue(all(not call["context_units"] for call in requests.calls if call["round"] < 3))
        final_calls = [call for call in requests.calls if call["round"] == 3]
        self.assertTrue(all([unit["block_id"] for unit in call["context_units"]] == ["b2"] for call in final_calls))
        self.assertTrue(all("votes" not in call and "proposals" not in call and "peer_answers" not in call for call in requests.calls))
        self.assertTrue(all(call["original_segments"] == requests.calls[0]["original_segments"] for call in requests.calls))
        self.assertTrue(all(system == SYSTEM for system, call in zip(requests.systems, requests.calls) if call["round"] < 3))
        self.assertTrue(all("Trace plural" in system and system.startswith(SYSTEM)
                            for system, call in zip(requests.systems, requests.calls) if call["round"] == 3))
        audit = result["provenance"]["speaker_completion"]
        self.assertEqual(audit["cases"][0]["resolution_round"], 3)

    def test_final_context_retrieval_is_bounded_unique_and_visible(self):
        document = missing_document(30, long=True)
        document["segments"][10]["attribution"]["candidates"] = ["mara"]
        document["cast"]["characters"][0]["evidence"] = [{"block_id": "b25", "quote": "Mara said softly"}]
        context = _final_context(document, 10, 1200, [document["segments"][10]], [])
        self.assertLessEqual(sum(len(item["text"]) for item in context), 1200)
        self.assertEqual(len({item["block_id"] for item in context}), len(context))
        self.assertNotIn("b11", {item["block_id"] for item in context})
        source = {block["id"]: block["text"] for block in document["source"]["blocks"]}
        self.assertTrue(all(item["text"] in source[item["block_id"]] for item in context))

    def test_duplicate_reviewer_or_request_is_not_three_way_consensus(self):
        votes = [{"reviewer_id": reviewer, "request_fingerprint": str(index), "status": "valid", "resolved": True,
                  "assignment_fingerprint": "same"} for index, (reviewer, _) in enumerate(REVIEWERS)]
        self.assertTrue(_unanimous(votes))
        duplicate = deepcopy(votes)
        duplicate[2] = deepcopy(duplicate[1])
        self.assertFalse(_unanimous(duplicate))
        duplicate[2]["reviewer_id"] = REVIEWERS[2][0]
        self.assertFalse(_unanimous(duplicate))

    def test_simultaneous_known_speakers_resolve_without_a_primary_voice(self):
        document = missing_document(1)
        document["cast"]["characters"].append({"id": "c2", "name": "Lena", "aliases": [], "description": "Another speaker.", "evidence": [], "voice_id": "v2"})
        document["cast"]["voices"].append({"id": "v2", "description": "Second voice.", "traits": []})
        document["cast"]["roles"].append({"id": "lena", "kind": "dialogue", "voice_id": "v2", "character_id": "c2", "description": "Lena speaking."})
        # Evidence is validated lexically here; the independent reviewers assess
        # its semantic support. The fixture explicitly says the two speak together.
        old = document["source"]["blocks"][0]["text"]
        text = "Mara and Lena said together, “Hello.”"
        document["source"]["blocks"][0]["text"] = text
        document["segments"][0].update(text=text, end=len(text), id=f"b1:0:{len(text)}")
        document["cast"]["characters"][0]["evidence"] = []
        def group(row, payload):
            before, after = text.split("“", 1)
            evidence = [{"block_id": "b1", "quote": "Mara and Lena said together"}]
            row.update(reason="Two named speakers speak together.", evidence=evidence)
            row["segments"][0].update(text=before, evidence=evidence)
            row["segments"][1].update(text="“" + after, role_id=None, simultaneous_role_ids=["mara", "lena"], evidence=evidence)
            if payload["reviewer_id"] == REVIEWERS[2][0]:
                row["segments"][1]["simultaneous_role_ids"].reverse()
            return row
        # Use a response builder independent of correction()'s single-speaker tag.
        class GroupRequests(Requests):
            def response(self, payload):
                base = response({"target_units": [{"block_id": "b1", "text": old}]})
                return group(base, payload)
        result = complete_speakers(document, GroupRequests())
        self.assertEqual(validate_document(result), [])
        self.assertEqual(unassigned_segments(result), [])
        piece = result["segments"][1]
        self.assertIsNone(piece["role_id"])
        self.assertEqual({row["role_id"] for row in piece["simultaneous_speakers"]}, {"mara", "lena"})
        skipped = Requests()
        complete_speakers(result, skipped)
        self.assertEqual(skipped.calls, [])

    def test_global_limit_is_five_and_all_cases_covered(self):
        requests = Requests()
        result = complete_speakers(missing_document(8), requests, workers=5)
        self.assertEqual(requests.max_active, 5)
        self.assertEqual(len(requests.calls), 24)
        self.assertEqual(len(result["provenance"]["speaker_completion"]["cases"]), 8)

    def test_cache_replay_reproduces_results_without_new_model_calls(self):
        class Client(Requests):
            identity = {"model": "fixture", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = complete_speakers(missing_document(2), requests)
            count = len(client.calls)
            second = complete_speakers(missing_document(2), requests)
        self.assertEqual(count, 6)
        self.assertEqual(len(client.calls), count)
        self.assertEqual(first, second)

    def test_all_three_rounds_replay_from_cache_without_new_calls(self):
        class Client(Requests):
            identity = {"model": "fixture", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                row = self.response(payload)
                for piece in row["segments"]:
                    piece.update(role_id=None, basis="unresolved", review_required=True, candidates=["mara"])
                return row
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = complete_speakers(missing_document(1), requests)
            second = complete_speakers(missing_document(1), requests)
        self.assertEqual(len(client.calls), 9)
        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["speaker_completion"]["status"], "blocked")

    def test_transport_failure_leaves_a_case_for_every_target_and_propagates(self):
        class FailedRequests(Requests):
            def run(self, system, payload, validate, *, cancelled=None):
                raise ProviderError("Timed out")
        with tempfile.TemporaryDirectory() as directory:
            requests = FailedRequests()
            requests.cache_dir = Path(directory)
            with self.assertRaises(ProviderError):
                complete_speakers(missing_document(4), requests, workers=1)
            paths = list(Path(directory).glob("speaker-completion-audits/*.json"))
            self.assertEqual(len(paths), 1)
            record = json.loads(paths[0].read_text())
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(len(record["cases"]), 4)
            self.assertEqual(record["cases"][0]["rounds"][0]["votes"][0]["status"], "interrupted")

    def test_no_unassigned_speech_makes_no_calls_and_ignores_punctuation_only(self):
        requests = Requests()
        result = complete_speakers(fixture(), requests)
        self.assertEqual(requests.calls, [])
        self.assertEqual(result["provenance"]["speaker_completion"]["cases"], [])
        document = missing_document(1)
        document["source"]["blocks"][0]["text"] = "…"
        document["cast"]["characters"][0]["evidence"] = []
        document["segments"][0].update(text="…", end=1)
        complete_speakers(document, requests)
        self.assertEqual(requests.calls, [])


if __name__ == "__main__":
    unittest.main()
