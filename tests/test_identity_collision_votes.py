"""Batched collision vote responses must cover each case exactly once with the identity vote contract."""

from copy import deepcopy
from json import dumps
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from epub_tts_narrator.identity_collisions import COLLISION_VERSION, COLLISION_VOTE_SYSTEM, resolve_collisions
from epub_tts_narrator.identity_prompts import LENSES
from epub_tts_narrator.identity_vote_validation import make_batch_vote_accept, make_vote_accept
from epub_tts_narrator.pipeline import fingerprint
from test_identity import fixture


class CaptureVotes:
    """Offline fake recording identity_collision_votes payloads."""

    def __init__(self, limit=200000):
        self.calls = []
        self.client = SimpleNamespace(identity={"model": "offline-votes"})
        self.cache_dir = Path(tempfile.gettempdir())
        self.max_request_chars = limit

    def run(self, system, payload, validate, *, cancelled=None):
        self.calls.append(deepcopy(payload))
        votes = []
        for entry in payload["cases"]:
            votes.append({"case_id": entry["case_id"], "verdict": "approve",
                          "reason": "The narrative continues one person across both records beyond the shared name.",
                          "evidence": deepcopy(entry["evidence"][:1])})
        return validate({"votes": votes})


def votes_fixture():
    source, cast = fixture(("Trask", "Trask", "Trask", "Trask"))
    source["blocks"][0]["text"] = "Trask spoke. The captain was called Red."
    requests = CaptureVotes()
    _, _, provenance = resolve_collisions(source, cast, requests)
    payloads = [payload for payload in requests.calls
                if payload["original_cast_fingerprint"] == fingerprint(cast)]
    return source, payloads


def good_vote(payload, index, **overrides):
    entry = payload["cases"][index]
    vote = {"case_id": entry["case_id"], "verdict": "approve",
            "reason": "The narrative continues one person across both records beyond the shared name.",
            "evidence": deepcopy(entry["evidence"][:1])}
    return dict(vote, **overrides)


def accept_batch(payload, response, source, reviewer="identity_reviewer_1", lens=None):
    accept = make_batch_vote_accept(reviewer=reviewer, lens=lens or LENSES[0][1], payload=payload,
                                    source=source, request_fingerprint="request-fingerprint")
    return accept(response)


class BatchVoteValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.payloads = votes_fixture()
        cls.payload = cls.payloads[0]

    def test_exact_batch_coverage_returns_case_bound_identity_votes(self):
        response = {"votes": [good_vote(self.payload, 0), good_vote(self.payload, 1)]}
        votes = accept_batch(self.payload, response, self.source)
        self.assertEqual(len(votes), 2)
        entry = self.payload["cases"][0]
        vote = votes[0]
        self.assertEqual(vote["case_id"], entry["case_id"])
        self.assertEqual(vote["reviewer_id"], "identity_reviewer_1")
        self.assertEqual(vote["lens"], LENSES[0][1])
        self.assertEqual(vote["verdict"], "approve")
        self.assertEqual(vote["request_fingerprint"], "request-fingerprint")
        self.assertEqual(vote["source_fingerprint"], entry["source_fingerprint"])
        self.assertEqual(vote["original_cast_fingerprint"], entry["original_cast_fingerprint"])
        self.assertEqual(vote["proposed_cast_fingerprint"], entry["proposed_cast_fingerprint"])
        self.assertEqual(vote["patch_fingerprint"], entry["patch_fingerprint"])

    def test_batch_vote_matches_the_exact_single_identity_vote_contract(self):
        entry = self.payload["cases"][0]
        single = make_vote_accept(reviewer="identity_reviewer_1", lens=LENSES[0][1], source=self.source,
                                  source_units=entry["source_units"], cast_view=entry["original_cast"],
                                  identity_case={"kind": entry["kind"], "character_ids": entry["character_ids"]},
                                  proposed_cast=entry["proposed_cast"], request_fingerprint="request-fingerprint",
                                  fingerprints={"source_fingerprint": entry["source_fingerprint"],
                                                "original_cast_fingerprint": entry["original_cast_fingerprint"],
                                                "proposed_cast_fingerprint": entry["proposed_cast_fingerprint"],
                                                "patch_fingerprint": entry["patch_fingerprint"]})
        response = {"verdict": "approve", "reason": good_vote(self.payload, 0)["reason"],
                    "evidence": deepcopy(entry["evidence"][:1])}
        batched = accept_batch(self.payload, {"votes": [good_vote(self.payload, 0), good_vote(self.payload, 1)]},
                               self.source)[0]
        self.assertEqual({key: value for key, value in batched.items() if key != "case_id"}, single(response))

    def test_malformed_batches_are_rejected_without_accepting_any_vote(self):
        votes = [good_vote(self.payload, 0), good_vote(self.payload, 1)]
        duplicates = [deepcopy(votes[0]), deepcopy(votes[0])]
        unknown = [deepcopy(votes[0]), dict(deepcopy(votes[1]), case_id="identity_collision_unknown")]
        extra_field = [dict(deepcopy(votes[0]), confidence=1), deepcopy(votes[1])]
        missing_field = [{key: value for key, value in votes[0].items() if key != "evidence"}, deepcopy(votes[1])]
        bad_verdict = [dict(deepcopy(votes[0]), verdict="maybe"), deepcopy(votes[1])]
        empty_reason = [dict(deepcopy(votes[0]), reason="   "), deepcopy(votes[1])]
        long_reason = [dict(deepcopy(votes[0]), reason="x" * 2401), deepcopy(votes[1])]
        name_only = [dict(deepcopy(votes[0]), evidence=[{"block_id": "b1", "quote": "Trask"}]), deepcopy(votes[1])]
        hidden_quote = [dict(deepcopy(votes[0]), evidence=[{"block_id": "b1", "quote": "They left."}]),
                        deepcopy(votes[1])]
        unknown_block = [dict(deepcopy(votes[0]), evidence=[{"block_id": "bZ", "quote": "Trask spoke."}]),
                         deepcopy(votes[1])]
        not_a_dict = ["approve", deepcopy(votes[1])]
        matrices = [
            ("votes key missing", {"Votes": votes}, "exactly a votes array"),
            ("extra envelope field", {"votes": votes, "note": "extra"}, "exactly a votes array"),
            ("votes not a list", {"votes": {"0": votes[0]}}, "exactly a votes array"),
            ("omitted case", {"votes": votes[:1]}, "exactly 2 vote"),
            ("duplicate case", {"votes": duplicates}, "duplicates case_id"),
            ("unknown case", {"votes": unknown}, "unknown case_id"),
            ("extra vote field", {"votes": extra_field}, "exactly case_id, verdict, reason, and evidence"),
            ("missing vote field", {"votes": missing_field}, "exactly case_id, verdict, reason, and evidence"),
            ("vote not an object", {"votes": not_a_dict}, "exactly case_id, verdict, reason, and evidence"),
            ("invalid verdict", {"votes": bad_verdict}, "approve, reject, or uncertain"),
            ("blank reason", {"votes": empty_reason}, "concise source-bound reason"),
            ("oversized reason", {"votes": long_reason}, "concise source-bound reason"),
            ("name-only approval", {"votes": name_only}, "beyond names or aliases alone"),
            ("citation outside visible units", {"votes": hidden_quote}, "visible"),
            ("unknown source block", {"votes": unknown_block}, "unknown source block"),
        ]
        for name, response, expected in matrices:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected):
                    accept_batch(self.payload, response, self.source)

    def test_rejection_needs_visible_evidence_but_not_context_beyond_names(self):
        entry = self.payload["cases"][0]
        vote = good_vote(self.payload, 0, verdict="reject",
                         evidence=[{"block_id": "b1", "quote": "Trask spoke."}])
        votes = accept_batch(self.payload, {"votes": [vote, good_vote(self.payload, 1)]}, self.source)
        self.assertEqual(votes[0]["verdict"], "reject")

    def test_every_reviewer_sees_identical_case_regions_with_private_identity(self):
        self.assertEqual(len(self.payloads), 3)
        case_regions = {dumps(payload["cases"], ensure_ascii=False, sort_keys=True) for payload in self.payloads}
        self.assertEqual(len(case_regions), 1)
        self.assertEqual({payload["reviewer_id"] for payload in self.payloads}, {row[0] for row in LENSES})
        self.assertEqual({payload["lens"] for payload in self.payloads}, {row[1] for row in LENSES})
        for payload in self.payloads:
            self.assertEqual(payload["mode"], "identity_collision_votes")
            self.assertEqual(payload["collision_version"], COLLISION_VERSION)
            self.assertEqual(list(payload)[-2:], ["reviewer_id", "lens"])

    def test_capacity_limit_splits_identical_halves_for_every_reviewer(self):
        source, cast = fixture(tuple(["Trask"] * 16))
        probe = CaptureVotes()
        resolve_collisions(source, cast, probe)
        worst_reviewer, worst_lens = max(LENSES, key=lambda row: len(row[0]) + len(row[1]))
        base = max(len(COLLISION_VOTE_SYSTEM) + len(dumps(dict(payload, reviewer_id=worst_reviewer,
                                                               lens=worst_lens), ensure_ascii=False))
                   for payload in probe.calls)
        limited = CaptureVotes(limit=base - 1)
        updated, issues, provenance = resolve_collisions(source, cast, limited)
        wave1 = [payload for payload in limited.calls
                 if payload["original_cast_fingerprint"] == fingerprint(cast)]
        self.assertEqual(len(wave1), 6)
        by_reviewer = {}
        for payload in wave1:
            by_reviewer.setdefault(payload["reviewer_id"], []).append(
                sorted(entry["case_id"] for entry in payload["cases"]))
        partitions = {dumps(sorted(batches), ensure_ascii=False) for batches in by_reviewer.values()}
        self.assertEqual(len(partitions), 1)
        self.assertTrue(all(len(batch) == 4 for batches in by_reviewer.values() for batch in batches))
        self.assertEqual({record["status"] for record in provenance["cases"]}, {"applied"})
        self.assertEqual(issues, [])
        self.assertEqual(len(updated["characters"]), 1)


if __name__ == "__main__":
    unittest.main()
