from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity_collisions import COLLISION_VERSION, prior_merge_conflict, resolve_collisions
from epub_tts_narrator.pipeline import fingerprint, merge_patch
from test_identity_collisions import BatchRequests, collision_fixture


class CollisionConstraintTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_approved_split_parts_cannot_be_recombined_by_the_name_screen(self):
        source, cast = collision_fixture(2)
        requests = BatchRequests(cache_dir=self.root)
        split = {"kind": "split", "character_ids": ["c1"], "status": "applied",
                 "patch": {"characters": [{"id": "c1"}, {"id": "c2"}]}}

        updated, _, provenance = resolve_collisions(source, cast, requests, source_cases=[split])

        self.assertEqual(updated, cast)
        self.assertEqual(requests.calls, [])
        self.assertEqual(provenance["state"]["forbidden_pairs"], [["c1", "c2"]])
        self.assertEqual(provenance["state"]["frozen_ids"], ["c1", "c2"])

    def test_prior_identity_constraints_are_bound_to_the_original_book(self):
        source, cast = collision_fixture(2)
        first = BatchRequests(verdicts={("c1", "c2"): "reject"}, cache_dir=self.root / "first")
        _, _, prior = resolve_collisions(source, cast, first)
        other = deepcopy(source)
        other["book"]["id"] = "another-book"
        requests = BatchRequests(cache_dir=self.root / "other")

        with self.assertRaisesRegex(ValueError, "source_fingerprint"):
            resolve_collisions(other, cast, requests, prior=prior)

        self.assertEqual(requests.calls, [])

    def test_conflicting_applied_source_merge_cannot_erase_a_previous_negative_constraint(self):
        source, cast = collision_fixture(2)
        first = BatchRequests(verdicts={("c1", "c2"): "reject"}, cache_dir=self.root / "first")
        _, _, prior = resolve_collisions(source, cast, first)
        patch = {"characters": [], "roles": [], "voices": [], "issues": [], "summary": "Conflicting hypothesis.",
                 "merges": [{"from_character_id": "c2", "into_character_id": "c1",
                             "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}]}
        merged = merge_patch(source, cast, patch, {key: [] for key in ("characters", "roles", "voices")})[0]
        requests = BatchRequests(cache_dir=self.root / "conflict")

        with self.assertRaisesRegex(ValueError, "conflict"):
            resolve_collisions(source, merged, requests, original_cast=cast, prior=prior,
                               source_cases=[{"kind": "merge", "character_ids": ["c1", "c2"], "status": "applied"}])

        self.assertEqual(requests.calls, [])

    def test_source_merge_guard_checks_all_original_members_of_a_merged_lineage(self):
        source, _ = collision_fixture(3)
        source_hash = fingerprint(source)
        state = {"version": COLLISION_VERSION, "source_fingerprint": source_hash,
                 "lineage_members": {"c1": ["c1", "c2"]}, "forbidden_pairs": [["c2", "c3"]],
                 "unresolved_pairs": [], "frozen_ids": [], "retired_ids": []}

        self.assertTrue(prior_merge_conflict(state, ["c1", "c3"], source_hash))
        self.assertFalse(prior_merge_conflict(state, ["c1", "new_candidate"], source_hash))
