"""Adversarial identity decisions use source checks plus a blind unanimous jury."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from concurrent.futures import CancelledError
from threading import Event, Lock
import time
import unittest

from epub_tts_narrator.identity import _accept_scan, _accept_split, _split_ids, _votes, audit_identities
from epub_tts_narrator.identity_prompts import SERIES_VOTE, VOTE
from epub_tts_narrator.pipeline import CachedRequests, check_cast, fingerprint, make_units
from epub_tts_narrator.prompts import PROMPT_VERSION
from test_contracts import valid_document


def fixture(names=("Mara", "Red")):
    source = deepcopy(valid_document()["source"])
    source["blocks"][0]["text"] = "Mara, the captain, spoke to Mara, the baker. The captain was called Red."
    source["blocks"][1]["text"] = "Red returned after a year. She was still the captain. The baker stayed home."
    cast = {"characters": [], "roles": [{"id": "narrator_main", "kind": "narration", "character_id": None,
             "voice_id": "vn", "description": "Independent narrator."}],
            "voices": [{"id": "vn", "description": "Measured narrator.", "traits": []}]}
    for i, name in enumerate(names, 1):
        cast["characters"].append({"id": f"c{i}", "name": name, "aliases": [],
            "description": "A source-described character.", "voice_id": f"v{i}",
            "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]})
        cast["roles"].append({"id": f"r{i}", "kind": "dialogue", "character_id": f"c{i}",
                              "voice_id": f"v{i}", "description": "Character dialogue."})
        cast["voices"].append({"id": f"v{i}", "description": "A distinct clear voice.", "traits": []})
    return source, cast


def case(kind="merge", ids=("c1", "c2")):
    return {"kind": kind, "character_ids": list(ids), "reason": "Narrative evidence identifies the people.",
            "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}


def split_patch(payload):
    cast, available = payload["original_cast"], payload["available_ids"]
    first = deepcopy(cast["characters"][0])
    first.update(name="Mara the captain", description="The captain, called Red.", aliases=["Red"],
                 evidence=[{"block_id": "b1", "quote": "The captain was called Red."}])
    second = {"id": available["characters"][0], "name": "Mara the baker", "aliases": [],
              "description": "The baker, distinct from the captain.", "voice_id": available["voices"][0],
              "evidence": [{"block_id": "b1", "quote": "Mara, the baker"}]}
    roles = [deepcopy(role) for role in cast["roles"] if role["character_id"] == first["id"]]
    roles.append({"id": available["roles"][0], "kind": "dialogue", "character_id": second["id"],
                  "voice_id": second["voice_id"], "description": "The baker speaking."})
    return {"characters": [first, second], "roles": roles,
            "voices": [{"id": second["voice_id"], "description": "Distinct warm baker voice.", "traits": []}],
            "merges": [], "issues": [], "summary": "Separate the captain and baker."}


class Requests:
    def __init__(self, cases=(), verdicts=None, split=None):
        self.cases, self.verdicts, self.split = list(cases), verdicts or {}, split or split_patch
        self.calls = []
        self.lock = Lock()
        self.active = self.max_active = 0

    def response(self, payload):
        if payload["mode"] == "identity_scan":
            return {"cases": deepcopy(self.cases)}
        if payload["mode"] == "series_identity_scan":
            return {"decisions": [{"character_id": character["id"], "decision": "match",
                "prior_character_ids": [character["series_identity"]["character_id"]],
                "reason": character["series_identity"]["reason"],
                "evidence": deepcopy(character["series_identity"]["evidence"])}
                for character in payload["current_characters"]]}
        if payload["mode"] == "identity_prepare_split":
            return self.split(payload)
        if payload["mode"] != "identity_vote":
            raise AssertionError(payload["mode"])
        return {"verdict": self.verdicts.get(payload["reviewer_id"], "approve"),
                "reason": "The named captain is explicitly called Red.",
                "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}

    def run(self, system, payload, validate, *, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Fixture request cancelled")
        with self.lock:
            self.calls.append(deepcopy(payload))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if payload["mode"] == "identity_vote":
                time.sleep(.005)
            return validate(self.response(payload))
        finally:
            with self.lock:
                self.active -= 1


class IdentityTests(unittest.TestCase):
    def audit(self, requests, source=None, cast=None, **kwargs):
        default_source, default_cast = fixture()
        source, cast = source or default_source, cast or default_cast
        return audit_identities(source, cast, requests, make_units(source), **kwargs)

    def test_approved_returning_nickname_merge_uses_three_blind_bound_votes(self):
        source, cast = fixture()
        original = deepcopy(cast)
        requests = Requests([case()])
        updated, issues, provenance = self.audit(requests, source, cast, whole_book=True, workers=3)
        self.assertEqual(len(updated["characters"]), 1)
        self.assertEqual(updated["characters"][0]["aliases"], ["Red"])
        self.assertEqual(cast, original)
        self.assertEqual(issues, [])
        votes = [payload for payload in requests.calls if payload["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertEqual(len({payload["reviewer_id"] for payload in votes}), 3)
        self.assertTrue(all(payload["original_cast"] == original for payload in votes))
        self.assertTrue(all("votes" not in payload and "votes" not in payload["case"] for payload in votes))
        self.assertTrue(all(len(payload["source_units"]) == 2 for payload in votes))
        self.assertEqual(provenance["cases"][0]["status"], "applied")
        self.assertEqual(len({vote["request_fingerprint"] for vote in provenance["cases"][0]["votes"]}), 3)

    def test_one_dissent_keeps_both_characters_and_creates_unresolved_issue(self):
        source, cast = fixture()
        updated, issues, provenance = self.audit(Requests([case()], {"identity_reviewer_2": "reject"}), source, cast)
        self.assertEqual(updated, cast)
        self.assertEqual(issues[0]["kind"], "identity_consensus")
        self.assertEqual(provenance["cases"][0]["status"], "unresolved")

    def test_uncertainty_is_not_a_positive_vote(self):
        updated, issues, provenance = self.audit(Requests([case()], {"identity_reviewer_3": "uncertain"}))
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(len(issues), 1)
        self.assertEqual(provenance["cases"][0]["votes"][-1]["verdict"], "uncertain")

    def test_invalid_vote_is_an_error_never_agreement(self):
        updated, issues, provenance = self.audit(Requests([case()], {"identity_reviewer_3": "maybe"}))
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(provenance["cases"][0]["votes"][-1]["verdict"], "error")
        self.assertEqual(len(issues), 1)

    def test_same_or_similar_names_trigger_review_even_when_scanner_misses(self):
        for names in (("Mara", "Mara"), ("Marabella", "Marabelle")):
            with self.subTest(names=names):
                source, cast = fixture(names)
                requests = Requests()
                updated, issues, provenance = self.audit(requests, source, cast)
                self.assertEqual(updated, cast)
                self.assertEqual(issues, [])
                self.assertEqual(provenance["cases"][0]["origin"], "name_collision_screen")
                self.assertEqual(provenance["cases"][0]["status"], "confirmed_separate")
                self.assertEqual(len([p for p in requests.calls if p["mode"] == "identity_vote"]), 3)

    def test_pending_merge_is_seeded_and_survivor_direction_preserved(self):
        merge = {"from_character_id": "c1", "into_character_id": "c2", "evidence": case()["evidence"]}
        updated, issues, provenance = self.audit(Requests(), pending_merges=[merge])
        self.assertEqual(updated["characters"][0]["id"], "c2")
        self.assertEqual(issues, [])
        self.assertEqual(provenance["cases"][0]["patch"]["merges"], [merge])

    def test_overlapping_merges_cannot_create_unreviewed_transitive_identity(self):
        source, cast = fixture(("Mara", "Red", "Captain"))
        requests = Requests([case(ids=("c1", "c2")), case(ids=("c2", "c3"))])
        updated, issues, provenance = self.audit(requests, source, cast)
        self.assertEqual(updated, cast)
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(row["status"] == "overlapping_cases" for row in provenance["cases"]))

    def test_conflicting_keep_separate_blocks_approved_merge(self):
        updated, issues, provenance = self.audit(Requests([case(), case("keep_separate")]))
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(len(issues), 1)
        self.assertEqual(provenance["cases"][0]["status"], "overlapping_cases")

    def test_same_name_conflation_split_preserves_original_and_adds_distinct_voice(self):
        source, cast = fixture(("Mara",))
        cast["characters"][0]["series_identity"] = {"character_id": "old", "basis": "context",
            "reason": "Stale identity now split.", "evidence": case()["evidence"], "review_required": False}
        updated, issues, provenance = self.audit(Requests([case("split", ("c1",))]), source, cast)
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(updated["characters"][0]["id"], "c1")
        self.assertEqual(updated["characters"][0]["voice_id"], "v1")
        self.assertEqual(len({row["voice_id"] for row in updated["characters"]}), 2)
        self.assertTrue(all("series_identity" not in row for row in updated["characters"]))
        self.assertEqual(issues, [])
        check_cast(source, updated)
        self.assertEqual(provenance["cases"][0]["status"], "applied")

    def test_split_cannot_silently_leave_an_owned_narration_role_unchanged(self):
        source, cast = fixture(("Mara",))
        cast["roles"][0]["character_id"] = "c1"

        def omit_narration(payload):
            response = split_patch(payload)
            response["roles"] = [row for row in response["roles"] if row["id"] != "narrator_main"]
            return response

        requests = Requests([case("split", ("c1",))], split=omit_narration)
        updated, issues, provenance = self.audit(requests, source, cast)
        self.assertEqual(updated, cast)
        self.assertIn("redistribute every role", issues[0]["reason"])
        self.assertFalse(any(payload["mode"] == "identity_vote" for payload in requests.calls))
        self.assertEqual(provenance["cases"][0]["status"], "invalid_proposal")

    def test_split_cannot_change_unrelated_narrator_or_existing_voice(self):
        source, cast = fixture(("Mara",))
        for collection, row in (("roles", cast["roles"][0]), ("voices", cast["voices"][1])):
            def unrelated(payload):
                response = split_patch(payload)
                response[collection].append(dict(deepcopy(row), description="An unrelated modification."))
                return response
            updated, issues, _ = self.audit(Requests([case("split", ("c1",))], split=unrelated), source, cast)
            self.assertEqual(updated, cast)
            self.assertIn("unrelated", issues[0]["reason"])

    def test_split_tolerates_exact_repeated_records_without_editing_them(self):
        source, cast = fixture(("Mara",))

        def repeated(payload):
            response = split_patch(payload)
            response["roles"].append(deepcopy(cast["roles"][0]))
            response["voices"].extend(deepcopy(cast["voices"]))
            return response

        updated, issues, provenance = self.audit(Requests([case("split", ("c1",))], split=repeated), source, cast)
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(issues, [])
        self.assertEqual(updated["roles"][0], cast["roles"][0])
        self.assertEqual(updated["voices"][:2], cast["voices"])
        patch = provenance["cases"][0]["patch"]
        self.assertNotIn("narrator_main", {row["id"] for row in patch["roles"]})
        self.assertTrue(all(row["id"] not in {"vn", "v1"} for row in patch["voices"]))

    def test_scan_cannot_cite_source_valid_but_hidden_quote(self):
        source, cast = fixture()
        units = make_units(source)[0][:1]
        response = {"cases": [case()]}
        response["cases"][0]["evidence"] = [{"block_id": "b2", "quote": "She was still the captain."}]
        with self.assertRaisesRegex(ValueError, "visible"):
            _accept_scan(response, source, cast, units)

    def test_missing_reordered_or_rewritten_source_is_rejected_before_calls(self):
        source, cast = fixture()
        all_units = [unit for batch in make_units(source) for unit in batch]
        changed = deepcopy(all_units)
        changed[0]["text"] = "Invented source"
        for units in ([all_units[:1]], [list(reversed(all_units))], [changed], [[*all_units, all_units[0]]]):
            with self.subTest(units=units):
                requests = Requests()
                with self.assertRaisesRegex(ValueError, "source|text"):
                    audit_identities(source, cast, requests, units)
                self.assertEqual(requests.calls, [])

    def test_names_alone_cannot_approve_even_if_all_reviewers_agree(self):
        class NameOnlyRequests(Requests):
            def response(self, payload):
                response = super().response(payload)
                if payload["mode"] == "identity_vote":
                    response["evidence"] = [{"block_id": "b1", "quote": "Mara"}]
                return response

        source, cast = fixture()
        updated, issues, provenance = self.audit(NameOnlyRequests([case()]), source, cast)
        self.assertEqual(updated, cast)
        self.assertEqual(len(issues), 1)
        self.assertTrue(all(vote["verdict"] == "error" for vote in provenance["cases"][0]["votes"]))

    def test_empty_evidence_cannot_approve_even_if_all_reviewers_agree(self):
        class EmptyRequests(Requests):
            def response(self, payload):
                response = super().response(payload)
                if payload["mode"] == "identity_vote":
                    response["evidence"] = []
                return response

        updated, issues, provenance = self.audit(EmptyRequests([case()]))
        self.assertEqual(len(updated["characters"]), 2)
        self.assertEqual(len(issues), 1)
        self.assertTrue(all(vote["verdict"] == "error" for vote in provenance["cases"][0]["votes"]))

    def test_preparation_issues_are_preserved_after_approved_split(self):
        def uncertain_split(payload):
            result = split_patch(payload)
            result["issues"] = [{"kind": "identity_merge", "target": "c1", "reason": "A different unresolved identity concern."}]
            return result

        source, cast = fixture(("Mara",))
        _, issues, _ = self.audit(Requests([case("split", ("c1",))], split=uncertain_split), source, cast)
        self.assertEqual(issues, [{"kind": "identity_merge", "target": "c1", "reason": "A different unresolved identity concern."}])

    def test_scan_deduplicates_same_case_across_batches(self):
        source, cast = fixture()
        source["blocks"][1]["text"] = source["blocks"][0]["text"]
        source["blocks"][1]["chapter_id"] = "ch2"
        source["chapters"][0]["block_ids"] = ["b1"]
        source["chapters"].append(dict(source["chapters"][0], id="ch2", block_ids=["b2"]))

        class ScanRequests(Requests):
            def response(self, payload):
                result = super().response(payload)
                if payload["mode"] == "identity_scan":
                    result["cases"][0]["evidence"][0]["block_id"] = payload["source_units"][0]["block_id"]
                return result

        requests = ScanRequests([case()])
        _, _, provenance = self.audit(requests, source, cast)
        self.assertEqual(len(provenance["scans"]), 2)
        self.assertEqual(len(provenance["cases"]), 1)
        self.assertEqual(provenance["cases"][0]["scan_indexes"], [0, 1])
        self.assertEqual(len([p for p in requests.calls if p["mode"] == "identity_vote"]), 3)

    def test_concurrency_limit_is_shared_across_case_reviews(self):
        source, cast = fixture(("Mara", "Red", "Other", "Else"))
        requests = Requests([case(ids=("c1", "c2")), case(ids=("c3", "c4"))])
        self.audit(requests, source, cast, workers=2)
        self.assertEqual(requests.max_active, 2)

    def test_fatal_provider_failure_does_not_dispatch_remaining_cases(self):
        class FailedRequests(Requests):
            def __init__(self, cases):
                super().__init__(cases)
                self.second_started = Event()
                self.failure_observed = Event()
                self.calls_after_failure = []
                self.provider_error = RuntimeError("Provider rejected the revoked key")

            def run(self, system, payload, validate, *, cancelled=None):
                if payload["mode"] == "identity_vote":
                    self.cancellation_token = cancelled
                    if self.failure_observed.is_set():
                        self.calls_after_failure.append(payload)
                return super().run(system, payload, validate, cancelled=cancelled)

            def response(self, payload):
                if payload["mode"] == "identity_vote":
                    if payload["reviewer_id"] == "identity_reviewer_1":
                        if not self.second_started.wait(5):
                            raise AssertionError("Second in-flight review did not start")
                        self.failure_observed.set()
                        raise self.provider_error
                    self.second_started.set()
                    if not self.cancellation_token.wait(5):
                        raise AssertionError("Worker did not signal the fatal failure")
                return super().response(payload)

        source, cast = fixture(("Mara", "Red", "Other", "Else", "Tama", "Gale", "Seth", "Hugh"))
        requests = FailedRequests([case(ids=(f"c{i}", f"c{i + 1}")) for i in (1, 3, 5, 7)])
        with self.assertRaisesRegex(RuntimeError, "revoked key") as raised:
            self.audit(requests, source, cast, workers=2)
        self.assertIs(raised.exception, requests.provider_error)
        self.assertEqual(len([payload for payload in requests.calls if payload["mode"] == "identity_vote"]), 2)
        self.assertEqual(requests.calls_after_failure, [])

    def test_split_allocation_cannot_overwrite_an_existing_record(self):
        source, cast = fixture()
        candidate = dict(case("split", ("c1",)), id="identity_fixture")
        original = _split_ids(candidate)
        cast["characters"][1]["id"] = original["characters"][0]
        cast["roles"][2]["character_id"] = original["characters"][0]
        cast["roles"][2]["id"] = original["roles"][0]
        cast["voices"][2]["id"] = original["voices"][0]
        cast["characters"][1]["voice_id"] = original["voices"][0]
        cast["roles"][2]["voice_id"] = original["voices"][0]
        available = _split_ids(candidate, cast)
        for collection in ("characters", "roles", "voices"):
            self.assertNotIn(original[collection][0], available[collection])
            self.assertEqual(len(available[collection]), 8)
        self.assertEqual(available, _split_ids(candidate, cast))

    def test_cache_keeps_three_distinct_jury_responses_and_reuses_them(self):
        class Client:
            identity = {"model": "offline", "base_url": "http://invalid.test"}

            def __init__(self):
                self.requests = Requests([case()])
                self.calls = []

            def complete(self, system, payload):
                self.calls.append(payload)
                return self.requests.response(payload)

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = self.audit(requests)
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 4)
            client.calls.clear()
            second = self.audit(requests)
            self.assertEqual(first, second)
            self.assertEqual(client.calls, [])

    def test_serialized_juries_share_source_and_case_prefixes_but_keep_private_reviews(self):
        source, cast = fixture(("Mara", "Red", "Other", "Else"))
        requests = Requests([case(ids=("c1", "c2")), case(ids=("c3", "c4"))])
        self.audit(requests, source, cast, whole_book=True)
        votes = [payload for payload in requests.calls if payload["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 6)
        serialized = [json.dumps(payload, ensure_ascii=False, allow_nan=False) for payload in votes]
        self.assertTrue(all({"source_units", "original_cast"} <= set(payload) for payload in votes))
        self.assertEqual(len({value.partition(', "case":')[0] for value in serialized}), 1)
        for identity in {payload["case"]["id"] for payload in votes}:
            group = [json.dumps(payload, ensure_ascii=False, allow_nan=False) for payload in votes if payload["case"]["id"] == identity]
            self.assertEqual(len({value.partition(', "reviewer_id":')[0] for value in group}), 1)
            self.assertEqual(len(set(group)), 3)
        self.assertTrue(all({"reviewer_id", "lens"} <= set(payload) for payload in votes))

    def test_series_jury_prefix_order_reuses_current_prompt_cache_entries_regardless_of_key_order(self):
        source, cast = fixture()
        units = [unit for batch in make_units(source) for unit in batch]
        candidate = {"case": dict(case(), id="identity_fixture", kind="series_match", prior_character_id="c1"),
                     "patch": {"series_identity": {"character_id": "c1"}}, "proposed_cast": deepcopy(cast),
                     "source_units": units, "prior_series": deepcopy(cast),
                     "series_registry_fingerprint": "existing-series-fingerprint"}
        requests = Requests()
        progress = []
        expected = _votes(requests, source, cast, [candidate], workers=1,
                          progress=progress.append, label="Series identity jury (round 1/2)")
        self.assertEqual(progress, [
            "Series identity jury (round 1/2): 1/3 reviewer responses complete",
            "Series identity jury (round 1/2): 3/3 reviewer responses complete",
        ])
        self.assertEqual(len(requests.calls), 3)
        self.assertTrue(all(list(payload)[:3] == ["source_units", "original_cast", "prior_series"] for payload in requests.calls))
        serialized = [json.dumps(payload, ensure_ascii=False, allow_nan=False) for payload in requests.calls]
        self.assertEqual(len({value.partition(', "reviewer_id":')[0] for value in serialized}), 1)
        legacy_order = ("mode", "identity_version", "reviewer_id", "lens", "case", "source_fingerprint",
                        "original_cast_fingerprint", "proposed_cast_fingerprint", "patch_fingerprint",
                        "original_cast", "proposed_cast", "patch", "source_units", "prior_series", "series_registry_fingerprint")

        class NoCallsClient:
            identity = {"model": "offline", "base_url": "http://invalid.test"}

            def complete(self, system, payload):
                raise AssertionError("An existing validated jury response should have been reused")

        with tempfile.TemporaryDirectory() as directory:
            client = NoCallsClient()
            for payload in requests.calls:
                legacy = {key: payload[key] for key in legacy_order}
                self.assertEqual(legacy, payload)
                self.assertEqual(fingerprint(legacy), fingerprint(payload))
                key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                   "system": SERIES_VOTE, "payload": legacy})
                Path(directory, key + ".json").write_text(json.dumps(requests.response(payload)), encoding="utf-8")
            cached = CachedRequests(client, directory, repair_attempts=0)
            replay_progress = []
            self.assertEqual(_votes(cached, source, cast, [candidate], workers=1,
                                    progress=replay_progress.append), expected)
            self.assertEqual(replay_progress[-1], "Identity jury: 3/3 reviewer responses complete")

    def test_series_match_disagreement_withholds_reuse_and_preserves_original(self):
        source, cast = fixture(("Mara",))
        prior_cast = deepcopy(cast)
        prior_cast["characters"][0]["id"] = "prior_mara"
        prior_cast["roles"][1]["character_id"] = "prior_mara"
        registry = {"format": "epub-audiobook-series", "version": 1,
                    "series_id": "00000000-0000-0000-0000-000000000001", "books": [
            {"source_fingerprint": "prior-book", "book": {"title": "Prior book"}, "cast": prior_cast}]}
        cast["characters"][0]["series_identity"] = {"character_id": "prior_mara", "basis": "context",
            "reason": "The returning captain has the established nickname.", "evidence": case()["evidence"], "review_required": False}
        original = deepcopy(cast)
        updated, issues, provenance = self.audit(Requests(verdicts={"identity_reviewer_2": "uncertain"}), source, cast,
                                                series_registry=registry)
        self.assertEqual(cast, original)
        self.assertEqual(updated["characters"][0]["series_identity"]["basis"], "uncertain")
        self.assertTrue(updated["characters"][0]["series_identity"]["review_required"])
        self.assertEqual(issues[0]["kind"], "identity_consensus")
        self.assertEqual(provenance["series_cases"][0]["status"], "unresolved")
        registry_hash = provenance["series_registry_fingerprint"]
        self.assertTrue(all(vote["series_registry_fingerprint"] == registry_hash for vote in provenance["series_cases"][0]["votes"]))


if __name__ == "__main__":
    unittest.main()
