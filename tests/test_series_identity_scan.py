"""Continuity is an explicit decision for every current character, never omission."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity import audit_identities, _accept_series_scan, _series_scan_requests
from epub_tts_narrator.pipeline import CachedRequests, check_cast, fingerprint, make_units
from epub_tts_narrator.series import reconcile_series_cast, series_context
from test_contracts import valid_document
from test_identity import Requests


def fixture():
    source = deepcopy(valid_document()["source"])
    source["blocks"][0]["text"] = "Captain Mara returned aboard the Dawn with her old navigator Ivo. Her crew still called her Red."
    source["blocks"][1]["text"] = "Bex was a newly hired baker from the distant colony, unrelated to Captain Mara."
    cast = {"characters": [], "roles": [{"id": "narrator_main", "kind": "narration", "character_id": None,
             "voice_id": "vn", "description": "Independent narrator."}],
            "voices": [{"id": "vn", "description": "Measured narrator.", "traits": []}]}
    for index, (name, description) in enumerate((("Mara", "Captain of the Dawn with navigator Ivo."),
                                                ("Bex", "A new baker from the distant colony.")), 1):
        cast["characters"].append({"id": f"c{index}", "name": name, "aliases": ["Red"] if index == 1 else [],
            "description": description, "voice_id": f"v{index}",
            "evidence": [{"block_id": f"b{index}", "quote": source["blocks"][index - 1]["text"]}]})
        cast["roles"].append({"id": f"r{index}", "kind": "dialogue", "character_id": f"c{index}",
                             "voice_id": f"v{index}", "description": "Character dialogue."})
        cast["voices"].append({"id": f"v{index}", "description": "Distinct clear voice.", "traits": []})
    prior_cast = {"characters": [{"id": "prior_mara", "name": "Mara", "aliases": ["Red"],
        "description": "Captain of the Dawn, whose old navigator is Ivo.", "voice_id": "prior_voice_mara", "evidence": []}],
        "roles": [{"id": "prior_role_mara", "kind": "dialogue", "character_id": "prior_mara", "voice_id": "prior_voice_mara", "description": "The captain speaking."}],
        "voices": [{"id": "prior_voice_mara", "description": "Established warm, measured captain voice.", "traits": []}]}
    registry = {"format": "epub-audiobook-series", "version": 1,
        "series_id": "00000000-0000-0000-0000-000000000001", "books": [{"source_fingerprint": "earlier-book",
        "book": {"id": "earlier", "title": "Earlier Journey"}, "cast": prior_cast}]}
    return source, cast, registry


def decision(character, kind=None, candidates=None):
    kind = kind or ("match" if character["id"] == "c1" else "new")
    return {"character_id": character["id"], "decision": kind,
            "prior_character_ids": candidates if candidates is not None else (["prior_mara"] if kind == "match" else []),
            "reason": "The captain returns with her established ship and navigator." if kind == "match" else "The text introduces a new baker.",
            "evidence": deepcopy(character["evidence"])}


class ScanRequests(Requests):
    def __init__(self, decisions=None, modify=None, **kwargs):
        super().__init__(**kwargs)
        self.decisions, self.modify = decisions or {}, modify
        self.max_request_chars = 200000

    def response(self, payload):
        if payload["mode"] == "series_identity_scan":
            response = {"decisions": [deepcopy(self.decisions.get(row["id"], decision(row))) for row in payload["current_characters"]]}
            return self.modify(response, payload) if self.modify else response
        if payload["mode"] == "identity_vote":
            return {"verdict": self.verdicts.get(payload["reviewer_id"], "approve"),
                    "reason": "The captain returns with the established ship, navigator, and nickname.",
                    "evidence": deepcopy(payload["case"]["evidence"])}
        return super().response(payload)


class SeriesIdentityScanTests(unittest.TestCase):
    def audit(self, requests, source=None, cast=None, registry=None, **kwargs):
        default_source, default_cast, default_registry = fixture()
        return audit_identities(source or default_source, cast or default_cast, requests,
            make_units(source or default_source), series_registry=registry or default_registry, **kwargs)

    def test_omitted_discovery_links_receive_explicit_scan_then_three_votes_and_reuse_voice(self):
        source, cast, registry = fixture()
        before, requests = deepcopy(cast), ScanRequests()
        updated, issues, provenance = self.audit(requests, source, cast, registry, whole_book=True, workers=5)
        self.assertEqual(cast, before)
        self.assertEqual(issues, [])
        self.assertEqual(updated["characters"][0]["series_identity"]["character_id"], "prior_mara")
        self.assertNotIn("series_identity", updated["characters"][1])
        scans = [row for row in requests.calls if row["mode"] == "series_identity_scan"]
        self.assertEqual(len(scans), 1)
        self.assertEqual([row["id"] for row in scans[0]["current_characters"]], ["c1", "c2"])
        self.assertEqual(scans[0]["source_scope"], "complete_source")
        self.assertEqual("".join(unit["text"] for unit in scans[0]["source_units"]), "".join(block["text"] for block in source["blocks"]))
        votes = [row for row in requests.calls if row["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertEqual(len({row["reviewer_id"] for row in votes}), 3)
        self.assertEqual(provenance["series_cases"][0]["status"], "confirmed")
        self.assertEqual(provenance["series_scan"]["request_count"], 1)
        self.assertEqual(provenance["series_scan"]["requests"][0]["request_fingerprint"], fingerprint(scans[0]))
        linked, _, _ = reconcile_series_cast(source, updated, registry)
        self.assertEqual(linked["characters"][0]["id"], "prior_mara")
        self.assertEqual(linked["characters"][0]["voice_id"], "prior_voice_mara")
        for character in updated["characters"]:
            character.pop("series_identity", None)
        self.assertEqual(updated, before)

    def test_returning_character_under_established_alias_is_not_split(self):
        source, cast, registry = fixture()
        cast["characters"][0].update(name="Red", aliases=[])
        updated, issues, provenance = self.audit(ScanRequests(), source, cast, registry)
        linked, _, _ = reconcile_series_cast(source, updated, registry)
        self.assertEqual(issues, [])
        self.assertEqual(linked["characters"][0]["voice_id"], "prior_voice_mara")
        self.assertEqual(provenance["series_cases"][0]["status"], "confirmed")

    def test_explicit_new_same_name_clears_stale_match_without_merging(self):
        source, cast, registry = fixture()
        source["blocks"][0]["text"] = "Mara was a new baker who had never served on the Dawn; Captain Mara had died years ago."
        character = cast["characters"][0]
        character.update(aliases=[], description="New baker, distinct from the late captain.",
                         evidence=[{"block_id": "b1", "quote": source["blocks"][0]["text"]}])
        character["series_identity"] = {"character_id": "prior_mara", "basis": "context", "reason": "Stale guess.", "evidence": deepcopy(character["evidence"]), "review_required": False}
        requests = ScanRequests({"c1": decision(character, "new")})
        updated, _, provenance = self.audit(requests, source, cast, registry)
        self.assertNotIn("series_identity", updated["characters"][0])
        self.assertEqual(provenance["series_cases"], [])
        linked, _, _ = reconcile_series_cast(source, updated, registry)
        self.assertNotEqual(linked["characters"][0]["id"], "prior_mara")
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_voice_mara")
        self.assertTrue(all(row["mode"] != "identity_vote" for row in requests.calls))

    def test_uncertain_multiple_prior_candidates_stay_separate_with_explicit_review(self):
        source, cast, registry = fixture()
        prior = registry["books"][0]["cast"]
        prior["characters"].append(dict(deepcopy(prior["characters"][0]), id="prior_other", voice_id="prior_voice_other", description="A different Mara, a baker."))
        prior["voices"].append({"id": "prior_voice_other", "description": "Separate baker voice.", "traits": []})
        requests = ScanRequests({"c1": decision(cast["characters"][0], "uncertain", ["prior_mara", "prior_other"])})
        updated, issues, provenance = self.audit(requests, source, cast, registry)
        self.assertNotIn("series_identity", updated["characters"][0])
        self.assertEqual(provenance["series_cases"], [])
        self.assertEqual(provenance["series_scan"]["decisions"][0]["prior_character_ids"], ["prior_mara", "prior_other"])
        self.assertTrue(any(issue["target"] == "c1" and "prior_other" in issue["reason"] for issue in issues))
        linked, _, _ = reconcile_series_cast(source, updated, registry)
        self.assertNotIn(linked["characters"][0]["id"], {"prior_mara", "prior_other"})

    def test_one_jury_dissent_withholds_returning_voice_despite_scanner_match(self):
        source, cast, registry = fixture()
        updated, issues, provenance = self.audit(ScanRequests(verdicts={"identity_reviewer_2": "reject"}), source, cast, registry)
        self.assertEqual(updated["characters"][0]["series_identity"]["basis"], "uncertain")
        self.assertTrue(updated["characters"][0]["series_identity"]["review_required"])
        self.assertEqual(provenance["series_cases"][0]["status"], "unresolved")
        self.assertTrue(issues)
        linked, _, _ = reconcile_series_cast(source, updated, registry)
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_voice_mara")

    def test_missing_duplicate_extra_reordered_and_unknown_character_decisions_are_invalid(self):
        mutations = [lambda response: response["decisions"].pop(),
            lambda response: response["decisions"].append(deepcopy(response["decisions"][0])),
            lambda response: response["decisions"].reverse(),
            lambda response: response["decisions"][1].update(character_id="c1"),
            lambda response: response["decisions"][0].update(character_id="invented"),
            lambda response: response.update(cast={}),
            lambda response: response["decisions"][0].update(voice_id="new_voice")]
        for mutation in mutations:
            def modify(response, payload):
                mutation(response)
                return response
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.audit(ScanRequests(modify=modify))

    def test_forged_invisible_name_only_evidence_and_invalid_prior_ids_are_rejected(self):
        source, cast, registry = fixture()
        prior = series_context(registry, source)
        units = make_units(source)[0]
        valid = {"decisions": [decision(row) for row in cast["characters"]]}
        mutations = [lambda row: row.update(prior_character_ids=["invented"]),
                     lambda row: row.update(prior_character_ids=["prior_mara", "prior_mara"]),
                     lambda row: row.update(evidence=[{"block_id": "b1", "quote": "Mara"}]),
                     lambda row: row.update(evidence=[{"block_id": "b1", "quote": "Fabrication"}]),
                     lambda row: row.update(evidence=[])]
        for mutate in mutations:
            response = deepcopy(valid)
            mutate(response["decisions"][0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                _accept_series_scan(response, source, cast["characters"], prior, units)
        with self.assertRaisesRegex(ValueError, "visible"):
            _accept_series_scan(valid, source, cast["characters"], prior, units[:1])

    def test_missing_decisions_are_repaired_and_all_responses_replay_from_cache(self):
        class Client(ScanRequests):
            identity = {"model": "test", "base_url": "https://example.invalid"}
            series_attempts = 0
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                response = self.response(payload)
                if payload["mode"] == "series_identity_scan":
                    self.series_attempts += 1
                    if self.series_attempts == 1:
                        response["decisions"].pop()
                return response
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=1)
            first = self.audit(requests)
            count = len(client.calls)
            second = self.audit(requests)
        self.assertEqual(client.series_attempts, 2)
        self.assertEqual(len(client.calls), count)
        self.assertEqual(first, second)
        repair = next(row for row in client.calls if row["mode"] == "series_identity_scan" and "correction" in row)
        self.assertIn("EVERY current character", repair["correction"]["validator_error"])

    def test_bounded_mode_uses_complete_source_when_it_fits(self):
        requests = ScanRequests()
        self.audit(requests, whole_book=False)
        scans = [row for row in requests.calls if row["mode"] == "series_identity_scan"]
        self.assertEqual(len(scans), 1)
        self.assertEqual(scans[0]["source_scope"], "complete_source")

    def test_bounded_mode_discloses_selected_full_blocks_and_covers_every_character(self):
        source, cast, registry = fixture()
        for index in range(3, 35):
            source["blocks"].append({"id": f"b{index}", "chapter_id": "ch1",
                "text": "Mara watched the Dawn. " + "The ship crossed the sea. " * 35,
                "source": {"href": "text/ch1.xhtml", "tag": "p", "element_id": None}})
            source["chapters"][0]["block_ids"].append(f"b{index}")
        check_cast(source, cast)
        requests = ScanRequests()
        requests.max_request_chars = 14000
        payloads = _series_scan_requests(source, cast, series_context(registry, source), fingerprint(registry),
            [unit for batch in make_units(source) for unit in batch], requests, False)
        self.assertEqual(len(payloads), 2)
        self.assertEqual([row["current_characters"][0]["id"] for row in payloads], ["c1", "c2"])
        for payload in payloads:
            self.assertEqual(payload["source_scope"], "selected_blocks")
            self.assertGreater(payload["context_selection"]["omitted_source_blocks"], 0)
            for unit in payload["source_units"]:
                self.assertEqual(unit["start"], 0)
                self.assertEqual(unit["text"], next(block["text"] for block in source["blocks"] if block["id"] == unit["block_id"]))

    def test_first_book_skips_continuity_scan(self):
        source, cast, registry = fixture()
        registry["books"] = []
        requests = ScanRequests()
        updated, issues, provenance = self.audit(requests, source, cast, registry)
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])
        self.assertNotIn("series_scan", provenance)
        self.assertTrue(all(row["mode"] != "series_identity_scan" for row in requests.calls))


if __name__ == "__main__":
    unittest.main()
