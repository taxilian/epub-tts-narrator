"""Opt-in continuity keeps global casting stable without borrowing source evidence."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.config import load_config
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import analyze, fingerprint, write_json
from epub_tts_narrator.identity_prompts import LENSES
from epub_tts_narrator.series import (link_analysis, load_registry, reconcile_series_cast,
                                   save_registry, series_context, validate_registry, validate_series_proposals)
from test_contracts import valid_document
from test_pipeline import FakeClient


def fixture(*, sequel=False, narrator_character=False):
    document = valid_document()
    document["cast"]["roles"][0]["id"] = "narrator_main"
    if narrator_character:
        document["cast"]["roles"][0]["character_id"] = "c1"
    for segment in document["segments"]:
        if segment["role_id"] == "narrator":
            segment["role_id"] = "narrator_main"
    if sequel:
        document["source"]["book"].update(id="book2", title="The Return")
        # All first-book voice quotations are absent from this book, despite
        # the deliberately reused local block IDs.
        document["source"]["blocks"][0]["text"] = "Mara Vale, the lighthouse keeper, returned."
        document["cast"]["characters"][0].update(name="Mara Vale", aliases=["Mara"],
            evidence=[{"block_id": "b1", "quote": "Mara Vale"}])
        document["cast"]["voices"][1]["traits"] = [{"name": "timbre", "value": "new proposal", "basis": "casting", "evidence": []}]
        document["segments"] = []
        for block in document["source"]["blocks"]:
            document["segments"].append({"id": block["id"], "block_id": block["id"], "start": 0,
                "end": len(block["text"]), "text": block["text"], "kind": "narration",
                "role_id": "narrator_main", "voice_id": "vn", "delivery": {"direction": "", "evidence": []},
                "attribution": {"basis": "explicit", "reason": "Narrative prose.", "evidence": [],
                                "candidates": [], "review_required": False}})
    return document


def propose(document, character_id):
    document["cast"]["characters"][0]["series_identity"] = {
        "character_id": character_id, "basis": "context", "review_required": False,
        "reason": "The earlier name and lighthouse occupation establish narrative continuity.",
        "evidence": [{"block_id": "b1", "quote": document["source"]["blocks"][0]["text"]}],
    }


def consensus_for(document, registry):
    """Build the provenance shape emitted by the independent identity jury."""
    source, cast = document["source"], document["cast"]
    source_hash, cast_hash, registry_hash = fingerprint(source), fingerprint(cast), fingerprint(registry)
    cases = []
    for character in cast["characters"]:
        proposal = character.get("series_identity")
        if proposal is None:
            continue
        case = {"id": "identity_series_" + fingerprint([character["id"], proposal["character_id"]])[:20],
                "kind": "series_match", "character_ids": [character["id"]],
                "prior_character_id": proposal["character_id"], "reason": proposal["reason"],
                "evidence": deepcopy(proposal["evidence"]), "scan_indexes": [], "status": "confirmed"}
        case["votes"] = [{"reviewer_id": reviewer, "lens": lens, "verdict": "approve",
                          "reason": proposal["reason"], "evidence": deepcopy(proposal["evidence"]),
                          "source_fingerprint": source_hash, "series_registry_fingerprint": registry_hash,
                          "original_cast_fingerprint": cast_hash, "proposed_cast_fingerprint": cast_hash,
                          "patch_fingerprint": fingerprint({"series_identity": proposal}),
                          "request_fingerprint": fingerprint({"case": case["id"], "reviewer": reviewer})}
                         for reviewer, lens in LENSES]
        cases.append(case)
    return {"policy": "unanimous_3", "source_fingerprint": source_hash,
            "series_registry_fingerprint": registry_hash, "final_cast_fingerprint": cast_hash,
            "series_cases": cases}


class SeriesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "series.json"
        self.empty = load_registry(self.path)

    def seed(self, document=None):
        linked = link_analysis(document or fixture(), self.empty)
        save_registry(self.path, self.empty, linked)
        return linked, load_registry(self.path)

    def test_seed_reimport_is_idempotent_and_original_is_unchanged(self):
        original = fixture()
        before = deepcopy(original)
        first, registry = self.seed(original)
        second = link_analysis(original, registry)
        self.assertEqual(first, second)
        save_registry(self.path, registry, second)
        self.assertEqual(load_registry(self.path), registry)
        self.assertEqual(original, before)
        self.assertEqual(validate_document(second), [])
        self.assertTrue(first["cast"]["characters"][0]["id"].startswith("series_char_"))
        self.assertEqual(first, link_analysis(first, registry))

    def test_renamed_alias_context_reuses_id_and_rich_voice_without_stale_evidence(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        old = first["cast"]["characters"][0]
        propose(current, old["id"])
        linked = link_analysis(current, registry)
        character = linked["cast"]["characters"][0]
        self.assertEqual(character["id"], old["id"])
        self.assertEqual(character["voice_id"], old["voice_id"])
        voice = next(v for v in linked["cast"]["voices"] if v["id"] == character["voice_id"])
        self.assertEqual(voice["description"], first["cast"]["voices"][1]["description"])
        self.assertEqual(voice["traits"][0]["value"], "soft")
        self.assertEqual(voice["traits"][0]["basis"], "casting")
        self.assertEqual(voice["traits"][0]["evidence"], [])
        self.assertEqual(voice["traits"][0]["series_origin"]["original_basis"], "text")
        self.assertEqual(character["evidence"], current["cast"]["characters"][0]["evidence"])
        self.assertEqual(validate_document(linked), [])
        context = series_context(registry, current["source"])
        self.assertNotIn('"quote"', json.dumps(context))

    def test_shared_name_without_context_proposal_stays_separate_and_flagged(self):
        first, registry = self.seed()
        linked = link_analysis(fixture(sequel=True), registry)
        self.assertNotEqual(linked["cast"]["characters"][0]["id"], first["cast"]["characters"][0]["id"])
        self.assertEqual(linked["issues"][0]["kind"], "series_identity")
        self.assertTrue(linked["readiness"]["ready"])

    def test_unanimous_source_bound_identity_reuses_voice_without_any_name_overlap(self):
        first, registry = self.seed()
        current, old = fixture(sequel=True), first["cast"]["characters"][0]
        current["cast"]["characters"][0].update(name="The returning visitor", aliases=[])
        propose(current, old["id"])
        # Neither a model flag nor an ordinary import can bypass the name guard.
        current["cast"]["characters"][0]["consensus_confirmed"] = True
        plain, _, _ = reconcile_series_cast(current["source"], current["cast"], registry)
        self.assertNotEqual(plain["characters"][0]["id"], old["id"])
        consensus = consensus_for(current, registry)
        updated, issues, provenance = reconcile_series_cast(current["source"], current["cast"], registry,
                                                           confirmed_identity_matches=consensus)
        self.assertEqual(updated["characters"][0]["id"], old["id"])
        self.assertEqual(updated["characters"][0]["voice_id"], old["voice_id"])
        self.assertEqual(issues, [])
        self.assertEqual(provenance["identity_matches"][0]["basis"],
                         "unanimous_identity_consensus_with_current_source_evidence")
        inherited = next(voice for voice in updated["voices"] if voice["id"] == old["voice_id"])
        self.assertEqual(inherited["description"], first["cast"]["voices"][1]["description"])

    def test_supplied_consensus_requires_three_distinct_bound_approve_votes(self):
        first, registry = self.seed()
        current, old = fixture(sequel=True), first["cast"]["characters"][0]
        propose(current, old["id"])
        # Even an overlapping name cannot bypass a supplied but failing jury.
        consensus = consensus_for(current, registry)
        mutations = [
            lambda votes: votes[1].update(verdict="reject"),
            lambda votes: votes[1].update(verdict="uncertain"),
            lambda votes: votes[1].update(reviewer_id=votes[0]["reviewer_id"]),
            lambda votes: votes[1].update(request_fingerprint=votes[0]["request_fingerprint"]),
            lambda votes: votes[1].update(source_fingerprint="0" * 64),
            lambda votes: votes[1].update(series_registry_fingerprint="0" * 64),
            lambda votes: votes[1].update(original_cast_fingerprint="0" * 64),
            lambda votes: votes[1].update(proposed_cast_fingerprint="0" * 64),
            lambda votes: votes[1].update(patch_fingerprint="0" * 64),
            lambda votes: votes[1].update(evidence=[{"block_id": "b1", "quote": "not in this source"}]),
            lambda votes: votes[1].update(evidence=[{"block_id": "b1", "quote": "Mara Vale"}]),
            lambda votes: votes.pop(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                invalid = deepcopy(consensus)
                mutate(invalid["series_cases"][0]["votes"])
                updated, issues, _ = reconcile_series_cast(current["source"], current["cast"], registry,
                                                           confirmed_identity_matches=invalid)
                self.assertNotEqual(updated["characters"][0]["id"], old["id"])
                self.assertEqual(issues[0]["kind"], "series_identity")

    def test_consensus_rejects_stale_source_registry_cast_and_mismatched_cases(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        consensus = consensus_for(current, registry)
        for field in ("source_fingerprint", "series_registry_fingerprint", "final_cast_fingerprint"):
            with self.subTest(field=field):
                invalid = deepcopy(consensus)
                invalid[field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "current source, registry, and cast"):
                    reconcile_series_cast(current["source"], current["cast"], registry,
                                          confirmed_identity_matches=invalid)
        for changes in ({"prior_character_id": "unknown"}, {"character_ids": ["unknown"]},
                        {"evidence": [{"block_id": "b1", "quote": "Mara"}]}, {"id": "wrong-case"}):
            with self.subTest(changes=changes):
                invalid = deepcopy(consensus)
                invalid["series_cases"][0].update(changes)
                with self.assertRaisesRegex(ValueError, "consensus"):
                    reconcile_series_cast(current["source"], current["cast"], registry,
                                          confirmed_identity_matches=invalid)

    def test_two_unanimous_current_claims_cannot_share_one_prior_identity(self):
        first, registry = self.seed()
        current, old = fixture(sequel=True), first["cast"]["characters"][0]
        propose(current, old["id"])
        extra = deepcopy(current["cast"]["characters"][0])
        extra.update(id="c2", name="A separate visitor", aliases=[], voice_id="v2")
        current["cast"]["characters"].append(extra)
        current["cast"]["voices"].append({"id": "v2", "description": "Another distinct voice.", "traits": []})
        current["cast"]["roles"].append({"id": "other", "kind": "dialogue", "character_id": "c2",
                                         "voice_id": "v2", "description": "Separate visitor."})
        updated, issues, _ = reconcile_series_cast(current["source"], current["cast"], registry,
                                                   confirmed_identity_matches=consensus_for(current, registry))
        self.assertNotIn(old["id"], {character["id"] for character in updated["characters"]})
        self.assertEqual(len({character["voice_id"] for character in updated["characters"]}), 2)
        self.assertEqual(len(issues), 2)

    def test_verified_ambiguous_alias_match_survives_another_rejected_case_downgrade(self):
        original = fixture()
        extra = deepcopy(original["cast"]["characters"][0])
        extra.update(id="c2", name="Other Mara", aliases=["Mara"], voice_id="v2")
        original["cast"]["characters"].append(extra)
        original["cast"]["voices"].append({"id": "v2", "description": "Another distinct voice.", "traits": []})
        original["cast"]["roles"].append({"id": "other", "kind": "dialogue", "character_id": "c2",
                                           "voice_id": "v2", "description": "Another Mara."})
        first, registry = self.seed(original)
        current = fixture(sequel=True)
        old_ids = [character["id"] for character in first["cast"]["characters"]]
        propose(current, old_ids[0])
        second = deepcopy(current["cast"]["characters"][0])
        second.update(id="c2", name="Second visitor", aliases=[], voice_id="v2")
        second["series_identity"]["character_id"] = old_ids[1]
        current["cast"]["characters"].append(second)
        current["cast"]["voices"].append({"id": "v2", "description": "Another distinct voice.", "traits": []})
        current["cast"]["roles"].append(deepcopy(original["cast"]["roles"][2]))
        consensus = consensus_for(current, registry)
        consensus["series_cases"][1]["status"] = "unresolved"
        consensus["series_cases"][1]["votes"][0]["verdict"] = "reject"
        second["series_identity"].update(basis="uncertain", review_required=True)
        consensus["final_cast_fingerprint"] = fingerprint(current["cast"])
        updated, issues, _ = reconcile_series_cast(current["source"], current["cast"], registry,
                                                   confirmed_identity_matches=consensus)
        self.assertEqual(updated["characters"][0]["id"], old_ids[0])
        self.assertNotEqual(updated["characters"][1]["id"], old_ids[1])
        self.assertEqual(len(issues), 1)

    def test_uncertain_series_scan_retains_nameless_candidates_without_voice_reuse(self):
        from epub_tts_narrator.identity import audit_identities
        from epub_tts_narrator.pipeline import make_units
        from test_series_identity_scan import fixture as scan_fixture, ScanRequests, decision
        source, cast, registry = scan_fixture()
        cast["characters"][0].update(name="An unnamed visitor", aliases=[])
        prior = registry["books"][0]["cast"]
        prior["characters"].append(dict(deepcopy(prior["characters"][0]), id="prior_other",
                                         name="Another captain", aliases=[], voice_id="prior_voice_other"))
        prior["voices"].append({"id": "prior_voice_other", "description": "Distinct second captain.", "traits": []})
        original_source = deepcopy(source)
        for candidates in (["prior_mara", "prior_other"], []):
            with self.subTest(candidates=candidates):
                requests = ScanRequests({"c1": decision(cast["characters"][0], "uncertain", candidates)})
                audited, _, consensus = audit_identities(source, cast, requests, make_units(source),
                                                         series_registry=registry, whole_book=True, workers=5)
                plain, _, _ = reconcile_series_cast(source, audited, registry)
                linked, issues, provenance = reconcile_series_cast(source, audited, registry,
                                                                   confirmed_identity_matches=consensus)
                self.assertEqual(linked, plain)
                self.assertEqual(source, original_source)
                current_id = linked["characters"][0]["id"]
                self.assertEqual(provenance["unresolved_identity_candidates"],
                                 [{"character_id": current_id, "candidates": candidates}])
                self.assertEqual(provenance["identity_matches"], [])
                self.assertNotIn(linked["characters"][0]["voice_id"], {"prior_voice_mara", "prior_voice_other"})
                self.assertTrue(any(issue["target"] == current_id and issue["kind"] == "series_identity" for issue in issues))
                self.assertFalse(any(payload["mode"] == "identity_vote" for payload in requests.calls))
                for invalid_ids in (["unknown_prior"], ["prior_mara", "prior_mara"]):
                    invalid = deepcopy(consensus)
                    invalid["series_scan"]["decisions"][0]["prior_character_ids"] = invalid_ids
                    with self.assertRaisesRegex(ValueError, "Uncertain series scan candidates"):
                        reconcile_series_cast(source, audited, registry, confirmed_identity_matches=invalid)
                invalid = deepcopy(consensus)
                invalid["series_scan"]["decisions"][0]["character_id"] = "unknown_current"
                with self.assertRaisesRegex(ValueError, "Uncertain series scan candidates"):
                    reconcile_series_cast(source, audited, registry, confirmed_identity_matches=invalid)

    def test_ambiguous_shared_alias_does_not_allow_model_to_pick_arbitrarily(self):
        original = fixture()
        extra = deepcopy(original["cast"]["characters"][0])
        extra.update(id="c2", name="Other Mara", aliases=["Mara"], voice_id="v2")
        original["cast"]["characters"].append(extra)
        voice = deepcopy(original["cast"]["voices"][1])
        voice["id"] = "v2"
        original["cast"]["voices"].append(voice)
        original["cast"]["roles"].append({"id": "other", "kind": "dialogue", "character_id": "c2", "voice_id": "v2", "description": "Another speaker."})
        first, registry = self.seed(original)
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        self.assertNotIn(linked["cast"]["characters"][0]["id"], [c["id"] for c in first["cast"]["characters"]])
        self.assertEqual(len(linked["provenance"]["series"]["unresolved_identity_candidates"][0]["candidates"]), 2)

    def test_identified_narrator_keeps_separate_prior_narration_voice(self):
        first, registry = self.seed(fixture(narrator_character=True))
        current = fixture(sequel=True, narrator_character=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        narration, dialogue = linked["cast"]["roles"]
        self.assertEqual(narration["voice_id"], first["cast"]["roles"][0]["voice_id"])
        self.assertNotEqual(narration["voice_id"], dialogue["voice_id"])
        self.assertEqual(narration["character_id"], dialogue["character_id"])
        self.assertEqual(len(linked["cast"]["voices"]), 2)

    def test_default_narrator_reuses_casting_but_other_pov_and_afterword_do_not(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        extra = deepcopy(current["cast"]["voices"][0])
        extra["id"] = "va"
        current["cast"]["voices"].append(extra)
        current["cast"]["roles"].append({"id": "author_afterword", "kind": "narration", "character_id": None,
                                           "voice_id": "va", "description": "Author afterword."})
        linked = link_analysis(current, registry)
        old_voice = first["cast"]["roles"][0]["voice_id"]
        self.assertEqual(linked["cast"]["roles"][0]["voice_id"], old_voice)
        self.assertNotEqual(linked["cast"]["roles"][2]["voice_id"], old_voice)
        current = fixture(sequel=True, narrator_character=True)
        current["cast"]["characters"][0].update(name="Lena", aliases=[])
        linked = link_analysis(current, registry)
        self.assertNotEqual(linked["cast"]["roles"][0]["voice_id"], old_voice)

    def test_old_book_quotation_cannot_support_current_identity(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        current["cast"]["characters"][0]["series_identity"]["evidence"][0]["quote"] = "softly"
        with self.assertRaisesRegex(ValueError, "does not occur"):
            validate_series_proposals(current["source"], current["cast"], registry)

    def test_supported_identity_reuses_ids_but_prior_identity_review_remains_unresolved(self):
        original = fixture()
        original["issues"] = [{"id": "unclear", "kind": "identity", "target": "c1", "reason": "Two possible Maras.", "resolved": False}]
        first, registry = self.seed(original)
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        self.assertEqual(linked["cast"]["characters"][0]["id"], first["cast"]["characters"][0]["id"])
        self.assertEqual(linked["cast"]["characters"][0]["voice_id"], first["cast"]["characters"][0]["voice_id"])
        self.assertTrue(series_context(registry, current["source"])["prior_books"][0]["unresolved_issues"])
        issue = linked["issues"][0]
        self.assertFalse(issue["resolved"])
        self.assertEqual(issue["series_origin"]["issue_id"], "unclear")
        self.assertEqual(issue["series_origin"]["source_fingerprint"], first["provenance"]["series"]["source_fingerprint"])
        self.assertTrue(linked["provenance"]["series"]["identity_matches"][0]["prior_review_required"])
        self.assertTrue(linked["readiness"]["ready"])

    def test_casting_details_and_informational_separation_do_not_create_new_identity(self):
        original = fixture()
        original["issues"] = [
            {"id": "distinct", "kind": "identity", "target": "c1", "reason": "Mara and Mary are distinct people; keep them separate.", "resolved": False},
            {"id": "vocal", "kind": "casting", "target": "v1", "reason": "The species and vocal capability are unspecified.", "resolved": False},
        ]
        first, registry = self.seed(original)
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        self.assertEqual(linked["cast"]["characters"][0]["id"], first["cast"]["characters"][0]["id"])
        self.assertEqual(linked["cast"]["characters"][0]["voice_id"], first["cast"]["characters"][0]["voice_id"])
        self.assertEqual({issue["series_origin"]["issue_id"] for issue in linked["issues"]}, {"distinct", "vocal"})
        self.assertTrue(linked["readiness"]["ready"])

    def test_narration_interpretation_review_keeps_unique_existing_voice(self):
        original = fixture()
        original["issues"] = [{"id": "thought-policy", "kind": "narrator", "target": "narrator_main",
                                "reason": "Review the boundary between direct thoughts and narration.", "resolved": False}]
        first, registry = self.seed(original)
        linked = link_analysis(fixture(sequel=True), registry)
        self.assertEqual(linked["cast"]["roles"][0]["voice_id"], first["cast"]["roles"][0]["voice_id"])
        carried = [issue for issue in linked["issues"] if issue.get("series_origin", {}).get("issue_id") == "thought-policy"]
        self.assertEqual(len(carried), 1)
        self.assertEqual(carried[0]["target"], "narrator_main")
        self.assertFalse(carried[0]["resolved"])
        self.assertTrue(linked["readiness"]["ready"])

    def test_afterword_role_review_does_not_leak_to_same_characters_main_narration(self):
        original = fixture(narrator_character=True)
        original["cast"]["roles"].append({"id": "author_afterword", "kind": "narration", "character_id": "c1",
                                           "voice_id": "vn", "description": "The author's afterword."})
        original["issues"] = [{"id": "afterword", "kind": "narrator", "target": "author_afterword",
                                "reason": "Review the interpretation of the afterword.", "resolved": False}]
        first, registry = self.seed(original)
        current = fixture(sequel=True, narrator_character=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        self.assertEqual(linked["cast"]["roles"][0]["voice_id"], first["cast"]["roles"][0]["voice_id"])
        self.assertEqual(linked["issues"], [])
        # An explicitly retained matching afterword scope does inherit it.
        current["cast"]["roles"].append(deepcopy(original["cast"]["roles"][2]))
        with_afterword = link_analysis(current, registry)
        self.assertEqual(len(with_afterword["issues"]), 1)
        self.assertEqual(with_afterword["issues"][0]["target"], "author_afterword")

    def test_current_uncertain_proposal_stays_separate_despite_exact_quoted_context(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        current["cast"]["characters"][0]["series_identity"].update(basis="uncertain", review_required=True)
        linked = link_analysis(current, registry)
        self.assertNotEqual(linked["cast"]["characters"][0]["id"], first["cast"]["characters"][0]["id"])
        self.assertEqual(linked["issues"][0]["kind"], "series_identity")

    def test_inherited_review_is_deduplicated_across_book_snapshots(self):
        original = fixture()
        original["issues"] = [{"id": "vocal", "kind": "casting", "target": "c1", "reason": "Vocal capability is unspecified.", "resolved": False}]
        first, registry = self.seed(original)
        second = fixture(sequel=True)
        propose(second, first["cast"]["characters"][0]["id"])
        linked = link_analysis(second, registry)
        save_registry(self.path, registry, linked)
        third = fixture(sequel=True)
        third["source"]["book"].update(id="book3", title="A Third Visit")
        propose(third, first["cast"]["characters"][0]["id"])
        final = link_analysis(third, load_registry(self.path))
        self.assertEqual(len(final["issues"]), 1)
        self.assertEqual(final["issues"][0], linked["issues"][0])
        self.assertTrue(final["readiness"]["ready"])
        # The origin's explanation can be revised without creating a second
        # descendant of the same source-qualified issue.
        updated = load_registry(self.path)
        updated["books"][0]["issues"][0]["reason"] = "Vocal capability still needs a production decision."
        revised = link_analysis(third, updated)
        self.assertEqual(len(revised["issues"]), 1)
        self.assertEqual(revised["issues"][0]["series_origin"]["reason"], updated["books"][0]["issues"][0]["reason"])

    def test_saved_source_qualified_resolution_is_not_resurrected_by_earlier_snapshot(self):
        original = fixture()
        original["issues"] = [{"id": "vocal", "kind": "casting", "target": "c1", "reason": "Choose a production voice.", "resolved": False}]
        first, registry = self.seed(original)
        second = fixture(sequel=True)
        propose(second, first["cast"]["characters"][0]["id"])
        linked = link_analysis(second, registry)
        # Explicit edit of a completed review, then a registry save; matching
        # itself never performs this resolution.
        linked["issues"][0]["resolved"] = True
        save_registry(self.path, registry, linked)
        updated = load_registry(self.path)
        third = fixture(sequel=True)
        third["source"]["book"]["id"] = "book3"
        propose(third, first["cast"]["characters"][0]["id"])
        final = link_analysis(third, updated)
        self.assertEqual(final["issues"], [])
        self.assertTrue(final["readiness"]["ready"])
        self.assertFalse(any(book["unresolved_issues"] for book in series_context(updated, third["source"])["prior_books"]))

    def test_malformed_inherited_review_origin_is_a_validation_error(self):
        _, registry = self.seed()
        for origin in (None, "invalid", {"kind": "inherited_series_review"}):
            registry["books"][0]["issues"] = [{"id": "bad", "kind": "casting", "target": "c1", "reason": "Review.",
                                               "resolved": False, "series_origin": origin}]
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "review origin"):
                validate_registry(registry)

    def test_explicit_separate_registry_means_separate_ids(self):
        first, registry = self.seed()
        unrelated = load_registry(self.root / "unrelated.json")
        other = link_analysis(fixture(), unrelated)
        self.assertNotEqual(first["cast"]["characters"][0]["id"], other["cast"]["characters"][0]["id"])
        with self.assertRaisesRegex(ValueError, "different series"):
            link_analysis(first, unrelated)

    def test_concurrent_registry_change_does_not_overwrite_other_update(self):
        first, registry = self.seed()
        changed = deepcopy(registry)
        changed["books"][0]["book"]["title"] = "Concurrent metadata change"
        write_json(self.path, changed, overwrite=True)
        with self.assertRaisesRegex(ValueError, "changed during"):
            save_registry(self.path, registry, first)
        self.assertEqual(load_registry(self.path), changed)
        self.assertFalse(self.path.with_name("series.json.lock").exists())

    def test_reimport_keeps_generated_identity_review_flags(self):
        _, registry = self.seed()
        second_raw = fixture(sequel=True)
        linked = link_analysis(second_raw, registry)
        save_registry(self.path, registry, linked)
        updated = load_registry(self.path)
        repeated = link_analysis(second_raw, updated)
        self.assertEqual(repeated, linked)
        self.assertTrue(repeated["readiness"]["ready"])
        save_registry(self.path, updated, repeated)
        self.assertEqual(updated, load_registry(self.path))

    def test_current_source_voice_corrections_keep_their_evidence_and_stable_id(self):
        first, registry = self.seed()
        corrected = fixture()
        corrected["cast"]["voices"][1]["description"] = "A corrected, quieter production voice."
        linked = link_analysis(corrected, registry)
        voice_id = first["cast"]["characters"][0]["voice_id"]
        voice = next(item for item in linked["cast"]["voices"] if item["id"] == voice_id)
        self.assertEqual(voice["description"], corrected["cast"]["voices"][1]["description"])
        self.assertEqual(voice["traits"], corrected["cast"]["voices"][1]["traits"])
        self.assertEqual(linked["cast"]["characters"][0]["id"], first["cast"]["characters"][0]["id"])
        self.assertEqual(validate_document(linked), [])

    def test_reimport_preserves_new_review_issues_in_the_current_analysis(self):
        _, registry = self.seed()
        current = fixture()
        current["issues"] = [{"id": "new-review", "kind": "identity", "target": "c1", "reason": "A new source review finding.", "resolved": False}]
        linked = link_analysis(current, registry)
        self.assertEqual(linked["issues"][0]["id"], "new-review")
        self.assertEqual(linked["issues"][0]["target"], linked["cast"]["characters"][0]["id"])
        self.assertTrue(linked["readiness"]["ready"])

    def test_exact_and_linked_reimports_refresh_new_prior_voice_review_without_recasting(self):
        first, registry = self.seed()
        current = fixture(sequel=True)
        propose(current, first["cast"]["characters"][0]["id"])
        linked = link_analysis(current, registry)
        save_registry(self.path, registry, linked)
        updated = load_registry(self.path)
        updated["books"][0]["issues"].append({"id": "new-prior-review", "kind": "casting",
            "target": first["cast"]["characters"][0]["voice_id"], "reason": "A newly noticed prior voice concern.", "resolved": False})
        for input_document in (current, linked):
            with self.subTest(already_linked="series" in input_document["provenance"]):
                refreshed = link_analysis(input_document, updated)
                self.assertEqual(refreshed["cast"], linked["cast"])
                self.assertEqual(refreshed["segments"], linked["segments"])
                self.assertEqual(len(refreshed["issues"]), 1)
                self.assertEqual(refreshed["issues"][0]["series_origin"]["issue_id"], "new-prior-review")
                self.assertTrue(refreshed["readiness"]["ready"])
                self.assertTrue(refreshed["provenance"]["series"]["identity_matches"][0]["prior_review_required"])
                self.assertIn("review_context_fingerprint", refreshed["provenance"]["series"])
                self.assertEqual(refreshed, link_analysis(refreshed, updated))

    def test_default_narrator_respects_explicit_shared_character_voice(self):
        _, registry = self.seed()
        current = fixture(sequel=True)
        current["cast"]["roles"][0]["voice_id"] = "v1"
        for segment in current["segments"]:
            segment["voice_id"] = "v1"
        linked = link_analysis(current, registry)
        self.assertEqual(linked["cast"]["roles"][0]["voice_id"], linked["cast"]["characters"][0]["voice_id"])
        self.assertTrue(any(issue["kind"] == "series_narrator" for issue in linked["issues"]))

    def test_narrator_voice_issue_target_survives_profile_reuse(self):
        _, registry = self.seed()
        current = fixture(sequel=True)
        current["issues"] = [{"id": "voice-review", "kind": "casting", "target": "vn", "reason": "Review the narrator casting.", "resolved": False}]
        linked = link_analysis(current, registry)
        voice_ids = {voice["id"] for voice in linked["cast"]["voices"]}
        self.assertIn(linked["issues"][0]["target"], voice_ids)
        self.assertEqual(linked["issues"][0]["target"], linked["cast"]["roles"][0]["voice_id"])

    def test_pipeline_opt_in_passes_context_and_reuses_validated_cache(self):
        first, registry = self.seed()
        prior_id = first["cast"]["characters"][0]["id"]

        class ContinuityClient(FakeClient):
            def discover(self, payload):
                response = super().discover(payload)
                for character in response["characters"]:
                    character["series_identity"] = {"character_id": prior_id, "basis": "context", "review_required": False,
                        "reason": "The same named speaker and scene establish continuity.",
                        "evidence": [{"block_id": "b1", "quote": "Mara said softly"}]}
                return response

        client = ContinuityClient()
        source = fixture()["source"]
        output = analyze(source, client, self.root / "cache", series_registry=registry, identity_consensus=False)
        self.assertEqual(output["cast"]["characters"][0]["id"], prior_id)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all("series_cast" in payload for _, payload in client.calls if "mode" in payload))
        cached = analyze(source, client, self.root / "cache", series_registry=registry, identity_consensus=False)
        self.assertEqual(output, cached)
        self.assertEqual(len(client.calls), 3)
        default_client = FakeClient()
        default = analyze(source, default_client, self.root / "plain-cache", identity_consensus=False)
        self.assertNotIn("series", default["provenance"])
        self.assertFalse(any("series_cast" in payload for _, payload in default_client.calls))

    def test_cli_import_without_provider_and_relative_file_config(self):
        original = self.root / "analysis.json"
        output = self.root / "linked.json"
        write_json(original, fixture())
        original_bytes = original.read_bytes()
        with patch("epub_tts_narrator.cli.OpenAICompatibleClient") as provider, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["series-import", str(original), "--series", str(self.path), "-o", str(output)]), 0)
            self.assertEqual(main(["series-import", str(original), "--series", str(self.path)]), 0)
        provider.assert_not_called()
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(len(load_registry(self.path)["books"]), 1)
        self.assertEqual(validate_document(json.loads(output.read_text())), [])
        config = self.root / "settings.toml"
        config.write_text('[analysis]\nseries_registry="series.json"\n')
        self.assertEqual(load_config(config)["analysis"]["series_registry"], str(self.path.resolve()))
        for value in ("true", "[]", '""'):
            config.write_text('[analysis]\nseries_registry=' + value)
            with self.assertRaisesRegex(ValueError, "series_registry"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
