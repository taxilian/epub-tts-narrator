"""Offline tests for adaptive, source-region-independent cast discovery.

Proves region independence (seed-only requests), no accidental name merging
across regions, unbounded response-local ID allocation, exact-evidence
fail-closed acceptance, deferred merges, narrator seed preservation on
conflicting observations, and deterministic cache reuse — all without network
or model calls.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

from epub_tts_narrator import casting_discovery
from epub_tts_narrator.casting_discovery import discover_candidates, flatten_atoms
from epub_tts_narrator.pipeline import DEFAULT_POLICY, CachedRequests, make_units, seed_cast


class _Capacity(Exception):
    """Internal preflight signal for the reference runner stub."""


def _run_partitioned(items, build, execute, *, fits=None, progress=None, label="", requests=None):
    """Reference stub of the shared adaptive runner contract.

    Preflight fits check and explicit request-budget failures bisect the source
    atom list; no depth or leaf cap applies. Replaced by the real module once
    epub_tts_narrator.adaptive_requests lands.
    """
    progress = progress or (lambda message: None)
    if not items:
        return []
    results, pending = [], [list(items)]
    while pending:
        subset = pending.pop(0)
        try:
            system, payload = build(subset)
            if fits is not None and not fits(payload):
                raise _Capacity()
            result = execute(system, payload)
        except _Capacity:
            pass
        except ValueError as error:
            if "max_request_chars" not in str(error):
                raise
        else:
            results.append((subset, result))
            continue
        if len(subset) == 1:
            raise ValueError("A single source atom cannot satisfy the request budget")
        middle = len(subset) // 2
        pending[:0] = [subset[:middle], subset[middle:]]
        progress(f"{label}: splitting {len(subset)} source atom(s) to fit the request budget")
    return results


def _install_runner_stub():
    if "epub_tts_narrator.adaptive_requests" in sys.modules:
        return
    try:
        import epub_tts_narrator.adaptive_requests  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("epub_tts_narrator.adaptive_requests")
        module.run_partitioned = _run_partitioned
        sys.modules["epub_tts_narrator.adaptive_requests"] = module


_install_runner_stub()


def make_source(*paragraphs):
    blocks = [{"id": f"b{i}", "chapter_id": "ch1", "text": text,
               "source": {"href": "text/ch1.xhtml", "element_id": f"p{i}", "tag": "p"}}
              for i, text in enumerate(paragraphs, 1)]
    return {"book": {"id": "book1", "title": "Two Regions", "authors": ["A. Writer"], "language": "en"},
            "chapters": [{"id": "ch1", "title": "Chapter One", "source_href": "text/ch1.xhtml",
                          "block_ids": [block["id"] for block in blocks]}],
            "blocks": blocks, "exclusions": [], "warnings": [], "extraction_policy": "Test extraction."}


TWO_SCENES = (
    "Mara held the lantern steady and said, “Keep the flame high tonight, traveler, "
    "for the road beyond the bridge is dark and the river waits for no one at all.”",
    "Mara the ferrywoman counted the coins twice and muttered, “The crossing costs "
    "one silver by day and two by night, and I never refund the drowned.”",
)


class DiscoveryFake:
    """Deterministic discovery fixture answering with response-local c/r/v IDs."""

    identity = {"base_url": "http://offline.invalid/v1", "model": "fixture-discovery"}

    def __init__(self, source=None, *, candidates_per_unit=1, distinct_names=False, character_count=None,
                 narrator_pattern=None, bad_evidence=False, cross_region_id=False, merge=False,
                 foreign_evidence=False, narrator_character=False, local_issue_targets=False):
        self.calls = []
        self.blocks = {block["id"]: block["text"] for block in source["blocks"]} if source else {}
        self.candidates_per_unit = candidates_per_unit
        self.distinct_names = distinct_names
        self.character_count = character_count
        self.narrator_pattern = narrator_pattern
        self.bad_evidence = bad_evidence
        self.cross_region_id = cross_region_id
        self.merge = merge
        self.foreign_evidence = foreign_evidence
        self.narrator_character = narrator_character
        self.local_issue_targets = local_issue_targets

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if payload.get("mode") != "casting_discovery":
            raise AssertionError(f"Unexpected mode {payload.get('mode')!r}")
        return self.discover(payload)

    def discover(self, payload):
        units = payload["target_units"]
        total = self.character_count or self.candidates_per_unit * len(units)
        characters, roles, voices = [], [], []
        for index in range(1, total + 1):
            unit = units[(index - 1) % len(units)]
            character_id = "char_du0000002-u0000002_c9" if self.cross_region_id else f"c{index}"
            voice_id, role_id = f"v{index}", f"r{index}"
            if self.bad_evidence:
                quote = "text that never occurs anywhere"
            elif self.foreign_evidence:
                quote = self.blocks[unit["block_id"]][:12]
            else:
                quote = unit["text"][:12]
            evidence = [{"block_id": unit["block_id"], "quote": quote}]
            characters.append({"id": character_id, "aliases": [],
                               "name": f"Speaker {index}" if self.distinct_names else "Mara",
                               "description": "A candidate anchored in this region's visible text.",
                               "evidence": deepcopy(evidence), "voice_id": voice_id})
            roles.append({"id": role_id, "kind": "dialogue", "character_id": character_id,
                          "voice_id": voice_id, "description": f"Dialogue spoken by {character_id}."})
            voices.append({"id": voice_id, "description": "A brief initial audible voice direction."})
        merges = []
        if self.merge and len(characters) >= 2:
            merges = [{"from_character_id": "c2", "into_character_id": "c1",
                       "evidence": deepcopy(characters[0]["evidence"])}]
        roles = list(roles)
        if self.narrator_pattern is not None:
            roles.append({"id": "narrator_main", "kind": "narration", "character_id": None,
                          "voice_id": "voice_narrator_main",
                          "description": self.narrator_pattern.format(region=payload["region"])})
        if self.narrator_character:
            local = total + 1
            characters.append({"id": f"c{local}", "aliases": [], "name": "Elias",
                               "description": "The first-person narrator of this region.",
                               "evidence": [{"block_id": units[0]["block_id"], "quote": units[0]["text"][:12]}],
                               "voice_id": f"v{local}"})
            roles.append({"id": f"r{local}", "kind": "dialogue", "character_id": f"c{local}",
                          "voice_id": f"v{local}", "description": "Elias's spoken lines."})
            voices.append({"id": f"v{local}", "description": "A quiet, reflective voice."})
            roles.append({"id": "narrator_main", "kind": "narration", "character_id": f"c{local}",
                          "voice_id": "voice_narrator_main",
                          "description": f"First-person narration by Elias in region {payload['region']}."})
        issues = []
        if self.local_issue_targets:
            issues = [{"kind": "narrator", "target": "r1", "reason": "Narration scope is uncertain here."},
                      {"kind": "casting", "target": "v1", "reason": "This voice may duplicate a shared seed voice."}]
        return {"characters": characters, "roles": roles, "voices": voices, "merges": merges,
                "issues": issues, "summary": f"Region {payload['region']} summary."}


def _budget_for(source, *, extra_units=0):
    """A request budget that admits one unit per region but not more."""
    seed = seed_cast()
    atoms = flatten_atoms(make_units(source, 256))
    payload, _token = casting_discovery._region_request(source, seed, DEFAULT_POLICY, [atoms[0]])
    single = len(casting_discovery.DISCOVERY_SYSTEM) + len(json.dumps(payload, ensure_ascii=False))
    if extra_units:
        payload, _token = casting_discovery._region_request(source, seed, DEFAULT_POLICY, atoms[:1 + extra_units])
        multi = len(casting_discovery.DISCOVERY_SYSTEM) + len(json.dumps(payload, ensure_ascii=False))
        return single + max(1, (multi - single) // (extra_units + 1) - 1)
    return single + 64


class DiscoveryCandidatesTests(unittest.TestCase):
    def test_one_repair_receives_all_invalid_character_evidence_locations(self):
        class CorrectReportedFields(DiscoveryFake):
            def discover(self, payload):
                response = super().discover(payload)
                feedback = payload.get("correction", {}).get("validator_error", "")
                for index, character in enumerate(response["characters"]):
                    if f"characters[{index}].evidence[0]" not in feedback:
                        character["evidence"][0]["quote"] = "DO NOT ECHO fabricated evidence"
                return response

        client = CorrectReportedFields(self.source, character_count=2)
        requests = CachedRequests(client, self.cache, repair_attempts=1)
        result = discover_candidates(self.source, requests, make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(len(result["cast"]["characters"]), 2)
        self.assertEqual(len(client.calls), 2)
        feedback = client.calls[1][1]["correction"]["validator_error"]
        self.assertIn("characters[0].evidence[0]", feedback)
        self.assertIn("characters[1].evidence[0]", feedback)
        self.assertNotIn("DO NOT ECHO", feedback)

    def test_malformed_model_ids_are_not_echoed_in_diagnostics(self):
        secret = "DO NOT ECHO private passage or credential"

        class InvalidId(DiscoveryFake):
            def discover(self, payload):
                response = super().discover(payload)
                response["characters"][0]["id"] = secret
                return response

        with self.assertRaises(ValueError) as caught:
            discover_candidates(self.source, self._requests(InvalidId(self.source)),
                                make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("characters[0].id", str(caught.exception))

    def test_new_voice_trait_cannot_borrow_evidence_from_another_region(self):
        class ForeignVoiceFact(DiscoveryFake):
            def discover(self, payload):
                response = super().discover(payload)
                hidden = next(block for block in self.blocks if block not in
                              {unit["block_id"] for unit in payload["target_units"]})
                response["voices"][0]["traits"] = [
                    {"name": "tone", "value": "hoarse", "basis": "text",
                     "evidence": [{"block_id": hidden, "quote": self.blocks[hidden][:12]}]}]
                return response

        client = ForeignVoiceFact(self.source)
        with self.assertRaisesRegex(ValueError, "voices.*visible"):
            discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertFalse(list(self.cache.glob("*.json")))

    def setUp(self):
        self.source = make_source(*TWO_SCENES)
        self.before = deepcopy(self.source)
        self.seed = seed_cast()
        self.atoms = flatten_atoms(make_units(self.source, 256))
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.cache = Path(self._temporary.name)

    def _requests(self, client, *, budget=None):
        return CachedRequests(client, self.cache, repair_attempts=0,
                              max_request_chars=budget if budget is not None else 200000)

    def test_independent_regions_keep_equal_names_separate_without_merging(self):
        client = DiscoveryFake()
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        characters = result["cast"]["characters"]
        self.assertEqual([character["name"] for character in characters], ["Mara", "Mara"])
        self.assertNotEqual(characters[0]["id"], characters[1]["id"])
        for character in characters:
            self.assertTrue(character["id"].startswith("char_d"), character["id"])
            self.assertTrue(character["id"].endswith("_c1"), character["id"])
        self.assertEqual(result["pending_merges"], [])
        region_units = [unit for region in result["regions"] for unit in region]
        self.assertEqual(sorted(unit["unit_id"] for unit in region_units),
                         sorted(unit["unit_id"] for unit in self.atoms))
        self.assertEqual(result["regions"][0][0], self.atoms[0])
        self.assertEqual(len(client.calls), len(result["regions"]))
        for system, payload in client.calls:
            self.assertEqual(payload["known_cast"], self.seed)
        self.assertEqual(self.source, self.before)

    def test_scene_summaries_bind_region_units_and_source_intervals(self):
        client = DiscoveryFake()
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(len(result["scene_summaries"]), len(result["regions"]))
        for index, summary in enumerate(result["scene_summaries"]):
            region = result["regions"][index]
            self.assertEqual(summary["region"], index + 1)
            self.assertEqual(summary["unit_ids"], [unit["unit_id"] for unit in region])
            self.assertEqual(summary["summary"], f"Region {summary['token']} summary.")
            covered = {(interval["block_id"], interval["start"], interval["end"])
                       for interval in summary["intervals"]}
            expected = {(unit["block_id"], unit["start"], unit["start"] + len(unit["text"])) for unit in region}
            self.assertEqual(covered, expected)

    def test_conflicting_narrator_observations_preserve_shared_seed(self):
        client = DiscoveryFake(self.source, narrator_pattern="Narration scope observed in region {region}.")
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        narrator = next(role for role in result["cast"]["roles"] if role["id"] == "narrator_main")
        self.assertEqual(narrator["description"], self.seed["roles"][0]["description"])
        issues = result["provenance"]["issues"]
        conflicts = [issue for issue in issues if issue["kind"] == "shared_record_conflict"]
        self.assertEqual([issue["target"] for issue in conflicts], ["narrator_main"])
        observations = result["provenance"]["shared_record_observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(observations[0]["records"]), 2)
        descriptions = {record["description"] for record in observations[0]["records"]}
        self.assertEqual(len(descriptions), 2)
        self.assertEqual(result["provenance"]["narration_scopes"], [])

    def test_first_person_narrator_conflicts_keep_scoped_role_candidates(self):
        client = DiscoveryFake(self.source, narrator_character=True)
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        narrator = next(role for role in result["cast"]["roles"] if role["id"] == "narrator_main")
        self.assertIsNone(narrator["character_id"])
        self.assertEqual(narrator["description"], self.seed["roles"][0]["description"])
        scoped_roles = [role for role in result["cast"]["roles"] if role["kind"] == "narration"
                        and role["id"] != "narrator_main"]
        self.assertEqual(len(scoped_roles), 2)
        elias = {character["id"]: character for character in result["cast"]["characters"]
                 if character["name"] == "Elias"}
        self.assertEqual(len(elias), 2)
        for role in scoped_roles:
            self.assertIn(role["character_id"], elias)
            self.assertEqual(role["voice_id"], "voice_narrator_main")
            self.assertTrue(role["id"].startswith("role_d"))
            self.assertTrue(role["source_scope"]["unit_ids"])
            self.assertTrue(role["source_scope"]["intervals"])
        scopes = result["provenance"]["narration_scopes"]
        self.assertEqual(len(scopes), 2)
        for scope, region in zip(scopes, result["regions"]):
            self.assertEqual(scope["unit_ids"], [unit["unit_id"] for unit in region])
            self.assertEqual(scope["intervals"],
                             casting_discovery._block_intervals(region))
            self.assertIn(scope["role_id"], {role["id"] for role in scoped_roles})
            self.assertIn(scope["character_id"], elias)
        issues = result["provenance"]["issues"]
        self.assertTrue(any(issue["kind"] == "shared_record_conflict" and issue["target"] == "narrator_main"
                            for issue in issues))

    def test_local_issue_targets_are_namespaced_for_every_collection(self):
        client = DiscoveryFake(self.source, local_issue_targets=True)
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        issues = result["provenance"]["issues"]
        targets = sorted(issue["target"] for issue in issues if issue["kind"] in ("narrator", "casting"))
        self.assertEqual(len(targets), 4)
        self.assertEqual(sum(target.startswith("role_d") for target in targets), 2)
        self.assertEqual(sum(target.startswith("voice_d") for target in targets), 2)

    def test_new_candidate_evidence_must_be_visible_inside_its_own_region(self):
        paragraph = ("The keeper walked the length of the gallery deck every hour, and every hour the fog pressed "
                     "a little closer against the glass, until the lamp room itself seemed to float free of the "
                     "tower and the sea below forgot the shape of the rocks it had guarded for a hundred years.")
        source = make_source("Mara held the lantern steady while the wind picked up around the tower. " + paragraph,
                             "Mara counted the coins twice and muttered about the crossing fees.")
        client = DiscoveryFake(source, foreign_evidence=True)
        with self.assertRaises(ValueError) as raised:
            discover_candidates(source, self._requests(client, budget=_budget_for(source)),
                                make_units(source, 256), policy=DEFAULT_POLICY)
        self.assertIn("visible in the supplied units", str(raised.exception))

    def test_unbounded_response_local_ids_exceed_legacy_allocation_caps(self):
        client = DiscoveryFake(distinct_names=True, character_count=70)
        result = discover_candidates(self.source, self._requests(client),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(len(result["cast"]["characters"]), 70)
        allocated = {character["id"] for character in result["cast"]["characters"]}
        self.assertEqual(len(allocated), 70)
        self.assertTrue(all(character_id.startswith("char_d") for character_id in allocated))

    def test_allocation_guidance_is_machine_consumed_and_never_a_cap(self):
        client = DiscoveryFake()
        discover_candidates(self.source, self._requests(client), make_units(self.source, 256),
                            policy=DEFAULT_POLICY)
        _system, payload = client.calls[0]
        self.assertEqual(payload["available_ids"]["characters"], ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"])
        self.assertEqual(payload["id_allocation"]["policy"], "unbounded_response_local")
        self.assertIn("no cap", payload["id_allocation"]["available_ids_meaning"])

    def test_invalid_new_candidate_evidence_fails_closed_and_writes_failed_response(self):
        client = DiscoveryFake(bad_evidence=True)
        with self.assertRaises(ValueError) as raised:
            discover_candidates(self.source, self._requests(client), make_units(self.source, 256),
                                policy=DEFAULT_POLICY)
        message = str(raised.exception)
        self.assertIn("evidence", message.casefold())
        failed = list((self.cache / "failed").glob("*.json"))
        self.assertEqual(len(failed), 1)

    def test_cross_region_allocated_reference_is_rejected(self):
        client = DiscoveryFake(cross_region_id=True)
        with self.assertRaises(ValueError) as raised:
            discover_candidates(self.source, self._requests(client), make_units(self.source, 256),
                                policy=DEFAULT_POLICY)
        self.assertIn("another source region", str(raised.exception))

    def test_local_merges_stay_deferred_and_ledger_stays_complete(self):
        client = DiscoveryFake(candidates_per_unit=2, merge=True)
        result = discover_candidates(self.source, self._requests(client, budget=_budget_for(self.source)),
                                     make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(len(result["cast"]["characters"]), 4)
        self.assertEqual(len(result["pending_merges"]), 2)
        for merge in result["pending_merges"]:
            self.assertTrue(merge["from_character_id"].startswith("char_d"))
            self.assertTrue(merge["into_character_id"].startswith("char_d"))
            self.assertTrue(merge["evidence"])

    def test_request_too_small_for_even_one_atom_is_rejected(self):
        client = DiscoveryFake()
        with self.assertRaises(ValueError) as raised:
            discover_candidates(self.source, self._requests(client, budget=64),
                                make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertIn("never trimmed", str(raised.exception))

    def test_completed_regions_replay_from_cache_without_new_model_calls(self):
        first = DiscoveryFake()
        discover_candidates(self.source, self._requests(first, budget=_budget_for(self.source)),
                            make_units(self.source, 256), policy=DEFAULT_POLICY)
        second = DiscoveryFake()
        discover_candidates(self.source, self._requests(second, budget=_budget_for(self.source)),
                            make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(second.calls, [])


if __name__ == "__main__":
    unittest.main()
