"""Offline tests for finite, source-anchored cast coverage verification.

Proves exact source anchoring, deterministic anchor-derived finding IDs,
candidate additions and bindings, dirty-atom-only re-verification, termination
on repeated findings with explicit unresolved issues, fail-closed certification
(missing/empty findings never certify), canonical record protection, and an
untouched source — all without network or model calls.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator import casting_coverage
from epub_tts_narrator.casting_coverage import finding_id, verify_coverage
from epub_tts_narrator.pipeline import DEFAULT_POLICY, CachedRequests, check_cast, fingerprint, make_units, seed_cast


KEEPER_QUOTE = "“Open the gate,” said the keeper"


def make_source():
    blocks = [
        ("b1", "Mara unpacked the tent while the fire crackled and the night air grew colder around the small camp."),
        ("b2", f"{KEEPER_QUOTE}, “before the bridge closes for the night, stranger.”"),
        ("b3", "The road bent toward the mountains and disappeared behind a wall of pines."),
    ]
    return _source_from_blocks(blocks)


STRANGER_QUOTE = "“before the bridge closes for the night, stranger.”"


def make_long_source():
    paragraph = ("The keeper walked the length of the gallery deck every hour, and every hour the fog pressed a "
                 "little closer against the glass, until the lamp room itself seemed to float free of the tower "
                 "and the sea below forgot the shape of the rocks it had guarded for a hundred years.")
    return _source_from_blocks([("b1", "Mara watched the tower from the camp and waited for morning light."),
                                ("b2", f"{KEEPER_QUOTE} {paragraph}"),
                                ("b3", "The road bent toward the mountains and disappeared behind a wall of pines.")])


def _source_from_blocks(blocks):
    return {"book": {"id": "book1", "title": "The Gate", "authors": ["A. Writer"], "language": "en"},
            "chapters": [{"id": "ch1", "title": "Chapter One", "source_href": "text/ch1.xhtml",
                          "block_ids": [block_id for block_id, _text in blocks]}],
            "blocks": [{"id": block_id, "chapter_id": "ch1", "text": text,
                        "source": {"href": "text/ch1.xhtml", "element_id": f"p{i}", "tag": "p"}}
                       for i, (block_id, text) in enumerate(blocks, 1)],
            "exclusions": [], "warnings": [], "extraction_policy": "Test extraction."}


def mara_cast(source, *, size=1):
    cast = seed_cast()
    cast["characters"].append({"id": "char_mara", "name": "Mara", "aliases": [],
                               "description": "A traveler making camp for the night.",
                               "evidence": [{"block_id": "b1", "quote": "Mara"}], "voice_id": "voice_mara"})
    cast["roles"].append({"id": "role_mara", "kind": "dialogue", "character_id": "char_mara",
                          "voice_id": "voice_mara", "description": "Mara's dialogue."})
    cast["voices"].append({"id": "voice_mara", "description": "A light, quick, youthful voice.", "traits": []})
    for index in range(1, size):
        suffix = f"{index:03d}"
        evidence = {"block_id": "b3", "quote": "The road bent"}
        cast["characters"].append({"id": f"char_extra{suffix}", "name": f"Extra {index}", "aliases": [],
                                   "description": "A background traveler near the road.",
                                   "evidence": [dict(evidence)], "voice_id": f"voice_extra{suffix}"})
        cast["roles"].append({"id": f"role_extra{suffix}", "kind": "dialogue", "character_id": f"char_extra{suffix}",
                              "voice_id": f"voice_extra{suffix}", "description": f"Extra {index} speaking."})
        cast["voices"].append({"id": f"voice_extra{suffix}", "description": "A plain, passing voice.", "traits": []})
    check_cast(source, cast)
    return cast


PEOPLE_SPANS = ("Open the gate", "said the keeper", "before the bridge", "closes for the night")


class CoverageFake:
    """Deterministic coverage fixture anchored to exact source spans.

    Behaves like a compliant model: it reads cast_view, binds findings to
    supplied canonical characters when one already accounts for the passage,
    and only defines response-local newcomers otherwise.
    """

    identity = {"base_url": "http://offline.invalid/v1", "model": "fixture-coverage"}

    def __init__(self, source, *, findings=True, certified=True, bind=None, repeat=False,
                 legacy=False, rename=False, voice_profile=False, outside=False,
                 second_speaker=False, alias_proposal=False, span_units=False, sequential_people=0):
        self.calls = []
        self.blocks = {block["id"]: block["text"] for block in source["blocks"]}
        self.findings = findings
        self.certified = certified
        self.bind = bind
        self.repeat = repeat
        self.legacy = legacy
        self.rename = rename
        self.voice_profile = voice_profile
        self.outside = outside
        self.second_speaker = second_speaker
        self.alias_proposal = alias_proposal
        self.span_units = span_units
        self.sequential_people = sequential_people
        self.people_reported = 0
        self.reported = False

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if payload.get("mode") != "casting_coverage":
            raise AssertionError(f"Unexpected mode {payload.get('mode')!r}")
        return self.respond(payload)

    def finding(self, block_id, character_id):
        text = self.blocks[block_id]
        start = text.index(KEEPER_QUOTE)
        return {"block_id": block_id, "start": start, "end": start + len(KEEPER_QUOTE),
                "quote": text[start:start + len(KEEPER_QUOTE)], "character_id": character_id,
                "reason": "This passage has a distinct solo speaker without an accountable candidate."}

    def stranger_finding(self, block_id, character_id):
        text = self.blocks[block_id]
        start = text.index(STRANGER_QUOTE)
        return {"block_id": block_id, "start": start, "end": start + len(STRANGER_QUOTE),
                "quote": text[start:start + len(STRANGER_QUOTE)], "character_id": character_id,
                "reason": "A second distinct speaker in this passage lacks a candidate."}

    def keeper_records(self, characters, roles, voices):
        characters.append({"id": "c1", "name": "The gate keeper", "aliases": [],
                           "description": "The keeper of the bridge gate.",
                           "evidence": [{"block_id": "b2", "quote": "said the keeper"}],
                           "voice_id": "v1"})
        roles.append({"id": "r1", "kind": "dialogue", "character_id": "c1", "voice_id": "v1",
                      "description": "The keeper's dialogue at the gate."})
        voices.append({"id": "v1", "description": "A dry, commanding voice."})

    def stranger_records(self, characters, roles, voices):
        characters.append({"id": "c2", "name": "The night stranger", "aliases": [],
                           "description": "A stranger asking for the gate at night.",
                           "evidence": [{"block_id": "b2", "quote": "for the night"}],
                           "voice_id": "v2"})
        roles.append({"id": "r2", "kind": "dialogue", "character_id": "c2", "voice_id": "v2",
                      "description": "The stranger's dialogue."})
        voices.append({"id": "v2", "description": "A hoarse, uncertain voice."})

    def respond(self, payload):
        records = payload["cast_view"]["records"]["characters"]
        by_name = {row["name"]: row["id"] for row in records if isinstance(row, dict)}
        keeper_id = by_name.get("The gate keeper")
        stranger_id = by_name.get("The night stranger")
        characters, roles, voices, findings = [], [], [], []
        needs_keeper, needs_stranger = False, False
        units = payload["target_units"]
        if self.span_units and len(units) >= 2:
            grouped = {}
            for unit in units:
                grouped.setdefault(unit["block_id"], []).append(unit)
            block_id, group = next((key, value) for key, value in grouped.items() if len(value) >= 2)
            start = group[0]["start"]
            end = group[1]["start"] + 20
            text = self.blocks[block_id]
            findings.append({"block_id": block_id, "start": start, "end": end, "quote": text[start:end],
                             "character_id": "c1", "reason": "An anchor spanning two supplied units."})
            needs_keeper = True
        for unit in units:
            if self.outside:
                text = self.blocks["b3"]
                start = text.index("The road")
                findings.append({"block_id": "b3", "start": start, "end": start + 8,
                                 "quote": text[start:start + 8], "character_id": "c1"})
                if not any(row["id"] == "c1" for row in characters):
                    first = units[0]
                    characters.append({"id": "c1", "name": "The gate keeper", "aliases": [],
                                       "description": "The keeper of the bridge gate.",
                                       "evidence": [{"block_id": first["block_id"], "quote": first["text"][:8]}],
                                       "voice_id": "v1"})
                    roles.append({"id": "r1", "kind": "dialogue", "character_id": "c1", "voice_id": "v1",
                                  "description": "The keeper's dialogue at the gate."})
                    voices.append({"id": "v1", "description": "A dry, commanding voice."})
                continue
            if self.sequential_people:
                fresh = self.people_reported < self.sequential_people
            else:
                fresh = self.repeat or not self.reported
            if unit["block_id"] != "b2" or not self.findings or not fresh:
                continue
            self.reported = True
            if self.sequential_people:
                person = self.people_reported + 1
                self.people_reported = person
                snippet = PEOPLE_SPANS[person - 1]
                text = self.blocks["b2"]
                start = text.index(snippet)
                findings.append({"block_id": "b2", "start": start, "end": start + len(snippet),
                                 "quote": text[start:start + len(snippet)], "character_id": "c1",
                                 "reason": "A distinct solo speaker in this atom lacks a candidate."})
                characters.append({"id": "c1", "name": f"Person {person}", "aliases": [],
                                   "description": f"A distinct speaker {person} in the same atom.",
                                   "evidence": [{"block_id": "b2", "quote": snippet}],
                                   "voice_id": "v1"})
                roles.append({"id": "r1", "kind": "dialogue", "character_id": "c1", "voice_id": "v1",
                              "description": f"Person {person} speaking."})
                voices.append({"id": "v1", "description": f"A plain passing voice {person}."})
                continue
            if self.alias_proposal:
                findings.append(self.finding("b2", "char_mara"))
                characters.append({"id": "char_mara", "name": "Mara", "aliases": ["Mara of the camp"],
                                   "description": "A traveler making camp for the night.",
                                   "evidence": [{"block_id": "b1", "quote": "Mara"}], "voice_id": "voice_mara"})
                continue
            if self.repeat:
                findings.append(self.finding("b2", "c1"))
                needs_keeper = True
            elif self.bind is not None:
                findings.append(self.finding("b2", self.bind))
            elif keeper_id is None:
                findings.append(self.finding("b2", "c1"))
                needs_keeper = True
            if self.second_speaker and stranger_id is None and not self.repeat:
                findings.append(self.stranger_finding("b2", "c2"))
                needs_stranger = True
        if needs_keeper:
            self.keeper_records(characters, roles, voices)
        if needs_stranger:
            self.stranger_records(characters, roles, voices)
        if self.rename:
            characters.append({"id": "char_mara", "name": "Mara Renamed", "aliases": [],
                               "description": "A traveler making camp for the night.",
                               "evidence": [{"block_id": "b1", "quote": "Mara"}], "voice_id": "voice_mara"})
        if self.voice_profile:
            voices.append({"id": "voice_mara", "description": "A completely redesigned voice profile."})
        response = {"characters": characters, "roles": roles, "voices": voices, "merges": [],
                    "issues": [], "summary": "Checked every target unit against the cast."}
        if not self.legacy:
            response["findings"] = findings
            response["certified"] = self.certified
        return response


class VerifyCoverageTests(unittest.TestCase):
    def test_duplicate_anchor_for_different_people_is_rejected_before_caching(self):
        class DuplicateFinding(CoverageFake):
            def respond(self, payload):
                response = super().respond(payload)
                if len(response["findings"]) == 2:
                    for field in ("block_id", "start", "end", "quote"):
                        response["findings"][1][field] = response["findings"][0][field]
                return response

        client = DuplicateFinding(self.source, second_speaker=True)
        with self.assertRaisesRegex(ValueError, "duplicate source anchor"):
            verify_coverage(self.source, mara_cast(self.source), self._requests(client),
                             [self.atoms], policy=DEFAULT_POLICY)
        self.assertFalse(list(self.cache.glob("*.json")))

    def test_merge_nominated_during_repair_survives_the_clean_followup_scan(self):
        class MergeFinding(CoverageFake):
            def respond(self, payload):
                response = super().respond(payload)
                if any(row["id"] == "c1" for row in response["characters"]):
                    response["merges"] = [{"from_character_id": "c1", "into_character_id": "char_mara",
                                           "reason": "This possible identity link needs independent review.",
                                           "evidence": [{"block_id": "b2", "quote": KEEPER_QUOTE}]}]
                return response

        result = verify_coverage(self.source, mara_cast(self.source), self._requests(MergeFinding(self.source)),
                                 [self.atoms], policy=DEFAULT_POLICY)
        self.assertEqual(len(result["cast"]["characters"]), 2)
        self.assertEqual(len(result["pending_merges"]), 1)
        self.assertEqual(result["pending_merges"][0]["into_character_id"], "char_mara")

    def setUp(self):
        self.source = make_source()
        self.before = deepcopy(self.source)
        self.atoms = [unit for batch in make_units(self.source, 256) for unit in batch]
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.cache = Path(self._temporary.name)

    def _requests(self, client, *, budget=None, repair_attempts=0):
        return CachedRequests(client, self.cache, repair_attempts=repair_attempts,
                              max_request_chars=budget if budget is not None else 200000)

    def _budget_for(self, cast):
        payload, _token = casting_coverage._region_request(self.source, cast, DEFAULT_POLICY, [self.atoms[0]])
        return len(casting_coverage.COVERAGE_SYSTEM) + len(json.dumps(payload, ensure_ascii=False)) + 64

    def test_adds_missing_speaker_from_exact_anchor_and_reverifies_only_dirty_atoms(self):
        client = CoverageFake(self.source)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        added = [character for character in result["cast"]["characters"] if character["id"] != "char_mara"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["name"], "The gate keeper")
        self.assertTrue(added[0]["id"].startswith("char_c"))
        self.assertEqual(result["issues"], [])
        round_two = [payload for _system, payload in client.calls if payload["target_units"] == [self.atoms[1]]]
        round_one = [payload for _system, payload in client.calls if len(payload["target_units"]) == 3]
        self.assertEqual(len(round_one), 1)
        self.assertGreaterEqual(len(round_two), 1)
        for _system, payload in client.calls[1:]:
            self.assertEqual([unit["unit_id"] for unit in payload["target_units"]], ["u0000002"])
        expected = finding_id(fingerprint(self.source), "b2",
                              self.client_anchor()[0], self.client_anchor()[1])
        self.assertEqual([record["finding_id"] for record in result["provenance"]["findings"]], [expected])
        self.assertEqual(result["provenance"]["findings"][0]["disposition"], "addition")
        self.assertEqual(result["provenance"]["rounds"], 2)
        self.assertEqual([[unit["unit_id"] for unit in region] for region in result["dirty_regions"]], [["u0000002"]])
        self.assertEqual(self.source, self.before)

    def client_anchor(self):
        text = self.source["blocks"][1]["text"]
        start = text.index(KEEPER_QUOTE)
        return start, start + len(KEEPER_QUOTE)

    def test_binding_existing_candidate_applies_no_cast_change(self):
        cast = mara_cast(self.source)
        client = CoverageFake(self.source, bind="char_mara")
        result = verify_coverage(self.source, cast, self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(result["cast"], cast)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["provenance"]["findings"][0]["disposition"], "binding")
        _system, payload = client.calls[0]
        self.assertEqual(payload["cast_view"]["records"], cast)
        self.assertEqual(payload["cast_view"]["omitted_character_ids"], [])
        self.assertEqual(payload["cast_view"]["full_cast_fingerprint"], fingerprint(cast))
        self.assertIn("char_mara", payload["cast_view"]["included_character_ids"])
        self.assertEqual([[unit["unit_id"] for unit in region] for region in result["dirty_regions"]], [["u0000002"]])

    def test_repeated_finding_becomes_unresolved_issue_and_terminates(self):
        client = CoverageFake(self.source, repeat=True)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(result["provenance"]["rounds"], 2)
        added = [character for character in result["cast"]["characters"]]
        self.assertEqual(len(added), 1)
        unresolved = [issue for issue in result["issues"] if issue["kind"] == "coverage_finding_unresolved"]
        self.assertEqual(len(unresolved), 1)
        self.assertTrue(unresolved[0]["target"].startswith("cov_"))
        self.assertEqual(result["provenance"]["findings"][0]["repeats"], 1)
        self.assertEqual(len(result["provenance"]["unapplied_candidates"]), 3)
        self.assertEqual({entry["collection"] for entry in result["provenance"]["unapplied_candidates"]},
                         {"characters", "roles", "voices"})

    def test_missing_findings_and_certification_never_counts_as_clean(self):
        client = CoverageFake(self.source, legacy=True)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(result["cast"], seed_cast())
        self.assertEqual(result["dirty_regions"], [])
        uncertified = [issue for issue in result["issues"] if issue["kind"] == "coverage_scan_uncertified"]
        self.assertGreaterEqual(len(uncertified), 1)

    def test_explicit_empty_findings_with_certification_are_clean(self):
        client = CoverageFake(self.source, findings=False, certified=True)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["dirty_regions"], [])
        self.assertEqual(result["provenance"]["rounds"], 1)
        self.assertTrue(all(region["certified"] for region in result["provenance"]["regions"]))

    def test_uncertified_regions_are_flagged_while_findings_still_apply(self):
        client = CoverageFake(self.source, certified=False)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(len(result["cast"]["characters"]), 1)
        uncertified = [issue for issue in result["issues"] if issue["kind"] == "coverage_scan_uncertified"]
        self.assertGreaterEqual(len(uncertified), 1)

    def test_renames_and_voice_profile_changes_are_rejected(self):
        cases = [(CoverageFake(self.source, rename=True), "must stay unchanged in coverage"),
                 (CoverageFake(self.source, voice_profile=True), "ensemble")]
        for client, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError) as raised:
                    verify_coverage(self.source, mara_cast(self.source), self._requests(client),
                                    make_units(self.source, 256), policy=DEFAULT_POLICY)
                self.assertIn(expected, str(raised.exception))

    def test_out_of_region_anchors_are_rejected_when_regions_are_split(self):
        client = CoverageFake(self.source, outside=True)
        with self.assertRaises(ValueError) as raised:
            verify_coverage(self.source, seed_cast(), self._requests(client, budget=self._budget_for(seed_cast())),
                            make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertIn("outside this region's target_units", str(raised.exception))

    def test_budget_too_small_for_even_one_atom_is_rejected(self):
        client = CoverageFake(self.source)
        with self.assertRaises(ValueError) as raised:
            verify_coverage(self.source, seed_cast(), self._requests(client, budget=64),
                            make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertIn("never trimmed", str(raised.exception))

    def test_two_missing_speakers_in_one_atom_alloc_distinct_sequential_candidates(self):
        first_client = CoverageFake(self.source)
        first = verify_coverage(self.source, seed_cast(), self._requests(first_client),
                                make_units(self.source, 256), policy=DEFAULT_POLICY)
        keeper = first["cast"]["characters"][0]
        self.assertEqual(keeper["name"], "The gate keeper")
        second_client = CoverageFake(self.source, second_speaker=True)
        second = verify_coverage(self.source, first["cast"], self._requests(second_client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        by_name = {character["name"]: character for character in second["cast"]["characters"]}
        self.assertEqual(sorted(by_name), ["The gate keeper", "The night stranger"])
        self.assertEqual(by_name["The gate keeper"], keeper)
        self.assertNotEqual(by_name["The gate keeper"]["id"], by_name["The night stranger"]["id"])
        tokens = {record["finding_id"]: record for record in second["provenance"]["findings"]}
        self.assertEqual(len(tokens), 1)
        self.assertEqual(second["provenance"]["findings"][0]["disposition"], "addition")
        first_token, second_token = self._region_tokens(first_client, second_client)
        self.assertNotEqual(first_token, second_token)
        self.assertIn("u0000002-u0000002", first_token)
        self.assertIn("u0000002-u0000002", second_token)

    def _region_tokens(self, first_client, second_client):
        units = lambda payload: [unit["unit_id"] for unit in payload["target_units"]]
        first = [payload["region"] for _system, payload in first_client.calls
                 if units(payload) == ["u0000002"]]
        second = [payload["region"] for _system, payload in second_client.calls
                  if units(payload) == ["u0000002"]]
        self.assertTrue(first and second)
        return first[0], second[0]

    def test_sequential_newcomers_in_one_atom_keep_distinct_canonical_ids(self):
        client = CoverageFake(self.source, sequential_people=4)
        result = verify_coverage(self.source, seed_cast(), self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        people = [character for character in result["cast"]["characters"]]
        self.assertEqual([character["name"] for character in people],
                         ["Person 1", "Person 2", "Person 3", "Person 4"])
        self.assertEqual(len({character["id"] for character in people}), 4)
        self.assertEqual(result["provenance"]["rounds"], 5)
        self.assertEqual([record["disposition"] for record in result["provenance"]["findings"]],
                         ["addition", "addition", "addition", "addition"])
        self.assertFalse([issue for issue in result["issues"]
                          if issue["kind"] in ("coverage_round_budget", "coverage_repair_cycle")])
        self.assertFalse(result["provenance"]["round_budget_exhausted"])
        self.assertEqual([[unit["unit_id"] for unit in region] for region in result["dirty_regions"]],
                         [["u0000002"]])
        self.assertEqual(self.source, self.before)

    def test_anchor_spanning_two_unit_intervals_is_rejected(self):
        source = make_long_source()
        client = CoverageFake(source, span_units=True)
        with self.assertRaises(ValueError) as raised:
            verify_coverage(source, seed_cast(), self._requests(client),
                            make_units(source, 256), policy=DEFAULT_POLICY)
        self.assertIn("not fully contained", str(raised.exception))

    def test_canonical_metadata_updates_surface_identity_proposals_without_applying(self):
        cast = mara_cast(self.source)
        client = CoverageFake(self.source, alias_proposal=True)
        result = verify_coverage(self.source, cast, self._requests(client),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        self.assertEqual(result["cast"], cast)
        proposals = [issue for issue in result["issues"] if issue["kind"] == "identity_proposal"]
        self.assertEqual([issue["target"] for issue in proposals], ["char_mara"])
        observations = result["provenance"]["shared_record_observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["id"], "char_mara")
        self.assertEqual(observations[0]["records"][0]["aliases"], ["Mara of the camp"])

    def test_oversized_cast_switches_to_scoped_complete_record_views(self):
        cast = mara_cast(self.source, size=120)
        client = CoverageFake(self.source)
        budget = self._scoped_budget(cast)
        result = verify_coverage(self.source, cast, self._requests(client, budget=budget),
                                 make_units(self.source, 256), policy=DEFAULT_POLICY)
        _system, payload = client.calls[0]
        self.assertFalse(payload["cast_view"]["complete"])
        self.assertGreater(len(payload["cast_view"]["omitted_character_ids"]), 0)
        self.assertNotIn("char_extra001", payload["cast_view"]["included_character_ids"])
        self.assertIn("char_mara", payload["cast_view"]["included_character_ids"])
        self.assertTrue(payload["cast_view"]["records"]["characters"])
        self.assertEqual(len(result["cast"]["characters"]), 121)
        self.assertEqual(result["issues"], [])

    def _scoped_budget(self, cast):
        """A budget that rejects the full-cast view but admits every scoped region."""
        source_fingerprint = fingerprint(self.source)
        cast_fingerprint = fingerprint(cast)
        full_payload, _token = casting_coverage._region_request(
            self.source, cast, DEFAULT_POLICY, [self.atoms[0]],
            source_fingerprint=source_fingerprint, cast_fingerprint=cast_fingerprint, scoped=False)
        full_size = len(casting_coverage.COVERAGE_SYSTEM) + len(json.dumps(full_payload, ensure_ascii=False))
        scoped_sizes = []
        for atom in self.atoms:
            scoped_payload, _token = casting_coverage._region_request(
                self.source, cast, DEFAULT_POLICY, [atom],
                source_fingerprint=source_fingerprint, cast_fingerprint=cast_fingerprint, scoped=True)
            scoped_sizes.append(len(casting_coverage.COVERAGE_SYSTEM)
                                + len(json.dumps(scoped_payload, ensure_ascii=False)))
        self.assertGreater(full_size, max(scoped_sizes))
        return max(scoped_sizes) + 64



if __name__ == "__main__":
    unittest.main()
