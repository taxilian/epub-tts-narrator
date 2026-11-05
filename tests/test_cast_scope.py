"""Explicit complete-record cast scoping discloses exactly what requests omit."""

from copy import deepcopy
import unittest

from epub_tts_narrator.cast_scope import scoped_cast
from epub_tts_narrator.pipeline import fingerprint, make_units
from test_identity import fixture


def scope_fixture():
    source, cast = fixture()
    cast["characters"].append({"id": "c3", "name": "Quillon", "aliases": [],
        "description": "A person the supplied units never mention.", "voice_id": "v3", "evidence": []})
    cast["roles"].append({"id": "r3", "kind": "dialogue", "character_id": "c3",
                          "voice_id": "v3", "description": "Character dialogue."})
    cast["voices"].append({"id": "v3", "description": "A distinct quiet voice.", "traits": []})
    return source, cast


def all_units(source):
    return [unit for batch in make_units(source) for unit in batch]


class CastScopeTests(unittest.TestCase):
    def test_locally_relevant_records_are_copied_whole_with_explicit_disclosure(self):
        source, cast = scope_fixture()
        units = all_units(source)
        view, scope = scoped_cast(source, cast, units[:1])
        self.assertEqual(scope["included_ids"], ["c1", "c2"])
        self.assertFalse(scope["complete"])
        self.assertEqual(scope["omitted_counts"], {"characters": 1, "roles": 1, "voices": 1})
        self.assertEqual(scope["full_cast_fingerprint"], fingerprint(cast))
        self.assertEqual([row["id"] for row in view["characters"]], ["c1", "c2"])
        for collection in ("characters", "roles", "voices"):
            supplied = {row["id"] for row in view[collection]}
            self.assertTrue(supplied <= {row["id"] for row in cast[collection]})

    def test_supplied_records_are_never_trimmed_or_reordered_evidence(self):
        source, cast = scope_fixture()
        units = all_units(source)
        view, _ = scoped_cast(source, cast, units[:1])
        originals = {row["id"]: row for collection in ("characters", "roles", "voices") for row in cast[collection]}
        for collection in ("characters", "roles", "voices"):
            for row in view[collection]:
                self.assertEqual(row, originals[row["id"]])

    def test_requested_characters_are_included_even_when_locally_invisible(self):
        source, cast = scope_fixture()
        units = all_units(source)
        view, scope = scoped_cast(source, cast, units[1:], character_ids=["c1"])
        self.assertEqual(scope["included_ids"], ["c1", "c2"])
        self.assertEqual([row["id"] for row in view["characters"]], ["c1", "c2"])
        alone_view, alone_scope = scoped_cast(source, cast, units[1:])
        self.assertEqual(alone_scope["included_ids"], ["c2"])
        self.assertNotIn("c1", [row["id"] for row in alone_view["characters"]])

    def test_alias_mentions_count_as_local_visibility(self):
        source, cast = scope_fixture()
        cast["characters"][0]["aliases"] = ["the captain"]
        units = all_units(source)
        _, scope = scoped_cast(source, cast, units[1:])
        self.assertIn("c1", scope["included_ids"])

    def test_narration_roles_stay_and_bind_their_owner_characters(self):
        source, cast = scope_fixture()
        cast["roles"][0]["character_id"] = "c1"
        units = all_units(source)
        view, scope = scoped_cast(source, cast, units[1:])
        self.assertIn("c1", scope["included_ids"])
        self.assertIn("narrator_main", {row["id"] for row in view["roles"]})
        self.assertIn("vn", {row["id"] for row in view["voices"]})

    def test_complete_scope_reports_zero_omissions_when_every_record_applies(self):
        source, cast = scope_fixture()
        units = all_units(source)
        view, scope = scoped_cast(source, cast, units, character_ids=["c3"])
        self.assertTrue(scope["complete"])
        self.assertEqual(scope["omitted_counts"], {"characters": 0, "roles": 0, "voices": 0})
        self.assertEqual(scope["included_ids"], ["c1", "c2", "c3"])
        for collection in ("characters", "roles", "voices"):
            self.assertEqual(len(view[collection]), len(cast[collection]))

    def test_unknown_requested_character_is_rejected(self):
        source, cast = scope_fixture()
        with self.assertRaisesRegex(ValueError, "unknown character"):
            scoped_cast(source, cast, [], character_ids=["c1", "absent"])


if __name__ == "__main__":
    unittest.main()
