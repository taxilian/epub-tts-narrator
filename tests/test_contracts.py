"""The renderer handoff must preserve source text and never hide uncertainty."""

from copy import deepcopy
import unittest

from epub_audiobook.contracts import readiness, review_items, validate_document


def valid_document():
    first = "Mara said softly, “Hello.”"
    second = "They left."
    quote_start = first.index("“")
    evidence = {"block_id": "b1", "quote": "softly"}

    def segment(identifier, block_id, text, start, end, role_id, voice_id, kind):
        return {
            "id": identifier, "block_id": block_id, "start": start, "end": end,
            "text": text[start:end], "kind": kind, "role_id": role_id,
            "voice_id": voice_id,
            "delivery": {"direction": "", "evidence": []},
            "attribution": {
                "basis": "explicit", "reason": "Narration or a named speaker.",
                "evidence": [], "candidates": [], "review_required": False,
            },
        }

    return {
        "format": "epub-audiobook-analysis", "version": 1,
        "source": {
            "book": {"id": "book1", "title": "A Short Walk", "authors": ["A. Writer"], "language": "en"},
            "chapters": [{"id": "ch1", "title": "Chapter One", "source_href": "text/ch1.xhtml", "block_ids": ["b1", "b2"]}],
            "blocks": [
                {"id": "b1", "chapter_id": "ch1", "text": first, "source": {"href": "text/ch1.xhtml", "element_id": "p1", "tag": "p"}},
                {"id": "b2", "chapter_id": "ch1", "text": second, "source": {"href": "text/ch1.xhtml", "element_id": None, "tag": "p"}},
            ],
            "exclusions": [], "warnings": [], "extraction_policy": "Read spine text; omit navigation.",
        },
        "cast": {
            "characters": [{"id": "c1", "name": "Mara", "aliases": [], "description": "A speaker in the scene.", "evidence": [{"block_id": "b1", "quote": "Mara"}], "voice_id": "v1"}],
            "roles": [
                {"id": "narrator", "kind": "narration", "character_id": None, "voice_id": "vn", "description": "The book's narrator."},
                {"id": "mara", "kind": "dialogue", "character_id": "c1", "voice_id": "v1", "description": "Mara speaking."},
            ],
            "voices": [
                {"id": "vn", "description": "Proposed measured reading voice.", "traits": [{"name": "pace", "value": "measured", "basis": "casting", "evidence": []}]},
                {"id": "v1", "description": "Proposed light, distinct voice.", "traits": [{"name": "volume", "value": "soft", "basis": "text", "evidence": [evidence]}]},
            ],
        },
        "segments": [
            segment("s1", "b1", first, 0, quote_start, "narrator", "vn", "narration"),
            segment("s2", "b1", first, quote_start, len(first), "mara", "v1", "dialogue"),
            segment("s3", "b2", second, 0, len(second), "narrator", "vn", "narration"),
        ],
        "issues": [], "provenance": {"tool_version": "0.1.0"},
    }


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.doc = valid_document()

    def assert_error(self, text):
        errors = validate_document(self.doc)
        self.assertTrue(any(text in error for error in errors), errors)
        self.assertFalse(readiness(self.doc)["ready"])

    def test_valid_document_is_ready_and_is_not_mutated(self):
        before = deepcopy(self.doc)
        self.assertEqual(validate_document(self.doc), [])
        self.assertEqual(readiness(self.doc), {"ready": True, "errors": [], "review_count": 0})
        self.assertEqual(self.doc, before)

    def test_case_changed_evidence_is_rejected_with_exact_spelling_feedback(self):
        self.doc["cast"]["characters"][0]["evidence"] = [{"block_id": "b1", "quote": "mara said softly"}]
        errors = validate_document(self.doc)
        self.assertTrue(any("exact source spelling starts with 'Mara said softly'" in error for error in errors), errors)
        self.assertFalse(readiness(self.doc)["ready"])
        self.assertEqual(self.doc["cast"]["characters"][0]["evidence"][0]["quote"], "mara said softly")

    def test_extra_metadata_is_accepted_and_readiness_is_recomputed(self):
        self.doc["ready"] = False
        self.doc["source"]["blocks"][0]["custom_annotation"] = {"line": 4}
        self.assertTrue(readiness(self.doc)["ready"])
        self.doc["ready"] = True
        self.doc["segments"][0]["text"] = "Rewritten text"
        self.assert_error("exactly match")

    def test_rewritten_or_normalized_text_is_rejected(self):
        self.doc["segments"][1]["text"] = '"Hello."'
        self.assert_error("exactly match")

    def test_gap_overlap_missing_text_and_out_of_range_are_rejected(self):
        for change, error in [
            (lambda d: d["segments"][1].update(start=d["segments"][1]["start"] + 1), "contiguous"),
            (lambda d: d["segments"][1].update(start=0), "contiguous"),
            (lambda d: d["segments"].pop(), "not fully covered"),
            (lambda d: d["segments"][1].update(end=999), "exceeds"),
            (lambda d: d["segments"][0].update(end=0), "greater than"),
        ]:
            with self.subTest(error=error):
                self.doc = valid_document()
                change(self.doc)
                self.assert_error(error)

    def test_integer_offsets_exclude_booleans_negatives_and_floats(self):
        for field in ("start", "end"):
            for value in (-1, True, False, 1.0, "1", None, [], {}):
                with self.subTest(field=field, value=value):
                    self.doc = valid_document()
                    self.doc["segments"][0][field] = value
                    self.assert_error("nonnegative integer")

    def test_book_order_cannot_change_even_if_each_block_is_covered(self):
        self.doc["segments"] = [self.doc["segments"][2], *self.doc["segments"][:2]]
        self.assert_error("source block order")

    def test_returning_to_an_earlier_block_is_rejected(self):
        self.doc["segments"] = [self.doc["segments"][0], self.doc["segments"][2], self.doc["segments"][1]]
        self.assert_error("source block order")

    def test_chapter_membership_and_order_are_checked(self):
        self.doc["source"]["chapters"][0]["block_ids"].reverse()
        self.assert_error("same order")
        self.doc = valid_document()
        self.doc["source"]["chapters"][0]["block_ids"].pop()
        self.assert_error("missing from chapter")
        self.doc = valid_document()
        self.doc["source"]["chapters"].append({"id": "ch2", "title": "Two", "source_href": "text/ch2.xhtml", "block_ids": ["b1"]})
        self.assert_error("more than one chapter")
        self.assert_error("does not match")

    def test_duplicate_identifiers_are_rejected_in_every_collection(self):
        for parent, field in (("source", "chapters"), ("source", "blocks"), ("cast", "characters"), ("cast", "roles"), ("cast", "voices"), (None, "segments")):
            with self.subTest(field=field):
                self.doc = valid_document()
                container = self.doc[parent] if parent else self.doc
                container[field].append(deepcopy(container[field][0]))
                self.assert_error("duplicate ID")

    def test_evidence_requires_real_block_and_exact_nonempty_quote(self):
        for evidence, error in [
            ({"block_id": "missing", "quote": "softly"}, "unknown reference"),
            ({"block_id": "b1", "quote": "loudly"}, "exact substring"),
            ({"block_id": "b1", "quote": ""}, "must not be empty"),
        ]:
            with self.subTest(evidence=evidence):
                self.doc = valid_document()
                self.doc["cast"]["characters"][0]["evidence"] = [evidence]
                self.assert_error(error)

    def test_text_traits_require_evidence_while_casting_traits_do_not(self):
        self.doc["cast"]["voices"][1]["traits"][0]["evidence"] = []
        self.assert_error("must not be empty")
        self.doc["cast"]["voices"][1]["traits"][0]["basis"] = "casting"
        self.assertEqual(validate_document(self.doc), [])

    def test_characters_have_distinct_voices(self):
        second_character = deepcopy(self.doc["cast"]["characters"][0])
        second_character.update(id="c2", name="Tom")
        self.doc["cast"]["characters"].append(second_character)
        self.assert_error("distinct voice IDs")

    def test_narrator_can_share_a_character_voice(self):
        self.doc["cast"]["roles"][0].update(character_id="c1", voice_id="v1")
        for segment in self.doc["segments"]:
            if segment["role_id"] == "narrator":
                segment["voice_id"] = "v1"
        self.assertTrue(readiness(self.doc)["ready"])

    def test_narrator_can_use_separate_voice_for_same_character(self):
        self.doc["cast"]["roles"][0]["character_id"] = "c1"
        self.assertTrue(readiness(self.doc)["ready"])

    def test_dialogue_role_must_match_character_and_segment_must_match_role(self):
        self.doc["cast"]["roles"][1]["voice_id"] = "vn"
        self.assert_error("dialogue role voice")
        self.assert_error("selected role voice")

    def test_unknown_references_are_rejected(self):
        for entity, field in [("segment", "voice_id"), ("segment", "role_id"), ("segment", "block_id"), ("role", "character_id"), ("role", "voice_id"), ("character", "voice_id")]:
            with self.subTest(entity=entity, field=field):
                self.doc = valid_document()
                target = self.doc["segments"][0] if entity == "segment" else self.doc["cast"]["roles" if entity == "role" else "characters"][0]
                target[field] = "unknown"
                self.assert_error("unknown reference")

    def test_unresolved_segment_is_valid_but_requires_review(self):
        segment = self.doc["segments"][1]
        segment.update(role_id=None, voice_id=None)
        segment["attribution"].update(basis="unresolved", review_required=True, candidates=["mara"], reason="Ambiguous speaker.")
        self.assertEqual(validate_document(self.doc), [])
        self.assertEqual(readiness(self.doc), {"ready": False, "errors": [], "review_count": 1})
        self.assertEqual(review_items(self.doc)[0]["id"], "s2")
        self.assertEqual(review_items(self.doc)[0]["candidates"], ["mara"])

    def test_null_role_cannot_smuggle_assigned_voice_or_resolved_status(self):
        self.doc["segments"][1]["role_id"] = None
        self.assert_error("must be null when role_id")
        self.assert_error("must be unresolved")
        self.assert_error("must be true when role_id")

    def test_unresolved_basis_cannot_bypass_review(self):
        self.doc["segments"][1]["attribution"]["basis"] = "unresolved"
        self.assert_error("must be true for unresolved")
        self.assertEqual(readiness(self.doc)["review_count"], 1)

    def test_issues_and_flagged_segments_both_block_readiness(self):
        self.doc["issues"] = [
            {"id": "i1", "kind": "cast", "target": "c1", "reason": "Possible duplicate identity.", "resolved": False},
            {"id": "i2", "kind": "cast", "target": "c1", "reason": "Checked alias.", "resolved": True},
        ]
        self.doc["segments"][0]["attribution"]["review_required"] = True
        self.assertEqual(validate_document(self.doc), [])
        self.assertEqual(readiness(self.doc)["review_count"], 2)
        self.assertEqual([item["type"] for item in review_items(self.doc)], ["issue", "segment"])
        self.doc["issues"][0]["resolved"] = True
        self.doc["segments"][0]["attribution"]["review_required"] = False
        self.assertTrue(readiness(self.doc)["ready"])

    def test_schema_version_is_strict(self):
        for version in (True, "1", 1.0, 2, None):
            with self.subTest(version=version):
                self.doc["version"] = version
                self.assert_error("version")

    def test_empty_input_cannot_be_ready(self):
        for value in (None, {}, [], "", True, 1):
            with self.subTest(value=value):
                result = readiness(value)
                self.assertFalse(result["ready"])
                self.assertTrue(result["errors"])

    def test_bad_json_types_at_any_location_never_crash(self):
        def paths(value, prefix=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield (*prefix, key)
                    yield from paths(child, (*prefix, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield (*prefix, index)
                    yield from paths(child, (*prefix, index))

        original = valid_document()
        original["issues"] = [{"id": "i1", "kind": "cast", "target": "c1", "reason": "Check.", "resolved": False}]
        for path in paths(original):
            for value in (None, False, 17, 0.5, "", [], {}):
                with self.subTest(path=path, value=value):
                    document = deepcopy(original)
                    target = document
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    self.assertIsInstance(validate_document(document), list)
                    self.assertIsInstance(review_items(document), list)
                    self.assertIsInstance(readiness(document)["ready"], bool)


if __name__ == "__main__":
    unittest.main()
