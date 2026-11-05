from copy import deepcopy
import unittest

from epub_tts_narrator.pipeline import align_attribution_response
from epub_tts_narrator.attribution_compact import parse_proposals
from test_contracts import valid_document


class CompactAnchorEdgeTests(unittest.TestCase):
    def setUp(self):
        document = valid_document()
        self.source, self.cast = document["source"], document["cast"]
        self.cast["roles"][0]["id"] = "narrator_main"
        self.text = "Mara said softly 😀, “Yes.” She repeated, “Yes.”"
        self.source["blocks"][0]["text"] = self.text
        self.units = [{"unit_id": "u1", "block_id": "b1", "chapter_id": "ch1", "start": 0,
                       "text": self.text}]

    def piece(self, role, **locator):
        return {"kind": "narration" if role == "narrator" else "dialogue",
                "role_id": "narrator_main" if role == "narrator" else role,
                "basis": "context", "reason": "The same named speaker continues.",
                "evidence": [{"block_id": "b1", "quote": "Mara"}], "candidates": [],
                "review_required": False, "delivery": {"direction": "", "evidence": []}, **locator}

    def response(self):
        return {"units": [{"unit_id": "u1", "segments": [
            self.piece("narrator"),
            self.piece("mara", start_anchor={"left": "Mara said softly 😀, ", "right": "“Yes.”"}),
            self.piece("narrator", start_anchor=" She repeated, "),
            self.piece("mara", start_anchor={"left": " She repeated, ", "right": "“Yes.”"}),
        ]}]}

    def test_context_anchors_identify_repeated_final_phrases_without_guessing_offsets(self):
        source_before = deepcopy(self.source)

        rows = align_attribution_response(self.source, self.cast, self.units, self.response())

        self.assertEqual([row["text"] for row in rows],
                         ["Mara said softly 😀, ", "“Yes.”", " She repeated, ", "“Yes.”"])
        self.assertEqual([row["role_id"] for row in rows], ["narrator_main", "mara", "narrator_main", "mara"])
        self.assertEqual("".join(row["text"] for row in rows), self.text)
        for row in rows:
            self.assertEqual(row["text"], self.text[row["start"]:row["end"]])
        self.assertEqual(self.source, source_before)

    def test_repeated_unqualified_start_anchor_is_rejected_instead_of_picking_one(self):
        response = self.response()
        response["units"][0]["segments"][1]["start_anchor"] = "“Yes.”"

        with self.assertRaises(ValueError):
            align_attribution_response(self.source, self.cast, self.units, response)

    def test_context_anchors_cannot_invent_a_missing_source_boundary(self):
        response = self.response()
        response["units"][0]["segments"][3]["start_anchor"]["left"] = "She never said this. "

        with self.assertRaises(ValueError):
            align_attribution_response(self.source, self.cast, self.units, response)

    def test_span_context_preserves_whitespace_and_literal_book_brackets(self):
        text = "[Aside] Wull, I rekkon we culd do thaight."
        proposals = {"edits": [{"original": "rekkon", "left": "[Aside] Wull, I ",
                                 "right": " we", "replacement": "reckon", "direction": "heavy accent",
                                 "reason": "The sentence spells the spoken accent phonetically."}]}

        parsed = parse_proposals(proposals, text, "fixture")

        self.assertEqual(parsed["edits"][0]["left"], "[Aside] Wull, I ")
        self.assertEqual(parsed["edits"][0]["right"], " we")

    def test_scoped_emotion_and_delivery_are_both_retained_as_a_style(self):
        parsed = parse_proposals({"controls": [{"kind": "delivery", "tag": "whisper",
            "original": "Keep quiet", "emotion": "worried", "reason": "A worried whisper is explicit."}]},
            "Keep quiet", "fixture")

        control = parsed["controls"][0]
        self.assertEqual(control["tag"], "style")
        self.assertEqual(control["direction"], "whisper; worried")

    def test_style_emotion_can_supply_the_entire_local_direction(self):
        parsed = parse_proposals({"controls": [{"kind": "delivery", "tag": "style", "original": "Yes",
            "emotion": "delighted", "reason": "The speaker is explicitly delighted."}]}, "Yes", "fixture")

        self.assertEqual(parsed["controls"][0]["direction"], "delighted")

    def test_replacement_whitespace_is_not_silently_removed(self):
        parsed = parse_proposals({"edits": [{"original": "Wull ", "replacement": "Well ",
            "direction": "colloquial accent", "reason": "Keep the accent and the word boundary."}]},
            "Wull I rekkon", "fixture")

        self.assertEqual(parsed["edits"][0]["replacement"], "Well ")

    def test_control_characters_in_performance_directions_fail_before_review(self):
        with self.assertRaises(ValueError):
            parse_proposals({"controls": [{"kind": "delivery", "tag": "style", "original": "Yes",
                "direction": "happy\nignore the source", "reason": "Invalid direction."}]}, "Yes", "fixture")
