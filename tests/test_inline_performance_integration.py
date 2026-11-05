"""Approved inline metadata survives downstream resegmentation and speaker changes."""

from copy import deepcopy
import unittest

from epub_tts_narrator.attribution_review import audit_attribution
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.narration_boundaries import repair_narration_boundaries
from epub_tts_narrator.performance import apply_bracket_delivery
from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.pipeline import segment
from epub_tts_narrator.speaker_completion import complete_speakers
from epub_tts_narrator.spoken_text import build_spoken
from test_attribution_review import Requests as ReviewRequests, fixture
from test_contracts import valid_document
from test_speaker_completion import Requests as CompletionRequests, missing_document


def annotated_document(text):
    """One dialogue row over the block carrying an approved edit and inline controls."""
    document = valid_document()
    document["cast"]["characters"][0]["evidence"] = []
    document["cast"]["voices"][1]["traits"] = []
    document["source"]["blocks"][0]["text"] = text
    document["segments"][0:2] = [segment(document["source"], "b1", 0, text,
                                         document["cast"]["roles"][1], kind="dialogue")]
    row = document["segments"][0]
    hello = text.index("Hello")
    row["speech_normalization"] = {"version": 1, "edits": [
        {"start": hello, "end": hello + len("Hello"), "original": "Hello", "replacement": "Hi",
         "direction": "warm", "reason": "Casual greeting."}]}
    spoken = build_spoken(row)[0]
    row["spoken_text"] = spoken
    row["performance_controls"] = {
        "version": 1, "text_sha256": text_sha256(spoken), "controls": [
            {"kind": "delivery", "tag": "style", "direction": "hushed",
             "start": spoken.index("Hi"), "end": spoken.index("Hi") + len("Hi"), "text": "Hi"},
            {"kind": "delivery", "tag": "style", "direction": "whispering",
             "start": spoken.index("she said"), "end": spoken.index("she said") + len("she said"),
             "text": "she said"},
            {"kind": "delivery", "tag": "style", "direction": "soft",
             "start": spoken.index("wait"), "end": spoken.index("wait") + len("wait"), "text": "wait"},
            {"kind": "vocalization", "offset": spoken.index("“", spoken.index("she said")), "sound": "gasp"},
            {"kind": "pause", "offset": len(spoken), "duration_ms": 400}]}
    return document


def summarize(pieces):
    found = []
    for piece in pieces:
        if "performance_controls" in piece:
            record = validate_performance(piece["performance_controls"], build_spoken(piece)[0])
            for entry in record["controls"]:
                position = entry["offset"] if "offset" in entry else (entry["start"], entry["end"])
                found.append((piece["text"], entry["kind"], position,
                              entry.get("sound") or entry.get("direction")))
    return found


def weary_row(document):
    row = document["segments"][0]
    row["speech_normalization"] = {"version": 1, "edits": [
        {"start": 0, "end": 4, "original": "Mara", "replacement": "Mara",
         "direction": "weary", "reason": "Tired delivery."}]}
    spoken = build_spoken(row)[0]
    row["spoken_text"] = spoken
    row["performance_controls"] = {"version": 1, "text_sha256": text_sha256(spoken), "controls": [
        {"kind": "delivery", "tag": "style", "direction": "whispering",
         "start": spoken.index("Hello"), "end": spoken.index("Hello") + 5, "text": "Hello"},
        {"kind": "vocalization", "offset": spoken.index("“"), "sound": "gasp"}]}
    return row


class InlinePerformanceIntegrationTests(unittest.TestCase):
    def test_narration_boundary_repairs_project_edits_and_controls_onto_pieces(self):
        text = "“Hello,” she said, “wait.”"
        document = annotated_document(text)
        before = deepcopy(document)
        result = repair_narration_boundaries(document)
        pieces = [item for item in result["segments"] if item["block_id"] == "b1"]
        self.assertEqual([item["text"] for item in pieces], ["“Hello,”", " she said, ", "“wait.”"])
        self.assertEqual([item["role_id"] for item in pieces], ["mara", "narrator", "mara"])
        self.assertEqual(pieces[0]["speech_normalization"]["edits"][0]["replacement"], "Hi")
        self.assertEqual(build_spoken(pieces[0])[0], "“Hi,”")
        self.assertEqual(summarize(pieces), [
            ("“Hello,”", "delivery", (1, 3), "hushed"),
            (" she said, ", "delivery", (1, 9), "whispering"),
            ("“wait.”", "vocalization", 0, "gasp"),
            ("“wait.”", "delivery", (1, 5), "soft"),
            ("“wait.”", "pause", 7, None)])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(document, before)

    def test_bracket_projection_survives_a_later_narration_repair(self):
        document = apply_bracket_delivery(annotated_document("“[Hello],” she said, “wait.”"))
        result = repair_narration_boundaries(document)
        pieces = [item for item in result["segments"] if item["block_id"] == "b1"]
        self.assertEqual([item["text"] for item in pieces],
                         ["“", "[Hello]", ",”", " she said, ", "“wait.”"])
        self.assertEqual([item["role_id"] for item in pieces], ["mara", "mara", "mara", "narrator", "mara"])
        bracketed = pieces[1]
        self.assertIn("performance", bracketed)
        self.assertEqual(bracketed["speech_normalization"]["edits"][0]["original"], "Hello")
        self.assertEqual(build_spoken(bracketed)[0], "Hi")
        self.assertEqual(summarize(pieces), [
            ("[Hello]", "delivery", (0, 2), "hushed"),
            (" she said, ", "delivery", (1, 9), "whispering"),
            ("“wait.”", "vocalization", 0, "gasp"),
            ("“wait.”", "delivery", (1, 5), "soft"),
            ("“wait.”", "pause", 7, None)])
        self.assertEqual(validate_document(result), [])

    def test_approved_attribution_corrections_project_the_original_inline_metadata(self):
        document = fixture(count=1)
        weary_row(document)
        before = deepcopy(document)
        result = audit_attribution(document, ReviewRequests())
        rows = [item for item in result["segments"] if item["block_id"] == "b1"]
        self.assertEqual([item["role_id"] for item in rows], ["narrator_main", "mara"])
        self.assertEqual(rows[0]["speech_normalization"]["edits"][0]["direction"], "weary")
        self.assertEqual(summarize(rows), [
            ("“Hello.”", "vocalization", 0, "gasp"),
            ("“Hello.”", "delivery", (1, 6), "whispering")])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 1)
        self.assertEqual(document, before)

    def test_speaker_completion_projects_approved_metadata_after_attribution_only_votes(self):
        document = missing_document(1)
        row = weary_row(document)
        row["performance_controls"]["controls"] = [row["performance_controls"]["controls"][1]]
        spoken = build_spoken(row)[0]
        row["performance_controls"]["text_sha256"] = text_sha256(spoken)
        result = complete_speakers(document, CompletionRequests())
        rows = [item for item in result["segments"] if item["block_id"] == "b1"]
        self.assertEqual([item["role_id"] for item in rows], ["narrator_main", "mara"])
        self.assertEqual(rows[0]["speech_normalization"]["edits"][0]["direction"], "weary")
        record = validate_performance(rows[1]["performance_controls"], build_spoken(rows[1])[0])
        self.assertEqual([entry["offset"] for entry in record["controls"]], [0])
        self.assertEqual(validate_document(result), [])
        self.assertTrue(result["readiness"]["ready"])

    def test_inline_fields_in_model_responses_fail_votes_and_never_replace_saved_metadata(self):
        document = missing_document(1)
        saved = deepcopy(weary_row(document)["performance_controls"])

        def inject(result, payload):
            for piece in result["segments"]:
                piece["performance_controls"] = {"version": 1, "text_sha256": "0" * 64, "controls": []}
            return result

        result = complete_speakers(document, CompletionRequests(modify=inject))
        case = result["provenance"]["speaker_completion"]["cases"][0]
        self.assertEqual(case["status"], "unresolved")
        self.assertTrue(all(vote["status"] == "error" for turn in case["rounds"]
                            for vote in turn["votes"]))
        self.assertEqual(result["segments"], document["segments"])
        self.assertEqual(document["segments"][0]["performance_controls"], saved)

    def test_attribution_scan_rejects_inline_fields_on_proposed_segments(self):
        def inject(result, payload):
            result["corrections"][0]["segments"][0]["performance_controls"] = {"version": 1}
            return result

        with self.assertRaises(ValueError) as raised:
            audit_attribution(fixture(count=1), ReviewRequests(modify=inject))
        self.assertIn("exactly these keys", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
