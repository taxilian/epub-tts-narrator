"""Approved inline controls validate against the spoken form and survive resegmentation."""

from copy import deepcopy
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.inline_performance import MAX_DROPS_RECORDED, project_segments
from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.spoken_text import build_spoken
from test_contracts import valid_document


def plain(text, start=0, block="b1", **extra):
    row = {"id": f"{block}:{start}:{start + len(text)}", "block_id": block, "start": start,
           "end": start + len(text), "text": text, "kind": "dialogue"}
    row.update(extra)
    return row


def controls(text, entries):
    return {"version": 1, "text_sha256": text_sha256(text), "controls": entries}


def style(text, word, direction="whispering"):
    return {"kind": "delivery", "tag": "style", "direction": direction,
            "start": text.index(word), "end": text.index(word) + len(word), "text": word}


def event(offset, kind="vocalization", value="gasp"):
    item = {"kind": kind, "offset": offset}
    item["sound" if kind == "vocalization" else "duration_ms"] = value
    return item


def summary(entry):
    if "offset" in entry:
        return (entry["kind"], entry["offset"], entry.get("sound", entry.get("duration_ms")))
    return (entry["kind"], (entry["start"], entry["end"]), entry["direction"])


def collected(rows):
    found = []
    for item in rows:
        if "performance_controls" in item:
            record = validate_performance(item["performance_controls"], build_spoken(item)[0])
            found.extend((item["text"], summary(entry)) for entry in record["controls"])
    return found


class InlinePerformanceTests(unittest.TestCase):
    def test_split_styles_fragment_at_boundaries_and_events_land_exactly_once(self):
        text = "Mara said softly, “Hello there.”"
        entries = [style(text, "softly, “Hello"), style(text, "there", "warm"),
                   event(5, "vocalization", "breath"), event(text.index("“")),
                   event(len(text), "pause", 400)]
        original = [plain(text, performance_controls=controls(text, entries))]
        cut = text.index("“")
        result = project_segments(original, [plain(text[:cut]), plain(text[cut:], start=cut)])
        self.assertEqual([item["text"] for item in result], [text[:cut], text[cut:]])
        self.assertTrue(all("speech_normalization" not in item for item in result))
        self.assertEqual(sorted(collected(result)), [
            (text[:cut], ("delivery", (10, 18), "whispering")),
            (text[:cut], ("vocalization", 5, "breath")),
            (text[cut:], ("delivery", (0, 6), "whispering")),
            (text[cut:], ("delivery", (7, 12), "warm")),
            (text[cut:], ("pause", 14, 400)),
            (text[cut:], ("vocalization", 0, "gasp"))])

    def test_merged_original_rows_rebase_edits_and_controls_onto_one_new_row(self):
        first = plain("Wull said, ")
        first["speech_normalization"] = {"version": 1, "edits": [
            {"start": 0, "end": 4, "original": "Wull", "replacement": "Well",
             "direction": "rural accent", "reason": "Keep the dialect sound."}]}
        second = plain("“Aye.”", start=11)
        second["performance_controls"] = controls("“Aye.”", [event(0), style("“Aye.”", "Aye")])
        result = project_segments([first, second], [plain("Wull said, “Aye.”")])
        spoken = build_spoken(result[0])[0]
        self.assertEqual(spoken, "Well said, “Aye.”")
        self.assertEqual(result[0]["speech_normalization"], first["speech_normalization"])
        record = validate_performance(result[0]["performance_controls"], spoken)
        self.assertEqual([summary(entry) for entry in record["controls"]],
                         [("vocalization", 11, "gasp"), ("delivery", (12, 15), "whispering")])

    def test_a_boundary_cut_drops_only_that_edit_and_the_cut_pronunciation(self):
        text = "Wull said hellay there"
        row = plain(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 0, "end": 4, "original": "Wull", "replacement": "Well",
             "direction": "rural", "reason": "Dialect."},
            {"start": 10, "end": 16, "original": "hellay", "replacement": "hella",
             "direction": "drawl", "reason": "Stretched."}]}
        spoken = build_spoken(row)[0]
        row["performance_controls"] = controls(spoken, [
            {"kind": "pronunciation", "alphabet": "respelling", "start": spoken.index("hella"),
             "end": spoken.index("hella") + 5, "text": "hella", "replacement": "hell-uh"},
            style(spoken, "said", "calm")])
        drops = []
        result = project_segments([row], [plain(text[:13]), plain(text[13:], start=13)], drops=drops)
        self.assertEqual("".join(item["text"] for item in result), text)
        self.assertEqual(result[0]["speech_normalization"]["edits"], [row["speech_normalization"]["edits"][0]])
        self.assertEqual(sorted(collected(result)), [(text[:13], ("delivery", (5, 9), "calm"))])
        self.assertTrue(all("speech_normalization" not in item and "performance_controls" not in item
                            for item in result[1:]))
        self.assertEqual(sorted((drop["kind"], drop["reason"].split(";")[0]) for drop in drops),
                         [("edit", "a new segment boundary cuts this edit"),
                          ("pronunciation", "a new segment boundary cuts this pronunciation")])

    def test_controls_ambiguous_inside_a_replacement_drop_without_globalizing(self):
        text = "said waaait now"
        row = plain(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 5, "end": 11, "original": "waaait", "replacement": "waaaait",
             "direction": "stretched", "reason": "Trail off."}]}
        spoken = build_spoken(row)[0]
        row["performance_controls"] = controls(spoken, [
            style(spoken, "waaaait"), style(spoken, "now", "warm"), event(8), event(9), event(12)])
        drops = []
        result = project_segments([row], [plain(text)], drops=drops)
        record = validate_performance(result[0]["performance_controls"], build_spoken(result[0])[0])
        self.assertEqual([summary(entry) for entry in record["controls"]],
                         [("delivery", (5, 12), "whispering"), ("vocalization", 12, "gasp"),
                          ("delivery", (13, 16), "warm")])
        self.assertEqual([(drop["kind"], drop["reason"]) for drop in drops],
                         [("vocalization", "position falls inside an approved replacement; "
                                           "no source-bound anchor exists")] * 2)

    def test_stale_approved_records_are_rejected_not_silently_dropped(self):
        row = plain("They left.", performance_controls={"version": 1, "text_sha256": "0" * 64, "controls": []})
        with self.assertRaises(ValueError) as raised:
            project_segments([row], [plain("They left.")])
        self.assertIn("Performance record requires", str(raised.exception))

    def test_legacy_rows_pass_through_detached_and_inputs_stay_untouched(self):
        text = "Mara said, “Hello.”"
        original, new = [plain(text)], [plain(text[:5]), plain(text[5:], start=5)]
        before, new_before = deepcopy(original), deepcopy(new)
        result = project_segments(original, new)
        self.assertEqual(result, new_before)
        self.assertEqual(original, before)
        self.assertEqual(new, new_before)
        self.assertIsNot(result[0], new[0])
        self.assertTrue(all("performance_controls" not in item and "speech_normalization" not in item
                            for item in result))

    def test_changed_source_bounds_and_gaps_are_rejected(self):
        text = "One two three"
        row = plain(text, performance_controls=controls(text, []))
        with self.subTest(case="bounds"):
            with self.assertRaises(ValueError) as raised:
                project_segments([row], [plain(text[:9], performance_controls=controls(text[:9], []))])
            self.assertIn("identical source bounds", str(raised.exception))
        with self.subTest(case="gap"):
            with self.assertRaises(ValueError) as raised:
                project_segments([row], [plain("One", performance_controls=controls("One", [])),
                                         plain("three", start=8, performance_controls=controls("three", []))])
            self.assertIn("must tile", str(raised.exception))

    def test_the_drop_audit_is_bounded_with_an_overflow_marker(self):
        filler = "h" * 200
        text = "oh " + filler + " now"
        row = plain(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 3, "end": 3 + len(filler), "original": filler, "replacement": "h" * 210,
             "direction": "stretched", "reason": "Long trailing sound."}]}
        spoken = build_spoken(row)[0]
        row["performance_controls"] = controls(
            spoken, [style(spoken, "h" * 210)] + [event(offset) for offset in range(4, 74)])
        drops = []
        result = project_segments([row], [plain(text)], drops=drops)
        self.assertEqual(len(drops), MAX_DROPS_RECORDED + 1)
        self.assertTrue(drops[-1]["reason"].startswith("6 additional drop(s)"))
        self.assertEqual(len(collected(result)), 1)

    def test_annotated_blocks_project_independently_and_preserve_row_order(self):
        first = plain("One two three",
                      performance_controls=controls("One two three", [style("One two three", "One")]))
        second = plain("Four five", block="b2")
        new = [plain("One"), plain(" two three", start=3),
               plain("Four", block="b2"), plain(" five", start=4, block="b2")]
        result = project_segments([first, second], new)
        self.assertEqual([item["block_id"] for item in result], ["b1", "b1", "b2", "b2"])
        self.assertEqual(collected(result[:1]), [("One", ("delivery", (0, 3), "whispering"))])
        self.assertTrue(all("performance_controls" not in item for item in result[1:]))
        self.assertEqual(result[2:], new[2:])

    def test_contracts_bind_controls_to_the_approved_spoken_form(self):
        document = valid_document()
        row = document["segments"][-1]
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 0, "end": 4, "original": "They", "replacement": "They",
             "direction": "slowly drawn", "reason": "Weary narration."}]}
        spoken = build_spoken(row)[0]
        row["spoken_text"] = spoken
        row["performance_controls"] = controls(spoken, [
            {"kind": "delivery", "tag": "slow", "direction": "", "start": 0, "end": 4, "text": "They"},
            {"kind": "pronunciation", "alphabet": "respelling", "start": 5, "end": 9,
             "text": "left", "replacement": "lefft"}])
        self.assertEqual(validate_document(document), [])
        row["performance_controls"]["text_sha256"] = "0" * 64
        self.assertTrue(any("performance_controls" in error for error in validate_document(document)))
        row["performance_controls"]["controls"] = "nope"
        self.assertTrue(any("performance_controls" in error for error in validate_document(document)))


if __name__ == "__main__":
    unittest.main()
