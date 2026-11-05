from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.performance import apply_bracket_delivery
from epub_tts_narrator.inline_performance import project_segments
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.spoken_text import build_spoken
from test_contracts import valid_document
from test_pipeline import FakeClient
from epub_tts_narrator.pipeline import analyze


class BracketInlinePerformanceTests(unittest.TestCase):
    def document(self, text):
        document = valid_document()
        document["source"]["blocks"][1]["text"] = text
        row = document["segments"][-1]
        row.update(id=f"b2:0:{len(text)}", text=text, start=0, end=len(text))
        return document, row

    def test_bracket_splits_rebase_scoped_styles_and_deliver_boundary_events_once(self):
        text = "Say [secret] now."
        document, row = self.document(text)
        row["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(text), "controls": [
                {"kind": "delivery", "tag": "style", "direction": "whispering",
                 "start": text.index("secret"), "end": text.index("secret") + len("secret"), "text": "secret"},
                {"kind": "delivery", "tag": "style", "direction": "warm",
                 "start": text.index("now"), "end": text.index("now") + len("now"), "text": "now"},
                {"kind": "vocalization", "offset": text.index("]") + 1, "sound": "chuckle"},
            ],
        }
        original = deepcopy(document)

        result = apply_bracket_delivery(document)

        rows = [item for item in result["segments"] if item["block_id"] == "b2"]
        self.assertEqual([item["text"] for item in rows], ["Say ", "[secret]", " now."])
        self.assertEqual([build_spoken(item)[0] for item in rows], ["Say ", "secret", " now."])
        controls = []
        for item in rows:
            if "performance_controls" in item:
                record = validate_performance(item["performance_controls"], build_spoken(item)[0])
                controls.extend((item["text"], control) for control in record["controls"])
        styles = [(source, control["text"], control["direction"]) for source, control in controls
                  if control["kind"] == "delivery"]
        self.assertEqual(styles, [("[secret]", "secret", "whispering"), (" now.", "now", "warm")])
        events = [(source, control["offset"]) for source, control in controls if control["kind"] == "vocalization"]
        self.assertEqual(events, [(" now.", 0)])
        self.assertEqual(document, original)
        self.assertEqual(result["source"], original["source"])
        self.assertEqual(apply_bracket_delivery(result), result)

    def test_bracket_splits_preserve_contained_spelling_edits_and_their_control_mapping(self):
        text = "Wull [secrettt] now."
        document, row = self.document(text)
        row["speech_normalization"] = {"version": 1, "edits": [
            {"start": 0, "end": 4, "original": "Wull", "replacement": "Well",
             "direction": "colloquial accent", "reason": "Retain the source's dialect while clarifying the word."},
            {"start": text.index("secrettt"), "end": text.index("secrettt") + len("secrettt"),
             "original": "secrettt", "replacement": "secret", "direction": "sustained final consonant",
             "reason": "Spelling expresses a stretched final consonant."},
        ]}
        spoken = build_spoken(row)[0]
        row["spoken_text"] = spoken
        row["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(spoken), "controls": [
                {"kind": "delivery", "tag": "style", "direction": "whispering",
                 "start": spoken.index("secret"), "end": spoken.index("secret") + len("secret"), "text": "secret"},
            ],
        }

        result = apply_bracket_delivery(document)

        rows = [item for item in result["segments"] if item["block_id"] == "b2"]
        self.assertEqual([build_spoken(item)[0] for item in rows], ["Well ", "secret", " now."])
        self.assertEqual([item["speech_normalization"]["edits"][0]["original"] for item in rows[:2]],
                         ["Wull", "secrettt"])
        record = validate_performance(rows[1]["performance_controls"], "secret")
        self.assertEqual(record["controls"][0]["text"], "secret")
        self.assertEqual((record["controls"][0]["start"], record["controls"][0]["end"]), (0, 6))
        self.assertEqual("".join(item["text"] for item in rows), text)
        self.assertEqual(apply_bracket_delivery(result), result)

    def test_projection_rejects_changed_canonical_text_even_when_offsets_still_match(self):
        document, row = self.document("Keep this secret.")
        row["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(row["text"]), "controls": [
                {"kind": "delivery", "tag": "whisper", "direction": "", "start": 0,
                 "end": len(row["text"]), "text": row["text"]},
            ],
        }
        changed = deepcopy(document["segments"])
        changed[-1]["text"] = changed[-1]["text"].replace("secret", "SECRET")

        with self.assertRaises(ValueError):
            project_segments(document["segments"], changed)

    def test_control_only_attribution_may_store_unchanged_spoken_text_without_bracket_metadata(self):
        document, row = self.document("Keep this secret.")
        row["spoken_text"] = row["text"]
        row["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(row["text"]), "controls": [
                {"kind": "delivery", "tag": "whisper", "direction": "", "start": 0,
                 "end": len(row["text"]), "text": row["text"]},
            ],
        }

        self.assertEqual(validate_document(document), [])

        row["spoken_text"] = "Rewrite the book instead."
        self.assertTrue(validate_document(document))

    def test_analysis_reviews_cross_bracket_styles_before_projecting_onto_source_pieces(self):
        source = valid_document()["source"]
        source["blocks"][0]["text"] = "Mara said softly in a whisper, “A [spark] lights the way.”"

        class Model(FakeClient):
            def complete(self, system, payload):
                if payload.get("mode") == "inline_performance_review":
                    self.calls.append((system, deepcopy(payload)))
                    return {"decisions": [{"id": item["id"], "verdict": "keep"}
                                          for item in payload["proposals"]]}
                return super().complete(system, payload)

            def attribute(self, payload):
                response = super().attribute(payload)
                for unit in response["units"]:
                    for piece in unit["segments"]:
                        if piece["kind"] == "dialogue":
                            piece["proposals"] = {"controls": [{"kind": "delivery", "tag": "whisper",
                                "original": piece["text"], "reason": "The source explicitly specifies a whisper."}]}
                return response

        client = Model()
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(source, client, Path(temporary), identity_consensus=False)

        styled = [row for row in result["segments"] if row.get("performance_controls", {}).get("controls")]
        self.assertEqual([row["text"] for row in styled], ["“A ", "[spark]", " lights the way.”"])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(sum(payload.get("mode") == "inline_performance_review" for _, payload in client.calls), 1)
